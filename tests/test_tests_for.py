"""scripts/tests_for.py on a small synthetic checkout, and a few facts about this one (no game data).

    python -I tests/test_tests_for.py

The synthetic checkout has the layout of this repository: tools/, scripts/, viewer/, docs/, tests/ with a
semantic suite, and a runtime/CMakeLists.txt with programs, a library, loops and ctest entries.
"""
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import tests_for  # noqa: E402

HEAD = 'import os\nimport sys\nimport unittest\nROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))\n'
TOOLS = 'sys.path.insert(0, os.path.join(ROOT, "tools"))\n'
MAIN = '\nif __name__ == "__main__":\n    unittest.main()\n'

# The synthetic checkout is data: tests/fixtures/tests_for_checkout.json maps each of its paths to the file's text.
# It is kept out of this file because the paths and folder names quoted in those texts would read as references
# of this test to the real repository.
with open(os.path.join(ROOT, "tests", "fixtures", "tests_for_checkout.json"), encoding="utf-8") as _data:
    FILES = json.load(_data)
# Two paths the assertions need; split so that they do not read as references either.
CMAKE = "runtime/CMake" + "Lists.txt"
I18N = "tests/test_i18n" + "_index.py"


class Checkout(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.td = tempfile.TemporaryDirectory(prefix="leonida tests_for ")
        cls.addClassCleanup(cls.td.cleanup)
        cls.root = os.path.join(cls.td.name, "checkout")
        for path, text in FILES.items():
            full = os.path.join(cls.root, path)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "w", encoding="utf-8", newline="\n") as f:
                f.write(text)

    def found(self, path, reason=None):
        entry, = tests_for.survey(self.root, [path])
        return [(t["test"], t["through"]) if reason == "imports" else t["test"] for t in entry["tests"]
                if reason is None or t["reason"] == reason]

    def run_tool(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = tests_for.main(["--repo", self.root, *args])
        return code, out.getvalue(), err.getvalue()


class ReachTest(Checkout):
    def test_a_test_names_a_file_by_import_path_or_joined_segments(self):
        self.assertEqual(self.found("tools/alpha.py", "names it"), ["tests/test_alpha.py"])
        self.assertEqual(self.found("tools/carve.py", "names it"), ["tests/test_paths.py"])
        self.assertEqual(self.found("build_all.py", "names it"), ["tests/test_paths.py"])
        self.assertEqual(self.found("viewer/page.html", "names it"), ["tests/test_joined.py", "tests/test_page.cjs"])
        self.assertEqual(self.found("viewer/other.html", "names it"), ["tests/test_pathlib.py"])
        self.assertEqual(self.found("tools/lonely.py"), ["tests/test_numpy_marker.py"])

    def test_imports_are_followed_and_say_which_module_they_go_through(self):
        self.assertEqual(self.found("tools/helper.py", "names it"), ["tests/test_pytest_marker.py"])
        self.assertEqual(self.found("tools/helper.py", "imports"),
                         [("tests/semantic/test_one.py", "tests/semantic/common.py"), ("tests/test_alpha.py", "tools/alpha.py"),
                          ("tests/test_through_helper.py", "tests/helpers.py")])
        # Two imports deep, and one of two modules on an import line.
        self.assertEqual(self.found("tools/deeper.py", "imports"),
                         [("tests/semantic/test_one.py", "tests/semantic/common.py"), ("tests/test_alpha.py", "tools/alpha.py")])
        self.assertEqual(self.found("tools/deeper.py", "names it"), ["tests/test_import_path.py"])
        # A script the build runs by path is reached through the build.
        self.assertEqual(self.found("tools/carve.py", "imports"), [])
        self.assertEqual(tests_for.Repository(self.root).names["build_all.py"], {"tools/carve.py"})

    def test_a_folder_counts_when_the_test_names_it_and_nothing_in_it(self):
        self.assertEqual(self.found("viewer/page.html", "folder"), ["tests/test_viewer_pages.py"])
        self.assertEqual(self.found("viewer/server.py", "folder"), ["tests/test_viewer_pages.py"])
        # tools/ is named only to import from it, in one line or in two.
        self.assertEqual(self.found("tools/lonely.py", "folder"), [])
        self.assertEqual(self.found("tools/carve.py", "folder"), [])
        # A folder called game or docs under some other root is not the repository's.
        self.assertEqual(self.found("game/manifest.json"), [])
        # Only the files right in the folder, not what lies deeper.
        self.assertEqual(self.found("runtime/shims/net.cpp", "folder"), [])

    def test_markdown_goes_to_the_documentation_tests_ci_runs(self):
        for path in ("docs/topic.md", "GUIDE.md", "docs/i18n/fr/sample.md"):
            self.assertEqual(self.found(path), ["tests/test_docs_sample.py", I18N])
            self.assertEqual(self.found(path, "folder"), ["tests/test_docs_sample.py", I18N])

    def test_a_fixture_goes_to_the_tests_beside_it_that_name_it(self):
        self.assertEqual(self.found("tests/semantic/fixtures/ps_case.asm"), ["tests/semantic/test_one.py"])
        self.assertEqual(self.found("tests/fixtures/sample.bin"), ["tests/test_fixture_user.py"])
        self.assertEqual(self.found("tests/notes.txt"), [])

    def test_a_test_file_is_its_own_test(self):
        entry, = tests_for.survey(self.root, ["tests/test_alpha.py"])
        self.assertEqual((entry["is_test"], entry["tests"], entry["ctest"]), (True, [], []))
        self.assertFalse(tests_for.survey(self.root, ["tests/helpers.py"])[0]["is_test"])

    def test_how_ci_runs_each_test_file(self):
        for test, runner in (("tests/test_alpha.py", "python"), ("tests/test_pathlib.py", "pytest"),
                             ("tests/test_pytest_marker.py", "pytest"), ("tests/test_numpy_marker.py", "pytest"),
                             ("tests/semantic/test_one.py", "semantic"), ("tests/test_page.cjs", "node")):
            self.assertEqual(tests_for.runner_of(self.root, test), runner, test)


class EdgeTest(Checkout):
    def test_a_module_name_goes_to_the_file_beside_the_importer_first(self):
        self.assertEqual(self.found("tools/common.py"), ["tests/test_tools_common.py"])
        self.assertEqual(self.found("tests/semantic/common.py", "names it"), ["tests/semantic/test_one.py"])

    def test_a_test_that_another_test_imports_is_a_module_too(self):
        self.assertEqual(self.found("tests/test_base.py", "names it"), ["tests/test_uses_test.py"])
        # A module is reached by imports only: a test that quotes its bare name does not count, as it would for
        # a fixture.
        self.assertEqual(self.found("tests/helpers.py"), ["tests/test_through_helper.py"])

    def test_a_path_among_other_arguments_and_one_under_a_dot_folder(self):
        self.assertEqual(self.found("scripts/runner.py", "names it"), ["tests/test_argv.py", "tests/test_joined.py"])
        self.assertEqual(self.found(".github/workflows/build.yml"), ["tests/test_ci.py"])

    def test_a_test_that_names_its_own_path_is_not_its_own_test(self):
        self.assertEqual(self.found("tests/test_paths.py"), [])

    def test_tests_are_listed_by_reason_then_by_name(self):
        self.assertEqual(self.found("tools/helper.py"), ["tests/test_pytest_marker.py", "tests/semantic/test_one.py",
                                                         "tests/test_alpha.py", "tests/test_through_helper.py"])

    def test_the_folder_reason_is_for_the_files_right_in_the_folder(self):
        self.assertEqual(self.found(CMAKE, "folder"), ["tests/test_runtime_dir.py"])
        self.assertEqual(self.found("runtime/shims/net.cpp", "folder"), [])

    def test_ctest_entries_are_read_as_declared_and_loops_leave_no_placeholder(self):
        names = [name for name, _, _ in tests_for.Runtime(self.root).tests]
        self.assertEqual(names, ["net", "net_again", "core", "core", "loop_a", "loop_b", "run_modes", "rejects_one",
                                 "rejects_two", "part"])
        # Declared once per platform, listed once.
        self.assertEqual(tests_for.survey(self.root, ["runtime/src/core.cpp"])[0]["ctest"].count("core"), 1)

    def test_a_header_is_looked_up_beside_the_file_that_includes_it(self):
        self.assertEqual(tests_for.survey(self.root, ["runtime/extra/part.h"])[0]["ctest"], ["part"])

    def test_a_changed_test_alone_is_run(self):
        out = self.run_tool("tests/test_docs_sample.py")[1]
        self.assertTrue(out.endswith("Run:\n  python -I tests/test_docs_sample.py\n"), out)

    def test_two_files_of_one_program_name_its_tests_once(self):
        report = json.loads(self.run_tool("--json", "runtime/shims/net.cpp", "runtime/test/net_test.cpp")[1])
        self.assertEqual(report["commands"], ['ctest --test-dir runtime/build -C RelWithDebInfo --output-on-failure '
                                              '-R "^(net|net_again|run_modes|rejects_one|rejects_two)$"'])
        out = self.run_tool("runtime/test/core_test.cpp")[1]
        self.assertIn("  ctest (1): core\n", out)
        self.assertNotIn("no test names it", out)

    def test_cmake_comments_and_nested_parentheses(self):
        text = ('add_executable(x a.cpp   # (was: b.cpp\n  c.cpp)\nadd_test(NAME n COMMAND prog "(x)" tail)\n'
                "if(A AND (B OR C))\n")
        self.assertEqual(tests_for.cmake_commands(text), [
            ("add_executable", ["x", "a.cpp", "c.cpp"]), ("add_test", ["NAME", "n", "COMMAND", "prog", "(x)", "tail"]),
            ("if", ["A", "AND", "B", "OR", "C"])])


class RuntimeTest(Checkout):
    def ctest(self, path):
        return tests_for.survey(self.root, [path])[0]["ctest"]

    def test_a_source_goes_to_the_tests_of_every_program_built_from_it(self):
        self.assertEqual(self.ctest("runtime/test/net_test.cpp"), ["net", "net_again"])
        # In the test's program and, through the set() list, in rt.
        self.assertEqual(self.ctest("runtime/shims/net.cpp"), ["net", "net_again", "run_modes", "rejects_one", "rejects_two"])
        self.assertEqual(self.ctest("runtime/shims/late.cpp"), ["run_modes", "rejects_one", "rejects_two"])
        self.assertEqual(self.ctest("runtime/shims/unused.cpp"), [])
        self.assertEqual(self.ctest("tools/alpha.py"), [])

    def test_loops_are_followed(self):
        self.assertEqual(self.ctest("runtime/test/loop_a_test.cpp"), ["loop_a"])
        self.assertEqual(self.ctest("runtime/test/loop_b_test.cpp"), ["loop_b"])
        self.assertEqual(self.ctest("runtime/shims/audio.cpp"), ["loop_a", "loop_b", "run_modes", "rejects_one", "rejects_two"])

    def test_a_library_goes_to_the_programs_that_link_it(self):
        self.assertEqual(self.ctest("runtime/src/core.cpp"), ["core", "run_modes", "rejects_one", "rejects_two"])

    def test_a_header_goes_to_the_sources_that_include_it(self):
        # types.h <- net.h <- net.cpp, entry.cpp and net_test.cpp
        self.assertEqual(self.ctest("runtime/shims/types.h"), ["net", "net_again", "run_modes", "rejects_one", "rejects_two"])
        self.assertEqual(self.ctest("runtime/src/core.h"), ["core", "run_modes", "rejects_one", "rejects_two"])

    def test_a_script_a_ctest_runs_is_that_test(self):
        self.assertEqual(self.ctest("runtime/test/check_modes.py"), ["run_modes"])

    def test_cmake_text_is_read_as_commands(self):
        text = 'set(A x.cpp "y z.cpp") # add_test(NAME no COMMAND no)\nadd_test(NAME t\n  COMMAND prog --flag)\n'
        self.assertEqual(tests_for.cmake_commands(text), [("set", ["A", "x.cpp", "y z.cpp"]),
                                                          ("add_test", ["NAME", "t", "COMMAND", "prog", "--flag"])])


class CommandsTest(Checkout):
    def test_commands_follow_the_runners(self):
        code, out, err = self.run_tool("tools/helper.py", "viewer/page.html", "runtime/shims/net.cpp", "tests/test_alpha.py")
        self.assertEqual((code, err), (0, ""))
        self.assertIn("tools/helper.py\n  names it (1): tests/test_pytest_marker.py\n"
                      "  imports (3): tests/semantic/test_one.py (through tests/semantic/common.py), "
                      "tests/test_alpha.py (through tools/alpha.py), tests/test_through_helper.py (through tests/helpers.py)\n", out)
        self.assertIn("tests/test_alpha.py  (a test)\n", out)
        self.assertIn("runtime/shims/net.cpp\n  ctest (5): net, net_again, run_modes, rejects_one, rejects_two\n", out)
        self.assertTrue(out.endswith(
            "Run:\n"
            "  python -I tests/test_alpha.py\n"
            "  python -I tests/test_joined.py\n"
            "  python -I tests/test_through_helper.py\n"
            "  python -I tests/test_viewer_pages.py\n"
            "  python -m pytest tests/test_pytest_marker.py   # needs tests/requirements.txt\n"
            "  python -I scripts/run_semantic_tests.py tests/semantic -p test_one.py\n"
            "  node --test tests/test_page.cjs\n"
            '  ctest --test-dir runtime/build -C RelWithDebInfo --output-on-failure -R "^(net|net_again|run_modes|rejects_one|rejects_two)$"\n'),
            out)

    def test_a_file_nothing_names_says_so(self):
        code, out, _ = self.run_tool("runtime/shims/unused.cpp")
        self.assertEqual((code, out), (0, "runtime/shims/unused.cpp\n  no test names it\n\nNothing to run was found for these files.\n"))

    def test_wide_reasons_are_left_out_of_the_commands_when_they_are_many(self):
        report = [{"file": "tools/x.py", "is_test": False, "ctest": [f"t{i}" for i in range(41)],
                   "tests": [{"test": "tests/test_alpha.py", "reason": "names it", "through": None},
                             # Reached a second way: it is among the chosen ones, not one more wide test.
                             {"test": "tests/test_alpha.py", "reason": "imports", "through": "tools/y.py"}]
                   + [{"test": f"tests/semantic/test_w{i:02}.py", "reason": "imports", "through": "tools/y.py"} for i in range(21)]}]
        lines, left_out = tests_for.commands(self.root, report, False)
        self.assertEqual((lines, left_out), (["python -I tests/test_alpha.py",
                                              "ctest --test-dir runtime/build -C RelWithDebInfo --output-on-failure   # 41 of its tests"], 21))
        lines, left_out = tests_for.commands(self.root, report, True)
        self.assertEqual((lines[:2], left_out), (["python -I tests/test_alpha.py",
                                                  "python -I scripts/run_semantic_tests.py tests/semantic   # 21 of its files"], 0))
        # Twenty are still listed one by one, and forty ctest names still make a pattern.
        report[0]["tests"].pop()
        report[0]["ctest"].pop()
        lines, left_out = tests_for.commands(self.root, report, False)
        self.assertEqual((len(lines), left_out), (22, 0))
        self.assertEqual(lines[1], "python -I scripts/run_semantic_tests.py tests/semantic -p test_w00.py")
        self.assertTrue(lines[-1].endswith('-R "^(' + "|".join(f"t{i}" for i in range(40)) + ')$"'))
        self.assertIn("21 more test(s) reach these files", self.text_for(report, 21))

    def text_for(self, report, left_out):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            tests_for.print_report(report, ["x"], left_out)
        return out.getvalue()

    def test_a_folder_argument_stands_for_its_files(self):
        report = json.loads(self.run_tool("--json", "viewer")[1])
        self.assertEqual([entry["file"] for entry in report["files"]], ["viewer/other.html", "viewer/page.html", "viewer/server.py"])
        self.assertEqual(sorted(report), ["commands", "files", "left_out"])
        self.assertIn("node --test tests/test_page.cjs", report["commands"])
        with contextlib.chdir(os.path.join(self.root, "tools")):
            self.assertEqual(json.loads(self.run_tool("--json", "alpha.py")[1])["files"][0]["file"], "tools/alpha.py")
        code, out, err = self.run_tool(os.path.join("..", "elsewhere.py"))
        self.assertEqual((code, out), (2, ""))
        self.assertIn("tests_for: ", err)
        self.assertIn("is not inside the repository", err)


class BranchTest(unittest.TestCase):
    def test_without_paths_the_branchs_files_are_taken(self):
        td = tempfile.TemporaryDirectory(prefix="leonida tests_for git ")
        self.addCleanup(td.cleanup)
        root = os.path.join(td.name, "repo")
        env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
        env.update({"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"})
        patcher = mock.patch.dict(os.environ, env, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)

        def git(*args):
            r = subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                                "-c", "commit.gpgsign=false", "-C", root, *args], capture_output=True, encoding="utf-8")
            self.assertEqual(r.returncode, 0, r.stderr)

        def write(path, text):
            full = os.path.join(root, path)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "w", encoding="utf-8", newline="\n") as f:
                f.write(text)

        for path, text in (("tools/alpha.py", ""), ("tools/beta.py", ""), ("tools/gamma.py", ""),
                           ("tests/test_alpha.py", HEAD + TOOLS + "import alpha\n" + MAIN),
                           ("tests/test_beta.py", HEAD + TOOLS + "import beta\n" + MAIN),
                           ("tests/test_gamma.py", HEAD + TOOLS + "import gamma\nimport delta\n" + MAIN)):
            write(path, text)
        git("init", "-q", "-b", "main")
        git("add", "-A")
        git("commit", "-q", "-m", "start")
        git("checkout", "-q", "-b", "work")
        write("tools/alpha.py", "A = 1\n")
        git("commit", "-q", "-am", "alpha")
        write("tools/beta.py", "B = 1\n")           # modified, not committed
        write("tools/delta.py", "D = 1\n")          # untracked

        def run(*args):
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                code = tests_for.main(["--repo", root, "--base", "main", *args])
            return code, out.getvalue(), err.getvalue()

        report = json.loads(run("--json")[1])
        self.assertEqual([entry["file"] for entry in report["files"]], ["tools/alpha.py", "tools/beta.py", "tools/delta.py"])
        self.assertEqual(report["commands"], ["python -I tests/test_alpha.py", "python -I tests/test_beta.py",
                                              "python -I tests/test_gamma.py"])
        # Another head is compared as committed, without the working tree.
        git("branch", "other", "main")
        self.assertEqual(json.loads(run("--json", "--head", "work")[1])["commands"], ["python -I tests/test_alpha.py"])
        self.assertEqual(run("--head", "other")[1], "No changed files: name the files or folders to look up.\n")
        code, out, err = run("--head", "nowhere")
        self.assertEqual((code, out), (2, ""))
        self.assertTrue(err.startswith("tests_for: git diff"), err)


def real(words):
    """A path of the real repository, written with spaces so that this file does not name it."""
    return words.replace(" ", "/")


class ThisRepositoryTest(unittest.TestCase):
    SCRIPT, THIS = real("scripts tests_for.py"), real("tests test_tests_for.py")

    def test_this_file_names_only_the_script_and_its_fixture(self):
        # If the texts of the synthetic checkout were in this file, their paths (a viewer page, a docs page, the
        # CMake file) would make this test come up for the real files of those names.
        files, folders = tests_for.Repository(ROOT).references(self.THIS, tests_for.read(__file__))
        self.assertEqual((files, folders), ({self.SCRIPT, real("tests fixtures tests_for_checkout.json")}, set()))

    def test_this_test_is_what_the_script_is_tested_by(self):
        entry, = tests_for.survey(ROOT, [self.SCRIPT])
        self.assertEqual([t["test"] for t in entry["tests"] if t["reason"] == "names it"], [self.THIS])
        # The fixture is a file of its own: changing it brings this test up as well.
        entry, = tests_for.survey(ROOT, [real("tests fixtures tests_for_checkout.json")])
        self.assertEqual([t["test"] for t in entry["tests"]], [self.THIS])

    def test_the_runtime_has_programs_and_tests_to_map(self):
        runtime = tests_for.Runtime(ROOT)
        self.assertGreater(len(runtime.tests), 300)
        self.assertIn("ws2_offline", runtime.tests_for(real("runtime shims ws2_offline.cpp")))
        self.assertEqual(runtime.tests_for(real("runtime test loader_test.cpp")), ["loader"])


if __name__ == "__main__":
    unittest.main()
