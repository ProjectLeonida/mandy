"""tools/gpu_results.py on synthetic GPU test records, and on the repository's own (no GPU, no game data).

    python -I tests/test_gpu_results.py

The synthetic records use the layouts the files in gpu-test-results/ use, with invented devices and numbers.
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
sys.path.insert(0, os.path.join(ROOT, "tools"))

import gpu_results  # noqa: E402

FULL = """\
GPU:        NVIDIA GeForce RTX 9990 (Zeta / ZT104, PCI 10de:ffff), subgroup 32..32
Driver:     NVIDIA 700.10 (proprietary, Windows), Vulkan SDK 1.4.400.0 (loader and tools)
Subgroup:   32..32 (the run_gpu_tests.py device line shows subgroup 32..32)
OS:         Windows 11 Pro 10.0.30000 (en-GB)
Commit:     0123456789abcdef0123456789abcdef01234567 (main, 2026-11-24)
Tools:      llvm-mc 23.1.6 (conda llvm-tools with the AMDGPU target), SPIRV-Tools v2026.4,
            CMake 3.31.6 + Ninja, Python 3.13.6
run_gpu_tests.py: 257 shaders: 257 translated, 257 pass spirv-val
            native:                 197/248 passed, 49 skipped, 2 failed
            LEONIDA_W64_EMULATE=1:  229/248 passed, 17 skipped, 2 failed
Failed:     cs_alpha, cs_beta_d16 (exit 1, native and emulated alike). These are the same
            two that fail on another card, where the cause is known;
            cs_gamma and cs_delta pass with the fix applied.
Skipped:    native: 31 "no 64-wide compute subgroup on this device" and 16 delegated cases.
Notes:
- Built with scripts\\build_runtime.ps1 at the commit above.
- AI-assisted. No game files, nothing online.
"""


def record(gpu="AMD Radeon RX 1 (RDNA9 / GFX9999, Navi 99)", driver="RADV (Mesa 30.1.2-arch1), Vulkan 1.5.1",
           commit="aaaaaaa (main)", runs="native: 10/12 passed, 2 skipped, 0 failed", failed="none", extra=""):
    return (f"GPU:        {gpu}\nDriver:     {driver}\nOS:         Arch Linux, kernel 9.9\n"
            f"Commit:     {commit}\nrun_gpu_tests.py: 20 shaders: 20 translated\n            {runs}\n"
            f"Failed:     {failed}\n{extra}")


class RecordsCase(unittest.TestCase):
    def setUp(self):
        td = tempfile.TemporaryDirectory(prefix="leonida gpu_results ")
        self.addCleanup(td.cleanup)
        self.dir = td.name

    def write(self, name, text, folder=None):
        path = os.path.join(folder or self.dir, name)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        return path

    def parse(self, text, name="R.txt"):
        return gpu_results.parse_record(self.write(name, text))

    def run_tool(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = gpu_results.main(list(args))
        return code, out.getvalue(), err.getvalue()


class ParseTest(RecordsCase):
    def test_a_full_record(self):
        r = self.parse(FULL, "RTX9990.txt")
        self.assertEqual((r["file"], r["gpu"], r["family"], r["vendor"]),
                         ("RTX9990.txt", "NVIDIA GeForce RTX 9990", "Zeta", "NVIDIA"))
        self.assertEqual((r["driver"], r["os"]), ("NVIDIA 700.10", "Windows 11 Pro 10.0.30000"))
        self.assertEqual((r["commit"], r["subgroup"]), ("0123456789abcdef0123456789abcdef01234567", [32, 32]))
        self.assertEqual(r["runs"], [
            {"mode": "native", "label": "native", "passed": 197, "total": 248, "skipped": 49, "failed": 2},
            {"mode": "emulated", "label": "LEONIDA_W64_EMULATE=1", "passed": 229, "total": 248, "skipped": 17,
             "failed": 2}])
        # The wrapped sentence that starts with cs_gamma names fixtures that pass.
        self.assertEqual(r["failed"], ["cs_alpha", "cs_beta_d16"])
        self.assertEqual((r["problems"], r["notes"]), ([], []))
        self.assertNotIn("path", r, "a report must not carry the folder it was read from")

    def test_fields_start_in_the_first_column_and_continue_until_the_next(self):
        fields = gpu_results.parse_fields(
            "GPU:  one\n      two\n\n      three\nNot a field, just prose with a colon later: here\n"
            "Driver:\n  indented: not a field\nrun_gpu_tests.py (native):   1/1 passed\n- bullet: no\nOS: last\n")
        self.assertEqual(fields, [
            ("GPU", "one\ntwo\n\nthree\nNot a field, just prose with a colon later: here"),
            ("Driver", "indented: not a field"), ("run_gpu_tests.py (native)", "1/1 passed\n- bullet: no"),
            ("OS", "last")])
        self.assertEqual(gpu_results.parse_fields("prose before any field\n"), [])

    def test_names_are_shortened_without_losing_the_version(self):
        for driver, want in (("RADV (Mesa 30.1.2-arch1), Vulkan 1.5.1", "RADV (Mesa 30.1.2-arch1)"),
                             ("AMD proprietary 26.11.2 (LLPC, driverVersion 2.0.416),\n  Vulkan 1.4", "AMD proprietary 26.11.2"),
                             ("MoltenVK 1.4.2 (Homebrew bottle), Vulkan 1.4.357", "MoltenVK 1.4.2"),
                             # A comma inside the parenthesis does not end the name.
                             ("ANV (Mesa 30.2.7, LLVM 30), Vulkan 1.5", "ANV (Mesa 30.2.7, LLVM 30)"),
                             ("llvmpipe", "llvmpipe")):
            self.assertEqual(self.parse(record(driver=driver))["driver"], want)
        # Only the first line of a field is its name.
        r = self.parse("GPU: AMD W\n     second line of the GPU field\nDriver: d 1\nOS: Linux Mint 30\n"
                       "    kernel 9.1 and more\nCommit: abcdef0\n")
        self.assertEqual((r["gpu"], r["os"]), ("AMD W", "Linux Mint 30"))
        for gpu, name, family, vendor in (
                ("AMD Radeon Graphics, Ryzen 5 0000X iGPU (RDNA9, gfx9999), subgroup 32..64",
                 "AMD Radeon Graphics, Ryzen 5 0000X iGPU", "RDNA9", "AMD"),
                ("Intel Graphics (Meteor Lake-P, Xe-LPG, integrated)", "Intel Graphics", "Meteor Lake-P", "Intel"),
                ("Apple M9 (Apple GPU, 12 cores)", "Apple M9", "Apple GPU", "Apple"),
                ("GeForce GTX 1, subgroup 32", "GeForce GTX 1", "", "NVIDIA"),
                ("none - Mesa llvmpipe (LLVM 19.1.7)", "none - Mesa llvmpipe", "LLVM 19.1.7", "software"),
                ("Imagination BXM-8-256", "Imagination BXM-8-256", "", "other")):
            r = self.parse(record(gpu=gpu))
            self.assertEqual((r["gpu"], r["family"], r["vendor"]), (name, family, vendor))
        self.assertEqual(self.parse(record(gpu="AMD X (RDNA9), subgroup 32..64"))["subgroup"], [32, 64])
        self.assertIsNone(self.parse(record())["subgroup"])

    def test_runs_are_labelled_by_what_stands_before_or_after_the_counts(self):
        def runs(text):
            return [(run["mode"], run["passed"], run["total"], run["skipped"], run["failed"])
                    for run in self.parse(record(runs=text))["runs"]]

        self.assertEqual(runs("228/248 passed, 16 skipped, 4 failed"), [("native", 228, 248, 16, 4)])
        self.assertEqual(runs("183/248 passed, 61 skipped, 4 failed\n            214/248 passed, 30 skipped, 4 failed "
                              "(LEONIDA_W64_EMULATE=1)"),
                         [("native", 183, 248, 61, 4), ("emulated", 214, 248, 30, 4)])
        self.assertEqual(runs("218/235 passed, 16 skipped, 1 failed (native and LEONIDA_W64_EMULATE=1 alike)"),
                         [("native", 218, 235, 16, 1), ("emulated", 218, 235, 16, 1)])
        self.assertEqual(runs("Native:                198/249 passed, 49 skipped, 2 failed (exit 1).\n"
                              "            LEONIDA_W64_EMULATE=1: 97/97 passed"),
                         [("native", 198, 249, 49, 2), ("emulated", 97, 97, None, None)])
        self.assertEqual(runs("81/97 passed, 16 skipped, 0 failed (exit 0)"), [("native", 81, 97, 16, 0)])
        # A run of another kind keeps its own label and is neither the native nor the emulated run.
        self.assertEqual(runs("Native + validation: 202/263 passed, 53 skipped, 8 failed\n"
                              "            LEONIDA_W64_EMULATE=1 --filter cs_ (as CI):    93/121 passed, 28 skipped, 0 failed"),
                         [("Native + validation", 202, 263, 53, 8),
                          ("LEONIDA_W64_EMULATE=1 --filter cs_ (as CI)", 93, 121, 28, 0)])
        self.assertEqual(runs("257 shaders: 257 translated, 257 pass spirv-val"), [])

    def test_a_run_can_be_named_in_the_field_itself(self):
        r = self.parse("GPU: AMD Y\nDriver: d 1\nOS: o\nCommit: abcdef0\n"
                       "run_gpu_tests.py (native):                 258/258 translated and pass spirv-val; "
                       "231/249 passed, 16 skipped, 2 failed\n"
                       "run_gpu_tests.py (LEONIDA_W64_EMULATE=1):  258/258 translated; 230/249 passed, 16 skipped, 3 failed\n")
        self.assertEqual([(run["mode"], run["passed"], run["failed"]) for run in r["runs"]],
                         [("native", 231, 2), ("emulated", 230, 3)])

    def test_failed_fixtures_are_the_ones_the_field_opens_with_or_gives_a_line_of_their_own(self):
        def failed(text):
            return self.parse(record(failed=text))["failed"]

        for nothing in ("none", "None.", "none, in either run. cs_alpha and cs_beta pass on this commit.", "",
                        "none. What the earlier record listed:\n            cs_alpha: passes since the fix"):
            self.assertEqual(failed(nothing), [])
        self.assertEqual(failed("cs_alpha and cs_beta, in every run: \"64 bad\", the expected 71 plus 2000."), ["cs_alpha", "cs_beta"])
        self.assertEqual(failed("ms_clip, ms_wave_s3: the harness exits without a result. The other four\n"
                                "            (ms_tri, ms_pack) pass.\n"
                                "            ms_v5 and ms_ballot) pass, and the extension is enabled."), ["ms_clip", "ms_wave_s3"])
        self.assertEqual(failed("cs_one: got 8192, want 7682\n            cs_two: got 8224\n"
                                "            The fixtures of another fix, not on main yet.\n"
                                "            cs_three (both modes): exit 1.\n            ps_four and vs_five: flake\n"
                                "            cs_one: listed twice"),
                         ["cs_one", "cs_two", "cs_three", "ps_four", "vs_five"])
        self.assertEqual(failed("cs_ext in all four profiles:\n            6 of 64 words differ"), ["cs_ext"])
        self.assertEqual(failed("- gs_tri: wrong winding"), ["gs_tri"])
        # Not a fixture name: no stage prefix, or prose first.
        self.assertEqual(failed("two fixtures, cs_alpha and cs_beta"), [])
        self.assertEqual(failed("d3d12_execute_indirect_validation: 5 cases"), [])

    def test_what_a_record_lacks_is_a_problem_and_what_could_not_be_read_a_note(self):
        r = self.parse("GPU: AMD Z\nOS: o\nCommit: the tip of main\nNotes:\n- nothing else\n")
        self.assertEqual(r["problems"], ["no Driver field", "the Commit field has no commit hash"])
        self.assertEqual(r["notes"], ["no run_gpu_tests.py line with `N/M passed`"])
        self.assertEqual((r["commit"], r["runs"], r["failed"]), (None, [], []))
        r = self.parse(record(runs="native: 1/9 passed, 0 skipped, 8 failed", failed="cs_only: bad"))
        self.assertEqual((r["problems"], r["notes"]), ([], ["8 failed in a run, 1 fixture name(s) read from Failed"]))
        self.assertEqual(self.parse("")["problems"], [f"no {key} field" for key in ("GPU", "Driver", "OS", "Commit")])
        # A field that is there but empty is as good as missing.
        self.assertEqual(self.parse("GPU: AMD Z\nDriver:\nOS: o\nCommit: abcdef0\n")["problems"], ["no Driver field"])

    def test_the_first_field_of_a_name_is_the_records(self):
        # A record can quote another one further down; the header block comes first.
        r = self.parse(record(gpu="AMD First (RDNA9)", commit="abcdef0")
                       + "Notes:\n- compared with\nGPU: NVIDIA Second (Zeta)\nCommit: 1234567\n")
        self.assertEqual((r["gpu"], r["vendor"], r["commit"]), ("AMD First", "AMD", "abcdef0"))


class OverviewTest(RecordsCase):
    def three(self):
        self.write("A.txt", record(gpu="NVIDIA A (Zeta)", commit="1111111", failed="cs_x, cs_a: bad",
                                   runs="native: 8/12 passed, 2 skipped, 2 failed"))
        self.write("B.txt", record(gpu="AMD B (RDNA9)", commit="2222222", failed="cs_x: bad",
                                   runs="native: 9/12 passed, 2 skipped, 1 failed"))
        self.write("C.txt", record(gpu="NVIDIA C (Zeta)", commit="3333333"))
        self.write("README.md", "# not a record\n")
        return [gpu_results.parse_record(p) for p in gpu_results.find_records([self.dir])]

    def test_fixtures_are_listed_with_the_records_and_vendors_they_fail_on(self):
        report = gpu_results.overview(self.three(), None)
        self.assertEqual([r["file"] for r in report["records"]], ["A.txt", "B.txt", "C.txt"])
        self.assertEqual(report["vendors"], {"NVIDIA": 2, "AMD": 1})
        # The fixture that fails most often comes first, not the first of the alphabet.
        self.assertEqual(report["fixtures"], [
            {"fixture": "cs_x", "records": ["A.txt", "B.txt"], "vendors": {"NVIDIA": 1, "AMD": 1},
             "last_behind": None, "last_record": None, "newer_without": None},
            {"fixture": "cs_a", "records": ["A.txt"], "vendors": {"NVIDIA": 1},
             "last_behind": None, "last_record": None, "newer_without": None}])

    def test_newer_records_that_no_longer_list_a_fixture_are_counted(self):
        records = self.three()
        for r, behind in zip(records, (40, 12, 3)):
            r["behind"] = behind
        records.append({**records[2], "file": "D.txt", "behind": 0, "runs": []})   # no run: says nothing
        records.append({**records[2], "file": "E.txt", "behind": None})            # commit unknown here
        records.append({**records[2], "file": "F.txt", "behind": 12})              # as old as B: not newer
        by = {e["fixture"]: e for e in gpu_results.fixtures(records)}
        self.assertEqual((by["cs_x"]["last_record"], by["cs_x"]["last_behind"], by["cs_x"]["newer_without"]),
                         ("B.txt", 12, 1))
        self.assertEqual((by["cs_a"]["last_record"], by["cs_a"]["last_behind"], by["cs_a"]["newer_without"]),
                         ("A.txt", 40, 3))

    def test_commit_distances_come_from_git(self):
        repo = os.path.join(self.dir, "repo")
        os.mkdir(repo)
        env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
        env.update({"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"})

        def git(*args):
            r = subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                                "-c", "commit.gpgsign=false", "-C", repo, *args], capture_output=True, encoding="utf-8",
                               env=env)
            self.assertEqual(r.returncode, 0, r.stderr)
            return r.stdout.strip()

        git("init", "-q", "-b", "main")
        shas = []
        for n in range(4):
            git("commit", "-q", "--allow-empty", "-m", f"commit {n}")
            shas.append(git("rev-parse", "HEAD"))
        records = [{"commit": shas[0], "behind": None}, {"commit": shas[2][:9], "behind": None},
                   {"commit": shas[3], "behind": None}, {"commit": "fffffff", "behind": None},
                   {"commit": None, "behind": None}]
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertTrue(gpu_results.commits_behind(records, "main", repo=repo))
            self.assertEqual([r["behind"] for r in records], [3, 1, 0, None, None])
            self.assertFalse(gpu_results.commits_behind(records, "no-such-branch", repo=repo))
            self.assertFalse(gpu_results.commits_behind(records, "main", repo=self.dir))


class CommandLineTest(RecordsCase):
    def setUp(self):
        super().setUp()
        self.write("RTX9990.txt", FULL)
        self.write("Vega.txt", record(failed="cs_alpha: bad", runs="native: 11/12 passed, 0 skipped, 1 failed"))
        self.write("notes.md", "GPU: not a record\n")

    def test_text_report(self):
        code, out, err = self.run_tool(self.dir, "--no-git")
        self.assertEqual((code, err), (0, ""))
        self.assertIn("2 record(s): NVIDIA 1, AMD 1\n", out)
        self.assertNotIn("behind:", out)
        lines = out.splitlines()
        header = next(i for i, line in enumerate(lines) if line.startswith("record"))
        self.assertEqual(lines[header].split(), ["record", "GPU", "driver", "behind", "native", "emulated", "failed"])
        # Rows go by vendor, then GPU: AMD before NVIDIA, whatever the files are called. Columns line up.
        self.assertRegex(lines[header + 1], r"^Vega\.txt +AMD Radeon RX 1 \(RDNA9\) +RADV \(Mesa 30\.1\.2-arch1\) +\? +"
                                            r"11/12 0s 1f +- +cs_alpha$")
        self.assertRegex(lines[header + 2], r"^RTX9990\.txt +NVIDIA GeForce RTX 9990 \(Zeta\) +NVIDIA 700\.10 +\? +"
                                            r"197/248 49s 2f +229/248 17s 2f +cs_alpha, cs_beta_d16$")
        self.assertEqual(lines[header + 1].index("11/12"), lines[header].index("native"))
        self.assertIn("\nFixtures that fail\n  cs_alpha: 2 record(s), NVIDIA 1, AMD 1\n      RTX9990.txt, Vega.txt\n"
                      "  cs_beta_d16: 1 record(s), NVIDIA 1\n      RTX9990.txt\n", out)
        self.assertNotIn("Other runs", out)
        self.assertNotIn("Not read in full", out)
        self.assertNotIn(self.dir, out)

    def test_other_runs_and_unread_records_are_named(self):
        self.write("C.txt", record(runs="Native + validation: 5/9 passed, 1 skipped, 3 failed", failed="cs_q: bad"))
        self.write("D.txt", "GPU: Intel D\nDriver: ANV 1\nOS: o\nCommit: abcdef1\n")
        out = self.run_tool(self.dir, "--no-git")[1]
        self.assertIn("\nOther runs\n  C.txt: Native + validation: 5/9 1s 3f\n", out)
        self.assertIn("\nNot read in full\n  C.txt: 3 failed in a run, 1 fixture name(s) read from Failed\n"
                      "  D.txt: no run_gpu_tests.py line with `N/M passed`\n", out)

    def test_markdown_table(self):
        self.write("P.txt", record(gpu="AMD P|Q (RDNA9)", failed="none"))
        code, out, _ = self.run_tool(self.dir, "--markdown", "--no-git", "--folder-url", "records/")
        self.assertEqual(code, 0)
        lines = out.splitlines()
        self.assertEqual(lines[:2], ["| Record | GPU | Driver | OS | Commit | Native | Emulated | Failed |",
                                     "|---|---|---|---|---|---|---|---|"])
        self.assertIn("| [P.txt](records/P.txt) | AMD P\\|Q (RDNA9) | RADV (Mesa 30.1.2-arch1) | Arch Linux | `aaaaaaa` | "
                      "10/12 2s 0f | - | none |", lines)
        self.assertIn("| [RTX9990.txt](records/RTX9990.txt) | NVIDIA GeForce RTX 9990 (Zeta) | NVIDIA 700.10 | "
                      "Windows 11 Pro 10.0.30000 | `012345678` | 197/248 49s 2f | 229/248 17s 2f | `cs_alpha`, `cs_beta_d16` |", lines)
        self.assertEqual(len(lines), 5)
        self.assertIn("](../gpu-test-results/P.txt)", self.run_tool(self.dir, "--markdown", "--no-git")[1])

    def test_json_is_the_whole_report_without_local_paths(self):
        code, out, _ = self.run_tool(self.dir, "--json", "--no-git")
        report = json.loads(out)
        self.assertEqual((code, sorted(report), report["main"]), (0, ["fixtures", "main", "records", "vendors"], None))
        self.assertEqual([r["file"] for r in report["records"]], ["RTX9990.txt", "Vega.txt"])
        self.assertEqual(report["fixtures"][0]["fixture"], "cs_alpha")
        self.assertNotIn(os.path.basename(self.dir), out)

    def test_check_names_the_records_to_fix_and_exits_1(self):
        code, out, _ = self.run_tool(self.dir, "--check")
        self.assertEqual((code, out), (0, "2 record(s) checked, 0 to fix\n"))
        bad = self.write("Bad.txt", "GPU: AMD Bad\nDriver: d\nCommit: soon\nrun_gpu_tests.py: 1/1 passed\n")
        self.write("Thin.txt", "GPU: AMD Thin\nDriver: d\nOS: o\nCommit: abcdef1\n")
        code, out, _ = self.run_tool(self.dir, "--check")
        self.assertEqual(code, 1)
        self.assertEqual(out, "Bad.txt: no OS field\nBad.txt: the Commit field has no commit hash\n"
                              "Thin.txt: note: no run_gpu_tests.py line with `N/M passed`\n4 record(s) checked, 1 to fix\n")
        # One file can be checked alone, as for a pull request that adds it.
        self.assertEqual(self.run_tool(bad, "--check")[0], 1)
        self.assertEqual(self.run_tool(os.path.join(self.dir, "Vega.txt"), "--check"), (0, "1 record(s) checked, 0 to fix\n", ""))

    def test_a_missing_target_is_a_usage_error(self):
        code, out, err = self.run_tool(os.path.join(self.dir, "nowhere"))
        self.assertEqual((code, out), (2, ""))
        self.assertIn("gpu_results: ", err)
        self.assertIn("no such record or folder", err)
        with self.assertRaises(SystemExit) as stop, contextlib.redirect_stderr(io.StringIO()):
            gpu_results.main([self.dir, "--json", "--check"])
        self.assertEqual(stop.exception.code, 2)


class RepositoryRecordsTest(unittest.TestCase):
    def test_every_record_in_the_repository_has_the_required_fields(self):
        paths = gpu_results.find_records([gpu_results.DEFAULT_FOLDER])
        self.assertGreater(len(paths), 40)
        for path in paths:
            record_ = gpu_results.parse_record(path)
            with self.subTest(record=record_["file"]):
                self.assertEqual(record_["problems"], [], "gpu-test-results/README.md has the fields a record needs")
                self.assertTrue(record_["gpu"] and record_["driver"] and record_["os"] and record_["commit"])

    def test_the_default_folder_is_the_repositorys(self):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = gpu_results.main(["--check"])
        self.assertEqual((code, err.getvalue()), (0, ""))
        self.assertRegex(out.getvalue(), r"\d+ record\(s\) checked, 0 to fix\n$")


if __name__ == "__main__":
    unittest.main()
