"""scripts/pr_overlap.py on throwaway git repositories (no network, no game data).

    python -I tests/test_pr_overlap.py

An "upstream" repository gets pull request refs the way GitHub lays them out: refs/pull/N/head for every pull request
and refs/pull/N/merge only while it is open. The script runs in a clone of it, where a branch plays the contributor's
own work.
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

import pr_overlap  # noqa: E402

PAGE = "".join(f"line {i}\n" for i in range(1, 13))


def page(**changed):
    """PAGE with some lines replaced: page(line5="mine") changes the fifth line."""
    lines = PAGE.splitlines(keepends=True)
    for key, text in changed.items():
        lines[int(key[4:]) - 1] = text + "\n"
    return "".join(lines)


class PrOverlapTest(unittest.TestCase):
    def setUp(self):
        td = tempfile.TemporaryDirectory(prefix="leonida pr_overlap ")
        self.addCleanup(td.cleanup)
        self.root = td.name
        env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
        env.update({"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1", "XDG_CONFIG_HOME": td.name})
        patcher = mock.patch.dict(os.environ, env, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.upstream = os.path.join(self.root, "upstream")
        self.clone = os.path.join(self.root, "clone")
        os.mkdir(self.upstream)
        self.git(self.upstream, "init", "-q", "-b", "main")
        for path, text in (("viewer/index.html", PAGE), ("viewer/studio.html", PAGE), ("tools/a.py", "a = 1\n"),
                           ("docs/world.md", "# World\n")):
            self.write(self.upstream, path, text)
        self.commit(self.upstream, "start")
        self.git(self.root, "clone", "-q", self.upstream, self.clone)

    def git(self, repo, *args):
        r = subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                            "-c", "commit.gpgsign=false", "-c", "protocol.file.allow=always", "-C", repo, *args],
                           capture_output=True, encoding="utf-8")
        self.assertEqual(r.returncode, 0, r.stderr)
        return r.stdout.strip()

    def write(self, repo, path, text):
        full = os.path.join(repo, path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)

    def commit(self, repo, message):
        self.git(repo, "add", "-A")
        self.git(repo, "commit", "-q", "-m", message)
        return self.git(repo, "rev-parse", "HEAD")

    def pull(self, number, changes, subject=None, is_open=True, base="main"):
        """A pull request in upstream: a commit on `base` that writes `changes`, with the refs GitHub would keep."""
        self.git(self.upstream, "checkout", "-q", "--detach", base)
        for path, text in changes.items():
            self.write(self.upstream, path, text)
        sha = self.commit(self.upstream, subject or f"change for pull request {number}")
        self.git(self.upstream, "update-ref", f"refs/pull/{number}/head", sha)
        if is_open:
            # GitHub's test merge is a commit of its own. Which one does not matter here, only that it is not the
            # head: the script must take a pull request's commit from the head ref.
            self.git(self.upstream, "update-ref", f"refs/pull/{number}/merge",
                     self.git(self.upstream, "rev-parse", "main"))
        self.git(self.upstream, "checkout", "-q", "main")
        return sha

    def close(self, number):
        self.git(self.upstream, "update-ref", "-d", f"refs/pull/{number}/merge")

    def run_tool(self, *args, repo=None):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = pr_overlap.main(["--repo", repo or self.clone, *args])
        return code, out.getvalue(), err.getvalue()

    def report(self, *args, **kwargs):
        code, out, err = self.run_tool("--json", *args, **kwargs)
        self.assertEqual(code, 0, err)
        return json.loads(out)

    def numbers(self, *args, **kwargs):
        return [pr["number"] for pr in self.report(*args, **kwargs)["pull_requests"]]

    def kept_refs(self):
        out = self.git(self.clone, "for-each-ref", "--format=%(refname) %(objectname)", "refs/pr-overlap/")
        return dict(line.split(" ") for line in out.splitlines())

    def four_pull_requests(self):
        return {1: self.pull(1, {"viewer/index.html": page(line5="pull request 1")}, "viewer: one"),
                2: self.pull(2, {"tools/a.py": "a = 2\n", "viewer2/x.txt": "x\n"}, "tools: two"),
                3: self.pull(3, {"viewer/index.html": page(line2="closed")}, "viewer: three", is_open=False),
                4: self.pull(4, {"viewer/new.js": "new\n", "docs/world.md": "# World\n\nMore.\n"}, "viewer: four")}


class PathsTest(PrOverlapTest):
    def test_paths_find_the_open_pull_requests_that_change_them(self):
        self.four_pull_requests()
        report = self.report("viewer/index.html")
        self.assertEqual((report["mode"], report["open"], report["paths"]), ("paths", 3, ["viewer/index.html"]))
        # Pull request 3 changes the file too, but it is closed: it has no merge ref.
        self.assertEqual([(pr["number"], pr["files"], pr["changed_files"]) for pr in report["pull_requests"]],
                         [(1, ["viewer/index.html"], 1)])
        self.assertNotIn("merge", report["pull_requests"][0])
        # A folder takes everything below it, and only that: viewer2/x.txt is not in viewer.
        self.assertEqual(self.numbers("viewer"), [1, 4])
        self.assertEqual(self.report("viewer")["pull_requests"][1]["files"], ["viewer/new.js"])
        self.assertEqual(self.numbers("viewer/"), [1, 4])
        self.assertEqual(self.numbers("viewer2"), [2])
        self.assertEqual(self.numbers("docs/world.md", "tools/a.py"), [2, 4])
        self.assertEqual(self.numbers("viewer/studio.html"), [])
        self.assertEqual(self.numbers("."), [1, 2, 4])

    def test_a_pull_request_is_shown_by_its_head_commit(self):
        sha = self.pull(1, {"viewer/index.html": page(line5="x")}, "viewer: a subject (SOME-ID)")
        pr, = self.report("viewer")["pull_requests"]
        self.assertEqual((pr["head"], pr["author"], pr["subject"]), (sha, "Test", "viewer: a subject (SOME-ID)"))
        self.assertRegex(pr["date"], r"^\d{4}-\d\d-\d\d$")
        code, out, _ = self.run_tool("viewer")
        self.assertEqual(code, 0)
        self.assertIn(f"#1     {pr['date']}  Test: viewer: a subject (SOME-ID)\n         viewer/index.html\n", out)
        self.assertIn("1 open pull request(s) change these paths.", out)
        # The report above fetched the head already.
        self.assertIn("1 open pull request(s) on origin (0 head(s) fetched, 0 no longer open)", out)

    def test_what_main_changed_since_is_not_the_pull_requests(self):
        self.pull(1, {"tools/a.py": "a = 2\n"})
        self.write(self.upstream, "docs/world.md", "# World\n\nChanged on main after the pull request branched.\n")
        self.commit(self.upstream, "main moves on")
        self.git(self.clone, "fetch", "-q", "origin")
        # A two-dot diff of main against the old branch would list docs/world.md as well.
        self.assertEqual(self.numbers("docs/world.md"), [])
        self.assertEqual(self.report("tools")["pull_requests"][0]["files"], ["tools/a.py"])

    def test_a_rename_counts_for_the_old_and_the_new_path(self):
        self.git(self.upstream, "checkout", "-q", "--detach", "main")
        self.git(self.upstream, "mv", "viewer/studio.html", "viewer/inspector.html")
        sha = self.commit(self.upstream, "rename the studio page")
        for kind in ("head", "merge"):
            self.git(self.upstream, "update-ref", f"refs/pull/1/{kind}", sha)
        self.git(self.upstream, "checkout", "-q", "main")
        self.assertEqual(self.numbers("viewer/studio.html"), [1])
        self.assertEqual(self.numbers("viewer/inspector.html"), [1])
        self.assertEqual(self.report("viewer")["pull_requests"][0]["files"],
                         ["viewer/inspector.html", "viewer/studio.html"])

    def test_paths_are_relative_to_the_current_folder_inside_the_repository(self):
        odd = "docs/with space/ünï.md"
        self.four_pull_requests()
        self.pull(5, {odd: "text\n"})
        with contextlib.chdir(os.path.join(self.clone, "viewer")):
            self.assertEqual(self.report("index.html")["paths"], ["viewer/index.html"])
            self.assertEqual(self.numbers("index.html"), [1])
            self.assertEqual(self.numbers(".."), [1, 2, 4, 5])
            self.assertEqual(self.numbers(os.path.join("..", "tools")), [2])
        # From outside the repository a path is relative to its root; names need not be ASCII or free of spaces.
        self.assertEqual(self.report(odd)["pull_requests"][0]["files"], [odd])
        self.assertIn(odd, self.run_tool("docs")[1])
        code, _, err = self.run_tool(os.path.join("..", "elsewhere"))
        self.assertEqual(code, 2)
        self.assertIn("is not inside the repository", err)

    def test_a_pull_request_without_common_history_is_counted_not_listed(self):
        self.pull(1, {"viewer/index.html": page(line5="x")})
        self.git(self.upstream, "checkout", "-q", "--orphan", "island")
        sha = self.commit(self.upstream, "another history")
        self.git(self.upstream, "update-ref", "refs/pull/2/head", sha)
        self.git(self.upstream, "update-ref", "refs/pull/2/merge", sha)
        self.git(self.upstream, "checkout", "-q", "main")
        report = self.report("viewer")
        self.assertEqual((report["open"], report["unrelated"], [pr["number"] for pr in report["pull_requests"]]),
                         (2, 1, [1]))
        self.assertIn("1 pull request(s) share no history with origin/main and were skipped",
                      self.run_tool("viewer")[1])


    def test_a_shallow_clone_is_told_why_nothing_was_found(self):
        self.pull(1, {"viewer/index.html": page(line5="x")})
        self.write(self.upstream, "tools/a.py", "a = 3\n")
        self.commit(self.upstream, "main moves on")
        shallow = os.path.join(self.root, "shallow")
        self.git(self.root, "clone", "-q", "--depth", "1", "file://" + self.upstream.replace(os.sep, "/"), shallow)
        report = self.report("viewer", repo=shallow)
        self.assertEqual((report["open"], report["unrelated"], report["shallow"], report["pull_requests"]),
                         (1, 1, True, []))
        self.assertIn("1 pull request(s) share no history with origin/main and were skipped: this clone is shallow, "
                      "and `git fetch --unshallow` brings the history a merge base needs",
                      self.run_tool("viewer", repo=shallow)[1])
        # In a full clone the same pull request is found, and nothing is said about shallowness.
        report = self.report("viewer")
        self.assertEqual((report["unrelated"], report["shallow"], [pr["number"] for pr in report["pull_requests"]]),
                         (0, False, [1]))


class BranchTest(PrOverlapTest):
    def work(self):
        """The contributor's branch: two commits, a modified file and an untracked one."""
        self.git(self.clone, "checkout", "-q", "-b", "work")
        self.write(self.clone, "viewer/index.html", page(line5="mine"))
        self.commit(self.clone, "viewer: my change")
        self.write(self.clone, "tests/new_test.py", "mine = True\n")
        head = self.commit(self.clone, "tests: my test")
        self.write(self.clone, "docs/world.md", "# World\n\nNot committed yet.\n")
        self.write(self.clone, "notes/todo.txt", "untracked\n")
        # Edited again since its commit: a file of the branch, not one that is only in the working tree.
        self.write(self.clone, "tests/new_test.py", "mine = True  # and more\n")
        return head

    def test_the_branchs_files_and_how_it_merges_with_each_pull_request(self):
        head = self.work()
        self.pull(1, {"viewer/index.html": page(line5="theirs")}, "same line")
        self.pull(2, {"viewer/index.html": page(line11="theirs")}, "another line")
        self.pull(3, {"docs/world.md": "# World\n\nTheirs.\n"}, "the file modified in the working tree")
        self.pull(4, {"tests/new_test.py": "theirs = True\n"}, "the same new file")
        self.pull(5, {"notes/todo.txt": "theirs\n"}, "the untracked file")
        self.pull(6, {"tools/a.py": "a = 6\n"}, "a file the branch leaves alone")
        self.pull(7, {"viewer/index.html": page(line5="closed")}, "closed", is_open=False)
        report = self.report()
        self.assertEqual((report["mode"], report["head"], report["head_sha"], report["open"]),
                         ("branch", "HEAD", head, 6))
        self.assertEqual(report["files"], ["docs/world.md", "notes/todo.txt", "tests/new_test.py", "viewer/index.html"])
        self.assertEqual(report["uncommitted"], ["docs/world.md", "notes/todo.txt"])
        # Conflicts first, then by number. The merge is of commits: the working tree's files overlap but cannot
        # conflict.
        self.assertEqual([(pr["number"], pr["merge"], pr["conflicts"]) for pr in report["pull_requests"]],
                         [(1, "conflict", ["viewer/index.html"]), (4, "conflict", ["tests/new_test.py"]),
                          (2, "clean", []), (3, "clean", []), (5, "clean", [])])
        code, out, _ = self.run_tool()
        self.assertEqual(code, 0)
        self.assertIn("4 file(s) changed by HEAD against origin/main, 2 of them only in the working tree: "
                      "docs/world.md, notes/todo.txt, tests/new_test.py, viewer/index.html", out)
        self.assertRegex(out, r"#1     conflict \d{4}-\d\d-\d\d  Test: same line\n         viewer/index.html\n"
                              r"         conflicts in: viewer/index.html\n")
        self.assertIn("5 other open pull request(s) change these files; 2 conflict with this branch: #1, #4; "
                      "3 merge cleanly with it: #2, #3, #5.", out)
        self.assertIn("The merge check covers committed work only", out)

    def test_another_head_is_compared_without_the_working_tree(self):
        self.work()
        self.pull(1, {"viewer/index.html": page(line5="theirs")})
        self.pull(3, {"docs/world.md": "# World\n\nTheirs.\n"})
        self.git(self.clone, "branch", "other", "HEAD~1")
        report = self.report("--head", "other")
        self.assertEqual((report["files"], report["uncommitted"]), (["viewer/index.html"], []))
        self.assertEqual([(pr["number"], pr["merge"]) for pr in report["pull_requests"]], [(1, "conflict")])
        self.assertNotIn("committed work only", self.run_tool("--head", "other")[1])

    def test_the_branchs_own_pull_request_and_one_it_is_built_on(self):
        self.work()
        self.git(self.clone, "stash", "-q", "--include-untracked")
        self.git(self.clone, "push", "-q", "origin", "HEAD:refs/pull/8/head", "HEAD:refs/pull/8/merge")
        self.pull(2, {"viewer/index.html": page(line11="theirs")})
        def standing():
            return [(pr["number"], pr["merge"]) for pr in self.report()["pull_requests"]]

        self.assertEqual(standing(), [(2, "clean"), (8, "same")])
        out = self.run_tool()[1]
        self.assertIn("1 other open pull request(s) change these files; 1 merge cleanly with it: #2.", out)
        self.assertIn("#8 is this branch's own pull request.", out)
        # A commit that is not pushed yet: the pull request's head is now behind the branch.
        self.write(self.clone, "tools/a.py", "a = 8\n")
        self.commit(self.clone, "one more")
        self.assertEqual(standing(), [(2, "clean"), (8, "included")])
        self.assertIn("#8 is already in this branch", self.run_tool()[1])

    def test_a_branch_without_changes_has_nothing_to_compare(self):
        self.pull(1, {"viewer/index.html": page(line5="theirs")})
        report = self.report()
        self.assertEqual((report["files"], report["pull_requests"]), ([], []))
        out = self.run_tool()[1]
        self.assertIn("0 file(s) changed by HEAD against origin/main", out)
        self.assertIn("Name the files or folders you plan to change", out)

    def test_a_pull_request_that_is_on_main_already_changes_nothing(self):
        # GitHub removes the merge ref of a merged pull request; one that still has it must not be listed either,
        # since its head is its own merge base with main.
        sha = self.pull(1, {"viewer/index.html": page(line5="merged")})
        self.git(self.upstream, "merge", "-q", "--no-ff", "-m", "Merge pull request #1", sha)
        self.git(self.clone, "pull", "-q", "origin", "main")
        self.git(self.clone, "checkout", "-q", "-b", "work")
        self.write(self.clone, "viewer/index.html", page(line5="merged", line9="mine"))
        self.commit(self.clone, "mine")
        self.assertEqual(self.report()["open"], 1)
        self.assertEqual(self.numbers(), [])
        self.assertEqual(self.numbers("viewer"), [])


class RefsTest(PrOverlapTest):
    def test_the_kept_refs_are_the_open_pull_requests_of_the_last_run(self):
        shas = self.four_pull_requests()
        self.git(self.upstream, "tag", "v1", shas[1])
        report = self.report("viewer")
        self.assertEqual((report["open"], report["fetched"], report["removed"]), (3, 3, 0))
        self.assertEqual(self.kept_refs(), {f"refs/pr-overlap/{n}": shas[n] for n in (1, 2, 4)})
        # Nothing else is touched: no tags, no FETCH_HEAD.
        self.assertEqual(self.git(self.clone, "tag"), "")
        self.assertFalse(os.path.exists(os.path.join(self.clone, ".git", "FETCH_HEAD")))

        # Pull request 1 closes, 2 gets a new commit, 9 is opened; 4 stays as it is.
        self.close(1)
        moved = self.pull(2, {"tools/a.py": "a = 22\n", "viewer/studio.html": page(line3="two")},
                          base="refs/pull/2/head")
        opened = self.pull(9, {"viewer/index.html": page(line7="nine")})
        report = self.report("viewer")
        self.assertEqual((report["open"], report["fetched"], report["removed"]), (3, 2, 1))
        self.assertEqual(self.kept_refs(), {"refs/pr-overlap/2": moved, "refs/pr-overlap/4": shas[4],
                                            "refs/pr-overlap/9": opened})
        self.assertEqual([pr["number"] for pr in report["pull_requests"]], [2, 4, 9])
        report = self.report("viewer")
        self.assertEqual((report["fetched"], report["removed"]), (0, 0))

    def test_a_repository_whose_pull_requests_are_all_closed_is_not_taken_for_a_fork(self):
        self.pull(1, {"viewer/index.html": page(line5="closed")}, is_open=False)
        code, out, err = self.run_tool("viewer")
        self.assertEqual((code, err), (0, ""))
        self.assertIn("0 open pull request(s) on origin (0 head(s) fetched, 0 no longer open)", out)
        self.assertEqual(self.kept_refs(), {})

    def test_no_fetch_works_from_the_kept_refs_without_the_remote(self):
        self.four_pull_requests()
        code, _, err = self.run_tool("--no-fetch", "viewer")
        self.assertEqual(code, 2)
        self.assertIn("run once without --no-fetch", err)
        self.assertEqual(self.numbers("viewer"), [1, 4])
        self.git(self.clone, "remote", "set-url", "origin", os.path.join(self.root, "gone"))
        report = self.report("--no-fetch", "viewer")
        self.assertEqual((report["open"], report["fetched"], report["removed"]), (3, None, None))
        self.assertEqual([pr["number"] for pr in report["pull_requests"]], [1, 4])
        self.assertIn("3 open pull request(s) on origin as of the last fetch", self.run_tool("--no-fetch", "viewer")[1])
        code, out, err = self.run_tool("viewer")
        self.assertEqual((code, out), (2, ""))
        self.assertIn("pr_overlap: git ls-remote", err)

    def test_the_remote_is_the_one_of_the_base_or_the_one_named(self):
        self.four_pull_requests()
        # origin becomes a fork: it has the branches, not the pull request refs.
        fork = os.path.join(self.root, "fork.git")
        self.git(self.root, "clone", "-q", "--bare", self.upstream, fork)
        self.git(self.clone, "remote", "set-url", "origin", fork)
        self.git(self.clone, "remote", "add", "upstream", self.upstream)
        self.git(self.clone, "fetch", "-q", "upstream")
        code, out, err = self.run_tool("viewer")
        self.assertEqual(code, 0)
        self.assertIn("0 open pull request(s) on origin", out)
        self.assertIn("origin has no pull request refs. Is it a fork?", err)
        report = self.report("--base", "upstream/main", "viewer")
        self.assertEqual((report["remote"], [pr["number"] for pr in report["pull_requests"]]), ("upstream", [1, 4]))
        report = self.report("--remote", self.upstream, "viewer")
        self.assertEqual((report["remote"], [pr["number"] for pr in report["pull_requests"]]), (self.upstream, [1, 4]))


class ErrorsTest(PrOverlapTest):
    def test_usage_and_git_errors_exit_2_with_one_line(self):
        self.pull(1, {"viewer/index.html": page(line5="x")})
        # What to look up is checked before the remote is asked: a mistyped path or branch costs no fetch.
        self.assertEqual(self.run_tool(os.path.join("..", "elsewhere"))[0], 2)
        self.assertEqual(self.run_tool("--head", "nowhere")[0], 2)
        self.assertEqual(self.kept_refs(), {})
        for args, message in ((["--base", "origin/nowhere", "viewer"], "--base origin/nowhere is not a commit here"),
                              (["--head", "nowhere"], "--head nowhere is not a commit here"),
                              (["--remote", "nowhere", "viewer"], "git ls-remote nowhere")):
            code, out, err = self.run_tool(*args)
            self.assertEqual((code, out), (2, ""), args)
            self.assertIn(message, err)
            self.assertTrue(err.startswith("pr_overlap: "), err)
        code, out, err = self.run_tool("viewer", repo=self.root)
        self.assertEqual((code, out), (2, ""))
        self.assertIn("pr_overlap: git rev-parse", err)
        with self.assertRaises(SystemExit) as stop, contextlib.redirect_stderr(io.StringIO()):
            pr_overlap.main(["--repo", self.clone, "--jobs", "0"])
        self.assertEqual(stop.exception.code, 2)

    def test_a_missing_git_is_an_error_message_not_a_traceback(self):
        with mock.patch.object(pr_overlap.subprocess, "run", side_effect=FileNotFoundError(2, "No such file")):
            code, out, err = self.run_tool("viewer")
        self.assertEqual((code, out), (2, ""))
        self.assertIn("pr_overlap: cannot run git: No such file", err)

    def test_one_job_gives_the_same_answer(self):
        self.four_pull_requests()
        self.assertEqual(self.numbers("--jobs", "1", "."), [1, 2, 4])


if __name__ == "__main__":
    unittest.main()
