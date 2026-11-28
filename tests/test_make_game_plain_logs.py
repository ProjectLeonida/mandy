"""make_game.py without the LEONIDA board (--no-tui, or stdout redirected) still writes <work>/logs/<stage>.log:
log() and fail() lines and the output of the commands run() starts, which also reach the console byte for byte.
No game files needed.

    python -I tests/test_make_game_plain_logs.py
"""
import contextlib
import io
import os
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import make_game  # noqa: E402

leonida_tui = make_game.leonida_tui


def console():
    """A stdout with a byte buffer, like the real one, so the mirrored bytes can be read back."""
    return io.TextIOWrapper(io.BytesIO(), encoding="utf-8", newline="")


def console_bytes(stream) -> bytes:
    stream.flush()
    return stream.buffer.getvalue()


def child(code: str) -> list:
    return [sys.executable, "-I", "-c", code]


class PlainStageLogsTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.logs_dir = os.path.join(self.dir.name, "logs")
        self.assertIsNone(leonida_tui.current_board())

    def read(self, stage: str) -> list:
        with open(os.path.join(self.logs_dir, f"{stage}.log"), encoding="utf-8") as f:
            return f.read().splitlines()

    def test_log_and_fail_lines_go_to_the_running_stage(self):
        out = console()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            with make_game.PlainStageLogs(["shaders", "assets"], self.logs_dir):
                with leonida_tui.stage_scope("shaders"):
                    make_game.log("translating")
                    with self.assertRaises(SystemExit):
                        make_game.fail("llvm-mc is missing")
                with leonida_tui.stage_scope("assets"):
                    make_game.log("exporting")
                make_game.log("outside every stage")
        shaders, assets = self.read("shaders"), self.read("assets")
        self.assertEqual(len(shaders), 2)
        self.assertTrue(shaders[0].endswith("  translating"))
        self.assertEqual(shaders[1], "ERROR: llvm-mc is missing")
        self.assertEqual(len(assets), 1)
        self.assertTrue(assets[0].endswith("  exporting"))
        text = console_bytes(out).decode()
        for line in ("translating", "exporting", "outside every stage", f"stage logs in {self.logs_dir}"):
            self.assertIn(line, text)

    def test_run_mirrors_the_bytes_and_logs_each_piece(self):
        # A \r progress line, a \r\n ending, stderr after stdout and a last piece with no ending. The child writes
        # bytes, so its own text mode (\n -> \r\n on Windows) does not change what it sends.
        code = ("import sys; o, e = sys.stdout.buffer, sys.stderr.buffer; "
                "o.write(b'shard 1/2\\rshard 2/2\\r\\ndone\\n'); o.flush(); e.write(b'a warning\\n'); e.flush(); "
                "o.write(b'tail'); o.flush()")
        out = console()
        with contextlib.redirect_stdout(out):
            with make_game.PlainStageLogs(["shaders"], self.logs_dir):
                with leonida_tui.stage_scope("shaders"):
                    self.assertEqual(make_game.run(child(code)), 0)
        self.assertIn(b"shard 1/2\rshard 2/2\r\ndone\na warning\ntail", console_bytes(out))
        lines = self.read("shaders")
        self.assertIn(os.path.basename(sys.executable), lines[0])   # run() logs the command first
        self.assertEqual(lines[1:], ["shard 1/2", "shard 2/2", "done", "a warning", "tail"])

    def test_a_failed_command_stops_the_build_with_its_code_in_the_log(self):
        with contextlib.redirect_stdout(console()), contextlib.redirect_stderr(io.StringIO()):
            with make_game.PlainStageLogs(["runtime"], self.logs_dir):
                with leonida_tui.stage_scope("runtime"):
                    self.assertEqual(make_game.run(child("import sys; sys.exit(3)"), check=False), 3)
                    with self.assertRaises(SystemExit):
                        make_game.run(child("import sys; sys.exit(4)"), hint="see the log")
        self.assertEqual(self.read("runtime")[-1], f"ERROR: {os.path.basename(sys.executable)} exited with 4. "
                                                   "see the log")

    def test_caller_pipes_and_unlogged_stages_keep_subprocess_run(self):
        done = subprocess.CompletedProcess([], 0)
        with contextlib.redirect_stdout(console()), patch("make_game.subprocess.run", return_value=done) as plain:
            with make_game.PlainStageLogs(["shaders"], self.logs_dir):
                with leonida_tui.stage_scope("shaders"):
                    make_game.run(["tool", "a"], stdout=subprocess.PIPE)        # the caller reads the output
                with leonida_tui.stage_scope("check"):
                    make_game.run(["tool", "b"])                                # not a build stage with a log
                make_game.run(["tool", "c"])                                    # no stage at all
        self.assertEqual([c.args[0] for c in plain.call_args_list], [["tool", "a"], ["tool", "b"], ["tool", "c"]])

    def test_extra_arguments_reach_the_child(self):
        with contextlib.redirect_stdout(console()):
            with make_game.PlainStageLogs(["pipelines"], self.logs_dir):
                with leonida_tui.stage_scope("pipelines"):
                    env = dict(os.environ, LEONIDA_PLAIN_LOG_TEST="from the caller")
                    make_game.run(child("import os; print(os.getcwd()); print(os.environ['LEONIDA_PLAIN_LOG_TEST'])"),
                                  cwd=self.dir.name, env=env, text=True)
        lines = self.read("pipelines")
        self.assertEqual(os.path.normcase(lines[1]), os.path.normcase(os.path.realpath(self.dir.name)))
        self.assertEqual(lines[2], "from the caller")

    def test_inactive_after_the_build_and_appends_like_the_board(self):
        for word in ("first", "second"):
            with contextlib.redirect_stdout(console()):
                with make_game.PlainStageLogs(["assets"], self.logs_dir):
                    with leonida_tui.stage_scope("assets"):
                        make_game.log(word)
            self.assertIsNone(make_game._plain_logs)
        self.assertEqual([line.split("  ", 1)[1] for line in self.read("assets")], ["first", "second"])

    def test_ctrl_c_stops_the_child(self):
        # Ctrl+C while the output is pumped: the interrupt goes on, and the child does not keep running.
        slow = child("import sys, time; print('started', flush=True); time.sleep(60)")
        started = time.monotonic()
        with contextlib.redirect_stdout(console()), \
                patch.object(make_game.PlainStageLogs, "_write_pieces", side_effect=KeyboardInterrupt):
            with make_game.PlainStageLogs(["assets"], self.logs_dir):
                with leonida_tui.stage_scope("assets"), self.assertRaises(KeyboardInterrupt):
                    make_game.run(slow)
        self.assertLess(time.monotonic() - started, 20)

    def test_an_unwritable_log_folder_does_not_stop_the_build(self):
        blocker = os.path.join(self.dir.name, "a file")
        with open(blocker, "w", encoding="utf-8", newline="\n") as f:
            f.write("not a folder\n")
        out, err = console(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            with make_game.PlainStageLogs(["shaders"], os.path.join(blocker, "logs")):
                with leonida_tui.stage_scope("shaders"):
                    make_game.log("still logged to the console")
                    self.assertEqual(make_game.run(child("print('child output')")), 0)
        text = console_bytes(out).decode()
        self.assertIn("still logged to the console", text)
        self.assertIn("child output", text)
        self.assertNotIn("stage logs in", text)
        self.assertEqual(err.getvalue().count("cannot write the stage logs"), 1)

    def test_no_build_stage_writes_nothing(self):
        out = console()
        with contextlib.redirect_stdout(out):
            with make_game.PlainStageLogs([], self.logs_dir):
                make_game.log("check only")
        self.assertFalse(os.path.exists(self.logs_dir))
        self.assertNotIn(b"stage logs in", console_bytes(out))


if __name__ == "__main__":
    unittest.main()
