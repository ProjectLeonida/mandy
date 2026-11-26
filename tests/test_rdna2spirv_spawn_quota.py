"""tools/rdna2spirv.py: _run_spirv_tool starts a tool again when Windows refuses it with ERROR_NOT_ENOUGH_QUOTA.

    python -I tests/test_rdna2spirv_spawn_quota.py -v

The Windows compute worker (scripts/compute_workers_win.py) runs in a job that allows two active processes: itself
and one tool. The job can still count a tool that has just exited when the next one is started, and CreateProcess
then fails with winerror 1816. The unit tests fake that error; the Windows-only test starts real processes in such
a job. No game data.
"""
import errno
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "scripts"))

import rdna2spirv  # noqa: E402

QUOTA = 1816  # ERROR_NOT_ENOUGH_QUOTA


def win_error(code):
    """An OSError as CreateProcess raises it, with the Win32 code in .winerror (a plain attribute off Windows)."""
    e = OSError(errno.ENOMEM, "Not enough quota is available to process this command")
    e.winerror = code
    return e


class Clock:
    """time.monotonic and time.sleep stand-ins: sleeping moves the clock, nothing waits."""

    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class RunSpirvToolTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        for name in ("monotonic", "sleep"):
            patcher = mock.patch.object(rdna2spirv.time, name, getattr(self.clock, name))
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_quota_refusals_are_started_again(self):
        done = subprocess.CompletedProcess(["spirv-val"], 0)
        with mock.patch.object(rdna2spirv.subprocess, "run",
                               side_effect=[win_error(QUOTA), win_error(QUOTA), done]) as run:
            self.assertIs(rdna2spirv._run_spirv_tool(["spirv-val", "x.spv"], capture_output=True), done)
        self.assertEqual(run.call_count, 3)
        for call in run.call_args_list:
            self.assertEqual(call, mock.call(["spirv-val", "x.spv"], capture_output=True))
        self.assertEqual(len(self.clock.sleeps), 2)

    def test_a_refusal_that_lasts_is_raised_once_the_time_is_up(self):
        with mock.patch.object(rdna2spirv.subprocess, "run", side_effect=win_error(QUOTA)) as run:
            with self.assertRaises(OSError) as raised:
                rdna2spirv._run_spirv_tool(["spirv-as", "-"])
        self.assertEqual(raised.exception.winerror, QUOTA)
        self.assertGreater(run.call_count, 1)
        self.assertLessEqual(self.clock.now, rdna2spirv.SPAWN_RETRY_SECONDS)
        self.assertGreaterEqual(self.clock.now, rdna2spirv.SPAWN_RETRY_SECONDS - max(self.clock.sleeps))
        # The traceback says the retry ran out, not only what the first refusal said.
        self.assertEqual(raised.exception.__notes__,
                         [f"spirv-as: refused {run.call_count} times in {self.clock.now:.1f} s "
                          f"(SPAWN_RETRY_SECONDS = {rdna2spirv.SPAWN_RETRY_SECONDS})"])

    def test_other_errors_are_not_retried(self):
        for error in (win_error(5), OSError(errno.ENOMEM, "out of memory")):
            with self.subTest(error=error):
                with mock.patch.object(rdna2spirv.subprocess, "run", side_effect=error) as run:
                    with self.assertRaises(OSError) as raised:
                        rdna2spirv._run_spirv_tool(["spirv-val"])
                self.assertIs(raised.exception, error)
                self.assertFalse(hasattr(error, "__notes__"))
                self.assertEqual(run.call_count, 1)
        self.assertEqual(self.clock.sleeps, [])

    def test_a_missing_tool_still_names_itself(self):
        missing = FileNotFoundError(errno.ENOENT, "The system cannot find the file specified")
        missing.winerror = 2
        with mock.patch.object(rdna2spirv.subprocess, "run", side_effect=missing) as run:
            with self.assertRaises(FileNotFoundError) as raised:
                rdna2spirv._run_spirv_tool(["spirv-as", "-"])
        self.assertEqual(raised.exception.filename, "spirv-as")
        self.assertEqual(run.call_count, 1)


CHILD = """
import json, sys
sys.path.insert(0, sys.argv[1])
import rdna2spirv
sys.stdin.readline()  # in the job from here on
for _ in range(int(sys.argv[2])):
    rdna2spirv._run_spirv_tool(["cmd", "/c", "rem"], capture_output=True, check=True)
print(json.dumps("ok"))
"""


@unittest.skipUnless(os.name == "nt", "Windows job objects")
class TwoProcessJobTests(unittest.TestCase):
    def test_back_to_back_tools_in_the_worker_job(self):
        # Before the retry, 15 to 28 of 300 such starts failed (5-9 %) with winerror 1816 on a Windows 11 host.
        import compute_workers_win as win
        mib = 1024
        with win.job(mib, True) as job:
            p = subprocess.Popen([sys.executable, "-I", "-c", CHILD, str(ROOT / "tools"), "150"],
                                 stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                 creationflags=win.LOW)
            try:
                win.check(win.assign(job, int(p._handle)))
                win.verify(job, int(p._handle), mib, True)
                out, err = p.communicate("go\n", timeout=300)
            finally:
                if p.poll() is None:
                    p.kill()
                    p.wait(timeout=10)
        self.assertEqual(p.returncode, 0, err)
        self.assertEqual(json.loads(out), "ok")


if __name__ == "__main__":
    unittest.main()
