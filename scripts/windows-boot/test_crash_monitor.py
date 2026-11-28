"""Exercise the native Windows crash monitor with an entirely synthetic child."""
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[2]


@unittest.skipUnless(__name__ == "__main__", "run through the windows_crash_monitor CTest target")
class CrashMonitorTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="leonida crash test ")
        self.addCleanup(tmp.cleanup)
        self.report = Path(tmp.name) / "private run"

    def run_fixture(self, mode, *args, seconds=10):
        env = dict(os.environ, LEONIDA_TEST_CRASH_MODE=mode)
        result = subprocess.run(
            [MONITOR, REPO, self.report, str(seconds), FIXTURE,
             subprocess.list2cmdline([str(FIXTURE), *args])],
            env=env, capture_output=True, timeout=30,
        )
        return result.returncode & 0xffffffff

    def check_artifacts(self, code, exception):
        data = (self.report / "crash.dmp").read_bytes()
        self.assertEqual(data[:4], b"MDMP")
        count, offset = struct.unpack_from("<II", data, 8)
        streams = dict((kind, pos) for kind, size, pos in
                       (struct.unpack_from("<III", data, offset + 12*i) for i in range(count)))
        self.assertIn(3, streams)  # ThreadListStream
        trace = (self.report / "stack.txt").read_text()
        self.assertIn(f"code=0x{code:08x}", trace)
        if exception:
            pos = streams[6]  # ExceptionStream: thread ID, padding, exception record
            thread_id, _, actual_code = struct.unpack_from("<III", data, pos)
            self.assertEqual(actual_code, code)
            self.assertIn(f"thread={thread_id}\n", trace)
            self.assertIn("kind=unhandled_exception", trace)
            self.assertIn("#0 ", trace)
        else:
            self.assertNotIn(6, streams)
            self.assertIn("kind=abnormal_exit", trace)
        self.assertFalse((self.report / "crash.dmp.partial").exists())
        return trace

    def test_worker(self):
        code = 0xe0421234
        self.assertEqual(self.run_fixture("worker"), code)
        trace = self.check_artifacts(code, True)
        if FIXTURE.with_suffix(".pdb").exists():
            self.assertIn("crash_worker", trace)

    def test_breakpoint(self):
        self.assertEqual(self.run_fixture("break"), 0x80000003)
        self.check_artifacts(0x80000003, True)

    def test_explicit_exit(self):
        self.assertEqual(self.run_fixture("exit"), 23)
        trace = self.check_artifacts(23, False)
        self.assertIn("#0 ", trace)

    def test_exit_with_exception_number(self):
        self.assertEqual(self.run_fixture("numeric"), 0xc0000005)
        self.check_artifacts(0xc0000005, False)

    def test_handled(self):
        self.assertEqual(self.run_fixture("handled"), 0)
        self.assertFalse(any(self.report.iterdir()))

    def test_arguments_and_clean_exit(self):
        self.assertEqual(self.run_fixture("argv", "a b\\", 'a"b'), 0)
        self.assertFalse(any(self.report.iterdir()))

    def test_timeout(self):
        self.assertEqual(self.run_fixture("timer", seconds=1), 0)
        self.assertEqual([p.name for p in self.report.iterdir()], ["timed-out"])

    def test_existing_directory(self):
        self.report.mkdir()
        sentinel = self.report / "keep"
        sentinel.write_bytes(b"original")
        self.assertNotEqual(self.run_fixture("worker"), 0)
        self.assertEqual(list(self.report.iterdir()), [sentinel])
        self.assertEqual(sentinel.read_text(), "original")

    def test_repo_rejected(self):
        self.report = REPO / (self.report.parent.name + "-forbidden")
        self.assertNotEqual(self.run_fixture("worker"), 0)
        self.assertFalse(self.report.exists())

    def test_boot_entry_point(self):
        root = self.report.parent
        (root / "MicrosoftGame.Config").touch()
        (root / "index.json").write_bytes(b"{}")
        (root / "exe").mkdir()
        (root / "exe" / "index.json").write_bytes(b"{}")
        (root / "gta6rt_tls_host.dll").touch()
        fixture = shutil.copy2(FIXTURE, root)
        shutil.copy2(MONITOR, root)
        out = root / "output"
        script = REPO / "scripts" / "windows-boot" / "boot.ps1"
        # Isolate the fixture from a developer's normal boot settings and paths.
        env = {k: v for k, v in os.environ.items() if not k.upper().startswith("LEONIDA_")}
        for mode, expected in (("exit", 23), ("timer", 0)):
            with self.subTest(mode=mode):
                result = subprocess.run(
                    ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", script,
                     "-CrashDump", "-Name", mode, "-Runtime", fixture, "-Exe", fixture, "-GameDir", root,
                     "-Shaders", root, "-OutDir", out, "-Python", sys.executable, "-Seconds",
                     "1" if mode == "timer" else "10", "-LockWaitSeconds", "1", "-NoInput", "-NoCapture"],
                    env=dict(env, LEONIDA_TEST_CRASH_MODE=mode), capture_output=True, timeout=30,
                )
                self.assertEqual(result.returncode, expected, (result.stdout, result.stderr))
                self.assertTrue((out / f"{mode}.digest.txt").is_file())
                self.assertIn(b"fixture stdout", (out / f"{mode}.stdout.log").read_bytes())
                self.assertIn(b"fixture stderr", (out / f"{mode}.log").read_bytes())
                report, = out.glob(f"{mode}-crash-*")
                if mode == "exit":
                    self.assertTrue((report / "crash.dmp").is_file())
                else:
                    self.assertTrue((report / "timed-out").is_file())
                    self.assertFalse((report / "crash.dmp").exists())


if __name__ == "__main__":
    MONITOR, FIXTURE = map(Path, sys.argv[1:3])
    unittest.main(argv=sys.argv[:1], verbosity=2)
