"""Windows Job Object, output lock, and gate validation tests for scripts/compute_workers_win.py."""

import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if os.name == "nt":
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    import compute_workers_win as cww  # noqa: E402
else:
    cww = None


@unittest.skipUnless(os.name == "nt", "Windows kernel32 Job Object and CreateFileW APIs")
class ComputeWorkersWinTests(unittest.TestCase):
    def test_job_creation_flags_and_mib_validation(self):
        import ctypes as ct

        for worker in (False, True):
            with self.subTest(worker=worker):
                with cww.job(64, worker) as h:
                    info = cww.Extended()
                    cww.check(cww.query_job(h, 9, ct.byref(info), ct.sizeof(info), None))
                    self.assertEqual(info.basic.flags, cww.flags(worker))
                    self.assertEqual(info.process_mem, 64 << 20)

        for bad_mib in (True, False, 1.5, 0, -1, 1 << 50):
            with self.subTest(bad_mib=bad_mib):
                with self.assertRaises(ValueError):
                    with cww.job(bad_mib, False):
                        pass

    def test_output_lock_exclusivity_and_null_handle_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock_path = Path(tmp) / "out.lock"
            with cww.output_lock(lock_path):
                with self.assertRaises(PermissionError):
                    with cww.output_lock(lock_path):
                        pass
            # Re-acquiring succeeds after clean exit.
            with cww.output_lock(lock_path):
                pass

            with (
                mock.patch.object(cww, "create_file", return_value=0),
                mock.patch.object(cww, "close", side_effect=AssertionError("close(0) must not be called")),
                self.assertRaises(OSError),
            ):
                with cww.output_lock(lock_path):
                    pass

    def test_start_rejects_bad_lock_mib_and_oversized_gate_before_popen(self):
        cases = [
            ("req.json", 64, False),
            ("req.json", 64, 0),
            ("req.json", 64, -1),
            ("req.json", True, 4),
            ("req.json", 0, 4),
            ("x" * 70000, 64, 4),
        ]
        for req, mib, lock in cases:
            with self.subTest(mib=mib, lock=lock, req_len=len(req)):
                with (
                    mock.patch.object(cww, "job") as mk_job,
                    mock.patch.object(cww.subprocess, "Popen") as popen,
                    self.assertRaises(ValueError),
                ):
                    with cww.start(req, mib, lock):
                        pass
                mk_job.assert_not_called()
                popen.assert_not_called()

    def test_child_rejects_bad_gates_and_closes_handle_when_verify_fails(self):
        bad_gates = [
            b'["req", 64, 4]',                # no trailing newline
            b"x" * 65537,                     # oversized
            b'\xff\n',                        # non-ASCII
            b'{\n',                           # malformed JSON
            b'{}\n',                          # non-list
            b'["req", 64]\n',                 # wrong length
            b'["req", true, 4]\n',            # bool mib
            b'["", 64, 4]\n',                 # empty request
            b'["req", 64, 0]\n',              # zero handle
            b'["req", 64, -5]\n',             # negative handle
        ]
        for raw in bad_gates:
            with self.subTest(raw=raw[:32]):
                fake_stdin = mock.Mock(buffer=io.BytesIO(raw))
                with mock.patch.object(cww.sys, "stdin", fake_stdin), self.assertRaises(ValueError):
                    cww.child()

        fake_stdin = mock.Mock(buffer=io.BytesIO(b'["req.json", 64, 123]\n'))
        with (
            mock.patch.object(cww.sys, "stdin", fake_stdin),
            mock.patch.object(cww, "verify", side_effect=RuntimeError("no job")),
            mock.patch.object(cww, "close", return_value=1) as close_fn,
            self.assertRaisesRegex(RuntimeError, "no job"),
        ):
            cww.child()
        close_fn.assert_called_once_with(123)


if __name__ == "__main__":
    unittest.main()
