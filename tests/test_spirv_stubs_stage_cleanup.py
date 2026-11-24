"""Staged partner stubs get the target's mode on every OS and are made writable on failure.

Synthetic index only; no game data. Run: python3 -I tests/test_spirv_stubs_stage_cleanup.py
"""
import os
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import spirv_stubs as S  # noqa: E402

INDEX = {"ps": {"stage": "ps", "inputs": [0]}}


class StageCleanupTest(unittest.TestCase):
    def test_windows_staged_stub_gets_published_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(S.os, "name", "nt"):
                written = S.generate_for_index(tmp, INDEX)
            self.assertEqual(len(written), 1)
            target = os.path.join(tmp, "stubs", written[0])
            self.assertEqual(os.stat(target).st_mode & 0o777, S.published_mode(target) & 0o777)

    def test_failed_replace_makes_staged_stub_writable_before_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            chmods = []
            real_chmod = os.chmod

            def chmod(path, mode, *args, **kwargs):
                chmods.append((str(path), mode))
                return real_chmod(path, mode, *args, **kwargs)

            with mock.patch.object(S, "_replace_stub", side_effect=OSError("replace denied")), \
                    mock.patch.object(S.os, "chmod", chmod):
                with self.assertRaises(OSError):
                    S.generate_for_index(tmp, INDEX)
            self.assertTrue(any(mode == 0o600 for _, mode in chmods), chmods)
            self.assertEqual(os.listdir(os.path.join(tmp, "stubs")), [])


if __name__ == "__main__":
    unittest.main()
