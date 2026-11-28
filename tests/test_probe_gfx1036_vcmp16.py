"""tools/probe_gfx1036_vcmp16.py names what it is missing in one line instead of a traceback.

    python -I tests/test_probe_gfx1036_vcmp16.py

The probe itself needs a gfx1036 GPU; these checks need none, and no game files.
"""
import contextlib
import io
import os
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import probe_gfx1036_vcmp16 as probe  # noqa: E402


class MissingPrerequisiteTest(unittest.TestCase):
    def run_main(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            rc = probe.main()
        return rc, err.getvalue().strip().splitlines()

    def test_missing_vk_compute_test_is_one_line(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(probe, "VK_COMPUTE_TEST", os.path.join(tmp, "vk_compute_test.exe")):
            rc, lines = self.run_main()
        self.assertEqual(rc, 1)
        self.assertEqual(len(lines), 1, lines)
        self.assertTrue(lines[0].startswith("ERROR: vk_compute_test not found"), lines)

    def test_missing_spirv_as_is_one_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            exe = os.path.join(tmp, "vk_compute_test.exe")
            with open(exe, "wb"):
                pass
            with mock.patch.object(probe, "VK_COMPUTE_TEST", exe), \
                    mock.patch.object(probe.shutil, "which", return_value=None):
                rc, lines = self.run_main()
        self.assertEqual(rc, 1)
        self.assertEqual(lines, ["ERROR: spirv-as not found on PATH; install the Vulkan SDK (https://vulkan.lunarg.com/)"])

    def test_usage_names_this_file(self):
        doc = probe.__doc__
        self.assertIn("tools/probe_gfx1036_vcmp16.py", doc)
        self.assertNotIn("Leonida-worktrees", doc)


if __name__ == "__main__":
    unittest.main()
