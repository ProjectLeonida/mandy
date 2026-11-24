"""Synthetic output failure tests for the skeleton report CLI."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "tools"))
import anim_skel_match as sm


class BrokenWriter(io.StringIO):
    def write(self, text):
        raise OSError(28, "synthetic output failure")


class ReportErrorTest(unittest.TestCase):
    def invoke(self, target, fail_write=False):
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(sm, "stream_names", return_value={"synthetic": []}), \
             mock.patch.object(sm, "read_skeletons", return_value=[]), \
             contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            if fail_write:
                with mock.patch.object(sm, "open", return_value=BrokenWriter(), create=True):
                    code = sm.main(["input-a", "input-b", "--json", str(target)])
            else:
                code = sm.main(["input-a", "input-b", "--json", str(target)])
        return code, out.getvalue(), err.getvalue()

    def test_directory_as_report_is_a_clean_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, _, err = self.invoke(tmp)
        self.assertEqual(code, 2)
        self.assertIn("cannot write", err)
        self.assertNotIn("Traceback", err)

    def test_write_failure_is_a_clean_error(self):
        code, _, err = self.invoke("synthetic.json", fail_write=True)
        self.assertEqual(code, 2)
        self.assertIn("synthetic output failure", err)

    def test_successful_report_remains_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "report.json"
            code, _, err = self.invoke(target)
            doc = json.loads(target.read_text(encoding="utf-8"))
        self.assertEqual((code, err), (0, ""))
        self.assertEqual(doc["skeletons"], 0)


if __name__ == "__main__":
    unittest.main()
