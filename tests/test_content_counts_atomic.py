"""Synthetic multi-input CLI checks for transactional aggregate counts."""
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
import content_definitions as cd


class AtomicCountsTest(unittest.TestCase):
    def test_failed_directory_discards_partial_counts_and_continues(self):
        valid = cd.parse_content([b'<SSetupData><contentChangeSetGroups><Item/></contentChangeSetGroups></SSetupData>'])
        for error in (OSError("synthetic"), cd.ContentError("synthetic"), ValueError("synthetic")):
            with self.subTest(error=type(error).__name__), tempfile.TemporaryDirectory() as tmp:
                bad = Path(tmp) / "broken"
                good = Path(tmp) / "good"
                bad.mkdir()
                good.mkdir()
                def scan(path):
                    yield (0, 0, valid)
                    if path == str(bad):
                        yield (1, 0, None)
                        raise error
                out = io.StringIO()
                with mock.patch.object(cd, "carve_documents", side_effect=scan), \
                     mock.patch.object(sys, "argv", ["content_definitions", str(bad), str(good)]), \
                     contextlib.redirect_stdout(out):
                    code = cd.main()
                result = json.loads(out.getvalue())
                self.assertEqual(code, 1)
                self.assertEqual(result["failed_inputs"], 1)
                self.assertEqual(result["documents"], 1)
                self.assertEqual(result["setups"], 1)
                self.assertEqual(result["groups"], 1)
                self.assertEqual(result["other_xml"], 0)

    def test_successful_inputs_keep_all_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "synthetic"
            folder.mkdir()
            doc = cd.parse_content([b'<SSetupData/>'])
            out = io.StringIO()
            with mock.patch.object(cd, "carve_documents", return_value=iter([(0, 0, doc), (1, 0, None)])), \
                 mock.patch.object(sys, "argv", ["content_definitions", str(folder)]), \
                 contextlib.redirect_stdout(out):
                code = cd.main()
            result = json.loads(out.getvalue())
            self.assertEqual(code, 0)
            self.assertEqual((result["documents"], result["other_xml"], result["failed_inputs"]), (1, 1, 0))


if __name__ == "__main__":
    unittest.main()
