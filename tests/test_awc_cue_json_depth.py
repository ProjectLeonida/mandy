"""Synthetic malformed cue sheets do not crash or publish label tracks."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "awc_cue_labels.py"


class CueDepthTest(unittest.TestCase):
    def test_deep_json_reports_error_without_creating_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            sheet = Path(tmp) / "synthetic.json"
            output = Path(tmp) / "labels"
            sheet.write_text("[" * 3000 + "0" + "]" * 3000, encoding="utf-8", newline="\n")
            result = subprocess.run([sys.executable, "-I", str(TOOL), str(sheet), str(output)],
                                    capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 1)
            self.assertIn("cue sheet JSON nesting is too deep", result.stderr)
            self.assertNotIn("Traceback", result.stderr)
            self.assertFalse(output.exists())

    def test_valid_sheet_still_creates_expected_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            sheet = Path(tmp) / "synthetic.json"
            output = Path(tmp) / "labels"
            sheet.write_text(json.dumps({"sample_rate": 48000, "regions": [{"file": "00000001.ogg",
                "name_hash": "00000002", "start_sample": 0, "end_sample": 48000}]}),
                encoding="utf-8", newline="\n")
            result = subprocess.run([sys.executable, "-I", str(TOOL), str(sheet), str(output)],
                                    capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual((output / "00000001.txt").read_text(encoding="utf-8"),
                             "0.000000000\t1.000000000\t00000002 #1\n")


if __name__ == "__main__":
    unittest.main()
