"""Synthetic D3D12 trace rows exceeding Python's integer parsing limit."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import boot_trace as bt


class SequenceLimitTest(unittest.TestCase):
    def setUp(self):
        self.old_limit = sys.get_int_max_str_digits()
        sys.set_int_max_str_digits(640)
        self.addCleanup(sys.set_int_max_str_digits, self.old_limit)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "synthetic.log"
        self.corrupt = "[d3d12-trace] #" + "9" * 641 + " CorruptMethod failed"

    def write(self, lines):
        self.path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

    def test_corrupt_rows_preserve_valid_calls_and_sequence(self):
        self.write([self.corrupt, "[d3d12-trace] #1 First ok", self.corrupt,
                    "[d3d12-trace] #2 Second failed", self.corrupt])
        trace = bt.parse_d3d12(str(self.path))
        self.assertEqual(trace["ignored_lines"], 3)
        self.assertEqual(trace["calls"], 2)
        self.assertEqual(trace["outcomes"], {"ok": 1, "failed": 1})
        self.assertEqual(trace["sequence_break_count"], 0)
        self.assertEqual(trace["methods"], ["First", "Second"])
        self.assertEqual(trace["last_call"], {"seq": 2, "method": "Second", "outcome": "failed"})

    def test_only_corrupt_rows_give_clean_cli_error(self):
        self.write([self.corrupt])
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = bt.main(["--d3d12", str(self.path), "--counts-only"])
        self.assertEqual(code, 2)
        self.assertEqual(out.getvalue(), "")
        self.assertIn("trace unavailable", err.getvalue())
        self.assertNotIn("Traceback", err.getvalue())

    def test_cli_counts_report_keeps_later_valid_call(self):
        self.write([self.corrupt, "[d3d12-trace] #1 First ok"])
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = bt.main(["--d3d12", str(self.path), "--counts-only"])
        self.assertEqual(code, 0)
        report = json.loads(out.getvalue())
        self.assertEqual(report["d3d12"]["calls"], 1)
        self.assertEqual(report["d3d12"]["ignored_lines"], 1)
        self.assertNotIn("CorruptMethod", out.getvalue())


if __name__ == "__main__":
    unittest.main()
