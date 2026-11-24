"""Synthetic truncated streams must not become successful partial counts."""
import contextlib
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path[:0] = [os.path.join(os.path.dirname(os.path.dirname(__file__)), p) for p in ("tools", "tests")]
import awc_markers as am
from test_awc_markers import build_bank, event


class ShortReadTest(unittest.TestCase):
    def test_truncated_header_scan_fails_instead_of_returning_partial(self):
        data = build_bank([(7, [(am.TAG_MARKER, event(1, 2, 3))])])
        with self.assertRaisesRegex(OSError, "short read"):
            list(am.iter_banks(io.BytesIO(data[:-1]), len(data)))
        self.assertEqual(len(list(am.iter_banks(io.BytesIO(data), len(data)))), 1)

    def test_truncated_marker_and_metadata_reads_fail(self):
        for tag in (am.TAG_MARKER, am.TAG_FORMAT):
            with self.subTest(tag=tag):
                bank = {"offset": 0, "entries": [(7, [(am.TAG_MARKER, 0, 16), (tag, 0, 32)])]}
                with self.assertRaisesRegex(OSError, "short read"):
                    am.bank_chunks(io.BytesIO(b"\0" * 15), bank)

    def test_cli_does_not_publish_when_scan_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "synthetic.awc"
            output = Path(tmp) / "report.json"
            source.write_bytes(b"synthetic")
            output.write_bytes(b"previous report")
            err = io.StringIO()
            with mock.patch.object(am, "iter_banks", side_effect=OSError("short read")), \
                 contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
                code = am.main([str(source), "--out", str(output)])
            self.assertEqual(code, 1)
            self.assertIn("short read", err.getvalue())
            self.assertEqual(output.read_bytes(), b"previous report")


if __name__ == "__main__":
    unittest.main()
