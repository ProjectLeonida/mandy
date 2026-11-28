"""Synthetic mid-stream I/O errors preserve the other trace's summary."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import boot_trace as bt


class BrokenStream(io.StringIO):
    def __next__(self):
        raise OSError(5, "synthetic read failure")


class ReadErrorTests(unittest.TestCase):
    def test_midstream_error_keeps_other_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            startup = str(Path(tmp) / "startup.jsonl")
            d3d = str(Path(tmp) / "d3d.log")
            Path(startup).write_text('{"kind":"exit","code":0}\n', encoding="utf-8", newline="\n")
            Path(d3d).write_text('[d3d12-trace] #1 Synthetic ok\n', encoding="utf-8", newline="\n")
            original_open = open
            for broken in (startup, d3d):
                def fake_open(path, *args, **kwargs):
                    if path == broken and kwargs.get("encoding"):
                        return BrokenStream()
                    return original_open(path, *args, **kwargs)
                out, err = io.StringIO(), io.StringIO()
                with self.subTest(broken=broken), mock.patch("builtins.open", side_effect=fake_open):
                    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                        code = bt.main([startup, "--d3d12", d3d, "--counts-only"])
                self.assertEqual(code, 2)
                doc = json.loads(out.getvalue())
                self.assertIsNone(doc["startup" if broken == startup else "d3d12"])
                self.assertIsNotNone(doc["d3d12" if broken == startup else "startup"])
                self.assertIn("trace unavailable", err.getvalue())
                self.assertNotIn(broken, err.getvalue())

    def test_reader_wraps_midstream_error(self):
        with mock.patch.object(bt, "open", side_effect=[io.BytesIO(b"{}"), BrokenStream()], create=True):
            with self.assertRaisesRegex(bt.TraceError, "cannot read"):
                bt.parse_startup("synthetic")


if __name__ == "__main__":
    unittest.main()
