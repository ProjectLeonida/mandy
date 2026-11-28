"""Synthetic class-table shape checks, before reading any PE image."""
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import gdk_runtime_calls as g


VALID = {"source": "synthetic", "classes": {"Example": {"id": "class-id", "implements": ["IExample"]}},
         "interfaces": {"IExample": {"id": "interface-id", "class": "Example", "inherits": None,
                                     "methods": ["ExampleMethod"]}}}


class TableShapeTest(unittest.TestCase):
    def invalid_documents(self):
        for doc in (None, [], "table", 1, True, {}):
            yield doc
        for field in ("classes", "interfaces"):
            for value in (None, [], "entries", 1):
                doc = copy.deepcopy(VALID)
                doc[field] = value
                yield doc
            for entry in (None, [], "entry", 1, {}, {"id": None}, {"id": []}, {"id": ""}):
                doc = copy.deepcopy(VALID)
                doc[field] = {"Broken": entry}
                yield doc

    def test_constructor_rejects_invalid_lookup_shapes(self):
        for doc in self.invalid_documents():
            with self.subTest(doc=doc), self.assertRaisesRegex(ValueError, "class table"):
                g.RuntimeTable(doc)

    def test_cli_reports_table_error_before_reading_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "table.json"
            for doc in self.invalid_documents():
                path.write_text(json.dumps(doc), encoding="utf-8", newline="\n")
                out, err = io.StringIO(), io.StringIO()
                with self.subTest(doc=doc), mock.patch.object(g, "report") as report:
                    with mock.patch.object(g, "read_runtime_source") as source:
                        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                            code = g.main(["unused.exe", "--table", str(path)])
                        source.assert_not_called()
                    report.assert_not_called()
                    self.assertEqual(code, 1)
                    self.assertEqual(out.getvalue(), "")
                    self.assertIn("cannot read the class table", err.getvalue())
                    self.assertNotIn("Traceback", err.getvalue())

    def test_valid_lookup_and_empty_tables_still_work(self):
        table = g.RuntimeTable(copy.deepcopy(VALID))
        self.assertEqual(table.label("class-id"), "Example")
        self.assertEqual(table.class_of("interface-id"), "Example")
        self.assertEqual(table.method("IExample", 3), ("ExampleMethod", None))
        self.assertEqual(g.RuntimeTable({"classes": {}, "interfaces": {}}).ids(), set())


if __name__ == "__main__":
    unittest.main()
