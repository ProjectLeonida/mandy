"""Unit tests for the synthetic test inventory tool (DX-10)."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
INVENTORY_TOOL = TOOLS / "test_inventory.py"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import test_inventory  # noqa: E402


class TestInventoryTests(unittest.TestCase):
    def invoke(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-I", "-B", str(INVENTORY_TOOL), *args],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )

    def test_help_exits_cleanly(self):
        proc = self.invoke("--help")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("DX-10", proc.stdout)
        self.assertIn("--summary", proc.stdout)
        self.assertIn("--json", proc.stdout)
        self.assertIn("--subsystem", proc.stdout)

    def test_summary_output(self):
        proc = self.invoke("--summary")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("Subsystem Map (DX-10)", proc.stdout)
        self.assertIn("runtime", proc.stdout)
        self.assertIn("shader", proc.stdout)
        self.assertIn("tools", proc.stdout)
        self.assertIn("viewer", proc.stdout)

    def test_json_schema_validation(self):
        proc = self.invoke("--json")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        data = json.loads(proc.stdout)
        self.assertEqual(data["schema_version"], 1)
        self.assertGreater(data["total_test_files"], 1000)
        self.assertGreater(data["synthetic_coverage_pct"], 90.0)
        self.assertIn("by_subsystem", data)
        self.assertIn("runtime", data["by_subsystem"])
        self.assertIn("shader", data["by_subsystem"])
        self.assertIn("tests", data)
        self.assertIsInstance(data["tests"], list)

    def test_subsystem_filter(self):
        proc = self.invoke("--subsystem", "shader", "--json")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        data = json.loads(proc.stdout)
        for t in data["tests"]:
            self.assertEqual(t["subsystem"], "shader")

    def test_gaps_filter(self):
        proc = self.invoke("--gaps", "--json")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        data = json.loads(proc.stdout)
        for t in data["tests"]:
            self.assertTrue(t["needs_retail_game_data"])
            self.assertIn(t["retail_requirement"], ("optional", "required"))

    def test_ignores_pycache_and_matches_semantic_discovery_rules(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_root = Path(tmp)
            sem = tmp_root / "tests" / "semantic"
            cache = sem / "__pycache__"
            pkg = sem / "pkg"
            rt = tmp_root / "runtime" / "test"
            cache.mkdir(parents=True)
            pkg.mkdir(parents=True)
            rt.mkdir(parents=True)

            (sem / "test_sample.py").write_text("import unittest\n", encoding="utf-8", newline="\n")
            (sem / "testCase.py").write_text("import unittest\n", encoding="utf-8", newline="\n")
            (pkg / "__init__.py").write_text("import unittest\n", encoding="utf-8", newline="\n")
            (cache / "test_sample.cpython-314.pyc").write_bytes(b"\x00" * 16)
            (rt / "optional_retail_test.cpp").write_text(
                '// optional retail leg\nif (!getenv("LEONIDA_GAME_DIR")) printf("SKIP\\n");\n',
                encoding="utf-8",
                newline="\n",
            )
            (rt / "required_retail_test.py").write_text(
                "requires_retail_package = True\n", encoding="utf-8", newline="\n"
            )

            items = {i["path"]: i for i in test_inventory.scan_tests(tmp_root)}
            self.assertEqual(
                sorted(items),
                [
                    "runtime/test/optional_retail_test.cpp",
                    "runtime/test/required_retail_test.py",
                    "tests/semantic/pkg/__init__.py",
                    "tests/semantic/testCase.py",
                    "tests/semantic/test_sample.py",
                ],
            )
            self.assertEqual(items["runtime/test/optional_retail_test.cpp"]["retail_requirement"], "optional")
            self.assertTrue(items["runtime/test/optional_retail_test.cpp"]["is_synthetic"])
            self.assertEqual(items["runtime/test/required_retail_test.py"]["retail_requirement"], "required")
            self.assertFalse(items["runtime/test/required_retail_test.py"]["is_synthetic"])


if __name__ == "__main__":
    unittest.main()
