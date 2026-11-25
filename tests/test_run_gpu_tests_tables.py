#!/usr/bin/env python3
"""Verify scripts/run_gpu_tests.py keeps compute-dump fixture tables sorted one entry per line."""
import ast
import pathlib
import sys
import unittest

REPO = pathlib.Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_gpu_tests  # noqa: E402


def _parse_module_table(tree: ast.Module, name: str) -> ast.AST:
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name:
            return node.value
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return node.value
    raise AssertionError(f"{name} not found at module level in scripts/run_gpu_tests.py")


class RunGpuTestsTablesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = (SCRIPTS_DIR / "run_gpu_tests.py").read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)

    def test_cs_dump_wants_sorted_one_per_line(self) -> None:
        node = _parse_module_table(self.tree, "CS_DUMP_WANTS")
        self.assertIsInstance(node, ast.Dict)
        keys = [k.value for k in node.keys]
        lines = [k.lineno for k in node.keys]
        self.assertEqual(len(keys), len(set(keys)), "Duplicate key in CS_DUMP_WANTS")
        self.assertEqual(keys, sorted(keys), "CS_DUMP_WANTS keys must be sorted alphabetically")
        self.assertEqual(len(lines), len(set(lines)), "CS_DUMP_WANTS entries must be one per line")
        self.assertGreaterEqual(len(keys), 70)

    def test_cs_dump_non_wave64_sorted_subset_one_per_line(self) -> None:
        node = _parse_module_table(self.tree, "CS_DUMP_NON_WAVE64")
        self.assertIsInstance(node, ast.Call)
        self.assertEqual(len(node.args), 1)
        set_node = node.args[0]
        self.assertIsInstance(set_node, ast.Set)
        names = [e.value for e in set_node.elts]
        lines = [e.lineno for e in set_node.elts]
        self.assertEqual(len(names), len(set(names)), "Duplicate name in CS_DUMP_NON_WAVE64")
        self.assertEqual(names, sorted(names), "CS_DUMP_NON_WAVE64 must be sorted alphabetically")
        self.assertEqual(len(lines), len(set(lines)), "CS_DUMP_NON_WAVE64 entries must be one per line")
        self.assertTrue(run_gpu_tests.CS_DUMP_NON_WAVE64.issubset(run_gpu_tests.CS_DUMP_WANTS))

    def test_cs_dump_hexwords_sorted_subset_one_per_line(self) -> None:
        node = _parse_module_table(self.tree, "CS_DUMP_HEXWORDS")
        self.assertIsInstance(node, ast.Call)
        self.assertEqual(len(node.args), 1)
        set_node = node.args[0]
        self.assertIsInstance(set_node, ast.Set)
        names = [e.value for e in set_node.elts]
        lines = [e.lineno for e in set_node.elts]
        self.assertEqual(len(names), len(set(names)), "Duplicate name in CS_DUMP_HEXWORDS")
        self.assertEqual(names, sorted(names), "CS_DUMP_HEXWORDS must be sorted alphabetically")
        self.assertEqual(len(lines), len(set(lines)), "CS_DUMP_HEXWORDS entries must be one per line")
        self.assertTrue(run_gpu_tests.CS_DUMP_HEXWORDS.issubset(run_gpu_tests.CS_DUMP_WANTS))

    def test_dispatch_uses_cs_dump_wants_directly(self) -> None:
        self.assertIn("elif name in CS_DUMP_WANTS:", self.source)
        self.assertIn("wave64 = name not in CS_DUMP_NON_WAVE64", self.source)
        self.assertIn("hexwords=name in CS_DUMP_HEXWORDS", self.source)


if __name__ == "__main__":
    unittest.main()
