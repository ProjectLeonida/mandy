"""Regression tests for rdna2spirv.stage_of's process-local carve-index cache.

Standard library only; no game data.

    python -I tests/test_rdna2spirv_stage_cache.py
"""
import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import rdna2spirv  # noqa: E402


def tree(root, stage):
    """One synthetic carve root with <root>/index.json and <root>/disasm/shader.s."""
    disasm = os.path.join(root, "disasm")
    os.makedirs(disasm)
    with open(os.path.join(root, "index.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump([{"file": "shader.dxbc", "stage": stage}], f)
    listing = os.path.join(disasm, "shader.s")
    with open(listing, "w", encoding="utf-8", newline="\n") as f:
        f.write("\ts_endpgm\n")
    return listing


class StageOfCacheTest(unittest.TestCase):
    def setUp(self):
        self.old = rdna2spirv._STAGES
        rdna2spirv._STAGES = None

    def tearDown(self):
        rdna2spirv._STAGES = self.old

    def test_second_root_does_not_reuse_the_first_roots_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = tree(os.path.join(tmp, "first"), "ps")
            second = tree(os.path.join(tmp, "second"), "cs")
            self.assertEqual(rdna2spirv.stage_of("shader", first), "ps")
            self.assertEqual(rdna2spirv.stage_of("shader", second), "cs")

    def test_same_root_is_still_cached(self):
        with tempfile.TemporaryDirectory() as tmp:
            listing = tree(os.path.join(tmp, "root"), "ms")
            self.assertEqual(rdna2spirv.stage_of("shader", listing), "ms")
            os.remove(os.path.join(os.path.dirname(listing), "..", "index.json"))
            self.assertEqual(rdna2spirv.stage_of("shader", listing), "ms")

    def test_a_missing_index_is_not_cached(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = os.path.join(tmp, "root")
            disasm = os.path.join(root, "disasm")
            os.makedirs(disasm)
            listing = os.path.join(disasm, "shader.s")
            with open(listing, "w", encoding="utf-8", newline="\n") as f:
                f.write("\ts_endpgm\n")
            self.assertEqual(rdna2spirv.stage_of("shader", listing), "?")
            with open(os.path.join(root, "index.json"), "w", encoding="utf-8", newline="\n") as f:
                json.dump([{"file": "shader.dxbc", "stage": "cs"}], f)
            self.assertEqual(rdna2spirv.stage_of("shader", listing), "cs")


if __name__ == "__main__":
    unittest.main()
