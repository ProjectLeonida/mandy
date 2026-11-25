"""Tests for tools/fill_fspx_library_states.py.

Synthetic index and .state sidecars only: no game data, no GPU.

Run:
    python -I tests/test_fill_fspx_library_states.py
"""

import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "fill_fspx_library_states.py")

spec = importlib.util.spec_from_file_location("fill_fspx_library_states_under_test", TOOL)
fill_states_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fill_states_module)


class FillStatesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        self.sidecars = os.path.join(self.dir, "fspx")
        os.makedirs(self.sidecars)
        self.index_path = os.path.join(self.dir, "index.json")

    def tearDown(self):
        self.tmp.cleanup()

    def write_state(self, name, data=b"state"):
        with open(os.path.join(self.sidecars, name + ".state"), "wb") as sink:
            sink.write(data)
        return hashlib.sha256(data).hexdigest()

    def test_fills_only_missing_keys_with_a_sidecar(self):
        digest = self.write_state("lib_cs_00001", b"kind0")
        index = {
            "lib_cs_00001": {"fspx_code_sha256": "aa", "stage": "cs"},
            "lib_cs_00002": {"fspx_code_sha256": "bb", "fspx_state_sha256": "kept", "stage": "cs"},
            "lib_cs_00003": {"fspx_code_sha256": "cc", "stage": "cs"},
            "game_cs_00004": {"fssm_sha256": "dd", "stage": "cs"},
        }
        added, without = fill_states_module.fill_states(index, self.sidecars)
        self.assertEqual((added, without), (1, 1))
        self.assertEqual(index["lib_cs_00001"]["fspx_state_sha256"], digest)
        self.assertEqual(index["lib_cs_00002"]["fspx_state_sha256"], "kept")
        self.assertNotIn("fspx_state_sha256", index["lib_cs_00003"])
        self.assertNotIn("fspx_state_sha256", index["game_cs_00004"])

    def test_publish_writes_lf_sorted_json(self):
        index = {"b": {"fspx_code_sha256": "2"}, "a": {"fspx_code_sha256": "1"}}
        fill_states_module.publish(self.index_path, index)
        with open(self.index_path, "rb") as source:
            raw = source.read()
        self.assertNotIn(b"\r\n", raw)
        self.assertLess(raw.index(b'"a"'), raw.index(b'"b"'))
        self.assertEqual(json.loads(raw.decode("utf-8")), index)

    def test_main_fills_an_index_in_place(self):
        digest = self.write_state("lib_cs_00010", b"state-10")
        with open(self.index_path, "w", encoding="utf-8", newline="\n") as sink:
            json.dump({"lib_cs_00010": {"fspx_code_sha256": "ee", "stage": "cs"}}, sink)
        self.assertEqual(fill_states_module.main([self.index_path, self.sidecars]), 0)
        with open(self.index_path, encoding="utf-8") as source:
            index = json.load(source)
        self.assertEqual(index["lib_cs_00010"]["fspx_state_sha256"], digest)

    def test_main_refuses_a_non_object_index(self):
        with open(self.index_path, "w", encoding="utf-8", newline="\n") as sink:
            sink.write("[]\n")
        self.assertEqual(fill_states_module.main([self.index_path, self.sidecars]), 1)

    def test_main_refuses_an_unreadable_index(self):
        self.assertEqual(fill_states_module.main([self.index_path, self.sidecars]), 1)


if __name__ == "__main__":
    unittest.main()
