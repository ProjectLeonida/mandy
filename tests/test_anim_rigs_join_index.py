"""Synthetic parity and scale checks for rig lookup."""
import collections
import os
import random
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "tools"))
import anim_rigs as ar


def reference(dicts, rigs):
    keys = collections.defaultdict(set)
    for dictionary in dicts:
        if dictionary.get("name"):
            keys[dictionary["name"]].update(e["key"] for e in dictionary.get("entries", ()))
    entries = [e for group in rigs["rbf0"]["rigsets"] for e in group["entries"]]
    counts = collections.Counter()
    for e in entries:
        h = ar.anim_index.joaat32(e["rig"].lower())
        if h in keys.get(e["source"], ()):
            counts["resolved"] += 1
        elif any(h in values for values in keys.values()):
            counts["in_another_dictionary"] += 1
        elif e["source"] not in keys:
            counts["no_dictionary_named_source"] += 1
        else:
            counts["key_missing"] += 1
    distinct = {(e["source"], e["rig"].lower()) for e in entries}
    used = {(src, ar.anim_index.joaat32(rig)) for src, rig in distinct}
    counts["distinct_rigs"] = len(distinct)
    counts["keys"] = sum(map(len, keys.values()))
    counts["keys_named_by_an_entry"] = sum((name, h) in used for name, values in keys.items() for h in values)
    return dict(sorted(counts.items()))


class JoinIndexTest(unittest.TestCase):
    def test_randomized_parity_and_edge_cases(self):
        rng = random.Random(734)
        names = [f"rig_{i}" for i in range(60)]
        for iteration in range(50):
            dictionaries = []
            for _ in range(rng.randrange(15)):
                entries = [{"key": ar.anim_index.joaat32(rng.choice(names))} for _ in range(rng.randrange(8))]
                dictionaries.append({"name": rng.choice([None, "", "a", "b", "c"]), "entries": entries})
            entries = [{"source": rng.choice(["a", "b", "c", "missing"]),
                        "rig": rng.choice(names).upper() if rng.randrange(2) else rng.choice(names)}
                       for _ in range(35)]
            rigs = {"rbf0": {"rigsets": [{"entries": entries[:15]}, {"entries": entries[15:]}]}}
            with self.subTest(iteration=iteration):
                self.assertEqual(ar.join_rbf0(dictionaries, rigs), reference(dictionaries, rigs))

    def test_many_unmatched_entries_use_one_global_union(self):
        dictionaries = [{"name": f"dictionary_{i}", "entries": [{"key": i}]} for i in range(2000)]
        entries = [{"source": "unknown", "rig": f"name_{i}"} for i in range(20000)]
        rigs = {"rbf0": {"rigsets": [{"entries": entries}]}}
        with mock.patch.object(ar.anim_index, "joaat32", return_value=0xffffffff):
            result = ar.join_rbf0(dictionaries, rigs)
        self.assertEqual(result["no_dictionary_named_source"], 20000)
        self.assertEqual(result["keys"], 2000)
        self.assertEqual(result["distinct_rigs"], 20000)


if __name__ == "__main__":
    unittest.main()
