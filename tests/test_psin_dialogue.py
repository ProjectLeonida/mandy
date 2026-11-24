"""Synthetic conversation records for the native speaker report."""

import csv
import pathlib
import struct
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from test_psin_data import make_container, ptr  # noqa: E402
from test_psin_index import joaat  # noqa: E402
from psin_dialogue import speaker_rows  # noqa: E402


ROOT = pathlib.Path(__file__).resolve().parents[1]
# Schema identifiers and offsets describe the disk format. Payloads and names below are invented.
SCHEMA = [
    (0xEDAF33B4, 1, 88, 3, [(0x827F53FD, 16, 0x52, 0, 0, 0), (0xCDE324D0, 32, 0x52, 0, 0, 0)]),
    (0x1638FDAA, 2, 20, 2, [(0xB2A5A79B, 0, 0x4A, 0, 0, 0), (0xBD6915B4, 4, 0x14, 0, 0, 0)]),
    (0x06A04D25, 3, 64, 3, [(0x53C6B42D, 0, 0x52, 0, 0, 0), (0x59519816, 24, 0x15, 0, 0, 0)]),
    (0x7622C7F9, 4, 40, 3, [(0x9624A3B6, 0, 0x42, 0, 0, 0), (0x66FB3A21, 8, 0x42, 0, 0, 0)]),
]
SPEAKERS = (joaat("InventedFirst"), joaat("InventedSecond"))


def sample(*, second_slot=7, array_capacity=1, invalid_string=False, wrong_schema=False, duplicate_slot=False):
    # The same cue, label and slot are used in two conversations with different participants.
    groups = bytearray(176)
    participants = bytearray(40)
    turns = bytearray(128)
    lines = bytearray(80)
    cue = b"FISHING_CUE\0"
    strings = cue + b"FICTIONAL_AB_1\0"
    for i, character in enumerate(SPEAKERS):
        struct.pack_into("<QHHI", groups, i * 88 + 16, ptr(2, i * 20), 1, 1, 0)
        struct.pack_into("<QHHI", groups, i * 88 + 32, ptr(3, i * 64), 1, 1, 0)
        struct.pack_into("<II", participants, i * 20, character, second_slot if i else 7)
        struct.pack_into("<QHHI", turns, i * 64, ptr(4, i * 40), 1, array_capacity if i else 1, 0)
        struct.pack_into("<I", turns, i * 64 + 24, 7)
        label_at = len(strings) - 1 if i and invalid_string else len(cue)
        struct.pack_into("<QQ", lines, i * 40, ptr(5), ptr(5, label_at))
    if duplicate_slot:
        struct.pack_into("<QHHI", groups, 16, ptr(2), 2, 2, 0)
    schema = SCHEMA if not wrong_schema else SCHEMA[:-1] + [(0x7622C7F9, 4, 40, 3, [])]
    return make_container([(0xEDAF33B4, groups), (0x1638FDAA, participants), (0x06A04D25, turns),
                           (0x7622C7F9, lines), (0x10, strings)], schema)


class TestDialogue(unittest.TestCase):
    def test_slot_is_local_to_the_conversation(self):
        rows = list(speaker_rows(sample()))
        self.assertEqual(len(rows), 2)
        self.assertEqual([r[4] for r in rows], list(SPEAKERS))
        self.assertEqual([r[3] for r in rows], [7, 7])
        self.assertEqual([r[5:] for r in rows], [("FISHING_CUE", "FICTIONAL_AB_1")] * 2)
        self.assertEqual([b - a for a, b in zip(rows[0][:3], rows[1][:3])], [88, 64, 40])

    def test_unresolved_slot_stays_empty(self):
        self.assertEqual([r[4] for r in speaker_rows(sample(second_slot=8))], [SPEAKERS[0], 0])

    def test_duplicate_slot_stays_empty_in_its_conversation(self):
        self.assertEqual([r[4] for r in speaker_rows(sample(duplicate_slot=True))], [0, SPEAKERS[1]])

    def test_bad_records_stay_in_bounds(self):
        for blob in (sample(array_capacity=2), sample(invalid_string=True)):
            with self.subTest(size=len(blob)):
                self.assertEqual([r[4] for r in speaker_rows(blob)], [SPEAKERS[0]])
        self.assertEqual(list(speaker_rows(sample(wrong_schema=True))), [])
        self.assertEqual(list(speaker_rows(sample()[:-3])), [])

    def test_cli_writes_a_separate_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = pathlib.Path(tmp)
            resource, legacy, native = (folder / n for n in ("sample.bin", "dialogue.csv", "native.csv"))
            resource.write_bytes(sample())
            subprocess.run([sys.executable, "-I", str(ROOT / "tools/text_tables.py"), str(resource),
                            "--dialogue", str(legacy), "--native-speakers", str(native)],
                           check=True, capture_output=True, text=True)
            self.assertEqual(legacy.read_text(), "wave,key,label,speaker\n")
            with native.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["character_hash"] for row in rows], [f"{n:08x}" for n in SPEAKERS])
            self.assertEqual({row["subtitle_hash"] for row in rows}, {f"{joaat('fictional_ab_1'):08x}"})
            self.assertEqual({row["offset"] for row in rows}, {"0x0"})
            self.assertEqual({row["source"] for row in rows}, {str(resource)})


if __name__ == "__main__":
    unittest.main()
