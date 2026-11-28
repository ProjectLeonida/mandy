"""Verify production BC1-BC7 decoding against independent synthetic GDK goldens."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import bcpack


def goldens():
    for line in (ROOT / "tests/fixtures/bcpack-standalone.txt").read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            codec, name, packed, expected = line.split()
            yield int(codec), name, bytes.fromhex(packed), bytes.fromhex(expected)


class StandaloneDecoder(unittest.TestCase):
    def test_independent_goldens(self):
        cases = list(goldens())
        self.assertEqual(len(cases), 1076)
        for codec, name, packed, expected in cases:
            with self.subTest(name=name, path="python"), \
                    patch.object(bcpack, "_decode_lanes_native", return_value=None):
                self.assertEqual(bcpack.decode(packed, codec, len(expected)), expected)
            if bcpack._native_unpack() is not None:
                with self.subTest(name=name, path="native-with-python-fallback"):
                    self.assertEqual(bcpack.decode(packed, codec, len(expected)), expected)

    def test_profiles_cannot_change_production_output(self):
        for codec, name, packed, expected in goldens():
            if not name.endswith("-request"):
                continue
            with self.subTest(codec=codec):
                got, info = bcpack.decode_with_info(packed, codec, len(expected),
                                                    allow_provisional=True, history="missing-profile.json")
                self.assertEqual(got, expected)
                self.assertEqual(info, set())

    def test_malformed_input(self):
        for codec, name, packed, expected in goldens():
            if not name.endswith("-request"):
                continue
            for cut in (0, 1, len(packed) // 2):
                with self.subTest(codec=codec, cut=cut), self.assertRaises(bcpack.BcpackError):
                    bcpack.decode(packed[:cut], codec, len(expected))
            with self.subTest(codec=codec), self.assertRaises(bcpack.BcpackError):
                bcpack.decode(packed + b"\xff", codec, len(expected))
            with self.subTest(codec=codec), self.assertRaises(bcpack.BcpackError):
                bcpack.decode(packed, codec, len(expected) - 1)


if __name__ == "__main__":
    unittest.main()
