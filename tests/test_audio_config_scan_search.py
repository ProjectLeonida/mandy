"""Synthetic equivalence and sparse-input work checks for audio config scans."""
import os
import random
import struct
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "tools"))
import audio_config as a


class ObservedBytes(bytes):
    def __new__(cls, data):
        obj = super().__new__(cls, data)
        obj.slices = 0
        return obj

    def __getitem__(self, item):
        obj = super().__getitem__(item)
        if isinstance(item, slice):
            self.slices += 1
        return obj


class ScanSearchTest(unittest.TestCase):
    def test_sparse_data_does_not_slice_every_byte(self):
        data = ObservedBytes(b"x" * 100000)
        self.assertEqual(a.scan(data), [])
        self.assertLess(data.slices, 10)

    def test_scan_agrees_with_stream_across_boundaries(self):
        rng = random.Random(472)
        for size in (0, 23, 24, 50, 127, 1024):
            for iteration in range(20):
                blob = bytearray(rng.randbytes(size))
                for _ in range(6):
                    if len(blob) < 24:
                        break
                    i = rng.randrange(4, len(blob) - 19)
                    blob[i-4:i] = struct.pack("<I", 30000)
                    blob[i:i+2] = a.MARKER_1
                    blob[i+4:i+8] = a.MARKER_2
                    blob[i+16:i+20] = a.MARKER_3
                data = bytes(blob)
                chunks = [data[i:i+7] for i in range(0, len(data), 7)]
                with self.subTest(size=size, iteration=iteration):
                    self.assertEqual(a.scan(data), list(a.scan_stream(chunks)))


if __name__ == "__main__":
    unittest.main()
