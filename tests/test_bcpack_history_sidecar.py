"""tools/bcpack_history_sidecar.py writes the BC6H/BC7 history sidecar the runtime decoder reads (synthetic history only).

The layout is the one runtime/src/bcpack.cpp's read_history_sidecar reads: "BCPHIST1", 256 BC6H entries of 70 bytes,
256 BC7 entries of 168. These checks put known values into a made-up history and find them at their offsets, so a
change on either side shows here; runtime/test/bcpack_test.cpp decodes with a sidecar of this tool against
tools/bcpack.py with the same history. No game files.
"""
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "bcpack_history_sidecar.py")
sys.path.insert(0, os.path.join(ROOT, "tools"))
import bcpack_history_sidecar as sidecar  # noqa: E402

BC6H, BC7 = 70, 168
BC7_BASE = 8 + 256 * BC6H


def _history():
    bc6h = {
        "endpointHistory": {"5": [[0x1234, 0x2AB, 0x15, 0x2A], [1, 2, 3, 4], [65535, 1023, 63, 0]]},
        "indexHistory": {"7": list(range(16))},
        "endpointClass": {"5": 3},
        "indexClass": {"7": 1},
        "endpointConfirmedBits": {"5": [[16, 10, 6, 6], [8, 4, 2, 1], [0, 0, 0, 0]]},
        "indexConfirmedBits": {"7": [4] * 15 + [2]},
    }
    bc7 = {
        "historyEndpoints": {"9": [i * 10 for i in range(24)]},
        "historyEndpointBits": {"9": [8] * 23 + [3]},
        "historyEndpointConfirmedBits": {"9": [4] * 24},
        "historyIndices": {"11": [i % 16 for i in range(32)]},
        "historyIndexBits": {"11": [4] * 32},
        "historyIndexConfirmedBits": {"11": [2] * 32},
    }
    return {"bc6h": bc6h, "bc7": bc7}


class SidecarLayout(unittest.TestCase):
    def test_empty_history_is_all_unknown(self):
        data = sidecar.render(None)
        self.assertEqual(len(data), sidecar.SIZE)
        self.assertEqual(len(data), 60936)
        self.assertEqual(data[:8], b"BCPHIST1")
        self.assertFalse(any(data[8:]), "no history: every value, depth and class is 0")

    def test_bc6h_entry(self):
        data = sidecar.render(_history())
        at = 8 + 5 * BC6H
        self.assertEqual(struct.unpack_from("<12H", data, at),
                         (0x1234, 0x2AB, 0x15, 0x2A, 1, 2, 3, 4, 65535, 1023, 63, 0))
        self.assertEqual(tuple(data[at + 40:at + 52]), (16, 10, 6, 6, 8, 4, 2, 1, 0, 0, 0, 0))
        self.assertEqual(data[at + 68], 3, "endpoint class")
        at = 8 + 7 * BC6H
        self.assertEqual(tuple(data[at + 24:at + 40]), tuple(range(16)), "index history")
        self.assertEqual(tuple(data[at + 52:at + 68]), (4,) * 15 + (2,), "index confirmed bits")
        self.assertEqual(data[at + 69], 1, "index class")

    def test_bc7_entry(self):
        data = sidecar.render(_history())
        at = BC7_BASE + 9 * BC7
        endpoints = tuple(data[at:at + 24])
        # load_bc7's mid-range fill: a value known to d < 8 leading bits keeps them and sets the next one.
        self.assertEqual(endpoints[:23], tuple(i * 10 for i in range(23)))
        self.assertEqual(endpoints[23], ((230 >> 5) << 5) | (1 << 4))
        self.assertEqual(tuple(data[at + 24:at + 48]), (8,) * 23 + (3,), "endpoint bits")
        self.assertEqual(tuple(data[at + 48:at + 72]), (4,) * 24, "endpoint confirmed bits")
        at = BC7_BASE + 11 * BC7
        self.assertEqual(tuple(data[at + 72:at + 104]), tuple(i % 16 for i in range(32)), "indices")
        self.assertEqual(tuple(data[at + 104:at + 136]), (4,) * 32, "index bits")
        self.assertEqual(tuple(data[at + 136:at + 168]), (2,) * 32, "index confirmed bits")

    def test_command_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "history.json")
            with open(src, "w", encoding="utf-8", newline="\n") as f:
                json.dump(_history(), f)
            out = os.path.join(tmp, "bcpack_history.bin")
            r = subprocess.run([sys.executable, "-I", TOOL, src, out], capture_output=True, text=True, timeout=120)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            with open(out, "rb") as f:
                self.assertEqual(f.read(), sidecar.render(_history()))
            r = subprocess.run([sys.executable, "-I", TOOL, src, src], capture_output=True, text=True, timeout=120)
            self.assertEqual(r.returncode, 2, "a sidecar over its own input is refused")
            r = subprocess.run([sys.executable, "-I", TOOL, os.path.join(tmp, "missing.json"), out],
                               capture_output=True, text=True, timeout=120)
            self.assertEqual(r.returncode, 2, "a missing history is an error, not a traceback")


if __name__ == "__main__":
    unittest.main()
