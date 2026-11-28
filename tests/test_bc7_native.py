"""The optional Rust BC7 decoder (tools/bc7_native) against the Python one in tools/dds_png.py.

Synthetic random blocks only (no game data). The comparisons skip when the library is not built:

    cargo build --release --locked --offline --manifest-path tools/bc7_native/Cargo.toml
    python -I tests/test_bc7_native.py
"""
import importlib.util
import os
import random
import subprocess
import sys
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_module():
    path = os.path.join(ROOT, "tools", "dds_png.py")
    spec = importlib.util.spec_from_file_location("dds_png_bc7_native", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dds = load_module()
NATIVE = dds._native_bc7()


def random_blocks(rng, count, mode=None):
    """`count` random 16-byte blocks; `mode` 0-7 forces that mode, 8 the reserved one, None any."""
    out = bytearray(rng.randbytes(16 * count))
    if mode is not None:
        for i in range(0, len(out), 16):
            out[i] = 0 if mode == 8 else ((1 << mode) | (out[i] << (mode + 1))) & 0xFF
    return bytes(out)


def python_decode(data, offset, width, height):
    with mock.patch.object(dds, "_BC7_NATIVE", [None]):
        return dds._decode_bc7(data, offset, width, height)


@unittest.skipIf(NATIVE is None, "tools/bc7_native is not built")
class NativeMatchesPython(unittest.TestCase):
    def assert_same(self, data, width, height, offset=0):
        native = dds._decode_bc7(data, offset, width, height)
        expected = python_decode(data, offset, width, height)
        self.assertEqual(len(native), width * height * 4)
        if native != expected:
            px = next(i for i in range(len(native)) if native[i] != expected[i]) // 4
            self.fail(f"{width}x{height}: first difference at pixel ({px % width}, {px // width})")

    def test_every_mode(self):
        # 4,096 blocks per mode (and the reserved mode), 64x64 blocks each: 36,864 blocks.
        rng = random.Random(0xBC7)
        for mode in range(9):
            with self.subTest(mode=mode):
                self.assert_same(random_blocks(rng, 64 * 64, mode), 256, 256)

    def test_unconstrained_blocks(self):
        # 16,384 blocks with every byte random, so the mode follows the first byte's lowest set bit.
        rng = random.Random(7)
        self.assert_same(random_blocks(rng, 128 * 128), 512, 512)

    def test_edge_sizes_and_offset(self):
        rng = random.Random(11)
        for width, height in ((1, 1), (2, 3), (3, 5), (4, 4), (5, 4), (13, 7), (17, 33), (517, 259)):
            with self.subTest(size=(width, height)):
                count = ((width + 3) // 4) * ((height + 3) // 4)
                prefix = rng.randbytes(9)
                data = prefix + random_blocks(rng, count) + rng.randbytes(16)
                self.assert_same(memoryview(data), width, height, offset=len(prefix))

    def test_repeated_blocks(self):
        # Runs of the same block exercise the native decoder's previous-block reuse.
        rng = random.Random(3)
        pool = [random_blocks(rng, 1, m) for m in range(9)]
        data = b"".join(rng.choice(pool) * rng.randint(1, 9) for _ in range(800))
        count = len(data) // 16
        self.assert_same(data[:16 * (count // 40) * 40], 160, 4 * (count // 40))

    def test_truncated_payload_still_raises(self):
        with self.assertRaises(dds.DdsError):
            dds._decode_bc7(bytes(31), 0, 8, 4)

    def test_decode_dds_uses_the_native_path(self):
        rng = random.Random(5)
        width, height = 37, 21
        payload = random_blocks(rng, ((width + 3) // 4) * ((height + 3) // 4))
        header = bytearray(148)
        header[0:4] = b"DDS "
        header[4:8] = (124).to_bytes(4, "little")
        header[8:12] = (0x1007).to_bytes(4, "little")
        header[12:16] = height.to_bytes(4, "little")
        header[16:20] = width.to_bytes(4, "little")
        header[76:80] = (32).to_bytes(4, "little")
        header[80:84] = (0x4).to_bytes(4, "little")
        header[84:88] = b"DX10"
        header[128:132] = (98).to_bytes(4, "little")
        header[132:136] = (3).to_bytes(4, "little")
        header[140:144] = (1).to_bytes(4, "little")
        data = bytes(header) + payload
        with mock.patch.object(dds, "_decode_bc7_block", side_effect=AssertionError("Python path used")):
            native = dds.decode_dds(data)
        self.assertEqual(native[2], python_decode(payload, 0, width, height))


class Fallback(unittest.TestCase):
    def test_disabled_by_environment(self):
        # A fresh interpreter with LEONIDA_BC7_NATIVE=0 finds no library and decodes in Python.
        code = (
            "import importlib.util, random, sys\n"
            "spec = importlib.util.spec_from_file_location('d', sys.argv[1])\n"
            "d = importlib.util.module_from_spec(spec); spec.loader.exec_module(d)\n"
            "assert d._native_bc7() is None\n"
            "data = random.Random(1).randbytes(16 * 12)\n"
            "sys.stdout.write(d._decode_bc7(data, 0, 13, 9).hex())\n"
        )
        env = dict(os.environ, LEONIDA_BC7_NATIVE="0")
        result = subprocess.run([sys.executable, "-I", "-c", code, os.path.join(ROOT, "tools", "dds_png.py")],
                                env=env, capture_output=True, text=True, check=True)
        data = random.Random(1).randbytes(16 * 12)
        self.assertEqual(bytes.fromhex(result.stdout), python_decode(data, 0, 13, 9))

    def test_missing_library_falls_back(self):
        with mock.patch.object(dds, "_BC7_NATIVE", []), mock.patch.object(dds.os.path, "isfile",
                                                                          return_value=False):
            self.assertIsNone(dds._native_bc7())
            data = random.Random(2).randbytes(16 * 4)
            self.assertEqual(len(dds._decode_bc7(data, 0, 8, 8)), 8 * 8 * 4)

    def test_refused_call_falls_back(self):
        # A native refusal (-1) is not an error: Python decodes the surface.
        data = random.Random(4).randbytes(16 * 4)
        refuse = (lambda *args: -1, None, None, None)
        with mock.patch.object(dds, "_BC7_NATIVE", [refuse]):
            self.assertEqual(dds._decode_bc7(data, 0, 8, 8), python_decode(data, 0, 8, 8))


if __name__ == "__main__":
    unittest.main()
