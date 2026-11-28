"""Byte-for-byte Rust/Python BVH rehydration checks using hand-built inputs.

    cargo build --release --locked --offline --manifest-path tools/bvh_native/Cargo.toml
    python -I tests/test_bvh_native.py
"""
import os
import random
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

# These tests set the per-module switch themselves; the global LEONIDA_NATIVE=0 opt-out would override it.
os.environ.pop("LEONIDA_NATIVE", None)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_bvh_rehydrate import BLAS_DEHYDRATED, QUADS, R, box_group, build, header, leaf_chunk, level_table, ptr
import bvh_native as native


ORDERS = (0x01, 0x02, 0x04, 0x06, 0x08, 0x09)


def tree(leaves, rng, compression=R.NO_TRANSFORM, root_wide=True, updatable=True, overrides=True):
    """Build mixed BOX16/BOX32 levels over leaves, bottom-up in groups of up to 64 nodes."""
    # Room for all parent entries, including the entries of both units of a wide node.
    elements = ((64 + (2 * len(leaves) + 8) * 4 + 63) // 64) * 64
    nodes = elements + 64 * len(leaves)
    hierarchy = bytearray()
    levels = []
    children = [ptr(elements + 64 * i, R.TRI1 if leaf[1] & 0x80 and i & 1 else R.TRI0)
                for i, leaf in enumerate(leaves)]

    while children:
        groups = [children[i:i + 4] for i in range(0, len(children), 4)]
        count = len(groups)
        wide_count = (count + 1) // 2 if root_wide else count // 2
        children = []
        level_ends = []
        for wide, boxes in ((True, groups[:wide_count]), (False, groups[wide_count:])):
            node_size = 128 if wide else 64
            for start in range(0, len(boxes), 64):
                batch = []
                for kids in boxes[start:start + 64]:
                    values = {}
                    if overrides:
                        for bit in rng.sample(range(6 * len(kids)), rng.randrange(min(4, 6 * len(kids)))):
                            value = rng.uniform(-2.0, 2.0)
                            values[bit] = value if wide else R.float_to_half(value)
                    children.append(ptr(nodes + len(hierarchy) + node_size * len(batch),
                                        R.BOX32 if wide else R.BOX16))
                    batch.append((kids, values))
                packed = box_group(batch, wide)
                hierarchy.extend(packed.ljust(node_size * len(batch), b"\0"))
            level_ends.append(len(hierarchy))
        levels.append(level_ends)
        if len(children) == 1:
            break

    table = level_table(levels)
    src = bytearray((nodes + len(hierarchy) + len(table) + 63) // 64 * 64)
    flags = BLAS_DEHYDRATED | (R.FLAG_UPDATABLE if updatable else 0)
    src[:64] = header(len(src), len(leaves), (1, -4, -3, 4, -1, 2), nodes, children[0], elements, 64,
                      compression, flags)
    for i in range(0, len(leaves), R.LEAVES_PER_CHUNK):
        packed = leaf_chunk(leaves[i:i + R.LEAVES_PER_CHUNK])
        start = elements + 64 * i
        src[start:start + len(packed)] = packed
    src[nodes:nodes + len(hierarchy)] = hierarchy
    src[-len(table):] = table
    return bytes(src)


def python_rehydrate(src, geometries, offset=0):
    with mock.patch.object(R, "_try_native", return_value=None):
        return R.rehydrate(src, geometries, offset)


def result_or_error(rehydrate, src, geometries):
    try:
        return rehydrate(src, geometries)
    except (IndexError, OverflowError, TypeError, ValueError, struct.error) as error:
        return type(error), str(error)


class TestFallback(unittest.TestCase):
    def setUp(self):
        self.src, _, _ = build([(0, 1, 0, None, 0x09)])

    def test_missing_library(self):
        native._load.cache_clear()
        self.addCleanup(native._load.cache_clear)
        with tempfile.TemporaryDirectory() as folder:
            with mock.patch.object(native, "__file__", os.path.join(folder, "__init__.py")):
                self.assertIsNone(native._load())
                self.assertEqual(R.rehydrate(self.src, iter([QUADS])), python_rehydrate(self.src, [QUADS]))

    def test_disabled_library(self):
        with mock.patch.dict(os.environ, LEONIDA_BVH_NATIVE="0"), mock.patch.object(native, "_load") as load:
            self.assertEqual(R.rehydrate(self.src, [QUADS]), python_rehydrate(self.src, [QUADS]))
            load.assert_not_called()

    def test_refused_and_failed_calls_leave_output_clean(self):
        def dirty_output(*args):
            args[-1][0] = 0xff
            return -1

        expected = python_rehydrate(self.src, [QUADS])
        for failure in (dirty_output, OSError("test FFI failure")):
            with self.subTest(failure=failure), mock.patch.dict(os.environ, LEONIDA_BVH_NATIVE="1"):
                with mock.patch.object(native, "_load") as load:
                    load.return_value.side_effect = failure
                    self.assertEqual(R.rehydrate(self.src, iter([QUADS])), expected)
                    load.return_value.assert_called_once()

    def test_custom_geometry_uses_python(self):
        class CustomGeometry(R.Geometry):
            def vertex(self, index):
                return 1.0, 2.0, 3.0

        geometry = CustomGeometry(QUADS.vertices)
        expected = python_rehydrate(self.src, [geometry])
        with mock.patch.object(native, "_load") as load:
            self.assertEqual(R.rehydrate(self.src, [geometry]), expected)
            load.return_value.assert_not_called()


class TestNativeParity(unittest.TestCase):
    def setUp(self):
        if os.environ.get("LEONIDA_BVH_NATIVE") == "0":
            self.skipTest("native BVH path is disabled")
        self.decode = native._load()
        if self.decode is None:
            self.skipTest("build tools/bvh_native to run the native parity checks")

    def assert_native(self, src, geometries, offset=0):
        expected = python_rehydrate(src, geometries, offset)
        with mock.patch.object(R, "write_leaves", side_effect=AssertionError("unexpected Python fallback")):
            actual = R.rehydrate(src, iter(geometries), offset)
        self.assertIs(type(actual), bytearray)
        self.assertEqual(actual, expected)

    def assert_fallback(self, src, geometries):
        with mock.patch.object(native, "_load", return_value=mock.Mock(wraps=self.decode)) as load:
            with mock.patch.object(R, "write_leaves", wraps=R.write_leaves) as write_leaves:
                actual = result_or_error(R.rehydrate, src, iter(geometries))
                write_leaves.assert_called_once()
            load.return_value.assert_called_once()
        self.assertEqual(actual, result_or_error(python_rehydrate, src, geometries))

    def test_randomized_trees(self):
        rng = random.Random(0x425648)
        geometries = []
        for i in range(24):
            stride = rng.choice((12, 16, 23))
            vertices = b"".join(struct.pack("<3f", *(rng.uniform(-10.0, 10.0) for _ in range(3))).ljust(stride, b"X")
                                for _ in range(24))
            index16 = i % 3 == 1
            indices = None if i % 3 == 0 else struct.pack("<24" + ("H" if index16 else "I"),
                                                          *rng.sample(range(24), 24))
            transform = None if i & 1 else tuple(tuple(rng.uniform(-1.0, 1.0) for _ in range(4)) for _ in range(3))
            geometries.append(R.Geometry(vertices, stride, indices, index16, transform))

        for trial in range(2048):
            leaves = []
            for _ in range(rng.randrange(1, 20)):
                paired = bool(rng.getrandbits(1))
                flags = rng.randrange(128) | (0x80 if paired else 0)
                leaves.append((rng.randrange(len(geometries)), flags, rng.randrange(8),
                               rng.randrange(8) if paired else None, rng.choice(ORDERS) | rng.randrange(16) << 4))
            compression = R.NO_TRANSFORM
            if trial % 4:
                compression = ((trial % 64) << 24) | ((trial % 4) << 30)
                for axis in range(3):
                    compression |= rng.randrange(122, 133) << (8 * axis)
            src = tree(leaves, rng, compression, root_wide=bool(trial & 1), updatable=bool(trial & 2))
            prefix = bytes(trial % 9)
            with self.subTest(trial=trial):
                self.assert_native(prefix + src, geometries, len(prefix))

    def test_isolated_cli_loads_the_library(self):
        src, _, _ = build([(0, 1, 0, None, 0x09)])
        expected = python_rehydrate(src, [QUADS])
        probe = """
import ctypes
import runpy
import sys

class ProbedLibrary(ctypes.CDLL):
    def __getattr__(self, symbol):
        function = super().__getattr__(symbol)
        if symbol == "bvh_rehydrate_v1":
            print("loaded the BVH native entry point")
        return function

ctypes.CDLL = ProbedLibrary
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
"""
        with tempfile.TemporaryDirectory() as folder:
            source = os.path.join(folder, "source.bin")
            vertices = os.path.join(folder, "vertices.bin")
            output = os.path.join(folder, "output.bin")
            with open(source, "wb") as stream:
                stream.write(src)
            with open(vertices, "wb") as stream:
                stream.write(QUADS.vertices)
            result = subprocess.run([sys.executable, "-I", "-c", probe, os.path.abspath(R.__file__), source,
                                     "--geom", vertices, "--out", output],
                                    check=True, text=True, capture_output=True)
            self.assertIn("loaded the BVH native entry point", result.stdout)
            with open(output, "rb") as stream:
                self.assertEqual(stream.read(), expected)

    def test_empty_structure_with_no_geometry(self):
        src = header(64, 0, (0, 0, 0, 1, 1, 1), 0, R.TRI0, 64, 64)
        self.assert_native(src, [])

    def test_multiple_leaf_chunks_and_box_groups(self):
        rng = random.Random(0x4096)
        leaves = [(0, 0x81, i % 2, (i + 1) % 2, ORDERS[i % len(ORDERS)]) for i in range(4225)]
        self.assert_native(tree(leaves, rng, compression=0xcb7d7e7f), [QUADS])

    def test_half_rounding_and_nonfinite_bounds(self):
        points = (2 ** -25, -2 ** -25, -0.0, 1 + 2 ** -11, -1 - 2 ** -11, 0.0,
                  65504.0, -65504.0, float("inf"), -float("inf"), 2 ** -14, -2 ** -14)
        geometry = R.Geometry(struct.pack("<12f", *points), indices=QUADS.indices)
        leaves = [(0, 0x81, 0, 1, 0x09)] * 8
        for root_wide in (True, False):
            self.assert_native(tree(leaves, random.Random(24), root_wide=root_wide, overrides=False), [geometry])
        # BOX16 overrides include signed zeros, subnormals and infinities. A BOX32 parent reads
        # them back, exercising the reduction's tie order.
        self.assert_native(self.half_override_tree(dict(enumerate((0x8000, 0x0001, 0x8001, 0xfc00, 0x7c00, 0xfbff)))),
                           [QUADS])

    def half_override_tree(self, overrides):
        src = bytearray(tree([(0, 1, 0, None, 0x09)] * 8, random.Random(24), overrides=False))
        hdr = R.Header(src)
        narrow = hdr.offset_to_nodes + 128
        data = box_group([([ptr(hdr.offset_to_elements + 64 * i, R.TRI0) for i in range(4, 8)], overrides)], False)
        src[narrow:narrow + len(data)] = data
        return bytes(src)

    def test_half_nan_bounds_fall_back_to_python(self):
        # struct.unpack("<e") canonicalizes half NaN payloads on CPython <= 3.12 and keeps them on 3.14,
        # so a stream with a half NaN is left to whichever Python is running.
        for nan in (0x7c01, 0x7d01, 0x7e00, 0x7e01, 0xfc01, 0xfe01, 0xfeff, 0xffff):
            with self.subTest(nan=hex(nan)):
                self.assert_fallback(self.half_override_tree({3: 0x0001, 11: nan, 17: 0x8000}), [QUADS])

    def test_fallback_on_float_edges(self):
        src, _, _ = build([(0, 1, 0, None, 0x09)], root_wide=False)
        for bits in (0x7fa12345, 0xffc12345):
            vertex = struct.pack("<3I", bits, 0x80000000, 0)
            self.assert_fallback(src, [R.Geometry(vertex * 3)])
        wide, _, _ = build([(0, 1, 0, None, 0x09)], overrides={0: float("nan")})
        self.assert_fallback(wide, [QUADS])
        # The inwards half bound may change with the Python half-range correction.
        self.assert_fallback(src, [R.Geometry(struct.pack("<3f", 70000.0, -70000.0, 0.0) * 3)])
        # Python float32 conversion can raise on a finite double product.
        overflow = ((2.0 ** 1023, 0.0, 0.0, 0.0), (0, 1, 0, 0), (0, 0, 1, 0))
        self.assert_fallback(src, [R.Geometry(struct.pack("<3f", 1.0, 0.0, 0.0) * 3, transform=overflow)])
        inf = R.Geometry(struct.pack("<3f", float("inf"), 0.0, 0.0) * 3, transform=overflow)
        self.assert_fallback(src, [inf])

    def test_truncations_and_bad_records_keep_python_errors(self):
        src, _, _ = build([(0, 1, 0, None, 0x09)])
        self.assertEqual(result_or_error(R.rehydrate, src[:-1], [QUADS]),
                         result_or_error(python_rehydrate, src[:-1], [QUADS]))
        self.assert_fallback(src, [R.Geometry(QUADS.vertices, indices=QUADS.indices[:11])])
        self.assert_fallback(src, [R.Geometry(QUADS.vertices[:35])])
        self.assert_fallback(src, [])
        self.assert_fallback(build([(0, 1, 0, None, 0xff)])[0], [QUADS])
        for at, value in ((52, len(src) - 1), (40, len(src) - 3), (len(src) - 4, 0xffffffff), (0x200, 5 << 24)):
            damaged = bytearray(src)
            struct.pack_into("<I", damaged, at, value)
            self.assert_fallback(bytes(damaged), [QUADS])


if __name__ == "__main__":
    unittest.main()
