"""Parity, fallback and timing checks for the optional Rust kernels of tools/glb_writer.py.

Every mesh here is generated in memory from a seeded random source; no game data is read.
Build the library first (``cargo build --release --manifest-path tools/glb_native/Cargo.toml``);
without it the parity tests are skipped and only the fallback tests run.
"""
from __future__ import annotations

import fractions
import math
import os
import random
import struct
import subprocess
import sys
import time
import unittest
from contextlib import contextmanager
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import glb_native  # noqa: E402
import glb_writer  # noqa: E402

LIB = glb_native._load()
OPTED_OUT = os.environ.get("LEONIDA_GLB_NATIVE") == "0"
needs_lib = unittest.skipIf(LIB is None or OPTED_OUT, "tools/glb_native is not built, or LEONIDA_GLB_NATIVE=0")


@contextmanager
def python_only():
    old = os.environ.get("LEONIDA_GLB_NATIVE")
    os.environ["LEONIDA_GLB_NATIVE"] = "0"
    try:
        yield
    finally:
        if old is None:
            del os.environ["LEONIDA_GLB_NATIVE"]
        else:
            os.environ["LEONIDA_GLB_NATIVE"] = old


def outcome(fn, *args):
    try:
        return ("ok", fn(*args))
    except Exception as exc:  # noqa: BLE001 - the class and message are what is compared
        return ("error", type(exc), str(exc))


def bits(value):
    return struct.unpack("<Q", struct.pack("<d", value))[0]


def same(a, b):
    """Equal results, with floats compared by bit pattern (so 0.0 and -0.0 differ)."""
    if type(a) is not type(b):
        return False
    if isinstance(a, float):
        return bits(a) == bits(b)
    if isinstance(a, dict):
        return list(a) == list(b) and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def rand_float(rng, spread):
    kind = rng.random()
    if kind < 0.02:
        return 0.0
    if kind < 0.04:
        return -0.0
    if kind < 0.06:
        return rng.uniform(-1.0, 1.0) * 2.0 ** rng.randint(-1074, -1000)
    if kind < 0.08:
        return rng.uniform(-1.0, 1.0) * 2.0 ** rng.randint(60, 127)
    return rng.uniform(-spread, spread)


def rand_primitive(rng, *, valid_bias=0.85):
    """A random parse_obj-style primitive; most are valid, some carry what Python rejects."""
    vertices = rng.choice((1, 2, 3, 4, 7, 16, 61, 200, 777))
    spread = rng.choice((1.0, 100.0, 5000.0, 1e-3))
    int_mode = rng.random() < 0.08
    positions = []
    for _ in range(vertices):
        if int_mode:
            positions.append(tuple(rng.choice((rng.randint(-50, 50), rng.uniform(-50, 50))) for _ in range(3)))
        else:
            positions.append(tuple(rand_float(rng, spread) for _ in range(3)))
    if rng.random() < 0.3:
        positions = [list(p) for p in positions]
    tris = rng.randint(1, max(1, vertices * 2))
    indices = [rng.randrange(vertices) for _ in range(tris * 3)]
    if rng.random() < 0.1 and vertices >= 3:
        indices += [0, 1, 1, 2, 2, 2]  # degenerate triangles
    prim = {"name": "p", "positions": positions, "indices": indices, "material": rng.choice(("m", 3, True, None, ""))}
    choice = rng.random()
    if choice < 0.4:
        prim["normals"] = [tuple(rand_float(rng, rng.choice((1.0, 1e-300, 1e300)))
                                 for _ in range(3)) for _ in range(vertices)]
    elif choice < 0.5:
        prim["normals"] = []
    elif choice < 0.55:
        prim["normals"] = None
    if rng.random() < 0.5:
        prim["uvs"] = [(rand_float(rng, 4.0), rand_float(rng, 4.0)) for _ in range(vertices)]
    if rng.random() < 0.3:
        prim["material_index"] = rng.randint(0, 4)
    if rng.random() > valid_bias:
        breakage = rng.randrange(12)
        if breakage == 0:
            prim["indices"] = indices + [vertices]
        elif breakage == 1:
            prim["indices"] = indices[:-1]
        elif breakage == 2:
            prim["indices"] = indices + [-1, 0, 0]
        elif breakage == 3:
            positions[rng.randrange(vertices)] = (math.inf, 0.0, 0.0)
        elif breakage == 4:
            positions[rng.randrange(vertices)] = (1e39, 0.0, 0.0)
        elif breakage == 5:
            positions[rng.randrange(vertices)] = (1.0, 2.0)
        elif breakage == 6 and "uvs" in prim:
            prim["uvs"][0] = (1e300, 0.0)
        elif breakage == 7:
            prim["normals"] = [(0.0, 0.0, 1.0)] * (vertices + 1)
        elif breakage == 8:
            prim["uvs"] = [(0.0, 0.0)] * (vertices + 2)
        elif breakage == 9:
            prim["indices"] = indices + [0.0, 0, 0]
        elif breakage == 10:
            positions[0] = (2 ** 30, 1, 0)
        else:
            prim["indices"] = []
    return prim


class NativeParityTests(unittest.TestCase):
    def assert_parity(self, fn, *args):
        native = outcome(fn, *args)
        with python_only():
            reference = outcome(fn, *args)
        self.assertTrue(same(native, reference), f"{fn.__name__}: native {native!r} != Python {reference!r}")
        return native

    @needs_lib
    def test_random_primitives_match_python(self):
        rng = random.Random(20261123)
        taken = declined = errors = 0
        for case in range(1500):
            prim = rand_primitive(rng)
            result = self.assert_parity(glb_writer.pack_primitive, prim)
            if result[0] == "error":
                errors += 1
            elif glb_native.try_pack(prim) is not None:
                taken += 1
            else:
                declined += 1
        print(f"\n  pack_primitive: {taken} native, {declined} declined, {errors} Python errors, all identical")
        self.assertGreater(taken, 1000)

    @needs_lib
    def test_api_function_readers_match_python(self):
        """The readers used when the layout probe fails give the same results."""
        api = LIB[1]
        self.assertEqual(api.direct, 1, "the layout probe should pass on a default CPython build")
        rng = random.Random(77)
        api.direct = 0
        try:
            for case in range(400):
                prim = rand_primitive(rng)
                self.assert_parity(glb_writer.pack_primitive, prim)
                self.assert_parity(glb_writer.compute_bounds, [prim])
        finally:
            api.direct = 1

    def test_layout_probe_rejects_a_wrong_layout(self):
        if LIB is None or OPTED_OUT:
            self.skipTest("tools/glb_native is not built, or LEONIDA_GLB_NATIVE=0")
        lib, api = LIB
        wrong = glb_native._PyApi.from_buffer_copy(api)
        wrong.tuple_items = api.tuple_items - 8  # reads ob_size where the first item should be
        self.assertFalse(glb_native._layout_agrees(lib, wrong))
        self.assertTrue(glb_native._layout_agrees(lib, api))

    @needs_lib
    def test_random_bounds_and_builds_match_python(self):
        rng = random.Random(8683)
        for case in range(300):
            prims = [rand_primitive(rng, valid_bias=1.0) for _ in range(rng.randint(1, 4))]
            if rng.random() < 0.2:
                prims[0]["positions"] = [(math.nan, 0.0, 0.0), (math.inf, 1.0, 2.0)]
            self.assert_parity(glb_writer.compute_bounds, prims)
            self.assert_parity(glb_writer._build_glb_with_stats, prims, None)
        self.assertIsNotNone(glb_native.try_bounds([{"positions": [(1.0, -0.0, 2.0)]}]))
        self.assertIs(glb_native.try_bounds([{"positions": [(math.nan, 0.0, 0.0)]}, {}]), glb_native.NO_BOUNDS)

    @needs_lib
    def test_one_shot_iterator_of_primitives_matches_python(self):
        def prims():
            return [
                {"positions": [(5.0, 6.0, 7.0), (-1.0, 2.0, 3.0)]},
                {"positions": iter([(1.0, 1.0, 1.0)])},  # native declines this one after reading the first
                {"positions": [(9.0, -9.0, 0.5)]},
            ]

        native = outcome(lambda: glb_writer.compute_bounds(iter(prims())))
        with python_only():
            reference = outcome(lambda: glb_writer.compute_bounds(iter(prims())))
        self.assertEqual(native[0], "ok")
        self.assertTrue(same(native, reference), f"native {native!r} != Python {reference!r}")
        self.assertIsNone(glb_native.try_bounds(iter(prims())))

    @needs_lib
    def test_uint32_indices(self):
        rng = random.Random(5)
        vertices = 70000
        prim = {
            "positions": [(rng.uniform(-9, 9), rng.uniform(-9, 9), rng.uniform(-9, 9)) for _ in range(vertices)],
            "indices": [rng.randrange(vertices) for _ in range(3 * 30000)] + [vertices - 1, 0, 1],
            "uvs": [(rng.random(), rng.random()) for _ in range(vertices)],
        }
        self.assertIsNotNone(glb_native.try_pack(prim))
        packed = self.assert_parity(glb_writer.pack_primitive, prim)[1]
        self.assertEqual(packed["index_type"], glb_writer.COMPONENT_UNSIGNED_INT)

    @needs_lib
    def test_edge_inputs_match_python_or_decline(self):
        tri = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
        cases = [
            {"positions": tri, "indices": [0, 1, 2]},
            {"positions": iter(tri), "indices": iter([0, 1, 2])},
            {"positions": [], "indices": []},
            {"positions": tri, "indices": []},
            {"positions": tri, "indices": [True, 1, 2]},
            {"positions": [(True, 0, 0), (1, 0, 0), (0, 1, 0)], "indices": [0, 1, 2]},
            {"positions": [(2 ** 24, 0, 0), (0, 2 ** 24, 0), (0, 0, 2 ** 24)], "indices": [0, 1, 2]},
            {"positions": [(2 ** 40 + 1, 0, 0), (0, 2 ** 40, 0), (0, 0, 2 ** 40)], "indices": [0, 1, 2]},
            {"positions": [(10 ** 400, 0, 0)] + tri[1:], "indices": [0, 1, 2]},
            {"positions": [(fractions.Fraction(1, 3), 0, 0)] + tri[1:], "indices": [0, 1, 2]},
            {"positions": [("1", 0.0, 0.0)] + tri[1:], "indices": [0, 1, 2]},
            {"positions": [(0.0, 0.0, 0.0, 9.0)] + tri[1:], "indices": [0, 1, 2]},
            {"positions": tri, "indices": [0, 1, 2], "normals": [("1", "0", "0")] * 3},
            {"positions": tri, "indices": [0, 1, 2], "normals": [(1, 0, 0, 7)] * 3},
            {"positions": tri, "indices": [0, 1, 2], "normals": [(1e308, 1e308, 1e308)] * 3},
            {"positions": tri, "indices": [0, 1, 2], "normals": [(5e-324, -5e-324, 0.0)] * 3},
            {"positions": tri, "indices": [0, 1, 2], "uvs": [(math.nan, -math.inf)] * 3},
            {"positions": tri, "indices": [0, 1, 2], "uvs": [(-0.0, 3.4e38)] * 3},
            {"positions": tri, "indices": [0, 1, 2], "uvs": [(0.5,)] * 3},
            {"positions": [(1e30, 0.0, 0.0), (-1e30, 1e30, 0.0), (0.0, 0.0, 1e30)], "indices": [0, 1, 2]},
            {"positions": tri, "indices": [0, 1, 2], "material": 1.5},
        ]
        for prim in cases:
            with self.subTest(prim=repr(prim)[:120]):
                copy = dict(prim)
                if not isinstance(prim["positions"], list):
                    copy["positions"], copy["indices"] = list(tri), [0, 1, 2]
                    prim = {"positions": iter(tri), "indices": iter([0, 1, 2])}
                    native = outcome(glb_writer.pack_primitive, prim)
                    with python_only():
                        reference = outcome(glb_writer.pack_primitive, copy)
                    self.assertTrue(same(native, reference))
                    continue
                self.assert_parity(glb_writer.pack_primitive, prim)


class FallbackTests(unittest.TestCase):
    PRIM = {"positions": [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)], "indices": [0, 1, 2]}

    def test_opt_out_skips_the_library(self):
        with python_only(), mock.patch.object(glb_native, "_load", side_effect=AssertionError("loaded")):
            self.assertIsNone(glb_native.try_pack(self.PRIM))
            self.assertIsNone(glb_native.try_bounds([self.PRIM]))
            glb_writer.pack_primitive(self.PRIM)
            glb_writer.compute_bounds([self.PRIM])

    def test_missing_library_runs_python(self):
        with python_only():
            expected = glb_writer.build_glb([self.PRIM])
        with mock.patch.object(glb_native, "_load", return_value=None):
            self.assertEqual(glb_writer.build_glb([self.PRIM]), expected)

    def test_library_that_disagrees_on_hypot_is_not_used(self):
        self.assertFalse(glb_native._hypot_agrees(lambda x, y, z: math.hypot(x, y, z) + 1.0))

    def test_unloadable_library_path(self):
        glb_native._load.cache_clear()
        try:
            with mock.patch.object(glb_native, "_library_name", return_value="missing-glb-native.bin"):
                self.assertIsNone(glb_native._load())
        finally:
            glb_native._load.cache_clear()

    def test_refusal_code_runs_python(self):
        fake = mock.Mock()
        fake.glb_pack_obj_v1.return_value = 1
        fake.glb_bounds_obj_v1.return_value = 1
        with mock.patch.object(glb_native, "_load", return_value=(fake, glb_native._PyApi())):
            self.assertIsNone(glb_native.try_pack(self.PRIM))
            self.assertIsNone(glb_native.try_bounds([self.PRIM]))
            with python_only():
                expected = glb_writer.pack_primitive(self.PRIM)
            self.assertEqual(glb_writer.pack_primitive(self.PRIM), expected)

    def test_writer_without_the_package(self):
        code = ("import sys; sys.path.insert(0, 'tools'); sys.modules['glb_native'] = None; import glb_writer; "
                "assert glb_writer._glb_native is None; "
                "print(len(glb_writer.build_glb([{'positions': [(0,0,0),(1,0,0),(0,1,0)], 'indices': [0,1,2]}])))")
        out = subprocess.run([sys.executable, "-I", "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
        self.assertTrue(out.stdout.strip().isdigit())


def grid_primitive(rng, side, *, normals, uvs):
    positions = [(x * 0.25 + rng.uniform(-0.01, 0.01), rng.uniform(-1.0, 1.0), z * 0.25)
                 for z in range(side) for x in range(side)]
    indices = []
    for z in range(side - 1):
        for x in range(side - 1):
            a = z * side + x
            indices += (a, a + side, a + 1, a + 1, a + side, a + side + 1)
    prim = {"positions": positions, "indices": indices, "material": "m"}
    if normals:
        prim["normals"] = [(rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1)) for _ in positions]
    if uvs:
        prim["uvs"] = [(p[0], p[2]) for p in positions]
    return prim


@needs_lib
class Benchmark(unittest.TestCase):
    def test_timing(self):
        rng = random.Random(1)
        rows = []
        for label, normals, uvs in (("pack_primitive, 65,536 vertices, smooth normals", False, False),
                                    ("pack_primitive, 65,536 vertices, smooth normals + UV", False, True),
                                    ("pack_primitive, 65,536 vertices, given normals + UV", True, True)):
            prim = grid_primitive(rng, 256, normals=normals, uvs=uvs)  # 65,536 vertices, 130,050 triangles
            times = {}
            for mode in ("python", "native", "python", "native", "python", "native"):
                if mode == "python":
                    with python_only():
                        start = time.perf_counter()
                        ref = glb_writer.pack_primitive(prim)
                        elapsed = time.perf_counter() - start
                else:
                    start = time.perf_counter()
                    got = glb_writer.pack_primitive(prim)
                    elapsed = time.perf_counter() - start
                times[mode] = min(times.get(mode, math.inf), elapsed)
            self.assertTrue(same(got, ref))
            rows.append((label, times["python"], times["native"]))
        for label, normals, uvs in (("build_glb, 4 x 16,384 vertices, smooth normals", False, False),
                                    ("build_glb, 4 x 16,384 vertices, given normals + UV", True, True)):
            prims = [grid_primitive(rng, 128, normals=normals, uvs=uvs) for _ in range(4)]
            times = {}
            for mode in ("python", "native") * 3:
                if mode == "python":
                    with python_only():
                        start = time.perf_counter()
                        ref = glb_writer._build_glb_with_stats(prims)
                        elapsed = time.perf_counter() - start
                else:
                    start = time.perf_counter()
                    got = glb_writer._build_glb_with_stats(prims)
                    elapsed = time.perf_counter() - start
                times[mode] = min(times.get(mode, math.inf), elapsed)
            self.assertTrue(same(got, ref))
            rows.append((label, times["python"], times["native"]))
        print()
        for label, py, nat in rows:
            print(f"  {label}: Python {py * 1e3:.1f} ms, native {nat * 1e3:.2f} ms, {py / nat:.1f}x")
            self.assertLess(nat, py)


if __name__ == "__main__":
    unittest.main(verbosity=2)
