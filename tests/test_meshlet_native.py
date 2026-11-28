"""Synthetic bit and summary parity for the optional meshlet kernels."""

import json
import os
from pathlib import Path
import random
import struct
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

# These tests set the per-module switch themselves; the global LEONIDA_NATIVE=0 opt-out would override it.
os.environ.pop("LEONIDA_NATIVE", None)
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "tests")]

import meshlet_native as native
import meshlet_page as mp
from test_meshlet_page import block, other_block, page


ALL = dict(faces=True, centres=True, normals=True, cones=True)
TAGS = ((0, 0x42, 8), (2, 0x21, 4), (11, 0x27, 4), (18, 0x27, 4))


def mesh_group(meshlets):
    """Pack (refs, local triangles, radius bits, center words) for each meshlet."""
    records, triangles, refs = bytearray(), bytearray(), bytearray()
    for i, (vertices, faces, radius, xy, z_cone) in enumerate(meshlets):
        counts = (len(vertices) - 1) | ((len(faces) - 1) << 7) | (radius << 16)
        bases = ((len(refs) // 2) & 65535) | (((len(triangles) // 4) & 65535) << 16)
        records.extend(struct.pack("<4I", counts, bases, xy, z_cone))
        for a, b, c in faces:
            triangles.extend(struct.pack("<I", a | (b << 7) | (c << 14) | (i << 21)))
        for ref in vertices:
            refs.extend(struct.pack("<H", ref))
    return struct.pack("<3I", 0x80000000 | len(meshlets), len(triangles) // 4, len(refs) // 2) + records + triangles + refs


def sample(rng, i):
    n = rng.randrange(3, 33)
    mode = (0, 1, 7)[i % 3]
    tags = ((0, 0x42, 8), (11, 0x27, 4)) if mode == 0 else (
        (0, 0x42, 8), (2, 0x20 | mode, 4), (11, 0x27, 4), (18, 0x27, 4))
    bias = tuple(rng.uniform(-20, 20) for _ in range(3))
    scale = tuple(rng.uniform(-0.01, 0.01) for _ in range(3))
    if i % 11 == 0:
        bias = (-0.0, 0.0, -0.0)
        scale = (-0.0, -(2 ** -149), 3.4028234663852886e38)
    if i % 19 == 0:
        bias = (float("nan"), float("inf"), float("-inf"))
    words = [(rng.randrange(1 << 24), rng.randrange(1 << 24), rng.randrange(1 << 16)) for _ in range(n)]
    packed = bytearray(other_block(n, tags, axis=i % 3, packed=words, bias=bias, scale=scale))
    # Use arbitrary IEEE bit patterns and independent normal/tangent fields.
    packed[128 + 8 * n:] = rng.randbytes(len(packed) - (128 + 8 * n))
    if i % 97 == 0:
        struct.pack_into("<I", packed, 0x60, 0xFFA1_1234)
    groups = []
    for _ in range(1 + (i % 4 == 0)):
        meshlets = []
        for _ in range(rng.randrange(1, 6)):
            vertices = [n - 1] + [rng.randrange(n) for _ in range(rng.randrange(1, 10))]
            nv = len(vertices)
            faces = [(0, 0, 0)] + [tuple(rng.randrange(nv) for _ in range(3)) for _ in range(rng.randrange(1, 18))]
            radius = rng.randrange(0x7c00) if i % 23 else rng.choice((0x7c00, 0xfc00, 0x7e00))
            xy, z_cone = rng.getrandbits(32), rng.getrandbits(32)
            if i % 7 == 0:
                xy, z_cone = 0x007ff7ff, 0x7ff
            meshlets.append((vertices, faces, radius, xy, z_cone))
        groups.append(mesh_group(meshlets))
    chunks = [packed, *groups]
    if i % 17 == 0:
        chunks.insert(0, block(n))
    if i % 13 == 0:
        chunks.insert(-1, packed)
    return page(*chunks)


def outcome(call, encode):
    try:
        return "value", encode(call())
    except (ValueError, IndexError, OverflowError, struct.error) as exc:
        return type(exc), exc.args


def bits(rows):
    return b"".join(struct.pack("<3d", *row) for row in rows) if rows is not None else None


def tangent_bits(slots):
    return [(slot, b"".join(struct.pack("<3di", *row) for row in rows)) for slot, rows in slots.items()]


def json_bytes(result):
    return json.dumps(result, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def finite_page(n=128, groups=1):
    # Overlapping local triangles make both opposite windings and shared faces.
    packed = [(v % 16, v // 16, (v * 37) % 29) for v in range(n)]
    q = other_block(n, TAGS, packed=packed, scale=(0.25, 0.25, 0.125))
    meshlets = []
    for start in range(0, n, 64):
        refs = list(range(start, min(start + 64, n)))
        faces = [(i, i + 1, i + 2) for i in range(len(refs) - 2)]
        meshlets.append((refs, faces, 0x4c00, 0x7ff7ff, 0x700007ff))
    group = mesh_group(meshlets)
    return page(q, *([group] * groups))


class FallbackTests(unittest.TestCase):
    def test_missing_library(self):
        src = finite_page()
        with patch.object(mp, "_native", None):
            expected = mp.summarise_page(src, "missing", **ALL)
        with patch.dict(os.environ, LEONIDA_MESHLET_NATIVE="1"), patch.object(native.ctypes, "CDLL",
                                                                             side_effect=OSError):
            native._load.cache_clear()
            try:
                self.assertIsNone(native._load())
                self.assertEqual(json_bytes(expected), json_bytes(mp.summarise_page(src, "missing", **ALL)))
            finally:
                native._load.cache_clear()

    def test_disabled_does_not_load(self):
        src = finite_page()
        with patch.object(mp, "_native", None):
            expected = mp.summarise_page(src, "disabled", **ALL)
        with patch.dict(os.environ, LEONIDA_MESHLET_NATIVE="0"), patch.object(native, "_load",
                                                                             side_effect=AssertionError):
            self.assertEqual(json_bytes(expected), json_bytes(mp.summarise_page(src, "disabled", **ALL)))

    def test_negative_indices_and_errors(self):
        positions = [(0., 0., 0.), (1., 0., 0.), (0., 1., 0.), (0., 1., 0.)]
        normals = [(0., 0., 1.)] * len(positions)
        for triangles in ([(-4, -3, -2), (0, 1, 3)], [(-5, 1, 2)], [(-5, 1, 4)], [(0, 0, 0)]):
            with self.subTest(triangles=triangles):
                for call in (lambda: mp.face_agreement(positions, triangles),
                             lambda: mp.normal_agreement(positions, normals, triangles)):
                    with patch.object(mp, "_native", None):
                        expected = outcome(call, json_bytes)
                    self.assertEqual(expected, outcome(call, json_bytes))


class NativeTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ, LEONIDA_MESHLET_NATIVE="1"))
        if native._load() is None:
            self.skipTest("optional meshlet release library is absent")

    def test_random_page_and_ieee_parity(self):
        rng = random.Random(0x8367)
        for i in range(2400):
            src = sample(rng, i)
            chunks = mp.parse_page(src)
            flags = ALL if i % 2 == 0 else {key: bool(i & (1 << j)) for j, key in enumerate(ALL)}
            summaries = []
            rows = []
            for option in (None, native):
                with patch.object(mp, "_native", option):
                    summaries.append(outcome(lambda: mp.summarise_page(src, "synthetic", **flags), json_bytes))
                    rows.append([
                        (bits(mp.positions_4240(src, chunk)), bits(mp.normals_4240(src, chunk)),
                         tangent_bits(mp.tangents_4240(src, chunk)))
                        for chunk in chunks if chunk["kind"] == "block4240"
                    ])
            self.assertEqual(summaries[0], summaries[1], f"page {i}")
            self.assertEqual(rows[0], rows[1], f"IEEE rows in page {i}")

    def test_group_and_summary_take_native_path(self):
        src = finite_page(groups=3)
        chunks = mp.parse_page(src)
        with patch.object(mp, "_native", None):
            expected = mp.summarise_page(src, "finite", **ALL)
        for record in chunks[1:]:
            result = native.validate_group(src, record["offset"])
            self.assertEqual(result, tuple(record[key] for key in ("full_meshlets", "vertex_span", "unique_refs")))
            self.assertIsNotNone(native.geometry(src, record, chunks[0], **ALL))
        self.assertIsNotNone(native.positions(src, chunks[0]))
        self.assertIsNotNone(native.normals(src, chunks[0]))
        self.assertIsNotNone(native.tangents(src, chunks[0]))
        self.assertIsNotNone(native.block_checks(src, chunks[0], normals=True))
        with patch.multiple(mp, **{name: lambda *args: self.fail("Python geometry ran") for name in (
                "group_triangles", "face_agreement", "normal_agreement", "cone_agreement", "centre_agreement")}):
            self.assertEqual(json_bytes(expected), json_bytes(mp.summarise_page(src, "finite", **ALL)))

    def test_refusals_preserve_python_results_and_errors(self):
        src = finite_page()
        packed, record = mp.parse_page(src)
        at = record["offset"]
        for offset, value in ((0, 0), (4, 1), (8, 1), (12, 0xffff), (16, 1), (44, 0xffff_ffff)):
            corrupt = bytearray(src)
            struct.pack_into("<I", corrupt, at + offset, value)
            corrupt = bytes(corrupt)
            with self.subTest(offset=offset):
                self.assertIsNone(native.validate_group(corrupt, at))
                call = lambda: mp.parse_group(corrupt, at)
                with patch.object(mp, "_native", None):
                    expected = outcome(call, json_bytes)
                self.assertEqual(expected, outcome(call, json_bytes))
        for cut in (0, 1, at + 10, len(src) - 16):
            truncated = src[:cut]
            self.assertIsNone(native.validate_group(truncated, at))
            call = lambda: mp.summarise_page(truncated, "truncated", **ALL)
            with patch.object(mp, "_native", None):
                expected = outcome(call, json_bytes)
            self.assertEqual(expected, outcome(call, json_bytes))
        for cut in (0x72, 0x81, 0x88):
            truncated = src[:cut]
            self.assertIsNone(native.positions(truncated, packed))
            call = lambda: mp.positions_4240(truncated, packed)
            with patch.object(mp, "_native", None):
                expected = outcome(call, bits)
            self.assertEqual(expected, outcome(call, bits))

    def test_native_call_error_retries_python(self):
        src = finite_page(groups=2)
        with patch.object(mp, "_native", None):
            expected = mp.summarise_page(src, "retry", **ALL)
        library = native._load()
        symbols = ("meshlet_validate_v1", "meshlet_geometry_v1", "meshlet_block_v1", "meshlet_decode_v1")
        with patch.multiple(library, **dict.fromkeys(symbols, lambda *_: -1)):
            self.assertEqual(json_bytes(expected), json_bytes(mp.summarise_page(src, "retry", **ALL)))
        with patch.object(library, "meshlet_geometry_v1", side_effect=OSError):
            self.assertEqual(json_bytes(expected), json_bytes(mp.summarise_page(src, "retry", **ALL)))

    def test_offset_wrap_uses_python(self):
        refs = [0] * 128
        faces = [(0, 0, 0)] * 128
        src = mesh_group([(refs, faces, 0, 0x7ff7ff, 0x7ff)] * 513)
        self.assertIsNone(native.validate_group(src, 0))
        with patch.object(mp, "_native", None):
            expected = mp.parse_group(src, 0)
        self.assertEqual(expected, mp.parse_group(src, 0))

    def test_buffer_types_use_python(self):
        src = finite_page()
        packed, group = mp.parse_page(src)
        for data in (bytearray(src), memoryview(src)):
            self.assertIsNone(native.geometry(data, group, packed, **ALL))
            self.assertEqual(json_bytes(mp.summarise_page(src, "buffer", **ALL)),
                             json_bytes(mp.summarise_page(data, "buffer", **ALL)))

    def test_guard_band_uses_python_for_centre_boundary(self):
        packed = other_block(1, TAGS, packed=[(1, 1, 1)], scale=(0.03125,) * 3)
        src = page(packed, mesh_group([([0], [(0, 0, 0)], 0, 0x7ff7ff, 0x7ff)]))
        b, g = mp.parse_page(src)
        self.assertIsNone(native.geometry(src, g, b, **ALL))
        with patch.object(mp, "_native", None):
            expected = mp.summarise_page(src, "radius", **ALL)
        self.assertEqual(expected["checks"]["centres_4240_radius_covers_within_rounding"], 1)
        self.assertEqual(json_bytes(expected), json_bytes(mp.summarise_page(src, "radius", **ALL)))

    def test_shared_block_sparse_groups_decode_only_referenced_vertices(self):
        # One 65,536-vertex block shared by 1,024 one-triangle groups that reference only its last
        # three vertices: the work per group must follow the group, not the block.
        n, groups = 65536, 1024
        packed = [(v & 255, (v >> 8) & 255, (v * 37) & 0xffff) for v in range(n)]
        q = other_block(n, TAGS, packed=packed, scale=(0.25, 0.25, 0.125))
        group = mesh_group([([n - 3, n - 2, n - 1], [(0, 1, 2)], 0x4c00, 0x7ff7ff, 0x700007ff)])
        src = page(q, *([group] * groups))
        block_record, *records = mp.parse_page(src)
        self.assertEqual(len(records), groups)
        self.assertEqual({r["vertex_span"] for r in records}, {n})
        flags = dict(faces=True, centres=True, normals=False, cones=True)
        for record in records[:8]:
            for options in (flags, ALL):
                self.assertIsNotNone(native.geometry(src, record, block_record, **options))

        def run(option, **options):
            with patch.object(mp, "_native", option):
                start = time.perf_counter()
                result = mp.summarise_page(src, "shared", **options)
                return json_bytes(result), time.perf_counter() - start

        expected, python_time = run(None, **flags)
        actual, native_time = run(native, **flags)
        self.assertEqual(expected, actual)
        self.assertLess(native_time, python_time, f"native {native_time:.3f}s, Python {python_time:.3f}s")
        # Normals make the Python reference decode the whole block per group; compare a prefix of the groups.
        short = page(q, *([group] * 8))
        with patch.object(mp, "_native", None):
            expected = mp.summarise_page(short, "shared", **ALL)
        self.assertEqual(json_bytes(expected), json_bytes(mp.summarise_page(short, "shared", **ALL)))

    def test_isolated_cli_json(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "synthetic.bin"
            report = Path(directory) / "report.json"
            data.write_bytes(finite_page(groups=4))
            reports = []
            for setting in ("0", "1"):
                env = dict(os.environ, LEONIDA_MESHLET_NATIVE=setting)
                args = [sys.executable, "-I", str(ROOT / "tools" / "meshlet_page.py"), str(data),
                        "--json", str(report), "--faces", "--centres", "--normals", "--cones"]
                output = subprocess.check_output(args, env=env, cwd=directory)
                reports.append((output, report.read_bytes()))
            self.assertEqual(*reports)


if __name__ == "__main__":
    unittest.main()
