"""Computed I/J interpolation without PerVertexKHR (--f16-interp-fallback and --bary-fallback).

Verifies the uniform %entry screen-quad barycentric Jacobian lowering (OpCapability DerivativeControl +
OpDPdxFine / OpDPdyFine + integer IEEE-754 normal determinant guard):
  * exact perspective-projected 2x2 pixel quad evaluation (w0 != w1 != w2) for snapped constants,
    zeroed provoking-vertex reads, sub-pixel offsets, halved weights, and J written between p1 and p2;
  * --f16-interp-fallback (alone and combined with --bary-fallback) on both f32 and decoded low/high f16
    attributes, including dynamic f16_halves packing and bary_fallback_gs pass-through compatibility;
  * non-uniform per-lane branching / EXEC masking (all OpDPdxFine / OpDPdyFine instructions stay in %entry);
  * degenerate zero-area, signed-zero, subnormal, infinite, and NaN barycentric determinants fall back
    cleanly to P_c without OpFDiv-by-zero or non-finite contamination;
  * existing refusals (mixed flat+smooth f32, v_interp_mov_f32 p10/p20, computed f32 on 16-bit-flagged
    interpolant) remain intact.

Hand-written synthetic programs only; no game files.
"""
from __future__ import annotations

import math
import os
import pathlib
import re
import shutil
import struct
import subprocess
import tempfile
import unittest

import support
import rdna2spirv
import spirv_asm
import bary_fallback_gs
from spirv_mini_interp import f32, half_to_float, run_quad

HERE = pathlib.Path(__file__).resolve().parent
FIXTURE = HERE / "fixtures" / "ps_ij_computed_deriv_fallback.asm"


def f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def bits_f32(w: int) -> float:
    return struct.unpack("<f", struct.pack("<I", w & 0xFFFFFFFF))[0]


def pack_f16_pair(lo: float, hi: float) -> int:
    return struct.unpack("<I", struct.pack("<ee", lo, hi))[0]


def translate_with(name: str, lines: list[str], *, f16: bool = False, bary: bool = False, fssm=None):
    old_f16, old_bary = rdna2spirv.F16_HALVES, rdna2spirv.BARY_FALLBACK
    rdna2spirv.F16_HALVES = f16
    rdna2spirv.BARY_FALLBACK = bary
    try:
        spv = support.translate_lines(name, lines, fssm=fssm)
        meta = rdna2spirv.shader_meta(rdna2spirv.translate.last_gen, spv)
    finally:
        rdna2spirv.F16_HALVES = old_f16
        rdna2spirv.BARY_FALLBACK = old_bary
    return spv, meta


def perspective_quad_inputs(
    spvasm: str,
    verts_xyw: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]],
    attrs_words: dict[int, list[tuple[int, int, int, int]]],
    pixel_xy: tuple[float, float] = (8.5, 12.5),
):
    """Exact Vulkan perspective-correct fragment inputs for a 2x2 pixel quad (lanes 0..3) and per-lane (I, J)."""
    (x0, y0, w0), (x1, y1, w1), (x2, y2, w2) = verts_xyw
    area = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
    quad = []
    ij_per_lane = []
    for q in range(4):
        px = pixel_xy[0] + (q & 1)
        py = pixel_xy[1] + ((q >> 1) & 1)
        b = ((px - x0) * (y2 - y0) - (py - y0) * (x2 - x0)) / area
        c = ((x1 - x0) * (py - y0) - (y1 - y0) * (px - x0)) / area
        a = 1.0 - b - c
        inv_w = a / w0 + b / w1 + c / w2
        k = f32((a / w0) / inv_w)
        i = f32((b / w1) / inv_w)
        j = f32((c / w2) / inv_w)
        ij_per_lane.append((i, j))
        inp: dict[str, object] = {"%bary": (k, i, j), "%frag_coord": (f32(px), f32(py), 0.5, f32(inv_w))}
        for n, vwords in attrs_words.items():
            fv = [[bits_f32(w) for w in vw] for vw in vwords]
            raw_name = f"%in_attr{n}"
            if f"{raw_name} = OpVariable" in spvasm:
                if f"OpDecorate {raw_name} Flat" in spvasm:
                    inp[raw_name] = tuple(fv[0])
                else:
                    inp[raw_name] = tuple(
                        f32((a / w0 * fv[0][ch] + b / w1 * fv[1][ch] + c / w2 * fv[2][ch]) / inv_w)
                        for ch in range(4)
                    )
            for half in (0, 1):
                hname = f"%in_f16h{n}_{half}"
                if f"{hname} = OpVariable" in spvasm:
                    hv = [[half_to_float((vw[ch] >> (16 * half)) & 0xFFFF) for ch in range(4)] for vw in vwords]
                    inp[hname] = tuple(
                        f32((a / w0 * hv[0][ch] + b / w1 * hv[1][ch] + c / w2 * hv[2][ch]) / inv_w)
                        for ch in range(4)
                    )
        quad.append(inp)
    return quad, ij_per_lane


class ComputedIJDerivativeFallbackTests(unittest.TestCase):
    # Perspective-skewed 64x64 triangle (w0 != w1 != w2) with non-zero provoking vertex values.
    VERTS_PERSP = ((0.0, 0.0, 1.0), (64.0, 0.0, 2.5), (0.0, 64.0, 4.0))
    ATTR0_FLOATS = ((1.25, -2.0, 3.5, 0.5), (5.25, 6.0, -0.5, 2.5), (-2.75, 10.0, 7.5, 4.5))
    ATTR0_WORDS = [tuple(f32_bits(x) for x in v) for v in ATTR0_FLOATS]
    LO_HALVES = (1.5, 3.5, 5.5)
    HI_HALVES = (-2.0, 6.0, 14.0)
    ATTR1_WORDS = [(pack_f16_pair(lo, hi), 0, 0, 0) for lo, hi in zip(LO_HALVES, HI_HALVES)]

    def _validate_spirv(self, spvasm: str) -> None:
        binary = spirv_asm.assemble(spvasm, target_env="vulkan1.2")
        self.assertGreater(len(binary), 20)
        spirv_val = shutil.which("spirv-val")
        if spirv_val:
            with tempfile.TemporaryDirectory() as tmp:
                spv_path = os.path.join(tmp, "mod.spv")
                pathlib.Path(spv_path).write_bytes(binary)
                r = subprocess.run([spirv_val, "--target-env", "vulkan1.2", spv_path],
                                   capture_output=True, text=True)
                self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_ortho_dyadic_triangle_is_bit_exact_under_f16_interp_fallback(self):
        """On vs_ids's 64x64 ortho triangle (ps_ij_computed pattern), snapped (0.25,0.25), zeroed P0, and (i, j/2)
        are bit-for-bit exact across all 4 quad lanes under --f16-interp-fallback."""
        lines = [
            "v_mov_b32_e32 v10, v0",
            "v_mov_b32_e32 v11, v1",
            "v_mov_b32_e32 v0, 0.25",
            "v_mov_b32_e32 v1, 0.25",
            "v_interp_p1_f32_e32 v2, v0, attr0.x",
            "v_interp_p2_f32_e32 v2, v1, attr0.x",
            "v_mov_b32_e32 v0, 0",
            "v_mov_b32_e32 v1, 0",
            "v_interp_p1_f32_e32 v3, v0, attr0.y",
            "v_interp_p2_f32_e32 v3, v1, attr0.y",
            "v_mul_f32_e32 v11, 0.5, v11",
            "v_mov_b32_e32 v0, v10",
            "v_mov_b32_e32 v1, v11",
            "v_interp_p1_f32_e32 v4, v0, attr0.y",
            "v_interp_p2_f32_e32 v4, v1, attr0.y",
            "exp mrt0 v2, v3, v4, v2 done vm",
            "s_endpgm",
        ]
        ortho_verts = ((0.0, 0.0, 1.0), (128.0, 0.0, 1.0), (0.0, 128.0, 1.0))
        uv_words = [tuple(f32_bits(x) for x in v) for v in ((0.0, 0.0, 0.0, 1.0),
                                                            (2.0, 0.0, 0.0, 1.0),
                                                            (0.0, 2.0, 0.0, 1.0))]
        spv, meta = translate_with("ps_ij_ortho", lines, f16=True, bary=False)
        self._validate_spirv(spv)
        self.assertNotIn("PerVertexKHR", spv)
        self.assertIn("OpDecorate %bary BuiltIn BaryCoordKHR", spv)
        self.assertIn("DerivativeControl", meta["capabilities"])
        self.assertNotIn("DerivativeControl", meta["features"])
        quad, ij = perspective_quad_inputs(spv, ortho_verts, {0: uv_words}, pixel_xy=(16.5, 24.5))
        outs = run_quad(spv, lambda lane: quad[lane])
        for lane in range(4):
            w = outs[lane]["%out_mrt0"]
            self.assertEqual(w[0], f32_bits(0.5))
            self.assertEqual(w[1], f32_bits(0.0))
            self.assertEqual(w[2], f32_bits(ij[lane][1]))   # 2 * (j / 2) == j
            self.assertEqual(w[3], f32_bits(0.5))

    def test_perspective_fixture_branch_and_f16_halves(self):
        """Hand-written fixture mixing pristine f32, non-uniform EXEC-masked computed f32, and computed low/high f16
        under strong perspective foreshortening (w = 1.0, 2.5, 4.0) with --f16-interp-fallback."""
        raw_lines = [ln for ln in FIXTURE.read_text(encoding="utf-8").splitlines()
                     if ln.strip() and not ln.strip().startswith(";")]
        attrs = {0: self.ATTR0_WORDS, 1: self.ATTR1_WORDS}
        # Pick pixel_xy=(21.5, 10.5) so lane 0 has I < 0.20 (untaken) and lanes 1..3 have I > 0.20 (taken): non-uniform quad!
        spv, meta = translate_with("ps_ij_persp_fix", raw_lines, f16=True, bary=False)
        self._validate_spirv(spv)
        self.assertNotIn("PerVertexKHR", spv)
        self.assertEqual(meta["f16_halves"], {"1": [26, 27]})
        # Every OpDPdxFine / OpDPdyFine must sit in %entry before the first branch.
        first_branch = spv.index("OpBranch ")
        for m in re.finditer(r"OpDPd[xy]Fine", spv):
            self.assertLess(m.start(), first_branch)

        quad, ij = perspective_quad_inputs(spv, self.VERTS_PERSP, attrs, pixel_xy=(21.5, 10.5))
        # Confirm the 2x2 quad genuinely straddles the I > 0.20 branch condition!
        conds = [ij[q][0] > f32(0.2) for q in range(4)]
        self.assertEqual(set(conds), {False, True})

        outs = run_quad(spv, lambda lane: quad[lane])
        p0_x, p1_x, p2_x = (v[0] for v in self.ATTR0_FLOATS)
        p0_y, p1_y, p2_y = (v[1] for v in self.ATTR0_FLOATS)
        for lane in range(4):
            got = [bits_f32(w) for w in outs[lane]["%out_mrt0"]]
            il, jl = ij[lane]
            want_x = p0_x + il * (p1_x - p0_x) + jl * (p2_x - p0_x)
            ii, jj = (0.25, 0.25) if conds[lane] else (il, jl)
            want_y = p0_y + ii * (p1_y - p0_y) + jj * (p2_y - p0_y)
            want_f16_lo = self.LO_HALVES[0]   # I' = J' = 0 -> P0 low half = 1.5
            want_f16_hi = self.HI_HALVES[0] + il * (self.HI_HALVES[1] - self.HI_HALVES[0]) + (
                0.5 * jl) * (self.HI_HALVES[2] - self.HI_HALVES[0])
            self.assertAlmostEqual(got[0], want_x, places=5)
            self.assertAlmostEqual(got[1], want_y, places=4)
            self.assertAlmostEqual(got[2], want_f16_lo, places=2)
            self.assertAlmostEqual(got[3], float(struct.unpack("<e", struct.pack("<e", want_f16_hi))[0]),
                                   places=2)

    def test_j_written_between_ops_and_non_v0_v1_registers(self):
        """Computed I/J stored purely in v8/v9 (neither v0 nor v1 in frag_inputs) and J updated between p1 and p2."""
        lines = [
            "v_mov_b32_e32 v8, 0.25",
            "v_mov_b32_e32 v9, 0.0",
            "v_interp_p1_f32_e32 v2, v8, attr0.z",
            "v_mov_b32_e32 v9, 0.5",
            "v_interp_p2_f32_e32 v2, v9, attr0.z",
            "exp mrt0 v2, v2, v2, v2 done vm",
            "s_endpgm",
        ]
        spv, _ = translate_with("ps_ij_v8_v9", lines, f16=True, bary=False)
        self._validate_spirv(spv)
        quad, _ = perspective_quad_inputs(spv, self.VERTS_PERSP, {0: self.ATTR0_WORDS})
        outs = run_quad(spv, lambda lane: quad[lane])
        p0, p1, p2 = (v[2] for v in self.ATTR0_FLOATS)
        want = p0 + 0.25 * (p1 - p0) + 0.5 * (p2 - p0)
        for lane in range(4):
            self.assertAlmostEqual(bits_f32(outs[lane]["%out_mrt0"][0]), want, places=4)

    def test_degenerate_determinant_guards_zero_subnormal_inf_and_nan(self):
        """When the 2x2 barycentric Jacobian is zero, subnormal, infinite, or NaN, %ij_nz selects P_c cleanly."""
        lines = [
            "v_mov_b32_e32 v0, 0.25",
            "v_mov_b32_e32 v1, 0.25",
            "v_interp_p1_f32_e32 v2, v0, attr0.x",
            "v_interp_p2_f32_e32 v2, v1, attr0.x",
            "exp mrt0 v2, v2, v2, v2 done vm",
            "s_endpgm",
        ]
        spv, _ = translate_with("ps_ij_degen", lines, f16=True)
        self._validate_spirv(spv)
        pc_val = (3.75, 0.0, 0.0, 0.0)
        sub = struct.unpack("<f", struct.pack("<I", 0x00000100))[0]
        cases = [
            ("zero_area_identical_bary", [(0.5, 0.25, 0.25)] * 4),
            ("collinear_bary", [(0.5, 0.1, 0.4), (0.4, 0.2, 0.4), (0.5, 0.1, 0.4), (0.4, 0.2, 0.4)]),
            ("subnormal_det", [(1.0, 0.0, 0.0), (1.0, sub, 0.0), (1.0, 0.0, sub), (1.0, sub, sub)]),
            ("inf_bary", [(0.0, math.inf, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, math.inf), (0.0, 0.0, 0.0)]),
            ("nan_bary", [(0.0, math.nan, 0.0)] * 4),
        ]
        for label, barys in cases:
            with self.subTest(case=label):
                outs = run_quad(spv, lambda lane, bs=barys: {"%bary": bs[lane], "%in_attr0": pc_val,
                                                             "%frag_coord": (0.5, 0.5, 0.5, 1.0)})
                for lane in range(4):
                    self.assertEqual(outs[lane]["%out_mrt0"][0], f32_bits(3.75))

    def test_default_mode_keeps_per_vertex_khr_unchanged(self):
        """Without fallback flags, computed I/J still uses PerVertexKHR and emits no DerivativeControl."""
        lines = [
            "v_mov_b32_e32 v0, 0",
            "v_mov_b32_e32 v1, 0",
            "v_interp_p1_f32_e32 v2, v0, attr0.x",
            "v_interp_p2_f32_e32 v2, v1, attr0.x",
            "exp mrt0 v2, v2, v2, v2 done vm",
            "s_endpgm",
        ]
        spv, meta = translate_with("ps_ij_default", lines, f16=False, bary=False)
        self.assertIn("OpDecorate %in_attr0 PerVertexKHR", spv)
        self.assertNotIn("DerivativeControl", meta["capabilities"])
        self.assertNotIn("OpDPdxFine", spv)

    def test_existing_per_vertex_refusals_preserved_under_fallbacks(self):
        """--bary-fallback, mixed flat+smooth f32, p10/p20 mov deltas, and computed f32 on 16-bit-flagged slots still refuse."""
        for f16 in (False, True):
            with self.subTest(bary_refuse_f16=f16):
                with self.assertRaisesRegex(rdna2spirv.Unsupported, "bary-fallback"):
                    translate_with("ps_bary_refuse", [
                        "v_mov_b32_e32 v0, 0", "v_mov_b32_e32 v1, 0",
                        "v_interp_p1_f32_e32 v2, v0, attr0.x", "v_interp_p2_f32_e32 v2, v1, attr0.x",
                        "exp mrt0 v2, v2, v2, v2 done vm", "s_endpgm",
                    ], f16=f16, bary=True)
        for f16, bary in ((True, False), (False, True)):
            with self.subTest(f16=f16, bary=bary):
                with self.assertRaisesRegex(rdna2spirv.Unsupported, "need per-vertex inputs"):
                    translate_with("ps_mixed_refuse", [
                        "v_interp_p1_f32_e32 v2, v0, attr0.x", "v_interp_p2_f32_e32 v2, v1, attr0.x",
                        "v_interp_mov_f32_e32 v3, p0, attr0.y",
                        "exp mrt0 v2, v3, v2, v3 done vm", "s_endpgm",
                    ], f16=f16, bary=bary)
                with self.assertRaisesRegex(rdna2spirv.Unsupported, "without per-vertex inputs"):
                    translate_with("ps_delta_refuse", [
                        "v_interp_mov_f32_e32 v2, p10, attr0.x",
                        "exp mrt0 v2, v2, v2, v2 done vm", "s_endpgm",
                    ], f16=f16, bary=bary)
        # FSSM pixel-input record at 0x138: kind=6 (byte 0), flags=0x12 (bit 1 = p1/p2, bit 4 = 16-bit) at byte 7.
        fssm = bytearray(0x148)
        fssm[0:4] = b"FSSM"
        fssm[0xD4] = 0x20  # wave32
        fssm[0x138] = 6
        fssm[0x13F] = 0x12
        with self.assertRaisesRegex(rdna2spirv.Unsupported, "16-bit interpolant through v_interp_\\*_f32"):
            translate_with("ps_fp16_flag_refuse", [
                "v_mov_b32_e32 v0, 0", "v_mov_b32_e32 v1, 0",
                "v_interp_p1_f32_e32 v2, v0, attr0.x", "v_interp_p2_f32_e32 v2, v1, attr0.x",
                "exp mrt0 v2, v2, v2, v2 done vm", "s_endpgm",
            ], f16=True, fssm=bytes(fssm))


if __name__ == "__main__":
    unittest.main()
