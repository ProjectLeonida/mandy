"""Computed I/J interpolation without PerVertexKHR (--f16-interp-fallback).

Verifies the uniform %entry screen-quad barycentric Jacobian shortcut lowering (OpCapability DerivativeControl +
FragmentBarycentricKHR + OpDPdxFine / OpDPdyFine + integer IEEE-754 normal determinant guard):
  * perspective-projected 2x2 pixel quad evaluation (w0 != w1 != w2) at 64x64, 1920x1080, and 2048x2048
    for snapped constants, zeroed provoking-vertex reads, sub-pixel offsets, halved weights, and J written
    between p1 and p2, pinned against a binary64 Jacobian oracle and the analytical finite-difference
    error bound (1 + 2*|alpha| + 2*|beta|) * m_ulp;
  * --f16-interp-fallback (without --bary-fallback) on both f32 and decoded low/high f16 attributes,
    including fragmentShaderBarycentric feature recording and v_interp_p1lv_f16 vsrc2 invariance;
  * non-uniform per-lane branching / EXEC masking (all OpDPdxFine / OpDPdyFine instructions stay in %entry);
  * degenerate zero-area, 1D line-primitive (constant J -> det == 0), signed-zero, subnormal, infinite,
    and NaN barycentric determinants fall back cleanly to P_c without OpFDiv-by-zero or non-finite contamination;
  * existing refusals (--bary-fallback, mixed flat+smooth f32, v_interp_mov_f32 p10/p20, computed f32 on
    16-bit-flagged interpolant) remain intact.

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
from spirv_mini_interp import f32, half_to_float, run_quad

HERE = pathlib.Path(__file__).resolve().parent
FIXTURE = HERE / "fixtures" / "ps_ij_computed_deriv_fallback.asm"


def f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def bits_f32(w: int) -> float:
    return struct.unpack("<f", struct.pack("<I", w & 0xFFFFFFFF))[0]


def pack_f16_pair(lo: float, hi: float) -> int:
    return struct.unpack("<I", struct.pack("<ee", lo, hi))[0]


def ulp32(x: float) -> float:
    """Binary32 ULP of f32(x) (subnormal-aware, distinct from Python's binary64 math.ulp)."""
    v = abs(f32(x))
    if v == 0.0:
        return math.ldexp(1.0, -149)
    _, exp = math.frexp(v)
    return math.ldexp(1.0, max(exp - 24, -149))


def quad_jacobian_ref64(quad: list[dict[str, object]], lane: int, target_i: float, target_j: float,
                        var_name: str, ch: int) -> tuple[float, float, float]:
    """Exact binary64 Jacobian solve on the binary32-quantized 2x2 quad inputs for lane."""
    x0, x1 = lane & ~1, (lane & ~1) | 1
    y0, y1 = lane & ~2, (lane & ~2) | 2
    _, ic, jc = quad[lane]["%bary"]
    pc = quad[lane][var_name][ch]
    dix = quad[x1]["%bary"][1] - quad[x0]["%bary"][1]
    diy = quad[y1]["%bary"][1] - quad[y0]["%bary"][1]
    djx = quad[x1]["%bary"][2] - quad[x0]["%bary"][2]
    djy = quad[y1]["%bary"][2] - quad[y0]["%bary"][2]
    det64 = dix * djy - diy * djx
    alp64 = ((target_i - ic) * djy - (target_j - jc) * diy) / det64
    bet64 = (dix * (target_j - jc) - djx * (target_i - ic)) / det64
    dpx = quad[x1][var_name][ch] - quad[x0][var_name][ch]
    dpy = quad[y1][var_name][ch] - quad[y0][var_name][ch]
    return pc + alp64 * dpx + bet64 * dpy, alp64, bet64


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
    """Binary32-quantized Vulkan perspective-correct fragment inputs for a 2x2 pixel quad (lanes 0..3) and (I, J)."""
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
        self.assertIn("FragmentBarycentricKHR", meta["capabilities"])
        self.assertIn("fragmentShaderBarycentric", meta["features"])
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
            ref_y, _, _ = quad_jacobian_ref64(quad, lane, ii, jj, "%in_attr0", 1)
            ref_lo, _, _ = quad_jacobian_ref64(quad, lane, 0.0, 0.0, "%in_f16h1_0", 0)
            ref_hi, _, _ = quad_jacobian_ref64(quad, lane, il, f32(0.5 * jl), "%in_f16h1_1", 0)
            # Channel 0 is a pristine center read: bit-exact to the smooth input word.
            self.assertEqual(outs[lane]["%out_mrt0"][0], f32_bits(quad[lane]["%in_attr0"][0]))
            self.assertLess(abs(got[0] - want_x), 2e-6)
            self.assertLess(abs(got[1] - ref_y), 2e-6)
            self.assertLess(abs(got[1] - want_y), 1e-4)
            # On 64x64 (|alpha| + |beta| <= 32 px), 1.5 narrows to exact f16 (0 ulp); see 2048px test below for wide drift.
            self.assertEqual(got[2], float(struct.unpack("<e", struct.pack("<e", ref_lo))[0]))
            self.assertEqual(got[2], want_f16_lo)
            self.assertEqual(got[3], float(struct.unpack("<e", struct.pack("<e", ref_hi))[0]))
            self.assertLessEqual(abs(got[3] - float(struct.unpack("<e", struct.pack("<e", want_f16_hi))[0])), 2**-10)

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

    def test_p1lv_f16_computed_ij_uses_interpolated_attribute_not_vsrc2(self):
        """Like the non-computed path, computed-I/J v_interp_p1lv_f16 takes P_0 from the interpolated attribute
        rather than reading vsrc2: varying v10 leaves %reg_v10 unread and produces identical quad outputs."""
        def make_lines(v10_imm: str) -> list[str]:
            return [
                f"v_mov_b32_e32 v10, {v10_imm}",
                "v_mov_b32_e32 v0, 0.25",
                "v_mov_b32_e32 v1, 0.5",
                "v_interp_p1lv_f16 v2, v0, attr1.x, v10",
                "v_interp_p2_f16 v2, v1, attr1.x, v2",
                "exp mrt0 v2, v2, v2, v2 done vm",
                "s_endpgm",
            ]

        spv_zero, _ = translate_with("ps_p1lv_ij_0", make_lines("0"), f16=True, bary=False)
        spv_one, _ = translate_with("ps_p1lv_ij_1", make_lines("0x3f800000"), f16=True, bary=False)
        self._validate_spirv(spv_zero)
        self._validate_spirv(spv_one)
        self.assertIn("OpStore %reg_v10 ", spv_zero)
        self.assertIn("OpStore %reg_v10 ", spv_one)
        self.assertNotIn("OpLoad %uint %reg_v10", spv_zero)
        self.assertNotIn("OpLoad %uint %reg_v10", spv_one)
        strip_v10 = lambda s: [ln for ln in s.splitlines()
                               if "OpStore %reg_v10 " not in ln and not ln.startswith("%c3f800000 =")]
        self.assertEqual(strip_v10(spv_zero), strip_v10(spv_one))

        quad0, _ = perspective_quad_inputs(spv_zero, self.VERTS_PERSP, {1: self.ATTR1_WORDS})
        quad1, _ = perspective_quad_inputs(spv_one, self.VERTS_PERSP, {1: self.ATTR1_WORDS})
        outs0 = run_quad(spv_zero, lambda lane: quad0[lane])
        outs1 = run_quad(spv_one, lambda lane: quad1[lane])
        # LO_HALVES = (1.5, 3.5, 5.5) at (0.25, 0.5) -> 1.5 + 0.25*2.0 + 0.5*4.0 = 4.0 (f16 0x4400).
        for lane in range(4):
            self.assertEqual(outs0[lane]["%out_mrt0"], outs1[lane]["%out_mrt0"])
            self.assertEqual(outs0[lane]["%out_mrt0"][0] & 0xFFFF, 0x4400)

    def test_1080p_perspective_precision_subpixel_vs_full_viewport_jump(self):
        """At 1920x1080 with perspective W = (1.0, 2.5, 4.0), sub-pixel barycentric shifts stay within ~5e-6, while
        ~1,000-pixel jumps to fixed constants reflect 1-pixel binary32 input quantization (< 2e-3, <= 1 f16 ulp) and
        agree with an exact binary64 Jacobian solve on the same binary32 quad inputs within 2e-6."""
        verts_1080p = ((0.0, 0.0, 1.0), (1920.0, 0.0, 2.5), (0.0, 1080.0, 4.0))
        subpix_lines = [
            "v_add_f32_e32 v8, 0x38800000, v0",     # I + 2^-14 (~0.1 px at 1080p)
            "v_add_f32_e32 v9, 0xb8800000, v1",     # J - 2^-14
            "v_interp_p1_f32_e32 v2, v8, attr0.y",
            "v_interp_p2_f32_e32 v2, v9, attr0.y",
            "exp mrt0 v2, v2, v2, v2 done vm",
            "s_endpgm",
        ]
        spv_sub, _ = translate_with("ps_ij_1080p_sub", subpix_lines, f16=True, bary=False)
        self._validate_spirv(spv_sub)

        raw_lines = [ln for ln in FIXTURE.read_text(encoding="utf-8").splitlines()
                     if ln.strip() and not ln.strip().startswith(";")]
        spv_fix, _ = translate_with("ps_ij_1080p_fix", raw_lines, f16=True, bary=False)
        attrs = {0: self.ATTR0_WORDS, 1: self.ATTR1_WORDS}
        p0_y, p1_y, p2_y = (v[1] for v in self.ATTR0_FLOATS)
        delta = bits_f32(0x38800000)

        for qx, qy in ((1794.5, 50.5), (1580.5, 96.5), (640.5, 360.5), (32.5, 900.5)):
            with self.subTest(quad_xy=(qx, qy)):
                q_sub, ij_sub = perspective_quad_inputs(spv_sub, verts_1080p, {0: self.ATTR0_WORDS}, pixel_xy=(qx, qy))
                outs_sub = run_quad(spv_sub, lambda lane, qs=q_sub: qs[lane])
                for lane in range(4):
                    it = f32(ij_sub[lane][0] + delta)
                    jt = f32(ij_sub[lane][1] - delta)
                    want_sub = p0_y + it * (p1_y - p0_y) + jt * (p2_y - p0_y)
                    self.assertLess(abs(bits_f32(outs_sub[lane]["%out_mrt0"][0]) - want_sub), 5e-6)

                q_fix, ij_fix = perspective_quad_inputs(spv_fix, verts_1080p, attrs, pixel_xy=(qx, qy))
                outs_fix = run_quad(spv_fix, lambda lane, qf=q_fix: qf[lane])
                for lane in range(4):
                    got = [bits_f32(w) for w in outs_fix[lane]["%out_mrt0"]]
                    if ij_fix[lane][0] > f32(0.2):
                        # Binary64 Jacobian solve on the identical binary32 quad inputs: proves error is input rounding.
                        x0, x1 = lane & ~1, (lane & ~1) | 1
                        y0, y1 = lane & ~2, (lane & ~2) | 2
                        ic, jc = ij_fix[lane]
                        pc = q_fix[lane]["%in_attr0"][1]
                        dix = q_fix[x1]["%bary"][1] - q_fix[x0]["%bary"][1]
                        diy = q_fix[y1]["%bary"][1] - q_fix[y0]["%bary"][1]
                        djx = q_fix[x1]["%bary"][2] - q_fix[x0]["%bary"][2]
                        djy = q_fix[y1]["%bary"][2] - q_fix[y0]["%bary"][2]
                        det64 = dix * djy - diy * djx
                        alp64 = ((0.25 - ic) * djy - (0.25 - jc) * diy) / det64
                        bet64 = (dix * (0.25 - jc) - djx * (0.25 - ic)) / det64
                        ref64 = pc + alp64 * (q_fix[x1]["%in_attr0"][1] - q_fix[x0]["%in_attr0"][1]) + (
                            bet64 * (q_fix[y1]["%in_attr0"][1] - q_fix[y0]["%in_attr0"][1]))
                        self.assertLess(abs(got[1] - 3.0), 2e-3)
                        self.assertLess(abs(got[1] - ref64), 2e-6)
                    # Channel 2 exports v_cvt_f32_f16(v3), where I' = J' = 0 targets LO_HALVES[0] = 1.5 (<= 1 f16 ulp).
                    self.assertLessEqual(abs(got[2] - 1.5), 2**-10)

    def test_wide_2048px_perspective_extrapolation_bound_and_line_det_zero_fallback(self):
        """Characterizes the shortcut's finite-difference amplification on a 2048px perspective triangle with P in
        [1.5, 4096.0] (distant (0, 0) jump reaches >0.30 / >300 f16 ulps while obeying (1 + 2|alpha| + 2|beta|)*m_ulp,
        whereas local sub-pixel shifts stay <2e-3), and verifies 1D line primitives (constant J -> det == 0) return P_c."""
        lines = [
            "v_add_f32_e32 v8, 0x39800000, v0",     # local sub-pixel offset I + 2^-12 (< 1 px on 2048px)
            "v_add_f32_e32 v9, 0xb9800000, v1",     # J - 2^-12
            "v_interp_p1_f32_e32 v4, v8, attr0.x",
            "v_interp_p2_f32_e32 v4, v9, attr0.x",
            "v_mov_b32_e32 v0, 0",
            "v_mov_b32_e32 v1, 0",
            "v_interp_p1_f32_e32 v2, v0, attr0.x",
            "v_interp_p2_f32_e32 v2, v1, attr0.x",
            "v_interp_p1ll_f16 v3, v0, attr1.x",
            "v_interp_p2_f16 v3, v1, attr1.x, v3",
            "v_cvt_f32_f16_e32 v3, v3",
            "exp mrt0 v2, v3, v4, v2 done vm",
            "s_endpgm",
        ]
        spv, _ = translate_with("ps_ij_wide_2048", lines, f16=True, bary=False)
        self._validate_spirv(spv)

        verts_2048 = ((0.0, 0.0, 1.0), (2048.0, 0.0, 1.3), (0.0, 2048.0, 1.7))
        p_vals = (1.5, 4096.0, 5.5)
        p10, p20 = p_vals[1] - p_vals[0], p_vals[2] - p_vals[0]
        a0 = [tuple(f32_bits(x) for x in (p, 0.0, 0.0, 0.0)) for p in p_vals]
        a1 = [(pack_f16_pair(p, 0.0), 0, 0, 0) for p in p_vals]
        attrs = {0: a0, 1: a1}
        delta = bits_f32(0x39800000)

        peak_far_f32_err = 0.0
        peak_far_f16_err = 0.0
        for qx in (64.5, 512.5, 1024.5, 1387.5, 1750.5, 2000.5):
            quad, ij = perspective_quad_inputs(spv, verts_2048, attrs, pixel_xy=(qx, 10.5))
            outs = run_quad(spv, lambda lane, qs=quad: qs[lane])
            m_ulp = max(
                0.5 * ulp32(quad[q]["%in_attr0"][0])
                + abs(p10) * 0.5 * ulp32(quad[q]["%bary"][1])
                + abs(p20) * 0.5 * ulp32(quad[q]["%bary"][2])
                for q in range(4)
            )
            residuals = [
                quad[q]["%in_attr0"][0] - (p_vals[0] + quad[q]["%bary"][1] * p10 + quad[q]["%bary"][2] * p20)
                for q in range(4)
            ]
            for lane in range(4):
                got_far_f32, got_far_f16, got_local_f32, _ = [bits_f32(w) for w in outs[lane]["%out_mrt0"]]
                ref_far, alp_far, bet_far = quad_jacobian_ref64(quad, lane, 0.0, 0.0, "%in_attr0", 0)
                x0, x1 = lane & ~1, (lane & ~1) | 1
                y0, y1 = lane & ~2, (lane & ~2) | 2
                # Exact linear-functional identity on quantized inputs: ref64 - P0 == e_c + alp*(e_x1-e_x0) + bet*(e_y1-e_y0).
                linear_err = (residuals[lane]
                              + alp_far * (residuals[x1] - residuals[x0])
                              + bet_far * (residuals[y1] - residuals[y0]))
                self.assertLess(abs((ref_far - p_vals[0]) - linear_err), 1e-9)

                dpx = quad[x1]["%in_attr0"][0] - quad[x0]["%in_attr0"][0]
                dpy = quad[y1]["%in_attr0"][0] - quad[y0]["%in_attr0"][0]
                alu_tol = 8.0 * ulp32(max(abs(quad[lane]["%in_attr0"][0]), abs(alp_far * dpx), abs(bet_far * dpy), 1.0))
                self.assertLessEqual(abs(got_far_f32 - ref_far), alu_tol)
                bound_far = (1.0 + 2.0 * abs(alp_far) + 2.0 * abs(bet_far)) * m_ulp + alu_tol
                far_err = abs(got_far_f32 - p_vals[0])
                self.assertLessEqual(far_err, bound_far)
                peak_far_f32_err = max(peak_far_f32_err, far_err)
                peak_far_f16_err = max(peak_far_f16_err, abs(got_far_f16 - p_vals[0]))

                # Local sub-pixel offset on the same 2048px wide triangle stays small (< 2e-3).
                it = f32(ij[lane][0] + delta)
                jt = f32(ij[lane][1] - delta)
                want_local = p_vals[0] + it * p10 + jt * p20
                _, alp_loc, bet_loc = quad_jacobian_ref64(quad, lane, it, jt, "%in_attr0", 0)
                self.assertLess(max(abs(alp_loc), abs(bet_loc)), 1.0)
                bound_loc = (1.0 + 2.0 * abs(alp_loc) + 2.0 * abs(bet_loc)) * m_ulp + 8.0 * ulp32(want_local)
                self.assertLessEqual(abs(got_local_f32 - want_local), bound_loc)
                self.assertLess(abs(got_local_f32 - want_local), 2e-3)

        # Pin that distant jumps across a 2048px primitive genuinely incur O(span * ulp) drift (> 300 f16 ulps at 1.5).
        self.assertGreater(peak_far_f32_err, 0.30)
        self.assertGreater(peak_far_f16_err, 300 * (2**-10))

        # 1D line primitive: J == 0 across all 4 quad lanes -> dJ/dx = dJ/dy = 0 -> det == 0 -> returns P_c, not P(0,0).
        line_asm = [
            "v_mov_b32_e32 v0, 0",
            "v_mov_b32_e32 v1, 0",
            "v_interp_p1_f32_e32 v2, v0, attr0.x",
            "v_interp_p2_f32_e32 v2, v1, attr0.x",
            "exp mrt0 v2, v2, v2, v2 done vm",
            "s_endpgm",
        ]
        spv_def, _ = translate_with("ps_line_def", line_asm, f16=False, bary=False)
        spv_f16, _ = translate_with("ps_line_f16", line_asm, f16=True, bary=False)
        line_bary = [(0.75, 0.25, 0.0), (0.50, 0.50, 0.0), (0.75, 0.25, 0.0), (0.50, 0.50, 0.0)]
        line_pc = [(2.0, 0.0, 0.0, 0.0), (4.0, 0.0, 0.0, 0.0), (2.0, 0.0, 0.0, 0.0), (4.0, 0.0, 0.0, 0.0)]
        line_pv = [(0.0, 0.0, 0.0, 0.0), (8.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0)]
        outs_def = run_quad(spv_def, lambda l: {"%bary": line_bary[l], "%in_attr0": line_pv,
                                                "%frag_coord": (float(l & 1), float(l >> 1), 0.5, 1.0)})
        outs_f16 = run_quad(spv_f16, lambda l: {"%bary": line_bary[l], "%in_attr0": line_pc[l],
                                                "%frag_coord": (float(l & 1), float(l >> 1), 0.5, 1.0)})
        for l in range(4):
            self.assertEqual(bits_f32(outs_def[l]["%out_mrt0"][0]), 0.0)
            self.assertEqual(bits_f32(outs_f16[l]["%out_mrt0"][0]), line_pc[l][0])


if __name__ == "__main__":
    unittest.main()
