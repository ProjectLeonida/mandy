"""Hardware DenormFlushToZero 32 + canonical Mesa NIR fmulz lowering (Item 3B.1).

Verifies:
  * Default translation declares OpCapability DenormFlushToZero and OpExecutionMode %main DenormFlushToZero 32
    alongside SignedZeroInfNanPreserve 32 and DenormPreserve 16, and records "shaderDenormFlushToZeroFloat32"
    in shader_meta()["float_controls"] (core since SPIR-V 1.4, no SPV_KHR_float_controls extension at vulkan1.2).
  * Software flush_f32() wrappers (OpFOrdLessThan against 2^-126) are elided on f32 MAD/MAC/legacy/fma_mix/dot2 ops
    under hardware FTZ32 and restored under --no-ftz32.
  * DenormFlushToZero 32 is gated off when the shader contains v_div_scale_f32, v_div_fmas_f32, s_setreg* on
    HW_REG_MODE, s_denorm_mode, or under --no-ftz32 / --no-float-controls.
  * v_mul_legacy_f32 emits the canonical Mesa NIR fmulz pattern (OpSelect %float (OpLogicalOr (OpFOrdEqual x %f0)
    (OpFOrdEqual y %f0)) %f0 (OpFMul x y)) and preserves exact edge-case semantics (0*NaN=+0, 0*Inf=+0, (-0)*1=+0,
    subnormal inputs -> +0, signed subnormal underflow, normal products) in both FTZ32 and --no-ftz32 modes.
  * Emitted SPIR-V passes spirv-as + spirv-val --target-env vulkan1.2, and CLI --no-ftz32 works end to end.
"""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")
sys.dont_write_bytecode = True
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
from shader_ir import parse_line  # noqa: E402
import support  # noqa: E402


def _translate(lines, *, ftz32=True, float_controls=True, name="ps_ftz32_test"):
    old_fc, old_ftz = rdna2spirv.FLOAT_CONTROLS, rdna2spirv.FTZ32
    rdna2spirv.FLOAT_CONTROLS = float_controls
    rdna2spirv.FTZ32 = ftz32
    try:
        spv = support.translate_lines(name, lines)
        meta = rdna2spirv.shader_meta(rdna2spirv.translate.last_gen, spv)
        return spv, meta
    finally:
        rdna2spirv.FLOAT_CONTROLS = old_fc
        rdna2spirv.FTZ32 = old_ftz


def _validate(spv):
    if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
        return
    with tempfile.TemporaryDirectory() as tmp:
        asm, out = os.path.join(tmp, "m.spvasm"), os.path.join(tmp, "m.spv")
        with open(asm, "w", encoding="utf-8", newline="\n") as f:
            f.write(spv)
        r1 = subprocess.run(["spirv-as", "--target-env", "vulkan1.2", asm, "-o", out],
                            capture_output=True, text=True)
        assert r1.returncode == 0, r1.stderr
        r2 = subprocess.run(["spirv-val", "--target-env", "vulkan1.2", out],
                            capture_output=True, text=True)
        assert r2.returncode == 0, r2.stderr


class Ftz32AndFmulzTests(unittest.TestCase):
    def test_hand_written_fixture_values_and_header(self):
        path = Path(HERE) / "fixtures" / "ftz32_fmulz.asm"
        lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()
                 if ln.strip() and not ln.lstrip().startswith(";")]
        spv_hw, meta_hw = _translate(lines, ftz32=True)
        spv_sw, meta_sw = _translate(lines, ftz32=False)

        self.assertIn("OpCapability DenormFlushToZero", spv_hw)
        self.assertIn("OpExecutionMode %main DenormFlushToZero 32", spv_hw)
        self.assertIn("OpExecutionMode %main DenormPreserve 16", spv_hw)
        self.assertIn("OpExecutionMode %main SignedZeroInfNanPreserve 32", spv_hw)
        self.assertNotIn("SPV_KHR_float_controls", spv_hw)
        self.assertEqual(meta_hw["float_controls"], [
            "shaderSignedZeroInfNanPreserveFloat32",
            "shaderDenormPreserveFloat16",
            "shaderDenormFlushToZeroFloat32",
        ])
        # Software flush_f32() comparisons against 2^-126 are elided under FTZ32.
        self.assertEqual(spv_hw.count("OpFOrdLessThan %bool"), 0)
        self.assertGreaterEqual(spv_sw.count("OpFOrdLessThan %bool"), 10)
        self.assertNotIn("DenormFlushToZero", spv_sw)
        self.assertEqual(meta_sw["float_controls"], [
            "shaderSignedZeroInfNanPreserveFloat32",
            "shaderDenormPreserveFloat16",
        ])

        # Expected mrt0 words:
        # v2 = legacy_mul(subnorm, qNaN) = +0.0 (0x00000000)
        # v5 = legacy_mul(-FLT_MIN, 0.5) = -0.0 (0x80000000)
        # v8 = mad_f32(2.0, 3.0, subnorm) = 6.0 (0x40C00000)
        # v9 = mad_legacy_f32(subnorm, qNaN, 2.0) = 2.0 (0x40000000)
        want = [0x00000000, 0x80000000, 0x40C00000, 0x40000000]
        self.assertEqual(support.mrt0_words(spv_hw), want)
        self.assertEqual(support.mrt0_words(spv_sw), want)
        _validate(spv_hw)
        _validate(spv_sw)

    def test_gated_off_on_div_fmas_div_scale_and_setreg_mode(self):
        # 1. s_setreg_imm32_b32 on HW_REG_MODE disables FTZ32 and restores software flushes.
        setreg_lines = [
            "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 3",
            "v_mov_b32 v0, 0x00000001",
            "v_mov_b32 v1, 0x7fc00000",
            "v_mul_legacy_f32 v2, v0, v1",
            "exp mrt0 v2, v2, v2, v2 done vm",
            "s_endpgm",
        ]
        spv_sr, meta_sr = _translate(setreg_lines, ftz32=True)
        self.assertNotIn("DenormFlushToZero", spv_sr)
        self.assertIn("OpExecutionMode %main DenormPreserve 16", spv_sr)
        self.assertNotIn("shaderDenormFlushToZeroFloat32", meta_sr["float_controls"])
        self.assertGreaterEqual(spv_sr.count("OpFOrdLessThan %bool"), 3)
        self.assertEqual(support.mrt0_words(spv_sr), [0] * 4)
        _validate(spv_sr)

        # 2. v_div_fmas_f32 disables FTZ32 (keeps subnormal inputs/intermediates on RDNA2 even under denorm32=0).
        div_fmas_lines = [
            "v_mov_b32 v1, 0x00080000",
            "v_mov_b32 v2, 0x67800000",
            "v_mov_b32 v3, 0",
            "v_cmp_eq_u32_e32 vcc_lo, 0, 1",
            "v_div_fmas_f32 v0, v1, v2, v3",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv_df, meta_df = _translate(div_fmas_lines, ftz32=True)
        self.assertNotIn("DenormFlushToZero", spv_df)
        self.assertNotIn("shaderDenormFlushToZeroFloat32", meta_df["float_controls"])
        self.assertEqual(support.mrt0_words(spv_df)[0], 0x26800000)
        _validate(spv_df)

        # 3. Direct predicate check for v_div_scale_f32, s_denorm_mode, and s_setreg on HW_REG_MODE vs non-MODE.
        self.assertTrue(rdna2spirv._disables_ftz32([parse_line("v_div_scale_f32 v0, vcc_lo, v1, v2, v1")]))
        self.assertTrue(rdna2spirv._disables_ftz32([parse_line("s_denorm_mode 0")]))
        self.assertTrue(rdna2spirv._disables_ftz32([parse_line("s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 4), 0")]))
        self.assertFalse(rdna2spirv._disables_ftz32([parse_line("s_setreg_imm32_b32 hwreg(HW_REG_FLAT_SCR_LO), 0")]))
        self.assertFalse(rdna2spirv._disables_ftz32([parse_line("v_mul_legacy_f32 v0, v1, v2")]))

    def test_v_mul_legacy_f32_edge_cases_both_modes(self):
        cases = [
            # 0 * NaN = +0.0, 0 * Inf = +0.0, (-0) * (+1) = +0.0, subnormal * NaN/Inf/normal = +0.0
            (0x00000000, 0x7FC00000, 0x00000000),
            (0x80000000, 0x7FC00001, 0x00000000),
            (0xFFC00000, 0x00000000, 0x00000000),
            (0x00000000, 0x7F800000, 0x00000000),
            (0x80000000, 0xFF800000, 0x00000000),
            (0x7F800000, 0x80000000, 0x00000000),
            (0x80000000, 0x3F800000, 0x00000000),
            (0x3F800000, 0x80000000, 0x00000000),
            (0x00000001, 0x7FC00000, 0x00000000),
            (0x807FFFFF, 0x7F800000, 0x00000000),
            # Underflow preserves product sign
            (0x00800000, 0x3F000000, 0x00000000),
            (0x80800000, 0x3F000000, 0x80000000),
            # Normal products and Inf
            (0x40000000, 0x40400000, 0x40C00000),
            (0xBF800000, 0x7F800000, 0xFF800000),
        ]
        for ftz32 in (True, False):
            for x, y, want in cases:
                with self.subTest(ftz32=ftz32, x=hex(x), y=hex(y)):
                    spv, _ = _translate([
                        f"v_mov_b32 v0, {x:#x}",
                        f"v_mov_b32 v1, {y:#x}",
                        "v_mul_legacy_f32 v2, v0, v1",
                        "exp mrt0 v2, v2, v2, v2 done vm",
                        "s_endpgm",
                    ], ftz32=ftz32)
                    self.assertEqual(support.mrt0_words(spv)[0], want)
        # Non-zero * NaN stays NaN
        for ftz32 in (True, False):
            spv, _ = _translate([
                "v_mov_b32 v0, 0x3f800000",
                "v_mov_b32 v1, 0x7fc00000",
                "v_mul_legacy_f32 v2, v0, v1",
                "exp mrt0 v2, v2, v2, v2 done vm",
                "s_endpgm",
            ], ftz32=ftz32)
            got = support.mrt0_words(spv)[0]
            self.assertEqual(got & 0x7F800000, 0x7F800000)
            self.assertNotEqual(got & 0x007FFFFF, 0)

    def test_cli_no_ftz32_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            dis = os.path.join(tmp, "disasm")
            out = os.path.join(tmp, "out")
            os.mkdir(dis)
            with open(os.path.join(dis, "ps_cli.s"), "w", encoding="utf-8") as f:
                f.write("; ps_cli\n\tv_mul_legacy_f32 v0, v1, v2\n\texp mrt0 v0, v0, v0, v0 done vm\n\ts_endpgm\n")
            with open(os.path.join(tmp, "ps_cli.dxbc"), "wb") as f:
                f.write(support.STUB_DXBC)
            cmd = [sys.executable, "-I", os.path.join(TOOLS_DIR, "rdna2spirv.py"),
                   dis, "--out", out, "--assembler", "python", "--no-validate", "--no-ftz32"]
            p = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            spvasm = Path(out, "ps_cli.spvasm").read_text(encoding="utf-8")
            self.assertNotIn("DenormFlushToZero", spvasm)
            self.assertGreaterEqual(spvasm.count("OpFOrdLessThan %bool"), 3)


if __name__ == "__main__":
    unittest.main()
