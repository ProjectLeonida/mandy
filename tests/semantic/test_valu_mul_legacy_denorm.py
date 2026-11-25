"""Bit-level DX9 legacy multiply under the inferred RDNA2 f32 denorm mode 0.

Synthetic-only tests: zero operands override infinity/NaN with positive zero,
including an input subnormal flushed to zero. A nonzero product that underflows
flushes to a zero with its sign. The product is flushed before VOP3 omod and
the scaled result is flushed again.
"""

import os
import sys
import unittest

# Allow this standalone test to run with python -I, as the other semantic tests do.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import support


def run_mul(src0, src1, mods=""):
    ins = "v_mul_legacy_f32_e64" if mods else "v_mul_legacy_f32"
    lines = [
        f"v_mov_b32 v0, {src0:#x}",
        f"v_mov_b32 v1, {src1:#x}",
        f"{ins} v2, v0, v1{(' ' + mods) if mods else ''}",
        "exp mrt0 v2, v2, v2, v2 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines("ps_mul_legacy_denorm_semantic", lines)
    words = support.mrt0_words(spv)
    assert len(words) == 4
    return words[0], spv


class LegacyMulDenormalTests(unittest.TestCase):
    def test_inputs_flush_before_legacy_zero_override(self):
        # With mode0, 2^-149 is treated as zero, so the nonzero result 2^-125
        # of an IEEE multiply must not survive. Both signs of a zero input
        # give +0 under the legacy DX9 zero-times-anything rule.
        for src0, src1 in [
            (0x00000001, 0x4B800000),
            (0x7F800000, 0x807FFFFF),
            (0xFF800000, 0x007FFFFF),
            (0x80000001, 0x4B800000),
            (0x00000001, 0x7F800000),
            (0x80000001, 0xFFC00000),
            (0x00000000, 0x7F800000),
            (0x80000000, 0xFF800000),
        ]:
            with self.subTest(src0=hex(src0), src1=hex(src1)):
                got, _ = run_mul(src0, src1)
                self.assertEqual(got, 0)

    def test_nonzero_products_flush_signed_underflow(self):
        for src0, src1, expected in [
            (0x00800000, 0x3F000000, 0x00000000),
            (0x80800000, 0x3F000000, 0x80000000),
            (0x00800000, 0x3F800000, 0x00800000),
            (0x80800000, 0x3F800000, 0x80800000),
            (0x40000000, 0x40400000, 0x40C00000),
        ]:
            with self.subTest(src0=hex(src0), src1=hex(src1)):
                got, _ = run_mul(src0, src1)
                self.assertEqual(got, expected)

    def test_product_flush_precedes_modifiers(self):
        for src0, src1, mods, expected in [
            (0x00800000, 0x3F000000, "mul:2", 0x00000000),
            (0x80800000, 0x3F000000, "mul:4", 0x00000000),
            (0x00800000, 0x3F800000, "div:2", 0x00000000),
            (0x80800000, 0x3F800000, "div:2", 0x00000000),
            (0x40000000, 0x40400000, "div:2", 0x40400000),
        ]:
            with self.subTest(src0=hex(src0), src1=hex(src1), mods=mods):
                got, _ = run_mul(src0, src1, mods)
                self.assertEqual(got, expected)

    def test_each_operand_is_tested_once(self):
        # The operand's magnitude test is the zero override itself: no flushed copy of the operand and no
        # second compare with zero. The product is flushed once, and again only after output scaling.
        frame = support.translate_lines("ps_mul_legacy_denorm_semantic", [
            "v_mov_b32 v0, 0x800000",
            "v_mov_b32 v1, 0x3f000000",
            "v_mov_b32 v2, v0",
            "exp mrt0 v2, v2, v2, v2 done vm",
            "s_endpgm",
        ])
        for mods, magnitude_tests, zero_compares in [
            ("", 3, 0),
            ("clamp", 3, 0),
            ("mul:2", 4, 1),
            ("div:2 clamp", 4, 1),
        ]:
            with self.subTest(mods=mods):
                _, spv = run_mul(0x00800000, 0x3F000000, mods)
                added = lambda op: spv.count(op) - frame.count(op)    # noqa: E731
                self.assertEqual(added("OpFOrdLessThan %bool"), magnitude_tests)
                self.assertEqual(added("OpFOrdEqual %bool"), zero_compares)
                self.assertEqual(added("OpFMul %float"), 2 if zero_compares else 1)

    def test_emits_explicit_f32_flushes_instead_of_a_driver_mode(self):
        _, spv = run_mul(0x00800000, 0x3F000000, "mul:2")
        # One check for each input, plus the product and final result.
        self.assertGreaterEqual(spv.count("OpFOrdLessThan %bool"), 4)
        self.assertIn("OpFMul %float", spv)
        self.assertIn("OpSelect %float", spv)
        # The software rule does not demand a vendor-specific float control.
        self.assertNotIn("DenormFlushToZero 32", spv)


if __name__ == "__main__":
    unittest.main()
