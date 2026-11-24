"""v_mul_f64, v_fma_f64, v_min_f64, v_max_f64 and v_cvt_f64_f32 through the translator (SHADER-F64-VALU-OPS).

Each program loads its operands into VGPR pairs, runs one instruction into v[4:5] and exports both words, which the CPU interpreter computes from the emitted SPIR-V. The
expected values are written out by hand from the RDNA2 ISA (docs/design.md, "Binary64 multiply, FMA, min/max and
conversions"); the integer conversions are in test_valu_cvt_int_f64.py. Until this change the translator refused
all of these ops by name (Unsupported).
Standard library only; no game data.
"""
import os
import struct
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from support import mrt0_words, translate_lines  # noqa: E402  (support puts tools/ on the path)
import rdna2spirv  # noqa: E402

QNAN = 0x7FF8000000000000
NEG_QNAN_PAYLOAD = 0xFFF8000000000123
INF = float("inf")


def bits(x):
    """The binary64 word of a Python float, or an int taken as the word itself."""
    return x if isinstance(x, int) else struct.unpack("<Q", struct.pack("<d", x))[0]


def value(word):
    return struct.unpack("<d", struct.pack("<Q", word))[0]


def run(op_line, x=0.0, y=0.0, z=0.0):
    """The binary64 word left in v[4:5] after op_line, with v[0:1] = x, v[2:3] = y and v[6:7] = z."""
    lines = []
    for reg, v in ((0, x), (2, y), (6, z)):
        w = bits(v)
        lines += [f"v_mov_b32 v{reg}, {w & 0xFFFFFFFF:#x}", f"v_mov_b32 v{reg + 1}, {w >> 32:#x}"]
    lines += ["v_mov_b32 v4, 0xdeadbeef", "v_mov_b32 v5, 0xdeadbeef", op_line,
              "exp mrt0 v4, v5, v4, v5 done vm", "s_endpgm"]
    lo, hi = mrt0_words(translate_lines("ps_f64_ops", lines))[:2]
    return lo | hi << 32


class MulF64(unittest.TestCase):
    def test_products(self):
        for x, y, want in ((1.5, -2.25, -3.375), (3.0, 1.0 / 3.0, 3.0 * (1.0 / 3.0)), (1e300, 1e10, INF),
                           (-0.0, 5.0, -0.0), (2.0 ** -1000, 2.0 ** -60, 2.0 ** -1060)):
            with self.subTest(x=x, y=y):
                self.assertEqual(run("v_mul_f64 v[4:5], v[0:1], v[2:3]", x, y), bits(want))

    def test_invalid_product_is_nan(self):
        w = run("v_mul_f64 v[4:5], v[0:1], v[2:3]", INF, 0.0)
        self.assertNotEqual(value(w), value(w))

    def test_source_modifiers_literal_and_output_modifiers(self):
        self.assertEqual(run("v_mul_f64 v[4:5], -v[0:1], |v[2:3]|", 1.5, -2.0), bits(-3.0))
        # A hex literal dword is bits 63:32 of the f64 operand (docs/design.md, f64 literal sources): 2.0.
        self.assertEqual(run("v_mul_f64 v[4:5], 0x40000000, v[0:1]", 1.5), bits(3.0))
        self.assertEqual(run("v_mul_f64 v[4:5], v[0:1], 0.5", 3.0), bits(1.5))
        self.assertEqual(run("v_mul_f64 v[4:5], v[0:1], v[2:3] clamp", 1.5, 2.0), bits(1.0))
        self.assertEqual(run("v_mul_f64 v[4:5], v[0:1], v[2:3] clamp", -1.5, 2.0), 0)


class FmaF64(unittest.TestCase):
    def test_one_rounding(self):
        # a * a - (1 + 2^-29) with a = 1 + 2^-30 is exactly 2^-60; rounding the product first gives 0.
        a = 1.0 + 2.0 ** -30
        self.assertEqual(run("v_fma_f64 v[4:5], v[0:1], v[0:1], v[2:3]", a, -(1.0 + 2.0 ** -29)), bits(2.0 ** -60))

    def test_values_and_modifiers(self):
        self.assertEqual(run("v_fma_f64 v[4:5], v[0:1], v[2:3], v[6:7]", 1.5, 2.0, 0.25), bits(3.25))
        self.assertEqual(run("v_fma_f64 v[4:5], v[0:1], 2.0, 0x3ff00000", 1.5), bits(4.0))
        self.assertEqual(run("v_fma_f64 v[4:5], -v[0:1], v[2:3], v[6:7]", 1.5, 2.0, 0.25), bits(-2.75))
        self.assertEqual(run("v_fma_f64 v[4:5], v[0:1], v[2:3], v[6:7] div:2", 1.5, 2.0, 1.0), bits(2.0))


class MinMaxF64(unittest.TestCase):
    def test_ordered_values(self):
        for x, y in ((1.5, -2.0), (-2.0, 1.5), (INF, -INF), (1e-310, 2e-310)):
            with self.subTest(x=x, y=y):
                self.assertEqual(run("v_min_f64 v[4:5], v[0:1], v[2:3]", x, y), bits(min(x, y)))
                self.assertEqual(run("v_max_f64 v[4:5], v[0:1], v[2:3]", x, y), bits(max(x, y)))

    def test_one_nan_gives_the_other_operand(self):
        for op in ("v_min_f64", "v_max_f64"):
            with self.subTest(op=op):
                self.assertEqual(run(f"{op} v[4:5], v[0:1], v[2:3]", QNAN, -2.0), bits(-2.0))
                self.assertEqual(run(f"{op} v[4:5], v[0:1], v[2:3]", 3.0, NEG_QNAN_PAYLOAD), bits(3.0))

    def test_two_nans_give_src0_bits(self):
        for op in ("v_min_f64", "v_max_f64"):
            with self.subTest(op=op):
                self.assertEqual(run(f"{op} v[4:5], v[0:1], v[2:3]", NEG_QNAN_PAYLOAD, QNAN), NEG_QNAN_PAYLOAD)

    def test_modifiers(self):
        self.assertEqual(run("v_max_f64 v[4:5], -v[0:1], |v[2:3]|", 1.5, -0.5), bits(0.5))
        self.assertEqual(run("v_max_f64 v[4:5], v[0:1], v[2:3] mul:4 clamp", 0.5, 0.125), bits(1.0))


class CvtF64F32(unittest.TestCase):
    def run_f32(self, word, line="v_cvt_f64_f32_e32 v[4:5], v0"):
        return run(line, word)       # the f32 source word is the low word of v[0:1]

    def test_exact_widening(self):
        for f32_word, want in ((0x3FC00000, 1.5), (0xC7EFFFFF, -122879.9921875), (0x7F7FFFFF, 3.4028234663852886e38),
                               (0x00800000, 2.0 ** -126), (0x7F800000, INF), (0xFF800000, -INF), (0x80000000, -0.0)):
            with self.subTest(f32=hex(f32_word)):
                self.assertEqual(self.run_f32(f32_word), bits(want))

    def test_denormal_source_reads_as_signed_zero(self):
        # The game's MODE flushes f32 denormals (docs/design.md, as for v_floor/v_ceil/v_fract_f32).
        self.assertEqual(self.run_f32(0x00000001), 0)
        self.assertEqual(self.run_f32(0x807FFFFF), bits(-0.0))

    def test_nan_stays_nan(self):
        w = self.run_f32(0x7FC00000)
        self.assertNotEqual(value(w), value(w))

    def test_modifiers(self):
        self.assertEqual(self.run_f32(0x3FC00000, "v_cvt_f64_f32_e64 v[4:5], -v0"), bits(-1.5))
        self.assertEqual(self.run_f32(0x3FC00000, "v_cvt_f64_f32_e64 v[4:5], v0 clamp"), bits(1.0))
        self.assertEqual(run("v_cvt_f64_f32_e32 v[4:5], 0.5"), bits(0.5))


class Refusals(unittest.TestCase):
    def refused(self, lines):
        with self.assertRaises(rdna2spirv.Unsupported):
            translate_lines("ps_f64_refused", lines + ["exp mrt0 v4, v5, v4, v5 done vm", "s_endpgm"])

    def test_directed_rounding_mode(self):
        # Only v_add_f64 models a MODE.FP_ROUND write for f64; the product would be rounded to nearest.
        for line in ("v_mul_f64 v[4:5], v[0:1], v[2:3]", "v_fma_f64 v[4:5], v[0:1], v[2:3], v[6:7]"):
            with self.subTest(line=line):
                self.refused(["s_round_mode 0x4", line])

    def test_bad_modifiers(self):
        self.refused(["v_mul_f64 v[4:5], v[0:1], v[2:3] mul:2 div:2"])
        self.refused(["v_cvt_i32_f64_e64 v4, v[0:1] clamp"])


if __name__ == "__main__":
    unittest.main()
