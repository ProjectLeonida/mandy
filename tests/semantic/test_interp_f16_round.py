"""How the hardware rounds a packed 16-bit varying's interpolation (SHADER-468, measured on gfx1036).

With FP16_INTERP_MODE the SPI writes P10/P20 as **per-half f16 differences** of the producer's words, and
`v_interp_p2_f16` narrows the f32 result **toward zero**; the weights and the sum are f32. Measured through a native
draw on the RDNA2 iGPU of a Ryzen 5 7600 (gfx1036, AMD Windows driver 24.10.24, Vulkan 1.3.280): a model with an f16
delta matches 65,536 of 65,536 samples (262,144 across seven draws) where the f32 difference does not, and toward-zero
narrowing matches all of them where nearest-even matches 39,607. `tools/f16_interp_probe.py` reproduces it.

The first case is the one that tells the two delta models apart: -65504 and 1000 are 66504 apart, past the largest
f16, so an f16 P10 overflows to inf while the f32 difference stays finite.
"""
import struct
import unittest

import support

BARY_QUARTERS = (0.5, 0.25, 0.25)                     # i = j = 0.25
OVERFLOW = ((-65504.0, 0.0), (1000.0, 1.0), (-65504.0, 0.0))     # (low half, high half) per vertex
BARY_TWO_FIFTHS = (0.6, 0.4, 0.0)                     # i = 0.4
THREE = ((0.0, 0.0), (3.0, 0.0), (0.0, 0.0))

SHADER = """v_interp_p1ll_f16 v2, v0, attr0.x
v_interp_p2_f16 v2, v1, attr0.x, v2
v_interp_p1ll_f16 v3, v0, attr0.x high
v_interp_p2_f16 v3, v1, attr0.x, v3 high
exp mrt0 v2, v3, v3, v3 done vm
s_endpgm
"""


def pack(lo, hi):
    return struct.unpack("<I", struct.pack("<ee", lo, hi))[0]


def run(halves, bary):
    attr0 = [(pack(lo, hi), 0, 0, 0) for lo, hi in halves]
    spv = support.translate_lines("ps_interp_f16_round", SHADER.splitlines())
    return support.mrt0_words(spv, inputs=support.varyings(spv, {0: attr0}, bary))


class InterpF16Round(unittest.TestCase):
    def test_p10_is_an_f16_difference(self):
        """P10 for the low half is f16(1000 - -65504), which is inf, not the f32 difference 66504."""
        words = run(OVERFLOW, BARY_QUARTERS)
        self.assertEqual(f"{words[0] & 0xFFFF:#06x}", "0x7c00")      # +inf: -65504 + 0.25 * inf
        self.assertEqual(f"{words[1] & 0xFFFF:#06x}", "0x3400")      # the high halves are 0, 1, 0: i = 0.25

    def test_p2_narrows_toward_zero(self):
        """0.4 * 3 is 1.20000005 in f32, between f16 1.19921875 and 1.2001953125: toward zero takes the lower one."""
        words = run(THREE, BARY_TWO_FIFTHS)
        self.assertEqual(f"{words[0] & 0xFFFF:#06x}", "0x3ccc")      # 1.19921875; nearest even gives 0x3ccd


if __name__ == "__main__":
    unittest.main()
