"""The FRONT_FACE pixel-shader system VGPR holds 0xffffffff (front) / 0 (back) (SHADER-892).

On RDNA2 the VGPR is 0xffffffff / 0 when SPI_BARYC_CNTL.FRONT_FACE_ALL_BITS = 1 and +1.0 / -1.0 when it is 0
(#5123). Every one of the game's 345 pixel shaders that reads it tests it as an integer against 0, which only the
mask form answers correctly (+1.0 and -1.0 are both non-zero). This translates a hand-written PS that reads it the
same way, after the two SV_Position components it reads, and checks what the translator stores. No game data.
"""
import os
import re
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tests"))

from test_shader_meta import isg1_chunk, make_dxbc, translate_synthetic  # noqa: E402

# SV_Position (system value 1) with x and y read, then SV_IsFrontFace (system value 9): v2, v3 = x, y; v4 = facing.
SIGNATURE = make_dxbc([isg1_chunk([("SV_Position", 0, 0, 0xF, 0x3, 1, 3),
                                   ("SV_IsFrontFace", 0, 1, 0x1, 0x1, 9, 1)])])
LINES = [
    "v_cmp_ne_u32_e32 vcc_lo, 0, v4",
    "v_cndmask_b32_e64 v0, 0, 1.0, vcc_lo",
    "exp mrt0 v0, v0, v0, v0 done vm",
    "s_endpgm",
]


class FrontFaceMaskTest(unittest.TestCase):
    def test_front_face_is_stored_as_all_bits_or_zero(self):
        asm, _ = translate_synthetic("ps_front_face", LINES, dxbc=SIGNATURE)
        self.assertIn("BuiltIn FrontFacing", asm)
        select = re.search(r"%ffu = OpSelect %uint %ff (%\S+) (%\S+)", asm)
        self.assertIsNotNone(select, "FRONT_FACE is not stored as a uint select")
        consts = dict(re.findall(r"(%\S+) = OpConstant %uint (\d+)", asm))
        self.assertEqual(int(consts[select.group(1)]), 0xFFFFFFFF)  # front
        self.assertEqual(int(consts[select.group(2)]), 0)           # back
        # The float form (+1.0 / -1.0) is gone: nothing negates 1.0 for the facing value.
        self.assertNotIn("%ffm1", asm)
        # And the value goes into the VGPR the shader reads, v4.
        store = re.search(r"%ffp = OpAccessChain %_ptr_Function_uint %vgpr (%\S+)", asm)
        self.assertIsNotNone(store)
        self.assertEqual(int(consts[store.group(1)]), 4)


if __name__ == "__main__":
    unittest.main()
