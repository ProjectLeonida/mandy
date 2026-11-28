"""The 4x8-bit dot products as SPV_KHR_integer_dot_product instructions, and their --integer-dot-fallback form.

v_dot4_i32_i8 / v_dot4c_i32_i8 become OpSDot on the two dwords read as packed signed bytes, v_dot4_u32_u8 /
v_dot4c_u32_u8 OpUDot on packed unsigned ones (PackedVectorFormat4x8Bit), and the addend is added with OpIAdd, which
wraps at 32 bits as the instructions do. Under clamp the hardware saturates the addend plus the exact sum, which is
what OpSDotAccSat / OpUDotAccSat compute. op_sel picks 16-bit halves of the sources (measured on gfx1036,
test_alu_dot4_i8.py); the halves are put together into one dword first (Gen.dot4_halves), so the instruction still
sees four bytes. The expanded form (four extracts, multiplies and adds) stays for --integer-dot-fallback and must give
the same words. docs/design.md, "Signed-byte dot products".

    python -m unittest discover -s tests/semantic -p test_packed_dot4.py
"""
import inspect
from pathlib import Path
import unittest

from support import HERE, mrt0_words, rdna2spirv, translate_listing

FIXTURES = Path(HERE) / "fixtures"
I8_WORDS = [0x8000FC03, 0x7FFFFFFF, 0x80000000, 0x0000003E]
U8_WORDS = [0x00004288, 0x0003F803, 0xFFFFFFFF, 0x00008506]


def translate_text(name, data, fallback=False):
    saved = rdna2spirv.INTEGER_DOT_FALLBACK
    rdna2spirv.INTEGER_DOT_FALLBACK = fallback
    try:
        return translate_listing(name, data)
    finally:
        rdna2spirv.INTEGER_DOT_FALLBACK = saved


def translate(name, fallback=False):
    return translate_text("ps_" + name, (FIXTURES / (name + ".asm")).read_bytes(), fallback)


def body(text):
    """The module from its first function on: what the instructions became, without the header."""
    return text[text.index(" = OpFunction "):]


class PackedDot4(unittest.TestCase):
    def test_signed_bytes(self):
        text = translate("packed_dot4_i8")
        self.assertEqual(mrt0_words(text), I8_WORDS)
        self.assertEqual(text.count("OpSDotKHR %int"), 2)          # the plain form and op_sel
        self.assertEqual(text.count("OpSDotAccSatKHR %int"), 2)    # both clamps
        self.assertNotIn("OpBitFieldSExtract", body(text))
        self.assertIn("OpCapability DotProductKHR", text)
        self.assertIn("OpCapability DotProductInput4x8BitPackedKHR", text)
        self.assertIn('OpExtension "SPV_KHR_integer_dot_product"', text)

    def test_unsigned_bytes(self):
        text = translate("packed_dot4_u8")
        self.assertEqual(mrt0_words(text), U8_WORDS)
        self.assertEqual(text.count("OpUDotKHR %uint"), 4)         # v_dot4_u32_u8 twice, v_dot4c_u32_u8 twice
        self.assertEqual(text.count("OpUDotAccSatKHR %uint"), 1)
        self.assertNotIn("OpBitFieldUExtract", body(text))

    def test_fallback_gives_the_same_words(self):
        for name, words in (("packed_dot4_i8", I8_WORDS), ("packed_dot4_u8", U8_WORDS)):
            with self.subTest(name=name):
                text = translate(name, fallback=True)
                self.assertEqual(mrt0_words(text), words)
                self.assertNotIn("DotProduct", text)
                self.assertNotIn("SPV_KHR_integer_dot_product", text)
                self.assertNotIn("Dot", body(text))

    def test_the_index_names_the_feature(self):
        # shader_meta maps each capability to the device feature the runtime must enable.
        self.assertIn('"DotProductKHR": "shaderIntegerDotProduct"', inspect.getsource(rdna2spirv.shader_meta))

    def test_the_halves_op_sel_picks(self):
        # Every op_sel / op_sel_hi pair the rule takes for S1, against the reference of test_alu_dot4_i8: bytes 0..1
        # from half `lo`, bytes 2..3 from half `hi`. Both forms must agree with it.
        def s8(byte):
            return byte - 256 if byte & 0x80 else byte
        a, b = 0x7F01FE80, 0x02FD7F81
        for lo, hi in ((0, 0), (0, 1), (1, 0), (1, 1)):
            halves = ((b >> (16 * lo)) & 0xFFFF) | (((b >> (16 * hi)) & 0xFFFF) << 16)
            want = sum(s8((a >> (8 * i)) & 0xFF) * s8((halves >> (8 * i)) & 0xFF) for i in range(4)) & 0xFFFFFFFF
            lines = (f"\tv_mov_b32 v1, 0x{a:08x}\n\tv_mov_b32 v2, 0x{b:08x}\n\tv_mov_b32 v3, 0\n"
                     f"\tv_dot4_i32_i8 v8, v1, v2, v3 op_sel:[0,{lo},0] op_sel_hi:[1,{hi},1]\n"
                     "\texp mrt0 v8, v8, v8, v8 done vm\n\ts_endpgm\n")
            for fallback in (False, True):
                with self.subTest(lo=lo, hi=hi, fallback=fallback):
                    text = translate_text("ps_dot4_halves", lines.encode(), fallback)
                    self.assertEqual(mrt0_words(text), [want] * 4)


if __name__ == "__main__":
    unittest.main()
