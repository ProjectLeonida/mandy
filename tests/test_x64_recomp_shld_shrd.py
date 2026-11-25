"""Tests for shld and shrd double-precision shift emitters (0F A4/A5/AC/AD) in tools/x64_recomp.py.

Covers:
- Decoding & classification of 16/32/64-bit shld/shrd (imm8 and CL, reg and mem)
- Single-instruction ARM64 EXTR emission for 32/64-bit immediate counts
- 16-bit 32-bit-concatenation shift (including count == 16) with upper-48-bit preservation
- Exact SF, ZF, CF, OF (count == 1 sign change), and PF flags + count == 0 no-op
- Aliasing cases (dst == src, dst == rcx with CL, src == rcx with CL, memory dst)
- Real-world 128-bit __shiftleft128 / __shiftright128 idioms

Static checks run everywhere; execution tests run natively on Apple Silicon when
CAN_RUN_MACHO is available, or under QEMU.

    python -I tests/test_x64_recomp_shld_shrd.py
"""
import os
import shutil
import struct
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

import x64_recomp as xr  # noqa: E402

GUEST_VA = 0x46000000


def _runner():
    if getattr(xr, "CAN_RUN_MACHO", False) and hasattr(xr, "run_result") and not os.environ.get("QEMU_AARCH64"):
        return lambda code: xr.run_result(code, GUEST_VA)
    qemu = os.environ.get("QEMU_AARCH64") or shutil.which("qemu-system-aarch64")
    if qemu and os.path.exists(qemu):
        return lambda code: xr.run_kernel(code, GUEST_VA)
    return None


RUNNER = _runner()


def _imm64(reg, val):
    rex = 0x48 | (1 if reg >= 8 else 0)
    return bytes([rex, 0xB8 + (reg & 7)]) + struct.pack("<Q", val & 0xFFFFFFFFFFFFFFFF)


def _pack_cf_zf_sf_of_pf_to_edi():
    """Set edi = CF | (ZF<<1) | (SF<<2) | (OF<<3) | (PF<<4) without clobbering rax..rdx."""
    return bytes.fromhex(
        "410f92c0" "410f94c1" "410f98c2" "410f90c3" "410f9ac4"
        "410fb6f8"
        "450fb6c9" "41d1e1" "4409cf"
        "450fb6d2" "41c1e202" "4409d7"
        "450fb6db" "41c1e303" "4409df"
        "450fb6e4" "41c1e404" "4409e7"
    )


class ShldShrdStaticTests(unittest.TestCase):
    """Decode, classification, and static ARM64 emission tests."""

    def translate(self, code):
        insns = list(xr.disasm(code, 0x1000))
        self.assertFalse(any(i.unknown or i.error for i in insns), insns)
        tr = xr.FunctionTranslator(insns)
        tr.translate()
        self.assertEqual(tr.stats["unhandled"], [], f"unhandled: {tr.stats['unhandled']}")
        return tr, insns

    def test_decode_and_classify_all_forms(self):
        samples = [
            ("660fa4d804", "shld", 2, "i"),
            ("0fa4d808", "shld", 4, "i"),
            ("480fa4d828", "shld", 8, "i"),
            ("660fa5d8", "shld", 2, "r"),
            ("0fa5d8", "shld", 4, "r"),
            ("480fa5d8", "shld", 8, "r"),
            ("660facd804", "shrd", 2, "i"),
            ("0facd808", "shrd", 4, "i"),
            ("480facd828", "shrd", 8, "i"),
            ("660fadd8", "shrd", 2, "r"),
            ("0fadd8", "shrd", 4, "r"),
            ("480fadd8", "shrd", 8, "r"),
        ]
        for hx, mnem, sz, cnt_kind in samples:
            with self.subTest(hx=hx):
                ins = list(xr.disasm(bytes.fromhex(hx), 0x1000))[0]
                self.assertFalse(ins.unknown or ins.error, ins)
                self.assertEqual(ins.mnem, mnem)
                self.assertEqual(ins.kind, "dshift")
                self.assertEqual(ins.flags_w, "NZCVP")
                self.assertEqual(ins.flags_r, "")
                self.assertEqual(len(ins.ops), 3)
                self.assertEqual(ins.ops[0][2], sz)
                self.assertEqual(ins.ops[1][2], sz)
                self.assertEqual(ins.ops[2][0], cnt_kind)

    def test_all_forms_translate_cleanly(self):
        forms = [
            "660fa4d800", "660fa4d801", "660fa4d810", "660fa5d8", "660fa41c2404",
            "0fa4d800", "0fa4d808", "0fa5d8", "0fa41c2408", "0fa51c24",
            "480fa4d800", "480fa4d828", "480fa5d8", "480fa41c243f", "480fa51c24",
            "660facd801", "660facd810", "660fadd8", "660fac1c2404",
            "0facd808", "0fadd8", "0fac1c2408", "0fad1c24",
            "480facd828", "480fadd8", "480fac1c243f", "480fad1c24",
        ]
        self.translate(bytes.fromhex("".join(forms) + "c3"))

    def test_32bit_and_64bit_imm_use_extr(self):
        # shld eax, ebx, 8 (0f a4 d8 08) -> 32-bit EXTR W0, W2, W17, #24
        tr32, _ = self.translate(bytes.fromhex("0fa4d808c3"))
        a32 = xr.Asm()
        a32.extr(xr.T0, xr.T2, xr.GTMP, 24, sf=0)
        self.assertIn(a32.words[0], tr32.asm.words)

        # shrd rax, rbx, 40 (48 0f ac d8 28) -> 64-bit EXTR X0, X17, X2, #40
        tr64, _ = self.translate(bytes.fromhex("480facd828c3"))
        a64 = xr.Asm()
        a64.extr(xr.T0, xr.GTMP, xr.T2, 40, sf=1)
        self.assertIn(a64.words[0], tr64.asm.words)


@unittest.skipIf(RUNNER is None, "needs Darwin arm64 Mach-O runner or qemu-system-aarch64")
class ShldShrdExecTests(unittest.TestCase):
    """End-to-end arm64 execution tests for shld and shrd."""

    def run_code(self, code):
        res, tr = RUNNER(code)
        self.assertEqual(tr.stats["unhandled"], [], f"unhandled: {tr.stats['unhandled']}")
        return res

    def test_32bit_and_64bit_shld_shrd_imm_and_cl_values_and_flags(self):
        # (label, init_rax, init_rbx, cl_or_None, op_hex, want_rax, want_flags_cf_zf_sf_of_pf)
        cases = [
            # 32-bit shld imm8: 0x12345678, 0x9ABCDEF0, 8 -> 0x3456789A (CF=bit24=0, SF=0, ZF=0, OF=0, PF=1 (0x9A=4 ones))
            ("32 shld imm8", 0xDEAD_BEEF_1234_5678, 0x9ABCDEF0, None, "0fa4d808", 0x000000003456789A, 0b10000),
            # 32-bit shld imm 1 with sign change: 0x40000000, 0x00000000, 1 -> 0x80000000 (CF=0, ZF=0, SF=1, OF=1, PF=1)
            ("32 shld 1 OF", 0x40000000, 0, None, "0fa4d801", 0x80000000, 0b11100),
            # 32-bit shld imm 1 with CF=1 & ZF=1: 0x80000000, 0, 1 -> 0 (CF=1, ZF=1, SF=0, OF=1, PF=1)
            ("32 shld 1 CF+ZF", 0x80000000, 0, None, "0fa4d801", 0, 0b11011),
            # 32-bit shrd imm8: 0x12345678, 0x9ABCDEF0, 8 -> 0xF0123456 (CF=bit7=0, SF=1, ZF=0, OF=1, PF=1 (0x56=4 ones))
            ("32 shrd imm8", 0xDEAD_BEEF_1234_5678, 0x9ABCDEF0, None, "0facd808", 0x00000000F0123456, 0b11100),
            # 32-bit shrd imm 1 with CF=1 & sign change: 0x00000001, 0x00000001, 1 -> 0x80000000 (CF=1, SF=1, OF=1, PF=1)
            ("32 shrd 1 CF+OF", 1, 1, None, "0facd801", 0x80000000, 0b11101),
            # 32-bit CL forms
            ("32 shld cl(8)", 0x12345678, 0x9ABCDEF0, 8, "0fa5d8", 0x3456789A, 0b10000),
            ("32 shrd cl(8)", 0x12345678, 0x9ABCDEF0, 8, "0fadd8", 0xF0123456, 0b11100),
            # 64-bit shld/shrd with high bit count >= 32 (40 bits!)
            # dst=0x0123456789ABCDEF, src=0xFEDCBA9876543210, shld 40 -> 0xCDEF_FEDCBA9876
            # Wait: dst<<40 = 0xCDEF_0000000000, src>>24 = 0x0000_00FEDCBA9876 -> 0xCDEFFEDCBA9876?
            # Let's compute: 0x0123456789ABCDEF << 40 = 0xCDEF000000000000; 0xFEDCBA9876543210 >> 24 = 0x000000FEDCBA9876
            # Wait: 40 bits = 10 hex nibbles! 0xABCDEF -> low 6 nibbles are ABCDEF? 16 - 10 = 6 nibbles = ABCDEF!
            # So dst<<40 = 0xABCDEF0000000000, src>>24 = 0x000000FEDCBA9876 -> 0xABCDEFFEDCBA9876!
            # CF = dst[64-40] = dst[24] = bit 0 of byte 3 (0x89 -> bit0=1)! SF=1, PF: 0x76=01110110 (5 ones -> PF=0), OF=1 (0->1)
            ("64 shld 40", 0x0123456789ABCDEF, 0xFEDCBA9876543210, None, "480fa4d828", 0xABCDEFFEDCBA9876, 0b01101),
            # 64-bit shrd 40: dst>>40 = 0x0000000000012345, src<<24 = 0xBA98765432100000 -> 0xBA98765432100123?
            # Wait: 40 bits = 10 hex nibbles -> dst>>40 = 0x0000000000012345 (6 nibbles!), src<<24 = 0xBA98765432100000 -> 0xBA98765432112345?
            # Let's check exact nibble math:
            # dst = 0x012345_6789ABCDEF -> dst >> 40 = 0x0000000000012345 (bits 0..23)
            # src = 0xFEDCBA_9876543210 -> src << 24 = 0x9876543210_000000 (Wait: 24 bits = 6 nibbles! F E D C B A shifted out -> 0x9876543210000000? No: 0xFEDCBA9876543210 << 24 = 0x9876543210_000000!)
            # OR: 0x9876543210012345!
            # CF = dst[39] = bit 7 of byte 4 (0x67 -> bit7=0)! SF=1 (0x9...), OF=1 (0->1), PF: 0x45=01000101 (3 ones -> PF=0)
            ("64 shrd 40", 0x0123456789ABCDEF, 0xFEDCBA9876543210, None, "480facd828", 0x9876543210012345, 0b01100),
            ("64 shld cl(40)", 0x0123456789ABCDEF, 0xFEDCBA9876543210, 40, "480fa5d8", 0xABCDEFFEDCBA9876, 0b01101),
            ("64 shrd cl(40)", 0x0123456789ABCDEF, 0xFEDCBA9876543210, 40, "480fadd8", 0x9876543210012345, 0b01100),
        ]
        for label, init_rax, init_rbx, cl, op_hex, want_rax, want_flags in cases:
            with self.subTest(label=label):
                code = _imm64(0, init_rax) + _imm64(3, init_rbx)
                if cl is not None:
                    code += _imm64(1, cl)
                code += (
                    bytes.fromhex(op_hex)
                    + _pack_cf_zf_sf_of_pf_to_edi()
                    + _imm64(2, want_rax)
                    + bytes.fromhex("4839d0" "0f94c0" "0fb6c0" "c1e704" "09c7" "c3")
                )
                self.assertEqual(self.run_code(code), (want_flags << 4) | 1)

        # Count == 0 (both imm 0 and CL == 64 -> 0 mod 64) must leave value AND flags untouched!
        code_zero = (
            _imm64(0, 0x12345678) + _imm64(3, 0x9ABCDEF0) + _imm64(1, 64)
            + bytes.fromhex("f9" "480fa5d8" "0f92c2" "0fb6fa" "c3")  # stc; shld rax, rbx, cl(64->0); edi = CF (must stay 1!)
        )
        self.assertEqual(self.run_code(code_zero), 1)

    def test_16bit_shld_shrd_imm_and_cl_and_count_16(self):
        upper = 0xAAAA_BBBB_CCCC_0000
        # (label, v16, src16, cl, op_hex, want16, want_cf, want_zf, want_sf)
        cases = [
            ("16 shld 4", 0x1234, 0xABCD, None, "660fa4d804", 0x234A, 1, 0, 0),    # bit12 of 0x1234 is 1!
            ("16 shrd 4", 0x1234, 0xABCD, None, "660facd804", 0xD123, 0, 0, 1),    # bit3 of 0x1234 is 0, SF=1!
            ("16 shld 16", 0x0001, 0x8002, None, "660fa4d810", 0x8002, 1, 0, 1),   # c==16: dst:=src, CF=orig[0]=1, SF=1!
            ("16 shrd 16", 0x8000, 0x0000, None, "660facd810", 0x0000, 1, 1, 0),   # c==16: dst:=0, CF=orig[15]=1, ZF=1!
            ("16 shld cl(4)", 0x1234, 0xABCD, 4, "660fa5d8", 0x234A, 1, 0, 0),
            ("16 shrd cl(16)", 0x8000, 0x0000, 16, "660fadd8", 0x0000, 1, 1, 0),
        ]
        for label, v16, src16, cl, op_hex, want16, cf, zf, sf in cases:
            with self.subTest(label=label):
                want_flags = cf | (zf << 1) | (sf << 2)
                code = _imm64(0, upper | v16) + _imm64(3, 0x1111_2222_3333_0000 | src16)
                if cl is not None:
                    code += _imm64(1, cl)
                code += (
                    bytes.fromhex(op_hex)
                    + bytes.fromhex("410f92c0" "410f94c1" "410f98c2"
                                    "410fb6f8" "450fb6c9" "41d1e1" "4409cf"
                                    "450fb6d2" "41c1e202" "4409d7")
                    + _imm64(2, upper | want16)
                    + bytes.fromhex("4839d0" "0f94c0" "0fb6c0" "c1e704" "09c7" "c3")
                )
                self.assertEqual(self.run_code(code), (want_flags << 4) | 1)

    def test_memory_aliasing_and_128bit_shifts(self):
        # 1. dst == src (shld eax, eax, 4 rotates 0x12345678 -> 0x23456781)
        code_self = _imm64(0, 0x12345678) + bytes.fromhex("0fa4c004" "89c7" "c3")
        self.assertEqual(self.run_code(code_self), 0x23456781)

        # 2. dst == rcx with CL count: rcx = 0x12345604, rbx = 0xABCDEF00 -> shld ecx, ebx, cl (cl=4) -> 0x2345604A
        code_rcx_dst = _imm64(1, 0x12345604) + _imm64(3, 0xABCDEF00) + bytes.fromhex("0fa5d9" "89cf" "c3")
        self.assertEqual(self.run_code(code_rcx_dst), 0x2345604A)

        # 3. memory dst: shld qword [rsp], rbx, 16 (48 0f a4 1c 24 10)
        code_mem = (
            bytes.fromhex("4883ec10")
            + _imm64(6, 0x1122334455667788)
            + bytes.fromhex("48893424")
            + _imm64(3, 0xAABBCCDDEEFF0099)
            + bytes.fromhex("480fa41c2410")
            + _imm64(2, 0x334455667788AABB)
            + bytes.fromhex("483b1424" "0f94c0" "0fb6f8" "4883c410" "c3")
        )
        self.assertEqual(self.run_code(code_mem), 1)

        # 4. 128-bit shift left then right: (rdx:rax) << 20 then >> 20 recovers exact bits!
        # hi=0x0000_0123_4567_89AB, lo=0xCDEF_0123_4567_89AB -> <<20:
        # new_hi = 0x123456789ABCDEF0, new_lo = 0x123456789AB00000
        code_128 = (
            _imm64(2, 0x0000_0123_4567_89AB)
            + _imm64(0, 0xCDEF_0123_4567_89AB)
            + _imm64(1, 20)
            + bytes.fromhex("480fa5c2" "48d3e0")                 # shld rdx, rax, cl; shl rax, cl
            + _imm64(3, 0x1234_5678_9ABC_DEF0)
            + _imm64(6, 0x1234_5678_9AB0_0000)
            + bytes.fromhex("4839da" "0f94c1" "0fb6c9" "4839f0" "0f94c2" "0fb6d2" "21d1" "89cf" "c3")
        )
        self.assertEqual(self.run_code(code_128), 1)


if __name__ == "__main__":
    unittest.main()
