"""Tests for 8/16-bit adc/sbb, sign-extended immediates, and AF stashing in tools/x64_recomp.py.

Regression coverage for:
- 8-bit and 16-bit adc and sbb (reg, hi8, mem, imm) with bit-exact result and SF/ZF/CF/OF/PF/AF
- Sign-extension of narrow immediates (83 /2 ib, 83 /3 ib, 48 81 /2 id, 48 81 /3 id) to dst width
- CTX_A / CTX_B stashing on adc/sbb for AF materialization across all 4 widths

Static checks run everywhere; execution tests run natively on Apple Silicon when
CAN_RUN_MACHO is available, or under QEMU.

    python -I tests/test_x64_recomp_adcsbb.py
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


class AdcSbbStaticTests(unittest.TestCase):
    """Static translation & ARM64 encoding checks that run on every platform."""

    def translate(self, code):
        insns = list(xr.disasm(code, 0x1000))
        self.assertFalse(any(i.unknown or i.error for i in insns), insns)
        tr = xr.FunctionTranslator(insns)
        tr.translate()
        self.assertEqual(tr.stats["unhandled"], [], f"unhandled: {tr.stats['unhandled']}")
        return tr

    def test_csinv_csetm_encoding(self):
        a = xr.Asm()
        a.csetm(2, "cs", sf=0)
        a.csetm(2, "cc", sf=0)
        a.csetm(2, "eq", sf=1)
        self.assertEqual(a.words, [0x5A9F33E2, 0x5A9F23E2, 0xDA9F13E2])

    def test_8bit_and_16bit_adc_sbb_translate_cleanly(self):
        forms = [
            "10d8", "12c3", "14ff", "80d0ff", "101c24",        # 8-bit adc reg/hi8/imm/mem
            "18d8", "1ac3", "1cff", "80d8ff", "181c24",        # 8-bit sbb reg/hi8/imm/mem
            "6611d8", "6615ffff", "6683d0ff", "66111c24",      # 16-bit adc reg/imm16/imm8/mem
            "6619d8", "661dffff", "6683d8ff", "66191c24",      # 16-bit sbb reg/imm16/imm8/mem
        ]
        self.translate(bytes.fromhex("".join(forms) + "c3"))

    def test_adc_sbb_sign_extend_imm8_to_dst_width(self):
        # 48 83 d0 ff (adc rax, -1): must load -1 via MOVN X1, #0 (0x92800001), NOT MOVZ X1, #0xFF (0xD2801FE1)!
        tr64 = self.translate(bytes.fromhex("4883d0ffc3"))
        a_ref = xr.Asm()
        a_ref.movn(xr.T1, 0)
        a_bad = xr.Asm()
        a_bad.movz(xr.T1, 0xFF)
        self.assertIn(a_ref.words[0], tr64.asm.words, "adc rax, -1 must sign-extend imm8 to 64-bit -1")
        self.assertNotIn(a_bad.words[0], tr64.asm.words)

        # 66 83 d0 ff (adc ax, -1): must load 0xFFFF via MOVZ X1, #0xFFFF (0xD29FFFE1), NOT 0xFF!
        tr16 = self.translate(bytes.fromhex("6683d0ffc3"))
        a_ref16 = xr.Asm()
        a_ref16.movz(xr.T1, 0xFFFF)
        self.assertIn(a_ref16.words[0], tr16.asm.words, "adc ax, -1 must sign-extend imm8 to 16-bit 0xFFFF")
        self.assertNotIn(a_bad.words[0], tr16.asm.words)

    def test_adc_sbb_stash_af_operands(self):
        a_ref = xr.Asm()
        a_ref.stur(xr.T0, xr.CTX_REG, xr.CTX_A, size=8)
        a_ref.stur(xr.T1, xr.CTX_REG, xr.CTX_B, size=8)
        stur_a, stur_b = a_ref.words

        for hx in ("11d89cc3", "19d89cc3", "10d89cc3", "6619d89cc3"):
            with self.subTest(hx=hx):
                tr = self.translate(bytes.fromhex(hx))
                self.assertIn(stur_a, tr.asm.words)
                self.assertIn(stur_b, tr.asm.words)


@unittest.skipIf(RUNNER is None, "needs Darwin arm64 Mach-O runner or qemu-system-aarch64")
class AdcSbbExecTests(unittest.TestCase):
    """End-to-end arm64 execution tests for 8/16/32/64-bit adc and sbb."""

    def run_code(self, code):
        res, tr = RUNNER(code)
        self.assertEqual(tr.stats["unhandled"], [], f"unhandled: {tr.stats['unhandled']}")
        return res

    def _run_8bit_case(self, m, dst8, src8, cf_in):
        upper_a = 0xDEAD_BEEF_CAFE_BA00
        upper_b = 0x1122_3344_5566_7700
        op_hex = "10d8" if m == "adc" else "18d8"  # adc/sbb al, bl
        cf_hex = "f9" if cf_in else "f8"           # stc / clc
        # Expected values & all 6 x86 flags (CF=0x1, PF=0x4, AF=0x10, ZF=0x40, SF=0x80, OF=0x800)
        s8 = lambda x: x - 256 if x >= 128 else x
        if m == "adc":
            full = dst8 + src8 + cf_in
            res8 = full & 0xFF
            cf_out = int(full >= 256)
            s_full = s8(dst8) + s8(src8) + cf_in
        else:
            full = dst8 - src8 - cf_in
            res8 = full & 0xFF
            cf_out = int(full < 0)
            s_full = s8(dst8) - s8(src8) - cf_in
        of_out = int(s_full < -128 or s_full > 127)
        zf_out = int(res8 == 0)
        sf_out = int((res8 & 0x80) != 0)
        pf_out = int((bin(res8).count("1") % 2) == 0)
        af_out = int(((dst8 ^ src8 ^ res8) & 0x10) != 0)
        want_ef = cf_out | (pf_out << 2) | (af_out << 4) | (zf_out << 6) | (sf_out << 7) | (of_out << 11)

        code = (
            _imm64(0, upper_a | dst8)
            + _imm64(3, upper_b | src8)
            + bytes.fromhex(cf_hex + op_hex)
            + bytes.fromhex("eb00" "9c" "5f" "81e7d5080000")  # jmp +0; pushf; pop rdi; and edi, 0x8D5
            + _imm64(2, upper_a | res8)
            + bytes.fromhex("4839d0" "0f94c0" "0fb6c0" "c1e704" "09c7" "c3")
        )
        self.assertEqual(self.run_code(code), (want_ef << 4) | 1, f"{m}({dst8:#x}, {src8:#x}, cf={cf_in})")

    def test_8bit_adc_and_sbb_all_flag_corner_cases(self):
        corners = [
            (0x00, 0x00, 0), (0x00, 0x00, 1),
            (0xFF, 0x00, 1), (0x00, 0xFF, 1), (0xFF, 0xFF, 0), (0xFF, 0xFF, 1),
            (0x7F, 0x00, 1), (0x7F, 0x01, 0), (0x7F, 0x7F, 1), (0x7F, 0x80, 1), (0x7F, 0xFF, 1),
            (0x80, 0x00, 1), (0x80, 0x7F, 1), (0x80, 0x80, 0), (0x80, 0x80, 1), (0x80, 0xFF, 1),
            (0x0F, 0x00, 1), (0x10, 0x00, 1), (0x0F, 0x01, 0), (0x10, 0x01, 0),
        ]
        for m in ("adc", "sbb"):
            for dst8, src8, cf in corners:
                with self.subTest(m=m, dst=hex(dst8), src=hex(src8), cf=cf):
                    self._run_8bit_case(m, dst8, src8, cf)

        # hi8 & memory 8-bit: adc ah, bh (10 fc) and sbb byte [rsp], bl (18 1c 24)
        code_hi8 = (
            _imm64(0, 0x1122_3344_5566_7F88)
            + _imm64(3, 0x9999_8888_7777_0000)
            + bytes.fromhex("f9" "10fc")                      # stc; adc ah, bh (0x7F + 0 + 1 = 0x80)
            + _imm64(2, 0x1122_3344_5566_8088)
            + bytes.fromhex("4839d0" "0f94c0" "0fb6f8" "c3")
        )
        self.assertEqual(self.run_code(code_hi8), 1)

        code_mem8 = (
            bytes.fromhex("4883ec10")
            + _imm64(6, 0xAABB_CCDD_EEFF_1100)
            + bytes.fromhex("48893424")
            + _imm64(3, 0x00)
            + bytes.fromhex("f9" "181c24")                    # stc; sbb byte [rsp], bl (0 - 0 - 1 = 0xFF)
            + _imm64(2, 0xAABB_CCDD_EEFF_11FF)
            + bytes.fromhex("483b1424" "0f94c0" "0fb6f8" "4883c410" "c3")
        )
        self.assertEqual(self.run_code(code_mem8), 1)

    def test_16bit_adc_sbb_and_multiword_chain(self):
        upper = 0x1234_5678_9ABC_0000
        # 16-bit adc: 0xFFFF + 0x0000 + CF(1) = 0x0000 -> CF|PF|AF|ZF = 0x55, upper 48 preserved
        code_adc16 = (
            _imm64(0, upper | 0xFFFF) + _imm64(3, 0x0000)
            + bytes.fromhex("f9" "6611d8" "eb00" "9c" "5f" "81e7d5080000")
            + _imm64(2, upper | 0x0000)
            + bytes.fromhex("4839d0" "0f94c0" "0fb6c0" "c1e704" "09c7" "c3")
        )
        self.assertEqual(self.run_code(code_adc16), (0x55 << 4) | 1)

        # 16-bit sbb: 0x8000 - 0x0000 - CF(1) = 0x7FFF -> OF|AF|PF = 0x814, upper 48 preserved
        code_sbb16 = (
            _imm64(0, upper | 0x8000) + _imm64(3, 0x0000)
            + bytes.fromhex("f9" "6619d8" "eb00" "9c" "5f" "81e7d5080000")
            + _imm64(2, upper | 0x7FFF)
            + bytes.fromhex("4839d0" "0f94c0" "0fb6c0" "c1e704" "09c7" "c3")
        )
        self.assertEqual(self.run_code(code_sbb16), (0x814 << 4) | 1)

        # 32-bit addition via chained 16-bit ops: (0x0001_FFFF + 0x0002_0001) = 0x0004_0000
        # ax=0xFFFF, bx=0x0001, cx=0x0001, dx=0x0002 -> add ax, cx; adc bx, dx -> bx=4, ax=0
        code_add32 = (
            _imm64(0, 0xFFFF) + _imm64(3, 0x0001) + _imm64(1, 0x0001) + _imm64(2, 0x0002)
            + bytes.fromhex("6601c8" "6611d3" "c1e310" "0fb7c0" "09c3" "89df" "c3")
        )
        self.assertEqual(self.run_code(code_add32), 0x00040000)

        # 32-bit subtraction via chained 16-bit ops: (0x0005_0000 - 0x0001_0001) = 0x0003_FFFF
        code_sub32 = (
            _imm64(0, 0x0000) + _imm64(3, 0x0005) + _imm64(1, 0x0001) + _imm64(2, 0x0001)
            + bytes.fromhex("6629c8" "6619d3" "c1e310" "0fb7c0" "09c3" "89df" "c3")
        )
        self.assertEqual(self.run_code(code_sub32), 0x0003FFFF)

    def test_adc_sbb_sign_extended_negative_imm_and_32_64_af(self):
        # 1. adc ax, -1 (66 83 d0 ff) with CF=1: 0x1234 + 0xFFFF + 1 = 0x1234 (CF=1, upper 48 preserved)
        upper = 0xAAAA_BBBB_CCCC_0000
        code_adc16_neg = (
            _imm64(0, upper | 0x1234)
            + bytes.fromhex("f9" "6683d0ff" "0f92c1" "0fb6f9")  # stc; adc ax, -1; edi = CF
            + _imm64(2, upper | 0x1234)
            + bytes.fromhex("4839d0" "0f94c0" "0fb6c0" "c1e704" "09c7" "c3")
        )
        self.assertEqual(self.run_code(code_adc16_neg), (1 << 4) | 1)

        # 2. adc rax, -1 (48 83 d0 ff) with CF=0: 0x0000_0001_0000_0000 + (-1) = 0x0000_0000_FFFF_FFFF (CF=1)
        code_adc64_neg = (
            _imm64(0, 0x0000_0001_0000_0000)
            + bytes.fromhex("f8" "4883d0ff" "0f92c1" "0fb6f9")
            + _imm64(2, 0x0000_0000_FFFF_FFFF)
            + bytes.fromhex("4839d0" "0f94c0" "0fb6c0" "c1e704" "09c7" "c3")
        )
        self.assertEqual(self.run_code(code_adc64_neg), (1 << 4) | 1)

        # 3. sbb eax, -1 (83 d8 ff) with CF=0: 10 - (-1) = 11, CF=1
        code_sbb32_neg = (
            _imm64(0, 10)
            + bytes.fromhex("f8" "83d8ff" "0f92c1" "0fb6f9")
            + bytes.fromhex("83f80b" "0f94c0" "0fb6c0" "c1e704" "09c7" "c3")
        )
        self.assertEqual(self.run_code(code_sbb32_neg), (1 << 4) | 1)

        # 4. 32-bit adc & 64-bit sbb AF flag check via pushf:
        # adc eax, 0 with eax=0x0F, CF=1 -> 0x10 -> AF=1 (0x10)
        code_af32 = (
            _imm64(0, 0x0F)
            + bytes.fromhex("f9" "83d000" "eb00" "9c" "5f" "81e7d5080000" "c3")
        )
        self.assertEqual(self.run_code(code_af32), 0x10)

        # sbb rax, 0 with rax=0x10, CF=1 -> 0x0F -> AF=1, PF=1 (0x14)
        code_af64 = (
            _imm64(0, 0x10)
            + bytes.fromhex("f9" "4883d800" "eb00" "9c" "5f" "81e7d5080000" "c3")
        )
        self.assertEqual(self.run_code(code_af64), 0x14)


if __name__ == "__main__":
    unittest.main()
