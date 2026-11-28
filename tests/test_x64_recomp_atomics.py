"""Tests for xchg (atomic memory), cmpxchg, and xadd emitters in tools/x64_recomp.py.

Covers:
- ARMv8.0 LL/SC instruction encodings (ldaxrb/ldaxrh/ldaxr, stlxrb/stlxrh/stlxr, clrex)
- _classify / FLAG_WRITERS / SUB_KIND metadata for cmpxchg and xadd
- Memory xchg implicit LOCK atomicity (aligned LL/SC + unaligned fallback) with NZCV preservation
- 8/16/32/64-bit cmpxchg and lock cmpxchg (reg, hi8, aligned/unaligned mem, upper-bit rules, all 6 flags)
- 8/16/32/64-bit xadd and lock xadd (reg, hi8, dst==src, aligned/unaligned mem, all 6 flags)

Static encoding/decode/translation checks run everywhere; execution tests run natively
on Apple Silicon when CAN_RUN_MACHO is available, or under QEMU.

    python -I tests/test_x64_recomp_atomics.py
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
    """Set edi = CF | (ZF<<1) | (SF<<2) | (OF<<3) | (PF<<4) without clobbering rax..rdx/rsi/rbp."""
    # setc r8b; setz r9b; sets r10b; seto r11b; setp r12b
    return bytes.fromhex(
        "410f92c0" "410f94c1" "410f98c2" "410f90c3" "410f9ac4"
        "410fb6f8"                                      # movzx edi, r8b
        "450fb6c9" "41d1e1" "4409cf"                    # r9d << 1 -> edi
        "450fb6d2" "41c1e202" "4409d7"                  # r10d << 2 -> edi
        "450fb6db" "41c1e303" "4409df"                  # r11d << 3 -> edi
        "450fb6e4" "41c1e404" "4409e7"                  # r12d << 4 -> edi
    )


class Arm64AtomicEncodingTests(unittest.TestCase):
    """Bit-exact checks for ARMv8.0 LL/SC encodings in Asm."""

    def test_ldaxr_stlxr_clrex_encodings(self):
        a = xr.Asm()
        a.ldaxr(0, 16, size=1)
        a.ldaxr(0, 16, size=2)
        a.ldaxr(0, 16, size=4)
        a.ldaxr(0, 16, size=8)
        a.stlxr(3, 2, 16, size=1)
        a.stlxr(3, 2, 16, size=2)
        a.stlxr(3, 2, 16, size=4)
        a.stlxr(3, 2, 16, size=8)
        a.clrex()
        self.assertEqual(a.words, [
            0x085FFE00,  # ldaxrb w0, [x16]
            0x485FFE00,  # ldaxrh w0, [x16]
            0x885FFE00,  # ldaxr  w0, [x16]
            0xC85FFE00,  # ldaxr  x0, [x16]
            0x0803FE02,  # stlxrb w3, w2, [x16]
            0x4803FE02,  # stlxrh w3, w2, [x16]
            0x8803FE02,  # stlxr  w3, w2, [x16]
            0xC803FE02,  # stlxr  w3, x2, [x16]
            0xD5033F5F,  # clrex
        ])

    def test_stlxr_rejects_status_aliasing_rt_or_rn(self):
        a = xr.Asm()
        with self.assertRaises(AssertionError):
            a.stlxr(2, 2, 16, size=4)
        with self.assertRaises(AssertionError):
            a.stlxr(16, 2, 16, size=4)


class AtomicDecodeAndTranslateTests(unittest.TestCase):
    """Decode, flag classification, and static ARM64 emission checks."""

    def translate(self, code):
        insns = list(xr.disasm(code, 0x1000))
        self.assertFalse(any(i.unknown or i.error for i in insns), insns)
        tr = xr.FunctionTranslator(insns)
        tr.translate()
        self.assertEqual(tr.stats["unhandled"], [], f"unhandled: {tr.stats['unhandled']}")
        return tr, insns

    def test_classify_flags_and_producer_tables(self):
        self.assertEqual(xr.FLAG_WRITERS["cmpxchg"], "NZCVPA")
        self.assertEqual(xr.FLAG_WRITERS["xadd"], "NZCVPA")
        self.assertIn("cmpxchg", xr.SUB_KIND)
        self.assertNotIn("xadd", xr.SUB_KIND)

        # Neither cmpxchg nor xadd reads flags; both write NZCVPA.
        insns = list(xr.disasm(bytes.fromhex("0fb1cb" "0fc1cb" "f00fb10c24" "f00fc10c24"), 0x1000))
        for ins in insns:
            self.assertEqual(ins.flags_w, "NZCVPA", ins)
            self.assertEqual(ins.flags_r, "", f"{ins} must not claim to read flags")

    def test_all_atomic_forms_translate_cleanly(self):
        forms = [
            # xchg reg/mem across 8/16/32/64
            "86c4", "861c24", "66871c24", "871c24", "48871c24",
            # cmpxchg reg/mem unlocked & locked across 8/16/32/64
            "0fb0cb", "660fb1cb", "0fb1cb", "480fb1cb",
            "0fb00c24", "660fb10c24", "0fb10c24", "480fb10c24",
            "f00fb00c24", "f0660fb10c24", "f00fb10c24", "f0480fb10c24",
            # xadd reg/mem unlocked & locked across 8/16/32/64
            "0fc0cb", "660fc1cb", "0fc1cb", "480fc1cb",
            "0fc00c24", "660fc10c24", "0fc10c24", "480fc10c24",
            "f00fc00c24", "f0660fc10c24", "f00fc10c24", "f0480fc10c24",
        ]
        code = bytes.fromhex("".join(forms) + "c3")
        self.translate(code)

    def test_memory_xchg_uses_llsc_and_preserves_nzcv(self):
        # xchg [rsp], ebx (87 1c 24) has implicit LOCK and must emit ldaxr/stlxr with NO flag-setting ops.
        tr, _ = self.translate(bytes.fromhex("871c24c3"))
        ldaxr = [w for w in tr.asm.words if (w & 0x3FFFFC00) == 0x085FFC00]
        stlxr = [w for w in tr.asm.words if (w & 0x3FE0FC00) == 0x0800FC00]
        self.assertEqual(len(ldaxr), 1, "memory xchg must emit ldaxr")
        self.assertEqual(len(stlxr), 1, "memory xchg must emit stlxr")
        # Must NOT emit any flag-setting instructions (ADDS/SUBS/ANDS/TST/CMP: bit29=1 in DP-imm/reg)
        for w in tr.asm.words:
            # ANDS/TST imm (0x72000000 mask 0x7F800000), ANDS/TST reg (0x6A000000 mask 0x7F200000),
            # ADDS/SUBS imm (0x31000000/0x71000000 mask 0x3F800000), ADDS/SUBS reg (0x2B000000/0x6B000000 mask 0x3F200000)
            self.assertNotEqual(w & 0x7F800000, 0x72000000, f"unexpected ANDS/TST imm {w:#x}")
            self.assertNotEqual(w & 0x7F200000, 0x6A000000, f"unexpected ANDS/TST reg {w:#x}")
            self.assertNotEqual(w & 0x3F800000, 0x31000000, f"unexpected ADDS/SUBS imm {w:#x}")
            self.assertNotEqual(w & 0x3F200000, 0x2B000000, f"unexpected ADDS/SUBS reg {w:#x}")

    def test_lock_vs_unlocked_cmpxchg_and_xadd_llsc(self):
        # Locked cmpxchg emits ldaxr, stlxr, clrex; unlocked does not.
        tr_lock_c, _ = self.translate(bytes.fromhex("f00fb10c24c3"))
        self.assertTrue(any((w & 0x3FFFFC00) == 0x085FFC00 for w in tr_lock_c.asm.words))
        self.assertTrue(any((w & 0x3FE0FC00) == 0x0800FC00 for w in tr_lock_c.asm.words))
        self.assertIn(0xD5033F5F, tr_lock_c.asm.words)

        tr_plain_c, _ = self.translate(bytes.fromhex("0fb10c24c3"))
        self.assertFalse(any((w & 0x3FFFFC00) == 0x085FFC00 for w in tr_plain_c.asm.words))
        self.assertNotIn(0xD5033F5F, tr_plain_c.asm.words)

        # Locked xadd emits ldaxr, stlxr; unlocked does not.
        tr_lock_x, _ = self.translate(bytes.fromhex("f00fc10c24c3"))
        self.assertTrue(any((w & 0x3FFFFC00) == 0x085FFC00 for w in tr_lock_x.asm.words))
        self.assertTrue(any((w & 0x3FE0FC00) == 0x0800FC00 for w in tr_lock_x.asm.words))

        tr_plain_x, _ = self.translate(bytes.fromhex("0fc10c24c3"))
        self.assertFalse(any((w & 0x3FFFFC00) == 0x085FFC00 for w in tr_plain_x.asm.words))


@unittest.skipIf(RUNNER is None, "needs Darwin arm64 Mach-O runner or qemu-system-aarch64")
class AtomicExecTests(unittest.TestCase):
    """End-to-end arm64 execution tests for xchg, cmpxchg, and xadd."""

    def run_code(self, code):
        res, tr = RUNNER(code)
        self.assertEqual(tr.stats["unhandled"], [], f"unhandled: {tr.stats['unhandled']}")
        return res

    def test_xchg_reg_and_memory_preserves_flags(self):
        # 1. xchg al, ah (86 c4): swaps low/high byte, preserves upper 48 bits.
        code = (
            _imm64(0, 0x1122_3344_5566_AB12)
            + bytes.fromhex("86c4")
            + _imm64(2, 0x1122_3344_5566_12AB)
            + bytes.fromhex("4839d0" "0f94c0" "0fb6f8" "c3")
        )
        self.assertEqual(self.run_code(code), 1)

        # 2. xchg eax, eax (87 c0): 32-bit self-xchg zero-extends upper 32 bits of rax.
        code = (
            _imm64(0, 0xDEAD_BEEF_1234_5678)
            + bytes.fromhex("87c0")
            + _imm64(2, 0x0000_0000_1234_5678)
            + bytes.fromhex("4839d0" "0f94c0" "0fb6f8" "c3")
        )
        self.assertEqual(self.run_code(code), 1)

        # 3. xchg ax, bx (66 87 d8): preserves upper 48 bits of both rax and rbx.
        code = (
            _imm64(0, 0xAAAA_BBBB_CCCC_1111)
            + _imm64(3, 0x1111_2222_3333_2222)
            + bytes.fromhex("6687d8")
            + _imm64(2, 0xAAAA_BBBB_CCCC_2222)
            + _imm64(1, 0x1111_2222_3333_1111)
            + bytes.fromhex("4839d0" "0f94c2" "4839cb" "0f94c1" "20ca" "0fb6fa" "c3")
        )
        self.assertEqual(self.run_code(code), 1)

        # 4. Memory xchg (8/16/32/64-bit, aligned + unaligned offset +1) AND flag preservation!
        # Setup flags via `sub eax, ebx` (0 - 1 -> CF=1, SF=1, ZF=0, OF=0, PF=1) then run xchg [rsp+off], reg
        # and verify flags survive intact AND swapped values match.
        cases = [
            # (label, size, off, init_mem64, init_reg, op_hex, want_reg, want_mem64)
            ("8-bit aligned", 1, 0, 0xAABBCCDDEEFF0011, 0x9988776655443322, "861424",
             0x9988776655443311, 0xAABBCCDDEEFF0022),
            ("16-bit aligned", 2, 0, 0xAABBCCDDEEFF1234, 0x9988776655445678, "66871424",
             0x9988776655441234, 0xAABBCCDDEEFF5678),
            ("32-bit aligned", 4, 0, 0xAABBCCDDEEFF1234, 0x9988776655445678, "871424",
             0x00000000EEFF1234, 0xAABBCCDD55445678),
            ("64-bit aligned", 8, 0, 0x0123456789ABCDEF, 0xFEDCBA9876543210, "48871424",
             0x0123456789ABCDEF, 0xFEDCBA9876543210),
            ("32-bit unaligned +1", 4, 1, 0x1122334455667788, 0xDEAD_BEEF_CAFE_BABE, "87542401",
             0x0000000044556677, 0x112233CAFEBABE88),
        ]
        for label, _sz, _off, init_mem, init_rdx, op_hex, want_rdx, want_mem in cases:
            with self.subTest(label=label):
                code = (
                    bytes.fromhex("4883ec10")               # sub rsp, 16
                    + _imm64(6, init_mem)                   # mov rsi, init_mem
                    + bytes.fromhex("48893424")             # mov [rsp], rsi
                    + _imm64(2, init_rdx)                   # mov rdx, init_rdx
                    + bytes.fromhex("31c0" "bb01000000" "29d8")  # xor eax,eax; mov ebx,1; sub eax,ebx (CF=1,SF=1,PF=1)
                    + bytes.fromhex(op_hex)                 # xchg [rsp+off], rdx/edx/dx/dl
                    + _pack_cf_zf_sf_of_pf_to_edi()         # edi = flags (want 0b10101 = 21)
                    + _imm64(0, want_rdx)
                    + bytes.fromhex("4839c2" "0f94c0" "0fb6c0")  # eax = (rdx == want_rdx)
                    + _imm64(1, want_mem)
                    + bytes.fromhex("483b0c24" "0f94c1" "0fb6c9" "21c8")  # eax &= ([rsp] == want_mem)
                    + bytes.fromhex("c1e704" "09c7" "4883c410" "c3")      # edi = (flags<<4)|ok
                )
                self.assertEqual(self.run_code(code), (0b10101 << 4) | 1)

        # 5. xchg [rax], rax (48 87 00): base register aliases the swapped register!
        code = (
            bytes.fromhex("4883ec10")                       # sub rsp, 16
            + _imm64(6, 0x0102030405060708)
            + bytes.fromhex("48893424" "4889e0" "488700")   # mov [rsp], rsi; mov rax, rsp; xchg [rax], rax
            # Now rax == 0x0102030405060708 and [rsp] == rsp!
            + _imm64(2, 0x0102030405060708)
            + bytes.fromhex("4839d0" "0f94c0" "0fb6c0" "48392424" "0f94c1" "0fb6c9" "21c8" "89c7" "4883c410" "c3")
        )
        self.assertEqual(self.run_code(code), 1)

    def test_cmpxchg_all_widths_reg_and_hi8(self):
        # (label, init_rax, init_rbx, init_rcx, op_hex, want_rax, want_rbx, want_flags_cf_zf_sf_of_pf)
        cases = [
            # 8-bit equal: al==bl -> bl:=cl, upper bits of rax & rbx preserved; ZF=1, PF=1
            ("8b eq", 0xAAAA_BBBB_CCCC_DD42, 0x1111_2222_3333_4442, 0x99, "0fb0cb",
             0xAAAA_BBBB_CCCC_DD42, 0x1111_2222_3333_4499, 0b10010),
            # 8-bit unequal: al(0x80) - bl(0x01) = 0x7F -> OF=1, CF=0, SF=0, ZF=0, PF=0 (7 bits = odd); al:=0x01
            ("8b ne ovf", 0xAAAA_BBBB_CCCC_DD80, 0x1111_2222_3333_4401, 0x99, "0fb0cb",
             0xAAAA_BBBB_CCCC_DD01, 0x1111_2222_3333_4401, 0b01000),
            # 8-bit hi8 equal: cmpxchg bh, cl (0f b0 cf): al==bh -> bh:=cl
            ("8b hi8 eq", 0x5555_0000_0000_0077, 0x1122_3344_5566_7788, 0xAA, "0fb0cf",
             0x5555_0000_0000_0077, 0x1122_3344_5566_AA88, 0b10010),
            # 8-bit hi8 unequal: cmpxchg bh, cl: al(0x10) - bh(0x20) = 0xF0 -> CF=1, SF=1, PF=1; al:=0x20
            ("8b hi8 ne", 0x5555_0000_0000_0010, 0x1122_3344_5566_2088, 0xAA, "0fb0cf",
             0x5555_0000_0000_0020, 0x1122_3344_5566_2088, 0b10101),
            # 16-bit equal: ax==bx -> bx:=cx, upper 48 bits preserved
            ("16b eq", 0xDEAD_BEEF_CAFE_1234, 0x1111_2222_3333_1234, 0x5678, "660fb1cb",
             0xDEAD_BEEF_CAFE_1234, 0x1111_2222_3333_5678, 0b10010),
            # 16-bit unequal: ax(0x0001) - bx(0x0002) = 0xFFFF -> CF=1, SF=1, PF=1
            ("16b ne", 0xDEAD_BEEF_CAFE_0001, 0x1111_2222_3333_0002, 0x5678, "660fb1cb",
             0xDEAD_BEEF_CAFE_0002, 0x1111_2222_3333_0002, 0b10101),
            # 32-bit equal: eax==ebx -> ebx:=ecx (zero-extends rbx!), upper 32 of rax PRESERVED!
            ("32b eq", 0xDEAD_BEEF_1234_5678, 0x9999_8888_1234_5678, 0xCAFE_BABE, "0fb1cb",
             0xDEAD_BEEF_1234_5678, 0x0000_0000_CAFE_BABE, 0b10010),
            # 32-bit unequal: eax(0x80000000) - ebx(1) = 0x7FFFFFFF -> OF=1, PF=1 (0xFF has 8 ones = even);
            # both rax AND rbx zero-extended!
            ("32b ne", 0xDEAD_BEEF_8000_0000, 0x9999_8888_0000_0001, 0xCAFE_BABE, "0fb1cb",
             0x0000_0000_0000_0001, 0x0000_0000_0000_0001, 0b11000),
            # 64-bit equal
            ("64b eq", 0x0123_4567_89AB_CDEF, 0x0123_4567_89AB_CDEF, 0xFEDC_BA98_7654_3210, "480fb1cb",
             0x0123_4567_89AB_CDEF, 0xFEDC_BA98_7654_3210, 0b10010),
            # 64-bit unequal: 0 - 1 -> CF=1, SF=1, PF=1
            ("64b ne", 0, 1, 0x42, "480fb1cb", 1, 1, 0b10101),
        ]
        for label, init_rax, init_rbx, init_rcx, op_hex, want_rax, want_rbx, want_flags in cases:
            with self.subTest(label=label):
                code = (
                    _imm64(0, init_rax)
                    + _imm64(3, init_rbx)
                    + _imm64(1, init_rcx)
                    + bytes.fromhex(op_hex)
                    + _pack_cf_zf_sf_of_pf_to_edi()
                    + _imm64(2, want_rax)
                    + _imm64(6, want_rbx)
                    + bytes.fromhex("4839d0" "0f94c0" "0fb6c0" "4839f3" "0f94c2" "0fb6d2" "21d0" "c1e704" "09c7" "c3")
                )
                self.assertEqual(self.run_code(code), (want_flags << 4) | 1)

        # Verify SUB_KIND borrow chaining: cmpxchg (CF=1) followed immediately by sbb edx, edx -> -1
        code_sbb = (
            _imm64(0, 5) + _imm64(3, 9) + _imm64(1, 0x77) + _imm64(2, 0)
            + bytes.fromhex("0fb1cb" "19d2" "89d7" "c3")  # cmpxchg ebx, ecx; sbb edx, edx; mov edi, edx
        )
        self.assertEqual(self.run_code(code_sbb), 0xFFFFFFFF)

    def test_cmpxchg_memory_unlocked_and_locked_aligned_and_unaligned(self):
        # Test 8/16/32/64-bit unlocked & LOCK memory cmpxchg (equal & unequal, aligned & unaligned +1)
        # plus cross-block pushf verifying all 6 flags (CF=0x1, PF=0x4, AF=0x10, ZF=0x40, SF=0x80, OF=0x800).
        cases = [
            # (label, init_mem64, init_rax, init_rcx, op_hex, want_mem64, want_rax, want_eflags_mask)
            ("8b lock eq", 0x1122334455667755, 0xAA00000000000055, 0x99, "f00fb00c24",
             0x1122334455667799, 0xAA00000000000055, 0x44),  # ZF=1, PF=1
            ("8b lock ne AF", 0x1122334455667701, 0xAA00000000000010, 0x99, "f00fb00c24",
             0x1122334455667701, 0xAA00000000000001, 0x14),  # 0x10 - 0x01 = 0x0F -> AF=1, PF=1 (4 ones)
            ("16b lock eq", 0xAABBCCDDEEFF1234, 0x5555666677771234, 0xBEEF, "f0660fb10c24",
             0xAABBCCDDEEFFBEEF, 0x5555666677771234, 0x44),
            ("16b lock ne", 0xAABBCCDDEEFF0002, 0x5555666677770001, 0xBEEF, "f0660fb10c24",
             0xAABBCCDDEEFF0002, 0x5555666677770002, 0x95),  # 1-2 = 0xFFFF -> CF|PF|AF|SF = 0x95
            ("32b lock eq", 0xAABBCCDDEEFF1234, 0x99998888EEFF1234, 0xCAFEBABE, "f00fb10c24",
             0xAABBCCDDCAFEBABE, 0x99998888EEFF1234, 0x44),
            ("32b lock ne", 0xAABBCCDD00000001, 0x9999888880000000, 0xCAFEBABE, "f00fb10c24",
             0xAABBCCDD00000001, 0x0000000000000001, 0x814),  # 0x80000000-1 = 0x7FFFFFFF -> OF|AF|PF = 0x814
            ("64b lock eq", 0x0123456789ABCDEF, 0x0123456789ABCDEF, 0xFEDCBA9876543210, "f0480fb10c24",
             0xFEDCBA9876543210, 0x0123456789ABCDEF, 0x44),
            ("64b unlocked ne", 0x0000000000000005, 0x0000000000000003, 0x99, "480fb10c24",
             0x0000000000000005, 0x0000000000000005, 0x91),  # 3-5 = -2 (0xFE: 7 ones -> PF=0) -> CF|AF|SF = 0x91
            ("32b lock unaligned +1 eq", 0x1122334455667788, 0xDEAD_BEEF_4455_6677, 0xAABBCCDD, "f00fb14c2401",
             0x112233AABBCCDD88, 0xDEAD_BEEF_4455_6677, 0x44),
            ("32b lock unaligned +1 ne", 0x1122334455667788, 0xDEAD_BEEF_0000_0000, 0xAABBCCDD, "f00fb14c2401",
             0x1122334455667788, 0x0000_0000_4455_6677, 0x91),  # 0 - 0x44556677 = 0xBBAA9989 -> CF|AF|SF = 0x91
        ]

        for label, init_mem, init_rax, init_rcx, op_hex, want_mem, want_rax, want_ef in cases:
            with self.subTest(label=label):
                code = (
                    bytes.fromhex("4883ec10")                       # sub rsp, 16
                    + _imm64(6, init_mem)
                    + bytes.fromhex("48893424")                     # mov [rsp], rsi
                    + _imm64(0, init_rax)
                    + _imm64(1, init_rcx)
                    + bytes.fromhex(op_hex)                         # [lock] cmpxchg [rsp+off], rcx/ecx/cx/cl
                    + bytes.fromhex("eb00" "9c" "5f" "81e7d5080000")  # jmp +0; pushf; pop rdi; and edi, 0x8D5
                    + _imm64(2, want_rax)
                    + bytes.fromhex("4839d0" "0f94c0" "0fb6c0")     # eax = (rax == want_rax)
                    + _imm64(3, want_mem)
                    + bytes.fromhex("483b1c24" "0f94c2" "0fb6d2" "21d0")  # eax &= ([rsp] == want_mem)
                    + bytes.fromhex("c1e704" "09c7" "4883c410" "c3")      # edi = (eflags<<4)|ok
                )
                self.assertEqual(self.run_code(code), (want_ef << 4) | 1)

        # Verify lock cmpxchg [rax], rcx (f0 48 0f b1 08) when rax is BOTH comparator and base pointer (mismatch case!)
        code_alias = (
            bytes.fromhex("4883ec10")                               # sub rsp, 16
            + _imm64(6, 0x123456789ABCDEF0)
            + bytes.fromhex("48893424" "4889e0")                    # mov [rsp], rsi; mov rax, rsp
            + _imm64(1, 0x55)
            + bytes.fromhex("f0480fb108")                           # lock cmpxchg [rax], rcx (rsp != 0x1234... -> rax := 0x1234...)
            + _imm64(2, 0x123456789ABCDEF0)
            + bytes.fromhex("4839d0" "0f94c0" "0fb6f8" "4883c410" "c3")
        )
        self.assertEqual(self.run_code(code_alias), 1)

    def test_xadd_all_widths_reg_hi8_same_reg_and_memory(self):
        # 1. Register xadd across 8/16/32/64-bit, hi8, and dst==src (xadd eax, eax -> 2*eax!)
        reg_cases = [
            # (label, init_rbx, init_rcx, op_hex, want_rbx, want_rcx, want_ef_0x8D5)
            # 8-bit: 0x7F + 0x01 = 0x80 -> SF=1, OF=1, AF=1, PF=0, ZF=0, CF=0 -> 0x890
            ("8b ovf", 0x112233445566777F, 0xAABBCCDDEEFF0001, "0fc0cb",
             0x1122334455667780, 0xAABBCCDDEEFF007F, 0x890),
            # 8-bit carry+zero: 0xFF + 0x01 = 0x00 -> CF=1, PF=1, AF=1, ZF=1 -> 0x55
            ("8b cz", 0x11223344556677FF, 0xAABBCCDDEEFF0001, "0fc0cb",
             0x1122334455667700, 0xAABBCCDDEEFF00FF, 0x55),
            # 16-bit: 0xFFFE + 0x0002 = 0x0000 -> CF|PF|AF|ZF = 0x55; upper 48 preserved
            ("16b cz", 0x111122223333FFFE, 0x4444555566660002, "660fc1cb",
             0x1111222233330000, 0x444455556666FFFE, 0x55),
            # 32-bit: zero-extends BOTH rbx and rcx
            ("32b", 0xDEAD_BEEF_7FFF_FFFF, 0xCAFE_BABE_0000_0001, "0fc1cb",
             0x0000_0000_8000_0000, 0x0000_0000_7FFF_FFFF, 0x894),  # 0x80000000: SF|OF|AF|PF(0x00=even) = 0x894
            # 64-bit
            ("64b", 0xFFFFFFFFFFFFFFFF, 0x0000000000000002, "480fc1cb",
             0x0000000000000001, 0xFFFFFFFFFFFFFFFF, 0x11),  # CF|AF = 0x11 (0x01 odd parity)
        ]
        for label, init_rbx, init_rcx, op_hex, want_rbx, want_rcx, want_ef in reg_cases:
            with self.subTest(label=label):
                code = (
                    _imm64(3, init_rbx)
                    + _imm64(1, init_rcx)
                    + bytes.fromhex(op_hex)
                    + bytes.fromhex("9c" "5f" "81e7d5080000")       # pushf; pop rdi; and edi, 0x8D5
                    + _imm64(0, want_rbx)
                    + _imm64(2, want_rcx)
                    + bytes.fromhex("4839c3" "0f94c0" "0fb6c0" "4839d1" "0f94c2" "0fb6d2" "21d0" "c1e704" "09c7" "c3")
                )
                self.assertEqual(self.run_code(code), (want_ef << 4) | 1)

        # 2. xadd eax, eax (0f c1 c0): dst == src MUST yield 2*eax (not old eax)!
        code_self = (
            _imm64(0, 0xDEAD_BEEF_0000_0015)
            + bytes.fromhex("0fc1c0")
            + _imm64(2, 0x0000_0000_0000_002A)
            + bytes.fromhex("4839d0" "0f94c0" "0fb6f8" "c3")
        )
        self.assertEqual(self.run_code(code_self), 1)

        # 3. xadd al, ah (0f c0 e0): al=10, ah=3 -> al=13, ah=10, upper 48 bits preserved!
        code_hi8 = (
            _imm64(0, 0x1122_3344_5566_030A)
            + bytes.fromhex("0fc0e0")
            + _imm64(2, 0x1122_3344_5566_0A0D)
            + bytes.fromhex("4839d0" "0f94c0" "0fb6f8" "c3")
        )
        self.assertEqual(self.run_code(code_hi8), 1)

        # 4. Memory xadd & lock xadd (8/16/32/64-bit, aligned & unaligned +1, plus adc chaining)
        mem_cases = [
            ("8b lock", 0x11223344556677F0, 0xAA00000000000020, "f00fc00424",
             0x1122334455667710, 0xAA000000000000F0, 0x01),  # 0xF0+0x20=0x110 -> CF=1, PF=0(0x10 odd), AF=0
            ("16b lock", 0xAABBCCDDEEFF7FFF, 0x1122334455660001, "f0660fc10424",
             0xAABBCCDDEEFF8000, 0x1122334455667FFF, 0x894),  # 0x8000: SF|OF|AF|PF(0x00 even) = 0x894
            ("32b lock", 0xAABBCCDDFFFFFFFE, 0x9999888800000003, "f00fc10424",
             0xAABBCCDD00000001, 0x00000000FFFFFFFE, 0x11),  # CF|AF = 0x11
            ("64b lock", 0x0000000000000010, 0x0000000000000025, "f0480fc10424",
             0x0000000000000035, 0x0000000000000010, 0x04),  # 0x35 = 00110101 (4 ones -> PF=1)
            ("32b unlocked", 0xAABBCCDD00000007, 0x1234567800000008, "0fc10424",
             0xAABBCCDD0000000F, 0x0000000000000007, 0x04),  # 0x0F (4 ones -> PF=1)
            ("32b lock unaligned +1", 0x1122330000000188, 0x0000000000000002, "f00fc1442401",
             0x1122330000000388, 0x0000000000000001, 0x04),  # 0x03 (2 ones -> PF=1)
        ]
        for label, init_mem, init_rax, op_hex, want_mem, want_rax, want_ef in mem_cases:
            with self.subTest(label=label):
                code = (
                    bytes.fromhex("4883ec10")
                    + _imm64(6, init_mem)
                    + bytes.fromhex("48893424")
                    + _imm64(0, init_rax)
                    + bytes.fromhex(op_hex)                         # [lock] xadd [rsp+off], rax/eax/ax/al
                    + bytes.fromhex("9c" "5f" "81e7d5080000")
                    + _imm64(2, want_rax)
                    + bytes.fromhex("4839d0" "0f94c0" "0fb6c0")
                    + _imm64(3, want_mem)
                    + bytes.fromhex("483b1c24" "0f94c2" "0fb6d2" "21d0")
                    + bytes.fromhex("c1e704" "09c7" "4883c410" "c3")
                )
                self.assertEqual(self.run_code(code), (want_ef << 4) | 1)

        # 5. lock xadd [rax], rax (f0 48 0f c1 00): base register aliases src!
        code_alias = (
            bytes.fromhex("4883ec10")
            + _imm64(6, 100)
            + bytes.fromhex("48893424" "4889e0" "f0480fc100")       # [rsp]=100; rax=rsp; lock xadd [rax], rax
            # Now rax == 100 and [rsp] == rsp + 100!
            + bytes.fromhex("488d542464")                           # lea rdx, [rsp + 100]
            + bytes.fromhex("48391424" "0f94c1" "0fb6c9")           # ecx = ([rsp] == rsp + 100)
            + _imm64(2, 100)
            + bytes.fromhex("4839d0" "0f94c0" "0fb6c0" "21c8" "89c7" "4883c410" "c3")
        )
        self.assertEqual(self.run_code(code_alias), 1)

        # 6. Verify ADD-kind carry chaining: xadd (CF=1) followed immediately by adc edi, 0 -> 1
        code_adc = (
            _imm64(3, 0xFFFFFFFF) + _imm64(1, 1) + _imm64(7, 0)
            + bytes.fromhex("0fc1cb" "83d700" "c3")                 # xadd ebx, ecx; adc edi, 0; ret
        )
        self.assertEqual(self.run_code(code_adc), 1)


if __name__ == "__main__":
    unittest.main()
