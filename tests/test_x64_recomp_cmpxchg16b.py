"""Tests for cmpxchg8b and cmpxchg16b (0F C7 /1) in tools/x64_recomp.py.

Covers:
- Decoding & classification of 0F C7 /1 (cmpxchg8b m64, REX.W cmpxchg16b m128,
  LOCK prefix, REX.R ignored for opcode extension, 66 prefix ignored, RIP-relative
  addressing, and rejection of mod==3 / non-/1 encodings as unknown with exact length)
- Bit-exact ARMv8.0 LDAXP / STLXP / CLREX encodings and operand constraints
- LOCK vs unlocked codegen (LL/SC pair loop + alignment guard vs direct pair load/store)
- Match & mismatch semantics for cmpxchg8b (ignoring upper 32 bits on compare,
  preserving upper 32 bits of RAX/RDX on match, zero-extending RAX/RDX on mismatch)
- Full 128-bit cmpxchg16b match, single-high-bit mismatch, unaligned fallback,
  base-register aliasing ([rax]), and Windows InterlockedCompareExchange128 retry loop
- Preservation of CF/SF/OF/PF (updating only ZF) on both ARM NZCV and materialized
  ctx.eflags paths, including dead-ZF preservation

Static checks run everywhere; execution tests run natively on Apple Silicon when
CAN_RUN_MACHO is available, or under QEMU.

    python -I tests/test_x64_recomp_cmpxchg16b.py
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


def _align_r8_stack_frame():
    """Allocate 64 bytes on rsp and point r8 at a 16-byte aligned slot inside it."""
    return bytes.fromhex(
        "4883ec40"      # sub rsp, 64
        "4989e0"        # mov r8, rsp
        "4983c00f"      # add r8, 15
        "49c1e804"      # shr r8, 4
        "49c1e004"      # shl r8, 4
    )


def _epilogue_ret():
    return bytes.fromhex(
        "4883c440"      # add rsp, 64
        "c3"            # ret
    )


def _pack_flags_to_edi():
    """Set edi = ZF | (CF<<1) | (SF<<2) | (OF<<3) | (PF<<4) using r11..r15."""
    return bytes.fromhex(
        "410f94c3"      # setz r11b
        "410f92c4"      # setc r12b
        "410f98c5"      # sets r13b
        "410f90c6"      # seto r14b
        "410f9ac7"      # setp r15b
        "410fb6fb"      # movzx edi, r11b
        "450fb6e4"      # movzx r12d, r12b
        "41d1e4"        # shl r12d, 1
        "4409e7"        # or edi, r12d
        "450fb6ed"      # movzx r13d, r13b
        "41c1e502"      # shl r13d, 2
        "4409ef"        # or edi, r13d
        "450fb6f6"      # movzx r14d, r14b
        "41c1e603"      # shl r14d, 3
        "4409f7"        # or edi, r14d
        "450fb6ff"      # movzx r15d, r15b
        "41c1e704"      # shl r15d, 4
        "4409ff"        # or edi, r15d
    )


class Cmpxchg16bStaticTests(unittest.TestCase):
    def test_decode_cmpxchg8b_and_cmpxchg16b(self):
        cases = [
            # (hex, mnem, size, mem_size, lock)
            ("0fc70f", "cmpxchg8b", 3, 8, False),               # cmpxchg8b [rdi]
            ("f00fc74c2408", "cmpxchg8b", 6, 8, True),          # lock cmpxchg8b [rsp+8]
            ("480fc70f", "cmpxchg16b", 4, 16, False),           # cmpxchg16b [rdi]
            ("f0490fc74810", "cmpxchg16b", 6, 16, True),        # lock cmpxchg16b [r8+0x10]
            ("4c0fc70f", "cmpxchg16b", 4, 16, False),           # REX.W + REX.R ignored for /1
            ("660fc70f", "cmpxchg8b", 4, 8, False),             # 66 prefix ignored on /1
            ("66480fc70f", "cmpxchg16b", 5, 16, False),         # 66 + REX.W -> cmpxchg16b
        ]
        for hx, mnem, size, mem_sz, lock in cases:
            with self.subTest(hx=hx):
                raw = bytes.fromhex(hx)
                ins = xr.decode_one(raw, 0, len(raw), GUEST_VA)
                self.assertFalse(ins.unknown)
                self.assertIsNone(ins.error)
                self.assertEqual(ins.mnem, mnem)
                self.assertEqual(ins.size, size)
                self.assertEqual(ins.kind, "cas2")
                self.assertEqual(ins.flags_w, "Z")
                self.assertEqual(ins.flags_r, "")
                self.assertEqual(ins.lock, lock)
                self.assertEqual(len(ins.ops), 1)
                self.assertEqual(ins.ops[0][0], "m")
                self.assertEqual(ins.ops[0][1]["size"], mem_sz)
                self.assertEqual(xr.FLAG_WRITERS[mnem], "Z")

        # RIP-relative addressing resolves abs = addr + size + disp
        rip_raw = bytes.fromhex("0fc70d10000000")
        ins = xr.decode_one(rip_raw, 0, len(rip_raw), GUEST_VA)
        self.assertEqual(ins.mnem, "cmpxchg8b")
        self.assertEqual(ins.size, 7)
        self.assertTrue(ins.ops[0][1]["rip"])
        self.assertEqual(ins.ops[0][1]["abs"], GUEST_VA + 7 + 0x10)

        # mod==3 or reg!=1 in group 9 remain unknown with accurate length
        for bad_hx, exp_len in (
            ("0fc7c8", 3),          # mod=3 /1 (invalid reg form)
            ("0fc7f0", 3),          # rdrand eax (mod=3 /6)
            ("0fc737", 3),          # vmptrld [rdi] (mod=0 /6)
            ("480fc7f8", 4),        # rdseed rax (mod=3 /7)
        ):
            with self.subTest(bad_hx=bad_hx):
                raw = bytes.fromhex(bad_hx)
                ins = xr.decode_one(raw, 0, len(raw), GUEST_VA)
                self.assertTrue(ins.unknown)
                self.assertEqual(ins.size, exp_len)

    def test_arm64_ldaxp_stlxp_clrex_encodings(self):
        a = xr.Asm()
        a.ldaxp(0, 1, 16, sf=0)
        a.ldaxp(0, 1, 16, sf=1)
        a.stlxp(3, 22, 20, 16, sf=0)
        a.stlxp(3, 22, 20, 16, sf=1)
        a.clrex()
        self.assertEqual(
            a.words,
            [0x887F8600, 0xC87F8600, 0x8823D216, 0xC823D216, 0xD5033F5F],
        )

        with self.assertRaises(AssertionError):
            xr.Asm().ldaxp(0, 0, 16, sf=1)
        for bad_rs in (22, 20, 16):
            with self.assertRaises(AssertionError):
                xr.Asm().stlxp(bad_rs, 22, 20, 16, sf=1)

    def test_translate_emits_llsc_only_for_lock(self):
        for hx, expect_words in (
            ("f00fc74c2408c3", {0x887F8600, 0x8823D216, 0xD5033F5F}),
            ("f0480fc74c2410c3", {0xC87F8600, 0xC823D216, 0xD5033F5F}),
        ):
            with self.subTest(lock_hx=hx):
                insns = xr.disasm(bytes.fromhex(hx), GUEST_VA)
                ft = xr.FunctionTranslator(insns)
                ft.translate()
                self.assertEqual(ft.stats["unhandled"], [])
                words = set(ft.asm.words)
                for w in expect_words:
                    self.assertIn(w, words)

        for hx in ("0fc74c2408c3", "480fc74c2410c3"):
            with self.subTest(unlocked_hx=hx):
                insns = xr.disasm(bytes.fromhex(hx), GUEST_VA)
                ft = xr.FunctionTranslator(insns)
                ft.translate()
                self.assertEqual(ft.stats["unhandled"], [])
                words = set(ft.asm.words)
                self.assertTrue(words.isdisjoint({0x887F8600, 0xC87F8600, 0x8823D216, 0xC823D216, 0xD5033F5F}))


@unittest.skipUnless(RUNNER is not None, "no arm64 runner available")
class Cmpxchg16bExecTests(unittest.TestCase):
    def _run(self, code: bytes) -> int:
        val, ft = RUNNER(code)
        self.assertEqual(ft.stats["unhandled"], [])
        return val

    def test_exec_cmpxchg8b_match_mismatch_unaligned_and_aliasing(self):
        # 1. Both unlocked and LOCK cmpxchg8b match when EDX:EAX == [r8],
        #    ignoring upper 32 bits of RAX/RDX and leaving those upper bits intact.
        for lock_pfx in ("", "f0"):
            with self.subTest(case="match", lock=bool(lock_pfx)):
                # Pack checks into a bitmask in edi:
                # bit 0: ZF == 1
                # bit 1: [r8] == 0x12345678AABBCCDD
                # bit 2: upper32(RAX) == 0xDEAD0000
                # bit 3: upper32(RDX) == 0xBEEF0000
                code = (
                    _align_r8_stack_frame()
                    + _imm64(9, 0x5566778811223344)
                    + bytes.fromhex("4d8908")
                    + _imm64(0, 0xDEAD000011223344)
                    + _imm64(2, 0xBEEF000055667788)
                    + _imm64(3, 0xAAAA0000AABBCCDD)
                    + _imm64(1, 0xBBBB000012345678)
                    + bytes.fromhex(lock_pfx + "410fc708")
                    + bytes.fromhex("410f94c3" "410fb6fb")  # setz r11b; movzx edi, r11b
                    + bytes.fromhex("4d8b08")               # mov r9, [r8]
                    + _imm64(10, 0x12345678AABBCCDD)
                    + bytes.fromhex("4d39d1" "410f94c3" "450fb6db" "41d1e3" "4409df")
                    + _imm64(10, 0xDEAD000011223344)
                    + bytes.fromhex("4c39d0" "410f94c3" "450fb6db" "41c1e302" "4409df")
                    + _imm64(10, 0xBEEF000055667788)
                    + bytes.fromhex("4c39d2" "410f94c3" "450fb6db" "41c1e303" "4409df")
                    + _epilogue_ret()
                )
                self.assertEqual(self._run(code), 0x0F)

        # 2. Mismatch on low half only and high half only:
        #    ZF=0, memory unchanged, RAX=zero_ext(lo), RDX=zero_ext(hi) (upper 32 cleared!).
        for lock_pfx in ("", "f0"):
            for exp_eax, exp_edx in (
                (0xDEAD000011223345, 0xBEEF000055667788),  # low mismatch
                (0xDEAD000011223344, 0xBEEF000055667789),  # high mismatch
            ):
                with self.subTest(case="mismatch", lock=bool(lock_pfx), exp_eax=hex(exp_eax)):
                    code = (
                        _align_r8_stack_frame()
                        + _imm64(9, 0x5566778811223344)
                        + bytes.fromhex("4d8908")
                        + _imm64(0, exp_eax)
                        + _imm64(2, exp_edx)
                        + _imm64(3, 0xAABBCCDD)
                        + _imm64(1, 0x12345678)
                        + bytes.fromhex(lock_pfx + "410fc708")
                        + bytes.fromhex("410f95c3" "410fb6fb")  # setnz r11b -> bit0 = (ZF==0)
                        + bytes.fromhex("4d8b08")               # mov r9, [r8]
                        + _imm64(10, 0x5566778811223344)
                        + bytes.fromhex("4d39d1" "410f94c3" "450fb6db" "41d1e3" "4409df")
                        + _imm64(10, 0x0000000011223344)        # upper 32 of RAX must be 0
                        + bytes.fromhex("4c39d0" "410f94c3" "450fb6db" "41c1e302" "4409df")
                        + _imm64(10, 0x0000000055667788)        # upper 32 of RDX must be 0
                        + bytes.fromhex("4c39d2" "410f94c3" "450fb6db" "41c1e303" "4409df")
                        + _epilogue_ret()
                    )
                    self.assertEqual(self._run(code), 0x0F)

        # 3. Unaligned lock cmpxchg8b at [r8+1] (exercises unaligned fallback without SIGBUS)
        for disp in (1, 4):
            with self.subTest(unaligned_disp=disp):
                code = (
                    _align_r8_stack_frame()
                    + _imm64(9, 0x8877665544332211)
                    + bytes.fromhex(f"4d8948{disp:02x}")        # mov [r8+disp], r9
                    + _imm64(0, 0x44332211)
                    + _imm64(2, 0x88776655)
                    + _imm64(3, 0xCAFEBABE)
                    + _imm64(1, 0xDEADBEEF)
                    + bytes.fromhex(f"f0410fc748{disp:02x}")    # lock cmpxchg8b [r8+disp]
                    + bytes.fromhex("410f94c3" "410fb6fb")      # bit0: ZF==1
                    + bytes.fromhex(f"4d8b48{disp:02x}")        # mov r9, [r8+disp]
                    + _imm64(10, 0xDEADBEEFCAFEBABE)
                    + bytes.fromhex("4d39d1" "410f94c3" "450fb6db" "41d1e3" "4409df")
                    + _epilogue_ret()
                )
                self.assertEqual(self._run(code), 0x03)

        # 4. Base-register aliasing: lock cmpxchg8b [rax] on mismatch reloads RAX cleanly
        code = (
            _align_r8_stack_frame()
            + _imm64(9, 0x1122334455667788)
            + bytes.fromhex("4d8908")           # mov [r8], r9
            + bytes.fromhex("4c89c0")           # mov rax, r8 (base == rax, EAX != 0x55667788)
            + _imm64(2, 0x11223344)             # edx matches hi, eax mismatches lo
            + _imm64(3, 0x10101010)
            + _imm64(1, 0x20202020)
            + bytes.fromhex("f00fc708")         # lock cmpxchg8b [rax]
            + bytes.fromhex("410f95c3" "410fb6fb")  # bit0: ZF==0
            + _imm64(10, 0x0000000055667788)
            + bytes.fromhex("4c39d0" "410f94c3" "450fb6db" "41d1e3" "4409df")
            + _imm64(10, 0x0000000011223344)
            + bytes.fromhex("4c39d2" "410f94c3" "450fb6db" "41c1e302" "4409df")
            + _epilogue_ret()
        )
        self.assertEqual(self._run(code), 0x07)

    def test_exec_cmpxchg16b_match_mismatch_unaligned_and_retry_loop(self):
        lo_init = 0x8000000011223344
        hi_init = 0x9000000055667788
        lo_new = 0xFEDCBA9876543210
        hi_new = 0x0123456789ABCDEF

        # 1. Match (unlocked and LOCK) on 16-byte aligned [r8]
        for lock_pfx in ("", "f0"):
            with self.subTest(case="match16", lock=bool(lock_pfx)):
                code = (
                    _align_r8_stack_frame()
                    + _imm64(9, lo_init)
                    + _imm64(10, hi_init)
                    + bytes.fromhex("4d8908" "4d895008")        # [r8]=lo_init, [r8+8]=hi_init
                    + _imm64(0, lo_init)
                    + _imm64(2, hi_init)
                    + _imm64(3, lo_new)
                    + _imm64(1, hi_new)
                    + bytes.fromhex(lock_pfx + "490fc708")      # [lock] cmpxchg16b [r8]
                    + bytes.fromhex("410f94c3" "410fb6fb")      # bit0: ZF==1
                    + bytes.fromhex("4d8b08")                   # mov r9, [r8]
                    + _imm64(10, lo_new)
                    + bytes.fromhex("4d39d1" "410f94c3" "450fb6db" "41d1e3" "4409df")
                    + bytes.fromhex("4d8b4808")                 # mov r9, [r8+8]
                    + _imm64(10, hi_new)
                    + bytes.fromhex("4d39d1" "410f94c3" "450fb6db" "41c1e302" "4409df")
                    + _epilogue_ret()
                )
                self.assertEqual(self._run(code), 0x07)

        # 2. Mismatch when ONLY bit 63 of RAX or bit 63 of RDX differs (proves full 128-bit compare)
        for lock_pfx in ("", "f0"):
            for exp_lo, exp_hi in (
                (lo_init ^ (1 << 63), hi_init),
                (lo_init, hi_init ^ (1 << 63)),
            ):
                with self.subTest(case="mismatch16", lock=bool(lock_pfx), exp_lo=hex(exp_lo)):
                    code = (
                        _align_r8_stack_frame()
                        + _imm64(9, lo_init)
                        + _imm64(10, hi_init)
                        + bytes.fromhex("4d8908" "4d895008")
                        + _imm64(0, exp_lo)
                        + _imm64(2, exp_hi)
                        + _imm64(3, lo_new)
                        + _imm64(1, hi_new)
                        + bytes.fromhex(lock_pfx + "490fc708")
                        + bytes.fromhex("410f95c3" "410fb6fb")  # bit0: ZF==0
                        + _imm64(10, lo_init)                   # RAX reloaded from [r8]
                        + bytes.fromhex("4c39d0" "410f94c3" "450fb6db" "41d1e3" "4409df")
                        + _imm64(10, hi_init)                   # RDX reloaded from [r8+8]
                        + bytes.fromhex("4c39d2" "410f94c3" "450fb6db" "41c1e302" "4409df")
                        + bytes.fromhex("4d8b08")               # [r8] still lo_init
                        + _imm64(10, lo_init)
                        + bytes.fromhex("4d39d1" "410f94c3" "450fb6db" "41c1e303" "4409df")
                        + _epilogue_ret()
                    )
                    self.assertEqual(self._run(code), 0x0F)

        # 3. Unaligned lock cmpxchg16b at [r8+8] hits fallback without SIGBUS
        code = (
            _align_r8_stack_frame()
            + _imm64(9, lo_init)
            + _imm64(10, hi_init)
            + bytes.fromhex("4d894808" "4d895010")              # [r8+8]=lo_init, [r8+16]=hi_init
            + _imm64(0, lo_init)
            + _imm64(2, hi_init)
            + _imm64(3, lo_new)
            + _imm64(1, hi_new)
            + bytes.fromhex("f0490fc74808")                     # lock cmpxchg16b [r8+8]
            + bytes.fromhex("410f94c3" "410fb6fb")              # bit0: ZF==1
            + bytes.fromhex("4d8b4808")
            + _imm64(10, lo_new)
            + bytes.fromhex("4d39d1" "410f94c3" "450fb6db" "41d1e3" "4409df")
            + bytes.fromhex("4d8b4810")
            + _imm64(10, hi_new)
            + bytes.fromhex("4d39d1" "410f94c3" "450fb6db" "41c1e302" "4409df")
            + _epilogue_ret()
        )
        self.assertEqual(self._run(code), 0x07)

        # 4. Realistic Windows InterlockedCompareExchange128 retry loop:
        #    start with stale RDX:RAX (0:0); first lock cmpxchg16b fails and loads
        #    live [r8] into RDX:RAX; second iteration computes (lo+5, hi+1) and succeeds.
        loop_body = bytes.fromhex(
            "488d5805"          # lea rbx, [rax+5]      (4)
            "488d4a01"          # lea rcx, [rdx+1]      (4)
            "41ffc1"            # inc r9d               (3)  # iteration counter
            "f0490fc708"        # lock cmpxchg16b [r8]  (5)
        )
        rel_back = (-(len(loop_body) + 2)) & 0xFF
        code = (
            _align_r8_stack_frame()
            + _imm64(9, 100)
            + _imm64(10, 7)
            + bytes.fromhex("4d8908" "4d895008")                # [r8] = {100, 7}
            + bytes.fromhex("31c0" "31d2" "4531c9")             # xor eax,eax; xor edx,edx; xor r9d,r9d
            + loop_body
            + bytes([0x75, rel_back])                           # jne loop_body
            # Verify: r9d == 2, [r8] == 105, [r8+8] == 8 -> return r9d*1000 + [r8] + [r8+8] = 2113
            + bytes.fromhex("418b38")                           # mov edi, [r8]      (105)
            + bytes.fromhex("41037808")                         # add edi, [r8+8]    (113)
            + bytes.fromhex("4569c9e8030000")                   # imul r9d, r9d, 1000 (2000)
            + bytes.fromhex("4401cf")                           # add edi, r9d       (2113)
            + _epilogue_ret()
        )
        self.assertEqual(self._run(code), 2113)

    def test_exec_preserves_other_flags_arm_and_ctx(self):
        # Test both same-block (ARM NZCV path) and cross-block (materialized ctx.eflags path).
        # Setup 1: sub (0 - 1) -> 0xFFFFFFFF: CF=1, SF=1, OF=0, PF=1 (0xFF has 8 ones), ZF=0.
        #          Then lock cmpxchg8b MATCH -> ZF becomes 1; CF/SF/OF/PF must stay (1,1,0,1)!
        #          Expected packed mask: ZF=1 | CF=2 | SF=4 | OF=0 | PF=16 = 0x17.
        for cross_block in (False, True):
            with self.subTest(case="match_preserves_cf_sf_of_pf", cross_block=cross_block):
                code = (
                    _align_r8_stack_frame()
                    + _imm64(9, 0x2222222211111111)
                    + bytes.fromhex("4d8908")
                    + _imm64(0, 0x11111111)
                    + _imm64(2, 0x22222222)
                    + _imm64(3, 0x33333333)
                    + _imm64(1, 0x44444444)
                    + bytes.fromhex("41b900000000" "4183e901")  # mov r9d, 0; sub r9d, 1 -> CF=1,SF=1,OF=0,PF=1,ZF=0
                    + bytes.fromhex("f0410fc708")               # lock cmpxchg8b [r8] -> match (ZF:=1)
                    + (bytes.fromhex("eb00") if cross_block else b"")
                    + _pack_flags_to_edi()
                    + _epilogue_ret()
                )
                self.assertEqual(self._run(code), 0x17)

        # Setup 2: sub (0x80000000 - 1) -> 0x7FFFFFFF: CF=0, SF=0, OF=1, PF=1 (0xFF), ZF=0;
        #          actually start with ZF=1? Use 0x80000000 - 0x80000000? That has OF=0.
        #          Use sub (0x80000000 - 1) -> CF=0, SF=0, OF=1, PF=1, ZF=0, then
        #          lock cmpxchg16b MATCH -> ZF=1 (0x19), and MISMATCH after xor -> ZF=0.
        for cross_block in (False, True):
            with self.subTest(case="cmpxchg16b_preserves_of_pf", cross_block=cross_block):
                code = (
                    _align_r8_stack_frame()
                    + _imm64(9, 0x1111)
                    + _imm64(10, 0x2222)
                    + bytes.fromhex("4d8908" "4d895008")
                    + _imm64(0, 0x1111)
                    + _imm64(2, 0x2222)
                    + _imm64(3, 0x3333)
                    + _imm64(1, 0x4444)
                    + bytes.fromhex("41b900000080" "4183e901")  # 0x80000000 - 1 -> OF=1,PF=1,CF=0,SF=0,ZF=0
                    + bytes.fromhex("f0490fc708")               # lock cmpxchg16b [r8] -> match (ZF:=1)
                    + (bytes.fromhex("eb00") if cross_block else b"")
                    + _pack_flags_to_edi()
                    + _epilogue_ret()
                )
                self.assertEqual(self._run(code), 0x19)         # ZF=1 | OF=8 | PF=16

        # Setup 3: Start with ZF=1, CF=1, PF=1, SF=0, OF=0 (xor r9d,r9d; stc),
        #          then lock cmpxchg16b MISMATCH -> ZF:=0 while CF=1, PF=1 stay!
        for cross_block in (False, True):
            with self.subTest(case="mismatch_clears_only_zf", cross_block=cross_block):
                code = (
                    _align_r8_stack_frame()
                    + _imm64(9, 0x1111)
                    + _imm64(10, 0x2222)
                    + bytes.fromhex("4d8908" "4d895008")
                    + _imm64(0, 0x9999)                         # mismatch!
                    + _imm64(2, 0x2222)
                    + _imm64(3, 0x3333)
                    + _imm64(1, 0x4444)
                    + bytes.fromhex("4531c9" "f9")              # xor r9d, r9d (ZF=1,PF=1,SF=0,OF=0); stc (CF=1)
                    + bytes.fromhex("f0490fc708")               # lock cmpxchg16b [r8] -> mismatch (ZF:=0)
                    + (bytes.fromhex("eb00") if cross_block else b"")
                    + _pack_flags_to_edi()
                    + _epilogue_ret()
                )
                self.assertEqual(self._run(code), 0x12)         # ZF=0 | CF=2 | PF=16

        # Setup 4: Dead-ZF path — nobody reads ZF after lock cmpxchg8b, only CF from earlier sub.
        # Proves alignment guard and compare inside _emit_cas2 do NOT clobber NZCV.
        code = (
            _align_r8_stack_frame()
            + _imm64(9, 0x2222222211111111)
            + bytes.fromhex("4d8908")
            + _imm64(0, 0x11111111)
            + _imm64(2, 0x22222222)
            + _imm64(3, 0x33333333)
            + _imm64(1, 0x44444444)
            + bytes.fromhex("41b900000000" "4183e901")          # sub 0-1 -> CF=1
            + bytes.fromhex("f0410fc708")                       # lock cmpxchg8b [r8] (ZF dead!)
            + bytes.fromhex("410f92c3" "410fb6fb")              # setc r11b; movzx edi, r11b -> 1
            + _epilogue_ret()
        )
        self.assertEqual(self._run(code), 1)


if __name__ == "__main__":
    unittest.main()
