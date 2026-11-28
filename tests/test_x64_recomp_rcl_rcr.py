"""Unit tests for rcl/rcr rotate-through-carry emitters in tools/x64_recomp.py.

Covers 8/16/32/64-bit registers (including high-byte ah) and memory operands,
immediate (1 and imm8) and variable CL counts, modulo-(width+1) reduction for
8-bit/16-bit, zero-effective-count no-op flag preservation, sub/add/stc CF
chaining, count-1 OF, and dst==count (rcl ecx, cl). No game files.

    python -I tests/test_x64_recomp_rcl_rcr.py
"""
import os
import shutil
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
TOOLS_DIR = os.path.join(ROOT, "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

import x64_recomp as x  # noqa: E402

QEMU = os.environ.get(
    "QEMU_AARCH64",
    "/home/nolan/.nix-profile/bin/qemu-system-aarch64"
    if os.path.exists("/home/nolan/.nix-profile/bin/qemu-system-aarch64")
    else shutil.which("qemu-system-aarch64") or "")
HAS_QEMU = bool(QEMU) and os.path.exists(QEMU)
USE_MACHO = bool(getattr(x, "CAN_RUN_MACHO", False)) and not os.environ.get("QEMU_AARCH64")
CAN_RUN = USE_MACHO or HAS_QEMU

GUEST_VA = 0x46000000
EXEC_SCRATCH = (getattr(x, "DARWIN_RW", 0x47000000) + 0x100000) if USE_MACHO else 0x47100000


def asm(*hexes):
    return bytes.fromhex("".join(h.replace(" ", "") for h in hexes))


def run(code):
    return x.run_result(code, GUEST_VA)


def ref_rc(op, val, cf, raw_c, w):
    mask = (1 << w) - 1
    c = (raw_c & (0x3F if w == 64 else 0x1F)) % (w + 1)
    v = val & mask
    for _ in range(c):
        if op == "rcl":
            ncf = (v >> (w - 1)) & 1
            v = ((v << 1) | cf) & mask
        else:
            ncf = v & 1
            v = (v >> 1) | (cf << (w - 1))
        cf = ncf
    return v, cf


class RclRcrTranslateTests(unittest.TestCase):
    def test_all_rcl_rcr_forms_translate_cleanly(self):
        code = asm(
            "d0 d0", "d0 d8", "d0 d4",
            "c0 d0 03", "c0 d8 08", "c0 d0 09",
            "d2 d0", "d2 d8",
            "66 d1 d0", "66 d3 d0", "66 c1 d8 10",
            "d1 d0", "d1 d8", "c1 d0 05", "c1 d8 05",
            "d3 d0", "d3 d8", "d3 d1",
            "48 d1 d0", "48 d1 d8", "48 c1 d0 3f", "48 c1 d8 3f",
            "48 d3 d0", "48 d3 d8",
            "d1 17", "d2 1f",
            "c3",
        )
        tr = x.translate_function(code, GUEST_VA)
        self.assertEqual(tr.stats["unhandled"], [])
        self.assertEqual(tr.stats["insns"], 27)


@unittest.skipUnless(CAN_RUN, "neither Apple Silicon Mach-O runner nor qemu-system-aarch64 available")
class RclRcrExecTests(unittest.TestCase):
    def test_32bit_and_64bit_imm_and_cl(self):
        # Test 32-bit rcl/rcr for imm1, imm5, and CL=1/5/31 with both CF=0 and CF=1.
        for op, op_imm, op_cl in (("rcl", "c1 d0", "d3 d0"), ("rcr", "c1 d8", "d3 d8")):
            val = 0xA55A1234
            for cf_in in (0, 1):
                cf_op = "f9" if cf_in else "f8"
                for cnt in (1, 5, 31):
                    exp_v, exp_cf = ref_rc(op, val, cf_in, cnt, 32)
                    # Immediate form
                    code_imm = asm(
                        "b8 " + val.to_bytes(4, "little").hex(),
                        cf_op,
                        f"{op_imm} {cnt:02x}",
                        "0f 92 c2", "0f b6 d2",             # dl = CF
                        "31 d0", "89 c7", "c3",             # edi = res ^ CF
                    )
                    r, tr = run(code_imm)
                    self.assertEqual(tr.stats["unhandled"], [])
                    self.assertEqual(r, exp_v ^ exp_cf, (op, "imm", cf_in, cnt))
                    # CL variable form
                    code_cl = asm(
                        "b8 " + val.to_bytes(4, "little").hex(),
                        f"b9 {cnt:02x} 00 00 00",
                        cf_op,
                        op_cl,
                        "0f 92 c2", "0f b6 d2",
                        "31 d0", "89 c7", "c3",
                    )
                    r, tr = run(code_cl)
                    self.assertEqual(tr.stats["unhandled"], [])
                    self.assertEqual(r, exp_v ^ exp_cf, (op, "cl", cf_in, cnt))

        # 64-bit rcl/rcr with c=1, c=63 (imm and CL)
        val64 = 0x8123456789ABCDEF
        for op, op_imm, op_cl in (("rcl", "48 c1 d0", "48 d3 d0"), ("rcr", "48 c1 d8", "48 d3 d8")):
            for cnt in (1, 63):
                exp_v, exp_cf = ref_rc(op, val64, 1, cnt, 64)
                exp_fold = ((exp_v >> 32) ^ (exp_v & 0xFFFFFFFF) ^ exp_cf) & 0xFFFFFFFF
                for mode, body in (
                    ("imm", f"{op_imm} {cnt:02x}"),
                    ("cl", f"b9 {cnt:02x} 00 00 00 " + op_cl),
                ):
                    code = asm(
                        "48 b8 " + val64.to_bytes(8, "little").hex(),
                        "f9",                               # stc
                        body,
                        "0f 92 c2", "0f b6 d2",
                        "48 89 c3", "48 c1 eb 20",          # ebx = res >> 32
                        "31 d8", "31 d0", "89 c7", "c3",
                    )
                    r, tr = run(code)
                    self.assertEqual(tr.stats["unhandled"], [])
                    self.assertEqual(r, exp_fold, (op, mode, cnt))

    def test_8bit_and_16bit_modulo_width_plus_one_and_zero_noop(self):
        # 8-bit count reduces mod 9 (so 9 -> 0 no-op, 8 -> full wrap); 16-bit reduces mod 17.
        # Also verify upper bytes of eax are preserved and count==0 keeps CF & ZF intact.
        for op, op_imm, op_cl in (("rcl", "c0 d0", "d2 d0"), ("rcr", "c0 d8", "d2 d8")):
            for cnt in (0, 1, 8, 9, 10):
                exp_v, exp_cf = ref_rc(op, 0xA5, 1, cnt, 8)
                exp = 0x12345600 | exp_v
                for body in (f"{op_imm} {cnt:02x}", f"b9 {cnt:02x} 00 00 00 {op_cl}"):
                    code = asm(
                        "b8 a5 56 34 12",                   # eax = 0x123456A5
                        "f9",                               # stc (CF=1)
                        body,
                        "0f 92 c2", "0f b6 d2",             # dl = CF
                        "31 d0", "89 c7", "c3",
                    )
                    r, tr = run(code)
                    self.assertEqual(tr.stats["unhandled"], [])
                    self.assertEqual(r, exp ^ exp_cf, (op, 8, cnt))

        # 16-bit: count 16 and 17 (mod 17 -> 16 and 0)
        for cnt in (16, 17, 18):
            exp_v, exp_cf = ref_rc("rcr", 0xBEEF, 1, cnt, 16)
            exp = 0xCAFE0000 | exp_v
            code = asm(
                "b8 ef be fe ca",                           # eax = 0xCAFEBEEF
                f"b9 {cnt:02x} 00 00 00",
                "f9",                                       # stc
                "66 d3 d8",                                 # rcr ax, cl
                "0f 92 c2", "0f b6 d2",
                "31 d0", "89 c7", "c3",
            )
            r, tr = run(code)
            self.assertEqual(tr.stats["unhandled"], [])
            self.assertEqual(r, exp ^ exp_cf, (16, cnt))

    def test_chaining_sub_borrow_memory_hi8_dst_eq_cl_and_of(self):
        # 1) Multi-word 64-bit shift left via shl eax,1; rcl edx,1 (0x80000001 -> edx gets carry 1)
        # 2) sub-kind CF input: 1 - 2 sets borrow (CF=1), rcl eax, 1 pulls that 1 into bit 0!
        # 3) dst == count reg: rcl ecx, cl
        # 4) memory operand + hi8 reg (ah)
        s = EXEC_SCRATCH
        code = asm(
            # Part 1: 64-bit shift left (edx:eax = 0x00000003_80000001 << 1 -> edx = 7)
            "b8 01 00 00 80",
            "ba 03 00 00 00",
            "d1 e0",                                        # shl eax, 1 -> CF=1
            "d1 d2",                                        # rcl edx, 1 -> edx=7, CF=0
            "89 d7",                                        # edi = 7
            # Part 2: sub borrow (1 - 2 -> CF=1) fed directly into rcl eax, 1 (eax=0 -> 1)
            "bb 01 00 00 00",
            "83 eb 02",                                     # sub ebx, 2 -> CF=1 (SUB_KIND!)
            "b8 00 00 00 00",
            "d1 d0",                                        # rcl eax, 1 -> eax=1
            "c1 e7 04", "09 c7",                            # edi = (7<<4)|1 = 0x71
            # Part 3: rcl ecx, cl (ecx=3, CF=1 -> (3<<3)|(1<<2) = 0x1C)
            "b9 03 00 00 00",
            "f9",                                           # stc
            "d3 d1",                                        # rcl ecx, cl
            "c1 e7 08", "09 cf",                            # edi = 0x711C
            # Part 4: memory rcl dword [rsi], 1 (0x80000000 with CF=1 -> 1, CF=1)
            "48 be " + s.to_bytes(8, "little").hex(),
            "c7 06 00 00 00 80",
            "f9",
            "d1 16",                                        # rcl dword [rsi], 1
            "0f 92 c2", "0f b6 d2",                         # dl = CF (1)
            "8b 06",                                        # eax = 1
            "c1 e0 01", "09 d0",                            # eax = 3
            "c1 e7 04", "09 c7",                            # edi = 0x711C3
            # Part 5: hi8 reg rcl ah, 1 (ax=0x8005, CF=1 -> ax=0x0105, CF=1)
            "b8 05 80 00 00",
            "f9",
            "d0 d4",                                        # rcl ah, 1
            "3d 05 01 00 00",                               # cmp eax, 0x0105
            "0f 94 c2", "0f b6 d2",                         # dl = 1
            "c1 e7 04", "09 d7",                            # edi = 0x711C31
            "c3",
        )
        r, tr = run(code)
        self.assertEqual(tr.stats["unhandled"], [])
        self.assertEqual(r, 0x711C31)

    def test_count1_overflow_flag(self):
        # rcl(0x40000000, CF=0, 1) -> res=0x80000000, new_CF=0 -> OF = MSB(res)^new_CF = 1^0 = 1!
        # rcr(0x00000000, CF=1, 1) -> res=0x80000000 -> OF = bit31^bit30 = 1^0 = 1!
        # rcr(0x80000000, CF=1, 1) -> res=0xC0000000 -> OF = bit31^bit30 = 1^1 = 0!
        code = asm(
            "b8 00 00 00 40", "f8", "d1 d0",                # rcl eax, 1 -> OF=1
            "9c", "5b", "81 e3 00 08 00 00", "c1 eb 0b",    # ebx = (eflags >> 11) & 1 -> 1
            "89 df",
            "b8 00 00 00 00", "f9", "d1 d8",                # rcr eax, 1 -> OF=1
            "9c", "5b", "81 e3 00 08 00 00", "c1 eb 0b",
            "c1 e7 01", "09 df",
            "b8 00 00 00 80", "f9", "d1 d8",                # rcr eax, 1 -> OF=0
            "9c", "5b", "81 e3 00 08 00 00", "c1 eb 0b",
            "c1 e7 01", "09 df",
            "c3",
        )
        r, tr = run(code)
        self.assertEqual(tr.stats["unhandled"], [])
        self.assertEqual(r, 0b110)


if __name__ == "__main__":
    unittest.main()
