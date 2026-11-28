"""Unit tests for std/cld (DF=1), reverse string ops, and lods in tools/x64_recomp.py.

Covers std/cld direction tracking in CTX_DF, pushf/popf DF round-trip (bit 10),
overlapping backward rep movsb, backward rep stosd, forward & backward lodsb/lodsd/lodsq,
backward repne scasb / repe cmpsb (including cross-block flags), and NZCV preservation
across std/cld/movs/stos/lods and zero-count rep scas. No game files.

    python -I tests/test_x64_recomp_std_str.py
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


class StdStrTranslateTests(unittest.TestCase):
    def test_std_cld_lods_and_rep_strings_translate_cleanly(self):
        code = asm(
            "fd",                   # std
            "f3 a4",                # rep movsb
            "f3 ab",                # rep stosd
            "ac",                   # lodsb
            "66 ad",                # lodsw
            "ad",                   # lodsd
            "48 ad",                # lodsq
            "f2 ae",                # repne scasb
            "f3 a6",                # repe cmpsb
            "fc",                   # cld
            "9c", "9d",             # pushf; popf
            "c3",
        )
        tr = x.translate_function(code, GUEST_VA)
        self.assertEqual(tr.stats["unhandled"], [])
        self.assertEqual(tr.stats["insns"], 13)


@unittest.skipUnless(CAN_RUN, "neither Apple Silicon Mach-O runner nor qemu-system-aarch64 available")
class StdStrExecTests(unittest.TestCase):
    def test_std_overlapping_backward_rep_movsb(self):
        # Overlapping memmove upward: copy [s+0..s+3] = 'ABCD' -> [s+2..s+5] using DF=1.
        # Forward copy would corrupt it to 'ABABAB'; backward copy yields [s+2..s+5] == 'ABCD'.
        s = EXEC_SCRATCH
        code = asm(
            "48 bb " + s.to_bytes(8, "little").hex(),
            "c7 03 41 42 43 44",                        # [s+0..s+3] = 'ABCD'
            "c7 43 04 00 00 00 00",                     # [s+4..s+7] = 0
            "48 8d 73 03",                              # lea rsi, [rbx+3] ('D')
            "48 8d 7b 05",                              # lea rdi, [rbx+5]
            "b9 04 00 00 00",                           # mov ecx, 4
            "fd",                                       # std (DF=1)
            "f3 a4",                                    # rep movsb backward
            "fc",                                       # cld (DF=0)
            "8b 7b 02",                                 # mov edi, [rbx+2] == 'ABCD' (0x44434241)
            "c3",
        )
        r, tr = run(code)
        self.assertEqual(tr.stats["unhandled"], [])
        self.assertEqual(r, 0x44434241)

    def test_std_stosd_and_lods_forward_and_backward(self):
        # Backward rep stosd fills [s+0..s+11] with 0x11223344, then lodsb/lodsd verify
        # forward & backward stepping plus partial-width rax preservation.
        s = EXEC_SCRATCH
        code = asm(
            "48 bb " + s.to_bytes(8, "little").hex(),
            "48 8d 7b 08",                              # rdi = rbx+8
            "b8 44 33 22 11",
            "b9 03 00 00 00",
            "fd", "f3 ab",                              # backward rep stosd -> [rbx+0..11] = 0x11223344
            "48 8d 73 03",                              # rsi = rbx+3
            "b8 00 55 00 00",                           # eax = 0x00005500
            "ac",                                       # lodsb (DF=1): al=0x11, ah=0x55 kept -> eax=0x5511
            "89 c7",                                    # edi = 0x5511
            "ac",                                       # lodsb (DF=1): al=0x22 -> rsi=rbx+1
            "0f b6 c0", "c1 e7 08", "09 c7",            # edi = 0x551122
            "fc",                                       # cld (DF=0)
            "ac",                                       # lodsb (DF=0): reads [rbx+1]=0x33 -> rsi=rbx+2
            "0f b6 c0", "c1 e7 08", "09 c7",            # edi = 0x55112233
            "48 8d 73 04",                              # rsi = rbx+4
            "ad",                                       # lodsd -> eax = 0x11223344
            "35 44 33 22 11",                           # xor eax, 0x11223344 -> 0
            "09 c7",                                    # edi stays 0x55112233 iff lodsd matched!
            "c3",
        )
        r, tr = run(code)
        self.assertEqual(tr.stats["unhandled"], [])
        self.assertEqual(r, 0x55112233)

    def test_std_scasb_cmpsb_and_cross_block_flags(self):
        # Backward repne scasb finds 'Q' from right end; backward repe cmpsb stops at mismatch
        # and propagates ZF across a block boundary.
        s = EXEC_SCRATCH
        code = asm(
            "48 bb " + s.to_bytes(8, "little").hex(),
            "c7 03 51 58 58 58",                        # [s+0..3] = 'QXXX'
            "48 8d 7b 03",                              # rdi = s+3
            "b0 51",                                    # al = 'Q'
            "b9 04 00 00 00",
            "fd", "f2 ae",                              # backward repne scasb -> matches at s+0, rcx=0, ZF=1
            "eb 00",                                    # split block -> forces CTX_EFLAGS
            "0f 94 c2", "0f b6 d2",                     # dl = ZF (1)
            # Now compare 'ABZX' vs 'ABYX' backward: matches 'X', mismatches 'Z' vs 'Y' -> ZF=0
            "c7 43 10 41 42 5a 58",
            "c7 43 14 41 42 59 58",
            "48 8d 73 13",                              # rsi = s+0x13 ('X')
            "48 8d 7b 17",                              # rdi = s+0x17 ('X')
            "b9 04 00 00 00",
            "f3 a6",                                    # backward repe cmpsb -> ZF=0, rcx=2
            "eb 00",                                    # split block
            "0f 95 c0", "0f b6 c0",                     # eax = (ZF==0) -> 1
            "fc",                                       # cld
            "c1 e2 04", "09 c2",                        # edx = (1<<4)|1 = 0x11
            "c1 e2 04", "09 ca",                        # edx = (0x11<<4)|rcx(2) = 0x112
            "89 d7",
            "c3",
        )
        r, tr = run(code)
        self.assertEqual(tr.stats["unhandled"], [])
        self.assertEqual(r, 0x112)

    def test_pushf_popf_df_roundtrip_and_nzcv_preservation(self):
        # std -> pushf (bit 10 set) -> cld -> popf (DF=1 restored!) -> backward lodsb confirms DF=1.
        # Also verify std/cld/rep movsb/zero-count repne scasb preserve live NZCV flags.
        s = EXEC_SCRATCH
        code = asm(
            "48 bb " + s.to_bytes(8, "little").hex(),
            "c7 03 11 22 33 44",                        # [s+0]=0x11, [s+1]=0x22, [s+2]=0x33
            "fd",                                       # std (DF=1)
            "9c",                                       # pushf
            "58",                                       # pop rax -> bit 10 (0x400) must be 1
            "50",                                       # push rax back
            "fc",                                       # cld (DF=0)
            "9d",                                       # popf -> restores DF=1!
            "48 8d 73 02",                              # rsi = s+2 (0x33)
            "ac",                                       # lodsb -> 0x33, steps backward to s+1!
            "ac",                                       # lodsb -> 0x22!
            "0f b6 e8",                                 # ebp = 0x22 (save away from rdi)
            # Verify live NZCV survives std, cld, rep movsb, and zero-count repne scasb:
            "b8 03 00 00 00", "83 f8 09",               # cmp 3, 9 -> lt (SF!=OF)
            "fc",                                       # cld (preserves NZCV)
            "48 8d 73 00", "48 8d 7b 10",               # rsi=s, rdi=s+16
            "b9 04 00 00 00", "f3 a4",                  # rep movsb (preserves NZCV)
            "b9 00 00 00 00", "f2 ae",                  # mov ecx,0 (no flag change); repne scasb (no-op)
            "0f 9c c0", "0f b6 c0",                     # eax = setl -> 1
            "89 ef",                                    # edi = ebp (0x22)
            "c1 e7 04", "09 c7",                        # edi = 0x221
            "c3",
        )
        r, tr = run(code)
        self.assertEqual(tr.stats["unhandled"], [])
        self.assertEqual(r, 0x221)


if __name__ == "__main__":
    unittest.main()
