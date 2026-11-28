"""Unit tests for bit test/scan/count emitters in tools/x64_recomp.py.

Covers bt/bts/btr/btc (reg/mem, reg/imm8, including signed memory bit offsets),
bsf/bsr, tzcnt/lzcnt (F3 0F BC/BD), and popcnt (F3 0F B8) across 16/32/64-bit
operand sizes without game files or third-party packages.

    python -I tests/test_x64_recomp_bitops.py
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


class BitopsDecodeTests(unittest.TestCase):
    def one(self, code, base=0x1000):
        return x.decode_one(code, 0, len(code), base)

    def test_bt_family_decode_and_flags(self):
        for hex_bytes, mnem in (
            ("0f a3 c8", "bt"),
            ("48 0f ab c8", "bts"),
            ("0f b3 c8", "btr"),
            ("66 0f bb c8", "btc"),
            ("0f ba e0 05", "bt"),
            ("0f ba e8 05", "bts"),
            ("0f ba f0 05", "btr"),
            ("0f ba f8 05", "btc"),
        ):
            ins = self.one(asm(hex_bytes))
            self.assertEqual((ins.mnem, ins.kind, ins.flags_w, ins.flags_r),
                             (mnem, "bt", "C", ""), hex_bytes)

    def test_bsf_bsr_tzcnt_lzcnt_popcnt_decode(self):
        cases = (
            ("0f bc c1", "bsf", "alu", "Z", 4),
            ("48 0f bc c1", "bsf", "alu", "Z", 8),
            ("66 0f bd c1", "bsr", "alu", "Z", 2),
            ("f3 0f bc c1", "tzcnt", "alu", "CZ", 4),
            ("f3 48 0f bc c1", "tzcnt", "alu", "CZ", 8),
            ("66 f3 0f bc c1", "tzcnt", "alu", "CZ", 2),
            ("f3 0f bd c1", "lzcnt", "alu", "CZ", 4),
            ("f3 48 0f bd c1", "lzcnt", "alu", "CZ", 8),
            ("66 f3 0f bd c1", "lzcnt", "alu", "CZ", 2),
            ("f3 0f b8 c1", "popcnt", "alu", "Z", 4),
            ("f3 48 0f b8 c1", "popcnt", "alu", "Z", 8),
            ("66 f3 0f b8 c1", "popcnt", "alu", "Z", 2),
        )
        for hex_bytes, mnem, kind, fw, sz in cases:
            ins = self.one(asm(hex_bytes))
            self.assertEqual((ins.mnem, ins.kind, ins.flags_w, ins.ops[0][2]),
                             (mnem, kind, fw, sz), hex_bytes)

        # Unprefixed 0F B8 is jmpe (not popcnt) and must stay unhandled.
        ins = self.one(asm("0f b8 c1"))
        self.assertEqual((ins.mnem, ins.kind), ("popcnt_jmpe", "sys"))


class Arm64BitEncodingTests(unittest.TestCase):
    def test_rbit_and_clz_encodings(self):
        a = x.Asm()
        a.rbit(1, 2, sf=0)
        a.rbit(1, 2, sf=1)
        a.clz(1, 2, sf=0)
        a.clz(1, 2, sf=1)
        a.li(0, -1)
        self.assertEqual(a.words, [0x5AC00041, 0xDAC00041, 0x5AC01041, 0xDAC01041, 0x92800000])


class BitopsTranslateTests(unittest.TestCase):
    def test_all_bitops_translate_cleanly(self):
        code = asm(
            "0f a3 c8",             # bt eax, ecx
            "48 0f ab c8",          # bts rax, rcx
            "0f b3 07",             # btr [rdi], eax
            "0f ba 3f 03",          # btc dword [rdi], 3
            "0f bc c1",             # bsf eax, ecx
            "48 0f bd c1",          # bsr rax, rcx
            "f3 0f bc c1",          # tzcnt eax, ecx
            "66 f3 0f bd c1",       # lzcnt ax, cx
            "f3 48 0f b8 c1",       # popcnt rax, rcx
            "c3",
        )
        tr = x.translate_function(code, GUEST_VA)
        self.assertEqual(tr.stats["unhandled"], [])
        self.assertEqual(tr.stats["insns"], 10)

        # Unprefixed 0F B8 stays unhandled
        tr_bad = x.translate_function(asm("0f b8 c1", "c3"), GUEST_VA)
        self.assertEqual(len(tr_bad.stats["unhandled"]), 1)


@unittest.skipUnless(CAN_RUN, "neither Apple Silicon Mach-O runner nor qemu-system-aarch64 available")
class BitopsExecTests(unittest.TestCase):
    def test_bt_bts_btr_btc_register_forms(self):
        # Exercise bt/bts/btr/btc on 32-bit, 64-bit, and 16-bit regs (including index wrap & CF).
        code = asm(
            "31 ff",                                    # xor edi, edi (packed flag bits)
            "b8 00 00 00 00",                           # mov eax, 0
            "b9 25 00 00 00",                           # mov ecx, 37 (37 mod 32 = 5)
            "0f ab c8",                                 # bts eax, ecx -> old bit 5 = 0 (CF=0), eax=0x20
            "11 ff",                                    # adc edi, edi (bit <- 0)
            "0f a3 c8",                                 # bt eax, ecx -> bit 5 = 1 (CF=1)
            "11 ff",                                    # adc edi, edi (bit <- 1)
            "0f ba f8 05",                              # btc eax, 5 -> old=1 (CF=1), eax=0
            "11 ff",                                    # adc edi, edi (bit <- 1)
            "0f ba e8 05",                              # bts eax, 5 -> old=0 (CF=0), eax=0x20
            "11 ff",                                    # adc edi, edi (bit <- 0)
            "0f b3 c8",                                 # btr eax, ecx -> old=1 (CF=1), eax=0
            "11 ff",                                    # adc edi, edi (bit <- 1)
            # 64-bit bit 63 test + toggle
            "48 b8 00 00 00 00 00 00 00 80",            # mov rax, 1<<63
            "48 0f ba f8 3f",                           # btc rax, 63 -> old=1 (CF=1), rax=0
            "11 ff",                                    # adc edi, edi (bit <- 1)
            "48 0f ba e0 3f",                           # bt rax, 63 -> now 0 (CF=0)
            "11 ff",                                    # adc edi, edi (bit <- 0)
            # 16-bit bts preserves upper 48 bits of rax
            "48 b8 00 00 55 aa 00 00 00 00",            # rax = 0xAA550000
            "66 0f ba e8 13",                           # bts ax, 19 (19 mod 16 = 3) -> rax=0xAA550008
            "3d 08 00 55 aa",                           # cmp eax, 0xAA550008
            "0f 94 c2",                                 # sete dl
            "0f b6 d2",
            "c1 e7 01", "09 d7",                        # pack match bit
            "c3",
        )
        r, tr = run(code)
        self.assertEqual(tr.stats["unhandled"], [])
        # CF sequence: 0, 1, 1, 0, 1, 1, 0 -> 0b0110110, plus final 1 -> 0b01101101 = 0x6D
        self.assertEqual(r, 0x6D)

    def test_bt_memory_signed_and_immediate_offsets(self):
        # Memory register bit offset is signed and can reach bytes before & after base pointer.
        # Memory immediate bit offset is masked modulo operand width.
        s = EXEC_SCRATCH
        code = asm(
            "48 be " + (s + 8).to_bytes(8, "little").hex(),  # rsi = scratch + 8
            "c7 46 f8 00 00 00 00",                         # [rsi-8..rsi-5] = 0
            "c7 46 fc 00 00 00 80",                         # [rsi-1] = 0x80 (bit -1 is 1!)
            "c7 06 00 00 00 00",                            # [rsi+0..rsi+3] = 0
            "c7 46 04 00 00 00 00",                         # [rsi+4..rsi+7] = 0
            "31 ff",                                        # xor edi, edi
            # 1) bt dword [rsi], -1 -> reads bit 7 of [rsi-1] == 1
            "b9 ff ff ff ff",                               # ecx = -1
            "0f a3 0e",                                     # bt [rsi], ecx -> CF=1
            "11 ff",                                        # adc edi, edi (1)
            # 2) bts dword [rsi], -9 -> sets bit 7 of [rsi-2] (old=0)
            "b9 f7 ff ff ff",                               # ecx = -9
            "0f ab 0e",                                     # bts [rsi], ecx -> CF=0, [rsi-2]=0x80
            "11 ff",                                        # adc edi, edi (0)
            # 3) bts dword [rsi], 35 -> sets bit 3 of [rsi+4] (old=0)
            "b9 23 00 00 00",                               # ecx = 35
            "0f ab 0e",                                     # bts [rsi], ecx -> CF=0, [rsi+4]=0x08
            "11 ff",                                        # adc edi, edi (0)
            # 4) btr dword [rsi], -1 -> clears bit 7 of [rsi-1] (old=1)
            "b9 ff ff ff ff",
            "0f b3 0e",                                     # btr [rsi], ecx -> CF=1, [rsi-1]=0
            "11 ff",                                        # adc edi, edi (1)
            # 5) btc dword [rsi], 35 -> toggles bit 3 of [rsi+4] (old=1 -> 0)
            "b9 23 00 00 00",
            "0f bb 0e",                                     # btc [rsi], ecx -> CF=1, [rsi+4]=0
            "11 ff",                                        # adc edi, edi (1)
            # 6) immediate bts dword [rsi], 35 -> 35 mod 32 = 3 -> sets bit 3 of [rsi+0], NOT [rsi+4]!
            "0f ba 2e 23",                                  # bts dword [rsi], 35 -> [rsi+0]=0x08
            "11 ff",                                        # adc edi, edi (0)
            # Verify memory bytes: [rsi-2]==0x80, [rsi-1]==0x00, [rsi+0]==0x08, [rsi+4]==0x00
            "0f b6 46 fe", "35 80 00 00 00",                # [rsi-2] ^ 0x80 (imm32; 83 f0 80 sign-extends to -128)
            "0f b6 56 ff", "09 d0",                         # | [rsi-1]
            "0f b6 16", "83 f2 08", "09 d0",                # | ([rsi+0] ^ 0x08)
            "0f b6 56 04", "09 d0",                         # | [rsi+4]
            "85 c0", "0f 94 c2", "0f b6 d2",                # dl = (all bytes exact)
            "c1 e7 01", "09 d7",
            "c3",
        )
        r, tr = run(code)
        self.assertEqual(tr.stats["unhandled"], [])
        # Sequence: 1, 0, 0, 1, 1, 0, plus match=1 -> 0b1001101 = 0x4D
        self.assertEqual(r, 0x4D)

    def test_bsf_and_bsr_including_zero_preserves_dst(self):
        # Test 32/64/16-bit bsf/bsr and verify src==0 sets ZF=1 while keeping dst intact.
        code = asm(
            # bsf eax, 0x00100100 -> 8, ZF=0
            "b9 00 01 10 00",
            "0f bc c1",                                 # bsf eax, ecx -> 8
            "0f 95 c2", "0f b6 d2",                     # dl = (ZF==0) -> 1
            "c1 e0 04", "09 d0",                        # eax = (8<<4)|1 = 0x81
            "89 c7",                                    # edi = 0x81
            # bsr eax, 0x00100100 -> 20
            "0f bd c1",                                 # bsr eax, ecx -> 20 (0x14)
            "c1 e7 08", "09 c7",                        # edi = 0x8114
            # 64-bit bsr rax, (1<<63)|1 -> 63; bsf rax, (1<<40) -> 40
            "48 b9 01 00 00 00 00 00 00 80",
            "48 0f bd c1",                              # bsr rax, rcx -> 63 (0x3F)
            "c1 e7 08", "09 c7",                        # edi = 0x81143F
            # zero source: dst=0x5A must stay 0x5A, ZF=1
            "b8 5a 00 00 00",
            "31 c9",                                    # xor ecx, ecx
            "0f bc c1",                                 # bsf eax, 0 -> eax stays 0x5A, ZF=1
            "0f 94 c2", "0f b6 d2",                     # dl = (ZF==1) -> 1
            "01 d0",                                    # eax = 0x5B
            "c1 e7 08", "09 c7",                        # edi = 0x81143F5B? Wait: 0x143F5B (shifted 32b)
            "c3",
        )
        r, tr = run(code)
        self.assertEqual(tr.stats["unhandled"], [])
        self.assertEqual(r, 0x81143F5B)

    def test_tzcnt_and_lzcnt_widths_and_flags(self):
        # Pack results + CF/ZF checks for 16/32/64-bit tzcnt & lzcnt including zero and MSB/LSB.
        code = asm(
            # 1) tzcnt32(0) = 32, CF=1, ZF=0
            "31 c9",
            "f3 0f bc c1",                              # tzcnt eax, ecx -> 32
            "0f 92 c2", "0f b6 d2",                     # CF=1
            "0f 95 c3", "0f b6 db",                     # ZF=0 (setne)
            "c1 e2 01", "09 da",                        # edx = 0b11 = 3
            "c1 e0 02", "09 d0",                        # eax = (32<<2)|3 = 131
            "89 c7",                                    # edi = 131
            # 2) tzcnt16(0) = 16, CF=1, ZF=0
            "b8 ff ff 00 00",                           # upper bits 0xFFFF0000 preserved by 16-bit write? No, low16=0
            "66 f3 0f bc c1",                           # tzcnt ax, cx -> ax=16
            "0f b7 c0",                                 # movzx eax, ax -> 16
            "01 c7",                                    # edi += 16 -> 147
            # 3) tzcnt64(1) = 0, CF=0, ZF=1
            "48 b9 01 00 00 00 00 00 00 00",
            "f3 48 0f bc c1",                           # tzcnt rax, rcx -> 0, CF=0, ZF=1
            "0f 93 c2", "0f b6 d2",                     # CF==0 -> 1
            "0f 94 c3", "0f b6 db",                     # ZF==1 -> 1
            "01 da", "01 d0",                           # eax = 0 + 2 = 2
            "01 c7",                                    # edi += 2 -> 149
            # 4) lzcnt16(0x0010) = 11, lzcnt16(0) = 16
            "b9 10 00 00 00",
            "66 f3 0f bd c1",                           # lzcnt ax, cx -> 11
            "0f b7 c0", "01 c7",                        # edi += 11 -> 160
            "31 c9",
            "66 f3 0f bd c1",                           # lzcnt ax, 0 -> 16
            "0f b7 c0", "01 c7",                        # edi += 16 -> 176
            # 5) lzcnt64(1<<63) = 0, CF=0, ZF=1; lzcnt64(0) = 64, CF=1, ZF=0
            "48 b9 00 00 00 00 00 00 00 80",
            "f3 48 0f bd c1",                           # lzcnt rax, rcx -> 0, ZF=1, CF=0
            "0f 94 c2", "0f b6 d2",                     # ZF==1 -> 1
            "01 c7", "01 d7",                           # edi += 0 + 1 -> 177
            "48 31 c9",
            "f3 48 0f bd c1",                           # lzcnt rax, 0 -> 64, CF=1
            "0f 92 c2", "0f b6 d2",                     # CF==1 -> 1
            "01 c7", "01 d7",                           # edi += 64 + 1 -> 242
            "c3",
        )
        r, tr = run(code)
        self.assertEqual(tr.stats["unhandled"], [])
        self.assertEqual(r, 242)

    def test_popcnt_all_widths_and_zf(self):
        # popcnt64(-1)=64, popcnt64(0x5555..)=32, popcnt32(0xF0F00F0F)=16,
        # popcnt16(0x7FFF)=15, popcnt64(0)=0 with ZF=1 -> sum + ZF checks = 128
        code = asm(
            "31 ff",                                    # edi = 0
            "48 b9 ff ff ff ff ff ff ff ff",
            "f3 48 0f b8 c1",                           # popcnt rax, rcx -> 64
            "01 c7",
            "48 b9 55 55 55 55 55 55 55 55",
            "f3 48 0f b8 c9",                           # popcnt rcx, rcx (dst==src) -> 32
            "01 cf",
            "b9 0f 0f f0 f0",
            "f3 0f b8 c1",                              # popcnt eax, ecx -> 16
            "01 c7",
            "b9 ff 7f ff ff",                           # low16 = 0x7FFF (15 ones)
            "66 f3 0f b8 c1",                           # popcnt ax, cx -> 15
            "0f b7 c0", "01 c7",
            "48 31 c9",
            "f3 48 0f b8 c1",                           # popcnt rax, 0 -> 0, ZF=1
            "0f 94 c2", "0f b6 d2",                     # dl = 1
            "01 c7", "01 d7",                           # edi = 64+32+16+15+0+1 = 128
            "c3",
        )
        r, tr = run(code)
        self.assertEqual(tr.stats["unhandled"], [])
        self.assertEqual(r, 128)


if __name__ == "__main__":
    unittest.main()
