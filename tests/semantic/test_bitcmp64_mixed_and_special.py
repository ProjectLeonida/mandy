"""Semantic tests for s_bitcmp0/1_b64 and s_bitcmp0/1_b32 on mixed pairs, wave32 VCC, and mask bit indices.

Covers four edge cases in Gen.bitcmp:
  1. Mixed SGPR pair (one dword holds a lane mask, the other holds plain scalar data or EITHER -1)
     in both wave32 and wave64.
  2. Swapped / duplicated wave64 half-mask SGPR pairs (e.g. s0 = exec_hi, s1 = exec_lo; or both exec_lo)
     in native wave64 and --wave64-emulate.
  3. Wave32 `s_bitcmp0/1_b64 vcc, bit` when `vcc_lo` holds a lane mask and `vcc_hi` holds plain scalar data
     (or a separate wave32 lane mask): bits 32..63 must read `vcc_hi`, not 0.
  4. Lane-mask bit-index operand `a[1]` (e.g. `s_bitcmp1_b32 s10, vcc_lo` or `s_bitcmp1_b64 s[10:11], vcc_lo`):
     must use the wave's 32-bit mask word rather than per-lane 0 / 0xFFFFFFFF.

    python -I tests/semantic/test_bitcmp64_mixed_and_special.py
"""
import gc
import json
import os
import struct
import sys
import tempfile
import unittest
import warnings

SEMANTIC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(SEMANTIC_DIR))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")
for _p in (TOOLS_DIR, SEMANTIC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
VSHARP = [0, 0, 0xFFFFFFFF, 0x31016FAC]


def translate_cs(name: str, body_lines: list[str], *, wave64: bool = False,
                 emulate: bool = False, fspx: bool = False) -> str:
    threads = 64 if wave64 else 32
    with tempfile.TemporaryDirectory() as td:
        os.mkdir(os.path.join(td, "disasm"))
        src = os.path.join(td, "disasm", f"cs_{name}.s")
        with open(src, "w", encoding="utf-8", newline="\n") as f:
            f.write(f"; cs_{name}\n")
            for ln in body_lines:
                f.write(f"\t{ln}\n")
            f.write("\ts_endpgm\n")
        with open(os.path.join(td, f"cs_{name}.dxbc"), "wb") as f:
            f.write(STUB_DXBC)
        if fspx:
            os.mkdir(os.path.join(td, "fspx"))
            meta = {"wave32": not wave64, "rsrc2": 0x8090, "threads": [threads, 1, 1]}
            with open(os.path.join(td, "fspx", f"cs_{name}.json"), "w", encoding="utf-8", newline="\n") as f:
                json.dump(meta, f)
        else:
            os.mkdir(os.path.join(td, "fssm"))
            fssm = bytearray(0x148)
            fssm[0:4] = b"FSSM"
            struct.pack_into("<I", fssm, 0xD4, 0x00 if wave64 else 0x20)
            struct.pack_into("<4I", fssm, 0x138, 0x8090, threads, 1, 1)
            with open(os.path.join(td, "fssm", f"cs_{name}.fssm"), "wb") as f:
                f.write(bytes(fssm))
        old_emu = rdna2spirv.WAVE64_EMULATE
        rdna2spirv.WAVE64_EMULATE = emulate
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                spv = rdna2spirv.translate(src)
                gc.collect()
        finally:
            rdna2spirv.WAVE64_EMULATE = old_emu
        return spv


def run_cs(spv: str, *, wave64: bool = False, emulate: bool = False) -> dict[int, int]:
    threads = 64 if wave64 else 32
    sub = 32 if emulate else threads
    pc = VSHARP + [0] * 28
    return si.run_workgroup(
        spv, threads, sub,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i % sub},
        push_constants=pc, memory={},
    )


class BitcmpMixedAndSpecialTests(unittest.TestCase):
    def test_wave32_mixed_mask_and_data_pair(self):
        # s10 = mask with lanes {0, 5} set (0x21); s11 = plain data 0x00000008 (bit 35 set).
        spv = translate_cs("w32_bitcmp64_mixed", [
            "s_and_b32 vcc_lo, exec_lo, 0x21",
            "s_mov_b32 s10, vcc_lo",
            "s_mov_b32 s11, 0x08",
            "s_bitcmp1_b64 s[10:11], 5",
            "s_cselect_b32 s12, 1, 0",
            "s_bitcmp1_b64 s[10:11], 4",
            "s_cselect_b32 s13, 1, 0",
            "s_bitcmp1_b64 s[10:11], 35",
            "s_cselect_b32 s14, 1, 0",
            "s_bitcmp0_b64 s[10:11], 36",
            "s_cselect_b32 s15, 1, 0",
            "v_mov_b32_e32 v1, s12",
            "v_mov_b32_e32 v2, s13",
            "v_mov_b32_e32 v3, s14",
            "v_mov_b32_e32 v4, s15",
            "v_lshlrev_b32_e32 v6, 2, v0",
            "buffer_store_dword v1, v6, s[0:3], 0 offen",
            "buffer_store_dword v2, v6, s[0:3], 0 offen offset:128",
            "buffer_store_dword v3, v6, s[0:3], 0 offen offset:256",
            "buffer_store_dword v4, v6, s[0:3], 0 offen offset:384",
        ], wave64=False)
        mem = run_cs(spv, wave64=False)
        self.assertEqual([mem[4 * i] for i in range(32)], [1] * 32)
        self.assertEqual([mem[128 + 4 * i] for i in range(32)], [0] * 32)
        self.assertEqual([mem[256 + 4 * i] for i in range(32)], [1] * 32)
        self.assertEqual([mem[384 + 4 * i] for i in range(32)], [1] * 32)

    def test_wave32_mask_and_either_minus1_pair_preserves_high_word(self):
        # s10 = mask (0x01), s11 = -1 (EITHER: all high bits 32..63 are 1).
        # On origin/main, holds_lane_mask(s[10:11]) was True and wave_mask dropped hi to 0.
        spv = translate_cs("w32_bitcmp64_either_hi", [
            "s_and_b32 vcc_lo, exec_lo, 0x01",
            "s_mov_b32 s10, vcc_lo",
            "s_mov_b32 s11, -1",
            "s_bitcmp1_b64 s[10:11], 0",
            "s_cselect_b32 s12, 1, 0",
            "s_bitcmp1_b64 s[10:11], 40",
            "s_cselect_b32 s13, 1, 0",
            "v_mov_b32_e32 v1, s12",
            "v_mov_b32_e32 v2, s13",
            "v_lshlrev_b32_e32 v6, 2, v0",
            "buffer_store_dword v1, v6, s[0:3], 0 offen",
            "buffer_store_dword v2, v6, s[0:3], 0 offen offset:128",
        ], wave64=False)
        mem = run_cs(spv, wave64=False)
        self.assertEqual([mem[4 * i] for i in range(32)], [1] * 32)
        self.assertEqual([mem[128 + 4 * i] for i in range(32)], [1] * 32)

    def test_wave32_vcc_mask_lo_and_data_hi(self):
        # In wave32, vcc = {vcc_hi, vcc_lo}. vcc_lo is a lane mask (0x04 -> bit 2),
        # vcc_hi is plain scalar data (0x10 -> bit 36).
        spv = translate_cs("w32_bitcmp64_vcc_mixed", [
            "s_and_b32 vcc_lo, exec_lo, 0x04",
            "s_mov_b32 vcc_hi, 0x10",
            "s_bitcmp1_b64 vcc, 2",
            "s_cselect_b32 s12, 1, 0",
            "s_bitcmp1_b64 vcc, 36",
            "s_cselect_b32 s13, 1, 0",
            "s_bitcmp1_b64 vcc, 35",
            "s_cselect_b32 s14, 1, 0",
            "v_mov_b32_e32 v1, s12",
            "v_mov_b32_e32 v2, s13",
            "v_mov_b32_e32 v3, s14",
            "v_lshlrev_b32_e32 v6, 2, v0",
            "buffer_store_dword v1, v6, s[0:3], 0 offen",
            "buffer_store_dword v2, v6, s[0:3], 0 offen offset:128",
            "buffer_store_dword v3, v6, s[0:3], 0 offen offset:256",
        ], wave64=False, fspx=True)
        mem = run_cs(spv, wave64=False)
        self.assertEqual([mem[4 * i] for i in range(32)], [1] * 32)
        self.assertEqual([mem[128 + 4 * i] for i in range(32)], [1] * 32)
        self.assertEqual([mem[256 + 4 * i] for i in range(32)], [0] * 32)

    def test_wave64_bitcmp32_high_half_mask(self):
        # In wave64, vcc has lane 2 (lo bit 2) and lane 35 (hi bit 3) set.
        # s_bitcmp1_b32 on vcc_hi or a MASK_HI SGPR must test hi (bit 3 = 1, bit 2 = 0),
        # not lo (bit 3 = 0, bit 2 = 1).
        lines = [
            "v_cmp_eq_u32_e64 s[14:15], v0, 2",
            "v_cmp_eq_u32_e64 vcc, v0, 35",
            "s_or_b64 vcc, vcc, s[14:15]",
            "s_mov_b32 s10, vcc_hi",
            "s_bitcmp1_b32 vcc_hi, 3",
            "s_cselect_b32 s16, 1, 0",
            "s_bitcmp1_b32 vcc_hi, 2",
            "s_cselect_b32 s17, 1, 0",
            "s_bitcmp1_b32 s10, 3",
            "s_cselect_b32 s18, 1, 0",
            "s_bitcmp1_b32 s10, 2",
            "s_cselect_b32 s19, 1, 0",
            "v_mov_b32_e32 v1, s16",
            "v_mov_b32_e32 v2, s17",
            "v_mov_b32_e32 v3, s18",
            "v_mov_b32_e32 v4, s19",
            "v_lshlrev_b32_e32 v6, 2, v0",
            "buffer_store_dword v1, v6, s[0:3], 0 offen",
            "buffer_store_dword v2, v6, s[0:3], 0 offen offset:256",
            "buffer_store_dword v3, v6, s[0:3], 0 offen offset:512",
            "buffer_store_dword v4, v6, s[0:3], 0 offen offset:768",
        ]
        for emu in (False, True):
            with self.subTest(emulate=emu):
                spv = translate_cs("w64_bitcmp32_hi", lines, wave64=True, emulate=emu)
                mem = run_cs(spv, wave64=True, emulate=emu)
                self.assertEqual([mem[4 * i] for i in range(64)], [1] * 64)
                self.assertEqual([mem[256 + 4 * i] for i in range(64)], [0] * 64)
                self.assertEqual([mem[512 + 4 * i] for i in range(64)], [1] * 64)
                self.assertEqual([mem[768 + 4 * i] for i in range(64)], [0] * 64)

    def test_wave64_swapped_halves_and_mixed_pair(self):
        # vcc has lanes 2 (lo bit 2) and 35 (hi bit 3) set.
        # Put swapped halves into s[10:11]: s10 = vcc_hi (0x08 -> bit 3), s11 = vcc_lo (0x04 -> bit 34).
        # Also test mixed pair s[12:13]: s12 = vcc_lo (0x04 -> bit 2), s13 = 0x20 (bit 37).
        lines = [
            "v_cmp_eq_u32_e64 s[14:15], v0, 2",
            "v_cmp_eq_u32_e64 vcc, v0, 35",
            "s_or_b64 vcc, vcc, s[14:15]",
            "s_mov_b32 s10, vcc_hi",
            "s_mov_b32 s11, vcc_lo",
            "s_mov_b32 s12, vcc_lo",
            "s_mov_b32 s13, 0x20",
            "s_bitcmp1_b64 s[10:11], 3",
            "s_cselect_b32 s16, 1, 0",
            "s_bitcmp1_b64 s[10:11], 34",
            "s_cselect_b32 s17, 1, 0",
            "s_bitcmp1_b64 s[12:13], 2",
            "s_cselect_b32 s18, 1, 0",
            "s_bitcmp1_b64 s[12:13], 37",
            "s_cselect_b32 s19, 1, 0",
            "v_mov_b32_e32 v1, s16",
            "v_mov_b32_e32 v2, s17",
            "v_mov_b32_e32 v3, s18",
            "v_mov_b32_e32 v4, s19",
            "v_lshlrev_b32_e32 v6, 2, v0",
            "buffer_store_dword v1, v6, s[0:3], 0 offen",
            "buffer_store_dword v2, v6, s[0:3], 0 offen offset:256",
            "buffer_store_dword v3, v6, s[0:3], 0 offen offset:512",
            "buffer_store_dword v4, v6, s[0:3], 0 offen offset:768",
        ]
        for emu in (False, True):
            with self.subTest(emulate=emu):
                spv = translate_cs("w64_bitcmp64_swapped", lines, wave64=True, emulate=emu)
                mem = run_cs(spv, wave64=True, emulate=emu)
                self.assertEqual([mem[4 * i] for i in range(64)], [1] * 64)
                self.assertEqual([mem[256 + 4 * i] for i in range(64)], [1] * 64)
                self.assertEqual([mem[512 + 4 * i] for i in range(64)], [1] * 64)
                self.assertEqual([mem[768 + 4 * i] for i in range(64)], [1] * 64)

    def test_lane_mask_bit_index_operand(self):
        # vcc_lo = 0x05 (lanes 0 and 2 active) -> bit index = 5.
        # s10 = 0x20 (bit 5 set, bit 0 clear, bit 31 clear), s11 = 0.
        spv = translate_cs("w32_bitcmp_mask_idx", [
            "s_mov_b32 s10, 0x20",
            "s_mov_b32 s11, 0",
            "s_and_b32 vcc_lo, exec_lo, 0x05",
            "s_bitcmp1_b32 s10, vcc_lo",
            "s_cselect_b32 s12, 1, 0",
            "s_bitcmp1_b64 s[10:11], vcc_lo",
            "s_cselect_b32 s13, 1, 0",
            "v_mov_b32_e32 v1, s12",
            "v_mov_b32_e32 v2, s13",
            "v_lshlrev_b32_e32 v6, 2, v0",
            "buffer_store_dword v1, v6, s[0:3], 0 offen",
            "buffer_store_dword v2, v6, s[0:3], 0 offen offset:128",
        ], wave64=False)
        mem = run_cs(spv, wave64=False)
        self.assertEqual([mem[4 * i] for i in range(32)], [1] * 32)
        self.assertEqual([mem[128 + 4 * i] for i in range(32)], [1] * 32)


if __name__ == "__main__":
    unittest.main()
