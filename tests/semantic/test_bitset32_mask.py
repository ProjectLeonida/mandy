"""Semantic tests for s_bitset0_b32 and s_bitset1_b32 on lane-mask destinations and sources.

On RDNA2, s_bitset0_b32 / s_bitset1_b32 mutate bit (S0 & 31) of D in place without modifying SCC.
When D holds a lane mask (exec_lo/exec_hi, vcc_lo/vcc_hi holding a mask, or a mask SGPR), each
invocation stores only its own lane's bit (0 or 0xFFFFFFFF), so mutating a bit mask directly on
that per-lane word corrupts every lane instead of flipping only lane (S0 & 31) (or lane (S0 & 31) + 32
for a wave64 high half). Likewise, a lane-mask bit-index source S0 must supply the wave's 32-bit
mask word rather than this lane's 0 / 0xFFFFFFFF.

    python -I tests/semantic/test_bitset32_mask.py
"""
import gc
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

import lane_masks  # noqa: E402
import rdna2spirv  # noqa: E402
import shader_ir  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

M32 = 0xFFFFFFFF
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
VSHARP = [0, 0, 0xFFFFFFFF, 0x31016FAC]


def translate_cs(name: str, body_lines: list[str], *, wave64: bool = False, emulate: bool = False) -> str:
    threads = 64 if wave64 else 32
    with tempfile.TemporaryDirectory() as td:
        os.mkdir(os.path.join(td, "disasm"))
        os.mkdir(os.path.join(td, "fssm"))
        src = os.path.join(td, "disasm", f"cs_{name}.s")
        with open(src, "w", encoding="utf-8", newline="\n") as f:
            f.write(f"; cs_{name}\n")
            for ln in body_lines:
                f.write(f"\t{ln}\n")
            f.write("\ts_endpgm\n")
        with open(os.path.join(td, f"cs_{name}.dxbc"), "wb") as f:
            f.write(STUB_DXBC)
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


class LaneMasksBitset32Tests(unittest.TestCase):
    def _analyse(self, lines, *, wave64: bool):
        insns = [shader_ir.parse_line(ln) for ln in lines]
        masks, _ = lane_masks.analyse([insns], [[]], wave64=wave64)
        return insns, masks

    def test_wave32_preserves_mask_sgpr_and_vcc(self):
        insns, masks = self._analyse([
            "s_mov_b32 s0, exec_lo",
            "s_bitset0_b32 s0, 5",
            "s_bcnt1_i32_b32 s1, s0",
            "v_cmp_lt_u32_e32 vcc_lo, v0, 10",
            "s_bitset1_b32 vcc_lo, 12",
            "s_bcnt1_i32_b32 s2, vcc_lo",
        ], wave64=False)
        self.assertEqual(masks[id(insns[2])].get(0), lane_masks.MASK)
        self.assertEqual(masks[id(insns[5])].get("vcc"), lane_masks.MASK)

    def test_wave64_preserves_lo_and_hi_halves(self):
        insns, masks = self._analyse([
            "s_mov_b32 s0, exec_lo",
            "s_mov_b32 s1, exec_hi",
            "s_bitset0_b32 s0, 3",
            "s_bitset1_b32 s1, 4",
            "s_bcnt1_i32_b32 s2, s0",
            "s_bcnt1_i32_b32 s3, s1",
            "v_cmp_lt_u32_e64 vcc, v0, 40",
            "s_bitset0_b32 vcc_lo, 2",
            "s_bitset1_b32 vcc_hi, 10",
            "s_bcnt1_i32_b64 s4, vcc",
        ], wave64=True)
        self.assertEqual(masks[id(insns[4])].get(0), lane_masks.MASK_LO)
        self.assertEqual(masks[id(insns[5])].get(1), lane_masks.MASK_HI)
        self.assertEqual(masks[id(insns[9])].get("vcc"), lane_masks.MASK)

    def test_either_constant_mutated_becomes_plain_data(self):
        insns, masks = self._analyse([
            "s_mov_b32 s0, 0",
            "s_bitset1_b32 s0, 5",
            "s_bcnt1_i32_b32 s1, s0",
        ], wave64=False)
        self.assertNotIn(0, masks[id(insns[2])])


class Bitset32MaskExecTests(unittest.TestCase):
    def test_wave32_exec_lo_bitset0_and_bitset1(self):
        # Start with EXEC = 0x15 (lanes 0, 2, 4). Clear bit 2 -> {0, 4}; set bit 7 -> {0, 4, 7}.
        spv = translate_cs("w32_exec_bitset", [
            "v_mov_b32_e32 v1, 0x1111",
            "s_mov_b32 exec_lo, 0x15",
            "s_bitset0_b32 exec_lo, 2",
            "s_bitset1_b32 exec_lo, 7",
            "v_mov_b32_e32 v1, 0xcafe",
            "s_mov_b32 exec_lo, -1",
            "v_lshlrev_b32_e32 v6, 2, v0",
            "buffer_store_dword v1, v6, s[0:3], 0 offen",
        ], wave64=False)
        mem = run_cs(spv, wave64=False)
        got = [mem[4 * i] for i in range(32)]
        expected = [0xCAFE if i in (0, 4, 7) else 0x1111 for i in range(32)]
        self.assertEqual(got, expected)

    def test_wave64_exec_lo_and_exec_hi_keep_other_half(self):
        # EXEC_LO = 0x0F (lanes 0..3), EXEC_HI = 0x03 (lanes 32, 33).
        # Clear exec_lo bit 1, set exec_lo bit 5, clear exec_hi bit 0 (lane 32), set exec_hi bit 4 (lane 36).
        lines = [
            "v_mov_b32_e32 v1, 0x2222",
            "s_mov_b32 exec_lo, 0x0f",
            "s_mov_b32 exec_hi, 0x03",
            "s_bitset0_b32 exec_lo, 1",
            "s_bitset1_b32 exec_lo, 5",
            "s_bitset0_b32 exec_hi, 0",
            "s_bitset1_b32 exec_hi, 4",
            "v_mov_b32_e32 v1, 0xbeef",
            "s_mov_b64 exec, -1",
            "v_lshlrev_b32_e32 v6, 2, v0",
            "buffer_store_dword v1, v6, s[0:3], 0 offen",
        ]
        expected = [0xBEEF if i in (0, 2, 3, 5, 33, 36) else 0x2222 for i in range(64)]
        for emu in (False, True):
            with self.subTest(emulate=emu):
                spv = translate_cs("w64_exec_bitset", lines, wave64=True, emulate=emu)
                mem = run_cs(spv, wave64=True, emulate=emu)
                self.assertEqual([mem[4 * i] for i in range(64)], expected)

    def test_wave32_vcc_lo_and_mask_sgpr_bitset(self):
        spv = translate_cs("w32_vcc_sgpr_bitset", [
            "v_cmp_lt_u32_e32 vcc_lo, v0, 4",
            "s_mov_b32 s10, vcc_lo",
            "s_bitset0_b32 vcc_lo, 1",
            "s_bitset1_b32 vcc_lo, 6",
            "s_bitset0_b32 s10, 3",
            "s_bitset1_b32 s10, 10",
            "v_cndmask_b32_e32 v1, 0, 1, vcc_lo",
            "v_cndmask_b32_e64 v2, 0, 1, s10",
            "s_bcnt1_i32_b32 s11, vcc_lo",
            "s_bcnt1_i32_b32 s12, s10",
            "v_mov_b32_e32 v3, s11",
            "v_mov_b32_e32 v4, s12",
            "v_lshlrev_b32_e32 v6, 2, v0",
            "buffer_store_dword v1, v6, s[0:3], 0 offen",
            "buffer_store_dword v2, v6, s[0:3], 0 offen offset:128",
            "buffer_store_dword v3, v6, s[0:3], 0 offen offset:256",
            "buffer_store_dword v4, v6, s[0:3], 0 offen offset:384",
        ], wave64=False)
        mem = run_cs(spv, wave64=False)
        self.assertEqual([mem[4 * i] for i in range(32)], [1 if i in (0, 2, 3, 6) else 0 for i in range(32)])
        self.assertEqual([mem[128 + 4 * i] for i in range(32)], [1 if i in (0, 1, 2, 10) else 0 for i in range(32)])
        self.assertEqual([mem[256 + 4 * i] for i in range(32)], [4] * 32)
        self.assertEqual([mem[384 + 4 * i] for i in range(32)], [4] * 32)

    def test_wave64_vcc_and_mask_sgpr_halves_bitset(self):
        lines = [
            "v_cmp_lt_u32_e64 vcc, v0, 34",
            "s_bitset0_b32 vcc_lo, 0",
            "s_bitset1_b32 vcc_hi, 5",
            "v_cndmask_b32_e64 v1, 0, 1, vcc",
            "s_bcnt1_i32_b64 s11, vcc",
            "s_mov_b32 s10, exec_hi",
            "s_bitset0_b32 s10, 2",
            "s_bcnt1_i32_b32 s12, s10",
            "v_mov_b32_e32 v2, s11",
            "v_mov_b32_e32 v3, s12",
            "v_lshlrev_b32_e32 v6, 2, v0",
            "buffer_store_dword v1, v6, s[0:3], 0 offen",
            "buffer_store_dword v2, v6, s[0:3], 0 offen offset:256",
            "buffer_store_dword v3, v6, s[0:3], 0 offen offset:512",
        ]
        want_vcc = [1 if (1 <= i <= 33 or i == 37) else 0 for i in range(64)]
        for emu in (False, True):
            with self.subTest(emulate=emu):
                spv = translate_cs("w64_vcc_sgpr_bitset", lines, wave64=True, emulate=emu)
                mem = run_cs(spv, wave64=True, emulate=emu)
                self.assertEqual([mem[4 * i] for i in range(64)], want_vcc)
                self.assertEqual([mem[256 + 4 * i] for i in range(64)], [34] * 64)
                self.assertEqual([mem[512 + 4 * i] for i in range(64)], [31] * 64)

    def test_lane_mask_bit_index_source_uses_wave_mask_word(self):
        # vcc_lo has lanes 0 and 2 active -> wave mask word = 0x5 -> bit 5 set (0x20) uniformly.
        spv = translate_cs("w32_mask_bit_index", [
            "s_mov_b32 s10, 0",
            "s_and_b32 vcc_lo, exec_lo, 0x05",
            "s_bitset1_b32 s10, vcc_lo",
            "v_mov_b32_e32 v1, s10",
            "v_lshlrev_b32_e32 v6, 2, v0",
            "buffer_store_dword v1, v6, s[0:3], 0 offen",
        ], wave64=False)
        mem = run_cs(spv, wave64=False)
        self.assertEqual([mem[4 * i] for i in range(32)], [1 << 5] * 32)


if __name__ == "__main__":
    unittest.main()
