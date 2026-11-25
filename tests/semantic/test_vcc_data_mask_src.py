"""Semantic tests for reading plain scalar data in VCC (vcc / vcc_lo / vcc_hi) as a lane mask.

When VCC is written as plain scalar data (e.g. `s_mov_b32 vcc_lo, K` / `s_mov_b32 vcc_hi, K` or from
SGPRs), `tools/lane_masks.py` leaves VCC uncertified (not a per-lane 0 / ~0 mask). Reading it through
`Gen.mask_src()` (v_cndmask_b32, v_add_co_ci_u32, EXEC writes, 64-bit scalar mask ops) must extract
bit `lane & 31` of `%vcc` for lanes 0..31 and (in wave64 64-bit mask reads) of `%vcc_hi` for lanes
32..63, and wave64 32-bit `vcc_hi` reads must extract bit `lane & 31` of `%vcc_hi`.

Certified wave64 lane-mask writes to `vcc` (`mask_dst` / `mov64` / `put_mask` high half) also mirror
this lane's 0 / ~0 bit into `%vcc_hi`, so a control-flow join between a certified `v_cmp` mask and a
plain-data `vcc_lo`/`vcc_hi` write stays exact on all 64 lanes on both incoming paths.

Standard library only; no LLVM, no GPU, no game files.
"""
import gc
import os
import struct
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
for p in (HERE, TOOLS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import rdna2spirv  # noqa: E402
import spirv_asm  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

M32 = 0xFFFFFFFF
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
VSHARP = [0, 0, 0xFFFFFFFF, 0x31016FAC]


def _fssm_blob(wave_size: int) -> bytes:
    blob = bytearray(0x138)
    blob[0:4] = b"FSSM"
    if wave_size == 32:
        struct.pack_into("<I", blob, 0xD4, 0x20)
    return bytes(blob) + struct.pack("<4I", 0x8090, wave_size, 1, 1)


def _translate_cs(name: str, lines, wave_size: int, *, emulate: bool = False) -> str:
    prev_emu = rdna2spirv.WAVE64_EMULATE
    try:
        rdna2spirv.WAVE64_EMULATE = emulate
        with tempfile.TemporaryDirectory() as tmp:
            os.mkdir(os.path.join(tmp, "disasm"))
            os.mkdir(os.path.join(tmp, "fssm"))
            src = os.path.join(tmp, "disasm", name + ".s")
            with open(src, "w", encoding="utf-8", newline="\n") as f:
                f.write(f"; {name} synthetic\n")
                for item in lines:
                    text, size = item if isinstance(item, tuple) else (item, 4)
                    f.write(f"\t{text} ; encoding: [{','.join(['0x00'] * size)}]\n")
            with open(os.path.join(tmp, name + ".dxbc"), "wb") as f:
                f.write(STUB_DXBC)
            with open(os.path.join(tmp, "fssm", name + ".fssm"), "wb") as f:
                f.write(_fssm_blob(wave_size))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                spv = rdna2spirv.translate(src)
                gc.collect()
            assert rdna2spirv.translate.last_gen.wave64 == (wave_size == 64)
            binary = spirv_asm.assemble(spv)
            assert len(binary) >= 20 and binary[:4] == b"\x03\x02\x23\x07"
            return spv
    finally:
        rdna2spirv.WAVE64_EMULATE = prev_emu


def _run_cs(spv: str, wave_size: int, *, s4: int = 0, s5: int = 0, s6: int = 0, emulate: bool = False):
    pc = VSHARP + [s4 & M32, s5 & M32, s6 & M32, 0] + [0] * 24
    sg_size = 32 if emulate else wave_size
    mem = si.run_workgroup(
        spv, wave_size, sg_size,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i % sg_size},
        push_constants=pc, memory={},
    )
    return [mem[4 * l] for l in range(wave_size)]


class Wave32VccDataMaskTests(unittest.TestCase):
    def test_vcc_lo_constant_and_sgpr_cndmask_extracts_lane_bit(self):
        spv_lit = _translate_cs("cs_w32_vcc_lit_cndmask", [
            ("s_mov_b32 vcc_lo, 0x80000002", 8),
            ("s_mov_b32 vcc_hi, 0xffffffff", 8),
            ("v_cndmask_b32_e64 v6, 3, 5, vcc_lo", 8),
            ("v_lshlrev_b32_e32 v5, 2, v0", 4),
            ("buffer_store_dword v6, v5, s[0:3], 0 offen", 8),
            ("s_endpgm", 4),
        ], 32)
        want_lit = [5 if (0x80000002 >> l) & 1 else 3 for l in range(32)]
        self.assertEqual(_run_cs(spv_lit, 32), want_lit)

        spv_sgpr = _translate_cs("cs_w32_vcc_sgpr_cndmask", [
            ("s_mov_b32 vcc_lo, s4", 4),
            ("s_mov_b32 vcc_hi, s5", 4),
            ("v_cndmask_b32_e64 v6, 10, 20, vcc_lo", 8),
            ("v_lshlrev_b32_e32 v5, 2, v0", 4),
            ("buffer_store_dword v6, v5, s[0:3], 0 offen", 8),
            ("s_endpgm", 4),
        ], 32)
        for lo in (0, 1, 2, 0xA55A0005, 0x80000000, M32):
            with self.subTest(lo=hex(lo)):
                want = [20 if (lo >> l) & 1 else 10 for l in range(32)]
                self.assertEqual(_run_cs(spv_sgpr, 32, s4=lo, s5=M32 ^ lo), want)

    def test_vcc_lo_plain_data_carry_in_and_exec_write(self):
        spv_ci = _translate_cs("cs_w32_vcc_ci", [
            ("s_mov_b32 vcc_lo, s4", 4),
            ("v_mov_b32_e32 v1, 100", 4),
            ("v_mov_b32_e32 v2, 7", 4),
            ("v_add_co_ci_u32_e64 v6, s8, v1, v2, vcc_lo", 8),
            ("v_lshlrev_b32_e32 v5, 2, v0", 4),
            ("buffer_store_dword v6, v5, s[0:3], 0 offen", 8),
            ("s_endpgm", 4),
        ], 32)
        mask = 0x5000000A
        self.assertEqual(_run_cs(spv_ci, 32, s4=mask), [107 + ((mask >> l) & 1) for l in range(32)])

        spv_exec = _translate_cs("cs_w32_vcc_exec", [
            ("v_mov_b32_e32 v6, 1", 4),
            ("s_mov_b32 vcc_lo, s4", 4),
            ("s_and_saveexec_b32 s8, vcc_lo", 4),
            ("v_mov_b32_e32 v6, 9", 4),
            ("s_mov_b32 exec_lo, s8", 4),
            ("v_lshlrev_b32_e32 v5, 2, v0", 4),
            ("buffer_store_dword v6, v5, s[0:3], 0 offen", 8),
            ("s_endpgm", 4),
        ], 32)
        self.assertEqual(_run_cs(spv_exec, 32, s4=mask), [9 if (mask >> l) & 1 else 1 for l in range(32)])


class Wave64VccDataMaskTests(unittest.TestCase):
    WAVE64_MARK = [("s_and_b64 s[12:13], exec, exec", 4)]

    def test_vcc_lo_and_vcc_hi_plain_data_cndmask_and_carry_in(self):
        spv_cnd = _translate_cs("cs_w64_vcc_data_cndmask", self.WAVE64_MARK + [
            ("s_mov_b32 vcc_lo, s4", 4),
            ("s_mov_b32 vcc_hi, s5", 4),
            ("v_cndmask_b32_e64 v6, 3, 5, vcc", 8),
            ("v_lshlrev_b32_e32 v5, 2, v0", 4),
            ("buffer_store_dword v6, v5, s[0:3], 0 offen", 8),
            ("s_endpgm", 4),
        ], 64)
        spv_cnd_emu = _translate_cs("cs_w64_vcc_data_cndmask_emu", self.WAVE64_MARK + [
            ("s_mov_b32 vcc_lo, s4", 4),
            ("s_mov_b32 vcc_hi, s5", 4),
            ("v_cndmask_b32_e64 v6, 3, 5, vcc", 8),
            ("v_lshlrev_b32_e32 v5, 2, v0", 4),
            ("buffer_store_dword v6, v5, s[0:3], 0 offen", 8),
            ("s_endpgm", 4),
        ], 64, emulate=True)
        for lo, hi in (
            (0x00000002, 0x00000001),
            (0x00000000, 0x80000001),
            (0x80000001, 0x00000000),
            (0xA55A0005, 0x12348001),
            (0, 0),
            (M32, M32),
        ):
            bits = lo | (hi << 32)
            want = [5 if (bits >> l) & 1 else 3 for l in range(64)]
            with self.subTest(lo=hex(lo), hi=hex(hi)):
                self.assertEqual(_run_cs(spv_cnd, 64, s4=lo, s5=hi), want)
                self.assertEqual(_run_cs(spv_cnd_emu, 64, s4=lo, s5=hi, emulate=True), want)

        spv_ci = _translate_cs("cs_w64_vcc_data_ci", self.WAVE64_MARK + [
            ("s_mov_b32 vcc_lo, s4", 4),
            ("s_mov_b32 vcc_hi, s5", 4),
            ("v_mov_b32_e32 v1, 50", 4),
            ("v_mov_b32_e32 v2, 4", 4),
            ("v_add_co_ci_u32_e64 v6, s[8:9], v1, v2, vcc", 8),
            ("v_lshlrev_b32_e32 v5, 2, v0", 4),
            ("buffer_store_dword v6, v5, s[0:3], 0 offen", 8),
            ("s_endpgm", 4),
        ], 64)
        lo, hi = 0x80000004, 0x00010002
        bits = lo | (hi << 32)
        self.assertEqual(_run_cs(spv_ci, 64, s4=lo, s5=hi), [54 + ((bits >> l) & 1) for l in range(64)])

    def test_wave64_vcc_hi_plain_data_half_mask_read(self):
        spv = _translate_cs("cs_w64_vcc_hi_data_exec_hi", self.WAVE64_MARK + [
            ("v_mov_b32_e32 v6, 1", 4),
            ("s_mov_b32 vcc_lo, 0", 4),
            ("s_mov_b32 vcc_hi, s5", 4),
            ("s_mov_b64 s[8:9], exec", 4),
            ("s_and_b32 exec_hi, exec_hi, vcc_hi", 4),
            ("v_mov_b32_e32 v6, 9", 4),
            ("s_mov_b64 exec, s[8:9]", 4),
            ("v_lshlrev_b32_e32 v5, 2, v0", 4),
            ("buffer_store_dword v6, v5, s[0:3], 0 offen", 8),
            ("s_endpgm", 4),
        ], 64)
        hi = 0x80000003
        want = [9] * 32 + [9 if (hi >> (l - 32)) & 1 else 1 for l in range(32, 64)]
        self.assertEqual(_run_cs(spv, 64, s5=hi), want)

    def test_wave64_mov_b64_vcc_from_data_sgpr_pair(self):
        # s_mov_b64 vcc, s[4:5] with uncertified data SGPR pair leaves vcc uncertified in lane_masks;
        # both v_cndmask_b32 and s_cbranch_vccz must still see all 64 lanes accurately.
        spv = _translate_cs("cs_w64_mov64_vcc_data", self.WAVE64_MARK + [
            ("s_mov_b32 vcc_hi, 0xffffffff", 8),  # poison %vcc_hi before s_mov_b64
            ("s_mov_b64 vcc, s[4:5]", 4),
            ("v_cndmask_b32_e64 v6, 3, 5, vcc", 8),
            ("v_lshlrev_b32_e32 v5, 2, v0", 4),
            ("buffer_store_dword v6, v5, s[0:3], 0 offen", 8),
            ("s_endpgm", 4),
        ], 64)
        for lo, hi in ((0, 0), (0, 0x80000001), (0x00000002, 0), (0x12340001, 0x80005678)):
            bits = lo | (hi << 32)
            want = [5 if (bits >> l) & 1 else 3 for l in range(64)]
            with self.subTest(lo=hex(lo), hi=hex(hi)):
                self.assertEqual(_run_cs(spv, 64, s4=lo, s5=hi), want)

    def test_wave64_join_between_certified_mask_and_plain_data_vcc(self):
        # If s6 != 0: vcc = v_cmp_lt_u32(v0, 35) (certified mask).
        # If s6 == 0: vcc_lo = s4, vcc_hi = s5 (plain scalar data).
        # At the join block, lane_masks drops vcc; mask_src must produce exact bits on both paths.
        spv = _translate_cs("cs_w64_vcc_join", [
            ("s_and_b64 s[12:13], exec, exec", 4),
            ("s_cmp_eq_u32 s6, 0", 4),
            ("s_cbranch_scc1 3", 4),
            ("v_cmp_lt_u32_e64 vcc, v0, 35", 8),
            ("s_branch 2", 4),
            ("s_mov_b32 vcc_lo, s4", 4),
            ("s_mov_b32 vcc_hi, s5", 4),
            ("v_cndmask_b32_e64 v6, 3, 5, vcc", 8),
            ("v_lshlrev_b32_e32 v5, 2, v0", 4),
            ("buffer_store_dword v6, v5, s[0:3], 0 offen", 8),
            ("s_endpgm", 4),
        ], 64)
        want_cmp = [5 if l < 35 else 3 for l in range(64)]
        self.assertEqual(_run_cs(spv, 64, s4=0, s5=0, s6=1), want_cmp)

        lo, hi = 0x00000005, 0x80000002
        bits = lo | (hi << 32)
        want_data = [5 if (bits >> l) & 1 else 3 for l in range(64)]
        self.assertEqual(_run_cs(spv, 64, s4=lo, s5=hi, s6=0), want_data)


if __name__ == "__main__":
    unittest.main()
