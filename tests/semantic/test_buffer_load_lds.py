"""Semantic tests for MUBUF loads with the `lds` modifier (RDNA2 ISA 8.1.9 "Memory Buffer Load to LDS").

Covers buffer_load_dword / buffer_load_{u,s}byte / buffer_load_{u,s}hort with `lds`: the fetched value is
written to LDS at LDS_BASE + M0[15:0] + inst_offset + TIDinWave*4 (bytes) instead of a VGPR, sub-dword loads
store the zero/sign-extended dword, an out-of-range buffer read writes 0, inactive lanes write nothing, and an
address outside the wave's LDS allocation drops the write. The three-operand llvm-mc spelling (no VDATA) and
the rejected forms (dwordxN, tfe, format loads) are checked too.

Standard library only; no GPU, no LLVM, no game files.
"""
import struct
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from support import rdna2spirv, translate_lines  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

# RSRC2 0x40088: 4 user SGPRs, workgroup id in s4, LDS_SIZE = 8 x 512 bytes = 4 KiB (1024 dwords).
FSSM = bytearray(0x148)
struct.pack_into("<I", FSSM, 0xD4, 0x20)          # wave32
struct.pack_into("<4I", FSSM, 0x138, 0x40088, 32, 1, 1)
FSSM = bytes(FSSM)
BASE = 0x1000          # input V# base: heap byte address
OUT = 0x8000           # output words through global_store_dword
BYTES = bytes(range(0x40, 0x40 + 64))
MEMORY = {BASE + i * 4: struct.unpack_from("<I", BYTES, i * 4)[0] for i in range(16)}
# V# word3: OOB_SELECT 3 (RAW), the identity DST_SELs, GFX10_FORMAT_32_UINT (ignored by untyped loads).
W3 = (3 << 28) | (4 | (5 << 3) | (6 << 6) | (7 << 9)) | (20 << 12)


def vsharp(num_records=0x1000):
    return [f"s_mov_b32 s4, {BASE:#x}", "s_mov_b32 s5, 0", f"s_mov_b32 s6, {num_records:#x}",
            f"s_mov_b32 s7, {W3:#x}"]


def store_out(reg, slot):
    """global_store_dword reg -> OUT + lane*4 + slot*4 (slot a Python int)."""
    return ["v_lshlrev_b32_e32 v30, 2, v0",
            f"v_add_nc_u32_e32 v30, {OUT + slot * 4:#x}, v30",
            f"global_store_dword v30, {reg}, s[0:1]"]


def read_lds(byte_vreg, out_reg):
    """ds_read_b32 the dword at VGPR byte address `byte_vreg` into out_reg."""
    return [f"ds_read_b32 {out_reg}, {byte_vreg}", "s_waitcnt lgkmcnt(0)"]


def lane_slot(m0, extra=0):
    """VGPR v20 = m0 + lane*4 + extra: this lane's LDS dword address for a load lds with offset:extra."""
    return [f"v_lshlrev_b32_e32 v20, 2, v0",
            f"v_add_nc_u32_e32 v20, {m0 + extra:#x}, v20"]


def run(body, memory=None, nlanes=8):
    spv = translate_lines("cs_buflds", list(body) + ["s_endpgm"], fssm=FSSM)
    mem = dict(MEMORY if memory is None else memory)
    si.run_workgroup(spv, nlanes, 32,
                     inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
                     push_constants=[0] * 36, memory=mem)
    return mem


class BufferLoadLdsTests(unittest.TestCase):
    def test_dword_writes_lds_per_lane(self):
        # m0 = 64: lane i's load of buffer dword i lands at LDS byte 64 + 4*i; ds_read_addtid reads it back.
        body = vsharp() + [
            "v_lshlrev_b32_e32 v1, 2, v0",          # v1 = lane * 4: the buffer byte offset
            "s_mov_b32 m0, 64",
            "buffer_load_dword v9, v1, s[4:7], 0 offen lds",
            "s_waitcnt vmcnt(0)",
            "s_barrier",
            "ds_read_addtid_b32 v2 offset:0",        # LDS[64 + lane*4]
            "s_waitcnt lgkmcnt(0)",
        ] + store_out("v2", 0)
        mem = run(body)
        for lane in range(8):
            self.assertEqual(mem.get(OUT + lane * 4), MEMORY[BASE + lane * 4], f"lane {lane}")

    def test_inst_offset_and_m0_high_bits(self):
        # offset:16 adds 16 to both the buffer and the LDS address; M0's high bits are ignored.
        body = vsharp() + [
            "v_lshlrev_b32_e32 v1, 2, v0",
            "s_mov_b32 m0, 0x12340040",               # M0[15:0] = 0x40
            "buffer_load_dword v9, v1, s[4:7], 0 offen offset:16 lds",
            "s_waitcnt vmcnt(0)",
            "s_barrier",
        ] + lane_slot(0x40, 16) + read_lds("v20", "v2") + store_out("v2", 1)
        mem = run(body)
        for lane in range(8):
            self.assertEqual(mem.get(OUT + 4 + lane * 4), MEMORY[BASE + 16 + lane * 4], f"lane {lane}")

    def test_llvm_mc_three_operand_form(self):
        # llvm-mc prints no VDATA operand for the lds forms: vaddr, srsrc, soffset.
        body = vsharp() + [
            "v_lshlrev_b32_e32 v1, 2, v0",
            "s_mov_b32 m0, 64",
            "buffer_load_dword v1, s[4:7], 0 offen lds",
            "s_waitcnt vmcnt(0)",
            "s_barrier",
        ] + lane_slot(64) + read_lds("v20", "v2") + store_out("v2", 2)
        mem = run(body)
        for lane in range(8):
            self.assertEqual(mem.get(OUT + 8 + lane * 4), MEMORY[BASE + lane * 4], f"lane {lane}")

    def test_subdword_loads_store_extended_dwords(self):
        # Byte offset 5 of the input dword: ubyte 0x45, sbyte 0x45; ushort at 6: 0x4746, sshort positive.
        cases = [("buffer_load_ubyte", 5, 0x45), ("buffer_load_sbyte", 5, 0x45),
                 ("buffer_load_ushort", 6, 0x4746), ("buffer_load_sshort", 6, 0x4746)]
        for slot, (op, off, want) in enumerate(cases):
            with self.subTest(op=op):
                body = vsharp() + [
                    f"v_mov_b32_e32 v1, {off}",
                    "s_mov_b32 m0, 64",
                    f"{op} v9, v1, s[4:7], 0 offen lds",
                    "s_waitcnt vmcnt(0)",
                    "s_barrier",
                ] + lane_slot(64) + read_lds("v20", "v2") + store_out("v2", slot)
                mem = run(body, nlanes=1)
                self.assertEqual(mem.get(OUT + slot * 4), want)

    def test_subdword_sign_extension(self):
        # BASE dword = 0x0145807F (bytes 7F 80 45 01), BASE+4 = 0x123480FF (bytes FF 80 34 12).
        mem_in = dict(MEMORY)
        mem_in[BASE] = 0x0145807F
        mem_in[BASE + 4] = 0x123480FF
        cases = [("buffer_load_ubyte", 1, 0x80), ("buffer_load_sbyte", 1, 0xFFFFFF80),
                 ("buffer_load_ushort", 1, 0x4580), ("buffer_load_sshort", 1, 0x4580),
                 ("buffer_load_ushort", 4, 0x80FF), ("buffer_load_sshort", 4, 0xFFFF80FF),
                 ("buffer_load_ubyte", 4, 0xFF), ("buffer_load_sbyte", 4, 0xFFFFFFFF)]
        for slot, (op, off, want) in enumerate(cases):
            with self.subTest(op=op, off=off):
                body = vsharp() + [
                    f"v_mov_b32_e32 v1, {off}",
                    "s_mov_b32 m0, 64",
                    f"{op} v9, v1, s[4:7], 0 offen lds",
                    "s_waitcnt vmcnt(0)",
                    "s_barrier",
                ] + lane_slot(64) + read_lds("v20", "v2") + store_out("v2", slot)
                mem = run(body, memory=mem_in, nlanes=1)
                self.assertEqual(mem.get(OUT + slot * 4), want)

    def test_out_of_range_buffer_read_writes_zero(self):
        # num_records 8: a dword load at byte offset 8 ends at 12 > 8 -> the lane writes 0 to LDS.
        body = vsharp(num_records=8) + [
            "v_mov_b32_e32 v1, 8",
            "s_mov_b32 m0, 64",
            "buffer_load_dword v9, v1, s[4:7], 0 offen lds",
            "s_waitcnt vmcnt(0)",
            "s_barrier",
        ] + lane_slot(64) + read_lds("v20", "v2") + store_out("v2", 0)
        mem = run(body, nlanes=1)
        self.assertEqual(mem.get(OUT), 0)

    def test_inactive_lane_writes_nothing(self):
        # Lane 0 only is active for the load; lane 1's LDS slot keeps the sentinel ds_write left.
        body = vsharp() + [
            # sentinel 0xdeadbeef at every lane's slot (m0 = 64)
            "v_lshlrev_b32_e32 v10, 2, v0",
            "v_add_nc_u32_e32 v10, 64, v10",
            "v_mov_b32_e32 v11, 0xdeadbeef",
            "ds_write_b32 v10, v11",
            "s_waitcnt lgkmcnt(0)",
            "v_lshlrev_b32_e32 v1, 2, v0",
            "s_mov_b32 m0, 64",
            "v_cmp_eq_u32_e32 vcc_lo, 0, v0",       # only lane 0 passes
            "s_and_saveexec_b32 s8, vcc_lo",
            "buffer_load_dword v9, v1, s[4:7], 0 offen lds",
            "s_mov_b32 exec_lo, s8",
            "s_waitcnt vmcnt(0)",
            "s_barrier",
        ] + lane_slot(64) + read_lds("v20", "v2") + store_out("v2", 0)
        mem = run(body, nlanes=2)
        self.assertEqual(mem.get(OUT), MEMORY[BASE])                 # lane 0 loaded
        self.assertEqual(mem.get(OUT + 4), 0xDEADBEEF)               # lane 1 kept the sentinel

    def test_lds_address_outside_allocation_is_dropped(self):
        # M0[15:0] = 0xFFF0: every lane's LDS dword index is past the 4 KiB allocation, so the write is
        # dropped and cannot wrap onto dword 1020 (byte 4080), which holds a sentinel.
        body = vsharp() + [
            "v_mov_b32_e32 v10, 4080",
            "v_mov_b32_e32 v11, 0x12345678",
            "ds_write_b32 v10, v11",
            "s_waitcnt lgkmcnt(0)",
            "v_lshlrev_b32_e32 v1, 2, v0",
            "s_mov_b32 m0, 0xFFF0",
            "buffer_load_dword v9, v1, s[4:7], 0 offen lds",
            "s_waitcnt vmcnt(0)",
            "s_barrier",
            "v_mov_b32_e32 v20, 4080",
        ] + read_lds("v20", "v2") + store_out("v2", 0)
        mem = run(body, nlanes=1)
        self.assertEqual(mem.get(OUT), 0x12345678)

    def test_rejected_forms(self):
        for line in ("buffer_load_dwordx2 v9, v1, s[4:7], 0 offen lds",
                     "buffer_load_dword v9, v1, s[4:7], 0 offen tfe lds",
                     "buffer_load_format_x v9, v1, s[4:7], 0 offen lds",
                     "buffer_load_ubyte v9, v1, s[4:7], 0 offen tfe lds"):
            with self.subTest(line=line):
                with self.assertRaises(rdna2spirv.Unsupported):
                    translate_lines("cs_buflds", vsharp() + ["v_mov_b32_e32 v1, 0", "s_mov_b32 m0, 64", line,
                                                             "s_endpgm"], fssm=FSSM)

    def test_spirv_stores_to_workgroup_memory(self):
        spv = translate_lines("cs_buflds", vsharp() + [
            "v_mov_b32_e32 v1, 0", "s_mov_b32 m0, 64",
            "buffer_load_dword v9, v1, s[4:7], 0 offen lds", "s_endpgm"], fssm=FSSM)
        self.assertIn("%lds", spv)
        self.assertIn("Workgroup", spv)


if __name__ == "__main__":
    unittest.main()
