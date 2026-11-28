"""Semantic interpreter tests for tfe on MUBUF/MTBUF loads.

Covers the RDNA2 ISA rule implemented in tools/rdna2spirv.py (docs/design.md, "Resident buffer_load* tfe"):
with tfe the VDATA range holds one status dword after the data VGPRs. The ISA writes it only when the fetch
NACKs (PRT, "Partially Resident Textures"); a Vulkan buffer range cannot NACK, so — exactly like the resident
image_load tfe case — the status VGPR keeps its prior value and the ordinary load lowering covers it.

  - buffer_load_dword / dwordx2, buffer_load_ushort and tbuffer_load_format_x with tfe: the data VGPRs get the
    loaded data and the status VGPR (vdst[data_count]) keeps the value the shader seeded.
  - tfe stays refused on stores (buffer_store_dword, buffer_store_byte) and atomics (buffer_atomic_add): the ISA
    defines no status dword for them and llvm-mc rejects the modifier there.

Standard library only; no GPU, no LLVM, no game files.
"""
from __future__ import annotations

import os
import struct
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.dont_write_bytecode = True
sys.path.insert(0, os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools"))
sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
FSSM = bytearray(0x148)
struct.pack_into("<4I", FSSM, 0x138, 0x88, 32, 1, 1)
FSSM = bytes(FSSM)
BASE = 0x1000
OUT = 0x3000
# Dwords at BASE..BASE+12 and the ushort at BASE+2 they imply.
MEMORY = {
    BASE: 0x11111111,
    BASE + 4: 0x22222222,
    BASE + 8: 0x33333333,
    BASE + 12: 0x44444444,
}


def translate(body):
    with tempfile.TemporaryDirectory() as tmp:
        for d in ("disasm", "fssm"):
            os.mkdir(os.path.join(tmp, d))
        with open(os.path.join(tmp, "cs_buftfe.dxbc"), "wb") as f:
            f.write(STUB_DXBC)
        with open(os.path.join(tmp, "fssm", "cs_buftfe.fssm"), "wb") as f:
            f.write(FSSM)
        src = os.path.join(tmp, "disasm", "cs_buftfe.s")
        with open(src, "w", encoding="utf-8") as f:
            f.write("; cs_buftfe\n" + "".join(f"\t{ln}\n" for ln in body) + "\ts_endpgm\n")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            return rdna2spirv.translate(src)


def run_cs(body, memory=None):
    spv = translate(body)
    mem = dict(MEMORY if memory is None else memory)
    si.run_workgroup(spv, 1, 32,
                     inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
                     push_constants=[0] * 36, memory=mem)
    return mem


def tfe_program(loads, results):
    """Seed the heap V# (s4..s7: base BASE, stride 0, NUM_RECORDS 0x40, OOB_SELECT 3) and the status VGPRs, run
    `loads`, then store each result VGPR in `results` to OUT+4*i."""
    body = [
        f"s_mov_b32 s4, {BASE:#x}", "s_mov_b32 s5, 0", "s_mov_b32 s6, 0x40", "s_mov_b32 s7, 0x31016fac",
        "v_mov_b32_e32 v1, 0",
    ]
    for reg, seed in ((5, 0x5eed0101), (8, 0x5eed0102), (10, 0x5eed0103), (12, 0x5eed0104), (14, 0x5eed0105)):
        body.append(f"v_mov_b32_e32 v{reg}, {seed:#x}")
    body += loads + ["s_waitcnt vmcnt(0)"]
    for i, reg in enumerate(results):
        body += [f"v_mov_b32_e32 v30, {OUT + 4 * i:#x}", f"global_store_dword v30, v{reg}, s[0:1]"]
    return body


class BufferLoadTfeTests(unittest.TestCase):
    def test_dword_load_status_vgpr_preserved(self):
        # buffer_load_dword v[4:5] ... tfe: v4 gets the dword, v5 is the status VGPR and keeps its seed.
        body = tfe_program(
            ["buffer_load_dword v[4:5], v1, s[4:7], 0 offen tfe"], [4, 5])
        mem = run_cs(body)
        self.assertEqual(mem.get(OUT), 0x11111111)
        self.assertEqual(mem.get(OUT + 4), 0x5EED0101)

    def test_dwordx2_load_status_vgpr_preserved(self):
        # buffer_load_dwordx2 v[6:8] ... tfe: v6/v7 get two dwords (inst_offset 4), v8 keeps its seed.
        body = tfe_program(
            ["buffer_load_dwordx2 v[6:8], v1, s[4:7], 0 offen tfe offset:4"], [6, 7, 8])
        mem = run_cs(body)
        self.assertEqual(mem.get(OUT), 0x22222222)
        self.assertEqual(mem.get(OUT + 4), 0x33333333)
        self.assertEqual(mem.get(OUT + 8), 0x5EED0102)

    def test_sub_dword_load_status_vgpr_preserved(self):
        # buffer_load_ushort v[9:10] ... tfe: v1 = 2, so the ushort at BASE+2; v10 keeps its seed.
        body = tfe_program(
            ["v_mov_b32_e32 v1, 2", "buffer_load_ushort v[9:10], v1, s[4:7], 0 offen tfe"], [9, 10])
        mem = run_cs(body)
        self.assertEqual(mem.get(OUT), 0x1111)
        self.assertEqual(mem.get(OUT + 4), 0x5EED0103)

    def test_d16_load_status_vgpr_preserved(self):
        # buffer_load_ubyte_d16 v[13:14] ... tfe: the byte zero-extends into the low half of v13 (the high half keeps
        # its seed, as a d16 load does), and the status VGPR v14 keeps its seed.
        body = tfe_program(
            ["v_mov_b32_e32 v13, 0xAAAA5555",
             "buffer_load_ubyte_d16 v[13:14], v1, s[4:7], 0 offen tfe"], [13, 14])
        mem = run_cs(body)
        self.assertEqual(mem.get(OUT), 0xAAAA0011)
        self.assertEqual(mem.get(OUT + 4), 0x5EED0105)

    def test_tbuffer_format_load_status_vgpr_preserved(self):
        # tbuffer_load_format_x v[11:12] ... tfe with BUF_FMT_32_UINT: v11 gets the raw dword, v12 keeps its seed.
        body = tfe_program(
            ["tbuffer_load_format_x v[11:12], v1, s[4:7], 0 format:[BUF_FMT_32_UINT] offen tfe"], [11, 12])
        mem = run_cs(body)
        self.assertEqual(mem.get(OUT), 0x11111111)
        self.assertEqual(mem.get(OUT + 4), 0x5EED0104)

    def test_tfe_still_refused_on_stores_and_atomics(self):
        for line in ("buffer_store_dword v2, v1, s[4:7], 0 offen tfe",
                     "buffer_store_dwordx2 v[2:3], v1, s[4:7], 0 offen tfe",
                     "buffer_store_byte v2, v1, s[4:7], 0 offen tfe",
                     "buffer_store_short_d16_hi v2, v1, s[4:7], 0 offen tfe",
                     "buffer_atomic_add v2, v1, s[4:7], 0 offen tfe",
                     "buffer_atomic_add v2, v1, s[4:7], 0 offen glc tfe"):
            with self.subTest(line=line):
                body = ["v_mov_b32_e32 v1, 0", "v_mov_b32_e32 v2, 0", line]
                with self.assertRaises(rdna2spirv.Unsupported):
                    translate(body)


if __name__ == "__main__":
    unittest.main()
