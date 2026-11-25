"""Semantic tests for s_wqm_b32 / s_wqm_b64 plain-data VCC sources and non-mask special destinations.

Verifies:
  1. s_wqm_b64 with plain 64-bit scalar data in VCC reads both %vcc and %vcc_hi (and null reads (0, 0)).
  2. s_wqm_b32 into m0 stores the full 32-bit WQM result rather than a single lane's mask bit.
  3. s_wqm_b32 / s_wqm_b64 into plain-data vcc_lo / vcc_hi / vcc stores the full 32/64-bit WQM words.
  4. s_wqm_b32 / s_wqm_b64 into null discards the result without requiring subgroup lane extraction while
     still updating SCC.
  5. Controls: s_wqm_b64 into exec and immediate-sourced s_wqm_b64 into vcc still write per-lane mask bits.
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
import spirv_mini_interp as si  # noqa: E402

M32 = 0xFFFFFFFF
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
VSHARP = [0, 0, 0xFFFFFFFF, 0x31016FAC]


def ref_wqm32(x):
    x &= M32
    out = 0
    for nib in range(8):
        sh = nib * 4
        if (x >> sh) & 0xF:
            out |= 0xF << sh
    return out


def _translate_cs(name, lines, *, wave32=False, emulate=False):
    with tempfile.TemporaryDirectory() as tmp:
        os.mkdir(os.path.join(tmp, "disasm"))
        os.mkdir(os.path.join(tmp, "fssm"))
        src = os.path.join(tmp, "disasm", name + ".s")
        with open(src, "w", encoding="utf-8", newline="\n") as f:
            f.write(f"; {name} synthetic\n")
            for text in lines:
                f.write(f"\t{text} ; encoding: [0x00,0x00,0x00,0x00]\n")
        with open(os.path.join(tmp, name + ".dxbc"), "wb") as f:
            f.write(STUB_DXBC)
        hdr = bytearray(0x148)
        hdr[0:4] = b"FSSM"
        struct.pack_into("<I", hdr, 0xD4, 0x20 if wave32 else 0)
        struct.pack_into("<4I", hdr, 0x138, 0x8020, 32 if wave32 else 64, 1, 1)
        with open(os.path.join(tmp, "fssm", name + ".fssm"), "wb") as f:
            f.write(hdr)
        prev = rdna2spirv.WAVE64_EMULATE
        rdna2spirv.WAVE64_EMULATE = emulate
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                spv = rdna2spirv.translate(src)
                gc.collect()
        finally:
            rdna2spirv.WAVE64_EMULATE = prev
        return spv


def _run_cs(spv, lo, hi, *, wave32=False, emulate=False, n_words=4):
    n_lanes = 32 if wave32 else 64
    sg_size = 32 if (wave32 or emulate) else 64
    pc = VSHARP + [lo & M32, hi & M32, 0, 0] + [0] * 24
    mem = si.run_workgroup(
        spv, n_lanes, sg_size,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i % sg_size},
        push_constants=pc, memory={},
    )
    return [mem[4 * i] for i in range(n_words)]


def _store_words(sregs):
    out = ["v_mov_b32_e32 v2, 0"]
    for i, sr in enumerate(sregs):
        out.append(f"v_mov_b32_e32 v1, {sr}")
        out.append(f"buffer_store_dword v1, v2, s[0:3], 0 offen offset:{4 * i}")
    out.append("s_endpgm")
    return out


class WqmDataVccAndDstTests(unittest.TestCase):
    def test_wqm_b64_reads_plain_vcc_both_words(self):
        lines = [
            "s_mov_b32 vcc_lo, s4",
            "s_mov_b32 vcc_hi, s5",
            "s_wqm_b64 s[6:7], vcc",
            "s_cselect_b32 s8, 1, 0",
            "s_wqm_b64 s[10:11], null",
            "s_cselect_b32 s9, 1, 0",
        ] + _store_words(["s6", "s7", "s8", "s10", "s11", "s9"])
        spv = _translate_cs("cs_w64_wqm_vcc_src", lines, wave32=False)
        for lo, hi in ((0x00000001, 0x00000010), (0x00000000, 0x80000000), (0x00000000, 0x00000000), (0x10204080, 0x08040201)):
            wlo, whi = ref_wqm32(lo), ref_wqm32(hi)
            scc = 1 if (wlo or whi) else 0
            with self.subTest(lo=hex(lo), hi=hex(hi)):
                self.assertEqual(_run_cs(spv, lo, hi, n_words=6), [wlo, whi, scc, 0, 0, 0])

    def test_wqm_b32_into_m0_stores_full_word(self):
        lines = [
            "s_wqm_b32 m0, s4",
            "s_cselect_b32 s6, 1, 0",
            "s_mov_b32 s7, m0",
        ] + _store_words(["s7", "s6"])
        for wave32 in (True, False):
            spv = _translate_cs(f"cs_wqm_m0_{32 if wave32 else 64}", lines, wave32=wave32)
            for val in (0x00000010, 0x00000001, 0x80000000, 0x00000000, 0x10200400):
                want = ref_wqm32(val)
                with self.subTest(wave32=wave32, val=hex(val)):
                    self.assertEqual(_run_cs(spv, val, 0, wave32=wave32, n_words=2), [want, int(want != 0)])

    def test_wqm_into_plain_data_vcc_preserves_words(self):
        # Plain-data SGPR source into vcc_lo / vcc_hi (b32) and vcc (b64): lane_masks treats VCC as plain data,
        # so the full WQM words must land in %vcc and %vcc_hi.
        lines_b32 = [
            "s_wqm_b32 vcc_lo, s4",
            "s_cselect_b32 s6, 1, 0",
            "s_wqm_b32 vcc_hi, s5",
            "s_cselect_b32 s7, 1, 0",
            "s_mov_b32 s8, vcc_lo",
            "s_mov_b32 s9, vcc_hi",
        ] + _store_words(["s8", "s6", "s9", "s7"])
        for wave32 in (True, False):
            spv = _translate_cs(f"cs_wqm_vcc_b32_{32 if wave32 else 64}", lines_b32, wave32=wave32)
            for lo, hi in ((0x00000010, 0x00000100), (0x00000000, 0x80000000), (0x00000001, 0x00000000)):
                wlo, whi = ref_wqm32(lo), ref_wqm32(hi)
                with self.subTest(wave32=wave32, lo=hex(lo), hi=hex(hi)):
                    self.assertEqual(_run_cs(spv, lo, hi, wave32=wave32, n_words=4),
                                     [wlo, int(wlo != 0), whi, int(whi != 0)])

        lines_b64 = [
            "s_wqm_b64 vcc, s[4:5]",
            "s_cselect_b32 s6, 1, 0",
            "s_wqm_b64 s[8:9], vcc",
            "s_mov_b32 s10, vcc_lo",
            "s_mov_b32 s11, vcc_hi",
        ] + _store_words(["s10", "s11", "s6", "s8", "s9"])
        spv64 = _translate_cs("cs_w64_wqm_vcc_b64", lines_b64, wave32=False)
        for lo, hi in ((0x00000010, 0x00000100), (0x00000000, 0x80000000), (0x00000000, 0x00000000)):
            wlo, whi = ref_wqm32(lo), ref_wqm32(hi)
            scc = int(bool(wlo or whi))
            with self.subTest(b64_lo=hex(lo), b64_hi=hex(hi)):
                self.assertEqual(_run_cs(spv64, lo, hi, wave32=False, n_words=5), [wlo, whi, scc, wlo, whi])

    def test_wqm_null_destination_sets_scc_without_subgroup_ops(self):
        lines = [
            "s_wqm_b32 null, s4",
            "s_cselect_b32 s6, 1, 0",
        ] + _store_words(["s6"])
        spv = _translate_cs("cs_w32_wqm_null", lines, wave32=True)
        self.assertNotIn("OpGroupNonUniform", spv)
        self.assertNotIn("SubgroupLocalInvocationId", spv)
        self.assertEqual(_run_cs(spv, 0, 0, wave32=True, n_words=1), [0])
        self.assertEqual(_run_cs(spv, 0x00000020, 0, wave32=True, n_words=1), [1])

    def test_wqm_b64_imm_into_vcc_and_exec_controls_stay_lane_masks(self):
        # Immediate into vcc (lane_masks marks vcc as MASK) and data into exec must still set per-lane mask bits.
        lines = [
            "s_wqm_b64 vcc, 0x10",             # quad 1 (lanes 4..7) active in VCC
            "v_cndmask_b32_e64 v6, 3, 5, vcc",
            "v_lshlrev_b32_e32 v5, 2, v0",
            "buffer_store_dword v6, v5, s[0:3], 0 offen",
            "s_wqm_b64 exec, s[4:5]",          # only quads active in (s4, s5) write offset:256
            "v_mov_b32_e32 v7, 9",
            "buffer_store_dword v7, v5, s[0:3], 0 offen offset:256",
            "s_endpgm",
        ]
        spv = _translate_cs("cs_w64_wqm_mask_ctrl", lines, wave32=False)
        pc = VSHARP + [0x00000001, 0x00000010, 0, 0] + [0] * 24  # quads 0 (lanes 0..3) and 9 (lanes 36..39)
        mem = si.run_workgroup(
            spv, 64, 64,
            inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
            push_constants=pc, memory={},
        )
        vcc_got = [mem[4 * l] for l in range(64)]
        vcc_want = [5 if 4 <= l <= 7 else 3 for l in range(64)]
        self.assertEqual(vcc_got, vcc_want)
        exec_got = [mem.get(256 + 4 * l, 0) for l in range(64)]
        exec_want = [9 if (0 <= l <= 3 or 36 <= l <= 39) else 0 for l in range(64)]
        self.assertEqual(exec_got, exec_want)


if __name__ == "__main__":
    unittest.main()
