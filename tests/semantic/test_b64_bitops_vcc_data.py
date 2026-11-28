"""Semantic tests for plain 64-bit scalar data in VCC (and null) on 64-bit scalar bit-count and bit-scan ops.

Covers s_bcnt1_i32_b64, s_bcnt0_i32_b64, s_ff1_i32_b64, s_ff0_i32_b64, s_flbit_i32_b64, and s_flbit_i32_i64
when VCC holds plain 64-bit scalar data rather than a lane mask, in both wave64 and wave32, plus null sources,
lane-mask-to-data transitions, and a real lane-mask control.
"""
import gc
import json
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


def _translate_cs(name, lines, *, wave32=False, emulate=False):
    with tempfile.TemporaryDirectory() as tmp:
        os.mkdir(os.path.join(tmp, "disasm"))
        src = os.path.join(tmp, "disasm", name + ".s")
        with open(src, "w", encoding="utf-8", newline="\n") as f:
            f.write(f"; {name} synthetic\n")
            for text in lines:
                f.write(f"\t{text} ; encoding: [0x00,0x00,0x00,0x00]\n")
        with open(os.path.join(tmp, name + ".dxbc"), "wb") as f:
            f.write(STUB_DXBC)
        if wave32:
            # Wave32 FSPX compute metadata: allows _b64 scalar ops on plain-data VCC pairs in wave32.
            os.mkdir(os.path.join(tmp, "fspx"))
            with open(os.path.join(tmp, "fspx", name + ".json"), "w", encoding="utf-8", newline="\n") as f:
                json.dump({"rsrc2": 0x8020, "threads": [32, 1, 1], "wave32": True}, f)
        else:
            os.mkdir(os.path.join(tmp, "fssm"))
            hdr = bytearray(0x148)
            hdr[0:4] = b"FSSM"
            struct.pack_into("<I", hdr, 0xD4, 0)
            struct.pack_into("<4I", hdr, 0x138, 0x8020, 64, 1, 1)
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


def _run_cs(spv, lo, hi, *, wave32=False, emulate=False, n_words=8):
    n_lanes = 32 if wave32 else 64
    sg_size = 32 if (wave32 or emulate) else 64
    pc = VSHARP + [lo & M32, hi & M32, 0, 0] + [0] * 24
    mem = si.run_workgroup(
        spv, n_lanes, sg_size,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i % sg_size},
        push_constants=pc, memory={},
    )
    return [mem[4 * i] for i in range(n_words)]


def _ref_b64_ops(lo, hi):
    v = (lo & M32) | ((hi & M32) << 32)
    bcnt1 = v.bit_count()
    bcnt1_scc = 1 if bcnt1 else 0
    bcnt0 = 64 - bcnt1
    bcnt0_scc = 1 if bcnt0 else 0
    ff1 = (v & -v).bit_length() - 1 if v else M32
    inv = (~v) & ((1 << 64) - 1)
    ff0 = (inv & -inv).bit_length() - 1 if inv else M32
    flbit_b64 = (64 - v.bit_length()) if v else M32
    signed_src = inv if (v >> 63) else v
    flbit_i64 = (64 - signed_src.bit_length()) if signed_src else M32
    return [bcnt1, bcnt1_scc, bcnt0, bcnt0_scc, ff1, ff0, flbit_b64, flbit_i64]


def _store_words(sregs):
    out = ["v_mov_b32_e32 v2, 0"]
    for i, sr in enumerate(sregs):
        out.append(f"v_mov_b32_e32 v1, {sr}")
        out.append(f"buffer_store_dword v1, v2, s[0:3], 0 offen offset:{4 * i}")
    out.append("s_endpgm")
    return out


class B64BitopsVccDataTests(unittest.TestCase):
    PAIRS = (
        (0x00000003, 0x0000000F),
        (0x00000000, 0x00000008),
        (0x00000000, 0x80000000),
        (0xFFFFFFFF, 0xFFFFFFFF),
        (0xFFFFFFFF, 0xFFFFFFF7),
        (0x00000000, 0x00100000),
        (0xFFFFFFFF, 0xFFEFFFFF),
        (0x00008000, 0x00000000),
        (0x00000000, 0x00000000),
    )

    def _program_vcc_data(self, prefix=()):
        return list(prefix) + [
            "s_mov_b32 vcc_lo, s4",
            "s_mov_b32 vcc_hi, s5",
            "s_bcnt1_i32_b64 s6, vcc",
            "s_cselect_b32 s7, 1, 0",
            "s_bcnt0_i32_b64 s8, vcc",
            "s_cselect_b32 s9, 1, 0",
            "s_ff1_i32_b64 s10, vcc",
            "s_ff0_i32_b64 s11, vcc",
            "s_flbit_i32_b64 s12, vcc",
            "s_flbit_i32_i64 s13, vcc",
        ] + _store_words(["s6", "s7", "s8", "s9", "s10", "s11", "s12", "s13"])

    def test_plain_vcc_data_wave64(self):
        spv = _translate_cs("cs_w64_b64_vcc_data", self._program_vcc_data(), wave32=False)
        for lo, hi in self.PAIRS:
            with self.subTest(lo=hex(lo), hi=hex(hi)):
                self.assertEqual(_run_cs(spv, lo, hi, wave32=False), _ref_b64_ops(lo, hi))

    def test_bcnt_and_ff_plain_vcc_data_reads_high_word(self):
        # Isolates s_bcnt[01]_i32_b64 and s_ff[01]_i32_b64 (without s_flbit) so a missing %vcc_hi load
        # fails with wrong numeric outputs rather than an earlier translation error.
        lines = [
            "s_mov_b32 vcc_lo, s4",
            "s_mov_b32 vcc_hi, s5",
            "s_bcnt1_i32_b64 s6, vcc",
            "s_cselect_b32 s7, 1, 0",
            "s_bcnt0_i32_b64 s8, vcc",
            "s_cselect_b32 s9, 1, 0",
            "s_ff1_i32_b64 s10, vcc",
            "s_ff0_i32_b64 s11, vcc",
        ] + _store_words(["s6", "s7", "s8", "s9", "s10", "s11"])
        spv = _translate_cs("cs_w64_bcnt_ff_vcc_data", lines, wave32=False)
        for lo, hi in ((0x3, 0xF), (0, 0x8), (M32, M32), (M32, 0xFFFFFFF7)):
            with self.subTest(lo=hex(lo), hi=hex(hi)):
                self.assertEqual(_run_cs(spv, lo, hi, wave32=False, n_words=6), _ref_b64_ops(lo, hi)[:6])

    def test_plain_vcc_data_wave64_emulate(self):
        spv = _translate_cs("cs_w64e_b64_vcc_data", self._program_vcc_data(), wave32=False, emulate=True)
        for lo, hi in ((0x3, 0xF), (0, 0x8), (M32, M32), (0, 0)):
            with self.subTest(lo=hex(lo), hi=hex(hi)):
                self.assertEqual(_run_cs(spv, lo, hi, wave32=False, emulate=True), _ref_b64_ops(lo, hi))

    def test_plain_vcc_data_flbit_wave32_fspx(self):
        # FSPX wave32 compute allows plain-data VCC pairs with s_flbit_i32_{b64,i64}
        # (s_bcnt/s_ff _b64 are strict wave64 markers in WAVE64_RE).
        lines = [
            "s_mov_b32 vcc_lo, s4",
            "s_mov_b32 vcc_hi, s5",
            "s_flbit_i32_b64 s6, vcc",
            "s_flbit_i32_i64 s7, vcc",
            "s_flbit_i32_b64 s8, null",
            "s_flbit_i32_i64 s9, null",
        ] + _store_words(["s6", "s7", "s8", "s9"])
        spv = _translate_cs("cs_w32_flbit_vcc_data", lines, wave32=True)
        for lo, hi in self.PAIRS:
            with self.subTest(lo=hex(lo), hi=hex(hi)):
                ref = _ref_b64_ops(lo, hi)
                self.assertEqual(_run_cs(spv, lo, hi, wave32=True, n_words=4), [ref[6], ref[7], M32, M32])

    def test_vcc_transition_from_lane_mask_to_plain_data(self):
        # VCC first holds a real lane mask, then is overwritten with arbitrary plain 64-bit scalar data.
        spv = _translate_cs(
            "cs_w64_b64_vcc_mask_then_data",
            self._program_vcc_data(prefix=("v_cmp_lt_u32_e64 vcc, v0, 37",)),
            wave32=False,
        )
        for lo, hi in ((0x3, 0xF), (0, 0x8), (M32, M32), (0, 0)):
            with self.subTest(lo=hex(lo), hi=hex(hi)):
                self.assertEqual(_run_cs(spv, lo, hi, wave32=False), _ref_b64_ops(lo, hi))

    def test_null_source_b64_bitops(self):
        lines = [
            "s_bcnt1_i32_b64 s6, null",
            "s_cselect_b32 s7, 1, 0",
            "s_bcnt0_i32_b64 s8, null",
            "s_cselect_b32 s9, 1, 0",
            "s_ff1_i32_b64 s10, null",
            "s_ff0_i32_b64 s11, null",
            "s_flbit_i32_b64 s12, null",
            "s_flbit_i32_i64 s13, null",
        ] + _store_words(["s6", "s7", "s8", "s9", "s10", "s11", "s12", "s13"])
        spv = _translate_cs("cs_b64_null_64", lines, wave32=False)
        self.assertEqual(_run_cs(spv, 0, 0, wave32=False), _ref_b64_ops(0, 0))

    def test_vcc_lane_mask_control_still_uses_wave_mask(self):
        # Negative control: when VCC holds a real lane mask (lanes 5..36 = 32 set bits),
        # the wave ballot path stays active.
        lines = [
            "v_cmp_ge_u32_e64 vcc, v0, 5",
            "v_cmp_lt_u32_e64 s[14:15], v0, 37",
            "s_and_b64 vcc, vcc, s[14:15]",
            "s_bcnt1_i32_b64 s6, vcc",
            "s_bcnt0_i32_b64 s7, vcc",
            "s_ff1_i32_b64 s8, vcc",
            "s_ff0_i32_b64 s9, vcc",
            "s_flbit_i32_b64 s10, vcc",
        ] + _store_words(["s6", "s7", "s8", "s9", "s10"])
        spv = _translate_cs("cs_w64_b64_vcc_mask_ctrl", lines, wave32=False)
        # Bits 5..36 set: 32 ones, 32 zeros, lowest 1 at 5, lowest 0 at 0, leading zeros = 64 - 37 = 27.
        self.assertEqual(_run_cs(spv, 0, 0, wave32=False, n_words=5), [32, 32, 5, 0, 27])


if __name__ == "__main__":
    unittest.main()
