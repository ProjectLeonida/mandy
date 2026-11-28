"""Semantic tests for s_mov_b64 with plain-data VCC sources and null sources/destinations.

Verifies:
  1. `s_mov_b64 s[D:D+1], vcc` when VCC holds plain 64-bit scalar data copies both %vcc and %vcc_hi
     (rather than duplicating %vcc into both words).
  2. `s_mov_b64 s[D:D+1], null` writes (0, 0) and `s_mov_b64 null, <src>` discards the move cleanly while
     invalid sources still raise Unsupported.
  3. Control: `s_mov_b64 s[D:D+1], vcc` when VCC holds a real lane mask still preserves the per-lane wave mask.
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


def _translate_cs(name, lines, *, wave32=False):
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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            spv = rdna2spirv.translate(src)
            gc.collect()
        return spv


def _run_cs(spv, lo, hi, *, wave32=False, n_words=4):
    n_lanes = 32 if wave32 else 64
    pc = VSHARP + [lo & M32, hi & M32, 0, 0] + [0] * 24
    mem = si.run_workgroup(
        spv, n_lanes, n_lanes,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
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


class Mov64VccDataAndNullTests(unittest.TestCase):
    PAIRS = (
        (0x11223344, 0x55667788),
        (0x00000000, 0xDEADBEEF),
        (0xCAFEBABE, 0x00000000),
        (0xFFFFFFFF, 0x80000001),
    )

    def test_mov64_from_plain_vcc_preserves_both_words(self):
        lines = [
            "s_mov_b32 vcc_lo, s4",
            "s_mov_b32 vcc_hi, s5",
            "s_mov_b64 s[6:7], vcc",
        ] + _store_words(["s6", "s7"])
        for wave32 in (False, True):
            spv = _translate_cs(f"cs_mov64_vcc_data_{32 if wave32 else 64}", lines, wave32=wave32)
            for lo, hi in self.PAIRS:
                with self.subTest(wave32=wave32, lo=hex(lo), hi=hex(hi)):
                    self.assertEqual(_run_cs(spv, lo, hi, wave32=wave32, n_words=2), [lo, hi])

    def test_mov64_from_vcc_after_mask_overwritten_with_data(self):
        lines = [
            "v_cmp_lt_u32_e64 vcc, v0, 37",
            "s_mov_b32 vcc_lo, s4",
            "s_mov_b32 vcc_hi, s5",
            "s_mov_b64 s[6:7], vcc",
        ] + _store_words(["s6", "s7"])
        spv = _translate_cs("cs_w64_mov64_vcc_mask_to_data", lines, wave32=False)
        for lo, hi in self.PAIRS:
            with self.subTest(lo=hex(lo), hi=hex(hi)):
                self.assertEqual(_run_cs(spv, lo, hi, wave32=False, n_words=2), [lo, hi])

    def test_mov64_null_source_and_destination(self):
        lines = [
            "s_mov_b32 s6, 0x12345678",
            "s_mov_b32 s7, 0x9abcdef0",
            "s_mov_b64 s[6:7], null",
            "s_mov_b64 null, s[4:5]",
            "s_mov_b64 null, vcc",
            "s_mov_b64 null, 0",
        ] + _store_words(["s6", "s7", "s4", "s5"])
        for wave32 in (False, True):
            spv = _translate_cs(f"cs_mov64_null_{32 if wave32 else 64}", lines, wave32=wave32)
            self.assertEqual(_run_cs(spv, 0xAA, 0xBB, wave32=wave32, n_words=4), [0, 0, 0xAA, 0xBB])

        # Invalid source with null destination still raises Unsupported.
        with self.assertRaises(rdna2spirv.Unsupported):
            _translate_cs("cs_w64_mov64_null_bad", ["s_mov_b64 null, s4", "s_endpgm"], wave32=False)

    def test_mov64_lane_mask_vcc_control_preserves_wave_mask(self):
        lines = [
            "v_cmp_lt_u32_e64 vcc, v0, 37",
            "s_mov_b64 s[6:7], vcc",
            "s_bcnt1_i32_b64 s8, s[6:7]",
            "v_cndmask_b32_e64 v6, 3, 5, s[6:7]",
            "v_lshlrev_b32_e32 v5, 2, v0",
            "buffer_store_dword v6, v5, s[0:3], 0 offen",
            "v_mov_b32_e32 v7, s8",
            "buffer_store_dword v7, v5, s[0:3], 0 offen offset:256",
            "s_endpgm",
        ]
        spv = _translate_cs("cs_w64_mov64_vcc_mask_ctrl", lines, wave32=False)
        pc = VSHARP + [0] * 28
        mem = si.run_workgroup(
            spv, 64, 64,
            inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
            push_constants=pc, memory={},
        )
        self.assertEqual([mem[4 * l] for l in range(64)], [5 if l < 37 else 3 for l in range(64)])
        self.assertEqual([mem[256 + 4 * l] for l in range(64)], [37] * 64)


if __name__ == "__main__":
    unittest.main()
