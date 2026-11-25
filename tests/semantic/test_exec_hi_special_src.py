"""Regression tests for s_mov_b32 exec_hi from non-mask special sources (m0, scc, plain-data vcc).

In wave64, s_mov_b32 exec_lo, <special> went through Gen.dst -> Gen.exec_bit and extracted bit
(lane & 31) of the 32-bit word. s_mov_b32 exec_hi, <special> instead took the special-cased
exec_half_write branch, which only extracted per-lane bits for SGPRs and passed raw src_u() for
special registers (m0, scc/src_scc, plain-data vcc_lo/vcc_hi). exec_half_write then tested
val != 0, enabling all 32 upper lanes (32..63) whenever any single bit was set.

All shaders here are synthetic and hand-written; no game files, LLVM, or GPU are used.

Run directly:
    python -I tests/semantic/test_exec_hi_special_src.py -v
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


def _translate_cs(name, lines, *, emulate=False):
    with tempfile.TemporaryDirectory() as tmp:
        os.mkdir(os.path.join(tmp, "disasm"))
        os.mkdir(os.path.join(tmp, "fssm"))
        src = os.path.join(tmp, "disasm", name + ".s")
        with open(src, "w", encoding="utf-8", newline="\n") as f:
            f.write(f"; {name} synthetic\n")
            for ln in lines:
                f.write(f"\t{ln}\n")
        with open(os.path.join(tmp, name + ".dxbc"), "wb") as f:
            f.write(STUB_DXBC)
        with open(os.path.join(tmp, "fssm", name + ".fssm"), "wb") as f:
            f.write(bytes(0x138) + struct.pack("<4I", 0x8090, 64, 1, 1))
        prev = rdna2spirv.WAVE64_EMULATE
        try:
            rdna2spirv.WAVE64_EMULATE = emulate
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                spv = rdna2spirv.translate(src)
                gc.collect()
        finally:
            rdna2spirv.WAVE64_EMULATE = prev
        return spv


def _run_cs(spv, s4, s5=0, *, emulate=False):
    pc = VSHARP + [s4 & M32, s5 & M32] + [0] * 26
    mem = si.run_workgroup(
        spv, 64, 32 if emulate else 64,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0),
                              "%sg_lane": i % 32 if emulate else i},
        push_constants=pc,
        memory={},
    )
    return [mem[4 * l] for l in range(64)]


def _observe_exec(setup_lines):
    """Run setup_lines (starting with all 64 lanes active), then record which lanes are active in EXEC."""
    return [
        "s_mov_b64 exec, -1",
        "v_mov_b32 v1, 0",
        "s_mov_b32 exec_lo, 0x00000003",  # lanes 0 and 1 active in low half; must stay intact on exec_hi writes
    ] + setup_lines + [
        "v_mov_b32 v1, 1",
        "s_mov_b64 exec, -1",
        "v_lshlrev_b32 v2, 2, v0",
        "buffer_store_dword v1, v2, s[0:3], 0 offen",
        "s_endpgm",
    ]


def _bits64(lo32, hi32):
    mask = (lo32 & M32) | ((hi32 & M32) << 32)
    return [1 if (mask >> l) & 1 else 0 for l in range(64)]


class ExecHiSpecialSrcTests(unittest.TestCase):
    def test_exec_hi_from_m0_and_plain_vcc_extracts_per_lane_bits(self):
        cases = [
            ("m0", ["s_mov_b32 m0, s4", "s_mov_b32 exec_hi, m0"]),
            ("vcc_lo_data", ["s_mov_b32 vcc_lo, s4", "s_mov_b32 vcc_hi, s5", "s_mov_b32 exec_hi, vcc_lo"]),
            ("vcc_hi_data", ["s_mov_b32 vcc_lo, s5", "s_mov_b32 vcc_hi, s4", "s_mov_b32 exec_hi, vcc_hi"]),
        ]
        patterns = [0x00000000, 0x00000001, 0x00000005, 0x80000002, 0x00FF00F0, 0xFFFFFFFF]
        for emulate in (False, True):
            for tag, setup in cases:
                spv = _translate_cs(f"cs_exec_hi_{tag}_{int(emulate)}", _observe_exec(setup), emulate=emulate)
                for pat in patterns:
                    with self.subTest(tag=tag, emulate=emulate, pat=hex(pat)):
                        got = _run_cs(spv, pat, 0x12345678, emulate=emulate)
                        self.assertEqual(got, _bits64(0x00000003, pat))

    def test_exec_hi_from_scc_sets_only_lane_32(self):
        for scc_val in (0, 1):
            with self.subTest(scc=scc_val):
                setup = [
                    "s_cmp_eq_u32 1, 1" if scc_val else "s_cmp_eq_u32 0, 1",
                    "s_mov_b32 exec_hi, src_scc",
                ]
                spv = _translate_cs(f"cs_exec_hi_scc_{scc_val}", _observe_exec(setup), emulate=False)
                self.assertEqual(_run_cs(spv, 0), _bits64(0x00000003, scc_val))

    def test_exec_hi_from_lane_mask_vcc_hi_control(self):
        # Control: when VCC holds a real wave64 lane mask (v_cmp), s_mov_b32 exec_hi, vcc_hi still
        # copies the upper 32 lanes' mask bits while keeping exec_lo intact.
        lines = [
            "s_mov_b64 exec, -1",
            "v_cmp_lt_u32_e32 vcc, v0, 40",   # lanes 0..39 true -> upper half is lanes 32..39 (0x000000FF)
            "v_mov_b32 v1, 0",
            "s_mov_b32 exec_lo, 0x00000003",
            "s_mov_b32 exec_hi, vcc_hi",
            "v_mov_b32 v1, 1",
            "s_mov_b64 exec, -1",
            "v_lshlrev_b32 v2, 2, v0",
            "buffer_store_dword v1, v2, s[0:3], 0 offen",
            "s_endpgm",
        ]
        spv = _translate_cs("cs_exec_hi_vcchi_mask_ctrl", lines, emulate=False)
        self.assertEqual(_run_cs(spv, 0), _bits64(0x00000003, 0x000000FF))


if __name__ == "__main__":
    unittest.main()
