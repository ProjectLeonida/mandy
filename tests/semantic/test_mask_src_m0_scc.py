"""Regression tests for m0 and scc/src_scc as lane-mask sources in mask_src and _maskable_b32.

_maskable_b32 in both tools/rdna2spirv.py and tools/lane_masks.py previously excluded m0 and
scc/src_scc, and Gen.mask_src rejected them as Unsupported("mask operand"). As a result:
  1. 32-bit bitwise mask ops into exec_lo/exec_hi (e.g. s_and_b32 exec_lo, exec_lo, m0 or
     s_andn2_b32 exec_lo, exec_lo, m0) fell back to raw src_u(), setting SCC=1 even when the
     resulting EXEC half was 0.
  2. 32-bit bitwise/not/select mask ops into vcc_lo/vcc_hi (e.g. s_and_b32 vcc_lo, vcc_lo, m0
     or s_not_b32 vcc_lo, m0) stored the raw non-zero scalar word into %vcc on lanes whose bit
     was 0 and dropped VCC from lane_masks.
  3. s_*_saveexec_b32 / s_*_wrexec_b32 / s_*_saveexec_b64 with m0 or src_scc raised Unsupported.

All shaders here are synthetic and hand-written; no game files, LLVM, or GPU are used.

Run directly:
    python -I tests/semantic/test_mask_src_m0_scc.py -v
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
            for ln in lines:
                f.write(f"\t{ln}\n")
        with open(os.path.join(tmp, name + ".dxbc"), "wb") as f:
            f.write(STUB_DXBC)
        if wave32:
            os.mkdir(os.path.join(tmp, "fspx"))
            with open(os.path.join(tmp, "fspx", name + ".json"), "w", encoding="utf-8", newline="\n") as f:
                json.dump({"rsrc2": 0x8020, "threads": [32, 1, 1], "wave32": True}, f)
        else:
            os.mkdir(os.path.join(tmp, "fssm"))
            with open(os.path.join(tmp, "fssm", name + ".fssm"), "wb") as f:
                f.write(b"FSSM" + bytes(0x138 - 4) + struct.pack("<4I", 0x8090, 64, 1, 1))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            spv = rdna2spirv.translate(src)
            gc.collect()
        return spv


def _run_cs(spv, s4, s5=0, *, wave32=False, n_words=2):
    wg = 32 if wave32 else 64
    pc = VSHARP + [s4 & M32, s5 & M32] + [0] * 26
    mem = si.run_workgroup(
        spv, wg, wg,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
        push_constants=pc,
        memory={},
    )
    return [tuple(mem[(l * n_words + w) * 4] for w in range(n_words)) for l in range(wg)]


class MaskSrcM0SccTests(unittest.TestCase):
    def test_exec_b32_bitwise_with_m0_gives_exact_lanes_and_scc(self):
        # exec_lo = 0x2 (only lane 1 active), m0 = s4:
        #   - s4 = 0x5 -> exec_lo & m0 = 0x0, SCC = 0
        #   - s4 = 0x7 -> exec_lo & m0 = 0x2, SCC = 1
        #   - s4 = 0x2 -> exec_lo & ~m0 = 0x0, SCC = 0
        #   - s4 = 0x5 -> exec_lo & ~m0 = 0x2, SCC = 1
        for wave32 in (True, False):
            wg = 32 if wave32 else 64
            for op, fn in (
                ("s_and_b32", lambda e, m: e & m),
                ("s_andn2_b32", lambda e, m: e & (~m & M32)),
                ("s_xor_b32", lambda e, m: e ^ m),
            ):
                lines = [
                    "s_mov_b32 exec_lo, 0x00000002",
                    "s_mov_b32 m0, s4",
                    f"{op} exec_lo, exec_lo, m0",
                    "s_cselect_b32 s10, 1, 0",
                    "v_mov_b32 v1, 1",
                    "s_mov_b32 exec_lo, -1",
                    "v_mov_b32 v2, s10",
                    "v_mul_lo_u32 v4, v0, 8",
                    "buffer_store_dword v1, v4, s[0:3], 0 offen",
                    "buffer_store_dword v2, v4, s[0:3], 0 offen offset:4",
                    "s_endpgm",
                ]
                spv = _translate_cs(f"cs_exec_{op}_m0_{wg}", lines, wave32=wave32)
                for m0_val in (0x0, 0x2, 0x5, 0x7, 0x80000002):
                    want_lo = fn(0x2, m0_val)
                    want_scc = int(want_lo != 0)
                    with self.subTest(wave=wg, op=op, m0=hex(m0_val)):
                        got = _run_cs(spv, m0_val, wave32=wave32, n_words=2)
                        expected = [
                            (1 if (l >= 32 or ((want_lo >> l) & 1)) else 0, want_scc)
                            for l in range(wg)
                        ]
                        self.assertEqual(got, expected)

    def test_vcc_b32_bitwise_and_not_with_m0_and_scc(self):
        # Wave32: vcc_lo starts as lanes 0..7 (0xFF), then combined with m0 (s4).
        # Verify both per-lane vcc_lo via v_cndmask_b32 AND wave SCC.
        for op, fn in (
            ("s_and_b32", lambda a, b: a & b),
            ("s_or_b32", lambda a, b: a | b),
            ("s_andn2_b32", lambda a, b: a & (~b & M32)),
        ):
            lines = [
                "v_cmp_lt_u32_e32 vcc_lo, v0, 8",
                "s_mov_b32 m0, s4",
                f"{op} vcc_lo, vcc_lo, m0",
                "s_cselect_b32 s10, 1, 0",
                "v_cndmask_b32_e64 v1, 0, 1, vcc_lo",
                "v_mov_b32 v2, s10",
                "v_mul_lo_u32 v4, v0, 8",
                "buffer_store_dword v1, v4, s[0:3], 0 offen",
                "buffer_store_dword v2, v4, s[0:3], 0 offen offset:4",
                "s_endpgm",
            ]
            spv = _translate_cs(f"cs_vcc_{op}_m0_32", lines, wave32=True)
            for m0_val in (0x0, 0x5, 0xFF, 0x0000FF00):
                want = fn(0xFF, m0_val)
                want_scc = int(want != 0)
                with self.subTest(op=op, m0=hex(m0_val)):
                    got = _run_cs(spv, m0_val, wave32=True, n_words=2)
                    expected = [((want >> l) & 1, want_scc) for l in range(32)]
                    self.assertEqual(got, expected)

        # s_not_b32 vcc_lo, m0 (wave32) and s_and_b32 vcc_hi, vcc_hi, src_scc (wave64)
        spv_not = _translate_cs(
            "cs_vcc_not_m0_32",
            [
                "s_mov_b32 m0, s4",
                "s_not_b32 vcc_lo, m0",
                "s_cselect_b32 s10, 1, 0",
                "v_cndmask_b32_e64 v1, 0, 1, vcc_lo",
                "v_mov_b32 v2, s10",
                "v_mul_lo_u32 v4, v0, 8",
                "buffer_store_dword v1, v4, s[0:3], 0 offen",
                "buffer_store_dword v2, v4, s[0:3], 0 offen offset:4",
                "s_endpgm",
            ],
            wave32=True,
        )
        for m0_val in (0x0, 0x5, M32):
            want = (~m0_val) & M32
            with self.subTest(op="s_not_b32", m0=hex(m0_val)):
                got = _run_cs(spv_not, m0_val, wave32=True, n_words=2)
                self.assertEqual(got, [((want >> l) & 1, int(want != 0)) for l in range(32)])

        # Wave64: vcc = lanes 0..35 (lo=0xFFFFFFFF, hi=0xF), s_and_b32 vcc_hi, vcc_hi, src_scc -> hi becomes 0x1 (lane 32 only)
        spv_w64_scc = _translate_cs(
            "cs_vcc_hi_and_scc_64",
            [
                "v_cmp_lt_u32_e32 vcc, v0, 36",
                "s_cmp_eq_u32 1, 1",
                "s_and_b32 vcc_hi, vcc_hi, src_scc",
                "s_cselect_b32 s10, 1, 0",
                "v_cndmask_b32_e64 v1, 0, 1, vcc",
                "v_mov_b32 v2, s10",
                "v_mul_lo_u32 v4, v0, 8",
                "buffer_store_dword v1, v4, s[0:3], 0 offen",
                "buffer_store_dword v2, v4, s[0:3], 0 offen offset:4",
                "s_endpgm",
            ],
            wave32=False,
        )
        got = _run_cs(spv_w64_scc, 0, wave32=False, n_words=2)
        expected = [(1 if (l < 32 or l == 32) else 0, 1) for l in range(64)]
        self.assertEqual(got, expected)

    def test_saveexec_b32_and_b64_with_m0_and_src_scc(self):
        # s_and_saveexec_b32 with m0 in wave32
        spv_se32 = _translate_cs(
            "cs_saveexec32_m0",
            [
                "s_mov_b32 m0, s4",
                "s_and_saveexec_b32 s6, m0",
                "s_cselect_b32 s10, 1, 0",
                "v_mov_b32 v1, 1",
                "s_mov_b32 exec_lo, s6",
                "v_mov_b32 v2, s10",
                "v_mul_lo_u32 v4, v0, 8",
                "buffer_store_dword v1, v4, s[0:3], 0 offen",
                "buffer_store_dword v2, v4, s[0:3], 0 offen offset:4",
                "s_endpgm",
            ],
            wave32=True,
        )
        for m0_val in (0x0, 0x00000005, 0x80000001):
            with self.subTest(op="saveexec_b32", m0=hex(m0_val)):
                got = _run_cs(spv_se32, m0_val, wave32=True, n_words=2)
                self.assertEqual(got, [((m0_val >> l) & 1, int(m0_val != 0)) for l in range(32)])

        # s_and_saveexec_b64 with m0 in wave64: m0 is 32-bit zero-extended to 64 bits (lanes 32..63 cleared)
        spv_se64 = _translate_cs(
            "cs_saveexec64_m0",
            [
                "s_mov_b32 m0, s4",
                "s_and_saveexec_b64 s[6:7], m0",
                "s_cselect_b32 s10, 1, 0",
                "v_mov_b32 v1, 1",
                "s_mov_b64 exec, s[6:7]",
                "v_mov_b32 v2, s10",
                "v_mul_lo_u32 v4, v0, 8",
                "buffer_store_dword v1, v4, s[0:3], 0 offen",
                "buffer_store_dword v2, v4, s[0:3], 0 offen offset:4",
                "s_endpgm",
            ],
            wave32=False,
        )
        got = _run_cs(spv_se64, 0x0000000A, wave32=False, n_words=2)
        expected = [(1 if (l < 32 and ((0x0000000A >> l) & 1)) else 0, 1) for l in range(64)]
        self.assertEqual(got, expected)


if __name__ == "__main__":
    unittest.main()
