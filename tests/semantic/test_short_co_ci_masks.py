"""Regression tests for 3-operand short-form v_add/sub/subrev_co_ci_u32 mask & RW handling.

The short VOP2 assembler spelling `v_add_co_ci_u32 vD, src0, src1` (3 operands) uses implicit VCC
for both carry-in and carry-out, so operand 1 is the 32-bit scalar/vector source `src0`, NOT the
carry-out mask destination:
  * `lane_masks.step()` must mark implicit `vcc` as MASK and must NOT mark `src0` (e.g. `s0`) as MASK.
  * `rdna2spirv.widen_wave64_masks()` must NOT widen `src0` SGPR to an SGPR pair (`n=2`), which
    otherwise crashes wave64 translation with `Unsupported("64-bit operand")`.
  * `rdna2spirv.writes_exec()` must NOT treat `src0 = exec_lo` on the 3-operand form as writing EXEC.
  * `wave64_halves._ins_rw()` must keep `src0` in `srcs` (not `dests`) and record `"vcc"` in both
    `defs` and `uses`.

    python -I tests/semantic/test_short_co_ci_masks.py
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
import wave64_halves  # noqa: E402

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
VSHARP = [0, 0, 0xFFFFFFFF, 0x31016FAC]


def translate_cs(name: str, body_lines: list[str], *, wave64: bool) -> str:
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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            spv = rdna2spirv.translate(src)
            gc.collect()
        return spv


def run_cs(spv: str, *, wave64: bool) -> dict[int, int]:
    threads = 64 if wave64 else 32
    pc = VSHARP + [0] * 28
    return si.run_workgroup(
        spv, threads, threads,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
        push_constants=pc, memory={},
    )


class ShortCoCiAnalysisTests(unittest.TestCase):
    def test_lane_masks_marks_vcc_not_src0(self):
        for op in ("v_add_co_ci_u32", "v_sub_co_ci_u32", "v_subrev_co_ci_u32"):
            for wave64 in (False, True):
                with self.subTest(op=op, wave64=wave64):
                    insns = [
                        shader_ir.parse_line("s_mov_b32 s10, 0x12345678"),
                        shader_ir.parse_line("s_mov_b32 vcc_lo, 0x00000007"),
                        shader_ir.parse_line(f"{op} v1, s10, v0"),
                        shader_ir.parse_line("s_bcnt1_i32_b32 s11, s10"),
                        shader_ir.parse_line("s_bcnt1_i32_b32 s12, vcc_lo"),
                    ]
                    masks, _ = lane_masks.analyse([insns], [[]], wave64=wave64)
                    # s10 is plain data input (src0), NOT a lane mask; implicit VCC carry-out IS a lane mask.
                    self.assertNotIn(10, masks[id(insns[3])])
                    self.assertEqual(masks[id(insns[4])].get("vcc"), lane_masks.MASK)

    def test_widen_wave64_masks_does_not_widen_short_src0(self):
        for op in ("v_add_co_ci_u32", "v_sub_co_ci_u32", "v_subrev_co_ci_u32"):
            with self.subTest(op=op):
                short_ins = shader_ir.parse_line(f"{op} v1, s10, v0")
                long_ins = shader_ir.parse_line(f"{op}_e64 v1, s12, s10, v0, s14")
                rdna2spirv.widen_wave64_masks([short_ins, long_ins])
                self.assertEqual(short_ins["args"][1]["n"], 1)
                self.assertEqual(long_ins["args"][1]["n"], 2)
                self.assertEqual(long_ins["args"][4]["n"], 2)

    def test_writes_exec_ignores_short_src0(self):
        short_ins = shader_ir.parse_line("v_add_co_ci_u32 v1, exec_lo, v0")
        long_ins = shader_ir.parse_line("v_add_co_ci_u32_e64 v1, exec_lo, s10, v0, vcc_lo")
        self.assertFalse(rdna2spirv.writes_exec(short_ins))
        self.assertTrue(rdna2spirv.writes_exec(long_ins))

    def test_wave64_halves_rw_keeps_src0_and_tracks_implicit_vcc(self):
        for op in ("v_add_co_ci_u32", "v_sub_co_ci_u32", "v_subrev_co_ci_u32"):
            with self.subTest(op=op):
                ins = shader_ir.parse_line(f"{op} v1, s10, v0")
                dests, srcs, defs, uses = wave64_halves.operands(ins)
                self.assertEqual([d["r"] for d in dests], [1])
                self.assertEqual([s["r"] for s in srcs], [10, 0])
                self.assertIn("vcc", defs)
                self.assertIn("vcc", uses)


class ShortCoCiExecTests(unittest.TestCase):
    def test_wave32_short_co_ci_preserves_src0_data_and_vcc_mask(self):
        # s10 = 0xFFFFFFF0 (popcnt 28). Carry-in = 1 for lane 0 only (v_cmp_eq_u32 v0, 0).
        # Lane L computes s10 + L + (L == 0): overflows for L >= 16 (16 lanes carry out).
        # Downstream s_bcnt1 on s10 must see popcnt(0xFFFFFFF0) == 28 (plain data),
        # and s_bcnt1 on vcc_lo must see 16 (wave carry-out mask).
        spv = translate_cs("w32_short_co_ci", [
            "s_mov_b32 s10, 0xfffffff0",
            "v_cmp_eq_u32_e32 vcc_lo, 0, v0",
            "v_add_co_ci_u32 v1, s10, v0",
            "s_bcnt1_i32_b32 s11, s10",
            "s_bcnt1_i32_b32 s12, vcc_lo",
            "v_mov_b32_e32 v2, s11",
            "v_mov_b32_e32 v3, s12",
            "v_lshlrev_b32_e32 v6, 2, v0",
            "buffer_store_dword v1, v6, s[0:3], 0 offen",
            "buffer_store_dword v2, v6, s[0:3], 0 offen offset:128",
            "buffer_store_dword v3, v6, s[0:3], 0 offen offset:256",
        ], wave64=False)
        mem = run_cs(spv, wave64=False)
        want_sum = [(0xFFFFFFF0 + i + (1 if i == 0 else 0)) & 0xFFFFFFFF for i in range(32)]
        self.assertEqual([mem[4 * i] for i in range(32)], want_sum)
        self.assertEqual([mem[128 + 4 * i] for i in range(32)], [28] * 32)
        self.assertEqual([mem[256 + 4 * i] for i in range(32)], [16] * 32)

    def test_wave64_short_co_ci_translates_and_runs(self):
        # On origin/main, widen_wave64_masks widens s10 -> s[10:11] and translation crashes.
        spv = translate_cs("w64_short_co_ci", [
            "s_mov_b32 s10, 0xffffffe0",
            "v_cmp_lt_u32_e64 vcc, v0, 2",
            "v_add_co_ci_u32 v1, s10, v0",
            "s_bcnt1_i32_b32 s11, s10",
            "s_bcnt1_i32_b64 s12, vcc",
            "v_mov_b32_e32 v2, s11",
            "v_mov_b32_e32 v3, s12",
            "v_lshlrev_b32_e32 v6, 2, v0",
            "buffer_store_dword v1, v6, s[0:3], 0 offen",
            "buffer_store_dword v2, v6, s[0:3], 0 offen offset:256",
            "buffer_store_dword v3, v6, s[0:3], 0 offen offset:512",
        ], wave64=True)
        mem = run_cs(spv, wave64=True)
        want_sum = [(0xFFFFFFE0 + i + (1 if i < 2 else 0)) & 0xFFFFFFFF for i in range(64)]
        self.assertEqual([mem[4 * i] for i in range(64)], want_sum)
        self.assertEqual([mem[256 + 4 * i] for i in range(64)], [27] * 64)
        self.assertEqual([mem[512 + 4 * i] for i in range(64)], [32] * 64)


if __name__ == "__main__":
    unittest.main()
