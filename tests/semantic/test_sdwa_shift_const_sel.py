"""Semantic tests for SDWA source selects on constant shift amounts (v_{lshlrev,lshrrev,ashrrev}_*_sdwa).

`Gen.imm_val()` must not fold an inline integer or seeded SGPR constant when the operand carries a
non-DWORD `sdwa_sel` (`BYTE_0`..`BYTE_3`, `WORD_0`, `WORD_1`) or `sext()`. Otherwise `SHIFT_REV`
(`v_lshlrev_b32`, `v_lshrrev_b32`, `v_ashrrev_i32`) shifts by the unselected 32-bit constant instead
of the SDWA-narrowed field (`tools/sdwa_diff.py` `shift-const-sel`).

Standard library only; no LLVM, no GPU, no game files.
"""
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
import sdwa_diff  # noqa: E402
import spirv_asm  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)


def _run_ps(name: str, lines, export: str = "v8, v9, v10, v11"):
    with tempfile.TemporaryDirectory() as tmp:
        os.mkdir(os.path.join(tmp, "disasm"))
        src = os.path.join(tmp, "disasm", name + ".s")
        with open(os.path.join(tmp, name + ".dxbc"), "wb") as f:
            f.write(STUB_DXBC)
        with open(src, "w", encoding="utf-8", newline="\n") as f:
            f.write(f"; {name}\n")
            for ln in lines:
                f.write(f"\t{ln}\n")
            f.write(f"\texp mrt0 {export} done vm\n\ts_endpgm\n")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            spv = rdna2spirv.translate(src)
    binary = spirv_asm.assemble(spv)
    assert len(binary) >= 20 and binary[:4] == b"\x03\x02\x23\x07"
    return list(si.run_fragment(spv)["%out_mrt0"])


class SdwaShiftConstSelTests(unittest.TestCase):
    def test_inline_const_shift_with_src0_sel(self):
        # 2 has BYTE_3 == 0 -> 64 << 0 = 0x40 merged into low byte of 0x80000000 -> 0x80000040.
        # -16 is 0xFFFFFFF0: BYTE_1 == 0xFF -> shift by 31; 0x80000000 >> 31 = 1 (unselected -16 & 31 = 16 -> 0x8000).
        # -16 WORD_1 == 0xFFFF -> shift by 31; ashr(0x80000000, 31) = 0xFFFFFFFF (unselected -> 0xFFFF8000).
        # DWORD select still folds cleanly: 3 << 0 = 3, 0x10 << 3 = 0x80.
        got = _run_ps("ps_sdwa_shift_inline", [
            "v_mov_b32 v1, 0x80000000",
            "v_mov_b32 v2, 0x80000000",
            "v_mov_b32 v3, 0x80000000",
            "v_mov_b32 v4, 0x00000010",
            "v_lshlrev_b32_sdwa v1, 2, 64 dst_sel:BYTE_0 dst_unused:UNUSED_PRESERVE src0_sel:BYTE_3 src1_sel:WORD_0",
            "v_lshrrev_b32_sdwa v2, -16, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:DWORD",
            "v_ashrrev_i32_sdwa v3, -16, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD",
            "v_lshlrev_b32_sdwa v4, 3, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD",
            "v_mov_b32 v8, v1",
            "v_mov_b32 v9, v2",
            "v_mov_b32 v10, v3",
            "v_mov_b32 v11, v4",
        ])
        self.assertEqual(got, [0x80000040, 1, 0xFFFFFFFF, 0x80])

    def test_seeded_sgpr_const_shift_with_src0_sel_and_sext(self):
        # s4 = 0x05040300: BYTE_1 = 3, BYTE_2 = 4, BYTE_3 = 5, WORD_1 = 0x0504 (&31 = 4); unselected s4 & 31 = 0.
        # s5 = 0x00008100: BYTE_1 = 0x81, sext -> 0xFFFFFF81 (&31 = 1); unselected s5 & 31 = 0.
        got = _run_ps("ps_sdwa_shift_sgpr", [
            "s_mov_b32 s4, 0x05040300",
            "s_mov_b32 s5, 0x00008100",
            "v_mov_b32 v1, 0x00000001",
            "v_mov_b32 v2, 0x00000100",
            "v_mov_b32 v3, 0x80000000",
            "v_mov_b32 v4, 0x00000003",
            "v_lshlrev_b32_sdwa v1, s4, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:DWORD",
            "v_lshrrev_b32_sdwa v2, s4, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2 src1_sel:DWORD",
            "v_ashrrev_i32_sdwa v3, s4, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3 src1_sel:DWORD",
            "v_lshlrev_b32_sdwa v4, sext(s5), v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:DWORD",
            "v_mov_b32 v8, v1",
            "v_mov_b32 v9, v2",
            "v_mov_b32 v10, v3",
            "v_mov_b32 v11, v4",
        ])
        self.assertEqual(got, [1 << 3, 0x100 >> 4, 0xFC000000, 3 << 1])

    def test_matches_sdwa_diff_reference_on_all_three_shift_ops(self):
        cases = [
            "v_lshlrev_b32_sdwa v1, 2, 64 dst_sel:BYTE_0 dst_unused:UNUSED_PRESERVE src0_sel:BYTE_3 src1_sel:WORD_0",
            "v_lshrrev_b32_sdwa v1, s4, v2 dst_sel:WORD_0 dst_unused:UNUSED_SEXT src0_sel:BYTE_1 src1_sel:DWORD",
            "v_ashrrev_i32_sdwa v1, sext(s5), v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD",
        ]
        for idx, insn in enumerate(cases):
            with self.subTest(insn=insn):
                prog = [
                    "v_mov_b32 v1, 0x80000000",
                    "v_mov_b32 v2, 0x12345678",
                    "v_mov_b32 v3, 0x80000040",
                    "v_mov_b32 v4, 0",
                    "v_mov_b32 v5, 0",
                    "v_mov_b32 v6, 0",
                    "v_mov_b32 v7, 0",
                    "s_mov_b32 s4, 0x05040300",
                    "s_mov_b32 s5, 0x80030000",
                    "s_mov_b32 s6, 0",
                    "s_mov_b32 s7, 0",
                    "s_mov_b32 vcc_lo, 0",
                    insn,
                    "v_mov_b32 v8, v1",
                    "v_mov_b32 v9, v2",
                    "v_mov_b32 v10, v3",
                    "v_cndmask_b32_e64 v11, 0, 1, vcc_lo",
                ]
                ref = sdwa_diff.reference(prog)
                got = _run_ps(f"ps_sdwa_ref_{idx}", prog)
                self.assertEqual(got, ref)


if __name__ == "__main__":
    unittest.main()
