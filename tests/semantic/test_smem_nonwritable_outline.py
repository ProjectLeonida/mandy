"""Per-load NonWritable SMEM restoration + outlined %sbuf_slow_xN partial-OOB helper (Item 3B.4).

Verifies:
  * Part A (NonWritable restoration & alias safety):
      - Read-only shaders keep NonWritable on all scalar loads.
      - Shaders with buffer stores/atomics still keep NonWritable (%sl_blk1/2/4) on provably non-aliasing scalar
        loads: disjoint V# roots, disjoint descriptor-table slots, 32-bit descriptor-heap s_load* (user SGPR + hi=0),
        s_getpc_b64 rodata loads, and loads with no PSB write reachable before or after them.
      - Scalar loads that can alias a store/atomic (same V# root, derived base from the written V# as in
        valu_fuzz --family mem, or unknown/constant base with a reaching/following write) use the mutable
        %sl_rw_blk2 / %sl_rw_blk4 / %_ptr_PSB_arr path without NonWritable.
      - Mixed shaders emit both NonWritable (%sl_blk*) and mutable (%sl_rw_blk*) loads cleanly in the same module.
      - Any s_dcache_* instruction alongside a PSB write conservatively disables NonWritable across the module.
  * Part B (Outlined %sbuf_slow_xN partial-OOB fallback):
      - Repeated wide s_buffer_load_dwordxN sites share one module-level %sbuf_slow_xN OpFunction per used width
        (emitting only widths actually used by the shader).
      - Partial-OOB and all-OOB runs match exact hardware bounds (zero out-of-range reads, exact values).
      - All generated SPIR-V modules pass spirv-as + spirv-val --target-env vulkan1.2.
"""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")
sys.dont_write_bytecode = True
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
from shader_ir import parse_line  # noqa: E402
import support  # noqa: E402
import test_value_semantics as tv  # noqa: E402

OUT = 0x2000
CB = 0x4000
TBL = 0x6000
OUT_V = 0x31016FAC
FSSM_12 = tv._fssm(12 << 1, (64, 1, 1))


def _validate(spv):
    if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
        return
    with tempfile.TemporaryDirectory() as tmp:
        asm, out = os.path.join(tmp, "m.spvasm"), os.path.join(tmp, "m.spv")
        with open(asm, "w", encoding="utf-8", newline="\n") as f:
            f.write(spv)
        r1 = subprocess.run(["spirv-as", "--target-env", "vulkan1.2", asm, "-o", out],
                            capture_output=True, text=True)
        assert r1.returncode == 0, r1.stderr
        r2 = subprocess.run(["spirv-val", "--target-env", "vulkan1.2", out],
                            capture_output=True, text=True)
        assert r2.returncode == 0, r2.stderr


class SmemNonWritableAndOutlineTests(unittest.TestCase):
    def test_fixture_restores_nonwritable_outlines_helpers_and_checks_oob(self):
        src = (Path(HERE) / "fixtures" / "cs_smem_nonwritable_outline.asm").read_bytes()
        spv = support.translate_listing("cs_smem_nonwritable_outline", src, fssm=FSSM_12)
        _validate(spv)

        # Part A: disjoint CBV s[4:7] and descriptor-table s[8:9] keep NonWritable despite buffer_store to s[0:3].
        self.assertIn("OpMemberDecorate %sl_blk4 0 NonWritable", spv)
        self.assertNotIn("%sl_rw_blk", spv)

        # Part B: two x4 + two x16 loads share ONE %sbuf_slow_x4 and ONE %sbuf_slow_x16 (no unused x2/x8).
        self.assertEqual(spv.count("%sbuf_slow_x4 = OpFunction "), 1)
        self.assertEqual(spv.count("%sbuf_slow_x16 = OpFunction "), 1)
        self.assertNotIn("%sbuf_slow_x2", spv)
        self.assertNotIn("%sbuf_slow_x8", spv)
        self.assertEqual(spv.count("OpFunctionCall %arr_sbuf_x4 %sbuf_slow_x4"), 2)
        self.assertEqual(spv.count("OpFunctionCall %arr_sbuf_x16 %sbuf_slow_x16"), 2)
        # Per-dword fallback loads exist once per width (4 + 16 = 20), not duplicated per call site (2*4 + 2*16 = 40).
        self.assertEqual(sum("OpLoad %uint" in ln and "Aligned 4" in ln for ln in spv.splitlines()), 20)

        # Execute with partial OOB: NUM_RECORDS = 0x18 (6 dwords: 0x00..0x14 in range, 0x18.. out of range).
        cb_bytes = 0x18
        mem = tv.ReadLog({a: 0xB0000000 | a for base in (CB, TBL) for a in range(base, base + 0x80, 4)})
        support.run_fragment(spv, inputs={"%wg_id": (0, 0, 0), "%local_id": (0, 0, 0)},
                             push_constants=[OUT, 0, 0x100, OUT_V, CB, 0, cb_bytes, OUT_V, TBL] + [0] * 23,
                             memory=mem)
        got = [dict.get(mem, OUT + 4 * i) for i in range(8)]
        self.assertEqual(got, [
            0xB0000000 | TBL,          # s12: table dword 0
            0xB0000000 | CB,           # s16: x4 @ 0x00 dword 0 (in bounds)
            0xB0000000 | (CB + 0x10),  # s20: x4 @ 0x10 dword 0 (in bounds, dwords 2..3 OOB)
            0xB0000000 | (CB + 0x10),  # s36: x16 @ 0x00 dword 4 (in bounds)
            0xB0000000 | (CB + 0x10),  # s48: x16 @ 0x10 dword 0 (in bounds)
            0,                         # s51: x16 @ 0x10 dword 3 (+0x1c -> OOB -> 0)
            0,                         # s62: x16 @ 0x10 dword 14 (OOB -> 0)
            0,                         # s63: x16 @ 0x10 dword 15 (OOB -> 0)
        ])
        cb_reads = sorted({a for a in mem.reads if CB <= a < CB + 0x80})
        self.assertEqual(cb_reads, list(range(CB, CB + cb_bytes, 4)))

        # Execute with all-OOB (NUM_RECORDS = 0): zero CB reads, all CB results 0.
        mem0 = tv.ReadLog({a: 0xC0000000 | a for base in (CB, TBL) for a in range(base, base + 0x80, 4)})
        support.run_fragment(spv, inputs={"%wg_id": (0, 0, 0), "%local_id": (0, 0, 0)},
                             push_constants=[OUT, 0, 0x100, OUT_V, CB, 0, 0, OUT_V, TBL] + [0] * 23,
                             memory=mem0)
        self.assertEqual([dict.get(mem0, OUT + 4 * i) for i in range(8)],
                         [0xC0000000 | TBL, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual([a for a in mem0.reads if CB <= a < CB + 0x80], [])

    def test_aliasing_derived_vsharp_drops_nonwritable(self):
        # valu_fuzz --family mem shape: s[16:19] is derived from s[0:3] (same root user SGPR s0),
        # so s_buffer_load_dwordx4 after buffer_store to s[0:3] must NOT carry NonWritable.
        lines = [
            "s_add_u32 s16, s0, 0x1000",
            "s_and_b32 s17, s1, 0xffff",
            "s_mov_b32 s18, 0x1000",
            "s_mov_b32 s19, 0x31016fac",
            "buffer_store_dword v0, v1, s[0:3], 0 offen",
            "s_waitcnt vmcnt(0)",
            "s_buffer_load_dwordx4 s[20:23], s[16:19], 0x0",
            "s_waitcnt lgkmcnt(0)",
            "s_endpgm",
        ]
        spv = support.translate_lines("cs_alias_smem", lines, fssm=FSSM_12)
        _validate(spv)
        self.assertIn("%sl_rw_blk4 = OpTypeStruct %v4uint_sl", spv)
        self.assertNotIn("OpMemberDecorate %sl_rw_blk4 0 NonWritable", spv)
        self.assertNotIn("%sl_blk4", spv)

    def test_mixed_ro_and_rw_loads_in_same_shader(self):
        # Disjoint CBV s[4:7] gets NonWritable (%sl_blk4); same-V# s[0:3] after store gets mutable (%sl_rw_blk4).
        lines = [
            "s_buffer_load_dwordx4 s[16:19], s[4:7], 0x0",
            "buffer_store_dword v0, v1, s[0:3], 0 offen",
            "s_waitcnt vmcnt(0)",
            "s_buffer_load_dwordx4 s[20:23], s[0:3], 0x0",
            "s_waitcnt lgkmcnt(0)",
            "s_endpgm",
        ]
        spv = support.translate_lines("cs_mixed_ro_rw", lines, fssm=FSSM_12)
        _validate(spv)
        self.assertIn("OpMemberDecorate %sl_blk4 0 NonWritable", spv)
        self.assertIn("%sl_rw_blk4 = OpTypeStruct %v4uint_sl", spv)
        self.assertNotIn("OpMemberDecorate %sl_rw_blk4 0 NonWritable", spv)
        self.assertIn("OpConvertUToPtr %_ptr_PSB_sl_blk4", spv)
        self.assertIn("OpConvertUToPtr %_ptr_PSB_sl_rw_blk4", spv)

    def test_descriptor_heap_load_stays_nonwritable_with_global_store(self):
        # 32-bit descriptor-table pointer in user SGPR s8 with hi=0 whose loaded dwords are consumed solely
        # as a V# descriptor stays NonWritable even alongside global_store with unknown VGPR address (saddr=off),
        # while a constant-address data s_load stays mutable.
        lines = [
            "s_mov_b32 s9, 0",
            "s_load_dwordx4 s[12:15], s[8:9], 0x0",
            "s_waitcnt lgkmcnt(0)",
            "buffer_load_dword v3, v0, s[12:15], 0 offen",
            "global_store_dword v[0:1], v2, off",
            "s_mov_b32 s20, 0x1000",
            "s_mov_b32 s21, 0",
            "s_load_dwordx4 s[24:27], s[20:21], 0x0",
            "v_mov_b32 v4, s24",
            "s_endpgm",
        ]
        spv = support.translate_lines("cs_heap_with_gstore", lines, fssm=FSSM_12)
        _validate(spv)
        self.assertIn("OpMemberDecorate %sl_blk4 0 NonWritable", spv)
        self.assertIn("OpConvertUToPtr %_ptr_PSB_sl_blk4", spv)
        self.assertIn("OpConvertUToPtr %_ptr_PSB_sl_rw_blk4", spv)

    def test_32bit_rw_buffer_data_sload_drops_nonwritable(self):
        # Negative control (P0): a 32-bit RW buffer pointer in user SGPR s0 with hi=0 that is written via
        # global_store (both via s[0:1] and via unknown off) and read as data via s_load MUST stay mutable.
        for store_line in (
            "global_store_dword v0, v1, s[0:1]",
            "global_store_dword v[0:1], v2, off",
        ):
            with self.subTest(store=store_line):
                lines = [
                    "s_mov_b32 s1, 0",
                    store_line,
                    "s_waitcnt vmcnt(0)",
                    "s_load_dwordx4 s[4:7], s[0:1], 0x0",
                    "s_waitcnt lgkmcnt(0)",
                    "v_mov_b32 v3, s4",
                    "s_endpgm",
                ]
                spv = support.translate_lines("cs_rw32_sload", lines, fssm=FSSM_12)
                _validate(spv)
                self.assertIn("%sl_rw_blk4 = OpTypeStruct %v4uint_sl", spv)
                self.assertNotIn("OpMemberDecorate %sl_rw_blk4 0 NonWritable", spv)
                self.assertNotIn("%sl_blk4", spv)

    def test_unknown_plus_user_sgpr_does_not_forge_disjoint_origin(self):
        # Negative control (P0): v_readfirstlane_b32 produces an unknown pointer in s10; adding user SGPR s4
        # (or s_lshl2_add_u32 with an unknown base) must NOT inherit ("in", 4) as a disjoint origin.
        for arith in (
            "s_add_u32 s10, s10, s4",
            "s_lshl2_add_u32 s10, s4, s10",
            "s_lshl2_add_u32 s10, s10, s4",
        ):
            with self.subTest(arith=arith):
                lines = [
                    "buffer_store_dword v0, v1, s[0:3], 0 offen",
                    "s_waitcnt vmcnt(0)",
                    "v_readfirstlane_b32 s10, v0",
                    arith,
                    "s_mov_b32 s11, 0",
                    "s_load_dwordx4 s[16:19], s[10:11], 0x0",
                    "s_waitcnt lgkmcnt(0)",
                    "v_mov_b32 v2, s16",
                    "s_endpgm",
                ]
                spv = support.translate_lines("cs_unknown_add_sgpr", lines, fssm=FSSM_12)
                _validate(spv)
                self.assertIn("%sl_rw_blk4 = OpTypeStruct %v4uint_sl", spv)
                self.assertNotIn("%sl_blk4", spv)

    def test_multi_sgpr_clobbers_clear_provenance(self):
        # Negative control (P1): s_movreld_b32 / s_swappc_b64 / s_setpc_b64 / s_call_b64 clobber arbitrary SGPRs,
        # so prior provenance of s[4:7] must not survive across them.
        for clobber in (
            "s_movreld_b32 s4, s0",
            "s_swappc_b64 s[20:21], s[22:23]",
            "s_setpc_b64 s[22:23]",
            "s_call_b64 s[20:21], 4",
        ):
            with self.subTest(clobber=clobber):
                insns = [
                    parse_line("buffer_store_dword v0, v1, s[0:3], 0 offen"),
                    parse_line(clobber),
                    parse_line("s_buffer_load_dwordx4 s[16:19], s[4:7], 0x0"),
                    parse_line("s_endpgm"),
                ]
                ro = rdna2spirv.analyse_scalar_load_ro([insns], [[]], user_limit=12)
                self.assertEqual(ro, frozenset())

    def test_disjoint_table_slots_stay_nonwritable(self):
        # Two V#s loaded from distinct static offsets (0x0 and 0x20) of the same descriptor table s[8:9]:
        # CBV at +0x0 stays NonWritable even after buffer_store through UAV at +0x20.
        lines = [
            "s_mov_b32 s9, 0",
            "s_load_dwordx4 s[12:15], s[8:9], 0x0",
            "s_load_dwordx4 s[16:19], s[8:9], 0x20",
            "s_waitcnt lgkmcnt(0)",
            "buffer_store_dword v0, v1, s[16:19], 0 offen",
            "s_waitcnt vmcnt(0)",
            "s_buffer_load_dwordx4 s[20:23], s[12:15], 0x0",
            "s_endpgm",
        ]
        spv = support.translate_lines("cs_disjoint_table_slots", lines, fssm=FSSM_12)
        _validate(spv)
        self.assertIn("OpMemberDecorate %sl_blk4 0 NonWritable", spv)
        self.assertNotIn("%sl_rw_blk", spv)

    def test_dcache_inv_disables_nonwritable_conservatively(self):
        insns = [
            parse_line("s_buffer_load_dwordx4 s[16:19], s[4:7], 0x0"),
            parse_line("buffer_store_dword v0, v1, s[0:3], 0 offen"),
            parse_line("s_dcache_inv"),
            parse_line("s_endpgm"),
        ]
        ro = rdna2spirv.analyse_scalar_load_ro([insns], [[]], user_limit=12)
        self.assertEqual(ro, frozenset())


if __name__ == "__main__":
    unittest.main()
