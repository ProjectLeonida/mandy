"""Round 4 Translator items 4T.3A + 4T.3B tests:
  * 4T.3A: Sound improvements to `analyse_scalar_load_ro`:
      - Forward CFG store-reachability (`not store_reached`), including loop back-edge soundness.
      - Explicit same-origin WAR protection (load before store to the same known origin stays mutable).
      - Generalized descriptor-only `s_load_dword*` provenance: 64-bit user SGPR table pointers `(in(k), in(k+1))`,
        2-level descriptor table loads (`s_load` -> `s_load` -> T#/S#/V#), and descriptors/V#s carried across
        scalar-spill VGPR lanes (`v_writelane_b32` / `v_readlane_b32`).
      - Branch-merge widening of static table slots to `("load_dyn", base)` without poisoning disjoint CBVs.
      - Negative controls: loop back-edge stores, 64-bit RW buffer data `s_load`, 2-level data `s_load`,
        spilled VGPR read as non-spill data or dynamic lane, and descriptor chain from a written root.
  * 4T.3B: Sound improvements to `float_only_image_ops`:
      - Complete f32 / f16 pure-float consumer coverage (`v_fmaak_f32`, `v_fmamk_f32`, `v_min3_f32`, `v_max3_f32`,
        `v_fma_legacy_f32`, `v_cubema_f32`, `v_div_fixup_f32`, `v_ldexp_f32`/`f16` src0, `v_fma_mix*` op_sel_hi,
        `v_mov_b32`/`v_cndmask_b32` VOP3 float `neg`/`abs`, `v_pk_fma_f16`, `v_cvt_pknorm_u16_f16`, etc.).
      - Transitive pure register-copy / select / pack propagation (`v_mov_b32`, `v_cndmask_b32`, `v_pack_b32_f16`)
        requiring all transitive consumers to be float-only.
      - Per-path EXEC-superset kill tracking (allows safe register-reuse kills in shaders that modify EXEC,
        while preserving soundness when EXEC narrows after `image_load` or switches via `s_xor`).
      - Full 32-bit packed overwrite kills (`v_cvt_pkrtz_f16_f32`).
      - Comprehensive negative controls for every soundness boundary.
"""
import os
import shutil
import struct
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


def _cs_fssm():
    hdr = bytearray(0x148)
    struct.pack_into("<4I", hdr, 0x138, 0x8088, 64, 1, 1)
    return bytes(hdr)


class ScalarLoadRoR4Tests(unittest.TestCase):
    def test_pre_store_disjoint_load_nonwritable_and_war_unknown_store_or_origin_stays_mutable(self):
        # Positive (CFG path-sensitive): Block 0 branches to Block 1 (disjoint CBV s[4:7] load -> buffer_store s[0:3])
        # and Block 2 (unknown global_store ... off -> endpgm). Block 1's load has `not store_reached` and its only
        # reachable following write is disjoint known s[0:3] -> gets NonWritable even though global_store exists in Block 2!
        b0 = [parse_line("s_cmp_eq_u32 s10, 0")]
        b1 = [
            parse_line("s_buffer_load_dwordx4 s[16:19], s[4:7], 0x0"),
            parse_line("buffer_store_dword v0, v1, s[0:3], 0 offen"),
            parse_line("s_endpgm"),
        ]
        b2 = [
            parse_line("global_store_dword v[0:1], v2, off"),
            parse_line("s_endpgm"),
        ]
        ro_cfg = rdna2spirv.analyse_scalar_load_ro([b0, b1, b2], [[1, 2], [], []], user_limit=12)
        self.assertIn(id(b1[0]), ro_cfg)

        # Negative control A1 (WAR hazard with later unknown store): s_load from s[0:1] before global_store ... off
        # could alias v[0:1]; NonWritable would allow ACO to sink the load past the store -> MUST stay mutable!
        lines_a1 = [
            "s_load_dwordx4 s[16:19], s[0:1], 0x0",
            "s_waitcnt lgkmcnt(0)",
            "global_store_dword v[0:1], v2, off",
            "s_waitcnt vmcnt(0)",
            "v_add_nc_u32_e32 v3, 1, s16",
            "s_endpgm",
        ]
        spv_a1 = support.translate_lines("cs_r4_war_unknown_store", lines_a1, fssm=FSSM_12)
        _validate(spv_a1)
        self.assertIn("%sl_rw_blk4 = OpTypeStruct %v4uint_sl", spv_a1)
        self.assertNotIn("%sl_blk4", spv_a1)

        # Negative control A2 (WAR hazard with unknown-origin load before known store): v_readfirstlane pointer has
        # origin None and could alias s[0:3] -> MUST stay mutable!
        lines_a2 = [
            "v_readfirstlane_b32 s10, v0",
            "v_readfirstlane_b32 s11, v1",
            "s_load_dwordx4 s[16:19], s[10:11], 0x0",
            "s_waitcnt lgkmcnt(0)",
            "buffer_store_dword v2, v3, s[0:3], 0 offen",
            "s_waitcnt vmcnt(0)",
            "v_add_nc_u32_e32 v4, 1, s16",
            "s_endpgm",
        ]
        spv_a2 = support.translate_lines("cs_r4_war_unknown_load", lines_a2, fssm=FSSM_12)
        _validate(spv_a2)
        self.assertIn("%sl_rw_blk4 = OpTypeStruct %v4uint_sl", spv_a2)
        self.assertNotIn("%sl_blk4", spv_a2)

    def test_pre_store_same_origin_war_stays_mutable(self):
        # Negative control (WAR safety): reading from s[0:3] before writing to the same known s[0:3] keeps
        # mutable loads so the driver cannot sink a NonWritable load past the store.
        lines = [
            "s_buffer_load_dwordx4 s[16:19], s[0:3], 0x0",
            "s_waitcnt lgkmcnt(0)",
            "buffer_store_dword v0, v1, s[0:3], 0 offen",
            "s_waitcnt vmcnt(0)",
            "v_mov_b32 v2, s16",
            "s_endpgm",
        ]
        spv = support.translate_lines("cs_r4_war_same_origin", lines, fssm=FSSM_12)
        _validate(spv)
        self.assertIn("%sl_rw_blk4 = OpTypeStruct %v4uint_sl", spv)
        self.assertNotIn("%sl_blk4", spv)

    def test_loop_backedge_store_reaches_earlier_load_in_loop(self):
        # Negative control (CFG loop back-edge): a load in a loop header before a store in the loop body IS
        # reached by the store via the back-edge (`store_reached = True`), while a load on a disjoint write-free
        # exit path is RO.
        b0 = [parse_line("s_cmp_eq_u32 s10, 0")]
        b1 = [parse_line("s_buffer_load_dwordx4 s[16:19], s[8:11], 0x0")]
        b2 = [parse_line("global_store_dword v[0:1], v2, off")]
        b3 = [parse_line("s_endpgm")]
        b4 = [
            parse_line("s_buffer_load_dwordx4 s[12:15], s[4:7], 0x0"),
            parse_line("s_endpgm"),
        ]
        ro = rdna2spirv.analyse_scalar_load_ro([b0, b1, b2, b3, b4], [[1, 4], [2, 3], [1], [], []], user_limit=12)
        self.assertIn(id(b4[0]), ro)
        self.assertNotIn(id(b1[0]), ro)

        # Single-block self-loop: load before store in same block with self back-edge must also stay mutable.
        loop_blk = [
            parse_line("s_buffer_load_dwordx4 s[16:19], s[8:11], 0x0"),
            parse_line("global_store_dword v[0:1], v2, off"),
        ]
        ro_self = rdna2spirv.analyse_scalar_load_ro([loop_blk, b3], [[0, 1], []], user_limit=12)
        self.assertNotIn(id(loop_blk[0]), ro_self)

    def test_pointer_arith_offset_table_load_does_not_forge_static_slot(self):
        # Negative control (P0): adding +0x20 to table base s2 before loading V# s[16:19] at imm 0x0 hits the
        # SAME slot as s[12:15] loaded from s[2:3]+0x20 -> must widen to load_dyn so s_buffer_load stays mutable!
        lines = [
            "s_load_dwordx4 s[12:15], s[2:3], 0x20",
            "s_add_u32 s4, s2, 0x20",
            "s_addc_u32 s5, s3, 0",
            "s_load_dwordx4 s[16:19], s[4:5], 0x0",
            "s_waitcnt lgkmcnt(0)",
            "buffer_store_dword v0, v1, s[16:19], 0 offen",
            "s_waitcnt vmcnt(0)",
            "s_buffer_load_dwordx4 s[20:23], s[12:15], 0x0",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32 v2, s20",
            "s_endpgm",
        ]
        insns = [parse_line(ln) for ln in lines]
        ro = rdna2spirv.analyse_scalar_load_ro([insns], [[]], user_limit=12)
        self.assertNotIn(id(insns[7]), ro)

    def test_special_reg_copy_vlist_spill_overwrite_and_2arg_vcmp_vcc_clobber(self):
        # Negative control B1: s_mov_b32 m0 / s_cselect_b32 exec_lo consumes s12 as data -> must drop NonWritable!
        for extra in ("s_mov_b32 m0, s12", "s_cselect_b32 exec_lo, s12, s13"):
            with self.subTest(case="special_copy", op=extra):
                insns = [
                    parse_line("global_store_dword v[0:1], v2, off"),
                    parse_line("s_load_dwordx4 s[12:15], s[2:3], 0x0"),
                    parse_line("buffer_load_dword v3, v0, s[12:15], 0 offen"),
                    parse_line(extra),
                    parse_line("s_endpgm"),
                ]
                self.assertNotIn(id(insns[1]), rdna2spirv.analyse_scalar_load_ro([insns], [[]], user_limit=12))

        # Negative control B2: vlist overwrite of spill VGPR v40 clears its spill provenance -> write origin becomes None!
        insns_b2 = [
            parse_line("s_load_dwordx4 s[12:15], s[2:3], 0x0"),
            parse_line("v_writelane_b32 v40, s12, 0"),
            parse_line("global_load_dwordx2 [v40,v41], v[0:1], off"),
            parse_line("v_readlane_b32 s20, v40, 0"),
            parse_line("buffer_store_dword v2, v3, s[20:23], 0 offen"),
            parse_line("s_buffer_load_dwordx4 s[24:27], s[4:7], 0x0"),
            parse_line("s_endpgm"),
        ]
        self.assertNotIn(id(insns_b2[5]), rdna2spirv.analyse_scalar_load_ro([insns_b2], [[]], user_limit=12))

        # Negative control VCC: 2-arg v_cmp_lt_u32_e32 v0, v1 clobbers implicit vcc -> s_load via vcc stays mutable!
        insns_vcc = [
            parse_line("s_load_dwordx2 vcc, s[2:3], 0x0"),
            parse_line("v_cmp_lt_u32_e32 v0, v1"),
            parse_line("buffer_store_dword v2, v3, s[0:3], 0 offen"),
            parse_line("s_load_dwordx4 s[16:19], vcc, 0x0"),
            parse_line("v_mov_b32 v4, s16"),
            parse_line("s_endpgm"),
        ]
        self.assertNotIn(id(insns_vcc[3]), rdna2spirv.analyse_scalar_load_ro([insns_vcc], [[]], user_limit=12))

    def test_64bit_and_2level_descriptor_table_loads_stay_nonwritable(self):
        # 64-bit user SGPR table pointer s[2:3] -> L1 loads sub-table pointer s[10:11] -> L2 loads V# s[16:19]
        # consumed solely by buffer_load_dword, even AFTER an unknown global_store!
        lines = [
            "global_store_dword v[0:1], v2, off",
            "s_waitcnt vmcnt(0)",
            "s_load_dwordx2 s[10:11], s[2:3], 0x10",
            "s_waitcnt lgkmcnt(0)",
            "s_load_dwordx4 s[16:19], s[10:11], 0x20",
            "s_waitcnt lgkmcnt(0)",
            "buffer_load_dword v3, v0, s[16:19], 0 offen",
            "s_endpgm",
        ]
        spv = support.translate_lines("cs_r4_2level_desc", lines, fssm=FSSM_12)
        _validate(spv)
        self.assertIn("OpMemberDecorate %sl_blk2 0 NonWritable", spv)
        self.assertIn("OpMemberDecorate %sl_blk4 0 NonWritable", spv)
        self.assertNotIn("%sl_rw_blk", spv)

    def test_spilled_descriptors_and_write_vsharp_provenance_survive_writelane_readlane(self):
        # Realistic heavy compute shader pattern:
        #   - 64-bit descriptor table s[2:3] loads CBV V# (+0x0) and UAV V# (+0x20).
        #   - UAV V# s[48:51] is spilled to v40 lanes 0..3 via v_writelane_b32 and reloaded via v_readlane_b32.
        #   - buffer_store_format_xyzw writes through s[48:51]; provenance stays ("load", ("in", 2), 0x20), NOT None!
        #   - Subsequent s_buffer_load through CBV s[12:15] (+0x0) stays NonWritable!
        lines = [
            "s_load_dwordx4 s[12:15], s[2:3], 0x0",
            "s_load_dwordx4 s[48:51], s[2:3], 0x20",
            "s_waitcnt lgkmcnt(0)",
            "v_writelane_b32 v40, s48, 0",
            "v_writelane_b32 v40, s49, 1",
            "v_writelane_b32 v40, s50, 2",
            "v_writelane_b32 v40, s51, 3",
            "v_readlane_b32 s48, v40, 0",
            "v_readlane_b32 s49, v40, 1",
            "v_readlane_b32 s50, v40, 2",
            "v_readlane_b32 s51, v40, 3",
            "buffer_store_format_xyzw v[0:3], v4, s[48:51], 0 idxen",
            "s_waitcnt vmcnt(0)",
            "s_buffer_load_dwordx4 s[20:23], s[12:15], 0x0",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32 v5, s20",
            "s_endpgm",
        ]
        spv = support.translate_lines("cs_r4_spill_vsharp", lines, fssm=FSSM_12)
        _validate(spv)
        self.assertIn("OpMemberDecorate %sl_blk4 0 NonWritable", spv)
        self.assertNotIn("%sl_rw_blk", spv)

    def test_64bit_rw_data_sload_and_2level_data_and_nonspill_vgpr_stay_mutable(self):
        # 1. 64-bit RW buffer data s_load after store MUST stay mutable.
        for store_line in (
            "global_store_dword v0, v1, s[0:1]",
            "global_store_dword v[0:1], v2, off",
        ):
            with self.subTest(case="rw64_data", store=store_line):
                lines = [
                    store_line,
                    "s_waitcnt vmcnt(0)",
                    "s_load_dwordx4 s[4:7], s[0:1], 0x0",
                    "s_waitcnt lgkmcnt(0)",
                    "v_mov_b32 v3, s4",
                    "s_endpgm",
                ]
                spv = support.translate_lines("cs_r4_rw64_data", lines, fssm=FSSM_12)
                _validate(spv)
                self.assertIn("%sl_rw_blk4 = OpTypeStruct %v4uint_sl", spv)
                self.assertNotIn("%sl_blk4", spv)

        # 2. 2-level load where L2 is consumed as data -> NEITHER L1 nor L2 may be NonWritable.
        lines_2lvl_data = [
            "global_store_dword v[0:1], v2, off",
            "s_waitcnt vmcnt(0)",
            "s_load_dwordx2 s[10:11], s[2:3], 0x10",
            "s_waitcnt lgkmcnt(0)",
            "s_load_dwordx4 s[16:19], s[10:11], 0x20",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32 v3, s16",
            "s_endpgm",
        ]
        spv2 = support.translate_lines("cs_r4_2lvl_data", lines_2lvl_data, fssm=FSSM_12)
        _validate(spv2)
        self.assertIn("%sl_rw_blk2 = OpTypeStruct %v2uint_sl", spv2)
        self.assertIn("%sl_rw_blk4 = OpTypeStruct %v4uint_sl", spv2)
        self.assertNotIn("%sl_blk2", spv2)
        self.assertNotIn("%sl_blk4", spv2)

        # 3. Spilled descriptor VGPR read directly by VALU (not pure spill) -> MUST stay mutable.
        lines_bad_spill = [
            "global_store_dword v[0:1], v2, off",
            "s_waitcnt vmcnt(0)",
            "s_load_dwordx4 s[12:15], s[2:3], 0x0",
            "s_waitcnt lgkmcnt(0)",
            "v_writelane_b32 v40, s12, 0",
            "v_add_nc_u32_e32 v1, 1, v40",
            "v_readlane_b32 s12, v40, 0",
            "buffer_load_dword v3, v0, s[12:15], 0 offen",
            "s_endpgm",
        ]
        insns_bs = [parse_line(ln) for ln in lines_bad_spill]
        self.assertEqual(rdna2spirv.analyse_scalar_load_ro([insns_bs], [[]], user_limit=12), frozenset())

        # 4. 2-level descriptor load whose root table s[0:1] is explicitly written -> NEITHER may be NonWritable.
        lines_written_root = [
            "global_store_dword v0, v1, s[0:1]",
            "s_waitcnt vmcnt(0)",
            "s_load_dwordx2 s[10:11], s[0:1], 0x10",
            "s_waitcnt lgkmcnt(0)",
            "s_load_dwordx4 s[16:19], s[10:11], 0x20",
            "s_waitcnt lgkmcnt(0)",
            "buffer_load_dword v3, v0, s[16:19], 0 offen",
            "s_endpgm",
        ]
        insns_wr = [parse_line(ln) for ln in lines_written_root]
        self.assertEqual(rdna2spirv.analyse_scalar_load_ro([insns_wr], [[]], user_limit=12), frozenset())

    def test_branch_merge_table_slots_widens_without_poisoning_disjoint_cbv(self):
        # Merging V#s from +0x20 and +0x40 of table s[2:3] widens the write origin to ("load_dyn", ("in", 2)):
        #   - A disjoint CBV s[4:7] AFTER the store still gets NonWritable.
        #   - A static slot (+0x0) from the SAME table s[2:3] AFTER the store MUST stay mutable (may alias load_dyn).
        b0 = [
            parse_line("s_load_dwordx4 s[12:15], s[2:3], 0x0"),
        ]
        b1 = [parse_line("s_load_dwordx4 s[48:51], s[2:3], 0x20")]
        b2 = [parse_line("s_load_dwordx4 s[48:51], s[2:3], 0x40")]
        b3 = [
            parse_line("buffer_store_dword v0, v1, s[48:51], 0 offen"),
            parse_line("s_buffer_load_dwordx4 s[20:23], s[4:7], 0x0"),
            parse_line("s_buffer_load_dwordx4 s[24:27], s[12:15], 0x0"),
            parse_line("s_endpgm"),
        ]
        ro = rdna2spirv.analyse_scalar_load_ro([b0, b1, b2, b3], [[1, 2], [3], [3], []], user_limit=12)
        self.assertIn(id(b0[0]), ro)    # descriptor load
        self.assertIn(id(b1[0]), ro)    # descriptor load
        self.assertIn(id(b2[0]), ro)    # descriptor load
        self.assertIn(id(b3[1]), ro)    # disjoint CBV s[4:7]
        self.assertNotIn(id(b3[2]), ro) # same table s[2:3] (+0x0) may alias load_dyn(s[2:3])!


class FloatOnlyImageOpsR4Tests(unittest.TestCase):
    def _assert_elided(self, name, lines, *, fssm=None):
        asm = support.translate_lines(name, lines, fssm=fssm)
        self.assertNotIn("%itx1_0", asm, f"{name} should elide %itx1_0")
        self.assertNotIn("%itx2_0", asm, f"{name} should elide %itx2_0")
        self.assertEqual(asm.count("OpImageFetch "), 1, f"{name} should emit 1 OpImageFetch")
        _validate(asm)

    def _assert_not_elided(self, name, lines, *, fssm=None):
        asm = support.translate_lines(name, lines, fssm=fssm)
        self.assertIn("%itx1_0", asm, f"{name} MUST keep %itx1_0")
        self.assertIn("%itx2_0", asm, f"{name} MUST keep %itx2_0")
        self.assertEqual(asm.count("OpImageFetch "), 3, f"{name} MUST keep all 3 OpImageFetch branches")

    def test_new_f32_and_f16_pure_float_consumers_elide(self):
        f32_cases = [
            ("fmaak_f32", ["v_fmaak_f32 v0, v0, 0x3f000000, 0x3f800000"]),
            ("fmamk_f32", ["v_fmamk_f32 v0, v0, 0x3f000000, v2"]),
            ("min3_f32", ["v_min3_f32 v0, v0, v2, 1.0"]),
            ("max3_f32", ["v_max3_f32 v0, v0, v2, 0.0"]),
            ("mac_legacy_f32", ["v_mac_legacy_f32_e32 v0, 0x3f800000, v2"]),
            ("cubema_f32", ["v_cubema_f32 v0, v0, v2, v3"]),
            ("div_fixup_f32", ["v_div_fixup_f32 v0, v0, v2, 1.0"]),
            ("ldexp_f32_src0", ["v_ldexp_f32 v0, v0, 2"]),
            ("fma_mix_f32_s0_f32", ["v_fma_mix_f32 v0, v0, 1.0, 0.0 op_sel_hi:[0,0,0]"]),
            ("mov_neg_f32", ["v_mov_b32_e64 v0, -v0"]),
            ("mov_abs_f32", ["v_mov_b32_e64 v0, |v0|"]),
            ("cndmask_neg_f32", ["v_cndmask_b32_e64 v0, -v0, v2, vcc"]),
        ]
        for label, consumer in f32_cases:
            with self.subTest(case=label):
                lines = [
                    "v_mov_b32 v4, 1",
                    "v_mov_b32 v5, 2",
                    "v_mov_b32 v2, 0x3f800000",
                    "v_mov_b32 v3, 0x40000000",
                    "image_load v0, v[4:5], s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm",
                    "s_waitcnt vmcnt(0)",
                    *consumer,
                    "exp mrt0 v0, v0, v0, v0 done vm",
                    "s_endpgm",
                ]
                self._assert_elided(f"ps_r4_{label}", lines)

        f16_cases = [
            ("pk_fma_f16", ["v_pk_fma_f16 v0, v0, v2, v3"]),
            ("min3_f16", ["v_min3_f16 v1, v0, v2, v3"]),
            ("max3_f16", ["v_max3_f16 v1, v0, v2, v3"]),
            ("cvt_pknorm_u16_f16", ["v_cvt_pknorm_u16_f16 v0, v0, v2"]),
            ("fma_mix_f16_src", ["v_fma_mix_f32 v0, v0, 1.0, 0.0 op_sel_hi:[1,0,0]"]),
            ("pack_b32_f16_neg", ["v_pack_b32_f16 v0, -v0, v2"]),
        ]
        for label, consumer in f16_cases:
            with self.subTest(case=label):
                lines = [
                    "v_mov_b32 v4, 1",
                    "v_mov_b32 v5, 2",
                    "v_mov_b32 v2, 0x3c003c00",
                    "v_mov_b32 v3, 0x3c003c00",
                    "image_load v0, v[4:5], s[0:7] dmask:0x3 dim:SQ_RSRC_IMG_2D unorm d16",
                    "s_waitcnt vmcnt(0)",
                    *consumer,
                    "exp mrt0 v1, v1, v1, v1 done vm",
                    "s_endpgm",
                ]
                self._assert_elided(f"ps_r4_d16_{label}", lines)

    def test_transitive_copy_select_pack_and_cvt_pk_elide_when_float_only(self):
        # 1. v_mov_b32 chain -> v_fmac_f32 elides.
        self._assert_elided("ps_r4_copy_fmac", [
            "v_mov_b32 v4, 1",
            "v_mov_b32 v5, 2",
            "v_mov_b32 v3, 0x3f800000",
            "image_load v0, v[4:5], s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm",
            "s_waitcnt vmcnt(0)",
            "v_mov_b32 v1, v0",
            "v_mov_b32 v2, v1",
            "v_fmac_f32_e32 v3, 0x40000000, v2",
            "exp mrt0 v3, v3, v3, v3 done vm",
            "s_endpgm",
        ])

        # 2. v_cndmask_b32 -> v_mul_f32 elides.
        self._assert_elided("ps_r4_cndmask_mul", [
            "v_mov_b32 v4, 1",
            "v_mov_b32 v5, 2",
            "v_mov_b32 v6, 0x3f800000",
            "image_load v0, v[4:5], s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm",
            "s_waitcnt vmcnt(0)",
            "v_cndmask_b32_e32 v1, v0, v6, vcc",
            "v_mul_f32_e32 v2, 0x40000000, v1",
            "exp mrt0 v2, v2, v2, v2 done vm",
            "s_endpgm",
        ])

        # 3. d16 image_load -> plain v_pack_b32_f16 -> v_pk_fma_f16 elides.
        self._assert_elided("ps_r4_d16_pack_pkfma", [
            "v_mov_b32 v4, 1",
            "v_mov_b32 v5, 2",
            "v_mov_b32 v6, 0x3c003c00",
            "image_load v0, v[4:5], s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm d16",
            "s_waitcnt vmcnt(0)",
            "v_pack_b32_f16 v1, v0, v6",
            "v_pk_fma_f16 v2, v1, v6, v6",
            "exp mrt0 v2, v2, v2, v2 done vm",
            "s_endpgm",
        ])

        # 4. v_cvt_pkrtz_f16_f32 overwrites all 32 bits of v0 after consuming v0/v1 as f32 -> buffer_store v0 ok!
        self._assert_elided("cs_r4_cvt_pkrtz_store", [
            "v_mov_b32 v4, 1",
            "v_mov_b32 v5, 2",
            "image_load v[0:1], v[4:5], s[0:7] dmask:0x3 dim:SQ_RSRC_IMG_2D unorm",
            "s_waitcnt vmcnt(0)",
            "v_cvt_pkrtz_f16_f32 v0, v0, v1",
            "buffer_store_dword v0, v4, s[8:11], 0 offen",
            "s_endpgm",
        ], fssm=_cs_fssm())

    def test_transitive_copy_select_and_partial_consumer_negative_controls(self):
        # 1. v_mov_b32 -> v_add_nc_u32 MUST NOT elide.
        self._assert_not_elided("ps_r4_neg_mov_int", [
            "v_mov_b32 v4, 1",
            "v_mov_b32 v5, 2",
            "image_load v0, v[4:5], s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm",
            "s_waitcnt vmcnt(0)",
            "v_mov_b32 v1, v0",
            "v_add_nc_u32_e32 v2, 1, v1",
            "exp mrt0 v2, v2, v2, v2 done vm",
            "s_endpgm",
        ])

        # 2. v_cndmask_b32 -> buffer_store_dword MUST NOT elide.
        self._assert_not_elided("cs_r4_neg_cndmask_store", [
            "v_mov_b32 v4, 1",
            "v_mov_b32 v5, 2",
            "v_mov_b32 v6, 0",
            "image_load v0, v[4:5], s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm",
            "s_waitcnt vmcnt(0)",
            "v_cndmask_b32_e32 v1, v0, v6, vcc",
            "buffer_store_dword v1, v4, s[8:11], 0 offen",
            "s_endpgm",
        ], fssm=_cs_fssm())

        # 3. v_mov_b32 -> exp (or exit without any float consumer) MUST NOT elide.
        self._assert_not_elided("ps_r4_neg_mov_exp", [
            "v_mov_b32 v4, 1",
            "v_mov_b32 v5, 2",
            "image_load v0, v[4:5], s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm",
            "s_waitcnt vmcnt(0)",
            "v_mov_b32 v1, v0",
            "exp mrt0 v1, v1, v1, v1 done vm",
            "s_endpgm",
        ])
        self._assert_not_elided("cs_r4_neg_mov_exit", [
            "v_mov_b32 v4, 1",
            "v_mov_b32 v5, 2",
            "image_load v0, v[4:5], s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm",
            "s_waitcnt vmcnt(0)",
            "v_mov_b32 v1, v0",
            "s_endpgm",
        ], fssm=_cs_fssm())

        # 4. d16 image_load -> plain v_pack_b32_f16 -> buffer_store_dword MUST NOT elide.
        self._assert_not_elided("cs_r4_neg_d16_pack_store", [
            "v_mov_b32 v4, 1",
            "v_mov_b32 v5, 2",
            "v_mov_b32 v6, 0",
            "image_load v0, v[4:5], s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm d16",
            "s_waitcnt vmcnt(0)",
            "v_pack_b32_f16 v1, v0, v6",
            "buffer_store_dword v1, v4, s[8:11], 0 offen",
            "s_endpgm",
        ], fssm=_cs_fssm())

        # 5. d16 image_load -> v_add_f16 + v_lshrrev_b32 MUST NOT elide (upper 16 bits preserved!).
        self._assert_not_elided("ps_r4_neg_d16_partial_shr", [
            "v_mov_b32 v4, 1",
            "v_mov_b32 v5, 2",
            "image_load v0, v[4:5], s[0:7] dmask:0x3 dim:SQ_RSRC_IMG_2D unorm d16",
            "s_waitcnt vmcnt(0)",
            "v_add_f16_e32 v0, 0x3c00, v0",
            "v_lshrrev_b32_e32 v1, 16, v0",
            "exp mrt0 v0, v1, v0, v1 done vm",
            "s_endpgm",
        ])

        # 6. v_ldexp_f32 integer exponent operand (src1) MUST NOT count as float consumer.
        self._assert_not_elided("ps_r4_neg_ldexp_exp", [
            "v_mov_b32 v4, 1",
            "v_mov_b32 v5, 2",
            "v_mov_b32 v2, 0x3f800000",
            "image_load v0, v[4:5], s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm",
            "s_waitcnt vmcnt(0)",
            "v_ldexp_f32 v1, v2, v0",
            "exp mrt0 v1, v1, v1, v1 done vm",
            "s_endpgm",
        ])

        # 7. v_cvt_pk_u8_f32 integer byte-index operand (src1) MUST NOT count as float consumer.
        self._assert_not_elided("ps_r4_neg_cvt_pk_u8_idx", [
            "v_mov_b32 v4, 1",
            "v_mov_b32 v5, 2",
            "v_mov_b32 v2, 0x3f800000",
            "v_mov_b32 v3, 0",
            "image_load v0, v[4:5], s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm",
            "s_waitcnt vmcnt(0)",
            "v_cvt_pk_u8_f32 v1, v2, v0, v3",
            "exp mrt0 v1, v1, v1, v1 done vm",
            "s_endpgm",
        ])

        # 8. v_fma_mix_f32 reading an f32-loaded VGPR as f16 (op_sel_hi:[1,0,0]) MUST NOT elide.
        self._assert_not_elided("ps_r4_neg_fma_mix_mismatch", [
            "v_mov_b32 v4, 1",
            "v_mov_b32 v5, 2",
            "image_load v0, v[4:5], s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm",
            "s_waitcnt vmcnt(0)",
            "v_fma_mix_f32 v1, v0, 1.0, 0.0 op_sel_hi:[1,0,0]",
            "exp mrt0 v1, v1, v1, v1 done vm",
            "s_endpgm",
        ])

    def test_exec_superset_register_reuse_positive_and_negative_controls(self):
        # Positive: shader modifies EXEC before image_load and widens EXEC afterward with s_or_b32;
        # v0 is consumed by v_fma_f32 and overwritten under EXEC >= E0, then reused as an integer counter.
        self._assert_elided("cs_r4_exec_superset_reuse", [
            "v_cmp_lt_u32_e32 vcc_lo, v0, 64",
            "s_and_saveexec_b32 s10, vcc_lo",
            "v_mov_b32 v4, 1",
            "v_mov_b32 v5, 2",
            "image_load v0, v[4:5], s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm",
            "s_waitcnt vmcnt(0)",
            "v_fma_f32 v0, v0, 0x40000000, 0x3f800000",
            "buffer_store_dword v0, v4, s[8:11], 0 offen",
            "s_or_b32 exec_lo, exec_lo, s10",
            "v_mov_b32 v0, 7",
            "v_add_nc_u32_e32 v0, 1, v0",
            "buffer_store_dword v0, v4, s[8:11], 0 offen",
            "s_endpgm",
        ], fssm=_cs_fssm())

        # Negative 1: EXEC narrows AFTER image_load -> v_mul_f32 only overwrites active lanes, leaving
        # image_load's v0 live in inactive lanes -> later v_add_nc_u32 reads image_load's v0 -> MUST NOT elide!
        self._assert_not_elided("cs_r4_neg_exec_narrow_after_load", [
            "v_mov_b32 v4, 1",
            "v_mov_b32 v5, 2",
            "image_load v0, v[4:5], s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm",
            "s_waitcnt vmcnt(0)",
            "v_cmp_lt_u32_e32 vcc_lo, v4, 32",
            "s_and_saveexec_b32 s10, vcc_lo",
            "v_mul_f32_e32 v0, 0x40000000, v0",
            "s_or_b32 exec_lo, exec_lo, s10",
            "v_add_nc_u32_e32 v1, 1, v0",
            "buffer_store_dword v1, v4, s[8:11], 0 offen",
            "s_endpgm",
        ], fssm=_cs_fssm())

        # Negative 2: image_load inside `if`, s_xor_b32 switches to `else` and overwrites v0 there;
        # `if` lanes STILL hold image_load's v0 after s_or_b32 merge -> v_add_nc_u32 reads it -> MUST NOT elide!
        self._assert_not_elided("cs_r4_neg_exec_xor_else_overwrite", [
            "v_mov_b32 v4, 1",
            "v_mov_b32 v5, 2",
            "v_cmp_lt_u32_e32 vcc_lo, v4, 32",
            "s_and_saveexec_b32 s10, vcc_lo",
            "image_load v0, v[4:5], s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm",
            "s_waitcnt vmcnt(0)",
            "v_mul_f32_e32 v2, 0x40000000, v0",
            "s_xor_b32 exec_lo, exec_lo, s10",
            "v_mov_b32 v0, 0",
            "s_or_b32 exec_lo, exec_lo, s10",
            "v_add_nc_u32_e32 v1, 1, v0",
            "buffer_store_dword v1, v4, s[8:11], 0 offen",
            "s_endpgm",
        ], fssm=_cs_fssm())

    def test_permlane16_without_fi_or_bc_preserves_dst_and_never_elides(self):
        # Negative control F1: v_permlane16_b32 / v_permlanex16_b32 with FI=0 and BOUND_CTRL=0 preserves vdst in
        # inactive-source lanes, so v0 still reaches v_add_nc_u32 -> MUST NOT elide!
        for op in ("v_permlane16_b32", "v_permlanex16_b32"):
            with self.subTest(op=op, mode="preserve"):
                self._assert_not_elided(f"cs_r4_neg_{op}", [
                    "v_mov_b32 v4, 1",
                    "v_mov_b32 v5, 2",
                    "v_mov_b32 v3, 0",
                    "image_load v0, v[4:5], s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm",
                    "s_waitcnt vmcnt(0)",
                    "v_add_f32_e32 v2, 1.0, v0",
                    f"{op} v0, v3, s0, s1",
                    "v_add_nc_u32_e32 v4, 1, v0",
                    "buffer_store_dword v4, v5, s[8:11], 0 offen",
                    "s_endpgm",
                ], fssm=_cs_fssm())

        # Positive control: with op_sel:[1,0] (FI=1), v_permlane16_b32 unconditionally overwrites v0 -> elides!
        self._assert_elided("cs_r4_pos_permlane16_fi1", [
            "v_mov_b32 v4, 1",
            "v_mov_b32 v5, 2",
            "v_mov_b32 v3, 0",
            "image_load v0, v[4:5], s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm",
            "s_waitcnt vmcnt(0)",
            "v_add_f32_e32 v2, 1.0, v0",
            "v_permlane16_b32 v0, v3, s0, s1 op_sel:[1,0]",
            "v_add_nc_u32_e32 v4, 1, v0",
            "buffer_store_dword v4, v5, s[8:11], 0 offen",
            "s_endpgm",
        ], fssm=_cs_fssm())


if __name__ == "__main__":
    unittest.main()
