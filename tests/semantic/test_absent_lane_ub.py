"""Regression suite for absent-lane / inactive-lane subgroup UB across all stages and ops.

Covers both static SPIR-V invariants (tests/semantic/check_absent_lane_ub.py) and strict
partial-wave execution in spirv_wave_interp.py (strict_subgroup_ub=True, active_lanes without
lane 0, single-lane waves, EXEC == 0, BDA fault_on_unmapped=True) and spirv_mini_interp.py:
  * wave32 PS (in 32-wide and 64-wide subgroups, lanes 0..3 absent, single lane, helper-only)
  * wave64 PS (lanes 0..3 absent, only upper lanes 35..38 running, EXEC == 0)
  * wave32 MS & CS (partial wave with lane 0 absent, EXEC == 0)
  * wave32 VS (partial wave with lane 0 absent, EXEC == 0, no BallotFindLSB)
  * v_readfirstlane_b32 followed by s_load_dword / s_buffer_load_dword (no unmapped BDA fault)
  * v_readlane_b32 (pure spill, mixed-VGPR definite spill, non-spill varying & dynamic)
  * v_permlane16_b32 / v_permlanex16_b32 (FI=0 and FI=1 on partial waves)
  * DPP (row_shr / row_shl out-of-bounds clamp + dpp8 fi:1 on partial waves)
  * quad_any (s_wqm_b32 / s_wqm_b64 in CS/VS/MS partial quads and PS full quads)
  * ds_append / ds_consume with EXEC == 0 and partial waves
  * Negative controls proving both spirv_wave_interp and check_absent_lane_ub catch `%uint_0`
    fallbacks and unguarded shuffles!

Hand-written synthetic listings only; no game data.
"""
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import check_absent_lane_ub as checker  # noqa: E402
import spirv_asm  # noqa: E402
import spirv_mini_interp as si  # noqa: E402
import spirv_wave_interp as wi  # noqa: E402
import support  # noqa: E402

SPIRV_AS = shutil.which("spirv-as")
SPIRV_VAL = shutil.which("spirv-val")


def fbits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def compute_fssm(threads, wave64=False, user_sgprs=4):
    """Minimal synthetic FSSM header for CS/MS tests."""
    hdr = bytearray(0x148)
    rsrc2 = (user_sgprs << 1) | 0x8080  # USER_SGPR + TGID_X + LDS
    if not wave64:
        struct.pack_into("<I", hdr, 0xD4, 1 << 25)  # wave32 bit
    struct.pack_into("<4I", hdr, 0x138, rsrc2, threads, 1, 1)
    return bytes(hdr)


def spirv_validate(spv_text):
    """Assemble and validate SPIR-V when spirv-as & spirv-val are installed; returns binary bytes or None."""
    if not (SPIRV_AS and SPIRV_VAL):
        return None
    with tempfile.TemporaryDirectory() as td:
        asm_p = os.path.join(td, "t.spvasm")
        spv_p = os.path.join(td, "t.spv")
        with open(asm_p, "w", encoding="utf-8") as f:
            f.write(spv_text)
        subprocess.run([SPIRV_AS, "--target-env", "vulkan1.3", asm_p, "-o", spv_p], check=True)
        subprocess.run([SPIRV_VAL, "--target-env", "vulkan1.3", spv_p], check=True)
        with open(spv_p, "rb") as f:
            return f.read()


class StaticCheckerSelfTests(unittest.TestCase):
    """Verify check_absent_lane_ub catches every class of absent-lane UB (negative controls)."""

    BASE = """OpCapability Shader
OpCapability GroupNonUniformBallot
OpCapability GroupNonUniformShuffle
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main" %sg_lane
OpDecorate %sg_lane BuiltIn SubgroupLocalInvocationId
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%bool = OpTypeBool
%v4uint = OpTypeVector %uint 4
%pin = OpTypePointer Input %uint
%sg_lane = OpVariable %pin Input
%true = OpConstantTrue %bool
%c0 = OpConstant %uint 0
%c3 = OpConstant %uint 3
%c5 = OpConstant %uint 5
%main = OpFunction %void None %fn
%entry = OpLabel
%l = OpLoad %uint %sg_lane
"""
    TAIL = "OpReturn\nOpFunctionEnd\n"

    def test_flags_fallback_to_constant_zero(self):
        body = (
            "%bal = OpGroupNonUniformBallot %v4uint %c3 %true\n"
            "%lsb = OpGroupNonUniformBallotFindLSB %uint %c3 %bal\n"
            "%any = OpGroupNonUniformAny %bool %c3 %true\n"
            "%first = OpSelect %uint %any %lsb %c0\n"
            "%r = OpGroupNonUniformShuffle %uint %c3 %l %first\n"
        )
        src = self.BASE + body + self.TAIL
        viols = checker.check_absent_lane_ub(src)
        self.assertEqual(len(viols), 1)
        self.assertIn("constant lane 0", viols[0])
        # Binary .spv negative control (assembled via tools/spirv_asm.py, no external tool needed).
        spv_bytes = spirv_asm.assemble(src)
        bin_viols = checker.check_absent_lane_ub(spv_bytes)
        self.assertEqual(len(bin_viols), 1)
        self.assertIn("constant lane 0", bin_viols[0])

    def test_flags_constant_lane_shuffle(self):
        src = self.BASE + "%r = OpGroupNonUniformShuffle %uint %c3 %l %c5\n" + self.TAIL
        viols = checker.check_absent_lane_ub(src)
        self.assertEqual(len(viols), 1)
        self.assertIn("constant lane 5", viols[0])
        bin_viols = checker.check_absent_lane_ub(spirv_asm.assemble(src))
        self.assertEqual(len(bin_viols), 1)
        self.assertIn("constant lane 5", bin_viols[0])

    def test_flags_quad_swap_in_compute(self):
        src = self.BASE.replace(
            "OpCapability GroupNonUniformShuffle",
            "OpCapability GroupNonUniformShuffle\nOpCapability GroupNonUniformQuad",
        ) + "%r = OpGroupNonUniformQuadSwap %uint %c3 %l %c0\n" + self.TAIL
        viols = checker.check_absent_lane_ub(src)
        self.assertEqual(len(viols), 1)
        self.assertIn("non-Fragment", viols[0])
        bin_viols = checker.check_absent_lane_ub(spirv_asm.assemble(src))
        self.assertEqual(len(bin_viols), 1)
        self.assertIn("non-Fragment", bin_viols[0])

    def test_flags_relative_and_xor_shuffles(self):
        for op in ("OpGroupNonUniformShuffleXor", "OpGroupNonUniformShuffleUp", "OpGroupNonUniformShuffleDown"):
            with self.subTest(op=op):
                cap = "GroupNonUniformShuffle" if op.endswith("Xor") else "GroupNonUniformShuffleRelative"
                src = self.BASE.replace(
                    "OpCapability GroupNonUniformShuffle",
                    f"OpCapability GroupNonUniformShuffle\nOpCapability {cap}",
                ) + f"%r = {op} %uint %c3 %l %c5\n" + self.TAIL
                self.assertEqual(len(checker.check_absent_lane_ub(src)), 1)
                self.assertEqual(len(checker.check_absent_lane_ub(spirv_asm.assemble(src))), 1)

    def test_accepts_ballot_extract_redirect_to_self(self):
        src = (
            self.BASE
            + "%bal = OpGroupNonUniformBallot %v4uint %c3 %true\n"
            + "%ok = OpGroupNonUniformBallotBitExtract %bool %c3 %bal %c5\n"
            + "%safe = OpSelect %uint %ok %c5 %l\n"
            + "%r = OpGroupNonUniformShuffle %uint %c3 %l %safe\n"
            + self.TAIL
        )
        self.assertEqual(checker.check_absent_lane_ub(src), [])
        self.assertEqual(checker.check_absent_lane_ub(spirv_asm.assemble(src)), [])

    def test_strict_wave_interp_catches_absent_lane_shuffle(self):
        body = "%r = OpGroupNonUniformShuffle %uint %c3 %l %c0\n"
        with self.assertRaisesRegex(wi.UnsupportedConstruct, r"Shuffle from invocation 0, which is not running"):
            wi.run_wave(self.BASE + body + self.TAIL, wave=32, active_lanes=[5], strict_subgroup_ub=True)


class AbsentLaneTranslatorTests(unittest.TestCase):
    """End-to-end tests: translate -> static check (text + binary) -> spirv-val -> strict partial-wave run."""

    def assert_clean_module(self, spv):
        self.assertEqual(checker.check_absent_lane_ub(spv), [])
        raw = spirv_validate(spv)
        if raw is not None:
            self.assertEqual(checker.check_absent_lane_ub(raw), [])

    def test_wave32_ps_readfirstlane_and_sload_with_absent_lane0_and_zero_exec(self):
        # Every running lane holds a valid 64-bit pointer in v[4:5]; EXEC goes to 0 before
        # v_readfirstlane_b32 + s_load_dword. Lane 0 is NOT launched (only lane 5 or lanes 4..7).
        # With the old %uint_0 fallback this read absent lane 0 and dereferenced garbage (run93).
        base = 0x2000
        spv = support.translate_lines("ps_w32_rfl_sload_zero_exec", [
            f"v_mov_b32_e32 v4, {base:#x}",
            "v_mov_b32_e32 v5, 0",
            "s_mov_b32 exec_lo, 0",
            "v_readfirstlane_b32 s10, v4",
            "v_readfirstlane_b32 s11, v5",
            "s_load_dword s12, s[10:11], 0x0",
            "s_waitcnt lgkmcnt(0)",
            "s_mov_b32 exec_lo, -1",
            "v_mov_b32_e32 v6, s12",
            "exp mrt0 v6, v6, v6, v6 done vm",
            "s_endpgm",
        ])
        self.assert_clean_module(spv)
        mem = {base: 0xCAFEBABE}
        for wave, active in ((32, [5]), (32, [31]), (32, [4, 5, 6, 7]), (64, [35, 36, 37, 38])):
            with self.subTest(wave=wave, active=active):
                res = wi.run_wave(spv, wave=wave, memory=mem, active_lanes=active,
                                  strict_subgroup_ub=True, fault_on_unmapped=True)
                for lane in active:
                    self.assertEqual(res["outputs"][lane]["%out_mrt0"], (0xCAFEBABE,) * 4)

    def test_wave64_ps_readfirstlane_and_sload_with_absent_lane0_and_zero_exec(self):
        # Wave64 PS hits the non-wave32_fragment path of v_readfirstlane_b32, which previously
        # fell back directly to %uint_0 when EXEC == 0.
        base = 0x3000
        spv = support.translate_lines("ps_w64_rfl_sload_zero_exec", [
            "s_mov_b64 exec, -1",
            f"v_mov_b32_e32 v4, {base:#x}",
            "v_mov_b32_e32 v5, 0",
            "s_mov_b64 exec, 0",
            "v_readfirstlane_b32 s10, v4",
            "v_readfirstlane_b32 s11, v5",
            "s_load_dword s12, s[10:11], 0x0",
            "s_waitcnt lgkmcnt(0)",
            "s_mov_b64 exec, -1",
            "v_mov_b32_e32 v6, s12",
            "exp mrt0 v6, v6, v6, v6 done vm",
            "s_endpgm",
        ])
        self.assert_clean_module(spv)
        mem = {base: 0x12345678}
        for active in ([5], [31], [35, 36, 37, 38], [63]):
            with self.subTest(active=active):
                res = wi.run_wave(spv, wave=64, memory=mem, active_lanes=active,
                                  strict_subgroup_ub=True, fault_on_unmapped=True)
                for lane in active:
                    self.assertEqual(res["outputs"][lane]["%out_mrt0"], (0x12345678,) * 4)

    def test_wave32_vs_and_cs_readfirstlane_with_absent_lane0_and_zero_exec(self):
        # VS (ballot_lsb_by_words path) and CS (BallotFindLSB path) must both avoid %uint_0.
        vs_spv = support.translate_lines("vs_w32_rfl_absent0", [
            "v_mov_b32_e32 v4, 0x4242",
            "s_mov_b32 exec_lo, 0",
            "v_readfirstlane_b32 s10, v4",
            "s_mov_b32 exec_lo, -1",
            "v_mov_b32_e32 v5, 0",
            "v_mov_b32_e32 v6, s10",
            "exp pos0 v5, v5, v5, v5 done",
            "exp param0 v6, v5, v5, v5",
            "s_endpgm",
        ])
        self.assert_clean_module(vs_spv)
        self.assertNotIn("OpGroupNonUniformBallotFindLSB", vs_spv)
        res = wi.run_wave(vs_spv, wave=32, active_lanes=[7, 8], strict_subgroup_ub=True,
                          builtins={"VertexIndex": lambda lane: lane, "InstanceIndex": lambda lane: 0})
        for lane in (7, 8):
            self.assertEqual(res["outputs"][lane]["%out_param0"][0], 0x4242)

        cs_spv = support.translate_lines("cs_w32_rfl_absent0", [
            "v_mov_b32_e32 v4, 0x55aa",
            "s_mov_b32 exec_lo, 0",
            "v_readfirstlane_b32 s10, v4",
            "s_mov_b32 exec_lo, -1",
            "v_mov_b32_e32 v6, s10",
            "v_mov_b32_e32 v7, 0",
            "buffer_store_dword v6, v7, s[0:3], 0 offen",
            "s_endpgm",
        ], fssm=compute_fssm(32, wave64=False))
        self.assert_clean_module(cs_spv)
        vsharp = [0x4000, 0, 0x100, 0x31016FAC]
        mem = {}
        si.run_workgroup(
            cs_spv, 1, 32, lanes=[5],
            inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (5, 0, 0), "%sg_lane": 5},
            push_constants=vsharp + [0] * 28, memory=mem,
        )
        self.assertEqual(mem.get(0x4000), 0x55AA)

    def test_mixed_vgpr_scalar_spill_and_nonspill_readlane_absent_lanes(self):
        # v48 is used as a vector register FIRST, then reused as a constant-lane scalar spill
        # (lanes 0 and 19); definite_spill_reads parks those reloads so absent lanes 0/19 stay exact!
        # v12 is a non-spill varying VGPR read from absent lane 0; guarded shuffle redirects to first
        # running lane (wave-uniform, zero UB).
        spv = support.translate_lines("ps_w32_mixed_spill_and_readlane", [
            "v_mov_b32_e32 v48, 0x9999",        # vector write disqualifies pure scalar_spills()
            "s_mov_b32 s20, 0x11223344",
            "s_mov_b32 s21, 0x55667788",
            "v_writelane_b32 v48, s20, 0",
            "v_writelane_b32 v48, s21, 19",
            "s_mov_b32 s20, 0",
            "s_mov_b32 s21, 0",
            "v_readlane_b32 s4, v48, 0",        # definite mixed-VGPR spill reload -> parked
            "v_readlane_b32 s5, v48, 19",       # definite mixed-VGPR spill reload -> parked
            "v_mov_b32_e32 v12, 0x7777",        # non-spill VGPR read from absent lane 0 -> guarded
            "v_readlane_b32 s6, v12, 0",
            "v_mov_b32_e32 v6, s4",
            "v_mov_b32_e32 v7, s5",
            "v_mov_b32_e32 v8, s6",
            "exp mrt0 v6, v7, v8, v8 done vm",
            "s_endpgm",
        ])
        self.assert_clean_module(spv)
        # Only one shuffle (for v12), zero shuffles for mixed spill v48!
        self.assertEqual(spv.count("OpGroupNonUniformShuffle"), 1)
        for wave, active in ((32, [5]), (32, [4, 5, 6, 7]), (64, [37, 38])):
            with self.subTest(wave=wave, active=active):
                res = wi.run_wave(spv, wave=wave, active_lanes=active, strict_subgroup_ub=True)
                for lane in active:
                    self.assertEqual(res["outputs"][lane]["%out_mrt0"],
                                     (0x11223344, 0x55667788, 0x7777, 0x7777))

    def test_permlane16_and_permlanex16_fi1_absent_lanes(self):
        # FI=1 reads EXEC-off launched lanes, and must redirect unlaunched/absent lanes to own
        # subgroup lane (yielding 0) without triggering UB.
        spv = support.translate_lines("ps_w32_permlane_fi1_absent", [
            "v_mbcnt_lo_u32_b32 v1, -1, 0",
            "v_add_nc_u32_e32 v2, 100, v1",
            # lane 5 reads lane 0 (absent) via permlane16 FI=1, and lane 21 (absent) via permlanex16 FI=1
            "v_permlane16_b32 v6, v2, 0, 0 op_sel:[1,0]",
            "v_permlanex16_b32 v7, v2, 0x55555555, 0x55555555 op_sel:[1,0]",
            # lane 5 reads lane 5 (present!) via permlane16 FI=1
            "v_permlane16_b32 v8, v2, 0x55555555, 0x55555555 op_sel:[1,0]",
            "exp mrt0 v6, v7, v8, v2 done vm",
            "s_endpgm",
        ])
        self.assert_clean_module(spv)
        res = wi.run_wave(spv, wave=32, active_lanes=[5], strict_subgroup_ub=True)
        # v_mbcnt_lo_u32_b32 with -1 gives lane id 5 -> v2 = 105.
        self.assertEqual(res["outputs"][5]["%out_mrt0"], (0, 0, 105, 105))

    def test_dpp_in_bounds_and_fi1_absent_lanes(self):
        # row_shr:1 in lane 16 is out-of-bounds (previously clamped safe_sg to %uint_0!);
        # dpp8 fi:1 reads absent neighbours in partial wave (previously unguarded!).
        spv = support.translate_lines("ps_w32_dpp_absent", [
            "v_mov_b32_e32 v2, 50",
            "v_mov_b32_e32 v6, 99",
            "v_add_nc_u32_dpp v6, v2, 1 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1",
            "v_add_nc_u32_dpp v7, v2, 2 dpp8:[0,0,0,0,0,0,0,0] fi:1",
            "exp mrt0 v6, v7, v2, v2 done vm",
            "s_endpgm",
        ])
        self.assert_clean_module(spv)
        # Only lane 16 running: row_shr:1 is OOB -> 0+1 = 1; dpp8 lane 0 of group (lane 16) IS present -> 50+2 = 52!
        res16 = wi.run_wave(spv, wave=32, active_lanes=[16], strict_subgroup_ub=True)
        self.assertEqual(res16["outputs"][16]["%out_mrt0"], (1, 52, 50, 50))
        # Only lane 17 running: row_shr:1 reads absent lane 16 -> 0+1 = 1; dpp8 reads absent lane 16 -> 0+2 = 2!
        res17 = wi.run_wave(spv, wave=32, active_lanes=[17], strict_subgroup_ub=True)
        self.assertEqual(res17["outputs"][17]["%out_mrt0"], (1, 2, 50, 50))

    def test_quad_any_partial_quad_in_vs_and_cs(self):
        # Partial quad (only lane 5 running in quad 4..7): VS and CS use OpGroupNonUniformBallot
        # (never QuadSwap or unguarded 4x Shuffle).
        vs_spv = support.translate_lines("vs_w32_wqm_partial", [
            "v_cmp_eq_u32_e32 vcc_lo, 5, v0",
            "s_wqm_b32 s4, vcc_lo",
            "v_cndmask_b32_e64 v6, 0, 1, s4",
            "v_mov_b32_e32 v5, 0",
            "exp pos0 v5, v5, v5, v5 done",
            "exp param0 v6, v5, v5, v5",
            "s_endpgm",
        ])
        self.assert_clean_module(vs_spv)
        self.assertNotIn("OpGroupNonUniformShuffle", vs_spv)
        self.assertNotIn("OpGroupNonUniformQuadSwap", vs_spv)
        res = wi.run_wave(vs_spv, wave=32, active_lanes=[5, 6], strict_subgroup_ub=True,
                          builtins={"VertexIndex": lambda lane: lane, "InstanceIndex": lambda lane: 0})
        self.assertEqual(res["outputs"][5]["%out_param0"][0], 1)
        self.assertEqual(res["outputs"][6]["%out_param0"][0], 1)

        cs_spv = support.translate_lines("cs_w32_wqm_partial", [
            "v_cmp_eq_u32_e32 vcc_lo, 5, v0",
            "s_wqm_b32 s4, vcc_lo",
            "v_cndmask_b32_e64 v6, 0, 1, s4",
            "v_lshlrev_b32_e32 v7, 2, v0",
            "buffer_store_dword v6, v7, s[0:3], 0 offen",
            "s_endpgm",
        ], fssm=compute_fssm(5, wave64=False))
        self.assert_clean_module(cs_spv)
        self.assertNotIn("OpGroupNonUniformQuadSwap", cs_spv)
        mem = {}
        si.run_workgroup(
            cs_spv, 2, 32, lanes=[5, 6],
            inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": ([5, 6][i], 0, 0), "%sg_lane": [5, 6][i]},
            push_constants=[0x1000, 0, 0x100, 0x31016FAC] + [0] * 28, memory=mem,
        )
        self.assertEqual(mem.get(0x1000 + 20), 1)
        self.assertEqual(mem.get(0x1000 + 24), 1)

    def test_ds_append_zero_exec_and_absent_lane0(self):
        # ds_append with EXEC == 0 in a partial wave where lane 0 is absent previously masked
        # first=64 & 31 -> 0 and shuffled from absent lane 0!
        cs_spv = support.translate_lines("cs_w32_ds_append_zero_exec", [
            "v_mov_b32_e32 v8, 0xdead",
            "s_mov_b32 exec_lo, 0",
            "ds_append v8",
            "s_mov_b32 exec_lo, -1",
            "v_mov_b32_e32 v7, 0",
            "buffer_store_dword v8, v7, s[0:3], 0 offen",
            "s_endpgm",
        ], fssm=compute_fssm(32, wave64=False))
        self.assert_clean_module(cs_spv)
        mem = {}
        si.run_workgroup(
            cs_spv, 1, 32, lanes=[5],
            inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (5, 0, 0), "%sg_lane": 5},
            push_constants=[0x2000, 0, 0x100, 0x31016FAC] + [0] * 28, memory=mem,
        )
        self.assertEqual(mem.get(0x2000), 0xDEAD)

    def test_wave32_mesh_shader_readfirstlane_sload_and_wqm(self):
        # Synthetic wave32 mesh shader (like the run93 MS shape) exercising v_readfirstlane_b32
        # feeding s_load_dword, s_wqm_b32, and v_readlane_b32: zero unsafe shuffles, passes spirv-val.
        r = bytearray(48)
        struct.pack_into("<I", r, 0, 256)
        struct.pack_into("<HH", r, 12, 32, 32)
        r[24], r[27] = 13, 2
        struct.pack_into("<III", r, 36, 32, 1, 1)
        part = b"PSV0" + struct.pack("<I", 4 + len(r)) + struct.pack("<I", len(r)) + bytes(r)
        head = 36
        dxbc = b"DXBC" + bytes(16) + struct.pack("<IIII", 1, head + len(part), 1, head) + part
        ms_spv = support.translate_lines("ms_w32_absent_lane_safe", [
            "v_mov_b32_e32 v4, 0x2000",
            "v_mov_b32_e32 v5, 0",
            "s_mov_b32 exec_lo, 0",
            "v_readfirstlane_b32 s10, v4",
            "v_readfirstlane_b32 s11, v5",
            "s_load_dword s12, s[10:11], 0x0",
            "s_waitcnt lgkmcnt(0)",
            "s_mov_b32 exec_lo, -1",
            "v_cmp_eq_u32_e32 vcc_lo, 5, v0",
            "s_wqm_b32 s14, vcc_lo",
            "v_readlane_b32 s15, v4, 0",
            "s_endpgm",
        ], dxbc=dxbc)
        self.assert_clean_module(ms_spv)
        self.assertNotIn("OpGroupNonUniformQuadSwap", ms_spv)

    def test_mixed_vgpr_spill_killed_by_vector_write_or_dynamic_writelane(self):
        # Soundness check for lane_masks.definite_spill_reads: overwriting v48 as a vector or with
        # a dynamic v_writelane_b32 must invalidate the parked slot so the later v_readlane_b32
        # reads the overwritten VGPR value rather than the stale parked scalar.
        spv_vec = support.translate_lines("ps_w32_mixed_spill_killed_vec", [
            "s_mov_b32 s20, 0x11111111",
            "v_writelane_b32 v48, s20, 0",
            "v_mov_b32_e32 v48, 0x22222222",    # kills parked slot (48, 0)
            "v_readlane_b32 s4, v48, 0",
            "v_mov_b32_e32 v6, s4",
            "exp mrt0 v6, v6, v6, v6 done vm",
            "s_endpgm",
        ])
        self.assert_clean_module(spv_vec)
        res = wi.run_wave(spv_vec, wave=32, active_lanes=[0, 1], strict_subgroup_ub=True)
        self.assertEqual(res["outputs"][0]["%out_mrt0"], (0x22222222,) * 4)

        spv_dyn = support.translate_lines("ps_w32_mixed_spill_killed_dyn", [
            "s_mov_b32 s20, 0x11111111",
            "s_mov_b32 s21, 0x33333333",
            "s_mov_b32 s22, 0",
            "v_writelane_b32 v48, s20, 0",
            "v_writelane_b32 v48, s21, s22",    # dynamic lane write kills all (48, *) slots
            "v_readlane_b32 s4, v48, 0",
            "v_mov_b32_e32 v6, s4",
            "exp mrt0 v6, v6, v6, v6 done vm",
            "s_endpgm",
        ])
        self.assert_clean_module(spv_dyn)
        res_d = wi.run_wave(spv_dyn, wave=32, active_lanes=[0, 1], strict_subgroup_ub=True)
        self.assertEqual(res_d["outputs"][0]["%out_mrt0"], (0x33333333,) * 4)

    def test_negative_control_sabotaged_fallback_caught_by_both_checker_and_interp(self):
        # Sabotage wave32_first_running_lane's fallback back to %c0 (the run93 bug!) and prove
        # BOTH check_absent_lane_ub AND spirv_wave_interp catch it immediately.
        base = 0x2000
        spv = support.translate_lines("ps_w32_neg_ctrl", [
            f"v_mov_b32_e32 v4, {base:#x}",
            "v_mov_b32_e32 v5, 0",
            "s_mov_b32 exec_lo, 0",
            "v_readfirstlane_b32 s10, v4",
            "s_mov_b32 exec_lo, -1",
            "v_mov_b32_e32 v6, s10",
            "exp mrt0 v6, v6, v6, v6 done vm",
            "s_endpgm",
        ])
        self.assertEqual(checker.check_absent_lane_ub(spv), [])
        # Replace the OpSelect fallback in v_readfirstlane_b32's first-lane select with %c0.
        bad, n = re.subn(
            r"(%\w+ = OpSelect %uint %\w+ %\w+ )%\w+(\n\s*%\w+ = OpLoad %uint %sg_lane\n\s*%\w+ = OpBitwiseAnd %uint %\w+ %c20)",
            r"\1%c0\2",
            spv,
            count=1,
        )
        self.assertEqual(n, 1)
        viols = checker.check_absent_lane_ub(bad)
        self.assertTrue(viols, "static checker must flag sabotaged %c0 fallback")
        bin_viols = checker.check_absent_lane_ub(spirv_asm.assemble(bad))
        self.assertTrue(bin_viols, "binary static checker must flag sabotaged %c0 fallback")
        with self.assertRaisesRegex(wi.UnsupportedConstruct, r"Shuffle from invocation 0, which is not running"):
            wi.run_wave(bad, wave=32, active_lanes=[5], strict_subgroup_ub=True)

    def test_hand_written_subgroup_fixtures_pass_static_checker(self):
        # Every hand-written subgroup fixture in tests/semantic/fixtures/disasm/ must emit zero
        # unsafe subgroup shuffles/broadcasts/non-fragment quad ops.
        fixtures = [
            ("cs_w32_lane_wrap", compute_fssm(32, wave64=False)),
            ("cs_w32_permlane_fi", compute_fssm(32, wave64=False)),
            ("cs_w32_ds_append", compute_fssm(32, wave64=False)),
            ("cs_w64_ds_append", compute_fssm(64, wave64=True)),
            ("cs_w64x_readlane", compute_fssm(64, wave64=True)),
            ("cs_w64x_writelane", compute_fssm(64, wave64=True)),
            ("cs_w64x_bpermute", compute_fssm(64, wave64=True)),
            ("cs_w64x_partial", compute_fssm(40, wave64=True)),
            ("cs_w64x_readfirstlane", compute_fssm(64, wave64=True)),
            ("cs_wavemask", compute_fssm(64, wave64=True)),
        ]
        for name, fssm in fixtures:
            with self.subTest(fixture=name):
                spv = support.translate_fixture(name, fssm=fssm)
                self.assertEqual(checker.check_absent_lane_ub(spv), [])


if __name__ == "__main__":
    unittest.main()
