"""Tests for intra-block %vgpr/%sgpr value forwarding in tools/rdna2spirv.py."""
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import support  # noqa: E402
from support import rdna2spirv  # noqa: E402
from spirv_wave_interp import run_wave  # noqa: E402


def _make_fssm_cs(user_sgprs=4, tgid_mask=1, tid_cnt=0, threads=(64, 1, 1), wave32=True):
    buf = bytearray(0x160)
    buf[0xD4] = 0x20 if wave32 else 0x00
    rsrc2 = ((user_sgprs & 31) << 1) | ((tgid_mask & 7) << 7) | ((tid_cnt & 3) << 11)
    struct.pack_into("<IIII", buf, 0x138, rsrc2, threads[0], threads[1], threads[2])
    return bytes(buf)


def _encode_listing(name, body_lines):
    lbl_idx = {}
    ins_count = 0
    for ln in body_lines:
        if ln.endswith(":"):
            lbl_idx[ln[:-1]] = ins_count
        else:
            ins_count += 1
    out = [f"; {name}  synthetic reg-forwarding test"]
    idx = 0
    for ln in body_lines:
        if ln.endswith(":"):
            continue
        enc = 0
        if ln.startswith(("s_cbranch_", "s_branch ")):
            op, target = ln.split()
            rel = (lbl_idx[target] - idx - 1) & 0xFFFF
            enc = 0xBF880000 | rel
            signed = rel if rel < 0x8000 else rel - 0x10000
            ln = f"{op} {signed}"
        b = enc.to_bytes(4, "little")
        enc_s = ",".join(f"0x{x:02x}" for x in b)
        out.append(f"\t{ln:<44} ; encoding: [{enc_s}]")
        idx += 1
    return ("\n".join(out) + "\n").encode("utf-8")


def _validate_spirv(spvasm):
    sa = shutil.which("spirv-as")
    sv = shutil.which("spirv-val")
    if not (sa and sv):
        if os.environ.get("LEONIDA_REQUIRE_SPIRV_TOOLS"):
            raise AssertionError("spirv-as / spirv-val required but not found on PATH")
        return
    with tempfile.TemporaryDirectory() as td:
        asm_p = os.path.join(td, "mod.spvasm")
        spv_p = os.path.join(td, "mod.spv")
        with open(asm_p, "w", encoding="utf-8", newline="\n") as f:
            f.write(spvasm)
        subprocess.run([sa, "--target-env", "vulkan1.2", asm_p, "-o", spv_p], check=True)
        subprocess.run([sv, "--target-env", "vulkan1.2", spv_p], check=True)


def _count_reg_loads(spvasm):
    ptrs = {}
    loads = 0
    for raw in spvasm.splitlines():
        s = raw.strip()
        if "OpAccessChain %_ptr_Function_uint %vgpr " in s or "OpAccessChain %_ptr_Function_uint %sgpr " in s:
            ptrs[s.split()[0]] = True
        elif "= OpLoad %uint " in s and s.split()[-1] in ptrs:
            loads += 1
    return loads


class RegLoadForwardingTests(unittest.TestCase):
    def test_same_block_forwarding_and_masked_exec_semantics(self):
        """Repeated reads/writes to the same VGPR/SGPR forward SSA values and preserve masked lanes."""
        body = [
            "s_mov_b32 s4, 7",
            "v_mov_b32 v0, 10",
            "v_mov_b32 v1, 20",
            "v_mov_b32 v2, 30",
            "v_mov_b32 v3, 0",
            "v_writelane_b32 v3, 1, 5",
            "v_writelane_b32 v3, 1, 17",
            # Mask off lanes 5 and 17
            "v_cmp_eq_u32 vcc_lo, v3, 0",
            "s_and_saveexec_b32 s5, vcc_lo",
            # Heavy read-after-write & self-aliasing chain in one basic block
            "v_add_nc_u32 v0, v0, v0",
            "v_add_nc_u32 v0, v0, s4",
            "v_add_nc_u32 v0, v0, s4",
            "v_add_nc_u32 v1, v1, v0",
            "v_add_nc_u32 v2, v2, v1",
            "v_add_nc_u32 v2, v2, s4",
            # Restore EXEC and update v1 for all lanes (must forward masked result for lanes 5/17!)
            "s_or_b32 exec_lo, exec_lo, s5",
            "v_add_nc_u32 v1, v1, 1",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spvasm = support.translate_lines("ps_reg_fwd_chain", body)
        _validate_spirv(spvasm)

        # Forwarding eliminates redundant intra-block reloads (drops loads from 22 on main to 10 across the
        # active_run blocks introduced by #3549).
        self.assertLessEqual(_count_reg_loads(spvasm), 10, f"too many register loads: {_count_reg_loads(spvasm)}")

        res = run_wave(spvasm, wave=32)
        for lane in range(32):
            if lane in (5, 17):
                self.assertEqual(tuple(res["outputs"][lane]["%out_mrt0"]), (10, 21, 30, 1))
            else:
                v0 = 10 + 10 + 7 + 7       # 34
                v1 = 20 + v0 + 1           # 55
                v2 = 30 + (20 + v0) + 7    # 91
                self.assertEqual(tuple(res["outputs"][lane]["%out_mrt0"]), (v0, v1, v2, 0))

    def test_divergent_loop_reloads_at_block_boundary_and_reduces_loads(self):
        """Register cache clears at every basic-block boundary so loop-carried values stay exact."""
        body = [
            "s_mov_b32 s4, 0",
            "s_mov_b32 s5, 4",
            "v_mov_b32 v0, 1",
            "v_mov_b32 v1, 2",
            "v_mov_b32 v2, 3",
            "v_mov_b32 v3, 0",
            "v_writelane_b32 v3, 1, 9",
            "BB_loop:",
            "v_cmp_eq_u32 vcc_lo, v3, 0",
            "s_and_saveexec_b32 s6, vcc_lo",
            "s_cbranch_execz BB_skip",
            "v_add_nc_u32 v0, v0, s4",
            "v_add_nc_u32 v0, v0, s4",
            "v_add_nc_u32 v1, v1, v0",
            "v_add_nc_u32 v2, v2, v1",
            "v_add_nc_u32 v2, v2, v0",
            "BB_skip:",
            "s_or_b32 exec_lo, exec_lo, s6",
            "s_add_u32 s4, s4, 1",
            "s_cmp_lt_u32 s4, s5",
            "s_cbranch_scc1 BB_loop",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spvasm = support.translate_listing("ps_reg_fwd_loop", _encode_listing("ps_reg_fwd_loop", body))
        _validate_spirv(spvasm)

        res = run_wave(spvasm, wave=32)
        v0, v1, v2 = 1, 2, 3
        for s4 in range(4):
            v0 = (v0 + s4 + s4) & 0xFFFFFFFF
            v1 = (v1 + v0) & 0xFFFFFFFF
            v2 = (v2 + v1 + v0) & 0xFFFFFFFF
        for lane in range(32):
            want = (1, 2, 3, 1) if lane == 9 else (v0, v1, v2, 0)
            self.assertEqual(tuple(res["outputs"][lane]["%out_mrt0"]), want)

    def test_dynamic_movreld_and_swaprel_invalidate_cached_vgprs(self):
        """v_movreld_b32, v_movrelsd_b32, v_movrelsd_2_b32, and v_swaprel_b32 invalidate cached VGPRs."""
        lines = [
            "v_mov_b32 v4, 0x11",
            "v_mov_b32 v5, 0x22",
            "v_mov_b32 v6, 0x33",
            "v_mov_b32 v7, 0x99",
            # Dynamic store: writes v[4 + 1] = v5 = 0x99; must invalidate cached v5=0x22 immediately.
            "s_mov_b32 m0, 1",
            "v_movreld_b32 v4, v7",
            # Read v5 BEFORE swaprel so removing v_movreld invalidation fails directly;
            # also read v4 and v6 so both are cached right before v_swaprel_b32.
            "v_mov_b32 v8, v5",
            "v_add_nc_u32 v9, v4, v6",
            # Swap v[4 + 2] (v6=0x33) and v[4 + 0] (v4=0x11) -> v4=0x33, v6=0x11; must invalidate cached v4/v6!
            "s_mov_b32 m0, 0x00020000",
            "v_swaprel_b32 v4, v4",
            # Dynamic read of v[4 + 1] (v5=0x99) after movreld -> v3=0x99 (verifies OpStore landed in %vgpr!)
            "s_mov_b32 m0, 1",
            "v_movrels_b32 v3, v4",
            "v_mov_b32 v0, v4",
            "v_mov_b32 v1, v8",
            "v_mov_b32 v2, v6",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spvasm = support.translate_lines("ps_reg_fwd_movrel", lines)
        _validate_spirv(spvasm)
        self.assertEqual(support.mrt0_words(spvasm), [0x33, 0x99, 0x11, 0x99])

    def test_single_masked_write_outside_active_run_and_single_op_loop(self):
        """A single masked dst() write caches the OpSelect result (not raw val), and single-op loops reset cache."""
        # Part 1: single VALU op (< MIN_ACTIVE_RUN) under partial EXEC uses dst() OpSelect, not if (active).
        body1 = [
            "v_mov_b32 v0, 10",
            "v_mov_b32 v1, 0",
            "v_writelane_b32 v1, 1, 3",
            "v_cmp_eq_u32 vcc_lo, v1, 0",
            "s_and_saveexec_b32 s4, vcc_lo",
            "v_add_nc_u32 v0, v0, 50",
            "s_or_b32 exec_lo, exec_lo, s4",
            "v_add_nc_u32 v2, v0, 1",
            "exp mrt0 v0, v1, v2, v2 done vm",
            "s_endpgm",
        ]
        spv1 = support.translate_lines("ps_reg_fwd_single_masked", body1)
        _validate_spirv(spv1)
        res1 = run_wave(spv1, wave=32)
        for lane in range(32):
            want = (10, 1, 11, 11) if lane == 3 else (60, 0, 61, 61)
            self.assertEqual(tuple(res1["outputs"][lane]["%out_mrt0"]), want)

        # Part 2: loop body with a single lane-local instruction has no active-run label before the first read.
        body2 = [
            "s_mov_b32 s4, 0",
            "s_mov_b32 s5, 5",
            "v_mov_b32 v0, 100",
            "BB_loop:",
            "v_add_nc_u32 v0, v0, 7",
            "s_add_u32 s4, s4, 1",
            "s_cmp_lt_u32 s4, s5",
            "s_cbranch_scc1 BB_loop",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv2 = support.translate_listing("ps_reg_fwd_single_loop", _encode_listing("ps_reg_fwd_single_loop", body2))
        _validate_spirv(spv2)
        res2 = run_wave(spv2, wave=32)
        for lane in range(32):
            self.assertEqual(tuple(res2["outputs"][lane]["%out_mrt0"]), (135, 135, 135, 135))

    def test_dynamic_s_movrels_reads_stored_sgpr_values(self):
        """store_reg still writes %sgpr so s_movrels_b32 reads the updated scalar values in the same block."""
        lines = [
            "s_mov_b32 s10, 0x101",
            "s_mov_b32 s11, 0x202",
            "s_mov_b32 s12, 0x303",
            "s_mov_b32 m0, 2",
            "s_movrels_b32 s13, s10",
            "v_mov_b32 v0, s13",
            "v_mov_b32 v1, s10",
            "v_mov_b32 v2, s11",
            "v_mov_b32 v3, s12",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spvasm = support.translate_lines("ps_reg_fwd_smovrels", lines)
        _validate_spirv(spvasm)
        self.assertEqual(support.mrt0_words(spvasm), [0x303, 0x101, 0x202, 0x303])

    def test_wave64_emulate_validates_with_reg_forwarding(self):
        """Cached register values never leak across x_split segments under --wave64-emulate."""
        body = [
            "s_mov_b32 s4, 0",
            "s_mov_b32 s5, 2",
            "v_mov_b32 v0, 1",
            "v_mov_b32 v1, 2",
            "BB_loop:",
            "v_cmp_gt_u32 vcc, v0, 0",
            "s_and_saveexec_b64 s[6:7], vcc",
            "s_cbranch_execz BB_skip",
            "v_add_nc_u32 v0, v0, v1",
            "v_add_nc_u32 v1, v1, v0",
            "v_add_nc_u32 v0, v0, s4",
            "BB_skip:",
            "s_or_b64 exec, exec, s[6:7]",
            "s_add_u32 s4, s4, 1",
            "s_cmp_lt_u32 s4, s5",
            "s_cbranch_scc1 BB_loop",
            "s_endpgm",
        ]
        fssm = _make_fssm_cs(user_sgprs=4, tgid_mask=1, tid_cnt=0, threads=(64, 1, 1), wave32=False)
        old_emu = rdna2spirv.WAVE64_EMULATE
        try:
            rdna2spirv.WAVE64_EMULATE = True
            spvasm = support.translate_listing("cs_reg_fwd_w64", _encode_listing("cs_reg_fwd_w64", body), fssm=fssm)
        finally:
            rdna2spirv.WAVE64_EMULATE = old_emu
        _validate_spirv(spvasm)

    def test_direct_gen_defaults_to_uncached_reg_loads(self):
        """Direct Gen() unit-test callers default to cache_regs=False for backwards compatibility."""
        g = rdna2spirv.Gen("cs")
        a = g.load_reg("vgpr", 3)
        b = g.load_reg("vgpr", 3)
        self.assertNotEqual(a, b)
        self.assertEqual(sum(1 for ln in g.body if "= OpLoad %uint " in ln), 2)


if __name__ == "__main__":
    unittest.main()
