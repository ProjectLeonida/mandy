"""Compile-time literal S# samplers via OpSpecConstant (Item 3B.5).

For each compile-time literal S# in `meta["literal_samplers"]` (exact sorted list order):
  * Declares `OpSpecConstant %uint 4294967295` (`0xFFFFFFFF`) with `OpDecorate %lit_smp_<i> SpecId <i>`.
  * Emits `meta["literal_sampler_spec_base"] = 0`.
  * At each sample/gather/get_lod site using that literal S#, branches on `%lit_smp_<i> != 0xFFFFFFFF`:
      - When specialized (`!= 0xFFFFFFFF`), uses `%lit_smp_<i>` directly (bypassing `%desc_lookup_fn`).
      - When unspecialized (`0xFFFFFFFF`), falls back to the bindless hash-table probe.
"""
import os
import re
import shutil
import subprocess
import tempfile
import unittest

import support
import rdna2spirv
import spirv_asm

S0 = (0x00000092, 0x00C00000, 0x00000000, 0x00000000)   # point filter literal S#
S1 = (0x00000092, 0x00C00000, 0x00500000, 0x00000000)   # bilinear filter literal S#


def translate_with_meta(name, lines):
    asm = support.translate_lines(name, lines)
    meta = rdna2spirv.shader_meta(rdna2spirv.translate.last_gen, asm)
    return asm, meta


def validate_v12(testcase, asm, label):
    if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
        return
    with tempfile.TemporaryDirectory() as tmp:
        sa = os.path.join(tmp, f"{label}.spvasm")
        sv = os.path.join(tmp, f"{label}.spv")
        with open(sa, "w", encoding="utf-8") as f:
            f.write(asm)
        res = rdna2spirv.assemble_spirv(sa, sv)
        testcase.assertEqual(res.returncode, 0, f"{label} spirv-as/val failed:\n{res.stderr}\n{res.stdout}")
        sv_py = os.path.join(tmp, f"{label}_py.spv")
        with open(sv_py, "wb") as f:
            f.write(spirv_asm.assemble(asm, target_env="vulkan1.2"))
        r2 = subprocess.run(["spirv-val", "--target-env", "vulkan1.2", sv_py], capture_output=True, text=True)
        testcase.assertEqual(r2.returncode, 0, f"{label} spirv_asm+val failed:\n{r2.stderr}\n{r2.stdout}")


def eval_sampler_select(asm, spec_val, hash_result):
    """Evaluate the spec-constant selection branch for %lit_smp_0 and return (selected_index, probed_hash)."""
    defs = {}
    order = []
    for line in asm.splitlines():
        m = re.match(r"\s*(%\w+) = (Op\w+)\s*(.*)$", line)
        if m:
            defs[m.group(1)] = (m.group(2), m.group(3).split())
            order.append((m.group(1), m.group(2), m.group(3).split()))
        else:
            m2 = re.match(r"\s*(OpBranchConditional)\s+(.*)$", line)
            if m2:
                order.append((None, m2.group(1), m2.group(2).split()))
    # Locate the OpINotEqual comparing %lit_smp_0 against 0xFFFFFFFF and its merge OpPhi.
    cmp_id = next(r for r, op, a in order if op == "OpINotEqual" and "%lit_smp_0" in a)
    br = next(a for _, op, a in order if op == "OpBranchConditional" and a[0] == cmp_id)
    fast_lbl, probe_lbl = br[1], br[2]
    phi_id, _, phi_args = next((r, op, a) for r, op, a in order if op == "OpPhi" and "%lit_smp_0" in a)
    # %lit_smp_0 != 0xFFFFFFFF branches straight to fast_lbl (skipping probe_lbl).
    is_specialized = (spec_val & 0xFFFFFFFF) != 0xFFFFFFFF
    if is_specialized:
        assert phi_args[1] == "%lit_smp_0"
        return spec_val, False
    # Unspecialized path enters probe_lbl, runs %desc_lookup_fn, and arrives from probe_lbl.
    assert probe_lbl in phi_args
    # Verify the probe block actually calls %desc_lookup_fn.
    probe_idx = next(i for i, (r, op, _) in enumerate(order) if r == probe_lbl and op == "OpLabel")
    fast_idx = next(i for i, (r, op, _) in enumerate(order) if r == fast_lbl and op == "OpLabel")
    probe_calls = [a for _, op, a in order[probe_idx:fast_idx] if op == "OpFunctionCall" and a[1] == "%desc_lookup_fn"]
    assert len(probe_calls) == 1, f"expected 1 hash lookup call in probe block, got {probe_calls}"
    return hash_result, True


class LiteralSamplerSpecConstantTests(unittest.TestCase):
    def test_single_literal_sampler_spec_constant_and_both_paths(self):
        lines = [
            f"s_mov_b32 s12, {S0[0]:#x}",
            f"s_mov_b32 s13, {S0[1]:#x}",
            f"s_mov_b32 s14, {S0[2]:#x}",
            f"s_mov_b32 s15, {S0[3]:#x}",
            "v_mov_b32 v4, 0x3f000000",
            "v_mov_b32 v5, 0x3f000000",
            "image_sample v[0:3], v[4:5], s[0:7], s[12:15] dmask:0xf dim:SQ_RSRC_IMG_2D",
            "s_waitcnt vmcnt(0)",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        asm, meta = translate_with_meta("ps_lit_smp1", lines)
        self.assertEqual(meta["literal_samplers"], [list(S0)])
        self.assertEqual(meta["literal_sampler_spec_base"], 0)
        self.assertIn("OpDecorate %lit_smp_0 SpecId 0", asm)
        self.assertIn("%lit_smp_0 = OpSpecConstant %uint 4294967295", asm)
        validate_v12(self, asm, "lit_smp1")

        # Specialized path: uses spec constant directly without calling %desc_lookup_fn.
        idx_spec, probed_spec = eval_sampler_select(asm, spec_val=7, hash_result=99)
        self.assertEqual((idx_spec, probed_spec), (7, False))

        # Unspecialized path (0xFFFFFFFF): falls back to %desc_lookup_fn hash probe.
        idx_fb, probed_fb = eval_sampler_select(asm, spec_val=0xFFFFFFFF, hash_result=13)
        self.assertEqual((idx_fb, probed_fb), (13, True))

    def test_two_literal_samplers_preserve_exact_sorted_order(self):
        # Emit S1 before S0 in instruction order; literal_samplers is sorted so S0 -> SpecId 0, S1 -> SpecId 1.
        lines = [
            f"s_mov_b32 s12, {S1[0]:#x}",
            f"s_mov_b32 s13, {S1[1]:#x}",
            f"s_mov_b32 s14, {S1[2]:#x}",
            f"s_mov_b32 s15, {S1[3]:#x}",
            f"s_mov_b32 s16, {S0[0]:#x}",
            f"s_mov_b32 s17, {S0[1]:#x}",
            f"s_mov_b32 s18, {S0[2]:#x}",
            f"s_mov_b32 s19, {S0[3]:#x}",
            "v_mov_b32 v4, 0x3f000000",
            "v_mov_b32 v5, 0x3f000000",
            "image_sample v[0:3], v[4:5], s[0:7], s[12:15] dmask:0xf dim:SQ_RSRC_IMG_2D",
            "image_sample_lz v[6:9], v[4:5], s[0:7], s[16:19] dmask:0xf dim:SQ_RSRC_IMG_2D",
            "s_waitcnt vmcnt(0)",
            "exp mrt0 v0, v6, v1, v7 done vm",
            "s_endpgm",
        ]
        asm, meta = translate_with_meta("ps_lit_smp2", lines)
        self.assertEqual(meta["literal_samplers"], [list(S0), list(S1)])
        self.assertEqual(meta["literal_sampler_spec_base"], 0)
        self.assertIn("OpDecorate %lit_smp_0 SpecId 0", asm)
        self.assertIn("OpDecorate %lit_smp_1 SpecId 1", asm)
        self.assertIn("%lit_smp_0 = OpSpecConstant %uint 4294967295", asm)
        self.assertIn("%lit_smp_1 = OpSpecConstant %uint 4294967295", asm)
        # First sample used S1 (SpecId 1), second sample used S0 (SpecId 0).
        pos1 = asm.index("OpINotEqual %bool %lit_smp_1")
        pos0 = asm.index("OpINotEqual %bool %lit_smp_0")
        self.assertLess(pos1, pos0)
        validate_v12(self, asm, "lit_smp2")

    def test_dynamic_sampler_omits_spec_constants_and_meta_key(self):
        lines = [
            "v_mov_b32 v4, 0x3f000000",
            "v_mov_b32 v5, 0x3f000000",
            "image_sample v[0:3], v[4:5], s[0:7], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D",
            "s_waitcnt vmcnt(0)",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        asm, meta = translate_with_meta("ps_dyn_smp", lines)
        self.assertNotIn("literal_samplers", meta)
        self.assertNotIn("literal_sampler_spec_base", meta)
        self.assertNotIn("OpSpecConstant", asm)
        self.assertNotIn("SpecId", asm)
        validate_v12(self, asm, "dyn_smp")


if __name__ == "__main__":
    unittest.main()
