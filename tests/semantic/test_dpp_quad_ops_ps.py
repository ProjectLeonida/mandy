"""quad_perm DPP in a pixel shader reads its quad through quad operations (SHADER-PS-DPP-QUAD-OPS).

Without MaximallyReconvergesKHR, Vulkan ("Helper Invocations") lets an implementation treat helper invocations as
inactive for group operations other than quad and derivative ones. The translator used to lower every quad_perm to
OpGroupNonUniformShuffle (and, with fi:0, the source lane's EXEC bit to a ballot), so a pixel next to a triangle edge
could read an undefined value from its helper neighbour. In fragment shaders it now uses OpGroupNonUniformQuadSwap
for the three swaps and OpGroupNonUniformQuadBroadcast, one per lane read, for every other selector.

The values are checked by running the translated program in tests/semantic/spirv_wave_interp.py, whose `helpers`
option takes the strict reading: a shuffle from a helper is undefined and a ballot does not see it.

Standard library only; no GPU, no LLVM, no SPIRV-Tools, no game files.
"""
import os
import re
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import support  # noqa: E402  (puts tools/ on the path)
import spirv_wave_interp as wi  # noqa: E402
from test_dpp_eager_subgroup import conditional_subgroup_ops  # noqa: E402

SWAPS = {(1, 0, 3, 2): 0, (2, 3, 0, 1): 1, (3, 2, 1, 0): 2}
SELECTORS = [(1, 0, 3, 2), (2, 3, 0, 1), (3, 2, 1, 0),            # the three swaps
             (0, 0, 0, 0), (1, 1, 1, 1), (2, 2, 2, 2), (3, 3, 3, 3),  # broadcasts
             (1, 1, 3, 3), (2, 3, 2, 3), (0, 0, 2, 2), (0, 1, 0, 1),  # the other selectors seen in pixel shaders
             (3, 1, 2, 0)]                                             # all four lanes, no pattern


def form(sel, fi, bound=1):
    return f"quad_perm:[{','.join(map(str, sel))}] row_mask:0xf bank_mask:0xf bound_ctrl:{bound} fi:{fi}"


def dpp_ps(dpp, exec_off=None):
    """A pixel shader: v6 = 100, then v6 = v0 (the lane) + 10 through DPP `dpp`, with lane `exec_off` off in EXEC
    for the DPP; mrt0.x exports v6's bits."""
    lines = ["v_mbcnt_lo_u32_b32 v0, -1, 0", "v_mov_b32 v1, 10", "v_mov_b32 v6, 100", "s_mov_b32 s10, exec_lo"]
    if exec_off is not None:
        lines.append(f"v_cmpx_ne_u32_e32 {exec_off}, v0")
    lines += [f"v_add_nc_u32_dpp v6, v0, v1 {dpp}", "s_mov_b32 exec_lo, s10",
              "exp mrt0 v6, v6, v6, v6 done vm", "s_endpgm"]
    return support.translate_lines("ps_dpp_quad", lines)


def reference(sel, fi, bound=1, exec_off=None, wave=32):
    """v6 per lane after dpp_ps(form(sel, fi, bound), exec_off): the RDNA2 DPP rules for quad_perm
    (docs/design.md, Gen.dpp_insn). A helper invocation is an ordinary active lane to the hardware."""
    out = []
    for lane in range(wave):
        src = (lane & ~3) | sel[lane & 3]
        valid = fi or src != exec_off
        if lane == exec_off or (not valid and not bound):
            out.append(100)                                   # not written: keeps its old value
        else:
            out.append((src if valid else 0) + 10)
    return out


def lanes(spv, helpers=()):
    return [o and o["%out_mrt0"][0] for o in wi.run_wave(spv, wave=32, helpers=helpers)["outputs"]]


class PixelShaderQuadPerm(unittest.TestCase):
    def test_no_shuffle_or_ballot(self):
        for sel in SELECTORS:
            for fi in (0, 1):
                with self.subTest(sel=sel, fi=fi):
                    spv = dpp_ps(form(sel, fi))
                    self.assertNotIn("OpGroupNonUniformShuffle", spv)
                    self.assertNotIn("OpGroupNonUniformBallot", spv)
                    self.assertIn("OpCapability GroupNonUniformQuad", spv)
                    self.assertEqual(conditional_subgroup_ops(spv), [])

    def test_swaps_and_broadcasts(self):
        for sel in SELECTORS:
            with self.subTest(sel=sel):
                spv = dpp_ps(form(sel, 1))
                swaps = re.findall(r"OpGroupNonUniformQuadSwap %uint %c3 %\w+ %c(\d+)", spv)
                reads = re.findall(r"OpGroupNonUniformQuadBroadcast %uint %c3 %\w+ %c(\d+)", spv)
                if sel in SWAPS:
                    self.assertEqual((swaps, reads), ([str(SWAPS[sel])], []))
                else:
                    self.assertEqual((swaps, sorted(reads)), ([], [str(k) for k in sorted(set(sel))]))

    def test_identity_reads_no_neighbour(self):
        spv = dpp_ps(form((0, 1, 2, 3), 1))
        self.assertNotIn("OpGroupNonUniformQuad", spv)
        self.assertEqual(lanes(spv), reference((0, 1, 2, 3), 1))

    def test_values(self):
        for sel in SELECTORS:
            for fi, bound in ((1, 1), (0, 1), (0, 0)):
                for exec_off in (None, 5):
                    with self.subTest(sel=sel, fi=fi, bound=bound, exec_off=exec_off):
                        spv = dpp_ps(form(sel, fi, bound), exec_off)
                        self.assertEqual(lanes(spv), reference(sel, fi, bound, exec_off))

    def test_helper_neighbours_are_read(self):
        # Lanes 2, 5 and 12 run as helper invocations. Their neighbours read their v0 as hardware does; with a
        # shuffle (fi:1) or a ballot of EXEC (fi:0) they read an undefined value or miss the bit.
        helpers = (2, 5, 12)
        for sel in SELECTORS:
            for fi in (0, 1):
                with self.subTest(sel=sel, fi=fi):
                    want = [None if lane in helpers else v for lane, v in enumerate(reference(sel, fi))]
                    self.assertEqual(lanes(dpp_ps(form(sel, fi)), helpers=helpers), want)


class OtherStagesKeepTheShuffle(unittest.TestCase):
    def test_vertex_shader(self):
        # Quad operations are guaranteed in fragment and compute shaders only (quadOperationsInAllStages).
        lines = ["v_mbcnt_lo_u32_b32 v0, -1, 0", "v_mov_b32 v1, 10", "v_mov_b32 v6, 100",
                 f"v_add_nc_u32_dpp v6, v0, v1 {form((1, 0, 3, 2), 1)}", "v_mov_b32 v5, 0",
                 "exp pos0 v5, v5, v5, v5 done", "exp param0 v6, v5, v5, v5", "s_endpgm"]
        spv = support.translate_lines("vs_dpp_quad", lines)
        self.assertIn("OpGroupNonUniformShuffle", spv)
        self.assertNotIn("OpGroupNonUniformQuad", spv)


class HelperModel(unittest.TestCase):
    """The interpreter's helpers option itself: a shuffle that reads a helper lane gives an undefined value."""

    def test_shuffle_from_a_helper_is_undefined(self):
        lines = ["v_mbcnt_lo_u32_b32 v0, -1, 0", "v_mov_b32 v1, 10", "v_mov_b32 v6, 100",
                 f"v_add_nc_u32_dpp v6, v0, v1 {form((1, 0, 3, 2), 1)}", "v_mov_b32 v5, 0",
                 "exp pos0 v5, v5, v5, v5 done", "exp param0 v6, v5, v5, v5", "s_endpgm"]
        spv = support.translate_lines("vs_dpp_quad", lines)
        with self.assertRaisesRegex(wi.UnsupportedConstruct, "undefined"):
            wi.run_wave(spv, wave=32, helpers=(2,),
                        builtins={"VertexIndex": lambda lane: lane, "InstanceIndex": lambda lane: 0})


if __name__ == "__main__":
    unittest.main()
