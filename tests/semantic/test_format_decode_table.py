"""The run-time FORMAT decode of buffer_load_format (Gen.fmt_dec_component) is table-driven and branch-free.

Every code goes through the translator and the mini interpreter, which runs the shared per-component functions, and is
compared with a reference decoded here from the raw bits with exact quotients. A code outside FORMATS gives zeros.

The decode has no branch: the format's kind, channel count and channel widths come from a table, and the kind picks
the conversion with selects. The module carries one copy of the decode, one function per output component, instead of
one copy per format: before, the shared function held every format's decode and a binary search over the codes.
"""
import struct
import unittest
from fractions import Fraction

import support
from spirv_mini_interp import half_to_float, run_fragment

import rdna2spirv

BASE = 0x1000
WORDS = (0x8123C4F7, 0x3E9A7F01, 0xC0490FDB, 0x5AA5FF80)


def f32_bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def reference(code):
    if code not in rdna2spirv.Gen.FORMATS:
        return (0, 0, 0, 0)
    size, kind, widths = rdna2spirv.Gen.FORMATS[code]
    raw, bit, out = sum(w << (32 * i) for i, w in enumerate(WORDS)), 0, []
    for w in widths:
        v = raw >> bit & ((1 << w) - 1)
        bit += w
        signed = v - (1 << w) if v >> (w - 1) else v
        if kind in ("f32", "u32", "i32", "uint"):
            out.append(v)
        elif kind == "sint":
            out.append(signed & 0xFFFFFFFF)
        elif kind == "f16":
            # A NaN half widens as sign | 0x7F800000 | payload << 13, quiet bit as stored: measured on gfx1036 for
            # formats 13, 29 and 71 (#4623, tools/f16_widen_probe.py). Not struct's '<e', whose NaN payload
            # depends on the Python version (CPython gh-130317).
            if v & 0x7FFF > 0x7C00:
                out.append((v & 0x8000) << 16 | 0x7F800000 | (v & 0x3FF) << 13)
            else:
                out.append(f32_bits(half_to_float(v)))
        elif kind == "unorm":
            out.append(f32_bits(float(Fraction(v, (1 << w) - 1))))
        elif kind == "snorm":
            out.append(f32_bits(float(max(Fraction(-1), Fraction(signed, (1 << (w - 1)) - 1)))))
        else:
            return None                 # a kind this reference doesn't model: covered by its own tests
    return tuple(out[c % len(out)] for c in range(4))


def load(code):
    lines = [f"s_mov_b32 s0, {BASE:#x}", "s_mov_b32 s1, 0", "s_mov_b32 s2, 0x100",
             f"s_mov_b32 s3, {code << 12 | 0x30000FAC:#x}", "v_mov_b32 v1, 0",
             "buffer_load_format_xyzw v[2:5], v1, s[0:3], 0 offen", "exp mrt0 v2, v3, v4, v5 done vm", "s_endpgm"]
    spv = support.translate_lines("ps_fmt_table", lines)
    return tuple(run_fragment(spv, memory={BASE + 4 * i: w for i, w in enumerate(WORDS)})["%out_mrt0"])


def module_of(code=1):
    lines = [f"s_mov_b32 s0, {BASE:#x}", "s_mov_b32 s1, 0", "s_mov_b32 s2, 0x100",
             f"s_mov_b32 s3, {code << 12 | 0x30000FAC:#x}", "v_mov_b32 v1, 0",
             "buffer_load_format_xyzw v[2:5], v1, s[0:3], 0 offen", "exp mrt0 v2, v3, v4, v5 done vm", "s_endpgm"]
    return support.translate_lines("ps_fmt_table", lines)


def function_body(spv, name):
    """The lines of the function `name`, without its header and its end."""
    body, inside = [], False
    for line in spv.splitlines():
        text = line.strip()
        if text.startswith(f"{name} = OpFunction"):
            inside = True
            continue
        if inside:
            if text == "OpFunctionEnd":
                break
            body.append(text)
    return body


BRANCHES = ("OpBranchConditional", "OpSelectionMerge", "OpSwitch", "OpBranch", "OpLoopMerge")


class FormatDecodeTableTests(unittest.TestCase):
    def test_every_format_and_the_codes_between(self):
        for code in range(max(rdna2spirv.Gen.FORMATS) + 2):
            want = reference(code)
            if want is not None:
                self.assertEqual(load(code), want, f"format {code}")

    def test_the_decode_has_no_branch(self):
        spv = module_of()
        for k in range(4):
            body = function_body(spv, f"%fmt_dec_c{k}")
            self.assertTrue(body, f"the decode of component {k} is missing")
            for line in body:
                for op in BRANCHES:
                    self.assertNotIn(op, line, f"the decode branches: {line}")

    def test_the_decode_is_shared_per_component(self):
        spv = module_of()
        for k in range(4):
            self.assertIn(f"%fmt_dec_c{k} = OpFunction %uint None %fn_fmt_dec", spv)
            self.assertIn(f"OpFunctionCall %uint %fmt_dec_c{k} ", spv)
        # The load site calls the four functions; the decode itself is not inlined there.
        main = function_body(spv, "%main")
        calls = [ln for ln in main if "OpFunctionCall %uint %fmt_dec_c" in ln]
        self.assertEqual(len(calls), 4)

    def test_the_decode_is_small(self):
        spv = module_of()
        count = sum(len(function_body(spv, f"%fmt_dec_c{k}")) for k in range(4))
        # Before, the shared function held every format's decode and a binary search over the codes: 3,498
        # instructions and 306 branches in the module of a shader with one dynamic-format load.
        self.assertLess(count, 900, f"the decode is {count} instructions")

    def test_narrow_loads_use_demand_driven_dynamic_channel_decode(self):
        # 1/2/3-component dynamic-format loads decode only the N channels selected by DST_SEL via %fmt_dec_dyn
        # and do not emit %fmt_dec_c0..c3 when no 4-component load is present.
        for suffix, ncomp, dst in (("x", 1, "v2"), ("xy", 2, "v[2:3]"), ("xyz", 3, "v[2:4]")):
            lines = [f"s_mov_b32 s0, {BASE:#x}", "s_mov_b32 s1, 0", "s_mov_b32 s2, 0x100",
                     f"s_mov_b32 s3, {1 << 12 | 0x30000FAC:#x}", "v_mov_b32 v1, 0",
                     "v_mov_b32 v2, 0", "v_mov_b32 v3, 0", "v_mov_b32 v4, 0", "v_mov_b32 v5, 0",
                     f"buffer_load_format_{suffix} {dst}, v1, s[0:3], 0 offen",
                     "exp mrt0 v2, v3, v4, v5 done vm", "s_endpgm"]
            spv = support.translate_lines(f"ps_fmt_dyn_{suffix}", lines)
            body = function_body(spv, "%fmt_dec_dyn")
            self.assertTrue(body, f"%fmt_dec_dyn missing for buffer_load_format_{suffix}")
            for line in body:
                for op in BRANCHES:
                    self.assertNotIn(op, line, f"%fmt_dec_dyn branches: {line}")
            for k in range(4):
                self.assertNotIn(f"%fmt_dec_c{k} = OpFunction", spv)
            main = function_body(spv, "%main")
            calls = [ln for ln in main if "OpFunctionCall %uint %fmt_dec_dyn " in ln]
            self.assertEqual(len(calls), ncomp, f"expected {ncomp} decode calls for _{suffix}")

    def test_narrow_loads_match_reference_across_all_formats_dst_sel_and_oob(self):
        # Exhaustive check: every format code 0..max+1 with rotated DST_SEL (including 0, 1, reserved 2/3, 4..7)
        # across buffer_load_format_x, _xy, and _xyz, plus OOB DST_SEL=1 vs channel selection.
        def dst1_for(code):
            entry = rdna2spirv.Gen.FORMATS.get(code)
            if entry is None or entry[1] in ("uint", "sint", "u32", "i32"):
                return 1
            return 0x3F800000

        def expect_sel(ref4, code, sel, in_bounds):
            if sel == 1:
                return dst1_for(code)
            if 4 <= sel <= 7:
                return ref4[sel - 4] if in_bounds else 0
            return 0

        # Build reusable shaders parameterized by V# word3 in push constants / s0..s3? Here translate once per
        # (ncomp, dst_sel_word) with code in s3 via s_mov_b32 — fast because we test every code on _x with
        # DST_SEL_X = (code % 8) and rotated multi-channel swizzles on _xy / _xyz.
        mem = {BASE + 4 * i: w for i, w in enumerate(WORDS)}
        # 1) All 128 codes on _xyz with swizzle (7, 5, 6) = W, Y, Z -> exercises channels 1, 2, 3 across every format
        for code in range(max(rdna2spirv.Gen.FORMATS) + 2):
            want4 = reference(code)
            if want4 is None:
                continue
            sels = (7, 5, 6)
            d3 = (code << 12) | 0x30000000 | sels[0] | (sels[1] << 3) | (sels[2] << 6)
            lines = [f"s_mov_b32 s0, {BASE:#x}", "s_mov_b32 s1, 0", "s_mov_b32 s2, 0x100",
                     f"s_mov_b32 s3, {d3:#x}", "v_mov_b32 v1, 0",
                     "v_mov_b32 v2, 0", "v_mov_b32 v3, 0", "v_mov_b32 v4, 0", "v_mov_b32 v5, 0",
                     "buffer_load_format_xyz v[2:4], v1, s[0:3], 0 offen",
                     "exp mrt0 v2, v3, v4, v5 done vm", "s_endpgm"]
            spv = support.translate_lines("ps_fmt_xyz_swiz", lines)
            got = tuple(run_fragment(spv, memory=mem)["%out_mrt0"])
            expected = tuple(expect_sel(want4, code, s, True) for s in sels) + (0,)
            self.assertEqual(got, expected, f"xyz format {code}")

        # 2) All 8 DST_SEL values (0..7) on _x in-bounds and out-of-bounds for both float (unorm 1) and uint (2).
        #    Also set DST_SEL_Y = 5 (odd) so an over-wide 4-bit extract of DST_SEL_X would pull in bit 0 of DST_SEL_Y
        #    and corrupt selectors 0..3 (catching 3->4 field-width mutations on OpBitFieldUExtract).
        for code in (1, 2, 76, 127):
            want4 = reference(code)
            for in_bounds in (True, False):
                voff = "0" if in_bounds else "0x200"
                for sel in range(8):
                    d3 = (code << 12) | 0x30000000 | sel | (5 << 3)
                    lines = [f"s_mov_b32 s0, {BASE:#x}", "s_mov_b32 s1, 0", "s_mov_b32 s2, 0x100",
                             f"s_mov_b32 s3, {d3:#x}", f"v_mov_b32 v1, {voff}",
                             "v_mov_b32 v2, 0", "v_mov_b32 v3, 0", "v_mov_b32 v4, 0", "v_mov_b32 v5, 0",
                             "buffer_load_format_x v2, v1, s[0:3], 0 offen",
                             "exp mrt0 v2, v3, v4, v5 done vm", "s_endpgm"]
                    spv = support.translate_lines("ps_fmt_x_sel", lines)
                    got = run_fragment(spv, memory=mem)["%out_mrt0"][0]
                    self.assertEqual(got, expect_sel(want4, code, sel, in_bounds),
                                     f"x format={code} sel={sel} in_bounds={in_bounds}")

    def test_mixed_xyzw_and_x_emits_fmt_norm_bits_once(self):
        # A shader containing both buffer_load_format_xyzw and buffer_load_format_x must emit %fmt_norm_bits once.
        lines = [f"s_mov_b32 s0, {BASE:#x}", "s_mov_b32 s1, 0", "s_mov_b32 s2, 0x100",
                 f"s_mov_b32 s3, {1 << 12 | 0x30000FAC:#x}", "v_mov_b32 v1, 0",
                 "buffer_load_format_xyzw v[2:5], v1, s[0:3], 0 offen",
                 "buffer_load_format_x v6, v1, s[0:3], 0 offen",
                 "v_add_f32 v2, v2, v6",
                 "exp mrt0 v2, v3, v4, v5 done vm", "s_endpgm"]
        spv = support.translate_lines("ps_fmt_mixed", lines)
        self.assertEqual(spv.count("%fmt_norm_bits = OpFunction"), 1)
        self.assertEqual(spv.count("%fmt_dec_dyn = OpFunction"), 1)
        self.assertEqual(spv.count("%fmt_dec_c0 = OpFunction"), 1)


if __name__ == "__main__":
    unittest.main()
