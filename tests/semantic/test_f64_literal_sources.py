"""Semantic tests for binary64 literal sources on v_div_fixup_f64, v_div_fmas_f64, v_cmp(x)_*_f64, and v_cmp(x)_class_f64.

On RDNA2, a 32-bit literal dword on a binary64 operand populates bits 63:32 with bits 31:0 zero
(e.g. `0x3ff00000` -> +1.0, `0x1` -> (0, 1)), an inline integer (-16..64) sign-extends to 64 bits
(e.g. `1` -> (1, 0), `-1` -> (0xffffffff, 0xffffffff)), and a spelling wider than 8 hex digits is the
whole binary64 constant (e.g. `0x000000003ff00000` -> (0x3ff00000, 0), a positive subnormal).
All four op families route through `Gen.f64_source_words()` using `tools/shader_ir.py`'s
`f64_literal` / `literal_hex_digits` tags. Standard library only; no GPU, no LLVM, no game files.
"""
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS_DIR = os.path.dirname(HERE)
ROOT = os.path.dirname(TESTS_DIR)
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")
for path in (HERE, TESTS_DIR, TOOLS_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

import f64_diff  # noqa: E402
import spirv_asm  # noqa: E402
import spirv_mini_interp as si  # noqa: E402
from support import translate_lines  # noqa: E402
import test_alu_div_fmas as div_fmas_mod  # noqa: E402

D_POS0 = 0x0000000000000000
D_NEG0 = 0x8000000000000000
D_ONE = 0x3FF0000000000000
D_TWO = 0x4000000000000000
D_THREE = 0x4008000000000000
D_INF = 0x7FF0000000000000
D_NINF = 0xFFF0000000000000
D_CANON_NAN = 0xFFF8000000000000


def _translate(name, lines, export="v0, v1, v2, v3"):
    return translate_lines(name, list(lines) + [f"exp mrt0 {export} done vm", "s_endpgm"])


def _run_words(name, lines, export="v0, v1, v2, v3"):
    spv = _translate(name, lines, export=export)
    binary = spirv_asm.assemble(spv)
    assert len(binary) >= 20 and binary[:4] == b"\x03\x02\x23\x07"
    return list(si.run_fragment(spv)["%out_mrt0"])


def _run_u64(name, lines):
    out = _run_words(name, lines)
    return (out[1] << 32) | out[0]


def _const_uints(spv):
    consts = {}
    for raw in spv.splitlines():
        lhs, sep, rhs = raw.strip().partition(" = OpConstant %uint ")
        if sep:
            consts[lhs] = int(rhs)
    return consts


def _constructed_f64_pairs(spv):
    consts = _const_uints(spv)
    pairs = []
    for raw in spv.splitlines():
        _, sep, rhs = raw.strip().partition(" = OpCompositeConstruct %v2uint_d ")
        if sep:
            lo, hi = rhs.split()
            if lo in consts and hi in consts:
                pairs.append((consts[lo], consts[hi]))
    return pairs


def _d2u(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", x))[0]


class DivFixupF64LiteralTests(unittest.TestCase):
    def test_one_dword_literals_in_all_three_source_slots(self):
        # 3.0 as 0x40080000 in S0, 1.0 as 0x3ff00000 in S1, -2.0 as 0xc0000000 in S2 -> -3.0.
        got = _run_u64("ps_fixup_lit_normal", [
            "v_div_fixup_f64 v[0:1], 0x40080000, 0x3ff00000, 0xc0000000",
        ])
        self.assertEqual(got, D_THREE | (1 << 63))

        # Finite / +inf via 32-bit literal 0x7ff00000 in S1 -> +0.0 (previously treated as subnormal).
        got_inf = _run_u64("ps_fixup_lit_s1_inf", [
            "v_div_fixup_f64 v[0:1], 0x40080000, 0x7ff00000, 1.0",
        ])
        self.assertEqual(got_inf, D_POS0)

        # NaN in S0 via 32-bit literal 0x7ff80000 -> signed infinity (-inf with S1=-2.0, S2=+1.0).
        got_nan_s0 = _run_u64("ps_fixup_lit_s0_nan", [
            "v_div_fixup_f64 v[0:1], 0x7ff80000, -2.0, 1.0",
        ])
        self.assertEqual(got_nan_s0, D_NINF)

        # -0.0 / -0.0 via 32-bit literal 0x80000000 in S1 and S2 -> canonical negative qNaN.
        got_zz = _run_u64("ps_fixup_lit_zz", [
            "v_div_fixup_f64 v[0:1], 1.0, 0x80000000, 0x80000000",
        ])
        self.assertEqual(got_zz, D_CANON_NAN)

        # Small hex literal 0x1 sits in bits 63:32 (0x00000001_00000000), while inline 1 is 0x00000000_00000001.
        got_hex1 = _run_u64("ps_fixup_lit_0x1", [
            "v_div_fixup_f64 v[0:1], 0x1, 1.0, 1.0",
        ])
        self.assertEqual(got_hex1, 0x0000000100000000)

        got_dec1 = _run_u64("ps_fixup_lit_dec1", [
            "v_div_fixup_f64 v[0:1], 1, 1.0, 1.0",
        ])
        self.assertEqual(got_dec1, 0x0000000000000001)

    def test_full_width_and_inline_literals_match_f64_diff_reference(self):
        cases = [
            "v_div_fixup_f64 v[8:9], 0x000000003ff00000, 0x3ff00000, 0x40000000",
            "v_div_fixup_f64 v[8:9], 0x3ff8000000000001, 0x000000007ff00000, 0x000000007ff00000",
            "v_div_fixup_f64 v[8:9], 0x40000000, -1, 1.0",
            "v_div_fixup_f64 v[8:9], 0x40080000, neg(0x3ff00000), 0x40000000",
        ]
        for idx, insn in enumerate(cases):
            with self.subTest(insn=insn):
                ref = f64_diff.ref_program([insn])[:2]
                got = _run_words(f"ps_fixup_ref_{idx}", [insn], export="v8, v9, v10, v11")[:2]
                self.assertEqual(got, ref)


def _eval_fmas_u64(spv):
    out = div_fmas_mod._eval_spv_f64_mrt0(spv)
    return (out[1] << 32) | out[0]


class DivFmasF64LiteralTests(unittest.TestCase):
    def test_one_dword_and_padded_literals_build_correct_words_and_evaluate(self):
        # VCC = 0: fma(2.0, 1.5, 1.0) = 4.0 using 32-bit one-dword hex literals in all three slots.
        spv_plain = _translate("ps_fmas_lit_plain", [
            "v_cmp_eq_u32_e32 vcc_lo, 0, 1",
            "v_div_fmas_f64 v[0:1], 0x40000000, 0x3ff80000, 0x3ff00000",
        ])
        div_fmas_mod.validate_spirv(spv_plain)
        pairs = _constructed_f64_pairs(spv_plain)
        self.assertEqual(pairs[:3], [(0, 0x40000000), (0, 0x3FF80000), (0, 0x3FF00000)])
        self.assertEqual(_eval_fmas_u64(spv_plain), _d2u(4.0))

        # VCC = 1 with S2 = 0x40000000 (2.0, |S2| >= 2.0 -> scale UP by 2^128):
        # (1.0 * 1.0 + 2.0) * 2^128 = 3.0 * 2^128. Previously 0x40000000 decoded as subnormal (< 2.0 -> scale DOWN).
        spv_up = _translate("ps_fmas_lit_scale_up", [
            "v_cmp_eq_u32_e32 vcc_lo, 1, 1",
            "v_div_fmas_f64 v[0:1], 0x3ff00000, 0x3ff00000, 0x40000000",
        ])
        div_fmas_mod.validate_spirv(spv_up)
        self.assertEqual(_eval_fmas_u64(spv_up), _d2u(3.0 * (2.0 ** 128)))

        # 0x1 (one-dword literal -> (0, 1)) vs inline 1 ((1, 0)) vs 16-digit padded 0x000000003ff00000 ((0x3ff00000, 0)).
        spv_widths = _translate("ps_fmas_lit_widths", [
            "v_cmp_eq_u32_e32 vcc_lo, 0, 1",
            "v_div_fmas_f64 v[0:1], 0x1, 1, 0x000000003ff00000",
        ])
        div_fmas_mod.validate_spirv(spv_widths)
        self.assertEqual(_constructed_f64_pairs(spv_widths)[:3], [(0, 1), (1, 0), (0x3FF00000, 0)])


class CmpF64LiteralTests(unittest.TestCase):
    def test_hex_small_literal_and_padded_full_width_literals(self):
        # 0x1 is (0, 1) > inline 1 which is (1, 0); 0x000000003ff00000 is subnormal < 1.0 (not == 1.0).
        got = _run_words("ps_cmp_f64_lits", [
            "v_cmp_gt_f64_e64 s4, 0x1, 1",
            "v_cndmask_b32_e64 v0, 0, 1, s4",
            "v_cmp_eq_f64_e64 s4, 0x1, 1",
            "v_cndmask_b32_e64 v1, 0, 1, s4",
            "v_cmp_lt_f64_e64 s4, 0x000000003ff00000, 1.0",
            "v_cndmask_b32_e64 v2, 0, 1, s4",
            "v_cmp_eq_f64_e64 s4, 0x000000003ff00000, 1.0",
            "v_cndmask_b32_e64 v3, 0, 1, s4",
        ])
        self.assertEqual(got, [1, 0, 1, 0])

        # 2-operand v_cmpx_*_f64_e32 form (src0 is args[0], src1 is args[1]).
        got_cmpx = _run_words("ps_cmpx_f64_lits", [
            "v_mov_b32_e32 v0, 10",
            "v_mov_b32_e32 v1, 20",
            "v_mov_b32_e32 v2, 30",
            "v_mov_b32_e32 v3, 40",
            "s_mov_b32 s10, exec_lo",
            "v_cmpx_gt_f64_e32 0x1, 1",                    # true -> lane stays active
            "v_mov_b32_e32 v0, 11",
            "v_cmpx_eq_f64_e32 0x000000003ff00000, 1.0",   # false -> lane turns inactive
            "v_mov_b32_e32 v1, 21",                        # skipped
            "s_mov_b32 exec_lo, s10",
            "v_cmp_gt_f64_e64 s6, 0x3ff8000000000001, 1.5",
            "v_cndmask_b32_e64 v2, 31, 32, s6",
        ])
        self.assertEqual(got_cmpx, [11, 20, 32, 40])

    def test_cmp_f64_literals_match_f64_diff_reference(self):
        for idx, insn in enumerate((
            "v_cmp_gt_f64_e64 s4, 0x1, 1",
            "v_cmp_eq_f64_e64 s4, 0x000000003ff00000, 1.0",
            "v_cmp_lt_f64_e64 s4, 0x000000003ff00000, 0x3ff00000",
        )):
            with self.subTest(insn=insn):
                prog = [insn, "v_cndmask_b32_e64 v8, 0, 1, s4"]
                ref = f64_diff.ref_program(prog)[0]
                got = _run_words(f"ps_cmp_ref_{idx}", prog, export="v8, v9, v10, v11")[0]
                self.assertEqual(got, ref)


class CmpClassF64LiteralTests(unittest.TestCase):
    def test_padded_full_width_literals_and_small_hex_literal(self):
        # 16-digit padded literals live in the low word (hi = 0) -> all four are +denormal (bit 7 = 0x80),
        # NOT +normal (1.0), -0.0, +inf, or qNaN.
        got = _run_words("ps_cmp_class_f64_padded", [
            "v_cmp_class_f64_e64 s4, 0x000000003ff00000, 0x80",
            "v_cndmask_b32_e64 v0, 0, 1, s4",
            "v_cmp_class_f64_e64 s4, 0x0000000080000000, 0x80",
            "v_cndmask_b32_e64 v1, 0, 1, s4",
            "v_cmp_class_f64_e64 s4, 0x000000007ff00000, 0x80",
            "v_cndmask_b32_e64 v2, 0, 1, s4",
            "v_cmp_class_f64_e64 s4, 0x000000007ff80000, 0x80",
            "v_cndmask_b32_e64 v3, 0, 1, s4",
        ])
        self.assertEqual(got, [1, 1, 1, 1])

        # Verify 2-operand v_cmpx_class_f64_e32 distinguishes padded 0x000000003ff00000 (+denorm) from 0x3ff00000 (+normal).
        got_cmpx = _run_words("ps_cmpx_class_f64_padded", [
            "v_mov_b32_e32 v0, 10",
            "v_mov_b32_e32 v1, 20",
            "v_mov_b32_e32 v2, 0",
            "v_mov_b32_e32 v3, 0",
            "s_mov_b32 s10, exec_lo",
            "v_cmpx_class_f64_e32 0x000000003ff00000, 0x80",   # +denormal -> stays active
            "v_mov_b32_e32 v0, 11",
            "v_cmpx_class_f64_e32 0x000000003ff00000, 0x100",  # not +normal -> turns inactive
            "v_mov_b32_e32 v1, 21",                            # skipped
            "s_mov_b32 exec_lo, s10",
        ])
        self.assertEqual(got_cmpx, [11, 20, 0, 0])

        # Verify emitted high/low words for 0x1 vs inline 1: the sign shift extracts bits from hi.
        spv_hex1 = _translate("ps_cls_0x1", [
            "v_cmp_class_f64_e64 s4, 0x1, 0x80",
            "v_cndmask_b32_e64 v0, 0, 1, s4",
        ])
        spv_dec1 = _translate("ps_cls_dec1", [
            "v_cmp_class_f64_e64 s4, 1, 0x80",
            "v_cndmask_b32_e64 v0, 0, 1, s4",
        ])

        def _sign_hi_word(spv):
            consts = _const_uints(spv)
            for raw in spv.splitlines():
                _, sep, rhs = raw.strip().partition(" = OpShiftRightLogical %uint ")
                if sep:
                    parts = rhs.split()
                    if len(parts) == 2 and consts.get(parts[1]) == 31:
                        return consts[parts[0]]
            raise AssertionError("sign shift not found")

        self.assertEqual(_sign_hi_word(spv_hex1), 1)
        self.assertEqual(_sign_hi_word(spv_dec1), 0)

    def test_cmp_class_f64_literals_match_f64_diff_reference(self):
        for idx, insn in enumerate((
            "v_cmp_class_f64_e64 s4, 0x000000003ff00000, 0x100",
            "v_cmp_class_f64_e64 s4, 0x000000003ff00000, 0x80",
            "v_cmp_class_f64_e64 s4, 0x3ff00000, 0x100",
            "v_cmp_class_f64_e64 s4, 0x0000000080000000, 0x20",
        )):
            with self.subTest(insn=insn):
                prog = [insn, "v_cndmask_b32_e64 v8, 0, 1, s4"]
                ref = f64_diff.ref_program(prog)[0]
                got = _run_words(f"ps_cls_ref_{idx}", prog, export="v8, v9, v10, v11")[0]
                self.assertEqual(got, ref)


if __name__ == "__main__":
    unittest.main()
