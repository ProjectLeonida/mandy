"""SDWA source selects on the VOP2 carry ops land on their sources (SHADER-SDWA-CO-CI-SELECT).

v_add_co_ci_u32, v_sub_co_ci_u32 and v_subrev_co_ci_u32 are printed in a long form,
vD, carry out, src0, src1, carry in, with the carry out between the destination and the
sources. Gen.insn() used to attach src0_sel/src1_sel from the second operand on, as for
every other VOP2 op, so for the long form src0_sel landed on the carry out, src1_sel
narrowed src0 and src1 was read whole: a silent wrong result. The short form
vD, src0, src1 was already right and is kept as a control.

Semantics (RDNA2 ISA VOP2 0x28/0x29/0x2a with the SDWA source rule): each source is
narrowed to its select, zero-extended, then
    add:    D = S0 + S1 + C,  carry out = the bit past 32
    sub:    D = S0 - S1 - C,  borrow out = (S1 + C > S0)
    subrev: D = S1 - S0 - C,  borrow out = (S0 + C > S1)
Hand-written listings are translated by tools/rdna2spirv.py and run by spirv_mini_interp.py;
expected words come from the plain-Python reference below.
"""

import random
import unittest

from support import mrt0_words, translate_lines  # noqa: F401

SEL = {"BYTE_0": (0, 8), "BYTE_1": (8, 8), "BYTE_2": (16, 8), "BYTE_3": (24, 8),
       "WORD_0": (0, 16), "WORD_1": (16, 16), "DWORD": (0, 32)}
OPS = ("v_add_co_ci_u32", "v_sub_co_ci_u32", "v_subrev_co_ci_u32")


def narrow(v, sel):
    off, width = SEL[sel]
    return (v >> off) & ((1 << width) - 1)


def ref(op, x, y, sel0, sel1, cin):
    s0, s1 = narrow(x, sel0), narrow(y, sel1)
    if op == "v_add_co_ci_u32":
        total = s0 + s1 + cin
        return total & 0xFFFFFFFF, int(total > 0xFFFFFFFF)
    if op == "v_subrev_co_ci_u32":
        s0, s1 = s1, s0
    return (s0 - s1 - cin) & 0xFFFFFFFF, int(s1 + cin > s0)


def run(op, x, y, sel0, sel1, cin, long_form=True):
    vcc = "-1" if cin else "0"
    if long_form:
        insn = f"{op}_sdwa v2, vcc, v1, v4, vcc"
    else:
        insn = f"{op}_sdwa v2, v1, v4"
    lines = [
        f"v_mov_b32 v1, 0x{x:08x}",
        f"v_mov_b32 v4, 0x{y:08x}",
        f"s_mov_b32 vcc_lo, {vcc}",
        f"s_mov_b32 vcc_hi, {vcc}",
        f"{insn} dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:{sel0} src1_sel:{sel1}",
        "v_cndmask_b32 v3, 0, 1, vcc",
        "exp mrt0 v2, v3, v2, v3 done vm",
        "s_endpgm",
    ]
    words = mrt0_words(translate_lines("ps_sdwa_carry", lines))
    return words[0], words[1]


class SdwaCarrySelect(unittest.TestCase):
    def check(self, op, x, y, sel0, sel1, cin, long_form=True):
        got = run(op, x, y, sel0, sel1, cin, long_form)
        want = ref(op, x, y, sel0, sel1, cin)
        self.assertEqual(got, want, f"{op} {'long' if long_form else 'short'} x={x:#010x} y={y:#010x} "
                                    f"{sel0}/{sel1} cin={cin}: got {got[0]:#010x}/{got[1]}, "
                                    f"want {want[0]:#010x}/{want[1]}")

    def test_reported_case(self):
        # The words from the claim: v1 = v4 = 0x00050003, src0 = WORD_1 (5), src1 = WORD_0 (3), no carry in.
        self.assertEqual(run("v_sub_co_ci_u32", 0x00050003, 0x00050003, "WORD_1", "WORD_0", 0), (2, 0))
        self.assertEqual(run("v_add_co_ci_u32", 0x00050003, 0x00050003, "WORD_1", "WORD_0", 0), (8, 0))
        self.assertEqual(run("v_subrev_co_ci_u32", 0x00050003, 0x00050003, "WORD_1", "WORD_0", 0),
                         (0xFFFFFFFE, 1))

    def test_long_form_selects_each_source(self):
        x, y = 0x80FF017F, 0x0102FE03
        for op in OPS:
            for sel0, sel1 in (("WORD_1", "WORD_0"), ("WORD_0", "WORD_1"), ("BYTE_3", "BYTE_0"),
                               ("BYTE_1", "BYTE_2"), ("DWORD", "WORD_1"), ("BYTE_2", "DWORD")):
                for cin in (0, 1):
                    self.check(op, x, y, sel0, sel1, cin)

    def test_short_form_unchanged(self):
        for op in OPS:
            for cin in (0, 1):
                self.check(op, 0x00050003, 0x00050003, "WORD_1", "WORD_0", cin, long_form=False)
                self.check(op, 0x80FF017F, 0x0102FE03, "BYTE_3", "BYTE_1", cin, long_form=False)

    def test_seeded(self):
        rng = random.Random(20261124)
        sels = sorted(SEL)
        for _ in range(24):
            op = rng.choice(OPS)
            self.check(op, rng.getrandbits(32), rng.getrandbits(32), rng.choice(sels), rng.choice(sels),
                       rng.randrange(2))


if __name__ == "__main__":
    unittest.main()
