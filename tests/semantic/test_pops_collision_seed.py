"""POPS: the collision word a pixel shader's ordered section reads is seeded so its wait loop ends at once.

The hardware gives a pixel shader with an ordered section (POPS) its collision word in the SGPR after the user SGPRs
and the primitive mask: collision wave id in bits 0-9, this wave's id in bits 16-25, overlap in bit 31. Console code
waits in a loop until wave_id + ~collision_id is below src_pops_exiting_wave_id + ~collision_id (unsigned). The
translator orders the section with fragment-shader interlock instead (docs/design.md "POPS"), reads the exiting id as 0
and seeds the word with POPS_COLLISION_SEED (wave id 1, collision id 0, no overlap). The overlap test before the loop
skips it for a wave with a lane in EXEC; a wave (here: a host subgroup) with EXEC all off falls into it. With the
word the push constants held (0: wave id == collision id) the test 0xffffffff < 0xffffffff never held, and the loop
ran until the loop guard (65,536 rounds by default) in every such subgroup. On a device with 32-wide subgroups one
half of a wave64 pixel program is such a subgroup whenever its lanes are all off.

The shader is runtime/test/shaders/ps_pops_wait_loop.asm (also run on the GPU by check.py and run_gpu_tests.py); its
listing tests/semantic/fixtures/disasm/ps_pops_wait_loop.s is llvm-mc output, for the branch encodings. The runs here
use spirv_wave_interp.run_wave, whole subgroups of 64 and of 32 lanes. The idea comes from Marvisak/VIctory 7ac1f54.
"""
import os
import re
import unittest
from unittest.mock import patch

import support
import rdna2spirv
import shader_ir
from spirv_wave_interp import run_wave
from test_loop_ballot_barrier import _resolve_labels

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
NAME = "ps_pops_wait_loop"
FIXTURE = os.path.join(HERE, "fixtures", "disasm", NAME + ".s")
ASM = os.path.join(ROOT, "runtime", "test", "shaders", NAME + ".asm")
LIMIT = 40          # a small loop guard, so a loop that never ends on its own costs 40 rounds here
WORD_SGPR = 18      # the collision word's SGPR in the fixture and below


def header_words():
    with open(ASM, encoding="utf-8") as f:
        m = re.search(r"Expected mrt0 bits: (.*)", f.read())
    return tuple(int(x, 16) for x in m.group(1).split(","))


def translate_fixture(seed=True):
    """The fixture listing, translated with LIMIT as the loop guard; without the seed when `seed` is False (the
    translator before the change: the word stays as the push constants give it)."""
    with open(FIXTURE, "rb") as f:
        data = f.read()
    with patch.object(rdna2spirv, "LOOP_LIMIT", LIMIT):
        if seed:
            return support.translate_listing(NAME, data)
        with patch.object(rdna2spirv, "pops_collision_sgpr", lambda insns, reads: None):
            return support.translate_listing(NAME, data)


def mrt0(text, wave, word=0):
    """{mrt0 words} over the lanes of one subgroup run, with `word` in push-constant dword WORD_SGPR."""
    run = run_wave(text, wave=wave, push_constants=[0] * WORD_SGPR + [word] + [0] * (31 - WORD_SGPR))
    return {tuple(out["%out_mrt0"]) if out else None for out in run["outputs"]}


PROLOGUE = [
    "s_bfe_u32 s2, s18, 0xa0000",
    "s_bfe_u32 s5, s18, 0xa0010",
    "s_not_b32 s4, s2",
    "s_lshr_b32 s0, s18, 31",
    "s_add_i32 s5, s5, s4",
    "s_cmp_eq_u32 s0, 0",
    "s_cselect_b64 s[0:1], exec, 0",
    "s_add_i32 s2, src_pops_exiting_wave_id, s4",
    "s_cmp_gt_u32 s2, s5",
    "s_cselect_b64 s[2:3], exec, 0",
    "s_or_b64 s[0:1], s[0:1], s[2:3]",
    "s_and_b64 vcc, exec, s[0:1]",
    "s_cbranch_vccnz body",
    "wait:",
    "s_sleep 0xffff",
    "s_add_i32 s12, s12, 1",
    "s_add_i32 s0, src_pops_exiting_wave_id, s4",
    "s_cmp_lt_u32 s5, s0",
    "s_cbranch_scc0 wait",
    "body:",
]


def lanes_on(*predicate):
    """Instructions that leave EXEC on in the lanes `predicate` (ending in a v_cmpx on the lane number in v1)
    names."""
    return ["v_mbcnt_lo_u32_b32 v1, -1, 0", "v_mbcnt_hi_u32_b32 v1, -1, v1", *predicate]


def translate_program(exec_lines, seed=True):
    """A pixel program: EXEC set by `exec_lines`, the ordered-section entry counting wait-loop rounds in s12, EXEC
    back on, the count exported."""
    items = ["s_mov_b32 s12, 0", "s_mov_b64 s[20:21], exec", *exec_lines, *PROLOGUE,
             "s_mov_b64 exec, s[20:21]", "v_mov_b32_e32 v4, s12", "exp mrt0 v4, v4, v4, v4 done vm", "s_endpgm"]
    text = "; pops\n" + "".join(f"\t{t} ; encoding: [{','.join(['0x00'] * size)}]\n"
                                for t, size in _resolve_labels(items))
    with patch.object(rdna2spirv, "LOOP_LIMIT", LIMIT):
        if seed:
            return support.translate_listing("ps_pops_rounds", text.encode("utf-8"))
        with patch.object(rdna2spirv, "pops_collision_sgpr", lambda insns, reads: None):
            return support.translate_listing("ps_pops_rounds", text.encode("utf-8"))


def rounds(text, wave):
    """{wait-loop rounds} over the lanes of one subgroup run (push constants all 0, as before the seed)."""
    return {out["%out_mrt0"][0] for out in run_wave(text, wave=wave)["outputs"] if out}


class Fixture(unittest.TestCase):
    def test_listing_is_the_asm(self):
        """The listing has the `; encoding:` comments the branches need (regen_fixtures.py writes none), so it is
        llvm-mc output; its mnemonics must be those of the .asm, in order."""
        def mnemonics(lines):
            return [re.sub(r"_e32$", "", ln.split()[0]) for ln in lines]
        with open(ASM, encoding="utf-8") as f:
            asm = [ln.strip() for ln in f if ln.strip() and not ln.lstrip().startswith(";")]
        with open(FIXTURE, encoding="utf-8") as f:
            listing = [ln.strip() for ln in f.read().splitlines()[1:] if ln.strip()]
        self.assertEqual(mnemonics(listing), mnemonics(asm))

    def test_the_three_tables_agree(self):
        """The .asm header, check.py's PIXEL_BITS and scripts/run_gpu_tests.py state the same four words."""
        import sys
        sys.path.insert(0, os.path.join(ROOT, "runtime", "test", "shaders"))
        import check
        with open(os.path.join(ROOT, "scripts", "run_gpu_tests.py"), encoding="utf-8") as f:
            m = re.search(r'"%s": \(([^)]*)\)' % NAME, f.read())
        self.assertEqual(check.PIXEL_BITS[NAME], header_words())
        self.assertEqual(tuple(int(x, 16) for x in m.group(1).split(",")), header_words())

    def test_words(self):
        """One round with EXEC off, none with EXEC on, whatever the push constants hold at the word's dword."""
        text = translate_fixture()
        for wave in (64, 32):
            for word in (0, 0x80000000, 0x03ff03ff):
                with self.subTest(wave=wave, word=hex(word)):
                    self.assertEqual(mrt0(text, wave, word), {header_words()})

    def test_without_the_seed_the_loop_runs_to_the_guard(self):
        """The translator before the change: with 0 in the word, EXEC off ran the loop until the guard."""
        text = translate_fixture(seed=False)
        for wave in (64, 32):
            with self.subTest(wave=wave):
                self.assertEqual(mrt0(text, wave), {(LIMIT,) + header_words()[1:]})


class Rounds(unittest.TestCase):
    """Rounds of the wait loop by which lanes are on at the ordered section's entry, in a wave64 program run as
    64-wide subgroups and as 32-wide ones (a device whose subgroups are narrower than the program's waves)."""
    def test_every_lane_on_skips_the_loop(self):
        for wave in (64, 32):
            self.assertEqual(rounds(translate_program([]), wave), {0})

    def test_a_lane_on_in_each_subgroup_skips_the_loop(self):
        exec_lines = lanes_on("v_and_b32_e32 v1, 31, v1", "v_cmpx_gt_u32_e32 8, v1")   # lanes 0-7 and 32-39
        for wave in (64, 32):
            self.assertEqual(rounds(translate_program(exec_lines), wave), {0})

    def test_every_lane_off_leaves_after_one_round(self):
        for wave in (64, 32):
            self.assertEqual(rounds(translate_program(["s_mov_b64 exec, 0"]), wave), {1})
            self.assertEqual(rounds(translate_program(["s_mov_b64 exec, 0"], seed=False), wave), {LIMIT})

    def test_a_half_with_every_lane_off(self):
        """Lanes 41-63 on: one 64-lane subgroup skips the loop. A 32-lane subgroup runs the low half (lanes 0-31,
        all off), which falls into it and leaves after one round; before the seed it stayed until the guard."""
        exec_lines = lanes_on("v_cmpx_lt_u32_e32 40, v1")
        self.assertEqual(rounds(translate_program(exec_lines), 64), {0})
        self.assertEqual(rounds(translate_program(exec_lines), 32), {1})
        self.assertEqual(rounds(translate_program(exec_lines, seed=False), 32), {LIMIT})


class WhichSgpr(unittest.TestCase):
    """pops_collision_sgpr: the source of the s_bfe_u32 that takes out the wave id, when its entry value is read."""
    def insns(self, *lines):
        return [shader_ir.parse_line("\t" + ln) for ln in lines]

    def test_the_wave_id_extract_names_the_sgpr(self):
        insns = self.insns("s_bfe_u32 s2, s18, 0xa0000", "s_bfe_u32 s5, s18, 0xa0010")
        self.assertEqual(rdna2spirv.pops_collision_sgpr(insns, {18}), 18)

    def test_a_written_sgpr_is_not_the_word(self):
        insns = self.insns("s_mov_b32 s18, 7", "s_bfe_u32 s5, s18, 0xa0010")
        self.assertIsNone(rdna2spirv.pops_collision_sgpr(insns, set()))

    def test_another_field_is_not_the_wave_id(self):
        insns = self.insns("s_bfe_u32 s2, s18, 0xa0000", "s_bfe_u32 s1, s18, 0x2001c")
        self.assertIsNone(rdna2spirv.pops_collision_sgpr(insns, {18}))

    def test_the_seed_needs_an_ordered_section(self):
        """The same extract in a program that never reads src_pops_exiting_wave_id gets no seed."""
        text = support.translate_lines("ps_no_pops", ["s_bfe_u32 s5, s18, 0xa0010", "v_mov_b32_e32 v4, s5",
                                                      "exp mrt0 v4, v4, v4, v4 done vm", "s_endpgm"])
        self.assertNotIn("%pops_word", text)
        self.assertIn("%pops_word", translate_fixture())


if __name__ == "__main__":
    unittest.main()
