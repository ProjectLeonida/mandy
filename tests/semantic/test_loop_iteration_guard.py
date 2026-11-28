"""Per-lane readback for terminating and unreachable loop bounds."""
from pathlib import Path
import unittest
from unittest.mock import patch

from test_loop_ballot_barrier import _translate_cs, rdna2spirv
from spirv_mini_interp import run_workgroup

FIXTURE = Path(__file__).with_name("fixtures") / "loop_iteration_guard.asm"


def module(lanes, limit, structured=True):
    lines = [s for s in FIXTURE.read_text().splitlines() if s and not s.startswith(";")]
    lines.insert(0, "s_mov_b64 exec, -1" if lanes == 64 else "s_mov_b32 exec_lo, -1")
    with patch.object(rdna2spirv, "LOOP_LIMIT", limit):
        text, gen = _translate_cs("cs_guard", lines, threads=lanes, emulate=lanes == 64, structured=structured)
    assert gen.wave64 == (lanes == 64) and gen.emulate == (lanes == 64)
    return text, gen.control_flow


def execute(text, lanes, bound):
    mem = run_workgroup(
        text, lanes, 32,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0),
                              "%sg_lane": i % 32, "%local_index": i},
        push_constants=[0, 0, lanes * 4, 0x31016FAC, bound] + [0] * 27, memory={})
    return [mem[i * 4] for i in range(lanes)]


class LoopGuardTests(unittest.TestCase):
    def test_finite_and_nonterminating(self):
        for lanes in (32, 64):
            with self.subTest(lanes=lanes):
                (plain, plain_flow), (guarded, flow) = module(lanes, 0), module(lanes, 16)
                # A single-wave emulated program keeps the structured form (no round fallback for `inline_all`).
                expected_flow = "structured" if lanes == 32 else "emulated wave64 structured"
                self.assertEqual((plain_flow, flow), (expected_flow, expected_flow))
                self.assertNotIn("%loop_guard_", plain)
                self.assertIn("%loop_guard_", guarded)
                self.assertEqual(execute(plain, lanes, 6), [6] * lanes)
                self.assertEqual(execute(guarded, lanes, 6), [6] * lanes)
                self.assertEqual(execute(guarded, lanes, 1), [32] * lanes)

    def test_round_budget_uses_the_scheduler_exit(self):
        text, flow = module(64, 16, structured=False)
        self.assertEqual(flow, "emulated wave64 rounds")
        # Counters and synchronization execute on the final round too. The guard
        # augments the scheduler's termination condition at its original branch.
        tail = text.split("%x_ran = OpLabel", 1)[1].split("%loop_exit = OpLabel", 1)[0]
        counters, latch = tail.split("%loop_cont = OpLabel")
        self.assertIn("OpAtomicIAdd", counters)
        self.assertTrue(counters.rstrip().endswith("OpBranch %loop_cont"))
        self.assertLess(latch.index("OpControlBarrier"), latch.index("%x_fin ="))
        self.assertRegex(latch, r"OpLogicalOr %bool %x_fin (%loop_guard_\w+_full)")
        self.assertRegex(latch, r"OpBranchConditional %loop_guard_\w+_take %loop_exit %loop_head\s*$")

    def test_dispatch_loop_also_stops(self):
        for lanes in (32, 64):
            with self.subTest(lanes=lanes):
                guarded, _ = module(lanes, 16, structured=False)
                self.assertEqual(execute(guarded, lanes, 6), [6] * lanes)
                words = execute(guarded, lanes, 1)
                self.assertEqual(words, [30] * lanes)


    def test_inner_budget_resets_on_each_entry(self):
        lines = ["s_mov_b32 s8, 0", "s_mov_b32 s9, 0", "outer:", "s_mov_b32 s10, 0",
                 "inner:", "s_add_u32 s9, s9, 1", "s_add_u32 s10, s10, 1", "s_cmp_eq_u32 s10, 3",
                 "s_cbranch_scc0 inner", "s_add_u32 s8, s8, 1", "s_cmp_eq_u32 s8, 3", "s_cbranch_scc0 outer",
                 "v_mov_b32_e32 v1, s9", "v_lshlrev_b32_e32 v2, 2, v0",
                 "buffer_store_dword v1, v2, s[0:3], 0 offen", "s_endpgm"]
        with patch.object(rdna2spirv, "LOOP_LIMIT", 3):
            text, _ = _translate_cs("cs_guard_nested", lines, threads=32)
        self.assertEqual(execute(text, 32, 0), [9] * 32)

    # Two nested loops that never end (as an inclusive exit against a bound of 0xFFFFFFFF makes them, once the counter
    # wraps). With a count per loop the limits multiply: LIMIT visits of the inner loop in each of LIMIT outer visits.
    # --loop-total bounds them all together. s9 counts the inner loop's visits.
    NEVER = ["s_mov_b32 s8, 0", "s_mov_b32 s9, 0", "outer:", "s_mov_b32 s10, 0",
             "inner:", "s_add_u32 s9, s9, 1", "s_add_u32 s10, s10, 1", "s_cmp_eq_u32 s10, 0",
             "s_cbranch_scc0 inner", "s_add_u32 s8, s8, 1", "s_cmp_eq_u32 s8, 0", "s_cbranch_scc0 outer",
             "v_mov_b32_e32 v1, s9", "v_lshlrev_b32_e32 v2, 2, v0",
             "buffer_store_dword v1, v2, s[0:3], 0 offen", "s_endpgm"]

    def nested(self, lines, limit, total, lanes=32):
        with patch.object(rdna2spirv, "LOOP_LIMIT", limit), patch.object(rdna2spirv, "LOOP_TOTAL", total):
            text, _ = _translate_cs("cs_guard_total", lines, threads=lanes, emulate=lanes == 64)
        return text, execute(text, lanes, 0)

    def test_total_bounds_nested_loops(self):
        # Without a total: 50 visits of the inner loop in each of 50 outer visits.
        text, words = self.nested(self.NEVER, 50, 0)
        self.assertNotIn("%loop_guard_total", text)
        self.assertEqual(words, [2500] * 32)
        # A total of 40: the first inner run takes 41 visits (the 41st finds the total spent), and the outer loop
        # leaves at once after it.
        text, words = self.nested(self.NEVER, 50, 40)
        self.assertIn("%loop_guard_total", text)
        self.assertEqual(words, [41] * 32)
        # Emulated wave64: the round scheduler's own loop is guarded too and spends from the same total, so fewer
        # visits are left for the program's loops, never more.
        text, words = self.nested(self.NEVER, 50, 40, lanes=64)
        self.assertIn("%loop_guard_total", text)
        self.assertEqual(len(set(words)), 1)
        self.assertTrue(0 < words[0] <= 41, words[0])

    def test_total_leaves_loops_that_end_alone(self):
        # Loops that end by themselves within the total compute what they did: 3 x 3 visits.
        lines = [s.replace("s10, 0", "s10, 3").replace("s8, 0", "s8, 3") if s.startswith("s_cmp") else s
                 for s in self.NEVER]
        _, words = self.nested(lines, 50, 16)
        self.assertEqual(words, [9] * 32)
