"""CPU checks for the batch-4 GPU fixtures (docs/design.md rows 218, 226, 265, 379/380, 381).

Each fixture runs the translator on its generated disassembly (tests/semantic/fixtures/disasm, made by
regen_fixtures.py) and executes all 32 invocations in the mini interpreter: lane 0 alone passes the fixture's EXEC
cut and stores the eight results, so the memory holds the words its header derives. The same words are asserted
against the three places that state them: the .asm header, check.py's CS_DUMPS and scripts/run_gpu_tests.py's table.
A GPU run of either runner then compares them with what the device writes.
"""

import os
import re
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SHADERS = os.path.join(ROOT, "runtime", "test", "shaders")
sys.path.insert(0, HERE)
sys.path.insert(0, SHADERS)

import support  # noqa: E402
import spirv_mini_interp as si  # noqa: E402
import check  # noqa: E402  (importing runtime/test/shaders/check.py runs nothing)
from compute_fixture import compute_header  # noqa: E402

FIXTURES = {
    # sdst under EXEC = 0xAAAAAAAA: 16 set lanes | lowest lane 1 << 8
    "cs_w32_sdst_exec": (272, 272, 272, 272, 272, 272, 272, 272),
    "cs_global_atomic32_int": (0x80000004, 2, 0xFFFFFFFB, 3, 3, 0xFFFFFFFB, 0xF000F000, 0xFF0FFF0F),
    "cs_global_atomic32_rtn": (0xFFFFFFFF, 0xCAFEF00D, 6, 4, 0x11111111, 0x100, 0xF0F0, 0xAAAA),
    "cs_global_atomic32_float": (0x3F800000, 0x40000000, 0x40400000, 0, 0x43, 0x42, 0x55, 0x55),
    "cs_global_atomic32_x2": (0xF000F000, 0x0F0F0F0F, 0xFF0FFF0F, 0xFFFFFFFF,
                              0x55555555, 0xEDCBA987, 0xAAAABBBB, 0xCCCCDDDD),
    "cs_w32_clamp32": (0xFFFFFFFF, 0, 0, 0x7FFFFFFF, 0x80000000, 3, 12, 0),
    "cs_w32_clamp16": (0xDEADFFFF, 0xDEAD0000, 0xDEAD0000, 0xFFFFFFFF, 0xFFFFFFFF, 0, 0, 0),
    "cs_w32_mad_denorm": (0, 0, 0, 0, 0x40E00000, 0, 0, 0),
    "cs_w32_vcmp_f64": (0xC, 0xC, 0xC, 0xC, 0, 0, 0xF, 0xF),
}
# The atomic fixtures also seed and rewrite words at 0x40..0x5C through the global pointer; the others write only
# the eight output dwords.
ATOMIC = tuple(n for n in FIXTURES if n.startswith("cs_global_atomic32"))
# The output V# in s[0:3]: base 0, num_records max, FORMAT 32_FLOAT, DST_SEL xyzw (as vk_compute_test sets it); the
# heap base in push-constant dwords 32/33 is 0, so an Xbox VA is the memory word index below.
OUT_V = [0, 0, 0xFFFFFFFF, 0x31016FAC]


def read(name):
    with open(os.path.join(SHADERS, name + ".asm"), encoding="utf-8") as f:
        return f.read()


def header_words(name):
    """The eight decimal words of the fixture's "; Run:" line."""
    m = re.search(r"out\[0\.\.7\] = ((?:[0-9]+ ){7}[0-9]+)", read(name))
    if not m:
        raise AssertionError(f"{name}.asm has no '; Run:' words")
    return tuple(int(w) for w in m.group(1).split())


def runner_words(name):
    """The tuple scripts/run_gpu_tests.py expects for `name` (its table is local to the run loop, so read the source)."""
    with open(os.path.join(ROOT, "scripts", "run_gpu_tests.py"), encoding="utf-8") as f:
        m = re.search(r'"%s": \[([^\]]*)\]' % re.escape(name), f.read())
    return tuple(int(w, 0) for w in m.group(1).split(",")) if m else None


def interpreter_words(name):
    spv = support.translate_fixture(
        name, fssm=bytes(compute_header(os.path.join(SHADERS, name + ".asm")))
    )
    memory = {}
    si.run_workgroup(
        spv,
        32,
        32,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0)},
        push_constants=OUT_V + [0] * 28,
        memory=memory,
    )
    return memory


class Batch4GpuFixtureTests(unittest.TestCase):
    def test_interpreter_stores_the_header_words(self):
        for name, want in FIXTURES.items():
            with self.subTest(name=name):
                memory = interpreter_words(name)
                self.assertEqual(tuple(memory.get(4 * i, 0xDEAD0000) for i in range(8)), want)
                if name not in ATOMIC:
                    # lane 0 stored the eight dwords and nothing else did
                    self.assertEqual(sorted(memory), [4 * i for i in range(8)])

    def test_header_check_and_runner_agree(self):
        cs_dumps = dict(check.CS_DUMPS)
        for name, want in FIXTURES.items():
            with self.subTest(name=name):
                self.assertEqual(header_words(name), want)
                self.assertEqual(tuple(cs_dumps.get(name, ())), want)
                self.assertEqual(runner_words(name), want)

    def test_wave32_registration(self):
        for name in FIXTURES:
            with self.subTest(name=name):
                self.assertIn(name, check.WAVE32_DUMPS)
                if name in ATOMIC or name == "cs_w32_sdst_exec":
                    # 32 EXEC bits wide, and the atomics are not idempotent: one wave must hold the only lane 0
                    self.assertIn(name, check.WAVE32_LANE_DUMPS)
                else:
                    self.assertNotIn(name, check.WAVE32_LANE_DUMPS)


if __name__ == "__main__":
    unittest.main()
