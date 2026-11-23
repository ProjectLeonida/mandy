"""CPU checks for the batch-5 GPU fixtures (docs/design.md rows 321, 322, 324).

Each fixture runs the translator on its generated disassembly (tests/semantic/fixtures/disasm, made by
regen_fixtures.py) and executes all 32 invocations in the mini interpreter: lane 0 alone passes the fixture's EXEC
cut and runs the LDS atomics and stores the eight results, so the memory holds the words its header derives. The
same words are asserted against the three places that state them: the .asm header, check.py's CS_DUMPS and
scripts/run_gpu_tests.py's table. A GPU run of either runner then compares them with what the device writes.
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
    # ds_mskor_b32 / ds_mskor_rtn_b32, DATA2 bits outside DATA ORed in
    "cs_w32_lds_mskor": (0x11223344, 0x11220008, 0xFFFF0000, 0xFF000001,
                         0xDEADBEEF, 0x5F5F5F5F, 0xFEEDFACE, 0x12345678),
    # ds_inc/dec/rsub_u32 boundaries, DATA = 5
    "cs_w32_lds_atomics_incdec": (5, 0, 0, 5, 0, 4, 5, 0xFFFFFFFE),
    # the _rtn_ forms return the pre-op word
    "cs_w32_lds_atomics_rtn": (4, 5, 0, 5, 7, 0xFFFFFFFE, 5, 0),
    # ds_min/max_f32: NaN, +-0 and +-inf
    "cs_w32_lds_minmax_f32": (0x80000000, 0x00000000, 0x3F800000, 0x3F800000,
                              0x7FC00000, 0x3F800000, 0x3F800000, 0x3F800000),
    # ds_wrxchg_rtn_b64 and ds_wrxchg2_rtn_b64, old value in vdst order, *8 offsets
    "cs_w32_lds_wrxchg_b64": (0x11111111, 0x88888888, 0x22222222, 0x44444444,
                              0x33333333, 0xCCCCCCCC, 0xDDDDDDDD, 0xEEEEEEEE),
    # ds_wrxchg2st64_rtn_b64, old values then the words it wrote at *64 offsets
    "cs_w32_lds_wrxchg2st64_b64": (0x11111111, 0x44444444, 0x55555555, 0x88888888,
                                   0x33333333, 0x77777777, 0xAAAAAAAA, 0xEEEEEEEE),
}
# The output V# in s[0:3]: base 0, num_records max, FORMAT 32_FLOAT, DST_SEL xyzw (as vk_compute_test sets it); the
# heap base in push-constant dwords 32/33 is 0, so an Xbox VA is the memory word index below.
OUT_V = [0, 0, 0xFFFFFFFF, 0x31016FAC]


def read(name):
    with open(os.path.join(SHADERS, name + ".asm"), encoding="utf-8") as f:
        return f.read()


def header_words(name):
    """The eight words of the fixture's "; Run:" line, decimal or 0x hex."""
    m = re.search(r"out\[0\.\.7\] = ([0-9a-fA-Fx ]+)", read(name))
    if not m:
        raise AssertionError(f"{name}.asm has no '; Run:' words")
    words = m.group(1).split()
    if len(words) != 8:
        raise AssertionError(f"{name}.asm names {len(words)} words, not 8")
    return tuple(int(w, 0) for w in words)


def runner_words(name):
    """The tuple scripts/run_gpu_tests.py expects for `name` (its table is local to the run loop, so read the source)."""
    with open(os.path.join(ROOT, "scripts", "run_gpu_tests.py"), encoding="utf-8") as f:
        m = re.search(r'"%s": \[([^\]]*)\]' % re.escape(name), f.read())
    return tuple(int(w, 0) for w in m.group(1).replace("\n", " ").split(",")) if m else None


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


class Batch5GpuFixtureTests(unittest.TestCase):
    def test_interpreter_stores_the_header_words(self):
        for name, want in FIXTURES.items():
            with self.subTest(name=name):
                memory = interpreter_words(name)
                self.assertEqual(tuple(memory.get(4 * i, 0xDEAD0000) for i in range(8)), want)
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
                # 32 EXEC bits wide, and the LDS atomics are not idempotent: one wave must hold the only lane 0
                self.assertIn(name, check.WAVE32_LANE_DUMPS)


if __name__ == "__main__":
    unittest.main()
