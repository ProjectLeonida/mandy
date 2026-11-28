"""CPU checks for the packed 16-bit integer GPU fixtures (cs_pk_int16_u, cs_pk_int16_i).

Each fixture runs the translator on its generated disassembly (tests/semantic/fixtures/disasm, made by
regen_fixtures.py) and executes all 32 invocations in the mini interpreter: lane 0 alone passes the fixture's EXEC
cut and stores the eight results, so the memory holds exactly the words its header derives. The same words are
asserted against the three places that state them: the .asm header, check.py's CS_DUMPS and
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
    "cs_pk_int16_u": (
        0xFFFF0007,
        0x0005FFFD,
        0xFFFA0004,
        0x0000000D,
        0xFFFE0004,
        0x00030001,
        0xFFFF0005,
        0x00000000,
    ),
    "cs_pk_int16_i": (
        0x00018000,
        0x00038002,
        0x80010001,
        0x0002FFFF,
        0xFFFF8001,
        0x80008000,
        0x7FFF8000,
        0x7FFF7FFF,
    ),
}
# The output V# in s[0:3]: base 0, 32 bytes, FORMAT 32_FLOAT, DST_SEL xyzw (as in cs_cvt_pk_u8.asm).
OUT_V = [0, 0, 32, 0x31016FAC]


def read(name):
    with open(os.path.join(SHADERS, name + ".asm"), encoding="utf-8") as f:
        return f.read()


def header_words(name):
    """The eight hex words of the fixture's "; Run:" line."""
    m = re.search(r"Run:.*?out\[0\.\.7\] = (.*?) \(hex\)", read(name))
    if not m:
        raise AssertionError(f"{name}.asm has no '; Run:' words")
    return tuple(int(w, 16) for w in m.group(1).split())


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


class PkInt16GpuFixtureTests(unittest.TestCase):
    def test_interpreter_stores_the_header_words(self):
        for name, want in FIXTURES.items():
            with self.subTest(name=name):
                memory = interpreter_words(name)
                self.assertEqual(
                    sorted(memory), [4 * i for i in range(8)]
                )  # lane 0 stored, the other 31 did not
                self.assertEqual(tuple(memory[4 * i] for i in range(8)), want)

    def test_header_check_and_runner_agree(self):
        cs_dumps = dict(check.CS_DUMPS)
        for name, want in FIXTURES.items():
            with self.subTest(name=name):
                self.assertEqual(header_words(name), want)
                self.assertEqual(cs_dumps.get(name), list(want))
                self.assertEqual(runner_words(name), want)

    def test_wave32_registration(self):
        for name in FIXTURES:
            with self.subTest(name=name):
                self.assertIn(name, check.WAVE32_DUMPS)
                self.assertNotIn(name, check.WAVE32_LANE_DUMPS)


if __name__ == "__main__":
    unittest.main()
