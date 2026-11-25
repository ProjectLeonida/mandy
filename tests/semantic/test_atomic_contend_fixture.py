"""cs_w32_atomic_contend: the GPU test of buffer_atomic_inc / buffer_atomic_dec under contention, run on the CPU too.

Standard library only; no GPU, no LLVM, no game files.

runtime/test/shaders/cs_w32_atomic_contend.asm (derivations in its header) has all 32 lanes of a wave32 group run
every op on the same dwords of the harness buffer (every byte 0xEE, a V# with num_records 0xFFFFFFFF). spirv_mini_interp
runs the lanes in lockstep, so their compare-exchanges collide and retry here as well; the GPU is where they race
for real. The reference applies the 32 ops one after another: the same op on every lane leaves the same final word
in any order, and the returned old values are the sequential ones in some order, which is all the sums and the
umax / umin depend on. The test also checks that check.py and scripts/run_gpu_tests.py expect the same words.
"""
import os
import re
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)  # python -I leaves the script's folder off sys.path

import support  # noqa: E402  (support puts tools/ on sys.path, honouring LEONIDA_TOOLS_DIR)
import spirv_mini_interp as si  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))
M32 = 0xFFFFFFFF
F = 0xEEEEEEEE                                               # the harness's fill
LANES = 32

FSSM = bytearray(0x148)
struct.pack_into("<4I", FSSM, 0x138, 0x88, 32, 1, 1)        # 4 user SGPRs, wg id in s4, no LDS, 32x1x1
FSSM = bytes(FSSM)
VSHARP = [0, 0, 0xFFFFFFFF, 0x31016FAC]                      # vk_compute_test's V#: base 0, num_records max, raw


def inc(m, d):
    return 0 if m >= d else (m + 1) & M32


def dec(m, d):
    return d if m == 0 or m > d else m - 1


def sequential(fn, d):
    """(final word, returned old values) of LANES fn ops on a word starting at F."""
    m, olds = F, []
    for _ in range(LANES):
        olds.append(m)
        m = fn(m, d)
    return m, olds


INC_MAX, INC_OLDS = sequential(inc, M32)
DEC_MAX, DEC_OLDS = sequential(dec, M32)
EXPECTED = [
    INC_MAX,
    (F + sum(1 << (o - F) for o in INC_OLDS)) & M32,
    sequential(inc, 19)[0],
    DEC_MAX,
    sequential(dec, 5)[0],
    max([F] + INC_OLDS),
    min([F] + DEC_OLDS),
    (F + sum(1 << (F - o) for o in DEC_OLDS)) & M32,
]


class AtomicContendFixture(unittest.TestCase):
    def test_reference(self):
        self.assertEqual(EXPECTED, [0xEEEEEF0E, 0xEEEEEEED, 0xB, 0xEEEEEECE, 0x4, 0xEEEEEF0D, 0xEEEEEECF, 0xEEEEEEED])
        # word 2 must tell inc from dec: with D = 20 (or 2, 6, 8) both end at 10 (@sovljanskii on #4331)
        self.assertNotEqual(sequential(inc, 19)[0], sequential(dec, 19)[0])

    def test_a_lost_update_changes_the_sums(self):
        # one returned value twice and another missing: the bit sum is not F + ~0 any more
        olds = INC_OLDS[:-1] + [INC_OLDS[0]]
        self.assertNotEqual((F + sum(1 << (o - F) for o in olds)) & M32, EXPECTED[1])

    def test_translated_shader(self):
        spv = support.translate_fixture("cs_w32_atomic_contend", fssm=FSSM)
        mem = {4 * k: F for k in range(64)}
        si.run_workgroup(spv, LANES, 32, inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0),
                                                                "%sg_lane": i},
                         push_constants=VSHARP + [0] * 28, memory=mem)
        for k, want in enumerate(EXPECTED):
            with self.subTest(word=k):
                self.assertEqual(mem[4 * k], want)

    def test_gpu_runners_expect_the_same_words(self):
        for rel in (("runtime", "test", "shaders", "check.py"), ("scripts", "run_gpu_tests.py")):
            with open(os.path.join(ROOT, *rel), encoding="utf-8") as f:
                text = f.read()
            m = re.search(r'"cs_w32_atomic_contend"\)?[,:]\s*\[([^\]]*)\]', text)
            with self.subTest(file="/".join(rel)):
                self.assertIsNotNone(m)
                self.assertEqual([int(v, 0) for v in m.group(1).split(",")], EXPECTED)


if __name__ == "__main__":
    unittest.main()
