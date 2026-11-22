"""cs_w32_saveexec_ext: the GPU test of #1657's saveexec / wrexec kinds, run here on the CPU too.

Standard library only; no GPU, no LLVM, no game files.

runtime/test/shaders/cs_w32_saveexec_ext.asm runs s_xor/andn1/orn1/orn2/nand/nor/xnor_saveexec_b32 and
s_andn1_wrexec_b32 with EXEC = lanes 0..23 and S0 = lanes 10..29 (both from compares) and stores, per op,
(s_bcnt1 << 8) | s_ff1 of the new EXEC. This runs its fixture (tests/semantic/regen_fixtures.py) as one wave32 group
through spirv_mini_interp, checks the words against tools/cf_ref.py's _SAVEEXEC (the FUZZ-8 ISA model), and checks
that check.py and scripts/run_gpu_tests.py expect the same words on the GPU.
"""
import os
import re
import struct
import unittest

import support
import spirv_mini_interp as si
import cf_ref

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
M32 = 0xFFFFFFFF

FSSM = bytearray(0x148)
struct.pack_into("<4I", FSSM, 0x138, 0x88, 32, 1, 1)     # 4 user SGPRs (the V#), workgroup id in s4, 32x1x1
FSSM = bytes(FSSM)
VSHARP = [0, 0, 0x1000, 0x31016FAC]

S0 = sum(1 << i for i in range(10, 30))
EXEC = sum(1 << i for i in range(24))
KINDS = ("xor", "andn1", "orn1", "orn2", "nand", "nor", "xnor", "andn1")   # the last is s_andn1_wrexec_b32


def word(mask):
    return bin(mask).count("1") << 8 | ((mask & -mask).bit_length() - 1)


EXPECTED = [word(cf_ref._SAVEEXEC[k](S0, EXEC, M32) & M32) for k in KINDS]


class SaveexecExtFixture(unittest.TestCase):
    def test_reference(self):
        self.assertEqual(EXPECTED, [0x1000, 0xA00, 0x1A00, 0x160A, 0x1200, 0x21E, 0x100A, 0xA00])

    def test_translated_shader(self):
        spv = support.translate_fixture("cs_w32_saveexec_ext", fssm=FSSM)
        mem = si.run_workgroup(spv, 32, 32, inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0),
                                                                   "%sg_lane": i},
                               push_constants=VSHARP + [0] * 28, memory={})
        self.assertEqual(sorted(mem), [4 * k for k in range(8)])
        for k, want in enumerate(EXPECTED):
            with self.subTest(word=k, op=KINDS[k]):
                self.assertEqual(mem[4 * k], want)

    def test_gpu_runners_expect_the_same_words(self):
        for rel in (("runtime", "test", "shaders", "check.py"), ("scripts", "run_gpu_tests.py")):
            with open(os.path.join(ROOT, *rel), encoding="utf-8") as f:
                text = f.read()
            m = re.search(r'"cs_w32_saveexec_ext"\)?[,:]\s*\[([^\]]*)\]', text)
            with self.subTest(file="/".join(rel)):
                self.assertIsNotNone(m)
                self.assertEqual([int(v, 0) for v in m.group(1).split(",")], EXPECTED)


if __name__ == "__main__":
    unittest.main()
