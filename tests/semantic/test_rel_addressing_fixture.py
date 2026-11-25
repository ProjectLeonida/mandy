"""cs_swaprel and cs_movrels: the GPU fixtures for v_swaprel_b32 and s_movrels_b32/b64, run on the CPU too.

Standard library only; no GPU, no LLVM, no game files.

runtime/test/shaders/cs_swaprel.asm swaps VGPR[D + M0[25:16]] with VGPR[S0 + M0[9:0]] three times, including
a self-swap and a set of ignored M0 bits; cs_movrels.asm reads SGPR[S0 + M0] (and the b64 pair) through M0,
including an overlapping destination and the SCC the instruction leaves alone. Derivations are in each .asm
header. These run the fixtures (tests/semantic/regen_fixtures.py) through spirv_mini_interp, compare the words
with plain-Python register files, and check that check.py and scripts/run_gpu_tests.py expect the same words
on the GPU.
"""

import os
import re
import struct
import unittest

import support
import spirv_mini_interp as si

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
M32 = 0xFFFFFFFF

# 4 user SGPRs (the output V#), workgroup id in s4, 512 bytes of LDS (LDS_SIZE 1), 32x1x1 threads.
FSSM = bytearray(0x148)
struct.pack_into("<4I", FSSM, 0x138, 0x8088, 32, 1, 1)
FSSM = bytes(FSSM)
VSHARP = [0, 0, 0x1000, 0x31016FAC]


def m0_of(dst_off: int, src_off: int, extra: int = 0) -> int:
    return ((dst_off << 16) | src_off | extra) & M32


def swaprel(vgpr, dst, src, m0):
    """v_swaprel_b32 on a copy of the VGPR file: the two M0 fields are the only bits used."""
    vgpr = list(vgpr)
    a, b = dst + ((m0 >> 16) & 0x3FF), src + (m0 & 0x3FF)
    vgpr[a], vgpr[b] = vgpr[b], vgpr[a]
    return vgpr


def movrels(sgpr, dst, src, m0, width=1):
    """s_movrels_b32/b64 on a copy of the SGPR file: every read before the write, M0 used whole."""
    sgpr = list(sgpr)
    vals = [sgpr[(src + m0 + w) % 128] for w in range(width)]
    for w, v in enumerate(vals):
        sgpr[(dst + w) % 128] = v
    return sgpr


def run(name):
    spv = support.translate_fixture(name, fssm=FSSM)
    mem = si.run_workgroup(
        spv,
        1,
        32,
        inputs_for=lambda i: {
            "%wg_id": (0, 0, 0),
            "%local_id": (i, 0, 0),
            "%sg_lane": i,
        },
        push_constants=VSHARP + [0] * 28,
        memory={},
    )
    return [mem[4 * k] for k in range(8)]


class SwaprelFixture(unittest.TestCase):
    def test_reference(self):
        vgpr = [0] * 256
        for r in range(1, 9):
            vgpr[r] = 11 * r
        vgpr = swaprel(vgpr, 1, 2, m0_of(4, 2))  # v5 <-> v4
        first = [vgpr[r] for r in (1, 2, 4, 5)]
        vgpr = swaprel(
            vgpr, 1, 5, m0_of(3, 2, 0xFC00FC00)
        )  # v4 <-> v7, ignored bits set
        second = [vgpr[r] for r in (4, 7)]
        vgpr = swaprel(vgpr, 1, 1, m0_of(5, 5))  # v6 with itself
        self.assertEqual(
            first + second + [vgpr[6], vgpr[8]], [11, 22, 55, 44, 77, 55, 66, 88]
        )

    def test_translated_shader(self):
        self.assertEqual(run("cs_swaprel"), [11, 22, 55, 44, 77, 55, 66, 88])

    def test_gpu_runners_expect_the_same_words(self):
        for rel in (
            ("runtime", "test", "shaders", "check.py"),
            ("scripts", "run_gpu_tests.py"),
        ):
            with open(os.path.join(ROOT, *rel), encoding="utf-8") as f:
                text = f.read()
            m = re.search(r'"cs_swaprel"\)?[,:]\s*\[([^\]]*)\]', text)
            with self.subTest(file="/".join(rel)):
                self.assertIsNotNone(m)
                got = [
                    int(v, 0)
                    for v in m.group(1).replace("\n", " ").split(",")
                    if v.strip()
                ]
                self.assertEqual(got, [11, 22, 55, 44, 77, 55, 66, 88])


class MovrelsFixture(unittest.TestCase):
    def test_reference(self):
        sgpr = [0] * 128
        for r, v in ((10, 0xAA), (11, 0xBB), (12, 0xCC), (13, 0xDD), (14, 0xEE)):
            sgpr[r] = v
        sgpr = movrels(sgpr, 4, 10, 0)  # s4 = s10
        sgpr = movrels(sgpr, 5, 10, 2)  # s5 = s12
        sgpr = movrels(sgpr, 6, 10, 1, width=2)  # s6:s7 = s11:s12
        sgpr = movrels(sgpr, 10, 10, 1)  # reads s11 before writing s10
        scc = 1  # s_cmp_eq_u32 s4, 0xAA
        sgpr = movrels(sgpr, 8, 10, 0)  # s8 = the new s10
        s9 = 1 if scc else 0  # s_cselect reads the SCC the movrels left alone
        self.assertEqual(
            [sgpr[r] for r in (4, 5, 6, 7, 10, 11)] + [s9, sgpr[14]],
            [0xAA, 0xCC, 0xBB, 0xCC, 0xBB, 0xBB, 1, 0xEE],
        )

    def test_translated_shader(self):
        self.assertEqual(run("cs_movrels"), [170, 204, 187, 204, 187, 187, 1, 238])

    def test_gpu_runners_expect_the_same_words(self):
        for rel in (
            ("runtime", "test", "shaders", "check.py"),
            ("scripts", "run_gpu_tests.py"),
        ):
            with open(os.path.join(ROOT, *rel), encoding="utf-8") as f:
                text = f.read()
            m = re.search(r'"cs_movrels"\)?[,:]\s*\[([^\]]*)\]', text)
            with self.subTest(file="/".join(rel)):
                self.assertIsNotNone(m)
                got = [
                    int(v, 0)
                    for v in m.group(1).replace("\n", " ").split(",")
                    if v.strip()
                ]
                self.assertEqual(got, [170, 204, 187, 204, 187, 187, 1, 238])


if __name__ == "__main__":
    unittest.main()
