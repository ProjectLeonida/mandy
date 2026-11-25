"""SHADER-930: the 64-bit scalar data ops lower to 2x32-bit SPIR-V, with no Int64 capability.

Standard library only; no GPU, no LLVM, no game files.

s_brev_b64, s_bfe_u64, s_bfe_i64, s_bitset0_b64, s_bitset1_b64, s_flbit_i32_b64 and s_flbit_i32_i64 work on an SGPR
pair of 64-bit data. tools/rdna2spirv.py lowers each of them on the two dwords with 32-bit ops; until SHADER-930
s_bfe_u64/s_bfe_i64 built a 64-bit OpTypeInt value instead, which pulled the Int64 capability (and the PSB
declarations it comes with) into every module that used them. This file checks that:

  * no module that uses one of the seven ops declares Int64 or a 64-bit integer type (SGPR pair and constant sources);
  * s_bfe_u64/s_bfe_i64 give the RDNA2 result and SCC over an offset x width grid at the dword edges (offset 0, 1,
    31, 32, 33, 63; width 0, 1, 31, 32, 33, 63, 64, 65, 127), with SCC preset to the opposite value;
  * the GPU fixtures runtime/test/shaders/cs_w32_salu64_bfe.asm and cs_w32_salu64_bits.asm (their translator inputs
    in tests/semantic/fixtures/disasm, written by tests/semantic/regen_fixtures.py) give, through spirv_mini_interp,
    the words their headers derive, which are the words check.py and scripts/run_gpu_tests.py expect on the GPU.

Reference: RDNA2 ISA SOP1/SOP2 pseudocode. s_bfe_*64: offset = S1[5:0], width = S1[22:16]; a field running past bit 63
stops there (width = min(width, 64 - offset), the 64-bit form of the measured s_bfe_u32 rule and LLVM
constantFoldBFE), s_bfe_i64 sign-extends from the top bit kept, SCC = D != 0. The others do not write SCC.
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
M64 = (1 << 64) - 1
X = 0xF0E1D2C3_89ABCDEF

# 4 user SGPRs (the output V#), workgroup id in s4, 32x1x1 threads, as in test_salu64_fixture.py.
FSSM = bytearray(0x148)
struct.pack_into("<4I", FSSM, 0x138, 0x88, 32, 1, 1)
FSSM = bytes(FSSM)
VSHARP = [0, 0, 0x1000, 0x31016FAC]


def ctl(off: int, width: int) -> int:
    return (off & 63) | ((width & 127) << 16)


def bfe64(x: int, s1: int, signed: bool) -> int:
    off, width = s1 & 63, (s1 >> 16) & 127
    width = min(width, 64 - off)
    if width == 0:
        return 0
    v = (x >> off) & ((1 << width) - 1)
    if signed and v >> (width - 1):
        v -= 1 << width
    return v & M64


def brev64(x: int) -> int:
    return int(f"{x & M64:064b}"[::-1], 2)


def flbit_b64(x: int) -> int:
    return 64 - x.bit_length() if x else M32


def flbit_i64(x: int) -> int:
    sign = x >> 63
    for i in range(62, -1, -1):
        if (x >> i) & 1 != sign:
            return 63 - i
    return M32


def _scc_word(pairs):
    return 0x5CC00000 | sum(int(bfe64(x, c, False) != 0) << i for i, (x, c) in enumerate(pairs))


P = M64 & ~(1 << 32) & ~(1 << (127 & 63)) & ~(1 << 31)       # s_bitset0_b64 at 32, 127 and 31
EXPECTED = {
    "cs_w32_salu64_bfe": [
        bfe64(X, ctl(0, 64), False) & M32,
        bfe64(X, ctl(0, 64), False) >> 32,
        bfe64(X, ctl(33, 32), False) & M32,
        bfe64(X, ctl(1, 127), True) >> 32,
        bfe64(X, ctl(1, 127), True) & M32,
        bfe64(X, ctl(8, 32), True) >> 32,
        bfe64(X, ctl(4, 36), False) >> 32,
        _scc_word([(X, 0xFF80FFC5), (X, ctl(0, 64)), (0xFFFFFFFF, ctl(32, 32)), (1 << 63, ctl(0, 64))]),
    ],
    "cs_w32_salu64_bits": [
        brev64(X) & M32,
        brev64(X) >> 32,
        flbit_b64(0),
        flbit_b64(1),
        flbit_i64(M64),
        flbit_i64(0xFFFFFFFF_7FFFFFFF),
        P >> 32,
        P & M32,                                                  # SCC kept at 1 by all of them
    ],
}
HAND = {
    "cs_w32_salu64_bfe": [0x89ABCDEF, 0xF0E1D2C3, 0x7870E961, 0xF870E961, 0xC4D5E6F7, 0xFFFFFFFF, 0xC, 0x5CC0000A],
    "cs_w32_salu64_bits": [0xC34B870F, 0xF7B3D591, 0xFFFFFFFF, 63, 0xFFFFFFFF, 32, 0x7FFFFFFE, 0x7FFFFFFF],
}

OFFSETS = (0, 1, 31, 32, 33, 63)
WIDTHS = (0, 1, 31, 32, 33, 63, 64, 65, 127)
SOURCES = (X, 0x80000000_00000001)


def assert_no_int64(test, text, what):
    with test.subTest(no_int64=what):
        test.assertNotIn("OpCapability Int64", text)
        test.assertIsNone(re.search(r"OpTypeInt 64\b", text), "a 64-bit integer type")


def ps(name, lines):
    """A pixel shader of the lines, exporting v0..v3 (the translator picks the stage from the ps_ prefix)."""
    return support.translate_lines("ps_" + name, list(lines) + ["exp mrt0 v0, v1, v2, v3 done vm", "s_endpgm"])


def run_bfe(op, x, c, scc_in):
    """D (64 bits) and SCC of `op s[0:1], s[2:3], s4` with SCC preset to scc_in, and the module text."""
    spv = ps(op, [f"s_mov_b32 s2, {x & M32:#x}", f"s_mov_b32 s3, {x >> 32:#x}", f"s_mov_b32 s4, {c:#x}",
                  f"s_mov_b32 s5, {scc_in}", "s_cmp_eq_u32 s5, 1", f"{op} s[0:1], s[2:3], s4",
                  "s_cselect_b32 s6, 1, 0", "v_mov_b32 v0, s0", "v_mov_b32 v1, s1", "v_mov_b32 v2, s6",
                  "v_mov_b32 v3, 0"])
    w = si.run_fragment(spv)["%out_mrt0"]
    return w[0] | w[1] << 32, w[2], spv


class NoInt64(unittest.TestCase):
    PROGRAMS = {
        "s_brev_b64": ["s_brev_b64 s[0:1], s[2:3]", "s_brev_b64 s[0:1], -1"],
        "s_bfe_u64": ["s_bfe_u64 s[0:1], s[2:3], s4", "s_bfe_u64 s[0:1], -2, 0x200010"],
        "s_bfe_i64": ["s_bfe_i64 s[0:1], s[2:3], s4", "s_bfe_i64 s[0:1], -2, 0x200010"],
        "s_bitset0_b64": ["s_bitset0_b64 s[0:1], s4", "s_bitset0_b64 s[0:1], 40"],
        "s_bitset1_b64": ["s_bitset1_b64 s[0:1], s4", "s_bitset1_b64 s[0:1], 40"],
        "s_flbit_i32_b64": ["s_flbit_i32_b64 s0, s[2:3]", "s_flbit_i32_b64 s0, -1"],
        "s_flbit_i32_i64": ["s_flbit_i32_i64 s0, s[2:3]", "s_flbit_i32_i64 s0, -1"],
    }

    def test_single_ops(self):
        for op, forms in self.PROGRAMS.items():
            for form in forms:
                text = ps(op, ["s_mov_b32 s2, 0x89abcdef", "s_mov_b32 s3, 0xf0e1d2c3", "s_mov_b32 s4, 0x240004",
                               form, "v_mov_b32 v0, s0", "v_mov_b32 v1, s1", "v_mov_b32 v2, 0",
                               "v_mov_b32 v3, 0"])
                self.assertIn(op, form)
                assert_no_int64(self, text, form)

    def test_gpu_fixtures(self):
        # The compute fixtures store through the V#, whose buffer addressing declares Int64 with the PSB types on
        # its own; their scalar part, as a pixel shader with no memory access, must not.
        for name in EXPECTED:
            with open(os.path.join(HERE, "fixtures", "disasm", name + ".s"), encoding="utf-8") as f:
                lines = [ln.strip() for ln in f.read().splitlines()[1:] if ln.strip()]
            scalar = [ln for ln in lines if ln.startswith("s_") and ln != "s_endpgm"]
            self.assertTrue(any(ln.startswith(("s_bfe_", "s_brev_b64", "s_flbit")) for ln in scalar))
            assert_no_int64(self, ps(name, scalar + ["v_mov_b32 v0, s8", "v_mov_b32 v1, s9", "v_mov_b32 v2, s30",
                                                     "v_mov_b32 v3, s17"]), name)


class Bfe64Edges(unittest.TestCase):
    def test_reference(self):
        self.assertEqual(bfe64(M64, ctl(0, 64), False), M64)
        self.assertEqual(bfe64(M64, ctl(0, 65), False), M64)                       # width cut to 64
        self.assertEqual(bfe64(1 << 63, ctl(63, 1), True), M64)                    # one-bit field, set: -1
        self.assertEqual(bfe64(1 << 31, ctl(0, 32), True), 0xFFFFFFFF_80000000)    # 32-bit field sign-extends
        self.assertEqual(bfe64(1 << 32, ctl(0, 33), True), 0xFFFFFFFF_00000000)    # 33-bit field sign-extends
        self.assertEqual(bfe64(X, ctl(32, 32), True), 0xFFFFFFFF_F0E1D2C3)
        self.assertEqual(bfe64(X, ctl(17, 0), True), 0)

    def test_grid_values_and_scc(self):
        for op in ("s_bfe_u64", "s_bfe_i64"):
            for x in SOURCES:
                for off in OFFSETS:
                    for width in WIDTHS:
                        c = ctl(off, width)
                        want = bfe64(x, c, op == "s_bfe_i64")
                        scc = int(want != 0)
                        with self.subTest(op=op, x=hex(x), off=off, width=width):
                            got, got_scc, _ = run_bfe(op, x, c, 1 - scc)
                            self.assertEqual((hex(got), got_scc), (hex(want), scc))

    def test_ignored_control_bits(self):
        # S1[15:6] and S1[31:23] are not part of the offset or the width.
        for op in ("s_bfe_u64", "s_bfe_i64"):
            c = 0xFF80FFC0 | ctl(35, 20)
            want = bfe64(X, ctl(35, 20), op == "s_bfe_i64")
            with self.subTest(op=op):
                got, scc, spv = run_bfe(op, X, c, 0)
                self.assertEqual((got, scc), (want, 1))
                assert_no_int64(self, spv, op)


class GpuFixtures(unittest.TestCase):
    def test_reference(self):
        self.assertEqual(EXPECTED, HAND)

    def test_translated_shaders(self):
        for name, want in EXPECTED.items():
            spv = support.translate_fixture(name, fssm=FSSM)
            mem = si.run_workgroup(spv, 1, 32, inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0),
                                                                      "%sg_lane": i},
                                   push_constants=VSHARP + [0] * 28, memory={})
            with self.subTest(fixture=name):
                self.assertEqual(sorted(mem), [4 * k for k in range(8)])
                self.assertEqual([mem[4 * k] for k in range(8)], want)

    def test_headers_and_gpu_runners_expect_the_same_words(self):
        for name, want in EXPECTED.items():
            with open(os.path.join(ROOT, "runtime", "test", "shaders", name + ".asm"), encoding="utf-8") as f:
                m = re.search(r"out\[0\.\.7\] = ((?:\d+ ){7}\d+)", f.read())
            with self.subTest(fixture=name, file="asm header"):
                self.assertIsNotNone(m)
                self.assertEqual([int(v) for v in m.group(1).split()], want)
            for rel in (("runtime", "test", "shaders", "check.py"), ("scripts", "run_gpu_tests.py")):
                with open(os.path.join(ROOT, *rel), encoding="utf-8") as f:
                    text = f.read()
                m = re.search(r'"' + name + r'"\)?[,:]\s*\[([^\]]*)\]', text)
                with self.subTest(fixture=name, file="/".join(rel)):
                    self.assertIsNotNone(m)
                    self.assertEqual([int(v, 0) for v in m.group(1).split(",")], want)


if __name__ == "__main__":
    unittest.main()
