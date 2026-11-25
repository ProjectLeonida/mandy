"""cs_w32_lds_add_f32: the GPU test of ds_add_f32 / ds_add_rtn_f32 (#3120), run here on the CPU too.

Standard library only; no GPU, no LLVM, no game files.

runtime/test/shaders/cs_w32_lds_add_f32.asm (derivations in its header) stores eight words from lane 0. This runs its
fixture as one wave32 group through spirv_mini_interp with the buffer vk_compute_test gives it (every byte 0xEE, a V#
with num_records 0xFFFFFFFF), checks the words against a reference that adds in double precision and rounds once
to binary32 (exact for these inputs, whose exponents are close: in general a double sum of two binary32 values can
round, and rounding it again to binary32 could then differ from one rounding), that check.py and
scripts/run_gpu_tests.py expect the same words, and that EXEC is cut to lane 0 before the first LDS op.
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

FSSM = bytearray(0x148)
struct.pack_into("<4I", FSSM, 0x138, 0x8088, 32, 1, 1)      # 4 user SGPRs, wg id in s4, 512 bytes of LDS, 32x1x1
FSSM = bytes(FSSM)
VSHARP = [0, 0, 0xFFFFFFFF, 0x31016FAC]                      # vk_compute_test's V#: base 0, num_records max, raw
FILL = 0xEEEEEEEE


def f32(b):
    return struct.unpack("<f", struct.pack("<I", b))[0]


def bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def add(m, d):
    """binary32 m + d, rounded to nearest even (the double sum is exact for this file's inputs); past FLT_MAX +-inf."""
    total = f32(m) + f32(d)
    try:
        return bits(total)
    except OverflowError:
        return 0x7F800000 if total > 0 else 0xFF800000


EXPECTED = [
    0x3FC00000,                          # ds_add_rtn_f32: the old value, 1.5
    add(0x3FC00000, 0x40100000),         # 1.5 + 2.25
    add(0x4B800000, 0x3F800000),         # 2^24 + 1: tie, down to even
    add(0x4B800001, 0x3F800000),         # (2^24 + 2) + 1: tie, up to even
    add(0x80000000, 0x00000000),         # -0 + +0
    add(0x7F61B1E6, 0x7F61B1E6),         # 3e38 + 3e38: overflow
    add(0x3F800000, 0xBF400000),         # ds_add_f32: 1.0 + -0.75
    add(0x40000000, 0x3F000000),         # offset:4: 2.0 + 0.5
]


class LdsAddF32Fixture(unittest.TestCase):
    def test_reference(self):
        self.assertEqual(EXPECTED, [0x3FC00000, 0x40700000, 0x4B800000, 0x4B800002, 0, 0x7F800000, 0x3E800000,
                                    0x40200000])

    def test_translated_shader(self):
        spv = support.translate_fixture("cs_w32_lds_add_f32", fssm=FSSM)
        mem = {4 * k: FILL for k in range(64)}
        si.run_workgroup(spv, 32, 32, inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0),
                                                             "%sg_lane": i},
                         push_constants=VSHARP + [0] * 28, memory=mem)
        for k, want in enumerate(EXPECTED):
            with self.subTest(word=k):
                self.assertEqual(mem[4 * k], want)

    def test_gpu_runners_expect_the_same_words(self):
        for rel in (("runtime", "test", "shaders", "check.py"), ("scripts", "run_gpu_tests.py")):
            with open(os.path.join(ROOT, *rel), encoding="utf-8") as f:
                text = f.read()
            m = re.search(r'"cs_w32_lds_add_f32"\)?[,:]\s*\[([^\]]*)\]', text)
            with self.subTest(file="/".join(rel)):
                self.assertIsNotNone(m)
                self.assertEqual([int(v, 0) for v in m.group(1).split(",")], EXPECTED)

    def test_lds_ops_run_in_lane_0_only(self):
        with open(os.path.join(ROOT, "runtime", "test", "shaders", "cs_w32_lds_add_f32.asm"), encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith(";")]
        self.assertEqual(lines[0], "v_cmpx_eq_u32_e32 0, v0")


if __name__ == "__main__":
    unittest.main()
