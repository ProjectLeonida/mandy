"""cs_w32_pk_rel_ext: the GPU test of packed 16-bit ints, v_swaprel_b32, s_movrels_b32 and ds_or/xor_b64, on the CPU.

Standard library only; no GPU, no LLVM, no game files.

runtime/test/shaders/cs_w32_pk_rel_ext.asm (derivations in its header) has lane 0 of a wave32 group store
v_pk_add_u16 clamp, v_pk_sub_i16, v_pk_mad_u16 with op_sel, v_pk_max_i16, the two registers v_swaprel_b32 swapped,
an s_movrels_b32 result and an LDS word from ds_or_rtn_b64 / ds_xor_b64. This runs its fixture as one wave32 group
through spirv_mini_interp, checks the words against values computed here, and checks that check.py and
scripts/run_gpu_tests.py expect the same words.

spirv_mini_interp gives every invocation its own LDS, so this cannot see lanes racing on one LDS address. The first
version ran the LDS ops in all 32 lanes, which passed here and failed on GPUs (an even number of ds_xor_b64 cancel;
found by @borislisandro and @tumbione on #1850); the .asm now cuts EXEC to lane 0 first, and
test_lds_ops_run_in_lane_0_only checks that order in the source.
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

FSSM = bytearray(0x148)
struct.pack_into("<4I", FSSM, 0x138, 0x8088, 32, 1, 1)     # 4 user SGPRs, wg id in s4, 512 bytes of LDS, 32x1x1
FSSM = bytes(FSSM)
VSHARP = [0, 0, 0x1000, 0x31016FAC]


def halves(x):
    return x & 0xFFFF, x >> 16


def s16(x):
    return x - 0x10000 if x & 0x8000 else x


def pack(lo, hi):
    return (lo & 0xFFFF) | (hi & 0xFFFF) << 16


a0, a1 = halves(0xFFFF0001), halves(0x00010002)
b0, b1 = halves(0x80000005), halves(0x00010007)
m0, m1, m2 = halves(0x00030002), halves(0x00050004), halves(0x00070006)
x0, x1 = halves(0x8000FFFF), halves(0x00010000)
regs = {10 + k: 0x100 + k for k in range(6)}
d, s = 10 + 2, 11 + 4                        # M0 = (2 << 16) | 4
regs[d], regs[s] = regs[s], regs[d]
lds_hi = 0x11110000
old_hi = lds_hi
lds_hi |= 0x0000FF00                         # ds_or_rtn_b64, high dword of DATA
lds_hi ^= 0x00F00000                         # ds_xor_b64

EXPECTED = [
    pack(min(a0[0] + a1[0], 0xFFFF), min(a0[1] + a1[1], 0xFFFF)),            # v_pk_add_u16 clamp
    pack(s16(b0[0]) - s16(b1[0]), s16(b0[1]) - s16(b1[1])),                 # v_pk_sub_i16
    pack(m0[1] * m1[0] + m2[0], m0[0] * m1[1] + m2[1]),                     # v_pk_mad_u16 op_sel [1,0,0] / hi [0,1,1]
    pack(max(s16(x0[0]), s16(x1[0])), max(s16(x0[1]), s16(x1[1]))),         # v_pk_max_i16
    regs[12], regs[15],                                                     # v_swaprel_b32
    0x200 + 3,                                                              # s_movrels_b32 s30, s20 with M0 = 3
    old_hi ^ lds_hi,                                                        # ds_or_rtn_b64 old ^ LDS after xor
]


class PkRelExtFixture(unittest.TestCase):
    def test_reference(self):
        self.assertEqual(EXPECTED, [0xFFFF0003, 0x7FFFFFFE, 0x00110012, 0x00010000, 0x105, 0x102, 0x203, 0x00F0FF00])

    def test_translated_shader(self):
        spv = support.translate_fixture("cs_w32_pk_rel_ext", fssm=FSSM)
        mem = si.run_workgroup(spv, 32, 32, inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0),
                                                                   "%sg_lane": i},
                               push_constants=VSHARP + [0] * 28, memory={})
        self.assertEqual(sorted(mem), [4 * k for k in range(8)])
        for k, want in enumerate(EXPECTED):
            with self.subTest(word=k):
                self.assertEqual(mem[4 * k], want)

    def test_lds_ops_run_in_lane_0_only(self):
        with open(os.path.join(ROOT, "runtime", "test", "shaders", "cs_w32_pk_rel_ext.asm"), encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith(";")]
        cmpx = lines.index("v_cmpx_eq_u32_e32 0, v0")
        first_ds = min(i for i, ln in enumerate(lines) if ln.startswith("ds_"))
        self.assertLess(cmpx, first_ds)

    def test_gpu_runners_expect_the_same_words(self):
        for rel in (("runtime", "test", "shaders", "check.py"), ("scripts", "run_gpu_tests.py")):
            with open(os.path.join(ROOT, *rel), encoding="utf-8") as f:
                text = f.read()
            m = re.search(r'"cs_w32_pk_rel_ext"\)?[,:]\s*\[([^\]]*)\]', text)
            with self.subTest(file="/".join(rel)):
                self.assertIsNotNone(m)
                self.assertEqual([int(v, 0) for v in m.group(1).split(",")], EXPECTED)


if __name__ == "__main__":
    unittest.main()
