"""cs_w32_late_ext: the GPU test of ds_mskor, v_add/sub_nc_i16 clamp and the s_buffer_load constant SGPR offset,
run here on the CPU too.

Standard library only; no GPU, no LLVM, no game files.

runtime/test/shaders/cs_w32_late_ext.asm (derivations in its header) stores eight words from lane 0. This runs its
fixture as one wave32 group through spirv_mini_interp with the buffer vk_compute_test gives it: every byte 0xEE and
a V# with num_records 0xFFFFFFFF (runtime/test/vk_compute_test.cpp). It checks the words against values computed
here, that check.py and scripts/run_gpu_tests.py expect the same words, and that EXEC is cut to lane 0 before the LDS
ops (spirv_mini_interp gives each invocation its own LDS, so lanes racing on one address are invisible here, #1850).
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
struct.pack_into("<4I", FSSM, 0x138, 0x8088, 32, 1, 1)      # 4 user SGPRs, wg id in s4, 512 bytes of LDS, 32x1x1
FSSM = bytes(FSSM)
VSHARP = [0, 0, 0xFFFFFFFF, 0x31016FAC]                      # vk_compute_test's V#: base 0, num_records max, raw
FILL = 0xEEEEEEEE


def i16(x):
    x &= 0xFFFF
    return x - 0x10000 if x >> 15 else x


def sbuf(sgpr_off, imm, num_records=0xFFFFFFFF):
    off = (sgpr_off & 0xFFFFFFFC) + (imm & 0xFFFFFFFC)        # no 32-bit wrap (#1948)
    return 0 if off + 4 > num_records else FILL


EXPECTED = [
    0x12345678,                                                # ds_mskor_rtn_b32: old value
    (0x12345678 & ~0xFF00 | 0xAB00) & M32,                     # the LDS word after it
    (M32 & ~0xFFFF | 1) & M32,                                 # ds_mskor_b32
    max(-32768, min(32767, i16(0x7FFF) + 1)) & 0xFFFF,         # v_add_nc_i16 clamp
    max(-32768, min(32767, i16(0x8000) - 1)) & 0xFFFF,         # v_sub_nc_i16 clamp
    (i16(0x7FFF) + 1) & 0xFFFF,                                # v_add_nc_i16, wraps
    sbuf(-4, 20),                                              # s_buffer_load, s12 = -4: out of range
    sbuf(12, 20),                                              # s_buffer_load, s12 = 12: the fill
]


class LateExtFixture(unittest.TestCase):
    def test_reference(self):
        self.assertEqual(EXPECTED, [0x12345678, 0x1234AB78, 0xFFFF0001, 0x7FFF, 0x8000, 0x8000, 0, 0xEEEEEEEE])

    def test_translated_shader(self):
        spv = support.translate_fixture("cs_w32_late_ext", fssm=FSSM)
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
            m = re.search(r'"cs_w32_late_ext"\)?[,:]\s*\[([^\]]*)\]', text)
            with self.subTest(file="/".join(rel)):
                self.assertIsNotNone(m)
                self.assertEqual([int(v, 0) for v in m.group(1).split(",")], EXPECTED)

    def test_lds_ops_run_in_lane_0_only(self):
        with open(os.path.join(ROOT, "runtime", "test", "shaders", "cs_w32_late_ext.asm"), encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith(";")]
        self.assertLess(lines.index("v_cmpx_eq_u32_e32 0, v0"),
                        min(i for i, ln in enumerate(lines) if ln.startswith("ds_")))


if __name__ == "__main__":
    unittest.main()
