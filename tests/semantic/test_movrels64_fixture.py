"""cs_w32_movrels64: the GPU test of s_movrels_b64 (and s_movrels_b32 at a base register), run here on the CPU too.

Standard library only; no GPU, no LLVM, no game files.

runtime/test/shaders/cs_w32_movrels64.asm (derivations in its header) stores eight words from lane 0. This runs its
fixture as one wave32 group through spirv_mini_interp with the buffer vk_compute_test gives it (every byte 0xEE, a V#
with num_records 0xFFFFFFFF), checks the words against a model of the SGPR file (D = SGPR[S0 + M0], the b64 form the
pair, the source read before the destination is written), and that check.py and scripts/run_gpu_tests.py expect the
same words.
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
struct.pack_into("<4I", FSSM, 0x138, 0x88, 32, 1, 1)        # 4 user SGPRs, wg id in s4, no LDS, 32x1x1
FSSM = bytes(FSSM)
VSHARP = [0, 0, 0xFFFFFFFF, 0x31016FAC]                      # vk_compute_test's V#: base 0, num_records max, raw
FILL = 0xEEEEEEEE


def model():
    s = {20 + k: 0x5EED0000 + 0x111 * k for k in range(8)}

    def rels(src, m0, n):
        return [s[src + m0 + i] for i in range(n)]

    s[10], s[11] = rels(20, 1, 2)
    s[14], s[15] = rels(20, 4, 2)
    s[12], = rels(22, 3, 1)
    scc = 1                                                  # s_cmp_eq_u32 s5, 3 with s5 = 3; s_movrels keeps it
    s[22], s[23] = rels(20, 1, 2)                            # reads s21, s22 before writing s22
    return [s[10], s[11], s[14], s[15], s[12], s[22], s[23], scc]


EXPECTED = model()


class Movrels64Fixture(unittest.TestCase):
    def test_reference(self):
        self.assertEqual(EXPECTED, [0x5EED0111, 0x5EED0222, 0x5EED0444, 0x5EED0555, 0x5EED0555, 0x5EED0111,
                                    0x5EED0222, 1])

    def test_translated_shader(self):
        spv = support.translate_fixture("cs_w32_movrels64", fssm=FSSM)
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
            m = re.search(r'"cs_w32_movrels64"\)?[,:]\s*\[([^\]]*)\]', text)
            with self.subTest(file="/".join(rel)):
                self.assertIsNotNone(m)
                self.assertEqual([int(v, 0) for v in m.group(1).split(",")], EXPECTED)

    def test_sgpr_pairs_are_even_aligned(self):
        # LLVM 22+ rejects a 64-bit SGPR pair with an odd base ("invalid register alignment"), and one fixture
        # llvm-mc can't assemble stops the whole GPU run (TEST-961), so no SGPR range here may start at an odd SGPR
        with open(os.path.join(ROOT, "runtime", "test", "shaders", "cs_w32_movrels64.asm"), encoding="utf-8") as f:
            body = "".join(ln for ln in f if not ln.startswith(";"))
        pairs = [(int(a), int(b)) for a, b in re.findall(r"s\[(\d+):(\d+)\]", body)]
        self.assertTrue(pairs)
        self.assertEqual([p for p in pairs if p[0] % 2], [])        # any SGPR range: an even base

    def test_m0_reaches_the_dynamic_path(self):
        # two of the reads take M0 from v_readfirstlane, which the translator can't fold to a constant
        with open(os.path.join(ROOT, "runtime", "test", "shaders", "cs_w32_movrels64.asm"), encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith(";")]
        self.assertEqual(sum(ln.startswith("v_readfirstlane_b32 s5") for ln in lines), 2)
        self.assertEqual(sum(ln == "s_mov_b32 m0, s5" for ln in lines), 2)


if __name__ == "__main__":
    unittest.main()
