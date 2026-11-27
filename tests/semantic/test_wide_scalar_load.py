"""Multi-dword scalar loads as the fewest wide loads (uvec4, then uvec2), with the out-of-range result unchanged.

s_buffer_load_dwordxN: dword i is 0, and memory is not read, when off + 4*i + 4 > limit (docs/design.md,
Scalar/buffer memory, measured on gfx1030 and gfx1036). The ends grow with i, so when the last dword is in range all
are: Gen.scalar_load then reads the N dwords as uvec4 / uvec2 loads, one branch for the whole instruction, and keeps
the per-dword path (in range: read; out of range: 0, not read) for a load that is partly out of range. s_load_dwordxN
has no bound and is always wide. The per-dword form made RADV emit one global_load_b32, one v_readfirstlane and one
64-bit compare per dword (6,048 / 5,935 / 5,448 in one FSPX program with 365 s_buffer_load_dwordx16).

cs_wide_scalar_load.asm takes its V# from user SGPRs, so the bound is a run-time value: the fast path and the
per-dword path both run. test_scalar_mem_smem.py checks the addresses for every hardware rule on the same code.

    python -m unittest discover -s tests/semantic -p test_wide_scalar_load.py
"""
from pathlib import Path
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import support  # noqa: E402
import test_value_semantics as tv  # noqa: E402

OUT = 0x2000
OUT_V = 0x31016FAC          # raw output V#, 0x100 bytes at OUT (as test_null_vsharp)
CB = 0x4000                 # the constant buffer
CB_BYTES = 0x48             # NUM_RECORDS: 18 dwords
PTR = 0x6000                # the s_load pointer
# 12 user SGPRs: s[0:3] output V#, s[4:7] the constant buffer's V#, s[8:9] a pointer; 64x1x1 threads.
FSSM = tv._fssm(12 << 1, (64, 1, 1))


def word(address):
    return 0xA0000000 | address


class WideScalarLoad(unittest.TestCase):
    def run_fixture(self):
        text = support.translate_listing("cs_wide_scalar_load",
                                         (Path(HERE) / "fixtures" / "cs_wide_scalar_load.asm").read_bytes(),
                                         fssm=FSSM)
        mem = tv.ReadLog({a: word(a) for base in (CB, PTR) for a in range(base, base + 0x100, 4)})
        support.run_fragment(text, inputs={"%wg_id": (0, 0, 0), "%local_id": (0, 0, 0)},
                             push_constants=[OUT, 0, 0x100, OUT_V, CB, 0, CB_BYTES, OUT_V, PTR, 0] + [0] * 22,
                             memory=mem)
        return text, mem

    def test_values(self):
        _, mem = self.run_fixture()
        got = [dict.get(mem, OUT + 4 * i) for i in range(16)]
        self.assertEqual(got, [
            word(CB), word(CB + 0x14), word(CB + 0x28), word(CB + 0x3C),     # x16 at 0, in range
            word(CB + 0x40), word(CB + 0x44), 0, 0,                          # x8 at 0x40: dwords 18.. are 0
            word(CB + 0x10), word(CB + 0x1C), word(CB + 0x44), 0,            # x4 at 0x10; x2 at 0x44: 2nd is 0
            word(PTR), word(PTR + 0xC), word(PTR + 0x10), word(PTR + 0x1C),  # s_load_dwordx8, no bound
        ])

    def test_nothing_out_of_range_is_read(self):
        _, mem = self.run_fixture()
        cb_reads = sorted({a for a in mem.reads if CB <= a < CB + 0x100})
        self.assertEqual(cb_reads, list(range(CB, CB + CB_BYTES, 4)))

    def test_the_loads_are_wide(self):
        text, _ = self.run_fixture()
        # x16: 4, x8: 2, x4: 1, s_load x8: 2 uvec4 loads; x2: 1 uvec2 load. Each s_buffer_load also keeps its
        # per-dword path for a partly out-of-range load (16 + 8 + 4 + 2 dword loads, which run only then); the
        # s_load has none.
        self.assertEqual(text.count("OpLoad %v4uint_sl"), 9)
        self.assertEqual(text.count("OpLoad %v2uint_sl"), 1)
        self.assertEqual(sum("OpLoad %uint" in ln and "Aligned 4" in ln for ln in text.splitlines()), 16 + 8 + 4 + 2)

    def test_a_destination_that_is_its_own_vsharp(self):
        # cs_wide_scalar_load_overlap.asm: s_buffer_load_dwordx4 s[20:23], s[20:23] with its V# from memory, under
        # every NUM_RECORDS from nothing to all of both buffers; the dwords out of range are 0 and not read.
        text = support.translate_listing("cs_wide_scalar_load_overlap",
                                         (Path(HERE) / "fixtures" / "cs_wide_scalar_load_overlap.asm").read_bytes(),
                                         fssm=tv._fssm(10 << 1, (64, 1, 1)))
        cb0, cb1, table = 0x4000, 0x5000, 0x6000

        def cb(base, nr, off):
            return word(base + off) if off + 4 <= nr else 0
        for nr0 in (0, 4, 8, 12, 16, 20, 24, 0x100):
            for nr1 in (0, 4, 8, 12, 16, 0x100):
                with self.subTest(nr0=nr0, nr1=nr1):
                    mem = tv.ReadLog({a: word(a) for base in (cb0, cb1) for a in range(base, base + 0x40, 4)})
                    mem.update({table: cb1, table + 4: 0, table + 8: nr1, table + 12: 0})
                    support.run_fragment(text, inputs={"%wg_id": (0, 0, 0), "%local_id": (0, 0, 0)},
                                         push_constants=[OUT, 0, 0x100, OUT_V, cb0, 0, nr0, 0, table, 0] + [0] * 22,
                                         memory=mem)
                    self.assertEqual([dict.get(mem, OUT + 4 * i) for i in range(8)],
                                     [cb(cb0, nr0, 0), cb(cb0, nr0, 4), cb(cb1, nr1, 0), cb(cb1, nr1, 4),
                                      cb(cb1, nr1, 8), cb(cb1, nr1, 12), cb(cb0, nr0, 8), cb(cb0, nr0, 20)])
                    self.assertFalse([a for a in mem.reads if (cb0 <= a < cb0 + 0x40 and a - cb0 + 4 > nr0) or
                                      (cb1 <= a < cb1 + 0x40 and a - cb1 + 4 > nr1)])


if __name__ == "__main__":
    unittest.main()
