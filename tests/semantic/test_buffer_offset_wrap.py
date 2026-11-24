"""The byte offset of a MUBUF access is formed in 64 bits: it does not wrap at 4 GiB.

The hardware adds index * stride (a full product) and soffset to the 48-bit V# base without wrapping at 32 bits
(RDNA2 ISA 8.1.5: address = base + stride * index + offset). Only off = vaddr + inst_offset wraps. So index
0xFFFFFFFF with a 20-byte stride addresses base + 0x13FFFFFFEC. A translator that keeps the offset in 32 bits turns
the same access into base - 20, which OOB_SELECT 3 (no index check) and 2 (no check) let through into the bytes
below the buffer. (The non-wrapping product was observed on hardware by five-star-recomp, commit b9b57e2.)

No bound is added: an access above 4 GiB that the mode allows is made, at its 64-bit address. The expected
addresses below are written out by hand.

    python -I tests/semantic/test_buffer_offset_wrap.py -v
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from spirv_mini_interp import run_workgroup
from test_loop_ballot_barrier import _translate_cs

BASE, OUTPUT = 0x4000, 0x8000
STRIDE, BYTES = 20, 4096
LANES = 32
RAW, DISABLED, STRUCTURED, STRUCTURED_OFFSET = 0x31016FAC, 0x21016FAC, 0x11016FAC, 0x01016FAC
SENTINEL, VALUE = 0xA5A5A5A5, 77


SWIZZLE_ENABLE = 1 << 31   # V# word1


def indexed(index, literal=None, swizzle=False):
    """Store VALUE at `index` of the view in s[0:3], load it back and write what was read to the output view.
    literal = (word3, records): the same descriptor from immediates, which the translator folds."""
    lines = []
    view = "s[0:3]"
    if literal is not None:
        word3, records = literal
        lines += [f"s_mov_b32 s8, {BASE:#x}", f"s_mov_b32 s9, {STRIDE << 16 | (SWIZZLE_ENABLE if swizzle else 0):#x}", f"s_mov_b32 s10, {records:#x}",
                  f"s_mov_b32 s11, {word3:#x}"]
        view = "s[8:11]"
    return lines + [
        f"v_mov_b32_e32 v1, {index}",
        f"v_mov_b32_e32 v2, {VALUE}",
        f"buffer_store_dword v2, v1, {view}, 0 idxen",
        f"buffer_load_dword v3, v1, {view}, 0 idxen",
        "v_lshlrev_b32_e32 v4, 2, v0",
        "buffer_store_dword v3, v4, s[4:7], 0 offen",
        "s_endpgm"]


SOFFSET = [
    # off = 0xFFFFFFF0 (32-bit, as the hardware forms it), then soffset 0x20 is added in 64 bits: 0x1_0000_0010.
    "v_mov_b32_e32 v1, -16",
    f"v_mov_b32_e32 v2, {VALUE}",
    "s_mov_b32 s12, 32",
    "buffer_store_dword v2, v1, s[0:3], s12 offen",
    "buffer_load_dword v3, v1, s[0:3], s12 offen",
    "v_lshlrev_b32_e32 v4, 2, v0",
    "buffer_store_dword v3, v4, s[4:7], 0 offen",
    "s_endpgm"]


class BufferOffset64Tests(unittest.TestCase):
    def run_case(self, lines, word3, stride=STRIDE, records=BYTES, swizzle=False):
        """(the words around the base after the run, the same before it, every other address written, what each
        lane read back)."""
        module, _ = _translate_cs("cs_buffer_offset_wrap", lines, threads=LANES)
        memory = {address: SENTINEL for address in range(BASE - 64, BASE + 128, 4)}
        before = dict(memory)
        result = run_workgroup(
            module, LANES, LANES,
            inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i, "%local_index": i},
            push_constants=[BASE, stride << 16 | (SWIZZLE_ENABLE if swizzle else 0), records, word3, OUTPUT, 0, LANES * 4,
                            RAW] + [0] * 24,
            memory=memory)
        near = {address: result[address] for address in before}
        output = range(OUTPUT, OUTPUT + 4 * LANES, 4)
        far = {address: value for address, value in result.items() if address not in before and address not in output}
        return near, before, far, [result[address] for address in output]

    def test_index_inside_the_buffer_is_stored_and_read(self):
        for name, word3, literal in (("raw", RAW, None), ("raw literal", RAW, (RAW, BYTES)),
                                     ("disabled", DISABLED, None)):
            with self.subTest(mode=name):
                near, before, far, read = self.run_case(indexed(2, literal), word3)
                before[BASE + 2 * STRIDE] = VALUE
                self.assertEqual(near, before)
                self.assertEqual(far, {})
                self.assertEqual(read, [VALUE] * LANES)

    def test_index_times_stride_past_4_gib_is_not_the_bytes_below_the_base(self):
        # 0xFFFFFFFF * 20 = 0x13_FFFF_FFEC; modulo 2^32 it is -20, the five words below the base.
        target = BASE + 0x13FFFFFFEC
        for name, word3, literal in (("raw", RAW, None), ("raw literal", RAW, (RAW, BYTES)),
                                     ("disabled", DISABLED, None), ("disabled literal", DISABLED, (DISABLED, BYTES))):
            with self.subTest(mode=name):
                near, before, far, read = self.run_case(indexed(-1, literal), word3)
                self.assertEqual(near, before)
                self.assertEqual(far, {target: VALUE})
                self.assertEqual(read, [VALUE] * LANES)

    def test_an_offset_above_4_gib_that_the_mode_allows_is_made(self):
        # OOB_SELECT 2 checks only NUM_RECORDS != 0. Index 0x10000000 with a 20-byte stride is byte 0x1_4000_0000
        # (5 GiB): the store lands there and the load reads it, with NUM_RECORDS 1.
        target = BASE + 0x140000000
        for name, literal in (("run-time", None), ("literal", (DISABLED, 1))):
            with self.subTest(v_sharp=name):
                near, before, far, read = self.run_case(indexed(0x10000000, literal), DISABLED, records=1)
                self.assertEqual(near, before)
                self.assertEqual(far, {target: VALUE})
                self.assertEqual(read, [VALUE] * LANES)

    def test_a_structured_index_in_range_whose_product_passes_4_gib_is_made(self):
        # OOB_SELECT 0 and 1 check index < NUM_RECORDS, not the product. Index 0x0CCCCCCD with a 20-byte stride is
        # byte 0x1_0000_0004 and in range with NUM_RECORDS 0xFFFFFFFF, so the access is made there, the same with a
        # literal V# (the translator folds the mode) and a run-time one. Modulo 2^32 it would be base + 4.
        target = BASE + 0x100000004
        for mode, word3 in (("structured", STRUCTURED), ("structured with offset", STRUCTURED_OFFSET)):
            for name, literal in (("run-time", None), ("literal", (word3, 0xFFFFFFFF))):
                with self.subTest(mode=mode, v_sharp=name):
                    near, before, far, read = self.run_case(indexed(0x0CCCCCCD, literal), word3, records=0xFFFFFFFF)
                    self.assertEqual(near, before)
                    self.assertEqual(far, {target: VALUE})
                    self.assertEqual(read, [VALUE] * LANES)

    def test_a_swizzled_offset_past_4_gib_is_made(self):
        # SWIZZLE_ENABLE with idxen: offset = (index / IS * stride + (off & ~3)) * IS + (index % IS) * 4 + (off & 3),
        # IS = 8 here (INDEX_STRIDE 0). Index 0x0CCCCCD5 is group 0x0199999A, element 5:
        # 0x0199999A * 20 * 8 + 5 * 4 = 0x1_0000_0054. Modulo 2^32 it would be base + 0x54.
        target = BASE + 0x100000054
        for name, literal in (("run-time", None), ("literal", (DISABLED, BYTES))):
            with self.subTest(v_sharp=name):
                near, before, far, read = self.run_case(indexed(0x0CCCCCD5, literal, swizzle=True), DISABLED,
                                                        swizzle=True)
                self.assertEqual(near, before)
                self.assertEqual(far, {target: VALUE})
                self.assertEqual(read, [VALUE] * LANES)

    def test_a_swizzled_index_inside_the_buffer_is_stored_and_read(self):
        # The same formula below 4 GiB, as a control: index 13 is group 1, element 5: (1 * 20) * 8 + 5 * 4 = 180.
        for name, literal in (("run-time", None), ("literal", (DISABLED, BYTES))):
            with self.subTest(v_sharp=name):
                near, before, far, read = self.run_case(indexed(13, literal, swizzle=True), DISABLED, swizzle=True)
                self.assertEqual(near, before)
                self.assertEqual(far, {BASE + 180: VALUE})
                self.assertEqual(read, [VALUE] * LANES)

    def test_structured_modes_still_refuse_the_index(self):
        near, before, far, read = self.run_case(indexed(-1), STRUCTURED_OFFSET, records=8)
        self.assertEqual(near, before)
        self.assertEqual(far, {})
        self.assertEqual(read, [0] * LANES)

    def test_soffset_is_added_without_wrapping(self):
        # DISABLED checks only that the buffer is not empty; modulo 2^32 the access would land at base + 0x10.
        near, before, far, read = self.run_case(SOFFSET, DISABLED, stride=0)
        self.assertEqual(near, before)
        self.assertEqual(far, {BASE + 0x100000010: VALUE})
        self.assertEqual(read, [VALUE] * LANES)


if __name__ == "__main__":
    unittest.main()
