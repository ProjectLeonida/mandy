"""The bindless index in the T# (docs/design.md, "Bindless index in the T#").

A T# whose dword 6 the dataflow (tools/descriptor_sources.py) traces to a heap slot takes its index from that word:
TAG [31:28] | storage kind [27:23] | sampled kind [22:16] | index [15:0], which the layer writes into every heap
shaders read (runtime/src/gfx10_descriptors.h, tsharp_index_word). No hash-table lookup is emitted for it. A T# from
anywhere else keeps the lookup, and its dword-6 index wins when the word carries the tag. The fixture
fixtures/disasm/ps_tsharp_index_word.s has both, and a T# loaded in either arm of a branch.
"""
import os
import re
import struct
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, HERE)

import support  # noqa: E402
import spirv_mini_interp as si  # noqa: E402
import rdna2spirv  # noqa: E402
import descriptor_sources  # noqa: E402
import shader_ir  # noqa: E402

HEADER = os.path.join(support.ROOT, "runtime", "src", "gfx10_descriptors.h")
LOOKUP = re.compile(r"OpFunctionCall %\w+ %desc_lookup_fn ")
SAMPLED_2D = 1 << 16          # kind bit 0 (the 2D sampled array) in the word
SAMPLED_3D = 1 << (16 + 2)    # kind bit 2 (the 3D sampled array)
TEXELS = {0: (0.0, 0.0, 0.0, 0.0), 2: (0.25, 0.5, 0.75, 1.0), 5: (1.0, 0.75, 0.5, 0.25)}
EPILOGUE = ["s_waitcnt vmcnt(0)", "exp mrt0 v4, v5, v6, v7 done vm", "s_endpgm"]


def fbits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def word(index, kind=SAMPLED_2D, tag=0xA):
    return (tag << 28) | kind | index


def tsharp(index_word, word3=0x90000000):
    """A 2D T# (TYPE 9 in dword 3, a float format in dword 1) with `index_word` in dword 6."""
    return [0x11110000, 0, 0x22220000, word3, 0x33330000, 0x44440000, index_word, 0x55550000]


def load_program(pointer_high):
    """image_load through a T# loaded at s[0:1] + 0x20; `pointer_high` is how s1 is made."""
    return support.translate_lines("ps_tsharp_index", [
        *pointer_high,
        "s_load_dwordx8 s[8:15], s[0:1], 0x20",
        "s_waitcnt lgkmcnt(0)",
        "v_mov_b32_e32 v2, 0",
        "v_mov_b32_e32 v3, 0",
        "image_load v[4:7], v[2:3], s[8:15] dmask:0xf dim:SQ_RSRC_IMG_2D unorm",
        *EPILOGUE,
    ])


def run(spv, words, desc_map=None, base=0x1000):
    memory = {base + 0x20 + 4 * i: w for i, w in enumerate(words)}
    images = [(lambda coord, lod, t=TEXELS.get(k, (9.0,) * 4): t) for k in range(8)]
    # The module declares the table either way; without `desc_map` it is empty, so a lookup would fail to read it.
    resources = {"%images": images, "%desc_map": [] if desc_map is None else desc_map}
    _, outputs = si.run_workgroup(spv, 32, 32, inputs_for=lambda i: {"%sg_lane": i}, return_outputs=True,
                                  push_constants=[base, 0, 0, 0, 0x2000, 0], memory=memory, resources=resources)
    return outputs[0]["%out_mrt0"]


def expected(index):
    return tuple(fbits(c) for c in TEXELS[index])


class TsharpIndexWord(unittest.TestCase):
    def test_constants_match_the_runtime(self):
        text = open(HEADER, encoding="utf-8").read()
        self.assertIn(f"kTsharpIndexDword = {rdna2spirv.TSHARP_INDEX_DWORD};", text)
        self.assertIn(f"kTsharpIndexTag = 0x{rdna2spirv.TSHARP_INDEX_TAG:X};", text)
        self.assertEqual(descriptor_sources.INDEX_DWORD, rdna2spirv.TSHARP_INDEX_DWORD)
        # The bit layout the shader decodes, as tsharp_index_word writes it.
        self.assertIn("(kTsharpIndexTag << 28) | (((kind >> 8) & 0x1Fu) << 23) | ((kind & 0x7Fu) << 16) | index",
                      text)

    def test_heap_tsharp_reads_dword6_without_a_lookup(self):
        spv = load_program(["s_mov_b32 s1, 0"])
        self.assertIsNone(LOOKUP.search(spv))
        # No %desc_map is bound: a lookup would raise. The word's index picks the image.
        self.assertEqual(run(spv, tsharp(word(2))), expected(2))
        self.assertEqual(run(spv, tsharp(word(5))), expected(5))

    def test_untagged_word_wrong_array_or_no_image_read_the_fallback(self):
        spv = load_program(["s_mov_b32 s1, 0"])
        for name, words in (("no tag", tsharp(word(2, tag=0x1))),
                            ("3D index for a 2D load", tsharp(word(2, kind=SAMPLED_3D))),
                            ("dword 3 names no image", tsharp(word(2), word3=0x30000000)),
                            ("zero slot", [0] * 8)):
            with self.subTest(name):
                self.assertEqual(run(spv, words), expected(0))

    def test_tsharp_of_unknown_source_keeps_the_lookup_and_prefers_the_word(self):
        spv = load_program([])           # s[0:1] is a 64-bit pointer: not a descriptor-table pointer
        self.assertIsNotNone(LOOKUP.search(spv))
        untagged = tsharp(5)
        desc_map = [0, *untagged, 5, 0x80000001]   # mask 0: one slot, holding the T#'s words -> index 5, 2D
        self.assertEqual(run(spv, untagged, desc_map), expected(5))
        # Tagged: the word wins over the table (whose key no longer matches anyway).
        self.assertEqual(run(spv, tsharp(word(2)), desc_map), expected(2))

    def test_branches_and_the_module_meta(self):
        spv = support.translate_fixture("ps_tsharp_index_word")
        gen = rdna2spirv.translate.last_gen
        self.assertEqual(gen.tsharp_index_reads, 2)    # +0x20, and +0x40 / +0x60 joined after the branch
        self.assertEqual(gen.tsharp_lookups, 1)        # the T# from the 64-bit pointer s[4:5]
        self.assertEqual(len(LOOKUP.findall(spv)), 2)  # that T#, and the literal S# (all three samples share it)
        meta = rdna2spirv.shader_meta(gen, spv)
        self.assertEqual(meta["tsharp_index"], "dword6")
        self.assertEqual((meta["tsharp_index_reads"], meta["tsharp_lookups"]), (2, 1))


def verdicts(lines, user=32):
    text = "; t  hand-written\n" + "".join("\t" + ln + "\n" for ln in lines)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "t.s")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        insns = shader_ir.parse_file(path)["insns"]
    out = descriptor_sources.analyse([insns], [[]], user)
    return [out[id(i)] for i in insns if i["op"].startswith("image_")]


class DescriptorSources(unittest.TestCase):
    SAMPLE = "image_load v[0:3], v[4:5], s[8:15] dmask:0xf dim:SQ_RSRC_IMG_2D unorm"

    def test_table_pointer_static_and_dynamic_offsets(self):
        self.assertEqual(verdicts(["s_mov_b32 s1, 0", "s_load_dwordx8 s[8:15], s[0:1], 0x40", self.SAMPLE]), [True])
        self.assertEqual(verdicts(["s_mov_b32 s1, 0", "s_lshl_b32 s20, s3, 5",
                                   "s_load_dwordx8 s[8:15], s[0:1], s20", self.SAMPLE]), [True])

    def test_a_slot_read_in_two_halves(self):
        # Dwords 4..7 from offset 0x10: dword 6 is the load's dword 2, at byte 24 of the slot.
        self.assertEqual(verdicts(["s_mov_b32 s1, 0", "s_load_dwordx4 s[8:11], s[0:1], 0x0",
                                   "s_load_dwordx4 s[12:15], s[0:1], 0x10", self.SAMPLE]), [True])
        # The same registers from offset 0x20 and 0x30 + 8: dword 6 lands at byte 8 of a slot.
        self.assertEqual(verdicts(["s_mov_b32 s1, 0", "s_load_dwordx4 s[8:11], s[0:1], 0x20",
                                   "s_load_dwordx4 s[12:15], s[0:1], 0x38", self.SAMPLE]), [False])

    def test_a_table_pointer_spilled_to_vgpr_lanes(self):
        # A compiler parks SGPRs in VGPR lanes; the pointer read back is still the table pointer.
        spill = ["s_mov_b32 s1, 0", "v_writelane_b32 v63, s0, 10", "v_writelane_b32 v63, s1, 11",
                 "v_writelane_b32 v63, vcc_hi, 12", "v_readlane_b32 vcc_lo, v63, 12"]
        back = ["v_readlane_b32 s20, v63, 10", "v_readlane_b32 s21, v63, 11",
                "s_load_dwordx8 s[8:15], s[20:21], 0x40", self.SAMPLE]
        self.assertEqual(verdicts(spill + back), [True])
        # Anything else that names the VGPR, or a lane written again, forgets what the lanes held.
        self.assertEqual(verdicts(spill + ["v_mov_b32_e32 v63, 0"] + back), [False])
        self.assertEqual(verdicts(spill + ["s_load_dword s40, s[2:3], 0x0", "v_writelane_b32 v63, s40, 10"] + back),
                         [False])

    def test_a_tsharp_loaded_into_vcc(self):
        # Short of SGPRs, a compiler loads a T#'s last two dwords into VCC; dword 6 is then vcc_lo.
        load = ["s_mov_b32 s1, 0", "s_load_dwordx8 [s100,s101,s102,s103,s104,s105,vcc_lo,vcc_hi], s[0:1], 0x40",
                "v_mov_b32_e32 v8, 1.0"]
        use = "image_sample_lz v[0:3], v[4:5], [s100,s101,s102,s103,s104,s105,vcc_lo,vcc_hi], s[16:19] " \
              "dmask:0xf dim:SQ_RSRC_IMG_2D"
        self.assertEqual(verdicts(load + [use]), [True])
        # An instruction that writes VCC (printed: VOPC and carry instructions name it) forgets what it held.
        self.assertEqual(verdicts(load + ["v_cmp_lt_f32_e32 vcc_lo, v0, v1", use]), [False])
        # A pointer parked in VCC and copied back is still the table pointer.
        self.assertEqual(verdicts(["s_mov_b32 s1, 0", "v_writelane_b32 v63, s0, 10", "v_writelane_b32 v63, s1, 11",
                                   "v_readlane_b32 vcc_lo, v63, 10", "v_readlane_b32 vcc_hi, v63, 11",
                                   "s_mov_b64 s[20:21], vcc", "s_load_dwordx8 s[8:15], s[20:21], 0x40", self.SAMPLE]),
                         [True])

    def test_other_sources_keep_the_lookup(self):
        for name, lines in (
                ("64-bit pointer", ["s_load_dwordx8 s[8:15], s[0:1], 0x0"]),
                ("system SGPR", ["s_mov_b32 s41, 0", "s_load_dwordx8 s[8:15], s[40:41], 0x0"]),
                ("loaded pointer", ["s_mov_b32 s1, 0", "s_load_dword s20, s[0:1], 0x0", "s_mov_b32 s21, 0",
                                    "s_load_dwordx8 s[8:15], s[20:21], 0x0"]),
                ("constant buffer", ["s_buffer_load_dwordx8 s[8:15], s[0:3], 0x0"]),
                ("edited dword 6", ["s_mov_b32 s1, 0", "s_load_dwordx8 s[8:15], s[0:1], 0x0",
                                    "s_or_b32 s14, s14, 1"]),
                ("literal", [f"s_mov_b32 s{r}, 0" for r in range(8, 16)])):
            with self.subTest(name):
                self.assertEqual(verdicts(lines + [self.SAMPLE], user=40 if name == "system SGPR" else 32), [False])


if __name__ == "__main__":
    unittest.main()
