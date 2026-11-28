"""A waterfall loop over a first-lane table offset gives each lane its own entry (TEST-WATERFALL-DESCRIPTOR).

The console compiler serializes a per-lane descriptor: it reads the first active lane's byte offset into a
descriptor table (v_readfirstlane_b32), narrows EXEC to the lanes holding the same one (v_cmpx_eq_u32),
loads that entry's eight T# words (s_load_dwordx8) and drops those lanes (s_andn2_wrexec_b32) until EXEC is
empty. Translated, the offset stays uniform among the lanes of one iteration (the descriptor hash probe's
OpGroupNonUniformBroadcastFirst is an identity among them), so each lane must fetch through the entry its own
offset names. A translator that collapsed the loop to the first lane's entry draws every lane from entry 0
and fails the odd lanes here; the per-lane words come from the table words alone, so a stale mask or a wrong
half fails too.

The listing is hand-built (no llvm-mc, GPU or game files): eight lanes of one wave, v0 the lane id, each
lane's table byte offset (v0 & 1) * 32, the entry a T# whose words name one of two images through the
descriptor hash table, and the fetched texel's first word stored per lane. image_load (a typed fetch) stands
in for the console's image_sample_lz: the CPU interpreter runs OpImageFetch, and both take their T# words
through the same s_load -> hash-probe path (only the sampler probe would be added).
"""
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, HERE)

import support
import spirv_mini_interp as si


def fbits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


# COMPUTE_PGM_RSRC2 0x8090: 8 user SGPRs (s[0:3] the V#, s4:s5 the table base), TGID_X (as test_ballot_votes).
FSSM = bytes(0x138) + struct.pack("<4I", 0x8090, 8, 1, 1)
VSHARP = [0, 0, 0xFFFFFFFF, 0x31016FAC]      # the V# the per-lane store goes through, base 0
TABLE = 0x1000                               # the descriptor table's byte address, pushed in s4:s5

# The console waterfall. The back edge lands on the v_readfirstlane, so every iteration re-reads the first
# active lane's offset (targets are dwords from the branch's end, hence -10 over the uniform 4-byte encodings).
WATERFALL = [
    "v_and_b32_e32 v7, 1, v0",                    # the lane's table byte offset: (v0 & 1) * 32
    "v_lshlrev_b32_e32 v7, 5, v7",
    "s_mov_b32 s20, exec_lo",                     # the full mask, restored after the loop
    "s_mov_b32 s21, exec_lo",                     # the lanes not yet served
    "v_readfirstlane_b32 s16, v7",                # <- loop head: first active lane's offset
    "v_cmpx_eq_u32_e32 s16, v7",                  # EXEC: the lanes with that offset
    "s_load_dwordx8 s[8:15], s[4:5], s16",        # that entry's T# words
    "s_waitcnt lgkmcnt(0)",
    "v_mov_b32_e32 v2, 0",                        # the fetch's coordinates
    "v_mov_b32_e32 v3, 0",
    "image_load v[4:7], v[2:3], s[8:15] dmask:0xf dim:SQ_RSRC_IMG_2D unorm",
    "s_waitcnt vmcnt(0)",
    "s_andn2_wrexec_b32 s21, s21",                # drop the lanes just served
    "s_cbranch_execnz -10",                       # while any lane remains
    "s_mov_b32 exec_lo, s20",
    "v_lshlrev_b32_e32 v12, 2, v0",               # store the texel word at the lane's own slot
    "buffer_store_dword v4, v12, s[0:3], 0 offen",
    "s_endpgm",
]


def listing(lines):
    # Branch targets are dword offsets from the branch's end, so every line carries a fake uniform 4-byte
    # encoding comment; only a branch requires one ("branch without encoding").
    text = "; cs_wf_descriptor  hand-written test program\n"
    for ln in lines:
        text += "\t%s ; encoding: [0x00, 0x00, 0x00, 0x00]\n" % ln
    return text.encode("utf-8")


class WaterfallDescriptorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spv = support.translate_listing("cs_wf_descriptor", listing(WATERFALL), fssm=FSSM)

    def test_each_lane_fetches_the_entry_its_own_offset_names(self):
        # Word 1 (data format) is 0, a float format, so both entries read the float image binding (the T#
        # shape of test_wave32_pixel_subgroup64.py).
        tsharp = {half: [0 if w == 1 else tag + w for w in range(8)]
                  for half, tag in ((0, 0x11110000), (1, 0x22220000))}
        memory = {TABLE + 32 * half + 4 * w: word
                  for half, words in tsharp.items() for w, word in enumerate(words)}
        # The runtime's descriptor hash table: word 0 the slot mask, then 10-word slots (T# words at 1-8, the
        # bindless index at 9). Two slots and mask 1: every T# is found within two probes whatever its hash.
        desc_map = [1, *tsharp[0], 1, 0, *tsharp[1], 2]
        texels = {1: (0.25, 0.5, 0.75, 1.0), 2: (1.0, 0.75, 0.5, 0.25)}
        images = [None] + [(lambda coord, lod, t=texels[k]: t) for k in (1, 2)]
        memory = si.run_workgroup(
            self.spv, 8, 8,
            inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
            push_constants=VSHARP + [TABLE, 0, 0, 0] + [0] * 24,
            memory=memory, resources={"%desc_map": desc_map, "%images": images})
        got = [memory.get(4 * i) for i in range(8)]
        want = [fbits(texels[1][0]) if i % 2 == 0 else fbits(texels[2][0]) for i in range(8)]
        # Even lanes hold entry 0's offset, odd lanes entry 1's, so a loop that kept the first lane's offset
        # for the whole wave stores 0.25's bits in every slot and fails here.
        self.assertEqual(got, want)


if __name__ == "__main__":
    unittest.main()
