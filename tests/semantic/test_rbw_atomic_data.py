"""CPU execution and liveness tests for atomic VGPR operands (SHADER-RBW-ATOMIC-DATA).

Verifies that read_before_write_vgprs, interp_ij_analysis, and reg_reads_writes treat:
  - non-returning buffer/image/global/flat atomics (without glc), ds_store*, and non-_rtn DS
    atomics as reading every VGPR operand and writing none;
  - returning buffer_atomic_* / image_atomic_* (with glc) as both reading and writing all a[0] VGPRs;
  - returning global_atomic_* / flat_atomic_* (with glc) and ds_*_rtn* as writing a[0] and reading a[1:].
Standard library only; no GPU, no LLVM, no game files.
"""
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from support import mrt0_words, rdna2spirv, translate_lines, varyings  # noqa: E402
from shader_ir import parse_line  # noqa: E402

M32 = 0xFFFFFFFF


def _make_isg1_dxbc():
    """Minimal DXBC declaring input attr0.xyzw (sysv=0, reg=0, mask=0xF)."""
    sig = struct.pack("<II", 1, 8) + struct.pack("<6I", 0, 0, 0, 0, 3, 0) + bytes([0xF, 0xF, 0, 0]) + struct.pack("<I", 0)
    body = b"ISG1" + struct.pack("<I", len(sig)) + sig
    head = 36
    return b"DXBC" + bytes(16) + struct.pack("<IIII", 1, head + len(body), 1, head) + body


ATTR0_DXBC = _make_isg1_dxbc()
BARY = (0.125, 0.25, 0.625)  # v0 = i = 0.25 (0x3e800000), v1 = j = 0.625 (0x3f200000)
BARY_I_BITS = 0x3E800000
BARY_J_BITS = 0x3F200000
ATTR0_VERTS = ((0.0, 0.0, 0.0, 0.0), (2.0, 0.0, 0.0, 0.0), (0.0, 4.0, 0.0, 0.0))
ATTR0 = [tuple(struct.unpack("<I", struct.pack("<f", x))[0] for x in v) for v in ATTR0_VERTS]


def _setup_vsharp(addr_reg="v2", offset=0x40):
    return [
        "s_mov_b32 s8, 0",
        "s_mov_b32 s9, 0",
        "s_mov_b32 s10, 0x100",
        "s_mov_b32 s11, 0x31016fac",
        f"v_mov_b32_e32 {addr_reg}, {offset:#x}",
    ]


class AtomicVgprLivenessTests(unittest.TestCase):
    def test_ps_non_glc_buffer_atomic_reads_entry_barycentric_and_preserves_v0(self):
        spv = translate_lines("ps_rbw_atomic_noglc", _setup_vsharp("v2", 0x40) + [
            "buffer_atomic_add v0, v2, s[8:11], 0 offen",
            "s_waitcnt vmcnt(0)",
            "v_mov_b32 v3, v0",
            "exp mrt0 v3, v1, v0, v1 done vm",
            "s_endpgm",
        ])
        mem = {0x40: 0x10}
        words = mrt0_words(spv, inputs={"%bary": BARY}, memory=mem)
        self.assertEqual(mem[0x40], (0x10 + BARY_I_BITS) & M32)
        self.assertEqual(words, [BARY_I_BITS, BARY_J_BITS, BARY_I_BITS, BARY_J_BITS])

    def test_ps_glc_buffer_atomic_reads_entry_barycentric_then_returns_old(self):
        spv = translate_lines("ps_rbw_atomic_glc", _setup_vsharp("v2", 0x40) + [
            "buffer_atomic_add v0, v2, s[8:11], 0 offen glc",
            "s_waitcnt vmcnt(0)",
            "exp mrt0 v0, v1, v0, v1 done vm",
            "s_endpgm",
        ])
        prior = 0x11223344
        mem = {0x40: prior}
        words = mrt0_words(spv, inputs={"%bary": BARY}, memory=mem)
        self.assertEqual(mem[0x40], (prior + BARY_I_BITS) & M32)
        self.assertEqual(words, [prior, BARY_J_BITS, prior, BARY_J_BITS])

    def test_interp_ij_analysis_distinguishes_non_glc_from_glc(self):
        common = _setup_vsharp("v2", 0x40)
        tail = [
            "s_waitcnt vmcnt(0)",
            "v_interp_p1_f32 v4, v0, attr0.x",
            "v_interp_p2_f32 v4, v1, attr0.x",
            "v_interp_p1_f32 v5, v0, attr0.y",
            "v_interp_p2_f32 v5, v1, attr0.y",
            "exp mrt0 v4, v5, v4, v5 done vm",
            "s_endpgm",
        ]
        spv_noglc = translate_lines(
            "ps_interp_noglc", common + ["buffer_atomic_add v0, v2, s[8:11], 0 offen"] + tail, dxbc=ATTR0_DXBC
        )
        self.assertNotIn("PerVertexKHR", spv_noglc)
        mem = {0x40: 0x10}
        words = mrt0_words(spv_noglc, inputs=varyings(spv_noglc, {0: ATTR0}, BARY), memory=mem)
        self.assertEqual(mem[0x40], (0x10 + BARY_I_BITS) & M32)
        # Expected interpolated values at BARY=(0.125, 0.25, 0.625): attr0.x = 2*0.25 = 0.5, attr0.y = 4*0.625 = 2.5
        self.assertEqual(words, [0x3F000000, 0x40200000, 0x3F000000, 0x40200000])

        # With glc, v0 IS overwritten by the atomic return value, so PerVertexKHR computed-I/J kicks in.
        spv_glc = translate_lines(
            "ps_interp_glc", common + ["buffer_atomic_add v0, v2, s[8:11], 0 offen glc"] + tail, dxbc=ATTR0_DXBC
        )
        self.assertIn("PerVertexKHR", spv_glc)

    def test_unit_rules_across_atomic_families(self):
        cases = [
            ("buffer_atomic_add v0, v2, s[8:11], 0 offen", {0, 2}, set()),
            ("buffer_atomic_add v0, v2, s[8:11], 0 offen glc", {0, 2}, {0}),
            ("buffer_atomic_cmpswap v[0:1], v2, s[8:11], 0 offen", {0, 1, 2}, set()),
            ("buffer_atomic_cmpswap v[0:1], v2, s[8:11], 0 offen glc", {0, 1, 2}, {0, 1}),
            ("buffer_atomic_or_x2 v[0:1], v2, s[8:11], 0 offen", {0, 1, 2}, set()),
            ("buffer_atomic_or_x2 v[0:1], v2, s[8:11], 0 offen glc", {0, 1, 2}, {0, 1}),
            ("image_atomic_add v0, v[2:3], s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm", {0, 2, 3}, set()),
            ("image_atomic_add v0, [v2, v5], s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm glc", {0, 2, 5}, {0}),
            ("image_atomic_cmpswap v[0:1], [v2, v5], s[0:7] dmask:0x3 dim:SQ_RSRC_IMG_2D unorm", {0, 1, 2, 5}, set()),
            ("image_atomic_cmpswap v[0:1], v[2:3], s[0:7] dmask:0x3 dim:SQ_RSRC_IMG_2D unorm glc", {0, 1, 2, 3}, {0, 1}),
            ("global_atomic_add v[2:3], v0, off", {0, 2, 3}, set()),
            ("global_atomic_add v4, v[2:3], v0, off glc", {0, 2, 3}, {4}),
            ("flat_atomic_add v[2:3], v0", {0, 2, 3}, set()),
            ("flat_atomic_add v4, v[2:3], v0 glc", {0, 2, 3}, {4}),
            ("ds_add_u32 v2, v0", {0, 2}, set()),
            ("ds_min_i32 v2, v0", {0, 2}, set()),
            ("ds_cmpst_b32 v2, v0, v1", {0, 1, 2}, set()),
            ("ds_store_b32 v2, v0", {0, 2}, set()),
            ("ds_add_rtn_u32 v4, v2, v0", {0, 2}, {4}),
        ]
        for stmt, expected_reads, expected_writes in cases:
            with self.subTest(stmt=stmt):
                ins = parse_line(stmt)
                self.assertEqual(rdna2spirv.reg_reads_writes([ins]), (expected_reads, expected_writes))
                self.assertEqual(rdna2spirv.read_before_write_vgprs([ins]), expected_reads)

    def test_glc_write_does_not_suppress_prior_read_and_kills_entry_value(self):
        insns = [
            parse_line("buffer_atomic_cmpswap v[0:1], v2, s[8:11], 0 offen glc"),
            parse_line("v_mov_b32 v3, v0"),
            parse_line("v_mov_b32 v4, v1"),
        ]
        # Both v0 and v1 are read as input data/cmp before being overwritten by glc, and neither stays carried.
        self.assertEqual(rdna2spirv.read_before_write_vgprs(insns), {0, 1, 2})

        # Contrast: non-glc buffer_atomic does NOT write a[0], so a later read of v0 still sees entry v0;
        # after v_mov v0, 0 first, buffer_atomic_add does not count v0 as live-in, and later v_mov reads v0=0.
        written_first = [
            parse_line("v_mov_b32 v0, 0"),
            parse_line("buffer_atomic_add v0, v2, s[8:11], 0 offen"),
            parse_line("v_mov_b32 v3, v0"),
        ]
        self.assertEqual(rdna2spirv.read_before_write_vgprs(written_first), {2})

        # Without glc, reading v0 only via buffer_atomic_add still requires entry v0.
        still_live = [
            parse_line("buffer_atomic_add v2, v0, s[8:11], 0 offen"),
        ]
        self.assertEqual(rdna2spirv.read_before_write_vgprs(still_live), {0, 2})


if __name__ == "__main__":
    unittest.main()
