"""Item 3B.6 tests: float-only image_read_typed elision + compact table-driven buffer_store_format.

Part A:
  * Elides dead uint/sint branches, format-mask lookups, and %itx1/%itx2 descriptor array aliases
    when an image_load* / image_sample* is provably float-only (dest VGPRs consumed solely by
    float ops, or filtered literal S# on image_sample*).
  * Never drops uint/sint branches for integer-sampled/gathered/loaded ops (image_gather4,
    point-filtered image_sample with int consumers, or image_load consumed by int/untyped ops).

Part B:
  * Verifies %fmt_encode_fn / %fmt_enc_chan is branch-free and compact (no 77-way OpSwitch).
  * Verifies bit-for-bit parity between dynamic table-driven `buffer_store_format_*` and fixed
    `tbuffer_store_format_*` across ALL 77 GFX10 format codes in `Gen.FORMATS` (plus unknown codes,
    BGRA 0xf2e swizzle on codes 50/56, 2-component DST_SEL swizzles, and d16 sign extension).
"""
import os
import re
import shutil
import struct
import subprocess
import tempfile
import unittest

import support
import rdna2spirv
import spirv_asm
from spirv_mini_interp import run_fragment

BASE = 0x2000
BRANCHES = ("OpBranchConditional", "OpSelectionMerge", "OpSwitch", "OpBranch", "OpLoopMerge")
GFX10_FORMAT_NAMES = {code: name.removeprefix("GFX10_FORMAT_") for code, name in rdna2spirv.GFX10_FORMATS.items()}


def validate_v12(testcase, asm, label):
    if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
        return
    with tempfile.TemporaryDirectory() as tmp:
        sa = os.path.join(tmp, f"{label}.spvasm")
        sv = os.path.join(tmp, f"{label}.spv")
        with open(sa, "w", encoding="utf-8") as f:
            f.write(asm)
        res = rdna2spirv.assemble_spirv(sa, sv)
        testcase.assertEqual(res.returncode, 0, f"{label} spirv-as/val failed:\n{res.stderr}\n{res.stdout}")
        sv_py = os.path.join(tmp, f"{label}_py.spv")
        with open(sv_py, "wb") as f:
            f.write(spirv_asm.assemble(asm, target_env="vulkan1.2"))
        r2 = subprocess.run(["spirv-val", "--target-env", "vulkan1.2", sv_py], capture_output=True, text=True)
        testcase.assertEqual(r2.returncode, 0, f"{label} spirv_asm+val failed:\n{r2.stderr}\n{r2.stdout}")


def function_body(spv, name):
    body, inside = [], False
    for line in spv.splitlines():
        text = line.strip()
        if text.startswith(f"{name} = OpFunction"):
            inside = True
            continue
        if inside:
            if text == "OpFunctionEnd":
                break
            body.append(text)
    return body


class ImageReadTypedFloatElisionTests(unittest.TestCase):
    def test_float_only_image_load_elides_uint_sint_branches(self):
        lines = [
            "v_mov_b32 v4, 1",
            "v_mov_b32 v5, 2",
            "image_load v[0:1], v[4:5], s[0:7] dmask:0x3 dim:SQ_RSRC_IMG_2D unorm",
            "s_waitcnt vmcnt(0)",
            "v_mul_f32_e32 v0, 0x40000000, v0",
            "v_add_f32_e32 v1, 0x3f800000, v1",
            "exp mrt0 v0, v1, v0, v1 done vm",
            "s_endpgm",
        ]
        asm = support.translate_lines("ps_load_float_only", lines)
        self.assertNotIn("%itx1_0", asm)
        self.assertNotIn("%itx2_0", asm)
        self.assertNotIn("%fmt_uint_mask", asm)
        self.assertNotIn("%fmt_sint_mask", asm)
        self.assertEqual(asm.count("OpImageFetch "), 1)
        self.assertIn("OpImageFetch %v4float", asm)
        validate_v12(self, asm, "load_float_only")

    def test_float_only_image_load_d16_elides_int_views_keeps_d16_masks(self):
        lines = [
            "v_mov_b32 v4, 1",
            "v_mov_b32 v5, 2",
            "image_load v0, v[4:5], s[0:7] dmask:0x3 dim:SQ_RSRC_IMG_2D unorm d16",
            "s_waitcnt vmcnt(0)",
            "v_pk_mul_f16 v0, v0, v0",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        asm = support.translate_lines("ps_load_d16_float_only", lines)
        self.assertNotIn("%itx1_0", asm)
        self.assertNotIn("%itx2_0", asm)
        self.assertNotIn("%fmt_uint_mask", asm)
        self.assertIn("%fmt_norm_mask", asm)
        self.assertIn("%fmt_f32_mask", asm)
        self.assertEqual(asm.count("OpImageFetch "), 1)
        validate_v12(self, asm, "load_d16_float_only")

    def test_filtered_literal_sampler_sample_elides_uint_sint_branches(self):
        # S# word 2 = 0x00500000 -> XY_MAG_FILTER=1 (bilinear), XY_MIN_FILTER=1 (bilinear).
        for op, addr in (("image_sample", "v[4:5]"), ("image_sample_lz", "v[4:5]"), ("image_sample_l", "v[4:6]")):
            with self.subTest(op=op):
                lines = [
                    "s_mov_b32 s8, 0x92",
                    "s_mov_b32 s9, 0x00c00000",
                    "s_mov_b32 s10, 0x00500000",
                    "s_mov_b32 s11, 0",
                    "v_mov_b32 v4, 0x3f000000",
                    "v_mov_b32 v5, 0x3f000000",
                    "v_mov_b32 v6, 0",
                    f"{op} v[0:3], {addr}, s[0:7], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D",
                    "s_waitcnt vmcnt(0)",
                    "exp mrt0 v0, v1, v2, v3 done vm",
                    "s_endpgm",
                ]
                asm = support.translate_lines("ps_filt_smp", lines)
                self.assertNotIn("%itx1_0", asm)
                self.assertNotIn("%itx2_0", asm)
                self.assertNotIn("%fmt_uint_mask", asm)
                validate_v12(self, asm, "filt_" + op)

    def test_int_consumers_and_gather4_never_elide_uint_sint_branches(self):
        # 1. image_load consumed by integer op must keep all 3 typed fetches.
        int_load = [
            "v_mov_b32 v4, 1",
            "v_mov_b32 v5, 2",
            "image_load v0, v[4:5], s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm",
            "s_waitcnt vmcnt(0)",
            "v_add_nc_u32_e32 v0, 1, v0",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        asm_load = support.translate_lines("ps_int_load", int_load)
        self.assertIn("%itx1_0", asm_load)
        self.assertIn("%itx2_0", asm_load)
        self.assertEqual(asm_load.count("OpImageFetch "), 3)

        # 2. point-filtered literal sampler + integer consumer (like SINT sample) must keep all 3 branches.
        int_sample = [
            "s_mov_b32 s8, 0x92",
            "s_mov_b32 s9, 0x00c00000",
            "s_mov_b32 s10, 0x00000000",
            "s_mov_b32 s11, 0",
            "v_mov_b32 v4, 0x3f000000",
            "v_mov_b32 v5, 0x3f000000",
            "image_sample v0, v[4:5], s[0:7], s[8:11] dmask:0x1 dim:SQ_RSRC_IMG_2D d16",
            "s_waitcnt vmcnt(0)",
            "v_bfe_i32 v0, v0, 0, 16",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        asm_smp = support.translate_lines("ps_int_sample", int_sample)
        self.assertIn("%itx1_0", asm_smp)
        self.assertIn("%itx2_0", asm_smp)

        # 3. image_gather4_lz (even with float consumers) stays typed because gather4 works on int formats.
        gather = [
            "v_mov_b32 v4, 0x3f000000",
            "v_mov_b32 v5, 0x3f000000",
            "image_gather4_lz v[0:3], v[4:5], s[0:7], s[8:11] dmask:0x1 dim:SQ_RSRC_IMG_2D",
            "s_waitcnt vmcnt(0)",
            "v_add_f32_e32 v0, v0, v1",
            "v_add_f32_e32 v2, v2, v3",
            "exp mrt0 v0, v2, v0, v2 done vm",
            "s_endpgm",
        ]
        asm_g = support.translate_lines("ps_gather_typed", gather)
        self.assertIn("%itx1_0", asm_g)
        self.assertIn("%itx2_0", asm_g)
        self.assertEqual(asm_g.count("OpImageGather "), 3)

    def test_atomics_ds_indirect_and_partial_f16_writes_never_elide_uint_sint(self):
        # P0-1: a[0] read by buffer_atomic_add / ds_add_u32 / image_atomic_add must prevent float-only elision.
        for label, stage, tail in (
            ("buffer_atomic", "ps", [
                "image_load v[0:1], v[4:5], s[0:7] dmask:0x3 dim:SQ_RSRC_IMG_2D unorm",
                "s_waitcnt vmcnt(0)",
                "v_mul_f32_e32 v2, 0x40000000, v1",
                "buffer_atomic_add v0, v4, s[8:11], 0 offen",
                "exp mrt0 v2, v2, v2, v2 done vm",
                "s_endpgm",
            ]),
            ("ds_atomic", "cs", [
                "image_load v[0:1], v[4:5], s[0:7] dmask:0x3 dim:SQ_RSRC_IMG_2D unorm",
                "s_waitcnt vmcnt(0)",
                "v_mul_f32_e32 v2, 0x40000000, v1",
                "ds_add_u32 v0, v4",
                "s_endpgm",
            ]),
            ("image_atomic", "ps", [
                "image_load v[0:1], v[4:5], s[0:7] dmask:0x3 dim:SQ_RSRC_IMG_2D unorm",
                "s_waitcnt vmcnt(0)",
                "v_mul_f32_e32 v2, 0x40000000, v1",
                "image_atomic_add v0, v[4:5], s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_2D",
                "exp mrt0 v2, v2, v2, v2 done vm",
                "s_endpgm",
            ]),
            ("v_movrels", "ps", [
                "image_load v0, v[4:5], s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm",
                "s_waitcnt vmcnt(0)",
                "v_mul_f32_e32 v2, 0x40000000, v0",
                "v_movrels_b32_e32 v3, v0",
                "exp mrt0 v2, v3, v2, v3 done vm",
                "s_endpgm",
            ]),
            # P0-2: scalar 16-bit write v_add_f16 preserves v0[31:16], so downstream integer read sees upper half!
            ("partial_f16_write", "ps", [
                "image_load v0, v[4:5], s[0:7] dmask:0x3 dim:SQ_RSRC_IMG_2D unorm d16",
                "s_waitcnt vmcnt(0)",
                "v_add_f16_e32 v0, 0x3c00, v0",
                "v_lshrrev_b32_e32 v1, 16, v0",
                "exp mrt0 v0, v1, v0, v1 done vm",
                "s_endpgm",
            ]),
        ):
            with self.subTest(case=label):
                fssm = None
                if stage == "cs":
                    hdr = bytearray(0x148)
                    struct.pack_into("<4I", hdr, 0x138, 0x8088, 64, 1, 1)
                    fssm = bytes(hdr)
                asm = support.translate_lines(f"{stage}_neg_" + label, [
                    "v_mov_b32 v4, 1",
                    "v_mov_b32 v5, 2",
                ] + tail, fssm=fssm)
                self.assertIn("%itx1_0", asm, f"{label} must NOT elide %itx1_0")
                self.assertIn("%itx2_0", asm, f"{label} must NOT elide %itx2_0")
                self.assertEqual(asm.count("OpImageFetch "), 3, f"{label} must keep all 3 typed fetches")


def _store_dynamic(code, vals, dst_sel=0xFAC, d16=False):
    op = "buffer_store_format_d16_xyzw" if d16 else "buffer_store_format_xyzw"
    regs = "v[2:3]" if d16 else "v[2:5]"
    movs = [f"v_mov_b32 v{2 + i}, {w:#x}" for i, w in enumerate(vals)]
    d3 = 0x30000000 | (code << 12) | dst_sel
    lines = [
        f"s_mov_b32 s0, {BASE:#x}",
        "s_mov_b32 s1, 0",
        "s_mov_b32 s2, 0x100",
        f"s_mov_b32 s3, {d3:#x}",
        "v_mov_b32 v1, 0",
        *movs,
        f"{op} {regs}, v1, s[0:3], 0 offen",
        "exp mrt0 v1, v1, v1, v1 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines("ps_dyn_store", lines)
    mem = {BASE + 4 * i: 0xDEAD0000 + i for i in range(4)}
    run_fragment(spv, memory=mem)
    return tuple(mem[BASE + 4 * i] for i in range(4)), spv


def _store_fixed(code, vals, dst_sel=0xFAC, d16=False):
    op = "tbuffer_store_format_d16_xyzw" if d16 else "tbuffer_store_format_xyzw"
    regs = "v[2:3]" if d16 else "v[2:5]"
    movs = [f"v_mov_b32 v{2 + i}, {w:#x}" for i, w in enumerate(vals)]
    fmt_name = GFX10_FORMAT_NAMES[code]
    d3 = 0x30000000 | (code << 12) | dst_sel
    lines = [
        f"s_mov_b32 s0, {BASE:#x}",
        "s_mov_b32 s1, 0",
        "s_mov_b32 s2, 0x100",
        f"s_mov_b32 s3, {d3:#x}",
        "v_mov_b32 v1, 0",
        *movs,
        f"{op} {regs}, v1, s[0:3], 0 format:[BUF_FMT_{fmt_name}] offen",
        "exp mrt0 v1, v1, v1, v1 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines("ps_fix_store", lines)
    mem = {BASE + 4 * i: 0xDEAD0000 + i for i in range(4)}
    run_fragment(spv, memory=mem)
    return tuple(mem[BASE + 4 * i] for i in range(4))


class TableDrivenBufferStoreFormatTests(unittest.TestCase):
    def test_encoder_is_compact_and_branch_free(self):
        _, spv = _store_dynamic(56, (0x3F000000, 0x3E99999A, 0xBF000000, 0x3F800000))
        self.assertIn("%fmt_enc_desc_tab = OpVariable", spv)
        self.assertIn("%fmt_enc_chan = OpFunction %uint None %fn_fmt_enc_chan", spv)
        self.assertIn("%fmt_encode_fn = OpFunction %v4uint_fmt None %fn_fmt_encode", spv)
        chan_body = function_body(spv, "%fmt_enc_chan")
        enc_body = function_body(spv, "%fmt_encode_fn")
        self.assertTrue(chan_body and enc_body)
        for line in chan_body + enc_body:
            for br in BRANCHES:
                self.assertNotIn(br, line, f"format encoder must be branch-free, found: {line}")
        total = len(chan_body) + len(enc_body)
        # Old 77-way OpSwitch was ~3,668 ops; table-driven encoder is < 350 ops.
        self.assertLess(total, 350, f"format encoder too large: {total} ops")
        validate_v12(self, spv, "table_store_enc")

    def test_bit_for_bit_parity_across_all_77_gfx10_formats(self):
        # Test vectors exercising ties (0.3 UNORM16 -> 19661), negatives, overflow, NaN, inf, ufloat, int clamping.
        f_vec1 = (0x3E99999A, 0xBF000000, 0x42C80000, 0x7FC00000)  # 0.3, -0.5, 100.0, qNaN
        f_vec2 = (0x80000000, 0x3F800000, 0x7F800000, 0xFF800000)  # -0.0, 1.0, +inf, -inf
        i_vec = (0x0000007F, 0xFFFFFF80, 0x00012345, 0x80000001)   # +127, -128, large +, large -
        for code, (_, kind, widths) in sorted(rdna2spirv.Gen.FORMATS.items()):
            # Use XY01 (0x00000000 | 4 | 5<<3 = 0x24 -> XY00) for 2-component formats so MUBUF and MTBUF match.
            sel = 0x024 if len(widths) == 2 else 0xFAC
            vecs = (i_vec,) if kind in ("u32", "i32", "uint", "sint") else (f_vec1, f_vec2)
            for vec in vecs:
                with self.subTest(code=code, kind=kind, vec=tuple(hex(x) for x in vec)):
                    got, _ = _store_dynamic(code, vec, dst_sel=sel)
                    want = _store_fixed(code, vec, dst_sel=sel)
                    self.assertEqual(got, want, f"format {code} ({kind} {widths}) mismatch")

    def test_unknown_formats_bgra_swap_and_d16_signed_parity(self):
        # Unknown codes (0, 78, 127) write nothing (memory keeps sentinel 0xDEAD0000..3).
        sentinel = tuple(0xDEAD0000 + i for i in range(4))
        for code in (0, 78, 127):
            got, _ = _store_dynamic(code, (0x3F800000,) * 4)
            self.assertEqual(got, sentinel, f"unknown format {code} should not touch memory")
        # BGRA swap (dst_sel=0xF2E) on codes 50 (2_10_10_10_UNORM) and 56 (8_8_8_8_UNORM).
        vec = (0x3E800000, 0x3F000000, 0x3F400000, 0x3F800000)  # 0.25, 0.5, 0.75, 1.0
        for code in (50, 56):
            self.assertEqual(_store_dynamic(code, vec, dst_sel=0xF2E)[0], _store_fixed(code, vec, dst_sel=0xF2E))
        # D16 signed SINT (e.g. code 61 = 8_8_8_8_SINT, code 76 = 32_32_32_32_SINT) parity.
        d16_vals = (0xFF80007F, 0x80000001)  # (+127, -128, +1, -32768)
        for code in (60, 61, 69, 70, 75, 76):
            self.assertEqual(_store_dynamic(code, d16_vals, d16=True)[0], _store_fixed(code, d16_vals, d16=True))


if __name__ == "__main__":
    unittest.main()
