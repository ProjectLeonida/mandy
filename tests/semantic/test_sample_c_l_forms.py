"""image_sample_c_l and image_sample_c_l_o: explicit-LOD depth comparison in 1D, 2D, 2D array and cube images
(SHADER-SAMPLE-C-L-FORMS).

Standard library only (spirv-val is used when it is on PATH); no GPU, no LLVM, no game files.

Both forms are the explicit-LOD compare path of image_sample_c_lz with a LOD operand: OpImageSampleDrefExplicitLod
with Lod, plus Offset for _o. The operand positions below are written by hand from the two compilers, not read from the
translator: LLVM's AMDGPUSampleVariant (IntrinsicsAMDGPU.td) puts [offset,] zcompare, then the coordinates, then the
lod, and SIISelLowering.cpp lowerImage pushes the extra arguments (offset, bias, zcompare) before the gradients and
the coordinates, packing the coordinates *and the trailing lod* in 16-bit pairs under a16 but keeping each extra
argument in a dword of its own; Mesa ACO's visit_tex gathers args as offset, bias, compare, derivatives, then the
coordinates followed by the lod (and clamp) and packs those in pairs under a16. Nothing here was run on a GPU.
"""
import os
import re
import shutil
import struct
import subprocess
import tempfile
import unittest

import support

DESC = "s[8:15], s[16:19]"
CS_FSSM = bytearray(0x148)
struct.pack_into("<4I", CS_FSSM, 0x138, 0x8088, 32, 1, 1)
CS_FSSM = bytes(CS_FSSM)
PRO = [f"v_mov_b32 v{r}, 0.5" for r in range(0, 8)] + ["v_mov_b32 v9, 0"]

DIMS = {"1D": "dim:SQ_RSRC_IMG_1D", "2D": "dim:SQ_RSRC_IMG_2D", "2DA": "dim:SQ_RSRC_IMG_2D_ARRAY",
        "CUBE": "dim:SQ_RSRC_IMG_CUBE", "3D": "dim:SQ_RSRC_IMG_3D"}
SAMPLED_TYPE = {"1D": "simg1d", "2D": "simg2d", "2DA": "simg2da", "CUBE": "simg2da"}   # a cube reads its 6-layer array view

# Hand-written operand positions: (instruction, last VGPR of the address, offset, z-compare, coordinates, lod), each
# register as (VGPR, half) where half is None for a whole dword and 0 / 1 for the low / high f16 of an a16 dword.
W = None
CASES = {
    # no offset: zcompare, coordinates, lod
    "c_l 1D":          ("image_sample_c_l", "1D", False, 2, None, (0, W), [(1, W)], (2, W)),
    "c_l 2D":          ("image_sample_c_l", "2D", False, 3, None, (0, W), [(1, W), (2, W)], (3, W)),
    "c_l 2DA":         ("image_sample_c_l", "2DA", False, 4, None, (0, W), [(1, W), (2, W), (3, W)], (4, W)),
    "c_l CUBE":        ("image_sample_c_l", "CUBE", False, 4, None, (0, W), [(1, W), (2, W), (3, W)], (4, W)),
    # a16: the compare keeps a dword, the coordinates and the lod pack in pairs
    "c_l 1D a16":      ("image_sample_c_l", "1D", True, 1, None, (0, W), [(1, 0)], (1, 1)),
    "c_l 2D a16":      ("image_sample_c_l", "2D", True, 2, None, (0, W), [(1, 0), (1, 1)], (2, 0)),
    "c_l 2DA a16":     ("image_sample_c_l", "2DA", True, 2, None, (0, W), [(1, 0), (1, 1), (2, 0)], (2, 1)),
    "c_l CUBE a16":    ("image_sample_c_l", "CUBE", True, 2, None, (0, W), [(1, 0), (1, 1), (2, 0)], (2, 1)),
    # _o: the offset dword first
    "c_l_o 1D":        ("image_sample_c_l_o", "1D", False, 3, (0, W), (1, W), [(2, W)], (3, W)),
    "c_l_o 2D":        ("image_sample_c_l_o", "2D", False, 4, (0, W), (1, W), [(2, W), (3, W)], (4, W)),
    "c_l_o 2DA":       ("image_sample_c_l_o", "2DA", False, 5, (0, W), (1, W), [(2, W), (3, W), (4, W)], (5, W)),
    "c_l_o CUBE":      ("image_sample_c_l_o", "CUBE", False, 5, (0, W), (1, W), [(2, W), (3, W), (4, W)], (5, W)),
    "c_l_o 1D a16":    ("image_sample_c_l_o", "1D", True, 2, (0, W), (1, W), [(2, 0)], (2, 1)),
    "c_l_o 2D a16":    ("image_sample_c_l_o", "2D", True, 3, (0, W), (1, W), [(2, 0), (2, 1)], (3, 0)),
    "c_l_o 2DA a16":   ("image_sample_c_l_o", "2DA", True, 3, (0, W), (1, W), [(2, 0), (2, 1), (3, 0)], (3, 1)),
    "c_l_o CUBE a16":  ("image_sample_c_l_o", "CUBE", True, 3, (0, W), (1, W), [(2, 0), (2, 1), (3, 0)], (3, 1)),
}


def line_of(op, dim, a16, last, extra=""):
    regs = "v0" if last == 0 else f"v[0:{last}]"
    return (f"{op} v9, {regs}, {DESC} dmask:0x1 {DIMS[dim]}" + (" a16" if a16 else "") + (" " + extra if extra else ""))


def translate(name, line, pre=()):
    fssm = CS_FSSM if name.startswith("cs_") else None
    tail = ["exp mrt0 v9, v9, v9, v9 done vm"] if name.startswith("ps_") else []
    return support.translate_lines(name, PRO + list(pre) + [line] + tail + ["s_endpgm"], fssm=fssm)


def dref_lines(spv):
    return [ln.strip() for ln in spv.splitlines() if "= OpImageSampleDref" in ln]


class Module:
    """Definitions of a translated module, to follow a value back to the VGPR (and f16 half) it was read from."""

    def __init__(self, spv):
        self.spv = spv
        self.defs = {m[1]: (m[2], m[3]) for m in re.finditer(r"^\s*(%\w+) = (\w+) (.*)$", spv, re.M)}
        self.consts = {m[1]: int(m[2]) for m in re.finditer(r"^\s*(%\w+) = OpConstant %uint (\d+)", spv, re.M)}

    def source(self, value):
        """(VGPR index, half) behind `value`. Follows bitcasts, copies, the first operand of the cube coordinate
        arithmetic (OpFSub s - 1, layer - 8 * slice) and the extraction of a half from UnpackHalf2x16."""
        half = None
        for _ in range(16):
            op, rest = self.defs[value]
            args = rest.split()
            if op in ("OpBitcast", "OpCopyObject"):
                value = args[1]
            elif op == "OpFSub":
                value = args[1]
            elif op == "OpCompositeExtract":
                pair = self.defs[args[1]]
                assert pair[0] == "OpExtInst" and "UnpackHalf2x16" in pair[1], (value, pair)
                half = int(args[2])
                value = pair[1].split()[-1]
            elif op == "OpLoad":
                chain = self.defs[args[1]]
                assert chain[0] == "OpAccessChain" and chain[1].split()[1] == "%vgpr", (value, chain)
                return self.consts[chain[1].split()[2]], half
            else:
                raise AssertionError(f"{value} = {op} {rest}")
        raise AssertionError("no VGPR behind " + value)

    def components(self, value, n):
        """The `n` scalar components of a coordinate value: a scalar itself, or the OpCompositeConstruct."""
        if n == 1:
            return [value]
        op, rest = self.defs[value]
        assert op == "OpCompositeConstruct", (value, op)
        return rest.split()[1:]

    def offset_fields(self, off, n):
        """[(VGPR, half, first bit, width, signed)] of an Offset operand."""
        out = []
        for part in self.components(off, n):
            op, rest = self.defs[part]
            assert op == "OpBitcast", (part, op)
            ext_op, ext = self.defs[rest.split()[1]]
            assert ext_op in ("OpBitFieldSExtract", "OpBitFieldUExtract"), ext_op
            _, raw, first, width = ext.split()
            reg, half = self.source(raw)
            out.append((reg, half, self.consts[first], self.consts[width], ext_op == "OpBitFieldSExtract"))
        return out


DREF = re.compile(r"^(%\w+) = OpImageSampleDrefExplicitLod %float (%\w+) (%\w+) (%\w+) Lod(\|Offset)? (%\w+)(?: (%\w+))?$")
NCOORD = {"1D": 1, "2D": 2, "2DA": 3, "CUBE": 3}
NOFF = {"1D": 1, "2D": 2, "2DA": 2, "CUBE": 2}


def check_form(test, spv, case):
    op, dim, a16, _last, offset, ref, coords, lod = CASES[case]
    (line,) = dref_lines(spv)
    m = DREF.match(line)
    test.assertIsNotNone(m, line)
    _res, sampled, uv, ref_id, has_off, lod_id, off_id = m.groups()
    mod = Module(spv)
    test.assertEqual(mod.source(ref_id), ref, "z-compare")
    test.assertEqual([mod.source(c) for c in mod.components(uv, NCOORD[dim])], coords, "coordinates")
    test.assertEqual(mod.source(lod_id), lod, "lod")
    test.assertEqual(bool(has_off), offset is not None)
    if offset is not None:
        want = [(offset[0], offset[1], 8 * k, 6, True) for k in range(NOFF[dim])]
        test.assertEqual(mod.offset_fields(off_id, NOFF[dim]), want, "texel offset")
    # the sampled image is the dimension's view (a cube reads its 6-layer array view)
    test.assertRegex(spv, rf"{re.escape(sampled)} = OpSampledImage %{SAMPLED_TYPE[dim]} ")


class SampleCLForms(unittest.TestCase):
    def test_every_operand_comes_from_its_hand_written_register(self):
        for case, (op, dim, a16, last, *_rest) in CASES.items():
            with self.subTest(case=case):
                spv = translate("ps_samplecl", line_of(op, dim, a16, last))
                check_form(self, spv, case)

    def test_image_operands_are_lod_then_offset(self):
        spv = translate("ps_samplecl", line_of("image_sample_c_l", "2D", False, 3))
        self.assertRegex(dref_lines(spv)[0], r"Lod %\w+$")
        spv = translate("ps_samplecl", line_of("image_sample_c_l_o", "2D", False, 4))
        self.assertRegex(dref_lines(spv)[0], r"Lod\|Offset %\w+ %\w+$")        # SPIR-V mask order: Lod 0x2, Offset 0x10

    def test_a_1d_offset_is_a_scalar_and_the_others_a_pair(self):
        spv = translate("ps_samplecl", line_of("image_sample_c_l_o", "1D", False, 3))
        self.assertRegex(spv, r"= OpImageSampleDrefExplicitLod %float %\w+ %\w+ %\w+ Lod\|Offset %\w+ (%\w+)\n")
        offset = re.search(r"Lod\|Offset %\w+ (%\w+)", spv)[1]
        self.assertRegex(spv, rf"{offset} = OpBitcast %int ")
        spv = translate("ps_samplecl", line_of("image_sample_c_l_o", "2DA", False, 5))
        offset = re.search(r"Lod\|Offset %\w+ (%\w+)", spv)[1]
        self.assertRegex(spv, rf"{offset} = OpCompositeConstruct %v2int ")

    def test_the_result_is_one_value_and_d16_an_f16_in_the_low_half(self):
        # Written as f32 bits under d16 every compare result would read back 0 in the low half (1.0 = 0x3f800000); the
        # f16 shifts out of the bitcast of the compare when d16 is set and not otherwise (as for image_sample_c_lz).
        for d16 in (False, True):
            with self.subTest(d16=d16):
                spv = translate("ps_scldref", line_of("image_sample_c_l", "2D", False, 3, "d16" if d16 else ""))
                (line,) = dref_lines(spv)
                d = re.escape(DREF.match(line)[1])
                bits = re.search(rf"^\s*(%\w+) = OpBitcast %uint {d}$", spv, re.M)
                self.assertIsNotNone(bits)
                shifted = re.search(rf"^\s*%\w+ = OpShiftRightLogical %uint {re.escape(bits[1])} %c10$", spv, re.M)
                self.assertEqual(bool(shifted), d16)

    def test_explicit_lod_samples_in_every_stage(self):
        for name in ("vs_samplecl", "cs_samplecl"):
            for op, last in (("image_sample_c_l", 3), ("image_sample_c_l_o", 4)):
                with self.subTest(shader=name, op=op):
                    spv = translate(name, line_of(op, "2D", False, last))
                    (line,) = dref_lines(spv)
                    self.assertIsNotNone(DREF.match(line), line)
                    self.assertEqual(Module(spv).source(DREF.match(line)[6]), (3, None) if op == "image_sample_c_l" else (4, None))

    def test_refusals(self):
        cases = (
            (line_of("image_sample_c_l", "3D", False, 4), "3D depth comparison has no Vulkan form"),
            (line_of("image_sample_c_l_o", "3D", False, 5), "3D depth comparison has no Vulkan form"),
            (line_of("image_sample_c_l", "3D", True, 2), "3D depth comparison has no Vulkan form"),
            (line_of("image_sample_c_l", "2D", False, 3).replace("dmask:0x1", "dmask:0x3"), "needs scalar depth comparison"),
            (line_of("image_sample_c_l_o", "2D", False, 4).replace("dmask:0x1", "dmask:0x2"), "needs scalar depth comparison"),
            (line_of("image_sample_c_l", "2D", False, 3, "tfe"), "needs scalar depth comparison"),
            (line_of("image_sample_c_l_o", "2D", False, 4, "lwe"), "needs scalar depth comparison"),
            (line_of("image_sample_c_l", "2D", False, 3, "glc"), "glc 2D"),         # a sampled read has no coherent form
            (line_of("image_sample_c_l", "2D", False, 2), "address width"),         # the LOD is missing
            (line_of("image_sample_c_l", "2D", False, 4), "address width"),
            (line_of("image_sample_c_l_o", "2D", False, 3), "address width"),       # the offset is missing
            (line_of("image_sample_c_l", "2D", True, 3), "address width"),          # a16 packs the coordinates and the LOD
            (line_of("image_sample_c_l_o", "1D", True, 3), "address width"),
        )
        for line, message in cases:
            with self.subTest(line=line):
                with self.assertRaises(support.rdna2spirv.Unsupported) as caught:
                    translate("ps_samplecl", line)
                self.assertIn(message, str(caught.exception))

    @unittest.skipUnless(shutil.which("llvm-mc"), "needs llvm-mc on PATH")
    def test_the_address_widths_are_the_assembler_s(self):
        """llvm-mc checks the size of the address range of a MIMG instruction, so it is a third source for the
        number of dwords of each form (and under a16, for the packing of the coordinates and the LOD)."""
        def assembles(line, cpu):
            res = subprocess.run(["llvm-mc", "-triple=amdgcn--amdpal", f"-mcpu={cpu}", "-show-encoding"],
                                 input=line + "\n", capture_output=True, text=True)
            return res.returncode == 0
        for case, (op, dim, a16, last, *_rest) in CASES.items():
            for cpu in ("gfx1013", "gfx1030"):
                with self.subTest(case=case, cpu=cpu):
                    self.assertTrue(assembles(line_of(op, dim, a16, last), cpu))
                    self.assertTrue(assembles(line_of(op, dim, a16, last, "d16"), cpu))
                    if last > 0:                       # one register fewer is a different, invalid, address
                        self.assertFalse(assembles(line_of(op, dim, a16, last - 1), cpu))

    @unittest.skipUnless(shutil.which("spirv-as") and shutil.which("spirv-val"), "needs spirv-as and spirv-val on PATH")
    def test_modules_validate(self):
        for case, (op, dim, a16, last, *_rest) in CASES.items():
            for d16 in (False, True):
                with self.subTest(case=case, d16=d16):
                    spv = translate("ps_samplecl", line_of(op, dim, a16, last, "d16" if d16 else ""))
                    validate(self, spv, allow_offset=op.endswith("_o"))

    @unittest.skipUnless(shutil.which("spirv-as") and shutil.which("spirv-val"), "needs spirv-as and spirv-val on PATH")
    def test_the_texel_offset_fallback_needs_no_maintenance8(self):
        old = support.rdna2spirv.TEXEL_OFFSET_FALLBACK
        support.rdna2spirv.TEXEL_OFFSET_FALLBACK = True
        try:
            for case, (op, dim, a16, last, *_rest) in CASES.items():
                if not op.endswith("_o"):
                    continue
                with self.subTest(case=case):
                    spv = translate("ps_samplecl", line_of(op, dim, a16, last))
                    self.assertNotIn(" Offset ", dref_lines(spv)[0])
                    validate(self, spv, allow_offset=False)
        finally:
            support.rdna2spirv.TEXEL_OFFSET_FALLBACK = old


def validate(test, spv, allow_offset):
    with tempfile.TemporaryDirectory() as tmp:
        a, b = os.path.join(tmp, "m.spvasm"), os.path.join(tmp, "m.spv")
        with open(a, "w", encoding="utf-8", newline="\n") as f:
            f.write(spv)
        subprocess.run(["spirv-as", "--target-env", "vulkan1.2", a, "-o", b], check=True)
        cmd = ["spirv-val", "--target-env", "vulkan1.2"] + (["--allow-offset-texture-operand"] if allow_offset else []) + [b]
        res = subprocess.run(cmd, capture_output=True, text=True)
        test.assertEqual(res.returncode, 0, res.stdout + res.stderr)


class TexelOffsetFallback(unittest.TestCase):
    """--texel-offset-fallback folds the offset into the coordinate at the sampled level. An explicit-LOD form knows
    the level (the LOD register, rounded by the S#'s MIP_FILTER like image_sample_l_o), not level 0 as _lz does."""

    def setUp(self):
        self.old = support.rdna2spirv.TEXEL_OFFSET_FALLBACK
        support.rdna2spirv.TEXEL_OFFSET_FALLBACK = True

    def tearDown(self):
        support.rdna2spirv.TEXEL_OFFSET_FALLBACK = self.old

    def floors(self, spv):
        return re.findall(r"= OpExtInst %float %glsl Floor (%\w+)", spv)

    def test_the_level_is_the_lod_register(self):
        # S# word 2 is s18: MIP_FILTER [27:26] = 1 (point) rounds the LOD to a level, so the level chain reads the LOD
        for a16, last, lod in ((False, 4, (4, None)), (True, 3, (3, 0))):
            with self.subTest(a16=a16):
                spv = translate("ps_samplecl", line_of("image_sample_c_l_o", "2D", a16, last), ["s_mov_b32 s18, 0x04000000"])
                mod = Module(spv)
                floors = self.floors(spv)
                self.assertEqual(len(floors), 1, floors)
                self.assertEqual(mod.source(floors[0]), lod)
                (line,) = dref_lines(spv)
                self.assertNotIn(" Offset ", line)
                self.assertEqual(mod.source(DREF.match(line)[6]), lod)       # the sample itself still uses the LOD
                self.assertIn("OpImageQuerySizeLod", spv)

    def test_a_filter_without_mips_reads_level_zero(self):
        spv = translate("ps_samplecl", line_of("image_sample_c_l_o", "2D", False, 4), ["s_mov_b32 s18, 0"])
        self.assertEqual(self.floors(spv), [])
        self.assertIn("OpImageQuerySizeLod", spv)

    def test_lz_stays_at_level_zero(self):
        spv = translate("ps_samplecl", "image_sample_c_lz_o v9, v[0:3], s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D",
                        ["s_mov_b32 s18, 0x04000000"])
        self.assertEqual(self.floors(spv), [])


if __name__ == "__main__":
    unittest.main()
