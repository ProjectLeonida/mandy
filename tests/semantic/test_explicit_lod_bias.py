"""rdna2spirv.py --explicit-lod-bias: the S# LOD_BIAS applied in the shader, for a driver whose sampler drops
VkSamplerCreateInfo::mipLodBias (MoltenVK 1.4.2; docs/design.md, "S# LOD bias without the sampler's mipLodBias").

The bias is the S#'s word 2 [13:0], signed 5.8 fixed point. With the flag:
* an explicit-LOD sample (image_sample_lz, _l, and the implicit forms outside the PS) adds it to its Lod;
* image_sample_b adds it to its own bias;
* an implicit sample in the PS that reads through Grad (the typed 2D read) scales both derivatives by 2^bias, and
  one that stays implicit (3D) takes it as a Bias operand;
* a literal S# folds the bias to a constant, a zero bias adds nothing, and an S# loaded at run time is decoded from
  its SGPR;
* without the flag nothing changes.
The GPU check is runtime/test/shaders/ps_ssharp (check.py SAMPLER_CASES, whose +-0.3 cases tell a bias counted once
from twice); scripts/run_gpu_tests.py turns the flag on where tools/vk_subgroup_info.py reports explicit_lod_bias.
Standard library only; no GPU, no game files.
"""
import re
import struct
import unittest

import support

D2 = "dim:SQ_RSRC_IMG_2D"
D3 = "dim:SQ_RSRC_IMG_3D"
CS_FSSM = bytearray(0x148)
struct.pack_into("<4I", CS_FSSM, 0x138, 0x8088, 32, 1, 1)
CS_FSSM = bytes(CS_FSSM)


def literal_ssharp(bias_256):
    """s_movs for a T# in s[8:15] and a point-mip S# in s[16:19] with LOD_BIAS = bias_256 / 256."""
    t = [0x2000, 0x1400000, 0, 0, 0, 0, 0, 0]
    s = [0, 0, (bias_256 & 0x3fff) | 1 << 26, 0]
    return [f"s_mov_b32 s{8 + i}, 0x{w:x}" for i, w in enumerate(t)] + \
           [f"s_mov_b32 s{16 + i}, 0x{w:x}" for i, w in enumerate(s)]


LOADED = ["s_load_dwordx8 s[8:15], s[0:1], 0x0", "s_load_dwordx4 s[16:19], s[0:1], 0x20", "s_waitcnt lgkmcnt(0)"]
COORDS = ["v_mov_b32 v2, 0.5", "v_mov_b32 v3, 0.25", "v_mov_b32 v4, 0.75", "v_mov_b32 v5, 1.0"]


def translate(name, prologue, line, flag):
    fssm = CS_FSSM if name.startswith("cs_") else None
    body = prologue + COORDS + [line] + (["s_waitcnt vmcnt(0)", "exp mrt0 v6, v7, v6, v7 done vm"]
                                          if name.startswith("ps_") else []) + ["s_endpgm"]
    prev = support.rdna2spirv.EXPLICIT_LOD_BIAS
    support.rdna2spirv.EXPLICIT_LOD_BIAS = flag
    try:
        return support.translate_lines(name, body, fssm=fssm)
    finally:
        support.rdna2spirv.EXPLICIT_LOD_BIAS = prev


def samples(spv):
    return [ln.strip() for ln in spv.splitlines() if "= OpImageSample" in ln]


def defn(spv, ident):
    """The right-hand side of `ident = ...`, or ''."""
    m = re.search(rf"^\s*{re.escape(ident)} = (.*)$", spv, re.M)
    return m.group(1).strip() if m else ""


def float_const(spv, ident):
    """The value of a float constant: the translator writes it as OpBitcast %float of a uint constant %c<hex>."""
    m = re.match(r"OpBitcast %float %c([0-9a-f]+)$", defn(spv, ident))
    if m:
        return struct.unpack("<f", struct.pack("<I", int(m.group(1), 16)))[0]
    m = re.match(r"OpConstant %float (\S+)$", defn(spv, ident))
    return float(m.group(1)) if m else None


def lod_operand(line):
    m = re.search(r" Lod (%\w+)", line)
    return m.group(1) if m else None


class ExplicitLodBias(unittest.TestCase):
    def test_lz_literal_bias_is_added_to_the_lod(self):
        line = f"image_sample_lz v[6:7], v[2:3], s[8:15], s[16:19] dmask:0x3 {D2}"
        off = samples(translate("ps_lb_lz_off", literal_ssharp(0x100), line, False))
        on_spv = translate("ps_lb_lz_on", literal_ssharp(0x100), line, True)
        on = samples(on_spv)
        self.assertTrue(off and all(lod_operand(s) == "%f0" for s in off), off)
        lods = {lod_operand(s) for s in on}
        self.assertEqual(len(lods), 1, on)
        rhs = defn(on_spv, lods.pop())
        m = re.match(r"OpFAdd %float %f0 (%\w+)$", rhs)
        self.assertTrue(m, rhs)
        self.assertEqual(float_const(on_spv, m.group(1)), 1.0)

    def test_negative_fraction_and_zero_bias(self):
        line = f"image_sample_lz v[6:7], v[2:3], s[8:15], s[16:19] dmask:0x3 {D2}"
        spv = translate("ps_lb_neg", literal_ssharp(-77), line, True)     # -77 / 256
        rhs = defn(spv, lod_operand(samples(spv)[0]))
        m = re.match(r"OpFAdd %float %f0 (%\w+)$", rhs)
        self.assertTrue(m, rhs)
        self.assertAlmostEqual(float_const(spv, m.group(1)), -77 / 256)
        zero = samples(translate("ps_lb_zero", literal_ssharp(0), line, True))
        self.assertTrue(all(lod_operand(s) == "%f0" for s in zero), zero)

    def test_runtime_ssharp_is_decoded_from_word_2(self):
        line = f"image_sample_l v[6:7], v[2:4], s[8:15], s[16:19] dmask:0x3 {D2}"
        spv = translate("ps_lb_rt", LOADED, line, True)
        self.assertRegex(spv, r"OpBitFieldSExtract %uint %\w+ %\w+ %\w+")
        lod = lod_operand(samples(spv)[0])
        self.assertRegex(defn(spv, lod), r"^OpFAdd %float %\w+ %\w+$")
        plain = translate("ps_lb_rt_off", LOADED, line, False)
        self.assertNotIn("OpBitFieldSExtract", plain.split("OpImageSample")[0][-2000:])

    def test_typed_implicit_scales_the_derivatives(self):
        line = f"image_sample v[6:7], v[2:3], s[8:15], s[16:19] dmask:0x3 {D2}"
        off = translate("ps_lb_grad_off", literal_ssharp(0x100), line, False)
        on = translate("ps_lb_grad_on", literal_ssharp(0x100), line, True)
        self.assertNotIn("Exp2", off)
        self.assertIn("Exp2", on)
        grads = [re.search(r" Grad (%\w+) (%\w+)", s) for s in samples(on)]
        self.assertTrue(grads and all(grads), samples(on))
        for g in grads:
            for ident in g.groups():
                self.assertTrue(defn(on, ident).startswith("OpVectorTimesScalar %v2float"), defn(on, ident))

    def test_implicit_3d_takes_a_bias_operand(self):
        line = f"image_sample v[6:7], v[2:4], s[8:15], s[16:19] dmask:0x3 {D3}"
        off = samples(translate("ps_lb_3d_off", literal_ssharp(0x100), line, False))
        on = samples(translate("ps_lb_3d_on", literal_ssharp(0x100), line, True))
        self.assertTrue(off and not any(" Bias " in s for s in off), off)
        self.assertTrue(on and all("ImplicitLod" in s and " Bias " in s for s in on), on)

    def test_sample_b_adds_to_its_own_bias(self):
        line = f"image_sample_b v[6:7], v[2:5], s[8:15], s[16:19] dmask:0x3 {D3}"
        spv = translate("ps_lb_b", literal_ssharp(0x100), line, True)
        on = samples(spv)
        m = re.search(r" Bias (%\w+)", on[0])
        self.assertTrue(m, on)
        self.assertRegex(defn(spv, m.group(1)), r"^OpFAdd %float %\w+ %\w+$")

    def test_compute_implicit_sample_adds_to_lod_zero(self):
        line = f"image_sample v[6:7], v[2:4], s[8:15], s[16:19] dmask:0x3 {D3}"
        off = samples(translate("cs_lb_off", literal_ssharp(0x100), line, False))
        spv = translate("cs_lb_on", literal_ssharp(0x100), line, True)
        on = samples(spv)
        self.assertTrue(all(lod_operand(s) == "%f0" for s in off), off)
        self.assertRegex(defn(spv, lod_operand(on[0])), r"^OpFAdd %float %f0 %\w+$")

    def test_compute_cl_clamps_after_the_bias(self):
        # Vulkan clamps the biased LOD: max(0 + bias, minlod), as _b_cl's max(bias, minlod), not max(0, minlod) + bias
        for op, vaddr in (("image_sample_cl", "v[2:4]"), ("image_sample_c_cl", "v[2:5]")):
            with self.subTest(op=op):
                line = f"{op} v[6:7], {vaddr}, s[8:15], s[16:19] dmask:0x3 {D2}"
                off = translate("cs_lb_cl_off", literal_ssharp(0x100), line, False)
                self.assertRegex(defn(off, lod_operand(samples(off)[0])), r"^OpExtInst %float %glsl FMax %f0 %\w+$")
                spv = translate("cs_lb_cl_on", literal_ssharp(0x100), line, True)
                m = re.match(r"^OpExtInst %float %glsl FMax (%\w+) %\w+$", defn(spv, lod_operand(samples(spv)[0])))
                self.assertTrue(m, samples(spv))
                self.assertRegex(defn(spv, m.group(1)), r"^OpFAdd %float %f0 %\w+$")


if __name__ == "__main__":
    unittest.main()
