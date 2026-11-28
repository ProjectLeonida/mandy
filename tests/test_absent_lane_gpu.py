"""GPU regression test for absent-lane / inactive-lane subgroup safety on partial waves.

Translates small hand-written synthetic RDNA2 shaders (wave32 5-thread partial CS, 1x1 partial-quad
wave32 PS, and 1x1 partial-quad wave64 PS) exercising `v_readfirstlane_b32` with `EXEC == 0` feeding
`s_load_dword`, `ds_append` with `EXEC == 0`, mixed-VGPR `v_writelane_b32`/`v_readlane_b32` on an
absent lane (lane 19), `v_permlanex16_b32` `FI=1` reading absent lane 16, `dpp8 fi:1` reading absent
lane 7, and `s_wqm_b32` on a partial quad. Assembles with `spirv-as`, validates with `spirv-val`,
verifies with `check_absent_lane_ub`, and executes via `vk_compute_test` and `vk_ps_sweep`.

Set `MESA_VK_DEVICE_SELECT` (e.g. `1002:164e!`) and `LEONIDA_VK_VALIDATION=1` as needed.
Skips cleanly when `spirv-as`, `spirv-val`, or the built Vulkan test binaries are absent.
"""
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEMANTIC = os.path.join(ROOT, "tests", "semantic")
if SEMANTIC not in sys.path:
    sys.path.insert(0, SEMANTIC)

import check_absent_lane_ub as checker  # noqa: E402
import support  # noqa: E402

SPIRV_AS = shutil.which("spirv-as")
SPIRV_VAL = shutil.which("spirv-val")
VK_COMPUTE = os.path.join(ROOT, "runtime", "build", "vk_compute_test")
VK_PS_SWEEP = os.path.join(ROOT, "runtime", "build", "vk_ps_sweep")

FULLSCREEN_VS_SPVASM = """OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Vertex %main "main" %vid %pos
OpDecorate %vid BuiltIn VertexIndex
OpDecorate %pos BuiltIn Position
%void = OpTypeVoid
%bool = OpTypeBool
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%float = OpTypeFloat 32
%v4f = OpTypeVector %float 4
%pin_u = OpTypePointer Input %uint
%pout_v4 = OpTypePointer Output %v4f
%vid = OpVariable %pin_u Input
%pos = OpVariable %pout_v4 Output
%c1 = OpConstant %uint 1
%c2 = OpConstant %uint 2
%fm1 = OpConstant %float -1.0
%f3 = OpConstant %float 3.0
%f0 = OpConstant %float 0.0
%f1 = OpConstant %float 1.0
%main = OpFunction %void None %fn
%entry = OpLabel
%i = OpLoad %uint %vid
%is1 = OpIEqual %bool %i %c1
%is2 = OpIEqual %bool %i %c2
%x = OpSelect %float %is1 %f3 %fm1
%y = OpSelect %float %is2 %f3 %fm1
%p = OpCompositeConstruct %v4f %x %y %f0 %f1
OpStore %pos %p
OpReturn
OpFunctionEnd
"""


def assemble_and_check(spv_text, out_spv):
    asm_path = out_spv + ".spvasm"
    with open(asm_path, "w", encoding="utf-8") as f:
        f.write(spv_text)
    subprocess.run([SPIRV_AS, "--target-env", "vulkan1.3", asm_path, "-o", out_spv], check=True)
    subprocess.run([SPIRV_VAL, "--target-env", "vulkan1.3", out_spv], check=True)
    with open(out_spv, "rb") as f:
        raw = f.read()
    viols = checker.check_absent_lane_ub(raw)
    if viols:
        raise AssertionError(f"absent-lane UB violations in {out_spv}: {viols}")


@unittest.skipUnless(SPIRV_AS and SPIRV_VAL, "requires spirv-as and spirv-val")
class PartialWaveGpuTests(unittest.TestCase):
    @unittest.skipUnless(os.path.isfile(VK_COMPUTE), "requires built runtime/build/vk_compute_test")
    def test_partial_wave_compute_5_threads(self):
        hdr = bytearray(0x148)
        struct.pack_into("<I", hdr, 0xD4, 1 << 25)            # wave32
        struct.pack_into("<4I", hdr, 0x138, 0x8088, 5, 1, 1)  # 4 user SGPRs, 5 threads (lanes 5..31 absent)
        spv_text = support.translate_lines("cs_w32_gpu_partial_absent_lane", [
            "v_mov_b32_e32 v4, 0x100",
            "v_mov_b32_e32 v5, 0",
            "s_mov_b32 exec_lo, 0",
            "v_readfirstlane_b32 s10, v4",
            "v_readfirstlane_b32 s11, v5",
            "s_load_dword s12, s[10:11], 0x0",
            "s_waitcnt lgkmcnt(0)",
            "ds_append v8",
            "s_mov_b32 exec_lo, -1",
            "v_mov_b32_e32 v20, 0x9999",
            "s_mov_b32 s14, 0x11223344",
            "v_writelane_b32 v20, s14, 19",
            "s_mov_b32 s14, 0",
            "v_readlane_b32 s15, v20, 19",
            "v_mov_b32_e32 v2, 50",
            "v_permlanex16_b32 v6, v2, 0, 0 op_sel:[1,0]",
            "v_add_nc_u32_dpp v7, v2, 2 dpp8:[7,7,7,7,7,7,7,7] fi:1",
            "v_cmp_eq_u32_e32 vcc_lo, 4, v0",
            "s_wqm_b32 s16, vcc_lo",
            "v_cndmask_b32_e64 v9, 0, 1, s16",
            "v_mov_b32_e32 v10, s12",
            "v_mov_b32_e32 v11, s15",
            "v_cmp_eq_u32_e32 vcc_lo, 4, v0",
            "s_and_saveexec_b32 s18, vcc_lo",
            "v_mov_b32_e32 v12, 0",
            "buffer_store_dword v10, v12, s[0:3], 0 offen offset:0",
            "buffer_store_dword v11, v12, s[0:3], 0 offen offset:4",
            "buffer_store_dword v6,  v12, s[0:3], 0 offen offset:8",
            "buffer_store_dword v7,  v12, s[0:3], 0 offen offset:12",
            "buffer_store_dword v9,  v12, s[0:3], 0 offen offset:16",
            "s_endpgm",
        ], fssm=bytes(hdr))
        with tempfile.TemporaryDirectory() as td:
            cs_spv = os.path.join(td, "cs.spv")
            assemble_and_check(spv_text, cs_spv)
            heap_in = bytearray(0x104)
            struct.pack_into("<I", heap_in, 0x100, 0xCAFEBABE)
            heap_path = os.path.join(td, "heap.bin")
            dump_path = os.path.join(td, "dump.bin")
            with open(heap_path, "wb") as f:
                f.write(heap_in)
            env = dict(os.environ, VKC_HEAP_INIT=heap_path, VKC_DUMP=dump_path, VKC_OUT_WORDS="128")
            p = subprocess.run([VK_COMPUTE, cs_spv, "1", "dump"], env=env, capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stdout + "\n" + p.stderr)
            with open(dump_path, "rb") as f:
                got = struct.unpack_from("<5I", f.read(), 0)
            self.assertEqual(got, (0xCAFEBABE, 0x11223344, 0, 2, 1))

    @unittest.skipUnless(os.path.isfile(VK_PS_SWEEP), "requires built runtime/build/vk_ps_sweep")
    def test_partial_quad_1x1_pixel_wave32_and_wave64(self):
        with tempfile.TemporaryDirectory() as td:
            vs_spv = os.path.join(td, "vs.spv")
            assemble_and_check(FULLSCREEN_VS_SPVASM, vs_spv)
            # VKS_HEAP_FILL=0,16,7 seeds offset 0 with deterministic non-zero float bits in [0.25, 1).
            x = (7 * 2654435761 + 0x9E3779B9) & 0xFFFFFFFF
            x ^= (x << 13) & 0xFFFFFFFF
            x ^= (x >> 17) & 0xFFFFFFFF
            x ^= (x << 5) & 0xFFFFFFFF
            want_word0 = 0x3E800000 + (x % 0x01000000)

            for tag, exec_init, exec_zero, exec_all in (
                ("ps_w32", [], ["s_mov_b32 exec_lo, 0"], ["s_mov_b32 exec_lo, -1"]),
                ("ps_w64", ["s_mov_b64 exec, -1"], ["s_mov_b64 exec, 0"], ["s_mov_b64 exec, -1"]),
            ):
                with self.subTest(stage=tag):
                    ps_text = support.translate_lines(tag + "_gpu_partial_1x1", exec_init + [
                        "v_mov_b32_e32 v4, 0",
                        "v_mov_b32_e32 v5, 0",
                    ] + exec_zero + [
                        "v_readfirstlane_b32 s10, v4",
                        "v_readfirstlane_b32 s11, v5",
                        "s_load_dword s12, s[10:11], 0x0",
                        "s_waitcnt lgkmcnt(0)",
                    ] + exec_all + [
                        "v_mov_b32_e32 v20, 0x9999",
                        "s_mov_b32 s14, 0x3f800000",
                        "v_writelane_b32 v20, s14, 19",
                        "s_mov_b32 s14, 0",
                        "v_readlane_b32 s15, v20, 19",
                        "v_mov_b32_e32 v2, 50",
                        "v_permlanex16_b32 v6, v2, 0, 0 op_sel:[1,0]",
                        "v_add_nc_u32_dpp v7, v2, 2 dpp8:[7,7,7,7,7,7,7,7] fi:1",
                        "v_mov_b32_e32 v8, s12",
                        "v_mov_b32_e32 v9, s15",
                        "exp mrt0 v8, v9, v6, v7 done vm",
                        "s_endpgm",
                    ])
                    ps_spv = os.path.join(td, tag + ".spv")
                    assemble_and_check(ps_text, ps_spv)
                    out_raw = os.path.join(td, tag + ".raw")
                    env = dict(os.environ, VKS_HEAP_FILL="0,16,7")
                    p = subprocess.run([VK_PS_SWEEP, vs_spv, ps_spv, out_raw, "1", "1", "1"],
                                       env=env, capture_output=True, text=True)
                    self.assertEqual(p.returncode, 0, p.stdout + "\n" + p.stderr)
                    with open(out_raw, "rb") as f:
                        got = struct.unpack("<4I", f.read())
                    self.assertEqual(got, (want_word0, 0x3F800000, 0, 2))


if __name__ == "__main__":
    unittest.main()
