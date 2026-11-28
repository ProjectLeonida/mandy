"""Coherent glc/dlc buffer, global, and scalar memory accesses (Item 3B.2).

When a shader performs a coherent (`glc` or `dlc`) buffer/global/scalar load or store:
  * Emits `OpCapability VulkanMemoryModel`, `OpCapability VulkanMemoryModelDeviceScope`,
    `OpExtension "SPV_KHR_vulkan_memory_model"`, and `OpMemoryModel PhysicalStorageBuffer64 Vulkan`.
  * Coherent loads carry `Aligned|MakePointerVisible|NonPrivatePointer 4 %c1` (Device scope)
    and set `meta["reads_glc_buffer"] = True`.
  * Coherent stores carry `Aligned|MakePointerAvailable|NonPrivatePointer 4 %c1` (Device scope)
    and do NOT set `reads_glc_buffer`.
  * Plain non-glc/dlc buffer/global accesses stay on `OpMemoryModel PhysicalStorageBuffer64 GLSL450`
    and omit `reads_glc_buffer`.
  * Mixing coherent buffer ops with coherent image ops (`image_load/store glc`) omits the banned
    `Coherent` decoration under VulkanMemoryModel and uses `MakeTexelVisible/Available|NonPrivateTexel %c1`.
"""
import os
import shutil
import struct
import tempfile
import unittest

import support
import rdna2spirv
import spirv_asm
from spirv_mini_interp import run_fragment

BASE = 0x1000
VSHARP_LINES = [
    f"s_mov_b32 s4, {BASE:#x}",
    "s_mov_b32 s5, 0",
    "s_mov_b32 s6, 0x100",
    "s_mov_b32 s7, 0x30016fac",
]
EPILOGUE = ["exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"]


def translate_with_meta(name, lines):
    asm = support.translate_lines(name, lines)
    meta = rdna2spirv.shader_meta(rdna2spirv.translate.last_gen, asm)
    return asm, meta


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
        # Also verify the built-in Python assembler produces valid Vulkan 1.2 SPIR-V.
        sv_py = os.path.join(tmp, f"{label}_py.spv")
        with open(sv_py, "wb") as f:
            f.write(spirv_asm.assemble(asm, target_env="vulkan1.2"))
        import subprocess
        r2 = subprocess.run(["spirv-val", "--target-env", "vulkan1.2", sv_py], capture_output=True, text=True)
        testcase.assertEqual(r2.returncode, 0, f"{label} spirv_asm+val failed:\n{r2.stderr}\n{r2.stdout}")


class CoherentBufferVmmTests(unittest.TestCase):
    def test_buffer_load_glc_and_dlc_emit_vmm_and_reads_glc_buffer(self):
        for mod in ("glc", "dlc", "glc dlc"):
            with self.subTest(mod=mod):
                lines = VSHARP_LINES + [
                    "v_mov_b32 v1, 0",
                    f"buffer_load_dword v0, v1, s[4:7], 0 offen {mod}",
                    "s_waitcnt vmcnt(0)",
                ] + EPILOGUE
                asm, meta = translate_with_meta("ps_buf_glc", lines)
                self.assertIn("OpCapability VulkanMemoryModel", asm)
                self.assertIn("OpCapability VulkanMemoryModelDeviceScope", asm)
                self.assertIn('OpExtension "SPV_KHR_vulkan_memory_model"', asm)
                self.assertIn("OpMemoryModel PhysicalStorageBuffer64 Vulkan", asm)
                self.assertIn("Aligned|MakePointerVisible|NonPrivatePointer 4 %c1", asm)
                self.assertTrue(meta.get("reads_glc_buffer"))
                self.assertIn("vulkanMemoryModel", meta["features"])
                self.assertIn("vulkanMemoryModelDeviceScope", meta["features"])
                validate_v12(self, asm, "buf_load_" + mod.replace(" ", "_"))
                out = run_fragment(asm, memory={BASE: 0x3F800000})["%out_mrt0"]
                self.assertEqual(out[0], 0x3F800000)

    def test_buffer_store_glc_emits_vmm_without_reads_glc_buffer(self):
        lines = VSHARP_LINES + [
            "v_mov_b32 v0, 0x12345678",
            "v_mov_b32 v1, 0",
            "buffer_store_dword v0, v1, s[4:7], 0 offen glc",
            "s_waitcnt vmcnt(0)",
        ] + EPILOGUE
        asm, meta = translate_with_meta("ps_buf_st_glc", lines)
        self.assertIn("OpCapability VulkanMemoryModel", asm)
        self.assertIn("OpCapability VulkanMemoryModelDeviceScope", asm)
        self.assertIn("OpMemoryModel PhysicalStorageBuffer64 Vulkan", asm)
        self.assertIn("Aligned|MakePointerAvailable|NonPrivatePointer 4 %c1", asm)
        self.assertNotIn("reads_glc_buffer", meta)
        validate_v12(self, asm, "buf_store_glc")
        mem = {}
        run_fragment(asm, memory=mem)
        self.assertEqual(mem[BASE], 0x12345678)

    def test_global_load_and_store_glc(self):
        lines = [
            f"s_mov_b32 s0, {BASE:#x}",
            "s_mov_b32 s1, 0",
            "v_mov_b32 v1, 0",
            "global_load_dword v0, v1, s[0:1] glc",
            "s_waitcnt vmcnt(0)",
            "v_mov_b32 v2, 4",
            "global_store_dword v2, v0, s[0:1] glc",
            "s_waitcnt vmcnt(0)",
        ] + EPILOGUE
        asm, meta = translate_with_meta("ps_glob_glc", lines)
        self.assertIn("OpMemoryModel PhysicalStorageBuffer64 Vulkan", asm)
        self.assertIn("Aligned|MakePointerVisible|NonPrivatePointer 4 %c1", asm)
        self.assertIn("Aligned|MakePointerAvailable|NonPrivatePointer 4 %c1", asm)
        self.assertTrue(meta.get("reads_glc_buffer"))
        validate_v12(self, asm, "global_glc")
        mem = {BASE: 0xCAFEBABE}
        run_fragment(asm, memory=mem)
        self.assertEqual(mem[BASE + 4], 0xCAFEBABE)

    def test_scalar_load_glc_and_buffer_format_glc(self):
        lines = VSHARP_LINES + [
            f"s_mov_b32 s0, {BASE:#x}",
            "s_mov_b32 s1, 0",
            "s_load_dwordx2 s[8:9], s[0:1], 0 glc",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32 v1, 0",
            "buffer_load_format_x v0, v1, s[4:7], 0 offen glc",
            "s_waitcnt vmcnt(0)",
            "buffer_store_format_x v0, v1, s[4:7], 0 offen glc",
            "s_waitcnt vmcnt(0)",
        ] + EPILOGUE
        asm, meta = translate_with_meta("ps_smem_fmt_glc", lines)
        self.assertIn("OpMemoryModel PhysicalStorageBuffer64 Vulkan", asm)
        self.assertIn("Aligned|MakePointerVisible|NonPrivatePointer 4 %c1", asm)
        self.assertIn("Aligned|MakePointerAvailable|NonPrivatePointer 4 %c1", asm)
        self.assertTrue(meta.get("reads_glc_buffer"))
        validate_v12(self, asm, "smem_fmt_glc")

    def test_plain_loads_and_stores_stay_glsl450_without_reads_glc_buffer(self):
        lines = VSHARP_LINES + [
            "v_mov_b32 v1, 0",
            "buffer_load_dword v0, v1, s[4:7], 0 offen",
            "s_waitcnt vmcnt(0)",
            "buffer_store_dword v0, v1, s[4:7], 0 offen",
        ] + EPILOGUE
        asm, meta = translate_with_meta("ps_buf_plain", lines)
        self.assertIn("OpMemoryModel PhysicalStorageBuffer64 GLSL450", asm)
        self.assertNotIn("VulkanMemoryModel", asm)
        self.assertNotIn("reads_glc_buffer", meta)
        validate_v12(self, asm, "buf_plain")

    def test_mixed_coherent_buffer_and_coherent_image_validates_under_vmm(self):
        lines = VSHARP_LINES + [
            "v_mov_b32 v1, 0",
            "v_mov_b32 v4, 1",
            "v_mov_b32 v5, 2",
            "buffer_load_dword v0, v1, s[4:7], 0 offen glc",
            "image_load v[2:3], v[4:5], s[8:15] dmask:0x3 dim:SQ_RSRC_IMG_2D unorm glc dlc",
            "s_waitcnt vmcnt(0)",
            "image_store v[2:3], v[4:5], s[8:15] dmask:0x3 dim:SQ_RSRC_IMG_2D unorm glc",
            "exp mrt0 v0, v2, v3, v0 done vm",
            "s_endpgm",
        ]
        asm, meta = translate_with_meta("ps_mixed_vmm_img", lines)
        self.assertIn("OpMemoryModel PhysicalStorageBuffer64 Vulkan", asm)
        # VulkanMemoryModel forbids Coherent decoration; image ops must use MakeTexelVisible/Available|NonPrivateTexel.
        self.assertNotIn(" Coherent", asm)
        self.assertIn("MakeTexelVisible|NonPrivateTexel %c1", asm)
        self.assertIn("MakeTexelAvailable|NonPrivateTexel %c1", asm)
        self.assertTrue(meta.get("reads_glc_buffer"))
        validate_v12(self, asm, "mixed_vmm_img")

    def test_wide_s_buffer_load_glc_runtime_bound_dominates_slow_path(self):
        lines = [
            "s_buffer_load_dwordx4 s[8:11], s[4:7], s12 glc",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32 v0, s8",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        asm, meta = translate_with_meta("ps_wide_sbuf_glc_rt", lines)
        self.assertIn("OpMemoryModel PhysicalStorageBuffer64 Vulkan", asm)
        self.assertIn("Aligned|MakePointerVisible|NonPrivatePointer 4 %c1", asm)
        self.assertNotIn("%sbuf_slow_x4", asm)
        self.assertNotIn("%_ptr_PSB_sl_blk", asm)
        self.assertTrue(meta.get("reads_glc_buffer"))
        validate_v12(self, asm, "wide_sbuf_glc_rt")


if __name__ == "__main__":
    unittest.main()
