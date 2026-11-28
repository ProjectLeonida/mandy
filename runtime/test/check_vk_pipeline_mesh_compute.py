"""vk_pipeline_check: --mesh-compute entries ("stage": "ms" with "mesh_compute") compile as GLCompute pipelines.

No game data: tiny synthetic SPIR-V modules are assembled with tools/spirv_asm.py. Checks that vk_pipeline_check:
  - compiles a --mesh-compute GLCompute module via vkCreateComputePipelines (152-byte push constants, or 128-byte push
    constants + set-1 dynamic UBO under --pc-tail-ubo) on GPUs both with and without VK_EXT_mesh_shader, while still
    reporting its results under the "ms" corpus stage;
  - selects --mesh-compute entries under --stage cs (--compute-only), excludes them under --stage graphics
    (--graphics-only), and keeps native "ms" entries on the VK_EXT_mesh_shader path;
  - compares lds_bytes of a --mesh-compute entry against maxComputeSharedMemorySize (even without VK_EXT_mesh_shader);
  - aggregates "ms" counts identically under --isolate.

Skips when vk_pipeline_check is not built or no usable Vulkan device is present (exits 77 with --ctest for CTest).
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from spirv_asm import assemble  # noqa: E402
from spirv_stubs import build_ps_stub, build_vs_stub  # noqa: E402

WINDOWS = os.name == "nt"
EXE_OVERRIDE: str | None = None

# Minimal GLCompute shader matching the 152-byte --mesh-compute push-constant contract (s0..s31 + heap/code/out base)
# and PhysicalStorageBufferAddresses (using a 2x32-bit pointer bitcast so shaderInt64 is not required, SHADER-423).
MESH_COMPUTE_ASM = """OpCapability Shader
OpCapability PhysicalStorageBufferAddresses
OpExtension "SPV_KHR_physical_storage_buffer"
OpMemoryModel PhysicalStorageBuffer64 GLSL450
OpEntryPoint GLCompute %main "main" %pc
OpExecutionMode %main LocalSize 64 1 1
OpDecorate %arr_ud ArrayStride 4
OpDecorate %PC Block
OpMemberDecorate %PC 0 Offset 0
OpMemberDecorate %PC 1 Offset 128
OpMemberDecorate %PC 2 Offset 132
OpMemberDecorate %PC 3 Offset 136
OpMemberDecorate %PC 4 Offset 140
OpMemberDecorate %PC 5 Offset 144
OpMemberDecorate %PC 6 Offset 148
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%v2uint = OpTypeVector %uint 2
%u32_32 = OpConstant %uint 32
%u32_5 = OpConstant %uint 5
%u32_6 = OpConstant %uint 6
%u32_0 = OpConstant %uint 0
%arr_ud = OpTypeArray %uint %u32_32
%PC = OpTypeStruct %arr_ud %uint %uint %uint %uint %uint %uint
%ptr_pc_struct = OpTypePointer PushConstant %PC
%ptr_pc_u32 = OpTypePointer PushConstant %uint
%ptr_psb_u32 = OpTypePointer PhysicalStorageBuffer %uint
%pc = OpVariable %ptr_pc_struct PushConstant
%main = OpFunction %void None %fn
%entry = OpLabel
%p_lo = OpAccessChain %ptr_pc_u32 %pc %u32_5
%p_hi = OpAccessChain %ptr_pc_u32 %pc %u32_6
%lo = OpLoad %uint %p_lo
%hi = OpLoad %uint %p_hi
%pair = OpCompositeConstruct %v2uint %lo %hi
%dst = OpBitcast %ptr_psb_u32 %pair
OpStore %dst %u32_0 Aligned 4
OpReturn
OpFunctionEnd
"""

# Minimal GLCompute shader matching the --mesh-compute + --pc-tail-ubo contract: 128-byte push constants + set 1 UBO.
MESH_COMPUTE_PCTAIL_ASM = """OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main" %pc %pc_tail
OpExecutionMode %main LocalSize 32 1 1
OpDecorate %arr_ud ArrayStride 4
OpDecorate %PC Block
OpMemberDecorate %PC 0 Offset 0
OpDecorate %arr_tail ArrayStride 16
OpDecorate %Tail Block
OpMemberDecorate %Tail 0 Offset 0
OpDecorate %pc_tail DescriptorSet 1
OpDecorate %pc_tail Binding 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%v4uint = OpTypeVector %uint 4
%u32_32 = OpConstant %uint 32
%u32_2 = OpConstant %uint 2
%arr_ud = OpTypeArray %uint %u32_32
%PC = OpTypeStruct %arr_ud
%ptr_pc = OpTypePointer PushConstant %PC
%arr_tail = OpTypeArray %v4uint %u32_2
%Tail = OpTypeStruct %arr_tail
%ptr_tail = OpTypePointer Uniform %Tail
%pc = OpVariable %ptr_pc PushConstant
%pc_tail = OpVariable %ptr_tail Uniform
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"""

PLAIN_CS_ASM = """OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"""


def find_binary():
    env = EXE_OVERRIDE or os.environ.get("LEONIDA_VK_PIPELINE_CHECK")
    if env:
        return Path(env) if Path(env).is_file() else None
    build = ROOT / "runtime" / "build"
    paths = (
        (build / "vk_pipeline_check.exe", build / "RelWithDebInfo" / "vk_pipeline_check.exe",
         build / "Release" / "vk_pipeline_check.exe")
        if WINDOWS
        else (build / "linux" / "vk_pipeline_check", build / "vk_pipeline_check")
    )
    for p in paths:
        if p.is_file():
            return p
    return None


def mesh_compute_layout(push_bytes=152):
    return {
        "address_dwords": [36, 37],
        "clip_count": 0,
        "clip_offset": 16,
        "cull_count": 0,
        "cull_offset": 16,
        "max_primitives": 1,
        "max_vertices": 3,
        "params": {"0": 16},
        "primitive_stride": 16,
        "primitives": 112,
        "push_constant_bytes": push_bytes,
        "slot_bytes": 128,
        "vertex_stride": 32,
        "vertices": 16,
    }


class MeshComputePipelineCheckTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.exe = find_binary()
        if cls.exe is None:
            raise unittest.SkipTest("vk_pipeline_check isn't built (set LEONIDA_VK_PIPELINE_CHECK to its path)")
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        probe_dir = Path(cls.tmp.name) / "probe"
        probe_dir.mkdir()
        (probe_dir / "c_probe.spv").write_bytes(assemble(PLAIN_CS_ASM, target_env="vulkan1.2"))
        (probe_dir / "index.json").write_text(
            json.dumps({"c_probe": {"stage": "cs", "subgroup_size": None, "features": []}}),
            encoding="utf-8",
            newline="\n",
        )
        probe = cls._exec(probe_dir, [])
        if probe.returncode != 0 or "cs pipelines: 1 ok, 0 failed" not in probe.stdout:
            raise unittest.SkipTest(
                f"no usable Vulkan device (exit {probe.returncode}): {probe.stderr.strip()[-300:]}"
            )

    @classmethod
    def _exec(cls, spirv_dir: Path, args: list[str]) -> subprocess.CompletedProcess:
        env = dict(os.environ)
        env.pop("VK_PIPELINE_CHECK_SIMULATE_CRASH", None)
        cmd = [str(cls.exe), str(spirv_dir), "--jobs", "2", *args]
        return subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=120)

    def make_corpus(self, name: str, entries: dict, files: dict[str, bytes], stubs: dict[str, bytes] | None = None) -> Path:
        d = Path(self.tmp.name) / name
        d.mkdir()
        for fname, data in files.items():
            (d / fname).write_bytes(data)
        if stubs:
            sd = d / "stubs"
            sd.mkdir()
            for fname, data in stubs.items():
                (sd / fname).write_bytes(data)
        (d / "index.json").write_text(json.dumps(entries, indent=2), encoding="utf-8", newline="\n")
        return d

    def test_synthetic_modules_validate_with_spirv_val(self):
        spirv_val = shutil.which("spirv-val")
        if not spirv_val:
            self.skipTest("spirv-val not on PATH")
        for label, text in (("mesh_compute", MESH_COMPUTE_ASM), ("mesh_compute_pctail", MESH_COMPUTE_PCTAIL_ASM)):
            spv = Path(self.tmp.name) / f"{label}.spv"
            spv.write_bytes(assemble(text, target_env="vulkan1.2"))
            r = subprocess.run([spirv_val, "--target-env", "vulkan1.2", str(spv)], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, f"{label}: {r.stdout}{r.stderr}")

    def test_mesh_compute_compiles_as_compute_and_respects_stage_filters(self):
        mc_spv = assemble(MESH_COMPUTE_ASM, target_env="vulkan1.2")
        mc_tail_spv = assemble(MESH_COMPUTE_PCTAIL_ASM, target_env="vulkan1.2")
        cs_spv = assemble(PLAIN_CS_ASM, target_env="vulkan1.2")
        ps_spv = build_ps_stub((0,))
        vs_stub = build_vs_stub((0,))

        entries = {
            "cs_one": {
                "stage": "cs",
                "bindings": [],
                "push_constant_bytes": 144,
                "local_size": [1, 1, 1],
                "subgroup_size": None,
                "features": [],
            },
            "ms_comp": {
                "stage": "ms",
                "bindings": [],
                "features": ["bufferDeviceAddress"],
                "lds_bytes": 64,
                "local_size": [64, 1, 1],
                "max_primitives": 1,
                "max_vertices": 3,
                "mesh_compute": mesh_compute_layout(152),
                "params": [0],
                "push_constant_bytes": 152,
                "subgroup_size": 64,
                "wave64": True,
            },
            "ms_comp_tail": {
                "stage": "ms",
                "bindings": [],
                "features": [],
                "lds_bytes": 4,
                "local_size": [32, 1, 1],
                "max_primitives": 1,
                "max_vertices": 3,
                "mesh_compute": mesh_compute_layout(128),
                "params": [0],
                "pc_tail": {"set": 1, "binding": 0, "bytes": 24},
                "push_constant_bytes": 128,
                "subgroup_size": 32,
                "wave64": False,
            },
            "ps_one": {
                "stage": "ps",
                "bindings": [],
                "features": [],
                "inputs": [0],
                "outputs": [0],
                "push_constant_bytes": 144,
                "subgroup_size": 32,
                "wave64": False,
            },
        }
        corpus = self.make_corpus(
            "mixed",
            entries,
            {
                "cs_one.spv": cs_spv,
                "ms_comp.spv": mc_spv,
                "ms_comp_tail.spv": mc_tail_spv,
                "ps_one.spv": ps_spv,
            },
            stubs={"vs_0.spv": vs_stub},
        )

        # On a 128B maxPushConstantsSize device, ms_comp skips for push-constant bytes while ms_comp_tail compiles;
        # on >= 152B devices both compile. Neither ever skips with "no VK_EXT_mesh_shader" or fails compilation.
        r_all = self._exec(corpus, [str(corpus / "all.cache")])
        self.assertEqual(r_all.returncode, 0, r_all.stdout + r_all.stderr)
        self.assertNotIn("no VK_EXT_mesh_shader", r_all.stdout)
        self.assertNotIn("PIPELINE FAIL", r_all.stdout)
        self.assertIn("ms: 2 modules ok, 0 failed", r_all.stdout)
        self.assertTrue(
            "ms pipelines: 2 ok, 0 failed, 0 skipped" in r_all.stdout
            or "ms pipelines: 1 ok, 0 failed, 1 skipped" in r_all.stdout,
            r_all.stdout,
        )

        # --compute-only checks cs_one + both --mesh-compute entries, leaving ps_one out.
        r_cs = self._exec(corpus, ["--compute-only"])
        self.assertEqual(r_cs.returncode, 0, r_cs.stdout + r_cs.stderr)
        self.assertIn("checking 3 shaders [0, 3) (stage=cs)", r_cs.stderr)
        self.assertIn("cs: 1 modules ok, 0 failed", r_cs.stdout)
        self.assertIn("ms: 2 modules ok, 0 failed", r_cs.stdout)
        self.assertNotIn("ps:", r_cs.stdout)

        # --graphics-only checks only ps_one, leaving cs_one and both --mesh-compute entries out.
        r_gfx = self._exec(corpus, ["--graphics-only"])
        self.assertEqual(r_gfx.returncode, 0, r_gfx.stdout + r_gfx.stderr)
        self.assertIn("checking 1 shaders [0, 1) (stage=graphics)", r_gfx.stderr)
        self.assertIn("ps: 1 modules ok, 0 failed", r_gfx.stdout)
        self.assertNotIn("ms:", r_gfx.stdout)
        self.assertNotIn("cs:", r_gfx.stdout)

        # --isolate aggregates the "ms" stage identically.
        r_iso = self._exec(corpus, ["--isolate", "--isolate-batch", "2"])
        self.assertEqual(r_iso.returncode, 0, r_iso.stdout + r_iso.stderr)
        self.assertIn("ms: 2 modules ok, 0 failed", r_iso.stdout)
        self.assertNotIn("no VK_EXT_mesh_shader", r_iso.stdout)

        # Native "stage": "ms" (without mesh_compute, or with mesh_compute: null) stays on the graphics/mesh path.
        native_dir = self.make_corpus(
            "native_ms_control",
            {
                "ms_native_null": {
                    "stage": "ms",
                    "bindings": [],
                    "features": ["meshShader"],
                    "mesh_compute": None,
                    "params": [],
                    "push_constant_bytes": 144,
                    "subgroup_size": 32,
                }
            },
            {"ms_native_null.spv": cs_spv},
            stubs={"ps_empty.spv": build_ps_stub(())},
        )
        r_native_cs = self._exec(native_dir, ["--compute-only"])
        self.assertEqual(r_native_cs.returncode, 0, r_native_cs.stdout + r_native_cs.stderr)
        self.assertIn("checking 0 shaders [0, 0) (stage=cs)", r_native_cs.stderr)

    def test_mesh_compute_lds_limit_uses_max_compute_shared_memory_size(self):
        cs_spv = assemble(PLAIN_CS_ASM, target_env="vulkan1.2")
        entries = {
            "ms_oversized_lds": {
                "stage": "ms",
                "bindings": [],
                "features": [],
                "lds_bytes": 0x40000000,  # 1 GiB > any GPU's maxComputeSharedMemorySize
                "local_size": [32, 1, 1],
                "mesh_compute": mesh_compute_layout(128),
                "pc_tail": {"set": 1, "binding": 0, "bytes": 24},
                "push_constant_bytes": 128,
                "subgroup_size": 32,
            }
        }
        corpus = self.make_corpus("lds_limit", entries, {"ms_oversized_lds.spv": cs_spv})
        r = self._exec(corpus, [])
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn(
            "MODULE FAIL ms_oversized_lds (workgroup memory 1073741824 bytes exceeds maxComputeSharedMemorySize ",
            r.stdout,
        )
        self.assertIn("ms: 0 modules ok, 1 failed", r.stdout)


def main(argv=None):
    global EXE_OVERRIDE
    raw = list(sys.argv) if argv is None else list(argv)
    ctest_mode = False
    cleaned = [raw[0]]
    i = 1
    while i < len(raw):
        if raw[i] == "--ctest":
            ctest_mode = True
            i += 1
        elif raw[i] == "--exe" and i + 1 < len(raw):
            EXE_OVERRIDE = raw[i + 1]
            i += 2
        else:
            cleaned.append(raw[i])
            i += 1
    result = unittest.main(argv=cleaned, exit=False).result
    if result.failures or result.errors or result.unexpectedSuccesses:
        return 1
    if result.testsRun == 0 and not result.skipped:
        return 1
    if ctest_mode and result.testsRun == len(result.skipped):
        return 77
    return 0


if __name__ == "__main__":
    sys.exit(main())
