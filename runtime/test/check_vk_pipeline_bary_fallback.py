"""vk_pipeline_check: --bary-fallback pixel shaders ("bary_fallback_input") compile as VS + synthesized GS + PS.

No game data: tiny synthetic SPIR-V modules are built in pure Python (tools/spirv_asm.py and tools/spirv_stubs.py,
stdlib only, no external llvm-mc or spirv-as required). Checks that vk_pipeline_check:
  - validates "bary_fallback_input" in index.json before loading Vulkan (rejecting non-object/non-null values, missing
    "location", negative/fractional/overflow values, and location >= 32 with exit code 2);
  - synthesizes the triangle pass-through Geometry stage (runtime/src/spirv_bary_fallback.h) for a "ps" entry carrying
    "bary_fallback_input": {"location": N} paired with stubs/vs_<inputs>.spv, compiles the 3-stage VS + GS + PS
    graphics pipeline, writes a non-empty pipeline cache, and hits/reports cleanly on a seeded --cache-feedback run
    (or skips with "no geometryShader" on a device proven to lack geometryShader, such as MoltenVK);
  - handles unexported PS vec4 inputs (zero-filled by the synthesized GS) and VS ClipDistance forwarding;
  - fails with exit code 1 (PIPELINE FAIL, never silently skipping or exiting 0) when build_bary_fallback_gs rejects
    a broken VS/PS interface (such as a vec4 PS input colliding with bary_fallback_input.location), both directly and
    under --isolate.

Skips GPU tests when vk_pipeline_check is not built or no usable Vulkan device is present (exits 77 with --ctest when
all tests skip).
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
from spirv_stubs import build_ps_stub, build_vs_stub, generate_for_index  # noqa: E402

WINDOWS = os.name == "nt"
EXE_OVERRIDE: str | None = None

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

# Minimal --bary-fallback PS: reads vec4 attr0 at Location 0 and vec3 barycentrics at Location 1, writes Location 0.
BARY_PS_ASM = """OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %in_attr0 %in_bary %out_col
OpExecutionMode %main OriginUpperLeft
OpDecorate %in_attr0 Location 0
OpDecorate %in_bary Location 1
OpDecorate %out_col Location 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%float = OpTypeFloat 32
%v3float = OpTypeVector %float 3
%v4float = OpTypeVector %float 4
%ptr_in_v4 = OpTypePointer Input %v4float
%ptr_in_v3 = OpTypePointer Input %v3float
%ptr_out_v4 = OpTypePointer Output %v4float
%f0 = OpConstant %float 0
%in_attr0 = OpVariable %ptr_in_v4 Input
%in_bary = OpVariable %ptr_in_v3 Input
%out_col = OpVariable %ptr_out_v4 Output
%main = OpFunction %void None %fn
%entry = OpLabel
%a = OpLoad %v4float %in_attr0
%b = OpLoad %v3float %in_bary
%bx = OpCompositeExtract %float %b 0
%by = OpCompositeExtract %float %b 1
%bz = OpCompositeExtract %float %b 2
%b4 = OpCompositeConstruct %v4float %bx %by %bz %f0
%sum = OpFAdd %v4float %a %b4
OpStore %out_col %sum
OpReturn
OpFunctionEnd
"""

# --bary-fallback PS that also declares vec4 at Location 2 (omitted by vs_0.spv so GS zero-fills it) and vec3 at Location 3.
BARY_PS_ZEROFILL_ASM = """OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %in_attr0 %in_unwritten %in_bary %out_col
OpExecutionMode %main OriginUpperLeft
OpDecorate %in_attr0 Location 0
OpDecorate %in_unwritten Location 2
OpDecorate %in_bary Location 3
OpDecorate %out_col Location 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%float = OpTypeFloat 32
%v3float = OpTypeVector %float 3
%v4float = OpTypeVector %float 4
%ptr_in_v4 = OpTypePointer Input %v4float
%ptr_in_v3 = OpTypePointer Input %v3float
%ptr_out_v4 = OpTypePointer Output %v4float
%f0 = OpConstant %float 0
%in_attr0 = OpVariable %ptr_in_v4 Input
%in_unwritten = OpVariable %ptr_in_v4 Input
%in_bary = OpVariable %ptr_in_v3 Input
%out_col = OpVariable %ptr_out_v4 Output
%main = OpFunction %void None %fn
%entry = OpLabel
%a = OpLoad %v4float %in_attr0
%u = OpLoad %v4float %in_unwritten
%b = OpLoad %v3float %in_bary
%bx = OpCompositeExtract %float %b 0
%by = OpCompositeExtract %float %b 1
%bz = OpCompositeExtract %float %b 2
%b4 = OpCompositeConstruct %v4float %bx %by %bz %f0
%s0 = OpFAdd %v4float %a %u
%sum = OpFAdd %v4float %s0 %b4
OpStore %out_col %sum
OpReturn
OpFunctionEnd
"""

# Partner VS exporting Position, BuiltIn ClipDistance[2], and Location 0 (exercises GS ClipDistance forwarding).
VS_CLIP2_ASM = """OpCapability Shader
OpCapability ClipDistance
OpMemoryModel Logical GLSL450
OpEntryPoint Vertex %main "main" %out_pos %out_clip %out_attr0
OpDecorate %out_pos BuiltIn Position
OpDecorate %out_clip BuiltIn ClipDistance
OpDecorate %out_attr0 Location 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%u2 = OpConstant %uint 2
%arr_clip = OpTypeArray %float %u2
%ptr_out_v4 = OpTypePointer Output %v4float
%ptr_out_clip = OpTypePointer Output %arr_clip
%f0 = OpConstant %float 0
%f1 = OpConstant %float 1
%pos_val = OpConstantComposite %v4float %f0 %f0 %f0 %f1
%zero_v4 = OpConstantComposite %v4float %f0 %f0 %f0 %f0
%zero_clip = OpConstantComposite %arr_clip %f0 %f0
%out_pos = OpVariable %ptr_out_v4 Output
%out_clip = OpVariable %ptr_out_clip Output
%out_attr0 = OpVariable %ptr_out_v4 Output
%main = OpFunction %void None %fn
%entry = OpLabel
OpStore %out_pos %pos_val
OpStore %out_clip %zero_clip
OpStore %out_attr0 %zero_v4
OpReturn
OpFunctionEnd
"""

# Valid standalone Fragment SPIR-V whose Location 1 is vec4 instead of vec3 (collides with bary_fallback_input at 1).
BROKEN_BARY_VEC4_PS_ASM = """OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %in_attr0 %in_bad_bary %out_col
OpExecutionMode %main OriginUpperLeft
OpDecorate %in_attr0 Location 0
OpDecorate %in_bad_bary Location 1
OpDecorate %out_col Location 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%ptr_in_v4 = OpTypePointer Input %v4float
%ptr_out_v4 = OpTypePointer Output %v4float
%in_attr0 = OpVariable %ptr_in_v4 Input
%in_bad_bary = OpVariable %ptr_in_v4 Input
%out_col = OpVariable %ptr_out_v4 Output
%main = OpFunction %void None %fn
%entry = OpLabel
%a = OpLoad %v4float %in_attr0
%b = OpLoad %v4float %in_bad_bary
%sum = OpFAdd %v4float %a %b
OpStore %out_col %sum
OpReturn
OpFunctionEnd
"""

# Minimal Geometry shader used only as an independent oracle for whether the selected GPU supports geometryShader.
GS_PROBE_ASM = """OpCapability Geometry
OpMemoryModel Logical GLSL450
OpEntryPoint Geometry %main "main" %in_pos %out_pos
OpExecutionMode %main Triangles
OpExecutionMode %main Invocations 1
OpExecutionMode %main OutputTriangleStrip
OpExecutionMode %main OutputVertices 3
OpDecorate %in_pos BuiltIn Position
OpDecorate %out_pos BuiltIn Position
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%int = OpTypeInt 32 1
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%u3 = OpConstant %uint 3
%arr = OpTypeArray %v4float %u3
%ptr_in_arr = OpTypePointer Input %arr
%ptr_in_v4 = OpTypePointer Input %v4float
%ptr_out_v4 = OpTypePointer Output %v4float
%i0 = OpConstant %int 0
%in_pos = OpVariable %ptr_in_arr Input
%out_pos = OpVariable %ptr_out_v4 Output
%main = OpFunction %void None %fn
%entry = OpLabel
%p = OpAccessChain %ptr_in_v4 %in_pos %i0
%v = OpLoad %v4float %p
OpStore %out_pos %v
OpEmitVertex
OpStore %out_pos %v
OpEmitVertex
OpStore %out_pos %v
OpEmitVertex
OpEndPrimitive
OpReturn
OpFunctionEnd
"""


def find_binary() -> Path | None:
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


def run_checker(exe: Path, spirv_dir: Path, args: list[str] | None = None) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.pop("VK_PIPELINE_CHECK_SIMULATE_CRASH", None)
    env.setdefault("VK_LOADER_LAYERS_DISABLE", "~implicit~")
    cmd = [str(exe), str(spirv_dir), *(args or [])]
    return subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=120)


class BaryFallbackIndexValidationTest(unittest.TestCase):
    """Runs without a Vulkan GPU: index.json is validated before Vulkan is loaded."""

    @classmethod
    def setUpClass(cls):
        cls.exe = find_binary()
        if cls.exe is None:
            raise unittest.SkipTest("vk_pipeline_check isn't built (set LEONIDA_VK_PIPELINE_CHECK to its path)")

    def test_invalid_bary_fallback_input_rejected_at_index_parse(self):
        invalid_cases = (
            ('"bad"', "expected object for bary_fallback_input"),
            ("123", "expected object for bary_fallback_input"),
            ("true", "expected object for bary_fallback_input"),
            ("[]", "expected object for bary_fallback_input"),
            ("{}", "missing bary_fallback_input.location"),
            ('{"other": 1}', "missing bary_fallback_input.location"),
            ('{"location": 32}', "bary_fallback_input.location must be < 32"),
            ('{"location": 99}', "bary_fallback_input.location must be < 32"),
            ('{"location": -1}', "expected unsigned 32-bit bary_fallback_input.location"),
            ('{"location": 1.5}', "expected unsigned 32-bit bary_fallback_input.location"),
            ('{"location": "1"}', "expected unsigned 32-bit bary_fallback_input.location"),
            ('{"location": 4294967296}', "expected unsigned 32-bit bary_fallback_input.location"),
        )
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            for raw_value, expected_msg in invalid_cases:
                with self.subTest(bary_fallback_input=raw_value):
                    (d / "index.json").write_text(
                        f'{{"ps_bad":{{"stage":"ps","bary_fallback_input":{raw_value}}}}}\n',
                        encoding="utf-8",
                        newline="\n",
                    )
                    r = run_checker(self.exe, d)
                    self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
                    self.assertIn("malformed JSON", r.stderr)
                    self.assertIn("index.json", r.stderr)
                    self.assertIn(expected_msg, r.stderr)

    def test_null_and_unknown_keys_in_bary_fallback_input_parse_cleanly(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            # --count 0 exits after parsing index.json + opening/skipping 0 shaders, or fails at Vulkan init if no GPU,
            # but never reports a malformed JSON error on index.json.
            for raw_value in ("null", '{"extra": [1, 2], "location": 31}'):
                with self.subTest(bary_fallback_input=raw_value):
                    (d / "index.json").write_text(
                        f'{{"ps_ok":{{"stage":"ps","bary_fallback_input":{raw_value}}}}}\n',
                        encoding="utf-8",
                        newline="\n",
                    )
                    r = run_checker(self.exe, d, ["--count", "0"])
                    self.assertNotEqual(r.returncode, 2, r.stderr)
                    self.assertNotIn("malformed JSON", r.stderr)
                    self.assertIn("1 shaders in index", r.stderr)


class BaryFallbackPipelineCheckTest(unittest.TestCase):
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
        (probe_dir / "gs_probe.spv").write_bytes(assemble(GS_PROBE_ASM, target_env="vulkan1.2"))
        stubs_dir = probe_dir / "stubs"
        stubs_dir.mkdir()
        (stubs_dir / "vs_empty.spv").write_bytes(build_vs_stub(()))
        (stubs_dir / "ps_empty.spv").write_bytes(build_ps_stub(()))
        (probe_dir / "index.json").write_text(
            json.dumps(
                {
                    "c_probe": {"stage": "cs", "subgroup_size": None, "features": []},
                    "gs_probe": {
                        "stage": "gs",
                        "features": ["geometryShader"],
                        "gs_input_locations": [],
                        "gs_output_locations": [],
                        "subgroup_size": None,
                    },
                }
            ),
            encoding="utf-8",
            newline="\n",
        )
        probe = cls._exec(probe_dir, ["--compute-only"])
        if probe.returncode != 0 or "cs pipelines: 1 ok, 0 failed" not in probe.stdout:
            raise unittest.SkipTest(
                f"no usable Vulkan device (exit {probe.returncode}): {probe.stderr.strip()[-300:]}"
            )
        gs_probe = cls._exec(probe_dir, ["--graphics-only"])
        cls.has_geometry_shader = "gs pipelines: 1 ok, 0 failed, 0 skipped" in gs_probe.stdout

    @classmethod
    def _exec(cls, spirv_dir: Path, args: list[str]) -> subprocess.CompletedProcess:
        return run_checker(cls.exe, spirv_dir, ["--jobs", "2", *args])

    def make_corpus(
        self,
        name: str,
        entries: dict,
        files: dict[str, bytes],
        stubs: dict[str, bytes] | None = None,
        auto_stubs: bool = False,
    ) -> Path:
        d = Path(self.tmp.name) / name
        d.mkdir()
        for fname, data in files.items():
            (d / fname).write_bytes(data)
        (d / "index.json").write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8", newline="\n")
        if auto_stubs:
            generate_for_index(str(d), entries)
        if stubs:
            sd = d / "stubs"
            sd.mkdir(exist_ok=True)
            for fname, data in stubs.items():
                (sd / fname).write_bytes(data)
        return d

    def test_synthetic_modules_validate_with_spirv_val(self):
        spirv_val = shutil.which("spirv-val")
        if not spirv_val:
            self.skipTest("spirv-val not on PATH")
        for label, text in (
            ("bary_ps", BARY_PS_ASM),
            ("bary_ps_zerofill", BARY_PS_ZEROFILL_ASM),
            ("vs_clip2", VS_CLIP2_ASM),
            ("broken_bary_vec4_ps", BROKEN_BARY_VEC4_PS_ASM),
            ("gs_probe", GS_PROBE_ASM),
        ):
            spv = Path(self.tmp.name) / f"{label}.spv"
            spv.write_bytes(assemble(text, target_env="vulkan1.2"))
            r = subprocess.run([spirv_val, "--target-env", "vulkan1.2", str(spv)], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, f"{label}: {r.stdout}{r.stderr}")

    def test_valid_bary_fallback_ps_compiles_with_synthesized_gs_and_writes_cache(self):
        ps_spv = assemble(BARY_PS_ASM, target_env="vulkan1.2")
        entries = {
            "ps_bary": {
                "stage": "ps",
                "bindings": [],
                "features": [],
                "inputs": [0],
                "outputs": [0],
                "bary_fallback_input": {"location": 1},
                "push_constant_bytes": 144,
                "subgroup_size": 32,
                "wave64": False,
            }
        }
        # generate_for_index writes stubs/vs_0.spv exporting ONLY BuiltIn Position + Location 0 (NOT Location 1).
        corpus = self.make_corpus("valid_bary", entries, {"ps_bary.spv": ps_spv}, auto_stubs=True)
        self.assertTrue((corpus / "stubs" / "vs_0.spv").is_file())

        cache_file = corpus / "pipeline_cache.bin"
        r = self._exec(corpus, [str(cache_file), "--cache-feedback"])
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("ps: 1 modules ok, 0 failed", r.stdout)

        if not self.has_geometry_shader:
            self.assertIn("SKIPPED ps_bary (no geometryShader)", r.stdout)
            self.assertIn("ps pipelines: 0 ok, 0 failed, 1 skipped", r.stdout)
            return

        self.assertNotIn("SKIPPED", r.stdout)
        self.assertNotIn("PIPELINE FAIL", r.stdout)
        self.assertIn("ps pipelines: 1 ok, 0 failed, 0 skipped", r.stdout)
        self.assertTrue(cache_file.is_file(), "expected pipeline_cache.bin to be written")
        self.assertGreater(cache_file.stat().st_size, 0)

        # Second run seeded with cache_file and --isolate also compiles 1 ok, 0 failed, 0 skipped.
        r_seeded = self._exec(corpus, [str(cache_file), "--cache-feedback"])
        self.assertEqual(r_seeded.returncode, 0, r_seeded.stdout + r_seeded.stderr)
        self.assertIn("ps pipelines: 1 ok, 0 failed, 0 skipped", r_seeded.stdout)

        r_iso = self._exec(corpus, [str(cache_file), "--isolate", "--isolate-batch", "1"])
        self.assertEqual(r_iso.returncode, 0, r_iso.stdout + r_iso.stderr)
        self.assertIn("ps: 1 modules ok, 0 failed", r_iso.stdout)
        self.assertIn("ps pipelines: 1 ok, 0 failed, 0 skipped", r_iso.stdout)

    def test_bary_fallback_clip_distance_forwarding_and_unexported_ps_input_zerofill(self):
        if not self.has_geometry_shader:
            self.skipTest("device has no geometryShader")
        ps_spv = assemble(BARY_PS_ZEROFILL_ASM, target_env="vulkan1.2")
        vs_clip_spv = assemble(VS_CLIP2_ASM, target_env="vulkan1.2")
        entries = {
            "ps_bary_clip": {
                "stage": "ps",
                "bindings": [],
                "features": [],
                "inputs": [0],
                "outputs": [0],
                "bary_fallback_input": {"location": 3},
                "push_constant_bytes": 144,
                "subgroup_size": 32,
                "wave64": False,
            }
        }
        corpus = self.make_corpus(
            "clip_and_zerofill",
            entries,
            {"ps_bary_clip.spv": ps_spv},
            stubs={"vs_0.spv": vs_clip_spv},
        )
        r = self._exec(corpus, [])
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("ps: 1 modules ok, 0 failed", r.stdout)
        self.assertTrue(
            "ps pipelines: 1 ok, 0 failed, 0 skipped" in r.stdout
            or "SKIPPED ps_bary_clip (no shaderClipDistance)" in r.stdout,
            r.stdout + r.stderr,
        )

    def test_broken_bary_fallback_interface_fails_pipeline_check(self):
        if not self.has_geometry_shader:
            self.skipTest("device has no geometryShader")
        bad_ps_spv = assemble(BROKEN_BARY_VEC4_PS_ASM, target_env="vulkan1.2")
        entries = {
            "ps_bad_bary": {
                "stage": "ps",
                "bindings": [],
                "features": [],
                "inputs": [0],
                "outputs": [0],
                "bary_fallback_input": {"location": 1},
                "push_constant_bytes": 144,
                "subgroup_size": 32,
                "wave64": False,
            }
        }
        corpus = self.make_corpus("broken_bary", entries, {"ps_bad_bary.spv": bad_ps_spv}, auto_stubs=True)

        r = self._exec(corpus, [])
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn("ps: 1 modules ok, 0 failed", r.stdout)
        self.assertIn("PIPELINE FAIL ps_bad_bary", r.stdout)
        self.assertIn("ps pipelines: 0 ok, 1 failed, 0 skipped", r.stdout)
        self.assertIn("bary_fallback: bary_location 1 collides with a non-vec3 PS Input", r.stderr)

        # --isolate must also fail with exit code 1 rather than dropping the GS synthesis failure.
        r_iso = self._exec(corpus, ["--isolate", "--isolate-batch", "1"])
        self.assertEqual(r_iso.returncode, 1, r_iso.stdout + r_iso.stderr)
        self.assertIn("PIPELINE FAIL ps_bad_bary", r_iso.stdout)
        self.assertIn("ps pipelines: 0 ok, 1 failed, 0 skipped", r_iso.stdout)


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
