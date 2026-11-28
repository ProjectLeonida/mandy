"""scripts/gpu_test_setup.py with stand-in tools and folders (no GPU, no llvm-mc, no game data).

    python -I tests/test_gpu_test_setup.py

The programs it would start are replaced by their recorded output; the build folder and the listings are made in
a temporary checkout.
"""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import gpu_test_setup as setup  # noqa: E402

VULKANINFO = """\
Devices:
========
GPU0:
\tapiVersion         = 1.4.351
\tdeviceName         = Vendor One 9000
\tdriverInfo         = 617.42
GPU1:
\tdeviceName         = Vendor Two Graphics
\tdriverInfo         = 25.10.36 (proprietary)

GPU0:
\tdeviceName        = Vendor One 9000
\tminSubgroupSize                                = 32
\tmaxSubgroupSize                                = 32
GPU1:
\tdeviceName        = Vendor Two Graphics
\tminSubgroupSize                                = 32
\tmaxSubgroupSize                                = 64
"""
AMDGPU = "LLVM version 23.1.6\n  Registered Targets:\n    amdgcn - AMD GCN GPUs\n    x86 - 32-bit X86\n"
PLAIN = "LLVM version 21.1.0\n  Registered Targets:\n    x86 - 32-bit X86\n"


class SetupCase(unittest.TestCase):
    def setUp(self):
        td = tempfile.TemporaryDirectory(prefix="leonida gpu_test_setup ")
        self.addCleanup(td.cleanup)
        self.root = td.name
        self.outputs = {"vulkaninfo": VULKANINFO}
        self.on_path = set()
        for patcher in (mock.patch.object(setup, "output_of", self.output_of),
                        mock.patch.object(setup.shutil, "which", lambda name: name if name in self.on_path else None),
                        mock.patch.dict(os.environ, {"LLVM_MC": ""})):
            patcher.start()
            self.addCleanup(patcher.stop)

    def output_of(self, cmd, timeout=60):
        return self.outputs.get(os.path.basename(cmd[0]))

    def programs(self, folder, names=setup.PROGRAMS + setup.MORE_PROGRAMS):
        os.makedirs(os.path.join(self.root, folder), exist_ok=True)
        for name in names:
            with open(os.path.join(self.root, folder, name + (".exe" if os.name == "nt" else "")), "w"):
                pass

    def listings(self, count):
        folder = os.path.join(self.root, setup.FIXTURES)
        os.makedirs(folder, exist_ok=True)
        for i in range(count):
            with open(os.path.join(folder, f"cs_{i}.s"), "w"):
                pass
        with open(os.path.join(folder, "README.txt"), "w"):
            pass

    def survey(self, **kwargs):
        return setup.survey(root=self.root, **kwargs)

    def levels(self, **kwargs):
        return {level["level"]: level for level in self.survey(**kwargs)["levels"]}


class SurveyTest(SetupCase):
    def test_nothing_but_python_runs_the_semantic_suite(self):
        self.outputs.clear()
        state = self.survey()
        self.assertEqual((state["llvm_mc"], state["spirv_as"], state["spirv_val"], state["gpus"], state["build"],
                          state["listings"]), (None, None, None, [], None, 0))
        levels = {level["level"]: level for level in state["levels"]}
        self.assertEqual((levels["semantic"]["ready"], levels["semantic"]["missing"]), (True, []))
        self.assertEqual(levels["semantic"]["command"], "python -I scripts/run_semantic_tests.py tests/semantic")
        self.assertFalse(levels["subset"]["ready"] or levels["full"]["ready"])
        self.assertEqual(len(levels["subset"]["missing"]), 3)      # programs, a device, the listings
        self.assertEqual(len(levels["full"]["missing"]), 3)        # programs, a device, llvm-mc
        self.assertTrue(levels["full"]["missing"][2].startswith("llvm-mc: "))

    def test_the_subset_needs_the_programs_a_device_and_the_listings_only(self):
        self.programs("runtime/build")
        self.listings(3)
        state = self.survey()
        self.assertEqual((state["build"], state["listings"]), (os.path.normpath("runtime/build"), 3))
        levels = {level["level"]: level for level in state["levels"]}
        self.assertEqual((levels["subset"]["ready"], levels["subset"]["missing"]), (True, []))
        self.assertEqual(levels["subset"]["command"], "python -I scripts/run_gpu_tests.py --build runtime/build "
                                                      "--disasm-dir tests/semantic/fixtures/disasm")
        # No llvm-mc and no SPIRV-Tools: the full run waits for llvm-mc alone.
        self.assertEqual((levels["full"]["ready"], len(levels["full"]["missing"])), (False, 1))

    def test_the_full_run_needs_an_llvm_mc_with_the_amdgpu_target(self):
        self.programs("runtime/build")
        self.listings(1)
        self.on_path.add("llvm-mc")
        self.outputs["llvm-mc"] = PLAIN
        state = self.survey()
        self.assertEqual(state["llvm_mc"], {"command": "llvm-mc", "version": "21.1.0", "amdgpu": False})
        full = state["levels"][2]
        self.assertEqual((full["level"], full["ready"]), ("full", False))
        self.assertIn("an llvm-mc with the AMDGPU target: llvm-mc has none", full["missing"][0])
        self.outputs["llvm-mc"] = AMDGPU
        state = self.survey()
        self.assertEqual(state["llvm_mc"], {"command": "llvm-mc", "version": "23.1.6", "amdgpu": True})
        self.assertEqual((state["levels"][2]["ready"], state["levels"][2]["missing"]), (True, []))
        self.assertEqual(state["levels"][2]["command"], "python -I scripts/run_gpu_tests.py --build runtime/build")

    def test_the_full_run_needs_the_programs_its_texture_fixtures_start(self):
        # The runner checks for three programs; the full run starts four more and fails their fixtures without.
        self.programs("runtime/build", setup.PROGRAMS + ("vk_texture_test",))
        self.listings(1)
        self.on_path.add("llvm-mc")
        self.outputs["llvm-mc"] = AMDGPU
        state = self.survey()
        self.assertEqual(state["more_missing"], ["vk_bindless_test", "vk_sampler_test", "d3d12_texture_uav_test"])
        subset, full = state["levels"][1], state["levels"][2]
        self.assertEqual((subset["ready"], full["ready"], len(full["missing"])), (True, False, 1))
        self.assertIn("vk_bindless_test, vk_sampler_test, d3d12_texture_uav_test in runtime/build", full["missing"][0])
        self.programs("runtime/build")
        self.assertEqual((self.survey()["more_missing"], self.survey()["levels"][2]["ready"]), ([], True))

    def test_the_printed_commands_work_as_printed(self):
        # A build folder with a space is one argument, and an llvm-mc named only on this command line is passed on.
        self.programs("gpu build")
        self.listings(1)
        self.outputs["wsl"] = AMDGPU
        levels = self.levels(build="gpu build", raw_llvm_mc="wsl -e llvm-mc")
        self.assertEqual(levels["subset"]["command"], 'python -I scripts/run_gpu_tests.py --build "gpu build" '
                                                      "--disasm-dir tests/semantic/fixtures/disasm")
        self.assertEqual(levels["full"]["command"],
                         'python -I scripts/run_gpu_tests.py --build "gpu build" --llvm-mc "wsl -e llvm-mc"')
        self.assertTrue(levels["full"]["ready"])
        # From LLVM_MC the runner reads it itself.
        with mock.patch.dict(os.environ, {"LLVM_MC": "wsl -e llvm-mc"}):
            self.assertEqual(self.levels(build="gpu build")["full"]["command"],
                             'python -I scripts/run_gpu_tests.py --build "gpu build"')

    def test_llvm_mc_can_be_a_whole_command(self):
        self.outputs["wsl"] = AMDGPU
        mc = self.survey(raw_llvm_mc="wsl -d Ubuntu -e /usr/lib/llvm-21/bin/llvm-mc")["llvm_mc"]
        self.assertEqual((mc["command"], mc["amdgpu"]), ("wsl -d Ubuntu -e /usr/lib/llvm-21/bin/llvm-mc", True))
        with mock.patch.dict(os.environ, {"LLVM_MC": "wsl -e llvm-mc"}):
            self.assertEqual(self.survey()["llvm_mc"]["command"], "wsl -e llvm-mc")

    def test_the_programs_are_found_where_the_runner_looks(self):
        self.listings(1)
        self.programs("runtime/build/RelWithDebInfo")
        self.assertEqual(self.survey()["build"], os.path.normpath("runtime/build/RelWithDebInfo"))
        self.assertIn("--build runtime/build/RelWithDebInfo ", self.levels()["subset"]["command"])
        # Two of three is not a build.
        self.programs("runtime/build/linux", setup.PROGRAMS[:2])
        self.assertEqual(self.survey(build="runtime/build/linux")["build"], None)
        missing = self.levels(build="runtime/build/linux")["subset"]["missing"][0]
        self.assertIn("vk_compute_test, vk_store_test, vk_harness in runtime/build/linux", missing)
        other = os.path.join(self.root, "elsewhere", "Release")
        self.programs(os.path.join("elsewhere", "Release"))
        self.assertEqual(self.survey(build=os.path.join(self.root, "elsewhere"))["build"], os.path.normpath(other))

    def test_spirv_tools_count_only_when_they_start(self):
        self.on_path |= {"spirv-as", "spirv-val"}
        self.assertEqual((self.survey()["spirv_as"], self.survey()["spirv_val"]), (None, None))
        self.outputs["spirv-as"] = "SPIRV-Tools v2026.4"
        self.assertEqual((self.survey()["spirv_as"], self.survey()["spirv_val"]), ("spirv-as", None))


class GpuTest(unittest.TestCase):
    def test_devices_are_read_with_driver_and_subgroup_range(self):
        self.assertEqual(setup.gpus(VULKANINFO), [
            {"name": "Vendor One 9000", "driver": "617.42", "subgroup": [32, 32]},
            {"name": "Vendor Two Graphics", "driver": "25.10.36 (proprietary)", "subgroup": [32, 64]}])
        self.assertEqual(setup.gpus("GPU0:\n\tdeviceName = Only\n"), [{"name": "Only", "driver": None, "subgroup": None}])
        self.assertEqual(setup.gpus(None), [])
        self.assertEqual(setup.gpus("ERROR: no devices\n"), [])


class ReportTest(SetupCase):
    def run_tool(self, *args):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), mock.patch.object(setup, "ROOT", self.root):
            code = setup.main(list(args))
        return code, out.getvalue()

    def test_text_report(self):
        self.listings(2)
        code, out = self.run_tool()
        self.assertEqual(code, 0)
        self.assertIn("llvm-mc:    not found\n", out)
        self.assertIn("spirv-as:   not found (tools/spirv_asm.py assembles instead)\n", out)
        self.assertIn("spirv-val:  not found (validation is skipped)\n", out)
        self.assertIn("programs:   not built\nlistings:   2 in tests/semantic/fixtures/disasm\n", out)
        self.assertIn("GPU:        Vendor One 9000 (617.42), subgroup 32..32: wave64 fixtures need "
                      "LEONIDA_W64_EMULATE=1\n", out)
        self.assertIn("GPU:        Vendor Two Graphics (25.10.36 (proprietary)), subgroup 32..64\n", out)
        self.assertIn("semantic: ready\n  python -I scripts/run_semantic_tests.py tests/semantic\nsubset: not yet\n"
                      "  needs the test programs", out)
        self.assertNotIn("a second time", out)
        self.programs("runtime/build")
        self.on_path.add("llvm-mc")
        self.outputs["llvm-mc"] = AMDGPU
        out = self.run_tool()[1]
        self.assertIn("llvm-mc:    llvm-mc (23.1.6, AMDGPU target)\n", out)
        self.assertIn("programs:   runtime/build\n", out)
        self.assertIn("subset: ready\n  python -I scripts/run_gpu_tests.py --build runtime/build --disasm-dir", out)
        self.assertIn("full: ready\n  python -I scripts/run_gpu_tests.py --build runtime/build\n", out)
        self.assertIn("run the GPU command a second time with LEONIDA_W64_EMULATE=1", out)

    def test_no_device_and_json(self):
        self.outputs.clear()
        self.assertIn("GPU:        none reported by vulkaninfo\n", self.run_tool()[1])
        report = json.loads(self.run_tool("--json")[1])
        self.assertEqual(sorted(report), ["build", "gpus", "levels", "listings", "llvm_mc", "llvm_mc_option", "more_missing", "spirv_as",
                                          "spirv_val"])
        self.assertEqual([level["level"] for level in report["levels"]], ["semantic", "subset", "full"])
        self.assertIn("a Vulkan device: vulkaninfo reports none", report["levels"][1]["missing"])


class ThisRepositoryTest(unittest.TestCase):
    def test_the_names_it_relies_on_are_the_runners(self):
        with open(os.path.join(ROOT, "scripts", "run_gpu_tests.py"), encoding="utf-8") as f:
            runner = f.read()
        self.assertIn('REQUIRED_TEST_PROGRAMS = ("vk_compute_test", "vk_store_test", "vk_harness")', runner)
        self.assertIn('"--disasm-dir"', runner)
        self.assertTrue(os.path.isdir(os.path.join(ROOT, setup.FIXTURES)))


if __name__ == "__main__":
    unittest.main()
