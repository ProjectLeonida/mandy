"""make_game.py: pipelines stage preflight checks (gta6rt.exe, Wine off Windows, VULKAN_SDK on Windows).

    python -I tests/test_make_game_pipeline_preflight.py

Standard library only. Every path is synthetic and every tool is mocked: no game files, GPU, Vulkan SDK or
build output is needed.
"""
import io
import os
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import make_game  # noqa: E402


class PreflightEnv:
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = tmp.name
        self.work = os.path.join(self.root, "work")
        self.out = os.path.join(self.root, "out")
        os.makedirs(self.work)
        os.makedirs(self.out)
        # Satisfy every other prerequisite so only the pipelines requirements under test can make preflight fail.
        for patcher in (mock.patch("sys.version_info", (3, 12, 0)),
                        mock.patch.dict(os.environ, {"LLVM_MC": "llvm-mc"}),
                        mock.patch("shutil.which", return_value="/bin/tool"),
                        mock.patch("shutil.disk_usage", return_value=mock.Mock(free=500 * 2**30)),
                        mock.patch.object(make_game, "ROOT", self.root)):
            patcher.start()
            self.addCleanup(patcher.stop)

    def write_shader_index(self):
        spirv = os.path.join(self.work, "shaders", "spirv")
        os.makedirs(spirv, exist_ok=True)
        with open(os.path.join(spirv, "index.json"), "w", encoding="utf-8", newline="\n") as f:
            f.write('{"shader_0": {}}\n')

    def write_runtime_exe(self, folder=None):
        dest = folder or self.out
        os.makedirs(dest, exist_ok=True)
        path = os.path.join(dest, "gta6rt.exe")
        with open(path, "wb"):
            pass
        return path

    def assert_preflight_fails(self, stages, match, **kw):
        err = io.StringIO()
        with mock.patch("sys.stderr", err):
            with self.assertRaises(SystemExit) as ctx:
                make_game.preflight(stages, self.work, out=self.out, **kw)
        self.assertEqual(ctx.exception.code, 1)
        self.assertIn(match, err.getvalue())
        return err.getvalue()


class PipelinePreflightTest(PreflightEnv, unittest.TestCase):
    def test_pipelines_without_runtime_or_exe_fails_in_preflight(self):
        for platform in ("nt", "posix"):
            with self.subTest(platform=platform), mock.patch("os.name", platform):
                err = self.assert_preflight_fails(["shaders", "pipelines"], "gta6rt.exe")
                self.assertIn("Install these first, then run the build again:", err)

    def test_main_does_not_start_shaders_when_runtime_exe_is_missing(self):
        game = os.path.join(self.root, "game")
        os.makedirs(game)
        argv = ["make_game.py", "--game", game, "--work", self.work, "--out", self.out,
                "--stages", "shaders,pipelines", "--no-tui"]
        with mock.patch.object(sys, "argv", argv), \
                mock.patch("os.name", "nt"), \
                mock.patch.object(make_game, "below_normal"), \
                mock.patch.object(make_game, "check", return_value={"expected_exe": "game.exe"}), \
                mock.patch.object(make_game, "shaders") as shaders, \
                mock.patch.object(make_game, "pipelines") as pipelines, \
                mock.patch.object(make_game, "log"), \
                mock.patch("sys.stderr", io.StringIO()):
            with self.assertRaises(SystemExit) as ctx:
                make_game.main()
        self.assertEqual(ctx.exception.code, 1)
        shaders.assert_not_called()
        pipelines.assert_not_called()

    def test_exe_already_built_still_requires_shaders(self):
        self.write_runtime_exe()
        with mock.patch("os.name", "nt"):
            self.assert_preflight_fails(["pipelines"], "No translated shaders yet")

    def test_exe_in_out_passes_preflight(self):
        self.write_shader_index()
        self.write_runtime_exe(self.out)
        for platform in ("nt", "posix"):
            with self.subTest(platform=platform), mock.patch("os.name", platform):
                make_game.preflight(["pipelines"], self.work, out=self.out)

    def test_runtime_stage_builds_gta6rt_so_existing_exe_is_not_required(self):
        self.write_shader_index()
        with mock.patch("os.name", "nt"), \
                mock.patch.dict(os.environ, {"VULKAN_SDK": r"C:\Vulkan"}), \
                mock.patch.object(make_game, "visual_cpp_tools_ok", return_value=True):
            make_game.preflight(["runtime", "pipelines"], self.work, out=self.out)

    def test_windows_runtime_and_pipelines_without_vulkan_sdk_fails_preflight(self):
        for sdk in (None, ""):
            with self.subTest(sdk=sdk), mock.patch("os.name", "nt"), \
                    mock.patch.dict(os.environ), \
                    mock.patch.object(make_game, "visual_cpp_tools_ok", return_value=True):
                if sdk is None:
                    os.environ.pop("VULKAN_SDK", None)
                else:
                    os.environ["VULKAN_SDK"] = sdk
                err = self.assert_preflight_fails(["runtime", "pipelines"], "VULKAN_SDK")
                self.assertIn("gta6rt.exe", err)

    def test_windows_prebuilt_runtime_with_pipelines_does_not_require_vulkan_sdk(self):
        self.write_shader_index()
        build = os.path.join(self.root, "prebuilt")
        os.makedirs(build)
        for name in make_game.RUNTIME_FILES:
            with open(os.path.join(build, name), "wb"):
                pass
        env = {k: v for k, v in os.environ.items() if k != "VULKAN_SDK"}
        with mock.patch("os.name", "nt"), mock.patch.dict(os.environ, env, clear=True):
            make_game.preflight(["runtime", "pipelines"], self.work, runtime_build=build, out=self.out)

    def test_posix_pipelines_requires_wine_or_wine_env(self):
        self.write_shader_index()
        self.write_runtime_exe(self.out)
        env = {k: v for k, v in os.environ.items() if k != "WINE"}
        with mock.patch("os.name", "posix"), \
                mock.patch.dict(os.environ, env, clear=True), \
                mock.patch("shutil.which", side_effect=lambda t: None if t == "wine" else "/bin/tool"):
            self.assert_preflight_fails(["pipelines"], "Wine")
        # Setting WINE to an existing executable file satisfies preflight even when wine is not on PATH.
        with mock.patch("os.name", "posix"), \
                mock.patch.dict(os.environ, {"WINE": sys.executable}), \
                mock.patch("shutil.which", return_value=None):
            make_game.preflight(["pipelines"], self.work, out=self.out)

    def test_posix_runtime_and_pipelines_requires_wine(self):
        self.write_shader_index()
        env = {k: v for k, v in os.environ.items() if k != "WINE"}
        with mock.patch("os.name", "posix"), \
                mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.object(make_game, "cross_build_missing", return_value=[]), \
                mock.patch.object(make_game, "cross_configure"), \
                mock.patch("shutil.which", side_effect=lambda t: None if t == "wine" else "/bin/tool"):
            self.assert_preflight_fails(["runtime", "pipelines"], "Wine")

    def run_runtime_preflight(self, system, vulkan_sdk, runtime_build=None):
        with mock.patch("os.name", system), \
                mock.patch.dict(os.environ), \
                mock.patch.object(make_game, "visual_cpp_tools_ok", return_value=True), \
                mock.patch.object(make_game, "log") as log:
            if vulkan_sdk is None:
                os.environ.pop("VULKAN_SDK", None)
            else:
                os.environ["VULKAN_SDK"] = vulkan_sdk
            make_game.preflight(["runtime"], self.work, runtime_build=runtime_build, out=self.out)
        return [call.args[0] for call in log.call_args_list]

    def test_runtime_without_vulkan_sdk_warns_that_the_vulkan_device_may_be_missing(self):
        for sdk in (None, ""):
            with self.subTest(sdk=sdk):
                lines = self.run_runtime_preflight("nt", sdk)
                self.assertEqual(len([line for line in lines if "VULKAN_SDK is not set" in line]), 1)
                self.assertIn("LEONIDA_D3D12=vulkan", "".join(lines))

    def test_runtime_with_vulkan_sdk_does_not_warn(self):
        lines = self.run_runtime_preflight("nt", r"C:\Vulkan")
        self.assertFalse([line for line in lines if "VULKAN_SDK" in line])

    def test_runtime_warning_is_skipped_for_a_prebuilt_runtime_and_other_systems(self):
        build = os.path.join(self.root, "build")
        os.makedirs(build)
        for name in make_game.RUNTIME_FILES:
            with open(os.path.join(build, name), "wb"):
                pass
        self.assertFalse([line for line in self.run_runtime_preflight("nt", None, runtime_build=build)
                          if "VULKAN_SDK" in line])
        with mock.patch.object(make_game, "cross_build_missing", return_value=[]), \
                mock.patch.object(make_game, "cross_configure"):
            self.assertFalse([line for line in self.run_runtime_preflight("posix", None)
                              if "VULKAN_SDK" in line])

    def test_missing_exe_is_listed_with_other_missing_prerequisites(self):
        with mock.patch("os.name", "nt"), \
                mock.patch("shutil.which", side_effect=lambda t: None if t == "llvm-mc" else "/bin/tool"):
            err = self.assert_preflight_fails(["shaders", "pipelines"], "gta6rt.exe")
        self.assertIn("llvm-mc (LLVM 21+ with the AMDGPU target)", err)
        self.assertEqual(err.count("gta6rt.exe"), 1)


if __name__ == "__main__":
    unittest.main()
