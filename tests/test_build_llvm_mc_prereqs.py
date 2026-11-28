"""Regression tests for scripts/build_llvm_mc.ps1 validation, relative -Temp, VS lookup, tool fallback, and literal install."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_llvm_mc.ps1"


@unittest.skipUnless(os.name == "nt", "Windows PowerShell installer")
class BuildLlvmMcPrereqsTests(unittest.TestCase):
    @staticmethod
    def _run_ps(script_path: Path, *args: str, cwd: Path | None = None, env: dict[str, str] | None = None):
        return subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
             "-File", str(script_path), *args],
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )

    def test_invalid_version_rejected_before_prerequisite_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            for bad_version in ("", "   ", "21.1.8 extra", "21%TEMP%"):
                with self.subTest(version=bad_version):
                    res = self._run_ps(
                        SCRIPT, "-Version", bad_version, "-NoEnv",
                        cwd=Path(tmp), env={**os.environ, "PATH": ""},
                    )
                    self.assertNotEqual(res.returncode, 0)
                    self.assertIn("is not a valid LLVM release version", res.stderr)

    def test_relative_temp_resolves_against_provider_pwd_not_dotnet_cwd(self):
        text = SCRIPT.read_text(encoding="utf-8-sig")
        head_block = text[text.index('$ErrorActionPreference = "Stop"'):text.index("if (-not $Keep)")]
        writer_block = text[text.index('$bat = Join-Path $Temp "_build.cmd"'):text.index("$p = Start-Process", )]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ps_cwd = root / "ps_cwd"
            dotnet_cwd = root / "dotnet_cwd"
            ps_cwd.mkdir()
            dotnet_cwd.mkdir()
            (ps_cwd / "rel").mkdir()
            driver = root / "rel-temp.ps1"
            driver.write_text(
                "param([string]$Temp, [string]$PsCwd, [string]$DotnetCwd)\n"
                '$Version = "21.1.8"\n'
                'Set-Location -LiteralPath $PsCwd\n'
                '[Environment]::CurrentDirectory = $DotnetCwd\n'
                + head_block
                + "\n$installer = $PsCwd\n$vcvars = Join-Path $PsCwd 'vcvars64.bat'\n$body = 'echo ok'\n"
                + writer_block,
                encoding="utf-8",
                newline="\n",
            )
            res = self._run_ps(driver, "-Temp", r".\rel", "-PsCwd", str(ps_cwd), "-DotnetCwd", str(dotnet_cwd))
            self.assertEqual(res.returncode, 0, res.stdout + res.stderr)
            self.assertTrue((ps_cwd / "rel" / "_build.cmd").is_file())
            self.assertFalse((dotnet_cwd / "rel" / "_build.cmd").exists())

    def test_missing_vswhere_or_vcvars_fails_with_clear_error(self):
        text = SCRIPT.read_text(encoding="utf-8-sig")
        block = text[text.index('$installer = "${env:ProgramFiles(x86)}'):text.index("$jobs = ")]
        block = block.replace('${env:ProgramFiles(x86)}\\Microsoft Visual Studio\\Installer', '$FakeInstaller')

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_installer = root / "installer"
            fake_installer.mkdir()
            driver = root / "check-vs.ps1"
            driver.write_text(
                "param([string]$FakeInstaller)\n"
                '$ErrorActionPreference = "Stop"\n'
                + block,
                encoding="utf-8",
                newline="\n",
            )
            res = self._run_ps(driver, "-FakeInstaller", str(fake_installer))
            self.assertNotEqual(res.returncode, 0)
            self.assertIn("Visual Studio with the C++ x64 tools wasn't found.", res.stderr)
            self.assertNotIn("CommandNotFoundException", res.stderr)

            # vswhere returns a VS directory missing VC\Auxiliary\Build\vcvars64.bat
            fake_vs = root / "fake_vs"
            fake_vs.mkdir()
            (fake_installer / "vswhere.exe").write_text(f"@echo {fake_vs}\r\n", encoding="ascii")
            vswhere_cmd = fake_installer / "vswhere.cmd"
            vswhere_cmd.write_text(f"@echo {fake_vs}\r\n", encoding="ascii")
            driver2 = root / "check-vcvars.ps1"
            driver2.write_text(
                driver.read_text(encoding="utf-8").replace("vswhere.exe", "vswhere.cmd"),
                encoding="utf-8",
                newline="\n",
            )
            # Only test vcvars check when vswhere executable resolves cleanly.
            res2 = self._run_ps(driver2, "-FakeInstaller", str(fake_installer))
            self.assertNotEqual(res2.returncode, 0)
            self.assertIn("Visual Studio vcvars64.bat wasn't found", res2.stderr)

    def test_cmake_and_ninja_resolve_independently_between_vs_and_path(self):
        text = SCRIPT.read_text(encoding="utf-8-sig")
        block = text[text.index('$cmakeDir = Join-Path $vs'):text.index('$src = Join-Path $Temp "src"')]

        for vs_tool, path_tool in (("cmake", "ninja"), ("ninja", "cmake")):
            with self.subTest(vs_tool=vs_tool, path_tool=path_tool):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    vs = root / "vs"
                    path_dir = root / "pathbin"
                    path_dir.mkdir()
                    if vs_tool == "cmake":
                        p = vs / "Common7" / "IDE" / "CommonExtensions" / "Microsoft" / "CMake" / "CMake" / "bin" / "cmake.exe"
                        p.parent.mkdir(parents=True)
                        p.write_bytes(b"cmake")
                        q = path_dir / "ninja.cmd"
                        q.write_text("@echo off\r\n", encoding="ascii")
                    else:
                        p = vs / "Common7" / "IDE" / "CommonExtensions" / "Microsoft" / "CMake" / "Ninja" / "ninja.exe"
                        p.parent.mkdir(parents=True)
                        p.write_bytes(b"ninja")
                        q = path_dir / "cmake.cmd"
                        q.write_text("@echo off\r\n", encoding="ascii")

                    driver = root / "resolve-tools.ps1"
                    driver.write_text(
                        "param([string]$vs)\n"
                        '$ErrorActionPreference = "Stop"\n'
                        + block
                        + '\nWrite-Output "CMAKE=$cmake"\nWrite-Output "NINJA=$ninja"\n',
                        encoding="utf-8",
                        newline="\n",
                    )
                    res = self._run_ps(driver, "-vs", str(vs), env={**os.environ, "PATH": str(path_dir)})
                    self.assertEqual(res.returncode, 0, res.stdout + res.stderr)
                    self.assertIn("CMAKE=", res.stdout)
                    self.assertIn("NINJA=", res.stdout)

    def test_bracketed_source_and_dest_install_cleanly(self):
        text = SCRIPT.read_text(encoding="utf-8-sig")
        tail = text[text.index("New-Item -ItemType Directory -Force $Dest"):]
        setter = '[Environment]::SetEnvironmentVariable("LLVM_MC", $final, "User")'
        tail = tail.replace(setter, '$null = $final')

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src_dir = root / "build[1]"
            src_dir.mkdir()
            exe = src_dir / "llvm-mc.exe"
            exe.write_bytes(b"synthetic")
            dest = root / "tools[1]"
            driver = root / "install-brackets.ps1"
            driver.write_text(
                "param([string]$Dest, [string]$exe, [switch]$NoEnv)\n"
                '$ErrorActionPreference = "Stop"\n$Keep = $true\n'
                + tail,
                encoding="utf-8",
                newline="\n",
            )
            res = self._run_ps(driver, "-Dest", str(dest), "-exe", str(exe), "-NoEnv")
            self.assertEqual(res.returncode, 0, res.stdout + res.stderr)
            self.assertEqual((dest / "llvm-mc.exe").read_bytes(), b"synthetic")


if __name__ == "__main__":
    unittest.main()
