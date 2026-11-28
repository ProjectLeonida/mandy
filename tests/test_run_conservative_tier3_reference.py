"""Tests for scripts/run_conservative_tier3_reference.ps1 using synthetic cmake/ctest shims."""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_conservative_tier3_reference.ps1"


@unittest.skipUnless(os.name == "nt", "Windows PowerShell runner")
class RunConservativeTier3ReferenceTests(unittest.TestCase):
    def _stage_repo_copy(self, tmp: str):
        root = Path(tmp) / "repo"
        scripts_dir = root / "scripts"
        scripts_dir.mkdir(parents=True)
        staged_script = scripts_dir / "run_conservative_tier3_reference.ps1"
        staged_script.write_bytes(SCRIPT.read_bytes())
        bin_dir = Path(tmp) / "bin"
        bin_dir.mkdir()
        log_dir = Path(tmp) / "logs"
        log_dir.mkdir()
        return root, staged_script, bin_dir, log_dir

    @staticmethod
    def _write_ps1_shims(bin_dir: Path, log_dir: Path, exit_spec: dict[str, int] | None = None):
        exit_spec = exit_spec or {}
        (bin_dir / "cmake.ps1").write_text(
            "param([Parameter(ValueFromRemainingArguments=$true)][string[]]$Rest)\n"
            f"$log = '{log_dir}'\n"
            "if ($Rest -contains '--build') {\n"
            "  [IO.File]::WriteAllText((Join-Path $log 'cmake_build.json'), (ConvertTo-Json -Compress $Rest))\n"
            f"  $global:LASTEXITCODE = {exit_spec.get('build', 0)}\n"
            "} else {\n"
            "  [IO.File]::WriteAllText((Join-Path $log 'cmake_cfg.json'), (ConvertTo-Json -Compress $Rest))\n"
            f"  $global:LASTEXITCODE = {exit_spec.get('cfg', 0)}\n"
            "}\n",
            encoding="utf-8",
            newline="\n",
        )
        (bin_dir / "ctest.ps1").write_text(
            "param([Parameter(ValueFromRemainingArguments=$true)][string[]]$Rest)\n"
            f"$log = '{log_dir}'\n"
            "$record = @{ args = $Rest; device_set = (Test-Path Env:LEONIDA_VK_DEVICE); device = $env:LEONIDA_VK_DEVICE }\n"
            "[IO.File]::WriteAllText((Join-Path $log 'ctest.json'), ($record | ConvertTo-Json -Compress))\n"
            f"$global:LASTEXITCODE = {exit_spec.get('ctest', 0)}\n",
            encoding="utf-8",
            newline="\n",
        )

    @staticmethod
    def _run_ps_wrapper(command: str, env: dict[str, str]):
        return subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )

    def test_missing_cmake_or_ctest_fails_fast_before_configure(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, staged_script, bin_dir, log_dir = self._stage_repo_copy(tmp)
            self._write_ps1_shims(bin_dir, log_dir)

            # 1. Empty PATH -> missing cmake.
            res_no_cmake = subprocess.run(
                ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                 "-File", str(staged_script)],
                env={**os.environ, "PATH": ""},
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30,
            )
            self.assertNotEqual(res_no_cmake.returncode, 0)
            self.assertIn("cmake wasn't found on PATH.", res_no_cmake.stderr)

            # 2. cmake present, ctest removed -> fails before configure runs.
            (bin_dir / "ctest.ps1").unlink()
            res_no_ctest = subprocess.run(
                ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                 "-File", str(staged_script)],
                env={**os.environ, "PATH": str(bin_dir)},
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30,
            )
            self.assertNotEqual(res_no_ctest.returncode, 0)
            self.assertIn("ctest wasn't found on PATH.", res_no_ctest.stderr)
            self.assertFalse((log_dir / "cmake_cfg.json").exists())

    def test_exact_ctest_args_device_trimming_and_env_restoration(self):
        for initial_device, passed_device, ctest_exit in (
            ("PreSetGpu", "  Arc A770  ", 0),
            (None, "  Arc A770  ", 4),
            ("PreSetGpu", "", 0),
        ):
            with self.subTest(initial=initial_device, passed=passed_device, ctest_exit=ctest_exit):
                with tempfile.TemporaryDirectory() as tmp:
                    _, staged_script, bin_dir, log_dir = self._stage_repo_copy(tmp)
                    self._write_ps1_shims(bin_dir, log_dir, {"ctest": ctest_exit})
                    env = {k: v for k, v in os.environ.items() if k.upper() != "LEONIDA_VK_DEVICE"}
                    env["PATH"] = str(bin_dir)
                    if initial_device is not None:
                        env["LEONIDA_VK_DEVICE"] = initial_device

                    dev_arg = f"-Device '{passed_device}'" if passed_device else ""
                    cmd = (
                        f"$ErrorActionPreference = 'Stop'; "
                        f"try {{ & '{staged_script}' {dev_arg} }} "
                        f"catch {{ Write-Output ('ERR:' + $_.Exception.Message) }} "
                        f"finally {{ "
                        f"Write-Output ('POST_HAS:' + (Test-Path Env:LEONIDA_VK_DEVICE)); "
                        f"Write-Output ('POST_VAL:' + $env:LEONIDA_VK_DEVICE) }}"
                    )
                    res = self._run_ps_wrapper(cmd, env)
                    self.assertEqual(res.returncode, 0, res.stdout + res.stderr)

                    ctest_log = json.loads((log_dir / "ctest.json").read_text(encoding="utf-8-sig"))
                    args = ctest_log["args"]
                    self.assertIn("--no-tests=error", args)
                    self.assertIn("^d3d12_conservative_tier3_reference$", args)
                    if passed_device.strip():
                        self.assertTrue(ctest_log["device_set"])
                        self.assertEqual(ctest_log["device"], passed_device.strip())
                    else:
                        self.assertFalse(ctest_log["device_set"])

                    lines = dict(line.split(":", 1) for line in res.stdout.splitlines() if ":" in line)
                    if ctest_exit:
                        self.assertIn("The Tier 3 reference test failed (4).", lines.get("ERR", ""))
                    else:
                        self.assertNotIn("ERR", lines)

                    if initial_device is None:
                        self.assertEqual(lines.get("POST_HAS"), "False")
                    else:
                        self.assertEqual(lines.get("POST_HAS"), "True")
                        self.assertEqual(lines.get("POST_VAL"), initial_device)

    def test_benign_stderr_succeeds_under_stop_redirect_and_nonzero_phases_throw(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, staged_script, bin_dir, _ = self._stage_repo_copy(tmp)
            (bin_dir / "cmake.cmd").write_text(
                "@echo off\r\necho cmake warning line 1>&2\r\nexit /b 0\r\n", encoding="ascii"
            )
            (bin_dir / "ctest.cmd").write_text(
                "@echo off\r\necho ctest note line 1>&2\r\nexit /b 0\r\n", encoding="ascii"
            )
            env = {**os.environ, "PATH": str(bin_dir)}
            res = self._run_ps_wrapper(
                f"$ErrorActionPreference = 'Stop'; & '{staged_script}' 2>&1",
                env,
            )
            self.assertEqual(res.returncode, 0, res.stdout + res.stderr)

        # Configure and build failures still surface expected exit-code errors cleanly.
        for phase, exit_spec, expected_msg in (
            ("cfg", {"cfg": 3}, "CMake configure failed (3)."),
            ("build", {"build": 2}, "Building the Tier 3 reference test failed (2)."),
        ):
            with self.subTest(phase=phase):
                with tempfile.TemporaryDirectory() as tmp:
                    _, staged_script, bin_dir, log_dir = self._stage_repo_copy(tmp)
                    self._write_ps1_shims(bin_dir, log_dir, exit_spec)
                    res = subprocess.run(
                        ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                         "-File", str(staged_script)],
                        env={**os.environ, "PATH": str(bin_dir)},
                        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30,
                    )
                    self.assertNotEqual(res.returncode, 0)
                    self.assertIn(expected_msg, res.stderr)


if __name__ == "__main__":
    unittest.main()
