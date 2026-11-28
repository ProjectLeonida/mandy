"""Tests for Build.bat and Mods.bat quoted per-user Python lookup and non-interactive pause control.

    python -I tests/test_win_launchers.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD_BAT = ROOT / "Build.bat"
MODS_BAT = ROOT / "Mods.bat"

FAKE_PY_CS = r"""
using System;
using System.IO;
using System.Text;

public class FakePython {
    public static int Main(string[] args) {
        if (args.Length >= 2 && args[0] == "-c") {
            return Environment.GetEnvironmentVariable("FAKE_PY_OLD") == "1" ? 1 : 0;
        }
        string log = Environment.GetEnvironmentVariable("FAKE_PY_LOG");
        if (!string.IsNullOrEmpty(log)) {
            File.WriteAllLines(log, args, Encoding.UTF8);
        }
        string rc = Environment.GetEnvironmentVariable("FAKE_PY_RC");
        return string.IsNullOrEmpty(rc) ? 0 : int.Parse(rc);
    }
}
"""


class WinLaunchersTests(unittest.TestCase):
    def test_static_quoting_and_no_pause_guards(self) -> None:
        build_txt = BUILD_BAT.read_text(encoding="utf-8")
        mods_txt = MODS_BAT.read_text(encoding="utf-8")
        for txt in (build_txt, mods_txt):
            self.assertIn('set "PY_EXE=', txt)
            self.assertIn('set "PY_EXE=%~1"', txt)
            self.assertNotIn('set PY="%~1"', txt)
            self.assertIn("LEONIDA_NO_PAUSE", txt)
        self.assertIn(":find_user_python", mods_txt)
        self.assertIn("if not defined PYTMP", build_txt)

    @unittest.skipUnless(os.name == "nt" and shutil.which("powershell.exe"), "requires Windows cmd.exe and PowerShell")
    def test_per_user_python_discovery_with_ampersand_and_exit_propagation(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            repo = root / "repo"
            (repo / "tools").mkdir(parents=True)
            shutil.copy2(BUILD_BAT, repo / "Build.bat")
            shutil.copy2(MODS_BAT, repo / "Mods.bat")
            (repo / "make_game.py").write_text("# stub\n", encoding="utf-8", newline="\n")
            (repo / "tools" / "leonida_mods.py").write_text("# stub\n", encoding="utf-8", newline="\n")

            local_app = root / "user & test"
            py_dir = local_app / "Programs" / "Python" / "Python313"
            py_dir.mkdir(parents=True)
            plain_stub = root / "fake_py.exe"

            ps_base = ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass"]
            ps_env = {**os.environ, "FAKE_PY_EXE": str(plain_stub), "FAKE_PY_CS": FAKE_PY_CS}
            ps_env["PSModulePath"] = str(Path(os.environ["SystemRoot"]) / "System32/WindowsPowerShell/v1.0/Modules")
            compile_cmd = (
                "$ErrorActionPreference = 'Stop'; "
                "Add-Type -OutputAssembly $env:FAKE_PY_EXE -OutputType ConsoleApplication -TypeDefinition $env:FAKE_PY_CS"
            )
            subprocess.run([*ps_base, "-Command", compile_cmd], env=ps_env, check=True, timeout=60)
            shutil.move(str(plain_stub), str(py_dir / "python.exe"))

            sys32 = str(Path(os.environ["SystemRoot"]) / "System32")
            clean_env = {
                "SystemRoot": os.environ["SystemRoot"],
                "ComSpec": os.environ.get("ComSpec", os.path.join(sys32, "cmd.exe")),
                "PATH": sys32,
                "LOCALAPPDATA": str(local_app),
                "TEMP": str(root),
                "TMP": str(root),
                "LEONIDA_NO_PAUSE": "1",
                "LEONIDA_PY_INSTALL_DRYRUN": "1",
            }

            log_build = root / "build.args"
            r_build = subprocess.run(
                ["cmd.exe", "/c", "call", str(repo / "Build.bat"), "--out", "D:\\Out Dir"],
                env={**clean_env, "FAKE_PY_LOG": str(log_build), "FAKE_PY_RC": "0"},
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(r_build.returncode, 0, r_build.stdout + r_build.stderr)
            self.assertEqual(
                log_build.read_text(encoding="utf-8-sig").splitlines(),
                ["-I", str(repo / "make_game.py"), "--out", "D:\\Out Dir"],
            )

            log_mods = root / "mods.args"
            r_mods = subprocess.run(
                ["cmd.exe", "/c", "call", str(repo / "Mods.bat"), "--check"],
                env={**clean_env, "FAKE_PY_LOG": str(log_mods), "FAKE_PY_RC": "5"},
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(r_mods.returncode, 5, r_mods.stdout + r_mods.stderr)
            self.assertEqual(
                log_mods.read_text(encoding="utf-8-sig").splitlines(),
                ["-I", str(repo / "tools" / "leonida_mods.py"), "gui", "--check"],
            )

            # Without a qualifying Python, Mods.bat exits 1 cleanly without hanging on pause.
            r_missing = subprocess.run(
                ["cmd.exe", "/c", "call", str(repo / "Mods.bat")],
                env={**clean_env, "FAKE_PY_OLD": "1"},
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(r_missing.returncode, 1, r_missing.stdout + r_missing.stderr)
            self.assertIn("Python 3.12 or newer is needed", r_missing.stdout + r_missing.stderr)


if __name__ == "__main__":
    unittest.main()
