"""Safety tests for scripts/first_look.ps1 (-GameDir validation, stream cleanup, build/report guards)."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "first_look.ps1"
POWERSHELL = ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass"]

PE_INSPECT_STUB = (
    "from pathlib import Path\n"
    "import os\n"
    "import sys\n"
    "exe = Path(sys.argv[1])\n"
    "out = Path(sys.argv[sys.argv.index('--out') + 1])\n"
    "game_dir = sys.argv[sys.argv.index('--game-dir') + 1]\n"
    "(out / 'seen_gamedir.txt').write_text(game_dir, encoding='utf-8', newline='\\n')\n"
    "if os.environ.get('LEONIDA_SKIP_PE_REPORT') != '1':\n"
    "    (out / (exe.name + '.json')).write_text("
    "'{\"imports\":[],\"sections\":[]}', encoding='utf-8', newline='\\n')\n"
)

BUILD_RUNTIME_STUB = (
    "if ($env:LEONIDA_FAIL_BUILD -eq '1') {\n"
    "    & cmd.exe /c 'exit /b 7'\n"
    "    return\n"
    "}\n"
)


class FirstLookSafetyTests(unittest.TestCase):
    def test_static_guards_present(self):
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("GetUnresolvedProviderPathFromPSPath($GameDir)", text)
        self.assertIn("Test-Path -LiteralPath $GameDir -PathType Container", text)
        self.assertIn("Test-Path -LiteralPath $out -PathType Leaf", text)
        self.assertIn("[IO.Directory]::CreateDirectory($out)", text)
        self.assertIn("$fs.Dispose()", text)
        self.assertIn("$read -ne 2", text)
        self.assertIn("$global:LASTEXITCODE = 0", text)
        self.assertIn("build_runtime.ps1 failed ($LASTEXITCODE)", text)
        self.assertIn("Remove-Item -LiteralPath $reportPath -Force", text)
        self.assertIn("Test-Path -LiteralPath $reportPath -PathType Leaf", text)
        self.assertIn("pe_inspect did not write $reportPath", text)

    @unittest.skipUnless(os.name == "nt" and shutil.which("powershell.exe"), "Windows PowerShell required")
    def test_runtime_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            scripts = root / "scripts"
            tools = root / "tools"
            build = root / "runtime" / "build"
            src = root / "runtime" / "src"
            for folder in (scripts, tools, build, src):
                folder.mkdir(parents=True)
            wrapper = scripts / SCRIPT.name
            shutil.copyfile(SCRIPT, wrapper)
            (scripts / "build_runtime.ps1").write_text(BUILD_RUNTIME_STUB, encoding="utf-8", newline="\n")
            (tools / "pe_inspect.py").write_text(PE_INSPECT_STUB, encoding="utf-8", newline="\n")
            for stub in ("gdk_runtime_calls.py", "find_hash_lookup.py", "exe_checklist.py"):
                (tools / stub).write_text("import sys\n", encoding="utf-8", newline="\n")

            env = {k: v for k, v in os.environ.items() if k.upper() != "PSMODULEPATH"}
            env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
            env["PSModulePath"] = str(Path(os.environ["SystemRoot"]) / "System32/WindowsPowerShell/v1.0/Modules")

            # Compile stub loader outside any bracketed directory.
            bin_dir = Path(tmp) / "bin"
            bin_dir.mkdir()
            compile_cmd = (
                "$ErrorActionPreference = 'Stop'; "
                "Add-Type -OutputAssembly gta6rt.exe -OutputType ConsoleApplication -TypeDefinition "
                "'public class Loader { public static int Main() { return 0; } }'"
            )
            subprocess.run([*POWERSHELL, "-Command", compile_cmd], cwd=bin_dir, env=env, check=True, timeout=30)
            shutil.copyfile(bin_dir / "gta6rt.exe", build / "gta6rt.exe")
            (build / "gta6rt_tls_host.dll").touch()

            exe = root / "game.exe"
            exe.write_bytes(b"MZ" + b"\x00" * 62)

            def run_ps(*args, extra_env=None):
                run_env = env if not extra_env else {**env, **extra_env}
                return subprocess.run(
                    [*POWERSHELL, "-File", str(wrapper), *args],
                    cwd=root, env=run_env, capture_output=True, text=True, timeout=30,
                )

            out_dir = root / "reports" / "exe" / "game"

            # 1. Missing -GameDir fails before creating report dir.
            res = run_ps("-Exe", str(exe), "-GameDir", str(root / "missing-dir"))
            self.assertNotEqual(res.returncode, 0, res.stdout + res.stderr)
            self.assertIn("not found or not a directory", res.stdout + res.stderr)
            self.assertFalse(out_dir.exists())

            # 2. Regular file passed as -GameDir fails before creating report dir.
            not_a_dir = root / "notadir.txt"
            not_a_dir.write_text("x", encoding="utf-8", newline="\n")
            res = run_ps("-Exe", str(exe), "-GameDir", str(not_a_dir))
            self.assertNotEqual(res.returncode, 0, res.stdout + res.stderr)
            self.assertIn("not found or not a directory", res.stdout + res.stderr)
            self.assertFalse(out_dir.exists())

            # 3. Relative -GameDir containing brackets resolves literally and passes through cleanly.
            bracket_dir = root / "data [1]"
            bracket_dir.mkdir()
            res = run_ps("-Exe", str(exe), "-GameDir", "data [1]")
            self.assertEqual(res.returncode, 0, res.stdout + res.stderr)
            seen = Path((out_dir / "seen_gamedir.txt").read_text(encoding="utf-8").strip())
            self.assertEqual(seen.resolve(), bracket_dir.resolve())

            # 4. Empty (0-byte) non-MZ exe throws cleanly and does not leak the file handle.
            empty_exe = root / "empty.exe"
            empty_exe.write_bytes(b"")
            res = run_ps("-Exe", str(empty_exe))
            self.assertNotEqual(res.returncode, 0, res.stdout + res.stderr)
            self.assertIn("MZ header", res.stdout + res.stderr)
            empty_exe.unlink()  # Would fail on Windows if the handle were leaked.

            # 5. Report destination occupied by a file fails cleanly.
            file_exe = root / "fileout.exe"
            file_exe.write_bytes(b"MZ")
            (root / "reports" / "exe" / "fileout").write_text("blocker", encoding="utf-8", newline="\n")
            res = run_ps("-Exe", str(file_exe))
            self.assertNotEqual(res.returncode, 0, res.stdout + res.stderr)
            self.assertIn("is a file, not a directory", res.stdout + res.stderr)

            # 6. Failing build_runtime.ps1 throws when rebuild is triggered by newer source.
            src_file = src / "dummy.cpp"
            src_file.write_text("// new\n", encoding="utf-8", newline="\n")
            loader = build / "gta6rt.exe"
            old = time.time() - 3600
            os.utime(loader, (old, old))
            res = run_ps("-Exe", str(exe), extra_env={"LEONIDA_FAIL_BUILD": "1"})
            self.assertNotEqual(res.returncode, 0, res.stdout + res.stderr)
            self.assertIn("build_runtime.ps1 failed (7)", res.stdout + res.stderr)

            # Keep loader fresh for step 7.
            now = time.time() + 10
            os.utime(loader, (now, now))

            # 7. Stale report from a prior run is cleared before pe_inspect runs, so skipping output fails.
            (out_dir / "game.exe.json").write_text(
                '{"imports":[{"where":"shim","dll":"stale_old.dll","functions":["Foo"]}],"sections":[]}',
                encoding="utf-8",
                newline="\n",
            )
            res = run_ps("-Exe", str(exe), extra_env={"LEONIDA_SKIP_PE_REPORT": "1"})
            self.assertNotEqual(res.returncode, 0, res.stdout + res.stderr)
            self.assertIn("pe_inspect did not write", res.stdout + res.stderr)
            self.assertNotIn("stale_old.dll", res.stdout + res.stderr)


if __name__ == "__main__":
    unittest.main()
