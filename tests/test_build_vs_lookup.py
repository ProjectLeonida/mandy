"""The Windows build scripts find a Preview or Insiders Visual Studio when no stable one has the C++ x64 tools.

vswhere lists a prerelease install only when asked with -prerelease. scripts/build_llvm_mc.ps1 and
scripts/build_runtime.ps1 ask for a stable install first, so a machine with both keeps the stable one, then for a
prerelease one. Each script's lookup runs as written against a stub vswhere.exe built with the .NET Framework C#
compiler; no Visual Studio, CMake, GPU or game files are needed.
"""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
CSC = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Microsoft.NET/Framework64/v4.0.30319/csc.exe"
# Logs its arguments and prints the install vswhere would: LEONIDA_VS_PRERELEASE when asked with -prerelease and one
# is set, else LEONIDA_VS_STABLE; nothing when that is empty.
VSWHERE = r'''
using System;
using System.IO;
class VsWhere {
    static int Main(string[] args) {
        File.AppendAllText(Environment.GetEnvironmentVariable("LEONIDA_VSWHERE_LOG"), string.Join(" ", args) + "\n");
        string found = Environment.GetEnvironmentVariable("LEONIDA_VS_STABLE");
        string preview = Environment.GetEnvironmentVariable("LEONIDA_VS_PRERELEASE");
        if (Array.IndexOf(args, "-prerelease") >= 0 && !string.IsNullOrEmpty(preview)) found = preview;
        if (!string.IsNullOrEmpty(found)) Console.WriteLine(found);
        return 0;
    }
}
'''


@unittest.skipUnless(os.name == "nt" and CSC.is_file(), "requires Windows PowerShell and .NET Framework csc")
class VisualStudioLookupTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="leonida_vswhere_")
        cls.addClassCleanup(cls.temp.cleanup)
        cls.base = Path(cls.temp.name)
        installer = cls.base / "Microsoft Visual Studio" / "Installer"
        installer.mkdir(parents=True)
        source = cls.base / "vswhere.cs"
        source.write_text(VSWHERE, encoding="utf-8", newline="\n")
        subprocess.run([str(CSC), "/nologo", "/target:exe", "/out:" + str(installer / "vswhere.exe"), str(source)],
                       check=True, capture_output=True, timeout=30)

    def lookup(self, script, stable, prerelease):
        """The script's lookup, from $installer to the line before $vcvars, with ProgramFiles(x86) at the stub.
        Returns the PowerShell result and the arguments of each vswhere call."""
        text = (SCRIPTS / script).read_text(encoding="utf-8-sig")
        start = text.index('$installer = "${env:ProgramFiles(x86)}')
        block = text[start:text.index("$vcvars = Join-Path $vs", start)]
        with tempfile.TemporaryDirectory(dir=self.base) as tmp:
            driver = Path(tmp) / "lookup.ps1"
            driver.write_text("param([string]$ProgramFilesX86)\n"
                              "$ErrorActionPreference = 'Stop'\n"
                              "${env:ProgramFiles(x86)} = $ProgramFilesX86\n"
                              + block + 'Write-Output "VS=$vs"\n',
                              encoding="utf-8", newline="\n")
            log = Path(tmp) / "vswhere.log"
            log.write_bytes(b"")
            result = subprocess.run(
                ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                 "-File", str(driver), "-ProgramFilesX86", str(self.base)],
                env=dict(os.environ, LEONIDA_VSWHERE_LOG=str(log), LEONIDA_VS_STABLE=stable,
                         LEONIDA_VS_PRERELEASE=prerelease),
                capture_output=True, text=True, timeout=60,
            )
            return result, [line.split() for line in log.read_text(encoding="utf-8").splitlines()]

    def test_prerelease_only_when_no_stable_install(self):
        stable, preview = r"C:\VS\2022\Community", r"C:\VS\18\Insiders"
        for script in ("build_llvm_mc.ps1", "build_runtime.ps1"):
            with self.subTest(script=script, installed="prerelease only"):
                result, calls = self.lookup(script, "", preview)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("VS=" + preview, result.stdout.splitlines())
                self.assertEqual(["-prerelease" in args for args in calls], [False, True])
            with self.subTest(script=script, installed="stable and prerelease"):
                result, calls = self.lookup(script, stable, preview)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("VS=" + stable, result.stdout.splitlines())
                self.assertEqual(["-prerelease" in args for args in calls], [False])
            with self.subTest(script=script, installed="neither"):
                result, calls = self.lookup(script, "", "")
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn("Visual Studio with the C++ x64 tools wasn't found.", result.stderr)
                self.assertEqual(["-prerelease" in args for args in calls], [False, True])


if __name__ == "__main__":
    unittest.main()
