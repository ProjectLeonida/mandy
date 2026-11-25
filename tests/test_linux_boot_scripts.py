"""scripts/linux-boot: the scripts parse, are executable and refuse wrong input before they start anything.

No Wine, no systemd unit and no game files are involved: every case below ends in a script's own argument checks.

    python -I tests/test_linux_boot_scripts.py -v
"""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOOT = os.path.join(ROOT, "scripts", "linux-boot")
SCRIPTS = ("boot.sh", "boot-inner.sh", "boot-digest.sh", "run-boot.sh", "configure-worktree.sh")
BASH = shutil.which("bash")


@unittest.skipUnless(sys.platform.startswith("linux") and BASH, "the boot scripts are for Linux and need bash")
class LinuxBootScripts(unittest.TestCase):
    def run_script(self, name, *args, **env):
        """(exit code, stderr) of one script with only the given LEONIDA_*/GTA6RT variables set."""
        clean = {k: v for k, v in os.environ.items() if not k.startswith("LEONIDA_") and k != "GTA6RT"}
        clean.update(env)
        done = subprocess.run([BASH, os.path.join(BOOT, name), *args], env=clean, capture_output=True, text=True,
                              timeout=60)
        return done.returncode, done.stderr

    def test_scripts_parse_and_are_executable(self):
        for name in SCRIPTS:
            with self.subTest(script=name):
                path = os.path.join(BOOT, name)
                self.assertTrue(os.access(path, os.X_OK), "not executable")
                self.assertEqual(subprocess.run([BASH, "-n", path], capture_output=True).returncode, 0)
                with open(path, "rb") as f:
                    self.assertNotIn(b"\r", f.read())

    def test_boot_refuses_wrong_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            game = os.path.join(tmp, "game")
            os.mkdir(game)
            out = os.path.join(tmp, "boot-dir")

            def boot(*args, **env):
                return self.run_script("boot.sh", *args, LEONIDA_BOOT_DIR=out, **env)

            code, err = boot("play")
            self.assertEqual(code, 2)
            self.assertIn("dry-run, init or run", err)
            code, err = boot("dry-run")
            self.assertEqual(code, 2)
            self.assertIn("LEONIDA_GAME_DIR", err)
            code, err = boot("dry-run", LEONIDA_GAME_DIR=game)
            self.assertEqual(code, 2)
            self.assertIn("game_scarlett_master.exe not found", err)
            # An empty stand-in for the executable: the next check is the loader, then the shader library.
            open(os.path.join(game, "game_scarlett_master.exe"), "wb").close()
            missing = os.path.join(tmp, "none", "gta6rt.exe")
            code, err = boot("dry-run", LEONIDA_GAME_DIR=game, GTA6RT=missing)
            self.assertEqual(code, 2)
            self.assertIn("no loader at", err)
            loader = os.path.join(tmp, "gta6rt.exe")
            open(loader, "wb").close()
            code, err = boot("run", LEONIDA_GAME_DIR=game, GTA6RT=loader)
            self.assertEqual(code, 2)
            self.assertIn("LEONIDA_SHADERS", err)
            code, err = boot("run", LEONIDA_GAME_DIR=game, GTA6RT=loader, LEONIDA_SHADERS=os.path.join(tmp, "no-library"))
            self.assertEqual(code, 2)
            self.assertIn("no shader library", err)
            self.assertFalse(os.path.exists(out), "a refused boot must not create its folders")

    def test_run_boot_refuses_wrong_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "boot-dir")
            code, err = self.run_script("run-boot.sh", "bad name", "x.exe", LEONIDA_BOOT_DIR=out)
            self.assertEqual(code, 2)
            self.assertIn("letters, digits", err)
            code, err = self.run_script("run-boot.sh", "one", os.path.join(tmp, "gta6rt.exe"), LEONIDA_BOOT_DIR=out)
            self.assertEqual(code, 2)
            self.assertIn("no such gta6rt", err)
            loader = os.path.join(tmp, "gta6rt.exe")
            open(loader, "wb").close()
            code, err = self.run_script("run-boot.sh", "one", loader, LEONIDA_BOOT_DIR=out)
            self.assertEqual(code, 2)
            self.assertIn("gta6rt_tls_host.dll is missing", err)
            self.assertFalse(os.path.exists(out), "a refused boot must not create its run folder")


if __name__ == "__main__":
    unittest.main()
