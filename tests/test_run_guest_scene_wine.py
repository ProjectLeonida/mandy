"""run_guest_scene.py takes the Wine to use from $WINE when --wine is not given."""
import contextlib
import importlib.util
import io
import os
from pathlib import Path
import stat
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

path = Path(__file__).resolve().parents[1] / "scripts" / "run_guest_scene.py"
spec = importlib.util.spec_from_file_location("run_guest_scene_wine", path)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@unittest.skipIf(os.name == "nt", "the Wine checks only apply outside Windows")
class RunGuestSceneWineTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.loader, self.guest = self.root / "loader.exe", self.root / "guest.exe"
        self.loader.write_bytes(b"MZ")
        self.guest.write_bytes(b"MZ")
        (self.root / "shaders" / "spirv").mkdir(parents=True)
        self.wine = self.root / "umu-proton" / "files" / "bin" / "wine"
        self.wine.parent.mkdir(parents=True)
        self.wine.write_text("#!/bin/sh\n")
        self.wine.chmod(self.wine.stat().st_mode | stat.S_IXUSR)

    def parse(self, *extra, env=None):
        argv = ["run_guest_scene.py", "--loader", str(self.loader), "--guest", str(self.guest),
                "--shaders", str(self.root / "shaders"), *extra]
        environ = {k: v for k, v in os.environ.items() if k != "WINE"}
        environ.update(env or {})
        err = io.StringIO()
        with mock.patch.object(sys, "argv", argv), mock.patch.dict(os.environ, environ, clear=True), \
                contextlib.redirect_stderr(err):
            try:
                return runner.parse_args(), err.getvalue()
            except SystemExit as e:
                return e.code, err.getvalue()

    def test_wine_from_environment(self):
        args, _ = self.parse(env={"WINE": str(self.wine)})
        self.assertEqual(args.wine, str(self.wine.resolve()))

    def test_wine_args_with_wine_from_environment(self):
        args, _ = self.parse("--wine-arg=--debug", env={"WINE": str(self.wine)})
        self.assertEqual((args.wine, args.wine_arg), (str(self.wine.resolve()), ["--debug"]))

    def test_option_overrides_environment(self):
        other = self.root / "wine-other"
        other.write_text("#!/bin/sh\n")
        args, _ = self.parse("--wine", str(other), env={"WINE": str(self.wine)})
        self.assertEqual(args.wine, str(other.resolve()))

    def test_missing_wine_still_refused_outside_windows(self):
        for env in ({}, {"WINE": ""}):
            with self.subTest(env=env):
                code, err = self.parse(env=env)
                self.assertEqual(code, 2)
                self.assertIn("--wine (or WINE) is required outside Windows", err)

    def test_wine_from_environment_that_does_not_exist(self):
        code, err = self.parse(env={"WINE": str(self.root / "nowhere" / "wine")})
        self.assertEqual(code, 2)
        self.assertIn("--wine: not found", err)


class RunGuestSceneWindowsEnvTest(unittest.TestCase):
    """On native Windows, an ambient WINE env var must not hijack a native launch."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.loader, self.guest = self.root / "loader.exe", self.root / "guest.exe"
        self.loader.write_bytes(b"MZ")
        self.guest.write_bytes(b"MZ")
        (self.root / "shaders" / "spirv").mkdir(parents=True)
        self.wine = self.root / "wine.exe"
        self.wine.write_bytes(b"MZ")

    def parse_as_windows(self, *extra, env=None):
        argv = ["run_guest_scene.py", "--loader", str(self.loader), "--guest", str(self.guest),
                "--shaders", str(self.root / "shaders"), *extra]
        environ = {k: v for k, v in os.environ.items() if k != "WINE"}
        environ.update(env or {})
        err = io.StringIO()
        with mock.patch.object(runner, "os", SimpleNamespace(name="nt", environ=environ)), \
                mock.patch.object(sys, "argv", argv), \
                contextlib.redirect_stderr(err):
            try:
                return runner.parse_args(), err.getvalue()
            except SystemExit as e:
                return e.code, err.getvalue()

    def test_ambient_wine_is_ignored_on_windows(self):
        for value in (str(self.wine), str(self.root / "missing-wine.exe")):
            with self.subTest(WINE=value):
                args, err = self.parse_as_windows(env={"WINE": value})
                self.assertEqual(err, "")
                self.assertIsNone(args.wine)

    def test_explicit_wine_flag_still_works_on_windows(self):
        args, err = self.parse_as_windows("--wine", str(self.wine), env={"WINE": str(self.root / "missing.exe")})
        self.assertEqual(err, "")
        self.assertEqual(args.wine, str(self.wine.resolve()))

    def test_wine_arg_requires_explicit_wine_on_windows(self):
        code, err = self.parse_as_windows("--wine-arg=--debug", env={"WINE": str(self.wine)})
        self.assertEqual(code, 2)
        self.assertIn("--wine-arg requires --wine", err)


if __name__ == "__main__":
    unittest.main()
