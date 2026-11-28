"""Unit tests for make_game.py's --pick when the chosen folder's name isn't valid in the locale's encoding:
kdialog and zenity print the folder as raw bytes, and neither decoding the answer nor logging the chosen
folder may stop the build. The dialogs are stand-ins, so no display or game files are needed.

    python -I tests/test_make_game_pick_fsdecode.py
"""
import io
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import make_game

CHOSEN = b"/tmp/caf\xe9"        # "café" in latin-1 bytes, refused by a UTF-8 locale
UNDECODABLE = "/tmp/caf\udce9"  # what os.fsdecode builds from CHOSEN: the byte stays in as a surrogate
VERSION = {"title": "Test", "version": "1.0.2.1"}


@unittest.skipIf(os.name == "nt", "kdialog and zenity never run on Windows: Tk opens the dialog there")
class NativeAskBytesTest(unittest.TestCase):
    """native_ask reads the dialog's answer as bytes and decodes it itself, so a name the locale's encoding
    can't decode reaches the build anyway."""

    def test_returns_a_folder_the_locale_cannot_decode(self):
        with tempfile.TemporaryDirectory() as tmp:
            exe = os.path.join(tmp, "kdialog")   # a stand-in kdialog: prints the folder's bytes, exits 0
            with open(exe, "w", encoding="utf-8", newline="\n") as f:
                f.write("#!/usr/bin/env python3\nimport sys\n"
                        "sys.stdout.buffer.write(b'/tmp/caf\\xe9\\n')\n")
            os.chmod(exe, 0o755)
            if not os.access(exe, os.X_OK):
                self.skipTest("the temporary folder is mounted noexec: set TMPDIR to one that is not")
            d = make_game.native_ask("kdialog", exe, "/tmp")
        self.assertIsInstance(d, str)
        self.assertEqual(os.fsencode(d), CHOSEN)   # round trip: the dialog's own bytes, newline stripped


class ShowPathTest(unittest.TestCase):
    """The text pick_game logs about the chosen folder: a surrogate in the path is a byte the locale couldn't
    decode, and a UTF-8 terminal refuses to print it."""

    def strict_utf8(self):
        return io.TextIOWrapper(io.BytesIO(), encoding="utf-8", errors="strict")

    def test_leaves_a_folder_that_prints_alone(self):
        self.assertEqual(make_game.show_path("/games/leonida"), "/games/leonida")

    def test_shows_a_folder_a_utf8_stream_refuses_without_raising(self):
        out = self.strict_utf8()
        out.write(make_game.show_path(UNDECODABLE))        # the chosen path itself can't be written there
        out.flush()
        self.assertIn(b"/tmp/caf", out.buffer.getvalue())   # the folder still shows up, escaped
        with self.assertRaises(UnicodeEncodeError):       # what logging the raw path does there
            out.write(UNDECODABLE)

    @unittest.skipIf(os.name == "nt", "on Windows a surrogate isn't a byte of the folder's name")
    def test_shows_the_byte_that_could_not_be_decoded(self):
        self.assertEqual(make_game.show_path(UNDECODABLE), "/tmp/caf\\xe9")


@unittest.skipIf(os.name == "nt", "kdialog and zenity never run on Windows: Tk opens the dialog there")
class PickGameUndecodableTest(unittest.TestCase):
    """--pick on the desktop's dialog: the terminal gets the folder with the undecodable bytes escaped, the
    build gets them back."""

    def test_logs_the_shown_folder_and_returns_the_chosen_one(self):
        run = MagicMock(return_value=types.SimpleNamespace(returncode=0, stdout=CHOSEN + b"\n"))
        out = io.TextIOWrapper(io.BytesIO(), encoding="utf-8", errors="strict")
        with patch("make_game.native_picker", return_value=("kdialog", "/usr/bin/kdialog")), \
                patch("make_game.subprocess.run", run), patch("make_game.match_version", return_value=VERSION), \
                patch.object(sys, "stdout", out):
            d = make_game.pick_game()
        out.flush()
        self.assertEqual(os.fsencode(d), CHOSEN)   # the build's path is the bytes the dialog printed
        logged = out.buffer.getvalue().decode("utf-8")   # strict: nothing unprintable reached the terminal
        self.assertIn(f"chose Test 1.0.2.1 in {make_game.show_path(os.fsdecode(CHOSEN))}", logged)


if __name__ == "__main__":
    unittest.main()
