"""Unit tests for the BCPack predictor profile hint in make_game.py.

    python -I tests/test_make_game_dstorage_profile.py
"""
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import make_game


class DstorageProfileHintTest(unittest.TestCase):
    """The runtime stage must name a profile that build_dstorage_history.py did not produce."""

    def test_missing_profile_names_generator_and_consequence(self):
        with tempfile.TemporaryDirectory() as temp:
            profile = os.path.join(temp, "bcpack", "history.bcph")
            hint = make_game.dstorage_profile_hint(profile)
        self.assertIsNotNone(hint)
        self.assertIn(profile, hint)
        self.assertIn("tools/build_dstorage_history.py", hint)
        self.assertIn("BC7", hint)

    def test_present_profile_needs_no_hint(self):
        with tempfile.TemporaryDirectory() as temp:
            profile = os.path.join(temp, "history.bcph")
            with open(profile, "wb") as handle:
                handle.write(b"BCPHIST1")
            self.assertIsNone(make_game.dstorage_profile_hint(profile))


if __name__ == "__main__":
    unittest.main()