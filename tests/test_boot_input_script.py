"""Unit tests for tools/boot_input_script.py on synthetic scripts (no game data, no GPU)."""

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import boot_input_script as bis


class ParseTest(unittest.TestCase):
    def test_absolute_only(self):
        plan = bis.parse("a@45,a@60,start@120")
        self.assertEqual(plan["absolute"], "a@45,a@60,start@120")
        self.assertEqual(plan["state"], [])
        self.assertEqual(plan["rejected"], [])

    def test_state_only(self):
        plan = bis.parse("a@session=d+5,b@game=3")
        self.assertEqual(plan["absolute"], "")
        self.assertEqual(
            plan["state"],
            [
                {"button": "a", "field": "session", "value": "d", "delay": 5.0},
                {"button": "b", "field": "game", "value": "3", "delay": 0.0},
            ],
        )

    def test_mixed_spaces_and_case(self):
        plan = bis.parse(" A@45 , a@Session=D+5.5 , b@Game=0x3+0 ")
        self.assertEqual(plan["absolute"], "a@45")
        self.assertEqual(
            plan["state"],
            [
                {"button": "a", "field": "session", "value": "d", "delay": 5.5},
                {"button": "b", "field": "game", "value": "3", "delay": 0.0},
            ],
        )

    def test_rejects_bad_entries(self):
        plan = bis.parse("rsup@session=d+5,a@session=zz,a@-1,a@nan,nope@3,a@,a@game=+5")
        self.assertEqual(plan["absolute"], "")
        self.assertEqual(plan["state"], [])
        self.assertEqual(len(plan["rejected"]), 7)

    def test_empty(self):
        plan = bis.parse("")
        self.assertEqual(plan["absolute"], "")
        self.assertEqual(plan["state"], [])
        self.assertEqual(plan["rejected"], [])


class DueTest(unittest.TestCase):
    def plan(self):
        return bis.parse("a@session=d+5,b@game=3+1")

    def test_delay_not_reached(self):
        self.assertEqual(bis.due(self.plan(), now=14.9, session="d", seen={"session": 10}), [])

    def test_fires_after_delay(self):
        self.assertEqual(
            bis.due(self.plan(), now=15.0, session="d", seen={"session": 10}),
            [{"index": 0, "button": "a"}],
        )

    def test_leading_zero_and_case(self):
        plan = bis.parse("a@session=0D+0")
        self.assertEqual(bis.due(plan, now=0, session="d", seen={"session": 0}), [{"index": 0, "button": "a"}])

    def test_other_field_ignored(self):
        plan = bis.parse("a@session=d+5")
        self.assertEqual(bis.due(plan, now=100, game="3", session=None, seen={"game": 0}), [])

    def test_already_fired(self):
        self.assertEqual(bis.due(self.plan(), now=100, session="d", seen={"session": 0}, fired=[0]), [])

    def test_no_state_seen(self):
        self.assertEqual(bis.due(self.plan(), now=100, session="d", seen={}), [])

    def test_games_entry(self):
        self.assertEqual(
            bis.due(self.plan(), now=1.0, game="3", seen={"game": 0}),
            [{"index": 1, "button": "b"}],
        )


if __name__ == "__main__":
    unittest.main()
