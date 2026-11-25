"""AGENTS.md's translator-suite counts match the semantic test tree.

    python -I tests/test_agents_semantic_counts.py
"""
import os
import re
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEMANTIC_DIR = os.path.join(ROOT, "tests", "semantic")
AGENTS_PATH = os.path.join(ROOT, "AGENTS.md")
DEF_TEST_RE = re.compile(r"^    def test_", re.M)


def count_semantic_suite():
    files = sorted(
        name for name in os.listdir(SEMANTIC_DIR)
        if name.startswith("test_") and name.endswith(".py")
        and os.path.isfile(os.path.join(SEMANTIC_DIR, name))
    )
    methods = 0
    for name in files:
        with open(os.path.join(SEMANTIC_DIR, name), encoding="utf-8", errors="replace") as f:
            methods += len(DEF_TEST_RE.findall(f.read()))
    return len(files), methods


def test_without_game_paragraph(text):
    marker = "## Test without the game first"
    start = text.index(marker) + len(marker)
    after = text[start:].lstrip("\r\n")
    para = after.split("\n\n", 1)[0]
    return " ".join(line.strip() for line in para.splitlines())


class AgentsSemanticCounts(unittest.TestCase):
    def test_stale_counts_are_gone(self):
        with open(AGENTS_PATH, encoding="utf-8") as f:
            text = f.read()
        self.assertNotIn("465 files", text)
        self.assertNotIn("6 108", text)
        self.assertNotIn("6,108", text)

    def test_documented_counts_match_tree(self):
        with open(AGENTS_PATH, encoding="utf-8") as f:
            text = f.read()
        para = test_without_game_paragraph(text)
        m = re.search(r"translator suite[^()]*\(([^)]+)\)", para)
        self.assertIsNotNone(m, f"missing translator-suite parenthetical in: {para}")
        paren = m.group(1)

        files_n, methods_n = count_semantic_suite()
        self.assertIn(f"{files_n} ", paren)
        methods_variants = (str(methods_n), f"{methods_n:,}", f"{methods_n:,}".replace(",", " "))
        self.assertTrue(
            any(v in paren for v in methods_variants),
            f"expected {methods_n} (in one of {methods_variants}) in parenthetical: {paren!r}",
        )
        self.assertIn("test_*.py", paren)
        self.assertIn("def test_", paren)


if __name__ == "__main__":
    unittest.main()
