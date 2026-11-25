"""docs/sandbox.md labels #2519 as an old original-repository number, not a live issue.

    python -I tests/test_docs_sandbox_issue_ref.py
"""
import os
import re
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SANDBOX_PATH = os.path.join(ROOT, "docs", "sandbox.md")


class SandboxIssueRef(unittest.TestCase):
    def test_bare_2519_is_not_a_live_cite(self):
        with open(SANDBOX_PATH, encoding="utf-8") as f:
            text = f.read()
        self.assertNotIn("(#2519)", text)
        self.assertNotIn("ItsNotRexed/Leonida", text)
        self.assertNotIn("ubiquitarianism/leonida/issues/2519", text)

    def test_old_number_is_explained(self):
        with open(SANDBOX_PATH, encoding="utf-8") as f:
            text = f.read()
        m = re.search(r"## 2\. How the pair is chosen\s+(.*?)(?:\n\n|\r\n\r\n)", text, re.S)
        self.assertIsNotNone(m, "missing section 2 lead paragraph in docs/sandbox.md")
        para = " ".join(m.group(1).splitlines())
        self.assertIn("material", para)
        self.assertIn("program table", para)
        self.assertIn("executable", para)
        self.assertIn("#2519", para)
        self.assertRegex(para, re.compile(r"\b(old|original)\b", re.I))
        self.assertRegex(
            para,
            re.compile(r"\b(original repository|original repo|gone|private|links nowhere|does not resolve|no longer)\b", re.I),
        )


if __name__ == "__main__":
    unittest.main()
