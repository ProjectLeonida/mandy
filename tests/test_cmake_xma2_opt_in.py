"""Verify that runtime/CMakeLists.txt gates the configure-time Cargo XMA2 build behind opt-in."""
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
CMAKE = ROOT / "runtime" / "CMakeLists.txt"


class CMakeXma2OptInTest(unittest.TestCase):
    def test_cargo_build_is_gated_by_opt_in(self):
        text = CMAKE.read_text(encoding="utf-8")
        self.assertRegex(
            text,
            r'option\(LEONIDA_XMA2_DECODER\s+"[^"]+"\s+OFF\)',
        )
        self.assertIn("if(LEONIDA_XMA2_DECODER OR LEONIDA_REQUIRE_XMA2_DECODER)", text)
        opt_pos = text.index("if(LEONIDA_XMA2_DECODER OR LEONIDA_REQUIRE_XMA2_DECODER)")
        cargo_pos = text.index("find_program(LEONIDA_CARGO cargo)")
        exec_pos = text.index("COMMAND ${LEONIDA_CARGO} build")
        self.assertLess(opt_pos, cargo_pos)
        self.assertLess(cargo_pos, exec_pos)


if __name__ == "__main__":
    unittest.main()
