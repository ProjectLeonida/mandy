"""Every CMake rule that runs runtime/test/shaders/build.py hands it the folder of the spirv-as CMake found.

build.py looks for spirv-as and spirv-val on PATH, and puts LEONIDA_SPIRV_TOOLS in front of it when that is set. The
shader rules pass "LEONIDA_SPIRV_TOOLS=<folder of the spirv-as find_program found>", so a build works when the tools
are found through VULKAN_SDK or -DLEONIDA_SPIRV_AS without being on PATH. The d3d12_subgroup_test rule did not, and
was not gated on the tools either: with llvm-mc and glslangValidator found and spirv-as off PATH, its four shader
rules stopped the whole build with "ERROR: spirv-as not found" (MSVC + Ninja, SPIRV-Tools from a conda environment).

No compiler and no CMake run are needed: this reads runtime/CMakeLists.txt.
"""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CMAKE_LISTS = ROOT / "runtime" / "CMakeLists.txt"
TOOLS_ENV = '"LEONIDA_SPIRV_TOOLS=${LEONIDA_SPIRV_TOOLS_NATIVE}"'


def build_py_rules(text):
    """The add_custom_command(...) blocks that run test/shaders/build.py, as tests/test_cmake_translator_deps.py finds them."""
    return [m.group(0) for m in re.finditer(r"add_custom_command\((?:[^()]|\$<[^>]*>|\([^()]*\))*\)", text)
            if "test/shaders/build.py" in m.group(0)]


class SpirvToolsEnvTest(unittest.TestCase):
    def setUp(self):
        self.text = CMAKE_LISTS.read_text(encoding="utf-8")

    def test_every_build_py_rule_names_the_spirv_tools_folder(self):
        rules = build_py_rules(self.text)
        self.assertGreaterEqual(len(rules), 8)
        for rule in rules:
            comment = re.search(r'COMMENT "([^"]*)"', rule)
            with self.subTest(rule=comment.group(1) if comment else rule[:80]):
                self.assertIn(TOOLS_ENV, rule)

    def test_the_subgroup_test_is_built_only_when_the_spirv_tools_were_found(self):
        # The folder comes from LEONIDA_SPIRV_AS, so the block that uses it has to require the tools.
        start = self.text.index("set(LEONIDA_SG_DIR ")
        gate = self.text.rindex("\n    if(", 0, start)
        condition = self.text[gate:self.text.index("\n", gate + 1)]
        for name in ("LEONIDA_LLVM_MC", "LEONIDA_SPIRV_AS", "LEONIDA_SPIRV_VAL", "Vulkan_GLSLANG_VALIDATOR_EXECUTABLE"):
            self.assertIn(name, condition)
        block = self.text[gate:start]
        self.assertIn("get_filename_component(LEONIDA_SPIRV_TOOLS_DIR ${LEONIDA_SPIRV_AS} DIRECTORY)", block)
        self.assertIn('file(TO_NATIVE_PATH "${LEONIDA_SPIRV_TOOLS_DIR}" LEONIDA_SPIRV_TOOLS_NATIVE)', block)


if __name__ == "__main__":
    unittest.main()
