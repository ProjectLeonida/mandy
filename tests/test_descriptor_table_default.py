"""The runtime's descriptor-table layout with no shader library loaded is the translator's default.

`runtime/shims/d3d12_vulkan_device.cpp` takes the layout from the loaded library's vote. With no library (the GPU
tests, the tools) it uses `kNoLibraryDescriptorTableLayout` (`runtime/src/shader_library.h`). The modules those
paths create pipelines from are translated with `tools/rdna2spirv.py`'s default `--descriptor-table`, so the two
must name the same layout: when the runtime said plain and the translator tagged, `cs_texture_uav.spv` read an
all-zero kind word and its stores went to the fallback image.

Reads both files as text; needs no build, GPU or game files.
"""

import os
import re
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read(*parts):
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as f:
        return f.read()


def runtime_no_library_layout(header):
    m = re.search(r"kNoLibraryDescriptorTableLayout\s*=\s*DescriptorTableLayout::(\w+)\s*;", header)
    return m.group(1) if m else None


def translator_defaults(source):
    constant = re.search(r'^DESCRIPTOR_TABLE\s*=\s*"(\w+)"', source, re.M)
    option = re.search(r'"--descriptor-table"\s*,\s*choices=DESCRIPTOR_TABLES\s*,\s*default=("(\w+)"|DESCRIPTOR_TABLE)',
                       source)
    constant = constant.group(1) if constant else None
    if not option:
        return constant, None
    return constant, option.group(2) if option.group(2) else constant


class DescriptorTableDefault(unittest.TestCase):
    def test_runtime_matches_translator(self):
        runtime = runtime_no_library_layout(read("runtime", "src", "shader_library.h"))
        constant, option = translator_defaults(read("tools", "rdna2spirv.py"))
        self.assertIsNotNone(runtime, "kNoLibraryDescriptorTableLayout not found in runtime/src/shader_library.h")
        self.assertIsNotNone(constant, "DESCRIPTOR_TABLE not found in tools/rdna2spirv.py")
        self.assertIsNotNone(option, "--descriptor-table's default not found in tools/rdna2spirv.py")
        self.assertEqual(constant, option, "rdna2spirv.py's DESCRIPTOR_TABLE and --descriptor-table default differ")
        self.assertEqual(runtime, constant,
                         "the runtime's no-library descriptor table must be the translator's default layout")

    def test_parsers(self):
        self.assertEqual(runtime_no_library_layout(
            "inline constexpr DescriptorTableLayout kNoLibraryDescriptorTableLayout = DescriptorTableLayout::plain;"),
            "plain")
        self.assertIsNone(runtime_no_library_layout("DescriptorTableLayout descriptor_table = plain;"))
        self.assertEqual(translator_defaults('DESCRIPTOR_TABLE = "plain"\n'
                                             'ap.add_argument("--descriptor-table", choices=DESCRIPTOR_TABLES, '
                                             'default=DESCRIPTOR_TABLE)'), ("plain", "plain"))
        self.assertEqual(translator_defaults('DESCRIPTOR_TABLE = "tagged"\n'
                                             'ap.add_argument("--descriptor-table", choices=DESCRIPTOR_TABLES, '
                                             'default="plain")'), ("tagged", "plain"))


if __name__ == "__main__":
    unittest.main()
