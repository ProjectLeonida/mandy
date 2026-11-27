"""-DLEONIDA_VULKAN_HEADERS_FETCH=ON (or a FETCHCONTENT_SOURCE_DIR_VULKANHEADERS copy) must win over other headers.

runtime/CMakeLists.txt used to run find_package(Vulkan) and find_path(Vulkan_INCLUDE_DIR) before the fetch, so any
Vulkan headers on the machine (an SDK, conda-forge's vulkan-headers, /usr/include) defined Vulkan::Headers first and
the pinned Khronos Vulkan-Headers were never used, even with the flag on. That changes GPU results: against
conda-forge's Vulkan 1.3 headers run_gpu_tests.py on a Radeon 780M skipped ps_texoff, ps_texoff_forms and
ps_texoff_implicit (225/248 passed instead of 228/248 with the pin). The configure now also names its headers in a
"Vulkan headers:" line (runtime/cmake/vulkan_headers_report.cmake) that a GPU record can quote.

FetchContent_MakeAvailable(VulkanHeaders) can also hand back other headers without populating the pin: with
-DFETCHCONTENT_TRY_FIND_PACKAGE_MODE=ALWAYS it tried find_package(VulkanHeaders) first, and an installed
VulkanHeadersConfig.cmake with VK_HEADER_VERSION 111 was used and named as the pin. The declaration now carries
OVERRIDE_FIND_PACKAGE, and the configure fails when a dependency provider or another declaration still supplies
other headers, or when the populated copy is not the pinned version.

The configure cases run CMake on runtime/ with hand-written stand-ins for an installed SDK (VULKAN_SDK), an installed
Vulkan-Headers package (VulkanHeaders_DIR) and the pinned release (FETCHCONTENT_SOURCE_DIR_VULKANHEADERS, or a copy
already in FETCHCONTENT_BASE_DIR with FETCHCONTENT_FULLY_DISCONNECTED=ON), so nothing is downloaded and no real Vulkan
headers are needed. They need cmake and a C++ compiler and are skipped without them; the cmake -P cases need cmake
only.
"""
import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNTIME = ROOT / "runtime"
MODULE = RUNTIME / "cmake" / "vulkan_headers_report.cmake"
CMAKE_LISTS = RUNTIME / "CMakeLists.txt"
CMAKE = shutil.which("cmake")

SDK_VERSION = 231   # conda-forge's vulkan-headers next to llvm-tools
PACKAGE_VERSION = 111   # an installed Vulkan-Headers package found through VulkanHeaders_DIR
# The VK_HEADER_VERSION of the pin (363 for vulkan-sdk-1.4.363.0), which a stand-in for the pin must have.
_PIN_VERSION = re.search(r"^set\(LEONIDA_VULKAN_HEADERS_PIN_VERSION ([0-9]+)\)$",
                         CMAKE_LISTS.read_text(encoding="utf-8"), re.M)
PIN_VERSION = int(_PIN_VERSION.group(1)) if _PIN_VERSION else None

# Runs at the end of the top-level directory (cmake_language DEFER) and writes the include directories of the
# Vulkan::Headers target the tools link, which is what the compiler gets, independent of the "Vulkan headers:" line.
HOOK = """\
function(leonida_test_dump_vulkan_headers)
  if(TARGET Vulkan::Headers)
    get_target_property(dirs Vulkan::Headers INTERFACE_INCLUDE_DIRECTORIES)
  else()
    set(dirs NONE)
  endif()
  file(WRITE "${CMAKE_BINARY_DIR}/leonida_test_vulkan_headers.txt" "${dirs}")
endfunction()
cmake_language(DEFER CALL leonida_test_dump_vulkan_headers)
"""

# A stand-in for the Khronos Vulkan-Headers project: the same target names and include directory layout.
FAKE_PROJECT = """\
cmake_minimum_required(VERSION 3.24)
project(Vulkan-Headers LANGUAGES NONE)
add_library(Vulkan-Headers INTERFACE)
add_library(Vulkan::Headers ALIAS Vulkan-Headers)
target_include_directories(Vulkan-Headers INTERFACE "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>")
"""

# A stand-in for an installed Khronos Vulkan-Headers package: the same config file and target names.
FAKE_PACKAGE = """\
if(NOT TARGET Vulkan::Headers)
  add_library(Vulkan::Headers INTERFACE IMPORTED)
  set_target_properties(Vulkan::Headers PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_LIST_DIR}/include")
endif()
"""

# A dependency provider (CMAKE_PROJECT_TOP_LEVEL_INCLUDES) that answers FetchContent_MakeAvailable(VulkanHeaders) with
# the installed package's headers, as a package manager's provider would.
PROVIDER = """\
function(leonida_test_provider method name)
  if(name STREQUAL "VulkanHeaders")
    if(NOT TARGET Vulkan::Headers)
      add_library(Vulkan::Headers INTERFACE IMPORTED GLOBAL)
      set_target_properties(Vulkan::Headers PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "@INCLUDE@")
    endif()
    FetchContent_SetPopulated(${name})
  endif()
endfunction()
cmake_language(SET_DEPENDENCY_PROVIDER leonida_test_provider SUPPORTED_METHODS FETCHCONTENT_MAKEAVAILABLE_SERIAL)
"""

CHECK_FAILED = "Something other than that download answered"


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def write_headers(include, version):
    """vulkan/vulkan.h and vulkan/vulkan_core.h under include; version None writes no VK_HEADER_VERSION line."""
    line = f"#define VK_HEADER_VERSION {version}\n" if version is not None else ""
    write(include / "vulkan" / "vulkan_core.h", f"#ifndef VULKAN_CORE_H_\n#define VULKAN_CORE_H_ 1\n{line}#endif\n")
    write(include / "vulkan" / "vulkan.h", '#include "vulkan_core.h"\n')


def same_dir(a, b):
    return os.path.normcase(os.path.realpath(a)) == os.path.normcase(os.path.realpath(b))


def included(text):
    """The directories of an INTERFACE_INCLUDE_DIRECTORIES value, with $<BUILD_INTERFACE:...> unwrapped."""
    dirs = []
    for d in text.split(";"):
        m = re.fullmatch(r"\$<BUILD_INTERFACE:(.*)>", d)
        if m:
            d = m.group(1)
        if d and not d.startswith("$<"):
            dirs.append(d)
    return dirs


def headers_line(output):
    lines = [ln for ln in output.splitlines() if "Vulkan headers:" in ln]
    return lines[-1] if lines else ""


def run_module(call):
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "driver.cmake"
        write(script, f'include("{MODULE.as_posix()}")\n'
                      'set(LEONIDA_VULKAN_HEADERS_PIN 0123456789abcdef0123456789abcdef01234567)\n'
                      f'set(LEONIDA_VULKAN_HEADERS_PIN_VERSION {PIN_VERSION})\n'
                      f'{call}\n')
        return subprocess.run([CMAKE, "-P", str(script)], capture_output=True, text=True, timeout=60)


def run_report(dirs, origin):
    return run_module(f'leonida_vulkan_headers_report("{dirs}" {origin})')


def run_check(dirs, source_dir, origin):
    return run_module(f'leonida_vulkan_headers_check("{dirs}" "{source_dir}" {origin})')


@unittest.skipUnless(CMAKE, "cmake not on PATH")
class ReportLineTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.include = Path(tmp.name) / "include"

    def test_found_headers_are_named_and_not_called_the_pin(self):
        write_headers(self.include, SDK_VERSION)
        r = run_report(self.include.as_posix(), "FOUND")
        self.assertEqual(r.returncode, 0, r.stderr)
        line = headers_line(r.stdout)
        self.assertIn(f"Vulkan headers: {self.include.as_posix()} (VK_HEADER_VERSION {SDK_VERSION})", line)
        self.assertIn("not the pin", line)
        self.assertIn("-DLEONIDA_VULKAN_HEADERS_FETCH=ON", line)

    def test_pin_names_the_commit(self):
        write_headers(self.include, PIN_VERSION)
        r = run_report(f"$<BUILD_INTERFACE:{self.include.as_posix()}>", "PIN")
        self.assertEqual(r.returncode, 0, r.stderr)
        line = headers_line(r.stdout)
        self.assertIn(f"Vulkan headers: {self.include.as_posix()} (VK_HEADER_VERSION {PIN_VERSION})", line)
        self.assertIn("the pin: Khronos Vulkan-Headers 0123456789abcdef0123456789abcdef01234567", line)

    def test_a_cached_pin_does_not_claim_a_hash_check(self):
        # FETCHCONTENT_FULLY_DISCONNECTED=ON: the copy in FETCHCONTENT_BASE_DIR is used and no archive is checked.
        write_headers(self.include, PIN_VERSION)
        r = run_report(f"$<BUILD_INTERFACE:{self.include.as_posix()}>", "PIN_CACHED")
        self.assertEqual(r.returncode, 0, r.stderr)
        line = headers_line(r.stdout)
        self.assertIn(f"(VK_HEADER_VERSION {PIN_VERSION}), the pinned version, cached source: Khronos Vulkan-Headers "
                      "0123456789abcdef0123456789abcdef01234567; archive not checked in this configure", line)
        self.assertNotIn("hash-checked", line)

    def test_source_dir_copy_is_not_vouched_for(self):
        write_headers(self.include, PIN_VERSION)
        r = run_report(self.include.as_posix(), "SOURCE_DIR")
        self.assertEqual(r.returncode, 0, r.stderr)
        line = headers_line(r.stdout)
        self.assertIn("FETCHCONTENT_SOURCE_DIR_VULKANHEADERS", line)
        self.assertIn("not checked against it", line)

    def test_header_without_a_version_says_unknown(self):
        write_headers(self.include, None)
        r = run_report(self.include.as_posix(), "FOUND")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("(VK_HEADER_VERSION unknown)", headers_line(r.stdout))


@unittest.skipUnless(CMAKE, "cmake not on PATH")
class CheckTest(unittest.TestCase):
    """leonida_vulkan_headers_check: Vulkan::Headers must be the copy FetchContent populated, and the pin's version."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        base = Path(tmp.name)
        self.source = base / "vulkan-headers"
        self.package = base / "package"
        write_headers(self.package / "include", PACKAGE_VERSION)

    def test_the_populated_copy_passes(self):
        write_headers(self.source / "include", PIN_VERSION)
        r = run_check(f"$<BUILD_INTERFACE:{(self.source / 'include').as_posix()}>", self.source.as_posix(), "PIN")
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_other_headers_fail(self):
        write_headers(self.source / "include", PIN_VERSION)
        r = run_check((self.package / "include").as_posix(), self.source.as_posix(), "PIN")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn(CHECK_FAILED, " ".join(r.stderr.split()))

    def test_nothing_populated_fails(self):
        # find_package() or a provider answered FetchContent_MakeAvailable: vulkanheaders_SOURCE_DIR is empty.
        r = run_check((self.package / "include").as_posix(), "", "PIN")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("it populated none", " ".join(r.stderr.split()))

    def test_another_version_is_not_the_pin(self):
        write_headers(self.source / "include", PIN_VERSION - 1)
        for origin in ("PIN", "PIN_CACHED"):
            with self.subTest(origin=origin):
                r = run_check((self.source / "include").as_posix(), self.source.as_posix(), origin)
                self.assertNotEqual(r.returncode, 0)
                self.assertIn(f"has VK_HEADER_VERSION {PIN_VERSION - 1}", " ".join(r.stderr.split()))

    def test_a_source_dir_copy_may_be_another_version(self):
        write_headers(self.source / "include", PIN_VERSION - 1)
        r = run_check((self.source / "include").as_posix(), self.source.as_posix(), "SOURCE_DIR")
        self.assertEqual(r.returncode, 0, r.stderr)


@unittest.skipUnless(CMAKE, "cmake not on PATH")
class ConfigureTest(unittest.TestCase):
    """Configures runtime/ with a fake installed SDK on VULKAN_SDK and a fake copy of the pinned release."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        base = Path(tmp.name).resolve()
        self.sdk_include = base / "sdk" / "include"
        write_headers(self.sdk_include, SDK_VERSION)
        self.pin_source = base / "vulkan-headers"
        self.pin_include = self.pin_source / "include"
        write_headers(self.pin_include, PIN_VERSION)
        write(self.pin_source / "CMakeLists.txt", FAKE_PROJECT)
        self.package = base / "package"
        self.package_include = self.package / "include"
        write_headers(self.package_include, PACKAGE_VERSION)
        write(self.package / "VulkanHeadersConfig.cmake", FAKE_PACKAGE)
        self.hook = base / "hook.cmake"
        write(self.hook, HOOK)
        self.base = base
        self.env = dict(os.environ, VULKAN_SDK=str(base / "sdk"))

    def run_configure(self, build, *args):
        cmd = [CMAKE, "-S", str(RUNTIME), "-B", str(build), f"-DCMAKE_PROJECT_gta6rt_INCLUDE={self.hook.as_posix()}",
               "-DBUILD_TESTING=OFF", "-DLEONIDA_BUILD_WINDOWS_RUNTIME=OFF", *args]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=self.env)
        out = r.stdout + r.stderr
        if r.returncode and "CMAKE_CXX_COMPILER" in out and not (build / "leonida_test_vulkan_headers.txt").exists():
            self.skipTest("no working C++ compiler for CMake")
        return r.returncode, out

    def configure(self, build, *args):
        returncode, out = self.run_configure(build, *args)
        self.assertEqual(returncode, 0, out[-4000:])
        dirs = included((build / "leonida_test_vulkan_headers.txt").read_text(encoding="utf-8"))
        return dirs, headers_line(out)

    def assert_uses(self, dirs, include):
        self.assertTrue(any(same_dir(d, include) for d in dirs), f"Vulkan::Headers includes {dirs}, not {include}")

    def assert_pin(self, dirs, line):
        self.assert_uses(dirs, self.pin_include)
        self.assertFalse(any(same_dir(d, self.sdk_include) for d in dirs), dirs)
        self.assertIn(f"(VK_HEADER_VERSION {PIN_VERSION})", line)
        self.assertIn("FETCHCONTENT_SOURCE_DIR_VULKANHEADERS", line)

    def test_fetch_wins_over_installed_headers(self):
        dirs, line = self.configure(self.base / "build", "-DLEONIDA_VULKAN_HEADERS_FETCH=ON",
                                    f"-DFETCHCONTENT_SOURCE_DIR_VULKANHEADERS={self.pin_source.as_posix()}")
        self.assert_pin(dirs, line)

    def seeded_base_dir(self, version):
        """A FETCHCONTENT_BASE_DIR that already holds a downloaded copy, for FETCHCONTENT_FULLY_DISCONNECTED=ON."""
        deps = self.base / "deps"
        write_headers(deps / "vulkanheaders-src" / "include", version)
        write(deps / "vulkanheaders-src" / "CMakeLists.txt", FAKE_PROJECT)
        return deps

    def test_always_mode_does_not_swap_in_an_installed_package(self):
        # FETCHCONTENT_TRY_FIND_PACKAGE_MODE=ALWAYS used to make FetchContent_MakeAvailable take the installed
        # VulkanHeadersConfig.cmake and populate nothing, and the line still said "the pin".
        deps = self.seeded_base_dir(PIN_VERSION)
        dirs, line = self.configure(self.base / "build", "-DLEONIDA_VULKAN_HEADERS_FETCH=ON",
                                    "-DFETCHCONTENT_TRY_FIND_PACKAGE_MODE=ALWAYS",
                                    f"-DVulkanHeaders_DIR={self.package.as_posix()}",
                                    f"-DFETCHCONTENT_BASE_DIR={deps.as_posix()}", "-DFETCHCONTENT_FULLY_DISCONNECTED=ON")
        self.assert_uses(dirs, deps / "vulkanheaders-src" / "include")
        self.assertFalse(any(same_dir(d, self.package_include) for d in dirs), dirs)
        # The preseeded copy is the pinned version, but no archive was checked in this configure.
        self.assertIn(f"(VK_HEADER_VERSION {PIN_VERSION}), the pinned version, cached source: Khronos Vulkan-Headers ",
                      line)
        self.assertIn("archive not checked in this configure (FETCHCONTENT_FULLY_DISCONNECTED=ON)", line)
        self.assertNotIn("hash-checked", line)

    def test_a_dependency_provider_cannot_stand_in_for_the_pin(self):
        provider = self.base / "provider.cmake"
        write(provider, PROVIDER.replace("@INCLUDE@", self.package_include.as_posix()))
        returncode, out = self.run_configure(self.base / "build", "-DLEONIDA_VULKAN_HEADERS_FETCH=ON",
                                             f"-DCMAKE_PROJECT_TOP_LEVEL_INCLUDES={provider.as_posix()}")
        self.assertNotEqual(returncode, 0, headers_line(out))
        self.assertIn(CHECK_FAILED, " ".join(out.split()))

    def test_a_stale_download_is_not_called_the_pin(self):
        # A build folder populated before the pin changed keeps that copy while FETCHCONTENT_FULLY_DISCONNECTED is ON.
        deps = self.seeded_base_dir(PIN_VERSION - 1)
        returncode, out = self.run_configure(self.base / "build", "-DLEONIDA_VULKAN_HEADERS_FETCH=ON",
                                             f"-DFETCHCONTENT_BASE_DIR={deps.as_posix()}",
                                             "-DFETCHCONTENT_FULLY_DISCONNECTED=ON")
        self.assertNotEqual(returncode, 0, headers_line(out))
        self.assertIn(f"has VK_HEADER_VERSION {PIN_VERSION - 1}", " ".join(out.split()))

    def test_turning_fetch_on_in_a_build_that_found_headers(self):
        build = self.base / "build"
        # The installed SDK is visible: without the fetch it is used (and Vulkan_INCLUDE_DIR is cached).
        dirs, line = self.configure(build)
        self.assert_uses(dirs, self.sdk_include)
        self.assertIn(f"(VK_HEADER_VERSION {SDK_VERSION}), found on this machine, not the pin", line)
        # The same build folder with the fetch turned on: the pin, not the cached lookup.
        dirs, line = self.configure(build, "-DLEONIDA_VULKAN_HEADERS_FETCH=ON",
                                    f"-DFETCHCONTENT_SOURCE_DIR_VULKANHEADERS={self.pin_source.as_posix()}")
        self.assert_pin(dirs, line)
        # And off again: back to the installed headers.
        dirs, line = self.configure(build, "-DLEONIDA_VULKAN_HEADERS_FETCH=OFF",
                                    "-UFETCHCONTENT_SOURCE_DIR_VULKANHEADERS")
        self.assert_uses(dirs, self.sdk_include)
        self.assertIn("not the pin", line)


class CMakeListsTest(unittest.TestCase):
    def setUp(self):
        self.text = CMAKE_LISTS.read_text(encoding="utf-8")

    def test_fetch_comes_before_find_package(self):
        fetch = re.search(r"^if\(LEONIDA_VULKAN_HEADERS_FETCH OR FETCHCONTENT_SOURCE_DIR_VULKANHEADERS\)$", self.text,
                          re.M)
        self.assertIsNotNone(fetch, "the top-level fetch branch is missing from runtime/CMakeLists.txt")
        self.assertLess(fetch.start(), self.text.index("\nfind_package(Vulkan QUIET)\n"))

    def fetch_branch(self):
        m = re.search(r"^if\(LEONIDA_VULKAN_HEADERS_FETCH OR FETCHCONTENT_SOURCE_DIR_VULKANHEADERS\)\n(.*?)^endif\(\)$",
                      self.text, re.S | re.M)
        self.assertIsNotNone(m, "the top-level fetch branch is missing from runtime/CMakeLists.txt")
        return m

    def test_headers_line_is_printed_with_the_tools(self):
        m = re.search(r"^if\(TARGET Vulkan::Headers\)\n(.*?)^  foreach\(tool ", self.text, re.S | re.M)
        self.assertIsNotNone(m, "if(TARGET Vulkan::Headers) ... foreach(tool ...) not found")
        self.assertIn("leonida_vulkan_headers_report(", m.group(1))
        self.assertLess(self.text.index("/cmake/vulkan_headers_report.cmake\")"), self.fetch_branch().start())

    def test_the_fetch_is_checked(self):
        branch = self.fetch_branch().group(1)
        declare = re.search(r"^  FetchContent_Declare\(VulkanHeaders\n(.*?)^  FetchContent_MakeAvailable\(VulkanHeaders\)$",
                            branch, re.S | re.M)
        self.assertIsNotNone(declare, "FetchContent_Declare(VulkanHeaders ...) not found in the fetch branch")
        code = "\n".join(ln for ln in declare.group(1).splitlines() if not ln.lstrip().startswith("#"))
        self.assertIn("OVERRIDE_FIND_PACKAGE", code)
        self.assertLess(branch.index("FetchContent_MakeAvailable(VulkanHeaders)"),
                        branch.index("leonida_vulkan_headers_check("))

    def test_pin_matches_the_ci_checkout(self):
        m = re.search(r"^set\(LEONIDA_VULKAN_HEADERS_PIN ([0-9a-f]{40})\)$", self.text, re.M)
        self.assertIsNotNone(m, "set(LEONIDA_VULKAN_HEADERS_PIN <commit>) not found")
        self.assertIn("Vulkan-Headers/archive/${LEONIDA_VULKAN_HEADERS_PIN}.tar.gz", self.text)
        ci = (ROOT / ".github" / "workflows" / "runtime.yml").read_text(encoding="utf-8")
        refs = re.findall(r"^\s*ref: ([0-9a-f]{40}) # vulkan-sdk-[0-9]+\.[0-9]+\.([0-9]+)\.[0-9]+$", ci, re.M)
        self.assertTrue(refs, "no Vulkan-Headers checkout in runtime.yml")
        self.assertEqual({commit for commit, _ in refs}, {m.group(1)})
        # The SDK version's third number is the VK_HEADER_VERSION the configure checks the download against.
        self.assertIsNotNone(PIN_VERSION, "set(LEONIDA_VULKAN_HEADERS_PIN_VERSION <n>) not found")
        self.assertEqual({int(version) for _, version in refs}, {PIN_VERSION})


if __name__ == "__main__":
    unittest.main()
