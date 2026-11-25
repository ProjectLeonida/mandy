"""Regression test for TEST-PIPELINE-LIBRARY-HOST-RELEASE.

Verifies that runtime/test/d3d12_pipeline_library_host_test.cpp guards COM Release()
with SUCCEEDED(hr) to prevent fail-fast (0xc0000409) crashes on failed steps
where the output pointer was left uninitialized or written with an invalid sentinel.
"""
import re
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parent.parent
HOST_TEST_CPP = ROOT / "runtime" / "test" / "d3d12_pipeline_library_host_test.cpp"


class PipelineLibraryHostReleaseTest(unittest.TestCase):
    def test_release_guarded_by_succeeded(self):
        self.assertTrue(HOST_TEST_CPP.is_file(), f"{HOST_TEST_CPP} must exist")
        content = HOST_TEST_CPP.read_text(encoding="utf-8")

        # Find one_step definition
        m = re.search(r"void one_step\([^)]*\)\s*\{(.*?)\}", content, re.DOTALL)
        self.assertIsNotNone(m, "one_step function must be defined in d3d12_pipeline_library_host_test.cpp")
        body = m.group(1)

        # Ensure Release() is guarded by SUCCEEDED(hr)
        self.assertIn("SUCCEEDED(hr)", body, "one_step must guard Release() with SUCCEEDED(hr)")
        self.assertNotIn("hr != HRESULT(kFaulted)) static_cast<IUnknown*>(c.out)->Release()", body,
                         "Unchecked Release() on failed HRESULT causes 0xc0000409 fail-fast")


if __name__ == "__main__":
    unittest.main()
