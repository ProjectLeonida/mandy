"""The tools/native Cargo workspace: every loader finds its library there and honours LEONIDA_NATIVE=0.

Standard library only. The checks that need the built libraries skip until
`cargo build --release --workspace --manifest-path tools/native/Cargo.toml` has run.
"""
import os
import subprocess
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
RELEASE = os.path.join(TOOLS, "native", "target", "release")
CRATES = ("anim_clips_native", "bc7_native", "bvh_native", "glb_native", "meshlet_native", "rdna2_decode_native",
          "spirv_asm_native", "spirv_opt_native")

# Prints, for each loader, whether it found a usable native path.
PROBE = r"""
import sys
sys.path.insert(0, sys.argv[1])
import _anim_clips_native, bvh_native, dds_png, glb_native, meshlet_native, rdna2_decode, rdna2_native
import spirv_asm, spirv_opt
from unittest import mock


def asks_bvh_library():
    # try_rehydrate reads the switches before it loads the library.
    with mock.patch.object(bvh_native, "_load", return_value=None) as load:
        bvh_native.try_rehydrate(b"", [], [], None, object)
    return load.called


found = {
    "anim_clips_native": _anim_clips_native.native() is not None,
    "bc7_native": bool(dds_png._native_bc7()),
    "bvh_native": bvh_native._load() is not None and asks_bvh_library(),
    "glb_native": glb_native._enabled() and glb_native._load() is not None,
    "meshlet_native": meshlet_native._library(b"") is not None,
    "rdna2_decode_native": rdna2_native.load(vars(rdna2_decode)) is not None,
    "spirv_asm_native": spirv_asm._native_assemble("", "vulkan1.2") is not None,
    "spirv_opt_native": spirv_opt._native_optimize("", 4) is not None,
}
print(" ".join(sorted(name for name, ok in found.items() if ok)))
"""


def library_name(crate):
    return {"win32": f"{crate}.dll", "darwin": f"lib{crate}.dylib"}.get(sys.platform, f"lib{crate}.so")


def probe(**env):
    full = {k: v for k, v in os.environ.items() if not k.startswith("LEONIDA_")}
    full.update(env)
    out = subprocess.run([sys.executable, "-I", "-c", PROBE, TOOLS], capture_output=True, text=True, env=full,
                         check=True)
    return set(out.stdout.split())


class Manifest(unittest.TestCase):
    def test_every_crate_is_a_member_with_no_dependency(self):
        with open(os.path.join(TOOLS, "native", "Cargo.toml"), encoding="utf-8") as f:
            manifest = f.read()
        for crate in CRATES:
            with self.subTest(crate=crate):
                self.assertIn(f'"../{crate}"', manifest)
                with open(os.path.join(TOOLS, crate, "Cargo.toml"), encoding="utf-8") as f:
                    member = f.read()
                self.assertIn('workspace = "../native"', member)
                self.assertNotIn("[profile", member)  # profiles live in the workspace root only
                deps = member.split("[dependencies]", 1)[1] if "[dependencies]" in member else ""
                self.assertEqual(deps.split("[", 1)[0].strip(), "")
                self.assertFalse(os.path.exists(os.path.join(TOOLS, crate, "Cargo.lock")))


@unittest.skipUnless(all(os.path.isfile(os.path.join(RELEASE, library_name(c))) for c in CRATES),
                     "tools/native is not built")
class Loaders(unittest.TestCase):
    def test_every_loader_finds_the_workspace_build(self):
        self.assertEqual(probe(), set(CRATES))

    def test_global_switch_turns_every_loader_off(self):
        self.assertEqual(probe(LEONIDA_NATIVE="0"), set())

    def test_per_module_switch_turns_one_loader_off(self):
        self.assertEqual(probe(LEONIDA_SPIRV_OPT_NATIVE="0"), set(CRATES) - {"spirv_opt_native"})


if __name__ == "__main__":
    unittest.main()
