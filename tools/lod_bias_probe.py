"""Measure whether the Vulkan driver applies VkSamplerCreateInfo::mipLodBias, for rdna2spirv.py --explicit-lod-bias.

    python -I tools/lod_bias_probe.py --build <folder with vk_sampler_test> [--json]

The answer is measured, not looked up: MoltenVK 1.4.2 drops the sampler's bias on an Apple M3 (ps_ssharp fails on
main) but applies it on an Apple M5 Pro (ps_ssharp passes on main), so a driver ID can't decide. The probe
translates runtime/test/shaders/vs_ids and ps_ssharp *without* the flag into a temporary folder (build.py with
LEONIDA_EXPLICIT_LOD_BIAS=0), then draws them with vk_sampler_test three times: no bias, LOD_BIAS +1 and -1 (S# words
from check.py's ssharp_words). ps_ssharp's mrt0 is (lz.r, lz.g, implicit.r, implicit.g): the explicit sample at LOD
0 and the derivative one at LOD 1, on a two-level texture.

* no-bias draw wrong                      -> None (the probe can't judge this device)
* +1 and -1 both as check.py expects      -> False: the driver applies the bias, the flag would count it twice
* +1 and -1 both read the no-bias levels  -> True: the driver drops it, translate with --explicit-lod-bias
* anything else                           -> None

The last line printed is {"explicit_lod_bias": true|false|null, "reason": "..."}; the exit status is 0 either way
(2 on a usage error). It honours the environment the tests run under (VK_DRIVER_FILES, LEONIDA_VULKAN_LOADER, ...).
Standard library only; needs llvm-mc and the SPIR-V tools build.py needs, no game files.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SHADERS = os.path.join(ROOT, "runtime", "test", "shaders")
EXE = ".exe" if os.name == "nt" else ""
sys.path.insert(0, SHADERS)
import check  # noqa: E402  (ssharp_words, sampler_mismatch, the SAMPLER_SIZE readback layout)


def probe(build: str, timeout: int = 120):
    """(True | False | None, reason)."""
    tool = os.path.join(build, "vk_sampler_test" + EXE)
    if not os.path.isfile(tool):
        return None, f"no vk_sampler_test{EXE} in {build}"
    with tempfile.TemporaryDirectory(prefix="leonida-lod-bias-") as tmp:
        env = dict(os.environ, LEONIDA_EXPLICIT_LOD_BIAS="0")
        try:
            r = subprocess.run([sys.executable, "-I", os.path.join(SHADERS, "build.py"), tmp, "vs_ids", "ps_ssharp"],
                               capture_output=True, text=True, env=env, timeout=timeout)
        except (OSError, subprocess.SubprocessError) as e:
            return None, f"build.py: {e}"
        shaders = [os.path.join(tmp, "spirv", n + ".spv") for n in ("vs_ids", "ps_ssharp")]
        if r.returncode or not all(os.path.isfile(s) for s in shaders):
            return None, "build.py could not translate vs_ids and ps_ssharp: " + (r.stdout + r.stderr).strip()[-300:]
        view = (0, 1)

        def draw(bias):
            words = check.ssharp_words(1, bias=bias)
            raw = Path(tmp) / f"bias{bias:+}.raw"
            try:
                d = subprocess.run([tool, *shaders, str(raw), *(f"{w:x}" for w in words), *(str(v) for v in view)],
                                   capture_output=True, text=True, timeout=timeout)
            except (OSError, subprocess.SubprocessError) as e:
                return None, f"vk_sampler_test: {e}"
            if d.returncode or not raw.is_file():
                return None, f"vk_sampler_test exited {d.returncode}: " + (d.stdout + d.stderr).strip()[-200:]
            data = raw.read_bytes()
            return check.sampler_mismatch(data, words, view), data

        base = draw(0)
        if base[0] is None:
            return None, base[1]
        if base[0]:
            return None, f"the no-bias draw is already wrong ({base[0]})"
        results = {}
        for bias in (1, -1):
            problem, data = draw(bias)
            if problem is None:
                return None, data
            results[bias] = (problem, data)
        if not any(p for p, _ in results.values()):
            return False, "the sampler applies LOD_BIAS +1 and -1"
        if all(p and data == base[1] for p, data in results.values()):
            return True, "the sampler drops LOD_BIAS: +1 and -1 read the no-bias levels"
        return None, "; ".join(f"bias {b:+}: {p or 'ok'}" for b, (p, _) in results.items())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--build", required=True, help="the folder with vk_sampler_test (the GPU test programs)")
    a = ap.parse_args(argv)
    answer, reason = probe(a.build)
    print(json.dumps({"explicit_lod_bias": answer, "reason": reason}))
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    sys.exit(main())
