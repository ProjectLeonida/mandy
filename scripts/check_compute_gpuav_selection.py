"""Check compute-module debug-name selection with an installed Khronos GPU-AV layer.

Uses only d3d12_compute_test's synthetic shaders. This is an opt-in integration
check: it requires a Vulkan device and a layer supporting selective GPU-AV and
instrumented-shader dumps. It is not part of the default CTest suite.
"""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile


def run_case(executable, directory, selector):
    directory.mkdir()
    cache = directory / "cache"
    cache.mkdir()
    env = os.environ.copy()
    env.update({
        "LEONIDA_VK_VALIDATION": "1",
        "LEONIDA_PIPELINE_CACHE_DIR": str(cache),
        "VK_LAYER_GPUAV_ENABLE": "1",
        "VK_LAYER_GPUAV_SHADER_INSTRUMENTATION": "1",
        "VK_LAYER_GPUAV_SELECT_INSTRUMENTED_SHADERS": "1",
        "VK_LAYER_GPUAV_SHADERS_TO_INSTRUMENT": selector,
        "VK_LAYER_GPUAV_DEBUG_DUMP_INSTRUMENTED_SHADERS": "1",
    })
    result = subprocess.run([str(executable)], cwd=directory, env=env,
                            capture_output=True, timeout=120)
    if result.returncode == 77:
        return None
    if result.returncode:
        raise RuntimeError(f"compute fixture failed ({result.returncode}):\n"
                           + result.stderr.decode("utf-8", errors="replace"))
    dumps = list(directory.glob("dump_*_after.spv"))
    for after in dumps:
        before = after.with_name(after.name.replace("_after.spv", "_before.spv"))
        original, instrumented = before.read_bytes(), after.read_bytes()
        if original[:4] != b"\x03\x02\x23\x07" or instrumented[:4] != b"\x03\x02\x23\x07":
            raise RuntimeError("layer dump is not SPIR-V")
        if original == instrumented:
            raise RuntimeError("selected module was not instrumented")
    return len(dumps)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("compute_test", type=Path, help="path to d3d12_compute_test")
    args = parser.parse_args()
    executable = args.compute_test.resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix="leonida_gpuav_selection_") as scratch:
        root = Path(scratch)
        selected = run_case(executable, root / "selected", "cs_user_data")
        if selected is None:
            print("SKIP: no usable Vulkan device")
            return 77
        if not selected:
            raise RuntimeError("named selection produced no instrumented module; "
                               "check module naming and the installed validation layer")
        unmatched = run_case(executable, root / "unmatched", "__leonida_no_matching_shader__")
        if unmatched is None:
            raise RuntimeError("Vulkan device became unavailable during the check")
        if unmatched:
            raise RuntimeError("unmatched selection instrumented a shader")
        print(f"PASS: selected {selected} module(s), unmatched 0; compute readback checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
