"""What this machine can run of the GPU tests right now, and the command for it.

    python -I scripts/gpu_test_setup.py [--build DIR] [--llvm-mc CMD] [--json]

docs/testing.md lists everything the GPU tests can use: llvm-mc with the AMDGPU target, SPIRV-Tools, the Vulkan
test programs of runtime/, a GPU. Not all of it is needed for a first result, and which part is missing shows
only when a run stops. This looks once and says what is there and how far it gets:

- `semantic`: the translator suite. Needs Python alone.
- `subset`: `run_gpu_tests.py --disasm-dir tests/semantic/fixtures/disasm`, the fixtures that are kept as
  listings, assembled by tools/spirv_asm.py when spirv-as is missing. Needs the three test programs and a GPU,
  but neither llvm-mc nor SPIRV-Tools.
- `full`: every fixture of runtime/test/shaders. Needs llvm-mc with the AMDGPU target as well, and the test
  programs the texture, sampler and bindless fixtures start.

For each level it prints the command, or what is missing for it. It also names the GPUs `vulkaninfo` reports with
their subgroup range: a device whose largest subgroup is below 64 runs the wave64 fixtures only with
`LEONIDA_W64_EMULATE=1`, so a record for it needs both runs.

It starts `llvm-mc --version`, `spirv-as --version`, `spirv-val --version` and `vulkaninfo`, reads their output and
looks for files; it builds nothing, runs no test and writes nothing. Exit 0 when it ran, 2 on a usage error.
Standard library only; no game files.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import mc_command  # noqa: E402

PROGRAMS = ("vk_compute_test", "vk_store_test", "vk_harness")     # REQUIRED_TEST_PROGRAMS of run_gpu_tests.py
# What the full run starts as well: without one of them its fixtures fail with "could not start".
MORE_PROGRAMS = ("vk_texture_test", "vk_bindless_test", "vk_sampler_test", "d3d12_texture_uav_test")
CONFIGS = ("", "RelWithDebInfo", "Release")
BUILDS = ("runtime/build", "runtime/build/linux", "runtime/build/windows")
FIXTURES = "tests/semantic/fixtures/disasm"


def output_of(cmd, timeout=60):
    """What a command prints, or None when it cannot be started or fails."""
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None
    text = (r.stdout + r.stderr).decode("utf-8", "replace")
    return text if r.returncode == 0 else None


def llvm_mc(raw):
    """{"command", "version", "amdgpu"} for the llvm-mc that LLVM_MC, --llvm-mc or PATH names; None without one."""
    try:
        cmd = mc_command.mc_command(raw)
    except (ValueError, OSError):
        return None
    text = output_of([*cmd, "--version"])
    if text is None:
        return None
    version = re.search(r"LLVM version (\S+)", text)
    return {"command": " ".join(cmd), "version": version.group(1) if version else None,
            "amdgpu": "amdgcn" in text.lower()}


def tool(name):
    """The path of a SPIRV-Tools program that starts, or None."""
    path = shutil.which(name)
    return path if path and output_of([path, "--version"]) is not None else None


def gpus(text):
    """[{"name", "driver", "subgroup": [min, max] or None}] from `vulkaninfo` text, one per device."""
    found = []
    for block in re.split(r"(?m)^GPU\d+:\s*$", text or "")[1:]:
        name = re.search(r"(?m)^\s*deviceName\s*=\s*(.+?)\s*$", block)
        if not name:
            continue
        low = re.search(r"(?m)^\s*minSubgroupSize\s*=\s*(\d+)", block)
        high = re.search(r"(?m)^\s*maxSubgroupSize\s*=\s*(\d+)", block)
        driver = re.search(r"(?m)^\s*driverInfo\s*=\s*(.+?)\s*$", block)
        entry = next((g for g in found if g["name"] == name.group(1)), None)
        if entry is None:
            entry = {"name": name.group(1), "driver": None, "subgroup": None}
            found.append(entry)
        if driver:
            entry["driver"] = driver.group(1)
        if low and high:
            entry["subgroup"] = [int(low.group(1)), int(high.group(1))]
    return found


def test_programs(root, builds):
    """The first folder that holds all three test programs, as run_gpu_tests.py looks for them."""
    ext = ".exe" if os.name == "nt" else ""
    for build in builds:
        for config in CONFIGS:
            folder = os.path.join(root, build, config) if not os.path.isabs(build) else os.path.join(build, config)
            if all(os.path.isfile(os.path.join(folder, name + ext)) for name in PROGRAMS):
                return os.path.normpath(os.path.join(build, config))
    return None


def survey(root=ROOT, build=None, raw_llvm_mc=None):
    try:
        listings = sum(1 for name in os.listdir(os.path.join(root, FIXTURES)) if name.endswith(".s"))
    except OSError:
        listings = 0
    state = {
        "llvm_mc": llvm_mc(raw_llvm_mc),
        "spirv_as": tool("spirv-as"), "spirv_val": tool("spirv-val"),
        "gpus": gpus(output_of(["vulkaninfo"])),
        "build": test_programs(root, [build] if build else BUILDS),
        "llvm_mc_option": raw_llvm_mc,
        "more_missing": [],
        "listings": listings,
    }
    if state["build"]:
        folder = state["build"] if os.path.isabs(state["build"]) else os.path.join(root, state["build"])
        ext = ".exe" if os.name == "nt" else ""
        state["more_missing"] = [n for n in MORE_PROGRAMS if not os.path.isfile(os.path.join(folder, n + ext))]
    state["levels"] = levels(state, build)
    return state


def quoted(text):
    """`text` as one argument of a shell command: in double quotes when it has a space, which PowerShell, cmd
    and POSIX shells all take."""
    return f'"{text}"' if " " in text else text


def levels(state, build=None):
    """[{"level", "ready", "command", "missing": [..]}] for semantic, subset and full."""
    where = (state["build"] or build or BUILDS[0]).replace(os.sep, "/")
    no_programs = (f"the test programs {', '.join(PROGRAMS)} in {where}: cmake -S runtime -B runtime/build, then "
                   "cmake --build runtime/build (CMake builds them only when it finds the Vulkan headers)")
    no_gpu = "a Vulkan device: vulkaninfo reports none"
    common = ([] if state["build"] else [no_programs]) + ([] if state["gpus"] else [no_gpu])
    mc = state["llvm_mc"]
    if mc is None:
        no_mc = ["llvm-mc: " + mc_command.install_hint()]
    elif not mc["amdgpu"]:
        no_mc = [f"an llvm-mc with the AMDGPU target: {mc['command']} has none (the official Windows build lacks it)"]
    else:
        no_mc = []
    more = ([f"the test programs {', '.join(state['more_missing'])} in {where}: cmake --build without --target "
             "builds them (the d3d12 one on Windows only)"] if state.get("more_missing") else [])
    run = f"python -I scripts/run_gpu_tests.py --build {quoted(where)}"
    return [
        {"level": "semantic", "ready": True, "missing": [],
         "command": "python -I scripts/run_semantic_tests.py tests/semantic"},
        {"level": "subset", "ready": not common and state["listings"] > 0,
         "missing": common + ([] if state["listings"] else [f"the listings in {FIXTURES}"]),
         "command": f"{run} --disasm-dir {FIXTURES}"},
        {"level": "full", "ready": not common and not no_mc and not more, "missing": common + more + no_mc,
         # An llvm-mc named on this command line is not in LLVM_MC: the runner has to be told as well.
         "command": run + (f" --llvm-mc {quoted(state['llvm_mc_option'])}" if state.get("llvm_mc_option") else "")},
    ]


def print_report(state):
    mc = state["llvm_mc"]
    print("llvm-mc:    " + ("not found" if mc is None else
                            f"{mc['command']} ({mc['version'] or 'version unknown'}, "
                            + ("AMDGPU target" if mc["amdgpu"] else "no AMDGPU target") + ")"))
    for key, name in (("spirv_as", "spirv-as"), ("spirv_val", "spirv-val")):
        fallback = "tools/spirv_asm.py assembles instead" if key == "spirv_as" else "validation is skipped"
        print(f"{name + ':':<12}" + (state[key] or f"not found ({fallback})"))
    print("programs:   " + (state["build"].replace(os.sep, "/") if state["build"] else "not built"))
    print(f"listings:   {state['listings']} in {FIXTURES}")
    if not state["gpus"]:
        print("GPU:        none reported by vulkaninfo")
    for gpu in state["gpus"]:
        sub = gpu["subgroup"]
        note = ("" if sub is None else f", subgroup {sub[0]}..{sub[1]}"
                + ("" if sub[1] >= 64 else ": wave64 fixtures need LEONIDA_W64_EMULATE=1"))
        print(f"GPU:        {gpu['name']}" + (f" ({gpu['driver']})" if gpu["driver"] else "") + note)
    print()
    for level in state["levels"]:
        print(f"{level['level']}: " + ("ready" if level["ready"] else "not yet"))
        if level["ready"]:
            print(f"  {level['command']}")
        for item in level["missing"]:
            print(f"  needs {item}")
    if any(g["subgroup"] and g["subgroup"][1] < 64 for g in state["gpus"]) and state["levels"][1]["ready"]:
        print("\nOn a device without 64-wide subgroups, run the GPU command a second time with LEONIDA_W64_EMULATE=1 "
              "set;\nselect a device with LEONIDA_VK_DEVICE=<part of its name>.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--build", help="CMake build folder of runtime/ (default: runtime/build and its usual variants)")
    ap.add_argument("--llvm-mc", help="llvm-mc path or whole command (default: LLVM_MC, then PATH)")
    ap.add_argument("--json", action="store_true", help="one JSON object instead of text")
    args = ap.parse_args(argv)
    state = survey(ROOT, build=args.build, raw_llvm_mc=args.llvm_mc)
    if args.json:
        print(json.dumps(state, indent=2))
    else:
        print_report(state)
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
