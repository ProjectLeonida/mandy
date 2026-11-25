"""vk_pipeline_check --merge, --names-file and --done-file (PIPELINES-STAGE-SHARDS-AND-DONE-LIST).

make_game.py's pipelines stage gives each checker process its own small cache (a shard) and merges the shards into
pipeline_cache.bin once at the end. This builds two caches from six tiny synthetic compute shaders (three each, no
game data), merges them, and checks that the result is a cache of this device that holds all six: a run over the
six seeded with the merged cache reports six application-cache hits, one seeded with only the first shard three.
It also checks that a cache written for another device is refused and leaves the output as it was, and that
--names-file selects and --done-file lists the right modules.

    python -I check_vk_pipeline_merge.py <vk_pipeline_check> <work dir>

Exit 0 on success, 1 on a failure, 77 (skip) with no Vulkan device or a driver that reports no application-cache
hits even for a cache it wrote itself (llvmpipe). Standard library only, plus tools/spirv_asm.py from this checkout.
"""
import json
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from spirv_asm import assemble  # noqa: E402

SKIP = 77
ASM = """OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize {size} 1 1
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"""
FIRST = ["m_alpha", "m_bravo", "m_charlie"]
SECOND = ["m_delta", "m_echo", "m_foxtrot"]


def write_corpus(directory, names):
    os.makedirs(directory, exist_ok=True)
    index = {}
    for name in names:
        size = 1 << (FIRST + SECOND).index(name)   # a different module (and pipeline) per name
        with open(os.path.join(directory, name + ".spv"), "wb") as f:
            f.write(assemble(ASM.format(size=size), target_env="vulkan1.1"))
        index[name] = {"stage": "cs", "subgroup_size": None, "features": []}
    with open(os.path.join(directory, "index.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(index, f)


def run(cmd):
    p = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, errors="replace", timeout=300)
    return p.returncode, p.stdout, p.stderr


def hits(stdout):
    m = re.search(r"cs cache feedback: (\d+) hits", stdout)
    return int(m.group(1)) if m else None


def main(argv):
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    tool, work = os.path.abspath(argv[1]), argv[2]
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    failures = []

    def expect(condition, what, *output):
        if not condition:
            failures.append(what + ("\n" + "\n".join(o for o in output if o) if output else ""))

    code, out, err = run([tool, "--device-info"])
    if code != 0:
        print(f"SKIP: no Vulkan device (--device-info exited {code}): {err.strip()[-300:]}")
        return SKIP
    info = dict(re.findall(r"(\w+)=(\S+)", out.split(" name=")[0]))
    expect(out.startswith("device-info ") and len(info.get("uuid", "")) == 32, "--device-info line", out)

    first, second, both = (os.path.join(work, d) for d in ("first", "second", "both"))
    write_corpus(first, FIRST)
    write_corpus(second, SECOND)
    write_corpus(both, FIRST + SECOND)
    cache_a, cache_b = os.path.join(work, "a.bin"), os.path.join(work, "b.bin")
    done_a = os.path.join(work, "a.done")
    code, out, err = run([tool, first, cache_a, "--jobs", "2", "--done-file", done_a])
    expect(code == 0 and "cs pipelines: 3 ok" in out, "first shard", out, err)
    with open(done_a, encoding="utf-8") as f:
        expect(f.read() == "".join(n + "\n" for n in FIRST), "--done-file lists the three passed modules")
    code, out, err = run([tool, second, cache_b, "--jobs", "2"])
    expect(code == 0 and "cs pipelines: 3 ok" in out, "second shard", out, err)

    # A driver that reports no hit for a cache it wrote itself can't show what a merge holds.
    code, out, err = run([tool, first, cache_a, "--jobs", "2", "--cache-feedback"])
    if hits(out) != 3:
        print(f"SKIP: this driver reports {hits(out)} application-cache hits for its own cache, not 3")
        return SKIP

    merged = os.path.join(work, "merged.bin")
    code, out, err = run([tool, "--merge", merged, cache_a, cache_b])
    expect(code == 0 and "merged 2 cache(s)" in out, "--merge", out, err)
    with open(merged, "rb") as f:
        header = f.read(32)
    expect(len(header) == 32 and int.from_bytes(header[4:8], "little") == 1, "merged cache header version")
    expect(int.from_bytes(header[8:12], "little") == int(info.get("vendor", "0"), 16), "merged cache vendor")
    expect(header[16:32].hex() == info.get("uuid"), "merged cache pipelineCacheUUID")

    # Seeded with the merge, all six pipelines hit; with the first shard alone only its three.
    for seed, want in ((merged, 6), (cache_a, 3)):
        probe = os.path.join(work, "probe.bin")
        shutil.copyfile(seed, probe)
        code, out, err = run([tool, both, probe, "--jobs", "2", "--cache-feedback"])
        expect(code == 0 and hits(out) == want, f"{os.path.basename(seed)} seeds {want} hits, got {hits(out)}",
               out, err)

    # Merging into an existing output keeps what it held.
    grown = os.path.join(work, "grown.bin")
    shutil.copyfile(cache_a, grown)
    code, out, err = run([tool, "--merge", grown, cache_b])
    expect(code == 0, "--merge into an existing cache", out, err)
    code, out, err = run([tool, both, grown, "--jobs", "2", "--cache-feedback"])
    expect(hits(out) == 6, f"existing output plus one input holds six, got {hits(out)}", out, err)

    # Another device's cache is refused, and the output stays as it was.
    foreign = os.path.join(work, "foreign.bin")
    with open(cache_b, "rb") as f:
        data = bytearray(f.read())
    data[8] ^= 0xFF   # vendorID
    with open(foreign, "wb") as f:
        f.write(data)
    with open(merged, "rb") as f:
        before = f.read()
    code, out, err = run([tool, "--merge", merged, foreign])
    expect(code == 1 and "not this device's cache" in err, "a foreign input fails the merge", out, err)
    with open(merged, "rb") as f:
        expect(f.read() == before, "a failed merge leaves the output as it was")
    code, out, err = run([tool, "--merge", merged, merged])
    expect(code == 2, "the output can't be an input", out, err)
    code, out, err = run([tool, "--merge", merged, os.path.join(work, "missing.bin")])
    expect(code == 1, "an unreadable input fails the merge", out, err)

    # --names-file picks entries of the index; a name it doesn't have is a usage error.
    names = os.path.join(work, "names.txt")
    done = os.path.join(work, "both.done")
    with open(names, "w", encoding="utf-8", newline="\n") as f:
        f.write("m_echo\nm_alpha\n")
    code, out, err = run([tool, both, "--jobs", "2", "--names-file", names, "--done-file", done])
    expect(code == 0 and "cs pipelines: 2 ok" in out, "--names-file selects two", out, err)
    with open(done, encoding="utf-8") as f:
        expect(f.read() == "m_alpha\nm_echo\n", "--done-file of a --names-file run")
    with open(names, "w", encoding="utf-8", newline="\n") as f:
        f.write("m_missing\n")
    code, out, err = run([tool, both, "--names-file", names])
    expect(code == 2 and "not in" in err, "--names-file with a name the index lacks", out, err)
    code, out, err = run([tool, both, "--isolate", "--done-file", done])
    expect(code == 2, "--done-file is refused with --isolate", out, err)

    for failure in failures:
        print("FAIL:", failure)
    if not failures:
        print("vk_pipeline_check merge: ok")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
