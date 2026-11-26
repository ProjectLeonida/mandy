"""A mesh shader's Workgroup memory fits in maxMeshSharedMemorySize (SHADER-MESH-SHARED-LIMIT).

`rdna2spirv.MAX_MESH_SHARED_BYTES` (`--max-mesh-shared-memory`, default 0: no check;
28672 is the VK_EXT_mesh_shader minimum `maxMeshSharedMemorySize`) caps what a MeshEXT module declares in the Workgroup storage class: the LDS array
(PSV0's shared size grown to the furthest ds offset, a power of two), the 4-byte `%ms_counts` and, under
`--wave64-emulate`, `%x_buf` — the same total `lds_bytes` records for the runtime's device-limit check
(vk_pipeline_check / mesh_limit_refusal). Over the cap the shader is refused (`Unsupported`, with the byte
counts). A `--mesh-compute` module is GLCompute and runs as a compute shader, so it keeps the compute cap
(`MAX_SHARED_BYTES`, `--max-shared-memory`). There is no allocation to shrink to: the array is the program's need.

The program is fixtures/ms_shared_limit.asm (hand-written, no game data) and the PSV0 part is written here; the
Workgroup bytes are counted from the emitted SPIR-V. The last test runs the tool's CLI end to end.
"""
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)  # python -I leaves the script's folder off sys.path

import support  # noqa: E402  (support puts tools/ on sys.path, honouring LEONIDA_TOOLS_DIR)

rdna2spirv = support.rdna2spirv
FIXTURE = os.path.join(HERE, "fixtures", "ms_shared_limit.asm")
TOOL = os.path.join(support.ROOT, "tools", "rdna2spirv.py")


def mesh_dxbc(group_shared, threads=128):
    """DXBC with one PSV0 part: the fields mesh_psv() reads (stage 13, 64/64 triangles, NumThreads, shared bytes)."""
    r = bytearray(48)
    struct.pack_into("<I", r, 0, group_shared)
    struct.pack_into("<HH", r, 12, 64, 64)
    r[24], r[27] = 13, 2
    struct.pack_into("<III", r, 36, threads, 1, 1)
    part = b"PSV0" + struct.pack("<I", 4 + len(r)) + struct.pack("<I", len(r)) + bytes(r)
    head = 32 + 4
    return b"DXBC" + bytes(16) + struct.pack("<IIII", 1, head + len(part), 1, head) + part


def fixture_lines():
    with open(FIXTURE, encoding="utf-8") as f:
        return [ln.split(";")[0].strip() for ln in f if ln.split(";")[0].strip()]


def translate(group_shared=0, emulate=False, mesh_compute=False, limit=None):
    prev = rdna2spirv.WAVE64_EMULATE, rdna2spirv.MESH_COMPUTE, rdna2spirv.MAX_MESH_SHARED_BYTES
    rdna2spirv.WAVE64_EMULATE, rdna2spirv.MESH_COMPUTE = emulate, mesh_compute
    if limit is not None:
        rdna2spirv.MAX_MESH_SHARED_BYTES = limit
    try:
        spv = support.translate_lines("ms_shared_limit", fixture_lines(), dxbc=mesh_dxbc(group_shared))
    finally:
        rdna2spirv.WAVE64_EMULATE, rdna2spirv.MESH_COMPUTE, rdna2spirv.MAX_MESH_SHARED_BYTES = prev
    return spv


def workgroup_bytes(spv):
    """{variable: bytes} of every Workgroup-storage-class OpVariable, sized from its pointee type. Only the
    Workgroup pointees are sized: a mesh module's Output arrays (vectors, structs) need no size here."""
    consts, defs, ptrs, out = {}, {}, {}, {}
    for ln in spv.splitlines():
        t = ln.split()
        if len(t) < 3 or t[1] != "=":
            continue
        rid, op, args = t[0], t[2], t[3:]
        if op == "OpConstant":
            consts[rid] = int(args[1], 0)
        elif op in ("OpTypeInt", "OpTypeFloat", "OpTypeVector", "OpTypeBool", "OpTypeArray", "OpTypeStruct",
                    "OpTypeRuntimeArray"):
            defs[rid] = (op, args)
        elif op == "OpTypePointer" and args[0] == "Workgroup":
            ptrs[rid] = args[1]
        elif op == "OpVariable" and args[1:2] == ["Workgroup"]:
            out[rid] = ptrs[args[0]]

    def size(t):
        if t in sizes:
            return sizes[t]
        op, args = defs[t]
        if op in ("OpTypeInt", "OpTypeFloat"):
            s = int(args[0]) // 8
        elif op == "OpTypeVector":
            s = int(args[1]) * size(args[0])
        elif op == "OpTypeBool":
            s = 4
        elif op == "OpTypeArray":
            s = size(args[0]) * consts[args[1]]
        elif op == "OpTypeStruct":
            s = sum(size(a) for a in args)
        else:                                   # OpTypeRuntimeArray and anything unsized
            s = 0
        sizes[t] = s
        return s

    sizes = {}
    return {name: size(t) for name, t in out.items()}


class MeshSharedMemoryLimitTests(unittest.TestCase):
    def test_under_limit_translates(self):
        # 16 KiB of group-shared + the 4-byte %ms_counts: under every cap, translated as MeshEXT
        spv = translate(16384)
        self.assertIn("OpEntryPoint MeshEXT", spv)
        self.assertEqual(workgroup_bytes(spv), {"%lds": 16384, "%ms_counts": 4})
        meta = rdna2spirv.shader_meta(rdna2spirv.translate.last_gen, spv)
        self.assertEqual(meta["lds_bytes"], 16388)

    def test_over_limit_is_refused_with_the_byte_counts(self):
        # 28,672 B of group-shared rounds the array up to 32,768 B (a power of two), + 4 = 32,772 B
        # against the VK_EXT_mesh_shader minimum, 28672
        with self.assertRaises(rdna2spirv.Unsupported) as cm:
            translate(28672, limit=28672)
        msg = str(cm.exception)
        self.assertIn("MS workgroup memory 32772 B", msg)
        self.assertIn("LDS 32768 B", msg)
        self.assertIn("%ms_counts 4 B", msg)
        self.assertIn("maxMeshSharedMemorySize 28672 B", msg)

    def test_default_checks_nothing(self):
        # Without --max-mesh-shared-memory nothing a mesh shader translated to before is refused: a full 64 KiB LDS
        # array plus the 4-byte %ms_counts (65,540 B) still translates, and lds_bytes carries it to the runtime
        spv = translate(65536)
        self.assertEqual(sum(workgroup_bytes(spv).values()), 65540)
        self.assertEqual(rdna2spirv.shader_meta(rdna2spirv.translate.last_gen, spv)["lds_bytes"], 65540)

    def test_limit_is_the_whole_workgroup_total(self):
        # The boundary is the module's whole Workgroup total: exactly it passes, one byte under refuses
        spv = translate(28672, limit=32772)
        self.assertEqual(sum(workgroup_bytes(spv).values()), 32772)
        with self.assertRaises(rdna2spirv.Unsupported):
            translate(28672, limit=32771)

    def test_wave64_exchange_counts_too(self):
        # --wave64-emulate's %x_buf (128 threads = 2 waves: 2L + 6W + 7 dwords = 1,100 B) is Workgroup memory too
        with self.assertRaises(rdna2spirv.Unsupported) as cm:
            translate(28672, emulate=True, limit=28672)
        msg = str(cm.exception)
        self.assertIn("MS workgroup memory 33872 B", msg)
        self.assertIn("wave64 exchange 1100 B", msg)
        spv = translate(8192, emulate=True)
        self.assertEqual(workgroup_bytes(spv), {"%lds": 8192, "%ms_counts": 4, "%x_buf": 1100})

    def test_mesh_compute_uses_the_compute_cap(self):
        # --mesh-compute emits GLCompute and runs as a compute shader: 32,772 B is over the mesh minimum but
        # under the compute cap, so it translates; over the compute cap it is refused as compute
        spv = translate(28672, mesh_compute=True)
        self.assertIn("OpEntryPoint GLCompute", spv)
        with self.assertRaises(rdna2spirv.Unsupported) as cm:
            translate(65536, mesh_compute=True)
        self.assertIn("maxComputeSharedMemorySize", str(cm.exception))

    def test_cli_max_mesh_shared_memory(self):
        # The flag end to end: the default accepts the 32,772 B module and records the whole total as lds_bytes;
        # --max-mesh-shared-memory 28672 (the VK_EXT_mesh_shader minimum) refuses it
        with tempfile.TemporaryDirectory() as tmp:
            dis = os.path.join(tmp, "disasm")
            os.mkdir(dis)
            with open(os.path.join(dis, "ms_limit.s"), "w", encoding="utf-8", newline="\n") as f:
                f.write("; ms_limit  hand-written test program\n")
                for ln in fixture_lines():
                    f.write("\t" + ln + "\n")
            with open(os.path.join(tmp, "ms_limit.dxbc"), "wb") as f:
                f.write(mesh_dxbc(28672))

            def run(*extra):
                out = os.path.join(tmp, "out" + str(len(extra)))
                cmd = [sys.executable, "-I", TOOL, os.path.join(dis, "ms_limit.s"), "--out", out,
                       "--assembler", "python", "--no-validate"] + list(extra)
                return subprocess.run(cmd, capture_output=True, text=True), out

            r, out = run("--max-mesh-shared-memory", "28672")
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("over maxMeshSharedMemorySize 28672 B", r.stdout)
            self.assertFalse(os.path.exists(os.path.join(out, "ms_limit.spv")))

            r, out = run()
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertTrue(os.path.exists(os.path.join(out, "ms_limit.spv")))
            with open(os.path.join(out, "index.json"), encoding="utf-8") as f:
                self.assertEqual(json.load(f)["ms_limit"]["lds_bytes"], 32772)


if __name__ == "__main__":
    unittest.main()
