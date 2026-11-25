"""A run-time-bounded s_buffer_load dword is loaded without a branch, from the heap's base when out of range.

The translator bounds each dword of an s_buffer_load whose bound is only known at run time (a V# from registers):
`off + 4*i + 4 <= NUM_RECORDS * stride`, and the result is 0 outside it. It used to put the load in an
OpSelectionMerge on that check. llvmpipe (Mesa 26.0.8, LLVM 21) hoisted it anyway: #3039's s12 = -4, offset:20
(0xFFFFFFFC + 20, past NUM_RECORDS) read host memory 4 GB past the buffer and segfaulted. Even an address selected
inside the branch is no fix, because there the compiler knows the condition is true and folds the select (#4573's
first version passed one program by luck and failed another). So there is no branch: the dword is loaded through
OpSelect(dword 0 in range, the address, the heap's base), each dword at index i when in range, else 0, and the value
is OpSelect(in range, loaded, 0). Out of range the loads read the first dword of the heap (Xbox VA 0, the start of the
host buffer the runtime maps the GPU window into, so always readable) and the values are thrown away. Not the V#'s
base: a descriptor with no readable bytes (num_records 0) may point anywhere, and main read nothing there
(@memorypressure on #4573).

This file checks that shape on the translated SPIR-V, that values are unchanged, and, with test_scalar_mem_smem's
evaluator, that the only extra reads are discarded reads of VA 0, including for a zero-size V# whose base is far from
any memory. Standard library only; no GPU, no game files.
"""
import os
import re
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)  # python -I leaves the script's folder off sys.path

import test_smem_const_offset as sco  # noqa: E402  (its translate / run helpers, and tools/ on sys.path)

RUNTIME_V = ["s_mov_b32 s8, 0x100", "s_mov_b32 s9, 0", "v_mov_b32_e32 v20, 0x40", "v_readfirstlane_b32 s10, v20",
             "s_mov_b32 s11, 0x31016fac", "v_mov_b32_e32 v21, -4", "v_readfirstlane_b32 s12, v21"]


def defs_of(spv):
    defs = {}
    for ln in spv.split("\n"):
        m = re.match(r"\s*(%\S+) = (Op\w+) (.*)", ln)
        if m:
            defs[m.group(1)] = (m.group(2), m.group(3).split())
    return defs


class SmemOobSafeAddress(unittest.TestCase):
    def test_bounded_dwords_load_through_a_select_with_no_branch(self):
        spv = sco.translate(RUNTIME_V + ["s_buffer_load_dwordx2 s[14:15], s[8:11], s12 offset:20",
                                         "s_waitcnt lgkmcnt(0)", "v_mov_b32_e32 v1, s14", "v_mov_b32_e32 v2, 0x300",
                                         "buffer_store_dword v1, v2, s[0:3], 0 offen"])
        defs = defs_of(spv)
        loads = [(res, args) for res, (op, args) in defs.items()
                 if op == "OpLoad" and args[0] == "%uint" and defs.get(args[1], ("",))[0] == "OpAccessChain"
                 and defs[args[1]][1][0] == "%_ptr_PSB_uint"]
        bounded = []
        for res, args in loads:
            chain = defs[args[1]][1]                             # %_ptr_PSB_uint <ptr> <index>
            conv_op, conv = defs[chain[1]]
            if conv_op == "OpConvertUToPtr" and defs.get(conv[1], ("",))[0] == "OpSelect":
                bounded.append((res, defs[conv[1]][1]))
        self.assertEqual(len(bounded), 2)                     # dwordx2: both dwords
        uses = {res: [d for d in defs.values() if d[0] == "OpSelect" and res in d[1]] for res, _ in bounded}
        sels = {tuple(sel) for _, sel in bounded}
        self.assertEqual(len(sels), 1)                        # one selected pointer for the whole load
        sel = list(sels.pop())
        self.assertEqual(sel[0], "%ulong")
        self.assertNotEqual(sel[2], sel[3])                   # the address, or the V#'s base
        conds = []
        for res, _ in bounded:
            # each value is OpSelect(its own dword's bound, the load, 0)
            (u,) = uses[res]
            self.assertEqual(u[1][0], "%uint")
            self.assertEqual(u[1][2:4], [res, "%c0"])
            conds.append(u[1][1])
        self.assertEqual(conds[0], sel[1])                    # the pointer follows dword 0's bound
        self.assertEqual(len(set(conds)), 2)                  # dword 1 has its own
        # and the bounded loads are not inside a selection construct any more
        body = spv.split("\n")
        for res, _ in bounded:
            k = next(i for i, ln in enumerate(body) if ln.strip().startswith(res + " = OpLoad"))
            self.assertFalse(any("OpBranchConditional" in ln for ln in body[k - 4:k]), res)

    def test_out_of_range_reads_only_the_v_sharp_base_and_discards_it(self):
        import test_scalar_mem_smem as tsm
        rules = tsm.SmemHardwareRuleTests()
        for line, n in (("s_buffer_load_dword s20, s[24:27], s12 offset:20", 1),
                        ("s_buffer_load_dwordx2 s[20:21], s[24:27], s12 offset:16", 2)):
            with self.subTest(line=line):
                ev, reads, _ = rules.run_line(line, {12: 0xFFFFFFFC, "nr": 0x100})
                self.assertEqual(reads, [])                    # nothing the program sees is read
                # out of range every dword reads index 0 of the heap: VA 0 (the evaluator's heap base is 0)
                self.assertEqual(ev.discarded, [0] * n)
                self.assertEqual([ev.sgpr.get(20 + i) for i in range(n)], [0] * n)
        # in range nothing is discarded
        ev, reads, _ = rules.run_line("s_buffer_load_dword s20, s[24:27], s12 offset:4", {12: 8, "nr": 0x100})
        self.assertEqual((reads, ev.discarded), ([12], []))

    def test_zero_size_descriptor_reads_only_the_heap_base(self):
        # num_records 0 with the V# base far from any memory (@memorypressure on #4573): main read nothing, and the
        # branchless form must not read the V# base either; it reads VA 0 and the result is 0
        import test_scalar_mem_smem as tsm
        rules = tsm.SmemHardwareRuleTests()
        for line in ("s_buffer_load_dword s20, s[24:27], s12 offset:0", "s_buffer_load_dwordx4 s[20:23], s[24:27], s12 offset:0"):
            with self.subTest(line=line):
                ev, reads, _ = rules.run_line(line, {12: 0, "nr": 0}, base_low=0)
                n = 4 if "x4" in line else 1
                self.assertEqual(reads, [])
                self.assertEqual(ev.discarded, [0] * n)
                self.assertNotIn(rules.REGION, ev.discarded)
                self.assertEqual([ev.sgpr.get(20 + i) for i in range(n)], [0] * n)

    def test_values_are_unchanged(self):
        for sgpr_off, imm in ((-4, 20), (-16, 0x20), (4, 8), (0x3C, 0), (0x40, 0), (0x10, 0x2C)):
            with self.subTest(sgpr_off=sgpr_off, imm=imm):
                line = f"s_buffer_load_dword s14, s[8:11], s12 offset:{imm:#x}"
                self.assertEqual(sco.run(line, sgpr_off, constant=False), sco.sbuf_ref(sgpr_off, imm))
                self.assertEqual(sco.run(line, sgpr_off, constant=True), sco.sbuf_ref(sgpr_off, imm))


class Fixture(unittest.TestCase):
    """runtime/test/shaders/cs_smem_oob_far.asm, which segfaults vk_compute_test on llvmpipe without this rule."""
    WANT = [0, 0xEEEEEEEE, 0, 0, 0, 0xEEEEEEEE, 0, 0xEEEEEEEE]

    def test_translated_fixture(self):
        import struct
        import support
        import spirv_mini_interp as si
        fssm = bytearray(0x148)
        struct.pack_into("<4I", fssm, 0x138, 0x8088, 32, 1, 1)
        spv = support.translate_fixture("cs_smem_oob_far", fssm=bytes(fssm))
        mem = {4 * k: 0xEEEEEEEE for k in range(64)}
        si.run_workgroup(spv, 32, 32, inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0),
                                                             "%sg_lane": i},
                         push_constants=[0, 0, 0xFFFFFFFF, 0x31016FAC] + [0] * 28, memory=mem)
        self.assertEqual([mem[4 * k] for k in range(8)], self.WANT)

    def test_gpu_runners_expect_the_same_words(self):
        root = os.path.dirname(os.path.dirname(HERE))
        for rel in (("runtime", "test", "shaders", "check.py"), ("scripts", "run_gpu_tests.py")):
            with open(os.path.join(root, *rel), encoding="utf-8") as f:
                m = re.search(r'"cs_smem_oob_far"\)?[,:]\s*\[([^\]]*)\]', f.read())
            with self.subTest(file="/".join(rel)):
                self.assertEqual([int(v, 0) for v in m.group(1).split(",")], self.WANT)


if __name__ == "__main__":
    unittest.main()
