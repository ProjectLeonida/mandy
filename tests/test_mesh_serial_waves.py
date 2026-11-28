"""rdna2spirv.py --mesh-serial-waves: a mesh shader's waves one after the other in 64 invocations.

Under --wave64-emulate a 128-thread mesh shader is two waves side by side in one workgroup, and what the waves do
not do alike goes through mini-schedulers. With --mesh-serial-waves a program whose waves cannot pass each other
anything (no LDS, no store or atomic to memory) declares 64 invocations and runs one pass per wave, each pass a
single wave. The inputs are the hand-written GPU tests runtime/test/shaders/ms_*.asm with conftest's synthetic
PSV0 containers (no game data); on the GPU, scripts/run_gpu_tests.py with LEONIDA_W64_EMULATE=1 and
LEONIDA_MESH_SERIAL_WAVES=1 draws them. Here:
* a program that qualifies declares LocalSize 64, loops over the passes (and, when it has branches, has no
  OpSwitch left: one wave gets the inline plan), takes the invocation's index from the built-in plus 64 * pass, keeps one copy of the export variables per pass and sets
  the mesh outputs once; it passes spirv-val; the metadata records the number of passes and keeps the program's
  thread count;
* the exchange takes its place in %x_buf from the invocation's own index, not from the lane it runs;
* under --mesh-compute (devices without VK_EXT_mesh_shader, such as NVIDIA Pascal) the compute program does the
  same, and writes its slot once per pass: the header from lane 0 of pass 0 only, each pass's vertices and
  primitives at the real index + 64 * pass;
* ms_wave_info_s3, whose waves count lanes in LDS, keeps the side-by-side form, and so does every program
  without --wave64-emulate or of at most 64 threads;
* without the flag the module is unchanged.
"""
import os
import re
import sys

import pytest

from conftest import ROOT, SAMPLE_DXBC, requires_tools

sys.path.insert(0, os.path.join(ROOT, "tools"))
import rdna2spirv  # noqa: E402

SERIAL = ("ms_tri", "ms_clip", "ms_f16_pack", "ms_w64x_ballot", "ms_w64x_wave_votes", "ms_w64x_wave_loops")
SIDE_BY_SIDE = ("ms_wave_info_s3",      # LDS shared by the waves
                "ms_v5_workgroup")      # 32 threads: one partial wave


def _translate(make_shader, name, serial, emulate=True, compute=False):
    with open(os.path.join(ROOT, "runtime", "test", "shaders", name + ".asm"), encoding="utf-8") as f:
        shader = make_shader(f.read(), name, dxbc=SAMPLE_DXBC[name])
    old = rdna2spirv.MESH_SERIAL_WAVES, rdna2spirv.WAVE64_EMULATE, rdna2spirv.MESH_COMPUTE
    rdna2spirv.MESH_SERIAL_WAVES, rdna2spirv.WAVE64_EMULATE, rdna2spirv.MESH_COMPUTE = serial, emulate, compute
    try:
        spvasm = shader.translate()
        meta = rdna2spirv.shader_meta(rdna2spirv.translate.last_gen, spvasm)
    finally:
        rdna2spirv.MESH_SERIAL_WAVES, rdna2spirv.WAVE64_EMULATE, rdna2spirv.MESH_COMPUTE = old
    return shader, spvasm, meta


@requires_tools
@pytest.mark.parametrize("name", SERIAL)
def test_serial_module(make_shader, name):
    shader, spvasm, meta = _translate(make_shader, name, True)
    shader.validate(spvasm)
    assert "OpExecutionMode %main LocalSize 64 1 1" in spvasm
    assert meta["mesh_serial_waves"] == 2 and meta["local_size"] == [128, 1, 1]
    if name == "ms_w64x_wave_votes":    # a program with branches: one wave gets the inline plan, no scheduler
        assert "OpSwitch" not in spvasm     # (a straight-line program keeps the round loop over its segments)
    assert spvasm.count("OpLoopMerge %msw_exit %msw_cont None") == 1
    # The built-in is only read to make the pass's index and the exchange's own place.
    assert "OpDecorate %real_local_index BuiltIn LocalInvocationIndex" in spvasm
    assert "%local_index = OpVariable %_ptr_Function_uint Function" in spvasm
    assert re.search(r"%msw_li = OpIAdd %uint %msw_rl %msw_off\s+OpStore %local_index %msw_li", spvasm)
    assert "%x_li = OpLoad %uint %real_local_index" in spvasm
    assert "%x_li = OpLoad %uint %local_index" not in spvasm
    # One copy of what a pass exported per pass, one OpSetMeshOutputsEXT, one write block per pass.
    for w in (0, 1):
        assert f"OpCopyMemory %ms_pos__w{w} %ms_pos" in spvasm
        assert f"OpLoad %v4float %ms_pos__w{w}" in spvasm
        assert f"%mli_w{w} = OpIAdd %uint %msw_erl" in spvasm
    assert spvasm.count("OpSetMeshOutputsEXT") == 1
    assert spvasm.index("%msw_exit = OpLabel") < spvasm.index("OpSetMeshOutputsEXT")


@requires_tools
@pytest.mark.parametrize("name", SERIAL)
def test_initialised_variables_are_set_again_each_pass(make_shader, name):
    _, spvasm, _ = _translate(make_shader, name, True)
    body = spvasm[spvasm.index("%msw_body = OpLabel"):]
    variables = re.findall(r"^\s*(%\w+) = OpVariable \S+ Function (\S+)$", spvasm, re.M)
    assert variables
    for var, init in variables:
        if var in ("%local_index", "%msw_pass") or "__w" in var:
            continue                # the pass counter, the index it sets, and the per-pass copies
        assert re.search(rf"OpStore {re.escape(var)} {re.escape(init)}\n", body), var


@requires_tools
@pytest.mark.parametrize("name", SIDE_BY_SIDE)
def test_program_that_does_not_qualify_is_unchanged(make_shader, name):
    _, plain, plain_meta = _translate(make_shader, name, False)
    _, flagged, meta = _translate(make_shader, name, True)
    assert flagged == plain and meta == plain_meta
    assert "mesh_serial_waves" not in meta and "%msw_" not in flagged


@requires_tools
@pytest.mark.parametrize("compute", [False, True], ids=["mesh", "mesh-compute"])
def test_flag_needs_emulation(make_shader, compute):
    _, plain, _ = _translate(make_shader, "ms_tri", False, False, compute)
    _, flagged, meta = _translate(make_shader, "ms_tri", True, False, compute)
    assert flagged == plain and "mesh_serial_waves" not in meta


@requires_tools
@pytest.mark.parametrize("name", SERIAL)
def test_serial_mesh_compute_module(make_shader, name):
    shader, spvasm, meta = _translate(make_shader, name, True, compute=True)
    shader.validate(spvasm)
    assert "OpEntryPoint GLCompute %main" in spvasm and "OpSetMeshOutputsEXT" not in spvasm
    assert "OpExecutionMode %main LocalSize 64 1 1" in spvasm
    assert meta["mesh_serial_waves"] == 2 and meta["local_size"] == [128, 1, 1] and "mesh_compute" in meta
    if name == "ms_w64x_wave_votes":    # one wave per pass: the inline plan, no scheduler
        assert "OpSwitch" not in spvasm
    assert spvasm.count("OpLoopMerge %msw_exit %msw_cont None") == 1
    assert "OpDecorate %real_local_index BuiltIn LocalInvocationIndex" in spvasm
    assert "%x_li = OpLoad %uint %real_local_index" in spvasm
    main = spvasm[spvasm.index("%main = OpFunction"):]
    loop_end = main.index("%msw_exit = OpLabel")
    # One barrier and one read of the counts, after the passes; then the slot writes of each pass from its copies.
    assert main.count("%msc = OpLoad %uint %ms_counts") == 1 and loop_end < main.index("%msc = OpLoad")
    for w in (0, 1):
        assert f"OpCopyMemory %ms_pos__w{w} %ms_pos" in main
        assert re.search(rf"%mli_w{w} = OpIAdd %uint %msw_erl %\w+", main)
        assert f"OpLoad %v4float %ms_pos__w{w}" in main[loop_end:]
        # The header goes to lane 0 of the slot: pass w's lane 0 is index 64 * w, so only pass 0 writes it.
        assert re.search(rf"%mh0_w{w} = OpIEqual %bool %mli_w{w} %c0\b", main)
    assert "OpLoad %v4float %ms_pos\n" not in main[loop_end:]   # no write from the last pass's own variables
    offset = re.search(r"%mli_w1 = OpIAdd %uint %msw_erl (%\w+)", main).group(1)
    assert re.search(rf"^\s*{re.escape(offset)} = OpConstant %uint 64$", spvasm, re.M)
    # As many slot stores per pass as the module without passes has in all.
    _, plain, _ = _translate(make_shader, name, False, compute=True)
    plain_main = plain[plain.index("%main = OpFunction"):]
    stores = len(re.findall(r"%mp\d+ = OpConvertUToPtr %_ptr_PSB_uint", plain_main))   # the slot's
    assert stores
    for w in (0, 1):
        assert len(re.findall(rf"%mp\d+_w{w} = OpConvertUToPtr %_ptr_PSB_uint", main)) == stores


@requires_tools
@pytest.mark.parametrize("name", SIDE_BY_SIDE)
def test_mesh_compute_program_that_does_not_qualify_is_unchanged(make_shader, name):
    _, plain, plain_meta = _translate(make_shader, name, False, compute=True)
    _, flagged, meta = _translate(make_shader, name, True, compute=True)
    assert flagged == plain and meta == plain_meta and "%msw_" not in flagged


@requires_tools
@pytest.mark.parametrize("name", SERIAL)
def test_without_the_flag_the_waves_are_side_by_side(make_shader, name):
    shader, spvasm, meta = _translate(make_shader, name, False)
    shader.validate(spvasm)
    assert "OpExecutionMode %main LocalSize 128 1 1" in spvasm
    assert "%msw_" not in spvasm and "mesh_serial_waves" not in meta
    assert "OpDecorate %local_index BuiltIn LocalInvocationIndex" in spvasm


@requires_tools
def test_a_wave_that_ends_early_does_not_skip_the_later_passes(make_shader):
    # ms_w64x_wave_votes with an s_endpgm in the middle of the program, behind a branch on the wave's number (7,
    # so never taken here): the end of one wave's program must end its pass, not the invocation.
    name = "ms_w64x_wave_votes"
    with open(os.path.join(ROOT, "runtime", "test", "shaders", name + ".asm"), encoding="utf-8") as f:
        text = f.read()
    at = "s_bfe_u32 s4, s3, 0x40018\n"
    assert text.count(at) == 1
    text = text.replace(at, at + "s_cmp_eq_u32 s4, 7\ns_cbranch_scc0 1\ns_endpgm\n")
    shader = make_shader(text, name, dxbc=SAMPLE_DXBC[name])
    old = rdna2spirv.MESH_SERIAL_WAVES, rdna2spirv.WAVE64_EMULATE
    rdna2spirv.MESH_SERIAL_WAVES, rdna2spirv.WAVE64_EMULATE = True, True
    try:
        spvasm = shader.translate()
        meta = rdna2spirv.shader_meta(rdna2spirv.translate.last_gen, spvasm)
    finally:
        rdna2spirv.MESH_SERIAL_WAVES, rdna2spirv.WAVE64_EMULATE = old
    shader.validate(spvasm)
    assert meta["mesh_serial_waves"] == 2
    assert spvasm != _translate(make_shader, name, True)[1]     # the early end is in the module
    assert "OpExecutionMode %main LocalSize 64 1 1" in spvasm
    main = spvasm[spvasm.index("%main = OpFunction"):]
    main = main[:main.index("OpFunctionEnd")]
    # One return, after the loop over the passes and after the outputs are written.
    assert main.count("OpReturn") == 1 and "OpKill" not in main and "OpTerminateInvocation" not in main
    assert main.index("%msw_exit = OpLabel") < main.index("OpSetMeshOutputsEXT") < main.index("OpReturn")
