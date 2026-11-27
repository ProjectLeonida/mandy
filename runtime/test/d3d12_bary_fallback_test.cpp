// End-to-end Vulkan GPU test for `--bary-fallback` D3D12 graphics PSOs (RT-BARY-FALLBACK-PSO).
//
// Exercises graphics PSOs whose pixel shader carries `bary_fallback_input` in `index.json` through the Xbox-ABI
// device proxy on a real Vulkan 1.3 GPU (including GPUs without `VK_KHR_fragment_shader_barycentric`, such as NVIDIA
// Pascal). SPIR-V 1.5 modules are constructed directly in memory — no glslangValidator, llvm-mc or game files needed:
//   1. Perspective-correct barycentrics + pass-through vec4 attribute: draws a fullscreen-covering triangle with
//      non-uniform clip W = (1, 2, 4) and verifies all 16x16 pixels match exact perspective-weighted `(b0, b1, b2)`
//      plus pass-through `Location 0` alpha (`0.75`), separating perspective interpolation from affine by >40 LSB.
//   2. VS `ClipDistance[1]` forwarding (when `shaderClipDistance` is enabled): clips `ndc_x < 0` to the clear color
//      while keeping `ndc_x > 0` with intact barycentrics.
//   3. Unused PS Input pruning + `--f16-interp-fallback` combinations:
//      a) 16-param VS (`0..29`) + PS declaring 26 `Input`s (`0..13`, `16..27`) while only reading `Location 7` +
//         `bary_fallback_input` at `Location 14`: unread inputs are zero-filled in the GS so `VS -> GS` stays within
//         64 input components (`maxGeometryInputComponents = 64`, e.g. RADV).
//      b) 10-param VS (`0..9`) + PS with `bary_fallback_input` at `Location 12` and colliding `f16_halves` at `[9, 10]`
//         relocated into `[13, 14]`: exercises combined `--f16-interp-fallback` + `--bary-fallback` within 64 geometry
//         input components.
//      c) 16-param VS (`0..15`) + PS with `f16_halves` relocated to `[28, 29]`: draws and verifies all 256 pixels when
//         `maxGeometryInputComponents >= 120 && maxGeometryOutputComponents >= 120`, or verifies clean logged refusal +
//         skips the draw on devices with tighter geometry stage limits.
//   4. Refusals: non-triangle topology (`E_INVALIDARG`), explicit GS (`E_NOTIMPL`), disabled `geometryShader` feature
//      (`E_NOTIMPL`), and `maxGeometryInputComponents` / `VUID-RuntimeSpirv-Location-06272` limit refusals (`E_NOTIMPL`).
//   5. Computed I/J pairs (RT-BARY-FALLBACK-PER-VERTEX): a PS that reads an attribute as a `Flat vec4[3]` named by
//      `bary_fallback_input.per_vertex` and weights the three vertices' values with its own I/J. The VS gives each
//      vertex a different value, and the weights are chosen so that every other order of the three vertices lands
//      at least 15 LSB away. Drawn with the block at the attribute's own first locations and relocated next to an
//      ordinary input; without the map the same PS is refused (`E_FAIL`).
// Skips (77) when no usable Vulkan 1.3 device or `geometryShader` feature is present. No game data.
#include "vk_test_device.h"
#include "d3d12_test_shaders.h"
#include "../shims/d3d12_com.h"
#include "../shims/d3d12_xs_device.h"
#include "../src/shader_library.h"
#include "../src/spirv_bary_fallback.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <vector>

#define EXPECT(cond) do { \
    if (!(cond)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #cond); return 1; } \
} while (0)

using namespace d3d12test;
using namespace leonida::spirv_detail;
using XboxHeapHandleFn = std::uint64_t(WINAPI*)(void* heap);  // heap slot 9: handle in RAX

namespace {

void put_inst(std::vector<std::uint32_t>& w, std::uint16_t op, std::initializer_list<std::uint32_t> args) {
    w.push_back((std::uint32_t(1 + args.size()) << 16) | op);
    for (std::uint32_t a : args) w.push_back(a);
}

void put_inst_vec(std::vector<std::uint32_t>& w, std::uint16_t op, const std::vector<std::uint32_t>& args) {
    w.push_back((std::uint32_t(1 + args.size()) << 16) | op);
    w.insert(w.end(), args.begin(), args.end());
}

void write_spv(const std::filesystem::path& path, const std::vector<std::uint32_t>& words) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    f.write(reinterpret_cast<const char*>(words.data()), std::streamsize(words.size() * 4));
}

// Triangle VS from VertexIndex (0, 1, 2):
//   - If `non_uniform_w` is true: V0 = (-1, -1, 0, 1), V1 = (6, -2, 0, 2), V2 = (-4, 12, 0, 4)
//     (same NDC (-1,-1), (3,-1), (-1,3), with w = 1, 2, 4).
//   - Else: V0 = (-1, -1, 0, 1), V1 = (3, -1, 0, 1), V2 = (-1, 3, 0, 1).
// Exports Location `attr_loc` = vec4(0.5, 0.25, 0.75, 1.0), and optionally BuiltIn ClipDistance float[1] = {ndc_x}.
std::vector<std::uint32_t> make_bary_vs(bool non_uniform_w, bool with_clip_x, std::uint32_t attr_loc = 0) {
    std::uint32_t next_id = 1;
    const auto alloc = [&]() { return next_id++; };

    const std::uint32_t id_void = alloc();
    const std::uint32_t id_fn = alloc();
    const std::uint32_t id_float = alloc();
    const std::uint32_t id_v4float = alloc();
    const std::uint32_t id_int = alloc();
    const std::uint32_t id_uint = alloc();
    const std::uint32_t id_bool = alloc();
    const std::uint32_t id_ptr_in_int = alloc();
    const std::uint32_t id_ptr_out_v4 = alloc();

    const std::uint32_t id_u1 = with_clip_x ? alloc() : 0;
    const std::uint32_t id_clip_arr = with_clip_x ? alloc() : 0;
    const std::uint32_t id_ptr_out_clip = with_clip_x ? alloc() : 0;
    const std::uint32_t id_ptr_out_f = with_clip_x ? alloc() : 0;

    const std::uint32_t id_f0 = alloc();     //  0.0
    const std::uint32_t id_f1 = alloc();     //  1.0
    const std::uint32_t id_fm1 = alloc();    // -1.0
    const std::uint32_t id_f3 = alloc();     //  3.0
    const std::uint32_t id_f2 = alloc();     //  2.0
    const std::uint32_t id_f4 = alloc();     //  4.0
    const std::uint32_t id_f6 = alloc();     //  6.0
    const std::uint32_t id_fm2 = alloc();    // -2.0
    const std::uint32_t id_fm4 = alloc();    // -4.0
    const std::uint32_t id_f12 = alloc();    // 12.0
    const std::uint32_t id_f025 = alloc();   //  0.25
    const std::uint32_t id_f05 = alloc();    //  0.50
    const std::uint32_t id_f075 = alloc();   //  0.75

    const std::uint32_t id_p0 = alloc();
    const std::uint32_t id_p1 = alloc();
    const std::uint32_t id_p2 = alloc();
    const std::uint32_t id_attr0_val = alloc();
    const std::uint32_t id_i0 = alloc();
    const std::uint32_t id_i1 = alloc();

    const std::uint32_t id_vid_var = alloc();
    const std::uint32_t id_pos_var = alloc();
    const std::uint32_t id_out0_var = alloc();
    const std::uint32_t id_clip_var = with_clip_x ? alloc() : 0;

    const std::uint32_t id_main = alloc();
    const std::uint32_t id_entry = alloc();
    const std::uint32_t id_vid = alloc();
    const std::uint32_t id_is0 = alloc();
    const std::uint32_t id_is1 = alloc();
    const std::uint32_t id_p12 = alloc();
    const std::uint32_t id_pos = alloc();
    const std::uint32_t id_cx12 = with_clip_x ? alloc() : 0;
    const std::uint32_t id_cx = with_clip_x ? alloc() : 0;
    const std::uint32_t id_cptr = with_clip_x ? alloc() : 0;

    std::vector<std::uint32_t> w = {kSpirvMagic, 0x00010500u, 0, next_id, 0};
    put_inst(w, kOpCapability, {1});  // Shader
    if (with_clip_x) put_inst(w, kOpCapability, {32});  // ClipDistance
    put_inst(w, kOpMemoryModel, {0, 1});

    std::vector<std::uint32_t> ep = {kExecModelVertex, id_main, 0x6e69616du, 0x00000000u,
                                     id_vid_var, id_pos_var, id_out0_var};
    if (with_clip_x) ep.push_back(id_clip_var);
    put_inst_vec(w, kOpEntryPoint, ep);

    put_inst(w, kOpDecorate, {id_vid_var, 11, 42});  // BuiltIn VertexIndex
    put_inst(w, kOpDecorate, {id_pos_var, 11, 0});   // BuiltIn Position
    put_inst(w, kOpDecorate, {id_out0_var, kDecorationLocation, attr_loc});
    if (with_clip_x) put_inst(w, kOpDecorate, {id_clip_var, 11, 3});  // BuiltIn ClipDistance

    put_inst(w, 19, {id_void});
    put_inst(w, 33, {id_fn, id_void});
    put_inst(w, kOpTypeFloat, {id_float, 32});
    put_inst(w, kOpTypeVector, {id_v4float, id_float, 4});
    put_inst(w, kOpTypeInt, {id_int, 32, 1});
    put_inst(w, kOpTypeInt, {id_uint, 32, 0});
    put_inst(w, 20, {id_bool});
    put_inst(w, kOpTypePointer, {id_ptr_in_int, kStorageInput, id_int});
    put_inst(w, kOpTypePointer, {id_ptr_out_v4, kStorageOutput, id_v4float});
    if (with_clip_x) {
        put_inst(w, 43, {id_uint, id_u1, 1u});
        put_inst(w, 28, {id_clip_arr, id_float, id_u1});
        put_inst(w, kOpTypePointer, {id_ptr_out_clip, kStorageOutput, id_clip_arr});
        put_inst(w, kOpTypePointer, {id_ptr_out_f, kStorageOutput, id_float});
    }

    put_inst(w, 43, {id_float, id_f0, 0x00000000u});
    put_inst(w, 43, {id_float, id_f1, 0x3f800000u});
    put_inst(w, 43, {id_float, id_fm1, 0xbf800000u});
    put_inst(w, 43, {id_float, id_f3, 0x40400000u});
    put_inst(w, 43, {id_float, id_f2, 0x40000000u});
    put_inst(w, 43, {id_float, id_f4, 0x40800000u});
    put_inst(w, 43, {id_float, id_f6, 0x40c00000u});
    put_inst(w, 43, {id_float, id_fm2, 0xc0000000u});
    put_inst(w, 43, {id_float, id_fm4, 0xc0800000u});
    put_inst(w, 43, {id_float, id_f12, 0x41400000u});
    put_inst(w, 43, {id_float, id_f025, 0x3e800000u});
    put_inst(w, 43, {id_float, id_f05, 0x3f000000u});
    put_inst(w, 43, {id_float, id_f075, 0x3f400000u});

    put_inst(w, 44, {id_v4float, id_p0, id_fm1, id_fm1, id_f0, id_f1});
    if (non_uniform_w) {
        put_inst(w, 44, {id_v4float, id_p1, id_f6, id_fm2, id_f0, id_f2});
        put_inst(w, 44, {id_v4float, id_p2, id_fm4, id_f12, id_f0, id_f4});
    } else {
        put_inst(w, 44, {id_v4float, id_p1, id_f3, id_fm1, id_f0, id_f1});
        put_inst(w, 44, {id_v4float, id_p2, id_fm1, id_f3, id_f0, id_f1});
    }
    put_inst(w, 44, {id_v4float, id_attr0_val, id_f05, id_f025, id_f075, id_f1});
    put_inst(w, 43, {id_int, id_i0, 0u});
    put_inst(w, 43, {id_int, id_i1, 1u});

    put_inst(w, kOpVariable, {id_ptr_in_int, id_vid_var, kStorageInput});
    put_inst(w, kOpVariable, {id_ptr_out_v4, id_pos_var, kStorageOutput});
    put_inst(w, kOpVariable, {id_ptr_out_v4, id_out0_var, kStorageOutput});
    if (with_clip_x) put_inst(w, kOpVariable, {id_ptr_out_clip, id_clip_var, kStorageOutput});

    put_inst(w, kOpFunction, {id_void, id_main, 0, id_fn});
    put_inst(w, 248, {id_entry});
    put_inst(w, kOpLoad, {id_int, id_vid, id_vid_var});
    put_inst(w, 170, {id_bool, id_is0, id_vid, id_i0});      // OpIEqual vid 0
    put_inst(w, 170, {id_bool, id_is1, id_vid, id_i1});      // OpIEqual vid 1
    put_inst(w, 169, {id_v4float, id_p12, id_is1, id_p1, id_p2});
    put_inst(w, 169, {id_v4float, id_pos, id_is0, id_p0, id_p12});
    put_inst(w, kOpStore, {id_pos_var, id_pos});
    put_inst(w, kOpStore, {id_out0_var, id_attr0_val});
    if (with_clip_x) {
        // ndc_x = (-1 for V0, +3 for V1, -1 for V2) -> clips left half (ndc_x < 0).
        put_inst(w, 169, {id_float, id_cx12, id_is1, id_f3, id_fm1});
        put_inst(w, 169, {id_float, id_cx, id_is0, id_fm1, id_cx12});
        put_inst(w, kOpAccessChain, {id_ptr_out_f, id_cptr, id_clip_var, id_i0});
        put_inst(w, kOpStore, {id_cptr, id_cx});
    }
    put_inst(w, kOpReturn, {});
    put_inst(w, kOpFunctionEnd, {});
    return w;
}

// PS reading Location `attr_loc` (vec4 attr0) and Location `bary_loc` (vec3 bary), writing vec4(bary.x, bary.y, bary.z, attr0.z).
std::vector<std::uint32_t> make_bary_ps(std::uint32_t bary_loc, std::uint32_t attr_loc = 0) {
    // 1=void, 2=fn, 3=float, 4=v3float, 5=v4float, 6=ptr_in_v3, 7=ptr_in_v4, 8=ptr_out_v4,
    // 9=out0, 10=in_attr0, 11=in_bary, 12=main, 13=entry,
    // 14=attr0, 15=bary, 16=bx, 17=by, 18=bz, 19=az, 20=rgba
    constexpr std::uint32_t kBound = 21;
    std::vector<std::uint32_t> w = {kSpirvMagic, 0x00010500u, 0, kBound, 0};
    put_inst(w, kOpCapability, {1});
    put_inst(w, kOpMemoryModel, {0, 1});
    put_inst(w, kOpEntryPoint, {kExecModelFragment, 12, 0x6e69616du, 0x00000000u, 9, 10, 11});
    put_inst(w, kOpExecutionMode, {12, 7});  // OriginUpperLeft

    put_inst(w, kOpDecorate, {9, kDecorationLocation, 0});
    put_inst(w, kOpDecorate, {10, kDecorationLocation, attr_loc});
    put_inst(w, kOpDecorate, {11, kDecorationLocation, bary_loc});

    put_inst(w, 19, {1});
    put_inst(w, 33, {2, 1});
    put_inst(w, kOpTypeFloat, {3, 32});
    put_inst(w, kOpTypeVector, {4, 3, 3});
    put_inst(w, kOpTypeVector, {5, 3, 4});
    put_inst(w, kOpTypePointer, {6, kStorageInput, 4});
    put_inst(w, kOpTypePointer, {7, kStorageInput, 5});
    put_inst(w, kOpTypePointer, {8, kStorageOutput, 5});

    put_inst(w, kOpVariable, {8, 9, kStorageOutput});
    put_inst(w, kOpVariable, {7, 10, kStorageInput});
    put_inst(w, kOpVariable, {6, 11, kStorageInput});

    put_inst(w, kOpFunction, {1, 12, 0, 2});
    put_inst(w, 248, {13});
    put_inst(w, kOpLoad, {5, 14, 10});
    put_inst(w, kOpLoad, {4, 15, 11});
    put_inst(w, kOpCompositeExtract, {3, 16, 15, 0});
    put_inst(w, kOpCompositeExtract, {3, 17, 15, 1});
    put_inst(w, kOpCompositeExtract, {3, 18, 15, 2});
    put_inst(w, kOpCompositeExtract, {3, 19, 14, 2});
    put_inst(w, kOpCompositeConstruct, {5, 20, 16, 17, 18, 19});
    put_inst(w, kOpStore, {9, 20});
    put_inst(w, kOpReturn, {});
    put_inst(w, kOpFunctionEnd, {});
    return w;
}

// Multi-param VS (param0..param(num_raw-1), optional static f16_halves for 0..6,
// param7.x = packHalf2x16(0.25, 0.75) = 0x3a003400, param7.w = 1.0).
std::vector<std::uint32_t> make_nparam_vs(std::uint32_t num_raw, bool with_static_halves) {
    const std::uint32_t num_halves = with_static_halves ? 14u : 0u;
    const std::uint32_t last_out_id = 30u + num_raw + num_halves;
    const std::uint32_t bound = last_out_id + 1u;
    std::vector<std::uint32_t> w = {kSpirvMagic, 0x00010500u, 0, bound, 0};
    put_inst(w, kOpCapability, {1});
    put_inst(w, kOpMemoryModel, {0, 1});
    std::vector<std::uint32_t> ep = {kExecModelVertex, 24, 0x6e69616du, 0x00000000u, 8, 9};
    for (std::uint32_t id = 31; id <= last_out_id; ++id) ep.push_back(id);
    put_inst_vec(w, kOpEntryPoint, ep);

    put_inst(w, kOpDecorate, {8, 11, 42});  // BuiltIn VertexIndex
    put_inst(w, kOpDecorate, {9, 11, 0});   // BuiltIn Position
    for (std::uint32_t p = 0; p < num_raw; ++p) put_inst(w, kOpDecorate, {31 + p, kDecorationLocation, p});
    if (with_static_halves) {
        const std::uint32_t half_base = 31u + num_raw;
        for (std::uint32_t n = 0; n <= 6; ++n) {
            put_inst(w, kOpDecorate, {half_base + 2 * n, kDecorationLocation, 28 - 2 * n});
            put_inst(w, kOpDecorate, {half_base + 2 * n + 1, kDecorationLocation, 29 - 2 * n});
        }
    }

    put_inst(w, 19, {1});
    put_inst(w, 33, {2, 1});
    put_inst(w, kOpTypeFloat, {3, 32});
    put_inst(w, kOpTypeVector, {4, 3, 4});
    put_inst(w, kOpTypeInt, {5, 32, 1});
    put_inst(w, kOpTypePointer, {6, kStorageInput, 5});
    put_inst(w, kOpTypePointer, {7, kStorageOutput, 4});
    put_inst(w, 20, {20});

    put_inst(w, kOpConstantNull, {4, 10});
    put_inst(w, 43, {3, 11, 0xbf800000u});  // -1.0f
    put_inst(w, 43, {3, 12, 0x40400000u});  //  3.0f
    put_inst(w, 43, {3, 13, 0x00000000u});  //  0.0f
    put_inst(w, 43, {3, 14, 0x3f800000u});  //  1.0f
    put_inst(w, 44, {4, 15, 11, 11, 13, 14});
    put_inst(w, 44, {4, 16, 12, 11, 13, 14});
    put_inst(w, 44, {4, 17, 11, 12, 13, 14});
    put_inst(w, 43, {5, 18, 0u});
    put_inst(w, 43, {5, 19, 1u});
    put_inst(w, 43, {3, 21, 0x3a003400u});  // packHalf2x16(0.25, 0.75)
    put_inst(w, 43, {3, 22, 0x3c003800u});
    put_inst(w, 44, {4, 23, 21, 22, 13, 14});

    put_inst(w, kOpVariable, {6, 8, kStorageInput});
    put_inst(w, kOpVariable, {7, 9, kStorageOutput});
    for (std::uint32_t id = 31; id <= last_out_id; ++id) put_inst(w, kOpVariable, {7, id, kStorageOutput});

    put_inst(w, kOpFunction, {1, 24, 0, 2});
    put_inst(w, 248, {25});
    put_inst(w, kOpLoad, {5, 26, 8});
    put_inst(w, 170, {20, 27, 26, 18});
    put_inst(w, 170, {20, 28, 26, 19});
    put_inst(w, 169, {4, 29, 28, 16, 17});
    put_inst(w, 169, {4, 30, 27, 15, 29});
    put_inst(w, kOpStore, {9, 30});
    for (std::uint32_t p = 0; p < num_raw; ++p) put_inst(w, kOpStore, {31 + p, p == 7 ? 23u : 10u});
    if (with_static_halves) {
        for (std::uint32_t id = 31u + num_raw; id <= last_out_id; ++id) put_inst(w, kOpStore, {id, 10});
    }
    put_inst(w, kOpReturn, {});
    put_inst(w, kOpFunctionEnd, {});
    return w;
}

// PS declaring 26 vec4 Inputs (0..13 and 16..27) + vec3 bary at 14, loading ONLY Location 7 + bary 14,
// writing vec4(bary.x, bary.y, bary.z, param7.w) to MRT 0.
std::vector<std::uint32_t> make_bary_pruned_ps() {
    std::uint32_t next_id = 1;
    const auto alloc = [&]() { return next_id++; };

    const std::uint32_t id_void = alloc();
    const std::uint32_t id_fn = alloc();
    const std::uint32_t id_float = alloc();
    const std::uint32_t id_v3float = alloc();
    const std::uint32_t id_v4float = alloc();
    const std::uint32_t id_ptr_in_v3 = alloc();
    const std::uint32_t id_ptr_in_v4 = alloc();
    const std::uint32_t id_ptr_out_v4 = alloc();
    const std::uint32_t id_out0 = alloc();
    const std::uint32_t id_in_bary = alloc();
    const std::uint32_t id_main = alloc();
    const std::uint32_t id_entry = alloc();

    std::uint32_t id_in7 = 0;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> in_vars;
    for (std::uint32_t loc = 0; loc < 28; ++loc) {
        if (loc == 14 || loc == 15) continue;
        const std::uint32_t vid = alloc();
        in_vars.push_back({vid, loc});
        if (loc == 7) id_in7 = vid;
    }

    const std::uint32_t id_bary_val = alloc();
    const std::uint32_t id_v7 = alloc();
    const std::uint32_t id_bx = alloc();
    const std::uint32_t id_by = alloc();
    const std::uint32_t id_bz = alloc();
    const std::uint32_t id_w7 = alloc();
    const std::uint32_t id_rgba = alloc();

    std::vector<std::uint32_t> w = {kSpirvMagic, 0x00010500u, 0, next_id, 0};
    put_inst(w, kOpCapability, {1});
    put_inst(w, kOpMemoryModel, {0, 1});
    std::vector<std::uint32_t> ep = {kExecModelFragment, id_main, 0x6e69616du, 0x00000000u, id_out0, id_in_bary};
    for (const auto& [vid, loc] : in_vars) ep.push_back(vid);
    put_inst_vec(w, kOpEntryPoint, ep);
    put_inst(w, kOpExecutionMode, {id_main, 7});

    put_inst(w, kOpDecorate, {id_out0, kDecorationLocation, 0});
    put_inst(w, kOpDecorate, {id_in_bary, kDecorationLocation, 14});
    for (const auto& [vid, loc] : in_vars) put_inst(w, kOpDecorate, {vid, kDecorationLocation, loc});

    put_inst(w, 19, {id_void});
    put_inst(w, 33, {id_fn, id_void});
    put_inst(w, kOpTypeFloat, {id_float, 32});
    put_inst(w, kOpTypeVector, {id_v3float, id_float, 3});
    put_inst(w, kOpTypeVector, {id_v4float, id_float, 4});
    put_inst(w, kOpTypePointer, {id_ptr_in_v3, kStorageInput, id_v3float});
    put_inst(w, kOpTypePointer, {id_ptr_in_v4, kStorageInput, id_v4float});
    put_inst(w, kOpTypePointer, {id_ptr_out_v4, kStorageOutput, id_v4float});

    put_inst(w, kOpVariable, {id_ptr_out_v4, id_out0, kStorageOutput});
    put_inst(w, kOpVariable, {id_ptr_in_v3, id_in_bary, kStorageInput});
    for (const auto& [vid, loc] : in_vars) put_inst(w, kOpVariable, {id_ptr_in_v4, vid, kStorageInput});

    put_inst(w, kOpFunction, {id_void, id_main, 0, id_fn});
    put_inst(w, 248, {id_entry});
    put_inst(w, kOpLoad, {id_v3float, id_bary_val, id_in_bary});
    put_inst(w, kOpLoad, {id_v4float, id_v7, id_in7});
    put_inst(w, kOpCompositeExtract, {id_float, id_bx, id_bary_val, 0});
    put_inst(w, kOpCompositeExtract, {id_float, id_by, id_bary_val, 1});
    put_inst(w, kOpCompositeExtract, {id_float, id_bz, id_bary_val, 2});
    put_inst(w, kOpCompositeExtract, {id_float, id_w7, id_v7, 3});
    put_inst(w, kOpCompositeConstruct, {id_v4float, id_rgba, id_bx, id_by, id_bz, id_w7});
    put_inst(w, kOpStore, {id_out0, id_rgba});
    put_inst(w, kOpReturn, {});
    put_inst(w, kOpFunctionEnd, {});
    return w;
}

// Combined f16 + bary PS: reads raw 0..num_raw-1, bary vec3 at `bary_loc`, and f16 halves at `lo_loc`/`hi_loc`,
// writing vec4(lo.x, hi.x, bary.y, bary.z) to MRT 0.
std::vector<std::uint32_t> make_bary_f16_ps(std::uint32_t num_raw, std::uint32_t bary_loc,
                                            std::uint32_t lo_loc, std::uint32_t hi_loc) {
    // 1=void, 2=fn, 3=float, 4=v3float, 5=v4float, 6=ptr_in_v3, 7=ptr_in_v4, 8=ptr_out_v4,
    // 9=out0, 10=in_bary, 11=in_lo, 12=in_hi, 13=main, 14=entry,
    // 15=bary, 16=vlo, 17=vhi, 18=lox, 19=hix, 20=by, 21=bz, 22=rgba, 23..=raw
    const std::uint32_t bound = 23 + num_raw;
    std::vector<std::uint32_t> w = {kSpirvMagic, 0x00010500u, 0, bound, 0};
    put_inst(w, kOpCapability, {1});
    put_inst(w, kOpMemoryModel, {0, 1});
    std::vector<std::uint32_t> ep = {kExecModelFragment, 13, 0x6e69616du, 0x00000000u, 9, 10, 11, 12};
    for (std::uint32_t i = 0; i < num_raw; ++i) ep.push_back(23 + i);
    put_inst_vec(w, kOpEntryPoint, ep);
    put_inst(w, kOpExecutionMode, {13, 7});

    put_inst(w, kOpDecorate, {9, kDecorationLocation, 0});
    put_inst(w, kOpDecorate, {10, kDecorationLocation, bary_loc});
    put_inst(w, kOpDecorate, {11, kDecorationLocation, lo_loc});
    put_inst(w, kOpDecorate, {12, kDecorationLocation, hi_loc});
    for (std::uint32_t i = 0; i < num_raw; ++i) put_inst(w, kOpDecorate, {23 + i, kDecorationLocation, i});

    put_inst(w, 19, {1});
    put_inst(w, 33, {2, 1});
    put_inst(w, kOpTypeFloat, {3, 32});
    put_inst(w, kOpTypeVector, {4, 3, 3});
    put_inst(w, kOpTypeVector, {5, 3, 4});
    put_inst(w, kOpTypePointer, {6, kStorageInput, 4});
    put_inst(w, kOpTypePointer, {7, kStorageInput, 5});
    put_inst(w, kOpTypePointer, {8, kStorageOutput, 5});

    put_inst(w, kOpVariable, {8, 9, kStorageOutput});
    put_inst(w, kOpVariable, {6, 10, kStorageInput});
    put_inst(w, kOpVariable, {7, 11, kStorageInput});
    put_inst(w, kOpVariable, {7, 12, kStorageInput});
    for (std::uint32_t i = 0; i < num_raw; ++i) put_inst(w, kOpVariable, {7, 23 + i, kStorageInput});

    put_inst(w, kOpFunction, {1, 13, 0, 2});
    put_inst(w, 248, {14});
    put_inst(w, kOpLoad, {4, 15, 10});
    put_inst(w, kOpLoad, {5, 16, 11});
    put_inst(w, kOpLoad, {5, 17, 12});
    put_inst(w, kOpCompositeExtract, {3, 18, 16, 0});  // lo.x = 0.25
    put_inst(w, kOpCompositeExtract, {3, 19, 17, 0});  // hi.x = 0.75
    put_inst(w, kOpCompositeExtract, {3, 20, 15, 1});  // bary.y = i
    put_inst(w, kOpCompositeExtract, {3, 21, 15, 2});  // bary.z = j
    put_inst(w, kOpCompositeConstruct, {5, 22, 18, 19, 20, 21});
    put_inst(w, kOpStore, {9, 22});
    put_inst(w, kOpReturn, {});
    put_inst(w, kOpFunctionEnd, {});
    return w;
}

// The VS of the computed-I/J draws, from VertexIndex (0, 1, 2): the positions of make_bary_vs(true, ...) (w = 1, 2,
// 4), Location 0 = vec4(0.5, 0.25, 0.75, 1.0) at every vertex, and Location 1 = a different value per vertex:
// (1, 0, 0.5, 0.25), (0, 1, 0.25, 0.5), (0.5, 0.25, 1, 1).
std::vector<std::uint32_t> make_ij_vs() {
    std::uint32_t next_id = 1;
    const auto alloc = [&]() { return next_id++; };

    const std::uint32_t id_void = alloc();
    const std::uint32_t id_fn = alloc();
    const std::uint32_t id_float = alloc();
    const std::uint32_t id_v4float = alloc();
    const std::uint32_t id_int = alloc();
    const std::uint32_t id_bool = alloc();
    const std::uint32_t id_ptr_in_int = alloc();
    const std::uint32_t id_ptr_out_v4 = alloc();

    const std::uint32_t id_f0 = alloc();     //  0.0
    const std::uint32_t id_f1 = alloc();     //  1.0
    const std::uint32_t id_fm1 = alloc();    // -1.0
    const std::uint32_t id_f2 = alloc();     //  2.0
    const std::uint32_t id_f4 = alloc();     //  4.0
    const std::uint32_t id_f6 = alloc();     //  6.0
    const std::uint32_t id_fm2 = alloc();    // -2.0
    const std::uint32_t id_fm4 = alloc();    // -4.0
    const std::uint32_t id_f12 = alloc();    // 12.0
    const std::uint32_t id_f025 = alloc();   //  0.25
    const std::uint32_t id_f05 = alloc();    //  0.50
    const std::uint32_t id_f075 = alloc();   //  0.75

    const std::uint32_t id_p0 = alloc();
    const std::uint32_t id_p1 = alloc();
    const std::uint32_t id_p2 = alloc();
    const std::uint32_t id_attr0_val = alloc();
    const std::uint32_t id_c0 = alloc();
    const std::uint32_t id_c1 = alloc();
    const std::uint32_t id_c2 = alloc();
    const std::uint32_t id_i0 = alloc();
    const std::uint32_t id_i1 = alloc();

    const std::uint32_t id_vid_var = alloc();
    const std::uint32_t id_pos_var = alloc();
    const std::uint32_t id_out0_var = alloc();
    const std::uint32_t id_out1_var = alloc();

    const std::uint32_t id_main = alloc();
    const std::uint32_t id_entry = alloc();
    const std::uint32_t id_vid = alloc();
    const std::uint32_t id_is0 = alloc();
    const std::uint32_t id_is1 = alloc();
    const std::uint32_t id_p12 = alloc();
    const std::uint32_t id_pos = alloc();
    const std::uint32_t id_c12 = alloc();
    const std::uint32_t id_col = alloc();

    std::vector<std::uint32_t> w = {kSpirvMagic, 0x00010500u, 0, next_id, 0};
    put_inst(w, kOpCapability, {1});  // Shader
    put_inst(w, kOpMemoryModel, {0, 1});
    put_inst(w, kOpEntryPoint, {kExecModelVertex, id_main, 0x6e69616du, 0x00000000u,
                                id_vid_var, id_pos_var, id_out0_var, id_out1_var});

    put_inst(w, kOpDecorate, {id_vid_var, 11, 42});  // BuiltIn VertexIndex
    put_inst(w, kOpDecorate, {id_pos_var, 11, 0});   // BuiltIn Position
    put_inst(w, kOpDecorate, {id_out0_var, kDecorationLocation, 0});
    put_inst(w, kOpDecorate, {id_out1_var, kDecorationLocation, 1});

    put_inst(w, 19, {id_void});
    put_inst(w, 33, {id_fn, id_void});
    put_inst(w, kOpTypeFloat, {id_float, 32});
    put_inst(w, kOpTypeVector, {id_v4float, id_float, 4});
    put_inst(w, kOpTypeInt, {id_int, 32, 1});
    put_inst(w, 20, {id_bool});
    put_inst(w, kOpTypePointer, {id_ptr_in_int, kStorageInput, id_int});
    put_inst(w, kOpTypePointer, {id_ptr_out_v4, kStorageOutput, id_v4float});

    put_inst(w, 43, {id_float, id_f0, 0x00000000u});
    put_inst(w, 43, {id_float, id_f1, 0x3f800000u});
    put_inst(w, 43, {id_float, id_fm1, 0xbf800000u});
    put_inst(w, 43, {id_float, id_f2, 0x40000000u});
    put_inst(w, 43, {id_float, id_f4, 0x40800000u});
    put_inst(w, 43, {id_float, id_f6, 0x40c00000u});
    put_inst(w, 43, {id_float, id_fm2, 0xc0000000u});
    put_inst(w, 43, {id_float, id_fm4, 0xc0800000u});
    put_inst(w, 43, {id_float, id_f12, 0x41400000u});
    put_inst(w, 43, {id_float, id_f025, 0x3e800000u});
    put_inst(w, 43, {id_float, id_f05, 0x3f000000u});
    put_inst(w, 43, {id_float, id_f075, 0x3f400000u});

    put_inst(w, 44, {id_v4float, id_p0, id_fm1, id_fm1, id_f0, id_f1});
    put_inst(w, 44, {id_v4float, id_p1, id_f6, id_fm2, id_f0, id_f2});
    put_inst(w, 44, {id_v4float, id_p2, id_fm4, id_f12, id_f0, id_f4});
    put_inst(w, 44, {id_v4float, id_attr0_val, id_f05, id_f025, id_f075, id_f1});
    put_inst(w, 44, {id_v4float, id_c0, id_f1, id_f0, id_f05, id_f025});
    put_inst(w, 44, {id_v4float, id_c1, id_f0, id_f1, id_f025, id_f05});
    put_inst(w, 44, {id_v4float, id_c2, id_f05, id_f025, id_f1, id_f1});
    put_inst(w, 43, {id_int, id_i0, 0u});
    put_inst(w, 43, {id_int, id_i1, 1u});

    put_inst(w, kOpVariable, {id_ptr_in_int, id_vid_var, kStorageInput});
    put_inst(w, kOpVariable, {id_ptr_out_v4, id_pos_var, kStorageOutput});
    put_inst(w, kOpVariable, {id_ptr_out_v4, id_out0_var, kStorageOutput});
    put_inst(w, kOpVariable, {id_ptr_out_v4, id_out1_var, kStorageOutput});

    put_inst(w, kOpFunction, {id_void, id_main, 0, id_fn});
    put_inst(w, 248, {id_entry});
    put_inst(w, kOpLoad, {id_int, id_vid, id_vid_var});
    put_inst(w, 170, {id_bool, id_is0, id_vid, id_i0});      // OpIEqual vid 0
    put_inst(w, 170, {id_bool, id_is1, id_vid, id_i1});      // OpIEqual vid 1
    put_inst(w, 169, {id_v4float, id_p12, id_is1, id_p1, id_p2});
    put_inst(w, 169, {id_v4float, id_pos, id_is0, id_p0, id_p12});
    put_inst(w, 169, {id_v4float, id_c12, id_is1, id_c1, id_c2});
    put_inst(w, 169, {id_v4float, id_col, id_is0, id_c0, id_c12});
    put_inst(w, kOpStore, {id_pos_var, id_pos});
    put_inst(w, kOpStore, {id_out0_var, id_attr0_val});
    put_inst(w, kOpStore, {id_out1_var, id_col});
    put_inst(w, kOpReturn, {});
    put_inst(w, kOpFunctionEnd, {});
    return w;
}

// The weights the computed-I/J PS gives the three vertices' values: I = 0.5 and J = 0.125 in
// P0 + I * (P1 - P0) + J * (P2 - P0), so 0.375, 0.5 and 0.125.
constexpr float kIjWeights[3] = {0.375f, 0.5f, 0.125f};
constexpr float kIjValues[3][4] = {{1.0f, 0.0f, 0.5f, 0.25f}, {0.0f, 1.0f, 0.25f, 0.5f}, {0.5f, 0.25f, 1.0f, 1.0f}};

// A PS in the shape rdna2spirv.py --bary-fallback gives a computed I/J pair: the attribute is a Flat vec4[3] at
// `pv_base` (element k = vertex k) and the raw barycentrics a vec3 at `bary_loc`. comb = the kIjWeights sum of the
// three elements. Writes vec4(comb.rgb, bary.y), or with `with_attr0` (an ordinary vec4 Input at Location 0)
// vec4(comb.r, comb.g, bary.y, attr0.z).
std::vector<std::uint32_t> make_ij_ps(std::uint32_t pv_base, std::uint32_t bary_loc, bool with_attr0) {
    // 1=void, 2=fn, 3=float, 4=v3float, 5=v4float, 6=uint, 7=uint_3, 8=arr, 9=ptr_in_v3, 10=ptr_in_v4,
    // 11=ptr_in_arr, 12=ptr_out_v4, 13=int, 14..16=int 0..2, 17..19=weights, 20=out0, 21=in_pv, 22=in_bary,
    // 23=in_attr0, 24=main, 25=entry, 26..31=element pointers and values, 32..34=weighted, 35=sum01, 36=comb,
    // 37=bary, 38=comb.r, 39=comb.g, 40=comb.b, 41=bary.y, 42=attr0, 43=attr0.z, 44=rgba
    constexpr std::uint32_t kBound = 45;
    std::vector<std::uint32_t> w = {kSpirvMagic, 0x00010500u, 0, kBound, 0};
    put_inst(w, kOpCapability, {1});
    put_inst(w, kOpMemoryModel, {0, 1});
    std::vector<std::uint32_t> ep = {kExecModelFragment, 24, 0x6e69616du, 0x00000000u, 20, 21, 22};
    if (with_attr0) ep.push_back(23);
    put_inst_vec(w, kOpEntryPoint, ep);
    put_inst(w, kOpExecutionMode, {24, 7});  // OriginUpperLeft

    put_inst(w, kOpDecorate, {20, kDecorationLocation, 0});
    put_inst(w, kOpDecorate, {21, kDecorationLocation, pv_base});
    put_inst(w, kOpDecorate, {21, 14});  // Flat
    put_inst(w, kOpDecorate, {22, kDecorationLocation, bary_loc});
    if (with_attr0) put_inst(w, kOpDecorate, {23, kDecorationLocation, 0});

    put_inst(w, 19, {1});
    put_inst(w, 33, {2, 1});
    put_inst(w, kOpTypeFloat, {3, 32});
    put_inst(w, kOpTypeVector, {4, 3, 3});
    put_inst(w, kOpTypeVector, {5, 3, 4});
    put_inst(w, kOpTypeInt, {6, 32, 0});
    put_inst(w, 43, {6, 7, 3u});
    put_inst(w, 28, {8, 5, 7});
    put_inst(w, kOpTypePointer, {9, kStorageInput, 4});
    put_inst(w, kOpTypePointer, {10, kStorageInput, 5});
    put_inst(w, kOpTypePointer, {11, kStorageInput, 8});
    put_inst(w, kOpTypePointer, {12, kStorageOutput, 5});
    put_inst(w, kOpTypeInt, {13, 32, 1});
    put_inst(w, 43, {13, 14, 0u});
    put_inst(w, 43, {13, 15, 1u});
    put_inst(w, 43, {13, 16, 2u});
    put_inst(w, 43, {3, 17, 0x3ec00000u});  // 0.375
    put_inst(w, 43, {3, 18, 0x3f000000u});  // 0.5
    put_inst(w, 43, {3, 19, 0x3e000000u});  // 0.125

    put_inst(w, kOpVariable, {12, 20, kStorageOutput});
    put_inst(w, kOpVariable, {11, 21, kStorageInput});
    put_inst(w, kOpVariable, {9, 22, kStorageInput});
    if (with_attr0) put_inst(w, kOpVariable, {10, 23, kStorageInput});

    put_inst(w, kOpFunction, {1, 24, 0, 2});
    put_inst(w, 248, {25});
    for (std::uint32_t k = 0; k < 3; ++k) {
        put_inst(w, kOpAccessChain, {10, 26 + 2 * k, 21, 14 + k});
        put_inst(w, kOpLoad, {5, 27 + 2 * k, 26 + 2 * k});
        put_inst(w, 142, {5, 32 + k, 27 + 2 * k, 17 + k});  // OpVectorTimesScalar
    }
    put_inst(w, 129, {5, 35, 32, 33});  // OpFAdd
    put_inst(w, 129, {5, 36, 35, 34});
    put_inst(w, kOpLoad, {4, 37, 22});
    put_inst(w, kOpCompositeExtract, {3, 38, 36, 0});
    put_inst(w, kOpCompositeExtract, {3, 39, 36, 1});
    put_inst(w, kOpCompositeExtract, {3, 40, 36, 2});
    put_inst(w, kOpCompositeExtract, {3, 41, 37, 1});
    if (with_attr0) {
        put_inst(w, kOpLoad, {5, 42, 23});
        put_inst(w, kOpCompositeExtract, {3, 43, 42, 2});
        put_inst(w, kOpCompositeConstruct, {5, 44, 38, 39, 41, 43});
    } else {
        put_inst(w, kOpCompositeConstruct, {5, 44, 38, 39, 40, 41});
    }
    put_inst(w, kOpStore, {20, 44});
    put_inst(w, kOpReturn, {});
    put_inst(w, kOpFunctionEnd, {});
    return w;
}

int to_unorm8(float v) {
    if (v <= 0.0f) return 0;
    if (v >= 1.0f) return 255;
    return int(std::lround(v * 255.0f));
}

}  // namespace

int main() {
    vktest::Device d;
    const auto status = d.initialize();
    if (status == vktest::DeviceStatus::unavailable) {
        std::fprintf(stderr, "SKIP: no usable Vulkan 1.3 device\n");
        return 77;
    }
    EXPECT(status == vktest::DeviceStatus::ready);
    if (!d.enabled.geometryShader) {
        std::printf("SKIP: no geometryShader on this GPU\n");
        return 77;
    }

    VkPhysicalDeviceProperties props{};
    d.instance_fn<PFN_vkGetPhysicalDeviceProperties>("vkGetPhysicalDeviceProperties")(d.physical, &props);
    const std::uint32_t max_gs_in = props.limits.maxGeometryInputComponents;
    const std::uint32_t max_gs_out = props.limits.maxGeometryOutputComponents;
    std::printf("GPU limits: maxGeometryInputComponents=%u, maxGeometryOutputComponents=%u\n", max_gs_in, max_gs_out);

    const auto vs_persp_prog = fssm_program({0x5653ba01, 0xbf810000});
    const auto vs_clip_prog = fssm_program({0x5653ba02, 0xbf810000});
    const auto vs_16p_prog = fssm_program({0x5653ba03, 0xbf810000});
    const auto vs_10p_prog = fssm_program({0x5653ba04, 0xbf810000});
    const auto vs_loc16_prog = fssm_program({0x5653ba05, 0xbf810000});
    const auto gs_dummy_prog = fssm_program({0x4753ba01, 0xbf810000});
    const auto ps_bary_prog = fssm_program({0x5053ba01, 0xbf810000});
    const auto ps_f16_prog = fssm_program({0x5053ba02, 0xbf810000});
    const auto ps_f16_low_prog = fssm_program({0x5053ba03, 0xbf810000});
    const auto ps_pruned_prog = fssm_program({0x5053ba04, 0xbf810000});
    const auto ps_loc16_prog = fssm_program({0x5053ba05, 0xbf810000});
    const auto vs_ij_prog = fssm_program({0x5653ba06, 0xbf810000});
    const auto ps_ij_prog = fssm_program({0x5053ba06, 0xbf810000});
    const auto ps_ij_reloc_prog = fssm_program({0x5053ba07, 0xbf810000});
    const auto ps_ij_nomap_prog = fssm_program({0x5053ba08, 0xbf810000});

    const auto vs_persp_c = shader_container(vs_persp_prog);
    const auto vs_clip_c = shader_container(vs_clip_prog);
    const auto vs_16p_c = shader_container(vs_16p_prog);
    const auto vs_10p_c = shader_container(vs_10p_prog);
    const auto vs_loc16_c = shader_container(vs_loc16_prog);
    const auto gs_dummy_c = shader_container(gs_dummy_prog);
    const auto ps_bary_c = shader_container(ps_bary_prog);
    const auto ps_f16_c = shader_container(ps_f16_prog);
    const auto ps_f16_low_c = shader_container(ps_f16_low_prog);
    const auto ps_pruned_c = shader_container(ps_pruned_prog);
    const auto ps_loc16_c = shader_container(ps_loc16_prog);
    const auto vs_ij_c = shader_container(vs_ij_prog);
    const auto ps_ij_c = shader_container(ps_ij_prog);
    const auto ps_ij_reloc_c = shader_container(ps_ij_reloc_prog);
    const auto ps_ij_nomap_c = shader_container(ps_ij_nomap_prog);

    const auto vs_persp_spv = make_bary_vs(true, false);
    const auto ps_bary_spv = make_bary_ps(1);
    std::vector<std::uint32_t> gs_dummy_spv;
    EXPECT(leonida::build_bary_fallback_gs(vs_persp_spv, ps_bary_spv, 1, gs_dummy_spv));

    const auto lib_dir = std::filesystem::temp_directory_path() / "leonida_bary_fallback_gpu_test";
    std::filesystem::create_directories(lib_dir);
    write_spv(lib_dir / "vs_persp.spv", vs_persp_spv);
    write_spv(lib_dir / "vs_clip.spv", make_bary_vs(false, true));
    write_spv(lib_dir / "vs_16p.spv", make_nparam_vs(16, true));
    write_spv(lib_dir / "vs_10p.spv", make_nparam_vs(10, false));
    write_spv(lib_dir / "vs_loc16.spv", make_bary_vs(false, false, 16));
    write_spv(lib_dir / "gs_dummy.spv", gs_dummy_spv);
    write_spv(lib_dir / "ps_bary.spv", ps_bary_spv);
    write_spv(lib_dir / "ps_bary_f16.spv", make_bary_f16_ps(14, 14, 15, 16));
    write_spv(lib_dir / "ps_bary_f16_low.spv", make_bary_f16_ps(8, 12, 9, 10));
    write_spv(lib_dir / "ps_bary_pruned.spv", make_bary_pruned_ps());
    write_spv(lib_dir / "ps_loc16.spv", make_bary_ps(1, 16));
    write_spv(lib_dir / "vs_ij.spv", make_ij_vs());
    write_spv(lib_dir / "ps_ij.spv", make_ij_ps(0, 3, false));
    write_spv(lib_dir / "ps_ij_reloc.spv", make_ij_ps(4, 1, true));
    write_spv(lib_dir / "ps_ij_nomap.spv", make_ij_ps(0, 3, false));
    {
        std::ofstream idx(lib_dir / "index.json", std::ios::binary | std::ios::trunc);
        idx << "{\"vs_persp\": {\"stage\": \"vs\", \"fssm_sha256\": \"" << leonida::sha256_hex(vs_persp_prog)
            << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"input_signature\": []},\n"
               " \"vs_clip\": {\"stage\": \"vs\", \"fssm_sha256\": \"" << leonida::sha256_hex(vs_clip_prog)
            << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"input_signature\": []},\n"
               " \"vs_16p\": {\"stage\": \"vs\", \"fssm_sha256\": \"" << leonida::sha256_hex(vs_16p_prog)
            << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"input_signature\": [], "
               "\"f16_halves\": {\"0\": [28, 29], \"1\": [26, 27], \"2\": [24, 25], \"3\": [22, 23], "
               "\"4\": [20, 21], \"5\": [18, 19], \"6\": [16, 17]}},\n"
               " \"vs_10p\": {\"stage\": \"vs\", \"fssm_sha256\": \"" << leonida::sha256_hex(vs_10p_prog)
            << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"input_signature\": [], \"f16_halves\": {}},\n"
               " \"vs_loc16\": {\"stage\": \"vs\", \"fssm_sha256\": \"" << leonida::sha256_hex(vs_loc16_prog)
            << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"input_signature\": []},\n"
               " \"gs_dummy\": {\"stage\": \"gs\", \"fssm_sha256\": \"" << leonida::sha256_hex(gs_dummy_prog)
            << "\", \"bindings\": [], \"push_constant_bytes\": 144},\n"
               " \"ps_bary\": {\"stage\": \"ps\", \"fssm_sha256\": \"" << leonida::sha256_hex(ps_bary_prog)
            << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"outputs\": [0], \"inputs\": [0], "
               "\"bary_fallback_input\": {\"location\": 1}},\n"
               " \"ps_bary_f16\": {\"stage\": \"ps\", \"fssm_sha256\": \"" << leonida::sha256_hex(ps_f16_prog)
            << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"outputs\": [0], "
               "\"bary_fallback_input\": {\"location\": 14}, \"f16_halves\": {\"7\": [15, 16]}},\n"
               " \"vs_ij\": {\"stage\": \"vs\", \"fssm_sha256\": \"" << leonida::sha256_hex(vs_ij_prog)
            << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"input_signature\": []},\n"
               // Attribute 1 as a vec4[3] at Locations 0..2, barycentrics at 3 (the layout rdna2spirv gives a PS
               // whose only input is the computed pair's attribute).
               " \"ps_ij\": {\"stage\": \"ps\", \"fssm_sha256\": \"" << leonida::sha256_hex(ps_ij_prog)
            << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"outputs\": [0], \"inputs\": [1], "
               "\"bary_fallback_input\": {\"location\": 3, \"per_vertex\": {\"1\": 0}}},\n"
               // The block relocated to 4..6 next to an ordinary input at 0, barycentrics at 1 (the VS output
               // location the block reads).
               " \"ps_ij_reloc\": {\"stage\": \"ps\", \"fssm_sha256\": \""
            << leonida::sha256_hex(ps_ij_reloc_prog)
            << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"outputs\": [0], \"inputs\": [0, 1], "
               "\"bary_fallback_input\": {\"location\": 1, \"per_vertex\": {\"1\": 4}}},\n"
               " \"ps_ij_nomap\": {\"stage\": \"ps\", \"fssm_sha256\": \""
            << leonida::sha256_hex(ps_ij_nomap_prog)
            << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"outputs\": [0], \"inputs\": [1], "
               "\"bary_fallback_input\": {\"location\": 3}},\n"
               " \"ps_bary_f16_low\": {\"stage\": \"ps\", \"fssm_sha256\": \"" << leonida::sha256_hex(ps_f16_low_prog)
            << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"outputs\": [0], "
               "\"bary_fallback_input\": {\"location\": 12}, \"f16_halves\": {\"7\": [9, 10]}},\n"
               " \"ps_bary_pruned\": {\"stage\": \"ps\", \"fssm_sha256\": \"" << leonida::sha256_hex(ps_pruned_prog)
            << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"outputs\": [0], "
               "\"bary_fallback_input\": {\"location\": 14}},\n"
               " \"ps_loc16\": {\"stage\": \"ps\", \"fssm_sha256\": \"" << leonida::sha256_hex(ps_loc16_prog)
            << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"outputs\": [0], "
               "\"bary_fallback_input\": {\"location\": 1}}}";
    }

    constexpr UINT kSize = 16;
    {
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = d.instance_proc;
        info.instance = d.instance;
        info.physical = d.physical;
        info.device = d.device;
        info.queue = d.queue;
        info.queue_family = d.family;
        info.buffer_device_address = true;
        info.features = d.enabled;
        info.subgroup = d.subgroup;
        info.shader_dir = lib_dir.string();

        ID3D12Device* inner = nullptr;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&inner))));
        Ref<ID3D12Device> device;
        EXPECT(SUCCEEDED(rt::d3d12_xs::create_device_proxy(inner, __uuidof(ID3D12Device),
                                                           reinterpret_cast<void**>(&device.p))));
        rt::d3d12::set_device_outer(inner, device.p);

        D3D12_VERSIONED_ROOT_SIGNATURE_DESC root_desc{};
        root_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
        const auto root_blob = serialize_root_signature(root_desc);
        EXPECT(!root_blob.empty());
        Ref<ID3D12RootSignature> root;
        EXPECT(SUCCEEDED(device->CreateRootSignature(0, root_blob.data(), root_blob.size(), IID_PPV_ARGS(&root.p))));

        D3D12_COMMAND_QUEUE_DESC qd{};
        qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        Ref<ID3D12CommandQueue> queue;
        EXPECT(SUCCEEDED(device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue.p))));
        Ref<ID3D12CommandAllocator> alloc;
        EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc.p))));
        Ref<ID3D12GraphicsCommandList> list;
        EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.p, nullptr,
                                                   IID_PPV_ARGS(&list.p))));
        EXPECT(SUCCEEDED(list->Close()));

        Ref<ID3D12Fence> fence;
        EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));

        D3D12_HEAP_PROPERTIES default_hp{D3D12_HEAP_TYPE_DEFAULT};
        D3D12_RESOURCE_DESC rt_desc{};
        rt_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        rt_desc.Width = kSize;
        rt_desc.Height = kSize;
        rt_desc.DepthOrArraySize = 1;
        rt_desc.MipLevels = 1;
        rt_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        rt_desc.SampleDesc.Count = 1;
        rt_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        Ref<ID3D12Resource> rt_res;
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_hp, D3D12_HEAP_FLAG_NONE, &rt_desc,
                                                         D3D12_RESOURCE_STATE_RENDER_TARGET, nullptr,
                                                         IID_PPV_ARGS(&rt_res.p))));

        D3D12_DESCRIPTOR_HEAP_DESC rtv_hd{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
        Ref<ID3D12DescriptorHeap> rtv_heap;
        EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&rtv_hd, IID_PPV_ARGS(&rtv_heap.p))));
        const D3D12_CPU_DESCRIPTOR_HANDLE rtv_h{
            SIZE_T(reinterpret_cast<XboxHeapHandleFn>(slot(rtv_heap.p, 9))(rtv_heap.p))};
        device->CreateRenderTargetView(rt_res.p, nullptr, rtv_h);

        D3D12_HEAP_PROPERTIES rb_hp{D3D12_HEAP_TYPE_READBACK};
        const auto rb_desc = buffer_desc(kSize * 256);
        Ref<ID3D12Resource> rb_res;
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&rb_hp, D3D12_HEAP_FLAG_NONE, &rb_desc,
                                                         D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                         IID_PPV_ARGS(&rb_res.p))));

        const auto make_pso_desc = [&](const std::vector<std::uint8_t>& vs, const std::vector<std::uint8_t>& ps) {
            D3D12_GRAPHICS_PIPELINE_STATE_DESC pd{};
            pd.pRootSignature = root.p;
            pd.VS = {vs.data(), vs.size()};
            pd.PS = {ps.data(), ps.size()};
            pd.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
            pd.SampleMask = 0xffffffffu;
            pd.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
            pd.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
            pd.RasterizerState.DepthClipEnable = TRUE;
            pd.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
            pd.NumRenderTargets = 1;
            pd.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
            pd.SampleDesc.Count = 1;
            return pd;
        };

        UINT64 fence_val = 0;
        const auto draw_and_readback = [&](ID3D12PipelineState* pso, std::vector<std::uint8_t>& out_bytes) -> bool {
            if (FAILED(alloc->Reset()) || FAILED(list->Reset(alloc.p, pso))) return false;
            const float clear_col[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            list->ClearRenderTargetView(rtv_h, clear_col, 0, nullptr);
            list->OMSetRenderTargets(1, &rtv_h, FALSE, nullptr);
            const D3D12_VIEWPORT vp = {0.0f, 0.0f, float(kSize), float(kSize), 0.0f, 1.0f};
            const D3D12_RECT sc = {0, 0, LONG(kSize), LONG(kSize)};
            list->RSSetViewports(1, &vp);
            list->RSSetScissorRects(1, &sc);
            list->SetGraphicsRootSignature(root.p);
            list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            list->DrawInstanced(3, 1, 0, 0);

            const D3D12_RESOURCE_BARRIER to_copy =
                transition(rt_res.p, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
            list->ResourceBarrier(1, &to_copy);
            D3D12_TEXTURE_COPY_LOCATION dst{};
            dst.pResource = rb_res.p;
            dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
            dst.PlacedFootprint = {0, {DXGI_FORMAT_R8G8B8A8_UNORM, kSize, kSize, 1, 256}};
            D3D12_TEXTURE_COPY_LOCATION src{};
            src.pResource = rt_res.p;
            src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
            const D3D12_RESOURCE_BARRIER to_rt =
                transition(rt_res.p, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET);
            list->ResourceBarrier(1, &to_rt);
            if (FAILED(list->Close())) return false;

            ID3D12CommandList* cl = list.p;
            queue->ExecuteCommandLists(1, &cl);
            if (rt::d3d12::last_execute_result(queue.p) != leonida::ReplayResult::success) return false;
            ++fence_val;
            if (FAILED(queue->Signal(fence.p, fence_val))) return false;
            if (FAILED(fence->SetEventOnCompletion(fence_val, nullptr))) return false;

            void* mapped = nullptr;
            const D3D12_RANGE read_range = {0, kSize * 256};
            if (FAILED(rb_res->Map(0, &read_range, &mapped)) || !mapped) return false;
            out_bytes.assign(static_cast<const std::uint8_t*>(mapped),
                             static_cast<const std::uint8_t*>(mapped) + kSize * 256);
            rb_res->Unmap(0, nullptr);
            return true;
        };

        // 1. Perspective-correct barycentrics + pass-through vec4 attribute across all 16x16 pixels.
        {
            auto pd = make_pso_desc(vs_persp_c, ps_bary_c);
            Ref<ID3D12PipelineState> pso;
            EXPECT(SUCCEEDED(device->CreateGraphicsPipelineState(&pd, IID_PPV_ARGS(&pso.p))) && pso.p);
            std::vector<std::uint8_t> pixels;
            EXPECT(draw_and_readback(pso.p, pixels));
            for (UINT row = 0; row < kSize; ++row) {
                for (UINT col = 0; col < kSize; ++col) {
                    const float ndc_x = -1.0f + 2.0f * (float(col) + 0.5f) / float(kSize);
                    const float ndc_y = +1.0f - 2.0f * (float(row) + 0.5f) / float(kSize);
                    const float l1 = 0.25f * (ndc_x + 1.0f);
                    const float l2 = 0.25f * (ndc_y + 1.0f);
                    const float l0 = 1.0f - l1 - l2;
                    const float denom = l0 / 1.0f + l1 / 2.0f + l2 / 4.0f;
                    const int want_r = to_unorm8((l0 / 1.0f) / denom);
                    const int want_g = to_unorm8((l1 / 2.0f) / denom);
                    const int want_b = to_unorm8((l2 / 4.0f) / denom);
                    const auto* px = pixels.data() + row * 256 + col * 4;
                    if (std::abs(int(px[0]) - want_r) > 2 || std::abs(int(px[1]) - want_g) > 2 ||
                        std::abs(int(px[2]) - want_b) > 2 || std::abs(int(px[3]) - 191) > 1) {
                        std::fprintf(stderr, "persp (%u,%u): got (%u,%u,%u,%u) want (%d,%d,%d,191)\n",
                                     col, row, px[0], px[1], px[2], px[3], want_r, want_g, want_b);
                        return 1;
                    }
                }
            }
            const auto* mid = pixels.data() + (kSize / 2) * 256 + (kSize / 2) * 4;
            std::printf("ps_bary (perspective w=1,2,4): center=(%u,%u,%u,%u) OK\n",
                        mid[0], mid[1], mid[2], mid[3]);
        }

        // 2. VS ClipDistance[1] forwarding (clips ndc_x < 0 to clear color).
        if (d.enabled.shaderClipDistance) {
            auto pd = make_pso_desc(vs_clip_c, ps_bary_c);
            Ref<ID3D12PipelineState> pso;
            EXPECT(SUCCEEDED(device->CreateGraphicsPipelineState(&pd, IID_PPV_ARGS(&pso.p))) && pso.p);
            std::vector<std::uint8_t> pixels;
            EXPECT(draw_and_readback(pso.p, pixels));
            for (UINT row = 0; row < kSize; ++row) {
                for (UINT col = 0; col < kSize; ++col) {
                    const auto* px = pixels.data() + row * 256 + col * 4;
                    if (col < kSize / 2 - 1) {
                        EXPECT(px[0] == 0 && px[1] == 0 && px[2] == 0 && px[3] == 0);
                    } else if (col > kSize / 2) {
                        const float ndc_x = -1.0f + 2.0f * (float(col) + 0.5f) / float(kSize);
                        const float ndc_y = +1.0f - 2.0f * (float(row) + 0.5f) / float(kSize);
                        const float l1 = 0.25f * (ndc_x + 1.0f);
                        const float l2 = 0.25f * (ndc_y + 1.0f);
                        const float l0 = 1.0f - l1 - l2;
                        EXPECT(std::abs(int(px[0]) - to_unorm8(l0)) <= 2);
                        EXPECT(std::abs(int(px[1]) - to_unorm8(l1)) <= 2);
                        EXPECT(std::abs(int(px[2]) - to_unorm8(l2)) <= 2);
                        EXPECT(std::abs(int(px[3]) - 191) <= 1);
                    }
                }
            }
            std::printf("ps_bary + ClipDistance[1]: left half clipped, right half interpolated OK\n");
        }

        // 3a. Pruned unused PS Inputs (vs_16p exports 0..29, ps_bary_pruned declares 0..13 & 16..27, loads only 7 + bary 14):
        //     must succeed and draw accurately even on GPUs with maxGeometryInputComponents = 64 (e.g. RADV).
        {
            auto pd = make_pso_desc(vs_16p_c, ps_pruned_c);
            Ref<ID3D12PipelineState> pso;
            EXPECT(SUCCEEDED(device->CreateGraphicsPipelineState(&pd, IID_PPV_ARGS(&pso.p))) && pso.p);
            std::vector<std::uint8_t> pixels;
            EXPECT(draw_and_readback(pso.p, pixels));
            for (UINT row = 0; row < kSize; ++row) {
                for (UINT col = 0; col < kSize; ++col) {
                    const float ndc_x = -1.0f + 2.0f * (float(col) + 0.5f) / float(kSize);
                    const float ndc_y = +1.0f - 2.0f * (float(row) + 0.5f) / float(kSize);
                    const float l1 = 0.25f * (ndc_x + 1.0f);
                    const float l2 = 0.25f * (ndc_y + 1.0f);
                    const float l0 = 1.0f - l1 - l2;
                    const auto* px = pixels.data() + row * 256 + col * 4;
                    EXPECT(std::abs(int(px[0]) - to_unorm8(l0)) <= 2);
                    EXPECT(std::abs(int(px[1]) - to_unorm8(l1)) <= 2);
                    EXPECT(std::abs(int(px[2]) - to_unorm8(l2)) <= 2);
                    EXPECT(px[3] == 255);  // param7.w = 1.0 (unread params in vs_16p are 0.0)
                }
            }
            std::printf("ps_bary_pruned (25 unread PS inputs pruned; live Location 7 + bary 14): all 256 pixels OK\n");
        }

        // 3b. Combined --f16-interp-fallback + --bary-fallback within 64 geometry input components (10-param VS):
        //     runs on all GPUs including RADV (maxGeometryInputComponents = 64).
        {
            auto pd = make_pso_desc(vs_10p_c, ps_f16_low_c);
            Ref<ID3D12PipelineState> pso;
            EXPECT(SUCCEEDED(device->CreateGraphicsPipelineState(&pd, IID_PPV_ARGS(&pso.p))) && pso.p);
            std::vector<std::uint8_t> pixels;
            EXPECT(draw_and_readback(pso.p, pixels));
            for (UINT row = 0; row < kSize; ++row) {
                for (UINT col = 0; col < kSize; ++col) {
                    const float ndc_x = -1.0f + 2.0f * (float(col) + 0.5f) / float(kSize);
                    const float ndc_y = +1.0f - 2.0f * (float(row) + 0.5f) / float(kSize);
                    const int want_i = to_unorm8(0.25f * (ndc_x + 1.0f));
                    const int want_j = to_unorm8(0.25f * (ndc_y + 1.0f));
                    const auto* px = pixels.data() + row * 256 + col * 4;
                    EXPECT(std::abs(int(px[0]) - 64) <= 1);
                    EXPECT(std::abs(int(px[1]) - 191) <= 1);
                    EXPECT(std::abs(int(px[2]) - want_i) <= 2);
                    EXPECT(std::abs(int(px[3]) - want_j) <= 2);
                }
            }
            std::printf("ps_bary_f16_low (10-param VS + f16 relocation <=14 + bary GS): all 256 pixels OK\n");
        }

        // 3c. Combined --f16-interp-fallback + --bary-fallback on 16-param VS (relocated f16 halves at [28, 29],
        //     requiring maxGeometryInputComponents >= 120 and maxGeometryOutputComponents >= 120).
        {
            auto pd = make_pso_desc(vs_16p_c, ps_f16_c);
            Ref<ID3D12PipelineState> pso;
            const HRESULT hr = device->CreateGraphicsPipelineState(&pd, IID_PPV_ARGS(&pso.p));
            if (max_gs_in < 120 || max_gs_out < 120) {
                EXPECT(FAILED(hr) && !pso.p);
                std::printf(
                    "SKIP ps_bary_f16 16-param draw: maxGeometryInputComponents=%u, maxGeometryOutputComponents=%u "
                    "(< 120); refused cleanly\n",
                    max_gs_in, max_gs_out);
            } else {
                EXPECT(SUCCEEDED(hr) && pso.p);
                std::vector<std::uint8_t> pixels;
                EXPECT(draw_and_readback(pso.p, pixels));
                for (UINT row = 0; row < kSize; ++row) {
                    for (UINT col = 0; col < kSize; ++col) {
                        const float ndc_x = -1.0f + 2.0f * (float(col) + 0.5f) / float(kSize);
                        const float ndc_y = +1.0f - 2.0f * (float(row) + 0.5f) / float(kSize);
                        const int want_i = to_unorm8(0.25f * (ndc_x + 1.0f));
                        const int want_j = to_unorm8(0.25f * (ndc_y + 1.0f));
                        const auto* px = pixels.data() + row * 256 + col * 4;
                        EXPECT(std::abs(int(px[0]) - 64) <= 1);
                        EXPECT(std::abs(int(px[1]) - 191) <= 1);
                        EXPECT(std::abs(int(px[2]) - want_i) <= 2);
                        EXPECT(std::abs(int(px[3]) - want_j) <= 2);
                    }
                }
                std::printf("ps_bary_f16 (16-param f16_halves relocation [28,29] + bary GS): all 256 pixels OK\n");
            }
        }

        // 5. Computed I/J pairs: the GS hands the PS the triangle's three values of VS output 1 (element k = vertex
        //    k), which the PS weights itself. Every pixel gets the same sum whatever its position; the barycentric
        //    channel still has to be the perspective-correct one of draw 1.
        {
            int want[4]{};
            for (int c = 0; c < 4; ++c) {
                want[c] = to_unorm8(kIjWeights[0] * kIjValues[0][c] + kIjWeights[1] * kIjValues[1][c] +
                                    kIjWeights[2] * kIjValues[2][c]);
            }
            // The nearest wrong answer: the same weights on any other order of the three vertices.
            int nearest = 255;
            const int orders[5][3] = {{0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
            for (const auto& o : orders) {
                int worst = 0;
                for (int c = 0; c < 2; ++c) {  // red and green: the channels both draws write
                    const int v = to_unorm8(kIjWeights[0] * kIjValues[o[0]][c] + kIjWeights[1] * kIjValues[o[1]][c] +
                                            kIjWeights[2] * kIjValues[o[2]][c]);
                    worst = (std::max)(worst, std::abs(v - want[c]));
                }
                nearest = (std::min)(nearest, worst);
            }
            EXPECT(nearest >= 15);

            const auto want_bary_y = [&](UINT col, UINT row) {
                const float ndc_x = -1.0f + 2.0f * (float(col) + 0.5f) / float(kSize);
                const float ndc_y = +1.0f - 2.0f * (float(row) + 0.5f) / float(kSize);
                const float l1 = 0.25f * (ndc_x + 1.0f);
                const float l2 = 0.25f * (ndc_y + 1.0f);
                const float l0 = 1.0f - l1 - l2;
                return to_unorm8((l1 / 2.0f) / (l0 / 1.0f + l1 / 2.0f + l2 / 4.0f));
            };

            // 5a. The block at Locations 0..2, barycentrics at 3: vec4(comb.rgb, bary.y).
            {
                auto pd = make_pso_desc(vs_ij_c, ps_ij_c);
                Ref<ID3D12PipelineState> pso;
                EXPECT(SUCCEEDED(device->CreateGraphicsPipelineState(&pd, IID_PPV_ARGS(&pso.p))) && pso.p);
                std::vector<std::uint8_t> pixels;
                EXPECT(draw_and_readback(pso.p, pixels));
                for (UINT row = 0; row < kSize; ++row) {
                    for (UINT col = 0; col < kSize; ++col) {
                        const auto* px = pixels.data() + row * 256 + col * 4;
                        const int by = want_bary_y(col, row);
                        if (std::abs(int(px[0]) - want[0]) > 1 || std::abs(int(px[1]) - want[1]) > 1 ||
                            std::abs(int(px[2]) - want[2]) > 1 || std::abs(int(px[3]) - by) > 2) {
                            std::fprintf(stderr, "ps_ij (%u,%u): got (%u,%u,%u,%u) want (%d,%d,%d,%d)\n", col, row,
                                         px[0], px[1], px[2], px[3], want[0], want[1], want[2], by);
                            return 1;
                        }
                    }
                }
                const auto* mid = pixels.data() + (kSize / 2) * 256 + (kSize / 2) * 4;
                std::printf("ps_ij (per-vertex block at 0..2): center=(%u,%u,%u,%u), all 256 pixels OK\n",
                            mid[0], mid[1], mid[2], mid[3]);
            }

            // 5b. The block relocated to 4..6, an ordinary input at 0, barycentrics at 1:
            //     vec4(comb.r, comb.g, bary.y, attr0.z).
            {
                auto pd = make_pso_desc(vs_ij_c, ps_ij_reloc_c);
                Ref<ID3D12PipelineState> pso;
                EXPECT(SUCCEEDED(device->CreateGraphicsPipelineState(&pd, IID_PPV_ARGS(&pso.p))) && pso.p);
                std::vector<std::uint8_t> pixels;
                EXPECT(draw_and_readback(pso.p, pixels));
                for (UINT row = 0; row < kSize; ++row) {
                    for (UINT col = 0; col < kSize; ++col) {
                        const auto* px = pixels.data() + row * 256 + col * 4;
                        const int by = want_bary_y(col, row);
                        if (std::abs(int(px[0]) - want[0]) > 1 || std::abs(int(px[1]) - want[1]) > 1 ||
                            std::abs(int(px[2]) - by) > 2 || std::abs(int(px[3]) - 191) > 1) {
                            std::fprintf(stderr, "ps_ij_reloc (%u,%u): got (%u,%u,%u,%u) want (%d,%d,%d,191)\n", col,
                                         row, px[0], px[1], px[2], px[3], want[0], want[1], by);
                            return 1;
                        }
                    }
                }
                std::printf("ps_ij_reloc (per-vertex block at 4..6 + ordinary input): all 256 pixels OK\n");
            }

            // 5c. The same PS with no per_vertex in its index entry: the vec4[3] input is refused, not mislinked.
            {
                auto pd = make_pso_desc(vs_ij_c, ps_ij_nomap_c);
                Ref<ID3D12PipelineState> bad;
                EXPECT(device->CreateGraphicsPipelineState(&pd, IID_PPV_ARGS(&bad.p)) == E_FAIL && !bad.p);
            }
        }

        // 4a & 4b. Refusals: non-triangle topology -> E_INVALIDARG; explicit GS -> E_NOTIMPL.
        {
            auto pd_lines = make_pso_desc(vs_persp_c, ps_bary_c);
            pd_lines.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE;
            Ref<ID3D12PipelineState> bad;
            EXPECT(device->CreateGraphicsPipelineState(&pd_lines, IID_PPV_ARGS(&bad.p)) == E_INVALIDARG && !bad.p);

            auto pd_gs = make_pso_desc(vs_persp_c, ps_bary_c);
            pd_gs.GS = {gs_dummy_c.data(), gs_dummy_c.size()};
            EXPECT(device->CreateGraphicsPipelineState(&pd_gs, IID_PPV_ARGS(&bad.p)) == E_NOTIMPL && !bad.p);
        }

        // 4c. Refusal when geometryShader is not enabled on VulkanDeviceInfo -> E_NOTIMPL.
        {
            rt::d3d12::VulkanDeviceInfo info_no_gs = info;
            info_no_gs.features.geometryShader = VK_FALSE;
            ID3D12Device* inner_no_gs = nullptr;
            EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info_no_gs, IID_PPV_ARGS(&inner_no_gs))));
            Ref<ID3D12Device> dev_no_gs;
            EXPECT(SUCCEEDED(rt::d3d12_xs::create_device_proxy(inner_no_gs, __uuidof(ID3D12Device),
                                                               reinterpret_cast<void**>(&dev_no_gs.p))));
            rt::d3d12::set_device_outer(inner_no_gs, dev_no_gs.p);
            Ref<ID3D12RootSignature> root2;
            EXPECT(SUCCEEDED(dev_no_gs->CreateRootSignature(0, root_blob.data(), root_blob.size(),
                                                            IID_PPV_ARGS(&root2.p))));
            auto pd = make_pso_desc(vs_persp_c, ps_bary_c);
            pd.pRootSignature = root2.p;
            Ref<ID3D12PipelineState> bad;
            EXPECT(dev_no_gs->CreateGraphicsPipelineState(&pd, IID_PPV_ARGS(&bad.p)) == E_NOTIMPL && !bad.p);
        }

        // 4d. Simulated RADV-like max_geometry_input_components = 64 on any GPU:
        //     - pruned high-location PS inputs (ps_bary_pruned) and 10-param f16 relocation (ps_bary_f16_low) succeed;
        //     - 16-param f16 relocation (ps_bary_f16) and live Location 16 (VUID-06272: 4*16+4 = 68 > 64) refuse cleanly.
        {
            rt::d3d12::VulkanDeviceInfo info_gs64 = info;
            info_gs64.max_geometry_input_components = 64;
            ID3D12Device* inner_gs64 = nullptr;
            EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info_gs64, IID_PPV_ARGS(&inner_gs64))));
            Ref<ID3D12Device> dev_gs64;
            EXPECT(SUCCEEDED(rt::d3d12_xs::create_device_proxy(inner_gs64, __uuidof(ID3D12Device),
                                                               reinterpret_cast<void**>(&dev_gs64.p))));
            rt::d3d12::set_device_outer(inner_gs64, dev_gs64.p);
            Ref<ID3D12RootSignature> root64;
            EXPECT(SUCCEEDED(dev_gs64->CreateRootSignature(0, root_blob.data(), root_blob.size(),
                                                           IID_PPV_ARGS(&root64.p))));

            auto pd_pruned = make_pso_desc(vs_16p_c, ps_pruned_c);
            pd_pruned.pRootSignature = root64.p;
            Ref<ID3D12PipelineState> pso_pruned;
            EXPECT(SUCCEEDED(dev_gs64->CreateGraphicsPipelineState(&pd_pruned, IID_PPV_ARGS(&pso_pruned.p))) &&
                   pso_pruned.p);

            auto pd_low = make_pso_desc(vs_10p_c, ps_f16_low_c);
            pd_low.pRootSignature = root64.p;
            Ref<ID3D12PipelineState> pso_low;
            EXPECT(SUCCEEDED(dev_gs64->CreateGraphicsPipelineState(&pd_low, IID_PPV_ARGS(&pso_low.p))) && pso_low.p);

            auto pd_16p = make_pso_desc(vs_16p_c, ps_f16_c);
            pd_16p.pRootSignature = root64.p;
            Ref<ID3D12PipelineState> bad;
            EXPECT(dev_gs64->CreateGraphicsPipelineState(&pd_16p, IID_PPV_ARGS(&bad.p)) == E_NOTIMPL && !bad.p);

            auto pd_loc16 = make_pso_desc(vs_loc16_c, ps_loc16_c);
            pd_loc16.pRootSignature = root64.p;
            EXPECT(dev_gs64->CreateGraphicsPipelineState(&pd_loc16, IID_PPV_ARGS(&bad.p)) == E_NOTIMPL && !bad.p);
            std::printf("max_geometry_input_components=64 checks (pruned OK, 10p f16 OK, 16p & loc16 refused) OK\n");
        }
    }

    std::error_code ec;
    std::filesystem::remove_all(lib_dir, ec);
    EXPECT(!d.validation || d.validation_errors() == 0);
    return 0;
}
