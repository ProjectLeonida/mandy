// Differential GPU and validation tests for D3D12_RASTERIZER_DESC::ForcedSampleCount (Target-Independent Rasterization).
// Compares the Vulkan D3D12 layer against native D3D12 on the layer's own GPU and against Microsoft WARP using
// hand-encoded SPIR-V 1.3 modules for the layer and runtime-compiled HLSL (d3dcompiler_47.dll) for the host devices.
// Set LEONIDA_VK_VALIDATION=1 to run under VK_LAYER_KHRONOS_validation when installed. Synthetic inputs only; no game data.
#include "d3d12_copy_compare.h"
#include "d3d12_test_shaders.h"
#include "host_adapter.h"
#include "host_wine.h"
#include "test_temp_dir.h"
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"
#include "../src/shader_library.h"

#include <d3dcompiler.h>
#include <dxgi1_4.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#define CHECK(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return false; } \
} while (0)

#define REQUIRE(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

constexpr UINT kWidth = 16;
constexpr UINT kHeight = 16;
constexpr UINT kPitch = 256;  // D3D12_TEXTURE_DATA_PITCH_ALIGNMENT

struct Point {
    double x, y;
};
using Triangle = std::array<Point, 3>;

// ---------------------------------------------------------------------------------------------------------------
// Minimal hand-encoded SPIR-V 1.3 modules matching the translator runtime contract (zero external shader compiler).

struct SpirvBuilder {
    std::uint32_t next_id = 1;
    std::vector<std::uint32_t> caps, mem, entries, modes, decos, types, defs;

    std::uint32_t id() { return next_id++; }

    static void op(std::vector<std::uint32_t>& out, std::uint32_t code, std::initializer_list<std::uint32_t> args) {
        out.push_back((std::uint32_t(args.size() + 1) << 16) | code);
        out.insert(out.end(), args.begin(), args.end());
    }

    static void append_str(std::vector<std::uint32_t>& out, const char* s) {
        const std::size_t bytes = std::strlen(s) + 1;
        const std::size_t words = (bytes + 3) / 4;
        const std::size_t at = out.size();
        out.resize(at + words, 0u);
        std::memcpy(&out[at], s, bytes);
    }

    static std::uint32_t f32_bits(float v) {
        std::uint32_t bits = 0;
        std::memcpy(&bits, &v, 4);
        return bits;
    }

    void entry_point(std::uint32_t model, std::uint32_t fn, const char* name,
                     std::initializer_list<std::uint32_t> interfaces) {
        std::vector<std::uint32_t> inst = {0u, model, fn};
        append_str(inst, name);
        inst.insert(inst.end(), interfaces.begin(), interfaces.end());
        inst[0] = (std::uint32_t(inst.size()) << 16) | 15u;  // OpEntryPoint
        entries.insert(entries.end(), inst.begin(), inst.end());
    }

    std::vector<std::uint32_t> finish() const {
        std::vector<std::uint32_t> out = {0x07230203u, 0x00010300u, 0u, next_id, 0u};
        for (const auto* s : {&caps, &mem, &entries, &modes, &decos, &types, &defs})
            out.insert(out.end(), s->begin(), s->end());
        return out;
    }
};

std::vector<std::uint32_t> build_vs_tir() {
    SpirvBuilder b;
    const auto main_id = b.id(), void_t = b.id(), f32_t = b.id(), vec2_t = b.id(), vec4_t = b.id(), u32_t = b.id();
    const auto c_u0 = b.id(), c_half = b.id(), c_one = b.id(), pv_t = b.id();
    const auto ptr_in_v2 = b.id(), ptr_out_pv = b.id(), ptr_out_v4 = b.id(), fn_t = b.id();
    const auto in_pos = b.id(), out_pv = b.id(), entry_lbl = b.id();
    const auto v = b.id(), x = b.id(), y = b.id(), pos = b.id(), dst = b.id();

    SpirvBuilder::op(b.caps, 17, {1});            // OpCapability Shader
    SpirvBuilder::op(b.mem, 14, {0, 1});          // OpMemoryModel Logical GLSL450
    b.entry_point(0, main_id, "main", {in_pos, out_pv});
    SpirvBuilder::op(b.decos, 71, {in_pos, 30, 0});   // OpDecorate Location 0
    SpirvBuilder::op(b.decos, 71, {pv_t, 2});         // OpDecorate Block
    SpirvBuilder::op(b.decos, 72, {pv_t, 0, 11, 0});  // OpMemberDecorate BuiltIn Position

    SpirvBuilder::op(b.types, 19, {void_t});
    SpirvBuilder::op(b.types, 22, {f32_t, 32});
    SpirvBuilder::op(b.types, 23, {vec2_t, f32_t, 2});
    SpirvBuilder::op(b.types, 23, {vec4_t, f32_t, 4});
    SpirvBuilder::op(b.types, 21, {u32_t, 32, 0});
    SpirvBuilder::op(b.types, 43, {u32_t, c_u0, 0u});
    SpirvBuilder::op(b.types, 43, {f32_t, c_half, SpirvBuilder::f32_bits(0.5f)});
    SpirvBuilder::op(b.types, 43, {f32_t, c_one, SpirvBuilder::f32_bits(1.0f)});
    SpirvBuilder::op(b.types, 30, {pv_t, vec4_t});
    SpirvBuilder::op(b.types, 32, {ptr_in_v2, 1, vec2_t});
    SpirvBuilder::op(b.types, 32, {ptr_out_pv, 3, pv_t});
    SpirvBuilder::op(b.types, 32, {ptr_out_v4, 3, vec4_t});
    SpirvBuilder::op(b.types, 33, {fn_t, void_t});
    SpirvBuilder::op(b.types, 59, {ptr_in_v2, in_pos, 1});
    SpirvBuilder::op(b.types, 59, {ptr_out_pv, out_pv, 3});

    SpirvBuilder::op(b.defs, 54, {void_t, main_id, 0, fn_t});
    SpirvBuilder::op(b.defs, 248, {entry_lbl});
    SpirvBuilder::op(b.defs, 61, {vec2_t, v, in_pos});
    SpirvBuilder::op(b.defs, 81, {f32_t, x, v, 0});
    SpirvBuilder::op(b.defs, 81, {f32_t, y, v, 1});
    SpirvBuilder::op(b.defs, 80, {vec4_t, pos, x, y, c_half, c_one});
    SpirvBuilder::op(b.defs, 65, {ptr_out_v4, dst, out_pv, c_u0});
    SpirvBuilder::op(b.defs, 62, {dst, pos});
    SpirvBuilder::op(b.defs, 253, {});
    SpirvBuilder::op(b.defs, 56, {});
    return b.finish();
}

std::vector<std::uint32_t> build_ps_empty() {
    SpirvBuilder b;
    const auto main_id = b.id(), void_t = b.id(), fn_t = b.id(), entry_lbl = b.id();
    SpirvBuilder::op(b.caps, 17, {1});
    SpirvBuilder::op(b.mem, 14, {0, 1});
    b.entry_point(4, main_id, "main", {});
    SpirvBuilder::op(b.modes, 16, {main_id, 7});  // OriginUpperLeft
    SpirvBuilder::op(b.types, 19, {void_t});
    SpirvBuilder::op(b.types, 33, {fn_t, void_t});
    SpirvBuilder::op(b.defs, 54, {void_t, main_id, 0, fn_t});
    SpirvBuilder::op(b.defs, 248, {entry_lbl});
    SpirvBuilder::op(b.defs, 253, {});
    SpirvBuilder::op(b.defs, 56, {});
    return b.finish();
}

std::vector<std::uint32_t> build_ps_color() {
    SpirvBuilder b;
    const auto main_id = b.id(), void_t = b.id(), f32_t = b.id(), vec4_t = b.id();
    const auto ptr_out_v4 = b.id(), fn_t = b.id();
    const auto c_zero = b.id(), c_quarter = b.id(), c_one = b.id(), c_color = b.id();
    const auto out_color = b.id(), entry_lbl = b.id();

    SpirvBuilder::op(b.caps, 17, {1});
    SpirvBuilder::op(b.mem, 14, {0, 1});
    b.entry_point(4, main_id, "main", {out_color});
    SpirvBuilder::op(b.modes, 16, {main_id, 7});
    SpirvBuilder::op(b.decos, 71, {out_color, 30, 0});

    SpirvBuilder::op(b.types, 19, {void_t});
    SpirvBuilder::op(b.types, 22, {f32_t, 32});
    SpirvBuilder::op(b.types, 23, {vec4_t, f32_t, 4});
    SpirvBuilder::op(b.types, 32, {ptr_out_v4, 3, vec4_t});
    SpirvBuilder::op(b.types, 33, {fn_t, void_t});
    SpirvBuilder::op(b.types, 43, {f32_t, c_zero, SpirvBuilder::f32_bits(0.0f)});
    SpirvBuilder::op(b.types, 43, {f32_t, c_quarter, SpirvBuilder::f32_bits(0.25f)});
    SpirvBuilder::op(b.types, 43, {f32_t, c_one, SpirvBuilder::f32_bits(1.0f)});
    SpirvBuilder::op(b.types, 44, {vec4_t, c_color, c_one, c_quarter, c_zero, c_one});
    SpirvBuilder::op(b.types, 59, {ptr_out_v4, out_color, 3});

    SpirvBuilder::op(b.defs, 54, {void_t, main_id, 0, fn_t});
    SpirvBuilder::op(b.defs, 248, {entry_lbl});
    SpirvBuilder::op(b.defs, 62, {out_color, c_color});
    SpirvBuilder::op(b.defs, 253, {});
    SpirvBuilder::op(b.defs, 56, {});
    return b.finish();
}

std::vector<std::uint32_t> build_ps_cov() {
    SpirvBuilder b;
    const auto main_id = b.id(), void_t = b.id(), u32_t = b.id(), c_0 = b.id(), c_1 = b.id();
    const auto arr_u32_1 = b.id(), ptr_in_arr = b.id(), ptr_in_u32 = b.id(), ptr_out_u32 = b.id(), fn_t = b.id();
    const auto in_mask = b.id(), out_mask = b.id(), entry_lbl = b.id(), elem_ptr = b.id(), mask = b.id();

    SpirvBuilder::op(b.caps, 17, {1});
    SpirvBuilder::op(b.mem, 14, {0, 1});
    b.entry_point(4, main_id, "main", {in_mask, out_mask});
    SpirvBuilder::op(b.modes, 16, {main_id, 7});
    SpirvBuilder::op(b.decos, 71, {in_mask, 11, 20});   // BuiltIn SampleMask
    SpirvBuilder::op(b.decos, 71, {out_mask, 30, 0});   // Location 0

    SpirvBuilder::op(b.types, 19, {void_t});
    SpirvBuilder::op(b.types, 21, {u32_t, 32, 0});
    SpirvBuilder::op(b.types, 43, {u32_t, c_0, 0u});
    SpirvBuilder::op(b.types, 43, {u32_t, c_1, 1u});
    SpirvBuilder::op(b.types, 28, {arr_u32_1, u32_t, c_1});
    SpirvBuilder::op(b.types, 32, {ptr_in_arr, 1, arr_u32_1});
    SpirvBuilder::op(b.types, 32, {ptr_in_u32, 1, u32_t});
    SpirvBuilder::op(b.types, 32, {ptr_out_u32, 3, u32_t});
    SpirvBuilder::op(b.types, 33, {fn_t, void_t});
    SpirvBuilder::op(b.types, 59, {ptr_in_arr, in_mask, 1});
    SpirvBuilder::op(b.types, 59, {ptr_out_u32, out_mask, 3});

    SpirvBuilder::op(b.defs, 54, {void_t, main_id, 0, fn_t});
    SpirvBuilder::op(b.defs, 248, {entry_lbl});
    SpirvBuilder::op(b.defs, 65, {ptr_in_u32, elem_ptr, in_mask, c_0});
    SpirvBuilder::op(b.defs, 61, {u32_t, mask, elem_ptr});
    SpirvBuilder::op(b.defs, 62, {out_mask, mask});
    SpirvBuilder::op(b.defs, 253, {});
    SpirvBuilder::op(b.defs, 56, {});
    return b.finish();
}

std::vector<std::uint32_t> build_ps_mrt() {
    SpirvBuilder b;
    const auto main_id = b.id(), void_t = b.id(), f32_t = b.id(), vec4_t = b.id(), u32_t = b.id();
    const auto c_0 = b.id(), c_1 = b.id(), arr_u32_1 = b.id();
    const auto ptr_out_v4 = b.id(), ptr_in_arr = b.id(), ptr_in_u32 = b.id(), ptr_out_u32 = b.id(), fn_t = b.id();
    const auto c_zero = b.id(), c_quarter = b.id(), c_one = b.id(), c_color = b.id();
    const auto out_color = b.id(), in_mask = b.id(), out_mask = b.id();
    const auto entry_lbl = b.id(), elem_ptr = b.id(), mask = b.id();

    SpirvBuilder::op(b.caps, 17, {1});
    SpirvBuilder::op(b.mem, 14, {0, 1});
    b.entry_point(4, main_id, "main", {out_color, in_mask, out_mask});
    SpirvBuilder::op(b.modes, 16, {main_id, 7});
    SpirvBuilder::op(b.decos, 71, {out_color, 30, 0});  // Location 0
    SpirvBuilder::op(b.decos, 71, {in_mask, 11, 20});   // BuiltIn SampleMask
    SpirvBuilder::op(b.decos, 71, {out_mask, 30, 1});   // Location 1

    SpirvBuilder::op(b.types, 19, {void_t});
    SpirvBuilder::op(b.types, 22, {f32_t, 32});
    SpirvBuilder::op(b.types, 23, {vec4_t, f32_t, 4});
    SpirvBuilder::op(b.types, 21, {u32_t, 32, 0});
    SpirvBuilder::op(b.types, 43, {u32_t, c_0, 0u});
    SpirvBuilder::op(b.types, 43, {u32_t, c_1, 1u});
    SpirvBuilder::op(b.types, 28, {arr_u32_1, u32_t, c_1});
    SpirvBuilder::op(b.types, 32, {ptr_out_v4, 3, vec4_t});
    SpirvBuilder::op(b.types, 32, {ptr_in_arr, 1, arr_u32_1});
    SpirvBuilder::op(b.types, 32, {ptr_in_u32, 1, u32_t});
    SpirvBuilder::op(b.types, 32, {ptr_out_u32, 3, u32_t});
    SpirvBuilder::op(b.types, 33, {fn_t, void_t});
    SpirvBuilder::op(b.types, 43, {f32_t, c_zero, SpirvBuilder::f32_bits(0.0f)});
    SpirvBuilder::op(b.types, 43, {f32_t, c_quarter, SpirvBuilder::f32_bits(0.25f)});
    SpirvBuilder::op(b.types, 43, {f32_t, c_one, SpirvBuilder::f32_bits(1.0f)});
    SpirvBuilder::op(b.types, 44, {vec4_t, c_color, c_one, c_quarter, c_zero, c_one});
    SpirvBuilder::op(b.types, 59, {ptr_out_v4, out_color, 3});
    SpirvBuilder::op(b.types, 59, {ptr_in_arr, in_mask, 1});
    SpirvBuilder::op(b.types, 59, {ptr_out_u32, out_mask, 3});

    SpirvBuilder::op(b.defs, 54, {void_t, main_id, 0, fn_t});
    SpirvBuilder::op(b.defs, 248, {entry_lbl});
    SpirvBuilder::op(b.defs, 62, {out_color, c_color});
    SpirvBuilder::op(b.defs, 65, {ptr_in_u32, elem_ptr, in_mask, c_0});
    SpirvBuilder::op(b.defs, 61, {u32_t, mask, elem_ptr});
    SpirvBuilder::op(b.defs, 62, {out_mask, mask});
    SpirvBuilder::op(b.defs, 253, {});
    SpirvBuilder::op(b.defs, 56, {});
    return b.finish();
}

bool write_spv(const std::filesystem::path& path, const std::vector<std::uint32_t>& words) {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(words.data()), std::streamsize(words.size() * 4));
    return bool(out);
}

bool compile_hlsl(pD3DCompile compile, const char* src, const char* entry, const char* target,
                  copycmp::Ptr<ID3DBlob>& blob) {
    copycmp::Ptr<ID3DBlob> err;
    const HRESULT hr = compile(src, std::strlen(src), nullptr, nullptr, nullptr, entry, target, 0, 0, &blob.p, &err.p);
    if (SUCCEEDED(hr)) return true;
    std::fprintf(stderr, "D3DCompile %s failed: 0x%08x%s%.*s\n", entry, unsigned(hr), err.p ? ": " : "",
                 err.p ? int(err->GetBufferSize()) : 0, err.p ? static_cast<const char*>(err->GetBufferPointer()) : "");
    return false;
}

struct ShaderSet {
    D3D12_SHADER_BYTECODE vs{}, ps_empty{}, ps_color{}, ps_cov{}, ps_mrt{};
};

struct LayerBlobs {
    std::vector<std::uint8_t> vs, ps_empty, ps_color, ps_cov, ps_mrt;
    ShaderSet view() const {
        return {
            {vs.data(), vs.size()},
            {ps_empty.data(), ps_empty.size()},
            {ps_color.data(), ps_color.size()},
            {ps_cov.data(), ps_cov.size()},
            {ps_mrt.data(), ps_mrt.size()},
        };
    }
};

struct HostBlobs {
    copycmp::Ptr<ID3DBlob> vs, ps_empty, ps_color, ps_cov, ps_mrt;
    ShaderSet view() const {
        return {
            {vs->GetBufferPointer(), vs->GetBufferSize()},
            {ps_empty->GetBufferPointer(), ps_empty->GetBufferSize()},
            {ps_color->GetBufferPointer(), ps_color->GetBufferSize()},
            {ps_cov->GetBufferPointer(), ps_cov->GetBufferSize()},
            {ps_mrt->GetBufferPointer(), ps_mrt->GetBufferSize()},
        };
    }
};

struct DeviceContext {
    std::string label;
    bool ours = false;
    copycmp::Run run;
    copycmp::Ptr<ID3D12RootSignature> root;
    ShaderSet shaders{};
    UINT no_output_samples = 1;
    D3D12_CONSERVATIVE_RASTERIZATION_TIER conservative_tier = D3D12_CONSERVATIVE_RASTERIZATION_TIER_NOT_SUPPORTED;

    bool init(ID3D12Device* dev, const std::string& name, bool is_ours, const ShaderSet& s) {
        label = name;
        ours = is_ours;
        shaders = s;
        CHECK(run.init(dev, is_ours));
        D3D12_VERSIONED_ROOT_SIGNATURE_DESC rs{};
        rs.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
        rs.Desc_1_0.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
        const auto blob = d3d12test::serialize_root_signature(rs);
        CHECK(!blob.empty());
        CHECK(SUCCEEDED(dev->CreateRootSignature(0, blob.data(), blob.size(), IID_PPV_ARGS(&root.p))));
        D3D12_FEATURE_DATA_D3D12_OPTIONS opts{};
        if (SUCCEEDED(dev->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &opts, sizeof(opts))))
            conservative_tier = opts.ConservativeRasterizationTier;
        D3D12_FEATURE_DATA_D3D12_OPTIONS19 opts19{};
        if (SUCCEEDED(dev->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS19, &opts19, sizeof(opts19))))
            no_output_samples = opts19.SupportedSampleCountsWithNoOutputs;
        return true;
    }
};

D3D12_GRAPHICS_PIPELINE_STATE_DESC base_pso(const DeviceContext& ctx, D3D12_SHADER_BYTECODE ps, UINT rtv_count,
                                            DXGI_FORMAT fmt0, DXGI_FORMAT fmt1 = DXGI_FORMAT_UNKNOWN,
                                            UINT forced_samples = 0) {
    static const D3D12_INPUT_ELEMENT_DESC elem{
        "POSITION", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0};
    D3D12_GRAPHICS_PIPELINE_STATE_DESC d{};
    d.pRootSignature = ctx.root.p;
    d.VS = ctx.shaders.vs;
    d.PS = ps;
    d.InputLayout = {&elem, 1};
    for (auto& rt : d.BlendState.RenderTarget) {
        rt.SrcBlend = rt.SrcBlendAlpha = D3D12_BLEND_ONE;
        rt.DestBlend = rt.DestBlendAlpha = D3D12_BLEND_ZERO;
        rt.BlendOp = rt.BlendOpAlpha = D3D12_BLEND_OP_ADD;
        rt.LogicOp = D3D12_LOGIC_OP_NOOP;
        rt.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    }
    d.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    d.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    d.RasterizerState.DepthClipEnable = TRUE;
    d.RasterizerState.ForcedSampleCount = forced_samples;
    d.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    d.NumRenderTargets = rtv_count;
    d.RTVFormats[0] = fmt0;
    d.RTVFormats[1] = fmt1;
    d.SampleDesc.Count = 1;
    d.SampleMask = UINT_MAX;
    return d;
}

// Uploads two triangles (fullscreen quad, 12 bytes/vertex -> 6 vertices = 48 bytes) or one triangle (3 vertices = 24 bytes).
bool upload_vertices(copycmp::Run& r, const std::vector<Triangle>& tris, copycmp::Ptr<ID3D12Resource>& vb_out) {
    const UINT64 bytes = tris.size() * 3 * sizeof(float) * 2;
    CHECK(r.buffer(D3D12_HEAP_TYPE_UPLOAD, bytes, D3D12_RESOURCE_STATE_GENERIC_READ, vb_out));
    void* mapped = nullptr;
    const D3D12_RANGE none{};
    CHECK(SUCCEEDED(vb_out->Map(0, &none, &mapped)));
    auto* p = static_cast<float*>(mapped);
    for (const auto& t : tris) {
        for (const auto& pt : t) {
            *p++ = float(pt.x * 2.0 / kWidth - 1.0);
            *p++ = float(1.0 - pt.y * 2.0 / kHeight);
        }
    }
    vb_out->Unmap(0, nullptr);
    return true;
}

std::vector<Triangle> fullscreen_quad() {
    return {
        {{{0.0, 0.0}, {double(kWidth), 0.0}, {double(kWidth), double(kHeight)}}},
        {{{0.0, 0.0}, {double(kWidth), double(kHeight)}, {0.0, double(kHeight)}}},
    };
}

// ---------------------------------------------------------------------------------------------------------------
// Section 1: PSO validation matrix across Layer, Native Adapter, and WARP.

bool test_pso_validation(const DeviceContext& layer, const DeviceContext& native_host,
                         const DeviceContext* warp, bool has_mixed_ext) {
    struct Case {
        const char* name;
        bool need_conservative = false;
        bool is_mixed_rtv_tir = false;
        bool is_down_mixed = false;
        void (*configure)(const DeviceContext&, D3D12_GRAPHICS_PIPELINE_STATE_DESC&);
    };

    const Case table[] = {
        {"Forced=0 attachmentless", false, false, false, [](const DeviceContext&, D3D12_GRAPHICS_PIPELINE_STATE_DESC& d) {
            d.RasterizerState.ForcedSampleCount = 0;
        }},
        {"Forced=1 attachmentless", false, false, false, [](const DeviceContext&, D3D12_GRAPHICS_PIPELINE_STATE_DESC& d) {
            d.RasterizerState.ForcedSampleCount = 1;
        }},
        {"Forced=4 attachmentless", false, false, false, [](const DeviceContext&, D3D12_GRAPHICS_PIPELINE_STATE_DESC& d) {
            d.RasterizerState.ForcedSampleCount = 4;
        }},
        {"Forced=8 attachmentless", false, false, false, [](const DeviceContext&, D3D12_GRAPHICS_PIPELINE_STATE_DESC& d) {
            d.RasterizerState.ForcedSampleCount = 8;
        }},
        {"Forced=16 attachmentless", false, false, false, [](const DeviceContext&, D3D12_GRAPHICS_PIPELINE_STATE_DESC& d) {
            d.RasterizerState.ForcedSampleCount = 16;
        }},
        {"Forced=2 invalid", false, false, false, [](const DeviceContext&, D3D12_GRAPHICS_PIPELINE_STATE_DESC& d) {
            d.RasterizerState.ForcedSampleCount = 2;
        }},
        {"Forced=3 invalid", false, false, false, [](const DeviceContext&, D3D12_GRAPHICS_PIPELINE_STATE_DESC& d) {
            d.RasterizerState.ForcedSampleCount = 3;
        }},
        {"Forced=5 invalid", false, false, false, [](const DeviceContext&, D3D12_GRAPHICS_PIPELINE_STATE_DESC& d) {
            d.RasterizerState.ForcedSampleCount = 5;
        }},
        {"Forced=32 invalid", false, false, false, [](const DeviceContext&, D3D12_GRAPHICS_PIPELINE_STATE_DESC& d) {
            d.RasterizerState.ForcedSampleCount = 32;
        }},
        {"Forced=UINT_MAX invalid", false, false, false, [](const DeviceContext&, D3D12_GRAPHICS_PIPELINE_STATE_DESC& d) {
            d.RasterizerState.ForcedSampleCount = (std::numeric_limits<UINT>::max)();
        }},
        {"Forced=4 DepthEnable=TRUE (no DSV)", false, false, false, [](const DeviceContext&, D3D12_GRAPHICS_PIPELINE_STATE_DESC& d) {
            d.RasterizerState.ForcedSampleCount = 4;
            d.DepthStencilState.DepthEnable = TRUE;
        }},
        {"Forced=4 DepthEnable=TRUE + D32 DSV", false, false, false, [](const DeviceContext&, D3D12_GRAPHICS_PIPELINE_STATE_DESC& d) {
            d.RasterizerState.ForcedSampleCount = 4;
            d.DepthStencilState.DepthEnable = TRUE;
            d.DSVFormat = DXGI_FORMAT_D32_FLOAT;
        }},
        {"Forced=4 StencilEnable=TRUE (no DSV)", false, false, false, [](const DeviceContext&, D3D12_GRAPHICS_PIPELINE_STATE_DESC& d) {
            d.RasterizerState.ForcedSampleCount = 4;
            d.DepthStencilState.StencilEnable = TRUE;
        }},
        {"Forced=4 DepthEnable=FALSE DSV=D32", false, false, false, [](const DeviceContext&, D3D12_GRAPHICS_PIPELINE_STATE_DESC& d) {
            d.RasterizerState.ForcedSampleCount = 4;
            d.DSVFormat = DXGI_FORMAT_D32_FLOAT;
        }},
        {"Forced=1 DepthEnable=TRUE + D32 DSV", false, false, false, [](const DeviceContext&, D3D12_GRAPHICS_PIPELINE_STATE_DESC& d) {
            d.RasterizerState.ForcedSampleCount = 1;
            d.DepthStencilState.DepthEnable = TRUE;
            d.DSVFormat = DXGI_FORMAT_D32_FLOAT;
        }},
        {"Forced=1 StencilEnable=TRUE + D24S8 DSV", false, false, false, [](const DeviceContext&, D3D12_GRAPHICS_PIPELINE_STATE_DESC& d) {
            d.RasterizerState.ForcedSampleCount = 1;
            d.DepthStencilState.StencilEnable = TRUE;
            d.DSVFormat = DXGI_FORMAT_D24_UNORM_S8_UINT;
        }},
        {"Forced=1 DepthEnable=FALSE DSV=D32", false, false, false, [](const DeviceContext&, D3D12_GRAPHICS_PIPELINE_STATE_DESC& d) {
            d.RasterizerState.ForcedSampleCount = 1;
            d.DSVFormat = DXGI_FORMAT_D32_FLOAT;
        }},
        {"Forced=4 + SampleDesc=4 (1 RTV)", false, false, false, [](const DeviceContext& c, D3D12_GRAPHICS_PIPELINE_STATE_DESC& d) {
            d.PS = c.shaders.ps_color;
            d.NumRenderTargets = 1;
            d.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
            d.SampleDesc.Count = 4;
            d.RasterizerState.ForcedSampleCount = 4;
        }},
        {"Forced=4 + SampleDesc=4 (no RTV)", false, false, false, [](const DeviceContext&, D3D12_GRAPHICS_PIPELINE_STATE_DESC& d) {
            d.SampleDesc.Count = 4;
            d.RasterizerState.ForcedSampleCount = 4;
        }},
        {"Forced=1 + SampleDesc=4 (1 RTV)", false, false, true, [](const DeviceContext& c, D3D12_GRAPHICS_PIPELINE_STATE_DESC& d) {
            d.PS = c.shaders.ps_color;
            d.NumRenderTargets = 1;
            d.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
            d.SampleDesc.Count = 4;
            d.RasterizerState.ForcedSampleCount = 1;
        }},
        {"Forced=8 + SampleDesc=4 (1 RTV)", false, false, false, [](const DeviceContext& c, D3D12_GRAPHICS_PIPELINE_STATE_DESC& d) {
            d.PS = c.shaders.ps_color;
            d.NumRenderTargets = 1;
            d.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
            d.SampleDesc.Count = 4;
            d.RasterizerState.ForcedSampleCount = 8;
        }},
        {"Forced=4 + AlphaToCoverage (1x RTV)", false, true, false, [](const DeviceContext& c, D3D12_GRAPHICS_PIPELINE_STATE_DESC& d) {
            d.PS = c.shaders.ps_color;
            d.NumRenderTargets = 1;
            d.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
            d.BlendState.AlphaToCoverageEnable = TRUE;
            d.RasterizerState.ForcedSampleCount = 4;
        }},
        {"Forced=4 + 1x RTV", false, true, false, [](const DeviceContext& c, D3D12_GRAPHICS_PIPELINE_STATE_DESC& d) {
            d.PS = c.shaders.ps_color;
            d.NumRenderTargets = 1;
            d.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
            d.RasterizerState.ForcedSampleCount = 4;
        }},
        {"Forced=4 + ConservativeRaster (no RTV)", true, false, false, [](const DeviceContext&, D3D12_GRAPHICS_PIPELINE_STATE_DESC& d) {
            d.RasterizerState.ForcedSampleCount = 4;
            d.RasterizerState.ConservativeRaster = D3D12_CONSERVATIVE_RASTERIZATION_MODE_ON;
        }},
        {"Forced=4 + ConservativeRaster (1x RTV)", true, true, false, [](const DeviceContext& c, D3D12_GRAPHICS_PIPELINE_STATE_DESC& d) {
            d.PS = c.shaders.ps_color;
            d.NumRenderTargets = 1;
            d.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
            d.RasterizerState.ForcedSampleCount = 4;
            d.RasterizerState.ConservativeRaster = D3D12_CONSERVATIVE_RASTERIZATION_MODE_ON;
        }},
    };

    const auto eval = [](const DeviceContext& ctx, const Case& c) -> HRESULT {
        auto d = base_pso(ctx, ctx.shaders.ps_empty, 0, DXGI_FORMAT_UNKNOWN);
        c.configure(ctx, d);
        copycmp::Ptr<ID3D12PipelineState> pso;
        return ctx.run.device->CreateGraphicsPipelineState(&d, IID_PPV_ARGS(&pso.p));
    };

    unsigned mismatches = 0;
    std::printf("\n--- PSO Validation Matrix (ForcedSampleCount) ---\n");
    std::printf("%-42s | %-10s | %-10s | %-10s\n", "case", "layer", native_host.label.c_str(), warp ? "WARP" : "n/a");
    for (const auto& c : table) {
        if (c.need_conservative &&
            (layer.conservative_tier == D3D12_CONSERVATIVE_RASTERIZATION_TIER_NOT_SUPPORTED ||
             native_host.conservative_tier == D3D12_CONSERVATIVE_RASTERIZATION_TIER_NOT_SUPPORTED))
            continue;
        const HRESULT hr_layer = eval(layer, c);
        const HRESULT hr_host = eval(native_host, c);
        const HRESULT hr_warp = warp ? eval(*warp, c) : E_FAIL;
        std::printf("%-42s | 0x%08x | 0x%08x | 0x%08x\n",
                    c.name, unsigned(hr_layer), unsigned(hr_host), unsigned(hr_warp));
        if (c.is_down_mixed) {
            // Vulkan mixed-attachment extensions require rasterizationSamples >= attachmentSamples, so 1 < 4 is E_NOTIMPL.
            if (hr_layer != E_NOTIMPL || FAILED(hr_host) || (warp && FAILED(hr_warp))) {
                std::fprintf(stderr, "MISMATCH down-mixed '%s': layer=0x%08x host=0x%08x warp=0x%08x\n",
                             c.name, unsigned(hr_layer), unsigned(hr_host), unsigned(hr_warp));
                ++mismatches;
            }
            continue;
        }
        if (c.is_mixed_rtv_tir && !has_mixed_ext) {
            if (hr_layer != E_NOTIMPL) {
                std::fprintf(stderr, "EXPECTED E_NOTIMPL without mixed ext on '%s': got 0x%08x\n",
                             c.name, unsigned(hr_layer));
                ++mismatches;
            }
            continue;
        }
        if (std::string(c.name) == "Forced=16 attachmentless" && !(layer.no_output_samples & 16u)) {
            if (hr_layer != E_INVALIDARG) {
                std::fprintf(stderr, "EXPECTED E_INVALIDARG for unsupported 16x attachmentless: got 0x%08x\n",
                             unsigned(hr_layer));
                ++mismatches;
            }
            continue;
        }
        if (SUCCEEDED(hr_layer) != SUCCEEDED(hr_host)) {
            std::fprintf(stderr, "MISMATCH vs %s on '%s': layer=0x%08x host=0x%08x\n",
                         native_host.label.c_str(), c.name, unsigned(hr_layer), unsigned(hr_host));
            ++mismatches;
        }
        if (warp && SUCCEEDED(hr_layer) != SUCCEEDED(hr_warp)) {
            if (std::string(c.name) == "Forced=16 attachmentless" && !(warp->no_output_samples & 16u)) continue;
            std::fprintf(stderr, "MISMATCH vs WARP on '%s': layer=0x%08x warp=0x%08x\n",
                         c.name, unsigned(hr_layer), unsigned(hr_warp));
            ++mismatches;
        }
    }
    return mismatches == 0;
}

// ---------------------------------------------------------------------------------------------------------------
// Section 2: Attachmentless TIR Occlusion Queries.

struct OcclusionScene {
    const char* name;
    UINT forced_samples;
    UINT sample_mask;
    D3D12_RECT scissor;
    std::vector<Triangle> tris;
    bool use_null_ps = false;
};

bool run_occlusion_scene(DeviceContext& ctx, const OcclusionScene& s, UINT64& count_out) {
    auto pso_desc = base_pso(ctx, s.use_null_ps ? D3D12_SHADER_BYTECODE{} : ctx.shaders.ps_empty,
                             0, DXGI_FORMAT_UNKNOWN, DXGI_FORMAT_UNKNOWN, s.forced_samples);
    pso_desc.SampleMask = s.sample_mask;
    copycmp::Ptr<ID3D12PipelineState> pso;
    CHECK(SUCCEEDED(ctx.run.device->CreateGraphicsPipelineState(&pso_desc, IID_PPV_ARGS(&pso.p))));

    D3D12_QUERY_HEAP_DESC qh_desc{D3D12_QUERY_HEAP_TYPE_OCCLUSION, 1, 0};
    copycmp::Ptr<ID3D12QueryHeap> qh;
    CHECK(SUCCEEDED(ctx.run.device->CreateQueryHeap(&qh_desc, IID_PPV_ARGS(&qh.p))));

    copycmp::Ptr<ID3D12Resource> vb, rb;
    CHECK(upload_vertices(ctx.run, s.tris, vb));
    CHECK(ctx.run.buffer(D3D12_HEAP_TYPE_READBACK, 64, D3D12_RESOURCE_STATE_COPY_DEST, rb));

    ctx.run.begin();
    ctx.run.list->SetGraphicsRootSignature(ctx.root.p);
    ctx.run.list->SetPipelineState(pso.p);
    ctx.run.list->OMSetRenderTargets(0, nullptr, FALSE, nullptr);
    const D3D12_VIEWPORT vp{0.0f, 0.0f, float(kWidth), float(kHeight), 0.0f, 1.0f};
    ctx.run.list->RSSetViewports(1, &vp);
    ctx.run.list->RSSetScissorRects(1, &s.scissor);
    const D3D12_VERTEX_BUFFER_VIEW vbv{vb->GetGPUVirtualAddress(), UINT(s.tris.size() * 24), 8};
    ctx.run.list->IASetVertexBuffers(0, 1, &vbv);
    ctx.run.list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    ctx.run.list->BeginQuery(qh.p, D3D12_QUERY_TYPE_OCCLUSION, 0);
    ctx.run.list->DrawInstanced(UINT(s.tris.size() * 3), 1, 0, 0);
    ctx.run.list->EndQuery(qh.p, D3D12_QUERY_TYPE_OCCLUSION, 0);
    ctx.run.list->ResolveQueryData(qh.p, D3D12_QUERY_TYPE_OCCLUSION, 0, 1, rb.p, 0);

    const std::string err = ctx.run.submit();
    if (!err.empty()) {
        std::fprintf(stderr, "%s occlusion '%s' submit failed: %s\n", ctx.label.c_str(), s.name, err.c_str());
        return false;
    }

    void* data = nullptr;
    const D3D12_RANGE read_range{0, sizeof(UINT64)};
    CHECK(SUCCEEDED(rb->Map(0, &read_range, &data)));
    std::memcpy(&count_out, data, sizeof(UINT64));
    const D3D12_RANGE none{};
    rb->Unmap(0, &none);
    return true;
}

bool test_attachmentless_occlusion(DeviceContext& layer, DeviceContext& native_host, DeviceContext* warp,
                                   std::uint32_t vendor_id) {
    const D3D12_RECT full{0, 0, LONG(kWidth), LONG(kHeight)};
    const D3D12_RECT clipped{4, 4, 12, 12};
    const std::vector<Triangle> diag = {{{{0.0, 0.0}, {double(kWidth), 0.0}, {0.0, double(kHeight)}}}};
    // Horizontal sliver in [4.0625, 4.4375] misses 1x pixel center (y=4.5) while hitting upper subsamples on 4x/8x/16x.
    const std::vector<Triangle> sliver = {
        {{{2.0625, 4.0625}, {14.0625, 4.0625}, {14.0625, 4.4375}}},
        {{{2.0625, 4.0625}, {14.0625, 4.4375}, {2.0625, 4.4375}}},
    };

    const OcclusionScene scenes[] = {
        {"full Forced=0", 0, UINT_MAX, full, fullscreen_quad()},
        {"full Forced=1", 1, UINT_MAX, full, fullscreen_quad()},
        {"full Forced=4", 4, UINT_MAX, full, fullscreen_quad()},
        {"full Forced=4 null-PS", 4, UINT_MAX, full, fullscreen_quad(), true},
        {"full Forced=4 mask=0x1", 4, 0x1u, full, fullscreen_quad()},
        {"full Forced=4 mask=0x2", 4, 0x2u, full, fullscreen_quad()},
        {"full Forced=4 mask=0x5", 4, 0x5u, full, fullscreen_quad()},
        {"full Forced=4 mask=0x0", 4, 0x0u, full, fullscreen_quad()},
        {"scissor Forced=4", 4, UINT_MAX, clipped, fullscreen_quad()},
        {"diag Forced=4", 4, UINT_MAX, full, diag},
        {"sliver Forced=1 (expect 0)", 1, UINT_MAX, full, sliver},
        {"sliver Forced=4 (expect >0)", 4, UINT_MAX, full, sliver},
        {"full Forced=8", 8, UINT_MAX, full, fullscreen_quad()},
        {"full Forced=8 mask=0x55", 8, 0x55u, full, fullscreen_quad()},
        {"sliver Forced=8", 8, UINT_MAX, full, sliver},
        {"full Forced=16", 16, UINT_MAX, full, fullscreen_quad()},
        {"full Forced=16 mask=0x0F0F", 16, 0x0F0Fu, full, fullscreen_quad()},
        {"sliver Forced=16", 16, UINT_MAX, full, sliver},
    };

    std::printf("\n--- Attachmentless TIR Occlusion Queries ---\n");
    unsigned verified = 0;
    unsigned occ_mismatches = 0;
    bool host_skips_empty_attless = false;
    UINT64 baseline_l[17]{}, baseline_h[17]{}, baseline_w[17]{};
    const bool is_amd = (vendor_id == 0x1002u);
    for (const auto& s : scenes) {
        const UINT mask = s.forced_samples ? s.forced_samples : 1u;
        if (!(layer.no_output_samples & mask) || !(native_host.no_output_samples & mask)) continue;
        UINT64 got_layer = 0, got_host = 0, got_warp = 0;
        CHECK(run_occlusion_scene(layer, s, got_layer));
        CHECK(run_occlusion_scene(native_host, s, got_host));
        const bool do_warp = warp && (warp->no_output_samples & mask);
        if (do_warp) CHECK(run_occlusion_scene(*warp, s, got_warp));

        std::printf("%-28s | layer=%-5llu | host=%-5llu | warp=%s\n",
                    s.name, static_cast<unsigned long long>(got_layer),
                    static_cast<unsigned long long>(got_host),
                    do_warp ? std::to_string(got_warp).c_str() : "skip");

        if (std::string(s.name) == "full Forced=0" && got_host == 0 && do_warp && got_warp > 0) {
            host_skips_empty_attless = true;
            std::printf("note: %s D3D12 returns 0 for attachmentless empty-PS occlusion queries; comparing against WARP\n",
                        native_host.label.c_str());
        }

        if (s.use_null_ps) {
            // D3D12 defines 0 occlusion samples when both PS and DSV are null, whereas Vulkan counts rasterized
            // samples (1024 at 4x on RTX/Intel/AMD 32.0.21042.62). This row is an execution-only smoke check,
            // not a parity check.
            std::printf("note: null-PS + no-DSV occlusion is 0 in D3D12/WARP whereas Vulkan counts rasterized "
                        "samples; execution-only check\n");
            ++verified;
            continue;
        }
        const bool expect_zero = (s.sample_mask == 0u) || (s.forced_samples > 1 && (s.sample_mask & 1u) == 0u);
        if (s.forced_samples <= 1 || expect_zero) {
            const bool allow_host_mask_diff = (s.forced_samples > 1 && s.sample_mask != 0u);
            if ((!host_skips_empty_attless && !allow_host_mask_diff && got_layer != got_host) ||
                (expect_zero && got_layer != 0) || (do_warp && got_layer != got_warp)) {
                std::fprintf(stderr, "OCCLUSION MISMATCH '%s': layer=%llu host=%llu warp=%s\n",
                             s.name, static_cast<unsigned long long>(got_layer),
                             static_cast<unsigned long long>(got_host),
                             do_warp ? std::to_string(got_warp).c_str() : "skip");
                ++occ_mismatches;
            } else if (!host_skips_empty_attless && allow_host_mask_diff && got_host != got_layer) {
                std::printf("note: native %s D3D12 returned %llu for Forced=%u mask=0x%x (AMD applies SampleMask "
                            "to forced subsamples directly while WARP and Vulkan follow TIR bit-0 semantics)\n",
                            native_host.label.c_str(), static_cast<unsigned long long>(got_host),
                            s.forced_samples, s.sample_mask);
            }
        } else {
            if (got_layer == 0 || (!host_skips_empty_attless && got_host == 0) || (do_warp && got_warp == 0)) {
                std::fprintf(stderr, "OCCLUSION NONZERO MISMATCH '%s': layer=%llu host=%llu warp=%s\n",
                             s.name, static_cast<unsigned long long>(got_layer),
                             static_cast<unsigned long long>(got_host),
                             do_warp ? std::to_string(got_warp).c_str() : "skip");
                ++occ_mismatches;
            }
            if (!is_amd) {
                if (do_warp && got_layer != got_warp) {
                    std::fprintf(stderr, "OCCLUSION SAMPLE-RATE MISMATCH vs WARP '%s': layer=%llu warp=%llu\n",
                                 s.name, static_cast<unsigned long long>(got_layer),
                                 static_cast<unsigned long long>(got_warp));
                    ++occ_mismatches;
                }
                if (!do_warp && !host_skips_empty_attless && got_layer != got_host && got_host == 0) {
                    ++occ_mismatches;
                }
            } else if (do_warp && got_layer != got_warp) {
                std::printf("note: AMD Vulkan reported %llu vs WARP %llu on '%s' (RDNA4 driver 26.11.2 counts "
                            "~half-rate samples)\n",
                            static_cast<unsigned long long>(got_layer),
                            static_cast<unsigned long long>(got_warp), s.name);
            }
            if (s.name == std::string("full Forced=4") || s.name == std::string("full Forced=8") ||
                s.name == std::string("full Forced=16")) {
                baseline_l[s.forced_samples] = got_layer;
                baseline_h[s.forced_samples] = got_host;
                baseline_w[s.forced_samples] = got_warp;
                if (got_layer <= UINT64(kWidth) * kHeight) {
                    std::fprintf(stderr, "TIR MULTISAMPLE INVARIANT FAILED '%s': layer=%llu <= %llu\n",
                                 s.name, static_cast<unsigned long long>(got_layer),
                                 static_cast<unsigned long long>(UINT64(kWidth) * kHeight));
                    ++occ_mismatches;
                }
            } else if (std::string(s.name) == "scissor Forced=4") {
                if (baseline_l[4] && !(got_layer > 0 && got_layer < baseline_l[4])) {
                    std::fprintf(stderr, "SCISSOR INVARIANT FAILED: layer=%llu full4=%llu\n",
                                 static_cast<unsigned long long>(got_layer),
                                 static_cast<unsigned long long>(baseline_l[4]));
                    ++occ_mismatches;
                }
            } else if (std::string(s.name) == "diag Forced=4") {
                if (baseline_l[4] && got_layer != baseline_l[4] / 2) {
                    std::fprintf(stderr, "DIAG INVARIANT FAILED: layer=%llu expected=%llu\n",
                                 static_cast<unsigned long long>(got_layer),
                                 static_cast<unsigned long long>(baseline_l[4] / 2));
                    ++occ_mismatches;
                }
            } else if (std::string(s.name).find("full Forced=") == 0 && s.sample_mask != UINT_MAX) {
                if (got_layer != baseline_l[s.forced_samples] ||
                    (do_warp && got_warp != baseline_w[s.forced_samples])) {
                    std::fprintf(stderr, "ODD SAMPLEMASK MISMATCH '%s'\n", s.name);
                    ++occ_mismatches;
                } else if (!host_skips_empty_attless && got_host != baseline_h[s.forced_samples]) {
                    std::printf("note: native %s D3D12 returned %llu vs baseline %llu for '%s'\n",
                                native_host.label.c_str(), static_cast<unsigned long long>(got_host),
                                static_cast<unsigned long long>(baseline_h[s.forced_samples]), s.name);
                }
            }
        }
        ++verified;
    }
    if (baseline_l[4] && baseline_l[8] && !(baseline_l[8] > baseline_l[4])) {
        std::fprintf(stderr, "MONOTONIC 8x>4x INVARIANT FAILED: l4=%llu l8=%llu\n",
                     static_cast<unsigned long long>(baseline_l[4]),
                     static_cast<unsigned long long>(baseline_l[8]));
        ++occ_mismatches;
    }
    if (baseline_l[8] && baseline_l[16] && !(baseline_l[16] > baseline_l[8])) {
        std::fprintf(stderr, "MONOTONIC 16x>8x INVARIANT FAILED: l8=%llu l16=%llu\n",
                     static_cast<unsigned long long>(baseline_l[8]),
                     static_cast<unsigned long long>(baseline_l[16]));
        ++occ_mismatches;
    }
    if (occ_mismatches) return false;
    if (layer.no_output_samples & 4u) {
        UINT64 s1 = 0, s4 = 0;
        CHECK(run_occlusion_scene(layer, {"sliver1", 1, UINT_MAX, full, sliver}, s1));
        CHECK(run_occlusion_scene(layer, {"sliver4", 4, UINT_MAX, full, sliver}, s4));
        CHECK(s1 == 0 && s4 > 0);
    }
    std::printf("Attachmentless occlusion verified: %u scenes passed\n", verified);
    return verified > 0;
}

// ---------------------------------------------------------------------------------------------------------------
// Section 3: 1x-RTV Mixed-Sample TIR (RGBA8, R32_UINT SV_Coverage, MRT, Unbound Variant, Conservative + TIR).

struct RtvScene {
    const char* name;
    UINT forced_samples;
    UINT sample_mask;
    D3D12_RECT scissor;
    std::vector<Triangle> tris;
    bool conservative = false;
};

bool render_rtv(DeviceContext& ctx, DXGI_FORMAT fmt, D3D12_SHADER_BYTECODE ps, const RtvScene& s,
                std::vector<std::uint8_t>& out0, std::vector<std::uint8_t>* out1 = nullptr,
                DXGI_FORMAT fmt1 = DXGI_FORMAT_UNKNOWN, bool leave_unbound = false, UINT64* occ_out = nullptr) {
    const UINT rtv_count = fmt1 != DXGI_FORMAT_UNKNOWN ? 2u : 1u;
    auto pso_desc = base_pso(ctx, ps, rtv_count, fmt, fmt1, s.forced_samples);
    pso_desc.SampleMask = s.sample_mask;
    if (s.conservative) pso_desc.RasterizerState.ConservativeRaster = D3D12_CONSERVATIVE_RASTERIZATION_MODE_ON;
    copycmp::Ptr<ID3D12PipelineState> pso;
    CHECK(SUCCEEDED(ctx.run.device->CreateGraphicsPipelineState(&pso_desc, IID_PPV_ARGS(&pso.p))));

    D3D12_RESOURCE_DESC tex_desc{};
    tex_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    tex_desc.Width = kWidth;
    tex_desc.Height = kHeight;
    tex_desc.DepthOrArraySize = tex_desc.MipLevels = 1;
    tex_desc.SampleDesc.Count = 1;
    tex_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    tex_desc.Format = fmt;

    copycmp::Ptr<ID3D12Resource> rt0, rt1, rb0, rb1, vb, q_rb;
    copycmp::Ptr<ID3D12DescriptorHeap> heap;
    copycmp::Ptr<ID3D12QueryHeap> qh;
    CHECK(ctx.run.texture(tex_desc, D3D12_RESOURCE_STATE_RENDER_TARGET, rt0));
    CHECK(ctx.run.buffer(D3D12_HEAP_TYPE_READBACK, kPitch * kHeight, D3D12_RESOURCE_STATE_COPY_DEST, rb0));
    if (rtv_count == 2) {
        tex_desc.Format = fmt1;
        CHECK(ctx.run.texture(tex_desc, D3D12_RESOURCE_STATE_RENDER_TARGET, rt1));
        CHECK(ctx.run.buffer(D3D12_HEAP_TYPE_READBACK, kPitch * kHeight, D3D12_RESOURCE_STATE_COPY_DEST, rb1));
    }
    const D3D12_DESCRIPTOR_HEAP_DESC hd{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, rtv_count, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    CHECK(SUCCEEDED(ctx.run.device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&heap.p))));
    const UINT inc = ctx.run.device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    const D3D12_CPU_DESCRIPTOR_HANDLE h0 = heap->GetCPUDescriptorHandleForHeapStart();
    const D3D12_CPU_DESCRIPTOR_HANDLE h1{h0.ptr + inc};
    ctx.run.device->CreateRenderTargetView(rt0.p, nullptr, h0);
    if (rtv_count == 2) ctx.run.device->CreateRenderTargetView(rt1.p, nullptr, h1);

    if (occ_out) {
        D3D12_QUERY_HEAP_DESC qd{D3D12_QUERY_HEAP_TYPE_OCCLUSION, 1, 0};
        CHECK(SUCCEEDED(ctx.run.device->CreateQueryHeap(&qd, IID_PPV_ARGS(&qh.p))));
        CHECK(ctx.run.buffer(D3D12_HEAP_TYPE_READBACK, 64, D3D12_RESOURCE_STATE_COPY_DEST, q_rb));
    }
    CHECK(upload_vertices(ctx.run, s.tris, vb));

    ctx.run.begin();
    const float clear[4] = {};
    ctx.run.list->ClearRenderTargetView(h0, clear, 0, nullptr);
    if (rtv_count == 2) ctx.run.list->ClearRenderTargetView(h1, clear, 0, nullptr);
    if (leave_unbound) {
        ctx.run.list->OMSetRenderTargets(0, nullptr, FALSE, nullptr);
    } else {
        const D3D12_CPU_DESCRIPTOR_HANDLE rts[2] = {h0, h1};
        ctx.run.list->OMSetRenderTargets(rtv_count, rts, FALSE, nullptr);
    }
    ctx.run.list->SetGraphicsRootSignature(ctx.root.p);
    ctx.run.list->SetPipelineState(pso.p);
    const D3D12_VIEWPORT vp{0.0f, 0.0f, float(kWidth), float(kHeight), 0.0f, 1.0f};
    ctx.run.list->RSSetViewports(1, &vp);
    ctx.run.list->RSSetScissorRects(1, &s.scissor);
    const D3D12_VERTEX_BUFFER_VIEW vbv{vb->GetGPUVirtualAddress(), UINT(s.tris.size() * 24), 8};
    ctx.run.list->IASetVertexBuffers(0, 1, &vbv);
    ctx.run.list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    if (occ_out) ctx.run.list->BeginQuery(qh.p, D3D12_QUERY_TYPE_OCCLUSION, 0);
    ctx.run.list->DrawInstanced(UINT(s.tris.size() * 3), 1, 0, 0);
    if (occ_out) {
        ctx.run.list->EndQuery(qh.p, D3D12_QUERY_TYPE_OCCLUSION, 0);
        ctx.run.list->ResolveQueryData(qh.p, D3D12_QUERY_TYPE_OCCLUSION, 0, 1, q_rb.p, 0);
    }

    auto copy_out = [&](ID3D12Resource* src, ID3D12Resource* dst, DXGI_FORMAT f) {
        ctx.run.barrier(src, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
        D3D12_TEXTURE_COPY_LOCATION src_loc{}, dst_loc{};
        src_loc.pResource = src;
        src_loc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        dst_loc.pResource = dst;
        dst_loc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        dst_loc.PlacedFootprint.Footprint = {f, kWidth, kHeight, 1, kPitch};
        ctx.run.list->CopyTextureRegion(&dst_loc, 0, 0, 0, &src_loc, nullptr);
    };
    copy_out(rt0.p, rb0.p, fmt);
    if (rtv_count == 2) copy_out(rt1.p, rb1.p, fmt1);

    const std::string err = ctx.run.submit();
    if (!err.empty()) {
        std::fprintf(stderr, "%s rtv '%s' failed: %s\n", ctx.label.c_str(), s.name, err.c_str());
        return false;
    }

    auto read_texels = [](ID3D12Resource* rb, std::vector<std::uint8_t>& out) {
        out.resize(std::size_t(kWidth) * kHeight * 4);
        void* data = nullptr;
        const D3D12_RANGE rr{0, std::size_t(kPitch) * kHeight};
        if (FAILED(rb->Map(0, &rr, &data))) return false;
        for (UINT y = 0; y < kHeight; ++y)
            std::memcpy(out.data() + y * kWidth * 4, static_cast<const std::uint8_t*>(data) + y * kPitch, kWidth * 4);
        const D3D12_RANGE none{};
        rb->Unmap(0, &none);
        return true;
    };
    CHECK(read_texels(rb0.p, out0));
    if (rtv_count == 2 && out1) CHECK(read_texels(rb1.p, *out1));
    if (occ_out) {
        void* data = nullptr;
        const D3D12_RANGE rr{0, sizeof(UINT64)};
        CHECK(SUCCEEDED(q_rb->Map(0, &rr, &data)));
        std::memcpy(occ_out, data, sizeof(UINT64));
        const D3D12_RANGE none{};
        q_rb->Unmap(0, &none);
    }
    return true;
}

bool test_mixed_sample_rtvs(DeviceContext& layer, DeviceContext& native_host, DeviceContext* warp,
                            bool has_mixed_ext) {
    if (!has_mixed_ext) {
        std::printf("\nMixed-sample RTV tests skipped (mixed_attachment_samples=false; "
                    "VK_AMD_mixed_attachment_samples requires LEONIDA_VK_AMD_MIXED_SAMPLES=1 opt-in)\n");
        return true;
    }

    const D3D12_RECT full{0, 0, LONG(kWidth), LONG(kHeight)};
    const D3D12_RECT clipped{3, 3, 13, 13};
    const std::vector<Triangle> diag = {{{{0.0, 0.0}, {double(kWidth), 0.0}, {0.0, double(kHeight)}}}};
    const std::vector<Triangle> sliver = {
        {{{2.0625, 4.0625}, {14.0625, 4.0625}, {14.0625, 4.4375}}},
        {{{2.0625, 4.0625}, {14.0625, 4.4375}, {2.0625, 4.4375}}},
    };

    const auto nonzero = [](const std::vector<std::uint8_t>& v) {
        return std::any_of(v.begin(), v.end(), [](std::uint8_t b) { return b != 0; });
    };
    const RtvScene color_scenes[] = {
        {"rgba8 full Forced=1", 1, UINT_MAX, full, fullscreen_quad()},
        {"rgba8 full Forced=4", 4, UINT_MAX, full, fullscreen_quad()},
        {"rgba8 full Forced=4 mask=0x1", 4, 0x1u, full, fullscreen_quad()},
        {"rgba8 full Forced=4 mask=0x2", 4, 0x2u, full, fullscreen_quad()},
        {"rgba8 full Forced=4 mask=0x5", 4, 0x5u, full, fullscreen_quad()},
        {"rgba8 diag Forced=4", 4, UINT_MAX, full, diag},
        {"rgba8 sliver Forced=1", 1, UINT_MAX, full, sliver},
        {"rgba8 sliver Forced=4", 4, UINT_MAX, full, sliver},
        {"rgba8 sliver Forced=4 mask=0x1", 4, 0x1u, full, sliver},
        {"rgba8 full Forced=8", 8, UINT_MAX, full, fullscreen_quad()},
        {"rgba8 sliver Forced=8", 8, UINT_MAX, full, sliver},
        {"rgba8 full Forced=16", 16, UINT_MAX, full, fullscreen_quad()},
        {"rgba8 sliver Forced=16", 16, UINT_MAX, full, sliver},
    };

    std::printf("\n--- Mixed-Sample 1x-RTV TIR Tests ---\n");
    std::vector<std::uint8_t> sliver1_pixels, sliver4_pixels, full4_pixels, full4_mask02, full4_mask05;
    for (const auto& s : color_scenes) {
        const UINT mask = s.forced_samples ? s.forced_samples : 1u;
        if (!(layer.no_output_samples & mask) || !(native_host.no_output_samples & mask)) continue;
        std::vector<std::uint8_t> pix_l, pix_h, pix_w;
        CHECK(render_rtv(layer, DXGI_FORMAT_R8G8B8A8_UNORM, layer.shaders.ps_color, s, pix_l));
        CHECK(render_rtv(native_host, DXGI_FORMAT_R8G8B8A8_UNORM, native_host.shaders.ps_color, s, pix_h));
        CHECK(pix_l == pix_h);
        if (warp && (warp->no_output_samples & mask)) {
            CHECK(render_rtv(*warp, DXGI_FORMAT_R8G8B8A8_UNORM, warp->shaders.ps_color, s, pix_w));
            CHECK(pix_l == pix_w);
        }
        if (std::string(s.name) == "rgba8 sliver Forced=1") sliver1_pixels = pix_l;
        if (std::string(s.name) == "rgba8 sliver Forced=4") sliver4_pixels = pix_l;
        if (std::string(s.name) == "rgba8 full Forced=4") full4_pixels = pix_l;
        if (std::string(s.name) == "rgba8 full Forced=4 mask=0x2") full4_mask02 = pix_l;
        if (std::string(s.name) == "rgba8 full Forced=4 mask=0x5") full4_mask05 = pix_l;
        std::printf("PASS %-30s (exact byte parity vs %s%s)\n",
                    s.name, native_host.label.c_str(), (warp && (warp->no_output_samples & mask)) ? " + WARP" : "");
    }
    CHECK(!nonzero(sliver1_pixels) && nonzero(sliver4_pixels));
    CHECK(!nonzero(full4_mask02) && nonzero(full4_mask05) && full4_mask05 == full4_pixels);

    // R32_UINT + SV_Coverage tests: bit-exact sample coverage masks!
    const RtvScene cov_scenes[] = {
        {"cov full Forced=1", 1, UINT_MAX, full, fullscreen_quad()},
        {"cov full Forced=4", 4, UINT_MAX, full, fullscreen_quad()},
        {"cov full Forced=4 mask=0x2", 4, 0x2u, full, fullscreen_quad()},
        {"cov full Forced=4 mask=0x5", 4, 0x5u, full, fullscreen_quad()},
        {"cov scissor Forced=4", 4, UINT_MAX, clipped, fullscreen_quad()},
        {"cov diag Forced=4", 4, UINT_MAX, full, diag},
        {"cov sliver Forced=4", 4, UINT_MAX, full, sliver},
        {"cov full Forced=8", 8, UINT_MAX, full, fullscreen_quad()},
        {"cov full Forced=8 mask=0x33", 8, 0x33u, full, fullscreen_quad()},
        {"cov full Forced=16", 16, UINT_MAX, full, fullscreen_quad()},
        {"cov full Forced=16 mask=0x5555", 16, 0x5555u, full, fullscreen_quad()},
    };
    std::vector<std::uint8_t> cov_full4, cov_mask02, cov_mask05;
    for (const auto& s : cov_scenes) {
        const UINT mask = s.forced_samples ? s.forced_samples : 1u;
        if (!(layer.no_output_samples & mask) || !(native_host.no_output_samples & mask)) continue;
        std::vector<std::uint8_t> bytes_l, bytes_h, bytes_w;
        CHECK(render_rtv(layer, DXGI_FORMAT_R32_UINT, layer.shaders.ps_cov, s, bytes_l));
        CHECK(render_rtv(native_host, DXGI_FORMAT_R32_UINT, native_host.shaders.ps_cov, s, bytes_h));
        if (bytes_l != bytes_h) {
            std::fprintf(stderr, "COVERAGE MISMATCH vs %s on '%s'\n", native_host.label.c_str(), s.name);
            return false;
        }
        if (warp && (warp->no_output_samples & mask)) {
            CHECK(render_rtv(*warp, DXGI_FORMAT_R32_UINT, warp->shaders.ps_cov, s, bytes_w));
            if (bytes_l != bytes_w) {
                std::fprintf(stderr, "COVERAGE MISMATCH vs WARP on '%s'\n", s.name);
                return false;
            }
        }
        if (std::string(s.name) == "cov full Forced=4") cov_full4 = bytes_l;
        if (std::string(s.name) == "cov full Forced=4 mask=0x2") cov_mask02 = bytes_l;
        if (std::string(s.name) == "cov full Forced=4 mask=0x5") cov_mask05 = bytes_l;
        std::uint32_t sample0 = 0;
        std::memcpy(&sample0, bytes_l.data() + (5 * kWidth + 5) * 4, 4);
        std::printf("PASS %-30s interior_cov=0x%04x\n", s.name, sample0);
    }
    CHECK(!nonzero(cov_mask02) && cov_mask05 == cov_full4);

    // MRT 2x 1x-RTV TIR (RGBA8 + R32_UINT SV_Coverage): test both non-zero odd mask (0x5) and masked-out even (0xA).
    for (const RtvScene& mrt : {RtvScene{"mrt Forced=4 mask=0x5", 4, 0x5u, full, diag},
                                RtvScene{"mrt Forced=4 mask=0xA", 4, 0xAu, full, diag}}) {
        std::vector<std::uint8_t> c_l, m_l, c_h, m_h, c_w, m_w;
        CHECK(render_rtv(layer, DXGI_FORMAT_R8G8B8A8_UNORM, layer.shaders.ps_mrt, mrt, c_l, &m_l, DXGI_FORMAT_R32_UINT));
        CHECK(render_rtv(native_host, DXGI_FORMAT_R8G8B8A8_UNORM, native_host.shaders.ps_mrt, mrt, c_h, &m_h,
                         DXGI_FORMAT_R32_UINT));
        CHECK(c_l == c_h && m_l == m_h);
        if (mrt.sample_mask & 1u) CHECK(nonzero(c_l) && nonzero(m_l));
        else CHECK(!nonzero(c_l) && !nonzero(m_l));
        if (warp) {
            CHECK(render_rtv(*warp, DXGI_FORMAT_R8G8B8A8_UNORM, warp->shaders.ps_mrt, mrt, c_w, &m_w,
                             DXGI_FORMAT_R32_UINT));
            CHECK(c_l == c_w && m_l == m_w);
        }
        std::printf("PASS %-30s (both RTV0 RGBA8 and RTV1 SV_Coverage match)\n", mrt.name);
    }

    // Unbound RTV variant on a 1x-RTV TIR PSO: OMSetRenderTargets(0) with occlusion query.
    {
        const RtvScene unb{"unbound-rtv Forced=4", 4, UINT_MAX, full, fullscreen_quad()};
        std::vector<std::uint8_t> dummy_l, dummy_h, dummy_w;
        UINT64 occ_l = 0, occ_h = 0, occ_w = 0;
        CHECK(render_rtv(layer, DXGI_FORMAT_R8G8B8A8_UNORM, layer.shaders.ps_color, unb, dummy_l, nullptr,
                         DXGI_FORMAT_UNKNOWN, true, &occ_l));
        CHECK(render_rtv(native_host, DXGI_FORMAT_R8G8B8A8_UNORM, native_host.shaders.ps_color, unb, dummy_h, nullptr,
                         DXGI_FORMAT_UNKNOWN, true, &occ_h));
        if (warp) {
            CHECK(render_rtv(*warp, DXGI_FORMAT_R8G8B8A8_UNORM, warp->shaders.ps_color, unb, dummy_w, nullptr,
                             DXGI_FORMAT_UNKNOWN, true, &occ_w));
            CHECK(occ_w > 0);
        }
        CHECK(!nonzero(dummy_l) && !nonzero(dummy_h) && occ_l > 0 && occ_h > 0);
        CHECK(occ_l == occ_h || (warp && occ_l == occ_w));
        std::printf("PASS %-30s layer_occ=%llu host_occ=%llu warp_occ=%s\n", unb.name,
                    static_cast<unsigned long long>(occ_l), static_cast<unsigned long long>(occ_h),
                    warp ? std::to_string(occ_w).c_str() : "skip");
    }

    // ConservativeRaster = ON + ForcedSampleCount = 4.
    if (layer.conservative_tier != D3D12_CONSERVATIVE_RASTERIZATION_TIER_NOT_SUPPORTED &&
        native_host.conservative_tier != D3D12_CONSERVATIVE_RASTERIZATION_TIER_NOT_SUPPORTED) {
        const RtvScene cons{"conservative Forced=4", 4, UINT_MAX, full, sliver, true};
        std::vector<std::uint8_t> c_l, c_h;
        CHECK(render_rtv(layer, DXGI_FORMAT_R8G8B8A8_UNORM, layer.shaders.ps_color, cons, c_l));
        CHECK(render_rtv(native_host, DXGI_FORMAT_R8G8B8A8_UNORM, native_host.shaders.ps_color, cons, c_h));
        CHECK(nonzero(c_l));
        std::printf("PASS %-30s (non-empty conservative coverage with Forced=4; host parity=%s)\n",
                    cons.name, c_l == c_h ? "exact" : "differed (edge uncertainty)");
    }

    // Negative draw test: 4x RTV bound to a 1x-RTV TIR PSO (SampleDesc.Count=1, Forced=4) must fail replay cleanly.
    {
        auto pso_desc = base_pso(layer, layer.shaders.ps_color, 1, DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_UNKNOWN, 4);
        copycmp::Ptr<ID3D12PipelineState> pso;
        CHECK(SUCCEEDED(layer.run.device->CreateGraphicsPipelineState(&pso_desc, IID_PPV_ARGS(&pso.p))));
        D3D12_RESOURCE_DESC msaa{};
        msaa.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        msaa.Width = kWidth;
        msaa.Height = kHeight;
        msaa.DepthOrArraySize = msaa.MipLevels = 1;
        msaa.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        msaa.SampleDesc.Count = 4;
        msaa.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        copycmp::Ptr<ID3D12Resource> rt4x, vb;
        CHECK(layer.run.texture(msaa, D3D12_RESOURCE_STATE_RENDER_TARGET, rt4x));
        CHECK(upload_vertices(layer.run, fullscreen_quad(), vb));
        copycmp::Ptr<ID3D12DescriptorHeap> heap;
        const D3D12_DESCRIPTOR_HEAP_DESC hd{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1};
        CHECK(SUCCEEDED(layer.run.device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&heap.p))));
        const auto h0 = heap->GetCPUDescriptorHandleForHeapStart();
        layer.run.device->CreateRenderTargetView(rt4x.p, nullptr, h0);

        layer.run.begin();
        layer.run.list->OMSetRenderTargets(1, &h0, FALSE, nullptr);
        layer.run.list->SetGraphicsRootSignature(layer.root.p);
        layer.run.list->SetPipelineState(pso.p);
        const D3D12_VIEWPORT vp{0, 0, float(kWidth), float(kHeight), 0, 1};
        layer.run.list->RSSetViewports(1, &vp);
        layer.run.list->RSSetScissorRects(1, &full);
        const D3D12_VERTEX_BUFFER_VIEW vbv{vb->GetGPUVirtualAddress(), 48, 8};
        layer.run.list->IASetVertexBuffers(0, 1, &vbv);
        layer.run.list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        layer.run.list->DrawInstanced(6, 1, 0, 0);
        CHECK(SUCCEEDED(layer.run.list->Close()));
        ID3D12CommandList* lists[] = {layer.run.list.p};
        layer.run.queue->ExecuteCommandLists(1, lists);
        CHECK(rt::d3d12::last_execute_result(layer.run.queue.p) == leonida::ReplayResult::invalid_subresource);
        layer.run.refused = true;
        std::printf("PASS draw sample mismatch 4x-RTV into Forced=4 1x-PSO refused cleanly\n");
    }
    return true;
}

// ---------------------------------------------------------------------------------------------------------------
// Section 4: Capability gating (mixed_attachment_samples = false) & create_vulkan_device_standalone extension check.

bool test_capability_gating_and_standalone(const vktest::Device& vk, const ShaderSet& l_shaders,
                                           const std::filesystem::path& shader_dir) {
    std::printf("\n--- Capability Gating & Standalone Device Check ---\n");
    rt::d3d12::VulkanDeviceInfo info{};
    info.get_instance_proc = vk.instance_proc;
    info.instance = vk.instance;
    info.physical = vk.physical;
    info.device = vk.device;
    info.queue = vk.queue;
    info.queue_family = vk.family;
    info.buffer_device_address = vk.buffer_device_address;
    info.occlusion_query_precise = vk.occlusion_query_precise;
    info.conservative_rasterization = vk.conservative_rasterization;
    info.mixed_attachment_samples = false;
    info.features = vk.enabled;
    info.shader_dir = shader_dir.string();

    copycmp::Ptr<ID3D12Device> gated_dev;
    CHECK(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&gated_dev.p))));
    DeviceContext gated_ctx;
    CHECK(gated_ctx.init(gated_dev.p, "Gated layer", true, l_shaders));

    const D3D12_RECT full{0, 0, LONG(kWidth), LONG(kHeight)};
    const std::vector<Triangle> sliver = {
        {{{2.0625, 4.0625}, {14.0625, 4.0625}, {14.0625, 4.4375}}},
        {{{2.0625, 4.0625}, {14.0625, 4.4375}, {2.0625, 4.4375}}},
    };
    if (gated_ctx.no_output_samples & 4u) {
        UINT64 occ = 0;
        CHECK(run_occlusion_scene(gated_ctx, {"gated attachmentless Forced=4", 4, UINT_MAX, full, sliver}, occ));
        CHECK(occ > 0);
        std::printf("PASS attachmentless Forced=4 works when mixed_attachment_samples=false (occ=%llu)\n",
                    static_cast<unsigned long long>(occ));
    }

    auto rtv_pso = base_pso(gated_ctx, gated_ctx.shaders.ps_color, 1, DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_UNKNOWN, 4);
    copycmp::Ptr<ID3D12PipelineState> pso;
    CHECK(gated_ctx.run.device->CreateGraphicsPipelineState(&rtv_pso, IID_PPV_ARGS(&pso.p)) == E_NOTIMPL);
    std::printf("PASS 1x-RTV Forced=4 returns E_NOTIMPL when mixed_attachment_samples=false\n");
    if (gated_ctx.conservative_tier != D3D12_CONSERVATIVE_RASTERIZATION_TIER_NOT_SUPPORTED) {
        auto cons_rtv_pso = base_pso(gated_ctx, gated_ctx.shaders.ps_color, 1, DXGI_FORMAT_R8G8B8A8_UNORM,
                                     DXGI_FORMAT_UNKNOWN, 4);
        cons_rtv_pso.RasterizerState.ConservativeRaster = D3D12_CONSERVATIVE_RASTERIZATION_MODE_ON;
        copycmp::Ptr<ID3D12PipelineState> cons_pso;
        CHECK(gated_ctx.run.device->CreateGraphicsPipelineState(&cons_rtv_pso, IID_PPV_ARGS(&cons_pso.p)) == E_NOTIMPL);
        std::printf("PASS conservative 1x-RTV Forced=4 returns E_NOTIMPL when mixed_attachment_samples=false\n");
    }

    copycmp::Ptr<ID3D12Device> standalone;
    CHECK(SUCCEEDED(rt::d3d12::create_vulkan_device_standalone(IID_PPV_ARGS(&standalone.p))));
    const rt::d3d12::VulkanDeviceInfo* sinfo = rt::d3d12::vulkan_device_info(standalone.p);
    CHECK(sinfo && sinfo->get_instance_proc && sinfo->instance && sinfo->physical);
    const auto enum_ext = reinterpret_cast<PFN_vkEnumerateDeviceExtensionProperties>(
        sinfo->get_instance_proc(sinfo->instance, "vkEnumerateDeviceExtensionProperties"));
    CHECK(enum_ext != nullptr);
    std::uint32_t ext_count = 0;
    enum_ext(sinfo->physical, nullptr, &ext_count, nullptr);
    std::vector<VkExtensionProperties> exts(ext_count);
    enum_ext(sinfo->physical, nullptr, &ext_count, exts.data());
    char amd_env[16]{};
    const DWORD amd_len = GetEnvironmentVariableA("LEONIDA_VK_AMD_MIXED_SAMPLES", amd_env, sizeof(amd_env));
    const bool allow_amd_mixed = (amd_len == 1 && amd_env[0] == '1');
    const bool physical_has_nv_mixed = std::any_of(exts.begin(), exts.end(), [](const VkExtensionProperties& e) {
        return std::strcmp(e.extensionName, VK_NV_FRAMEBUFFER_MIXED_SAMPLES_EXTENSION_NAME) == 0;
    });
    const bool physical_has_amd_mixed = std::any_of(exts.begin(), exts.end(), [](const VkExtensionProperties& e) {
        return std::strcmp(e.extensionName, VK_AMD_MIXED_ATTACHMENT_SAMPLES_EXTENSION_NAME) == 0;
    });
    const bool expected_standalone_mixed = physical_has_nv_mixed || (allow_amd_mixed && physical_has_amd_mixed);
    CHECK(sinfo->mixed_attachment_samples == expected_standalone_mixed);
    std::printf("PASS create_vulkan_device_standalone sets mixed_attachment_samples=%d (nv=%d amd=%d opt_in=%d)\n",
                sinfo->mixed_attachment_samples ? 1 : 0, physical_has_nv_mixed ? 1 : 0,
                physical_has_amd_mixed ? 1 : 0, allow_amd_mixed ? 1 : 0);
    return true;
}

// ---------------------------------------------------------------------------------------------------------------
// Section 5: Multi-submission queue test switching 1x -> 4x attless -> 4x 1x-RTV -> 8x 1x-RTV -> 4x MSAA -> 1x.

bool render_msaa_resolve(DeviceContext& ctx, const std::vector<Triangle>& tris, std::vector<std::uint8_t>& out) {
    auto pso_desc = base_pso(ctx, ctx.shaders.ps_color, 1, DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_UNKNOWN, 0);
    pso_desc.SampleDesc.Count = 4;
    copycmp::Ptr<ID3D12PipelineState> pso;
    CHECK(SUCCEEDED(ctx.run.device->CreateGraphicsPipelineState(&pso_desc, IID_PPV_ARGS(&pso.p))));

    D3D12_RESOURCE_DESC msaa_desc{};
    msaa_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    msaa_desc.Width = kWidth;
    msaa_desc.Height = kHeight;
    msaa_desc.DepthOrArraySize = msaa_desc.MipLevels = 1;
    msaa_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    msaa_desc.SampleDesc.Count = 4;
    msaa_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;

    D3D12_RESOURCE_DESC resolved_desc = msaa_desc;
    resolved_desc.SampleDesc.Count = 1;
    resolved_desc.Flags = D3D12_RESOURCE_FLAG_NONE;

    copycmp::Ptr<ID3D12Resource> msaa, resolved, rb, vb;
    copycmp::Ptr<ID3D12DescriptorHeap> heap;
    CHECK(ctx.run.texture(msaa_desc, D3D12_RESOURCE_STATE_RENDER_TARGET, msaa));
    CHECK(ctx.run.texture(resolved_desc, D3D12_RESOURCE_STATE_RESOLVE_DEST, resolved));
    CHECK(ctx.run.buffer(D3D12_HEAP_TYPE_READBACK, kPitch * kHeight, D3D12_RESOURCE_STATE_COPY_DEST, rb));
    CHECK(upload_vertices(ctx.run, tris, vb));
    const D3D12_DESCRIPTOR_HEAP_DESC hd{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1};
    CHECK(SUCCEEDED(ctx.run.device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&heap.p))));
    const auto h0 = heap->GetCPUDescriptorHandleForHeapStart();
    ctx.run.device->CreateRenderTargetView(msaa.p, nullptr, h0);

    ctx.run.begin();
    const float clear[4] = {};
    ctx.run.list->ClearRenderTargetView(h0, clear, 0, nullptr);
    ctx.run.list->OMSetRenderTargets(1, &h0, FALSE, nullptr);
    ctx.run.list->SetGraphicsRootSignature(ctx.root.p);
    ctx.run.list->SetPipelineState(pso.p);
    const D3D12_VIEWPORT vp{0, 0, float(kWidth), float(kHeight), 0, 1};
    const D3D12_RECT full{0, 0, LONG(kWidth), LONG(kHeight)};
    ctx.run.list->RSSetViewports(1, &vp);
    ctx.run.list->RSSetScissorRects(1, &full);
    const D3D12_VERTEX_BUFFER_VIEW vbv{vb->GetGPUVirtualAddress(), UINT(tris.size() * 24), 8};
    ctx.run.list->IASetVertexBuffers(0, 1, &vbv);
    ctx.run.list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    ctx.run.list->DrawInstanced(UINT(tris.size() * 3), 1, 0, 0);

    ctx.run.barrier(msaa.p, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_RESOLVE_SOURCE);
    ctx.run.list->ResolveSubresource(resolved.p, 0, msaa.p, 0, DXGI_FORMAT_R8G8B8A8_UNORM);
    ctx.run.barrier(resolved.p, D3D12_RESOURCE_STATE_RESOLVE_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
    D3D12_TEXTURE_COPY_LOCATION src_loc{}, dst_loc{};
    src_loc.pResource = resolved.p;
    src_loc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    dst_loc.pResource = rb.p;
    dst_loc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst_loc.PlacedFootprint.Footprint = {DXGI_FORMAT_R8G8B8A8_UNORM, kWidth, kHeight, 1, kPitch};
    ctx.run.list->CopyTextureRegion(&dst_loc, 0, 0, 0, &src_loc, nullptr);

    const std::string err = ctx.run.submit();
    if (!err.empty()) {
        std::fprintf(stderr, "%s 4x MSAA resolve failed: %s\n", ctx.label.c_str(), err.c_str());
        return false;
    }
    out.resize(std::size_t(kWidth) * kHeight * 4);
    void* data = nullptr;
    const D3D12_RANGE rr{0, std::size_t(kPitch) * kHeight};
    CHECK(SUCCEEDED(rb->Map(0, &rr, &data)));
    for (UINT y = 0; y < kHeight; ++y)
        std::memcpy(out.data() + y * kWidth * 4, static_cast<const std::uint8_t*>(data) + y * kPitch, kWidth * 4);
    const D3D12_RANGE none{};
    rb->Unmap(0, &none);
    return true;
}

bool test_multi_submission_pso_switching(DeviceContext& layer, DeviceContext& native_host, DeviceContext* warp,
                                         bool has_mixed_ext) {
    if (!has_mixed_ext) return true;
    std::printf("\n--- Multi-Submission PSO Switching Sequence ---\n");
    const D3D12_RECT full{0, 0, LONG(kWidth), LONG(kHeight)};
    const std::vector<Triangle> diag = {{{{0.0, 0.0}, {double(kWidth), 0.0}, {0.0, double(kHeight)}}}};
    const std::vector<Triangle> sliver = {
        {{{2.0625, 4.0625}, {14.0625, 4.0625}, {14.0625, 4.4375}}},
        {{{2.0625, 4.0625}, {14.0625, 4.4375}, {2.0625, 4.4375}}},
    };
    const auto nonzero = [](const std::vector<std::uint8_t>& v) {
        return std::any_of(v.begin(), v.end(), [](std::uint8_t b) { return b != 0; });
    };

    // 1) 1x normal (Forced=0) RGBA8 diag
    {
        std::vector<std::uint8_t> l, h, w;
        const RtvScene s{"seq1 1x normal diag", 0, UINT_MAX, full, diag};
        CHECK(render_rtv(layer, DXGI_FORMAT_R8G8B8A8_UNORM, layer.shaders.ps_color, s, l));
        CHECK(render_rtv(native_host, DXGI_FORMAT_R8G8B8A8_UNORM, native_host.shaders.ps_color, s, h));
        CHECK(l == h && nonzero(l));
        if (warp) { CHECK(render_rtv(*warp, DXGI_FORMAT_R8G8B8A8_UNORM, warp->shaders.ps_color, s, w)); CHECK(l == w); }
        std::printf("PASS seq 1: 1x normal RGBA8 diag\n");
    }
    // 2) 4x attachmentless TIR occlusion on sliver
    if ((layer.no_output_samples & 4u) && (native_host.no_output_samples & 4u)) {
        UINT64 ol = 0, oh = 0, ow = 0;
        const OcclusionScene s{"seq2 4x attless sliver", 4, UINT_MAX, full, sliver};
        CHECK(run_occlusion_scene(layer, s, ol) && ol > 0);
        CHECK(run_occlusion_scene(native_host, s, oh) && oh > 0);
        if (warp && (warp->no_output_samples & 4u)) { CHECK(run_occlusion_scene(*warp, s, ow) && ow > 0); }
        std::printf("PASS seq 2: 4x attachmentless TIR occlusion (l=%llu h=%llu)\n",
                    static_cast<unsigned long long>(ol), static_cast<unsigned long long>(oh));
    }
    // 3) 4x 1x-RTV TIR R32_UINT SV_Coverage on diag
    if ((layer.no_output_samples & 4u) && (native_host.no_output_samples & 4u)) {
        std::vector<std::uint8_t> l, h, w;
        const RtvScene s{"seq3 4x SV_Coverage diag", 4, UINT_MAX, full, diag};
        CHECK(render_rtv(layer, DXGI_FORMAT_R32_UINT, layer.shaders.ps_cov, s, l));
        CHECK(render_rtv(native_host, DXGI_FORMAT_R32_UINT, native_host.shaders.ps_cov, s, h));
        CHECK(l == h && nonzero(l));
        if (warp && (warp->no_output_samples & 4u)) {
            CHECK(render_rtv(*warp, DXGI_FORMAT_R32_UINT, warp->shaders.ps_cov, s, w));
            CHECK(l == w);
        }
        std::printf("PASS seq 3: 4x 1x-RTV TIR SV_Coverage diag\n");
    }
    // 4) 8x 1x-RTV TIR R32_UINT SV_Coverage on sliver
    if ((layer.no_output_samples & 8u) && (native_host.no_output_samples & 8u)) {
        std::vector<std::uint8_t> l, h, w;
        const RtvScene s{"seq4 8x SV_Coverage sliver", 8, UINT_MAX, full, sliver};
        CHECK(render_rtv(layer, DXGI_FORMAT_R32_UINT, layer.shaders.ps_cov, s, l));
        CHECK(render_rtv(native_host, DXGI_FORMAT_R32_UINT, native_host.shaders.ps_cov, s, h));
        CHECK(l == h && nonzero(l));
        if (warp && (warp->no_output_samples & 8u)) {
            CHECK(render_rtv(*warp, DXGI_FORMAT_R32_UINT, warp->shaders.ps_cov, s, w));
            CHECK(l == w);
        }
        std::printf("PASS seq 4: 8x 1x-RTV TIR SV_Coverage sliver\n");
    }
    // 5) 4x ordinary MSAA RGBA8 sliver resolved to 1x
    {
        std::vector<std::uint8_t> l, h, w;
        CHECK(render_msaa_resolve(layer, sliver, l));
        CHECK(render_msaa_resolve(native_host, sliver, h));
        CHECK(l == h && nonzero(l));
        if (warp) {
            CHECK(render_msaa_resolve(*warp, sliver, w));
            CHECK(l.size() == w.size());
            // Hardware RTX 5060 (Vulkan and D3D12) resolves 0.5 to 127 while WARP rounds 0.5*255 to 128; require
            // identical non-zero coverage footprint and <= 1 LSB UNORM resolve difference.
            for (std::size_t i = 0; i < l.size(); ++i) {
                if ((l[i] == 0) != (w[i] == 0) || std::abs(int(l[i]) - int(w[i])) > 1) {
                    std::fprintf(stderr, "seq 5 MSAA resolve diff at byte %zu: layer=%u warp=%u\n", i, l[i], w[i]);
                    return false;
                }
            }
        }
        std::printf("PASS seq 5: 4x ordinary MSAA resolved sliver\n");
    }
    // 6) 1x normal (Forced=0) RGBA8 sliver -> back to 1x center sampling (all zeros)
    {
        std::vector<std::uint8_t> l, h, w;
        const RtvScene s{"seq6 1x normal sliver", 0, UINT_MAX, full, sliver};
        CHECK(render_rtv(layer, DXGI_FORMAT_R8G8B8A8_UNORM, layer.shaders.ps_color, s, l));
        CHECK(render_rtv(native_host, DXGI_FORMAT_R8G8B8A8_UNORM, native_host.shaders.ps_color, s, h));
        CHECK(l == h && !nonzero(l));
        if (warp) { CHECK(render_rtv(*warp, DXGI_FORMAT_R8G8B8A8_UNORM, warp->shaders.ps_color, s, w)); CHECK(l == w); }
        std::printf("PASS seq 6: 1x normal RGBA8 sliver (clean return to 1x)\n");
    }
    return true;
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (hostwine::host_is_wine()) {
        std::puts("SKIP: native Windows D3D12 and WARP required for ForcedSampleCount differential tests");
        return 77;
    }

    vktest::Device vk;
    vk.request_buffer_device_address = true;
    const auto status = vk.initialize();
    if (status == vktest::DeviceStatus::unavailable) return 77;
    REQUIRE(status == vktest::DeviceStatus::ready);
    if (!vk.buffer_device_address) {
        std::puts("SKIP: bufferDeviceAddress required");
        return 77;
    }

    // Prepare temporary shader library directory with hand-encoded SPIR-V modules + FSSM index.
    const testfiles::ScratchDir scratch("leonida_forced_sample_count");
    const auto dir = scratch.path();
    const auto vs_prog = d3d12test::fssm_program({0x54495201u, 0x01u});
    const auto pse_prog = d3d12test::fssm_program({0x54495202u, 0x02u});
    const auto psc_prog = d3d12test::fssm_program({0x54495203u, 0x03u});
    const auto psv_prog = d3d12test::fssm_program({0x54495204u, 0x04u});
    const auto psm_prog = d3d12test::fssm_program({0x54495205u, 0x05u});
    REQUIRE(write_spv(dir / "vs_tir.spv", build_vs_tir()));
    REQUIRE(write_spv(dir / "ps_empty.spv", build_ps_empty()));
    REQUIRE(write_spv(dir / "ps_color.spv", build_ps_color()));
    REQUIRE(write_spv(dir / "ps_cov.spv", build_ps_cov()));
    REQUIRE(write_spv(dir / "ps_mrt.spv", build_ps_mrt()));
    {
        std::ofstream idx(dir / "index.json", std::ios::binary);
        idx << "{\n"
               "  \"vs_tir\": {\"stage\": \"vs\", \"fssm_sha256\": \"" << leonida::sha256_hex(vs_prog) << "\", "
               "\"bindings\": [], \"push_constant_bytes\": 0, \"input_signature\": ["
               "{\"semantic\": \"POSITION\", \"semantic_index\": 0, \"register\": 0, \"mask\": 3, "
               "\"rw_mask\": 3, \"system_value\": 0, \"component_type\": \"float\"}]},\n"
               "  \"ps_empty\": {\"stage\": \"ps\", \"fssm_sha256\": \"" << leonida::sha256_hex(pse_prog) << "\", "
               "\"bindings\": [], \"push_constant_bytes\": 0, \"outputs\": []},\n"
               "  \"ps_color\": {\"stage\": \"ps\", \"fssm_sha256\": \"" << leonida::sha256_hex(psc_prog) << "\", "
               "\"bindings\": [], \"push_constant_bytes\": 0, \"outputs\": [0]},\n"
               "  \"ps_cov\": {\"stage\": \"ps\", \"fssm_sha256\": \"" << leonida::sha256_hex(psv_prog) << "\", "
               "\"bindings\": [], \"push_constant_bytes\": 0, \"outputs\": [0]},\n"
               "  \"ps_mrt\": {\"stage\": \"ps\", \"fssm_sha256\": \"" << leonida::sha256_hex(psm_prog) << "\", "
               "\"bindings\": [], \"push_constant_bytes\": 0, \"outputs\": [0, 1]}\n"
               "}\n";
    }
    LayerBlobs l_blobs{
        d3d12test::shader_container(vs_prog),
        d3d12test::shader_container(pse_prog),
        d3d12test::shader_container(psc_prog),
        d3d12test::shader_container(psv_prog),
        d3d12test::shader_container(psm_prog),
    };

    // Query device properties/extensions for layer adapter LUID & mixed-samples availability.
    VkPhysicalDeviceIDProperties id_props{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
    VkPhysicalDeviceProperties2 props2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &id_props};
    vk.instance_fn<PFN_vkGetPhysicalDeviceProperties2>("vkGetPhysicalDeviceProperties2")(vk.physical, &props2);
    if (id_props.deviceLUIDValid) {
        LUID luid{};
        std::memcpy(&luid, id_props.deviceLUID, sizeof(luid));
        copycmp::layer_adapter() = luid;
    }

    const bool has_mixed_ext = vk.mixed_attachment_samples;

    // Compile HLSL equivalents via d3dcompiler_47.
    const HMODULE comp_mod = LoadLibraryExA("d3dcompiler_47.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    const auto compile = comp_mod
                             ? reinterpret_cast<pD3DCompile>(reinterpret_cast<void*>(GetProcAddress(comp_mod, "D3DCompile")))
                             : nullptr;
    if (!compile) {
        std::puts("SKIP: d3dcompiler_47.dll required");
        return 77;
    }
    const char* hlsl =
        "float4 vs(float2 p : POSITION) : SV_Position { return float4(p, 0.5, 1.0); }\n"
        "void ps_empty() {}\n"
        "float4 ps_color() : SV_Target { return float4(1.0, 0.25, 0.0, 1.0); }\n"
        "uint ps_cov(uint cov : SV_Coverage) : SV_Target { return cov; }\n"
        "struct MrtOut { float4 c : SV_Target0; uint m : SV_Target1; };\n"
        "MrtOut ps_mrt(uint cov : SV_Coverage) {\n"
        "    MrtOut o; o.c = float4(1.0, 0.25, 0.0, 1.0); o.m = cov; return o;\n"
        "}\n";
    HostBlobs h_blobs;
    REQUIRE(compile_hlsl(compile, hlsl, "vs", "vs_5_0", h_blobs.vs));
    REQUIRE(compile_hlsl(compile, hlsl, "ps_empty", "ps_5_0", h_blobs.ps_empty));
    REQUIRE(compile_hlsl(compile, hlsl, "ps_color", "ps_5_0", h_blobs.ps_color));
    REQUIRE(compile_hlsl(compile, hlsl, "ps_cov", "ps_5_0", h_blobs.ps_cov));
    REQUIRE(compile_hlsl(compile, hlsl, "ps_mrt", "ps_5_0", h_blobs.ps_mrt));

    // Create Vulkan D3D12 layer device (using standalone wrapper OR vk.device if available).
    copycmp::Ptr<ID3D12Device> ours_dev, host_dev, warp_dev;
    rt::d3d12::VulkanDeviceInfo info{};
    info.get_instance_proc = vk.instance_proc;
    info.instance = vk.instance;
    info.physical = vk.physical;
    info.device = vk.device;
    info.queue = vk.queue;
    info.queue_family = vk.family;
    info.buffer_device_address = vk.buffer_device_address;
    info.occlusion_query_precise = vk.occlusion_query_precise;
    info.conservative_rasterization = vk.conservative_rasterization;
    info.mixed_attachment_samples = vk.mixed_attachment_samples;
    info.features = vk.enabled;
    info.shader_dir = dir.string();
    REQUIRE(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&ours_dev.p))));

    std::string host_name;
    if (!copycmp::create_host(host_dev, host_name)) {
        std::puts("SKIP: native host D3D12 adapter unavailable");
        return 77;
    }
    const HMODULE d3d12_mod = LoadLibraryExA("d3d12.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    const HMODULE dxgi_mod = LoadLibraryExA("dxgi.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    REQUIRE(d3d12_mod && dxgi_mod);
    const auto d3d12_create = reinterpret_cast<PFN_D3D12_CREATE_DEVICE>(
        reinterpret_cast<void*>(GetProcAddress(d3d12_mod, "D3D12CreateDevice")));
    using CreateFactoryFn = HRESULT(WINAPI*)(REFIID, void**);
    const auto create_factory = reinterpret_cast<CreateFactoryFn>(
        reinterpret_cast<void*>(GetProcAddress(dxgi_mod, "CreateDXGIFactory1")));
    copycmp::Ptr<IDXGIFactory4> factory;
    copycmp::Ptr<IDXGIAdapter> warp_adapter;
    bool have_warp = false;
    if (d3d12_create && create_factory && SUCCEEDED(create_factory(IID_PPV_ARGS(&factory.p))) &&
        SUCCEEDED(factory->EnumWarpAdapter(IID_PPV_ARGS(&warp_adapter.p))) &&
        SUCCEEDED(d3d12_create(warp_adapter.p, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&warp_dev.p)))) {
        have_warp = true;
    }

    DeviceContext layer_ctx, host_ctx, warp_ctx;
    REQUIRE(layer_ctx.init(ours_dev.p, "Vulkan layer", true, l_blobs.view()));
    REQUIRE(host_ctx.init(host_dev.p, host_name, false, h_blobs.view()));
    if (have_warp) REQUIRE(warp_ctx.init(warp_dev.p, "WARP", false, h_blobs.view()));

    const unsigned errors_before = vk.validation_errors();
    const bool ok_val = test_pso_validation(layer_ctx, host_ctx, have_warp ? &warp_ctx : nullptr, has_mixed_ext);
    const bool ok_occ = test_attachmentless_occlusion(layer_ctx, host_ctx, have_warp ? &warp_ctx : nullptr,
                                                      props2.properties.vendorID);
    const bool ok_rtv = test_mixed_sample_rtvs(layer_ctx, host_ctx, have_warp ? &warp_ctx : nullptr, has_mixed_ext);
    const bool ok_gate = test_capability_gating_and_standalone(vk, l_blobs.view(), dir);
    const bool ok_seq =
        test_multi_submission_pso_switching(layer_ctx, host_ctx, have_warp ? &warp_ctx : nullptr, has_mixed_ext);

    const unsigned val_errors = vk.validation_errors() - errors_before;
    REQUIRE(ok_val && ok_occ && ok_rtv && ok_gate && ok_seq && val_errors == 0);
    std::printf("\nd3d12_forced_sample_count: all checks passed\n");
    return 0;
}
