// The identity of a graphics pipeline description (d3d12_graphics_identity.h) with synthetic descriptions only: no
// D3D12 device, no GPU, no game data. The shader "containers" are hand-built DXBC headers with an invented checksum.
//
// The identity must tell two descriptions apart by what defines the pipeline (state blocks, the checksum of each
// stage's shader, formats, topology, an MS or AS stage) and must not move with what differs from run to run (the
// root signature the title builds, an address, the copy of the bytecode in memory, flags and a cached blob).
#include <d3d12.h>

#include <cctype>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "d3d12_graphics_identity.h"

namespace {

int g_failures = 0;

void check(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++g_failures;
}

using rt::d3d12::StencilFaceMasks;

// A DXBC header (magic, 16 checksum bytes, version, size, no parts) whose checksum is `seed` repeated.
std::vector<std::uint8_t> container(std::uint8_t seed, std::size_t size = 32) {
    std::vector<std::uint8_t> bytes(size, 0);
    std::memcpy(bytes.data(), "DXBC", 4);
    for (int i = 0; i < 16; ++i) bytes[4 + i] = std::uint8_t(seed + i);
    return bytes;
}

D3D12_SHADER_BYTECODE view(const std::vector<std::uint8_t>& bytes) { return {bytes.data(), bytes.size()}; }

struct Fixture {
    std::vector<std::uint8_t> vs = container(0x10), ps = container(0x40), ms = container(0x70), as = container(0x90);
    D3D12_INPUT_ELEMENT_DESC element{"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0,
                                     D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0};
    D3D12_GRAPHICS_PIPELINE_STATE_DESC desc{};

    Fixture() {
        desc.VS = view(vs);
        desc.PS = view(ps);
        for (auto& target : desc.BlendState.RenderTarget) {
            target.SrcBlend = target.SrcBlendAlpha = D3D12_BLEND_ONE;
            target.DestBlend = target.DestBlendAlpha = D3D12_BLEND_ZERO;
            target.BlendOp = target.BlendOpAlpha = D3D12_BLEND_OP_ADD;
            target.LogicOp = D3D12_LOGIC_OP_NOOP;
            target.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
        }
        desc.SampleMask = 0xFFFFFFFFu;
        desc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
        desc.RasterizerState.CullMode = D3D12_CULL_MODE_BACK;
        desc.RasterizerState.DepthClipEnable = TRUE;
        desc.DepthStencilState.DepthEnable = TRUE;
        desc.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
        desc.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_LESS;
        desc.InputLayout = {&element, 1};
        desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        desc.NumRenderTargets = 2;
        desc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        desc.RTVFormats[1] = DXGI_FORMAT_R16G16B16A16_FLOAT;
        desc.DSVFormat = DXGI_FORMAT_D32_FLOAT;
        desc.SampleDesc = {1, 0};
    }
};

std::string id(const Fixture& f) { return rt::d3d12::graphics_identity(f.desc); }

// A description with the VS and the input stage of a mesh pipeline left out and `f.ms` as its mesh stage.
std::string mesh_id(Fixture f, const std::vector<std::uint8_t>& mesh, const StencilFaceMasks* back = nullptr,
                    bool depth_bounds = false, const D3D12_SHADER_BYTECODE* task = nullptr) {
    f.desc.VS = {};
    f.desc.InputLayout = {};
    f.desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_UNDEFINED;
    const D3D12_SHADER_BYTECODE ms = view(mesh);
    return rt::d3d12::graphics_identity(f.desc, &ms, task, back, depth_bounds);
}

bool hex64(const std::string& s) {
    if (s.size() != 64) return false;
    for (const char c : s)
        if (!std::isdigit(static_cast<unsigned char>(c)) && !(c >= 'a' && c <= 'f')) return false;
    return true;
}

}  // namespace

int main() {
    const Fixture base;
    const std::string reference = id(base);
    check(hex64(reference), "the identity is 64 lowercase hex digits");
    check(id(Fixture{}) == reference, "two descriptions built the same way share an identity");

    // Frozen values. key_graphics_state also feeds pipeline_key, whose hash names entries of a cache on disk: a
    // field moved, widened or reordered there would orphan every user's cache without any other test noticing.
    // The first value was computed with the code of main before the function was shared.
    {
        leonida::cached_pso::KeyBuilder state;
        rt::d3d12::key_graphics_state(state, base.desc, true);
        check(leonida::sha256_hex(state.data()) == "484d5f08252d7e0e80ca370520841e2b9fe2cc59306208e70508d60e841ac3e1",
              "the bytes key_graphics_state feeds the cache key are the ones main hashed");
        check(reference == "d2b87fb92c4fbc33728f50dde1c1011baa729388403a47fce753bad3282c367e",
              "the identity of the reference description is the frozen one");
    }

    // What is not part of the description: a copy of the bytecode elsewhere in memory, the root signature, flags,
    // a cached blob.
    {
        Fixture f;
        const auto vs_copy = f.vs, ps_copy = f.ps;
        f.desc.VS = view(vs_copy);
        f.desc.PS = view(ps_copy);
        check(id(f) == reference, "the same bytecode at another address keeps the identity");
    }
    {
        Fixture f;
        f.desc.pRootSignature = reinterpret_cast<ID3D12RootSignature*>(std::uintptr_t(0x1000));
        const std::string first = id(f);
        f.desc.pRootSignature = reinterpret_cast<ID3D12RootSignature*>(std::uintptr_t(0x2468ace0));
        check(first == reference && id(f) == reference, "another root signature (or its address) keeps the identity");
    }
    {
        Fixture f;
        f.desc.Flags = D3D12_PIPELINE_STATE_FLAG_TOOL_DEBUG;
        const char blob[4] = {1, 2, 3, 4};
        f.desc.CachedPSO = {blob, sizeof blob};
        check(id(f) == reference, "flags and a cached blob keep the identity");
    }
    {
        Fixture f;
        const std::string name_copy = "POSITION";
        f.element.SemanticName = name_copy.c_str();
        check(id(f) == reference, "an input semantic name at another address keeps the identity");
    }

    // What defines it: each change moves the identity.
    const auto differs = [&](const char* what, auto&& change) {
        Fixture f;
        change(f);
        check(id(f) != reference, what);
    };
    differs("a vertex shader with another checksum changes it", [](Fixture& f) { f.vs = container(0x11); f.desc.VS = view(f.vs); });
    differs("a pixel shader with another checksum changes it", [](Fixture& f) { f.ps = container(0x41); f.desc.PS = view(f.ps); });
    differs("a pixel shader of another size changes it", [](Fixture& f) { f.ps = container(0x40, 48); f.desc.PS = view(f.ps); });
    differs("a geometry shader changes it", [](Fixture& f) { f.desc.GS = view(f.ms); });
    differs("no pixel shader changes it", [](Fixture& f) { f.desc.PS = {}; });
    differs("a blend factor changes it", [](Fixture& f) { f.desc.BlendState.RenderTarget[1].SrcBlend = D3D12_BLEND_SRC_ALPHA; });
    differs("blending on changes it", [](Fixture& f) { f.desc.BlendState.RenderTarget[0].BlendEnable = TRUE; });
    differs("a write mask changes it", [](Fixture& f) { f.desc.BlendState.RenderTarget[0].RenderTargetWriteMask = 0x7; });
    differs("the sample mask changes it", [](Fixture& f) { f.desc.SampleMask = 0xF; });
    differs("the cull mode changes it", [](Fixture& f) { f.desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE; });
    differs("the depth bias changes it", [](Fixture& f) { f.desc.RasterizerState.DepthBias = 4; });
    differs("the depth function changes it", [](Fixture& f) { f.desc.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_GREATER; });
    differs("stencil on changes it", [](Fixture& f) { f.desc.DepthStencilState.StencilEnable = TRUE; });
    differs("a stencil operation changes it", [](Fixture& f) { f.desc.DepthStencilState.BackFace.StencilPassOp = D3D12_STENCIL_OP_REPLACE; });
    differs("a render target format changes it", [](Fixture& f) { f.desc.RTVFormats[1] = DXGI_FORMAT_R11G11B10_FLOAT; });
    differs("the number of render targets changes it", [](Fixture& f) { f.desc.NumRenderTargets = 1; });
    differs("the depth format changes it", [](Fixture& f) { f.desc.DSVFormat = DXGI_FORMAT_D24_UNORM_S8_UINT; });
    differs("the sample count changes it", [](Fixture& f) { f.desc.SampleDesc.Count = 4; });
    differs("the topology type changes it", [](Fixture& f) { f.desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE; });
    differs("the strip cut value changes it", [](Fixture& f) { f.desc.IBStripCutValue = D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_0xFFFF; });
    differs("an input element format changes it", [](Fixture& f) { f.element.Format = DXGI_FORMAT_R32G32_FLOAT; });
    differs("an input semantic name changes it", [](Fixture& f) { f.element.SemanticName = "TEXCOORD"; });
    differs("an input offset changes it", [](Fixture& f) { f.element.AlignedByteOffset = 12; });
    differs("no input element changes it", [](Fixture& f) { f.desc.InputLayout = {}; });

    // The stream path: an MS stage, an AS stage, the back face's masks and the depth bounds test.
    const Fixture mesh_base;
    const std::string mesh_ref = mesh_id(mesh_base, mesh_base.ms);
    check(hex64(mesh_ref) && mesh_ref != reference, "a mesh description has an identity of its own");
    check(mesh_id(Fixture{}, container(0x70)) == mesh_ref, "the same mesh stage in another buffer keeps the identity");
    check(mesh_id(mesh_base, container(0x71)) != mesh_ref, "a mesh stage with another checksum changes it");
    {
        // Two pipelines that differ only in the mesh stage: the cache key of the stream path does not tell them apart.
        const std::string other = mesh_id(mesh_base, container(0x72));
        check(other != mesh_ref && other != mesh_id(mesh_base, container(0x71)), "two mesh stages give two identities");
    }
    {
        const D3D12_SHADER_BYTECODE task = view(mesh_base.as);
        check(mesh_id(mesh_base, mesh_base.ms, nullptr, false, &task) != mesh_ref, "an amplification stage changes it");
    }
    {
        const StencilFaceMasks masks{0x0f, 0xf0};
        const StencilFaceMasks other{0x0f, 0xf1};
        check(mesh_id(mesh_base, mesh_base.ms, &masks) != mesh_ref, "the back face's stencil masks change it");
        check(mesh_id(mesh_base, mesh_base.ms, &masks) != mesh_id(mesh_base, mesh_base.ms, &other),
              "another back face write mask changes it");
        check(mesh_id(mesh_base, mesh_base.ms, &masks) == mesh_id(Fixture{}, container(0x70), &masks),
              "the same back face masks keep it");
    }
    check(mesh_id(mesh_base, mesh_base.ms, nullptr, true) != mesh_ref, "the depth bounds test changes it");
    {
        // D3D12 ignores the input layout, the strip cut and the topology of a mesh pipeline, so they do not count.
        Fixture f;
        const D3D12_SHADER_BYTECODE ms = view(f.ms);
        f.desc.VS = {};
        const std::string with_input = rt::d3d12::graphics_identity(f.desc, &ms);
        check(with_input == mesh_ref, "a mesh pipeline ignores the input layout and topology the description carries");
    }
    {
        Fixture f;
        f.desc.pRootSignature = reinterpret_cast<ID3D12RootSignature*>(std::uintptr_t(0x77770));
        check(mesh_id(f, f.ms) == mesh_ref, "another root signature keeps a mesh identity");
    }

    std::printf(g_failures ? "FAILED: %d\n" : "all passed\n", g_failures);
    return g_failures ? 1 : 0;
}
