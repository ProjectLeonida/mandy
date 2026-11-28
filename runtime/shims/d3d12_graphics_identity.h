// The identity of a graphics pipeline description: a hash of what defines the pipeline and nothing of the run.
//
// pipeline_key (ID3D12PipelineState::GetCachedBlob, the `key=` of the pipeline-compile log line) is what the PC
// runtime matches a CachedPSO against. It names the root signature (its content, or its address for an object that is
// not ours, which can differ between runs), it is not set on the stream creation path, and it has no MS or AS.
// The identity below is for finding the same pipeline again in another run, so it takes none of those: the decoded
// state blocks, the checksum of the shader of every stage, the formats and the topology the description declares.
// The root signature, an address and a pointer are not in it. Nothing here reads a device or changes what is built.
#pragma once

#include "../src/cached_pso.h"
#include "../src/shader_library.h"

#include <d3d12.h>

#include <algorithm>
#include <bit>
#include <climits>
#include <cstdint>
#include <cstring>
#include <string>

namespace rt::d3d12 {

// The back face's stencil masks of a DEPTH_STENCIL2 subobject. D3D12_DEPTH_STENCIL_DESC has one read and one write
// mask for both faces, so the face masks travel beside the description.
struct StencilFaceMasks {
    UINT8 read = D3D12_DEFAULT_STENCIL_READ_MASK;
    UINT8 write = D3D12_DEFAULT_STENCIL_WRITE_MASK;
};

// A shader as the key sees it, without reading the whole program on every PSO: a DXBC container already carries a
// checksum of its contents (bytes 4..19), so that and the size stand for it. A container without one (all zero, as
// hand-built test containers have) or other bytecode is hashed with FNV-1a.
inline void key_shader(leonida::cached_pso::KeyBuilder& key, const char* stage, const D3D12_SHADER_BYTECODE& code) {
    const auto* bytes = static_cast<const std::uint8_t*>(code.pShaderBytecode);
    const std::size_t size = bytes ? code.BytecodeLength : 0;
    key.tag(stage).u32(std::uint32_t(size)).u32(std::uint32_t(std::uint64_t(size) >> 32));
    if (size >= 20 && std::memcmp(bytes, "DXBC", 4) == 0 &&
        std::any_of(bytes + 4, bytes + 20, [](std::uint8_t b) { return b != 0; })) {
        key.bytes(bytes + 4, 16);
        return;
    }
    std::uint64_t h = 0xcbf29ce484222325ull;
    for (std::size_t i = 0; i < size; ++i) h = (h ^ bytes[i]) * 0x100000001b3ull;
    key.u32(std::uint32_t(h)).u32(std::uint32_t(h >> 32));
}

// Everything of a graphics description but its root signature, in the order pipeline_key has always hashed it.
// `input_assembly` false leaves out the input layout, the strip cut and the topology type: D3D12 ignores them for a
// mesh pipeline.
inline void key_graphics_state(leonida::cached_pso::KeyBuilder& key, const D3D12_GRAPHICS_PIPELINE_STATE_DESC& desc,
                               bool input_assembly) {
    key_shader(key, "vs", desc.VS);
    key_shader(key, "ps", desc.PS);
    key_shader(key, "ds", desc.DS);
    key_shader(key, "hs", desc.HS);
    key_shader(key, "gs", desc.GS);
    const D3D12_STREAM_OUTPUT_DESC& so = desc.StreamOutput;
    key.u32(so.pSODeclaration ? so.NumEntries : 0);
    for (UINT i = 0; so.pSODeclaration && i < so.NumEntries; ++i) {
        const D3D12_SO_DECLARATION_ENTRY& e = so.pSODeclaration[i];
        key.u32(e.Stream).text(e.SemanticName).u32(e.SemanticIndex).u32(e.StartComponent).u32(e.ComponentCount);
        key.u32(e.OutputSlot);
    }
    key.u32(so.pBufferStrides ? so.NumStrides : 0);
    for (UINT i = 0; so.pBufferStrides && i < so.NumStrides; ++i) key.u32(so.pBufferStrides[i]);
    key.u32(so.RasterizedStream);
    const D3D12_BLEND_DESC& blend = desc.BlendState;
    key.u32(blend.AlphaToCoverageEnable).u32(blend.IndependentBlendEnable);
    for (const D3D12_RENDER_TARGET_BLEND_DESC& t : blend.RenderTarget)
        key.u32(t.BlendEnable).u32(t.LogicOpEnable).u32(t.SrcBlend).u32(t.DestBlend).u32(t.BlendOp).u32(t.SrcBlendAlpha)
            .u32(t.DestBlendAlpha).u32(t.BlendOpAlpha).u32(t.LogicOp).u32(t.RenderTargetWriteMask);
    key.u32(desc.SampleMask);
    const D3D12_RASTERIZER_DESC& r = desc.RasterizerState;
    std::uint32_t bias_clamp = 0, slope = 0;
    std::memcpy(&bias_clamp, &r.DepthBiasClamp, 4);
    std::memcpy(&slope, &r.SlopeScaledDepthBias, 4);
    key.u32(r.FillMode).u32(r.CullMode).u32(r.FrontCounterClockwise).u32(std::uint32_t(r.DepthBias)).u32(bias_clamp);
    key.u32(slope).u32(r.DepthClipEnable).u32(r.MultisampleEnable).u32(r.AntialiasedLineEnable);
    key.u32(r.ForcedSampleCount).u32(r.ConservativeRaster);
    const D3D12_DEPTH_STENCIL_DESC& d = desc.DepthStencilState;
    key.u32(d.DepthEnable).u32(d.DepthWriteMask).u32(d.DepthFunc).u32(d.StencilEnable).u32(d.StencilReadMask);
    key.u32(d.StencilWriteMask);
    for (const D3D12_DEPTH_STENCILOP_DESC& face : {d.FrontFace, d.BackFace})
        key.u32(face.StencilFailOp).u32(face.StencilDepthFailOp).u32(face.StencilPassOp).u32(face.StencilFunc);
    if (input_assembly) {
        const D3D12_INPUT_LAYOUT_DESC& input = desc.InputLayout;
        const UINT input_count = input.pInputElementDescs
                                     ? std::min<UINT>(input.NumElements, D3D12_IA_VERTEX_INPUT_STRUCTURE_ELEMENT_COUNT)
                                     : 0;
        key.u32(input_count);
        for (UINT i = 0; i < input_count; ++i) {
            const D3D12_INPUT_ELEMENT_DESC& e = input.pInputElementDescs[i];
            key.text(e.SemanticName).u32(e.SemanticIndex).u32(e.Format).u32(e.InputSlot).u32(e.AlignedByteOffset);
            key.u32(e.InputSlotClass).u32(e.InstanceDataStepRate);
        }
        key.u32(desc.IBStripCutValue).u32(std::bit_cast<std::uint32_t>(desc.PrimitiveTopologyType));
    }
    key.u32(desc.NumRenderTargets);
    for (const DXGI_FORMAT format : desc.RTVFormats) key.u32(format);
    key.u32(desc.DSVFormat).u32(desc.SampleDesc.Count).u32(desc.SampleDesc.Quality).u32(desc.NodeMask);
}

// The identity of what create_graphics is given: the description, an MS or AS stage, the back face's stencil masks
// and the depth bounds test of a pipeline state stream. 64 lowercase hex digits (SHA-256).
inline std::string graphics_identity(const D3D12_GRAPHICS_PIPELINE_STATE_DESC& desc,
                                     const D3D12_SHADER_BYTECODE* ms = nullptr,
                                     const D3D12_SHADER_BYTECODE* as = nullptr,
                                     const StencilFaceMasks* back_masks = nullptr, bool depth_bounds = false) {
    const bool mesh = ms && ms->pShaderBytecode && ms->BytecodeLength;
    leonida::cached_pso::KeyBuilder key;
    key.tag("graphics identity 1");
    key_graphics_state(key, desc, !mesh);
    key_shader(key, "ms", ms ? *ms : D3D12_SHADER_BYTECODE{});
    key_shader(key, "as", as ? *as : D3D12_SHADER_BYTECODE{});
    key.u32(back_masks ? 1u : 0u).u32(back_masks ? back_masks->read : 0u).u32(back_masks ? back_masks->write : 0u);
    key.u32(depth_bounds ? 1u : 0u);
    return leonida::sha256_hex(key.data());
}

}  // namespace rt::d3d12
