// Unit tests for D3D12 Xbox resource description cleaning (runtime/shims/d3d12_xs_desc_clean.h)
// Verifies PC flag masking, console layout normalization, depth format classification,
// depth-stencil UAV refusal, and format-specific storage stripping.
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>

#include <cstdint>
#include <cstdio>

#include "../shims/d3d12_xs_desc_clean.h"

namespace {

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const char* message) {
    ++g_checks;
    if (!condition) {
        ++g_failures;
        std::printf("FAIL: %s\n", message);
    } else {
        std::printf("PASS: %s\n", message);
    }
}

D3D12_RESOURCE_DESC make_base_desc(D3D12_RESOURCE_DIMENSION dim, DXGI_FORMAT format) {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = dim;
    desc.Alignment = 65536;
    desc.Width = 1024;
    desc.Height = 512;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.SampleDesc.Quality = 0;
    desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    desc.Flags = D3D12_RESOURCE_FLAG_NONE;
    return desc;
}

} // namespace

int main() {
    using namespace rt::d3d12_xs;

    // 1. PC Resource Flags Mask verification
    {
        check((kPcResourceFlagsMask & D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET) != 0, "mask contains ALLOW_RENDER_TARGET");
        check((kPcResourceFlagsMask & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL) != 0, "mask contains ALLOW_DEPTH_STENCIL");
        check((kPcResourceFlagsMask & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) != 0, "mask contains ALLOW_UNORDERED_ACCESS");
        check((kPcResourceFlagsMask & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE) != 0, "mask contains DENY_SHADER_RESOURCE");
        check((kPcResourceFlagsMask & D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER) != 0, "mask contains ALLOW_CROSS_ADAPTER");
        check((kPcResourceFlagsMask & D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS) != 0, "mask contains ALLOW_SIMULTANEOUS_ACCESS");
        check((kPcResourceFlagsMask & D3D12_RESOURCE_FLAG_VIDEO_DECODE_REFERENCE_ONLY) != 0, "mask contains VIDEO_DECODE_REFERENCE_ONLY");
        check((kPcResourceFlagsMask & D3D12_RESOURCE_FLAG_VIDEO_ENCODE_REFERENCE_ONLY) != 0, "mask contains VIDEO_ENCODE_REFERENCE_ONLY");
        check((kPcResourceFlagsMask & D3D12_RESOURCE_FLAG_RAYTRACING_ACCELERATION_STRUCTURE) != 0, "mask contains RAYTRACING_ACCELERATION_STRUCTURE");

        // High console bits masked out
        const std::uint32_t console_bits = 0x80000000u | 0x00010000u | 0x00008000u | 0x00000200u;
        check((console_bits & kPcResourceFlagsMask) == 0, "high console-specific bits excluded from mask");
    }

    // 2. is_xbox_resource_layout classification
    {
        check(!is_xbox_resource_layout(D3D12_TEXTURE_LAYOUT_UNKNOWN), "UNKNOWN is not xbox layout");
        check(!is_xbox_resource_layout(D3D12_TEXTURE_LAYOUT_ROW_MAJOR), "ROW_MAJOR is not xbox layout");
        check(!is_xbox_resource_layout(D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE), "64KB_UNDEFINED_SWIZZLE is not xbox layout");
        check(!is_xbox_resource_layout(D3D12_TEXTURE_LAYOUT_64KB_STANDARD_SWIZZLE), "64KB_STANDARD_SWIZZLE is not xbox layout");

        check(is_xbox_resource_layout(kXboxTiledResourceLayout), "0x101 is xbox tiled layout");
        check(is_xbox_resource_layout(static_cast<D3D12_TEXTURE_LAYOUT>(0x102)), "0x102 is xbox layout");
        check(is_xbox_resource_layout(static_cast<D3D12_TEXTURE_LAYOUT>(0x105)), "0x105 is xbox layout");
        check(is_xbox_resource_layout(static_cast<D3D12_TEXTURE_LAYOUT>(0x200)), "0x200 is xbox layout");
    }

    // 3. is_depth_resource_format classification
    {
        check(is_depth_resource_format(DXGI_FORMAT_D16_UNORM), "D16_UNORM is depth");
        check(is_depth_resource_format(DXGI_FORMAT_D24_UNORM_S8_UINT), "D24_UNORM_S8_UINT is depth");
        check(is_depth_resource_format(DXGI_FORMAT_D32_FLOAT), "D32_FLOAT is depth");
        check(is_depth_resource_format(DXGI_FORMAT_D32_FLOAT_S8X24_UINT), "D32_FLOAT_S8X24_UINT is depth");
        check(is_depth_resource_format(DXGI_FORMAT_R16_TYPELESS), "R16_TYPELESS is depth");
        check(is_depth_resource_format(DXGI_FORMAT_R24G8_TYPELESS), "R24G8_TYPELESS is depth");
        check(is_depth_resource_format(DXGI_FORMAT_R32_TYPELESS), "R32_TYPELESS is depth");
        check(is_depth_resource_format(DXGI_FORMAT_R32G8X24_TYPELESS), "R32G8X24_TYPELESS is depth");
        check(is_depth_resource_format(static_cast<DXGI_FORMAT>(118)), "format 118 (D16_UNORM_S8_UINT) is depth");

        check(!is_depth_resource_format(DXGI_FORMAT_UNKNOWN), "UNKNOWN is not depth");
        check(!is_depth_resource_format(DXGI_FORMAT_R8G8B8A8_UNORM), "R8G8B8A8_UNORM is not depth");
        check(!is_depth_resource_format(DXGI_FORMAT_R32_FLOAT), "R32_FLOAT is not depth");
        check(!is_depth_resource_format(DXGI_FORMAT_R16_UNORM), "R16_UNORM is not depth");
        check(!is_depth_resource_format(DXGI_FORMAT_B8G8R8A8_UNORM), "B8G8R8A8_UNORM is not depth");
        check(!is_depth_resource_format(DXGI_FORMAT_B5G5R5A1_UNORM), "B5G5R5A1_UNORM is not depth");
        check(!is_depth_resource_format(DXGI_FORMAT_BC1_UNORM), "BC1_UNORM is not depth");
        check(!is_depth_resource_format(DXGI_FORMAT_R32_UINT), "R32_UINT is not depth");
    }

    // 4. without_depth_stencil_uav behavior
    {
        const std::uint32_t ds_uav = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

        // Texture dimension strips UAV
        std::uint32_t stripped_2d = without_depth_stencil_uav(D3D12_RESOURCE_DIMENSION_TEXTURE2D, ds_uav);
        check((stripped_2d & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) == 0, "TEXTURE2D with DS strips UAV");
        check((stripped_2d & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL) != 0, "TEXTURE2D retains ALLOW_DEPTH_STENCIL");

        std::uint32_t stripped_1d = without_depth_stencil_uav(D3D12_RESOURCE_DIMENSION_TEXTURE1D, ds_uav);
        check((stripped_1d & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) == 0, "TEXTURE1D with DS strips UAV");

        std::uint32_t stripped_3d = without_depth_stencil_uav(D3D12_RESOURCE_DIMENSION_TEXTURE3D, ds_uav);
        check((stripped_3d & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) == 0, "TEXTURE3D with DS strips UAV");

        // Buffer dimension preserves UAV
        std::uint32_t buf_flags = without_depth_stencil_uav(D3D12_RESOURCE_DIMENSION_BUFFER, ds_uav);
        check((buf_flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) != 0, "BUFFER dimension preserves UAV");

        // Texture without ALLOW_DEPTH_STENCIL preserves UAV
        const std::uint32_t rt_uav = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        std::uint32_t preserved_2d = without_depth_stencil_uav(D3D12_RESOURCE_DIMENSION_TEXTURE2D, rt_uav);
        check((preserved_2d & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) != 0, "TEXTURE2D without DS preserves UAV");
    }

    // 5. pc_resource_desc layout normalization
    {
        // Console tiled layout on texture normalized to UNKNOWN
        D3D12_RESOURCE_DESC tex_xbox = make_base_desc(D3D12_RESOURCE_DIMENSION_TEXTURE2D, DXGI_FORMAT_R8G8B8A8_UNORM);
        tex_xbox.Layout = kXboxTiledResourceLayout;
        D3D12_RESOURCE_DESC tex_pc = pc_resource_desc(tex_xbox);
        check(tex_pc.Layout == D3D12_TEXTURE_LAYOUT_UNKNOWN, "texture console 0x101 layout normalized to UNKNOWN");

        tex_xbox.Layout = static_cast<D3D12_TEXTURE_LAYOUT>(0x102);
        check(pc_resource_desc(tex_xbox).Layout == D3D12_TEXTURE_LAYOUT_UNKNOWN, "texture console 0x102 layout normalized to UNKNOWN");

        tex_xbox.Layout = static_cast<D3D12_TEXTURE_LAYOUT>(0x105);
        check(pc_resource_desc(tex_xbox).Layout == D3D12_TEXTURE_LAYOUT_UNKNOWN, "texture console 0x105 layout normalized to UNKNOWN");

        // Console layout on buffer normalized to ROW_MAJOR
        D3D12_RESOURCE_DESC buf_xbox = make_base_desc(D3D12_RESOURCE_DIMENSION_BUFFER, DXGI_FORMAT_UNKNOWN);
        buf_xbox.Layout = kXboxTiledResourceLayout;
        D3D12_RESOURCE_DESC buf_pc = pc_resource_desc(buf_xbox);
        check(buf_pc.Layout == D3D12_TEXTURE_LAYOUT_ROW_MAJOR, "buffer console 0x101 layout normalized to ROW_MAJOR");

        buf_xbox.Layout = static_cast<D3D12_TEXTURE_LAYOUT>(0x102);
        check(pc_resource_desc(buf_xbox).Layout == D3D12_TEXTURE_LAYOUT_ROW_MAJOR, "buffer console 0x102 layout normalized to ROW_MAJOR");

        // PC standard layouts preserved
        tex_xbox.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
        check(pc_resource_desc(tex_xbox).Layout == D3D12_TEXTURE_LAYOUT_UNKNOWN, "PC UNKNOWN layout preserved");

        tex_xbox.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        check(pc_resource_desc(tex_xbox).Layout == D3D12_TEXTURE_LAYOUT_ROW_MAJOR, "PC ROW_MAJOR layout preserved");

        tex_xbox.Layout = D3D12_TEXTURE_LAYOUT_64KB_STANDARD_SWIZZLE;
        check(pc_resource_desc(tex_xbox).Layout == D3D12_TEXTURE_LAYOUT_64KB_STANDARD_SWIZZLE, "PC 64KB_STANDARD_SWIZZLE preserved");
    }

    // 6. pc_resource_desc flag cleaning and UAV rules
    {
        // Console high flag bits stripped
        D3D12_RESOURCE_DESC xbox = make_base_desc(D3D12_RESOURCE_DIMENSION_TEXTURE2D, DXGI_FORMAT_R8G8B8A8_UNORM);
        xbox.Flags = static_cast<D3D12_RESOURCE_FLAGS>(
            D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS | 0x80000000u | 0x00010000u);
        D3D12_RESOURCE_DESC pc = pc_resource_desc(xbox);
        check(pc.Flags == (D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
              "console high flag bits stripped by pc_resource_desc");

        // B5G5R5A1_UNORM strips UAV
        D3D12_RESOURCE_DESC b5g5_xbox = make_base_desc(D3D12_RESOURCE_DIMENSION_TEXTURE2D, DXGI_FORMAT_B5G5R5A1_UNORM);
        b5g5_xbox.Flags = static_cast<D3D12_RESOURCE_FLAGS>(D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        D3D12_RESOURCE_DESC b5g5_pc = pc_resource_desc(b5g5_xbox);
        check((b5g5_pc.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) == 0, "B5G5R5A1 strips ALLOW_UNORDERED_ACCESS");
        check((b5g5_pc.Flags & D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET) != 0, "B5G5R5A1 retains ALLOW_RENDER_TARGET");

        // Depth-stencil format with ALLOW_DEPTH_STENCIL strips UAV
        D3D12_RESOURCE_DESC ds_xbox = make_base_desc(D3D12_RESOURCE_DIMENSION_TEXTURE2D, DXGI_FORMAT_R32G8X24_TYPELESS);
        ds_xbox.Flags = static_cast<D3D12_RESOURCE_FLAGS>(D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        D3D12_RESOURCE_DESC ds_pc = pc_resource_desc(ds_xbox);
        check((ds_pc.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) == 0, "depth texture with DS strips ALLOW_UNORDERED_ACCESS");
        check((ds_pc.Flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL) != 0, "depth texture retains ALLOW_DEPTH_STENCIL");

        // Geometry and dimension preservation
        check(ds_pc.Dimension == ds_xbox.Dimension, "Dimension preserved");
        check(ds_pc.Alignment == ds_xbox.Alignment, "Alignment preserved");
        check(ds_pc.Width == ds_xbox.Width, "Width preserved");
        check(ds_pc.Height == ds_xbox.Height, "Height preserved");
        check(ds_pc.DepthOrArraySize == ds_xbox.DepthOrArraySize, "DepthOrArraySize preserved");
        check(ds_pc.MipLevels == ds_xbox.MipLevels, "MipLevels preserved");
        check(ds_pc.Format == ds_xbox.Format, "Format preserved");
        check(ds_pc.SampleDesc.Count == ds_xbox.SampleDesc.Count, "SampleDesc.Count preserved");
        check(ds_pc.SampleDesc.Quality == ds_xbox.SampleDesc.Quality, "SampleDesc.Quality preserved");
    }

    std::printf("\nSummary: %d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
