// Tests console allocation sizes and placement for render targets and depth-stencil targets
// (runtime/shims/d3d12_xs_alloc_size.h).
// Synthetic descriptions only; no game data or GPU required.
#include <windows.h>
#include <d3d12.h>

#include <cstdint>
#include <cstdio>
#include <vector>

#include "d3d12_xs_alloc_size.h"

namespace {

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const char* what) {
    ++g_checks;
    if (!condition) {
        ++g_failures;
        std::printf("FAIL %s\n", what);
    }
}

void test_align_up_saturating() {
    using rt::d3d12_xs::align_up_saturating;

    check(align_up_saturating(0, 0) == 0, "align_up_saturating(0, 0) == 0");
    check(align_up_saturating(1234, 0) == 1234, "align_up_saturating with to=0 returns value");
    check(align_up_saturating(0, 64) == 0, "align_up_saturating(0, 64) == 0");
    check(align_up_saturating(1, 64) == 64, "align_up_saturating(1, 64) == 64");
    check(align_up_saturating(64, 64) == 64, "align_up_saturating(64, 64) == 64");
    check(align_up_saturating(65, 64) == 128, "align_up_saturating(65, 64) == 128");
    check(align_up_saturating(100, 65536) == 65536, "align_up_saturating(100, 64K) == 64K");
    check(align_up_saturating(65536, 65536) == 65536, "align_up_saturating(64K, 64K) == 64K");
    check(align_up_saturating(65537, 65536) == 131072, "align_up_saturating(64K+1, 64K) == 128K");

    // Saturation behavior near UINT64_MAX
    const UINT64 saturated = align_up_saturating(UINT64_MAX - 10, 64);
    check(saturated == (UINT64_MAX / 64 * 64), "align_up_saturating saturates without wrapping");
    check(align_up_saturating(UINT64_MAX, 65536) == (UINT64_MAX / 65536 * 65536), "align_up_saturating UINT64_MAX saturates");
}

void test_render_target_element_bytes() {
    using rt::d3d12_xs::render_target_element_bytes;

    // 1-byte formats
    check(render_target_element_bytes(DXGI_FORMAT_R8_UNORM) == 1, "R8_UNORM is 1 byte");
    check(render_target_element_bytes(DXGI_FORMAT_R8_UINT) == 1, "R8_UINT is 1 byte");
    check(render_target_element_bytes(DXGI_FORMAT_R8_SNORM) == 1, "R8_SNORM is 1 byte");
    check(render_target_element_bytes(DXGI_FORMAT_R8_SINT) == 1, "R8_SINT is 1 byte");
    check(render_target_element_bytes(DXGI_FORMAT_A8_UNORM) == 1, "A8_UNORM is 1 byte");

    // 2-byte formats
    check(render_target_element_bytes(DXGI_FORMAT_R8G8_UNORM) == 2, "R8G8_UNORM is 2 bytes");
    check(render_target_element_bytes(DXGI_FORMAT_R16_FLOAT) == 2, "R16_FLOAT is 2 bytes");
    check(render_target_element_bytes(DXGI_FORMAT_R16_UNORM) == 2, "R16_UNORM is 2 bytes");
    check(render_target_element_bytes(DXGI_FORMAT_B5G6R5_UNORM) == 2, "B5G6R5_UNORM is 2 bytes");

    // 4-byte formats
    check(render_target_element_bytes(DXGI_FORMAT_R8G8B8A8_UNORM) == 4, "R8G8B8A8_UNORM is 4 bytes");
    check(render_target_element_bytes(DXGI_FORMAT_R8G8B8A8_UNORM_SRGB) == 4, "R8G8B8A8_UNORM_SRGB is 4 bytes");
    check(render_target_element_bytes(DXGI_FORMAT_B8G8R8A8_UNORM) == 4, "B8G8R8A8_UNORM is 4 bytes");
    check(render_target_element_bytes(DXGI_FORMAT_R10G10B10A2_UNORM) == 4, "R10G10B10A2_UNORM is 4 bytes");
    check(render_target_element_bytes(DXGI_FORMAT_R11G11B10_FLOAT) == 4, "R11G11B10_FLOAT is 4 bytes");
    check(render_target_element_bytes(DXGI_FORMAT_R16G16_FLOAT) == 4, "R16G16_FLOAT is 4 bytes");
    check(render_target_element_bytes(DXGI_FORMAT_R32_FLOAT) == 4, "R32_FLOAT is 4 bytes");

    // 8-byte formats
    check(render_target_element_bytes(DXGI_FORMAT_R16G16B16A16_FLOAT) == 8, "R16G16B16A16_FLOAT is 8 bytes");
    check(render_target_element_bytes(DXGI_FORMAT_R32G32_FLOAT) == 8, "R32G32_FLOAT is 8 bytes");

    // 16-byte formats
    check(render_target_element_bytes(DXGI_FORMAT_R32G32B32A32_FLOAT) == 16, "R32G32B32A32_FLOAT is 16 bytes");

    // Unsupported / non-render target formats
    check(render_target_element_bytes(DXGI_FORMAT_UNKNOWN) == 0, "UNKNOWN format is 0 bytes");
    check(render_target_element_bytes(DXGI_FORMAT_BC1_UNORM) == 0, "BC1_UNORM is 0 bytes");
    check(render_target_element_bytes(DXGI_FORMAT_D32_FLOAT) == 0, "D32_FLOAT is 0 bytes for RT");
}

void test_console_depth_planes() {
    using rt::d3d12_xs::console_depth_planes;

    auto p = console_depth_planes(DXGI_FORMAT_D32_FLOAT_S8X24_UINT);
    check(p.depth == 4 && p.stencil == 1, "D32_FLOAT_S8X24_UINT is 4+1 planes");

    p = console_depth_planes(DXGI_FORMAT_D24_UNORM_S8_UINT);
    check(p.depth == 4 && p.stencil == 1, "D24_UNORM_S8_UINT is 4+1 planes");

    p = console_depth_planes(static_cast<DXGI_FORMAT>(118));
    check(p.depth == 2 && p.stencil == 1, "DXGI 118 (console D16S8) is 2+1 planes");

    p = console_depth_planes(DXGI_FORMAT_D32_FLOAT);
    check(p.depth == 4 && p.stencil == 0, "D32_FLOAT is 4+0 planes");

    p = console_depth_planes(DXGI_FORMAT_D16_UNORM);
    check(p.depth == 2 && p.stencil == 0, "D16_UNORM is 2+0 planes");

    p = console_depth_planes(DXGI_FORMAT_R8G8B8A8_UNORM);
    check(p.depth == 0 && p.stencil == 0, "R8G8B8A8_UNORM has no depth planes");
}

void test_console_render_target_size() {
    using rt::d3d12_xs::console_render_target_size;

    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Alignment = 0;
    desc.Width = 1920;
    desc.Height = 1080;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.SampleDesc.Count = 1;
    desc.SampleDesc.Quality = 0;
    desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;

    // 1920x1080 R8G8B8A8 (4 bytes/texel):
    // Tile size is 64 KiB. For 4 bytes/texel, tile is 128x128.
    // 1920 / 128 = 15 tiles wide.
    // ceil(1080 / 128) = 9 tiles high.
    // 15 * 9 = 135 tiles = 135 * 65536 = 8,847,360 bytes.
    const UINT64 rt_size = console_render_target_size(desc);
    check(rt_size == 135 * 65536, "1920x1080 R8G8B8A8 render target is 135 64K tiles");

    // Two slices
    desc.DepthOrArraySize = 2;
    check(console_render_target_size(desc) == 2 * 135 * 65536, "2 slices double the allocation");
    desc.DepthOrArraySize = 1;

    // Multisampling (Count > 1) returns 0
    desc.SampleDesc.Count = 4;
    check(console_render_target_size(desc) == 0, "Multisampled target returns 0");
    desc.SampleDesc.Count = 1;

    // Missing ALLOW_RENDER_TARGET flag returns 0
    desc.Flags = D3D12_RESOURCE_FLAG_NONE;
    check(console_render_target_size(desc) == 0, "Missing ALLOW_RENDER_TARGET returns 0");
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;

    // Non-2D dimension returns 0
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE3D;
    check(console_render_target_size(desc) == 0, "3D texture returns 0");
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    check(console_render_target_size(desc) == 0, "Buffer returns 0");
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;

    // Unsupported format returns 0
    desc.Format = DXGI_FORMAT_BC1_UNORM;
    check(console_render_target_size(desc) == 0, "BC1 format returns 0");
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;

    // Zero dimensions return 0
    desc.Width = 0;
    check(console_render_target_size(desc) == 0, "Zero width returns 0");
    desc.Width = 1920;
    desc.Height = 0;
    check(console_render_target_size(desc) == 0, "Zero height returns 0");
    desc.Height = 1080;
    desc.DepthOrArraySize = 0;
    check(console_render_target_size(desc) == 0, "Zero depth returns 0");
    desc.DepthOrArraySize = 1;

    // Full mip chain: non-zero and >= single mip size
    desc.MipLevels = 0; // full chain
    const UINT64 full_chain_size = console_render_target_size(desc);
    check(full_chain_size > rt_size, "Full mip chain is larger than single mip");
}

void test_console_depth_stencil_size() {
    using rt::d3d12_xs::console_depth_stencil_size;

    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Alignment = 0;
    desc.Width = 1920;
    desc.Height = 1080;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_D32_FLOAT_S8X24_UINT;
    desc.SampleDesc.Count = 1;
    desc.SampleDesc.Quality = 0;
    desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;

    const UINT64 ds_size = console_depth_stencil_size(desc);
    check(ds_size > 0, "1920x1080 D32S8 depth-stencil target has non-zero size");

    // Missing ALLOW_DEPTH_STENCIL returns 0
    desc.Flags = D3D12_RESOURCE_FLAG_NONE;
    check(console_depth_stencil_size(desc) == 0, "Missing ALLOW_DEPTH_STENCIL returns 0");
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;

    // Count > 1 returns 0
    desc.SampleDesc.Count = 2;
    check(console_depth_stencil_size(desc) == 0, "Multisampled depth returns 0");
    desc.SampleDesc.Count = 1;

    // Non-depth format returns 0
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    check(console_depth_stencil_size(desc) == 0, "Colour format returns 0 for depth size");
}

void test_console_placement() {
    using rt::d3d12_xs::console_placement;

    // Null descs returns false
    D3D12_RESOURCE_ALLOCATION_INFO total{};
    check(!console_placement(nullptr, 1, [](UINT) -> D3D12_RESOURCE_ALLOCATION_INFO { return {0, 0}; }, total, nullptr),
          "console_placement rejects null descs");

    // Non-target list returns false (PC answer stands)
    D3D12_RESOURCE_DESC buffer_desc{};
    buffer_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    buffer_desc.Width = 1024;
    buffer_desc.Height = 1;
    buffer_desc.DepthOrArraySize = 1;
    buffer_desc.MipLevels = 1;
    buffer_desc.SampleDesc.Count = 1;
    check(!console_placement(&buffer_desc, 1, [](UINT) -> D3D12_RESOURCE_ALLOCATION_INFO { return {1024, 65536}; }, total, nullptr),
          "console_placement returns false for buffers (no target)");

    // Growing render target
    D3D12_RESOURCE_DESC rt_desc{};
    rt_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    rt_desc.Width = 1920;
    rt_desc.Height = 1080;
    rt_desc.DepthOrArraySize = 1;
    rt_desc.MipLevels = 1;
    rt_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    rt_desc.SampleDesc.Count = 1;
    rt_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;

    // Suppose PC device reported 1920*1080*4 = 8,294,400 bytes, aligned to 64 KiB
    // Console size is 135 * 65536 = 8,847,360 bytes (which grows).
    D3D12_RESOURCE_ALLOCATION_INFO1 each[1]{};
    bool placed = console_placement(&rt_desc, 1,
                                    [](UINT) -> D3D12_RESOURCE_ALLOCATION_INFO {
                                        return {8294400, 65536};
                                    },
                                    total, each);
    check(placed, "console_placement accepts growing render target");
    check(total.SizeInBytes == 8847360, "total size matches console tile-padded size");
    check(total.Alignment == 65536, "total alignment matches 64 KiB");
    check(each[0].Offset == 0, "each[0] offset is 0");
    check(each[0].SizeInBytes == 8847360, "each[0] size matches console tile-padded size");
    check(each[0].Alignment == 65536, "each[0] alignment matches 64 KiB");
}

}  // namespace

int main() {
    test_align_up_saturating();
    test_render_target_element_bytes();
    test_console_depth_planes();
    test_console_render_target_size();
    test_console_depth_stencil_size();
    test_console_placement();

    std::printf("%d/%d checks passed\n", g_checks - g_failures, g_checks);
    return g_failures ? 1 : 0;
}
