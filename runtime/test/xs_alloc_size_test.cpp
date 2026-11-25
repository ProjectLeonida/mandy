// CPU-only unit tests for runtime/shims/d3d12_xs_alloc_size.h (no GPU or game data).
#include "d3d12_xs_alloc_size.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

int g_failures = 0;

void check(bool cond, const char* what) {
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_failures;
    }
}

D3D12_RESOURCE_DESC make_tex2d(UINT64 w, UINT h, DXGI_FORMAT fmt, D3D12_RESOURCE_FLAGS flags, UINT16 mips = 1,
                               UINT16 slices = 1, UINT samples = 1) {
    D3D12_RESOURCE_DESC d{};
    d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    d.Width = w;
    d.Height = h;
    d.DepthOrArraySize = slices;
    d.MipLevels = mips;
    d.Format = fmt;
    d.SampleDesc.Count = samples;
    d.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    d.Flags = flags;
    return d;
}

void test_align_up_saturating() {
    using rt::d3d12_xs::align_up_saturating;
    check(align_up_saturating(123, 0) == 123, "align_up_saturating(to=0) returns value without div-by-zero");
    check(align_up_saturating(123, 1) == 123, "align_up_saturating(to=1) returns value");
    check(align_up_saturating(0, 65536) == 0, "align_up_saturating(0, 64KiB) == 0");
    check(align_up_saturating(1, 65536) == 65536, "align_up_saturating(1, 64KiB) == 64KiB");
    check(align_up_saturating(UINT64_MAX, 65536) == (UINT64_MAX / 65536) * 65536,
          "align_up_saturating saturates near UINT64_MAX");
    check(align_up_saturating(UINT64_MAX, 0) == UINT64_MAX, "align_up_saturating(UINT64_MAX, 0) returns UINT64_MAX");
}

void test_overflow_guards() {
    using rt::d3d12_xs::console_depth_stencil_size;
    using rt::d3d12_xs::console_render_target_size;

    // Width > UINT32_MAX (including UINT64_MAX) must return 0 cleanly without shift/addition wrap.
    const auto huge_w = make_tex2d(UINT64_MAX, 64, DXGI_FORMAT_R8G8B8A8_UNORM,
                                   D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, 0);
    check(console_render_target_size(huge_w) == 0, "render target width == UINT64_MAX returns 0");

    const auto huge_ds = make_tex2d(UINT64_MAX, 64, DXGI_FORMAT_D32_FLOAT_S8X24_UINT,
                                    D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL, 0);
    check(console_depth_stencil_size(huge_ds) == 0, "depth-stencil width == UINT64_MAX returns 0");

    // Multi-mip render target where mip 0 fits in uint64 (~0xFFFFFFFE00010000) and mip 0 + mip 1 overflows uint64.
    const auto rt_mip_ovf = make_tex2d(0x3FFFFFBFu, 0x3FFFFFBFu, DXGI_FORMAT_R32G32B32A32_FLOAT,
                                       D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, 2);
    check(console_render_target_size(rt_mip_ovf) == 0, "render target multi-mip slice accumulation overflow returns 0");

    // Multi-plane depth-stencil where depth fits in uint64 (~0xFFFFFFFE00010000) and depth + stencil overflows uint64.
    const auto ds_plane_ovf = make_tex2d(0x7FFFFF7Fu, 0x7FFFFF7Fu, DXGI_FORMAT_D32_FLOAT_S8X24_UINT,
                                         D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL, 1);
    check(console_depth_stencil_size(ds_plane_ovf) == 0,
          "depth-stencil multi-plane slice accumulation overflow returns 0");

    // Normal 1920x1080 RGBA8 render target: 15x9 tiles of 128x128 (64 KiB each) = 135 * 64 KiB.
    const auto rt_1080 = make_tex2d(1920, 1080, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, 1);
    check(console_render_target_size(rt_1080) == 135ull * 65536ull, "1920x1080 RGBA8 RT is 135 SW_64KB_D tiles");
}

void test_console_placement_single_pass_and_fail_closed() {
    using rt::d3d12_xs::console_placement;

    // 1. Null / empty list returns false and never calls pc_one.
    UINT calls = 0;
    D3D12_RESOURCE_ALLOCATION_INFO total{999, 128};
    check(!console_placement(nullptr, 1, [&](UINT) { ++calls; return D3D12_RESOURCE_ALLOCATION_INFO{65536, 65536}; },
                             total, nullptr),
          "null descs returns false");
    check(calls == 0 && total.SizeInBytes == 999, "null descs leaves total untouched and skips pc_one");

    D3D12_RESOURCE_DESC buf{};
    buf.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    buf.Width = 4096;
    buf.Height = 1;
    buf.DepthOrArraySize = 1;
    buf.MipLevels = 1;
    buf.SampleDesc.Count = 1;
    check(!console_placement(&buf, 0, [&](UINT) { ++calls; return D3D12_RESOURCE_ALLOCATION_INFO{4096, 65536}; },
                             total, nullptr),
          "count == 0 returns false");
    check(!console_placement(&buf, 1, [&](UINT) { ++calls; return D3D12_RESOURCE_ALLOCATION_INFO{4096, 65536}; },
                             total, nullptr),
          "buffer-only list returns false without calling pc_one");
    check(calls == 0, "no-target list makes 0 pc_one calls");

    // 2. Single-pass pc_one invocation when `each` is requested, and exact layout calculation.
    std::array<D3D12_RESOURCE_DESC, 3> descs = {
        make_tex2d(1920, 1080, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, 1),
        make_tex2d(512, 512, DXGI_FORMAT_BC7_UNORM, D3D12_RESOURCE_FLAG_NONE, 4),
        make_tex2d(1920, 1080, DXGI_FORMAT_D32_FLOAT_S8X24_UINT, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL, 1),
    };
    std::array<D3D12_RESOURCE_ALLOCATION_INFO1, 3> each{};
    calls = 0;
    const bool ok = console_placement(
        descs.data(), static_cast<UINT>(descs.size()),
        [&](UINT i) -> D3D12_RESOURCE_ALLOCATION_INFO {
            ++calls;
            if (i == 0) return {1920ull * 1080ull * 4ull, 65536};
            if (i == 1) return {262144, 65536};
            return {1920ull * 1080ull * 8ull, 65536};
        },
        total, each.data());
    check(ok, "mixed RT + CPU texture + DS placement succeeds");
    check(calls == descs.size(), "pc_one called exactly once per descriptor even when `each` is non-null");
    check(each[0].Offset == 0 && each[0].SizeInBytes == 135ull * 65536ull, "entry 0 uses console RT tile size");
    check(each[1].Offset >= each[0].SizeInBytes && each[1].SizeInBytes > 0, "entry 1 placed after entry 0");
    check(each[2].Offset >= each[1].Offset + each[1].SizeInBytes && each[2].SizeInBytes > 0,
          "entry 2 placed after entry 1");
    check(total.SizeInBytes >= each[2].Offset + each[2].SizeInBytes, "total covers all entries");

    // 3. Failure in pc_one mid-list leaves `total` and `each` completely untouched.
    const D3D12_RESOURCE_ALLOCATION_INFO sentinel_total{0x1234, 0x5678};
    const D3D12_RESOURCE_ALLOCATION_INFO1 sentinel_each{0xAA, 0xBB, 0xCC};
    total = sentinel_total;
    each.fill(sentinel_each);
    calls = 0;
    const bool failed = console_placement(
        descs.data(), static_cast<UINT>(descs.size()),
        [&](UINT i) -> D3D12_RESOURCE_ALLOCATION_INFO {
            ++calls;
            if (i == 1) return {UINT64_MAX, 65536};
            return {65536, 65536};
        },
        total, each.data());
    check(!failed && calls == 2, "pc_one failure stops immediately and returns false");
    check(total.SizeInBytes == sentinel_total.SizeInBytes && total.Alignment == sentinel_total.Alignment,
          "pc_one failure leaves total untouched");
    check(each[0].Offset == sentinel_each.Offset && each[0].SizeInBytes == sentinel_each.SizeInBytes,
          "pc_one failure leaves each untouched");

    // 4. Heap-spill path (count > 16) works identically.
    std::vector<D3D12_RESOURCE_DESC> many(24, descs[0]);
    std::vector<D3D12_RESOURCE_ALLOCATION_INFO1> many_each(24);
    calls = 0;
    const bool many_ok = console_placement(
        many.data(), static_cast<UINT>(many.size()),
        [&](UINT) -> D3D12_RESOURCE_ALLOCATION_INFO {
            ++calls;
            return {1920ull * 1080ull * 4ull, 65536};
        },
        total, many_each.data());
    check(many_ok && calls == 24, "count > 16 heap-spill path calls pc_one once per entry");
    check(total.SizeInBytes == 24ull * 135ull * 65536ull && many_each[23].Offset == 23ull * 135ull * 65536ull,
          "count > 16 layout matches exact tile totals");
}

}  // namespace

int main() {
    test_align_up_saturating();
    test_overflow_guards();
    test_console_placement_single_pass_and_fail_closed();
    std::printf("xs_alloc_size_test: %s (%d failure%s)\n", g_failures ? "FAILED" : "OK", g_failures,
                g_failures == 1 ? "" : "s");
    return g_failures ? 1 : 0;
}
