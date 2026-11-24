// CPU benchmark of the GFX10 format-table lookups (src/gfx10_descriptors.h): format_row, vulkan_format,
// format_from_vulkan, format_swizzle and vulkan_view_format, which run for every view a descriptor names
// (CreateShaderResourceView / CreateUnorderedAccessView encode a T#, and binding a T# into set 0 picks its view
// format and component mapping under the descriptor lock). It times the indexed lookups against the linear scans of
// the table they replaced, kept here as the reference, on the sequence of calls one SRV makes, over a mix of the
// formats a frame's textures use. No GPU, driver or game files.
//
// Usage:
//   gfx10_format_bench [--iters N] [--reps R] [--self-test]
// --self-test checks that both give the same answers for every format and code, quickly, and prints no timings.
#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include "../src/gfx10_descriptors.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace g = leonida::gfx10;

namespace reference {  // the lookups as they were: a scan of format_table each

const g::FormatRow* format_row(VkFormat format) {
    for (const g::FormatRow& row : g::format_table)
        if (row.vulkan == format) return &row;
    return nullptr;
}
VkFormat vulkan_format(std::uint32_t code) {
    for (const g::FormatRow& row : g::format_table)
        if (row.gfx10 == code) return row.vulkan;
    return VK_FORMAT_UNDEFINED;
}
std::uint32_t format_from_vulkan(VkFormat format) {
    const g::FormatRow* row = format_row(format);
    return row ? row->gfx10 : 0;
}
g::Swizzle format_swizzle(VkFormat format) {
    const g::FormatRow* row = format_row(format);
    return row ? row->channels : g::identity_swizzle;
}
VkFormat vulkan_view_format(VkFormat image_format, std::uint32_t code, bool allow_channel_cast) {
    const g::FormatRow* image = format_row(image_format);
    if (!image) return VK_FORMAT_UNDEFINED;
    if (image->gfx10 == code) return image_format;
    if (g::is_depth_stencil_format(image_format)) return VK_FORMAT_UNDEFINED;
    const g::FormatRow* fallback = nullptr;
    for (const g::FormatRow& row : g::format_table) {
        if (row.gfx10 != code || g::is_depth_stencil_format(row.vulkan)) continue;
        if (row.channels == image->channels) return row.vulkan;
        if (!fallback) fallback = &row;
    }
    return allow_channel_cast && fallback ? fallback->vulkan : VK_FORMAT_UNDEFINED;
}
VkComponentMapping vulkan_swizzle(const g::Swizzle& swizzle, VkFormat view_format) {
    const g::Swizzle channels = format_swizzle(view_format);
    const auto component = [&](std::uint8_t select) {
        if (select == g::select_one) return VK_COMPONENT_SWIZZLE_ONE;
        if (select >= g::select_x && select <= g::select_w)
            for (unsigned i = 0; i < 4; ++i)
                if (channels[i] == select) return VkComponentSwizzle(VK_COMPONENT_SWIZZLE_R + i);
        return VK_COMPONENT_SWIZZLE_ZERO;
    };
    return {component(swizzle[0]), component(swizzle[1]), component(swizzle[2]), component(swizzle[3])};
}

} // namespace reference

namespace {

// Formats probed: every one in the table, the ones around them, and a few that are not there.
std::vector<VkFormat> probe_formats() {
    std::vector<VkFormat> out;
    for (std::uint32_t f = 0; f < 400; ++f) out.push_back(VkFormat(f));
    for (const g::FormatRow& row : g::format_table) out.push_back(row.vulkan);
    for (const std::uint32_t f : {1000340000u, 1000340001u, 1000340002u, 1000156000u, 1000066000u, 0x7FFFFFFFu,
                                  0x80000000u, 0xFFFFFFFFu})
        out.push_back(VkFormat(f));
    return out;
}

bool same(const g::Swizzle& a, const g::Swizzle& b) { return a == b; }

// Every format and code, both ways; 0 when they agree.
int self_test() {
    const std::vector<VkFormat> formats = probe_formats();
    for (const VkFormat f : formats) {
        if (g::format_row(f) != reference::format_row(f)) {
            std::fprintf(stderr, "format_row(%u) differs\n", unsigned(f));
            return 1;
        }
        if (g::format_from_vulkan(f) != reference::format_from_vulkan(f) ||
            !same(g::format_swizzle(f), reference::format_swizzle(f))) {
            std::fprintf(stderr, "format_from_vulkan/format_swizzle(%u) differs\n", unsigned(f));
            return 1;
        }
    }
    std::vector<std::uint32_t> codes;
    for (std::uint32_t c = 0; c < 1100; ++c) codes.push_back(c);
    for (const std::uint32_t c : {0x7FFFFFFFu, 0x80000000u, 0xFFFFFFFFu}) codes.push_back(c);
    for (const std::uint32_t c : codes) {
        if (g::vulkan_format(c) != reference::vulkan_format(c)) {
            std::fprintf(stderr, "vulkan_format(%u) differs\n", c);
            return 1;
        }
    }
    std::size_t pairs = 0;
    for (const VkFormat f : formats)
        for (const std::uint32_t c : codes)
            for (const bool cast : {false, true}) {
                ++pairs;
                if (g::vulkan_view_format(f, c, cast) != reference::vulkan_view_format(f, c, cast)) {
                    std::fprintf(stderr, "vulkan_view_format(%u, %u, %d) differs\n", unsigned(f), c, int(cast));
                    return 1;
                }
            }
    std::printf("gfx10_format_bench self-test: %zu formats, %zu codes, %zu view-format queries agree\n",
                formats.size(), codes.size(), pairs);
    return 0;
}

// What one SRV of each texture does (d3d12_vulkan_device.cpp, image_descriptor and bind_image): the code of the
// named format, the view format of the image for that code, the channels of the view format, and the mapping.
// Formats as a frame mixes them: block-compressed albedo and normals, render targets, depth, a few odd ones.
struct View {
    VkFormat image;   // the resource's format
    VkFormat named;   // the view's format
};
const View mix[] = {
    {VK_FORMAT_BC7_SRGB_BLOCK, VK_FORMAT_BC7_SRGB_BLOCK},
    {VK_FORMAT_BC7_UNORM_BLOCK, VK_FORMAT_BC7_SRGB_BLOCK},
    {VK_FORMAT_BC1_RGBA_SRGB_BLOCK, VK_FORMAT_BC1_RGBA_SRGB_BLOCK},
    {VK_FORMAT_BC5_UNORM_BLOCK, VK_FORMAT_BC5_UNORM_BLOCK},
    {VK_FORMAT_BC4_UNORM_BLOCK, VK_FORMAT_BC4_UNORM_BLOCK},
    {VK_FORMAT_BC3_UNORM_BLOCK, VK_FORMAT_BC3_UNORM_BLOCK},
    {VK_FORMAT_BC6H_UFLOAT_BLOCK, VK_FORMAT_BC6H_UFLOAT_BLOCK},
    {VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_SRGB},
    {VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_B8G8R8A8_UNORM},
    {VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R16G16B16A16_SFLOAT},
    {VK_FORMAT_B10G11R11_UFLOAT_PACK32, VK_FORMAT_B10G11R11_UFLOAT_PACK32},
    {VK_FORMAT_A2B10G10R10_UNORM_PACK32, VK_FORMAT_A2B10G10R10_UNORM_PACK32},
    {VK_FORMAT_R32_SFLOAT, VK_FORMAT_R32_SFLOAT},
    {VK_FORMAT_D32_SFLOAT, VK_FORMAT_R32_SFLOAT},
    {VK_FORMAT_R16G16_SFLOAT, VK_FORMAT_R16G16_SFLOAT},
    {VK_FORMAT_R8_UNORM, VK_FORMAT_R8_UNORM},
    {VK_FORMAT_R32G32B32A32_SFLOAT, VK_FORMAT_R32G32B32A32_SFLOAT},
    {VK_FORMAT_R8G8B8A8_UINT, VK_FORMAT_R32_UINT},
};

template<typename Lookups>
std::uint64_t run(std::size_t iters, Lookups&& lookups) {
    std::uint64_t sink = 0;
    for (std::size_t i = 0; i < iters; ++i) {
        const View& v = mix[i % std::size(mix)];
        sink += lookups(v);
    }
    return sink;
}

struct Indexed {
    std::uint64_t operator()(const View& v) const {
        const std::uint32_t code = g::format_from_vulkan(v.named);
        const VkFormat view = g::vulkan_view_format(v.image, code, true);
        const g::Swizzle channels = g::format_swizzle(view);
        const VkComponentMapping m = g::vulkan_swizzle(g::identity_swizzle, view);
        return std::uint64_t(code) + std::uint64_t(view) + channels[0] + std::uint64_t(m.r) +
               std::uint64_t(g::vulkan_format(code));
    }
};
struct Scanned {
    std::uint64_t operator()(const View& v) const {
        const std::uint32_t code = reference::format_from_vulkan(v.named);
        const VkFormat view = reference::vulkan_view_format(v.image, code, true);
        const g::Swizzle channels = reference::format_swizzle(view);
        const VkComponentMapping m = reference::vulkan_swizzle(g::identity_swizzle, view);
        return std::uint64_t(code) + std::uint64_t(view) + channels[0] + std::uint64_t(m.r) +
               std::uint64_t(reference::vulkan_format(code));
    }
};

template<typename Lookups>
double median_ns(std::size_t iters, std::size_t reps, Lookups lookups, std::uint64_t& sink) {
    std::vector<double> times;
    for (std::size_t r = 0; r < reps + 2; ++r) {
        const auto start = std::chrono::steady_clock::now();
        sink += run(iters, lookups);
        const auto stop = std::chrono::steady_clock::now();
        if (r >= 2) times.push_back(std::chrono::duration<double, std::nano>(stop - start).count() / double(iters));
    }
    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}

int usage() {
    std::fprintf(stderr, "usage: gfx10_format_bench [--iters N] [--reps R] [--self-test]\n");
    return 2;
}

} // namespace

int main(int argc, char** argv) {
    std::size_t iters = 2000000, reps = 15;
    bool self = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--self-test") self = true;
        else if (arg == "--iters" && i + 1 < argc) iters = std::strtoull(argv[++i], nullptr, 10);
        else if (arg == "--reps" && i + 1 < argc) reps = std::strtoull(argv[++i], nullptr, 10);
        else return usage();
    }
    if (const int r = self_test(); r || self) return r;
    if (!iters || !reps) return usage();

    std::uint64_t sink_indexed = 0, sink_scanned = 0;
    const double scanned = median_ns(iters, reps, Scanned{}, sink_scanned);
    const double indexed = median_ns(iters, reps, Indexed{}, sink_indexed);
    // Both compute the same values for the mix, so the sums agree.
    if (sink_indexed != sink_scanned) {
        std::fprintf(stderr, "the two lookups disagree on the benchmark mix\n");
        return 1;
    }
    std::printf("per SRV view (format_from_vulkan + vulkan_view_format + format_swizzle + vulkan_swizzle + "
                "vulkan_format), median of %zu runs of %zu:\n", reps, iters);
    std::printf("  table scan  %8.2f ns\n", scanned);
    std::printf("  indexed     %8.2f ns  (%.1fx, %.0f%% less)\n", indexed, scanned / indexed,
                100.0 * (1.0 - indexed / scanned));
    return 0;
}
