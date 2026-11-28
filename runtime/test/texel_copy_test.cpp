// copy_elements (shims/xg_xs_layout.h) copies each texel or block with a compile-time-sized memcpy for element sizes
// 1/2/4/8/16 and with a runtime-sized one for every other size. This compares it, byte for byte over the whole
// destination buffer, with the original runtime-sized loop kept here as the reference: tiled and linear layouts, every
// mip (tails included), array slices, volumes, both directions, and unaligned pointers and pitches. Synthetic textures
// only; no game files and no GPU.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "xg_xs_layout.h"

namespace {

using namespace rt::xg_xs;

int g_checks = 0, g_failures = 0;

void check(bool condition, const char* what) {
    ++g_checks;
    if (condition) return;
    ++g_failures;
    std::printf("FAIL: %s\n", what);
}

// The loop as it was before the element size was dispatched: std::memcpy with `bytes` read at run time.
template<bool Into, class Byte>
void reference_copy(const Texture& t, uint32_t level, uint32_t slice, const Byte* source, Byte* destination,
                    uint32_t row_pitch, uint32_t slice_pitch, std::vector<uint64_t>& columns) {
    const auto& mip = t.layout.plane[0].mips[level];
    const uint32_t bytes = t.format.bytes;
    const uint32_t row_bytes = mip.width_elements * bytes;
    const uint32_t depth = t.layout.dimension == kTexture3D ? mip.depth : 1;
    const ElementAddress address(t);
    const auto tail = address.tail_origin(mip);
    columns.resize(mip.width_elements);
    for (uint32_t x = 0; x < mip.width_elements; ++x) columns[x] = address.column(mip, tail, x);
    for (uint32_t z = 0; z < depth; ++z) {
        for (uint32_t y = 0; y < mip.height_elements; ++y) {
            const uint64_t linear = uint64_t(z) * slice_pitch + uint64_t(y) * row_pitch;
            const uint64_t tiled = address.row(mip, tail, y, slice + z);
            if (t.mode == kLinear) {
                std::memcpy(destination + (Into ? tiled : linear), source + (Into ? linear : tiled), row_bytes);
                continue;
            }
            for (uint32_t x = 0; x < mip.width_elements; ++x) {
                const uint64_t at = tiled + columns[x];
                const uint64_t to = linear + uint64_t(x) * bytes;
                std::memcpy(destination + (Into ? at : to), source + (Into ? to : at), bytes);
            }
        }
    }
}

uint32_t g_seed = 12345;
uint8_t next_byte() {
    g_seed = g_seed * 1664525u + 1013904223u;
    return uint8_t(g_seed >> 24);
}

// `misalign` moves every buffer start off its allocation's natural alignment; `extra` pads the pitches.
void compare_texture(const Texture& t, const char* what, uint32_t misalign, uint32_t extra) {
    for (uint32_t slice = 0; slice < t.array; ++slice) {
        for (uint32_t level = 0; level < t.layout.mips; ++level) {
            const auto& mip = t.layout.plane[0].mips[level];
            const uint32_t depth = t.layout.dimension == kTexture3D ? mip.depth : 1;
            const uint32_t pitch = mip.width_elements * t.format.bytes + extra;
            const uint32_t slice_pitch = depth > 1 ? pitch * mip.height_elements + extra : 0;
            const size_t linear_size = size_t(slice_pitch) * depth + size_t(pitch) * mip.height_elements + 64;
            // Slack: a non-specialized width wider than the built one may read past the layout's last row.
            const size_t tiled_size = size_t(t.layout.size) + size_t(mip.width_elements) * 16 + 64;
            for (int direction = 0; direction < 2; ++direction) {
                const bool into = direction == 0;
                const size_t source_size = into ? linear_size : tiled_size;
                const size_t destination_size = into ? tiled_size : linear_size;
                std::vector<uint8_t> source(source_size + misalign), want(destination_size + misalign);
                for (auto& b : source) b = next_byte();
                for (auto& b : want) b = next_byte();  // the bytes a copy must leave alone
                std::vector<uint8_t> got = want;
                std::vector<uint64_t> columns, reference_columns;
                if (into) {
                    reference_copy<true>(t, level, slice, source.data() + misalign, want.data() + misalign, pitch,
                                         slice_pitch, reference_columns);
                    copy_elements<true>(t, level, slice, source.data() + misalign, got.data() + misalign, pitch,
                                        slice_pitch, columns);
                } else {
                    reference_copy<false>(t, level, slice, source.data() + misalign, want.data() + misalign, pitch,
                                          slice_pitch, reference_columns);
                    copy_elements<false>(t, level, slice, source.data() + misalign, got.data() + misalign, pitch,
                                         slice_pitch, columns);
                }
                if (want != got) {
                    std::printf("  %s slice %u level %u %s misalign %u extra %u\n", what, slice, level,
                                into ? "into" : "from", misalign, extra);
                    check(false, "copy_elements matches the runtime-sized reference byte for byte");
                    return;
                }
            }
        }
    }
    check(true, what);
}

struct Case {
    const char* name;
    uint32_t dimension, width, height, depth_or_array, mips, format, layout;
};

void run_case(const Case& c, bool non_specialized) {
    ResourceDesc desc{c.dimension, 0, c.width, c.height, uint16_t(c.depth_or_array), uint16_t(c.mips), c.format, 1, 0,
                      c.layout, 0};
    Texture t;
    if (!build_texture(desc, t)) {
        check(false, c.name);
        return;
    }
    if (non_specialized) {
        // Same addressing, an element size the dispatch does not specialize (the tiled offsets stay inside the
        // resource because the element is no wider than the one it was built for).
        const uint32_t widths[17] = {0, 3, 3, 0, 3, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 12};
        t.format.bytes = widths[t.format.bytes];
    }
    char name[160];
    std::snprintf(name, sizeof name, "%s: %u-byte elements, %s", c.name, t.format.bytes,
                  t.mode == kLinear ? "linear" : "tiled");
    for (uint32_t misalign : {0u, 1u, 3u})
        for (uint32_t extra : {0u, 1u, 7u}) compare_texture(t, name, misalign, extra);
}

}  // namespace

int main() {
    const Case cases[] = {
        {"R8 2D tiled 64kb", kTexture2D, 61, 35, 1, 6, DXGI_FORMAT_R8_UNORM, kSwizzled | k64kbS},
        {"R16 2D tiled 64kb", kTexture2D, 100, 60, 1, 7, DXGI_FORMAT_R16_UNORM, kSwizzled | k64kbS},
        {"RGBA8 2D tiled 64kb", kTexture2D, 130, 70, 1, 8, DXGI_FORMAT_R8G8B8A8_UNORM, kSwizzled | k64kbD},
        {"RGBA16F 2D tiled", kTexture2D, 67, 33, 1, 7, DXGI_FORMAT_R16G16B16A16_FLOAT, kSwizzled | k64kbS},
        {"RGBA32F 2D tiled", kTexture2D, 40, 36, 1, 6, DXGI_FORMAT_R32G32B32A32_FLOAT, kSwizzled | k4kbS},
        {"BC1 blocks tiled", kTexture2D, 109, 71, 1, 6, DXGI_FORMAT_BC1_UNORM, kSwizzled | k64kbS},
        {"BC3 blocks tiled", kTexture2D, 109, 71, 1, 6, DXGI_FORMAT_BC3_UNORM, kSwizzled | k64kbS},
        {"RGBA8 array tiled", kTexture2D, 70, 70, 3, 5, DXGI_FORMAT_R8G8B8A8_UNORM, kSwizzled | k4kbS},
        {"BC5 array tiled", kTexture2D, 64, 64, 4, 7, DXGI_FORMAT_BC5_UNORM, kSwizzled | k64kbS},
        {"R8 1D linear", kTexture1D, 300, 1, 1, 4, DXGI_FORMAT_R8_UNORM, 1},
        {"RGBA8 volume", kTexture3D, 61, 35, 20, 6, DXGI_FORMAT_R8G8B8A8_UNORM, kSwizzled | k4kbS},
        {"R8 volume", kTexture3D, 33, 17, 9, 5, DXGI_FORMAT_R8_UNORM, kSwizzled | k4kbS},
        {"RGBA16F volume", kTexture3D, 61, 35, 67, 7, DXGI_FORMAT_R16G16B16A16_FLOAT, kSwizzled | k4kbS},
        {"RGBA32F volume", kTexture3D, 30, 20, 11, 5, DXGI_FORMAT_R32G32B32A32_FLOAT, kSwizzled | k4kbS},
        {"RGBA8 linear", kTexture2D, 83, 31, 1, 6, DXGI_FORMAT_R8G8B8A8_UNORM, 1},
        {"RGBA16F linear array", kTexture2D, 50, 20, 3, 4, DXGI_FORMAT_R16G16B16A16_FLOAT, 1},
        {"BC1 linear", kTexture2D, 64, 64, 1, 5, DXGI_FORMAT_BC1_UNORM, 1},
    };
    for (const Case& c : cases) run_case(c, false);
    // Element sizes outside 1/2/4/8/16 take the runtime-sized path.
    for (const Case& c : cases) run_case(c, true);
    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
