#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>

namespace leonida::dstorage {
// The public Xbox-GDK-Samples DirectStorageTextureShuffling example stores each BC component
// plane across the whole request. The MDU interleaves a stripe of each plane per group. The
// sample's compute shader does the remaining intra-group shuffle; surface tiling is already baked in.
inline constexpr unsigned stripes[][5] = {{}, {1, 1, 2}, {1, 1, 6}, {1, 3, 1, 1, 2}, {1, 3, 1, 3}};

inline size_t group_size(unsigned mode) { return mode == 1 ? 16384 : 32768; }

inline bool unshuffle(unsigned mode, std::span<const uint8_t> input, std::span<uint8_t> out) {
    if (!mode || mode > 4 || input.size() != out.size() || input.size() % group_size(mode)) return false;
    const size_t size = input.size();
    if (!size) return true;
    const auto in_begin = reinterpret_cast<std::uintptr_t>(input.data());
    const auto out_begin = reinterpret_cast<std::uintptr_t>(out.data());
    if (in_begin < out_begin ? out_begin - in_begin < size : in_begin - out_begin < size) return false;
    const size_t group = group_size(mode), groups = size / group;
    size_t stripe_bytes[5]{};
    const uint8_t* src[5]{};
    size_t stripe_count = 0;
    size_t source = 0;
    for (unsigned pages : stripes[mode]) {
        if (!pages) break;
        const size_t bytes = size_t(pages) * 4096;
        stripe_bytes[stripe_count] = bytes;
        src[stripe_count] = input.data() + source;
        source += bytes * groups;
        ++stripe_count;
    }
    // Emit each output group sequentially in a single pass instead of striding across the whole destination once
    // per plane stripe, keeping destination writes contiguous for cache/write-combining efficiency.
    uint8_t* dst = out.data();
    for (size_t i = 0; i < groups; ++i) {
        for (size_t s = 0; s < stripe_count; ++s) {
            const size_t bytes = stripe_bytes[s];
            std::memcpy(dst, src[s], bytes);
            src[s] += bytes;
            dst += bytes;
        }
    }
    return true;
}
} // namespace leonida::dstorage
