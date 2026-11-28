// Two rules of the XboxBvhBuilder shim (xbox_bvh_builder.cpp) that need no device, so they are tested on the CPU
// (xbox_bvh_builder_rules_test):
//   - filter_postbuild: an element's postbuild descriptions without the offline-statistics type (0x10), which the
//     DLL refuses for the whole element. Every description is looked at, also past the 16 kept: a 0x10 one at index
//     17 or later still makes the element one the shim skips.
//   - add_scratch_region: one element's region of a rehydrate's shared scratch buffer, aligned, without the size or
//     the running total wrapping past 2^64 (which would size the buffer too small and overlap the regions).
#pragma once

#include <windows.h>
#include <d3d12.h>

#include <cstdint>

#include "../src/xdxr_descs.h"

namespace rt::xbox_bvh {

inline constexpr UINT kMaxKeptPostbuild = 16;

struct PostbuildFilter {
    UINT kept = 0;                    // descriptions copied to the output, at most kMaxKeptPostbuild
    bool skipped_statistics = false;  // a 0x10 description was among them: the element is skipped
    bool dropped = false;             // more than kMaxKeptPostbuild others: the rest were left out
};

inline PostbuildFilter filter_postbuild(const D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC* post,
                                        UINT count,
                                        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC (&out)[kMaxKeptPostbuild]) {
    PostbuildFilter f;
    for (UINT p = 0; post && p < count; ++p) {
        if (static_cast<uint32_t>(post[p].InfoType) == leonida::xdxr::kPostbuildInfoOfflineStatistics) {
            f.skipped_statistics = true;
            continue;
        }
        // Past the 16 kept the rest are dropped, but still scanned for 0x10.
        if (f.kept == kMaxKeptPostbuild) {
            f.dropped = true;
            continue;
        }
        out[f.kept++] = post[p];
    }
    return f;
}

// Adds a region of `size` bytes (0 counts as 1) at `align` (a power of two) to a scratch buffer of `total` bytes:
// on success `region` is its offset and `total` the new size. False, with both unchanged, when the aligned size or
// the new total would not fit in 64 bits.
inline bool add_scratch_region(UINT64& total, UINT64 size, UINT64 align, UINT64& region) {
    if (!size) size = 1;
    if (size > UINT64_MAX - (align - 1)) return false;
    const UINT64 aligned = (size + align - 1) & ~(align - 1);
    if (aligned > UINT64_MAX - total) return false;
    region = total;
    total += aligned;
    return true;
}

}  // namespace rt::xbox_bvh
