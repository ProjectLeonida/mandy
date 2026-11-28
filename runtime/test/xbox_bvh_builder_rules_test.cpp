// The XboxBvhBuilder shim's postbuild filter and rehydrate scratch regions (shims/xbox_bvh_builder_rules.h), on the
// CPU, following kevincommits' finding (#1608, Finding 19, item 3): an offline-statistics (0x10) postbuild
// description after 16 kept ones still skips the element, and a scratch size near 2^64 neither wraps its alignment
// nor the batch's total. No device, no game data.
#include "../shims/xbox_bvh_builder_rules.h"

#include <cstdio>
#include <initializer_list>

namespace {
int failures = 0;
void check(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++failures;
}

using Post = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC;
constexpr auto kCompacted = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_COMPACTED_SIZE;
const auto kStatistics = static_cast<D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_TYPE>(
    leonida::xdxr::kPostbuildInfoOfflineStatistics);
}  // namespace

int main() {
    using namespace rt::xbox_bvh;
    Post kept[kMaxKeptPostbuild];

    // 20 descriptions: 0x10 at `at`, the rest COMPACTED_SIZE.
    for (const UINT at : {0u, 5u, 16u, 17u, 19u}) {
        Post post[20];
        for (UINT i = 0; i < 20; ++i) post[i] = {0x1000 + 8ull * i, kCompacted};
        post[at].InfoType = kStatistics;
        const auto f = filter_postbuild(post, 20, kept);
        char what[96];
        std::snprintf(what, sizeof(what), "0x10 at index %u of 20 skips the element", at);
        check(f.skipped_statistics, what);
        check(f.kept == kMaxKeptPostbuild && f.dropped, "the other 19: 16 kept, the rest dropped");
    }
    {
        Post post[20];
        for (UINT i = 0; i < 20; ++i) post[i] = {0x1000 + 8ull * i, kCompacted};
        const auto f = filter_postbuild(post, 20, kept);
        check(!f.skipped_statistics && f.kept == 16 && f.dropped && kept[15].DestBuffer == 0x1000 + 8 * 15,
              "20 without 0x10: the first 16 kept, in order");
        const auto g = filter_postbuild(post, 3, kept);
        check(!g.skipped_statistics && g.kept == 3 && !g.dropped, "3 without 0x10: all kept");
        const auto none = filter_postbuild(nullptr, 5, kept);
        check(none.kept == 0 && !none.skipped_statistics, "no array: nothing kept");
    }

    // Scratch regions: aligned, in order; a size or total that would wrap is refused and changes nothing.
    constexpr UINT64 align = 256;
    UINT64 total = 0, region = 0;
    check(add_scratch_region(total, 1, align, region) && region == 0 && total == 256, "size 1: one aligned region");
    check(add_scratch_region(total, 0, align, region) && region == 256 && total == 512, "size 0 counts as 1");
    check(add_scratch_region(total, 300, align, region) && region == 512 && total == 1024, "size 300: 512 bytes");
    UINT64 before = total, region_before = region;
    check(!add_scratch_region(total, UINT64_MAX - 100, align, region) && total == before && region == region_before,
          "a size within 255 of 2^64 is refused (its alignment would wrap to 0)");
    check(!add_scratch_region(total, UINT64_MAX - 1023, align, region) && total == before,
          "a size whose aligned region would push the total past 2^64 is refused");
    UINT64 big = UINT64_MAX - 511, big_region = 0;
    check(add_scratch_region(big, 256, align, big_region) && big == UINT64_MAX - 255 && big_region == UINT64_MAX - 511,
          "a region that ends exactly within range fits");
    check(!add_scratch_region(big, 512, align, big_region) && big == UINT64_MAX - 255,
          "the next one does not fit, and the total stays");

    std::printf("%d failure(s)\n", failures);
    return failures ? 1 : 0;
}
