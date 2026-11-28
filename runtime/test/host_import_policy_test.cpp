// Portable CPU tests for leonida::HostImportRangeCheck and leonida::host_import_memory_types.
#include "../src/host_import_policy.h"

#include <array>
#include <cstdio>
#include <vector>

namespace {

int g_failures = 0;

void expect(bool ok, const char* what, int line) {
    if (ok) return;
    std::fprintf(stderr, "FAIL line %d: %s\n", line, what);
    ++g_failures;
}

#define EXPECT(cond) expect((cond), #cond, __LINE__)

void test_private_and_pagefile_regions() {
    for (bool wine : {false, true}) {
        leonida::HostImportRangeCheck check{wine, true};
        EXPECT(check.accept({0x10000, true, true, false, false}));
        // Same allocation base (e.g. a second query chunk inside one reservation) stays accepted.
        EXPECT(check.accept({0x10000, true, true, false, false}));
    }
}

void test_uncommitted_cache_modified_and_readonly() {
    for (bool wine : {false, true}) {
        {
            leonida::HostImportRangeCheck check{wine, true};
            EXPECT(!check.accept({0x10000, false, true, false, false}));
        }
        {
            leonida::HostImportRangeCheck check{wine, true};
            EXPECT(!check.accept({0x10000, true, true, true, false}));
        }
        {
            leonida::HostImportRangeCheck check{wine, true};
            EXPECT(!check.accept({0x10000, true, false, false, false}));
        }
        {
            // When writability is not required, a committed non-cache-modified region is accepted.
            leonida::HostImportRangeCheck check{wine, false};
            EXPECT(check.accept({0x10000, true, false, false, false}));
        }
    }
}

void test_file_backed_mapped_rejected_under_wine_only() {
    {
        leonida::HostImportRangeCheck wine_check{true, true};
        EXPECT(!wine_check.accept({0x20000, true, true, false, true}));
    }
    {
        leonida::HostImportRangeCheck win_check{false, true};
        EXPECT(win_check.accept({0x20000, true, true, false, true}));
    }
}

void test_multi_allocation_rejected_under_wine_only() {
    {
        leonida::HostImportRangeCheck wine_check{true, true};
        EXPECT(wine_check.accept({0x30000, true, true, false, false}));
        EXPECT(!wine_check.accept({0x40000, true, true, false, false}));
    }
    {
        leonida::HostImportRangeCheck win_check{false, true};
        EXPECT(win_check.accept({0x30000, true, true, false, false}));
        EXPECT(win_check.accept({0x40000, true, true, false, false}));
    }
}

void test_memory_type_order_deduplicates() {
    constexpr std::uint32_t kHostVisible = 0x2u;
    constexpr std::uint32_t kHostCoherent = 0x4u;
    constexpr std::uint32_t kHostCached = 0x8u;
    constexpr std::uint32_t kPreferred = kHostVisible | kHostCoherent;

    // Mirrors NVIDIA desktop layout: type 0 = sysmem(0), type 1 = VRAM(1), type 2 = host-vis(2),
    // type 3 = host-vis|coherent(6), type 4 = host-vis|coherent|cached(14).
    const std::array<std::uint32_t, 5> flags = {0u, 1u, kHostVisible, kPreferred, kPreferred | kHostCached};

    // Allowed types: 0, 2, 3, 4 (mask 0x1d). Preferred (3, 4) must come first, then 0, 2; no duplicates.
    const auto order = leonida::host_import_memory_types(0x1du, flags.data(), std::uint32_t(flags.size()), kPreferred);
    const std::vector<std::uint32_t> expected = {3u, 4u, 0u, 2u};
    EXPECT(order == expected);

    // Allowed types all preferred (mask 0x18 = types 3 and 4): each appears exactly once, not twice.
    const auto pref_only =
        leonida::host_import_memory_types(0x18u, flags.data(), std::uint32_t(flags.size()), kPreferred);
    const std::vector<std::uint32_t> expected_pref = {3u, 4u};
    EXPECT(pref_only == expected_pref);

    // Empty / out-of-range bits produce an empty list.
    EXPECT(leonida::host_import_memory_types(0u, flags.data(), std::uint32_t(flags.size()), kPreferred).empty());
    EXPECT(leonida::host_import_memory_types(0x100u, flags.data(), std::uint32_t(flags.size()), kPreferred).empty());
    EXPECT(leonida::host_import_memory_types(0x1du, nullptr, 5u, kPreferred).empty());
}

}  // namespace

int main() {
    test_private_and_pagefile_regions();
    test_uncommitted_cache_modified_and_readonly();
    test_file_backed_mapped_rejected_under_wine_only();
    test_multi_allocation_rejected_under_wine_only();
    test_memory_type_order_deduplicates();
    if (g_failures) return 1;
    std::printf("ok\n");
    return 0;
}
