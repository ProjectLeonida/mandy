// Which (device address - heap_base) offsets a shader-visible V# can name. The translated shaders read the 48-bit
// base field sign-extended from bit 47 (docs/design.md, "Heap-relative buffer device addresses"), so the range is
// [0, 2^47) above the heap and [2^64 - 2^47, 2^64) below it, and [2^47, 2^48) is not representable. CPU only, no
// Vulkan, no game data: the same predicate the device uses (src/vsharp_range.h).
#include "vsharp_range.h"

#include <cstdint>
#include <cstdio>

namespace {

int g_failures = 0;

void expect(bool condition, const char* text) {
    if (!condition) {
        std::fprintf(stderr, "[rt] vsharp-range: FAILED: %s\n", text);
        ++g_failures;
    }
}

void test_above_heap() {
    expect(leonida::vsharp_offset_fits((1ull << 47) - 1, 1), "2^47 - 1 fits");
    expect(!leonida::vsharp_offset_fits(1ull << 47, 1), "2^47 does not fit");
    expect(!leonida::vsharp_offset_fits((1ull << 48) - 1, 1), "2^48 - 1 does not fit");
    expect(!leonida::vsharp_offset_fits(0, (1ull << 47) + 1), "a span crossing 2^47 does not fit");
    expect(leonida::vsharp_offset_fits(0, 1ull << 47), "a span ending exactly at 2^47 fits");
    expect(!leonida::vsharp_offset_fits(0, 0), "an empty range does not fit");
}

void test_below_heap() {
    expect(leonida::vsharp_offset_fits(UINT64_MAX - ((1ull << 47) - 1), 1), "-2^47 fits");
    expect(!leonida::vsharp_offset_fits(UINT64_MAX - (1ull << 47), 1), "-2^47 - 1 does not fit");
    expect(leonida::vsharp_offset_fits(UINT64_MAX - (1ull << 30) + 1, 1ull << 20), "-1 GiB, 1 MiB fits");
    expect(leonida::vsharp_offset_fits(UINT64_MAX, 1), "-1 fits");
}

} // namespace

int main() {
    test_above_heap();
    test_below_heap();
    if (g_failures) {
        std::fprintf(stderr, "[rt] vsharp-range: %d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("[rt] vsharp-range: all checks passed\n");
    return 0;
}
