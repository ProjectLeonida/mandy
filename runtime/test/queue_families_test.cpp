#include "../src/queue_families.h"

#include <array>
#include <cstdint>
#include <cstdio>

#define EXPECT(cond) do { \
    if (!(cond)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #cond); return 1; } \
} while (0)

int main() {
    using namespace leonida;

    // Empty pools return SIZE_MAX.
    const std::span<const std::uint32_t> empty{};
    const auto none = pick_queue_slot({empty, empty});
    EXPECT(none.pool == SIZE_MAX && none.index == 0);

    // Unused slot wins, earliest on a tie.
    const std::array<std::uint32_t, 2> p0{2u, 1u};
    const std::array<std::uint32_t, 2> p1{0u, 0u};
    const auto unused = pick_queue_slot({p0, p1});
    EXPECT(unused.pool == 1 && unused.index == 0);

    // Regression: non-empty pool where every slot has UINT32_MAX users must still pick the earliest slot.
    const std::array<std::uint32_t, 2> saturated{UINT32_MAX, UINT32_MAX};
    const auto sat = pick_queue_slot({empty, saturated});
    EXPECT(sat.pool == 1 && sat.index == 0);

    // Timestamp bits and sparse fallback sanity checks.
    const std::array<std::uint32_t, 2> ts{64u, 0u};
    EXPECT(queue_family_timestamp_bits(ts, 0) == 64u);
    EXPECT(queue_family_timestamp_bits(ts, 5) == 0u);
    EXPECT(copy_queue_timestamp_families_supported(ts, 0, std::nullopt));
    EXPECT(!copy_queue_timestamp_families_supported(ts, 0, 1u));

    std::puts("queue_families_test: ok");
    return 0;
}
