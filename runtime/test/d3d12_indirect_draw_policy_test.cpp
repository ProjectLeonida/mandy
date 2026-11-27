// Portable C++20 tests for runtime/shims/d3d12_indirect_draw_policy.h
#include "../shims/d3d12_indirect_draw_policy.h"

#include <cstddef>
#include <cstdio>
#include <cwchar>

#define CHECK(condition) do {                                        \
    if (!(condition)) {                                              \
        std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); \
        return 1;                                                    \
    }                                                                \
} while (0)

int main() {
    using namespace rt::d3d12;

    // 1. Policy parse: parse_indirect_draws_disabled
    CHECK(parse_indirect_draws_disabled(L"off", 3));
    CHECK(!parse_indirect_draws_disabled(L"off", 2));
    CHECK(!parse_indirect_draws_disabled(L"off", 4));
    CHECK(!parse_indirect_draws_disabled(L"OFF", 3));
    CHECK(!parse_indirect_draws_disabled(L"Off", 3));
    CHECK(!parse_indirect_draws_disabled(L"raw", 3));
    CHECK(!parse_indirect_draws_disabled(L"on", 2));
    CHECK(!parse_indirect_draws_disabled(L"1", 1));
    CHECK(!parse_indirect_draws_disabled(L"0", 1));
    CHECK(!parse_indirect_draws_disabled(L"", 0));
    CHECK(!parse_indirect_draws_disabled(nullptr, 0));
    CHECK(!parse_indirect_draws_disabled(nullptr, 3));
    CHECK(!parse_indirect_draws_disabled(L"disabled", 8));

    // 2. Obsolete raw flag detection: is_indirect_draws_raw_obsolete
    CHECK(is_indirect_draws_raw_obsolete(L"raw", 3));
    CHECK(!is_indirect_draws_raw_obsolete(L"raw", 2));
    CHECK(!is_indirect_draws_raw_obsolete(L"RAW", 3));
    CHECK(!is_indirect_draws_raw_obsolete(L"off", 3));
    CHECK(!is_indirect_draws_raw_obsolete(L"", 0));
    CHECK(!is_indirect_draws_raw_obsolete(nullptr, 0));
    CHECK(!is_indirect_draws_raw_obsolete(nullptr, 3));

    // 3. Process environment interaction:
    // When LEONIDA_INDIRECT_DRAWS is unset, indirect_draws_disabled() returns false.
    SetEnvironmentVariableW(L"LEONIDA_INDIRECT_DRAWS", nullptr);
    const bool initially_disabled = indirect_draws_disabled();
    CHECK(!initially_disabled);

    std::printf("d3d12_indirect_draw_policy: parse and obsolete-flag helpers verified cleanly\n");
    return 0;
}
