// Portable unit tests for runtime/src/submit_phases.h (pure stdlib, no GPU or game data).
#include "../src/submit_phases.h"

#include <cstdint>
#include <cstdio>
#include <limits>
#include <string_view>

namespace {

int g_failures = 0;

void check(bool cond, const char* what) {
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_failures;
    }
}

void test_parse_period_ns() {
    using namespace leonida::submit_phases;
    constexpr std::int64_t kDef = kDefaultPeriodSeconds * kNsPerSecond;
    constexpr std::int64_t kMax = static_cast<std::int64_t>(kMaxPeriodSeconds) * kNsPerSecond;

    check(parse_period_ns(nullptr) == kDef, "null text uses 5s default");
    check(parse_period_ns("") == kDef, "empty text uses 5s default");
    check(parse_period_ns("   ") == kDef, "whitespace-only text uses 5s default");
    check(parse_period_ns("0") == 0, "0 disables periodic logging");
    check(parse_period_ns(" 0 ") == 0, "padded 0 disables periodic logging");
    check(parse_period_ns("1") == kNsPerSecond, "1s parses exactly");
    check(parse_period_ns("12") == 12 * kNsPerSecond, "12s parses exactly");
    check(parse_period_ns("  30\t") == 30 * kNsPerSecond, "trimmed decimal parses");

    // Negative or malformed inputs must never underflow/wrap to negative periods or disable logging.
    check(parse_period_ns("-1") == kDef, "negative value falls back to 5s default");
    check(parse_period_ns("-0") == kDef, "signed -0 falls back to 5s default");
    check(parse_period_ns("+5") == kDef, "signed +5 falls back to 5s default");
    check(parse_period_ns("abc") == kDef, "non-numeric text falls back to 5s default");
    check(parse_period_ns("5s") == kDef, "trailing junk falls back to 5s default");

    // Huge values must saturate cleanly without signed int64 overflow UB.
    check(parse_period_ns("9223372036") == kMax, "exact max seconds saturates cleanly");
    check(parse_period_ns("9223372037") == kMax, "one past max seconds saturates cleanly");
    check(parse_period_ns("99999999999999999999999999") == kMax, "huge value saturates without overflow");
    check(parse_period_ns("99999999999999999999999999") > 0, "saturated period remains positive");
}

void test_add_and_bounds() {
    using namespace leonida::submit_phases;
    auto& t = totals();
    t.ns[replay].store(0, std::memory_order_relaxed);
    t.max_ns[replay].store(0, std::memory_order_relaxed);

    add(replay, 1500);
    add(replay, 500);
    add(replay, 4000);
    check(t.ns[replay].load(std::memory_order_relaxed) == 6000, "add accumulates total nanoseconds");
    check(t.max_ns[replay].load(std::memory_order_relaxed) == 4000, "add tracks maximum single duration");

    // Out-of-range phase index must be ignored cleanly without touching memory.
    add(kCount, 12345);
    add(static_cast<Phase>(99), 12345);
    check(std::string_view(name(replay)) == "replay", "valid phase name");
    check(std::string_view(name(kCount)) == "?", "out-of-range phase name returns ?");
}

void test_scope_and_submitted() {
    using namespace leonida::submit_phases;
    auto& t = totals();
    const std::uint64_t before_subs = t.all_submissions.load(std::memory_order_relaxed);
    {
        const Scope s(textures);
        (void)s;
    }
    submitted();
    check(t.all_submissions.load(std::memory_order_relaxed) == before_subs + 1,
          "submitted increments all_submissions");
}

}  // namespace

int main() {
    test_parse_period_ns();
    test_add_and_bounds();
    test_scope_and_submitted();
    std::printf("submit_phases_test: %s (%d failure%s)\n", g_failures ? "FAILED" : "OK", g_failures,
                g_failures == 1 ? "" : "s");
    return g_failures ? 1 : 0;
}
