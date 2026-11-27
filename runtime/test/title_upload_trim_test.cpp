// leonida::title_upload_trim (src/title_upload_trim.h): when the title-texture upload ring releases an idle slot's
// buffer. A large idle slot whose transfer is done goes; a busy, recent, small or empty one stays; 0 ms turns it off;
// malformed settings fall back to the defaults. No GPU, no device.
#include "../src/title_upload_trim.h"

#include <cstdint>
#include <cstdio>
#include <initializer_list>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

namespace tt = leonida::title_upload_trim;
constexpr std::uint64_t kMiB = 1 << 20;
constexpr std::uint64_t kMs = 1000 * 1000;

int defaults() {
    const tt::Settings s = tt::settings_from(nullptr, nullptr);
    EXPECT(s.enabled());
    EXPECT(s.idle_ns == 2000 * kMs);
    EXPECT(s.keep_bytes == 4 * kMiB);
    return 0;
}

int rule() {
    const tt::Settings s = tt::settings_from(nullptr, nullptr);
    const std::uint64_t t0 = 10'000 * kMs;
    // 64 MiB, done, idle exactly the limit and past it: released.
    EXPECT(tt::release(64 * kMiB, t0, t0 + 2000 * kMs, true, s));
    EXPECT(tt::release(64 * kMiB, t0, t0 + 60'000 * kMs, true, s));
    // Idle for less than the limit: kept.
    EXPECT(!tt::release(64 * kMiB, t0, t0 + 1999 * kMs, true, s));
    // Its transfer is still running: kept however long ago it was used.
    EXPECT(!tt::release(64 * kMiB, t0, t0 + 60'000 * kMs, false, s));
    // At or under the size kept: kept; one byte over: released.
    EXPECT(!tt::release(4 * kMiB, t0, t0 + 60'000 * kMs, true, s));
    EXPECT(tt::release(4 * kMiB + 1, t0, t0 + 60'000 * kMs, true, s));
    // Nothing allocated: nothing to release.
    EXPECT(!tt::release(0, t0, t0 + 60'000 * kMs, true, s));
    // A clock that reads earlier than the last use (another thread stamped it after `now` was read): kept.
    EXPECT(!tt::release(64 * kMiB, t0, t0 - 1, true, s));
    return 0;
}

int settings() {
    // 0 ms: never released, the behaviour before the trim.
    const tt::Settings off = tt::settings_from("0", nullptr);
    EXPECT(!off.enabled());
    EXPECT(!tt::release(1024 * kMiB, 0, UINT64_MAX, true, off));
    // Values are read as given.
    const tt::Settings s = tt::settings_from("500", "0");
    EXPECT(s.idle_ns == 500 * kMs);
    EXPECT(s.keep_bytes == 0);
    EXPECT(tt::release(1, 0, 500 * kMs, true, s));
    // Malformed or empty: the defaults.
    for (const char* bad : {"", "-1", "2s", "1e3", " 5", "99999999999999999999999"}) {
        const tt::Settings d = tt::settings_from(bad, bad);
        EXPECT(d.idle_ns == 2000 * kMs);
        EXPECT(d.keep_bytes == 4 * kMiB);
    }
    // Large but valid values saturate instead of wrapping.
    const tt::Settings big = tt::settings_from("18446744073709551615", "18446744073709551615");
    EXPECT(big.idle_ns == UINT64_MAX);
    EXPECT(big.keep_bytes == UINT64_MAX);
    EXPECT(!tt::release(1024 * kMiB, 0, UINT64_MAX - 1, true, big));
    return 0;
}

}  // namespace

int main() {
    if (defaults() || rule() || settings()) return 1;
    std::puts("title_upload_trim_test: ok");
    return 0;
}
