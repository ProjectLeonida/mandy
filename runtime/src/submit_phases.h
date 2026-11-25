#pragma once
// Where a submission's CPU time goes, always on and cheap: each phase of VulkanQueue::execute_now adds its time here
// (two steady_clock reads and two relaxed atomic adds), and every LEONIDA_SUBMIT_PHASES_SECONDS (5 by default, 0 turns
// the line off) one line says, since the last line, how many submissions there were, how many per second, the
// milliseconds per submission of each phase with the largest single one, and the share of the wall time the
// submitting threads spent in them. The time is the process's age (GetProcessTimes), so it lines up with the boot.
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>

namespace leonida::submit_phases {

enum Phase : unsigned {
    wait_room,  // VulkanQueue::wait_for_room: the queue at its cap waits for the GPU
    heaps,      // DeviceState::refresh_console_heaps
    shadows,    // slot-133 shadow buffers and title windows (ShadowPages::refresh)
    geometry,   // GeometryViews::refresh
    textures,   // Device::refresh_title_textures
    replay,     // the rest of execute_now: preparing, recording and submitting the lists
    kCount
};

inline const char* name(unsigned p) {
    static const char* const names[kCount] = {"wait", "heaps", "shadows", "geometry", "textures", "replay"};
    return p < kCount ? names[p] : "?";
}

struct Totals {
    std::atomic<std::uint64_t> ns[kCount]{};
    std::atomic<std::uint64_t> max_ns[kCount]{};
    std::atomic<std::uint64_t> submissions{0}, all_submissions{0};
    std::atomic<std::int64_t> last_log{0};
    std::atomic<std::uint64_t> presents{0};  // frames the presenter showed (VulkanPresenter), since the last line
    std::atomic<std::uint64_t> present_serial{0};  // the same, never reset (LEONIDA_REFRESH_PER_FRAME)
    std::atomic<std::uint32_t> thread{0};  // the thread of the last submission (the thread sampler reports it apart)
};
inline Totals& totals() {
    static Totals t;
    return t;
}

inline std::int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

// Seconds since the process started.
inline double process_age() {
#ifdef _WIN32
    FILETIME created{}, exited{}, kernel{}, user{}, now{};
    if (!GetProcessTimes(GetCurrentProcess(), &created, &exited, &kernel, &user)) return 0;
    GetSystemTimeAsFileTime(&now);
    const auto ticks = [](const FILETIME& f) { return (std::uint64_t(f.dwHighDateTime) << 32) | f.dwLowDateTime; };
    const std::uint64_t n = ticks(now), c = ticks(created);
    return n >= c ? double(n - c) * 1e-7 : 0.0;
#else
    return 0;
#endif
}

inline constexpr std::int64_t kNsPerSecond = 1000000000LL;
inline constexpr std::int64_t kDefaultPeriodSeconds = 5;
inline constexpr std::uint64_t kMaxPeriodSeconds =
    static_cast<std::uint64_t>((std::numeric_limits<std::int64_t>::max)() / kNsPerSecond);

// Parses LEONIDA_SUBMIT_PHASES_SECONDS (non-negative decimal seconds; 0 disables the periodic line).
// Missing, empty, negative or non-numeric text uses the 5 s default; huge values saturate safely without
// signed int64 overflow.
inline std::int64_t parse_period_ns(const char* text) {
    if (!text) return kDefaultPeriodSeconds * kNsPerSecond;
    while (*text == ' ' || *text == '\t') ++text;
    if (*text < '0' || *text > '9') return kDefaultPeriodSeconds * kNsPerSecond;
    std::uint64_t seconds = 0;
    while (*text >= '0' && *text <= '9') {
        const std::uint64_t digit = static_cast<std::uint64_t>(*text - '0');
        if (seconds > (kMaxPeriodSeconds - digit) / 10)
            seconds = kMaxPeriodSeconds;
        else
            seconds = seconds * 10 + digit;
        ++text;
    }
    while (*text == ' ' || *text == '\t') ++text;
    if (*text != '\0') return kDefaultPeriodSeconds * kNsPerSecond;
    return static_cast<std::int64_t>(seconds) * kNsPerSecond;
}

inline std::int64_t period_ns() {
    static const std::int64_t value = parse_period_ns(std::getenv("LEONIDA_SUBMIT_PHASES_SECONDS"));
    return value;
}

inline void add(Phase p, std::uint64_t ns) {
    if (p >= kCount) return;
    Totals& t = totals();
    t.ns[p].fetch_add(ns, std::memory_order_relaxed);
    std::uint64_t m = t.max_ns[p].load(std::memory_order_relaxed);
    while (ns > m && !t.max_ns[p].compare_exchange_weak(m, ns, std::memory_order_relaxed)) {
    }
}

// Times its own lifetime into a phase. When the periodic line is disabled (period == 0), skips the
// steady_clock reads and atomic updates completely.
struct Scope {
    Phase phase;
    bool active = period_ns() > 0;
    std::int64_t start = active ? now_ns() : 0;
    explicit Scope(Phase p) : phase(p) {}
    ~Scope() {
        if (!active) return;
        const std::int64_t dt = now_ns() - start;
        if (dt > 0) add(phase, static_cast<std::uint64_t>(dt));
    }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;
};

// After each submission: counts it, and prints the line when the period has passed. One thread wins the line.
inline void submitted() {
    Totals& t = totals();
    t.submissions.fetch_add(1, std::memory_order_relaxed);
    t.all_submissions.fetch_add(1, std::memory_order_relaxed);
#ifdef _WIN32
    t.thread.store(GetCurrentThreadId(), std::memory_order_relaxed);
#endif
    const std::int64_t period = period_ns();
    if (period <= 0) return;
    const std::int64_t now = now_ns();
    std::int64_t last = t.last_log.load(std::memory_order_relaxed);
    if (!last) {
        t.last_log.compare_exchange_strong(last, now, std::memory_order_relaxed);
        return;
    }
    if (now - last < period || !t.last_log.compare_exchange_strong(last, now, std::memory_order_relaxed)) return;
    const double seconds = double(now - last) * 1e-9;
    const std::uint64_t n = t.submissions.exchange(0, std::memory_order_relaxed);
    const std::uint64_t frames = t.presents.exchange(0, std::memory_order_relaxed);
    double ms[kCount], max_ms[kCount], all = 0;
    for (unsigned p = 0; p < kCount; ++p) {
        const double total = double(t.ns[p].exchange(0, std::memory_order_relaxed)) * 1e-6;
        max_ms[p] = double(t.max_ns[p].exchange(0, std::memory_order_relaxed)) * 1e-6;
        ms[p] = n ? total / double(n) : 0;
        all += total;
    }
    std::fprintf(stderr,
                 "[rt] submit phases t=%.0fs: %llu submissions in %.1f s (%.2f/s, %llu in all), %.2f frames/s; ms per submission "
                 "(max): %s %.2f (%.0f) %s %.2f (%.0f) %s %.2f (%.0f) %s %.2f (%.0f) %s %.2f (%.0f) %s %.2f (%.0f); "
                 "%.0f%% of the wall time\n",
                 process_age(), static_cast<unsigned long long>(n), seconds, double(n) / seconds,
                 static_cast<unsigned long long>(t.all_submissions.load(std::memory_order_relaxed)),
                 double(frames) / seconds, name(0), ms[0],
                 max_ms[0], name(1), ms[1], max_ms[1], name(2), ms[2], max_ms[2], name(3), ms[3], max_ms[3], name(4),
                 ms[4], max_ms[4], name(5), ms[5], max_ms[5], 100.0 * all * 1e-3 / seconds);
}

}  // namespace leonida::submit_phases
