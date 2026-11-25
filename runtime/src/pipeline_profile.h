// Optional driver-call timing. Match begin/end by id even when startup warmup and replay overlap.
#pragma once

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#ifdef _WIN32
#include <windows.h>
#endif

namespace leonida::pipeline_profile {

inline bool enabled() {
    static const bool enable = [] {
        const char* value = std::getenv("LEONIDA_PSO_PROFILE");
        return value && std::strcmp(value, "1") == 0;
    }();
    return enable;
}

inline long long unix_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

// CPU charged to the calling thread distinguishes synchronous compilation from time waiting in the driver.
inline long long thread_cpu_us() {
#ifdef _WIN32
    FILETIME creation{}, exit{}, kernel{}, user{};
    if (GetThreadTimes(GetCurrentThread(), &creation, &exit, &kernel, &user)) {
        const auto ticks = [](FILETIME t) {
            return (static_cast<unsigned long long>(t.dwHighDateTime) << 32) | t.dwLowDateTime;
        };
        return static_cast<long long>((ticks(kernel) + ticks(user)) / 10);
    }
#endif
    return -1;
}

// Title pipeline builds that took long enough to have compiled rather than come from the pipeline cache: a cache hit
// takes a few milliseconds, also while the title keeps every core busy. Counted whether or not profiling is on; the
// session cache save compares the count with how much the cache grew (pipeline_cache_store::GrowthWatch).
inline constexpr long long compiled_build_us = 200000;  // 200 ms
inline std::atomic<unsigned long long>& compiled_builds() {
    static std::atomic<unsigned long long> count{0};
    return count;
}

class DriverCall {
public:
    DriverCall(const char* stage, const char* origin, const char* program, const char* fragment = "-",
               unsigned unbound = 0)
        : start_(std::chrono::steady_clock::now()) {
        if (!enabled()) return;
        static std::atomic<unsigned long long> sequence{0};
        id_ = sequence.fetch_add(1, std::memory_order_relaxed) + 1;
        cpu_start_ = thread_cpu_us();
        std::fprintf(stderr, "[rt] pso-driver begin id=%llu unix_ms=%lld stage=%s origin=%s program=%s ps=%s mask=%u\n",
                     id_, unix_ms(), stage, origin, program, fragment, unbound);
    }

    void finish(int result) const {
        const auto us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - start_).count();
        if (result == 0 && us >= compiled_build_us) compiled_builds().fetch_add(1, std::memory_order_relaxed);
        if (!id_) return;
        const long long cpu = thread_cpu_us();
        std::fprintf(stderr, "[rt] pso-driver end id=%llu unix_ms=%lld us=%lld cpu_us=%lld result=%d\n",
                     id_, unix_ms(), static_cast<long long>(us), cpu >= 0 && cpu_start_ >= 0 ? cpu - cpu_start_ : -1,
                     result);
    }
private:
    unsigned long long id_ = 0;
    long long cpu_start_ = -1;
    std::chrono::steady_clock::time_point start_{};
};

}  // namespace leonida::pipeline_profile
