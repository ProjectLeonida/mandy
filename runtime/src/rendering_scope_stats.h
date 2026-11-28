// Opt-in, aggregate-only diagnostics for application draw rendering scopes. No resource or shader identities.
#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>

namespace leonida::scope_stats {
enum class End { pipeline, attachments, barrier, copy, clear, query, predicate, dispatch, label,
                 indirect, skipped, helper, statistics, other, list_end, count };
inline constexpr std::array end_names{"pipeline", "attachments", "barrier", "copy", "clear", "query", "predicate",
    "dispatch", "label", "indirect", "skipped", "helper", "statistics", "other", "list_end"};
static_assert(end_names.size() == static_cast<unsigned>(End::count));
inline constexpr std::array run_names{"1", "2", "3", "4-7", "8-15", "16-63", "64-255", "256+"};

struct Counts {
    std::uint64_t lists = 0, draws = 0, eligible = 0, indirect = 0, skipped = 0, scopes = 0, max_run = 0;
    std::array<std::uint64_t, run_names.size()> runs{};
    std::array<std::uint64_t, end_names.size()> ends{};

    void add(const Counts& b) {
        lists += b.lists; draws += b.draws; eligible += b.eligible; indirect += b.indirect; skipped += b.skipped;
        scopes += b.scopes;
        max_run = std::max(max_run, b.max_run);
        for (unsigned i = 0; i < runs.size(); ++i) runs[i] += b.runs[i];
        for (unsigned i = 0; i < ends.size(); ++i) ends[i] += b.ends[i];
    }

    void finish(std::uint64_t length, End reason) {
        ++scopes;
        max_run = std::max(max_run, length);
        const unsigned bucket = length <= 3 ? unsigned(length - 1) : length < 8 ? 3 : length < 16 ? 4 :
                                length < 64 ? 5 : length < 256 ? 6 : 7;
        ++runs[bucket];
        ++ends[static_cast<unsigned>(reason)];
    }

    void report(std::FILE* out = stderr) const {
        // One bounded line: at most 31 unsigned 64-bit values, with fixed, non-title labels.
        char line[2048];
        int pos = std::snprintf(line, sizeof(line),
            "[rt] rendering scopes: lists=%llu draws=%llu eligible=%llu indirect=%llu skipped=%llu scopes=%llu "
            "shared_draws=%llu max_run=%llu runs=[",
            ull(lists), ull(draws), ull(eligible), ull(indirect), ull(skipped), ull(scopes),
            ull(draws - scopes), ull(max_run));
        for (unsigned i = 0; i < runs.size(); ++i)
            pos += std::snprintf(line + pos, sizeof(line) - pos, "%s%s=%llu", i ? "," : "", run_names[i], ull(runs[i]));
        pos += std::snprintf(line + pos, sizeof(line) - pos, "] ends=[");
        for (unsigned i = 0; i < ends.size(); ++i)
            pos += std::snprintf(line + pos, sizeof(line) - pos, "%s%s=%llu", i ? "," : "", end_names[i], ull(ends[i]));
        std::fprintf(out, "%s]\n", line);
    }
private:
    static unsigned long long ull(std::uint64_t n) { return static_cast<unsigned long long>(n); }
};

inline bool enabled() {
    static const bool value = [] {
        const char* env = std::getenv("LEONIDA_RENDERING_SCOPE_STATS");
        return env && env[0] == '1' && env[1] == '\0';
    }();
    return value;
}

// Merge once per emitted list, never once per draw. A lock keeps each report internally consistent when queues
// record on different threads. Powers of two bound the log volume; normal process exit reports the final totals.
struct Totals {
    std::mutex mutex;
    Counts counts;
    std::uint64_t last_report = 0;
    void add(const Counts& part) {
        const std::lock_guard<std::mutex> lock(mutex);
        counts.add(part);
        if ((counts.lists & (counts.lists - 1)) == 0) {
            counts.report();
            last_report = counts.lists;
        }
    }
    ~Totals() { if (counts.lists != last_report) counts.report(); }
};
inline void publish(const Counts& part) {
    static Totals totals;
    totals.add(part);
}
}  // namespace leonida::scope_stats
