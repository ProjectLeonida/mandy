// Persisting the runtime's VkPipelineCache across launches (D3D-704).
//
// Two files sit beside gta6rt.exe, and they are not interchangeable:
//
//   pipeline_cache.bin         the offline corpus seed the `pipelines` build stage writes (#367). Read only.
//   pipeline_cache.session.bin what the runtime compiled last time. Written on device teardown.
//
// Keeping them apart means a run can never replace the multi-GB corpus cache with the handful of pipelines it
// happened to compile. On the next launch both are loaded and merged, so the offline entries and the previous
// session's entries are available together.
//
// This header is pure stdlib on purpose (it does not include vulkan.h): pipeline_cache_store_test builds on hosts
// where CMake has no Vulkan headers. The byte layout below is VkPipelineCacheHeaderVersionOne, whose 32 bytes are
// part of Vulkan's ABI, not something a driver chooses.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

#include "cached_pso.h"

namespace leonida::pipeline_cache_store {

// A session cache normally holds what one boot compiled (a few MB to a few hundred MB). The cap is what keeps a
// cache seeded from the multi-GB corpus file -- whose data comes back as the whole union -- from being rewritten
// at every exit. Past it the previous session file is left alone; raise the limit with
// LEONIDA_PIPELINE_CACHE_BYTES when a session really compiles that much.
inline constexpr std::size_t default_save_limit = std::size_t{256} << 20;  // 256 MiB

struct SaveLimit {
    std::size_t bytes = default_save_limit;
    bool valid = true;  // false when the text was not a byte count; bytes then stays at the default
};

// LEONIDA_PIPELINE_CACHE_BYTES: a decimal byte count, 0 turning the session file off entirely. Empty or absent
// means the default; stray text, a sign or an overflow is reported by the caller and the default is used.
inline SaveLimit parse_save_limit(const char* text) {
    if (!text || !*text) return {};
    std::size_t value = 0;
    for (const char* p = text; *p; ++p) {
        if (*p < '0' || *p > '9') return {default_save_limit, false};
        const std::size_t digit = static_cast<std::size_t>(*p - '0');
        if (value > (std::numeric_limits<std::size_t>::max() - digit) / 10) return {default_save_limit, false};
        value = value * 10 + digit;
    }
    return {value, true};
}

inline constexpr std::size_t header_bytes = 32;
inline constexpr std::uint32_t header_version_one = 1;

inline std::uint32_t read_u32(const std::uint8_t* p) {
    return std::uint32_t(p[0]) | (std::uint32_t(p[1]) << 8) | (std::uint32_t(p[2]) << 16) |
           (std::uint32_t(p[3]) << 24);
}

// True when the blob's header names this physical device and pipeline-cache UUID, which is what a driver requires
// before it will read the opaque payload. The header is little-endian, like the host.
inline bool header_matches(const void* data, std::size_t size, const cached_pso::Identity& identity) {
    if (!data || size < header_bytes) return false;
    const auto* bytes = static_cast<const std::uint8_t*>(data);
    return read_u32(bytes) == header_bytes && read_u32(bytes + 4) == header_version_one &&
           read_u32(bytes + 8) == identity.vendor && read_u32(bytes + 12) == identity.device &&
           std::memcmp(bytes + 16, identity.cache_uuid, sizeof(identity.cache_uuid)) == 0;
}

inline constexpr std::uint64_t default_checkpoint_interval_ms = 30000;
inline constexpr std::uint64_t default_checkpoint_retry_ms = 5000;
inline constexpr std::size_t checkpoint_priority = 1;  // behind priority-0 awaited builds, ahead of warm-up (SPIR-V bytes)

enum class CheckpointOutcome : std::uint8_t {
    saved,     // snapshot written; wait interval_ms before the next checkpoint
    deferred,  // yielded to a priority-0 awaited build; retry after retry_ms
    skipped,   // empty cache, over size cap, or I/O failure; wait interval_ms before retrying
};

// Coalesces startup pipeline-cache checkpoint scheduling across compiling threads.
// At most one checkpoint runs at a time; a deferred checkpoint backs off briefly (retry_ms) instead of burning the
// full cooldown interval.
class CheckpointSchedule {
public:
    explicit CheckpointSchedule(std::uint64_t initial_due_ms,
                                std::uint64_t interval_ms = default_checkpoint_interval_ms,
                                std::uint64_t retry_ms = default_checkpoint_retry_ms)
        : next_due_ms_(clamp_due(initial_due_ms)), interval_ms_(interval_ms), retry_ms_(retry_ms) {}

    bool try_claim(std::uint64_t now_ms) {
        std::uint64_t expected = next_due_ms_.load(std::memory_order_acquire);
        while (expected != in_flight && now_ms >= expected) {
            if (next_due_ms_.compare_exchange_weak(expected, in_flight, std::memory_order_acq_rel,
                                                   std::memory_order_acquire)) {
                return true;
            }
        }
        return false;
    }

    void complete(std::uint64_t now_ms, CheckpointOutcome outcome) {
        const std::uint64_t wait = outcome == CheckpointOutcome::deferred ? retry_ms_ : interval_ms_;
        next_due_ms_.store(saturate_add(now_ms, wait), std::memory_order_release);
    }

    void abort_claim(std::uint64_t now_ms) {
        next_due_ms_.store(saturate_add(now_ms, retry_ms_), std::memory_order_release);
    }

    std::uint64_t next_due_ms() const { return next_due_ms_.load(std::memory_order_acquire); }
    bool in_progress() const { return next_due_ms_.load(std::memory_order_acquire) == in_flight; }

private:
    static constexpr std::uint64_t in_flight = (std::numeric_limits<std::uint64_t>::max)();
    static constexpr std::uint64_t max_due = in_flight - 1;
    static std::uint64_t clamp_due(std::uint64_t t) { return t > max_due ? max_due : t; }
    static std::uint64_t saturate_add(std::uint64_t a, std::uint64_t b) {
        return a > max_due - b ? max_due : a + b;
    }

    std::atomic<std::uint64_t> next_due_ms_;
    std::uint64_t interval_ms_;
    std::uint64_t retry_ms_;
};

}  // namespace leonida::pipeline_cache_store
