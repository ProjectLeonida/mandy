#pragma once

// When the title-texture upload ring (shims/d3d12_title_image_upload.h) gives an idle slot's buffer back. Each of its
// four slots grows to the largest whole-image upload that went through it and kept that capacity for the rest of the
// run, in host-visible memory (system RAM on a discrete GPU): after a burst of large textures, such as a cutscene's,
// up to four of the largest stayed allocated. A slot is now released once it has been idle for a while and its last
// transfer has completed; a small one is kept, so steady small uploads do not allocate again each time.
//
//   LEONIDA_TITLE_UPLOAD_IDLE_MS  how long a slot stays unused before it is released (default 2000; 0: never, the
//                                 behaviour before this)
//   LEONIDA_TITLE_UPLOAD_KEEP_MB  a slot this size or smaller is never released (default 4)
//
// Pure standard library, so the rule is tested without a GPU (title_upload_trim_test).

#include <cstdint>
#include <cstdlib>

namespace leonida::title_upload_trim {

struct Settings {
    std::uint64_t idle_ns = 2000ull * 1000 * 1000;
    std::uint64_t keep_bytes = 4ull << 20;
    bool enabled() const { return idle_ns != 0; }
};

// A whole decimal number, or `fallback` for a missing, empty or malformed value.
inline std::uint64_t parse_u64(const char* text, std::uint64_t fallback) {
    if (!text || !*text) return fallback;
    std::uint64_t value = 0;
    for (const char* p = text; *p; ++p) {
        if (*p < '0' || *p > '9') return fallback;
        const std::uint64_t digit = std::uint64_t(*p - '0');
        if (value > (UINT64_MAX - digit) / 10) return fallback;
        value = value * 10 + digit;
    }
    return value;
}

// The two variables' values (null when unset).
inline Settings settings_from(const char* idle_ms, const char* keep_mb) {
    Settings s;
    const std::uint64_t ms = parse_u64(idle_ms, 2000);
    s.idle_ns = ms > UINT64_MAX / 1000000 ? UINT64_MAX : ms * 1000000;
    const std::uint64_t mb = parse_u64(keep_mb, 4);
    s.keep_bytes = mb > (UINT64_MAX >> 20) ? UINT64_MAX : mb << 20;
    return s;
}

// Whether a slot's buffer is released now. `capacity`: its bytes (0: nothing to release); `last_used_ns`: when a
// write last used it; `transfer_done`: its last transfer has completed (known without waiting).
inline bool release(std::uint64_t capacity, std::uint64_t last_used_ns, std::uint64_t now_ns, bool transfer_done,
                    const Settings& s) {
    if (!s.enabled() || !capacity || capacity <= s.keep_bytes || !transfer_done) return false;
    return now_ns >= last_used_ns && now_ns - last_used_ns >= s.idle_ns;
}

}  // namespace leonida::title_upload_trim
