// Process-local policy for this shim's ordinary indirect graphics recorder, not a device feature.
#pragma once

#include <windows.h>
#include <cwchar>
#include <cstdio>

namespace rt::d3d12 {

// Evaluates policy from a given environment variable string without mutating process state or caching.
inline bool parse_indirect_draws_disabled(const wchar_t* value, std::size_t len) noexcept {
    if (!value || len != 3) return false;
    return std::wmemcmp(value, L"off", 3) == 0;
}

inline bool is_indirect_draws_raw_obsolete(const wchar_t* value, std::size_t len) noexcept {
    if (!value || len != 3) return false;
    return std::wmemcmp(value, L"raw", 3) == 0;
}

inline bool indirect_draws_disabled() {
    static const bool disabled = [] {
        wchar_t value[4] = {};
        // GetEnvironmentVariableW returns the required size (including NUL) on overflow.
        const DWORD len = GetEnvironmentVariableW(L"LEONIDA_INDIRECT_DRAWS", value, 4);
        if (len != 3) return false;
        if (is_indirect_draws_raw_obsolete(value, 3))
            std::fprintf(stderr, "[rt] LEONIDA_INDIRECT_DRAWS=raw is obsolete; using guarded stateful draws\n");
        return parse_indirect_draws_disabled(value, 3);
    }();
    return disabled;
}
}  // namespace rt::d3d12
