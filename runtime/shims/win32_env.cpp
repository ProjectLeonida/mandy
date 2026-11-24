// Win32 localization, process priority-boost, and named-pipe shims for mapped executables.
//
// game_scarlett_master.exe imports GetUserDefaultGeoName (api-ms-win-core-localization-l1-2-3.dll),
// GetUserDefaultLocaleName / LocaleNameToLCID (api-ms-win-core-localization-l1-2-0.dll), SetProcessPriorityBoost
// (api-ms-win-core-processthreads-l1-1-2.dll), and WaitNamedPipeW / SetNamedPipeHandleState
// (api-ms-win-core-namedpipe-l1-1-0.dll). Bound straight to the host:
//   - SetProcessPriorityBoost(GetCurrentProcess(), ...) mutates the host gta6rt scheduler priority-boost policy;
//   - GetUserDefaultGeoName (Win10 1709+) can be absent on older hosts or Wine;
//   - GetUserDefaultLocaleName and LocaleNameToLCID(nullptr, ...) reflect arbitrary host locales with no override;
//   - WaitNamedPipeW(nullptr, ...) faults on Windows 11, while WaitNamedPipeA(nullptr, ...) fails with
//     ERROR_BAD_PATHNAME and SetNamedPipeHandleState(nullptr, ...) fails with ERROR_INVALID_HANDLE.
#include <windows.h>

#include <atomic>
#include <cctype>
#include <cstring>
#include <cwchar>
#include <string>

#include "shims.h"

namespace rt::win32_env {
namespace {

using GetUserDefaultGeoNameFn = int(WINAPI*)(LPWSTR, int);

INIT_ONCE g_init_once = INIT_ONCE_STATIC_INIT;
GetUserDefaultGeoNameFn g_host_get_geo_name = nullptr;
bool g_has_geo_override = false;
wchar_t g_geo_override[8]{};
bool g_has_locale_override = false;
wchar_t g_locale_override[LOCALE_NAME_MAX_LENGTH]{};
std::atomic<bool> g_guest_priority_boost_disabled{false};

bool is_valid_geo_code(const char* text, size_t len) {
    if (len == 2) {
        return std::isalpha(static_cast<unsigned char>(text[0])) &&
               std::isalpha(static_cast<unsigned char>(text[1]));
    }
    if (len == 3) {
        return std::isdigit(static_cast<unsigned char>(text[0])) &&
               std::isdigit(static_cast<unsigned char>(text[1])) &&
               std::isdigit(static_cast<unsigned char>(text[2]));
    }
    return false;
}

bool is_valid_locale_tag(const char* text, size_t len) {
    if (len == 0 || len >= LOCALE_NAME_MAX_LENGTH) return false;
    for (size_t i = 0; i < len; ++i) {
        const unsigned char c = static_cast<unsigned char>(text[i]);
        if (!std::isalnum(c) && c != '-' && c != '_') return false;
    }
    return true;
}

BOOL CALLBACK init_env_once(PINIT_ONCE, PVOID, PVOID*) {
    if (const HMODULE kb = GetModuleHandleW(L"kernelbase.dll")) {
        g_host_get_geo_name = reinterpret_cast<GetUserDefaultGeoNameFn>(GetProcAddress(kb, "GetUserDefaultGeoName"));
    }
    if (!g_host_get_geo_name) {
        if (const HMODULE k32 = GetModuleHandleW(L"kernel32.dll")) {
            g_host_get_geo_name =
                reinterpret_cast<GetUserDefaultGeoNameFn>(GetProcAddress(k32, "GetUserDefaultGeoName"));
        }
    }

    char geo_buf[16]{};
    const DWORD geo_len = GetEnvironmentVariableA("LEONIDA_GEO", geo_buf, static_cast<DWORD>(sizeof(geo_buf)));
    if (geo_len > 0 && geo_len < sizeof(geo_buf) && is_valid_geo_code(geo_buf, geo_len)) {
        for (DWORD i = 0; i < geo_len; ++i) {
            const char c = geo_buf[i];
            g_geo_override[i] = static_cast<wchar_t>(
                (c >= 'a' && c <= 'z') ? (c - 32) : c);
        }
        g_geo_override[geo_len] = L'\0';
        g_has_geo_override = true;
    }

    char loc_buf[LOCALE_NAME_MAX_LENGTH]{};
    const DWORD loc_len = GetEnvironmentVariableA("LEONIDA_LOCALE", loc_buf, static_cast<DWORD>(sizeof(loc_buf)));
    if (loc_len > 0 && loc_len < sizeof(loc_buf) && is_valid_locale_tag(loc_buf, loc_len)) {
        for (DWORD i = 0; i < loc_len; ++i) g_locale_override[i] = static_cast<wchar_t>(loc_buf[i]);
        g_locale_override[loc_len] = L'\0';
        g_has_locale_override = true;
    }

    BOOL host_disabled = FALSE;
    if (::GetProcessPriorityBoost(GetCurrentProcess(), &host_disabled)) {
        g_guest_priority_boost_disabled.store(host_disabled != FALSE, std::memory_order_relaxed);
    }
    return TRUE;
}

void ensure_initialized() { InitOnceExecuteOnce(&g_init_once, init_env_once, nullptr, nullptr); }

int write_wide_zstring(const wchar_t* src, LPWSTR dst, int cap, bool allow_zero_cap_query) {
    if (cap < 0 || (!dst && cap > 0)) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return 0;
    }
    const int needed = static_cast<int>(std::wcslen(src) + 1);
    if (!dst) {
        if (allow_zero_cap_query) return needed;
        SetLastError(ERROR_INSUFFICIENT_BUFFER);
        return 0;
    }
    if (cap < needed) {
        SetLastError(ERROR_INSUFFICIENT_BUFFER);
        return 0;
    }
    std::wmemcpy(dst, src, static_cast<size_t>(needed));
    return needed;
}

int WINAPI get_user_default_geo_name(LPWSTR geo_name, int geo_name_count) {
    ensure_initialized();
    if (geo_name_count < 0 || (!geo_name && geo_name_count > 0)) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return 0;
    }
    if (g_has_geo_override) {
        return write_wide_zstring(g_geo_override, geo_name, geo_name_count, /*allow_zero_cap_query=*/true);
    }
    if (g_host_get_geo_name) {
        const int r = g_host_get_geo_name(geo_name, geo_name_count);
        if (r > 0 || GetLastError() == ERROR_INSUFFICIENT_BUFFER) return r;
    }
    return write_wide_zstring(L"US", geo_name, geo_name_count, /*allow_zero_cap_query=*/true);
}

int WINAPI get_user_default_locale_name(LPWSTR locale_name, int locale_name_count) {
    ensure_initialized();
    if (locale_name_count < 0 || (!locale_name && locale_name_count > 0)) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return 0;
    }
    if (g_has_locale_override) {
        return write_wide_zstring(g_locale_override, locale_name, locale_name_count, /*allow_zero_cap_query=*/false);
    }
    if (!locale_name) {
        SetLastError(ERROR_INSUFFICIENT_BUFFER);
        return 0;
    }
    const int r = ::GetUserDefaultLocaleName(locale_name, locale_name_count);
    if (r > 0 || GetLastError() == ERROR_INSUFFICIENT_BUFFER || GetLastError() == ERROR_INVALID_PARAMETER) return r;
    return write_wide_zstring(L"en-US", locale_name, locale_name_count, /*allow_zero_cap_query=*/false);
}

LCID WINAPI locale_name_to_lcid(LPCWSTR name, DWORD flags) {
    ensure_initialized();
    if (!name && g_has_locale_override) {
        return ::LocaleNameToLCID(g_locale_override, flags);
    }
    return ::LocaleNameToLCID(name, flags);
}

bool is_current_process_handle(HANDLE process) {
    if (!process) return false;
    if (process == GetCurrentProcess()) return true;
    return GetProcessId(process) == GetCurrentProcessId();
}

BOOL WINAPI set_process_priority_boost(HANDLE process, BOOL disable_priority_boost) {
    ensure_initialized();
    if (!process) {
        SetLastError(ERROR_INVALID_HANDLE);
        return FALSE;
    }
    if (is_current_process_handle(process)) {
        g_guest_priority_boost_disabled.store(disable_priority_boost != FALSE, std::memory_order_relaxed);
        return TRUE;
    }
    return ::SetProcessPriorityBoost(process, disable_priority_boost);
}

BOOL WINAPI get_process_priority_boost(HANDLE process, PBOOL disable_priority_boost) {
    ensure_initialized();
    if (!process) {
        SetLastError(ERROR_INVALID_HANDLE);
        return FALSE;
    }
    if (!disable_priority_boost) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return FALSE;
    }
    if (is_current_process_handle(process)) {
        *disable_priority_boost = g_guest_priority_boost_disabled.load(std::memory_order_relaxed) ? TRUE : FALSE;
        return TRUE;
    }
    return ::GetProcessPriorityBoost(process, disable_priority_boost);
}

BOOL WINAPI wait_named_pipe_w(LPCWSTR named_pipe_name, DWORD timeout) {
    if (!named_pipe_name) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return FALSE;
    }
    return ::WaitNamedPipeW(named_pipe_name, timeout);
}

BOOL WINAPI wait_named_pipe_a(LPCSTR named_pipe_name, DWORD timeout) {
    if (!named_pipe_name) {
        SetLastError(ERROR_BAD_PATHNAME);
        return FALSE;
    }
    return ::WaitNamedPipeA(named_pipe_name, timeout);
}

BOOL WINAPI set_named_pipe_handle_state(HANDLE named_pipe, LPDWORD mode, LPDWORD max_collection_count,
                                        LPDWORD collect_data_timeout) {
    if (!named_pipe || named_pipe == INVALID_HANDLE_VALUE) {
        SetLastError(ERROR_INVALID_HANDLE);
        return FALSE;
    }
    return ::SetNamedPipeHandleState(named_pipe, mode, max_collection_count, collect_data_timeout);
}

}  // namespace
}  // namespace rt::win32_env

namespace {

using namespace rt::win32_env;

#define LOC_GEO_API(dll) {dll, "GetUserDefaultGeoName", reinterpret_cast<void*>(&get_user_default_geo_name)}

#define LOC_LOCALE_API(dll)                                                                          \
    {dll, "GetUserDefaultLocaleName", reinterpret_cast<void*>(&get_user_default_locale_name)},       \
        {dll, "LocaleNameToLCID", reinterpret_cast<void*>(&locale_name_to_lcid)}

#define PRIORITY_BOOST_API(dll)                                                                      \
    {dll, "SetProcessPriorityBoost", reinterpret_cast<void*>(&set_process_priority_boost)},          \
        {dll, "GetProcessPriorityBoost", reinterpret_cast<void*>(&get_process_priority_boost)}

#define NAMED_PIPE_API(dll)                                                                          \
    {dll, "WaitNamedPipeW", reinterpret_cast<void*>(&wait_named_pipe_w)},                            \
        {dll, "WaitNamedPipeA", reinterpret_cast<void*>(&wait_named_pipe_a)},                        \
        {dll, "SetNamedPipeHandleState", reinterpret_cast<void*>(&set_named_pipe_handle_state)}

}  // namespace

SHIM_TABLE(win32_env,
    LOC_GEO_API("kernel32.dll"),
    LOC_GEO_API("kernelbase.dll"),
    LOC_GEO_API("api-ms-win-core-localization-l1-2-3.dll"),
    LOC_LOCALE_API("kernel32.dll"),
    LOC_LOCALE_API("kernelbase.dll"),
    LOC_LOCALE_API("api-ms-win-core-localization-l1-2-0.dll"),
    LOC_LOCALE_API("api-ms-win-core-localization-l1-2-1.dll"),
    LOC_LOCALE_API("api-ms-win-core-localization-l1-2-2.dll"),
    LOC_LOCALE_API("api-ms-win-core-localization-l1-2-3.dll"),
    PRIORITY_BOOST_API("kernel32.dll"),
    PRIORITY_BOOST_API("kernelbase.dll"),
    PRIORITY_BOOST_API("api-ms-win-core-processthreads-l1-1-0.dll"),
    PRIORITY_BOOST_API("api-ms-win-core-processthreads-l1-1-1.dll"),
    PRIORITY_BOOST_API("api-ms-win-core-processthreads-l1-1-2.dll"),
    NAMED_PIPE_API("kernel32.dll"),
    NAMED_PIPE_API("kernelbase.dll"),
    NAMED_PIPE_API("api-ms-win-core-namedpipe-l1-1-0.dll"),
);
