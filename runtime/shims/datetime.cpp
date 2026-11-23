// Locale-aware date and time formatting shims (api-ms-win-core-datetime-l1-1-1.dll, kernel32.dll, kernelbase.dll).
//
// game_scarlett_master.exe calls GetDateFormatEx and GetTimeFormatEx with L"" (user default locale) and explicit
// format pictures, throwing a C++ exception on any 0 return. This shim:
//   - honors LEONIDA_LOCALE when lpLocaleName is null or empty (matching GetUserDefaultLocaleName);
//   - delegates to host NLS first;
//   - falls back to LOCALE_NAME_INVARIANT / "en-US" and a built-in Gregorian format-picture renderer when a sandbox
//     or minimal host lacks locale tables, while preserving strict Win32 validation (ERROR_INVALID_PARAMETER /
//     ERROR_INSUFFICIENT_BUFFER).

#include <windows.h>

#include <cstdint>
#include <cstring>
#include <string>

#include "shims.h"

namespace rt::datetime {
namespace {

const wchar_t* const kShortDays[7] = {L"Sun", L"Mon", L"Tue", L"Wed", L"Thu", L"Fri", L"Sat"};
const wchar_t* const kLongDays[7] = {L"Sunday",   L"Monday", L"Tuesday", L"Wednesday",
                                      L"Thursday", L"Friday", L"Saturday"};
const wchar_t* const kShortMonths[12] = {L"Jan", L"Feb", L"Mar", L"Apr", L"May", L"Jun",
                                         L"Jul", L"Aug", L"Sep", L"Oct", L"Nov", L"Dec"};
const wchar_t* const kLongMonths[12] = {L"January",  L"February", L"March",    L"April",
                                        L"May",      L"June",     L"July",     L"August",
                                        L"September", L"October", L"November", L"December"};

bool is_valid_date(const SYSTEMTIME& st) {
    if (st.wYear < 1601 || st.wYear > 9999) return false;
    if (st.wMonth < 1 || st.wMonth > 12) return false;
    if (st.wDay < 1 || st.wDay > 31) return false;
    return true;
}

bool is_valid_time(const SYSTEMTIME& st) {
    return st.wHour <= 23 && st.wMinute <= 59 && st.wSecond <= 59 && st.wMilliseconds <= 999;
}

SYSTEMTIME resolve_system_time(const SYSTEMTIME* lpTime, bool want_date, bool& valid_out) {
    if (!lpTime) {
        SYSTEMTIME now{};
        GetLocalTime(&now);
        valid_out = true;
        return now;
    }
    valid_out = want_date ? is_valid_date(*lpTime) : is_valid_time(*lpTime);
    return *lpTime;
}

int compute_weekday(const SYSTEMTIME& st) {
    if (st.wDayOfWeek <= 6) return st.wDayOfWeek;
    // Sakamoto's algorithm for Gregorian calendar weekday (0 = Sunday).
    static const int offsets[12] = {0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4};
    int y = static_cast<int>(st.wYear);
    if (st.wMonth < 3) --y;
    return (y + y / 4 - y / 100 + y / 400 + offsets[st.wMonth - 1] + static_cast<int>(st.wDay)) % 7;
}

void append_uint(std::wstring& out, unsigned val, int min_digits) {
    wchar_t tmp[16]{};
    int len = 0;
    do {
        tmp[len++] = static_cast<wchar_t>(L'0' + (val % 10u));
        val /= 10u;
    } while (val != 0 && len < 16);
    for (int i = len; i < min_digits; ++i) out.push_back(L'0');
    for (int i = len - 1; i >= 0; --i) out.push_back(tmp[i]);
}

std::wstring format_date_fallback(const SYSTEMTIME& st, const wchar_t* picture) {
    const wchar_t* fmt = (picture && *picture) ? picture : L"yyyy-MM-dd";
    const int dow = compute_weekday(st);
    std::wstring out;
    for (size_t i = 0; fmt[i] != L'\0';) {
        const wchar_t c = fmt[i];
        if (c == L'\'') {
            ++i;
            if (fmt[i] == L'\'') {
                out.push_back(L'\'');
                ++i;
                continue;
            }
            while (fmt[i] != L'\0' && fmt[i] != L'\'') {
                out.push_back(fmt[i++]);
            }
            if (fmt[i] == L'\'') ++i;
            continue;
        }
        size_t run = 1;
        while (fmt[i + run] == c) ++run;

        if (c == L'd') {
            if (run == 1) append_uint(out, st.wDay, 1);
            else if (run == 2) append_uint(out, st.wDay, 2);
            else if (run == 3) out.append(kShortDays[dow]);
            else out.append(kLongDays[dow]);
            i += run;
        } else if (c == L'M') {
            if (run == 1) append_uint(out, st.wMonth, 1);
            else if (run == 2) append_uint(out, st.wMonth, 2);
            else if (run == 3) out.append(kShortMonths[st.wMonth - 1]);
            else out.append(kLongMonths[st.wMonth - 1]);
            i += run;
        } else if (c == L'y') {
            if (run <= 2) append_uint(out, st.wYear % 100u, static_cast<int>(run));
            else append_uint(out, st.wYear, static_cast<int>(run));
            i += run;
        } else {
            out.push_back(c);
            ++i;
        }
    }
    return out;
}

std::wstring format_time_fallback(const SYSTEMTIME& st, const wchar_t* picture) {
    const wchar_t* fmt = (picture && *picture) ? picture : L"HH:mm:ss";
    const unsigned hour12 = (st.wHour % 12u == 0) ? 12u : (st.wHour % 12u);
    const wchar_t* ampm = (st.wHour < 12) ? L"AM" : L"PM";
    std::wstring out;
    for (size_t i = 0; fmt[i] != L'\0';) {
        const wchar_t c = fmt[i];
        if (c == L'\'') {
            ++i;
            if (fmt[i] == L'\'') {
                out.push_back(L'\'');
                ++i;
                continue;
            }
            while (fmt[i] != L'\0' && fmt[i] != L'\'') {
                out.push_back(fmt[i++]);
            }
            if (fmt[i] == L'\'') ++i;
            continue;
        }
        size_t run = 1;
        while (fmt[i + run] == c) ++run;

        if (c == L'H') {
            append_uint(out, st.wHour, run >= 2 ? 2 : 1);
            i += run;
        } else if (c == L'h') {
            append_uint(out, hour12, run >= 2 ? 2 : 1);
            i += run;
        } else if (c == L'm') {
            append_uint(out, st.wMinute, run >= 2 ? 2 : 1);
            i += run;
        } else if (c == L's') {
            append_uint(out, st.wSecond, run >= 2 ? 2 : 1);
            i += run;
        } else if (c == L't') {
            if (run == 1) out.push_back(ampm[0]);
            else out.append(ampm);
            i += run;
        } else {
            out.push_back(c);
            ++i;
        }
    }
    return out;
}

int write_wide_result(const std::wstring& formatted, LPWSTR out_buf, int cch_out) {
    const int needed = static_cast<int>(formatted.size()) + 1;
    if (cch_out == 0) return needed;
    if (!out_buf || cch_out < 0) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return 0;
    }
    if (cch_out < needed) {
        SetLastError(ERROR_INSUFFICIENT_BUFFER);
        return 0;
    }
    std::memcpy(out_buf, formatted.data(), formatted.size() * sizeof(wchar_t));
    out_buf[formatted.size()] = L'\0';
    return needed;
}

std::wstring select_locale(LPCWSTR lpLocaleName, bool& has_override) {
    has_override = false;
    if (lpLocaleName && *lpLocaleName) return std::wstring(lpLocaleName);
    wchar_t env[128]{};
    const DWORD n = GetEnvironmentVariableW(L"LEONIDA_LOCALE", env, static_cast<DWORD>(sizeof(env) / sizeof(env[0])));
    if (n > 0 && n < (sizeof(env) / sizeof(env[0]))) {
        has_override = true;
        return std::wstring(env, n);
    }
    return {};
}

}  // namespace

int WINAPI shim_GetDateFormatEx(LPCWSTR lpLocaleName, DWORD dwFlags, const SYSTEMTIME* lpDate, LPCWSTR lpFormat,
                                 LPWSTR lpDateStr, int cchDate, LPCWSTR lpCalendar) {
    if (cchDate < 0 || (cchDate > 0 && !lpDateStr)) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return 0;
    }
    bool valid_time = true;
    const SYSTEMTIME st = resolve_system_time(lpDate, true, valid_time);
    if (!valid_time) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return 0;
    }

    bool has_override = false;
    const std::wstring loc = select_locale(lpLocaleName, has_override);
    const LPCWSTR effective_loc = (!loc.empty()) ? loc.c_str() : lpLocaleName;

    int res = ::GetDateFormatEx(effective_loc, dwFlags, &st, lpFormat, lpDateStr, cchDate, lpCalendar);
    if (res > 0) return res;
    if (GetLastError() == ERROR_INSUFFICIENT_BUFFER) return 0;

    res = ::GetDateFormatEx(LOCALE_NAME_INVARIANT, dwFlags, &st, lpFormat, lpDateStr, cchDate, nullptr);
    if (res > 0) return res;
    if (GetLastError() == ERROR_INSUFFICIENT_BUFFER) return 0;

    return write_wide_result(format_date_fallback(st, lpFormat), lpDateStr, cchDate);
}

int WINAPI shim_GetTimeFormatEx(LPCWSTR lpLocaleName, DWORD dwFlags, const SYSTEMTIME* lpTime, LPCWSTR lpFormat,
                                 LPWSTR lpTimeStr, int cchTime) {
    if (cchTime < 0 || (cchTime > 0 && !lpTimeStr)) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return 0;
    }
    bool valid_time = true;
    const SYSTEMTIME st = resolve_system_time(lpTime, false, valid_time);
    if (!valid_time) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return 0;
    }

    bool has_override = false;
    const std::wstring loc = select_locale(lpLocaleName, has_override);
    const LPCWSTR effective_loc = (!loc.empty()) ? loc.c_str() : lpLocaleName;

    int res = ::GetTimeFormatEx(effective_loc, dwFlags, &st, lpFormat, lpTimeStr, cchTime);
    if (res > 0) return res;
    if (GetLastError() == ERROR_INSUFFICIENT_BUFFER) return 0;

    res = ::GetTimeFormatEx(LOCALE_NAME_INVARIANT, dwFlags, &st, lpFormat, lpTimeStr, cchTime);
    if (res > 0) return res;
    if (GetLastError() == ERROR_INSUFFICIENT_BUFFER) return 0;

    return write_wide_result(format_time_fallback(st, lpFormat), lpTimeStr, cchTime);
}

}  // namespace rt::datetime

#define DATETIME_API(dll)                                                                     \
    {dll, "GetDateFormatEx", reinterpret_cast<void*>(&::rt::datetime::shim_GetDateFormatEx)}, \
        {dll, "GetTimeFormatEx", reinterpret_cast<void*>(&::rt::datetime::shim_GetTimeFormatEx)}

SHIM_TABLE(datetime,
    DATETIME_API("kernel32.dll"),
    DATETIME_API("kernelbase.dll"),
    DATETIME_API("api-ms-win-core-datetime-l1-1-1.dll"),
);
