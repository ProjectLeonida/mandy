#include <windows.h>

#include <cstdio>
#include <cwchar>

#include "shims.h"

namespace {

int g_fail = 0;

#define CHECK(cond, ...)                                                       \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::fprintf(stderr, "FAIL (%d): ", __LINE__);                     \
            std::fprintf(stderr, __VA_ARGS__);                                 \
            std::fprintf(stderr, "\n");                                        \
            ++g_fail;                                                          \
        }                                                                      \
    } while (0)

}  // namespace

int main() {
    using GetDateFn = int(WINAPI*)(LPCWSTR, DWORD, const SYSTEMTIME*, LPCWSTR, LPWSTR, int, LPCWSTR);
    using GetTimeFn = int(WINAPI*)(LPCWSTR, DWORD, const SYSTEMTIME*, LPCWSTR, LPWSTR, int);

    auto* date_api =
        reinterpret_cast<GetDateFn>(rt::shims::find("api-ms-win-core-datetime-l1-1-1.dll", "GetDateFormatEx"));
    auto* date_k32 = reinterpret_cast<GetDateFn>(rt::shims::find("kernel32.dll", "GetDateFormatEx"));
    auto* date_kb = reinterpret_cast<GetDateFn>(rt::shims::find("kernelbase.dll", "GetDateFormatEx"));
    auto* time_api =
        reinterpret_cast<GetTimeFn>(rt::shims::find("api-ms-win-core-datetime-l1-1-1.dll", "GetTimeFormatEx"));
    auto* time_k32 = reinterpret_cast<GetTimeFn>(rt::shims::find("kernel32.dll", "GetTimeFormatEx"));

    CHECK(date_api != nullptr && date_k32 == date_api && date_kb == date_api,
          "GetDateFormatEx shim missing on expected DLL keys");
    CHECK(time_api != nullptr && time_k32 == time_api,
          "GetTimeFormatEx shim missing on expected DLL keys");
    CHECK(!rt::shims::owns_dll("api-ms-win-core-datetime-l1-1-1.dll"),
          "owns_dll must remain false for api-ms datetime contract");
    CHECK(!rt::shims::owns_dll("kernel32.dll"), "owns_dll must remain false for kernel32.dll");
    CHECK(!rt::shims::owns_dll("kernelbase.dll"), "owns_dll must remain false for kernelbase.dll");
    if (g_fail) return 1;

    SYSTEMTIME st{};
    st.wYear = 2026;
    st.wMonth = 11;
    st.wDay = 24;
    st.wDayOfWeek = 2;  // Tuesday
    st.wHour = 14;
    st.wMinute = 5;
    st.wSecond = 9;

    wchar_t buf[128]{};

    // 1. User-default / en-US formatting with literal quotes (RFC-style & ISO date/time pictures).
    SetEnvironmentVariableW(L"LEONIDA_LOCALE", L"en-US");
    int n = date_api(L"", 0, &st, L"ddd',' dd MMM yyyy", buf, static_cast<int>(sizeof(buf) / sizeof(buf[0])), nullptr);
    CHECK(n > 0 && std::wcscmp(buf, L"Tue, 24 Nov 2026") == 0,
          "GetDateFormatEx RFC format mismatch");

    n = date_api(nullptr, 0, &st, L"yyyy-MM-dd", buf, static_cast<int>(sizeof(buf) / sizeof(buf[0])), nullptr);
    CHECK(n == 11 && std::wcscmp(buf, L"2026-11-24") == 0,
          "GetDateFormatEx ISO format mismatch (n=%d)", n);

    n = time_api(L"", 0, &st, L"HH':'mm':'ss", buf, static_cast<int>(sizeof(buf) / sizeof(buf[0])));
    CHECK(n == 9 && std::wcscmp(buf, L"14:05:09") == 0,
          "GetTimeFormatEx 24h format mismatch (n=%d)", n);

    // 2. Size query (cch == 0) & insufficient buffer handling
    int req = date_api(L"en-US", 0, &st, L"yyyy-MM-dd", nullptr, 0, nullptr);
    CHECK(req == 11, "GetDateFormatEx length query expected 11, got %d", req);
    SetLastError(0);
    CHECK(date_api(L"en-US", 0, &st, L"yyyy-MM-dd", buf, 4, nullptr) == 0 &&
              GetLastError() == ERROR_INSUFFICIENT_BUFFER,
          "GetDateFormatEx undersized buffer must fail with ERROR_INSUFFICIENT_BUFFER");

    // 3. Invalid inputs must fail with ERROR_INVALID_PARAMETER
    SetLastError(0);
    CHECK(date_api(L"", 0, &st, L"yyyy-MM-dd", nullptr, 16, nullptr) == 0 &&
              GetLastError() == ERROR_INVALID_PARAMETER,
          "null buffer with nonzero count must fail");
    SYSTEMTIME bad_date = st;
    bad_date.wMonth = 13;
    SetLastError(0);
    CHECK(date_api(L"", 0, &bad_date, L"yyyy-MM-dd", buf, 32, nullptr) == 0 &&
              GetLastError() == ERROR_INVALID_PARAMETER,
          "invalid month must fail with ERROR_INVALID_PARAMETER");

    SYSTEMTIME bad_time = st;
    bad_time.wHour = 25;
    SetLastError(0);
    CHECK(time_api(L"", 0, &bad_time, L"HH:mm:ss", buf, 32) == 0 &&
              GetLastError() == ERROR_INVALID_PARAMETER,
          "invalid hour must fail with ERROR_INVALID_PARAMETER");

    // 4. Unknown locale recovers cleanly instead of returning 0
    n = date_api(L"xx-LEONIDA-INVALID", 0, &st, L"yyyy-MM-dd", buf, static_cast<int>(sizeof(buf) / sizeof(buf[0])),
                 nullptr);
    CHECK(n == 11 && std::wcscmp(buf, L"2026-11-24") == 0,
          "unknown locale fallback failed (n=%d)", n);
    n = time_api(L"xx-LEONIDA-INVALID", 0, &st, L"HH':'mm':'ss", buf, static_cast<int>(sizeof(buf) / sizeof(buf[0])));
    CHECK(n == 9 && std::wcscmp(buf, L"14:05:09") == 0,
          "unknown locale time fallback failed (n=%d)", n);

    // 5. Null SYSTEMTIME uses local wall clock cleanly
    CHECK(date_api(nullptr, 0, nullptr, L"yyyy-MM-dd", buf, static_cast<int>(sizeof(buf) / sizeof(buf[0])), nullptr) > 0,
          "null lpDate current-time date format failed");
    CHECK(time_api(nullptr, 0, nullptr, L"HH':'mm':'ss", buf, static_cast<int>(sizeof(buf) / sizeof(buf[0]))) > 0,
          "null lpTime current-time time format failed");

    SetEnvironmentVariableW(L"LEONIDA_LOCALE", nullptr);
    std::printf("datetime_test: %s (%d failures)\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
