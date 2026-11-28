// Tests for runtime/shims/win32_env.cpp through rt::shims::find.
#include <windows.h>

#include <cstdio>
#include <cstring>
#include <cwchar>

#include "shims.h"

namespace {

using GetGeoFn = int(WINAPI*)(LPWSTR, int);
using GetLocaleFn = int(WINAPI*)(LPWSTR, int);
using LocaleToLcidFn = LCID(WINAPI*)(LPCWSTR, DWORD);
using SetBoostFn = BOOL(WINAPI*)(HANDLE, BOOL);
using GetBoostFn = BOOL(WINAPI*)(HANDLE, PBOOL);
using WaitPipeWFn = BOOL(WINAPI*)(LPCWSTR, DWORD);
using WaitPipeAFn = BOOL(WINAPI*)(LPCSTR, DWORD);
using SetPipeStateFn = BOOL(WINAPI*)(HANDLE, LPDWORD, LPDWORD, LPDWORD);

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const char* what) {
    ++g_checks;
    std::printf("%s %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition) ++g_failures;
}

template <typename Fn>
Fn resolve(const char* dll, const char* name) {
    return reinterpret_cast<Fn>(rt::shims::find(dll, name));
}

void test_registry_coverage() {
    check(resolve<GetGeoFn>("api-ms-win-core-localization-l1-2-3.dll", "GetUserDefaultGeoName") != nullptr,
          "GetUserDefaultGeoName resolves");
    check(resolve<GetLocaleFn>("api-ms-win-core-localization-l1-2-0.dll", "GetUserDefaultLocaleName") != nullptr,
          "GetUserDefaultLocaleName resolves");
    check(resolve<LocaleToLcidFn>("api-ms-win-core-localization-l1-2-0.dll", "LocaleNameToLCID") != nullptr,
          "LocaleNameToLCID resolves");
    check(resolve<SetBoostFn>("api-ms-win-core-processthreads-l1-1-2.dll", "SetProcessPriorityBoost") != nullptr,
          "SetProcessPriorityBoost resolves");
    check(resolve<GetBoostFn>("kernel32.dll", "GetProcessPriorityBoost") != nullptr,
          "GetProcessPriorityBoost resolves");
    check(resolve<WaitPipeWFn>("api-ms-win-core-namedpipe-l1-1-0.dll", "WaitNamedPipeW") != nullptr,
          "WaitNamedPipeW resolves");
    check(resolve<SetPipeStateFn>("api-ms-win-core-namedpipe-l1-1-0.dll", "SetNamedPipeHandleState") != nullptr,
          "SetNamedPipeHandleState resolves");
}

void test_localization(bool expect_overrides) {
    auto get_geo = resolve<GetGeoFn>("api-ms-win-core-localization-l1-2-3.dll", "GetUserDefaultGeoName");
    auto get_locale = resolve<GetLocaleFn>("api-ms-win-core-localization-l1-2-0.dll", "GetUserDefaultLocaleName");
    auto to_lcid = resolve<LocaleToLcidFn>("api-ms-win-core-localization-l1-2-0.dll", "LocaleNameToLCID");

    SetLastError(0);
    check(get_geo(nullptr, 2) == 0 && GetLastError() == ERROR_INVALID_PARAMETER,
          "GetUserDefaultGeoName rejects nullptr with positive count");
    wchar_t small_geo[2]{};
    SetLastError(0);
    check(get_geo(small_geo, 1) == 0 && GetLastError() == ERROR_INSUFFICIENT_BUFFER,
          "GetUserDefaultGeoName reports ERROR_INSUFFICIENT_BUFFER for undersized buffer");

    const int geo_needed = get_geo(nullptr, 0);
    check(geo_needed >= 3, "GetUserDefaultGeoName(nullptr, 0) returns required length");
    wchar_t geo[16]{};
    const int geo_written = get_geo(geo, 16);
    check(geo_written == geo_needed && geo[0] != L'\0', "GetUserDefaultGeoName fills non-empty region code");
    if (expect_overrides) {
        check(std::wcscmp(geo, L"JP") == 0, "GetUserDefaultGeoName honors and uppercases LEONIDA_GEO override");
    }

    SetLastError(0);
    check(get_locale(nullptr, 4) == 0 && GetLastError() == ERROR_INVALID_PARAMETER,
          "GetUserDefaultLocaleName rejects nullptr with positive count");
    wchar_t tiny_loc[2]{};
    SetLastError(0);
    check(get_locale(tiny_loc, 1) == 0 && GetLastError() == ERROR_INSUFFICIENT_BUFFER,
          "GetUserDefaultLocaleName reports ERROR_INSUFFICIENT_BUFFER on cap=1");
    wchar_t loc[LOCALE_NAME_MAX_LENGTH]{};
    const int loc_written = get_locale(loc, LOCALE_NAME_MAX_LENGTH);
    check(loc_written > 1 && loc[0] != L'\0', "GetUserDefaultLocaleName fills non-empty locale");
    if (expect_overrides) {
        check(std::wcscmp(loc, L"en-GB") == 0, "GetUserDefaultLocaleName honors LEONIDA_LOCALE override");
        check(to_lcid(nullptr, 0) == to_lcid(L"en-GB", 0),
              "LocaleNameToLCID(nullptr) matches overridden user locale");
    } else {
        check(to_lcid(L"en-US", 0) == 0x0409, "LocaleNameToLCID resolves en-US to 0x0409");
    }
}

void test_process_priority_boost() {
    auto set_boost = resolve<SetBoostFn>("api-ms-win-core-processthreads-l1-1-2.dll", "SetProcessPriorityBoost");
    auto get_boost = resolve<GetBoostFn>("kernel32.dll", "GetProcessPriorityBoost");

    BOOL host_before = FALSE;
    check(::GetProcessPriorityBoost(GetCurrentProcess(), &host_before) != FALSE,
          "host GetProcessPriorityBoost baseline readable");

    BOOL val = FALSE;
    SetLastError(0);
    check(!set_boost(nullptr, TRUE) && GetLastError() == ERROR_INVALID_HANDLE,
          "SetProcessPriorityBoost rejects null handle");
    SetLastError(0);
    check(!get_boost(nullptr, &val) && GetLastError() == ERROR_INVALID_HANDLE,
          "GetProcessPriorityBoost rejects null handle");
    SetLastError(0);
    check(!get_boost(GetCurrentProcess(), nullptr) && GetLastError() == ERROR_INVALID_PARAMETER,
          "GetProcessPriorityBoost rejects null out pointer");

    check(set_boost(GetCurrentProcess(), !host_before) != FALSE,
          "SetProcessPriorityBoost accepts GetCurrentProcess()");
    check(get_boost(GetCurrentProcess(), &val) != FALSE && val == (!host_before ? TRUE : FALSE),
          "GetProcessPriorityBoost returns guest-configured state");

    BOOL host_after = FALSE;
    check(::GetProcessPriorityBoost(GetCurrentProcess(), &host_after) != FALSE && host_after == host_before,
          "host process priority-boost state remains untouched by guest calls");

    check(set_boost(GetCurrentProcess(), host_before) != FALSE,
          "SetProcessPriorityBoost toggles cleanly");
}

void test_named_pipes() {
    auto wait_w = resolve<WaitPipeWFn>("api-ms-win-core-namedpipe-l1-1-0.dll", "WaitNamedPipeW");
    auto wait_a = resolve<WaitPipeAFn>("kernel32.dll", "WaitNamedPipeA");
    auto set_state = resolve<SetPipeStateFn>("api-ms-win-core-namedpipe-l1-1-0.dll", "SetNamedPipeHandleState");

    SetLastError(0);
    check(!wait_w(nullptr, 0) && GetLastError() == ERROR_INVALID_PARAMETER, "WaitNamedPipeW rejects nullptr cleanly");
    SetLastError(0);
    check(!wait_a(nullptr, 0) && GetLastError() == ERROR_BAD_PATHNAME,
          "WaitNamedPipeA rejects nullptr with ERROR_BAD_PATHNAME");
    SetLastError(0);
    check(!wait_w(L"\\\\.\\pipe\\leonida_nonexistent_test_pipe", 0) && GetLastError() == ERROR_FILE_NOT_FOUND,
          "WaitNamedPipeW reports ERROR_FILE_NOT_FOUND on missing pipe");

    DWORD mode = PIPE_READMODE_BYTE;
    SetLastError(0);
    check(!set_state(nullptr, &mode, nullptr, nullptr) && GetLastError() == ERROR_INVALID_HANDLE,
          "SetNamedPipeHandleState rejects nullptr cleanly");
    SetLastError(0);
    check(!set_state(INVALID_HANDLE_VALUE, &mode, nullptr, nullptr) && GetLastError() == ERROR_INVALID_HANDLE,
          "SetNamedPipeHandleState rejects INVALID_HANDLE_VALUE cleanly");
}

}  // namespace

int main(int argc, char** argv) {
    bool expect_overrides = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--expect-override") == 0) expect_overrides = true;
    }
    test_registry_coverage();
    test_localization(expect_overrides);
    test_process_priority_boost();
    test_named_pipes();
    std::printf("%d check(s), %d failure(s)\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
