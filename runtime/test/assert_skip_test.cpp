// LEONIDA_SKIP_ASSERTS=N (src/assert_skip.h), with synthetic code only:
//   - the value: unset, empty and 0 are off; a decimal number up to 1000000 is the limit; anything else is off;
//   - off, nothing is installed and an int3 reaches the next handler as before;
//   - on, an int3 in memory no host module owns (a page this test writes, as the loader maps a game image) is
//     stepped over, logged with its address, and the code after it runs;
//   - only N of them: the next one is passed on, as are an int3 in a host module (this executable's own
//     __debugbreak) and a breakpoint whose address holds no 0xCC (the two-byte `int 3`).
// A fallback vectored handler added after the one under test stands in for "the run ends": it counts what reaches it
// and steps over it, so the test keeps going. No game data, no mapped images.
#include <windows.h>

#include <cstdint>
#include <cstdio>
#include <cstring>

#include "assert_skip.h"

namespace {

int g_failed = 0;

void check(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "ok   " : "FAIL ", what);
    if (!ok) ++g_failed;
}

// What the handler under test passed on.
volatile LONG g_passed_on = 0;
// What it saw for RIP in the first breakpoint (Windows and Wine report the 0xCC's own address).
volatile LONG g_rip_at_int3 = -1;

LONG CALLBACK fallback(EXCEPTION_POINTERS* info) {
    if (info->ExceptionRecord->ExceptionCode != EXCEPTION_BREAKPOINT) return EXCEPTION_CONTINUE_SEARCH;
    const auto at = reinterpret_cast<DWORD64>(info->ExceptionRecord->ExceptionAddress);
    InterlockedIncrement(&g_passed_on);
    if (g_rip_at_int3 < 0) InterlockedExchange(&g_rip_at_int3, info->ContextRecord->Rip == at ? 1 : 0);
    if (info->ContextRecord->Rip == at) info->ContextRecord->Rip += 1;  // over the one-byte int3
    return EXCEPTION_CONTINUE_EXECUTION;
}

// int3; mov eax, imm32; ret. The value shows that the code after the int3 ran.
using Site = int (*)();
Site write_site(unsigned char* at, std::uint32_t value) {
    const unsigned char code[] = {0xCC, 0xB8, 0, 0, 0, 0, 0xC3};
    std::memcpy(at, code, sizeof code);
    std::memcpy(at + 2, &value, sizeof value);
    return reinterpret_cast<Site>(at);
}

__declspec(noinline) int host_int3() {
    __debugbreak();
    return 7;
}

}  // namespace

int main() {
    SetErrorMode(GetErrorMode() | SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    namespace as = rt::assert_skip;

    bool valid = false;
    check(as::parse_limit(nullptr, &valid) == 0 && valid, "a null value is off");
    check(as::parse_limit(L"", &valid) == 0 && valid, "an empty value is off");
    check(as::parse_limit(L"0", &valid) == 0 && valid, "0 is off");
    check(as::parse_limit(L"3", &valid) == 3 && valid, "3 is a limit of 3");
    check(as::parse_limit(L"1000000", &valid) == 1000000 && valid, "1000000 is the largest limit");
    check(as::parse_limit(L"1000001", &valid) == 0 && !valid, "1000001 is refused");
    check(as::parse_limit(L"-1", &valid) == 0 && !valid, "a negative number is refused");
    check(as::parse_limit(L" 2", &valid) == 0 && !valid, "a leading space is refused");
    check(as::parse_limit(L"2x", &valid) == 0 && !valid, "trailing text is refused");
    check(as::parse_limit(L"yes", &valid) == 0 && !valid, "text is refused");
    SetEnvironmentVariableW(L"LEONIDA_SKIP_ASSERTS", nullptr);
    check(as::limit_from_environment() == 0, "unset in the environment: off");
    SetEnvironmentVariableW(L"LEONIDA_SKIP_ASSERTS", L"2");
    check(as::limit_from_environment() == 2, "LEONIDA_SKIP_ASSERTS=2 in the environment: 2");
    SetEnvironmentVariableW(L"LEONIDA_SKIP_ASSERTS", L"many");
    check(as::limit_from_environment() == 0, "LEONIDA_SKIP_ASSERTS=many in the environment: off");

    // Executable memory no host module owns, the way the loader maps a title's image.
    auto* page = static_cast<unsigned char*>(
        VirtualAlloc(nullptr, 4096, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE));
    check(page != nullptr, "allocate an executable page");
    if (!page) return 1;
    HMODULE owner = nullptr;
    check(!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                              reinterpret_cast<LPCWSTR>(page), &owner),
          "the page is in no host module");
    const Site a = write_site(page, 0x41);
    const Site b = write_site(page + 16, 0x42);
    const Site c = write_site(page + 32, 0x43);
    FlushInstructionCache(GetCurrentProcess(), page, 4096);

    void* const fallback_handler = AddVectoredExceptionHandler(0, &fallback);
    check(fallback_handler != nullptr, "add the fallback handler");

    // Off: the int3 reaches the fallback, which stands in for the end of the run.
    check(as::install(0) && as::skipped() == 0, "install(0) adds nothing");
    check(a() == 0x41 && g_passed_on == 1, "off: a guest int3 is passed on");
    check(g_rip_at_int3 == 1, "the breakpoint's RIP is the 0xCC's own address");

    // On, limit 2. The handler is added last, so it must come before the fallback to see anything: remove the
    // fallback and add it again behind it.
    RemoveVectoredExceptionHandler(fallback_handler);
    check(as::install(2), "install(2) adds the handler");
    void* const fallback_again = AddVectoredExceptionHandler(0, &fallback);
    check(fallback_again != nullptr, "add the fallback handler behind it");
    check(a() == 0x41 && as::skipped() == 1 && g_passed_on == 1, "the first guest int3 is stepped over");
    check(b() == 0x42 && as::skipped() == 2 && g_passed_on == 1, "the second, at another address, too");
    check(host_int3() == 7 && as::skipped() == 2 && g_passed_on == 2, "an int3 in a host module is passed on");
    check(c() == 0x43 && as::skipped() == 2 && g_passed_on == 3, "past the limit, a guest int3 is passed on");
    check(a() == 0x41 && as::skipped() == 2 && g_passed_on == 4, "and so is every later one");

    // A breakpoint whose reported address holds no 0xCC: the two-byte `int 3` (CD 03), for which Windows and Wine
    // report the byte after the CD. It is passed on, whatever the budget.
    check(as::install(5), "install(5) starts over");
    RemoveVectoredExceptionHandler(fallback_again);
    void* const fallback_last = AddVectoredExceptionHandler(0, &fallback);
    page[48] = 0xCD;  // int 3, two bytes; then the site's mov eax, 0x44; ret
    page[49] = 0x03;
    const Site d = reinterpret_cast<Site>(page + 48);
    write_site(page + 49, 0x44);
    page[49] = 0x03;  // write_site's 0xCC is the second byte of the int 3
    FlushInstructionCache(GetCurrentProcess(), page, 4096);
    const LONG before = g_passed_on;
    check(d() == 0x44 && as::skipped() == 0 && g_passed_on == before + 1,
          "a breakpoint with no 0xCC at its address is passed on");
    check(b() == 0x42 && as::skipped() == 1 && g_passed_on == before + 1, "a guest int3 after a restart is stepped over");

    as::install(0);
    RemoveVectoredExceptionHandler(fallback_last);
    VirtualFree(page, 0, MEM_RELEASE);
    std::printf("%s\n", g_failed ? "FAILED" : "all checks passed");
    return g_failed ? 1 : 0;
}
