#pragma once

// LEONIDA_SKIP_ASSERTS=N (diagnostic, off by default): step over up to N int3 breakpoints in guest code.
//
// A title's assert ends in an int3 (0xCC). With a debugger attached that is a stop to look at; without one it is an
// unhandled EXCEPTION_BREAKPOINT and the run ends there. gta6rt installs this vectored handler when the variable is a
// number above 0 (main.cpp, before any image code runs). For an EXCEPTION_BREAKPOINT whose address is in no host
// module (the Windows loader knows none at that address: the mapped game images, or code a title wrote itself) and
// whose byte there is 0xCC, it logs one line, moves RIP past the int3 and continues; after N of them it stops doing so,
// says so once, and the next int3 ends the run as before. An int3 in a host module (gta6rt, its shims, a system DLL,
// DbgBreakPoint) is never stepped over.
//
// RIP: on Windows x64 an int3's ExceptionAddress and the context's RIP both name the 0xCC (the kernel backs RIP up
// over it), and Wine reports the same; the handler also takes a RIP one past the 0xCC, which it leaves as it is.
// Any other RIP is not this int3's, and the exception is passed on.
//
// Unlike LEONIDA_SKIP_INT3 (startup_trace.h), it needs no --trace, counts every int3 rather than each address once,
// and stops after N, so a loop of asserts can't spin forever. With --trace each skip is also an `int3_skipped` event,
// and the trace's own handler, which runs first, records the exception and its stack. The handler is added last, so
// every other vectored handler sees the breakpoint first. Each skipped address is a finding, not a fix: a correct run
// reaches none.
//
// The log lines quote addresses as module+RVA only; they stay on the player's machine like the startup trace does.

#include <windows.h>

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cwchar>
#include <string>

#include "../shims/shims.h"
#include "../shims/startup_trace.h"

namespace rt::assert_skip {

inline constexpr wchar_t kVariable[] = L"LEONIDA_SKIP_ASSERTS";
// Far above any useful count; it only keeps the number in an unsigned.
inline constexpr unsigned long kMaxLimit = 1000000;

// The limit LEONIDA_SKIP_ASSERTS asks for: 0 when the text is null or empty, "0", or not a decimal number in
// 0..kMaxLimit (then `*valid`, when given, is false).
inline unsigned parse_limit(const wchar_t* text, bool* valid = nullptr) {
    if (valid) *valid = true;
    if (!text || !*text) return 0;
    if (*text < L'0' || *text > L'9') {  // no sign, no leading space: wcstoul would take "-1" as ULONG_MAX
        if (valid) *valid = false;
        return 0;
    }
    errno = 0;
    wchar_t* end = nullptr;
    const unsigned long n = std::wcstoul(text, &end, 10);
    if (errno != 0 || !end || *end != L'\0' || n > kMaxLimit) {
        if (valid) *valid = false;
        return 0;
    }
    return static_cast<unsigned>(n);
}

// LEONIDA_SKIP_ASSERTS as a limit; a value that is not one is reported on stderr and counts as 0 (off).
inline unsigned limit_from_environment() {
    wchar_t value[32] = {};
    const DWORD length = GetEnvironmentVariableW(kVariable, value, 32);
    if (length == 0) return 0;
    bool valid = false;
    const unsigned limit = length < 32 ? parse_limit(value, &valid) : 0;
    if (length >= 32 || !valid)
        std::fprintf(stderr, "[rt] LEONIDA_SKIP_ASSERTS: expected a number of int3 to step over (0..%lu); off\n",
                     kMaxLimit);
    return limit;
}

namespace detail {
inline std::atomic<unsigned> limit{0};
inline std::atomic<unsigned> claimed{0};  // skips handed out, may pass `limit` by the declined ones
inline std::atomic<bool> exhausted_logged{false};
inline void* veh = nullptr;

// True when the Windows loader knows a module at `address`: gta6rt, its DLLs, system DLLs. The Loader's mapped
// images are not among them.
inline bool in_host_module(const void* address) {
    HMODULE module = nullptr;
    return GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                              static_cast<LPCWSTR>(address), &module) && module;
}

// The byte at `address`, read only when the page is committed and readable.
inline bool read_byte(const void* address, unsigned char& byte) {
    MEMORY_BASIC_INFORMATION mbi{};
    if (!address || !VirtualQuery(address, &mbi, sizeof mbi) || mbi.State != MEM_COMMIT) return false;
    if (mbi.Protect & (PAGE_GUARD | PAGE_NOACCESS)) return false;
    const DWORD readable = PAGE_READONLY | PAGE_READWRITE | PAGE_WRITECOPY | PAGE_EXECUTE_READ |
                           PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY;
    if (!(mbi.Protect & readable)) return false;
    byte = *static_cast<const unsigned char*>(address);
    return true;
}

inline LONG CALLBACK handler(EXCEPTION_POINTERS* info) {
    if (!info || !info->ExceptionRecord || !info->ContextRecord) return EXCEPTION_CONTINUE_SEARCH;
    const EXCEPTION_RECORD& r = *info->ExceptionRecord;
    if (r.ExceptionCode != EXCEPTION_BREAKPOINT) return EXCEPTION_CONTINUE_SEARCH;
    const unsigned max = limit.load(std::memory_order_relaxed);
    if (max == 0) return EXCEPTION_CONTINUE_SEARCH;
    const void* at = r.ExceptionAddress;
    const auto at_word = reinterpret_cast<DWORD64>(at);
    CONTEXT& c = *info->ContextRecord;
    if (c.Rip != at_word && c.Rip != at_word + 1) return EXCEPTION_CONTINUE_SEARCH;  // not this int3's RIP
    unsigned char byte = 0;
    if (in_host_module(at) || !read_byte(at, byte) || byte != 0xCC) return EXCEPTION_CONTINUE_SEARCH;
    const unsigned n = claimed.fetch_add(1, std::memory_order_relaxed) + 1;
    if (n > max) {
        if (!exhausted_logged.exchange(true, std::memory_order_relaxed))
            std::fprintf(stderr, "[rt] LEONIDA_SKIP_ASSERTS: %u int3 stepped over; the one at %s is not\n", max,
                         shims::module_offset(at_word).c_str());
        return EXCEPTION_CONTINUE_SEARCH;
    }
    const std::string where = shims::module_offset(at_word);
    std::fprintf(stderr, "[rt] LEONIDA_SKIP_ASSERTS: int3 at %s stepped over (%u of %u)\n", where.c_str(), n, max);
    std::fflush(stderr);
    startup_trace::event("int3_skipped", "\"at\":" + startup_trace::quote(where) + ",\"skip\":" + std::to_string(n));
    if (c.Rip == at_word) c.Rip = at_word + 1;
    return EXCEPTION_CONTINUE_EXECUTION;
}
}  // namespace detail

// Steps over up to `limit` guest int3 from now on (0 removes the handler). Returns false only when the handler could
// not be added. Called once, from main.cpp; tests call it again to start over.
inline bool install(unsigned limit) {
    detail::limit.store(0, std::memory_order_relaxed);
    if (detail::veh) RemoveVectoredExceptionHandler(detail::veh);
    detail::veh = nullptr;
    detail::claimed.store(0, std::memory_order_relaxed);
    detail::exhausted_logged.store(false, std::memory_order_relaxed);
    if (limit == 0) return true;
    detail::limit.store(limit, std::memory_order_relaxed);
    detail::veh = AddVectoredExceptionHandler(0, &detail::handler);  // last: the other handlers see it first
    if (!detail::veh) detail::limit.store(0, std::memory_order_relaxed);
    return detail::veh != nullptr;
}

// int3 stepped over since install.
inline unsigned skipped() {
    const unsigned claimed = detail::claimed.load(std::memory_order_relaxed);
    const unsigned max = detail::limit.load(std::memory_order_relaxed);
    return claimed < max ? claimed : max;
}

}  // namespace rt::assert_skip
