// Opt-in unhandled-exception log: LEONIDA_CRASH_LOG.
//
// `LEONIDA_CRASH_LOG` names a file (or is `1`, `on` or `true` for the caller's default path, the overlay's
// leonida-crash.log). When set, install() registers a top-level unhandled-exception filter that writes one block to
// stderr and to that file: the exception code, the faulting address as module+0xRVA and raw, the access kind and
// address for an access violation, the thread id, the context, and up to 16 frames as module+0xRVA.
//
// It is a SetUnhandledExceptionFilter handler, not a vectored one: an exception something else handles (a shim's
// __try, a C++ catch, the loader's own checks) is not logged, and the block is written only for the exception that
// would have ended the process. The handler returns the previous filter's answer (EXCEPTION_CONTINUE_SEARCH when
// there was none), so termination and its exit code are exactly what they would have been without the log. A
// debugger still gets the exception first and decides, as it does without the log.
//
// The block is written under SEH: a corrupt stack or a failing allocation drops the block instead of changing the
// process's fate. Nothing is read from the title's memory beyond the stack the walk needs.
//
// The file can quote addresses only, never title data; it stays on the player's machine.
#pragma once

#include <windows.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <string>

#include "shims.h"
#include "stack_walk.h"

namespace rt::crash_log {

namespace detail {

inline std::atomic<bool> on{false};
// Intentionally immortal: the filter can run during teardown, after ordinary statics would be destroyed.
inline std::mutex& mutex() {
    static auto* m = new std::mutex;
    return *m;
}
inline FILE* file = nullptr;
inline LPTOP_LEVEL_EXCEPTION_FILTER previous = nullptr;
inline bool installed = false;
inline thread_local bool in_handler = false;
inline constexpr int kMaxFrames = 16;

inline std::string hex(uint64_t value) {
    char b[24];
    std::snprintf(b, sizeof b, "0x%llx", static_cast<unsigned long long>(value));
    return b;
}

inline std::string timestamp() {
    SYSTEMTIME t{};
    GetSystemTime(&t);
    char b[40];
    std::snprintf(b, sizeof b, "%04u-%02u-%02uT%02u:%02u:%02u.%03uZ", t.wYear, t.wMonth, t.wDay, t.wHour, t.wMinute,
                  t.wSecond, t.wMilliseconds);
    return b;
}

inline std::string utf8_path(const std::filesystem::path& path) {
    const std::wstring wide = path.wstring();
    const int bytes = WideCharToMultiByte(CP_UTF8, 0, wide.c_str(), static_cast<int>(wide.size()), nullptr, 0, nullptr,
                                          nullptr);
    if (bytes <= 0) return "<path>";
    std::string out(static_cast<size_t>(bytes), '\0');
    WideCharToMultiByte(CP_UTF8, 0, wide.c_str(), static_cast<int>(wide.size()), out.data(), bytes, nullptr, nullptr);
    return out;
}

inline const char* access_kind(ULONG_PTR operation) {
    switch (operation) {
        case 0: return "reading";
        case 1: return "writing";
        case 8: return "executing";
        default: return "accessing";
    }
}

// The block the filter writes. Public within the header so a test can check the fields on a hand-built record.
inline std::string format(const EXCEPTION_POINTERS& info) {
    const EXCEPTION_RECORD& r = *info.ExceptionRecord;
    std::string out;
    char line[512];
    std::snprintf(line, sizeof line,
                  "[rt] crash: %s unhandled exception 0x%08lX at %s (%s), thread %lu\n", timestamp().c_str(),
                  static_cast<unsigned long>(r.ExceptionCode),
                  shims::module_offset(reinterpret_cast<uint64_t>(r.ExceptionAddress)).c_str(),
                  hex(reinterpret_cast<uint64_t>(r.ExceptionAddress)).c_str(), GetCurrentThreadId());
    out += line;
    if (r.ExceptionCode == EXCEPTION_ACCESS_VIOLATION && r.NumberParameters >= 2) {
        std::snprintf(line, sizeof line, "[rt] crash:   access violation %s address %s\n",
                      access_kind(r.ExceptionInformation[0]),
                      hex(static_cast<uint64_t>(r.ExceptionInformation[1])).c_str());
        out += line;
    }
    if (info.ContextRecord) {
        std::snprintf(line, sizeof line, "[rt] crash:   rip %s rsp %s rbp %s\n",
                      hex(info.ContextRecord->Rip).c_str(), hex(info.ContextRecord->Rsp).c_str(),
                      hex(info.ContextRecord->Rbp).c_str());
        out += line;
        uint64_t frames[kMaxFrames] = {};
        const int count = stack_walk::frames(*info.ContextRecord, frames, kMaxFrames);
        out += "[rt] crash:   stack:\n";
        for (int i = 0; i < count; ++i) {
            std::snprintf(line, sizeof line, "[rt] crash:     #%d %s\n", i, shims::module_offset(frames[i]).c_str());
            out += line;
        }
    }
    std::snprintf(line, sizeof line, "[rt] crash: end of unhandled exception 0x%08lX\n",
                  static_cast<unsigned long>(r.ExceptionCode));
    out += line;
    return out;
}

inline void write_block(const EXCEPTION_POINTERS& info) {
    const std::string text = format(info);
    std::fwrite(text.data(), 1, text.size(), stderr);
    std::fflush(stderr);
    const std::lock_guard lock(detail::mutex());
    if (detail::file) {
        std::fwrite(text.data(), 1, text.size(), detail::file);
        std::fflush(detail::file);
    }
}

inline LONG CALLBACK handler(EXCEPTION_POINTERS* info) {
    if (!detail::on.load(std::memory_order_acquire) || detail::in_handler || !info || !info->ExceptionRecord)
        return EXCEPTION_CONTINUE_SEARCH;
    detail::in_handler = true;
#ifdef _MSC_VER
    // Under /EHsc, catch(...) does not catch SEH faults; keep this frame free of objects with destructors.
    __try {
        detail::write_block(*info);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
    }
#else
    try {
        detail::write_block(*info);
    } catch (...) {
    }
#endif
    detail::in_handler = false;
    // The answer the process would have got without the log: EXCEPTION_CONTINUE_SEARCH lets Windows end it as usual.
    return detail::previous ? detail::previous(info) : EXCEPTION_CONTINUE_SEARCH;
}

}  // namespace detail

inline bool enabled() noexcept { return detail::on.load(std::memory_order_acquire); }

// Reads LEONIDA_CRASH_LOG once. `fallback` is used when the variable is `1`/`on`/`true` (gta6rt passes the overlay's
// leonida-crash.log). Returns false when the variable is unset, empty or `0`, and installs only once.
inline bool install(const std::filesystem::path& fallback = {}) {
    std::wstring value(32768, L'\0');
    const DWORD n = GetEnvironmentVariableW(L"LEONIDA_CRASH_LOG", value.data(), static_cast<DWORD>(value.size()));
    if (n == 0 || n >= value.size()) return false;
    value.resize(n);
    if (value == L"0") return false;
    const bool use_fallback = value == L"1" || value == L"on" || value == L"true";
    std::filesystem::path path = use_fallback ? fallback : std::filesystem::path(value);
    if (!path.empty()) {
        std::error_code ec;
        path = std::filesystem::absolute(path, ec);
        if (ec) path = std::filesystem::path(value);  // keep what was given; the open below reports the failure
        if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path(), ec);
    }
    {
        const std::lock_guard lock(detail::mutex());
        if (detail::installed) return true;
        if (!path.empty()) {
            const errno_t open_error = _wfopen_s(&detail::file, path.c_str(), L"ab");
            if (open_error != 0) detail::file = nullptr;
            if (!detail::file)
                std::fprintf(stderr, "[rt] crash: cannot create the log file; logging to stderr only\n");
        } else {
            std::fprintf(stderr, "[rt] crash: no log path; logging to stderr only\n");
        }
        if (!path.empty() && detail::file)
            std::fprintf(stderr, "[rt] crash: unhandled exceptions are logged to %s and stderr\n",
                         detail::utf8_path(path).c_str());
        detail::previous = SetUnhandledExceptionFilter(&detail::handler);
        detail::installed = true;
    }
    detail::on.store(true, std::memory_order_release);
    return true;
}

// Removes the filter and closes the file (tests, and symmetry with startup_trace::close). The process exit path does
// not need it: the filter is only called on the way to termination.
inline void close() {
    detail::on.store(false, std::memory_order_release);
    const std::lock_guard lock(detail::mutex());
    if (detail::installed) {
        SetUnhandledExceptionFilter(detail::previous);
        detail::installed = false;
        detail::previous = nullptr;
    }
    if (detail::file) std::fclose(detail::file);
    detail::file = nullptr;
}

}  // namespace rt::crash_log
