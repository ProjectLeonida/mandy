// Opt-in unhandled-exception log: LEONIDA_CRASH_LOG.
//
// `LEONIDA_CRASH_LOG` names a file (or is `1`, `on` or `true` for the caller's default path, the overlay's
// leonida-crash.log). When set, install() registers a top-level unhandled-exception filter that writes one block to
// stderr and to that file: the exception code, the faulting address as module+0xRVA and raw, the access kind and
// address for an access violation, the thread id, the context (rip/rsp/rbp), and up to 16 frames as module+0xRVA.
//
// It is a SetUnhandledExceptionFilter handler, not a vectored one: an exception something else handles (a shim's
// __try, a C++ catch, the loader's own checks) is not logged, and the block is written only for the exception that
// would have ended the process. The handler returns the previous filter's answer (EXCEPTION_CONTINUE_SEARCH when
// there was none), so termination and its exit code are exactly what they would have been without the log. A
// debugger still gets the exception first and decides, as it does without the log.
//
// The handler must not deadlock a process that was already dying with a lock held. It therefore does not call
// shims::module_offset or any Loader API: module names come from module_ranges.h, a table published by normal code
// and read with per-slot seqlocks (no mutex, no Loader lock). It builds the block in a fixed stack buffer and writes
// it with WriteFile on handles captured at install, so it does not allocate or take a CRT/stdio lock either. A
// corrupt stack, a seqlock that cannot settle or a failing write drops the block instead of changing the process's
// fate.
//
// Host modules are the ones loaded when install() runs. A host DLL loaded later (the game's LoadLibrary, the Vulkan
// loader) has no entry and its frames print as raw addresses; mapped images are always current because the Loader
// publishes them as it maps.
//
// The file can quote addresses only, never title data; it stays on the player's machine.
#pragma once

#include <windows.h>
#include <tlhelp32.h>

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <string>
#include <vector>

#include "module_ranges.h"
#include "stack_walk.h"

namespace rt::crash_log {

namespace detail {

inline std::atomic<bool> on{false};
// Install/close take this; the handler never does. Immortal so a late close cannot touch a destroyed static.
inline std::mutex& mutex() {
    static auto* m = new std::mutex;
    return *m;
}
inline HANDLE file = INVALID_HANDLE_VALUE;
inline HANDLE stderr_handle = INVALID_HANDLE_VALUE;
inline LPTOP_LEVEL_EXCEPTION_FILTER previous = nullptr;
inline bool installed = false;
inline thread_local bool in_handler = false;
inline constexpr int kMaxFrames = 16;
inline constexpr size_t kBlockBytes = 4096;

// A bounded writer over a fixed stack buffer: no allocation, no printf, no locale or CRT lock.
struct Text {
    char* start;
    char* p;
    char* end;

    Text(char* buffer, size_t size) : start(buffer), p(buffer), end(buffer + size) {}

    void put(char c) {
        if (p < end) *p++ = c;
    }
    void put(const char* s) {
        while (*s) put(*s++);
    }
    void hex(uint64_t value, int min_digits = 0) {
        char digits[16];
        int n = 0;
        do {
            digits[n++] = "0123456789abcdef"[value & 15u];
            value >>= 4;
        } while (value && n < 16);
        while (n < min_digits && n < 16) digits[n++] = '0';
        while (n) put(digits[--n]);
    }
    // Exception codes keep their conventional upper-case spelling (0xC0000005).
    void code(uint32_t value) {
        char digits[8];
        for (int i = 7; i >= 0; --i) {
            digits[i] = "0123456789ABCDEF"[value & 15u];
            value >>= 4;
        }
        for (int i = 0; i < 8; ++i) put(digits[i]);
    }
    void dec(uint64_t value) {
        char digits[20];
        int n = 0;
        do {
            digits[n++] = static_cast<char>('0' + value % 10u);
            value /= 10u;
        } while (value && n < 20);
        while (n) put(digits[--n]);
    }
    void two(unsigned value) {
        put(static_cast<char>('0' + (value / 10u) % 10u));
        put(static_cast<char>('0' + value % 10u));
    }
    void timestamp() {
        SYSTEMTIME t{};
        GetSystemTime(&t);
        dec(t.wYear);
        put('-');
        two(t.wMonth);
        put('-');
        two(t.wDay);
        put('T');
        two(t.wHour);
        put(':');
        two(t.wMinute);
        put(':');
        two(t.wSecond);
        put('.');
        two(static_cast<unsigned>(t.wMilliseconds / 10u));
        put(static_cast<char>('0' + t.wMilliseconds % 10u));
        put('Z');
    }
    // "module+0xRVA" for a published range, else the bare address.
    void where(uint64_t address) {
        char name[module_ranges::kNameBytes] = {};
        uint32_t rva = 0;
        bool host = false;
        if (module_ranges::find(address, name, sizeof name, &rva, &host)) {
            if (host) put("host:");
            put(name);
            put("+0x");
            hex(rva);
        } else {
            put("0x");
            hex(address);
        }
    }
    size_t size() const { return static_cast<size_t>(p - start); }
};

inline const char* access_kind(ULONG_PTR operation) {
    switch (operation) {
        case 0: return "reading";
        case 1: return "writing";
        case 8: return "executing";
        default: return "accessing";
    }
}

inline std::string utf8_from_wide(const wchar_t* wide) {
    const int bytes = WideCharToMultiByte(CP_UTF8, 0, wide, -1, nullptr, 0, nullptr, nullptr);
    if (bytes <= 1) return {};
    std::string s(static_cast<size_t>(bytes), '\0');
    WideCharToMultiByte(CP_UTF8, 0, wide, -1, s.data(), bytes, nullptr, nullptr);
    s.resize(static_cast<size_t>(bytes) - 1);
    return s;
}

// The host ranges at install time. Toolhelp rather than GetModuleHandleExW/EnumProcessModules so the install path
// does not depend on a Loader being alive; install runs before any game code, so taking the loader lock here is safe.
inline std::vector<module_ranges::Entry> host_modules() {
    std::vector<module_ranges::Entry> entries;
    const HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPMODULE, GetCurrentProcessId());
    if (snapshot == INVALID_HANDLE_VALUE) return entries;
    MODULEENTRY32W module{};
    module.dwSize = sizeof module;
    for (BOOL ok = Module32FirstW(snapshot, &module); ok; ok = Module32NextW(snapshot, &module)) {
        module_ranges::Entry entry;
        entry.base = reinterpret_cast<uint64_t>(module.modBaseAddr);
        entry.size = module.modBaseSize;
        entry.name = utf8_from_wide(module.szModule);
        entries.push_back(std::move(entry));
    }
    CloseHandle(snapshot);
    return entries;
}

inline void write_open_handle(HANDLE handle, const char* data, size_t size) {
    if (!handle || handle == INVALID_HANDLE_VALUE || size == 0) return;
    DWORD written = 0;
    WriteFile(handle, data, static_cast<DWORD>(size), &written, nullptr);
}

// Fixed buffers and WriteFile only: safe to run with the shim registry, the Loader's range lock or a CRT lock held.
inline void write_block(const EXCEPTION_POINTERS& info) {
    char buffer[kBlockBytes] = {};
    Text text(buffer, sizeof buffer);
    const EXCEPTION_RECORD& record = *info.ExceptionRecord;
    const uint64_t address = reinterpret_cast<uint64_t>(record.ExceptionAddress);
    text.put("[rt] crash: ");
    text.timestamp();
    text.put(" unhandled exception 0x");
    text.code(static_cast<uint32_t>(record.ExceptionCode));
    text.put(" at ");
    text.where(address);
    text.put(" (0x");
    text.hex(address);
    text.put("), thread ");
    text.dec(GetCurrentThreadId());
    text.put("\n");
    if (record.ExceptionCode == EXCEPTION_ACCESS_VIOLATION && record.NumberParameters >= 2) {
        text.put("[rt] crash:   access violation ");
        text.put(access_kind(record.ExceptionInformation[0]));
        text.put(" address 0x");
        text.hex(record.ExceptionInformation[1]);
        text.put("\n");
    }
    if (info.ContextRecord) {
        text.put("[rt] crash:   rip 0x");
        text.hex(info.ContextRecord->Rip);
        text.put(" rsp 0x");
        text.hex(info.ContextRecord->Rsp);
        text.put(" rbp 0x");
        text.hex(info.ContextRecord->Rbp);
        text.put("\n");
        uint64_t frames[kMaxFrames] = {};
        const int count = stack_walk::frames(*info.ContextRecord, frames, kMaxFrames);
        text.put("[rt] crash:   stack:\n");
        for (int i = 0; i < count; ++i) {
            text.put("[rt] crash:     #");
            text.dec(static_cast<uint64_t>(i));
            text.put(' ');
            text.where(frames[i]);
            text.put("\n");
        }
    }
    text.put("[rt] crash: end of unhandled exception 0x");
    text.code(static_cast<uint32_t>(record.ExceptionCode));
    text.put("\n");
    write_open_handle(stderr_handle, buffer, text.size());
    write_open_handle(file, buffer, text.size());
}

inline LONG CALLBACK handler(EXCEPTION_POINTERS* info) {
    if (!on.load(std::memory_order_acquire) || in_handler || !info || !info->ExceptionRecord)
        return EXCEPTION_CONTINUE_SEARCH;
    in_handler = true;
#ifdef _MSC_VER
    // Under /EHsc, catch(...) does not catch SEH faults; keep this frame free of objects with destructors.
    __try {
        write_block(*info);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
    }
#else
    try {
        write_block(*info);
    } catch (...) {
    }
#endif
    in_handler = false;
    // The answer the process would have got without the log: EXCEPTION_CONTINUE_SEARCH lets Windows end it as usual.
    return previous ? previous(info) : EXCEPTION_CONTINUE_SEARCH;
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
    const std::vector<module_ranges::Entry> host = detail::host_modules();
    {
        const std::lock_guard lock(detail::mutex());
        if (detail::installed) return true;
        detail::stderr_handle = GetStdHandle(STD_ERROR_HANDLE);
        if (!path.empty()) {
            detail::file = CreateFileW(path.c_str(), FILE_APPEND_DATA, FILE_SHARE_READ | FILE_SHARE_WRITE,
                                        nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
            if (detail::file == INVALID_HANDLE_VALUE)
                std::fprintf(stderr, "[rt] crash: cannot create the log file; logging to stderr only\n");
        } else {
            std::fprintf(stderr, "[rt] crash: no log path; logging to stderr only\n");
        }
        // Before the handler is enabled: the host ranges never change after this (later host DLLs fall back to raw
        // addresses), and mapped ranges come from the Loader as it maps.
        module_ranges::publish_host(host);
        if (detail::file != INVALID_HANDLE_VALUE)
            std::fprintf(stderr, "[rt] crash: unhandled exceptions are logged to %s and stderr\n",
                         detail::utf8_from_wide(path.c_str()).c_str());
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
    if (detail::file != INVALID_HANDLE_VALUE) CloseHandle(detail::file);
    detail::file = INVALID_HANDLE_VALUE;
}

}  // namespace rt::crash_log
