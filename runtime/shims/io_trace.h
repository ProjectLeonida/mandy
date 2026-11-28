#pragma once
// LEONIDA_IO_TRACE=<file> writes one line for every read the title asks for, as it asks (RT-IO-TRACE):
//   ds  a DirectStorage request (dstorage.cpp): file, file offset, source and destination sizes, zlib/BCPack, the
//       request's name;
//   rf  a ReadFile, ReadFileEx or ReadFileScatter on a disk file (file_system.cpp): path, offset (the OVERLAPPED's, else the handle's
//       current position), size.
// Each line starts with the local wall-clock time, as the mem-probe lines in the log do, and is flushed at once, so
// the last reads before a crash or a device loss are on disk; the file is shared for reading while the game runs.
// What the title loaded just before a stop names the asset it was working on. Off, and free, without the variable.

#include <windows.h>
#include <share.h>

#include <cstdarg>
#include <cstdio>
#include <mutex>
#include <string>

namespace rt::io_trace {

inline FILE* out() {
    static FILE* file = [] {
        wchar_t path[MAX_PATH];
        const DWORD n = GetEnvironmentVariableW(L"LEONIDA_IO_TRACE", path, MAX_PATH);
        if (!n || n >= MAX_PATH) return static_cast<FILE*>(nullptr);
        return _wfsopen(path, L"w", _SH_DENYNO);  // readable while the game runs
    }();
    return file;
}

inline bool enabled() { return out() != nullptr; }

inline std::mutex& mutex() {
    static std::mutex m;
    return m;
}

// printf-style; the caller checks enabled() first so a disabled trace formats nothing.
inline void line(const char* format, ...) {
    FILE* f = out();
    if (!f) return;
    SYSTEMTIME now;
    GetLocalTime(&now);
    std::lock_guard lock(mutex());
    std::fprintf(f, "%02u:%02u:%02u.%03u ", now.wHour, now.wMinute, now.wSecond, now.wMilliseconds);
    va_list args;
    va_start(args, format);
    std::vfprintf(f, format, args);
    va_end(args);
    std::fputc('\n', f);
    std::fflush(f);
}

// The path of a file handle, asked of the handle each time: a handle value is reused after a close, so a cache by
// value would name the wrong file. ReadFile is a few thousand calls a boot; this is not the hot path.
inline std::string path_of(HANDLE file) {
    wchar_t wide[1024];
    const DWORD n = GetFinalPathNameByHandleW(file, wide, 1024, FILE_NAME_NORMALIZED);
    if (!n || n >= 1024) return "?";
    const int len = WideCharToMultiByte(CP_UTF8, 0, wide, int(n), nullptr, 0, nullptr, nullptr);
    std::string utf8(size_t(len), ' ');
    WideCharToMultiByte(CP_UTF8, 0, wide, int(n), utf8.data(), len, nullptr, nullptr);
    return utf8;
}

// A request name the title passed, copied defensively (the pointer is the title's). No C++ objects: __try.
// The load is inline in the __try, so under clang-cl a source that calls this needs /EHa (dstorage.cpp and
// io_trace_test have it, runtime/CMakeLists.txt); MSVC catches it either way.
inline int copy_name(const char* name, char* copy, int capacity) {
    int n = 0;
    __try {
        while (n + 1 < capacity && name[n]) {
            copy[n] = name[n];
            ++n;
        }
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return -1;
    }
    return n;
}

inline std::string safe_name(const char* name) {
    if (!name) return "-";
    char copy[128];
    const int n = copy_name(name, copy, int(sizeof copy));
    return n < 0 ? std::string("<unreadable>") : std::string(copy, size_t(n));
}

inline void note_read_file(HANDLE file, DWORD bytes, const OVERLAPPED* overlapped) {
    if (!enabled()) return;
    if (GetFileType(file) != FILE_TYPE_DISK) return;
    unsigned long long offset = 0;
    if (overlapped) {
        offset = (unsigned long long)overlapped->Offset | ((unsigned long long)overlapped->OffsetHigh << 32);
    } else {
        LARGE_INTEGER zero{}, at{};
        if (SetFilePointerEx(file, zero, &at, FILE_CURRENT)) offset = (unsigned long long)at.QuadPart;
    }
    line("rf %s off=0x%llx size=%lu", path_of(file).c_str(), offset, (unsigned long)bytes);
}

}  // namespace rt::io_trace
