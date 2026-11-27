#pragma once

// LEONIDA_THREAD_SNAPSHOT=<seconds>[,<seconds>...] (with --run): what every thread of the process is doing at those
// times after the entry point is called. Each other thread is suspended in turn, its registers are taken and its stack
// is unwound into a fixed array of return addresses, and it is resumed before anything is formatted, so nothing a
// suspended thread might hold (the heap, the loader lock) is needed while it is stopped. Then, on stderr:
//
//   [rt] snapshot 30 s: 98 threads, 41 distinct stacks
//   [rt] snapshot 30 s: 12 thread(s) [RenderWorker x8, (unnamed) x4], waiting in NtWaitForSingleObject on an Event
//        (auto-reset, not set):
//   [rt]   #0 host:ntdll.dll+0x... NtWaitForSingleObject
//   [rt]   #3 game_scarlett_master.exe+0x...
//
// A thread's name is its SetThreadDescription name or the MS_VC_EXCEPTION name the title raises (recorded from the
// moment install() runs). For a thread inside an ntdll wait the handle the syscall received is read from R10 (the
// first argument at `syscall`), and its object type and, for an event, its kind and state are queried. Frames are
// module+RVA (rt::shims::module_offset in gta6rt); frames in modules the host loaded carry their symbol (dbghelp: a
// PDB where there is one, the export name otherwise).
//
// The lines name the title's own code by RVA and its threads by name: they stay on the machine that ran it.

#include <windows.h>
#include <dbghelp.h>
#include <tlhelp32.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#pragma comment(lib, "dbghelp.lib")

namespace rt::thread_snapshot {

using Namer = std::string (*)(std::uint64_t address);  // "module+0xRVA" for an address

namespace detail {

constexpr DWORD kSetThreadName = 0x406D1388;  // MS_VC_EXCEPTION
constexpr int kMaxFrames = 96;  // driver compiler stacks run 30+ frames deep before their caller

inline std::mutex& names_mutex() {
    static std::mutex m;
    return m;
}
inline std::map<DWORD, std::string>& names() {
    static std::map<DWORD, std::string> n;
    return n;
}

// No C++ objects with destructors here: the copy runs under __try (a bad pointer from the title must not fault).
inline bool copy_name(const char* from, char* to, size_t size) {
    __try {
        size_t i = 0;
        for (; from && i + 1 < size && from[i]; ++i) to[i] = from[i];
        to[i] = 0;
        return i > 0;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return false;
    }
}

inline void remember_name(DWORD tid, const char* name) {
    const std::lock_guard lock(names_mutex());
    names()[tid] = name;
}

inline LONG CALLBACK name_handler(EXCEPTION_POINTERS* info) {
    const EXCEPTION_RECORD* r = info ? info->ExceptionRecord : nullptr;
    // THREADNAME_INFO is {DWORD type; LPCSTR name; DWORD thread; DWORD flags}: the first word's upper half is padding,
    // whatever the raiser's stack held, so only its low DWORD is the type.
    if (!r || r->ExceptionCode != kSetThreadName || r->NumberParameters < 3 ||
        static_cast<DWORD>(r->ExceptionInformation[0]) != 0x1000)
        return EXCEPTION_CONTINUE_SEARCH;
    char name[64];
    if (copy_name(reinterpret_cast<const char*>(r->ExceptionInformation[1]), name, sizeof(name))) {
        DWORD tid = static_cast<DWORD>(r->ExceptionInformation[2]);
        if (tid == 0xFFFFFFFF) tid = GetCurrentThreadId();
        remember_name(tid, name);
    }
    return EXCEPTION_CONTINUE_SEARCH;  // the title's own handler (or the default one) still sees it
}

struct Captured {
    DWORD tid = 0;
    bool ok = false;
    CONTEXT context{};
    CONTEXT caller{};  // the registers as unwound into frame 1 (the caller of the ntdll wait), for its handle
    bool has_caller = false;
    bool unwind_timed_out = false;  // its stack could not be walked while it was suspended (see capture)
    DWORD64 frames[kMaxFrames]{};
    int count = 0;
};

// Runs while the thread is suspended: only the unwinder and reads of the thread's own stack, no allocation.
inline void unwind_frames(Captured& c) {
    CONTEXT ctx = c.context;
    for (c.count = 0; c.count < kMaxFrames && ctx.Rip; ++c.count) {
        if (c.count == 1) {
            c.caller = ctx;
            c.has_caller = true;
        }
        c.frames[c.count] = ctx.Rip;
        DWORD64 image_base = 0;
        PRUNTIME_FUNCTION fn = RtlLookupFunctionEntry(ctx.Rip, &image_base, nullptr);
        if (!fn) {  // a leaf function: the return address is at [rsp]
            if (!ctx.Rsp || IsBadReadPtr(reinterpret_cast<void*>(ctx.Rsp), 8)) {
                ++c.count;
                break;
            }
            ctx.Rip = *reinterpret_cast<DWORD64*>(ctx.Rsp);
            ctx.Rsp += 8;
            continue;
        }
        void* handler_data = nullptr;
        DWORD64 frame = 0;
        RtlVirtualUnwind(UNW_FLAG_NHANDLER, image_base, ctx.Rip, fn, &ctx, &handler_data, &frame, nullptr);
    }
}

// Guarded: after a timeout (capture) the walk finishes on a stack that is moving again, where a read can fault.
inline void unwind(Captured& c) {
    __try {
        unwind_frames(c);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
    }
}

// The unwinder takes the loader's function-table locks (RtlLookupFunctionEntry). A thread suspended while it holds
// one of them exclusively (a DLL loading, a growable table changing) would block the unwind forever and stay
// suspended. So the unwind runs on a helper thread with a deadline: past it, the target is resumed (which lets the
// lock go and the helper finish) and that thread's frames are dropped.
struct UnwindHelper {
    HANDLE start = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    HANDLE done = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    Captured* job = nullptr;
    DWORD tid = 0;  // never suspended by a snapshot: it does the walking
    UnwindHelper() {
        HANDLE t = CreateThread(nullptr, 0, [](void* self) -> DWORD {
            auto* h = static_cast<UnwindHelper*>(self);
            for (;;) {
                WaitForSingleObject(h->start, INFINITE);
                unwind(*h->job);
                SetEvent(h->done);
            }
        }, this, 0, &tid);
        if (t) CloseHandle(t);
    }
};

inline UnwindHelper& helper() {
    static UnwindHelper h;
    return h;
}

inline bool capture(DWORD tid, Captured& c) {
    c.tid = tid;
    HANDLE h = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION, FALSE, tid);
    if (!h) return false;
    if (SuspendThread(h) == static_cast<DWORD>(-1)) {
        CloseHandle(h);
        return false;
    }
    c.context.ContextFlags = CONTEXT_FULL;
    c.ok = GetThreadContext(h, &c.context) != FALSE;
    bool timed_out = false;
    if (c.ok) {
        UnwindHelper& u = helper();
        u.job = &c;
        SetEvent(u.start);
        if (WaitForSingleObject(u.done, 1000) != WAIT_OBJECT_0) {
            timed_out = true;
            ResumeThread(h);                       // releases whatever lock the unwind is waiting for
            WaitForSingleObject(u.done, INFINITE);  // the helper finishes on the running thread's stack; discarded
        }
    }
    if (!timed_out) ResumeThread(h);
    CloseHandle(h);
    if (timed_out) {
        c.count = 0;
        c.unwind_timed_out = true;
    }
    return c.ok;
}

inline std::string symbol_of(DWORD64 address) {
    alignas(SYMBOL_INFO) char buffer[sizeof(SYMBOL_INFO) + 256] = {};
    auto* info = reinterpret_cast<SYMBOL_INFO*>(buffer);
    info->SizeOfStruct = sizeof(SYMBOL_INFO);
    info->MaxNameLen = 255;
    DWORD64 displacement = 0;
    if (!SymFromAddr(GetCurrentProcess(), address, &displacement, info)) return {};
    return info->Name;
}

inline std::string thread_name(DWORD tid) {
    {
        const std::lock_guard lock(names_mutex());
        if (auto it = names().find(tid); it != names().end()) return it->second;
    }
    using GetDescription = HRESULT(WINAPI*)(HANDLE, PWSTR*);
    static const auto get = reinterpret_cast<GetDescription>(
        GetProcAddress(GetModuleHandleW(L"kernelbase.dll"), "GetThreadDescription"));
    std::string name;
    if (HANDLE h = get ? OpenThread(THREAD_QUERY_LIMITED_INFORMATION, FALSE, tid) : nullptr) {
        PWSTR wide = nullptr;
        if (SUCCEEDED(get(h, &wide)) && wide) {
            const int n = WideCharToMultiByte(CP_UTF8, 0, wide, -1, nullptr, 0, nullptr, nullptr);
            if (n > 1) {
                name.resize(static_cast<size_t>(n - 1));
                WideCharToMultiByte(CP_UTF8, 0, wide, -1, name.data(), n, nullptr, nullptr);
            }
            LocalFree(wide);
        }
        CloseHandle(h);
    }
    return name;
}

// The object type behind a handle ("Event", "Thread", ...), empty when it is not a handle of this process.
inline std::string object_type(HANDLE handle) {
    using NtQueryObjectFn = LONG(NTAPI*)(HANDLE, ULONG, PVOID, ULONG, PULONG);
    static const auto query_object = reinterpret_cast<NtQueryObjectFn>(
        GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "NtQueryObject"));
    const auto value = reinterpret_cast<std::uintptr_t>(handle);
    if (!query_object || !value || (value & 3) || value > 0xFFFFFF) return {};  // handles are small multiples of 4
    alignas(8) unsigned char buffer[1024] = {};
    ULONG length = 0;
    if (query_object(handle, 2 /* ObjectTypeInformation */, buffer, sizeof(buffer), &length) < 0) return {};
    struct TypeName {  // the UNICODE_STRING that opens OBJECT_TYPE_INFORMATION (winternl.h not needed)
        USHORT Length;
        USHORT MaximumLength;
        PWSTR Buffer;
    };
    const auto* type = reinterpret_cast<const TypeName*>(buffer);
    std::string kind;
    for (USHORT i = 0; type->Buffer && i < type->Length / 2; ++i) kind += static_cast<char>(type->Buffer[i]);
    return kind;
}

// The handle a thread blocked in NtWaitForSingleObject waits on. Windows does not keep the syscall's argument
// registers for a thread stopped in the kernel, but the caller (WaitForSingleObjectEx, or the title's own code) keeps
// the handle in a non-volatile register to retry the wait after an alert; the unwound frame 1 has those. The first
// candidate that is a waitable object of this process is taken.
inline HANDLE waited_handle(const Captured& c) {
    std::vector<DWORD64> candidates;
    if (c.has_caller) {  // the caller's non-volatile registers first: they are what it keeps for the retry
        const CONTEXT& k = c.caller;
        for (DWORD64 v : {k.Rbx, k.Rsi, k.Rdi, k.R12, k.R13, k.R14, k.R15, k.Rbp}) candidates.push_back(v);
    }
    candidates.push_back(c.context.R10);  // the syscall's own argument registers last: often stale in the kernel
    candidates.push_back(c.context.Rcx);
    for (DWORD64 v : candidates) {
        const std::string kind = object_type(reinterpret_cast<HANDLE>(v));
        if (kind == "Event" || kind == "Semaphore" || kind == "Mutant" || kind == "Thread" || kind == "Process" ||
            kind == "Timer" || kind == "IoCompletion")
            return reinterpret_cast<HANDLE>(v);
    }
    return nullptr;
}

// The object behind a handle a thread waits on: "an Event (auto-reset, not set)", "a Thread (id 1234)", ...
inline std::string describe_handle(HANDLE handle) {
    using NtQueryEventFn = LONG(NTAPI*)(HANDLE, ULONG, PVOID, ULONG, PULONG);
    static const auto query_event = reinterpret_cast<NtQueryEventFn>(
        GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "NtQueryEvent"));
    if (!handle) return "an object whose handle was not found";
    const std::string kind = object_type(handle);
    if (kind.empty()) return "a handle that cannot be queried";
    std::string out = (std::strchr("AEIOU", kind[0]) ? "an " : "a ") + kind;
    if (kind == "Event" && query_event) {
        struct {
            ULONG type;
            LONG state;
        } event{};
        ULONG length = 0;
        if (query_event(handle, 0 /* EventBasicInformation */, &event, sizeof(event), &length) >= 0)
            out += std::string(" (") + (event.type == 1 ? "auto-reset" : "manual-reset") +
                   (event.state ? ", set)" : ", not set)");
    } else if (kind == "Thread") {
        out += " (id " + std::to_string(GetThreadId(handle)) + ")";
    }
    return out;
}

// For a thread stopped in an ntdll wait: which wait, and on what.
inline std::string describe_wait(const Captured& c, const std::string& top_symbol) {
    if (top_symbol == "NtWaitForSingleObject" || top_symbol == "ZwWaitForSingleObject")
        return "waiting in NtWaitForSingleObject on " + describe_handle(waited_handle(c));
    if (top_symbol == "NtWaitForMultipleObjects" || top_symbol == "ZwWaitForMultipleObjects") {
        const DWORD64 count = c.context.R10;
        std::string out = "waiting in NtWaitForMultipleObjects on " + std::to_string(count) + " handle(s)";
        const auto* handles = reinterpret_cast<const HANDLE*>(c.context.Rdx);
        if (count && count <= 8 && !IsBadReadPtr(handles, count * sizeof(HANDLE))) {
            out += ":";
            for (DWORD64 i = 0; i < count; ++i) out += (i ? ", " : " ") + describe_handle(handles[i]);
        }
        return out;
    }
    if (top_symbol == "NtWaitForAlertByThreadId") return "waiting on a lock or condition variable (NtWaitForAlertByThreadId)";
    if (top_symbol == "NtDelayExecution") return "sleeping (NtDelayExecution)";
    if (top_symbol.rfind("NtRemoveIoCompletion", 0) == 0) return "waiting on an I/O completion port";
    if (top_symbol == "NtWaitForWorkViaWorkerFactory") return "an idle thread-pool worker";
    if (top_symbol.rfind("Nt", 0) == 0 || top_symbol.rfind("Zw", 0) == 0) return "in " + top_symbol;
    return "running";
}

inline std::string take(const char* label, Namer namer) {
    static std::once_flag sym_once;
    std::call_once(sym_once, [] {
        SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS | SYMOPT_NO_PROMPTS);
        SymInitialize(GetCurrentProcess(), nullptr, TRUE);
    });
    SymRefreshModuleList(GetCurrentProcess());

    std::vector<DWORD> tids;
    const DWORD self = GetCurrentThreadId(), pid = GetCurrentProcessId(), walker = helper().tid;
    if (HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0); snap != INVALID_HANDLE_VALUE) {
        THREADENTRY32 te{sizeof(te)};
        for (BOOL more = Thread32First(snap, &te); more; more = Thread32Next(snap, &te))
            if (te.th32OwnerProcessID == pid && te.th32ThreadID != self && te.th32ThreadID != walker)
                tids.push_back(te.th32ThreadID);
        CloseHandle(snap);
    }
    std::vector<Captured> captured(tids.size());
    for (size_t i = 0; i < tids.size(); ++i) capture(tids[i], captured[i]);  // suspend, read, unwind, resume

    struct Group {
        std::vector<DWORD> tids;
        std::vector<std::string> names;
        std::string wait;
        const Captured* sample = nullptr;
    };
    std::map<std::string, Group> groups;  // keyed by the wait plus the frames
    size_t ok = 0;
    for (const Captured& c : captured) {
        if (!c.ok) continue;
        ++ok;
        const std::string top = c.count ? symbol_of(c.frames[0]) : std::string();
        std::string wait = c.unwind_timed_out ? std::string("stack not walked (the unwind timed out; thread resumed)")
                                              : describe_wait(c, top);
        std::string key = wait;
        for (int f = 0; f < c.count; ++f) key += ' ' + std::to_string(c.frames[f]);
        Group& g = groups[key];
        if (!g.sample) {
            g.sample = &c;
            g.wait = wait;
        }
        g.tids.push_back(c.tid);
        const std::string name = thread_name(c.tid);
        g.names.push_back(name.empty() ? "(unnamed)" : name);
    }
    std::vector<const Group*> ordered;
    for (const auto& [key, g] : groups) ordered.push_back(&g);
    std::stable_sort(ordered.begin(), ordered.end(),
                     [](const Group* a, const Group* b) { return a->tids.size() > b->tids.size(); });

    std::string out = std::string("[rt] snapshot ") + label + ": " + std::to_string(ok) + " threads, " +
                      std::to_string(groups.size()) + " distinct stacks\n";
    for (const Group* g : ordered) {
        std::map<std::string, int> name_count;
        for (const auto& n : g->names) ++name_count[n];
        std::string names;
        for (const auto& [n, k] : name_count) names += (names.empty() ? "" : ", ") + n + " x" + std::to_string(k);
        out += std::string("[rt] snapshot ") + label + ": " + std::to_string(g->tids.size()) + " thread(s) [" + names +
               "], " + g->wait + ":\n";
        for (int f = 0; f < g->sample->count; ++f) {
            const DWORD64 a = g->sample->frames[f];
            std::string line = "[rt]   #" + std::to_string(f) + " " + (namer ? namer(a) : std::to_string(a));
            if (const std::string s = symbol_of(a); !s.empty()) line += " " + s;
            out += line + "\n";
        }
    }
    return out;
}

}  // namespace detail

// Starts recording the names threads give themselves; call before the threads of interest start.
inline void install() {
    static void* veh = AddVectoredExceptionHandler(1, &detail::name_handler);
    (void)veh;
}

// Every other thread of the process at this moment, as [rt] lines (see the top of this file).
inline std::string take(const char* label, Namer namer) { return detail::take(label, namer); }

// LEONIDA_THREAD_SNAPSHOT=<seconds>[,<seconds>...]: a detached thread prints take() to stderr at those times after
// this call. Returns false (and does nothing) when the variable is unset or names no positive time.
inline bool start_from_env(Namer namer) {
    char* value = nullptr;
    size_t length = 0;
    if (_dupenv_s(&value, &length, "LEONIDA_THREAD_SNAPSHOT") != 0 || !value) return false;
    std::vector<double> times;
    for (char* p = value; *p;) {
        char* end = nullptr;
        const double t = std::strtod(p, &end);
        if (end == p) break;
        if (t > 0) times.push_back(t);
        p = *end == ',' ? end + 1 : end;
    }
    std::free(value);
    if (times.empty()) return false;
    std::sort(times.begin(), times.end());
    install();
    std::fprintf(stderr, "[rt] snapshot: %zu thread snapshot(s) armed (LEONIDA_THREAD_SNAPSHOT)\n", times.size());
    std::thread([times, namer] {
        const auto start = std::chrono::steady_clock::now();
        for (const double t : times) {
            std::this_thread::sleep_until(start + std::chrono::duration<double>(t));
            char label[32];
            std::snprintf(label, sizeof(label), "%g s", t);
            const std::string lines = take(label, namer);
            std::fwrite(lines.data(), 1, lines.size(), stderr);
            std::fflush(stderr);
        }
    }).detach();
    return true;
}

}  // namespace rt::thread_snapshot
