#include "thread_stacks.h"

#include <windows.h>
#include <intrin.h>
#include <tlhelp32.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>

#include "crash_report.h"
#include "../shims/shims.h"
#include "../shims/startup_trace.h"

namespace rt {
namespace {

using Clock = std::chrono::steady_clock;
using GetDescription = HRESULT(WINAPI*)(HANDLE, PWSTR*);
using QueryThread = LONG(NTAPI*)(HANDLE, int, void*, ULONG, ULONG*);

struct CopiedStack {
    DWORD64 copy = 0;      // Where the bytes are now.
    DWORD64 original = 0;  // Where they were in the thread: [original, original + size).
    size_t size = 0;
};

// Suspended section: system calls and a copy only.
bool copy_context_and_stack(HANDLE thread, CONTEXT& context, std::byte* buffer, CopiedStack& stack) {
    if (SuspendThread(thread) == DWORD(-1)) return false;
    context.ContextFlags = CONTEXT_FULL;
    MEMORY_BASIC_INFORMATION region{};
    bool copied = false;
    if (GetThreadContext(thread, &context) &&
        VirtualQuery(reinterpret_cast<void*>(context.Rsp), &region, sizeof(region)) &&
        region.State == MEM_COMMIT && !(region.Protect & (PAGE_GUARD | PAGE_NOACCESS))) {
        // From RSP (rounded down to keep the 16-byte alignment the copy buffer also has) to the top of the committed
        // run of stack pages, at most kMaxStackBytes.
        stack.original = context.Rsp & ~DWORD64(15);
        const DWORD64 top = reinterpret_cast<DWORD64>(region.BaseAddress) + region.RegionSize;
        stack.size = static_cast<size_t>(std::min<DWORD64>(StackSampler::kMaxStackBytes, top - stack.original));
        SIZE_T read = 0;
        copied = ReadProcessMemory(GetCurrentProcess(), reinterpret_cast<void*>(stack.original), buffer, stack.size,
                                   &read) && read == stack.size;
    }
    ResumeThread(thread);
    stack.copy = reinterpret_cast<DWORD64>(buffer);
    return copied;
}

// A frame's code can be unloaded after its thread resumes and a damaged stack can hand the unwinder a bad pointer:
// both are access violations here, and end the walk.
bool unwind_frame(CONTEXT& context) {
    __try {
        DWORD64 image_base = 0;
        if (const auto* entry = RtlLookupFunctionEntry(context.Rip, &image_base, nullptr)) {
            void* handler_data = nullptr;
            DWORD64 establisher = 0;
            RtlVirtualUnwind(UNW_FLAG_NHANDLER, image_base, context.Rip, const_cast<PRUNTIME_FUNCTION>(entry),
                             &context, &handler_data, &establisher, nullptr);
        } else {
            // No function entry: a leaf (or a thunk without unwind data) that has not moved RSP.
            context.Rip = *reinterpret_cast<const DWORD64*>(context.Rsp);
            context.Rsp += sizeof(DWORD64);
        }
    } __except (GetExceptionCode() == EXCEPTION_ACCESS_VIOLATION || GetExceptionCode() == EXCEPTION_IN_PAGE_ERROR
                    ? EXCEPTION_EXECUTE_HANDLER : EXCEPTION_CONTINUE_SEARCH) {
        return false;
    }
    return true;
}

size_t walk_copy(CONTEXT& context, const CopiedStack& stack, std::uint64_t* frames) {
    // An address in the copied range now names the same word in the copy.
    const auto rebase = [&](DWORD64& address) {
        if (address >= stack.original && address - stack.original < stack.size) address += stack.copy - stack.original;
    };
    const auto in_copy = [&](DWORD64 address) {
        return address >= stack.copy && address - stack.copy + sizeof(DWORD64) <= stack.size;
    };
    size_t count = 0;
    rebase(context.Rsp);
    while (context.Rip && count < StackSampler::kMaxFrames) {
        frames[count++] = context.Rip;
        if (!in_copy(context.Rsp)) break;
        const DWORD64 rsp = context.Rsp;
        // Any nonvolatile register can be a frame pointer, including one just restored from the copy's bytes.
        for (DWORD64* reg : {&context.Rbx, &context.Rbp, &context.Rsi, &context.Rdi,
                             &context.R12, &context.R13, &context.R14, &context.R15})
            rebase(*reg);
        if (!unwind_frame(context)) break;
        rebase(context.Rsp);  // A machine frame restores RSP from the copy as well.
        if (context.Rsp <= rsp) break;
    }
    return count;
}

std::string thread_label(HANDLE thread, GetDescription describe, QueryThread query) {
    if (describe) {
        PWSTR description = nullptr;
        if (SUCCEEDED(describe(thread, &description)) && description) {
            std::string name = startup_trace::detail::utf8_from_wide(description, static_cast<int>(wcslen(description)));
            LocalFree(description);
            if (!name.empty()) return name;
        }
    }
    constexpr int kWin32StartAddress = 9;  // ThreadQuerySetWin32StartAddress
    DWORD64 start = 0;
    if (query && query(thread, kWin32StartAddress, &start, sizeof(start), nullptr) >= 0 && start)
        return "start " + shims::module_offset(start);
    return "start unknown";
}

struct ThreadTrace {
    DWORD tid = 0;
    std::string label;
    std::array<std::uint64_t, StackSampler::kMaxFrames> frames{};
    size_t count = 0;
};

// `also`, when given, receives the same text (the crash log's copy of a fatal report).
void print_traces(const std::string& tag, const std::vector<ThreadTrace>& traces, std::string* also = nullptr) {
    std::string out = "[rt] stacks " + tag + ": " + std::to_string(traces.size()) + " threads\n";
    for (auto it = traces.begin(); it != traces.end(); ++it) {
        out += "[rt] stacks " + tag + " thread " + std::to_string(it->tid) + " (" + it->label + "):\n";
        const auto same = std::find_if(traces.begin(), it, [&](const ThreadTrace& earlier) {
            return it->count && it->count == earlier.count &&
                   std::equal(it->frames.begin(), it->frames.begin() + it->count, earlier.frames.begin());
        });
        if (same != it) {
            out += "   same as thread " + std::to_string(same->tid) + "\n";
        } else if (!it->count) {
            out += "   stack unavailable\n";
        } else {
            for (size_t i = 0; i < it->count; ++i)
                out += "   #" + std::to_string(i) + " " + shims::module_offset(it->frames[i]) + "\n";
        }
    }
    std::fputs(out.c_str(), stderr);
    std::fflush(stderr);
    if (also) *also += out;
}

void snapshot_threads(const std::string& tag, StackSampler& sampler, GetDescription describe, QueryThread query,
                      DWORD only = 0, std::string* also = nullptr) {
    const HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
    if (snapshot == INVALID_HANDLE_VALUE) {
        std::fprintf(stderr, "[rt] stacks %s: thread list failed (error %lu)\n", tag.c_str(), GetLastError());
        return;
    }
    std::vector<ThreadTrace> traces;
    traces.reserve(512);
    const DWORD pid = GetCurrentProcessId();
    const DWORD self = GetCurrentThreadId();
    THREADENTRY32 entry{};
    entry.dwSize = sizeof(entry);
    for (BOOL ok = Thread32First(snapshot, &entry); ok; ok = Thread32Next(snapshot, &entry)) {
        entry.dwSize = sizeof(entry);
        if (entry.th32OwnerProcessID != pid || entry.th32ThreadID == self) continue;
        if (only && entry.th32ThreadID != only) continue;
        const HANDLE thread = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION,
                                         FALSE, entry.th32ThreadID);
        if (!thread) continue;
        if (GetProcessIdOfThread(thread) == pid) {  // The listed thread may have exited and its ID been reused.
            ThreadTrace& trace = traces.emplace_back();
            trace.tid = entry.th32ThreadID;
            trace.label = thread_label(thread, describe, query);  // Before sampling: it allocates.
            trace.count = sampler.sample(thread, trace.frames.data());
        }
        CloseHandle(thread);
    }
    CloseHandle(snapshot);
    print_traces(tag, traces, also);
}

// ---- The crash log (RT-CRASH-LOG, src/crash_report.h) ----

// CPUID's 48-byte brand string, e.g. "AMD Ryzen 9 7900X 12-Core Processor"; empty if the CPU has none.
std::string cpu_brand() {
    int regs[4]{};
    __cpuid(regs, 0x80000000);
    if (static_cast<unsigned>(regs[0]) < 0x80000004u) return {};
    char brand[49]{};
    for (int i = 0; i < 3; ++i) {
        __cpuid(regs, 0x80000002 + i);
        std::memcpy(brand + 16 * i, regs, sizeof regs);
    }
    return brand;
}

// The running Windows, from RtlGetVersion: GetVersionEx reports what the executable's manifest declares instead.
void windows_version(std::uint32_t& major, std::uint32_t& minor, std::uint32_t& build) {
    major = minor = build = 0;
    using GetVersion = LONG(WINAPI*)(OSVERSIONINFOW*);
    const auto get = reinterpret_cast<GetVersion>(
        reinterpret_cast<void*>(GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "RtlGetVersion")));
    OSVERSIONINFOW info{sizeof info};
    if (get && get(&info) >= 0) {
        major = info.dwMajorVersion;
        minor = info.dwMinorVersion;
        build = info.dwBuildNumber;
    }
}

// The mangled name of the type a C++ throw raised (".?AVruntime_error@std@@"), read through the throw info MSVC
// passes in the exception's parameters: ThrowInfo -> CatchableTypeArray -> its first CatchableType -> TypeDescriptor,
// every pointer an RVA from the image base in parameter 3. No C++ objects here: __try cannot unwind them.
bool read_thrown_type(const EXCEPTION_RECORD& record, char* name, std::size_t size) {
    if (record.ExceptionCode != crash_report::kCppException || record.NumberParameters < 4 || size == 0) return false;
    const auto base = static_cast<std::uintptr_t>(record.ExceptionInformation[3]);
    const auto throw_info = static_cast<std::uintptr_t>(record.ExceptionInformation[2]);
    if (!base || !throw_info) return false;
    name[0] = '\0';
    __try {
        // ThrowInfo: attributes, pmfnUnwind, pForwardCompat, pCatchableTypeArray (32-bit RVAs on x64).
        const auto array = base + *reinterpret_cast<const std::int32_t*>(throw_info + 12);
        // CatchableTypeArray: nCatchableTypes, then arrayOfCatchableTypes[] as RVAs.
        if (*reinterpret_cast<const std::int32_t*>(array) < 1) return false;
        const auto type = base + *reinterpret_cast<const std::int32_t*>(array + 4);
        // CatchableType: properties, pType (an RVA to a TypeDescriptor: vftable, spare, then the name).
        const auto descriptor = base + *reinterpret_cast<const std::int32_t*>(type + 4);
        const char* text = reinterpret_cast<const char*>(descriptor + 2 * sizeof(void*));
        std::size_t i = 0;
        for (; i + 1 < size && text[i]; ++i) name[i] = text[i];
        name[i] = '\0';
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        name[0] = '\0';
        return false;
    }
    return name[0] != '\0';
}

std::string thrown_type(const EXCEPTION_RECORD& record) {
    char name[128];
    return read_thrown_type(record, name, sizeof name) ? std::string(name) : std::string();
}

// Where reports are appended: LEONIDA_CRASH_LOG, else %LOCALAPPDATA%\Leonida\crashes.log; empty for none.
std::wstring crash_log_path() {
    wchar_t buffer[MAX_PATH * 2];
    DWORD n = GetEnvironmentVariableW(L"LEONIDA_CRASH_LOG", buffer, static_cast<DWORD>(std::size(buffer)));
    if (n >= std::size(buffer)) return {};
    std::string narrow;
    for (DWORD i = 0; i < n; ++i) narrow += buffer[i] < 0x80 ? static_cast<char>(buffer[i]) : '?';
    switch (crash_report::log_choice(narrow)) {
    case crash_report::LogChoice::off: return {};
    case crash_report::LogChoice::path: return std::wstring(buffer, n);
    case crash_report::LogChoice::default_file: break;
    }
    n = GetEnvironmentVariableW(L"LOCALAPPDATA", buffer, static_cast<DWORD>(std::size(buffer)));
    if (!n || n >= std::size(buffer)) return {};
    std::wstring dir = std::wstring(buffer, n) + L"\\Leonida";
    CreateDirectoryW(dir.c_str(), nullptr);  // already there is fine; a failure shows as the open below failing
    return dir + L"\\crashes.log";
}

// Appends `text` to the crash log with Win32 calls only, flushed, so the report outlives the console and the
// process. Says on stderr where it went, or why it did not.
void append_crash_log(const std::string& text) {
    const std::wstring path = crash_log_path();
    if (path.empty()) return;
    const HANDLE file = CreateFileW(path.c_str(), FILE_APPEND_DATA, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                                    OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) {
        std::fprintf(stderr, "[rt] crash log: could not open %ls (error %lu)\n", path.c_str(), GetLastError());
        return;
    }
    DWORD written = 0;
    const bool ok = WriteFile(file, text.data(), static_cast<DWORD>(text.size()), &written, nullptr) &&
                    written == text.size();
    FlushFileBuffers(file);
    CloseHandle(file);
    if (ok) std::fprintf(stderr, "[rt] crash log: report appended to %ls\n", path.c_str());
    else std::fprintf(stderr, "[rt] crash log: writing %ls failed (error %lu)\n", path.c_str(), GetLastError());
}

}  // namespace

StackSampler::StackSampler() : stack_(std::make_unique_for_overwrite<std::byte[]>(kMaxStackBytes)) {}

std::size_t StackSampler::sample(void* thread, std::uint64_t* frames) {
    CONTEXT context{};
    CopiedStack stack;
    if (!copy_context_and_stack(static_cast<HANDLE>(thread), context, stack_.get(), stack)) {
        if (!context.Rip) return 0;
        frames[0] = context.Rip;  // The context without its stack: the innermost frame only.
        return 1;
    }
    return walk_copy(context, stack, frames);
}

bool parse_stack_times(std::string_view text, std::vector<unsigned>& seconds) {
    seconds.clear();
    for (;;) {
        const auto comma = text.find(',');
        const auto token = text.substr(0, comma);
        unsigned value = 0;
        const auto [end, error] = std::from_chars(token.data(), token.data() + token.size(), value);
        if (token.empty() || error != std::errc{} || end != token.data() + token.size()) {
            seconds.clear();
            return false;
        }
        seconds.push_back(value);
        if (comma == std::string_view::npos) break;
        text.remove_prefix(comma + 1);
    }
    std::sort(seconds.begin(), seconds.end());
    seconds.erase(std::unique(seconds.begin(), seconds.end()), seconds.end());
    return true;
}

void print_thread_stacks_now(std::uint32_t tid, const char* why) noexcept {
    static std::mutex mutex;
    try {
        std::lock_guard lock(mutex);
        static StackSampler sampler;
        const auto describe = reinterpret_cast<GetDescription>(reinterpret_cast<void*>(
            GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "GetThreadDescription")));
        const auto query = reinterpret_cast<QueryThread>(reinterpret_cast<void*>(
            GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "NtQueryInformationThread")));
        snapshot_threads(why ? why : "now", sampler, describe, query, tid);
    } catch (...) {
    }
}

void print_fault_stacks(const EXCEPTION_RECORD* record, const CONTEXT* context, const char* why) noexcept {
    static std::mutex mutex;
    try {
        const std::lock_guard lock(mutex);
        const char* tag = why ? why : "fatal";
        const std::string prefix = "[rt] fault " + std::string(tag) + ": ";
        // Everything printed below, kept for the crash log as well (RT-CRASH-LOG).
        std::string report;
        const auto emit = [&report](const std::string& text) {
            std::fputs(text.c_str(), stderr);
            report += text;
        };
        {
            SYSTEMTIME now{};
            GetSystemTime(&now);
            report += crash_report::log_header(now.wYear, now.wMonth, now.wDay, now.wHour, now.wMinute, now.wSecond,
                                               GetCurrentProcessId());
        }
        if (record) {
            // ExceptionAddress is where the faulting instruction is, which for a fault raised by an instruction in a
            // leaf without unwind data can name nothing; the chain below says what called it either way.
            const std::string at = record->ExceptionAddress
                                       ? shims::module_offset(reinterpret_cast<uint64_t>(record->ExceptionAddress))
                                       : std::string("unknown");
            char line[512];
            if (record->ExceptionCode == EXCEPTION_ACCESS_VIOLATION && record->NumberParameters >= 2) {
                const char* access = record->ExceptionInformation[0] == 0 ? "read"
                                   : record->ExceptionInformation[0] == 1 ? "write" : "execute";
                std::snprintf(line, sizeof line, "%s%s of 0x%llx at %s\n", prefix.c_str(), access,
                              static_cast<unsigned long long>(record->ExceptionInformation[1]), at.c_str());
                emit(line);
            } else {
                std::snprintf(line, sizeof line, "%sexception 0x%08lx at %s\n", prefix.c_str(),
                              static_cast<unsigned long>(record->ExceptionCode), at.c_str());
                emit(line);
                // What its parameters say: a C++ throw's type, an in-page error's status, or the raw words.
                std::uint64_t params[EXCEPTION_MAXIMUM_PARAMETERS]{};
                const DWORD n = (std::min<DWORD>)(record->NumberParameters, EXCEPTION_MAXIMUM_PARAMETERS);
                for (DWORD i = 0; i < n; ++i) params[i] = record->ExceptionInformation[i];
                emit(crash_report::exception_detail(record->ExceptionCode, n, params, thrown_type(*record), prefix));
            }
        }
        if (context) {
            crash_report::Registers r;
            r.rax = context->Rax, r.rbx = context->Rbx, r.rcx = context->Rcx, r.rdx = context->Rdx;
            r.rsi = context->Rsi, r.rdi = context->Rdi, r.rbp = context->Rbp, r.rsp = context->Rsp;
            r.r8 = context->R8, r.r9 = context->R9, r.r10 = context->R10, r.r11 = context->R11;
            r.r12 = context->R12, r.r13 = context->R13, r.r14 = context->R14, r.r15 = context->R15;
            r.rip = context->Rip, r.eflags = context->EFlags;
            emit(crash_report::registers_lines(r, prefix));
        }
        {
            std::uint32_t major, minor, build;
            windows_version(major, minor, build);
            emit(crash_report::host_line(cpu_brand(), major, minor, build, prefix));
        }
        std::fflush(stderr);
        // The filter runs on the faulting thread and its stack is live under us, so unlike a sampled thread this one
        // unwinds directly, with no copy (the same frames the suspended thread's own unwinder would walk).
        if (context && context->Rip) {
            CONTEXT walk = *context;
            std::array<std::uint64_t, StackSampler::kMaxFrames> frames{};
            std::size_t count = 0;
            while (walk.Rip && count < frames.size()) {
                frames[count++] = walk.Rip;
                if (!unwind_frame(walk)) break;
            }
            std::string out = "[rt] stacks " + std::string(tag) + " thread " + std::to_string(GetCurrentThreadId()) +
                              " (faulting thread):\n";
            if (!count) out += "   stack unavailable\n";
            else
                for (std::size_t i = 0; i < count; ++i)
                    out += "   #" + std::to_string(i) + " " + shims::module_offset(frames[i]) + "\n";
            emit(out);
            std::fflush(stderr);
        }
        static StackSampler sampler;
        const auto describe = reinterpret_cast<GetDescription>(reinterpret_cast<void*>(
            GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "GetThreadDescription")));
        const auto query = reinterpret_cast<QueryThread>(reinterpret_cast<void*>(
            GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "NtQueryInformationThread")));
        snapshot_threads(tag, sampler, describe, query, 0, &report);
        append_crash_log(report);
    } catch (...) {
    }
}

std::jthread start_thread_stacks() {
    const auto start = Clock::now();
    char spec[1024];
    const DWORD length = GetEnvironmentVariableA("LEONIDA_STACKS", spec, sizeof(spec));
    if (!length) return {};
    std::vector<unsigned> times;
    if (length >= sizeof(spec) || !parse_stack_times(std::string_view(spec, length), times)) {
        std::fprintf(stderr, "[rt] LEONIDA_STACKS: expected comma-separated whole seconds\n");
        return {};
    }
    return std::jthread([start, times = std::move(times)](std::stop_token stop) {
        const auto describe = reinterpret_cast<GetDescription>(reinterpret_cast<void*>(
            GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "GetThreadDescription")));
        const auto query = reinterpret_cast<QueryThread>(reinterpret_cast<void*>(
            GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "NtQueryInformationThread")));
        StackSampler sampler;
        std::condition_variable_any wake;
        std::mutex mutex;
        std::unique_lock lock(mutex);
        for (const unsigned at : times) {
            wake.wait_until(lock, stop, start + std::chrono::seconds(at), [] { return false; });
            if (stop.stop_requested()) break;
            snapshot_threads("t=" + std::to_string(at), sampler, describe, query);
        }
    });
}

}  // namespace rt
