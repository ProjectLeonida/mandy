// Optional out-of-process observer used by boot.ps1 on x64 Windows.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <dbghelp.h>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cwchar>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <string>

namespace fs = std::filesystem;

namespace {
struct Handle {
    HANDLE value;
    ~Handle() { if (value && value != INVALID_HANDLE_VALUE) CloseHandle(value); }
    Handle(const Handle&) = delete;
    Handle& operator=(const Handle&) = delete;
    explicit Handle(HANDLE handle) : value(handle) {}
};

void report_error(const char* operation, DWORD error = GetLastError()) {
    std::fprintf(stderr, "[crash-monitor] %s failed (%lu)\n", operation, error);
}

bool write_dump(const fs::path& dir, HANDLE process, DWORD pid, MINIDUMP_EXCEPTION_INFORMATION* info) {
    const auto partial = dir / L"crash.dmp.partial";
    {
        Handle file(CreateFileW(partial.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr));
        if (file.value == INVALID_HANDLE_VALUE) { report_error("create dump"); return false; }
        if (!MiniDumpWriteDump(process, pid, file.value, MiniDumpWithThreadInfo, info, nullptr, nullptr)) {
            report_error("MiniDumpWriteDump");
            return false;
        }
    }
    if (!MoveFileExW(partial.c_str(), (dir / L"crash.dmp").c_str(), MOVEFILE_WRITE_THROUGH)) {
        report_error("publish dump");
        return false;
    }
    return true;
}

void write_stack(std::ostream& out, HANDLE process, HANDLE thread, CONTEXT ctx, const fs::path& symbols) {
    // Only look beside the runtime. Ignore debugger environment paths and embedded build-machine paths.
    SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS | SYMOPT_FAIL_CRITICAL_ERRORS | SYMOPT_NO_PROMPTS |
                  SYMOPT_IGNORE_NT_SYMPATH | SYMOPT_IGNORE_CVREC | SYMOPT_DISABLE_SYMSRV_AUTODETECT);
    const bool loaded = SymInitializeW(process, symbols.c_str(), TRUE) != FALSE;
    STACKFRAME64 frame{};
    frame.AddrPC = {ctx.Rip, 0, AddrModeFlat};
    frame.AddrStack = {ctx.Rsp, 0, AddrModeFlat};
    frame.AddrFrame = {ctx.Rbp, 0, AddrModeFlat};
    for (unsigned n = 0; n < 64 && frame.AddrPC.Offset; ++n) {
        const auto pc = frame.AddrPC.Offset;
        const auto sp = frame.AddrStack.Offset;
        out << '#' << std::dec << n << " 0x" << std::hex << pc;
        IMAGEHLP_MODULE64 module{};
        module.SizeOfStruct = sizeof(module);
        if (loaded && SymGetModuleInfo64(process, pc, &module))
            out << ' ' << module.ModuleName << "+0x" << pc - module.BaseOfImage;
        alignas(SYMBOL_INFO) char storage[sizeof(SYMBOL_INFO) + MAX_SYM_NAME]{};
        auto* symbol = reinterpret_cast<SYMBOL_INFO*>(storage);
        symbol->SizeOfStruct = sizeof(SYMBOL_INFO);
        symbol->MaxNameLen = MAX_SYM_NAME;
        DWORD64 displacement = 0;
        if (loaded && SymFromAddr(process, pc, &displacement, symbol))
            out << " (" << symbol->Name << "+0x" << displacement << ')';
        out << '\n';
        if (!loaded || !StackWalk64(IMAGE_FILE_MACHINE_AMD64, process, thread, &frame, &ctx, nullptr,
                                    SymFunctionTableAccess64, SymGetModuleBase64, nullptr))
            break;
        if (frame.AddrPC.Offset == pc && frame.AddrStack.Offset == sp) {
            // The first StackWalk64 call can describe the initial frame again.
            if (n != 0 || !StackWalk64(IMAGE_FILE_MACHINE_AMD64, process, thread, &frame, &ctx, nullptr,
                                       SymFunctionTableAccess64, SymGetModuleBase64, nullptr)) break;
            if (frame.AddrPC.Offset == pc && frame.AddrStack.Offset == sp) break;
        }
    }
    if (loaded) SymCleanup(process);
}

void capture(const fs::path& dir, const fs::path& symbols, HANDLE process, const DEBUG_EVENT& event) {
    const bool exception = event.dwDebugEventCode == EXCEPTION_DEBUG_EVENT;
    const DWORD code = exception ? event.u.Exception.ExceptionRecord.ExceptionCode : event.u.ExitProcess.dwExitCode;
    Handle thread(OpenThread(THREAD_QUERY_INFORMATION | THREAD_GET_CONTEXT, FALSE, event.dwThreadId));
    CONTEXT ctx{};
    ctx.ContextFlags = CONTEXT_ALL;
    const bool got_context = thread.value && GetThreadContext(thread.value, &ctx);
    EXCEPTION_RECORD record{};
    if (exception) {
        record = event.u.Exception.ExceptionRecord;
        record.ExceptionRecord = nullptr; // Nested records belong to the child address space.
    }
    EXCEPTION_POINTERS pointers{&record, &ctx};
    MINIDUMP_EXCEPTION_INFORMATION info{event.dwThreadId, &pointers, FALSE};
    if (!write_dump(dir, process, event.dwProcessId, exception && got_context ? &info : nullptr))
        DeleteFileW((dir / L"crash.dmp.partial").c_str());

    std::ofstream stack(dir / L"stack.txt");
    stack << "kind=" << (exception ? "unhandled_exception" : "abnormal_exit")
          << "\ncode=0x" << std::hex << std::setw(8) << std::setfill('0') << code
          << "\nthread=" << std::dec << event.dwThreadId << '\n';
    if (got_context) write_stack(stack, process, thread.value, ctx, symbols);
    else stack << "thread context unavailable at this debug event\n";
    stack.close();
    if (!stack) report_error("write stack.txt", ERROR_WRITE_FAULT);
}

DWORD monitor(HANDLE process, const fs::path& dir, const fs::path& symbols, ULONGLONG deadline) {
    bool initial_break = true;
    bool captured = false;
    bool stopped = false;
    for (;;) {
        DEBUG_EVENT event{};
        if (!WaitForDebugEventEx(&event, 100)) {
            if (GetLastError() != ERROR_SEM_TIMEOUT) { report_error("WaitForDebugEventEx"); return 1; }
        } else {
            DWORD disposition = DBG_CONTINUE;
            const bool exiting = event.dwDebugEventCode == EXIT_PROCESS_DEBUG_EVENT;
            if (event.dwDebugEventCode == CREATE_PROCESS_DEBUG_EVENT && event.u.CreateProcessInfo.hFile)
                CloseHandle(event.u.CreateProcessInfo.hFile);
            if (event.dwDebugEventCode == LOAD_DLL_DEBUG_EVENT && event.u.LoadDll.hFile)
                CloseHandle(event.u.LoadDll.hFile);
            if (event.dwDebugEventCode == EXCEPTION_DEBUG_EVENT) {
                const auto& ex = event.u.Exception;
                if (initial_break && ex.dwFirstChance && ex.ExceptionRecord.ExceptionCode == EXCEPTION_BREAKPOINT)
                    initial_break = false; // The loader breakpoint belongs to the debugger.
                else {
                    disposition = DBG_EXCEPTION_NOT_HANDLED;
                    if (!ex.dwFirstChance && !stopped && !captured) {
                        capture(dir, symbols, process, event);
                        captured = true;
                    }
                }
            } else if (exiting && event.u.ExitProcess.dwExitCode && !stopped && !captured) {
                capture(dir, symbols, process, event);
            }
            if (!ContinueDebugEvent(event.dwProcessId, event.dwThreadId, disposition)) {
                report_error("ContinueDebugEvent");
                return 1;
            }
            if (exiting) {
                if (!stopped) return event.u.ExitProcess.dwExitCode;
                std::ofstream marker(dir / L"timed-out");
                marker.close();
                return marker ? 0 : 1;
            }
        }
        if (!captured && !stopped && GetTickCount64() >= deadline) {
            if (TerminateProcess(process, ERROR_TIMEOUT)) stopped = true;
            else if (GetLastError() != ERROR_ACCESS_DENIED) { report_error("TerminateProcess"); return 1; }
        }
    }
}
} // namespace

int wmain(int argc, wchar_t** argv) {
    if (argc != 6) {
        std::fputs("usage: gta6rt_crash_monitor REPO NEW_REPORT_DIR SECONDS RUNTIME CHILD_COMMAND_LINE\n", stderr);
        return 2;
    }
    wchar_t* end = nullptr;
    const long seconds = std::wcstol(argv[3], &end, 10);
    if (end == argv[3] || *end || errno == ERANGE || seconds < 0) return 2;
    try {
        const fs::path repo = fs::canonical(argv[1]);
        const fs::path dir = fs::weakly_canonical(fs::absolute(argv[2]));
        const fs::path runtime = fs::canonical(argv[4]);
        // Resolve junctions and compare file identity (Windows path case can differ).
        for (auto p = dir.parent_path();; p = p.parent_path()) {
            if (fs::equivalent(p, repo)) {
                std::fputs("[crash-monitor] private output must be outside the checkout\n", stderr);
                return 2;
            }
            if (p == p.root_path()) break;
        }
        if (!fs::create_directory(dir)) {
            std::fputs("[crash-monitor] report directory already exists\n", stderr);
            return 2;
        }
        STARTUPINFOW startup{};
        startup.cb = sizeof(startup);
        startup.dwFlags = STARTF_USESTDHANDLES;
        startup.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
        startup.hStdOutput = GetStdHandle(STD_OUTPUT_HANDLE);
        startup.hStdError = GetStdHandle(STD_ERROR_HANDLE);
        PROCESS_INFORMATION pi{};
        std::wstring command(argv[5]);
        if (!CreateProcessW(runtime.c_str(), command.data(), nullptr, nullptr, TRUE, DEBUG_ONLY_THIS_PROCESS,
                            nullptr, nullptr, &startup, &pi)) {
            report_error("CreateProcessW");
            return 2;
        }
        Handle process(pi.hProcess);
        Handle thread(pi.hThread);
        return static_cast<int>(monitor(process.value, dir, runtime.parent_path(), GetTickCount64() + seconds * 1000ULL));
    } catch (const fs::filesystem_error& e) {
        report_error("resolve private output or runtime", e.code().value());
        return 2;
    }
}
