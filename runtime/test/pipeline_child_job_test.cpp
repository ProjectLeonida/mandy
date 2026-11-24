#include "pipeline_child_job.h"
#include "pipeline_isolation_retry.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cwchar>
#include <iterator>
#include <string>

using namespace leonida;

#define CHECK(expr) do { if (!(expr)) { std::fprintf(stderr, "line %d: %s (%s)\n", __LINE__, #expr, error.c_str()); return 1; } } while (0)

constexpr SIZE_T MiB = 1024 * 1024;
constexpr SIZE_T cap = 128 * MiB;
constexpr DWORD child_status = 0xc000008e;

bool spawn(const wchar_t* mode, HANDLE output, DWORD flags, PROCESS_INFORMATION& child) {
    wchar_t exe[32768];
    const DWORD len = GetModuleFileNameW(nullptr, exe, DWORD(std::size(exe)));
    if (!len || len == std::size(exe)) return false;
    std::wstring command = L"\"" + std::wstring(exe, len) + L"\" " + mode;
    STARTUPINFOW startup{};
    startup.cb = sizeof startup;
    startup.dwFlags = STARTF_USESTDHANDLES;
    startup.hStdOutput = output;
    startup.hStdError = output;
    startup.hStdInput = output;  // the synthetic child never reads stdin
    return CreateProcessW(exe, command.data(), nullptr, nullptr, TRUE, flags, nullptr, nullptr, &startup, &child);
}

int child_main(const char* mode) {
    if (std::strcmp(mode, "hold-pipe") == 0) {
        WaitForSingleObject(GetCurrentProcess(), 30000);
        return 0;
    }
    if (std::strcmp(mode, "plain") == 0) {
        std::puts("plain");
        std::fflush(stdout);
        ExitProcess(child_status);
    }
    BOOL member = FALSE;
    if (!IsProcessInJob(GetCurrentProcess(), nullptr, &member) || !member) return 2;
    if (std::strcmp(mode, "descendant") == 0) {
        PROCESS_INFORMATION child{};
        if (!spawn(L"hold-pipe", GetStdHandle(STD_OUTPUT_HANDLE), 0, child)) return 3;
        CloseHandle(child.hThread);
        CloseHandle(child.hProcess);
        std::printf("%lu\n", child.dwProcessId);
        return 0;
    }
    void* small = VirtualAlloc(nullptr, 16 * MiB, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (!small) return 4;
    const bool outer_denial = std::strcmp(mode, "outer-denial") == 0;
    if (std::strcmp(mode, "deny") == 0 || outer_denial) {
        // One finite request, even if enforcement regresses. Committed pages need not be touched.
        const SIZE_T request = outer_denial ? 65 * MiB : cap + MiB;
        void* large = VirtualAlloc(nullptr, request, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
        if (large) {
            VirtualFree(large, 0, MEM_RELEASE);
            VirtualFree(small, 0, MEM_RELEASE);
            return 5;
        }
        std::puts("allocation denied; recovered");
    } else {
        std::puts("committed");
    }
    return VirtualFree(small, 0, MEM_RELEASE) ? 0 : 6;
}

enum class Fault { none, assign, resume, pipe, pipe_and_terminate };

// Keep the original full-access handle through every assertion, even for faults
// injected with a restricted duplicate of that handle.
void reap_test_child(HANDLE process) {
    if (WaitForSingleObject(process, 0) != WAIT_OBJECT_0) {
        TerminateProcess(process, 1);
        if (WaitForSingleObject(process, 5000) != WAIT_OBJECT_0)
            std::fputs("test child cleanup could not be confirmed\n", stderr);
    }
    CloseHandle(process);
}

int run(const wchar_t* mode, bool guarded, Fault fault = Fault::none, HANDLE outer = nullptr) {
    std::string error;
    ChildHandle descendant(nullptr, CloseHandle);
    {
        PipelineChildJob job;
        if (guarded) CHECK(job.configure(cap, error));
        SECURITY_ATTRIBUTES inherited{sizeof inherited, nullptr, TRUE};
        HANDLE reader = nullptr, writer = nullptr;
        CHECK(CreatePipe(&reader, &writer, &inherited, 0));
        ChildHandle read_end(reader, CloseHandle), write_end(writer, CloseHandle);
        CHECK(SetHandleInformation(reader, HANDLE_FLAG_INHERIT, 0));
        PROCESS_INFORMATION child{};
        CHECK(spawn(mode, writer, guarded ? CREATE_SUSPENDED : 0, child));
        std::unique_ptr<void, decltype(&reap_test_child)> process(child.hProcess, reap_test_child);
        ChildHandle thread(child.hThread, CloseHandle);
        write_end.reset();
        if (outer && !AssignProcessToJobObject(outer, process.get())) {
            const DWORD code = GetLastError();
            // Some test hosts already impose an incompatible outer job policy.
            BOOL in_job = FALSE;
            CHECK(code == ERROR_ACCESS_DENIED && IsProcessInJob(GetCurrentProcess(), nullptr, &in_job) && in_job);
            TerminateProcess(process.get(), 1);
            CHECK(WaitForSingleObject(process.get(), 5000) == WAIT_OBJECT_0);
            std::puts("SKIP: compatible nesting unavailable under the host job policy");
            return 77;
        }
        if (fault == Fault::assign || fault == Fault::resume) {
            HANDLE limited = nullptr;
            if (fault == Fault::assign &&
                !DuplicateHandle(GetCurrentProcess(), process.get(), GetCurrentProcess(), &limited,
                                 PROCESS_TERMINATE | SYNCHRONIZE, FALSE, 0)) {
                error = "could not duplicate process handle";
                CHECK(false);
            }
            ChildHandle limited_process(limited, CloseHandle);
            if (fault == Fault::assign) {
                CHECK(!job.start(limited, thread.get(), error));
                CHECK(error == "AssignProcessToJobObject failed (Win32 error 5)");
            } else {
                CHECK(!job.start(process.get(), nullptr, error));
                CHECK(error == "ResumeThread failed (Win32 error 6)");
            }
            CHECK(WaitForSingleObject(process.get(), 0) == WAIT_OBJECT_0);
            char byte;
            DWORD read = 0;
            CHECK(!ReadFile(reader, &byte, 1, &read, nullptr) && GetLastError() == ERROR_BROKEN_PIPE);
            return 0;
        }
        if (!guarded) {
            CHECK(WaitForSingleObject(process.get(), 5000) == WAIT_OBJECT_0);
            DWORD status = 0;
            CHECK(GetExitCodeProcess(process.get(), &status) && status == child_status);
            char buffer[32]{};
            DWORD read = 0;
            CHECK(ReadFile(reader, buffer, sizeof buffer, &read, nullptr) && std::string(buffer, read) == "plain\r\n");
            return 0;
        }
        CHECK(job.start(process.get(), thread.get(), error));
        thread.reset();
        ChildJobReport report;
        std::string output;
        DWORD status = 0;
        if (fault == Fault::pipe || fault == Fault::pipe_and_terminate) {
            HANDLE limited = nullptr;
            if (fault == Fault::pipe_and_terminate)
                CHECK(DuplicateHandle(GetCurrentProcess(), process.get(), GetCurrentProcess(), &limited,
                                      SYNCHRONIZE, FALSE, 0));
            ChildHandle limited_process(limited, CloseHandle);
            CHECK(!job.collect(limited ? limited : process.get(), nullptr, output, status, report, error));
            CHECK(error == "PeekNamedPipe(child stdout) failed (Win32 error 6)");
            CHECK(WaitForSingleObject(process.get(), 0) == WAIT_OBJECT_0);
            CHECK(report.peak_bytes);
            return 0;
        }
        CHECK(job.collect(process.get(), reader, output, status, report, error));
        CHECK(status == 0);
        CHECK(report.peak_bytes && *report.peak_bytes > 0);
        std::printf("peak commit: %llu bytes (%s)\n",
                    static_cast<unsigned long long>(*report.peak_bytes), report.event_label());
        if (std::wcscmp(mode, L"descendant") == 0) {
            const auto pid = static_cast<DWORD>(std::strtoul(output.c_str(), nullptr, 10));
            CHECK(pid != 0);
            descendant.reset(OpenProcess(SYNCHRONIZE, FALSE, pid));
            CHECK(descendant && WaitForSingleObject(descendant.get(), 0) == WAIT_TIMEOUT);
        } else {
            CHECK(*report.peak_bytes >= 16 * MiB);
            const bool outer_denial = std::wcscmp(mode, L"outer-denial") == 0;
            const bool denied = std::wcscmp(mode, L"deny") == 0 || outer_denial;
            if (outer_denial || !denied) CHECK(*report.peak_bytes < cap);
            CHECK(output.find(denied ? "allocation denied; recovered" : "committed") != std::string::npos);
        }
    }
    if (descendant) CHECK(WaitForSingleObject(descendant.get(), 5000) == WAIT_OBJECT_0);
    return 0;
}

int check_retry(bool reproduce, bool abort) {
    std::string error;
    struct Outcome {
        bool ready = false;
        bool done = false;
        std::string runner_error;
        std::size_t count = 0;
    };
    bool setup_failed = false;
    std::size_t attempts = 0, committed = 0, recorded = 0, lost = 0;
    auto attempt = [&](std::size_t start, std::size_t count, std::string& reason) {
        ++attempts;
        // Even an interrupted batch can have partial output which must never be committed.
        const Outcome result{!abort, count == 1 && (!reproduce || start == 1), {}, count};
        const auto action = isolation_action(result, count);
        if (action == IsolationAction::complete) committed += result.count;
        if (action == IsolationAction::record) ++recorded;
        if (action == IsolationAction::split) reason = "interrupted";
        return action;
    };
    auto unreproduced = [&](std::size_t start, std::size_t count, const std::string& reason) {
        if (start == 0 && count == 2 && reason == "interrupted") ++lost;
    };
    const auto failures = isolate_range(0, 2, setup_failed, attempt, unreproduced);
    CHECK(setup_failed == abort);
    CHECK(attempts == (abort ? 1 : 3));
    CHECK(committed == (abort ? 0 : reproduce ? 1 : 2));
    CHECK(recorded == (reproduce ? 1 : 0));
    CHECK(lost == (!abort && !reproduce ? 1 : 0));
    CHECK(failures == (abort ? 0 : 1));
    CHECK(isolation_action(Outcome{true, true, "launcher failed", 2}, 2) == IsolationAction::abort);
    return 0;
}

int check_nested() {
    std::string error;
    ChildHandle outer(CreateJobObjectW(nullptr, nullptr), CloseHandle);
    CHECK(outer);
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
    limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_PROCESS_MEMORY;
    limits.ProcessMemoryLimit = 64 * MiB;
    CHECK(SetInformationJobObject(outer.get(), JobObjectExtendedLimitInformation, &limits, sizeof limits));
    return run(L"outer-denial", true, Fault::none, outer.get());
}

int main(int argc, char** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--nested-test") == 0) return check_nested();
    if (argc == 2) return child_main(argv[1]);
    std::string error;
    ChildJobReport report;
    CHECK(!PipelineChildJob::observe_message(1, 2, JOB_OBJECT_MSG_JOB_MEMORY_LIMIT, report));
    CHECK(!PipelineChildJob::observe_message(2, 2, JOB_OBJECT_MSG_PROCESS_MEMORY_LIMIT, report));
    CHECK(std::strcmp(report.event_label(), "job limit involvement unconfirmed") == 0);
    CHECK(PipelineChildJob::observe_message(2, 2, JOB_OBJECT_MSG_JOB_MEMORY_LIMIT, report));
    CHECK(std::strcmp(report.event_label(), "job memory limit event observed") == 0);
    CHECK(run(L"plain", false) == 0);
    CHECK(run(L"plain", true, Fault::assign) == 0);
    CHECK(run(L"plain", true, Fault::resume) == 0);
    CHECK(run(L"hold-pipe", true, Fault::pipe) == 0);
    CHECK(run(L"hold-pipe", true, Fault::pipe_and_terminate) == 0);
    CHECK(run(L"commit", true) == 0);
    CHECK(run(L"deny", true) == 0);
    CHECK(run(L"descendant", true) == 0);
    CHECK(check_retry(false, false) == 0);
    CHECK(check_retry(true, false) == 0);
    CHECK(check_retry(false, true) == 0);
    std::puts("pipeline_child_job: ok");
    return 0;
}
