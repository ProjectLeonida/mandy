#pragma once

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <algorithm>
#include <exception>
#include <memory>
#include <optional>
#include <string>

namespace leonida {

using ChildHandle = std::unique_ptr<void, decltype(&CloseHandle)>;

struct ChildJobReport {
    std::optional<SIZE_T> peak_bytes;
    DWORD peak_error = 0;
    bool limit_event = false;

    const char* event_label() const {
        return limit_event ? "job memory limit event observed" : "job limit involvement unconfirmed";
    }
};

class PipelineChildJob {
public:
    bool configure(SIZE_T limit_bytes, std::string& error) {
        port_.reset(CreateIoCompletionPort(INVALID_HANDLE_VALUE, nullptr, 0, 1));
        if (!port_) return fail(error, "CreateIoCompletionPort");
        job_.reset(CreateJobObjectW(nullptr, nullptr));
        if (!job_) return fail(error, "CreateJobObjectW");
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
        limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_JOB_MEMORY | JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        limits.JobMemoryLimit = limit_bytes;
        if (!SetInformationJobObject(job_.get(), JobObjectExtendedLimitInformation, &limits, sizeof limits))
            return fail(error, "SetInformationJobObject(memory limit)");
        JOBOBJECT_ASSOCIATE_COMPLETION_PORT association{this, port_.get()};
        if (!SetInformationJobObject(job_.get(), JobObjectAssociateCompletionPortInformation,
                                     &association, sizeof association))
            return fail(error, "SetInformationJobObject(completion port)");
        return true;
    }

    static bool observe_message(ULONG_PTR key, ULONG_PTR job_key, DWORD message, ChildJobReport& report) {
        if (key != job_key || message != JOB_OBJECT_MSG_JOB_MEMORY_LIMIT) return false;
        report.limit_event = true;
        return true;
    }

    bool start(HANDLE process, HANDLE thread, std::string& error) {
        if (!AssignProcessToJobObject(job_.get(), process))
            return stop(process, error, "AssignProcessToJobObject");
        if (ResumeThread(thread) == DWORD(-1))
            return stop(process, error, "ResumeThread");
        return true;
    }

    bool collect(HANDLE process, HANDLE pipe, std::string& out, DWORD& status,
                 ChildJobReport& report, std::string& error) {
        bool eof = false;
        for (;;) {
            const DWORD wait = WaitForSingleObject(process, eof ? INFINITE : 20);
            if (wait != WAIT_OBJECT_0 && wait != WAIT_TIMEOUT)
                return stop(process, error, "WaitForSingleObject", &report);
            drain_messages(report);
            DWORD available = 0;
            if (!eof && !PeekNamedPipe(pipe, nullptr, 0, nullptr, &available, nullptr)) {
                if (GetLastError() != ERROR_BROKEN_PIPE)
                    return stop(process, error, "PeekNamedPipe(child stdout)", &report);
                eof = true;
            }
            // Read only bytes reported available. After the primary exits, consume one snapshot
            // and return even if a descendant still holds or writes to the pipe.
            while (available) {
                char buffer[4096];
                DWORD read = 0;
                if (!ReadFile(pipe, buffer, (std::min)(available, DWORD(sizeof buffer)), &read, nullptr))
                    return stop(process, error, "ReadFile(child stdout)", &report);
                if (!read) break;
                try {
                    out.append(buffer, read);
                } catch (const std::exception&) {
                    return stop(process, error, "capturing child stdout", &report, ERROR_SUCCESS);
                }
                available -= read;
            }
            if (wait == WAIT_OBJECT_0) break;
        }
        const bool exited = GetExitCodeProcess(process, &status) != FALSE;
        if (!exited) fail(error, "GetExitCodeProcess");
        snapshot(report);
        return exited;
    }

private:
    static bool fail(std::string& error, const char* api) {
        const DWORD code = GetLastError();
        error = std::string(api) + " failed (Win32 error " + std::to_string(code) + ")";
        return false;
    }

    bool stop(HANDLE process, std::string& error, const char* api, ChildJobReport* report = nullptr,
              DWORD code = GetLastError()) {
        // The process may not have joined the job, so terminate it explicitly.
        const BOOL terminated = TerminateProcess(process, 1);
        const DWORD terminate_error = terminated ? 0 : GetLastError();
        if (!terminated && report) snapshot(*report);
        if (!terminated) job_.reset();
        // If explicit termination failed after assignment, allow kill-on-close to finish.
        const DWORD wait = WaitForSingleObject(process, 5000);
        const DWORD wait_error = wait == WAIT_FAILED ? GetLastError() : 0;
        if (terminated && report) snapshot(*report);
        job_.reset();
        if (code != ERROR_SUCCESS) {
            SetLastError(code);
            fail(error, api);
        } else {
            error = std::string(api) + " failed";
        }
        if (wait != WAIT_OBJECT_0) {
            error += "; child cleanup could not be confirmed";
            if (terminate_error)
                error += "; TerminateProcess failed (Win32 error " + std::to_string(terminate_error) + ")";
            if (wait == WAIT_TIMEOUT)
                error += "; cleanup wait timed out";
            if (wait_error)
                error += "; cleanup wait failed (Win32 error " + std::to_string(wait_error) + ")";
        }
        return false;
    }

    void snapshot(ChildJobReport& report) {
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
        if (QueryInformationJobObject(job_.get(), JobObjectExtendedLimitInformation, &limits, sizeof limits, nullptr))
            report.peak_bytes = limits.PeakJobMemoryUsed;
        else
            report.peak_error = GetLastError();
        drain_messages(report);
    }

    void drain_messages(ChildJobReport& report) {
        // Hard-limit notifications are best-effort. Never wait for one or infer anything from its absence.
        for (unsigned i = 0; i < 64; ++i) {
            DWORD message = 0;
            ULONG_PTR key = 0;
            OVERLAPPED* value = nullptr;
            if (!GetQueuedCompletionStatus(port_.get(), &message, &key, &value, 0)) break;
            observe_message(key, reinterpret_cast<ULONG_PTR>(this), message, report);
        }
    }

    ChildHandle port_{nullptr, CloseHandle};
    ChildHandle job_{nullptr, CloseHandle};
};

}  // namespace leonida
