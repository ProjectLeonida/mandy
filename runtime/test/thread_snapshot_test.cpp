// thread_snapshot.h: a snapshot names each waiting thread, the object it waits on and its frames, and leaves every
// thread running afterwards. Three waiters: one on an auto-reset event named with SetThreadDescription, one on a
// manual-reset event named with the MS_VC_EXCEPTION protocol, one joining another thread. Exit 0 when every check
// passes, 1 otherwise.
#include <windows.h>

#include <atomic>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>

#include "thread_snapshot.h"

namespace {

int failures = 0;

void check(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++failures;
}

std::string module_of(std::uint64_t address) {
    HMODULE m = nullptr;
    char name[MAX_PATH] = "?";
    if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           reinterpret_cast<LPCSTR>(address), &m)) {
        GetModuleFileNameA(m, name, MAX_PATH);
        const char* base = std::strrchr(name, '\\');
        char out[MAX_PATH + 32];
        std::snprintf(out, sizeof(out), "host:%s+0x%llx", base ? base + 1 : name,
                      static_cast<unsigned long long>(address - reinterpret_cast<std::uint64_t>(m)));
        return out;
    }
    return "unknown";
}

void name_with_exception(const char* name) {
#pragma pack(push, 8)
    struct {
        DWORD type;
        LPCSTR name;
        DWORD thread_id;
        DWORD flags;
    } info{0x1000, name, 0xFFFFFFFF, 0};
#pragma pack(pop)
    __try {
        RaiseException(0x406D1388, 0, sizeof(info) / sizeof(ULONG_PTR), reinterpret_cast<ULONG_PTR*>(&info));
    } __except (EXCEPTION_EXECUTE_HANDLER) {
    }
}

}  // namespace

int main() {
    rt::thread_snapshot::install();
    HANDLE auto_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    HANDLE manual_event = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    std::atomic<int> started{0};

    std::thread a([&] {
        SetThreadDescription(GetCurrentThread(), L"snap-auto");
        ++started;
        WaitForSingleObject(auto_event, INFINITE);
    });
    std::thread b([&] {
        name_with_exception("snap-manual");
        ++started;
        WaitForSingleObject(manual_event, INFINITE);
    });
    HANDLE b_handle = OpenThread(SYNCHRONIZE | THREAD_QUERY_LIMITED_INFORMATION, FALSE, GetThreadId(b.native_handle()));
    std::thread c([&] {
        SetThreadDescription(GetCurrentThread(), L"snap-joiner");
        ++started;
        WaitForSingleObject(b_handle, INFINITE);
    });
    while (started < 3) Sleep(10);
    Sleep(200);  // all three inside their waits

    const std::string out = rt::thread_snapshot::take("test", &module_of);
    std::fputs(out.c_str(), stdout);

    check(out.find("[rt] snapshot test:") != std::string::npos, "a header line");
    check(out.find("[snap-auto x1], waiting in NtWaitForSingleObject on an Event (auto-reset, not set)") !=
              std::string::npos,
          "the SetThreadDescription thread waits on an auto-reset event");
    check(out.find("[snap-manual x1], waiting in NtWaitForSingleObject on an Event (manual-reset, not set)") !=
              std::string::npos,
          "the MS_VC_EXCEPTION-named thread waits on a manual-reset event");
    check(out.find("[snap-joiner x1], waiting in NtWaitForSingleObject on a Thread (id " +
                   std::to_string(GetThreadId(b.native_handle())) + ")") != std::string::npos,
          "the joiner waits on the other thread");
    check(out.find("NtWaitForSingleObject\n") != std::string::npos ||
              out.find(" NtWaitForSingleObject") != std::string::npos,
          "frames carry the ntdll symbol");

    // Every thread was resumed: the waits complete once signalled.
    SetEvent(auto_event);
    SetEvent(manual_event);
    a.join();
    b.join();
    c.join();
    check(true, "all three threads ran to completion after the snapshot");

    CloseHandle(b_handle);
    CloseHandle(auto_event);
    CloseHandle(manual_event);
    return failures ? 1 : 0;
}
