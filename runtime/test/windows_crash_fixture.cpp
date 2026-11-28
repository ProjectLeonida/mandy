#include <windows.h>
#include <cstdio>
#include <cwchar>

constexpr DWORD test_exception = 0xe0421234;

__declspec(noinline) DWORD WINAPI crash_worker(void*) {
    RaiseException(test_exception, 0, 0, nullptr);
    return 0;
}

int wmain(int argc, wchar_t** argv) {
    wchar_t mode[32]{};
    GetEnvironmentVariableW(L"LEONIDA_TEST_CRASH_MODE", mode, 32);
    std::puts("fixture stdout");
    std::fputs("fixture stderr\n", stderr);
    std::fflush(stdout);
    if (wcscmp(mode, L"worker") == 0) {
        HANDLE worker = CreateThread(nullptr, 0, crash_worker, nullptr, 0, nullptr);
        if (!worker) return 4;
        WaitForSingleObject(worker, INFINITE);
        CloseHandle(worker);
    } else if (wcscmp(mode, L"handled") == 0) {
        __try { RaiseException(test_exception, 0, 0, nullptr); }
        __except (GetExceptionCode() == test_exception ? EXCEPTION_EXECUTE_HANDLER : EXCEPTION_CONTINUE_SEARCH) {}
    } else if (wcscmp(mode, L"break") == 0) {
        DebugBreak();
    } else if (wcscmp(mode, L"exit") == 0) {
        ExitProcess(23);
    } else if (wcscmp(mode, L"numeric") == 0) {
        ExitProcess(EXCEPTION_ACCESS_VIOLATION);
    } else if (wcscmp(mode, L"timer") == 0) {
        Sleep(30000);
    } else if (wcscmp(mode, L"argv") == 0) {
        return argc == 3 && wcscmp(argv[1], L"a b\\") == 0 && wcscmp(argv[2], L"a\"b") == 0 ? 0 : 5;
    }
    return 0;
}
