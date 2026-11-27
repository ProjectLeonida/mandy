// Checks GetModuleFileNameA/W and GetModuleHandleA/W(nullptr) through shims/library_loader.cpp against host controls.
#include <windows.h>

#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "pe_loader.h"
#include "shims.h"

namespace {

namespace fs = std::filesystem;
int g_failures = 0;

void check(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++g_failures;
}

using GetModuleHandleAFn = HMODULE(WINAPI*)(LPCSTR);
using GetModuleHandleWFn = HMODULE(WINAPI*)(LPCWSTR);
using GetModuleFileNameAFn = DWORD(WINAPI*)(HMODULE, LPSTR, DWORD);
using GetModuleFileNameWFn = DWORD(WINAPI*)(HMODULE, LPWSTR, DWORD);

std::wstring wide_from_acp(const char* s) {
    if (!s || !*s) return {};
    const int n = MultiByteToWideChar(CP_ACP, 0, s, -1, nullptr, 0);
    if (n <= 1) return {};
    std::wstring out(static_cast<size_t>(n), L'\0');
    MultiByteToWideChar(CP_ACP, 0, s, -1, out.data(), n);
    out.pop_back();
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: library_loader_test <tiny_exe.exe> <tiny_dll.dll>\n");
        return 2;
    }
    const fs::path exe_path = fs::absolute(argv[1]).lexically_normal();
    const fs::path dll_path = fs::absolute(argv[2]).lexically_normal();

    const auto get_handle_a =
        reinterpret_cast<GetModuleHandleAFn>(rt::shims::find("api-ms-win-core-libraryloader-l1-2-0.dll", "GetModuleHandleA"));
    const auto get_handle_w =
        reinterpret_cast<GetModuleHandleWFn>(rt::shims::find("api-ms-win-core-libraryloader-l1-2-0.dll", "GetModuleHandleW"));
    const auto get_name_a =
        reinterpret_cast<GetModuleFileNameAFn>(rt::shims::find("api-ms-win-core-libraryloader-l1-2-0.dll", "GetModuleFileNameA"));
    const auto get_name_w =
        reinterpret_cast<GetModuleFileNameWFn>(rt::shims::find("api-ms-win-core-libraryloader-l1-2-0.dll", "GetModuleFileNameW"));

    check(get_handle_a && get_handle_w && get_name_a && get_name_w,
          "libraryloader-l1-2-0 serves GetModuleHandleA/W and GetModuleFileNameA/W");
    if (!get_handle_a || !get_handle_w || !get_name_a || !get_name_w) return 1;

    // 1. Before any EXE is loaded, nullptr falls back to host executable (parity with native GetModuleFileNameW).
    wchar_t host_w[MAX_PATH] = {};
    const DWORD host_w_len = ::GetModuleFileNameW(nullptr, host_w, MAX_PATH);
    wchar_t shim_w[MAX_PATH] = {};
    const DWORD pre_w_len = get_name_w(nullptr, shim_w, MAX_PATH);
    check(pre_w_len == host_w_len && std::wstring(shim_w) == host_w,
          "GetModuleFileNameW(nullptr) falls back to host before an EXE is loaded");
    check(get_handle_w(nullptr) == ::GetModuleHandleW(nullptr),
          "GetModuleHandleW(nullptr) falls back to host before an EXE is loaded");

    // 2. Host truncation/error-code control check on native GetModuleFileNameW/A.
    std::wstring short_host_w(5, L'x');
    ::SetLastError(0);
    const DWORD host_trunc_w = ::GetModuleFileNameW(nullptr, short_host_w.data(), 5);
    const DWORD host_trunc_err = ::GetLastError();

    // 3. Load a DLL only: nullptr still falls back to host, explicit DLL HMODULE resolves to the DLL path.
    try {
        {
            rt::Loader dll_only(dll_path.parent_path());
            rt::Module& dll_mod = dll_only.load(dll_path);
            const HMODULE dll_h = reinterpret_cast<HMODULE>(dll_mod.base);
            std::wstring out_w(MAX_PATH, L'\0');
            const DWORD dll_len = get_name_w(dll_h, out_w.data(), MAX_PATH);
            out_w.resize(dll_len);
            check(dll_len == dll_path.wstring().size() && out_w == dll_path.wstring(),
                  "GetModuleFileNameW(dll_h) returns mapped bundled DLL path");
            std::wstring after_dll_null(MAX_PATH, L'\0');
            const DWORD n = get_name_w(nullptr, after_dll_null.data(), MAX_PATH);
            after_dll_null.resize(n);
            check(after_dll_null == host_w, "GetModuleFileNameW(nullptr) stays host when only DLLs are mapped");
        }

        // 4. Load an EXE (passed via a relative filename), then change CWD before querying GetModuleFileNameA/W.
        const fs::path orig_cwd = fs::current_path();
        const fs::path temp_cwd = fs::temp_directory_path();
        fs::current_path(exe_path.parent_path());
        rt::Loader loader(exe_path.parent_path());
        rt::Module& exe_mod = loader.load(exe_path.filename());
        fs::current_path(temp_cwd);

        const HMODULE exe_h = reinterpret_cast<HMODULE>(exe_mod.base);
        check(get_handle_w(nullptr) == exe_h && get_handle_a(nullptr) == exe_h,
              "GetModuleHandleA/W(nullptr) returns mapped EXE HMODULE");

        std::wstring mapped_w(MAX_PATH, L'\0');
        ::SetLastError(ERROR_INVALID_PARAMETER);
        const DWORD mapped_w_len = get_name_w(nullptr, mapped_w.data(), MAX_PATH);
        const DWORD mapped_w_err = ::GetLastError();
        mapped_w.resize(mapped_w_len);
        check(mapped_w == exe_path.wstring() && mapped_w_err == ERROR_SUCCESS,
              "GetModuleFileNameW(nullptr) returns absolute mapped EXE path even after SetCurrentDirectory");

        std::vector<char> mapped_a(MAX_PATH, '\0');
        ::SetLastError(ERROR_INVALID_PARAMETER);
        const DWORD mapped_a_len = get_name_a(exe_h, mapped_a.data(), MAX_PATH);
        const DWORD mapped_a_err = ::GetLastError();
        check(mapped_a_err == ERROR_SUCCESS && wide_from_acp(mapped_a.data()) == exe_path.wstring() &&
                  mapped_a_len == std::strlen(mapped_a.data()),
              "GetModuleFileNameA(exe_h) returns matching ACP path");

        // Truncation and size=0 parity against host Windows behavior.
        std::wstring exact_len_buf(mapped_w_len + 1, L'Z');
        ::SetLastError(0);
        const DWORD exact_len_ret = get_name_w(nullptr, exact_len_buf.data(), mapped_w_len);
        check(exact_len_ret == mapped_w_len && ::GetLastError() == ERROR_INSUFFICIENT_BUFFER &&
                  exact_len_buf[mapped_w_len - 1] == L'\0',
              "GetModuleFileNameW with nSize == len truncates with NUL, returns nSize, sets ERROR_INSUFFICIENT_BUFFER");

        std::wstring trunc_w(5, L'Z');
        ::SetLastError(0);
        const DWORD trunc_ret = get_name_w(nullptr, trunc_w.data(), 5);
        check(trunc_ret == host_trunc_w && ::GetLastError() == host_trunc_err &&
                  trunc_w.substr(0, 4) == exe_path.wstring().substr(0, 4) && trunc_w[4] == L'\0',
              "GetModuleFileNameW truncation matches native host semantics");

        std::vector<char> trunc_a(5, 'Z');
        ::SetLastError(0);
        const DWORD trunc_a_ret = get_name_a(nullptr, trunc_a.data(), 5);
        check(trunc_a_ret == 5 && ::GetLastError() == ERROR_INSUFFICIENT_BUFFER && trunc_a[4] == '\0',
              "GetModuleFileNameA truncation sets ERROR_INSUFFICIENT_BUFFER and NUL-terminates");

        wchar_t zero_w[4] = L"abc";
        ::SetLastError(0);
        check(get_name_w(nullptr, zero_w, 0) == 0 && ::GetLastError() == ERROR_INSUFFICIENT_BUFFER,
              "GetModuleFileNameW with nSize=0 returns 0 with ERROR_INSUFFICIENT_BUFFER");

        // Bundled dependency HMODULE (tiny_dll loaded by tiny_exe).
        const HMODULE dep_h = get_handle_a("tiny_dll.dll");
        check(dep_h != nullptr, "GetModuleHandleA finds loaded dependency tiny_dll.dll");
        if (dep_h) {
            std::wstring dep_w(MAX_PATH, L'\0');
            const DWORD dep_len = get_name_w(dep_h, dep_w.data(), MAX_PATH);
            dep_w.resize(dep_len);
            check(fs::path(dep_w).filename() == dll_path.filename(),
                  "GetModuleFileNameW(dep_h) resolves dependency path");
        }

        // Owned shim pseudo-handle (xmem.dll).
        const HMODULE shim_h = get_handle_a("xmem.dll");
        check(shim_h != nullptr, "GetModuleHandleA resolves owned shim xmem.dll");
        if (shim_h) {
            std::wstring xmem_w(MAX_PATH, L'\0');
            const DWORD xmem_len = get_name_w(shim_h, xmem_w.data(), MAX_PATH);
            xmem_w.resize(xmem_len);
            check(xmem_w == L"xmem.dll", "GetModuleFileNameW(pseudo) returns shim DLL name");
        }

        fs::current_path(orig_cwd);
    } catch (const std::exception& e) {
        std::printf("FAIL exception: %s\n", e.what());
        ++g_failures;
    }
    std::printf("library_loader_test: %s (%d failures)\n", g_failures ? "FAILED" : "OK", g_failures);
    return g_failures ? 1 : 0;
}
