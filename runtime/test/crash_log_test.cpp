// Tests for the opt-in unhandled-exception log (LEONIDA_CRASH_LOG, shims/crash_log.h).
//
// Shows a fault in this image and in a small mapped image, the opt-out, a previous filter, an unwritable log path,
// and a fault under the registry's lookup lock.
//
//   crash_log_test                                        the parent: checks the install() rules, then starts copies
//                                                         of itself and checks their logs and exit codes
//   crash_log_test --fault-child TIDFILE                  writes through a null pointer on a new thread of this image
//   crash_log_test --fault-mapped DIR TIDFILE             maps a minimal synthetic image, then faults in it
//   crash_log_test --fault-previous TIDFILE MARKERFILE    installs its own filter before the log, then faults
//   crash_log_test --fault-locked TIDFILE                 faults while the shim registry's lookup lock is held
//
// The parent checks that each child still terminates with 0xC0000005 (the log must not swallow or change the
// exception), that the log names the code, the faulting address as module+0xRVA and raw, the thread id and the top
// frames, and that the same block reached stderr. The mapped case has to name crash_image.bin+0x1220, the game's
// shape of address, not just host modules. The opt-out child runs with the variable unset and must crash the same
// way without writing anything. The previous-filter child must still run the filter it installed; the unwritable
// case must still print the block on stderr; the locked case proves the handler does not take the lookup lock (the
// old module_offset path would deadlock here). LEONIDA_TEST_DIR names the folder for the logs, as for loader_test;
// the default is the temp folder.
#include <windows.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "../shims/crash_log.h"
#include "../shims/shims.h"
#include "../src/pe_loader.h"

namespace {

namespace fs = std::filesystem;

int g_checks = 0;
int g_failed = 0;

void check(bool ok, const std::string& what) {
    ++g_checks;
    std::printf("%s %s\n", ok ? "ok   " : "FAIL ", what.c_str());
    if (!ok) ++g_failed;
}

fs::path test_dir() {
    const char* env = std::getenv("LEONIDA_TEST_DIR");
    fs::path dir = (env && *env ? fs::path(env) : fs::temp_directory_path()) /
                   ("leonida_crash_log_" + std::to_string(GetCurrentProcessId()));
    fs::create_directories(dir);
    return dir;
}

std::string read_file(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

bool has(const std::string& text, const std::string& what) { return text.find(what) != std::string::npos; }

void unset_variable() { SetEnvironmentVariableW(L"LEONIDA_CRASH_LOG", nullptr); }

void set_variable(const fs::path& path) { SetEnvironmentVariableW(L"LEONIDA_CRASH_LOG", path.c_str()); }

// What the child does on its faulting thread. The pointer is volatile so the compiler cannot fold the store into
// something other than a write to address 0.
volatile int* volatile g_null = nullptr;
volatile int g_sink = 0;

// Storing after each call keeps these from becoming tail calls, so the walk has two return addresses in this image.
__declspec(noinline) void crash_leaf() { *g_null = 1; }
__declspec(noinline) void crash_middle() {
    crash_leaf();
    g_sink = 1;
}
__declspec(noinline) void crash_outer() {
    crash_middle();
    g_sink = 2;
}

std::wstring g_tid_path;
std::wstring g_marker_path;

void record_tid() {
    FILE* f = nullptr;
    if (_wfopen_s(&f, g_tid_path.c_str(), L"w") == 0 && f) {
        std::fprintf(f, "%lu\n", GetCurrentThreadId());
        std::fclose(f);
    }
}

template <typename Fault>
int fault_on_thread(Fault fault) {
    std::thread t([fault] {
        record_tid();
        fault();
    });
    t.join();
    return 4;  // a handled fault would reach this; the parent treats it as a failure
}

int fault_child() {
    // No interactive fault box: the process still ends with the exception code, which the parent reads.
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX);
    rt::crash_log::install();  // the opt-out case runs this too: no variable, no handler, the same fault
    return fault_on_thread([] { crash_outer(); });
}

// A filter installed before the crash log: the log must call it and return its answer, not replace its behaviour.
LONG CALLBACK previous_filter(EXCEPTION_POINTERS*) {
    const HANDLE marker = CreateFileW(g_marker_path.c_str(), GENERIC_WRITE, FILE_SHARE_READ, nullptr, CREATE_ALWAYS,
                                      FILE_ATTRIBUTE_NORMAL, nullptr);
    if (marker != INVALID_HANDLE_VALUE) {
        DWORD written = 0;
        WriteFile(marker, "previous", 8, &written, nullptr);
        CloseHandle(marker);
    }
    return EXCEPTION_CONTINUE_SEARCH;
}

int fault_previous_child() {
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX);
    SetUnhandledExceptionFilter(&previous_filter);
    rt::crash_log::install();
    return fault_on_thread([] { crash_outer(); });
}

// The fault happens while the shim registry's module-lookup lock is held: the handler must not need it.
int fault_locked_child() {
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX);
    rt::crash_log::install();
    rt::shims::with_module_lookup_lock_for_test(
        [](void*) {
            record_tid();
            crash_outer();
        },
        nullptr);
    return 6;
}

// A minimal mapped image with one function that writes through a null pointer. No imports, relocations, TLS or
// .pdata: the loader maps it, and the crash handler has to name it as `crash_image.bin+0x1220`.
constexpr uint32_t kImageTextRva = 0x1000;
constexpr uint32_t kImageFaultRva = kImageTextRva + 0x220;
constexpr const char* kImageName = "crash_image.bin";

std::vector<uint8_t> make_image() {
    std::vector<uint8_t> b(0x600, 0);
    auto* dos = reinterpret_cast<IMAGE_DOS_HEADER*>(b.data());
    dos->e_magic = IMAGE_DOS_SIGNATURE;
    dos->e_lfanew = 0x80;
    auto* nt = reinterpret_cast<IMAGE_NT_HEADERS64*>(b.data() + dos->e_lfanew);
    nt->Signature = IMAGE_NT_SIGNATURE;
    nt->FileHeader.Machine = IMAGE_FILE_MACHINE_AMD64;
    nt->FileHeader.NumberOfSections = 1;
    nt->FileHeader.SizeOfOptionalHeader = sizeof(IMAGE_OPTIONAL_HEADER64);
    nt->FileHeader.Characteristics = IMAGE_FILE_EXECUTABLE_IMAGE | IMAGE_FILE_DLL | IMAGE_FILE_LARGE_ADDRESS_AWARE;
    nt->OptionalHeader.Magic = IMAGE_NT_OPTIONAL_HDR64_MAGIC;
    nt->OptionalHeader.ImageBase = 0x180000000;
    nt->OptionalHeader.SizeOfHeaders = 0x200;
    nt->OptionalHeader.SizeOfImage = 0x2000;
    nt->OptionalHeader.Subsystem = IMAGE_SUBSYSTEM_WINDOWS_CUI;
    nt->OptionalHeader.NumberOfRvaAndSizes = IMAGE_NUMBEROF_DIRECTORY_ENTRIES;
    auto* sec = IMAGE_FIRST_SECTION(nt);
    std::memcpy(sec->Name, ".text", 6);
    sec->VirtualAddress = kImageTextRva;
    sec->Misc.VirtualSize = 0x400;
    sec->PointerToRawData = 0x200;
    sec->SizeOfRawData = 0x400;
    sec->Characteristics = IMAGE_SCN_CNT_CODE | IMAGE_SCN_MEM_EXECUTE | IMAGE_SCN_MEM_READ;
    const uint8_t fault[] = {0x48, 0x8B, 0x04, 0x25, 0, 0, 0, 0,  // mov rax, [0]
                             0xC3};                               // ret
    std::memcpy(b.data() + 0x200 + (kImageFaultRva - kImageTextRva), fault, sizeof fault);
    return b;
}

void write_file(const fs::path& p, const std::vector<uint8_t>& bytes) {
    std::ofstream out(p, std::ios::binary);
    out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!out) throw std::runtime_error("cannot write " + p.string());
}

int fault_mapped_child(const fs::path& dir) {
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX);
    rt::crash_log::install();
    const fs::path image = dir / kImageName;
    write_file(image, make_image());
    rt::Loader loader(dir);
    rt::Module& m = loader.load(image);
    void* fn = m.base + kImageFaultRva;
    return fault_on_thread([fn] { reinterpret_cast<void (*)()>(fn)(); });
}

std::string hex_code(DWORD code) {
    char b[16];
    std::snprintf(b, sizeof b, "%08lX", static_cast<unsigned long>(code));
    return b;
}

struct ChildResult {
    DWORD exit_code = 0;
    DWORD wait_error = 0;
    std::string stderr_text;
    std::string thread_id;
};

ChildResult run_child(const std::vector<std::wstring>& args, const fs::path& log, const fs::path& stderr_file,
                      const fs::path& tid_file, const fs::path& exe) {
    ChildResult result;
    SECURITY_ATTRIBUTES inheritable{sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
    const HANDLE out = CreateFileW(stderr_file.c_str(), GENERIC_WRITE, FILE_SHARE_READ, &inheritable, CREATE_ALWAYS,
                                   FILE_ATTRIBUTE_NORMAL, nullptr);
    if (out == INVALID_HANDLE_VALUE) {
        result.wait_error = GetLastError();
        return result;
    }
    STARTUPINFOW si{};
    si.cb = sizeof si;
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput = out;
    si.hStdOutput = out;
    si.hStdError = out;
    std::wstring command = L"\"" + exe.wstring() + L"\"";
    for (const std::wstring& arg : args) command += L" \"" + arg + L"\"";
    std::vector<wchar_t> mutable_command(command.begin(), command.end());
    mutable_command.push_back(L'\0');
    PROCESS_INFORMATION pi{};
    unset_variable();
    if (log.empty()) {
        // The opt-out case: the variable stays unset for the child.
    } else {
        set_variable(log);
    }
    const BOOL started = CreateProcessW(exe.c_str(), mutable_command.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW,
                                        nullptr, nullptr, &si, &pi);
    unset_variable();
    CloseHandle(out);
    if (!started) {
        result.wait_error = GetLastError();
        return result;
    }
    const DWORD waited = WaitForSingleObject(pi.hProcess, 60000);
    if (waited != WAIT_OBJECT_0) {
        result.wait_error = waited;
        TerminateProcess(pi.hProcess, 0xDEAD);
    } else if (!GetExitCodeProcess(pi.hProcess, &result.exit_code)) {
        result.wait_error = GetLastError();
    }
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    result.stderr_text = read_file(stderr_file);
    result.thread_id = read_file(tid_file);
    while (!result.thread_id.empty() && (result.thread_id.back() == '\n' || result.thread_id.back() == '\r'))
        result.thread_id.pop_back();
    return result;
}

// The variable is read once per process: each case is its own process, so the parent covers only the rules that do
// not need a fault.
void test_install_rules(const fs::path& dir) {
    const fs::path fallback = dir / "fallback.log";
    unset_variable();
    check(!rt::crash_log::install(fallback), "no variable: the log stays off");
    check(!rt::crash_log::enabled(), "no variable: enabled() is false");

    SetEnvironmentVariableW(L"LEONIDA_CRASH_LOG", L"0");
    check(!rt::crash_log::install(fallback), "0: the log stays off");
    unset_variable();

    SetEnvironmentVariableW(L"LEONIDA_CRASH_LOG", L"1");
    check(rt::crash_log::install(fallback), "1: the log is on");
    check(rt::crash_log::enabled(), "1: enabled() is true");
    check(fs::exists(fallback), "1: the fallback file is created");
    rt::crash_log::close();
    check(!rt::crash_log::enabled(), "close() turns the log off");
    unset_variable();

    const fs::path named = dir / "named.log";
    set_variable(named);
    check(rt::crash_log::install(fallback), "a path: the log is on");
    check(fs::exists(named), "a path: that file is created");
    rt::crash_log::close();
    unset_variable();
}

void test_crash(const fs::path& dir, const fs::path& exe) {
    const fs::path log = dir / "crash.log";
    const fs::path stderr_file = dir / "crash.stderr.txt";
    const fs::path tid_file = dir / "crash.tid.txt";
    const std::vector<std::wstring> args = {L"--fault-child", tid_file.wstring()};
    const ChildResult r = run_child(args, log, stderr_file, tid_file, exe);
    check(r.wait_error == 0, "the crash child ran (error " + std::to_string(r.wait_error) + ")");
    check(r.exit_code == 0xC0000005u,
          "the log does not change termination: exit code 0xC0000005 (got 0x" + hex_code(r.exit_code) + ")");
    check(!r.thread_id.empty(), "the child recorded its faulting thread id");
    const std::string text = read_file(log);
    check(!text.empty(), "the crash log has content");
    check(has(text, "unhandled exception 0xC0000005"), "the log names the exception code");
    check(has(text, "host:crash_log_test.exe+0x"), "the faulting address is module+0xRVA");
    check(has(text, "thread " + r.thread_id), "the log names the faulting thread id");
    check(has(text, "access violation writing address 0x0"), "the log names the access kind and address");
    check(has(text, "stack:"), "the log has a stack");
    check(has(text, "#0 host:crash_log_test.exe+0x"), "the top frame is module+0xRVA");
    check(has(text, "#1 host:crash_log_test.exe+0x"), "the first return address is module+0xRVA");
    check(has(text, "#2 host:crash_log_test.exe+0x"), "the second return address is module+0xRVA");
    check(has(text, "end of unhandled exception 0xC0000005"), "the block has an end marker");
    check(has(r.stderr_text, "unhandled exception 0xC0000005"), "the same block reached stderr");
    check(has(r.stderr_text, "thread " + r.thread_id), "stderr names the faulting thread id");
}

void test_mapped_crash(const fs::path& dir, const fs::path& exe) {
    const fs::path log = dir / "mapped.log";
    const fs::path stderr_file = dir / "mapped.stderr.txt";
    const fs::path tid_file = dir / "mapped.tid.txt";
    const std::vector<std::wstring> args = {L"--fault-mapped", dir.wstring(), tid_file.wstring()};
    const ChildResult r = run_child(args, log, stderr_file, tid_file, exe);
    check(r.wait_error == 0, "the mapped-image child ran (error " + std::to_string(r.wait_error) + ")");
    check(r.exit_code == 0xC0000005u,
          "the mapped-image child still ends with 0xC0000005 (got 0x" + hex_code(r.exit_code) + ")");
    const std::string text = read_file(log);
    check(has(text, "at crash_image.bin+0x1220"), "the faulting address is the mapped image's module+RVA");
    check(has(text, "#0 crash_image.bin+0x1220"), "the top frame is the mapped image's module+RVA");
    check(has(text, "thread " + r.thread_id), "the log names the faulting thread id");
    check(has(r.stderr_text, "crash_image.bin+0x1220"), "the mapped module+RVA reached stderr");
}

void test_opt_out(const fs::path& dir, const fs::path& exe) {
    const fs::path stderr_file = dir / "optout.stderr.txt";
    const fs::path tid_file = dir / "optout.tid.txt";
    const std::vector<std::wstring> args = {L"--fault-child", tid_file.wstring()};
    const ChildResult r = run_child(args, {}, stderr_file, tid_file, exe);
    check(r.wait_error == 0, "the opt-out child ran (error " + std::to_string(r.wait_error) + ")");
    check(r.exit_code == 0xC0000005u, "without the variable the child still ends with 0xC0000005");
    check(!has(r.stderr_text, "[rt] crash:"), "without the variable nothing is written to stderr");
}

void test_previous_filter(const fs::path& dir, const fs::path& exe) {
    const fs::path log = dir / "previous.log";
    const fs::path stderr_file = dir / "previous.stderr.txt";
    const fs::path tid_file = dir / "previous.tid.txt";
    const fs::path marker = dir / "previous.marker.txt";
    const std::vector<std::wstring> args = {L"--fault-previous", tid_file.wstring(), marker.wstring()};
    const ChildResult r = run_child(args, log, stderr_file, tid_file, exe);
    check(r.wait_error == 0, "the previous-filter child ran (error " + std::to_string(r.wait_error) + ")");
    check(r.exit_code == 0xC0000005u, "the previous-filter child still ends with 0xC0000005");
    check(has(read_file(log), "unhandled exception 0xC0000005"), "the log is written before the previous filter");
    check(read_file(marker) == "previous", "the filter installed before the log still runs");
}

void test_unwritable_path(const fs::path& dir, const fs::path& exe) {
    const fs::path stderr_file = dir / "unwritable.stderr.txt";
    const fs::path tid_file = dir / "unwritable.tid.txt";
    const std::vector<std::wstring> args = {L"--fault-child", tid_file.wstring()};
    // A directory is an existing path CreateFileW cannot open for append.
    const ChildResult r = run_child(args, dir, stderr_file, tid_file, exe);
    check(r.wait_error == 0, "the unwritable-path child ran (error " + std::to_string(r.wait_error) + ")");
    check(r.exit_code == 0xC0000005u, "an unwritable log path does not change the exit code");
    check(has(r.stderr_text, "cannot create the log file"), "the unwritable path is reported once");
    check(has(r.stderr_text, "unhandled exception 0xC0000005"), "the block still reaches stderr without a file");
}

void test_locked_fault(const fs::path& dir, const fs::path& exe) {
    const fs::path log = dir / "locked.log";
    const fs::path stderr_file = dir / "locked.stderr.txt";
    const fs::path tid_file = dir / "locked.tid.txt";
    const std::vector<std::wstring> args = {L"--fault-locked", tid_file.wstring()};
    const ChildResult r = run_child(args, log, stderr_file, tid_file, exe);
    check(r.wait_error == 0, "the locked-fault child completed instead of deadlocking (error " +
                                 std::to_string(r.wait_error) + ")");
    check(r.exit_code == 0xC0000005u, "the locked-fault child still ends with 0xC0000005");
    const std::string text = read_file(log);
    check(has(text, "unhandled exception 0xC0000005"), "a fault under the lookup lock is logged");
    check(has(text, "host:crash_log_test.exe+0x"), "and its module+RVA is still resolved");
    check(has(text, "thread " + r.thread_id), "and its thread id is named");
}

}  // namespace

int wmain(int argc, wchar_t** argv) {
    if (argc == 3 && std::wstring(argv[1]) == L"--fault-child") {
        g_tid_path = argv[2];
        return fault_child();
    }
    if (argc == 4 && std::wstring(argv[1]) == L"--fault-mapped") {
        g_tid_path = argv[3];
        return fault_mapped_child(argv[2]);
    }
    if (argc == 4 && std::wstring(argv[1]) == L"--fault-previous") {
        g_tid_path = argv[2];
        g_marker_path = argv[3];
        return fault_previous_child();
    }
    if (argc == 3 && std::wstring(argv[1]) == L"--fault-locked") {
        g_tid_path = argv[2];
        return fault_locked_child();
    }
    if (argc != 1) {
        std::fprintf(stderr,
                     "usage: crash_log_test [--fault-child TIDFILE | --fault-mapped DIR TIDFILE | "
                     "--fault-previous TIDFILE MARKERFILE | --fault-locked TIDFILE]\n");
        return 2;
    }
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX);
    try {
        const fs::path dir = test_dir();
        std::wstring self(32768, L'\0');
        const DWORD n = GetModuleFileNameW(nullptr, self.data(), static_cast<DWORD>(self.size()));
        if (!n || n >= self.size()) throw std::runtime_error("cannot name this executable");
        self.resize(n);
        test_install_rules(dir);
        test_crash(dir, fs::path(self));
        test_mapped_crash(dir, fs::path(self));
        test_previous_filter(dir, fs::path(self));
        test_unwritable_path(dir, fs::path(self));
        test_locked_fault(dir, fs::path(self));
        test_opt_out(dir, fs::path(self));
        std::error_code ec;
        fs::remove_all(dir, ec);
    } catch (const std::exception& e) {
        std::printf("FAIL exception: %s\n", e.what());
        ++g_failed;
    }
    std::printf("%d check(s), %d failed\n", g_checks, g_failed);
    return g_failed ? 1 : 0;
}
