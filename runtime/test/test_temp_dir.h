// Scratch folders for tests that write files (a translator-style shader library, an index) under the temp directory.
//
// The folder name carries the process id. Under one fixed name, every copy of a test running at the same time (ctest
// -j, or two suites sharing one Wine prefix or one TMPDIR) used the same folder, and one copy's remove_all deleted the
// modules another copy was still loading: with four copies of d3d12_compute at once, 6 of 40 runs failed at PSO
// creation or in copy_file, while 125 runs one at a time all passed (Apple M3 Max, MoltenVK 1.4.3 under Wine 11).
#pragma once

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

namespace testfiles {

namespace detail {

// Creates and removes a probe folder and file under `base`: the answer is whether a test could actually write
// there, not what the environment claims (C:\Windows is the GetTempPathW fallback with TMP/TEMP/USERPROFILE unset,
// and it is not writable).
inline bool writable_probe(const std::filesystem::path& base) {
    if (base.empty()) return false;
#ifdef _WIN32
    const auto pid = std::to_string(long(_getpid()));
#else
    const auto pid = std::to_string(long(getpid()));
#endif
    const auto probe_dir = base / (".leonida_probe_" + pid);
    std::error_code ec;
    std::filesystem::create_directories(probe_dir, ec);
    if (ec && !std::filesystem::is_directory(probe_dir, ec)) return false;
    const auto probe_file = probe_dir / "probe.tmp";
    FILE* probe = nullptr;
#ifdef _WIN32
    // _wfopen_s on the wide path: a narrow fopen would mangle a temp folder outside the ANSI code page.
    if (_wfopen_s(&probe, probe_file.c_str(), L"wb") != 0) probe = nullptr;
#else
    probe = std::fopen(probe_file.string().c_str(), "wb");
#endif
    if (!probe) {
        std::filesystem::remove_all(probe_dir, ec);
        return false;
    }
    std::fputs("x", probe);
    std::fclose(probe);
    std::filesystem::remove_all(probe_dir, ec);
    return true;
}

}  // namespace detail

// A folder a test may write into even when the environment gives no usable temporary directory. The candidates, in
// order: LEONIDA_TEST_DIR when set (the override loader_test.cpp honors), the standard temp directory and the
// current directory; the first that takes a fresh probe folder and file wins.
//
// temp_directory_path() alone is not enough: with TMP, TEMP and USERPROFILE all unset, GetTempPathW falls back to
// C:\Windows, where create_directories throws access-denied and the uncaught exception ends the test as 0xc0000409
// instead of a skip. Empty when no candidate works; the caller answers with SKIP (77), not a crash.
inline std::filesystem::path writable_temp_dir() {
    static const std::filesystem::path resolved = [] {
        std::vector<std::filesystem::path> candidates;
#ifdef _WIN32
        wchar_t* env = nullptr;
        size_t env_size = 0;
        if (_wdupenv_s(&env, &env_size, L"LEONIDA_TEST_DIR") == 0 && env && *env) candidates.emplace_back(env);
        std::free(env);
#else
        if (const char* env = std::getenv("LEONIDA_TEST_DIR"); env && *env) candidates.emplace_back(env);
#endif
        {
            std::error_code ec;
            const auto temp = std::filesystem::temp_directory_path(ec);
            if (!ec && !temp.empty()) candidates.push_back(temp);
        }
        {
            std::error_code ec;
            const auto cwd = std::filesystem::current_path(ec);
            if (!ec && !cwd.empty()) candidates.push_back(cwd);
        }
        for (const auto& base : candidates) {
            if (detail::writable_probe(base)) return base;
        }
        return std::filesystem::path{};
    }();
    return resolved;
}

// Makes the process environment agree with writable_temp_dir() when the effective temp folder cannot be written:
// with TMP, TEMP and USERPROFILE all unset, GetTempPathW names C:\Windows — for temp_directory_path() AND for the
// loader's own TLS-clone directory (pe_loader.cpp), which the tests cannot reach through writable_temp_dir(). Runs
// once per process from a static initializer, before main, so every caller sees the fixed TMP/TEMP.
inline void ensure_temp_env() {
    std::error_code ec;
    const auto temp = std::filesystem::temp_directory_path(ec);
    if (!ec && detail::writable_probe(temp)) return;
    const auto base = writable_temp_dir();
    if (base.empty()) return;
#ifdef _WIN32
    const auto dir = base.wstring();
    _wputenv_s(L"TMP", dir.c_str());
    _wputenv_s(L"TEMP", dir.c_str());
#else
    const auto dir = base.string();
    setenv("TMPDIR", dir.c_str(), 1);
    setenv("TMP", dir.c_str(), 1);
    setenv("TEMP", dir.c_str(), 1);
#endif
}

namespace detail {
// Never read: the dynamic initializer is the point — it runs ensure_temp_env() once per process before main.
inline const bool kTempEnvEnsured = (ensure_temp_env(), true);
}  // namespace detail

// writable_temp_dir()/<name>, created. Empty when there is no writable base or the folder cannot be made; the
// caller answers with a SKIP rather than letting create_directories or copy_file throw.
inline std::filesystem::path scratch_dir(std::string_view name) {
    const auto base = writable_temp_dir();
    if (base.empty()) return {};
    const auto dir = base / std::string(name);
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec || !std::filesystem::is_directory(dir, ec)) return {};
    return dir;
}

// <temp>/<name>_<pid>: a folder no other running process uses. Under writable_temp_dir(), so a stripped environment
// (no TMP/TEMP/USERPROFILE) lands in a folder that exists instead of C:\Windows; the old temp_directory_path() only
// when even that found nothing, so ScratchDir still reports the real error.
inline std::filesystem::path process_temp_dir(std::string_view name) {
#ifdef _WIN32
    const long pid = long(_getpid());
#else
    const long pid = long(getpid());
#endif
    auto base = writable_temp_dir();
    if (base.empty()) {
        std::error_code ec;
        base = std::filesystem::temp_directory_path(ec);
        if (ec) {
            base = std::filesystem::current_path(ec);
        }
    }
    return base / (std::string(name) + "_" + std::to_string(pid));
}

// process_temp_dir(name), emptied and created on construction (a crashed run may have left the same name behind) and
// removed with its contents on destruction, so early returns don't leave one folder per failed run.
class ScratchDir {
public:
    explicit ScratchDir(std::string_view name) : path_(process_temp_dir(name)) {
        std::error_code ec;
        std::filesystem::remove_all(path_, ec);
        std::filesystem::create_directories(path_, ec);
        if (ec) std::fprintf(stderr, "cannot create %s: %s\n", path_.string().c_str(), ec.message().c_str());
    }
    ~ScratchDir() {
        std::error_code ec;
        std::filesystem::remove_all(path_, ec);
    }
    ScratchDir(const ScratchDir&) = delete;
    ScratchDir& operator=(const ScratchDir&) = delete;

    const std::filesystem::path& path() const { return path_; }

private:
    std::filesystem::path path_;
};

// Copies `from` (a file the build compiled into the test, LEONIDA_TEST_*_SPV) to `to`, overwriting it. Returns false
// and says why when it can't: std::filesystem::copy_file would throw instead, and the uncaught exception ends the
// test without a message (abort; exit code 3 under Wine), which reads as a crash rather than a missing file.
inline bool copy_file(const std::filesystem::path& from, const std::filesystem::path& to) {
    std::error_code ec;
    std::filesystem::copy_file(from, to, std::filesystem::copy_options::overwrite_existing, ec);
    if (ec) std::fprintf(stderr, "cannot copy %s to %s: %s\n", from.string().c_str(), to.string().c_str(),
                         ec.message().c_str());
    return !ec;
}

}  // namespace testfiles
