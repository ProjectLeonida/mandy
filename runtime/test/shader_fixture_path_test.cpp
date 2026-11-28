#include "shader_fixture_path.h"
#include "test_temp_dir.h"
#include <cstdio>
#include <chrono>
#include <fstream>

int main() {
    namespace fs = std::filesystem;
    // scratch_dir, not temp_directory_path()/create_directories: with TMP/TEMP/USERPROFILE unset the temp path is
    // C:\Windows and the throwing create_directories below ended the run as 0xc0000409 before the try starts.
    const auto root = testfiles::scratch_dir("leonida_fixture_path_" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    if (root.empty()) {
        std::puts("shader fixture lookup failed: no writable directory");
        return 1;
    }
    std::error_code ec;
    fs::create_directories(root / "generated", ec);
    if (ec) {
        std::puts("shader fixture lookup failed: cannot create the fixture folder");
        return 1;
    }
    const auto previous = fs::current_path();
    int result = 0;
    try {
        fs::current_path(root);
        std::ofstream("generated/test.spv") << "packaged";
        const auto original = root / "build" / "test.spv";
        if (vktest::shader_fixture_path(original) != fs::path("generated/test.spv")) result = 1;
        fs::create_directory("build");
        std::ofstream(original) << "local";
        if (vktest::shader_fixture_path(original) != original) result = 1;
        const auto missing = root / "build" / "missing.spv";
        if (vktest::shader_fixture_path(missing) != missing) result = 1;
        fs::remove(original);
        fs::remove("generated/test.spv");
        fs::create_directory("generated/test.spv");
        if (vktest::shader_fixture_path(original) != original) result = 1;
    } catch (...) { result = 1; }
    fs::current_path(previous);
    fs::remove_all(root);
    std::puts(result ? "shader fixture lookup failed" : "shader fixture lookup passed");
    return result;
}
