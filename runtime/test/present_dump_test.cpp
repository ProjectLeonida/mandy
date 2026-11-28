// CPU tests for LEONIDA_PRESENT_DUMP helpers and buffer validation (runtime/src/present_dump.h); no GPU.
#include "../src/present_dump.h"
#include "test_temp_dir.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

bool set_dump_env(const std::filesystem::path& dir, const char* suffix) {
#ifdef _WIN32
    std::wstring value = dir.wstring();
    for (const char* p = suffix; *p; ++p) value.push_back(static_cast<wchar_t>(*p));
    return _wputenv_s(L"LEONIDA_PRESENT_DUMP", value.c_str()) == 0;
#else
    const std::string value = dir.string() + suffix;
    return setenv("LEONIDA_PRESENT_DUMP", value.c_str(), 1) == 0;
#endif
}

bool clear_dump_env() {
#ifdef _WIN32
    return _wputenv_s(L"LEONIDA_PRESENT_DUMP", L"") == 0;
#else
    return unsetenv("LEONIDA_PRESENT_DUMP") == 0;
#endif
}

std::uint32_t read_be32(const std::uint8_t* p) {
    return (std::uint32_t(p[0]) << 24) | (std::uint32_t(p[1]) << 16) |
           (std::uint32_t(p[2]) << 8) | std::uint32_t(p[3]);
}

int test_splash_handoff_policy() {
    using leonida::present_dump::should_release_capture_after_splash;
    EXPECT(!should_release_capture_after_splash(false, false));
    EXPECT(!should_release_capture_after_splash(false, true));
    EXPECT(should_release_capture_after_splash(true, false));
    EXPECT(!should_release_capture_after_splash(true, true));
    return 0;
}

int test_valid_frame_buffer() {
    using leonida::present_dump::valid_frame_buffer;
    const std::array<std::uint8_t, 64> bytes{};
    size_t needed = 12345;

    EXPECT(valid_frame_buffer({2, 2}, VK_FORMAT_B8G8R8A8_UNORM, bytes.data(), 16, &needed));
    EXPECT(needed == 16);
    EXPECT(valid_frame_buffer({2, 2}, VK_FORMAT_R8G8B8A8_SRGB, bytes.data(), bytes.size(), &needed));
    EXPECT(needed == 16);
    EXPECT(valid_frame_buffer({2, 2}, VK_FORMAT_R16G16B16A16_SFLOAT, bytes.data(), 32, &needed));
    EXPECT(needed == 32);
    EXPECT(valid_frame_buffer({2, 2}, VK_FORMAT_R32G32B32A32_SFLOAT, bytes.data(), 64, &needed));
    EXPECT(needed == 64);

    needed = 999;
    EXPECT(!valid_frame_buffer({2, 2}, VK_FORMAT_B8G8R8A8_UNORM, nullptr, 16, &needed));
    EXPECT(needed == 999);
    EXPECT(!valid_frame_buffer({2, 2}, VK_FORMAT_B8G8R8A8_UNORM, bytes.data(), 15, &needed));
    EXPECT(!valid_frame_buffer({0, 2}, VK_FORMAT_B8G8R8A8_UNORM, bytes.data(), 16, &needed));
    EXPECT(!valid_frame_buffer({2, 0}, VK_FORMAT_B8G8R8A8_UNORM, bytes.data(), 16, &needed));
    EXPECT(!valid_frame_buffer({2, 2}, VK_FORMAT_UNDEFINED, bytes.data(), 16, &needed));
    // 32-bit width*height*stride would overflow; valid_frame_buffer must reject it cleanly.
    EXPECT(!valid_frame_buffer({0x80000000u, 2u}, VK_FORMAT_B8G8R8A8_UNORM, bytes.data(), 16, &needed));
    EXPECT(!valid_frame_buffer({0x80000000u, 2u}, VK_FORMAT_B8G8R8A8_UNORM, bytes.data(), bytes.size(), &needed));
    EXPECT(needed == 999);
    return 0;
}

int test_save_valid_and_guards() {
    namespace fs = std::filesystem;
    using leonida::present_dump::PresentDump;
    testfiles::ScratchDir scratch("leonida_present_dump_cpu");

    // 1. Valid PNG write (with extra trailing capacity beyond needed bytes).
    const fs::path ok_dir = scratch.path() / "ok";
    EXPECT(set_dump_env(ok_dir, ",1"));
    {
        PresentDump dump;
        EXPECT(dump.enabled());
        EXPECT(dump.due(0));
        std::array<std::uint8_t, 28> pixels{};  // 2x2 BGRA = 16 bytes + 12 bytes padding
        for (size_t i = 0; i < 4; ++i) {
            pixels[i * 4 + 0] = 0;
            pixels[i * 4 + 1] = 102;
            pixels[i * 4 + 2] = 255;
            pixels[i * 4 + 3] = 255;
        }
        dump.save(1, {2, 2}, VK_FORMAT_B8G8R8A8_UNORM, false, pixels.data(), pixels.size());
    }
    const fs::path png_path = ok_dir / "present_000001.png";
    EXPECT(fs::exists(png_path));
    std::ifstream in(png_path, std::ios::binary);
    EXPECT(in.good());
    const std::vector<std::uint8_t> file_bytes((std::istreambuf_iterator<char>(in)), {});
    constexpr std::array<std::uint8_t, 8> sig{137, 80, 78, 71, 13, 10, 26, 10};
    EXPECT(file_bytes.size() >= 8 + 25 + 12);
    EXPECT(std::memcmp(file_bytes.data(), sig.data(), sig.size()) == 0);
    EXPECT(read_be32(file_bytes.data() + 8) == 13);
    EXPECT(std::memcmp(file_bytes.data() + 12, "IHDR", 4) == 0);
    EXPECT(read_be32(file_bytes.data() + 16) == 2);
    EXPECT(read_be32(file_bytes.data() + 20) == 2);
    EXPECT(file_bytes[24] == 8 && file_bytes[25] == 6);
    EXPECT(read_be32(file_bytes.data() + file_bytes.size() - 12) == 0);
    EXPECT(std::memcmp(file_bytes.data() + file_bytes.size() - 8, "IEND", 4) == 0);

    // 2. Null source pointer (regression for post-splash release_capture leaving capture_ptr_ null).
    const fs::path null_dir = scratch.path() / "null_src";
    EXPECT(set_dump_env(null_dir, ",1"));
    {
        PresentDump dump;
        EXPECT(dump.enabled());
        dump.save(150, {1920, 1080}, VK_FORMAT_B8G8R8A8_UNORM, false, nullptr, size_t(1920) * 1080 * 4);
        EXPECT(!dump.enabled());
    }
    EXPECT(std::distance(fs::directory_iterator(null_dir), fs::directory_iterator{}) == 0);

    // 3. Truncated buffer must disable cleanly without reading past `size`.
    const fs::path short_dir = scratch.path() / "short_buf";
    EXPECT(set_dump_env(short_dir, ",1"));
    {
        PresentDump dump;
        EXPECT(dump.enabled());
        const std::array<std::uint8_t, 4> tiny{1, 2, 3, 4};
        dump.save(2, {2, 2}, VK_FORMAT_B8G8R8A8_UNORM, false, tiny.data(), tiny.size());
        EXPECT(!dump.enabled());
    }
    EXPECT(std::distance(fs::directory_iterator(short_dir), fs::directory_iterator{}) == 0);

    // 4. Overflowing dimensions and unsupported format must disable cleanly.
    const fs::path bad_dir = scratch.path() / "bad_meta";
    EXPECT(set_dump_env(bad_dir, ",1"));
    {
        PresentDump dump;
        EXPECT(dump.enabled());
        const std::array<std::uint8_t, 16> buf{};
        dump.save(3, {0x80000000u, 2u}, VK_FORMAT_B8G8R8A8_UNORM, false, buf.data(), buf.size());
        EXPECT(!dump.enabled());
    }
    {
        PresentDump dump;
        EXPECT(dump.enabled());
        const std::array<std::uint8_t, 16> buf{};
        dump.save(4, {2, 2}, VK_FORMAT_UNDEFINED, false, buf.data(), buf.size());
        EXPECT(!dump.enabled());
    }
    EXPECT(std::distance(fs::directory_iterator(bad_dir), fs::directory_iterator{}) == 0);

    EXPECT(clear_dump_env());
    return 0;
}

}  // namespace

int main() {
    if (test_splash_handoff_policy() != 0) return 1;
    if (test_valid_frame_buffer() != 0) return 1;
    if (test_save_valid_and_guards() != 0) return 1;
    return 0;
}
