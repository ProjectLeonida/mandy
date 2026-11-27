#include "test_temp_dir.h"
#include "vk_test_device.h"
#include "../src/shader_library.h"
#include "../src/vulkan_backend.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#define EXPECT(condition)                                                               \
    do {                                                                                \
        if (!(condition)) {                                                             \
            std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition);                \
            return 1;                                                                   \
        }                                                                               \
    } while (0)

namespace {

using namespace leonida;
namespace fs = std::filesystem;

constexpr std::uint32_t kTopologyTriangleList = 4;

constexpr std::uint32_t fullscreen_vs_spirv[] = {
    0x07230203u, 0x00010000u, 0x0008000bu, 0x00000029u, 0x00000000u, 0x00020011u, 0x00000001u, 0x0006000bu,
    0x00000001u, 0x4c534c47u, 0x6474732eu, 0x3035342eu, 0x00000000u, 0x0003000eu, 0x00000000u, 0x00000001u,
    0x0007000fu, 0x00000000u, 0x00000004u, 0x6e69616du, 0x00000000u, 0x0000000cu, 0x0000001fu, 0x00030003u,
    0x00000002u, 0x000001c2u, 0x00040005u, 0x00000004u, 0x6e69616du, 0x00000000u, 0x00030005u, 0x00000009u,
    0x00736f70u, 0x00060005u, 0x0000000cu, 0x565f6c67u, 0x65747265u, 0x646e4978u, 0x00007865u, 0x00060005u,
    0x0000001du, 0x505f6c67u, 0x65567265u, 0x78657472u, 0x00000000u, 0x00060006u, 0x0000001du, 0x00000000u,
    0x505f6c67u, 0x7469736fu, 0x006e6f69u, 0x00070006u, 0x0000001du, 0x00000001u, 0x505f6c67u, 0x746e696fu,
    0x657a6953u, 0x00000000u, 0x00070006u, 0x0000001du, 0x00000002u, 0x435f6c67u, 0x4470696cu, 0x61747369u,
    0x0065636eu, 0x00070006u, 0x0000001du, 0x00000003u, 0x435f6c67u, 0x446c6c75u, 0x61747369u, 0x0065636eu,
    0x00030005u, 0x0000001fu, 0x00000000u, 0x00040047u, 0x0000000cu, 0x0000000bu, 0x0000002au, 0x00030047u,
    0x0000001du, 0x00000002u, 0x00050048u, 0x0000001du, 0x00000000u, 0x0000000bu, 0x00000000u, 0x00050048u,
    0x0000001du, 0x00000001u, 0x0000000bu, 0x00000001u, 0x00050048u, 0x0000001du, 0x00000002u, 0x0000000bu,
    0x00000003u, 0x00050048u, 0x0000001du, 0x00000003u, 0x0000000bu, 0x00000004u, 0x00020013u, 0x00000002u,
    0x00030021u, 0x00000003u, 0x00000002u, 0x00030016u, 0x00000006u, 0x00000020u, 0x00040017u, 0x00000007u,
    0x00000006u, 0x00000002u, 0x00040020u, 0x00000008u, 0x00000007u, 0x00000007u, 0x00040015u, 0x0000000au,
    0x00000020u, 0x00000001u, 0x00040020u, 0x0000000bu, 0x00000001u, 0x0000000au, 0x0004003bu, 0x0000000bu,
    0x0000000cu, 0x00000001u, 0x0004002bu, 0x0000000au, 0x0000000eu, 0x00000001u, 0x00020014u, 0x0000000fu,
    0x0004002bu, 0x00000006u, 0x00000011u, 0x40400000u, 0x0004002bu, 0x00000006u, 0x00000012u, 0xbf800000u,
    0x0004002bu, 0x0000000au, 0x00000015u, 0x00000002u, 0x00040017u, 0x00000019u, 0x00000006u, 0x00000004u,
    0x00040015u, 0x0000001au, 0x00000020u, 0x00000000u, 0x0004002bu, 0x0000001au, 0x0000001bu, 0x00000001u,
    0x0004001cu, 0x0000001cu, 0x00000006u, 0x0000001bu, 0x0006001eu, 0x0000001du, 0x00000019u, 0x00000006u,
    0x0000001cu, 0x0000001cu, 0x00040020u, 0x0000001eu, 0x00000003u, 0x0000001du, 0x0004003bu, 0x0000001eu,
    0x0000001fu, 0x00000003u, 0x0004002bu, 0x0000000au, 0x00000020u, 0x00000000u, 0x0004002bu, 0x00000006u,
    0x00000022u, 0x3f000000u, 0x0004002bu, 0x00000006u, 0x00000023u, 0x3f800000u, 0x00040020u, 0x00000027u,
    0x00000003u, 0x00000019u, 0x00050036u, 0x00000002u, 0x00000004u, 0x00000000u, 0x00000003u, 0x000200f8u,
    0x00000005u, 0x0004003bu, 0x00000008u, 0x00000009u, 0x00000007u, 0x0004003du, 0x0000000au, 0x0000000du,
    0x0000000cu, 0x000500aau, 0x0000000fu, 0x00000010u, 0x0000000du, 0x0000000eu, 0x000600a9u, 0x00000006u,
    0x00000013u, 0x00000010u, 0x00000011u, 0x00000012u, 0x0004003du, 0x0000000au, 0x00000014u, 0x0000000cu,
    0x000500aau, 0x0000000fu, 0x00000016u, 0x00000014u, 0x00000015u, 0x000600a9u, 0x00000006u, 0x00000017u,
    0x00000016u, 0x00000011u, 0x00000012u, 0x00050050u, 0x00000007u, 0x00000018u, 0x00000013u, 0x00000017u,
    0x0003003eu, 0x00000009u, 0x00000018u, 0x0004003du, 0x00000007u, 0x00000021u, 0x00000009u, 0x00050051u,
    0x00000006u, 0x00000024u, 0x00000021u, 0x00000000u, 0x00050051u, 0x00000006u, 0x00000025u, 0x00000021u,
    0x00000001u, 0x00070050u, 0x00000019u, 0x00000026u, 0x00000024u, 0x00000025u, 0x00000022u, 0x00000023u,
    0x00050041u, 0x00000027u, 0x00000028u, 0x0000001fu, 0x00000020u, 0x0003003eu, 0x00000028u, 0x00000026u,
    0x000100fdu, 0x00010038u,
};

constexpr std::uint32_t solid_color_fs_spirv[] = {
    0x07230203u, 0x00010000u, 0x0008000bu, 0x00000012u, 0x00000000u, 0x00020011u, 0x00000001u, 0x0006000bu,
    0x00000001u, 0x4c534c47u, 0x6474732eu, 0x3035342eu, 0x00000000u, 0x0003000eu, 0x00000000u, 0x00000001u,
    0x0006000fu, 0x00000004u, 0x00000004u, 0x6e69616du, 0x00000000u, 0x00000009u, 0x00030010u, 0x00000004u,
    0x00000007u, 0x00030003u, 0x00000002u, 0x000001c2u, 0x00040005u, 0x00000004u, 0x6e69616du, 0x00000000u,
    0x00050005u, 0x00000009u, 0x4374756fu, 0x726f6c6fu, 0x00000000u, 0x00030005u, 0x0000000au, 0x00004350u,
    0x00050006u, 0x0000000au, 0x00000000u, 0x6f6c6f63u, 0x00000072u, 0x00030005u, 0x0000000cu, 0x00006370u,
    0x00040047u, 0x00000009u, 0x0000001eu, 0x00000000u, 0x00030047u, 0x0000000au, 0x00000002u, 0x00050048u,
    0x0000000au, 0x00000000u, 0x00000023u, 0x00000000u, 0x00020013u, 0x00000002u, 0x00030021u, 0x00000003u,
    0x00000002u, 0x00030016u, 0x00000006u, 0x00000020u, 0x00040017u, 0x00000007u, 0x00000006u, 0x00000004u,
    0x00040020u, 0x00000008u, 0x00000003u, 0x00000007u, 0x0004003bu, 0x00000008u, 0x00000009u, 0x00000003u,
    0x0003001eu, 0x0000000au, 0x00000007u, 0x00040020u, 0x0000000bu, 0x00000009u, 0x0000000au, 0x0004003bu,
    0x0000000bu, 0x0000000cu, 0x00000009u, 0x00040015u, 0x0000000du, 0x00000020u, 0x00000001u, 0x0004002bu,
    0x0000000du, 0x0000000eu, 0x00000000u, 0x00040020u, 0x0000000fu, 0x00000009u, 0x00000007u, 0x00050036u,
    0x00000002u, 0x00000004u, 0x00000000u, 0x00000003u, 0x000200f8u, 0x00000005u, 0x00050041u, 0x0000000fu,
    0x00000010u, 0x0000000cu, 0x0000000eu, 0x0004003du, 0x00000007u, 0x00000011u, 0x00000010u, 0x0003003eu,
    0x00000009u, 0x00000011u, 0x000100fdu, 0x00010038u,
};

constexpr std::uint32_t mrt_color_uint_fs_spirv[] = {
    0x07230203u, 0x00010000u, 0x0008000bu, 0x00000019u, 0x00000000u, 0x00020011u, 0x00000001u, 0x0006000bu,
    0x00000001u, 0x4c534c47u, 0x6474732eu, 0x3035342eu, 0x00000000u, 0x0003000eu, 0x00000000u, 0x00000001u,
    0x0007000fu, 0x00000004u, 0x00000004u, 0x6e69616du, 0x00000000u, 0x00000009u, 0x00000014u, 0x00030010u,
    0x00000004u, 0x00000007u, 0x00030003u, 0x00000002u, 0x000001c2u, 0x00040005u, 0x00000004u, 0x6e69616du,
    0x00000000u, 0x00050005u, 0x00000009u, 0x4374756fu, 0x726f6c6fu, 0x00000000u, 0x00030005u, 0x0000000bu,
    0x00004350u, 0x00050006u, 0x0000000bu, 0x00000000u, 0x6f6c6f63u, 0x00000072u, 0x00040006u, 0x0000000bu,
    0x00000001u, 0x00676174u, 0x00030005u, 0x0000000du, 0x00006370u, 0x00040005u, 0x00000014u, 0x5474756fu,
    0x00006761u, 0x00040047u, 0x00000009u, 0x0000001eu, 0x00000000u, 0x00030047u, 0x0000000bu, 0x00000002u,
    0x00050048u, 0x0000000bu, 0x00000000u, 0x00000023u, 0x00000000u, 0x00050048u, 0x0000000bu, 0x00000001u,
    0x00000023u, 0x00000010u, 0x00040047u, 0x00000014u, 0x0000001eu, 0x00000001u, 0x00020013u, 0x00000002u,
    0x00030021u, 0x00000003u, 0x00000002u, 0x00030016u, 0x00000006u, 0x00000020u, 0x00040017u, 0x00000007u,
    0x00000006u, 0x00000004u, 0x00040020u, 0x00000008u, 0x00000003u, 0x00000007u, 0x0004003bu, 0x00000008u,
    0x00000009u, 0x00000003u, 0x00040015u, 0x0000000au, 0x00000020u, 0x00000000u, 0x0004001eu, 0x0000000bu,
    0x00000007u, 0x0000000au, 0x00040020u, 0x0000000cu, 0x00000009u, 0x0000000bu, 0x0004003bu, 0x0000000cu,
    0x0000000du, 0x00000009u, 0x00040015u, 0x0000000eu, 0x00000020u, 0x00000001u, 0x0004002bu, 0x0000000eu,
    0x0000000fu, 0x00000000u, 0x00040020u, 0x00000010u, 0x00000009u, 0x00000007u, 0x00040020u, 0x00000013u,
    0x00000003u, 0x0000000au, 0x0004003bu, 0x00000013u, 0x00000014u, 0x00000003u, 0x0004002bu, 0x0000000eu,
    0x00000015u, 0x00000001u, 0x00040020u, 0x00000016u, 0x00000009u, 0x0000000au, 0x00050036u, 0x00000002u,
    0x00000004u, 0x00000000u, 0x00000003u, 0x000200f8u, 0x00000005u, 0x00050041u, 0x00000010u, 0x00000011u,
    0x0000000du, 0x0000000fu, 0x0004003du, 0x00000007u, 0x00000012u, 0x00000011u, 0x0003003eu, 0x00000009u,
    0x00000012u, 0x00050041u, 0x00000016u, 0x00000017u, 0x0000000du, 0x00000015u, 0x0004003du, 0x0000000au,
    0x00000018u, 0x00000017u, 0x0003003eu, 0x00000014u, 0x00000018u, 0x000100fdu, 0x00010038u,
};

constexpr std::uint32_t store_image_cs_spirv[] = {
    0x07230203u, 0x00010000u, 0x0008000bu, 0x0000001fu, 0x00000000u, 0x00020011u, 0x00000001u, 0x0006000bu,
    0x00000001u, 0x4c534c47u, 0x6474732eu, 0x3035342eu, 0x00000000u, 0x0003000eu, 0x00000000u, 0x00000001u,
    0x0006000fu, 0x00000005u, 0x00000004u, 0x6e69616du, 0x00000000u, 0x0000000eu, 0x00060010u, 0x00000004u,
    0x00000011u, 0x00000001u, 0x00000001u, 0x00000001u, 0x00030003u, 0x00000002u, 0x000001c2u, 0x00040005u,
    0x00000004u, 0x6e69616du, 0x00000000u, 0x00030005u, 0x00000009u, 0x00747364u, 0x00080005u, 0x0000000eu,
    0x475f6c67u, 0x61626f6cu, 0x766e496cu, 0x7461636fu, 0x496e6f69u, 0x00000044u, 0x00030005u, 0x00000016u,
    0x00004350u, 0x00050006u, 0x00000016u, 0x00000000u, 0x6f6c6f63u, 0x00000072u, 0x00030005u, 0x00000018u,
    0x00006370u, 0x00030047u, 0x00000009u, 0x00000019u, 0x00040047u, 0x00000009u, 0x00000021u, 0x00000000u,
    0x00040047u, 0x00000009u, 0x00000022u, 0x00000000u, 0x00040047u, 0x0000000eu, 0x0000000bu, 0x0000001cu,
    0x00030047u, 0x00000016u, 0x00000002u, 0x00050048u, 0x00000016u, 0x00000000u, 0x00000023u, 0x00000000u,
    0x00040047u, 0x0000001eu, 0x0000000bu, 0x00000019u, 0x00020013u, 0x00000002u, 0x00030021u, 0x00000003u,
    0x00000002u, 0x00030016u, 0x00000006u, 0x00000020u, 0x00090019u, 0x00000007u, 0x00000006u, 0x00000001u,
    0x00000000u, 0x00000000u, 0x00000000u, 0x00000002u, 0x00000004u, 0x00040020u, 0x00000008u, 0x00000000u,
    0x00000007u, 0x0004003bu, 0x00000008u, 0x00000009u, 0x00000000u, 0x00040015u, 0x0000000bu, 0x00000020u,
    0x00000000u, 0x00040017u, 0x0000000cu, 0x0000000bu, 0x00000003u, 0x00040020u, 0x0000000du, 0x00000001u,
    0x0000000cu, 0x0004003bu, 0x0000000du, 0x0000000eu, 0x00000001u, 0x00040017u, 0x0000000fu, 0x0000000bu,
    0x00000002u, 0x00040015u, 0x00000012u, 0x00000020u, 0x00000001u, 0x00040017u, 0x00000013u, 0x00000012u,
    0x00000002u, 0x00040017u, 0x00000015u, 0x00000006u, 0x00000004u, 0x0003001eu, 0x00000016u, 0x00000015u,
    0x00040020u, 0x00000017u, 0x00000009u, 0x00000016u, 0x0004003bu, 0x00000017u, 0x00000018u, 0x00000009u,
    0x0004002bu, 0x00000012u, 0x00000019u, 0x00000000u, 0x00040020u, 0x0000001au, 0x00000009u, 0x00000015u,
    0x0004002bu, 0x0000000bu, 0x0000001du, 0x00000001u, 0x0006002cu, 0x0000000cu, 0x0000001eu, 0x0000001du,
    0x0000001du, 0x0000001du, 0x00050036u, 0x00000002u, 0x00000004u, 0x00000000u, 0x00000003u, 0x000200f8u,
    0x00000005u, 0x0004003du, 0x00000007u, 0x0000000au, 0x00000009u, 0x0004003du, 0x0000000cu, 0x00000010u,
    0x0000000eu, 0x0007004fu, 0x0000000fu, 0x00000011u, 0x00000010u, 0x00000010u, 0x00000000u, 0x00000001u,
    0x0004007cu, 0x00000013u, 0x00000014u, 0x00000011u, 0x00050041u, 0x0000001au, 0x0000001bu, 0x00000018u,
    0x00000019u, 0x0004003du, 0x00000015u, 0x0000001cu, 0x0000001bu, 0x00040063u, 0x0000000au, 0x00000014u,
    0x0000001cu, 0x000100fdu, 0x00010038u,
};

constexpr std::uint32_t noop_cs_spirv[] = {
    0x07230203u, 0x00010000u, 0x0008000bu, 0x0000000au, 0x00000000u, 0x00020011u, 0x00000001u, 0x0006000bu,
    0x00000001u, 0x4c534c47u, 0x6474732eu, 0x3035342eu, 0x00000000u, 0x0003000eu, 0x00000000u, 0x00000001u,
    0x0005000fu, 0x00000005u, 0x00000004u, 0x6e69616du, 0x00000000u, 0x00060010u, 0x00000004u, 0x00000011u,
    0x00000001u, 0x00000001u, 0x00000001u, 0x00030003u, 0x00000002u, 0x000001c2u, 0x00040005u, 0x00000004u,
    0x6e69616du, 0x00000000u, 0x00040047u, 0x00000009u, 0x0000000bu, 0x00000019u, 0x00020013u, 0x00000002u,
    0x00030021u, 0x00000003u, 0x00000002u, 0x00040015u, 0x00000006u, 0x00000020u, 0x00000000u, 0x00040017u,
    0x00000007u, 0x00000006u, 0x00000003u, 0x0004002bu, 0x00000006u, 0x00000008u, 0x00000001u, 0x0006002cu,
    0x00000007u, 0x00000009u, 0x00000008u, 0x00000008u, 0x00000008u, 0x00050036u, 0x00000002u, 0x00000004u,
    0x00000000u, 0x00000003u, 0x000200f8u, 0x00000005u, 0x000100fdu, 0x00010038u,
};

bool set_target_dump_env(const fs::path* dir, const char* suffix) {
#ifdef _WIN32
    if (!dir) return _wputenv_s(L"LEONIDA_TARGET_DUMP", L"") == 0;
    std::wstring value = dir->wstring();
    for (const char* p = suffix; p && *p; ++p) value.push_back(static_cast<wchar_t>(*p));
    return _wputenv_s(L"LEONIDA_TARGET_DUMP", value.c_str()) == 0;
#else
    if (!dir) return unsetenv("LEONIDA_TARGET_DUMP") == 0;
    const std::string value = dir->string() + (suffix ? suffix : "");
    return setenv("LEONIDA_TARGET_DUMP", value.c_str(), 1) == 0;
#endif
}

std::uint32_t read_be32(const std::uint8_t* p) {
    return (std::uint32_t(p[0]) << 24) | (std::uint32_t(p[1]) << 16) | (std::uint32_t(p[2]) << 8) | std::uint32_t(p[3]);
}

std::uint32_t crc32_bytes(std::span<const std::uint8_t> data) {
    std::uint32_t crc = ~0u;
    for (const std::uint8_t byte : data) {
        crc ^= byte;
        for (int bit = 0; bit < 8; ++bit) crc = (crc >> 1) ^ ((crc & 1u) ? 0xedb88320u : 0u);
    }
    return ~crc;
}

bool read_file_bytes(const fs::path& path, std::vector<std::uint8_t>& out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    out.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    return in.good() || in.eof();
}

struct DecodedPng {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::vector<std::uint8_t> rgba;
};

bool decode_rgba8_png(const fs::path& path, DecodedPng& out) {
    std::vector<std::uint8_t> bytes;
    if (!read_file_bytes(path, bytes)) return false;
    constexpr std::array<std::uint8_t, 8> signature{137, 80, 78, 71, 13, 10, 26, 10};
    if (bytes.size() < signature.size() || !std::equal(signature.begin(), signature.end(), bytes.begin())) return false;

    std::size_t at = signature.size();
    bool saw_ihdr = false;
    bool saw_iend = false;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::vector<std::uint8_t> idat;

    while (at + 12 <= bytes.size()) {
        const std::uint32_t len = read_be32(bytes.data() + at);
        if (len > bytes.size() - at - 12) return false;
        const std::uint8_t* type = bytes.data() + at + 4;
        const std::uint8_t* data = bytes.data() + at + 8;
        const std::uint32_t stored_crc = read_be32(data + len);
        const std::uint32_t computed_crc = crc32_bytes({type, std::size_t(len) + 4});
        if (stored_crc != computed_crc) return false;

        if (std::memcmp(type, "IHDR", 4) == 0) {
            if (saw_ihdr || len != 13) return false;
            width = read_be32(data);
            height = read_be32(data + 4);
            if (!width || !height || data[8] != 8 || data[9] != 6 || data[10] != 0 || data[11] != 0 || data[12] != 0)
                return false;
            saw_ihdr = true;
        } else if (std::memcmp(type, "IDAT", 4) == 0) {
            if (!saw_ihdr || saw_iend) return false;
            idat.insert(idat.end(), data, data + len);
        } else if (std::memcmp(type, "IEND", 4) == 0) {
            if (!saw_ihdr || len != 0) return false;
            saw_iend = true;
            at += 12;
            break;
        }
        at += 12 + std::size_t(len);
    }

    if (!saw_ihdr || !saw_iend || at != bytes.size()) return false;
    const std::size_t row_bytes = 1 + std::size_t(width) * 4;
    const std::size_t expected_raw = row_bytes * std::size_t(height);
    std::vector<std::uint8_t> inflated;
    if (!zlib_inflate(idat, inflated, expected_raw) || inflated.size() != expected_raw) return false;

    out.width = width;
    out.height = height;
    out.rgba.resize(std::size_t(width) * std::size_t(height) * 4);
    for (std::uint32_t y = 0; y < height; ++y) {
        const std::uint8_t* src_row = inflated.data() + std::size_t(y) * row_bytes;
        if (src_row[0] != 0) return false;
        std::memcpy(out.rgba.data() + std::size_t(y) * std::size_t(width) * 4, src_row + 1, std::size_t(width) * 4);
    }
    return true;
}

bool all_rgba_equal(const DecodedPng& img, std::array<std::uint8_t, 4> expected) {
    if (img.rgba.size() != std::size_t(img.width) * std::size_t(img.height) * 4) return false;
    for (std::size_t i = 0; i < img.rgba.size(); i += 4) {
        if (!std::equal(expected.begin(), expected.end(), img.rgba.data() + i)) {
            std::fprintf(stderr, "pixel %zu: got %u,%u,%u,%u want %u,%u,%u,%u\n", i / 4, unsigned(img.rgba[i]),
                         unsigned(img.rgba[i + 1]), unsigned(img.rgba[i + 2]), unsigned(img.rgba[i + 3]),
                         unsigned(expected[0]), unsigned(expected[1]), unsigned(expected[2]), unsigned(expected[3]));
            return false;
        }
    }
    return true;
}

std::array<std::uint32_t, 32> pack_color_push(std::array<float, 4> color, std::uint32_t tag = 0) {
    std::array<std::uint32_t, 32> words{};
    for (std::size_t i = 0; i < 4; ++i) std::memcpy(&words[i], &color[i], sizeof(float));
    words[4] = tag;
    return words;
}

struct PipelineHandles {
    vktest::Device& device;
    VkShaderModule vs = VK_NULL_HANDLE;
    VkShaderModule fs_single = VK_NULL_HANDLE;
    VkShaderModule fs_mrt = VK_NULL_HANDLE;
    VkShaderModule cs_store = VK_NULL_HANDLE;
    VkShaderModule cs_noop = VK_NULL_HANDLE;
    VkPipelineLayout gfx_layout = VK_NULL_HANDLE;
    VkPipeline gfx_single = VK_NULL_HANDLE;
    VkPipeline gfx_mrt = VK_NULL_HANDLE;
    VkDescriptorSetLayout uav_set_layout = VK_NULL_HANDLE;
    VkDescriptorPool uav_pool = VK_NULL_HANDLE;
    VkDescriptorSet uav_set = VK_NULL_HANDLE;
    VkImageView uav_view = VK_NULL_HANDLE;
    VkPipelineLayout cs_store_layout = VK_NULL_HANDLE;
    VkPipelineLayout cs_noop_layout = VK_NULL_HANDLE;
    VkPipeline cs_store_pipe = VK_NULL_HANDLE;
    VkPipeline cs_noop_pipe = VK_NULL_HANDLE;

    ~PipelineHandles() {
        const auto destroy_pipe = device.fn<PFN_vkDestroyPipeline>("vkDestroyPipeline");
        const auto destroy_layout = device.fn<PFN_vkDestroyPipelineLayout>("vkDestroyPipelineLayout");
        const auto destroy_shader = device.fn<PFN_vkDestroyShaderModule>("vkDestroyShaderModule");
        const auto destroy_pool = device.fn<PFN_vkDestroyDescriptorPool>("vkDestroyDescriptorPool");
        const auto destroy_set_layout = device.fn<PFN_vkDestroyDescriptorSetLayout>("vkDestroyDescriptorSetLayout");
        const auto destroy_view = device.fn<PFN_vkDestroyImageView>("vkDestroyImageView");
        for (VkPipeline p : {gfx_single, gfx_mrt, cs_store_pipe, cs_noop_pipe})
            if (p) destroy_pipe(device.device, p, nullptr);
        for (VkPipelineLayout l : {gfx_layout, cs_store_layout, cs_noop_layout})
            if (l) destroy_layout(device.device, l, nullptr);
        for (VkShaderModule m : {vs, fs_single, fs_mrt, cs_store, cs_noop})
            if (m) destroy_shader(device.device, m, nullptr);
        if (uav_pool) destroy_pool(device.device, uav_pool, nullptr);
        if (uav_set_layout) destroy_set_layout(device.device, uav_set_layout, nullptr);
        if (uav_view) destroy_view(device.device, uav_view, nullptr);
    }

    bool create_module(std::span<const std::uint32_t> code, VkShaderModule& out) {
        VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        info.codeSize = code.size_bytes();
        info.pCode = code.data();
        return device.fn<PFN_vkCreateShaderModule>("vkCreateShaderModule")(device.device, &info, nullptr, &out) ==
               VK_SUCCESS;
    }

    bool create_graphics(VkShaderModule frag, std::span<const VkFormat> formats, VkPipeline& out) {
        VkPipelineShaderStageCreateInfo stages[2]{
            {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, vs, "main",
             nullptr},
            {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, frag,
             "main", nullptr},
        };
        VkPipelineVertexInputStateCreateInfo vertex_input{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        VkPipelineInputAssemblyStateCreateInfo input_assembly{
            VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkPipelineViewportStateCreateInfo viewport_state{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        viewport_state.viewportCount = 1;
        viewport_state.scissorCount = 1;
        VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        raster.polygonMode = VK_POLYGON_MODE_FILL;
        raster.cullMode = VK_CULL_MODE_NONE;
        raster.frontFace = VK_FRONT_FACE_CLOCKWISE;
        raster.lineWidth = 1.0f;
        VkPipelineMultisampleStateCreateInfo msaa{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        msaa.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        std::vector<VkPipelineColorBlendAttachmentState> attachments(formats.size());
        for (auto& a : attachments) {
            a.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
                               VK_COLOR_COMPONENT_A_BIT;
        }
        VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        blend.attachmentCount = std::uint32_t(attachments.size());
        blend.pAttachments = attachments.data();
        const VkDynamicState dynamic_states[] = {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR,
            VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY,
            VK_DYNAMIC_STATE_PRIMITIVE_RESTART_ENABLE,
            VK_DYNAMIC_STATE_STENCIL_REFERENCE,
            VK_DYNAMIC_STATE_BLEND_CONSTANTS,
        };
        VkPipelineDynamicStateCreateInfo dynamic_info{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dynamic_info.dynamicStateCount = std::uint32_t(std::size(dynamic_states));
        dynamic_info.pDynamicStates = dynamic_states;
        VkPipelineRenderingCreateInfo rendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
        rendering.colorAttachmentCount = std::uint32_t(formats.size());
        rendering.pColorAttachmentFormats = formats.data();
        VkGraphicsPipelineCreateInfo info{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &rendering};
        info.stageCount = 2;
        info.pStages = stages;
        info.pVertexInputState = &vertex_input;
        info.pInputAssemblyState = &input_assembly;
        info.pViewportState = &viewport_state;
        info.pRasterizationState = &raster;
        info.pMultisampleState = &msaa;
        info.pColorBlendState = &blend;
        info.pDynamicState = &dynamic_info;
        info.layout = gfx_layout;
        return device.fn<PFN_vkCreateGraphicsPipelines>("vkCreateGraphicsPipelines")(
                   device.device, VK_NULL_HANDLE, 1, &info, nullptr, &out) == VK_SUCCESS;
    }

    bool init(VkImage uav_image) {
        if (!create_module(fullscreen_vs_spirv, vs) || !create_module(solid_color_fs_spirv, fs_single) ||
            !create_module(mrt_color_uint_fs_spirv, fs_mrt) || !create_module(store_image_cs_spirv, cs_store) ||
            !create_module(noop_cs_spirv, cs_noop))
            return false;

        const VkPushConstantRange gfx_push{VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(ShaderUserData)};
        VkPipelineLayoutCreateInfo gfx_layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        gfx_layout_info.pushConstantRangeCount = 1;
        gfx_layout_info.pPushConstantRanges = &gfx_push;
        if (device.fn<PFN_vkCreatePipelineLayout>("vkCreatePipelineLayout")(device.device, &gfx_layout_info, nullptr,
                                                                            &gfx_layout) != VK_SUCCESS)
            return false;

        const VkFormat single_fmt[1] = {VK_FORMAT_R8G8B8A8_UNORM};
        const VkFormat mrt_fmts[2] = {VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R32_UINT};
        if (!create_graphics(fs_single, single_fmt, gfx_single) || !create_graphics(fs_mrt, mrt_fmts, gfx_mrt))
            return false;

        VkImageViewCreateInfo view_info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        view_info.image = uav_image;
        view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view_info.format = VK_FORMAT_R8G8B8A8_UNORM;
        view_info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        if (device.fn<PFN_vkCreateImageView>("vkCreateImageView")(device.device, &view_info, nullptr, &uav_view) !=
            VK_SUCCESS)
            return false;

        const VkDescriptorSetLayoutBinding binding{0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT,
                                                   nullptr};
        VkDescriptorSetLayoutCreateInfo set_layout_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        set_layout_info.bindingCount = 1;
        set_layout_info.pBindings = &binding;
        if (device.fn<PFN_vkCreateDescriptorSetLayout>("vkCreateDescriptorSetLayout")(
                device.device, &set_layout_info, nullptr, &uav_set_layout) != VK_SUCCESS)
            return false;

        const VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1};
        VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pool_info.maxSets = 1;
        pool_info.poolSizeCount = 1;
        pool_info.pPoolSizes = &pool_size;
        if (device.fn<PFN_vkCreateDescriptorPool>("vkCreateDescriptorPool")(device.device, &pool_info, nullptr,
                                                                            &uav_pool) != VK_SUCCESS)
            return false;

        VkDescriptorSetAllocateInfo alloc_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        alloc_info.descriptorPool = uav_pool;
        alloc_info.descriptorSetCount = 1;
        alloc_info.pSetLayouts = &uav_set_layout;
        if (device.fn<PFN_vkAllocateDescriptorSets>("vkAllocateDescriptorSets")(device.device, &alloc_info, &uav_set) !=
            VK_SUCCESS)
            return false;

        const VkDescriptorImageInfo img_info{VK_NULL_HANDLE, uav_view, VK_IMAGE_LAYOUT_GENERAL};
        VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write.dstSet = uav_set;
        write.dstBinding = 0;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        write.pImageInfo = &img_info;
        device.fn<PFN_vkUpdateDescriptorSets>("vkUpdateDescriptorSets")(device.device, 1, &write, 0, nullptr);

        const VkPushConstantRange cs_push{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ShaderUserData)};
        VkPipelineLayoutCreateInfo store_layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        store_layout_info.setLayoutCount = 1;
        store_layout_info.pSetLayouts = &uav_set_layout;
        store_layout_info.pushConstantRangeCount = 1;
        store_layout_info.pPushConstantRanges = &cs_push;
        if (device.fn<PFN_vkCreatePipelineLayout>("vkCreatePipelineLayout")(
                device.device, &store_layout_info, nullptr, &cs_store_layout) != VK_SUCCESS)
            return false;

        VkPipelineLayoutCreateInfo noop_layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        noop_layout_info.pushConstantRangeCount = 1;
        noop_layout_info.pPushConstantRanges = &cs_push;
        if (device.fn<PFN_vkCreatePipelineLayout>("vkCreatePipelineLayout")(
                device.device, &noop_layout_info, nullptr, &cs_noop_layout) != VK_SUCCESS)
            return false;

        VkComputePipelineCreateInfo store_pipe_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        store_pipe_info.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                                 VK_SHADER_STAGE_COMPUTE_BIT, cs_store, "main", nullptr};
        store_pipe_info.layout = cs_store_layout;
        if (device.fn<PFN_vkCreateComputePipelines>("vkCreateComputePipelines")(
                device.device, VK_NULL_HANDLE, 1, &store_pipe_info, nullptr, &cs_store_pipe) != VK_SUCCESS)
            return false;

        VkComputePipelineCreateInfo noop_pipe_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        noop_pipe_info.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                                VK_SHADER_STAGE_COMPUTE_BIT, cs_noop, "main", nullptr};
        noop_pipe_info.layout = cs_noop_layout;
        return device.fn<PFN_vkCreateComputePipelines>("vkCreateComputePipelines")(
                   device.device, VK_NULL_HANDLE, 1, &noop_pipe_info, nullptr, &cs_noop_pipe) == VK_SUCCESS;
    }
};

constexpr ResourceId kRt0Id = 1;
constexpr ResourceId kRt1Id = 2;
constexpr ResourceId kUav0Id = 3;
constexpr std::uintptr_t kPsoSingle = 101;
constexpr std::uintptr_t kPsoMrt = 102;
constexpr std::uintptr_t kPsoCsStore = 201;
constexpr std::uintptr_t kPsoCsNoop = 202;
constexpr std::uint32_t kRtWidth = 16;
constexpr std::uint32_t kRtHeight = 8;
constexpr std::uint32_t kUavWidth = 8;
constexpr std::uint32_t kUavHeight = 8;

bool record_three_pass_list(CommandList& list, std::array<float, 4> scope1_color, std::array<float, 4> scope2_color,
                            std::uint32_t scope2_tag, std::array<float, 4> dispatch_color) {
    if (!list.transition({kRt0Id, all_subresources, d3d12_state::common, d3d12_state::render_target})) return false;
    if (!list.transition({kRt1Id, all_subresources, d3d12_state::common, d3d12_state::render_target})) return false;

    auto views1 = std::make_shared<DrawViews>();
    views1->render_target_count = 1;
    views1->render_targets[0] = {kRt0Id, 0, 0, 1};
    Draw draw1{};
    draw1.pipeline = kPsoSingle;
    draw1.user_data = pack_color_push(scope1_color);
    draw1.views = views1;
    draw1.topology = kTopologyTriangleList;
    draw1.viewport = {0.0f, 0.0f, float(kRtWidth), float(kRtHeight), 0.0f, 1.0f};
    draw1.scissor = {0, 0, std::int32_t(kRtWidth), std::int32_t(kRtHeight)};
    draw1.count = 3;
    draw1.instances = 1;
    if (!list.draw(std::move(draw1))) return false;

    auto views2 = std::make_shared<DrawViews>();
    views2->render_target_count = 2;
    views2->render_targets[0] = {kRt0Id, 0, 0, 1};
    views2->render_targets[1] = {kRt1Id, 0, 0, 1};
    Draw draw2{};
    draw2.pipeline = kPsoMrt;
    draw2.user_data = pack_color_push(scope2_color, scope2_tag);
    draw2.views = views2;
    draw2.topology = kTopologyTriangleList;
    draw2.viewport = {0.0f, 0.0f, float(kRtWidth), float(kRtHeight), 0.0f, 1.0f};
    draw2.scissor = {0, 0, std::int32_t(kRtWidth), std::int32_t(kRtHeight)};
    draw2.count = 3;
    draw2.instances = 1;
    if (!list.draw(std::move(draw2))) return false;

    if (!list.transition({kUav0Id, all_subresources, d3d12_state::common, d3d12_state::unordered_access})) return false;

    Dispatch noop_dispatch{};
    noop_dispatch.pipeline = kPsoCsNoop;
    noop_dispatch.x = 1;
    noop_dispatch.y = 1;
    noop_dispatch.z = 1;
    if (!list.dispatch(noop_dispatch)) return false;

    Dispatch write_dispatch{};
    write_dispatch.pipeline = kPsoCsStore;
    write_dispatch.user_data = pack_color_push(dispatch_color);
    write_dispatch.x = kUavWidth;
    write_dispatch.y = kUavHeight;
    write_dispatch.z = 1;
    if (!list.dispatch(write_dispatch)) return false;

    if (!list.transition({kUav0Id, all_subresources, d3d12_state::unordered_access, d3d12_state::common})) return false;
    if (!list.transition({kRt0Id, all_subresources, d3d12_state::render_target, d3d12_state::common})) return false;
    if (!list.transition({kRt1Id, all_subresources, d3d12_state::render_target, d3d12_state::common})) return false;
    return list.close();
}

void register_scene(VulkanContext& context, VkImage rt0, VkImage rt1, VkImage uav0, const PipelineHandles& handles) {
    ResourceBinding rt0_binding{};
    rt0_binding.image = rt0;
    rt0_binding.format = VK_FORMAT_R8G8B8A8_UNORM;
    rt0_binding.aspects = VK_IMAGE_ASPECT_COLOR_BIT;
    rt0_binding.width = kRtWidth;
    rt0_binding.height = kRtHeight;
    rt0_binding.color_attachment = true;
    context.add_resource(kRt0Id, rt0_binding, VK_IMAGE_LAYOUT_UNDEFINED);

    ResourceBinding rt1_binding{};
    rt1_binding.image = rt1;
    rt1_binding.format = VK_FORMAT_R32_UINT;
    rt1_binding.aspects = VK_IMAGE_ASPECT_COLOR_BIT;
    rt1_binding.width = kRtWidth;
    rt1_binding.height = kRtHeight;
    rt1_binding.color_attachment = true;
    context.add_resource(kRt1Id, rt1_binding, VK_IMAGE_LAYOUT_UNDEFINED);

    ResourceBinding uav0_binding{};
    uav0_binding.image = uav0;
    uav0_binding.format = VK_FORMAT_R8G8B8A8_UNORM;
    uav0_binding.aspects = VK_IMAGE_ASPECT_COLOR_BIT;
    uav0_binding.width = kUavWidth;
    uav0_binding.height = kUavHeight;
    context.add_resource(kUav0Id, uav0_binding, VK_IMAGE_LAYOUT_UNDEFINED);

    GraphicsPipeline pso_single{};
    pso_single.pipeline = handles.gfx_single;
    pso_single.layout = handles.gfx_layout;
    pso_single.push_stages = VK_SHADER_STAGE_FRAGMENT_BIT;
    pso_single.color_formats = {VK_FORMAT_R8G8B8A8_UNORM};
    context.add_graphics_pipeline(kPsoSingle, pso_single);

    GraphicsPipeline pso_mrt{};
    pso_mrt.pipeline = handles.gfx_mrt;
    pso_mrt.layout = handles.gfx_layout;
    pso_mrt.push_stages = VK_SHADER_STAGE_FRAGMENT_BIT;
    pso_mrt.color_formats = {VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R32_UINT};
    context.add_graphics_pipeline(kPsoMrt, pso_mrt);

    ComputePipeline pso_cs_store{};
    pso_cs_store.pipeline = handles.cs_store_pipe;
    pso_cs_store.layout = handles.cs_store_layout;
    pso_cs_store.bindless_set = handles.uav_set;
    pso_cs_store.writes_storage_image = true;
    context.add_pipeline(kPsoCsStore, pso_cs_store);

    ComputePipeline pso_cs_noop{};
    pso_cs_noop.pipeline = handles.cs_noop_pipe;
    pso_cs_noop.layout = handles.cs_noop_layout;
    pso_cs_noop.bindless_set = VK_NULL_HANDLE;
    pso_cs_noop.writes_storage_image = false;
    context.add_pipeline(kPsoCsNoop, pso_cs_noop);
}

PFN_vkGetDeviceProcAddr g_real_get_device_proc = nullptr;
PFN_vkCmdCopyImageToBuffer g_real_copy_image_to_buffer = nullptr;
PFN_vkCreateBuffer g_real_create_buffer = nullptr;
std::uint32_t g_copy_image_to_buffer_calls = 0;
std::uint32_t g_create_buffer_calls = 0;

VKAPI_ATTR void VKAPI_CALL counting_copy_image_to_buffer(VkCommandBuffer cb, VkImage src, VkImageLayout layout,
                                                         VkBuffer dst, std::uint32_t count,
                                                         const VkBufferImageCopy* regions) {
    ++g_copy_image_to_buffer_calls;
    g_real_copy_image_to_buffer(cb, src, layout, dst, count, regions);
}

VKAPI_ATTR VkResult VKAPI_CALL counting_create_buffer(VkDevice dev, const VkBufferCreateInfo* info,
                                                      const VkAllocationCallbacks* alloc, VkBuffer* buffer) {
    ++g_create_buffer_calls;
    return g_real_create_buffer(dev, info, alloc, buffer);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL counting_get_device_proc(VkDevice dev, const char* name) {
    PFN_vkVoidFunction fn = g_real_get_device_proc(dev, name);
    if (!fn) return nullptr;
    if (std::strcmp(name, "vkCmdCopyImageToBuffer") == 0) {
        g_real_copy_image_to_buffer = reinterpret_cast<PFN_vkCmdCopyImageToBuffer>(fn);
        return reinterpret_cast<PFN_vkVoidFunction>(&counting_copy_image_to_buffer);
    }
    if (std::strcmp(name, "vkCreateBuffer") == 0) {
        g_real_create_buffer = reinterpret_cast<PFN_vkCreateBuffer>(fn);
        return reinterpret_cast<PFN_vkVoidFunction>(&counting_create_buffer);
    }
    return fn;
}

int check_dumped_frame(const fs::path& dir, std::uint64_t expected_frame, std::array<std::uint8_t, 4> expected_s1,
                       std::array<std::uint8_t, 4> expected_s2, std::uint32_t expected_tag,
                       std::array<std::uint8_t, 4> expected_uav) {
    std::vector<std::uint8_t> index_bytes;
    EXPECT(read_file_bytes(dir / "index.json", index_bytes));
    const std::string index_text(reinterpret_cast<const char*>(index_bytes.data()), index_bytes.size());
    EXPECT(index_text.find("\"frame\": " + std::to_string(expected_frame)) != std::string::npos);
    EXPECT(index_text.find("\"pass\": 0") != std::string::npos);
    EXPECT(index_text.find("\"pass\": 1") != std::string::npos);
    EXPECT(index_text.find("\"pass\": 2") != std::string::npos);
    EXPECT(index_text.find("\"pass\": 3") == std::string::npos);
    EXPECT(index_text.find("\"kind\": \"render\"") != std::string::npos);
    EXPECT(index_text.find("\"kind\": \"dispatch\"") != std::string::npos);
    EXPECT(index_text.find("\"format\": \"VK_FORMAT_R8G8B8A8_UNORM\"") != std::string::npos);
    EXPECT(index_text.find("\"format\": \"VK_FORMAT_R32_UINT\"") != std::string::npos);
    EXPECT(index_text.find("\"width\": 16") != std::string::npos);
    EXPECT(index_text.find("\"height\": 8") != std::string::npos);
    EXPECT(index_text.find("\"width\": 8") != std::string::npos);
    EXPECT(index_text.find("\"file\": \"pass_0000_rt0.png\"") != std::string::npos);
    EXPECT(index_text.find("\"file\": \"pass_0001_rt0.png\"") != std::string::npos);
    EXPECT(index_text.find("\"file\": \"pass_0001_rt1.raw\"") != std::string::npos);
    EXPECT(index_text.find("\"file\": \"pass_0002_uav0.png\"") != std::string::npos);

    DecodedPng pass0_rt0;
    EXPECT(decode_rgba8_png(dir / "pass_0000_rt0.png", pass0_rt0));
    EXPECT(pass0_rt0.width == kRtWidth && pass0_rt0.height == kRtHeight);
    EXPECT(all_rgba_equal(pass0_rt0, expected_s1));

    DecodedPng pass1_rt0;
    EXPECT(decode_rgba8_png(dir / "pass_0001_rt0.png", pass1_rt0));
    EXPECT(pass1_rt0.width == kRtWidth && pass1_rt0.height == kRtHeight);
    EXPECT(all_rgba_equal(pass1_rt0, expected_s2));

    std::vector<std::uint8_t> pass1_rt1;
    EXPECT(read_file_bytes(dir / "pass_0001_rt1.raw", pass1_rt1));
    EXPECT(pass1_rt1.size() == std::size_t(kRtWidth) * std::size_t(kRtHeight) * sizeof(std::uint32_t));
    for (std::size_t i = 0; i < pass1_rt1.size(); i += 4) {
        std::uint32_t got = 0;
        std::memcpy(&got, pass1_rt1.data() + i, sizeof(got));
        EXPECT(got == expected_tag);
    }

    DecodedPng pass2_uav0;
    EXPECT(decode_rgba8_png(dir / "pass_0002_uav0.png", pass2_uav0));
    EXPECT(pass2_uav0.width == kUavWidth && pass2_uav0.height == kUavHeight);
    EXPECT(all_rgba_equal(pass2_uav0, expected_uav));

    EXPECT(std::distance(fs::directory_iterator(dir), fs::directory_iterator{}) == 5);
    return 0;
}

int test_unit_helpers() {
    EXPECT(!target_dump::parse_config_utf8("").valid);
    EXPECT(!target_dump::parse_config_utf8("dumps").valid);
    EXPECT(!target_dump::parse_config_utf8(",1").valid);
    EXPECT(!target_dump::parse_config_utf8("dumps,0").valid);
    EXPECT(!target_dump::parse_config_utf8("dumps,-1").valid);
    EXPECT(!target_dump::parse_config_utf8("dumps,abc").valid);
    EXPECT(!target_dump::parse_config_utf8("dumps,1x").valid);
    const auto with_comma = target_dump::parse_config_utf8("a,b,3");
    EXPECT(with_comma.valid && with_comma.dir == fs::path("a,b") && with_comma.frame == 3);
    EXPECT(target_dump::uses_png(VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT));
    EXPECT(!target_dump::uses_png(VK_FORMAT_R32_UINT, VK_IMAGE_ASPECT_COLOR_BIT));
    EXPECT(!target_dump::uses_png(VK_FORMAT_D32_SFLOAT, VK_IMAGE_ASPECT_DEPTH_BIT));
    EXPECT(target_dump::raw_texel_size(VK_FORMAT_R32_UINT, VK_IMAGE_ASPECT_COLOR_BIT) == 4);
    EXPECT(target_dump::raw_texel_size(VK_FORMAT_D32_SFLOAT, VK_IMAGE_ASPECT_DEPTH_BIT) == 4);
    EXPECT(target_dump::raw_texel_size(VK_FORMAT_D16_UNORM, VK_IMAGE_ASPECT_DEPTH_BIT) == 2);
    EXPECT(target_dump::raw_texel_size(VK_FORMAT_BC7_UNORM_BLOCK, VK_IMAGE_ASPECT_COLOR_BIT) == 0);
    return 0;
}

}

int main(int argc, char** argv) {
    const std::string_view mode = argc == 2 ? argv[1] : "";
    EXPECT(mode.empty() || mode == "--off" || mode == "--frame2" || mode == "--invalid");

    if (mode == "--invalid") {
        if (test_unit_helpers() != 0) return 1;
    }

    vktest::Device d;
    const auto status = d.initialize(false);
    if (status == vktest::DeviceStatus::unavailable) return 77;
    EXPECT(status == vktest::DeviceStatus::ready);

    VkPhysicalDeviceMemoryProperties memory_props{};
    d.instance_fn<PFN_vkGetPhysicalDeviceMemoryProperties>("vkGetPhysicalDeviceMemoryProperties")(d.physical,
                                                                                                  &memory_props);

    const VkImage rt0 = d.image(VK_FORMAT_R8G8B8A8_UNORM, kRtWidth, kRtHeight, 1, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
    const VkImage rt1 = d.image(VK_FORMAT_R32_UINT, kRtWidth, kRtHeight, 1, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
    const VkImage uav0 = d.image(VK_FORMAT_R8G8B8A8_UNORM, kUavWidth, kUavHeight, 1, VK_IMAGE_USAGE_STORAGE_BIT);
    EXPECT(rt0 && rt1 && uav0);

    PipelineHandles handles{d};
    EXPECT(handles.init(uav0));

    testfiles::ScratchDir scratch("leonida_target_dump");
    const fs::path dir = scratch.path() / "passes";
    const fs::path sentinel = scratch.path() / "outside";

    if (mode == "--off") {
        EXPECT(set_target_dump_env(nullptr, nullptr));
    } else if (mode == "--invalid") {
        EXPECT(set_target_dump_env(&dir, ",0"));
    } else if (mode == "--frame2") {
        EXPECT(set_target_dump_env(&dir, ",2"));
    } else {
        EXPECT(set_target_dump_env(&dir, ",1"));
        EXPECT(fs::create_directory(dir));
        EXPECT(std::ofstream(sentinel).put('X'));
        std::error_code ec;
        fs::create_hard_link(sentinel, dir / "pass_0000_rt0.png", ec);
    }

    g_real_get_device_proc = d.device_proc;
    g_copy_image_to_buffer_calls = 0;
    g_create_buffer_calls = 0;
    const PFN_vkGetDeviceProcAddr proc = mode == "--off" ? &counting_get_device_proc : d.device_proc;

    submit_phases::totals().present_serial.store(0, std::memory_order_relaxed);

    {
        VulkanContext context(proc, d.device, true);
        EXPECT(context.loaded());
        context.set_graphics_family(d.family);
        context.set_occlusion_resolve_memory(memory_props);
        if (mode == "--off" || mode == "--invalid") {
            EXPECT(!context.target_dump_enabled());
        } else {
            EXPECT(context.target_dump_enabled());
        }

        register_scene(context, rt0, rt1, uav0, handles);

        VulkanQueue queue(context, d.queue, d.family);
        EXPECT(queue.ok());

        if (mode == "--frame2") {
            CommandList frame1_list;
            EXPECT(record_three_pass_list(frame1_list, {1.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 1.0f, 0.0f, 1.0f}, 0x11111111u,
                                          {0.0f, 0.0f, 1.0f, 1.0f}));
            const CommandList* l1 = &frame1_list;
            EXPECT(queue.execute({&l1, 1}) == ReplayResult::success);
            EXPECT(queue.idle());
            EXPECT(!fs::exists(dir / "index.json"));

            submit_phases::totals().present_serial.store(1, std::memory_order_relaxed);

            CommandList frame2_list;
            EXPECT(record_three_pass_list(frame2_list, {255.0f / 255.0f, 64.0f / 255.0f, 0.0f, 1.0f},
                                          {0.0f, 128.0f / 255.0f, 1.0f, 1.0f}, 0x13579bdfu,
                                          {32.0f / 255.0f, 224.0f / 255.0f, 96.0f / 255.0f, 1.0f}));
            const CommandList* l2 = &frame2_list;
            EXPECT(queue.execute({&l2, 1}) == ReplayResult::success);
            EXPECT(queue.idle());

            submit_phases::totals().present_serial.store(2, std::memory_order_relaxed);

            CommandList frame3_list;
            EXPECT(record_three_pass_list(frame3_list, {0.5f, 0.5f, 0.5f, 1.0f}, {0.25f, 0.25f, 0.25f, 1.0f},
                                          0x22222222u, {1.0f, 1.0f, 0.0f, 1.0f}));
            const CommandList* l3 = &frame3_list;
            EXPECT(queue.execute({&l3, 1}) == ReplayResult::success);
            EXPECT(queue.idle());
            EXPECT(!context.target_dump_enabled());
        } else {
            CommandList list;
            EXPECT(record_three_pass_list(list, {255.0f / 255.0f, 64.0f / 255.0f, 0.0f, 1.0f},
                                          {0.0f, 128.0f / 255.0f, 1.0f, 1.0f}, 0x13579bdfu,
                                          {32.0f / 255.0f, 224.0f / 255.0f, 96.0f / 255.0f, 1.0f}));
            const CommandList* l = &list;
            EXPECT(queue.execute({&l, 1}) == ReplayResult::success);
        }
    }

    EXPECT(d.validation_errors() == 0);

    if (mode == "--off") {
        EXPECT(g_copy_image_to_buffer_calls == 0);
        EXPECT(g_create_buffer_calls == 0);
        EXPECT(!fs::exists(dir));
        return 0;
    }

    if (mode == "--invalid") {
        EXPECT(!fs::exists(dir));
        return 0;
    }

    if (mode.empty()) {
        EXPECT(fs::file_size(sentinel) == 1 && std::ifstream(sentinel).get() == 'X');
    }

    const std::uint64_t expected_frame = mode == "--frame2" ? 2 : 1;
    return check_dumped_frame(dir, expected_frame, {255, 64, 0, 255}, {0, 128, 255, 255}, 0x13579bdfu,
                              {32, 224, 96, 255});
}
