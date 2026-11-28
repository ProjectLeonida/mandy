// rt::title_patch (runtime/src/title_patch.h): LEONIDA_TITLE_PATCH's parser and its checked writes, on a page of real
// executable memory standing in for the title's image. No game data, no GPU.
//   1. a specification with ',' and '+' between patches gives every patch, in order, with its bytes;
//   2. malformed items (no colons, odd hex, unequal lengths, empty fields, not hex) are reported and skipped;
//   3. a patch whose expected bytes are there is written, and the page keeps its protection;
//   4. a patch whose expected bytes are not there leaves the bytes alone;
//   5. a patch that reaches past the image, or starts past it, is refused without touching memory.
#include <windows.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "../src/title_patch.h"

namespace {

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const char* what) {
    ++g_checks;
    std::printf("%s %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition) ++g_failures;
}

}  // namespace

int main() {
    using rt::title_patch::Result;
    {
        std::vector<std::string> bad;
        const auto p = rt::title_patch::parse("10:0f45c1:0f1f00+2a:AABB:ccdd,3:01:02", &bad);
        check(p.size() == 3 && bad.empty(), "1. three patches, none rejected");
        check(p.size() == 3 && p[0].rva == 0x10 && p[0].expect == std::vector<std::uint8_t>{0x0f, 0x45, 0xc1} &&
                  p[0].replace == std::vector<std::uint8_t>{0x0f, 0x1f, 0x00},
              "1. first patch: rva and bytes");
        check(p.size() == 3 && p[1].rva == 0x2a && p[1].expect == std::vector<std::uint8_t>{0xaa, 0xbb} &&
                  p[2].rva == 3,
              "1. upper-case hex and both separators");
    }
    {
        std::vector<std::string> bad;
        const auto p = rt::title_patch::parse("10,10:0f4:0f1,10:0f45:0f,:00:01,10::,10:zz:00,20:01:02", &bad);
        check(p.size() == 1 && p[0].rva == 0x20, "2. only the well-formed item is kept");
        check(bad.size() == 6, "2. six items reported");
    }
    SYSTEM_INFO si{};
    GetSystemInfo(&si);
    const std::size_t size = si.dwPageSize;
    auto* image = static_cast<std::uint8_t*>(VirtualAlloc(nullptr, size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE));
    if (!image) {
        std::printf("FAIL VirtualAlloc\n");
        return 1;
    }
    for (std::size_t i = 0; i < size; ++i) image[i] = static_cast<std::uint8_t>(i * 7);
    image[0x40] = 0x0f;
    image[0x41] = 0x45;
    image[0x42] = 0xc1;
    DWORD old = 0;
    VirtualProtect(image, size, PAGE_EXECUTE_READ, &old);
    {
        const auto p = rt::title_patch::parse("40:0f45c1:0f1f00");
        check(p.size() == 1 && rt::title_patch::apply(image, size, p[0]) == Result::applied, "3. applied");
        check(image[0x40] == 0x0f && image[0x41] == 0x1f && image[0x42] == 0x00, "3. new bytes in place");
        check(image[0x3f] == static_cast<std::uint8_t>(0x3f * 7) && image[0x43] == static_cast<std::uint8_t>(0x43 * 7),
              "3. neighbours untouched");
        MEMORY_BASIC_INFORMATION mbi{};
        VirtualQuery(image + 0x40, &mbi, sizeof(mbi));
        check(mbi.Protect == PAGE_EXECUTE_READ, "3. protection restored");
    }
    {
        const auto p = rt::title_patch::parse("40:0f45c1:909090");
        check(p.size() == 1 && rt::title_patch::apply(image, size, p[0]) == Result::mismatch,
              "4. a second run finds the patched bytes and refuses");
        check(image[0x40] == 0x0f && image[0x41] == 0x1f && image[0x42] == 0x00, "4. bytes unchanged");
    }
    {
        char spec[64];
        std::snprintf(spec, sizeof(spec), "%zx:%02x%02x:0000", size - 1, image[size - 1], 0);
        const auto p = rt::title_patch::parse(spec);
        check(p.size() == 1 && rt::title_patch::apply(image, size, p[0]) == Result::outside, "5. reaches past the end");
        std::snprintf(spec, sizeof(spec), "%zx:00:01", size + 0x100);
        const auto q = rt::title_patch::parse(spec);
        check(q.size() == 1 && rt::title_patch::apply(image, size, q[0]) == Result::outside, "5. starts past the end");
    }
    VirtualFree(image, 0, MEM_RELEASE);
    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
