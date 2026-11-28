// d3d12_backend.h (RT-VULKAN-REQUESTED-NOT-BUILT): LEONIDA_D3D12=vulkan in a runtime that has no Vulkan-backed device
// gets a note; every other case gets none. CPU only: no device, no Vulkan headers, no game file.
#include "../shims/d3d12_backend.h"

#include <cstdio>
#include <cstring>

namespace backend = rt::d3d12_xs::backend;

static int failures = 0;
static void check(bool condition, const char* what) {
    std::printf("%s %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition) ++failures;
}

int main() {
    check(backend::vulkan_requested("vulkan"), "requested: vulkan");
    check(!backend::vulkan_requested(nullptr), "requested: the variable is not set");
    check(!backend::vulkan_requested(""), "requested: an empty value");
    check(!backend::vulkan_requested("Vulkan"), "requested: another case is another value, as the device branch reads it");
    check(!backend::vulkan_requested("vulkan "), "requested: a trailing space is another value");
    check(!backend::vulkan_requested("host"), "requested: another backend name");

    const char* note = backend::vulkan_not_built_note("vulkan", false);
    check(note != nullptr, "note: vulkan asked of a runtime without the device");
    check(note && std::strstr(note, "LEONIDA_D3D12=vulkan") && std::strstr(note, "host's D3D12 device"),
          "note: names the variable and the device the title gets");
    check(note && std::strstr(note, "LEONIDA_VULKAN_HEADERS_FETCH") && std::strstr(note, "LEONIDA_REQUIRE_VULKAN") &&
              std::strstr(note, "VULKAN_SDK"),
          "note: names the ways to get the headers and the option that fails the configure");

    check(backend::vulkan_not_built_note("vulkan", true) == nullptr, "note: none when the device is built in");
    check(backend::vulkan_not_built_note(nullptr, false) == nullptr, "note: none when the variable is not set");
    check(backend::vulkan_not_built_note("", false) == nullptr, "note: none for an empty value");
    check(backend::vulkan_not_built_note("Vulkan", false) == nullptr,
          "note: none for a value the Vulkan build would not take either");
    check(backend::vulkan_not_built_note(nullptr, true) == nullptr, "note: none, device built in and nothing asked");

    std::printf("%d failures\n", failures);
    return failures ? 1 : 0;
}
