// Which device D3D12XboxCreateDevice puts behind the Xbox-ABI proxy, as LEONIDA_D3D12 asks (RT-VULKAN-REQUESTED-NOT-BUILT).
// Header only, so the test needs neither a device nor the Vulkan headers.
#pragma once

#include <cstring>

namespace rt::d3d12_xs::backend {

// LEONIDA_D3D12=vulkan, exactly: the one value that selects the Vulkan-backed device.
inline bool vulkan_requested(const char* value) { return value && std::strcmp(value, "vulkan") == 0; }

// What to tell the user when the variable asks for the Vulkan-backed device and this runtime has none, else null.
// A runtime configured without the Vulkan headers is built without LEONIDA_VULKAN_DEVICE: it used to take the variable
// without a word and run the title on the host's D3D12 device, which stops somewhere else and looks like a boot result.
inline const char* vulkan_not_built_note(const char* value, bool vulkan_built) {
    if (vulkan_built || !vulkan_requested(value)) return nullptr;
    return "LEONIDA_D3D12=vulkan is set, but this runtime was built without the Vulkan-backed device (its configure "
           "found no Vulkan headers), so the title gets the host's D3D12 device instead. Configure again with "
           "VULKAN_SDK set or -DLEONIDA_VULKAN_HEADERS_FETCH=ON; -DLEONIDA_REQUIRE_VULKAN=ON makes a configure "
           "without the headers fail";
}

}  // namespace rt::d3d12_xs::backend
