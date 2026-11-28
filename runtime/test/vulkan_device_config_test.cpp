// leonida::device_create_config (src/vulkan_device_config.h): what of a VkDeviceCreateInfo the pipeline journal's stamp
// records, on the CPU.
#include "../src/vulkan_device_config.h"
#include "../src/vulkan_device_fault.h"

#include <cstdio>
#include <utility>
#include <vector>

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "%s:%d: CHECK(%s) failed\n", __FILE__, __LINE__, #cond); \
            return 1;                                                            \
        }                                                                        \
    } while (0)

namespace {

VkBool32 diagnostics_supported = VK_TRUE;

void VKAPI_PTR query_features(VkPhysicalDevice, VkPhysicalDeviceFeatures2* features) {
    for (auto* next = static_cast<VkBaseOutStructure*>(features->pNext); next; next = next->pNext) {
        if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FAULT_FEATURES_EXT)
            reinterpret_cast<VkPhysicalDeviceFaultFeaturesEXT*>(next)->deviceFault = VK_TRUE;
        else if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DIAGNOSTICS_CONFIG_FEATURES_NV)
            reinterpret_cast<VkPhysicalDeviceDiagnosticsConfigFeaturesNV*>(next)->diagnosticsConfig =
                diagnostics_supported;
    }
}

// The runtime's shape: core features, a 1.2/1.3 chain and a few extensions; `fault`: LEONIDA_DEVICE_FAULT's request
// through DeviceFaultFeatures (the device-fault test's fake device), 0 none, 1 with the diagnostics config, 2 without.
struct Device {
    VkPhysicalDeviceFeatures core{};
    VkPhysicalDeviceVulkan13Features v13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceVulkan12Features v12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &v13};
    std::vector<const char*> extensions{VK_KHR_SWAPCHAIN_EXTENSION_NAME, VK_EXT_ROBUSTNESS_2_EXTENSION_NAME};
    leonida::DeviceFaultFeatures fault_features;
    VkDeviceCreateInfo info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &v12};

    explicit Device(int fault = 0, VkDeviceDiagnosticsConfigFlagsNV extra = 0) {
        core.robustBufferAccess = VK_TRUE;
        core.shaderInt64 = VK_TRUE;
        v12.bufferDeviceAddress = VK_TRUE;
        v13.synchronization2 = VK_TRUE;
        if (fault) {
            static const VkExtensionProperties advertised[] = {{VK_EXT_DEVICE_FAULT_EXTENSION_NAME, 2},
                                                               {VK_NV_DEVICE_DIAGNOSTIC_CHECKPOINTS_EXTENSION_NAME, 2},
                                                               {VK_NV_DEVICE_DIAGNOSTICS_CONFIG_EXTENSION_NAME, 2}};
            fault_features.enable(VK_NULL_HANDLE, query_features, advertised, extensions, info,
                                  {true, fault == 1, extra});
        }
        info.pEnabledFeatures = &core;
    }
    std::uint64_t config() {
        info.enabledExtensionCount = std::uint32_t(extensions.size());
        info.ppEnabledExtensionNames = extensions.data();
        return leonida::device_create_config(info);
    }
};

}  // namespace

int main() {
    Device base;
    const std::uint64_t plain = base.config();
    CHECK(plain != 0);
    CHECK(Device().config() == plain);  // the same create info, another process: the same value

    // Extension order does not matter; the set does.
    Device reordered;
    std::swap(reordered.extensions[0], reordered.extensions[1]);
    CHECK(reordered.config() == plain);
    Device more;
    more.extensions.push_back(VK_EXT_CUSTOM_BORDER_COLOR_EXTENSION_NAME);
    CHECK(more.config() != plain);

    // Feature bits, core and chained.
    Device no_int64;
    no_int64.core.shaderInt64 = VK_FALSE;
    CHECK(no_int64.config() != plain);
    Device v13_last;
    v13_last.v13.maintenance4 = VK_TRUE;  // the last member: counted
    CHECK(v13_last.config() != plain);
    Device v12_last;
    v12_last.v12.subgroupBroadcastDynamicId = VK_TRUE;
    CHECK(v12_last.config() != plain);

    // LEONIDA_DEVICE_FAULT=1 (diagnostics config) and =2 (fault info and checkpoints only) differ from a plain device
    // and from each other; Aftermath's extra flags count.
    const std::uint64_t with_config = Device(1).config();
    const std::uint64_t without_config = Device(2).config();
    CHECK(with_config != plain && without_config != plain && with_config != without_config);
    CHECK(Device(1).config() == with_config);
    const std::uint64_t shader_errors =
        Device(1, VK_DEVICE_DIAGNOSTICS_CONFIG_ENABLE_SHADER_ERROR_REPORTING_BIT_NV).config();
    CHECK(shader_errors != with_config && shader_errors != plain);

    // A device without the config feature gets no config struct: =1 is then the same as =2.
    diagnostics_supported = VK_FALSE;
    CHECK(Device(1).config() == without_config);
    diagnostics_supported = VK_TRUE;

    // Nothing enabled at all still hashes to a nonzero value.
    VkDeviceCreateInfo empty{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    CHECK(leonida::device_create_config(empty) != 0 && leonida::device_create_config(empty) != plain);

    std::printf("vulkan_device_config: ok\n");
    return 0;
}
