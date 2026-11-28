// What a VkDeviceCreateInfo asks of the driver that its pipeline compiles can depend on, as one 64-bit value: the
// pipeline journal's stamp (pipeline_cache_journal.h) records it next to the writing binary's build.
//
// On NVIDIA (621.01, RTX 4070 SUPER) a journal file written by a device without the diagnostics config
// (VK_NV_device_diagnostics_config, LEONIDA_DEVICE_FAULT=1 or LEONIDA_AFTERMATH=1) gave a device with it no hits, and
// the other way round: the driver's own key holds device-create state that the blob header does not show. Hashed:
//   - the enabled extensions, in any order;
//   - pEnabledFeatures;
//   - each structure on the pNext chain by its type, and the contents of those listed in add_struct (the core feature
//     structures, robustness2, device fault and the diagnostics config with its flags); others count by presence.
// Queue create infos are left out: they do not change a compile.
//
// Header only; tested by vulkan_device_config_test.
#pragma once

#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string_view>
#include <vector>

namespace leonida {

namespace device_config_detail {

struct Fnv {
    std::uint64_t hash = 0xcbf29ce484222325ull;
    void bytes(const void* data, std::size_t size) {
        const auto* p = static_cast<const std::uint8_t*>(data);
        for (std::size_t i = 0; i < size; ++i) {
            hash ^= p[i];
            hash *= 0x100000001b3ull;
        }
    }
    void value(std::uint32_t v) { bytes(&v, sizeof(v)); }
};

// The members after sType and pNext, up to and including `last` (so trailing padding is never read).
template <typename T>
void add_members(Fnv& h, const T& s, std::size_t last_end) {
    const std::size_t first = offsetof(T, pNext) + sizeof(void*);
    h.bytes(reinterpret_cast<const std::uint8_t*>(&s) + first, last_end - first);
}

#define LEONIDA_DEVICE_CONFIG_MEMBERS(T, last) \
    add_members(h, *reinterpret_cast<const T*>(s), offsetof(T, last) + sizeof(T::last))

inline void add_struct(Fnv& h, const VkBaseInStructure* s) {
    h.value(std::uint32_t(s->sType));
    switch (s->sType) {
    case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2:
        LEONIDA_DEVICE_CONFIG_MEMBERS(VkPhysicalDeviceFeatures2, features);
        break;
    case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES:
        LEONIDA_DEVICE_CONFIG_MEMBERS(VkPhysicalDeviceVulkan11Features, shaderDrawParameters);
        break;
    case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES:
        LEONIDA_DEVICE_CONFIG_MEMBERS(VkPhysicalDeviceVulkan12Features, subgroupBroadcastDynamicId);
        break;
    case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES:
        LEONIDA_DEVICE_CONFIG_MEMBERS(VkPhysicalDeviceVulkan13Features, maintenance4);
        break;
    case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT:
        LEONIDA_DEVICE_CONFIG_MEMBERS(VkPhysicalDeviceRobustness2FeaturesEXT, nullDescriptor);
        break;
    case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FAULT_FEATURES_EXT:
        LEONIDA_DEVICE_CONFIG_MEMBERS(VkPhysicalDeviceFaultFeaturesEXT, deviceFaultVendorBinary);
        break;
    case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DIAGNOSTICS_CONFIG_FEATURES_NV:
        LEONIDA_DEVICE_CONFIG_MEMBERS(VkPhysicalDeviceDiagnosticsConfigFeaturesNV, diagnosticsConfig);
        break;
    case VK_STRUCTURE_TYPE_DEVICE_DIAGNOSTICS_CONFIG_CREATE_INFO_NV:
        LEONIDA_DEVICE_CONFIG_MEMBERS(VkDeviceDiagnosticsConfigCreateInfoNV, flags);
        break;
    default:
        break;
    }
}

#undef LEONIDA_DEVICE_CONFIG_MEMBERS

}  // namespace device_config_detail

// Never 0, so a stamp's 0 can mean "not recorded".
inline std::uint64_t device_create_config(const VkDeviceCreateInfo& info) {
    using namespace device_config_detail;
    Fnv h;
    h.bytes("device-create-config", 20);
    std::vector<std::string_view> extensions;
    for (std::uint32_t i = 0; i < info.enabledExtensionCount; ++i)
        if (info.ppEnabledExtensionNames && info.ppEnabledExtensionNames[i])
            extensions.emplace_back(info.ppEnabledExtensionNames[i]);
    std::sort(extensions.begin(), extensions.end());
    extensions.erase(std::unique(extensions.begin(), extensions.end()), extensions.end());
    h.value(std::uint32_t(extensions.size()));
    for (const std::string_view name : extensions) {
        h.bytes(name.data(), name.size());
        h.value(0);
    }
    h.value(info.pEnabledFeatures ? 1u : 0u);
    if (info.pEnabledFeatures) h.bytes(info.pEnabledFeatures, sizeof(VkPhysicalDeviceFeatures));
    // A chain is short; the bound only stops a cycle.
    std::size_t links = 0;
    for (auto* s = static_cast<const VkBaseInStructure*>(info.pNext); s && links < 256; s = s->pNext, ++links)
        add_struct(h, s);
    return h.hash ? h.hash : 1;
}

}  // namespace leonida
