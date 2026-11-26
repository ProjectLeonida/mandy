// Loads Vulkan at run time. The macOS test tools default to MoltenVK.
#pragma once

#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include <vulkan/vulkan.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#ifdef __APPLE__
// Behind the Vulkan loader (LEONIDA_VULKAN_LOADER, the way to reach VK_LAYER_KHRONOS_validation on a Mac), MoltenVK
// is a portability driver: the loader lists it only for an instance created with VK_KHR_portability_enumeration and
// VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR, and a device it exposes must enable VK_KHR_portability_subset.
// None of the test tools ask for either, so through the loader they found no device ("Found drivers that contain
// devices which support the portability subset, but the instance does not enumerate portability drivers").
// load_vulkan() then hands out these wrappers, which add both where the loader and the device offer them.
namespace leonida_vulkan_loader {
inline PFN_vkGetInstanceProcAddr& real_gipa() {
    static PFN_vkGetInstanceProcAddr gipa = nullptr;
    return gipa;
}
inline VkInstance& last_instance() {
    static VkInstance instance = VK_NULL_HANDLE;
    return instance;
}
inline bool requested(const char* const* names, uint32_t count, const char* name) {
    for (uint32_t i = 0; i < count; ++i)
        if (names[i] && !std::strcmp(names[i], name)) return true;
    return false;
}
inline bool offered(const std::vector<VkExtensionProperties>& extensions, const char* name) {
    for (const auto& e : extensions)
        if (!std::strcmp(e.extensionName, name)) return true;
    return false;
}
inline VKAPI_ATTR VkResult VKAPI_CALL create_instance(const VkInstanceCreateInfo* info, const VkAllocationCallbacks* a,
                                                      VkInstance* out) {
    const auto real = reinterpret_cast<PFN_vkCreateInstance>(real_gipa()(nullptr, "vkCreateInstance"));
    const auto enumerate = reinterpret_cast<PFN_vkEnumerateInstanceExtensionProperties>(
        real_gipa()(nullptr, "vkEnumerateInstanceExtensionProperties"));
    if (!real) return VK_ERROR_INITIALIZATION_FAILED;
    VkInstanceCreateInfo copy = *info;
    std::vector<const char*> names(info->ppEnabledExtensionNames,
                                   info->ppEnabledExtensionNames + info->enabledExtensionCount);
    uint32_t count = 0;
    std::vector<VkExtensionProperties> extensions;
    if (enumerate && enumerate(nullptr, &count, nullptr) == VK_SUCCESS) {
        extensions.resize(count);
        if (enumerate(nullptr, &count, extensions.data()) != VK_SUCCESS) extensions.clear();
        extensions.resize(count);
    }
    const char* portability = "VK_KHR_portability_enumeration";
    if (offered(extensions, portability)) {
        if (!requested(names.data(), uint32_t(names.size()), portability)) names.push_back(portability);
        copy.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
        copy.enabledExtensionCount = uint32_t(names.size());
        copy.ppEnabledExtensionNames = names.data();
    }
    const VkResult result = real(&copy, a, out);
    if (result == VK_SUCCESS) last_instance() = *out;
    return result;
}
inline VKAPI_ATTR VkResult VKAPI_CALL create_device(VkPhysicalDevice physical, const VkDeviceCreateInfo* info,
                                                    const VkAllocationCallbacks* a, VkDevice* out) {
    const VkInstance instance = last_instance();
    const auto real = reinterpret_cast<PFN_vkCreateDevice>(real_gipa()(instance, "vkCreateDevice"));
    const auto enumerate = reinterpret_cast<PFN_vkEnumerateDeviceExtensionProperties>(
        real_gipa()(instance, "vkEnumerateDeviceExtensionProperties"));
    if (!real) return VK_ERROR_INITIALIZATION_FAILED;
    VkDeviceCreateInfo copy = *info;
    std::vector<const char*> names(info->ppEnabledExtensionNames,
                                   info->ppEnabledExtensionNames + info->enabledExtensionCount);
    uint32_t count = 0;
    std::vector<VkExtensionProperties> extensions;
    if (enumerate && enumerate(physical, nullptr, &count, nullptr) == VK_SUCCESS) {
        extensions.resize(count);
        if (enumerate(physical, nullptr, &count, extensions.data()) != VK_SUCCESS) extensions.clear();
        extensions.resize(count);
    }
    const char* subset = "VK_KHR_portability_subset";
    if (offered(extensions, subset) && !requested(names.data(), uint32_t(names.size()), subset)) {
        names.push_back(subset);
        copy.enabledExtensionCount = uint32_t(names.size());
        copy.ppEnabledExtensionNames = names.data();
    }
    return real(physical, &copy, a, out);
}
inline VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL get_instance_proc_addr(VkInstance instance, const char* name);
inline PFN_vkVoidFunction wrapped(const char* name) {
    if (!std::strcmp(name, "vkCreateInstance")) return reinterpret_cast<PFN_vkVoidFunction>(&create_instance);
    if (!std::strcmp(name, "vkCreateDevice")) return reinterpret_cast<PFN_vkVoidFunction>(&create_device);
    if (!std::strcmp(name, "vkGetInstanceProcAddr"))
        return reinterpret_cast<PFN_vkVoidFunction>(&get_instance_proc_addr);
    return nullptr;
}
inline VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL get_instance_proc_addr(VkInstance instance, const char* name) {
    if (name)
        if (const auto f = wrapped(name)) return f;
    return real_gipa()(instance, name);
}
}  // namespace leonida_vulkan_loader
#endif

inline PFN_vkGetInstanceProcAddr load_vulkan() {
    // Keep the Vulkan library loaded for the lifetime of the tool.
#ifdef _WIN32
    constexpr auto library_name = "vulkan-1.dll";
    static const auto library = LoadLibraryA(library_name);
    const auto entry = library ? GetProcAddress(library, "vkGetInstanceProcAddr") : nullptr;
#elif defined(__APPLE__)
    static const auto entry = []() -> void* {
        // MoltenVK compiles MSL with fast-math on by default, which lets Metal fold an f32 compare of a NaN with
        // itself to "ordered" (v_cmpx_eq/le_f32 on (NaN, NaN) give true; ps_vcmpx_forms). Translated shaders need
        // IEEE results, so turn it off before MoltenVK reads its configuration, unless the user set the variable
        // (docs/design.md, "MoltenVK fast-math"). The same variable reaches MoltenVK behind the Vulkan loader.
        setenv("MVK_CONFIG_FAST_MATH_ENABLED", "0", 0);
        // The default is direct MoltenVK: a portability driver would otherwise require instance opt-in.
        const char* requested = std::getenv("LEONIDA_VULKAN_LOADER");
        const bool overridden = requested && *requested;
        const char* name = overridden ? requested : "libMoltenVK.dylib";
        auto library = dlopen(name, RTLD_NOW | RTLD_LOCAL);
        if (!library && !overridden) {
            name = "/opt/homebrew/lib/libMoltenVK.dylib";
            library = dlopen(name, RTLD_NOW | RTLD_LOCAL);
        }
        if (!library) {
            std::fprintf(stderr, "could not load %s: %s\n", name, dlerror());
            return nullptr;
        }
        const auto symbol = dlsym(library, "vkGetInstanceProcAddr");
        if (!symbol) std::fprintf(stderr, "%s has no vkGetInstanceProcAddr\n", name);
        if (!symbol || !overridden) return symbol;
        // Through the loader: opt in to portability drivers (MoltenVK) for the tool (see above).
        leonida_vulkan_loader::real_gipa() = reinterpret_cast<PFN_vkGetInstanceProcAddr>(symbol);
        return reinterpret_cast<void*>(&leonida_vulkan_loader::get_instance_proc_addr);
    }();
#else
    constexpr auto library_name = "libvulkan.so.1";
    static const auto library = dlopen(library_name, RTLD_NOW | RTLD_LOCAL);
    const auto entry = library ? dlsym(library, "vkGetInstanceProcAddr") : nullptr;
#endif
#ifndef __APPLE__
    if (!entry) std::fprintf(stderr, "could not load vkGetInstanceProcAddr from %s\n", library_name);
#endif
    return reinterpret_cast<PFN_vkGetInstanceProcAddr>(entry);
}
