#pragma once
// LEONIDA_ADDRESS_BINDINGS=1, with LEONIDA_DEVICE_FAULT=1: names what the driver had bound at a device-fault address
// (RT-FAULT-ADDRESS-OWNER). VK_EXT_device_address_binding_report has the driver report the GPU virtual address range
// of every buffer, image and allocation as it is bound and unbound, to the instance's debug messengers. This keeps
// the live ranges and the most recent unbound ones, so the device-fault report (vulkan_device_fault.h) can say which
// object contains a faulting address, which objects end right below it and start right above it, and which objects
// held it before. An address that none of them covers is a GPU address no object of this device was ever given.
//
// The record and the report are pure (record_binding, report); the Vulkan side is install_messenger before
// vkCreateDevice's bindings start and enable_device on the device's create chain.

#include <vulkan/vulkan.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <span>
#include <string>
#include <vector>

namespace leonida::address_binding {

struct Entry {
    std::uint64_t base = 0, size = 0;
    VkObjectType type = VK_OBJECT_TYPE_UNKNOWN;
    std::uint64_t handle = 0;
    bool internal = false;                 // VK_DEVICE_ADDRESS_BINDING_INTERNAL_OBJECT_BIT_EXT
    double bound_at = 0, unbound_at = 0;   // seconds since the record started; unbound_at 0 while live
};

class Record {
public:
    static constexpr std::size_t kept_unbound = 16384;

    // One report: bound, or unbound (matched to its live entry by base, handle and size).
    void record(Entry e, bool bind, double now) {
        std::lock_guard lock(mutex_);
        if (bind) {
            e.bound_at = now;
            live_.emplace(e.base, e);
            ++binds_;
            return;
        }
        ++unbinds_;
        auto [first, last] = live_.equal_range(e.base);
        for (auto it = first; it != last; ++it) {
            if (it->second.handle != e.handle || it->second.size != e.size) continue;
            e = it->second;
            live_.erase(it);
            break;
        }
        e.unbound_at = now;
        unbound_.push_back(e);
        if (unbound_.size() > kept_unbound) unbound_.pop_front();
    }

    // What was bound at and around the `precision`-sized page of `address`, one line per finding.
    std::string report(std::uint64_t address, std::uint64_t precision) const {
        std::lock_guard lock(mutex_);
        std::string out;
        char line[160];
        std::snprintf(line, sizeof line, "0x%" PRIx64 " (precision 0x%" PRIx64 "): %" PRIu64 " binds, %" PRIu64
                                         " unbinds, %zu live\n",
                      address, precision, binds_, unbinds_, live_.size());
        out += line;
        const std::uint64_t lo = precision > 1 ? address & ~(precision - 1) : address;
        const std::uint64_t hi = lo + (precision > 1 ? precision : 1);
        const auto overlaps = [&](const Entry& e) { return e.base < hi && e.base + e.size > lo; };
        int containing = 0;
        const Entry* below = nullptr;
        const Entry* above = nullptr;
        // "containing" only for an object the address itself is in; one that shares only the reported page with it is
        // "on the page", since the driver gives the fault address no closer than `precision`.
        const auto inside = [&](const Entry& e) { return address >= e.base && address - e.base < e.size; };
        for (const auto& [base, e] : live_) {
            if (overlaps(e)) {
                if (containing++ < 16) out += describe(inside(e) ? "live, containing" : "live, on the page", e, address);
                continue;
            }
            if (base + e.size <= lo && (!below || base + e.size > below->base + below->size)) below = &e;
            if (base >= hi && (!above || base < above->base)) above = &e;
        }
        if (below) out += describe("live, nearest below", *below, address);
        if (above) out += describe("live, nearest above", *above, address);
        int held = 0;
        for (auto it = unbound_.rbegin(); it != unbound_.rend() && held < 16; ++it)
            if (overlaps(*it)) {
                out += describe(inside(*it) ? "unbound, held it" : "unbound, held the page", *it, address);
                ++held;
            }
        if (!containing && !held) out += "  nothing live or recently unbound there\n";
        return out;
    }

private:
    static const char* type_name(VkObjectType t) {
        switch (t) {
            case VK_OBJECT_TYPE_BUFFER: return "buffer";
            case VK_OBJECT_TYPE_IMAGE: return "image";
            case VK_OBJECT_TYPE_DEVICE_MEMORY: return "memory";
            case VK_OBJECT_TYPE_ACCELERATION_STRUCTURE_KHR: return "acceleration structure";
            case VK_OBJECT_TYPE_COMMAND_BUFFER: return "command buffer";
            case VK_OBJECT_TYPE_PIPELINE: return "pipeline";
            case VK_OBJECT_TYPE_QUERY_POOL: return "query pool";
            case VK_OBJECT_TYPE_DESCRIPTOR_POOL: return "descriptor pool";
            default: return "object";
        }
    }

    static std::string describe(const char* what, const Entry& e, std::uint64_t address) {
        const std::uint64_t end = e.base + e.size;
        const char* relation;
        std::uint64_t distance;
        if (address >= e.base && address < end) {
            relation = "into";
            distance = address - e.base;
        } else if (address >= end) {
            relation = "past the end of";
            distance = address - end;
        } else {
            relation = "before";
            distance = e.base - address;
        }
        char line[320];
        std::snprintf(line, sizeof line,
                      "  %s: %s%s 0x%" PRIx64 "..0x%" PRIx64 " (%" PRIu64 " KiB) handle 0x%" PRIx64
                      " bound at %.1f s%s; the address is 0x%" PRIx64 " %s it\n",
                      what, type_name(e.type), e.internal ? " (driver-internal)" : "", e.base, end, e.size >> 10,
                      e.handle, e.bound_at, e.unbound_at ? (", unbound at " + std::to_string(e.unbound_at) + " s").c_str() : "",
                      distance, relation);
        return line;
    }

    mutable std::mutex mutex_;
    std::multimap<std::uint64_t, Entry> live_;  // by base address
    std::deque<Entry> unbound_;                 // the most recent last
    std::uint64_t binds_ = 0, unbinds_ = 0;
};

struct State {
    std::mutex install;  // the messenger fields below; the record has a lock of its own
    Record record;
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    VkPhysicalDeviceAddressBindingReportFeaturesEXT feature{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ADDRESS_BINDING_REPORT_FEATURES_EXT};
    VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
    VkInstance messenger_instance = VK_NULL_HANDLE;
    PFN_vkDestroyDebugUtilsMessengerEXT destroy_messenger = nullptr;
    // A device enabled the report and this process's messenger records it. Read at a device fault on whatever thread
    // notices it, so atomic.
    std::atomic<bool> on{false};
};

inline State& state() {
    static State s;
    return s;
}

inline bool requested() {
    const char* v = std::getenv("LEONIDA_ADDRESS_BINDINGS");
    return v && v[0] == '1' && !v[1];
}

inline VKAPI_ATTR VkBool32 VKAPI_CALL on_message(VkDebugUtilsMessageSeverityFlagBitsEXT,
                                                 VkDebugUtilsMessageTypeFlagsEXT types,
                                                 const VkDebugUtilsMessengerCallbackDataEXT* data, void*) {
    if (!(types & VK_DEBUG_UTILS_MESSAGE_TYPE_DEVICE_ADDRESS_BINDING_BIT_EXT) || !data) return VK_FALSE;
    const VkDeviceAddressBindingCallbackDataEXT* binding = nullptr;
    for (auto* p = static_cast<const VkBaseInStructure*>(data->pNext); p; p = p->pNext)
        if (p->sType == VK_STRUCTURE_TYPE_DEVICE_ADDRESS_BINDING_CALLBACK_DATA_EXT)
            binding = reinterpret_cast<const VkDeviceAddressBindingCallbackDataEXT*>(p);
    if (!binding) return VK_FALSE;
    State& s = state();
    Entry e;
    e.base = binding->baseAddress;
    e.size = binding->size;
    e.internal = (binding->flags & VK_DEVICE_ADDRESS_BINDING_INTERNAL_OBJECT_BIT_EXT) != 0;
    if (data->objectCount && data->pObjects) {
        e.type = data->pObjects[0].objectType;
        e.handle = data->pObjects[0].objectHandle;
    }
    const double now = std::chrono::duration<double>(std::chrono::steady_clock::now() - s.start).count();
    s.record.record(e, binding->bindingType == VK_DEVICE_ADDRESS_BINDING_TYPE_BIND_EXT, now);
    return VK_FALSE;
}

// Before vkCreateDevice: adds the extension and its feature to `device`'s chain when the device has them and
// `instance` has this process's messenger. Without the messenger nothing would record the reports, so the device is
// left as it is and a fault report says the bindings were not recorded.
inline bool enable_device(VkInstance instance, VkPhysicalDevice physical, PFN_vkGetPhysicalDeviceFeatures2 query,
                          std::span<const VkExtensionProperties> advertised, std::vector<const char*>& extensions,
                          VkDeviceCreateInfo& device) {
    if (!requested() || !query) return false;
    {
        State& s = state();
        std::lock_guard lock(s.install);
        if (!s.messenger || s.messenger_instance != instance) {
            std::fprintf(stderr, "[rt] address-binding: no messenger on this device's instance; bindings are not "
                                 "recorded\n");
            return false;
        }
    }
    const bool has = std::any_of(advertised.begin(), advertised.end(), [](const auto& e) {
        return std::strcmp(e.extensionName, VK_EXT_DEVICE_ADDRESS_BINDING_REPORT_EXTENSION_NAME) == 0;
    });
    State& s = state();
    if (!has) {
        std::fprintf(stderr, "[rt] address-binding: the device has no VK_EXT_device_address_binding_report\n");
        return false;
    }
    s.feature.pNext = nullptr;
    VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &s.feature};
    query(physical, &features);
    if (!s.feature.reportAddressBinding) {
        std::fprintf(stderr, "[rt] address-binding: reportAddressBinding is not supported\n");
        return false;
    }
    extensions.push_back(VK_EXT_DEVICE_ADDRESS_BINDING_REPORT_EXTENSION_NAME);
    s.feature.pNext = const_cast<void*>(device.pNext);
    device.pNext = &s.feature;
    s.on = true;
    return true;
}

// After vkCreateInstance with VK_EXT_debug_utils and before vkCreateDevice: bindings reach the instance's messengers
// as they happen.
inline void install_messenger(VkInstance instance, PFN_vkGetInstanceProcAddr get) {
    if (!requested() || !instance || !get) return;
    const auto create =
        reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(get(instance, "vkCreateDebugUtilsMessengerEXT"));
    if (!create) {
        std::fprintf(stderr, "[rt] address-binding: no vkCreateDebugUtilsMessengerEXT\n");
        return;
    }
    VkDebugUtilsMessengerCreateInfoEXT info{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
    info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_DEVICE_ADDRESS_BINDING_BIT_EXT;
    info.pfnUserCallback = &on_message;
    State& s = state();
    std::lock_guard lock(s.install);
    // One record per process: a second instance (a second device in one process) would mix its GPU addresses with
    // the first's, and overwriting the handle would leave the first messenger alive past its instance. So the
    // second one is refused, and its device is not enabled.
    if (s.messenger) {
        std::fprintf(stderr, "[rt] address-binding: a messenger is already installed for another instance; this "
                             "instance's bindings are not recorded\n");
        return;
    }
    VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
    const VkResult r = create(instance, &info, nullptr, &messenger);
    if (r == VK_SUCCESS) {
        s.messenger = messenger;
        s.messenger_instance = instance;
        s.destroy_messenger = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
            get(instance, "vkDestroyDebugUtilsMessengerEXT"));
    }
    std::fprintf(stderr, "[rt] address-binding: messenger %s\n", r == VK_SUCCESS ? "installed" : "failed");
}

// Before vkDestroyInstance: a messenger may not outlive its instance. What it recorded stays.
inline void uninstall_messenger(VkInstance instance) {
    State& s = state();
    std::lock_guard lock(s.install);
    if (!s.messenger || s.messenger_instance != instance) return;
    if (s.destroy_messenger) s.destroy_messenger(instance, s.messenger, nullptr);
    s.messenger = VK_NULL_HANDLE;
    s.messenger_instance = VK_NULL_HANDLE;
    s.destroy_messenger = nullptr;
    s.on = false;
}

// At a device fault: what the driver had bound at and around `address`, on stderr.
inline void describe(std::uint64_t address, std::uint64_t precision) {
    State& s = state();
    if (!s.on) {
        // Asked for but not recording: say so, rather than leave a report that reads as "no object there".
        if (requested())
            std::fprintf(stderr, "[rt] address-binding: 0x%" PRIx64 ": bindings were not recorded for this device (no "
                                 "messenger or no VK_EXT_device_address_binding_report; see the address-binding "
                                 "lines at startup)\n", address);
        return;
    }
    const std::string text = s.record.report(address, precision);
    std::size_t at = 0;
    while (at < text.size()) {
        const std::size_t end = text.find('\n', at);
        std::fprintf(stderr, "[rt] address-binding: %.*s\n", int((end == std::string::npos ? text.size() : end) - at),
                     text.c_str() + at);
        if (end == std::string::npos) break;
        at = end + 1;
    }
}

}  // namespace leonida::address_binding
