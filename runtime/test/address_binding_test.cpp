// src/address_binding.h: the record of VK_EXT_device_address_binding_report reports and what it says about a
// device-fault address. Synthetic reports only: no Vulkan device, no GPU.
#include "../src/address_binding.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#ifndef _WIN32
static int _putenv_s(const char* name, const char* value) { return setenv(name, value, 1); }
#endif

#define CHECK(condition) do {                                        \
    if (!(condition)) {                                              \
        std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); \
        return 1;                                                    \
    }                                                                \
} while (0)

using namespace leonida::address_binding;

static bool has(const std::string& text, const char* part) { return text.find(part) != std::string::npos; }

static Entry entry(std::uint64_t base, std::uint64_t size, VkObjectType type, std::uint64_t handle) {
    Entry e;
    e.base = base;
    e.size = size;
    e.type = type;
    e.handle = handle;
    return e;
}

static int test_record() {
    Record r;
    r.record(entry(0x10000, 0x1cf000, VK_OBJECT_TYPE_IMAGE, 0xa), true, 1.0);   // ends at 0x1df000
    r.record(entry(0x200000, 0x10000, VK_OBJECT_TYPE_BUFFER, 0xb), true, 2.0);
    r.record(entry(0x400000, 0x1000, VK_OBJECT_TYPE_BUFFER, 0xc), true, 3.0);
    r.record(entry(0x400000, 0x1000, VK_OBJECT_TYPE_BUFFER, 0xc), false, 4.0);  // unbound again

    // Inside the image: it contains the page.
    std::string text = r.report(0x1ce000, 0x1000);
    CHECK(has(text, "3 binds, 1 unbinds, 2 live"));
    CHECK(has(text, "live, containing: image 0x10000..0x1df000"));
    CHECK(has(text, "0x1be000 into it"));

    // The page just past the image and below the buffer: no owner, the neighbours either side.
    text = r.report(0x1e0123, 0x1000);
    CHECK(!has(text, "containing"));
    CHECK(has(text, "nearest below: image"));
    CHECK(has(text, "0x1123 past the end of it"));  // from the address, not its page
    CHECK(has(text, "nearest above: buffer 0x200000"));
    CHECK(has(text, "nothing live or recently unbound there"));

    // Where the unbound buffer was: it held the page, and nothing live does.
    text = r.report(0x400800, 0x1000);
    CHECK(has(text, "unbound, held it: buffer 0x400000..0x401000"));
    CHECK(has(text, "unbound at 4.0"));
    CHECK(!has(text, "nothing live"));

    // Far above everything: only the nearest below, which is the highest live object.
    text = r.report(0x29b52a80000, 0x1000);
    CHECK(has(text, "nearest below: buffer 0x200000"));
    CHECK(!has(text, "nearest above"));
    return 0;
}

// The edges of an object and of the reported page (review of RT-FAULT-ADDRESS-OWNER).
static int test_boundaries() {
    Record r;
    r.record(entry(0x10000, 0x1cf000, VK_OBJECT_TYPE_IMAGE, 0xa), true, 1.0);  // ends at 0x1df000
    // The last byte of the image is in it.
    std::string text = r.report(0x1defff, 0x1000);
    CHECK(has(text, "live, containing: image"));
    CHECK(has(text, "0x1cefff into it"));
    // The byte right after its end is not, whatever the precision of a page that starts there.
    text = r.report(0x1df000, 0x1000);
    CHECK(!has(text, "live, containing") && !has(text, "on the page"));
    CHECK(has(text, "nearest below: image") && has(text, "0x0 past the end of it"));
    text = r.report(0x1df000, 1);
    CHECK(!has(text, "live, containing") && has(text, "nearest below: image"));
    // A coarse page that reaches back into the image: the image is on the page, but does not contain the address.
    text = r.report(0x1df010, 0x2000);
    CHECK(has(text, "live, on the page: image") && !has(text, "live, containing"));
    CHECK(has(text, "0x10 past the end of it"));
    CHECK(!has(text, "nothing live"));
    // The same address at byte precision: no object there, only the neighbour.
    text = r.report(0x1df010, 1);
    CHECK(!has(text, "on the page") && has(text, "nearest below: image"));
    CHECK(has(text, "nothing live or recently unbound there"));
    // The first byte of the image, and the one before it.
    CHECK(has(r.report(0x10000, 1), "live, containing: image"));
    text = r.report(0xffff, 1);
    CHECK(has(text, "nearest above: image") && has(text, "0x1 before it"));
    // An unbound object that only shares the page is "held the page", not "held it".
    r.record(entry(0x300000, 0x800, VK_OBJECT_TYPE_BUFFER, 0xd), true, 2.0);
    r.record(entry(0x300000, 0x800, VK_OBJECT_TYPE_BUFFER, 0xd), false, 3.0);
    CHECK(has(r.report(0x300900, 0x1000), "unbound, held the page: buffer 0x300000..0x300800"));
    CHECK(has(r.report(0x300100, 0x1000), "unbound, held it: buffer 0x300000..0x300800"));
    return 0;
}

// An unbind removes only the live entry with the same base, handle and size; the unbound list is capped.
static int test_unbind_matching_and_cap() {
    Record r;
    r.record(entry(0x100000, 0x1000, VK_OBJECT_TYPE_BUFFER, 0x1), true, 1.0);
    r.record(entry(0x100000, 0x1000, VK_OBJECT_TYPE_BUFFER, 0x2), true, 1.0);  // aliased: same base, other handle
    r.record(entry(0x100000, 0x2000, VK_OBJECT_TYPE_BUFFER, 0x2), false, 2.0);  // size differs: no live entry goes
    CHECK(has(r.report(0x100000, 0x1000), "2 live"));
    r.record(entry(0x100000, 0x1000, VK_OBJECT_TYPE_BUFFER, 0x3), false, 2.0);  // handle differs: none goes
    CHECK(has(r.report(0x100000, 0x1000), "2 live"));
    r.record(entry(0x100000, 0x1000, VK_OBJECT_TYPE_BUFFER, 0x2), false, 3.0);  // matches the second only
    std::string text = r.report(0x100000, 0x1000);
    CHECK(has(text, "1 live"));
    CHECK(has(text, "live, containing: buffer 0x100000..0x101000 (4 KiB) handle 0x1"));
    CHECK(!has(text, "live, containing: buffer 0x100000..0x101000 (4 KiB) handle 0x2"));

    // Past kept_unbound, the oldest unbound objects are forgotten and the newest kept.
    Record capped;
    const std::uint64_t n = Record::kept_unbound + 8;
    for (std::uint64_t i = 0; i < n; ++i) {
        const auto e = entry(0x1000000 + i * 0x1000, 0x1000, VK_OBJECT_TYPE_BUFFER, 0x100 + i);
        capped.record(e, true, 1.0);
        capped.record(e, false, 2.0);
    }
    CHECK(has(capped.report(0x1000000, 0x1000), "nothing live or recently unbound there"));  // the first, dropped
    CHECK(has(capped.report(0x1000000 + 7 * 0x1000, 0x1000), "nothing live or recently unbound"));  // the 8th
    CHECK(has(capped.report(0x1000000 + 8 * 0x1000, 0x1000), "unbound, held it"));  // the oldest one kept
    CHECK(has(capped.report(0x1000000 + (n - 1) * 0x1000, 0x1000), "unbound, held it"));
    return 0;
}

// The messenger's thread records while a device-fault report reads.
static int test_threads() {
    Record r;
    constexpr int threads = 4, per = 2000;
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; ++t)
        pool.emplace_back([&r, t] {
            for (int i = 0; i < per; ++i) {
                const auto e = entry(0x10000000ull * (t + 1) + std::uint64_t(i) * 0x1000, 0x1000,
                                     VK_OBJECT_TYPE_BUFFER, std::uint64_t(t) << 32 | std::uint64_t(i));
                r.record(e, true, 1.0);
                if (i % 2) r.record(e, false, 2.0);
            }
        });
    pool.emplace_back([&r] {
        for (int i = 0; i < 200; ++i) (void)r.report(0x10000000ull + std::uint64_t(i) * 0x1000, 0x1000);
    });
    for (auto& thread : pool) thread.join();
    char expect[96];
    std::snprintf(expect, sizeof expect, "%d binds, %d unbinds, %d live", threads * per, threads * per / 2,
                  threads * per / 2);
    CHECK(has(r.report(0, 0x1000), expect));
    return 0;
}

static int test_messages() {
    // Through the messenger callback, as the driver delivers them.
    VkDeviceAddressBindingCallbackDataEXT binding{VK_STRUCTURE_TYPE_DEVICE_ADDRESS_BINDING_CALLBACK_DATA_EXT};
    binding.baseAddress = 0x800000;
    binding.size = 0x2000;
    binding.bindingType = VK_DEVICE_ADDRESS_BINDING_TYPE_BIND_EXT;
    binding.flags = VK_DEVICE_ADDRESS_BINDING_INTERNAL_OBJECT_BIT_EXT;
    VkDebugUtilsObjectNameInfoEXT object{VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT};
    object.objectType = VK_OBJECT_TYPE_DEVICE_MEMORY;
    object.objectHandle = 0x77;
    VkDebugUtilsMessengerCallbackDataEXT data{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CALLBACK_DATA_EXT, &binding};
    data.objectCount = 1;
    data.pObjects = &object;
    // A message of another type is not a binding.
    on_message(VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT, VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT, &data, nullptr);
    CHECK(has(state().record.report(0x800000, 0x1000), "0 binds"));
    on_message(VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT, VK_DEBUG_UTILS_MESSAGE_TYPE_DEVICE_ADDRESS_BINDING_BIT_EXT,
               &data, nullptr);
    const std::string text = state().record.report(0x801000, 0x1000);
    CHECK(has(text, "1 binds"));
    CHECK(has(text, "live, containing: memory (driver-internal) 0x800000..0x802000"));
    CHECK(has(text, "handle 0x77"));
    return 0;
}

static VKAPI_ATTR void VKAPI_CALL features(VkPhysicalDevice, VkPhysicalDeviceFeatures2* f) {
    for (auto* p = static_cast<VkBaseOutStructure*>(f->pNext); p; p = p->pNext)
        if (p->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ADDRESS_BINDING_REPORT_FEATURES_EXT)
            reinterpret_cast<VkPhysicalDeviceAddressBindingReportFeaturesEXT*>(p)->reportAddressBinding = VK_TRUE;
}

// A fake debug-utils instance: messengers are numbered as they are made and counted as they are destroyed.
static int created_messengers = 0, destroyed_messengers = 0;

static VKAPI_ATTR VkResult VKAPI_CALL fake_create(VkInstance, const VkDebugUtilsMessengerCreateInfoEXT*,
                                                   const VkAllocationCallbacks*, VkDebugUtilsMessengerEXT* out) {
    *out = reinterpret_cast<VkDebugUtilsMessengerEXT>(static_cast<std::uintptr_t>(++created_messengers));
    return VK_SUCCESS;
}

static VKAPI_ATTR void VKAPI_CALL fake_destroy(VkInstance, VkDebugUtilsMessengerEXT, const VkAllocationCallbacks*) {
    ++destroyed_messengers;
}

static VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL fake_get(VkInstance, const char* name) {
    if (std::strcmp(name, "vkCreateDebugUtilsMessengerEXT") == 0) return reinterpret_cast<PFN_vkVoidFunction>(&fake_create);
    if (std::strcmp(name, "vkDestroyDebugUtilsMessengerEXT") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(&fake_destroy);
    return nullptr;
}

static VkInstance fake_instance(std::uintptr_t n) { return reinterpret_cast<VkInstance>(n); }

static int test_enable() {
    VkExtensionProperties ext{};
    std::snprintf(ext.extensionName, sizeof ext.extensionName, "%s", VK_EXT_DEVICE_ADDRESS_BINDING_REPORT_EXTENSION_NAME);
    std::vector<const char*> extensions;
    VkDeviceCreateInfo device{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    // Not asked for: nothing changes.
    _putenv_s("LEONIDA_ADDRESS_BINDINGS", "");
    CHECK(!enable_device(VK_NULL_HANDLE, VK_NULL_HANDLE, &features, {&ext, 1}, extensions, device));
    CHECK(extensions.empty() && !device.pNext);
    _putenv_s("LEONIDA_ADDRESS_BINDINGS", "1");
    const VkInstance first = fake_instance(0x1000), second = fake_instance(0x2000);
    // Asked for, but no messenger records the reports: the device is left as it is, and a fault says so.
    CHECK(!enable_device(first, VK_NULL_HANDLE, &features, {&ext, 1}, extensions, device));
    CHECK(extensions.empty() && !device.pNext && !state().on);
    install_messenger(first, &fake_get);
    CHECK(created_messengers == 1 && state().messenger_instance == first);
    // A second instance in the same process: refused, so the first messenger is neither overwritten nor leaked.
    install_messenger(second, &fake_get);
    CHECK(created_messengers == 1 && state().messenger_instance == first);
    CHECK(!enable_device(second, VK_NULL_HANDLE, &features, {&ext, 1}, extensions, device));
    CHECK(extensions.empty() && !device.pNext && !state().on);
    uninstall_messenger(second);  // not its messenger: nothing happens
    CHECK(destroyed_messengers == 0 && state().messenger);
    // Asked for, the instance has the messenger and the device has the extension: it goes on the chain.
    CHECK(enable_device(first, VK_NULL_HANDLE, &features, {&ext, 1}, extensions, device));
    CHECK(extensions.size() == 1 && device.pNext == &state().feature && state().feature.reportAddressBinding);
    CHECK(state().on);
    // Asked for, and the device does not list it: refused.
    std::vector<const char*> none;
    VkDeviceCreateInfo other{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    CHECK(!enable_device(first, VK_NULL_HANDLE, &features, {}, none, other));
    CHECK(none.empty() && !other.pNext);
    // Teardown of the first instance destroys its messenger; the second instance can then have one.
    uninstall_messenger(first);
    CHECK(destroyed_messengers == 1 && !state().messenger && !state().on);
    install_messenger(second, &fake_get);
    CHECK(created_messengers == 2 && state().messenger_instance == second);
    uninstall_messenger(second);
    CHECK(destroyed_messengers == 2);
    return 0;
}

int main() {
    if (const int r = test_record()) return r;
    if (const int r = test_boundaries()) return r;
    if (const int r = test_unbind_matching_and_cap()) return r;
    if (const int r = test_threads()) return r;
    if (const int r = test_messages()) return r;
    if (const int r = test_enable()) return r;
    std::printf("address_binding: containing, neighbouring and unbound objects at a fault address\n");
    return 0;
}
