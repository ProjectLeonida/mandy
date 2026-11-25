// src/address_binding.h: the record of VK_EXT_device_address_binding_report reports and what it says about a
// device-fault address. Synthetic reports only: no Vulkan device, no GPU.
#include "../src/address_binding.h"

#include <cstdio>
#include <cstdlib>
#include <string>

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

static int test_enable() {
    VkExtensionProperties ext{};
    std::snprintf(ext.extensionName, sizeof ext.extensionName, "%s", VK_EXT_DEVICE_ADDRESS_BINDING_REPORT_EXTENSION_NAME);
    std::vector<const char*> extensions;
    VkDeviceCreateInfo device{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    // Not asked for: nothing changes.
    _putenv_s("LEONIDA_ADDRESS_BINDINGS", "");
    CHECK(!enable_device(VK_NULL_HANDLE, &features, {&ext, 1}, extensions, device));
    CHECK(extensions.empty() && !device.pNext);
    // Asked for, and the device has it: the extension and the feature go on the chain.
    _putenv_s("LEONIDA_ADDRESS_BINDINGS", "1");
    CHECK(enable_device(VK_NULL_HANDLE, &features, {&ext, 1}, extensions, device));
    CHECK(extensions.size() == 1 && device.pNext == &state().feature && state().feature.reportAddressBinding);
    // Asked for, and the device does not list it: refused.
    std::vector<const char*> none;
    VkDeviceCreateInfo other{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    CHECK(!enable_device(VK_NULL_HANDLE, &features, {}, none, other));
    CHECK(none.empty() && !other.pNext);
    return 0;
}

int main() {
    if (const int r = test_record()) return r;
    if (const int r = test_messages()) return r;
    if (const int r = test_enable()) return r;
    std::printf("address_binding: containing, neighbouring and unbound objects at a fault address\n");
    return 0;
}
