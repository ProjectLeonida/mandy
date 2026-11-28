// leonida::address_guard::table_buffer_info (src/address_guard_buffer.h): the address guard's page table is read by
// guarded shaders on the direct and the async-compute queue, so its buffer is shared by every distinct queue family
// of the device, and stays exclusive on a device with one family. No GPU, no device.
#include "../src/address_guard_buffer.h"

#include <cstdint>
#include <cstdio>
#include <vector>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

int main() {
    namespace ag = leonida::address_guard;
    const std::vector<std::uint32_t> two{0, 2};
    const VkBufferCreateInfo shared = ag::table_buffer_info(0x2000, two);
    EXPECT(shared.sType == VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO && shared.size == 0x2000);
    EXPECT(shared.usage == (VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT));
    EXPECT(shared.sharingMode == VK_SHARING_MODE_CONCURRENT);
    EXPECT(shared.queueFamilyIndexCount == 2 && shared.pQueueFamilyIndices == two.data());

    const std::vector<std::uint32_t> one{0};
    const VkBufferCreateInfo single = ag::table_buffer_info(0x2000, one);
    EXPECT(single.sharingMode == VK_SHARING_MODE_EXCLUSIVE && single.queueFamilyIndexCount == 0 &&
           single.pQueueFamilyIndices == nullptr);
    std::puts("address_guard_buffer_test: ok");
    return 0;
}
