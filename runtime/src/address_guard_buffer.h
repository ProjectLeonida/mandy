#pragma once

// The address guard's page-table buffer (address_guard.h; docs/d3d12_vulkan.md, "Address guard"). Shaders translated
// with --address-guard read it through its device address on every D3D12 queue, so it is shared by all of the
// device's queue families, like the other buffers the device's queues all read.

#include "queue_families.h"

#include <cstdint>
#include <span>

namespace leonida::address_guard {

inline VkBufferCreateInfo table_buffer_info(VkDeviceSize bytes, std::span<const std::uint32_t> families) {
    VkBufferCreateInfo create{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    create.size = bytes;
    create.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    create.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    share_across_queue_families(create, families);
    return create;
}

}  // namespace leonida::address_guard
