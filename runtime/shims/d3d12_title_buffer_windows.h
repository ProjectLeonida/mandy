#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>
#include <vulkan/vulkan.h>

namespace rt::d3d12 {

// Storage for shader-addressed mirrors when the main arena has no usable offset.
// A window lasts for the lifetime of the device, so these blocks only grow.
class TitleBufferWindows {
    struct Block {
        VkBuffer buffer{};
        VkDeviceMemory memory{};
        VkDeviceAddress address{};
        VkDeviceSize capacity{}, used{};
        std::uint8_t* mapped{};
    };
    std::vector<Block> blocks_;
    VkDeviceSize reserved_{};
    bool logged_host_fallback_ = false;

public:
    VkDeviceSize reserved() const { return reserved_; }

    template<class Device>
    bool allocate(Device& d, VkDeviceSize bytes, VkDeviceSize& offset, std::uint8_t*& mapped,
                  bool device_local = false, VkDeviceSize cap = UINT64_MAX, bool* at_cap = nullptr) {
        const VkDeviceSize alignment = device_local ? 16 : 65536;
        constexpr VkDeviceSize block_size = 4ull << 20;
        auto take = [&](Block& b) {
            const auto start = (b.used + alignment - 1) & ~(alignment - 1);
            if (start > b.capacity || bytes > b.capacity - start) return false;
            mapped = b.mapped + start;
            offset = b.address - d.arena.address + start;
            b.used = start + bytes;
            return true;
        };
        if (!blocks_.empty() && take(blocks_.back())) return true;
        // A large view may start a new slab while smaller views still fit earlier tails.
        // Use those tails even after the budget can no longer accommodate another slab.
        if (device_local && blocks_.size() > 1)
            for (std::size_t i = blocks_.size() - 1; i != 0; --i)
                if (take(blocks_[i - 1])) return true;
        auto& vk = d.context->vk;
        if (!vk.vkGetBufferDeviceAddress) return false;
        Block b{};
        b.capacity = std::max(bytes, block_size);
        if (b.capacity > cap - reserved_) {
            if (at_cap) *at_cap = true;
            return false;
        }
        VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        info.size = b.capacity;
        info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        d.share(info);
        if (vk.vkCreateBuffer(d.info.device, &info, nullptr, &b.buffer) != VK_SUCCESS) return false;
        VkMemoryRequirements req{};
        vk.vkGetBufferMemoryRequirements(d.info.device, b.buffer, &req);
        if (req.size > cap - reserved_) {
            if (at_cap) *at_cap = true;
            vk.vkDestroyBuffer(d.info.device, b.buffer, nullptr);
            return false;
        }
        constexpr VkMemoryPropertyFlags host_access =
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        b.memory = d.allocate(req, host_access | (device_local ? VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT : 0),
                              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, nullptr, true);
        if (!b.memory && device_local) {
            // Without Resizable BAR the only device-local host-visible heap is the 256 MiB BAR window, and the
            // default GPU address arena can fill it. A view in system memory is slower to read but still correct.
            b.memory = d.allocate(req, host_access, 0, nullptr, true);
            if (b.memory && !logged_host_fallback_) {
                logged_host_fallback_ = true;
                std::fprintf(stderr, "[rt] d3d12: no device-local host-visible memory left for a %llu KiB block of "
                                     "buffer views, so it is in system memory\n",
                             static_cast<unsigned long long>(b.capacity >> 10));
            }
        }
        void* host{};
        if (b.memory && vk.vkBindBufferMemory(d.info.device, b.buffer, b.memory, 0) == VK_SUCCESS &&
            vk.vkMapMemory(d.info.device, b.memory, 0, VK_WHOLE_SIZE, 0, &host) == VK_SUCCESS) {
            VkBufferDeviceAddressInfo address{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
            address.buffer = b.buffer;
            b.address = vk.vkGetBufferDeviceAddress(d.info.device, &address);
            const auto biased = b.address - d.arena.address;
            // A positive 47-bit offset works with both generations of the translated V# decoder.
            if (b.address && biased < (1ull << 47) && b.capacity <= (1ull << 47) - biased) {
                b.mapped = static_cast<std::uint8_t*>(host);
                blocks_.push_back(b);
                reserved_ += req.size;
                if (blocks_.size() == 1 || !(blocks_.size() & (blocks_.size() - 1)))
                    std::fprintf(stderr, "[rt] d3d12: title buffer shadows outside the arena: %zu blocks, "
                                         "new block %llu KiB at shader VA 0x%llx\n", blocks_.size(),
                                 static_cast<unsigned long long>(b.capacity >> 10),
                                 static_cast<unsigned long long>(biased));
                return take(blocks_.back());
            }
            vk.vkUnmapMemory(d.info.device, b.memory);
        }
        vk.vkDestroyBuffer(d.info.device, b.buffer, nullptr);
        if (b.memory) vk.vkFreeMemory(d.info.device, b.memory, nullptr);
        return false;
    }

    bool lookup(VkDeviceAddress address, std::uint64_t& host, std::uint64_t& remaining) const {
        for (const auto& b : blocks_) {
            const auto delta = address - b.address;
            if (delta < b.used) {
                host = reinterpret_cast<std::uintptr_t>(b.mapped) + delta;
                remaining = b.used - delta;
                return true;
            }
        }
        return false;
    }

    // Device idle. No tracker writes into the blocks after this point.
    template<class Device>
    void release(Device& d) {
        for (auto& b : blocks_) {
            d.context->vk.vkUnmapMemory(d.info.device, b.memory);
            d.context->vk.vkDestroyBuffer(d.info.device, b.buffer, nullptr);
            d.context->vk.vkFreeMemory(d.info.device, b.memory, nullptr);
        }
        blocks_.clear();
        reserved_ = 0;
    }
};
}  // namespace rt::d3d12
