// GPU integration test for TitleHostRegions + VulkanTitleRegions + VulkanContext:
// registers a page-aligned host allocation, resolves it through VulkanContext::replay().buffer_at / host_at /
// resolve_title_bda, records a host-write barrier + vkCmdCopyBuffer into a readback buffer, verifies the GPU read
// the CPU-written pattern, and retires/unregisters without VK_ERROR_DEVICE_LOST.

#include "../src/vulkan_backend.h"
#include "vk_test_device.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

namespace {

void* alloc_host_pages(std::size_t bytes, std::size_t alignment) {
#ifdef _WIN32
    (void)alignment;
    return VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
    return std::aligned_alloc(alignment, bytes);
#endif
}

void free_host_pages(void* ptr) {
#ifdef _WIN32
    VirtualFree(ptr, 0, MEM_RELEASE);
#else
    std::free(ptr);
#endif
}

std::uint32_t find_host_visible_memory(const VkPhysicalDeviceMemoryProperties& mem, std::uint32_t type_bits) {
    constexpr VkMemoryPropertyFlags kWant =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    for (std::uint32_t i = 0; i < mem.memoryTypeCount; ++i) {
        if ((type_bits & (1u << i)) && (mem.memoryTypes[i].propertyFlags & kWant) == kWant) return i;
    }
    return mem.memoryTypeCount;
}

}  // namespace

int main() {
    vktest::Device dev;
    dev.request_buffer_device_address = true;
    const auto status = dev.initialize();
    if (status == vktest::DeviceStatus::unavailable) {
        std::printf("title_host_region_gpu_test: skipped (no Vulkan device)\n");
        return 0;
    }
    if (status != vktest::DeviceStatus::ready) {
        std::fprintf(stderr, "title_host_region_gpu_test: device init failed\n");
        return 1;
    }
    if (!dev.host_pointer_alignment || !dev.buffer_device_address) {
        std::printf("title_host_region_gpu_test: skipped (missing VK_EXT_external_memory_host or BDA)\n");
        return 0;
    }

    const std::size_t align = static_cast<std::size_t>(dev.host_pointer_alignment);
    const std::size_t region_size = std::max<std::size_t>(align * 4, 16384);
    void* host_mem = alloc_host_pages(region_size, align);
    if (!host_mem) {
        std::fprintf(stderr, "title_host_region_gpu_test: host allocation failed\n");
        return 1;
    }
    auto* words = static_cast<std::uint32_t*>(host_mem);
    for (std::size_t i = 0; i < region_size / sizeof(std::uint32_t); ++i) {
        words[i] = 0xA5000000u ^ static_cast<std::uint32_t>(i);
    }

    const auto get_mem_props =
        dev.instance_fn<PFN_vkGetPhysicalDeviceMemoryProperties>("vkGetPhysicalDeviceMemoryProperties");
    VkPhysicalDeviceMemoryProperties mem_props{};
    get_mem_props(dev.physical, &mem_props);

    const auto get_host_props =
        dev.fn<PFN_vkGetMemoryHostPointerPropertiesEXT>("vkGetMemoryHostPointerPropertiesEXT");
    if (!get_host_props) {
        std::fprintf(stderr, "title_host_region_gpu_test: vkGetMemoryHostPointerPropertiesEXT missing\n");
        free_host_pages(host_mem);
        return 1;
    }

    leonida::TitleHostRegions regions;
    const std::uint64_t host_va = reinterpret_cast<std::uint64_t>(host_mem);
    regions.register_range(host_va, region_size, 1);

    int failures = 0;
    {
        leonida::VulkanContext ctx(dev.device_proc, dev.device, true);
        ctx.configure_title_regions(get_host_props, dev.host_pointer_alignment, mem_props, &regions);

        const auto bda = ctx.resolve_title_bda(host_va + 256);
        if (!bda || *bda == 0) {
            std::fprintf(stderr, "title_host_region_gpu_test: resolve_title_bda failed\n");
            ++failures;
        }

        const std::uint8_t* resolved_host = nullptr;
        if (ctx.replay_functions().host_at(host_va + 128, 64, resolved_host) != leonida::HostReadStatus::ok ||
            resolved_host != static_cast<const std::uint8_t*>(host_mem) + 128) {
            std::fprintf(stderr, "title_host_region_gpu_test: replay_functions().host_at failed\n");
            ++failures;
        }

        VkBuffer src_buf = VK_NULL_HANDLE;
        VkDeviceSize src_off = 0;
        std::uint64_t lookup_size = 256;
        if (!ctx.replay_functions().buffer_at(host_va + 64, lookup_size, src_buf, src_off) ||
            src_buf == VK_NULL_HANDLE || src_off != 64 || lookup_size != 256) {
            std::fprintf(stderr, "title_host_region_gpu_test: replay_functions().buffer_at failed\n");
            ++failures;
        }

        // Create a host-visible readback buffer and copy from the imported host region on the GPU.
        const auto create_buf = dev.fn<PFN_vkCreateBuffer>("vkCreateBuffer");
        const auto destroy_buf = dev.fn<PFN_vkDestroyBuffer>("vkDestroyBuffer");
        const auto get_reqs = dev.fn<PFN_vkGetBufferMemoryRequirements>("vkGetBufferMemoryRequirements");
        const auto alloc_mem = dev.fn<PFN_vkAllocateMemory>("vkAllocateMemory");
        const auto free_mem = dev.fn<PFN_vkFreeMemory>("vkFreeMemory");
        const auto bind_buf = dev.fn<PFN_vkBindBufferMemory>("vkBindBufferMemory");
        const auto map_mem = dev.fn<PFN_vkMapMemory>("vkMapMemory");
        const auto unmap_mem = dev.fn<PFN_vkUnmapMemory>("vkUnmapMemory");
        const auto create_pool = dev.fn<PFN_vkCreateCommandPool>("vkCreateCommandPool");
        const auto destroy_pool = dev.fn<PFN_vkDestroyCommandPool>("vkDestroyCommandPool");
        const auto alloc_cb = dev.fn<PFN_vkAllocateCommandBuffers>("vkAllocateCommandBuffers");
        const auto begin_cb = dev.fn<PFN_vkBeginCommandBuffer>("vkBeginCommandBuffer");
        const auto end_cb = dev.fn<PFN_vkEndCommandBuffer>("vkEndCommandBuffer");
        const auto cmd_copy = dev.fn<PFN_vkCmdCopyBuffer>("vkCmdCopyBuffer");
        const auto queue_submit = dev.fn<PFN_vkQueueSubmit>("vkQueueSubmit");
        const auto queue_wait = dev.fn<PFN_vkQueueWaitIdle>("vkQueueWaitIdle");

        VkBuffer dst_buf = VK_NULL_HANDLE;
        VkDeviceMemory dst_mem = VK_NULL_HANDLE;
        VkBufferCreateInfo dst_info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        dst_info.size = 256;
        dst_info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        dst_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (create_buf(dev.device, &dst_info, nullptr, &dst_buf) != VK_SUCCESS) {
            std::fprintf(stderr, "title_host_region_gpu_test: create readback buffer failed\n");
            free_host_pages(host_mem);
            return 1;
        }
        VkMemoryRequirements reqs{};
        get_reqs(dev.device, dst_buf, &reqs);
        const std::uint32_t mem_idx = find_host_visible_memory(mem_props, reqs.memoryTypeBits);
        VkMemoryAllocateInfo alloc_info{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        alloc_info.allocationSize = reqs.size;
        alloc_info.memoryTypeIndex = mem_idx;
        if (alloc_mem(dev.device, &alloc_info, nullptr, &dst_mem) != VK_SUCCESS ||
            bind_buf(dev.device, dst_buf, dst_mem, 0) != VK_SUCCESS) {
            std::fprintf(stderr, "title_host_region_gpu_test: readback memory failed\n");
            destroy_buf(dev.device, dst_buf, nullptr);
            free_host_pages(host_mem);
            return 1;
        }

        VkCommandPool pool = VK_NULL_HANDLE;
        VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pool_info.queueFamilyIndex = dev.family;
        create_pool(dev.device, &pool_info, nullptr, &pool);
        VkCommandBuffer cb = VK_NULL_HANDLE;
        VkCommandBufferAllocateInfo cb_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cb_info.commandPool = pool;
        cb_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cb_info.commandBufferCount = 1;
        alloc_cb(dev.device, &cb_info, &cb);

        VkCommandBufferBeginInfo begin_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        begin_cb(cb, &begin_info);
        ctx.record_title_barrier(cb);
        const VkBufferCopy copy_region{src_off, 0, 256};
        cmd_copy(cb, src_buf, dst_buf, 1, &copy_region);
        end_cb(cb);

        VkSubmitInfo submit_info{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &cb;
        if (queue_submit(dev.queue, 1, &submit_info, VK_NULL_HANDLE) != VK_SUCCESS ||
            queue_wait(dev.queue) != VK_SUCCESS) {
            std::fprintf(stderr, "title_host_region_gpu_test: GPU submit/wait failed\n");
            ++failures;
        }

        void* mapped = nullptr;
        if (map_mem(dev.device, dst_mem, 0, 256, 0, &mapped) == VK_SUCCESS) {
            if (std::memcmp(mapped, static_cast<const std::uint8_t*>(host_mem) + 64, 256) != 0) {
                std::fprintf(stderr, "title_host_region_gpu_test: GPU readback mismatch\n");
                ++failures;
            }
            unmap_mem(dev.device, dst_mem);
        } else {
            std::fprintf(stderr, "title_host_region_gpu_test: map readback failed\n");
            ++failures;
        }

        destroy_pool(dev.device, pool, nullptr);
        destroy_buf(dev.device, dst_buf, nullptr);
        free_mem(dev.device, dst_mem, nullptr);

        // Unregister and verify retired cleanup works before freeing host pages.
        regions.unregister_range(host_va, region_size);
        ctx.sync_title_regions();
        if (ctx.resolve_title_bda(host_va + 64).has_value()) {
            std::fprintf(stderr, "title_host_region_gpu_test: region still resolved after unregister\n");
            ++failures;
        }
    }

    free_host_pages(host_mem);
    if (dev.validation_errors() > dev.known_validation_errors()) {
        std::fprintf(stderr, "title_host_region_gpu_test: Vulkan validation errors: %u\n", dev.validation_errors());
        ++failures;
    }
    if (failures == 0) {
        std::printf("title_host_region_gpu_test: ok (align=%zu, bda supported)\n", align);
    }
    return failures ? 1 : 0;
}
