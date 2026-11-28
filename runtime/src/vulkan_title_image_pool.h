#pragma once

#include <vulkan/vulkan.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <vector>

namespace leonida {

// Only optimal-tiled, non-sparse images go in a page. A slot is returned after the owning
// image is destroyed by the device's normal GPU retirement queue.
struct TitleImagePage {
    VkDevice device{};
    VkDeviceMemory memory{};
    PFN_vkFreeMemory free_memory{};
    std::uint32_t type = 0, capacity = 0, allocated = 0;
    std::atomic<std::uint32_t> free_slots{0};
    std::mutex mutex;
    std::vector<std::uint32_t> returned;

    ~TitleImagePage() {
        if (memory && free_memory) free_memory(device, memory, nullptr);
    }

    bool has_free() const { return free_slots.load(std::memory_order_acquire) != 0; }

    bool take(std::uint32_t& slot) {
        std::lock_guard lock(mutex);
        if (!returned.empty()) {
            slot = returned.back();
            returned.pop_back();
        } else if (allocated < capacity) {
            slot = allocated++;
        } else {
            return false;
        }
        free_slots.fetch_sub(1, std::memory_order_relaxed);
        return true;
    }

    void release(std::uint32_t slot) {
        std::lock_guard lock(mutex);
        returned.push_back(slot);  // the page reserves capacity at creation
        free_slots.fetch_add(1, std::memory_order_release);
    }
};

struct PooledTitleImage {
    PooledTitleImage() = default;
    PooledTitleImage(const PooledTitleImage&) = delete;
    PooledTitleImage& operator=(const PooledTitleImage&) = delete;

    std::shared_ptr<TitleImagePage> page;
    std::uint32_t slot = 0;
    VkImage image{};
    PFN_vkDestroyImage destroy_image{};

    ~PooledTitleImage() {
        if (image && page && destroy_image) destroy_image(page->device, image, nullptr);
        if (page) page->release(slot);
    }
};

class TitleImagePool {
public:
    // The caller has already excluded dedicated-only images. All allocations in one page have
    // identical stride, compatible memory types and non-overlapping, individually aligned ranges.
    template<class Allocate>
    std::unique_ptr<PooledTitleImage> bind(VkDevice device, VkImage image, const VkMemoryRequirements& req,
                                           PFN_vkBindImageMemory bind_image, PFN_vkDestroyImage destroy_image,
                                           PFN_vkFreeMemory free_memory, Allocate allocate) {
        constexpr VkDeviceSize largest_slot = VkDeviceSize{2} << 20;
        if (!image || !bind_image || !destroy_image || !free_memory) return {};
        if (req.alignment && !std::has_single_bit(req.alignment)) return {};
        const VkDeviceSize needed = std::max(req.size, req.alignment);
        if (!needed || needed > largest_slot || !req.memoryTypeBits) return {};
        // Eight size classes per power of two keep padding below one eighth (besides required alignment).
        const VkDeviceSize quantum = std::max(req.alignment, std::max(VkDeviceSize{1}, std::bit_floor(needed) / 8));
        const VkDeviceSize stride = (needed + quantum - 1) & ~(quantum - 1);
        const VkDeviceSize page_size = std::min(VkDeviceSize{8} << 20, 256 * stride);
        auto allocation = std::make_unique<PooledTitleImage>();
        {
            std::lock_guard lock(mutex_);
            if (const auto it = classes_.find(stride); it != classes_.end()) {
                auto& pages = it->second;
                std::erase_if(pages, [](const auto& page) { return page.expired(); });
                for (const auto& weak : pages) {
                    auto page = weak.lock();
                    if (page && page->type < 32u && (req.memoryTypeBits & (1u << page->type)) &&
                        page->has_free() && page->take(allocation->slot)) {
                        allocation->page = std::move(page);
                        break;
                    }
                }
                if (pages.empty()) classes_.erase(it);
            }
        }
        bool created_page = false;
        if (!allocation->page) {
            auto page = std::make_shared<TitleImagePage>();
            page->device = device;
            page->free_memory = free_memory;
            page->capacity = static_cast<std::uint32_t>(page_size / stride);
            if (!page->capacity) return {};
            page->free_slots.store(page->capacity, std::memory_order_relaxed);
            page->returned.reserve(page->capacity);
            // Allocate outside mutex_ so vkAllocateMemory never stalls concurrent binds on other classes/pages.
            page->memory = allocate(page_size, page->type);
            if (!page->memory || page->type >= 32u || !(req.memoryTypeBits & (1u << page->type))) return {};
            if (!page->take(allocation->slot)) return {};
            allocation->page = std::move(page);
            created_page = true;
        }
        if (bind_image(device, image, allocation->page->memory, stride * allocation->slot) != VK_SUCCESS) return {};
        allocation->destroy_image = destroy_image;
        allocation->image = image;
        if (created_page) {
            // Publish only after bind succeeded; if weak_ptr storage hits OOM, keep the already-bound image intact.
            try {
                std::lock_guard lock(mutex_);
                auto& pages = classes_[stride];
                std::erase_if(pages, [](const auto& page) { return page.expired(); });
                pages.push_back(allocation->page);
            } catch (const std::bad_alloc&) {
            }
        }
        return allocation;
    }

private:
    std::mutex mutex_;
    std::map<VkDeviceSize, std::vector<std::weak_ptr<TitleImagePage>>> classes_;
};

}  // namespace leonida
