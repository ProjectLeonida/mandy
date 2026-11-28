#pragma once

#include <vulkan/vulkan.h>
#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>
#include <new>
#include <shared_mutex>
#include <span>
#include <unordered_map>
#include <vector>

namespace leonida {
class GraphicsShaderModules {
  public:
    struct Module {
        VkDevice device;
        PFN_vkDestroyShaderModule destroy;
        std::vector<std::uint32_t> words;
        VkShaderModule handle = VK_NULL_HANDLE;
        Module(VkDevice d, PFN_vkDestroyShaderModule fn, std::span<const std::uint32_t> input)
            : device(d), destroy(fn), words(input.begin(), input.end()) {}
        ~Module() {
            if (handle && destroy) destroy(device, handle, nullptr);
        }
    };
    using Reference = std::shared_ptr<const Module>;
    Reference acquire(VkDevice device, PFN_vkCreateShaderModule create, PFN_vkDestroyShaderModule destroy,
                      std::span<const std::uint32_t> words) {
        if (words.empty() || !create || !destroy) return {};
        std::uint64_t hash = 14695981039346656037ull;
        for (std::uint32_t word : words)
            hash = (hash ^ word) * 1099511628211ull;
        {
            std::shared_lock lock(mutex_);
            if (auto found = find_locked(device, hash, words)) return found;
        }
        auto result = std::make_shared<Module>(device, destroy, words);
        VkShaderModuleCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        info.pCode = words.data();
        info.codeSize = words.size_bytes();
        if (create(device, &info, nullptr, &result->handle) != VK_SUCCESS || result->handle == VK_NULL_HANDLE) {
            result->handle = VK_NULL_HANDLE;
            return {};
        }
        Reference existing;
        {
            std::unique_lock lock(mutex_);
            if (auto found = find_locked(device, hash, words)) {
                existing = std::move(found);
            } else {
                prune_bucket_locked(hash);
                if ((++created_ % 1024) == 0)
                    std::erase_if(modules_, [](const auto &entry) { return entry.second.expired(); });
                try {
                    modules_.emplace(hash, result);
                } catch (const std::bad_alloc &) {
                }
            }
        }
        // Destroy any losing duplicate outside mutex_ so vkDestroyShaderModule never runs under the cache lock.
        if (existing) return existing;
        return result;
    }

  private:
    Reference find_locked(VkDevice device, std::uint64_t hash, std::span<const std::uint32_t> words) const {
        const auto [first, last] = modules_.equal_range(hash);
        for (auto i = first; i != last; ++i)
            if (auto live = i->second.lock();
                live && live->device == device &&
                std::equal(words.begin(), words.end(), live->words.begin(), live->words.end()))
                return live;
        return {};
    }
    void prune_bucket_locked(std::uint64_t hash) {
        const auto [first, last] = modules_.equal_range(hash);
        for (auto i = first; i != last;) {
            if (i->second.expired())
                i = modules_.erase(i);
            else
                ++i;
        }
    }
    mutable std::shared_mutex mutex_;
    std::unordered_multimap<std::uint64_t, std::weak_ptr<const Module>> modules_;
    std::uint64_t created_ = 0;
};
} // namespace leonida
