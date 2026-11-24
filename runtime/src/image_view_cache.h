// The attachment image-view cache behind VulkanContext::view: one VkImageView per (image, aspect, base mip, base
// layer, layer count, format), made on first use and kept until the image is destroyed.
//
// The replay asks for a view for every colour and depth attachment of every draw and clear (vulkan_replay.h), so a
// lookup is on the per-draw path; it was a std::map keyed by a 6-tuple, a tree walk of tuple comparisons. Here it is
// one hash lookup. Destroying an image used to visit every cached view of every image to find its own; a per-image
// index visits only that image's views. Both run under VulkanContext::views_mutex_, which this class leaves to the
// caller. Pure standard library: `runtime/test/image_view_cache_test.cpp` checks it against the std::map it replaces.

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <unordered_map>
#include <utility>
#include <vector>

namespace leonida {

template <class Image, class View>
class ImageViewCache {
public:
    struct Key {
        Image image{};
        std::uint32_t aspect = 0;
        std::uint32_t mip = 0;
        std::uint32_t layer = 0;
        std::uint32_t layers = 0;
        std::uint32_t format = 0;

        bool operator==(const Key& o) const {
            return image == o.image && aspect == o.aspect && mip == o.mip && layer == o.layer && layers == o.layers &&
                   format == o.format;
        }
    };

    // The cached view, or a value-initialized View (VK_NULL_HANDLE) when there is none.
    View find(const Key& key) const {
        const auto it = views_.find(key);
        return it == views_.end() ? View{} : it->second;
    }

    // Records `view` for `key`; a key already cached keeps its view (the caller looked it up first) and returns false.
    bool insert(const Key& key, View view) {
        if (!views_.try_emplace(key, view).second) return false;
        by_image_[key.image].push_back(key);
        return true;
    }

    // Removes every view of `image`, passing each to `release` (the caller defers its destruction).
    template <class Release>
    void erase_image(Image image, Release&& release) {
        const auto it = by_image_.find(image);
        if (it == by_image_.end()) return;
        for (const Key& key : it->second) {
            const auto view = views_.find(key);
            if (view == views_.end()) continue;
            release(view->second);
            views_.erase(view);
        }
        by_image_.erase(it);
    }

    template <class Fn>
    void for_each(Fn&& fn) const {
        for (const auto& [key, view] : views_) fn(view);
    }

    std::size_t size() const { return views_.size(); }

    void clear() {
        views_.clear();
        by_image_.clear();
    }

private:
    static std::size_t mix(std::uint64_t h, std::uint64_t v) {
        h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        return static_cast<std::size_t>(h);
    }

    struct KeyHash {
        std::size_t operator()(const Key& k) const {
            std::uint64_t h = std::hash<Image>{}(k.image);
            h = mix(h, (std::uint64_t(k.aspect) << 32) | k.format);
            h = mix(h, (std::uint64_t(k.mip) << 32) | k.layer);
            return mix(h, k.layers);
        }
    };

    std::unordered_map<Key, View, KeyHash> views_;
    std::unordered_map<Image, std::vector<Key>> by_image_;  // each image's keys in views_, for erase_image
};

} // namespace leonida
