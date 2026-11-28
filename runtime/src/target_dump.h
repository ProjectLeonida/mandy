#pragma once

#include "present_dump.h"
#include "vulkan_memory.h"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <atomic>
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace leonida::target_dump {

struct Config {
    std::filesystem::path dir;
    std::uint64_t frame = 0;
    bool valid = false;
};

template<typename CharT>
inline Config parse_config_chars(std::basic_string_view<CharT> raw) {
    const auto comma = raw.rfind(static_cast<CharT>(','));
    if (comma == raw.npos || comma == 0 || comma + 1 >= raw.size()) return {};
    std::string number;
    number.reserve(raw.size() - comma - 1);
    for (std::size_t i = comma + 1; i < raw.size(); ++i) {
        const auto ch = raw[i];
        if (ch < static_cast<CharT>('0') || ch > static_cast<CharT>('9')) return {};
        number.push_back(static_cast<char>(ch));
    }
    std::uint64_t frame = 0;
    const auto end = number.data() + number.size();
    const auto parsed = std::from_chars(number.data(), end, frame);
    if (parsed.ec != std::errc{} || parsed.ptr != end || frame == 0) return {};
    return {std::filesystem::path(raw.substr(0, comma)), frame, true};
}

inline Config parse_config_utf8(std::string_view raw) {
    return parse_config_chars<char>(raw);
}

inline unsigned raw_texel_size(VkFormat format, VkImageAspectFlags aspect) {
    if (aspect == VK_IMAGE_ASPECT_STENCIL_BIT) return 1;
    if (aspect == VK_IMAGE_ASPECT_DEPTH_BIT) {
        switch (format) {
        case VK_FORMAT_D16_UNORM:
        case VK_FORMAT_D16_UNORM_S8_UINT:
            return 2;
        case VK_FORMAT_D32_SFLOAT:
        case VK_FORMAT_D32_SFLOAT_S8_UINT:
        case VK_FORMAT_X8_D24_UNORM_PACK32:
        case VK_FORMAT_D24_UNORM_S8_UINT:
            return 4;
        default:
            return 0;
        }
    }
    if (aspect != VK_IMAGE_ASPECT_COLOR_BIT) return 0;
    if (const unsigned png_size = present_dump::texel_size(format); png_size != 0) return png_size;
    switch (format) {
    case VK_FORMAT_R8_UNORM:
    case VK_FORMAT_R8_SNORM:
    case VK_FORMAT_R8_UINT:
    case VK_FORMAT_R8_SINT:
    case VK_FORMAT_R8_SRGB:
#ifdef VK_KHR_MAINTENANCE_5_EXTENSION_NAME
    case VK_FORMAT_A8_UNORM_KHR:
#endif
        return 1;
    case VK_FORMAT_R8G8_UNORM:
    case VK_FORMAT_R8G8_SNORM:
    case VK_FORMAT_R8G8_UINT:
    case VK_FORMAT_R8G8_SINT:
    case VK_FORMAT_R16_UNORM:
    case VK_FORMAT_R16_SNORM:
    case VK_FORMAT_R16_UINT:
    case VK_FORMAT_R16_SINT:
    case VK_FORMAT_R16_SFLOAT:
    case VK_FORMAT_R5G6B5_UNORM_PACK16:
    case VK_FORMAT_B5G6R5_UNORM_PACK16:
    case VK_FORMAT_A1R5G5B5_UNORM_PACK16:
    case VK_FORMAT_B5G5R5A1_UNORM_PACK16:
    case VK_FORMAT_R4G4B4A4_UNORM_PACK16:
    case VK_FORMAT_B4G4R4A4_UNORM_PACK16:
        return 2;
    case VK_FORMAT_R8G8B8A8_SNORM:
    case VK_FORMAT_R8G8B8A8_UINT:
    case VK_FORMAT_R8G8B8A8_SINT:
    case VK_FORMAT_B8G8R8A8_SNORM:
    case VK_FORMAT_B8G8R8A8_UINT:
    case VK_FORMAT_B8G8R8A8_SINT:
    case VK_FORMAT_A2B10G10R10_UINT_PACK32:
    case VK_FORMAT_B10G11R11_UFLOAT_PACK32:
    case VK_FORMAT_E5B9G9R9_UFLOAT_PACK32:
    case VK_FORMAT_R16G16_UNORM:
    case VK_FORMAT_R16G16_SNORM:
    case VK_FORMAT_R16G16_UINT:
    case VK_FORMAT_R16G16_SINT:
    case VK_FORMAT_R16G16_SFLOAT:
    case VK_FORMAT_R32_UINT:
    case VK_FORMAT_R32_SINT:
    case VK_FORMAT_R32_SFLOAT:
        return 4;
    case VK_FORMAT_R16G16B16A16_UNORM:
    case VK_FORMAT_R16G16B16A16_SNORM:
    case VK_FORMAT_R16G16B16A16_UINT:
    case VK_FORMAT_R16G16B16A16_SINT:
    case VK_FORMAT_R32G32_UINT:
    case VK_FORMAT_R32G32_SINT:
    case VK_FORMAT_R32G32_SFLOAT:
        return 8;
    case VK_FORMAT_R32G32B32A32_UINT:
    case VK_FORMAT_R32G32B32A32_SINT:
        return 16;
    default:
        return 0;
    }
}

inline bool uses_png(VkFormat format, VkImageAspectFlags aspect) {
    return aspect == VK_IMAGE_ASPECT_COLOR_BIT && present_dump::texel_size(format) != 0;
}

inline std::string format_name(VkFormat format) {
    switch (format) {
    case VK_FORMAT_R8_UNORM: return "VK_FORMAT_R8_UNORM";
    case VK_FORMAT_R8_SNORM: return "VK_FORMAT_R8_SNORM";
    case VK_FORMAT_R8_UINT: return "VK_FORMAT_R8_UINT";
    case VK_FORMAT_R8_SINT: return "VK_FORMAT_R8_SINT";
    case VK_FORMAT_R8_SRGB: return "VK_FORMAT_R8_SRGB";
#ifdef VK_KHR_MAINTENANCE_5_EXTENSION_NAME
    case VK_FORMAT_A8_UNORM_KHR: return "VK_FORMAT_A8_UNORM_KHR";
#endif
    case VK_FORMAT_R8G8_UNORM: return "VK_FORMAT_R8G8_UNORM";
    case VK_FORMAT_R8G8_SNORM: return "VK_FORMAT_R8G8_SNORM";
    case VK_FORMAT_R8G8_UINT: return "VK_FORMAT_R8G8_UINT";
    case VK_FORMAT_R8G8_SINT: return "VK_FORMAT_R8G8_SINT";
    case VK_FORMAT_R8G8B8A8_UNORM: return "VK_FORMAT_R8G8B8A8_UNORM";
    case VK_FORMAT_R8G8B8A8_SNORM: return "VK_FORMAT_R8G8B8A8_SNORM";
    case VK_FORMAT_R8G8B8A8_UINT: return "VK_FORMAT_R8G8B8A8_UINT";
    case VK_FORMAT_R8G8B8A8_SINT: return "VK_FORMAT_R8G8B8A8_SINT";
    case VK_FORMAT_R8G8B8A8_SRGB: return "VK_FORMAT_R8G8B8A8_SRGB";
    case VK_FORMAT_B8G8R8A8_UNORM: return "VK_FORMAT_B8G8R8A8_UNORM";
    case VK_FORMAT_B8G8R8A8_SNORM: return "VK_FORMAT_B8G8R8A8_SNORM";
    case VK_FORMAT_B8G8R8A8_UINT: return "VK_FORMAT_B8G8R8A8_UINT";
    case VK_FORMAT_B8G8R8A8_SINT: return "VK_FORMAT_B8G8R8A8_SINT";
    case VK_FORMAT_B8G8R8A8_SRGB: return "VK_FORMAT_B8G8R8A8_SRGB";
    case VK_FORMAT_A2B10G10R10_UNORM_PACK32: return "VK_FORMAT_A2B10G10R10_UNORM_PACK32";
    case VK_FORMAT_A2B10G10R10_UINT_PACK32: return "VK_FORMAT_A2B10G10R10_UINT_PACK32";
    case VK_FORMAT_A2R10G10B10_UNORM_PACK32: return "VK_FORMAT_A2R10G10B10_UNORM_PACK32";
    case VK_FORMAT_B10G11R11_UFLOAT_PACK32: return "VK_FORMAT_B10G11R11_UFLOAT_PACK32";
    case VK_FORMAT_E5B9G9R9_UFLOAT_PACK32: return "VK_FORMAT_E5B9G9R9_UFLOAT_PACK32";
    case VK_FORMAT_R16_UNORM: return "VK_FORMAT_R16_UNORM";
    case VK_FORMAT_R16_SNORM: return "VK_FORMAT_R16_SNORM";
    case VK_FORMAT_R16_UINT: return "VK_FORMAT_R16_UINT";
    case VK_FORMAT_R16_SINT: return "VK_FORMAT_R16_SINT";
    case VK_FORMAT_R16_SFLOAT: return "VK_FORMAT_R16_SFLOAT";
    case VK_FORMAT_R16G16_UNORM: return "VK_FORMAT_R16G16_UNORM";
    case VK_FORMAT_R16G16_SNORM: return "VK_FORMAT_R16G16_SNORM";
    case VK_FORMAT_R16G16_UINT: return "VK_FORMAT_R16G16_UINT";
    case VK_FORMAT_R16G16_SINT: return "VK_FORMAT_R16G16_SINT";
    case VK_FORMAT_R16G16_SFLOAT: return "VK_FORMAT_R16G16_SFLOAT";
    case VK_FORMAT_R16G16B16A16_UNORM: return "VK_FORMAT_R16G16B16A16_UNORM";
    case VK_FORMAT_R16G16B16A16_SNORM: return "VK_FORMAT_R16G16B16A16_SNORM";
    case VK_FORMAT_R16G16B16A16_UINT: return "VK_FORMAT_R16G16B16A16_UINT";
    case VK_FORMAT_R16G16B16A16_SINT: return "VK_FORMAT_R16G16B16A16_SINT";
    case VK_FORMAT_R16G16B16A16_SFLOAT: return "VK_FORMAT_R16G16B16A16_SFLOAT";
    case VK_FORMAT_R32_UINT: return "VK_FORMAT_R32_UINT";
    case VK_FORMAT_R32_SINT: return "VK_FORMAT_R32_SINT";
    case VK_FORMAT_R32_SFLOAT: return "VK_FORMAT_R32_SFLOAT";
    case VK_FORMAT_R32G32_UINT: return "VK_FORMAT_R32G32_UINT";
    case VK_FORMAT_R32G32_SINT: return "VK_FORMAT_R32G32_SINT";
    case VK_FORMAT_R32G32_SFLOAT: return "VK_FORMAT_R32G32_SFLOAT";
    case VK_FORMAT_R32G32B32A32_UINT: return "VK_FORMAT_R32G32B32A32_UINT";
    case VK_FORMAT_R32G32B32A32_SINT: return "VK_FORMAT_R32G32B32A32_SINT";
    case VK_FORMAT_R32G32B32A32_SFLOAT: return "VK_FORMAT_R32G32B32A32_SFLOAT";
    case VK_FORMAT_R5G6B5_UNORM_PACK16: return "VK_FORMAT_R5G6B5_UNORM_PACK16";
    case VK_FORMAT_B5G6R5_UNORM_PACK16: return "VK_FORMAT_B5G6R5_UNORM_PACK16";
    case VK_FORMAT_A1R5G5B5_UNORM_PACK16: return "VK_FORMAT_A1R5G5B5_UNORM_PACK16";
    case VK_FORMAT_B5G5R5A1_UNORM_PACK16: return "VK_FORMAT_B5G5R5A1_UNORM_PACK16";
    case VK_FORMAT_R4G4B4A4_UNORM_PACK16: return "VK_FORMAT_R4G4B4A4_UNORM_PACK16";
    case VK_FORMAT_B4G4R4A4_UNORM_PACK16: return "VK_FORMAT_B4G4R4A4_UNORM_PACK16";
    case VK_FORMAT_D16_UNORM: return "VK_FORMAT_D16_UNORM";
    case VK_FORMAT_D16_UNORM_S8_UINT: return "VK_FORMAT_D16_UNORM_S8_UINT";
    case VK_FORMAT_D32_SFLOAT: return "VK_FORMAT_D32_SFLOAT";
    case VK_FORMAT_D32_SFLOAT_S8_UINT: return "VK_FORMAT_D32_SFLOAT_S8_UINT";
    case VK_FORMAT_X8_D24_UNORM_PACK32: return "VK_FORMAT_X8_D24_UNORM_PACK32";
    case VK_FORMAT_D24_UNORM_S8_UINT: return "VK_FORMAT_D24_UNORM_S8_UINT";
    default: return "VK_FORMAT_" + std::to_string(static_cast<std::int64_t>(format));
    }
}

struct TargetImage {
    VkImage image = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkFormat image_format = VK_FORMAT_UNDEFINED;
    VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    VkImageAspectFlags layout_aspects = VK_IMAGE_ASPECT_COLOR_BIT;
    VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
    std::uint32_t width = 1;
    std::uint32_t height = 1;
    std::uint32_t mip = 0;
    std::uint32_t layer = 0;
    std::uint32_t samples = 1;
    bool volume = false;
    std::string slot;
};

struct UavImage {
    VkImage image = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkImageAspectFlags aspects = VK_IMAGE_ASPECT_COLOR_BIT;
    std::uint32_t width = 1;
    std::uint32_t height = 1;
    std::uint32_t mip_levels = 1;
    std::uint32_t array_layers = 1;
    std::uint32_t mip = 0;
    std::uint32_t layer = 0;
    std::uint32_t samples = 1;
    bool volume = false;
};

struct IndexTarget {
    std::string slot;
    VkFormat format = VK_FORMAT_UNDEFINED;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t mip = 0;
    std::uint32_t layer = 0;
    unsigned bytes_per_texel = 0;
    bool png = false;
    std::string file;
};

struct IndexPass {
    std::uint32_t pass = 0;
    std::string kind;
    std::vector<IndexTarget> targets;
};

inline std::string serialize_index_json(std::uint64_t frame, std::span<const IndexPass> passes) {
    std::string out;
    out.reserve(128 + passes.size() * 256);
    out += "{\n  \"frame\": ";
    out += std::to_string(frame);
    out += ",\n  \"passes\": [";
    for (std::size_t i = 0; i < passes.size(); ++i) {
        const auto& pass = passes[i];
        out += i ? ",\n    {\n" : "\n    {\n";
        out += "      \"pass\": ";
        out += std::to_string(pass.pass);
        out += ",\n      \"kind\": \"";
        out += pass.kind;
        out += "\",\n      \"targets\": [";
        for (std::size_t j = 0; j < pass.targets.size(); ++j) {
            const auto& t = pass.targets[j];
            out += j ? ",\n        {" : "\n        {";
            out += "\n          \"slot\": \"";
            out += t.slot;
            out += "\",\n          \"format\": \"";
            out += format_name(t.format);
            out += "\",\n          \"vk_format\": ";
            out += std::to_string(static_cast<std::int64_t>(t.format));
            out += ",\n          \"width\": ";
            out += std::to_string(t.width);
            out += ",\n          \"height\": ";
            out += std::to_string(t.height);
            out += ",\n          \"mip\": ";
            out += std::to_string(t.mip);
            out += ",\n          \"layer\": ";
            out += std::to_string(t.layer);
            out += ",\n          \"bytes_per_texel\": ";
            out += std::to_string(t.bytes_per_texel);
            out += ",\n          \"encoding\": \"";
            out += t.png ? "png" : "raw";
            out += "\",\n          \"file\": \"";
            out += t.file;
            out += "\"\n        }";
        }
        out += pass.targets.empty() ? "]\n    }" : "\n      ]\n    }";
    }
    out += passes.empty() ? "]\n}\n" : "\n  ]\n}\n";
    return out;
}

class TargetDump {
public:
    TargetDump() {
#ifdef _WIN32
        wchar_t* env = nullptr;
        std::size_t length = 0;
        if (_wdupenv_s(&env, &length, L"LEONIDA_TARGET_DUMP") != 0) {
            report_invalid_env();
            return;
        }
        const std::unique_ptr<wchar_t, decltype(&std::free)> owned(env, &std::free);
        if (!env) return;
        const Config cfg = parse_config_chars<wchar_t>(env);
#else
        const char* env = std::getenv("LEONIDA_TARGET_DUMP");
        if (!env) return;
        const Config cfg = parse_config_chars<char>(env);
#endif
        if (!cfg.valid) {
            report_invalid_env();
            return;
        }
        try {
            dir_ = std::filesystem::absolute(cfg.dir);
            std::filesystem::create_directories(dir_);
            configured_frame_ = cfg.frame;
            target_frame_.store(cfg.frame, std::memory_order_relaxed);
        } catch (const std::exception&) {
            report_invalid_env();
        }
    }

    ~TargetDump() { discard_pending(); }

    TargetDump(const TargetDump&) = delete;
    TargetDump& operator=(const TargetDump&) = delete;

    void initialize(PFN_vkGetDeviceProcAddr get, VkDevice device, bool conditional_rendering = false) {
        if (!enabled() || !get || !device) return;
        device_ = device;
        const auto load = [&](auto& fn, const char* name) {
            fn = reinterpret_cast<std::remove_reference_t<decltype(fn)>>(get(device, name));
        };
        load(fns_.create_buffer, "vkCreateBuffer");
        load(fns_.destroy_buffer, "vkDestroyBuffer");
        load(fns_.buffer_requirements, "vkGetBufferMemoryRequirements");
        load(fns_.allocate_memory, "vkAllocateMemory");
        load(fns_.free_memory, "vkFreeMemory");
        load(fns_.bind_buffer, "vkBindBufferMemory");
        load(fns_.map_memory, "vkMapMemory");
        load(fns_.unmap_memory, "vkUnmapMemory");
        load(fns_.pipeline_barrier, "vkCmdPipelineBarrier");
        load(fns_.copy_image_to_buffer, "vkCmdCopyImageToBuffer");
        if (conditional_rendering) {
            load(fns_.begin_conditional_rendering, "vkCmdBeginConditionalRenderingEXT");
            load(fns_.end_conditional_rendering, "vkCmdEndConditionalRenderingEXT");
        }
    }

    void set_memory_properties(const VkPhysicalDeviceMemoryProperties& memory,
                               std::uint32_t max_memory_allocations = 0) {
        memory_ = memory;
        if (max_memory_allocations > 0) {
            const std::uint32_t safe_cap = std::max(1u, max_memory_allocations / 8u);
            max_staging_allocations_ = std::min(max_staging_allocations_, safe_cap);
        }
    }

    void set_staging_limits(VkDeviceSize max_bytes, std::uint32_t max_allocations) {
        std::lock_guard lock(mutex_);
        max_staging_bytes_ = max_bytes;
        max_staging_allocations_ = max_allocations;
    }

    VkDeviceSize outstanding_staging_bytes() const noexcept {
        std::lock_guard lock(mutex_);
        return outstanding_staging_bytes_;
    }

    std::uint32_t outstanding_staging_allocations() const noexcept {
        std::lock_guard lock(mutex_);
        return outstanding_staging_allocations_;
    }

    bool enabled() const noexcept { return target_frame_.load(std::memory_order_relaxed) != 0; }
    std::uint64_t target_frame() const noexcept { return target_frame_.load(std::memory_order_relaxed); }
    bool has_pending() const noexcept { return has_pending_.load(std::memory_order_acquire); }
    bool submission_captured() const noexcept { return pending_captured_; }

    bool active_for_frame(std::uint64_t frame) {
        const std::uint64_t target = target_frame_.load(std::memory_order_relaxed);
        if (!target) return false;
        if (frame <= target) return true;
        if (!has_pending_.load(std::memory_order_acquire)) finalize_if_complete(frame);
        return false;
    }

    void begin_submission(std::uint64_t frame, VkQueue queue = VK_NULL_HANDLE) {
        pending_passes_.clear();
        pending_captured_ = false;
        current_queue_ = queue;
        {
            std::lock_guard lock(mutex_);
            saved_uavs_ = active_uavs_;
        }
        const std::uint64_t target = target_frame_.load(std::memory_order_relaxed);
        capturing_ = target != 0 && frame == target && device_ != VK_NULL_HANDLE && fns_.complete() &&
                     memory_.memoryTypeCount != 0;
        if (capturing_) saw_target_frame_.store(true, std::memory_order_relaxed);
        next_pass_ = committed_passes_;
    }

    void note_uav(std::uint64_t id, const UavImage& image, std::uint32_t mip_count = 1,
                  std::uint32_t layer_count = 1) {
        // Depth/stencil UAVs write to separate storage-plane child images rather than the parent depth image; skip them
        // so post-dispatch capture never dumps stale pre-dispatch parent depth.
        if (!id || !image.image || image.aspects != VK_IMAGE_ASPECT_COLOR_BIT) return;
        if (image.mip >= image.mip_levels || image.layer >= image.array_layers) return;
        const auto mip_end = image.mip + std::min(mip_count, image.mip_levels - image.mip);
        const auto layer_end = image.layer + std::min(layer_count, image.array_layers - image.layer);
        std::lock_guard lock(mutex_);
        for (auto mip = image.mip; mip < mip_end; ++mip) {
            for (auto layer = image.layer; layer < layer_end; ++layer) {
                UavImage subresource = image;
                subresource.mip = mip;
                subresource.layer = layer;
                active_uavs_.insert_or_assign(UavKey{id, mip, layer}, TrackedUav{current_queue_, subresource});
            }
        }
    }

    void clear_uav(std::uint64_t id, std::uint32_t base_mip, std::uint32_t level_count, std::uint32_t base_layer,
                   std::uint32_t layer_count) {
        std::lock_guard lock(mutex_);
        auto it = active_uavs_.lower_bound(UavKey{id, base_mip, 0});
        while (it != active_uavs_.end() && std::get<0>(it->first) == id) {
            const UavImage& entry = it->second.image;
            if (entry.mip - base_mip >= level_count) break;
            if (entry.layer >= base_layer && entry.layer - base_layer < layer_count)
                it = active_uavs_.erase(it);
            else
                ++it;
        }
    }

    void forget_resource(std::uint64_t id) {
        if (!enabled() && !has_pending()) return;
        std::lock_guard lock(mutex_);
        erase_resource(active_uavs_, id);
        erase_resource(saved_uavs_, id);
    }

    void after_rendering(VkCommandBuffer cb, std::span<const TargetImage> targets,
                         const VkConditionalRenderingBeginInfoEXT* predicate = nullptr) {
        if (!capturing_ || targets.empty()) return;
        capture_pass(cb, "render", targets, false, predicate);
    }

    template<typename IsGeneral>
    void after_dispatch(VkCommandBuffer cb, bool writes_storage_image, IsGeneral&& is_general,
                        const VkConditionalRenderingBeginInfoEXT* predicate = nullptr) {
        if (!capturing_ || !writes_storage_image) return;
        std::vector<TargetImage> targets;
        {
            std::lock_guard lock(mutex_);
            if (active_uavs_.empty()) return;
            std::uint32_t slot_index = 0;
            for (const auto& [key, tracked] : active_uavs_) {
                if (tracked.queue != current_queue_) continue;
                const UavImage& uav = tracked.image;
                if (!uav.image || uav.samples > 1 || uav.aspects != VK_IMAGE_ASPECT_COLOR_BIT) continue;
                if (!is_general(std::get<0>(key), uav.aspects, uav.mip, uav.layer)) continue;
                TargetImage target{};
                target.image = uav.image;
                target.format = uav.format;
                target.image_format = uav.format;
                target.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
                target.layout_aspects = VK_IMAGE_ASPECT_COLOR_BIT;
                target.layout = VK_IMAGE_LAYOUT_GENERAL;
                target.width = std::max(1u, uav.width >> uav.mip);
                target.height = std::max(1u, uav.height >> uav.mip);
                target.mip = uav.mip;
                target.layer = uav.layer;
                target.samples = uav.samples;
                target.volume = uav.volume;
                target.slot = "uav" + std::to_string(slot_index++);
                targets.push_back(std::move(target));
            }
        }
        if (!targets.empty()) capture_pass(cb, "dispatch", targets, true, predicate);
    }

    void commit_submission(bool submitted, std::uint64_t ticket) {
        if (!submitted) {
            {
                std::lock_guard lock(mutex_);
                active_uavs_ = std::move(saved_uavs_);
            }
            pending_passes_.clear();
            pending_captured_ = false;
            capturing_ = false;
            current_queue_ = VK_NULL_HANDLE;
            return;
        }
        if (!pending_passes_.empty()) {
            committed_passes_ = next_pass_;
            std::lock_guard lock(mutex_);
            in_flight_.push_back({ticket, std::move(pending_passes_)});
            has_pending_.store(true, std::memory_order_release);
        }
        {
            std::lock_guard lock(mutex_);
            saved_uavs_.clear();
        }
        pending_passes_.clear();
        pending_captured_ = false;
        capturing_ = false;
        current_queue_ = VK_NULL_HANDLE;
    }

    std::uint64_t max_pending_ticket() const {
        std::lock_guard lock(mutex_);
        std::uint64_t maximum = 0;
        for (const auto& batch : in_flight_) maximum = std::max(maximum, batch.ticket);
        return maximum;
    }

    void drain(std::uint64_t completed_ticket, std::uint64_t current_frame = 0) {
        std::vector<InFlightBatch> ready;
        {
            std::lock_guard lock(mutex_);
            if (!in_flight_.empty()) {
                for (auto it = in_flight_.begin(); it != in_flight_.end();) {
                    if (it->ticket <= completed_ticket) {
                        ready.push_back(std::move(*it));
                        it = in_flight_.erase(it);
                    } else {
                        ++it;
                    }
                }
                has_pending_.store(!in_flight_.empty(), std::memory_order_release);
            }
        }
        if (!ready.empty()) {
            std::lock_guard io_lock(io_mutex_);
            for (auto& batch : ready) {
                for (auto& pass : batch.passes) {
                    IndexPass entry;
                    entry.pass = pass.pass;
                    entry.kind = std::move(pass.kind);
                    for (auto& target : pass.targets) {
                        write_captured_target(pass.pass, target, entry.targets);
                        target.staging.reset();
                    }
                    if (!entry.targets.empty()) completed_passes_.push_back(std::move(entry));
                }
            }
            std::stable_sort(completed_passes_.begin(), completed_passes_.end(),
                             [](const IndexPass& a, const IndexPass& b) { return a.pass < b.pass; });
            write_index_locked();
        }
        if (current_frame > 0) finalize_if_complete(current_frame);
    }

    void finalize_for_teardown() {
        if (!configured_frame_) return;
        std::lock_guard io_lock(io_mutex_);
        if ((saw_target_frame_.load(std::memory_order_relaxed) || !completed_passes_.empty()) && !index_written_) {
            write_index_locked();
        }
        target_frame_.store(0, std::memory_order_relaxed);
    }

    void discard_pending() {
        pending_passes_.clear();
        std::vector<InFlightBatch> leftovers;
        {
            std::lock_guard lock(mutex_);
            leftovers.swap(in_flight_);
            has_pending_.store(false, std::memory_order_release);
        }
        leftovers.clear();
    }

private:
    static constexpr VkDeviceSize default_max_staging_bytes = VkDeviceSize(512) << 20;
    static constexpr std::uint32_t default_max_staging_allocations = 256u;

    struct TrackedUav {
        VkQueue queue = VK_NULL_HANDLE;
        UavImage image{};
    };

    using UavKey = std::tuple<std::uint64_t, std::uint32_t, std::uint32_t>;
    using UavMap = std::map<UavKey, TrackedUav>;

    static void erase_resource(UavMap& uavs, std::uint64_t id) {
        const auto first = uavs.lower_bound(UavKey{id, 0, 0});
        const auto last = uavs.upper_bound(UavKey{id, UINT32_MAX, UINT32_MAX});
        uavs.erase(first, last);
    }

    struct DeviceFunctions {
        PFN_vkCreateBuffer create_buffer = nullptr;
        PFN_vkDestroyBuffer destroy_buffer = nullptr;
        PFN_vkGetBufferMemoryRequirements buffer_requirements = nullptr;
        PFN_vkAllocateMemory allocate_memory = nullptr;
        PFN_vkFreeMemory free_memory = nullptr;
        PFN_vkBindBufferMemory bind_buffer = nullptr;
        PFN_vkMapMemory map_memory = nullptr;
        PFN_vkUnmapMemory unmap_memory = nullptr;
        PFN_vkCmdPipelineBarrier pipeline_barrier = nullptr;
        PFN_vkCmdCopyImageToBuffer copy_image_to_buffer = nullptr;
        PFN_vkCmdBeginConditionalRenderingEXT begin_conditional_rendering = nullptr;
        PFN_vkCmdEndConditionalRenderingEXT end_conditional_rendering = nullptr;

        bool complete() const noexcept {
            return create_buffer && destroy_buffer && buffer_requirements && allocate_memory && free_memory &&
                   bind_buffer && map_memory && unmap_memory && pipeline_barrier && copy_image_to_buffer;
        }
    };

    struct StagingAllocation {
        TargetDump* owner = nullptr;
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkDeviceSize bytes = 0;

        StagingAllocation() = default;
        StagingAllocation(TargetDump* owner_in, VkBuffer buf, VkDeviceMemory mem, VkDeviceSize size) noexcept
            : owner(owner_in), buffer(buf), memory(mem), bytes(size) {}
        ~StagingAllocation() { reset(); }
        StagingAllocation(const StagingAllocation&) = delete;
        StagingAllocation& operator=(const StagingAllocation&) = delete;
        StagingAllocation(StagingAllocation&& other) noexcept
            : owner(std::exchange(other.owner, nullptr)),
              buffer(std::exchange(other.buffer, VK_NULL_HANDLE)),
              memory(std::exchange(other.memory, VK_NULL_HANDLE)),
              bytes(std::exchange(other.bytes, 0)) {}
        StagingAllocation& operator=(StagingAllocation&& other) noexcept {
            if (this != &other) {
                reset();
                owner = std::exchange(other.owner, nullptr);
                buffer = std::exchange(other.buffer, VK_NULL_HANDLE);
                memory = std::exchange(other.memory, VK_NULL_HANDLE);
                bytes = std::exchange(other.bytes, 0);
            }
            return *this;
        }
        explicit operator bool() const noexcept { return buffer != VK_NULL_HANDLE && memory != VK_NULL_HANDLE; }
        void reset() noexcept {
            if (owner) {
                owner->release_staging(buffer, memory, bytes);
                owner = nullptr;
            }
            buffer = VK_NULL_HANDLE;
            memory = VK_NULL_HANDLE;
            bytes = 0;
        }
    };

    struct PendingTarget {
        std::string slot;
        VkFormat format = VK_FORMAT_UNDEFINED;
        VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;
        std::uint32_t width = 0;
        std::uint32_t height = 0;
        std::uint32_t mip = 0;
        std::uint32_t layer = 0;
        unsigned bytes_per_texel = 0;
        VkDeviceSize bytes = 0;
        StagingAllocation staging;
    };

    struct PendingPass {
        std::uint32_t pass = 0;
        std::string kind;
        std::vector<PendingTarget> targets;
    };

    struct InFlightBatch {
        std::uint64_t ticket = 0;
        std::vector<PendingPass> passes;
    };

    static void report_invalid_env() {
        static std::once_flag once;
        std::call_once(once, [] {
            std::fprintf(stderr,
                         "[rt] target-dump: invalid LEONIDA_TARGET_DUMP (expected <dir>,<frame>, frame > 0); disabled\n");
        });
    }

    static bool open_exclusive(const std::filesystem::path& path, std::FILE*& file) {
        file = nullptr;
#ifdef _WIN32
        return _wfopen_s(&file, path.c_str(), L"wbx") == 0 && file != nullptr;
#else
        file = std::fopen(path.c_str(), "wbx");
        return file != nullptr;
#endif
    }

    template<typename Writer>
    static bool publish_file(const std::filesystem::path& path, Writer&& writer) {
        auto tmp = path;
        tmp += ".tmp";
        std::FILE* file = nullptr;
        if (!open_exclusive(tmp, file)) return false;
        bool ok = false;
        try {
            ok = writer(file);
        } catch (const std::exception&) {
            ok = false;
        }
        if (std::fclose(file) != 0) ok = false;
        std::error_code ec;
        if (ok) {
            std::filesystem::rename(tmp, path, ec);
            if (!ec) return true;
        }
        std::filesystem::remove(tmp, ec);
        return false;
    }

    void release_staging(VkBuffer buffer, VkDeviceMemory memory, VkDeviceSize bytes) noexcept {
        if (device_) {
            if (buffer && fns_.destroy_buffer) fns_.destroy_buffer(device_, buffer, nullptr);
            if (memory && fns_.free_memory) fns_.free_memory(device_, memory, nullptr);
        }
        std::lock_guard lock(mutex_);
        if (outstanding_staging_allocations_ > 0) --outstanding_staging_allocations_;
        outstanding_staging_bytes_ = outstanding_staging_bytes_ >= bytes ? outstanding_staging_bytes_ - bytes : 0;
    }

    bool allocate_staging(VkDeviceSize bytes, StagingAllocation& out) {
        out.reset();
        {
            std::lock_guard lock(mutex_);
            if (outstanding_staging_allocations_ >= max_staging_allocations_ || bytes > max_staging_bytes_ ||
                outstanding_staging_bytes_ > max_staging_bytes_ - bytes) {
                if (!staging_limit_logged_.exchange(true, std::memory_order_relaxed)) {
                    std::fprintf(stderr,
                                 "[rt] target-dump: staging limit reached (%u/%u allocations, %llu/%llu bytes); "
                                 "skipping further captures until earlier submissions retire\n",
                                 outstanding_staging_allocations_, max_staging_allocations_,
                                 static_cast<unsigned long long>(outstanding_staging_bytes_),
                                 static_cast<unsigned long long>(max_staging_bytes_));
                }
                return false;
            }
            ++outstanding_staging_allocations_;
            outstanding_staging_bytes_ += bytes;
        }

        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        StagingAllocation guard(this, VK_NULL_HANDLE, VK_NULL_HANDLE, bytes);
        VkBufferCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        info.size = bytes;
        info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (fns_.create_buffer(device_, &info, nullptr, &buffer) != VK_SUCCESS) return false;
        guard.buffer = buffer;

        VkMemoryRequirements req{};
        fns_.buffer_requirements(device_, buffer, &req);
        const auto order = memory_type_order(
            memory_, req.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            VK_MEMORY_PROPERTY_HOST_CACHED_BIT, req.size);
        for (const std::uint32_t type : order) {
            VkMemoryAllocateInfo alloc{};
            alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            alloc.allocationSize = req.size;
            alloc.memoryTypeIndex = type;
            if (fns_.allocate_memory(device_, &alloc, nullptr, &memory) == VK_SUCCESS) break;
            memory = VK_NULL_HANDLE;
        }
        guard.memory = memory;
        if (!memory || fns_.bind_buffer(device_, buffer, memory, 0) != VK_SUCCESS) return false;

        out = std::move(guard);
        return true;
    }

    void capture_pass(VkCommandBuffer cb, const char* kind, std::span<const TargetImage> targets, bool is_dispatch,
                      const VkConditionalRenderingBeginInfoEXT* predicate) {
        const bool suspend_predicate =
            predicate && predicate->buffer && fns_.end_conditional_rendering && fns_.begin_conditional_rendering;
        if (suspend_predicate) fns_.end_conditional_rendering(cb);

        PendingPass pass;
        pass.pass = next_pass_;
        pass.kind = kind;
        for (const auto& target : targets) {
            if (!target.image || target.samples > 1 || target.width == 0 || target.height == 0) continue;
            VkFormat effective_format = target.format;
            const unsigned view_bpp = raw_texel_size(target.format, target.aspect);
            const unsigned image_bpp = target.image_format != VK_FORMAT_UNDEFINED
                                           ? raw_texel_size(target.image_format, target.aspect)
                                           : view_bpp;
            unsigned bpp = view_bpp;
            if (target.image_format != VK_FORMAT_UNDEFINED && view_bpp != image_bpp) {
                effective_format = target.image_format;
                bpp = image_bpp;
            }
            if (bpp == 0) continue;
            const std::uint64_t pixels = std::uint64_t(target.width) * std::uint64_t(target.height);
            if (pixels == 0 || pixels > (std::numeric_limits<VkDeviceSize>::max)() / bpp) continue;
            const VkDeviceSize bytes = VkDeviceSize(pixels * bpp);

            StagingAllocation staging;
            if (!allocate_staging(bytes, staging)) continue;
            const VkBuffer buffer = staging.buffer;

            const VkImageLayout copy_layout = target.layout == VK_IMAGE_LAYOUT_GENERAL
                                                  ? VK_IMAGE_LAYOUT_GENERAL
                                                  : VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            const VkImageSubresourceRange barrier_range{
                target.layout_aspects, target.mip, 1, target.volume ? 0u : target.layer,
                target.volume ? VK_REMAINING_ARRAY_LAYERS : 1u};

            VkPipelineStageFlags src_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            VkAccessFlags src_access = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            if (is_dispatch) {
                src_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
                src_access = VK_ACCESS_SHADER_WRITE_BIT;
            } else if (target.aspect & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)) {
                src_stage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
                src_access = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            }

            VkImageMemoryBarrier before{};
            before.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            before.srcAccessMask = src_access;
            before.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            before.oldLayout = target.layout;
            before.newLayout = copy_layout;
            before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            before.image = target.image;
            before.subresourceRange = barrier_range;
            fns_.pipeline_barrier(cb, src_stage, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &before);

            VkBufferImageCopy region{};
            region.bufferOffset = 0;
            region.bufferRowLength = 0;
            region.bufferImageHeight = 0;
            region.imageSubresource = {target.aspect, target.mip, target.volume ? 0u : target.layer, 1};
            region.imageOffset = {0, 0, target.volume ? static_cast<std::int32_t>(target.layer) : 0};
            region.imageExtent = {target.width, target.height, 1};
            fns_.copy_image_to_buffer(cb, target.image, copy_layout, buffer, 1, &region);

            VkImageMemoryBarrier after{};
            after.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            after.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            after.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
            after.oldLayout = copy_layout;
            after.newLayout = target.layout;
            after.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            after.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            after.image = target.image;
            after.subresourceRange = barrier_range;
            fns_.pipeline_barrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr,
                                  0, nullptr, 1, &after);

            VkBufferMemoryBarrier host{};
            host.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            host.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
            host.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            host.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            host.buffer = buffer;
            host.offset = 0;
            host.size = bytes;
            fns_.pipeline_barrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1,
                                  &host, 0, nullptr);

            PendingTarget captured;
            captured.slot = target.slot;
            captured.format = effective_format;
            captured.aspect = target.aspect;
            captured.width = target.width;
            captured.height = target.height;
            captured.mip = target.mip;
            captured.layer = target.layer;
            captured.bytes_per_texel = bpp;
            captured.bytes = bytes;
            captured.staging = std::move(staging);
            pass.targets.push_back(std::move(captured));
        }

        if (suspend_predicate) fns_.begin_conditional_rendering(cb, predicate);

        if (!pass.targets.empty()) {
            ++next_pass_;
            pending_captured_ = true;
            pending_passes_.push_back(std::move(pass));
        }
    }

    bool write_captured_target(std::uint32_t pass_number, const PendingTarget& target,
                               std::vector<IndexTarget>& out_targets) {
        if (!target.staging) return false;
        void* mapped = nullptr;
        if (fns_.map_memory(device_, target.staging.memory, 0, target.bytes, 0, &mapped) != VK_SUCCESS || !mapped)
            return false;
        const auto* bytes = static_cast<const std::uint8_t*>(mapped);
        const bool png = uses_png(target.format, target.aspect);
        char prefix[32];
        std::snprintf(prefix, sizeof(prefix), "pass_%04u_", pass_number);
        const std::string filename = std::string(prefix) + target.slot + (png ? ".png" : ".raw");
        const std::filesystem::path full_path = dir_ / filename;
        const bool written = publish_file(full_path, [&](std::FILE* file) {
            if (png) {
                return present_dump::write_png(file, {target.width, target.height}, target.format, false, bytes);
            }
            return std::fwrite(bytes, 1, static_cast<std::size_t>(target.bytes), file) ==
                   static_cast<std::size_t>(target.bytes);
        });
        fns_.unmap_memory(device_, target.staging.memory);
        if (!written) return false;
        out_targets.push_back({target.slot, target.format, target.width, target.height, target.mip, target.layer,
                               target.bytes_per_texel, png, filename});
        return true;
    }

    void write_index_locked() {
        const std::string json = serialize_index_json(configured_frame_, completed_passes_);
        if (publish_file(dir_ / "index.json", [&](std::FILE* file) {
                return std::fwrite(json.data(), 1, json.size(), file) == json.size();
            })) {
            index_written_ = true;
        }
    }

    void finalize_if_complete(std::uint64_t current_frame) {
        const std::uint64_t target = target_frame_.load(std::memory_order_relaxed);
        if (!target || current_frame <= target || has_pending_.load(std::memory_order_acquire)) return;
        std::lock_guard io_lock(io_mutex_);
        if (has_pending_.load(std::memory_order_acquire)) return;
        if (!index_written_) write_index_locked();
        target_frame_.store(0, std::memory_order_relaxed);
    }

    std::filesystem::path dir_;
    std::uint64_t configured_frame_ = 0;
    std::atomic<std::uint64_t> target_frame_{0};
    std::atomic<bool> has_pending_{false};
    std::atomic<bool> saw_target_frame_{false};
    VkDevice device_ = VK_NULL_HANDLE;
    DeviceFunctions fns_{};
    VkPhysicalDeviceMemoryProperties memory_{};

    mutable std::mutex mutex_;
    UavMap active_uavs_;
    UavMap saved_uavs_;
    std::vector<InFlightBatch> in_flight_;
    VkDeviceSize max_staging_bytes_ = default_max_staging_bytes;
    std::uint32_t max_staging_allocations_ = default_max_staging_allocations;
    VkDeviceSize outstanding_staging_bytes_ = 0;
    std::uint32_t outstanding_staging_allocations_ = 0;
    std::atomic<bool> staging_limit_logged_{false};

    std::mutex io_mutex_;
    std::vector<IndexPass> completed_passes_;
    bool index_written_ = false;

    bool capturing_ = false;
    bool pending_captured_ = false;
    VkQueue current_queue_ = VK_NULL_HANDLE;
    std::uint32_t committed_passes_ = 0;
    std::uint32_t next_pass_ = 0;
    std::vector<PendingPass> pending_passes_;
};

}

namespace leonida {
using target_dump::TargetDump;
}
