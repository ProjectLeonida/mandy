#pragma once

// Builds a VkSpecializationInfo binding literal-sampler SPIR-V specialization constants (SpecId spec_base + i)
// to their uint32_t bindless sampler array indices (or 0xFFFFFFFFu when a sampler failed to bind so the shader
// safely falls back to the descriptor-table hash probe).

#include <vulkan/vulkan.h>

#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace leonida {

struct LiteralSamplerSpec {
    static constexpr std::uint32_t kFallbackIndex = 0xFFFFFFFFu;

    std::vector<VkSpecializationMapEntry> entries;
    std::vector<std::uint32_t> values;
    VkSpecializationInfo info{};

    LiteralSamplerSpec() = default;
    LiteralSamplerSpec(const LiteralSamplerSpec&) = delete;
    LiteralSamplerSpec& operator=(const LiteralSamplerSpec&) = delete;
    LiteralSamplerSpec(LiteralSamplerSpec&&) = delete;
    LiteralSamplerSpec& operator=(LiteralSamplerSpec&&) = delete;

    void clear() {
        entries.clear();
        values.clear();
        info = {};
    }

    // Populates entries/values/info for SpecId [spec_base, spec_base + indices.size()).
    // Leaves the specialization empty if `indices` is empty or `spec_base + indices.size() - 1` would overflow uint32.
    void build(std::uint32_t spec_base, std::span<const std::uint32_t> indices) {
        clear();
        if (indices.empty()) return;
        const std::size_t max_count =
            std::size_t(std::numeric_limits<std::uint32_t>::max() - spec_base) + 1u;
        if (indices.size() > max_count) return;
        entries.reserve(indices.size());
        values.assign(indices.begin(), indices.end());
        for (std::size_t i = 0; i < values.size(); ++i) {
            entries.push_back({spec_base + static_cast<std::uint32_t>(i),
                               static_cast<std::uint32_t>(i * sizeof(std::uint32_t)),
                               sizeof(std::uint32_t)});
        }
        info.mapEntryCount = static_cast<std::uint32_t>(entries.size());
        info.pMapEntries = entries.data();
        info.dataSize = values.size() * sizeof(std::uint32_t);
        info.pData = values.data();
    }

    const VkSpecializationInfo* ptr() const noexcept { return entries.empty() ? nullptr : &info; }
    bool active() const noexcept { return !entries.empty(); }
};

}  // namespace leonida
