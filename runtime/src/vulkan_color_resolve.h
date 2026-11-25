#pragma once

#include "color_resolve_spv.h"

#include <vulkan/vulkan.h>

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <span>
#include <utility>

namespace leonida {

enum class ColorResolveType { unsupported, floating, uint, sint };

inline ColorResolveType color_resolve_type(VkFormat format) {
    switch (format) {
    case VK_FORMAT_A1R5G5B5_UNORM_PACK16:
    case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
    case VK_FORMAT_A4R4G4B4_UNORM_PACK16:
    case VK_FORMAT_B10G11R11_UFLOAT_PACK32:
    case VK_FORMAT_B8G8R8A8_SRGB:
    case VK_FORMAT_B8G8R8A8_UNORM:
    case VK_FORMAT_R16G16B16A16_SFLOAT:
    case VK_FORMAT_R16G16B16A16_UNORM:
    case VK_FORMAT_R16G16_SFLOAT:
    case VK_FORMAT_R16G16_UNORM:
    case VK_FORMAT_R16_SFLOAT:
    case VK_FORMAT_R16_UNORM:
    case VK_FORMAT_R32G32B32A32_SFLOAT:
    case VK_FORMAT_R32G32_SFLOAT:
    case VK_FORMAT_R32_SFLOAT:
    case VK_FORMAT_R4G4B4A4_UNORM_PACK16:
    case VK_FORMAT_R5G6B5_UNORM_PACK16:
    case VK_FORMAT_R8G8B8A8_SRGB:
    case VK_FORMAT_R8G8B8A8_UNORM:
    case VK_FORMAT_R8G8_UNORM:
    case VK_FORMAT_R8_UNORM:
        return ColorResolveType::floating;
    case VK_FORMAT_A2B10G10R10_UINT_PACK32:
    case VK_FORMAT_R16G16B16A16_UINT:
    case VK_FORMAT_R16G16_UINT:
    case VK_FORMAT_R16_UINT:
    case VK_FORMAT_R32G32B32A32_UINT:
    case VK_FORMAT_R32G32_UINT:
    case VK_FORMAT_R32_UINT:
    case VK_FORMAT_R8G8B8A8_UINT:
    case VK_FORMAT_R8G8_UINT:
    case VK_FORMAT_R8_UINT:
        return ColorResolveType::uint;
    case VK_FORMAT_R16G16B16A16_SINT:
    case VK_FORMAT_R16G16_SINT:
    case VK_FORMAT_R16_SINT:
    case VK_FORMAT_R32G32B32A32_SINT:
    case VK_FORMAT_R32G32_SINT:
    case VK_FORMAT_R32_SINT:
    case VK_FORMAT_R8G8B8A8_SINT:
    case VK_FORMAT_R8G8_SINT:
    case VK_FORMAT_R8_SINT:
        return ColorResolveType::sint;
    default: return ColorResolveType::unsupported;
    }
}

struct ColorResolveJob {
    VkImage scratch = VK_NULL_HANDLE;
    VkImageView target = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
};

// The temporary attachment is sized to the rectangle. The queue retains this allocation and the immutable
// descriptor set until completion, alongside the other internal passes' per-submit resources.
template<class Functions>
class ColorResolvePass {
public:
    using Allocate = std::function<VkDeviceMemory(const VkMemoryRequirements&)>;
    using Supported = std::function<bool(const VkImageCreateInfo&)>;

    ColorResolvePass(const Functions& vk, VkDevice device, Allocate allocate, Supported supported)
        : vk_(vk), device_(device), allocate_(std::move(allocate)), supported_(std::move(supported)) {}
    ~ColorResolvePass() {
        for (const auto& [format, pipeline] : pipelines_) vk_.vkDestroyPipeline(device_, pipeline, nullptr);
        if (layout_) vk_.vkDestroyPipelineLayout(device_, layout_, nullptr);
        if (set_layout_) vk_.vkDestroyDescriptorSetLayout(device_, set_layout_, nullptr);
    }

    std::shared_ptr<void> prepare(VkImage source, VkFormat format, const VkImageSubresourceRange& range,
                                  VkImageLayout source_layout, VkExtent3D extent, ColorResolveJob& job) {
        const auto type = color_resolve_type(format);
        if (type == ColorResolveType::unsupported || !available()) return {};
        std::lock_guard lock(mutex_);

        VkImageCreateInfo image{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        image.imageType = VK_IMAGE_TYPE_2D;
        image.format = format;
        image.extent = extent;
        image.mipLevels = image.arrayLayers = 1;
        image.samples = VK_SAMPLE_COUNT_1_BIT;
        image.tiling = VK_IMAGE_TILING_OPTIMAL;
        image.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        if (!supported_(image)) return {};
        const VkPipeline pipeline = get_pipeline(format, type);
        if (!pipeline) return {};

        auto allocation = std::make_shared<Allocation>(vk_, device_);
        if (vk_.vkCreateImage(device_, &image, nullptr, &allocation->scratch) != VK_SUCCESS) return {};
        VkMemoryRequirements requirements{};
        vk_.vkGetImageMemoryRequirements(device_, allocation->scratch, &requirements);
        allocation->memory = allocate_(requirements);
        if (!allocation->memory ||
            vk_.vkBindImageMemory(device_, allocation->scratch, allocation->memory, 0) != VK_SUCCESS) return {};

        VkImageViewUsageCreateInfo usage{VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO};
        usage.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
        VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        view.pNext = &usage;
        view.image = source;
        view.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view.format = format;
        view.subresourceRange = range;
        if (vk_.vkCreateImageView(device_, &view, nullptr, &allocation->source) != VK_SUCCESS) return {};
        view.pNext = nullptr;
        view.image = allocation->scratch;
        view.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        if (vk_.vkCreateImageView(device_, &view, nullptr, &allocation->target) != VK_SUCCESS) return {};

        const VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1};
        VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pool.maxSets = pool.poolSizeCount = 1;
        pool.pPoolSizes = &size;
        if (vk_.vkCreateDescriptorPool(device_, &pool, nullptr, &allocation->pool) != VK_SUCCESS) return {};
        VkDescriptorSetAllocateInfo set{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        set.descriptorPool = allocation->pool;
        set.descriptorSetCount = 1;
        set.pSetLayouts = &set_layout_;
        VkDescriptorSet descriptor = VK_NULL_HANDLE;
        if (vk_.vkAllocateDescriptorSets(device_, &set, &descriptor) != VK_SUCCESS) return {};
        const VkDescriptorImageInfo input{VK_NULL_HANDLE, allocation->source, source_layout};
        VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write.dstSet = descriptor;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        write.pImageInfo = &input;
        vk_.vkUpdateDescriptorSets(device_, 1, &write, 0, nullptr);
        job = {allocation->scratch, allocation->target, descriptor, pipeline, layout_};
        return allocation;
    }

private:
    struct Allocation {
        const Functions& vk;
        VkDevice device;
        VkImage scratch = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkImageView source = VK_NULL_HANDLE, target = VK_NULL_HANDLE;
        VkDescriptorPool pool = VK_NULL_HANDLE;

        ~Allocation() {
            if (pool) vk.vkDestroyDescriptorPool(device, pool, nullptr);
            if (source) vk.vkDestroyImageView(device, source, nullptr);
            if (target) vk.vkDestroyImageView(device, target, nullptr);
            if (scratch) vk.vkDestroyImage(device, scratch, nullptr);
            if (memory) vk.vkFreeMemory(device, memory, nullptr);
        }
    };

    bool available() const {
        return vk_.vkCreateImage && vk_.vkDestroyImage && vk_.vkGetImageMemoryRequirements &&
               vk_.vkBindImageMemory && vk_.vkFreeMemory && vk_.vkCreateImageView && vk_.vkDestroyImageView &&
               vk_.vkCreateDescriptorPool && vk_.vkDestroyDescriptorPool && vk_.vkAllocateDescriptorSets &&
               vk_.vkUpdateDescriptorSets && vk_.vkCreateDescriptorSetLayout && vk_.vkDestroyDescriptorSetLayout &&
               vk_.vkCreatePipelineLayout && vk_.vkDestroyPipelineLayout && vk_.vkCreateShaderModule &&
               vk_.vkDestroyShaderModule && vk_.vkCreateGraphicsPipelines && vk_.vkDestroyPipeline;
    }

    bool ensure_layout() {
        if (layout_) return true;
        if (!set_layout_) {
            const VkDescriptorSetLayoutBinding binding{0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1,
                                                       VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
            VkDescriptorSetLayoutCreateInfo set{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
            set.bindingCount = 1;
            set.pBindings = &binding;
            if (vk_.vkCreateDescriptorSetLayout(device_, &set, nullptr, &set_layout_) != VK_SUCCESS) return false;
        }
        const VkPushConstantRange push{VK_SHADER_STAGE_FRAGMENT_BIT, 0, 4 * sizeof(std::int32_t)};
        VkPipelineLayoutCreateInfo layout{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layout.setLayoutCount = layout.pushConstantRangeCount = 1;
        layout.pSetLayouts = &set_layout_;
        layout.pPushConstantRanges = &push;
        return vk_.vkCreatePipelineLayout(device_, &layout, nullptr, &layout_) == VK_SUCCESS;
    }

    VkPipeline get_pipeline(VkFormat format, ColorResolveType type) {
        if (const auto it = pipelines_.find(format); it != pipelines_.end()) return it->second;
        if (!ensure_layout()) return VK_NULL_HANDLE;
        using Words = std::span<const std::uint32_t>;
        const Words fragment = type == ColorResolveType::uint ? Words(color_resolve_uint_spv)
                             : type == ColorResolveType::sint ? Words(color_resolve_sint_spv)
                                                             : Words(color_resolve_float_spv);
        const auto module = [&](Words words) {
            VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
            info.codeSize = words.size_bytes();
            info.pCode = words.data();
            VkShaderModule shader = VK_NULL_HANDLE;
            if (vk_.vkCreateShaderModule(device_, &info, nullptr, &shader) != VK_SUCCESS) return VkShaderModule{};
            return shader;
        };
        const VkShaderModule vs = module(color_resolve_vertex_spv), fs = module(fragment);
        VkPipeline pipeline = VK_NULL_HANDLE;
        if (vs && fs) {
            const VkPipelineShaderStageCreateInfo stages[] = {
                {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, vs, "main"},
                {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, fs, "main"},
            };
            const VkPipelineVertexInputStateCreateInfo vertices{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
            VkPipelineInputAssemblyStateCreateInfo assembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
            assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
            VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
            viewport.viewportCount = viewport.scissorCount = 1;
            VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
            raster.polygonMode = VK_POLYGON_MODE_FILL;
            raster.lineWidth = 1.0f;
            VkPipelineMultisampleStateCreateInfo samples{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
            samples.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
            VkPipelineColorBlendAttachmentState output{};
            output.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                    VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
            VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
            blend.attachmentCount = 1;
            blend.pAttachments = &output;
            const VkDynamicState states[]{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
            VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
            dynamic.dynamicStateCount = 2;
            dynamic.pDynamicStates = states;
            VkPipelineRenderingCreateInfo rendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
            rendering.colorAttachmentCount = 1;
            rendering.pColorAttachmentFormats = &format;
            VkGraphicsPipelineCreateInfo info{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
            info.pNext = &rendering;
            info.stageCount = 2;
            info.pStages = stages;
            info.pVertexInputState = &vertices;
            info.pInputAssemblyState = &assembly;
            info.pViewportState = &viewport;
            info.pRasterizationState = &raster;
            info.pMultisampleState = &samples;
            info.pColorBlendState = &blend;
            info.pDynamicState = &dynamic;
            info.layout = layout_;
            if (vk_.vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline) == VK_SUCCESS)
                pipelines_.emplace(format, pipeline);
            else if (pipeline) {
                vk_.vkDestroyPipeline(device_, pipeline, nullptr);
                pipeline = VK_NULL_HANDLE;
            }
        }
        if (fs) vk_.vkDestroyShaderModule(device_, fs, nullptr);
        if (vs) vk_.vkDestroyShaderModule(device_, vs, nullptr);
        return pipeline;
    }

    const Functions& vk_;
    VkDevice device_;
    Allocate allocate_;
    Supported supported_;
    std::mutex mutex_;
    VkDescriptorSetLayout set_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    std::map<VkFormat, VkPipeline> pipelines_;
};

} // namespace leonida
