// GPU-only count gating for simple D3D12 ExecuteIndirect(DISPATCH) signatures.
// Vulkan has vkCmdDispatchIndirect but no vkCmdDispatchIndirectCount. We prepare a private indirect
// buffer with zero-group records past the GPU count, then execute all MaxCommandCount records.
#pragma once
#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include <vulkan/vulkan.h>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <charconv>
#include <string_view>
#include <limits>
#include <memory>
#include <vector>
#include "indirect_dispatch_count_spv.h"
#include "vulkan_aftermath.h"
#include "vulkan_memory.h"

namespace leonida {

// Vulkan guarantees at least 65535 groups along each axis. A total budget
// at or below it bounds both the workload and every dimension.
inline std::uint32_t indirect_group_budget() {
    static const auto budget = []() -> std::uint32_t {
        const char* value = std::getenv("LEONIDA_INDIRECT_GROUP_BUDGET");
        if (!value) return 0;
        const std::string_view text(value);
        std::uint32_t groups{};
        const auto end = value + text.size();
        const auto parsed = std::from_chars(value, end, groups);
        return parsed.ec == std::errc{} && parsed.ptr == end ? std::min(groups, 65535u) : 0;
    }();
    return budget;
}

struct CountedDispatch {
    VkBuffer indirect = VK_NULL_HANDLE;
    VkDeviceSize offset = 0;
    VkDeviceAddress arguments = 0;
    VkDeviceAddress count = 0;
    VkDeviceAddress output = 0;
    std::uint32_t stride = 0;
    std::uint32_t max_count = 0;
};

class IndirectDispatchCounter {
public:
    IndirectDispatchCounter(PFN_vkGetDeviceProcAddr get, VkDevice device,
                            const VkPhysicalDeviceMemoryProperties& memory, const VkPhysicalDeviceLimits& limits)
        : get_(get), device_(device), memory_(memory), limits_(limits) {}
    ~IndirectDispatchCounter() {
        if (pipeline_) vk_.destroy_pipeline(device_, pipeline_, nullptr);
        if (layout_) vk_.destroy_layout(device_, layout_, nullptr);
    }
    IndirectDispatchCounter(const IndirectDispatchCounter&) = delete;
    IndirectDispatchCounter& operator=(const IndirectDispatchCounter&) = delete;

    // Retained by the command-buffer slot until its submission's timeline ticket completes.
    struct Batch {
        IndirectDispatchCounter& owner;
        struct Allocation {
            VkBuffer buffer = VK_NULL_HANDLE;
            VkDeviceMemory memory = VK_NULL_HANDLE;
        };
        std::vector<Allocation> allocations;
        explicit Batch(IndirectDispatchCounter& value) : owner(value) {}
        ~Batch() {
            for (auto& a : allocations) {
                if (a.buffer) owner.vk_.destroy_buffer(owner.device_, a.buffer, nullptr);
                if (a.memory) owner.vk_.free_memory(owner.device_, a.memory, nullptr);
            }
        }
        Batch(const Batch&) = delete;
        Batch& operator=(const Batch&) = delete;

        bool prepare(VkBuffer arguments, VkDeviceSize argument_offset, VkBuffer count,
                     VkDeviceSize count_offset, std::uint32_t stride, std::uint32_t max_count,
                     CountedDispatch& out) {
            out = {};
            constexpr VkDeviceSize record_bytes = 3 * sizeof(std::uint32_t);
            if (!arguments || (!count && (count_offset || !indirect_group_budget())) || stride < record_bytes ||
                stride % 4 || (argument_offset % 4) != 0 || (count_offset % 4) != 0 || !max_count ||
                !owner.limits_.maxComputeWorkGroupCount[0] || !owner.make_pipeline()) return false;
            if (VkDeviceSize(max_count) > std::numeric_limits<VkDeviceSize>::max() / record_bytes)
                return false;
            const VkDeviceSize bytes = VkDeviceSize(max_count) * record_bytes;
            VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
            info.size = bytes;
            info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                         VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
            Allocation a{};
            const auto cleanup = [&] {
                if (a.buffer) owner.vk_.destroy_buffer(owner.device_, a.buffer, nullptr);
                if (a.memory) owner.vk_.free_memory(owner.device_, a.memory, nullptr);
                return false;
            };
            if (owner.vk_.create_buffer(owner.device_, &info, nullptr, &a.buffer) != VK_SUCCESS) {
                a.buffer = VK_NULL_HANDLE;
                return false;
            }
            VkMemoryRequirements requirements{};
            owner.vk_.get_reqs(owner.device_, a.buffer, &requirements);
            auto order = memory_type_order(owner.memory_, requirements.memoryTypeBits, 0,
                                           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, requirements.size);
            if (!resizable_bar(owner.memory_)) {
                std::stable_partition(order.begin(), order.end(), [&](std::uint32_t t) {
                    const auto f = owner.memory_.memoryTypes[t].propertyFlags;
                    return (f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) && !(f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
                });
            }
            if (order.empty()) return cleanup();
            VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
            flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
            VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &flags};
            allocation.allocationSize = requirements.size;
            for (const std::uint32_t memory_type : order) {
                allocation.memoryTypeIndex = memory_type;
                if (owner.vk_.allocate_memory(owner.device_, &allocation, nullptr, &a.memory) == VK_SUCCESS) break;
                a.memory = VK_NULL_HANDLE;
            }
            if (!a.memory || owner.vk_.bind_buffer(owner.device_, a.buffer, a.memory, 0) != VK_SUCCESS)
                return cleanup();
            VkBufferDeviceAddressInfo addr{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
            addr.buffer = arguments;
            const VkDeviceAddress args_va = owner.vk_.get_address(owner.device_, &addr);
            addr.buffer = count;
            const VkDeviceAddress count_va = count ? owner.vk_.get_address(owner.device_, &addr) : 0;
            addr.buffer = a.buffer;
            const VkDeviceAddress output_va = owner.vk_.get_address(owner.device_, &addr);
            if (!args_va || (count && !count_va) || !output_va ||
                argument_offset > UINT64_MAX - args_va || (count && count_offset > UINT64_MAX - count_va))
                return cleanup();
            allocations.push_back(a);
            out = {a.buffer, 0, args_va + argument_offset, count ? count_va + count_offset : 0,
                   output_va, stride, max_count};
            return true;
        }
    };

    std::unique_ptr<Batch> batch() { return std::make_unique<Batch>(*this); }

    void emit(VkCommandBuffer cb, const CountedDispatch& job) const {
        VkMemoryBarrier before{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        before.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
        before.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vk_.cmd_barrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        0, 1, &before, 0, nullptr, 0, nullptr);
        vk_.cmd_bind_pipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
        struct Push {
            VkDeviceAddress arguments, count, output;
            std::uint32_t stride_words, first_record, records, budget, has_count;
        };
        static_assert(sizeof(Push) == 48);
        const VkDeviceSize dispatch_limit = VkDeviceSize(limits_.maxComputeWorkGroupCount[0]) * 64;
        // Bound per-dispatch indexing to 32 bits, even when the D3D12 stride is very large.
        // The shader loads and stores three consecutive dwords per record, so leave room for the final +2.
        const std::uint32_t index_stride_words = std::max<std::uint32_t>(3u, job.stride / 4);
        const VkDeviceSize index_limit = (UINT32_MAX - 2u) / index_stride_words;
        const VkDeviceSize chunk_limit = std::min(dispatch_limit, index_limit + 1);
        for (VkDeviceSize first = 0; first < job.max_count;) {
            const std::uint32_t count = std::uint32_t(std::min<VkDeviceSize>(chunk_limit, job.max_count - first));
            const Push push{job.arguments + first * job.stride, job.count, job.output + first * 12,
                            job.stride / 4, std::uint32_t(first), count, indirect_group_budget(), job.count != 0};
            vk_.cmd_push_constants(cb, layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
            vk_.cmd_dispatch(cb, (count - 1u) / 64u + 1u, 1, 1);
            first += count;
        }
        VkMemoryBarrier after{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        after.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
        after.dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_MEMORY_READ_BIT |
                              VK_ACCESS_MEMORY_WRITE_BIT;
        vk_.cmd_barrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                        0, 1, &after, 0, nullptr, 0, nullptr);
    }

private:
    struct Functions {
        PFN_vkCreateBuffer create_buffer = nullptr;
        PFN_vkDestroyBuffer destroy_buffer = nullptr;
        PFN_vkGetBufferMemoryRequirements get_reqs = nullptr;
        PFN_vkAllocateMemory allocate_memory = nullptr;
        PFN_vkFreeMemory free_memory = nullptr;
        PFN_vkBindBufferMemory bind_buffer = nullptr;
        PFN_vkGetBufferDeviceAddress get_address = nullptr;
        PFN_vkCreateShaderModule create_shader = nullptr;
        PFN_vkDestroyShaderModule destroy_shader = nullptr;
        PFN_vkCreatePipelineLayout create_layout = nullptr;
        PFN_vkDestroyPipelineLayout destroy_layout = nullptr;
        PFN_vkCreateComputePipelines create_pipelines = nullptr;
        PFN_vkDestroyPipeline destroy_pipeline = nullptr;
        PFN_vkCmdPipelineBarrier cmd_barrier = nullptr;
        PFN_vkCmdBindPipeline cmd_bind_pipeline = nullptr;
        PFN_vkCmdPushConstants cmd_push_constants = nullptr;
        PFN_vkCmdDispatch cmd_dispatch = nullptr;
    };

    template<typename T> bool load_fn(T& out, const char* name) const {
        out = reinterpret_cast<T>(get_(device_, name));
        return out != nullptr;
    }

    bool make_pipeline() {
        if (pipeline_) return true;
        Functions f{};
        if (!load_fn(f.create_buffer, "vkCreateBuffer") || !load_fn(f.destroy_buffer, "vkDestroyBuffer") ||
            !load_fn(f.get_reqs, "vkGetBufferMemoryRequirements") ||
            !load_fn(f.allocate_memory, "vkAllocateMemory") || !load_fn(f.free_memory, "vkFreeMemory") ||
            !load_fn(f.bind_buffer, "vkBindBufferMemory") ||
            !load_fn(f.get_address, "vkGetBufferDeviceAddress") ||
            !load_fn(f.create_shader, "vkCreateShaderModule") ||
            !load_fn(f.destroy_shader, "vkDestroyShaderModule") ||
            !load_fn(f.create_layout, "vkCreatePipelineLayout") ||
            !load_fn(f.destroy_layout, "vkDestroyPipelineLayout") ||
            !load_fn(f.create_pipelines, "vkCreateComputePipelines") ||
            !load_fn(f.destroy_pipeline, "vkDestroyPipeline") ||
            !load_fn(f.cmd_barrier, "vkCmdPipelineBarrier") ||
            !load_fn(f.cmd_bind_pipeline, "vkCmdBindPipeline") ||
            !load_fn(f.cmd_push_constants, "vkCmdPushConstants") || !load_fn(f.cmd_dispatch, "vkCmdDispatch"))
            return false;
        VkShaderModule module = VK_NULL_HANDLE;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        VkPipeline pipeline = VK_NULL_HANDLE;
        VkShaderModuleCreateInfo shader{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        shader.codeSize = sizeof(indirect_dispatch_count_spirv);
        shader.pCode = indirect_dispatch_count_spirv;
        const aftermath::ShaderName aftermath_name("(runtime) indirect_dispatch_count");
        bool ok = f.create_shader(device_, &shader, nullptr, &module) == VK_SUCCESS;
        const VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, 48};
        VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layout_info.pushConstantRangeCount = 1;
        layout_info.pPushConstantRanges = &range;
        ok = ok && f.create_layout(device_, &layout_info, nullptr, &layout) == VK_SUCCESS;
        VkComputePipelineCreateInfo info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        info.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        info.stage.module = module;
        info.stage.pName = "main";
        info.layout = layout;
        ok = ok && f.create_pipelines(device_, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline) == VK_SUCCESS;
        if (module) f.destroy_shader(device_, module, nullptr);
        if (!ok) {
            if (pipeline) f.destroy_pipeline(device_, pipeline, nullptr);
            if (layout) f.destroy_layout(device_, layout, nullptr);
            return false;
        }
        vk_ = f;
        layout_ = layout;
        pipeline_ = pipeline;
        return true;
    }
    PFN_vkGetDeviceProcAddr get_;
    VkDevice device_;
    VkPhysicalDeviceMemoryProperties memory_;
    VkPhysicalDeviceLimits limits_;
    Functions vk_{};
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
};
}  // namespace leonida
