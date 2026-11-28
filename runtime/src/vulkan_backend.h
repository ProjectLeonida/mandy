#pragma once
#include "vulkan_gpu_labels.h"

// The execution side of the D3D12 layer, without COM: the per-device Vulkan context (resources, layout tracker,
// image views), queues that execute command lists, and timeline-semaphore fences. The COM objects in
// shims/d3d12_com.cpp are thin wrappers over these. Needs Vulkan 1.2 timeline semaphores; rectangle clears and
// enhanced barriers need the Vulkan 1.3 functions (dynamicRendering, synchronization2) to be enabled, as core
// features or through their KHR extensions on a 1.2 device (vulkan13_features.h).

#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include "binary_occlusion_spirv.h"
#include "buffer_va_index.h"
#include "image_memory.h"
#include "occlusion_sum_spirv.h"
#include "pipeline_workers.h"
#include "vulkan_aftermath.h"
#include "queue_holds.h"
#include "retire_tickets.h"
#include "statistics_sum_spirv.h"
#include "vulkan13_features.h"
#include "vulkan_depth24.h"
#include "vulkan_title_image_pool.h"
#include "vulkan_indirect_dispatch_count.h"
#include "vulkan_indirect_clamp.h"
#include "vulkan_scratch.h"
#include "vulkan_replay.h"
#include "queue_families.h"
#include "resource_tiling.h"
#include "pipeline_profile.h"
#include "submit_phases.h"
#include "pipeline_wait.h"
#include "title_va_alias.h"
#include "xdxr_build.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <chrono>
#include <unordered_set>
#include <shared_mutex>
#include <span>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace leonida {

// Which check of the queue's replay refused a batch ("ExecuteCommandLists failed: unsupported"), by source line,
// the first time each one does. The batch is then not submitted at all.
inline ReplayResult queue_unsupported(int line) {
    static std::atomic<bool> seen[4096];
    if (line < 0 || line >= 4096 || !seen[line].exchange(true, std::memory_order_relaxed))
        std::fprintf(stderr, "[rt] d3d12: queue replay refused a batch at vulkan_backend.h:%d\n", line);
    return ReplayResult::unsupported;
}

// A CPU-written Vulkan-address instance pointer array, kept until the submission that builds with it completes.
// Completed stages are pooled per queue and reused, so a per-frame TLAS build does not allocate memory each frame.
// The same stages carry the title-memory translation table of an XDXR TLAS build (VulkanQueue::stage_xdxr_table).
struct KhrInstanceStage {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceAddress address = 0;
    std::uint64_t size = 0;  // bytes available in buffer, so a larger later list can be refused for a new stage
    VkBufferUsageFlags usage = 0;  // a pooled stage is reused only for the same usage
};

// One per-queue query pool of VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR, retained on the
// CommandSlot until its submission completes and then recycled to the queue pool.
struct KhrAsQueryPool {
    VkQueryPool pool = VK_NULL_HANDLE;
    std::uint32_t capacity = 0;
    std::uint32_t used = 0;
};

// VK_EXT_debug_utils labels: an instance extension, so a device has them only when its instance enabled it.
struct DebugLabelFunctions {
    PFN_vkCmdInsertDebugUtilsLabelEXT cmd_insert = nullptr;
    PFN_vkCmdBeginDebugUtilsLabelEXT cmd_begin = nullptr;
    PFN_vkCmdEndDebugUtilsLabelEXT cmd_end = nullptr;
    PFN_vkQueueInsertDebugUtilsLabelEXT queue_insert = nullptr;
    PFN_vkQueueBeginDebugUtilsLabelEXT queue_begin = nullptr;
    PFN_vkQueueEndDebugUtilsLabelEXT queue_end = nullptr;

    bool complete() const { return cmd_insert && cmd_begin && cmd_end && queue_insert && queue_begin && queue_end; }
};

struct VulkanDeviceFunctions {
#define LEONIDA_VK_DEVICE_FUNCTIONS(X)                                                                         \
    X(vkCreateCommandPool) X(vkDestroyCommandPool) X(vkAllocateCommandBuffers) X(vkResetCommandBuffer)         \
    X(vkBeginCommandBuffer) X(vkEndCommandBuffer) X(vkQueueSubmit) X(vkQueueWaitIdle) X(vkCreateSemaphore)     \
    X(vkDestroySemaphore) X(vkGetSemaphoreCounterValue) X(vkWaitSemaphores) X(vkSignalSemaphore)               \
    X(vkCreateImageView) X(vkDestroyImageView) X(vkCmdPipelineBarrier) X(vkCmdPipelineBarrier2)                \
    X(vkCmdClearColorImage) X(vkCmdClearDepthStencilImage) X(vkCmdBeginRendering) X(vkCmdEndRendering)         \
    X(vkCmdClearAttachments) X(vkDeviceWaitIdle) X(vkCreateImage) X(vkDestroyImage)                         \
    X(vkGetImageMemoryRequirements) X(vkBindImageMemory) X(vkCreateBuffer) X(vkDestroyBuffer)                  \
    X(vkGetBufferMemoryRequirements) X(vkBindBufferMemory) X(vkAllocateMemory) X(vkFreeMemory) X(vkMapMemory)  \
    X(vkUnmapMemory) X(vkCmdCopyBuffer) X(vkCmdCopyBufferToImage) X(vkCmdCopyImageToBuffer) X(vkCmdCopyImage) X(vkCmdBlitImage)  \
    X(vkCmdResolveImage) X(vkCmdFillBuffer) X(vkCmdUpdateBuffer) X(vkCreateSampler) X(vkDestroySampler)        \
    X(vkCmdBeginConditionalRenderingEXT) X(vkCmdEndConditionalRenderingEXT) X(vkCreateQueryPool)               \
    X(vkDestroyQueryPool) X(vkCmdResetQueryPool) X(vkCmdBeginQuery) X(vkCmdEndQuery) X(vkCmdWriteTimestamp)    \
    X(vkCmdCopyQueryPoolResults) X(vkCmdDrawIndirect) X(vkCmdDrawIndexedIndirect) X(vkCmdDrawIndirectCount)    \
    X(vkCmdDrawIndexedIndirectCount) X(vkCmdDispatchIndirect) X(vkGetBufferDeviceAddress)                      \
    X(vkGetAccelerationStructureBuildSizesKHR) X(vkCreateAccelerationStructureKHR)                             \
    X(vkDestroyAccelerationStructureKHR) X(vkGetAccelerationStructureDeviceAddressKHR)                         \
    X(vkCmdBuildAccelerationStructuresKHR) X(vkCmdCopyAccelerationStructureKHR)                                \
    X(vkCmdWriteAccelerationStructuresPropertiesKHR)                                                           \
    X(vkCreateDescriptorSetLayout) X(vkDestroyDescriptorSetLayout) X(vkCreateDescriptorPool)                   \
    X(vkDestroyDescriptorPool) X(vkAllocateDescriptorSets) X(vkUpdateDescriptorSets)                          \
    X(vkCreatePipelineLayout) X(vkDestroyPipelineLayout) X(vkCreateShaderModule) X(vkDestroyShaderModule)      \
    X(vkCreateComputePipelines) X(vkDestroyPipeline) X(vkCmdBindPipeline) X(vkCmdBindDescriptorSets)            \
    X(vkCmdPushConstants) X(vkCmdDispatch)                                                                     \
    X(vkCmdBindVertexBuffers2) X(vkCmdBindIndexBuffer) X(vkCmdSetViewport) X(vkCmdSetScissor)                  \
    X(vkCmdSetViewportWithCount) X(vkCmdSetScissorWithCount)                                                   \
    X(vkCmdSetPrimitiveTopology) X(vkCmdSetPrimitiveRestartEnable) X(vkCmdSetStencilReference) X(vkCmdSetDepthBounds)                          \
    X(vkCmdSetBlendConstants) X(vkCmdDraw)                                                                              \
    X(vkCmdDrawIndexed) X(vkCreateGraphicsPipelines) X(vkQueueBindSparse) X(vkGetImageSparseMemoryRequirements)   \
    X(vkCreatePipelineCache) X(vkDestroyPipelineCache) X(vkGetPipelineCacheData) X(vkMergePipelineCaches)   \
    X(vkCreateEvent) X(vkDestroyEvent) X(vkResetEvent) X(vkCmdSetEvent) X(vkCmdWaitEvents) X(vkCmdDrawMeshTasksEXT) X(vkCmdDrawMeshTasksIndirectEXT) X(vkCmdDrawMeshTasksIndirectCountEXT) \
    X(vkCreateRenderPass2) X(vkDestroyRenderPass) X(vkCreateFramebuffer) X(vkDestroyFramebuffer)               \
    X(vkCmdBeginRenderPass) X(vkCmdEndRenderPass)                                                            \
    X(vkCmdBeginQueryIndexedEXT) X(vkCmdEndQueryIndexedEXT)
#define LEONIDA_VK_MEMBER(name) PFN_##name name = nullptr;
    LEONIDA_VK_DEVICE_FUNCTIONS(LEONIDA_VK_MEMBER)
#undef LEONIDA_VK_MEMBER

    // The Vulkan 1.3 entry points may be null on older devices, the extension ones (conditional rendering, mesh
    // shaders) when the extension is not enabled, and the query functions are optional (a list that uses queries
    // without them is unsupported). The compute and descriptor functions are needed for normalized or summed
    // occlusion results.
    bool load(PFN_vkGetDeviceProcAddr get, VkDevice device) {
#define LEONIDA_VK_LOAD(name) name = reinterpret_cast<PFN_##name>(leonida::device_proc_or_khr(get, device, #name));
        LEONIDA_VK_DEVICE_FUNCTIONS(LEONIDA_VK_LOAD)
#undef LEONIDA_VK_LOAD
        return vkCreateCommandPool && vkDestroyCommandPool && vkAllocateCommandBuffers && vkResetCommandBuffer &&
               vkBeginCommandBuffer && vkEndCommandBuffer && vkQueueSubmit && vkQueueWaitIdle && vkCreateSemaphore &&
               vkDestroySemaphore && vkGetSemaphoreCounterValue && vkWaitSemaphores && vkSignalSemaphore &&
               vkCreateImageView && vkDestroyImageView && vkCmdPipelineBarrier && vkCmdClearColorImage &&
               vkCmdClearDepthStencilImage && vkCmdClearAttachments && vkDeviceWaitIdle && vkCreateImage &&
               vkDestroyImage && vkGetImageMemoryRequirements && vkBindImageMemory && vkCreateBuffer &&
               vkDestroyBuffer && vkGetBufferMemoryRequirements && vkBindBufferMemory && vkAllocateMemory &&
               vkFreeMemory && vkMapMemory && vkUnmapMemory && vkCmdCopyBuffer && vkCmdCopyBufferToImage &&
               vkCmdCopyImageToBuffer && vkCmdCopyImage && vkCmdResolveImage && vkCmdFillBuffer &&
               vkCmdUpdateBuffer && vkCmdBindPipeline && vkCmdPushConstants && vkCmdDispatch &&
               vkCreateShaderModule && vkDestroyShaderModule && vkCreatePipelineLayout &&
               vkDestroyPipelineLayout && vkCreateComputePipelines && vkDestroyPipeline &&
               vkGetBufferDeviceAddress && vkCmdBindIndexBuffer && vkCmdSetViewport && vkCmdSetScissor &&
               vkCmdSetStencilReference && vkCmdSetBlendConstants && vkCmdDraw && vkCmdDrawIndexed &&
               vkCreateGraphicsPipelines;
    }

    // Graphics pipelines also need these Vulkan 1.3 entry points (dynamic rendering, extended dynamic state).
    bool graphics_complete() const {
        return vkCmdBeginRendering && vkCmdEndRendering && vkCmdBindVertexBuffers2 && vkCmdSetPrimitiveTopology &&
               vkCmdSetPrimitiveRestartEnable;
    }
    // VK_EXT_transform_feedback is enabled on the device: the indexed-query entry points are loaded, so an
    // SO_STATISTICS heap can be a real transform-feedback pool (D3D-302).
    bool transform_feedback = false;
};

// Both kinds of compute resolve use the same storage buffer, descriptor set and pipeline layout.
// The destination needs transfer usage only. Pipelines are created on first use.
class OcclusionResolve {
public:
    OcclusionResolve(const VulkanDeviceFunctions& vk, VkDevice device) : vk_(vk), device_(device) {}
    OcclusionResolve(const OcclusionResolve&) = delete;
    OcclusionResolve& operator=(const OcclusionResolve&) = delete;

    ~OcclusionResolve() {
        if (sum_pipeline_) vk_.vkDestroyPipeline(device_, sum_pipeline_, nullptr);
        if (binary_pipeline_) vk_.vkDestroyPipeline(device_, binary_pipeline_, nullptr);
        if (statistics_pipeline_) vk_.vkDestroyPipeline(device_, statistics_pipeline_, nullptr);
        if (layout_) vk_.vkDestroyPipelineLayout(device_, layout_, nullptr);
        if (pool_) vk_.vkDestroyDescriptorPool(device_, pool_, nullptr);
        if (set_layout_) vk_.vkDestroyDescriptorSetLayout(device_, set_layout_, nullptr);
        if (scratch_) vk_.vkDestroyBuffer(device_, scratch_, nullptr);
        if (memory_) vk_.vkFreeMemory(device_, memory_, nullptr);
    }

    bool initialize(const VkPhysicalDeviceMemoryProperties& memory) {
        if (!vk_.vkCreateDescriptorSetLayout || !vk_.vkDestroyDescriptorSetLayout ||
            !vk_.vkCreateDescriptorPool || !vk_.vkDestroyDescriptorPool || !vk_.vkAllocateDescriptorSets ||
            !vk_.vkUpdateDescriptorSets || !vk_.vkCreatePipelineLayout || !vk_.vkDestroyPipelineLayout ||
            !vk_.vkCreateShaderModule || !vk_.vkDestroyShaderModule || !vk_.vkCreateComputePipelines ||
            !vk_.vkDestroyPipeline || !vk_.vkCmdBindPipeline || !vk_.vkCmdBindDescriptorSets ||
            !vk_.vkCmdPushConstants || !vk_.vkCmdDispatch)
            return false;

        VkBufferCreateInfo buffer{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        buffer.size = scratch_size;
        buffer.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        if (vk_.vkCreateBuffer(device_, &buffer, nullptr, &scratch_) != VK_SUCCESS) return false;
        VkMemoryRequirements requirements{};
        vk_.vkGetBufferMemoryRequirements(device_, scratch_, &requirements);
        auto type = memory.memoryTypeCount;
        for (std::uint32_t i = 0; i < memory.memoryTypeCount; ++i) {
            if (!(requirements.memoryTypeBits & (1u << i))) continue;
            if (type == memory.memoryTypeCount) type = i;
            if (memory.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
                type = i;
                break;
            }
        }
        if (type == memory.memoryTypeCount) return false;
        VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocation.allocationSize = requirements.size;
        allocation.memoryTypeIndex = type;
        if (vk_.vkAllocateMemory(device_, &allocation, nullptr, &memory_) != VK_SUCCESS ||
            vk_.vkBindBufferMemory(device_, scratch_, memory_, 0) != VK_SUCCESS)
            return false;

        const VkDescriptorSetLayoutBinding binding{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                                                   VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo set_layout{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        set_layout.bindingCount = 1;
        set_layout.pBindings = &binding;
        if (vk_.vkCreateDescriptorSetLayout(device_, &set_layout, nullptr, &set_layout_) != VK_SUCCESS) return false;
        const VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
        VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pool.maxSets = 1;
        pool.poolSizeCount = 1;
        pool.pPoolSizes = &size;
        if (vk_.vkCreateDescriptorPool(device_, &pool, nullptr, &pool_) != VK_SUCCESS) return false;
        VkDescriptorSetAllocateInfo set{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        set.descriptorPool = pool_;
        set.descriptorSetCount = 1;
        set.pSetLayouts = &set_layout_;
        if (vk_.vkAllocateDescriptorSets(device_, &set, &set_) != VK_SUCCESS) return false;
        const VkDescriptorBufferInfo storage{scratch_, 0, scratch_size};
        VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write.dstSet = set_;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        write.pBufferInfo = &storage;
        vk_.vkUpdateDescriptorSets(device_, 1, &write, 0, nullptr);

        // The count, and for statistics_sum.comp whether the segments are mesh-pool results.
        const VkPushConstantRange count{VK_SHADER_STAGE_COMPUTE_BIT, 0, 2 * sizeof(std::uint32_t)};
        VkPipelineLayoutCreateInfo layout{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layout.setLayoutCount = 1;
        layout.pSetLayouts = &set_layout_;
        layout.pushConstantRangeCount = 1;
        layout.pPushConstantRanges = &count;
        return vk_.vkCreatePipelineLayout(device_, &layout, nullptr, &layout_) == VK_SUCCESS;
    }

    bool prepare(bool needs_sum, bool needs_binary, bool needs_statistics_sum = false,
                 bool needs_sum_binary = false) {
        // The composite (sum then normalize, D3D-252) uses both pipelines.
        needs_sum = needs_sum || needs_sum_binary;
        needs_binary = needs_binary || needs_sum_binary;
        return (!needs_sum || sum_pipeline_ || create_pipeline(occlusion_sum_spirv, sum_pipeline_)) &&
               (!needs_binary || binary_pipeline_ || create_pipeline(binary_occlusion_spirv, binary_pipeline_)) &&
               (!needs_statistics_sum || statistics_pipeline_ ||
                create_pipeline(statistics_sum_spirv, statistics_pipeline_));
    }

    // Adds `count` pipeline-statistics results into one D3D12_QUERY_DATA_PIPELINE_STATISTICS at dst + offset: the
    // segments of a statistics query that was split around helper dispatches (D3D-162) or mesh draws (D3D-291).
    // Segment i is in `mesh_queries` when mesh[i] is set (two counters: fragment and compute; see
    // statistics_sum.comp), otherwise in `queries` (all eleven). A mesh segment's clipping invocations come from
    // `primitives_queries`, the mesh-primitives count of that segment (D3D-MESH-CLIPPING-COUNTER); it is null when
    // the device has no meshShaderQueries, and the field then reads 0.
    void sum_statistics(VkCommandBuffer cb, VkQueryPool queries, VkQueryPool mesh_queries,
                        VkQueryPool primitives_queries, const std::uint8_t* mesh, std::uint32_t first,
                        std::uint32_t count, VkBuffer dst, VkDeviceSize offset) const {
        constexpr VkDeviceSize result = statistics_counters * word_size;
        constexpr VkDeviceSize mesh_slot = 7 * word_size;  // PSInvocations, where the mesh pool's first counter goes
        constexpr VkDeviceSize clipping_slot = 5 * word_size;  // CInvocations, where the mesh-primitives count goes
        begin_scratch_write(cb);
        vk_.vkCmdFillBuffer(cb, scratch_, 0, result, 0);
        bind(cb, statistics_pipeline_);
        while (count) {
            const bool in_mesh = mesh && mesh[first];
            std::uint32_t chunk = 1;
            while (chunk < std::min(count, statistics_capacity) && bool(mesh && mesh[first + chunk]) == in_mesh)
                ++chunk;
            if (in_mesh) {
                // The mesh pool holds only the fragment and compute counters, so the clipping field of a mesh
                // segment reads 0 unless the mesh-primitives count fills it (D3D-MESH-CLIPPING-COUNTER).
                vk_.vkCmdFillBuffer(cb, scratch_, result, chunk * result, 0);
                barrier(cb, transfer, transfer, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
            }
            vk_.vkCmdCopyQueryPoolResults(cb, in_mesh ? mesh_queries : queries, first, chunk, scratch_,
                                          result + (in_mesh ? mesh_slot : 0), result,
                                          VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
            if (in_mesh && primitives_queries)
                vk_.vkCmdCopyQueryPoolResults(cb, primitives_queries, first, chunk, scratch_, result + clipping_slot,
                                              result, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
            barrier(cb, transfer | compute, compute, VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                    shader_access);
            dispatch(cb, chunk, 1, in_mesh);
            first += chunk;
            count -= chunk;
            if (count) barrier(cb, compute, transfer, shader_access, VK_ACCESS_TRANSFER_WRITE_BIT);
        }
        copy_result(cb, dst, offset, result);
    }

    void sum(VkCommandBuffer cb, VkQueryPool queries, std::uint32_t first, std::uint32_t count,
             VkBuffer dst, VkDeviceSize offset) const {
        begin_scratch_write(cb);
        vk_.vkCmdFillBuffer(cb, scratch_, 0, word_size, 0);
        bind(cb, sum_pipeline_);
        while (count) {
            const auto chunk = std::min(count, sum_capacity);
            vk_.vkCmdCopyQueryPoolResults(cb, queries, first, chunk, scratch_, word_size, word_size,
                                          VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
            barrier(cb, transfer | compute, compute, VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                    shader_access);
            dispatch(cb, chunk, 1);
            first += chunk;
            count -= chunk;
            if (count) barrier(cb, compute, transfer, shader_access, VK_ACCESS_TRANSFER_WRITE_BIT);
        }
        copy_result(cb, dst, offset, word_size);
    }

    // Sums `count` binary-occlusion segments into one value and normalizes that total in place to 0 or 1: the
    // result of a binary query whose interval spans more than one physical segment (overlapping queries, D3D-252).
    void sum_binary(VkCommandBuffer cb, VkQueryPool queries, std::uint32_t first, std::uint32_t count,
                    VkBuffer dst, VkDeviceSize offset) const {
        begin_scratch_write(cb);
        vk_.vkCmdFillBuffer(cb, scratch_, 0, word_size, 0);
        bind(cb, sum_pipeline_);
        auto tail = first, left = count;
        while (left) {
            const auto chunk = std::min(left, sum_capacity);
            vk_.vkCmdCopyQueryPoolResults(cb, queries, tail, chunk, scratch_, word_size, word_size,
                                          VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
            barrier(cb, transfer | compute, compute, VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                    shader_access);
            dispatch(cb, chunk, 1);
            tail += chunk;
            left -= chunk;
            if (left) barrier(cb, compute, transfer, shader_access, VK_ACCESS_TRANSFER_WRITE_BIT);
        }
        barrier(cb, compute, compute, VK_ACCESS_SHADER_WRITE_BIT, shader_access);
        bind(cb, binary_pipeline_);
        dispatch(cb, 1, 1);
        copy_result(cb, dst, offset, word_size);
    }

    void normalize(VkCommandBuffer cb, VkQueryPool queries, std::uint32_t first, std::uint32_t count,
                   VkBuffer dst, VkDeviceSize offset) const {
        bind(cb, binary_pipeline_);
        while (count) {
            const auto chunk = std::min(count, binary_capacity);
            const VkDeviceSize bytes = VkDeviceSize(chunk) * word_size;
            begin_scratch_write(cb);
            vk_.vkCmdCopyQueryPoolResults(cb, queries, first, chunk, scratch_, 0, word_size,
                                          VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
            barrier(cb, transfer, compute, VK_ACCESS_TRANSFER_WRITE_BIT, shader_access);
            dispatch(cb, chunk, (chunk + binary_group_size - 1) / binary_group_size);
            copy_result(cb, dst, offset, bytes);
            first += chunk;
            count -= chunk;
            offset += bytes;
        }
    }

private:
    bool create_pipeline(std::span<const std::uint32_t> code, VkPipeline& out) const {
        VkShaderModuleCreateInfo module{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        module.codeSize = code.size_bytes();
        module.pCode = code.data();
        VkShaderModule shader = VK_NULL_HANDLE;
        const aftermath::ShaderName aftermath_name("(runtime) occlusion resolve");
        if (vk_.vkCreateShaderModule(device_, &module, nullptr, &shader) != VK_SUCCESS) return false;
        VkComputePipelineCreateInfo pipeline{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pipeline.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                          VK_SHADER_STAGE_COMPUTE_BIT, shader, "main", nullptr};
        pipeline.layout = layout_;
        VkPipeline handle = VK_NULL_HANDLE;
        const VkResult result = vk_.vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipeline, nullptr, &handle);
        vk_.vkDestroyShaderModule(device_, shader, nullptr);
        if (result == VK_SUCCESS) out = handle;
        else if (handle) vk_.vkDestroyPipeline(device_, handle, nullptr);
        return result == VK_SUCCESS;
    }

    void bind(VkCommandBuffer cb, VkPipeline pipeline) const {
        vk_.vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vk_.vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layout_, 0, 1, &set_, 0, nullptr);
    }

    void dispatch(VkCommandBuffer cb, std::uint32_t chunk, std::uint32_t groups, bool mesh = false) const {
        const std::uint32_t range[2] = {chunk, mesh ? 1u : 0u};
        vk_.vkCmdPushConstants(cb, layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(range), range);
        vk_.vkCmdDispatch(cb, groups, 1, 1);
    }

    void begin_scratch_write(VkCommandBuffer cb) const {
        // A sum or a normalization in an earlier submission may still use this buffer.
        barrier(cb, transfer | compute, transfer,
                VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | shader_access,
                VK_ACCESS_TRANSFER_WRITE_BIT);
    }

    void copy_result(VkCommandBuffer cb, VkBuffer dst, VkDeviceSize offset, VkDeviceSize bytes) const {
        barrier(cb, compute, transfer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
        const VkBufferCopy copy{0, offset, bytes};
        vk_.vkCmdCopyBuffer(cb, scratch_, dst, 1, &copy);
    }

    void barrier(VkCommandBuffer cb, VkPipelineStageFlags src, VkPipelineStageFlags dst,
                 VkAccessFlags before, VkAccessFlags after) const {
        VkBufferMemoryBarrier memory{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        memory.srcAccessMask = before;
        memory.dstAccessMask = after;
        memory.srcQueueFamilyIndex = memory.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        memory.buffer = scratch_;
        memory.size = scratch_size;
        vk_.vkCmdPipelineBarrier(cb, src, dst, 0, 0, nullptr, 1, &memory, 0, nullptr);
    }

    static constexpr auto transfer = VK_PIPELINE_STAGE_TRANSFER_BIT;
    static constexpr auto compute = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    static constexpr VkAccessFlags shader_access = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    static constexpr std::uint32_t sum_capacity = 256;  // one more word holds the running sum
    static constexpr std::uint32_t binary_capacity = 1024;
    static constexpr std::uint32_t binary_group_size = 64;  // local_size_x in binary_occlusion.comp
    static constexpr std::uint32_t statistics_counters = 11;  // local_size_x in statistics_sum.comp
    static constexpr std::uint32_t statistics_capacity = 64;  // segments per pass; 11 more words hold the sums
    static constexpr VkDeviceSize word_size = sizeof(std::uint64_t);
    static constexpr VkDeviceSize scratch_size =
        std::max({sum_capacity + 1, binary_capacity, (statistics_capacity + 1) * statistics_counters}) * word_size;
    const VulkanDeviceFunctions& vk_;
    VkDevice device_;
    VkBuffer scratch_ = VK_NULL_HANDLE;
    VkDeviceMemory memory_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout_ = VK_NULL_HANDLE;
    VkDescriptorPool pool_ = VK_NULL_HANDLE;
    VkDescriptorSet set_ = VK_NULL_HANDLE;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    VkPipeline sum_pipeline_ = VK_NULL_HANDLE;
    VkPipeline binary_pipeline_ = VK_NULL_HANDLE;
    VkPipeline statistics_pipeline_ = VK_NULL_HANDLE;
};

struct SparseResource;
using ImageMemory = ImageMemoryPool<VkDeviceMemory>;

// Zero-allocation O(1) Ring Buffer deferred deletion queue tied to the device timeline semaphore.
class DeferredDeletionQueue {
public:
    enum class Kind : std::uint8_t {
        bindless_slot,
        image_view,
        image,
        sparse_resource,
        pooled_title_image,
        buffer,
        device_memory,
        image_memory,  // an ImageMemory::Range*, deleted: its bytes go back to their block
        acceleration_structure,
        framebuffer,
    };

    struct Item {
        std::uint64_t ticket = 0;
        std::uint64_t handle = 0;  // for bindless_slot: (uint64_t(dimension) << 32) | index
        Kind kind = Kind::image_view;
    };

    using BindlessReleaseFn = void (*)(void* user, std::uint32_t dimension, std::uint32_t index);

    explicit DeferredDeletionQueue(std::size_t initial_capacity = 256) : pending_(next_pow2(initial_capacity)) {}

    void set_bindless_release(BindlessReleaseFn fn, void* user) {
        std::lock_guard lock(mutex_);
        bindless_release_fn_ = fn;
        bindless_release_user_ = user;
    }

    void push(std::uint64_t retire_ticket, Kind kind, std::uint64_t handle) {
        if (!handle && kind != Kind::bindless_slot) return;
        std::lock_guard lock(mutex_);
        if (count_ == pending_.size()) {
            const std::size_t old_cap = pending_.size();
            std::vector<Item> grown(old_cap * 2);
            const std::size_t first = old_cap - head_;
            std::memcpy(grown.data(), pending_.data() + head_, first * sizeof(Item));
            if (head_ > 0) std::memcpy(grown.data() + first, pending_.data(), head_ * sizeof(Item));
            pending_ = std::move(grown);
            head_ = 0;
            tail_ = count_;
        }
        pending_[tail_] = {retire_ticket, handle, kind};
        tail_ = (tail_ + 1) & (pending_.size() - 1);
        ++count_;
        head_ticket_.store(pending_[head_].ticket, std::memory_order_release);
        pending_count_.store(count_, std::memory_order_release);
    }

    bool has_pending() const noexcept { return pending_count_.load(std::memory_order_acquire) != 0; }

    bool has_ready(std::uint64_t current_ticket) const noexcept {
        return pending_count_.load(std::memory_order_acquire) != 0 &&
               head_ticket_.load(std::memory_order_acquire) <= current_ticket;
    }

    // Drains all items with ticket <= current_ticket in O(k), invoking vkDestroy* outside the lock.
    // Uses a stack-local batch vector so concurrent drain() calls from multiple queues never race.
    void drain(std::uint64_t current_ticket, VkDevice device, const VulkanDeviceFunctions& vk) {
        if (!has_ready(current_ticket)) return;
        BindlessReleaseFn release_fn = nullptr;
        void* release_user = nullptr;
        std::vector<Item> to_destroy;
        {
            std::lock_guard lock(mutex_);
            if (count_ == 0 || pending_[head_].ticket > current_ticket) return;
            release_fn = bindless_release_fn_;
            release_user = bindless_release_user_;
            const std::size_t mask = pending_.size() - 1;
            while (count_ > 0 && pending_[head_].ticket <= current_ticket) {
                to_destroy.push_back(pending_[head_]);
                head_ = (head_ + 1) & mask;
                --count_;
            }
            if (count_ == 0) {
                head_ = 0;
                tail_ = 0;
                head_ticket_.store(UINT64_MAX, std::memory_order_release);
            } else {
                head_ticket_.store(pending_[head_].ticket, std::memory_order_release);
            }
            pending_count_.store(count_, std::memory_order_release);
        }

        for (const Item& item : to_destroy) {
            switch (item.kind) {
            case Kind::bindless_slot:
                if (release_fn) {
                    release_fn(release_user, static_cast<std::uint32_t>(item.handle >> 32),
                               static_cast<std::uint32_t>(item.handle & 0xFFFFFFFFu));
                }
                break;
            case Kind::image_view:
                vk.vkDestroyImageView(device, reinterpret_cast<VkImageView>(item.handle), nullptr);
                break;
            case Kind::image:
                vk.vkDestroyImage(device, reinterpret_cast<VkImage>(item.handle), nullptr);
                break;
            case Kind::pooled_title_image:
                delete reinterpret_cast<PooledTitleImage*>(item.handle);
                break;
            case Kind::sparse_resource:
                // Releases the resource and its page backing together after outstanding submissions.
                delete reinterpret_cast<std::shared_ptr<SparseResource>*>(item.handle);
                break;
            case Kind::buffer:
                vk.vkDestroyBuffer(device, reinterpret_cast<VkBuffer>(item.handle), nullptr);
                break;
            case Kind::device_memory:
                vk.vkFreeMemory(device, reinterpret_cast<VkDeviceMemory>(item.handle), nullptr);
                break;
            case Kind::image_memory:
                delete reinterpret_cast<ImageMemory::Range*>(item.handle);
                break;
            case Kind::acceleration_structure:
                if (vk.vkDestroyAccelerationStructureKHR)
                    vk.vkDestroyAccelerationStructureKHR(
                        device, reinterpret_cast<VkAccelerationStructureKHR>(item.handle), nullptr);
                break;
            case Kind::framebuffer:
                vk.vkDestroyFramebuffer(device, reinterpret_cast<VkFramebuffer>(item.handle), nullptr);
                break;
            }
        }
    }

private:
    static std::size_t next_pow2(std::size_t n) {
        std::size_t p = 16;
        while (p < n) p <<= 1;
        return p;
    }

    std::mutex mutex_;
    std::vector<Item> pending_;
    std::size_t head_ = 0;
    std::size_t tail_ = 0;
    std::size_t count_ = 0;
    std::atomic<std::size_t> pending_count_{0};
    std::atomic<std::uint64_t> head_ticket_{UINT64_MAX};
    BindlessReleaseFn bindless_release_fn_ = nullptr;
    void* bindless_release_user_ = nullptr;
};

// Sets a device's lost flag and says so once, on stderr, at the moment it is first seen. From then on every fence of
// the device reads UINT64_MAX and device calls return DXGI_ERROR_DEVICE_REMOVED, so a title takes its own
// device-removed path (on the retail boot an int3 at game_scarlett_master.exe+0x2474606, reached from its frame wait
// reading a fence's completed value as ~0); without this line that later stop is the first sign of the loss.
// True when this call set the flag. device_lost_reports counts the lines said in this process (for tests).
inline std::atomic<unsigned> device_lost_reports{0};
inline bool set_device_lost(std::atomic<bool>& flag, const char* where) noexcept {
    if (flag.exchange(true, std::memory_order_acq_rel)) return false;
    device_lost_reports.fetch_add(1, std::memory_order_relaxed);
    std::fprintf(stderr,
                 "[rt] vulkan: the device was lost (%s); its fences now read UINT64_MAX and device calls return "
                 "DXGI_ERROR_DEVICE_REMOVED\n",
                 where);
    return true;
}
// LEONIDA_GPU_LATENCY=1 (diagnostic): for each queue kind, how long the GPU took from a submission's vkQueueSubmit to
// the dispatcher seeing it done, and how much of the wall time the queue had work (in-order queue: a submission is
// busy from the later of its submit and the previous one's completion to its own). One line per kind every 10 s.
namespace gpu_latency {
inline bool enabled() {
    static const bool on = [] {
        const char* v = std::getenv("LEONIDA_GPU_LATENCY");
        return v && std::strcmp(v, "1") == 0;
    }();
    return on;
}
struct Stats {
    std::mutex mutex;
    std::uint64_t count = 0;
    double sum_ms = 0, max_ms = 0, busy_ms = 0;
    std::chrono::steady_clock::time_point last_done{}, since = std::chrono::steady_clock::now();
};
inline Stats& stats(int kind) {
    static Stats s[3];
    return s[kind < 0 || kind > 2 ? 0 : kind];
}
inline void held_tick(int kind);  // below: the held-operations line, due on the same 10 s

inline void done(int kind, std::chrono::steady_clock::time_point submitted) {
    using clock = std::chrono::steady_clock;
    const auto now = clock::now();
    const auto ms = [](clock::duration d) { return std::chrono::duration<double, std::milli>(d).count(); };
    Stats& s = stats(kind);
    std::lock_guard lock(s.mutex);
    const double latency = ms(now - submitted);
    ++s.count;
    s.sum_ms += latency;
    s.max_ms = (std::max)(s.max_ms, latency);
    s.busy_ms += ms(now - (std::max)(submitted, s.last_done));
    s.last_done = now;
    if (now - s.since >= std::chrono::seconds(10)) {
        const double wall = ms(now - s.since);
        std::fprintf(stderr, "[rt] gpu latency (%s): %llu submissions in %.0f s, submit to done %.2f ms on average, "
                             "%.1f ms the longest; the queue had work %.1f%% of the time\n",
                     kind == 0 ? "direct" : kind == 1 ? "compute" : "copy",
                     static_cast<unsigned long long>(s.count), wall / 1000.0, s.sum_ms / double(s.count), s.max_ms,
                     100.0 * s.busy_ms / wall);
        s.count = 0;
        s.sum_ms = s.max_ms = s.busy_ms = 0;
        s.since = now;
    }
    held_tick(kind);
}
// A held queue operation (Wait before Signal), released: how long it waited on the CPU before it could run.
// A held queue operation (Wait before Signal), released: how long it waited on the CPU before it could run, and for
// the longest one of the window the fence (semaphore handle) and value it waited for, so the slow signaller can be
// named. The line is due every 10 s: held() prints it, and so does done() (held_tick), so a window with nothing
// released still reports.
struct HeldStats {
    std::mutex mutex;
    std::uint64_t count[3] = {};
    double sum_ms = 0, max_ms = 0;
    std::uint64_t max_fence = 0, max_value = 0;
    int max_op = 0;
    std::chrono::steady_clock::time_point since = std::chrono::steady_clock::now();
};
inline HeldStats& held_stats(int kind) {
    static HeldStats h[3];
    return h[kind < 0 || kind > 2 ? 0 : kind];
}
// held_stats(kind).mutex held.
inline void held_report_locked(int kind, HeldStats& s, std::chrono::steady_clock::time_point now) {
    if (now - s.since < std::chrono::seconds(10)) return;
    const std::uint64_t all = s.count[0] + s.count[1] + s.count[2];
    std::fprintf(stderr, "[rt] gpu latency (%s): %llu held operations released in %.0f s (%llu batches, %llu "
                         "signals, %llu markers), held %.2f ms on average, %.1f ms the longest",
                 kind == 0 ? "direct" : kind == 1 ? "compute" : "copy", static_cast<unsigned long long>(all),
                 std::chrono::duration<double>(now - s.since).count(), static_cast<unsigned long long>(s.count[0]),
                 static_cast<unsigned long long>(s.count[1]), static_cast<unsigned long long>(s.count[2]),
                 all ? s.sum_ms / double(all) : 0.0, s.max_ms);
    if (all && s.max_fence)
        std::fprintf(stderr, " (a %s waiting for fence 0x%llx value %llu)",
                     s.max_op == 0 ? "batch" : s.max_op == 1 ? "Signal" : "marker",
                     static_cast<unsigned long long>(s.max_fence), static_cast<unsigned long long>(s.max_value));
    std::fputc('\n', stderr);
    s.count[0] = s.count[1] = s.count[2] = 0;
    s.sum_ms = s.max_ms = 0;
    s.max_fence = s.max_value = 0;
    s.since = now;
}
inline void held(int kind, int op, std::chrono::steady_clock::time_point at, std::uint64_t fence = 0,
                 std::uint64_t value = 0) {
    using clock = std::chrono::steady_clock;
    HeldStats& s = held_stats(kind);
    const auto now = clock::now();
    const double ms = std::chrono::duration<double, std::milli>(now - at).count();
    std::lock_guard lock(s.mutex);
    ++s.count[op < 0 || op > 2 ? 0 : op];
    s.sum_ms += ms;
    if (ms > s.max_ms) {
        s.max_ms = ms;
        s.max_fence = fence;
        s.max_value = value;
        s.max_op = op;
    }
    held_report_locked(kind, s, now);
}
inline void held_tick(int kind) {
    HeldStats& s = held_stats(kind);
    std::lock_guard lock(s.mutex);
    held_report_locked(kind, s, std::chrono::steady_clock::now());
}
}  // namespace gpu_latency

// Centralized single-thread fence dispatcher using Vulkan 1.2 vkWaitSemaphores(VK_SEMAPHORE_WAIT_ANY_BIT).
// Eliminates per-fence OS threads and 10ms polling: a synthetic timeline semaphore (wakeup_sem_) at index 0
// wakes vkWaitSemaphores immediately whenever a new wait, a released fence, or shutdown arrives while waiting_ == true.
class FenceWaitDispatcher {
public:
    struct Entry {
        VkSemaphore semaphore = VK_NULL_HANDLE;
        std::uint64_t value = 0;
        void* event_handle = nullptr;
        std::function<void()> callback;
    };

    // A dispatch runs outside mutex_. Each entry in a ready batch gets its own slot: a callback may re-entrantly
    // call cancel() for a later entry in the same batch, which marks that entry cancelled so it is skipped before
    // it can use the semaphore. A callback running on another thread must still be waited for.
    // A cancelled entry still in pending_ is dispatched by cancel() itself (callback and bookkeeping both run);
    // one already in a ready batch is skipped and only its event is signalled. The held Signal op's count is
    // therefore not decremented for a skipped batch entry. That is safe because only ~Fence cancels a wait, but
    // keep it in mind if cancel() gains other callers.
    struct DispatchSlot {
        VkSemaphore semaphore = VK_NULL_HANDLE;
        std::thread::id owner;
        bool active = false;     // guarded by mutex_
        bool cancelled = false;  // guarded by mutex_
    };

    using SignalEventFn = void (*)(void* event_handle);

    // Longest single vkWaitSemaphores call in run(): bounds how long the thread can sit in an uninterruptible wait.
    static constexpr std::uint64_t kWaitSliceNs = 100'000'000;  // 100 ms

    FenceWaitDispatcher(VkDevice device, const VulkanDeviceFunctions& vk, std::atomic<bool>& device_lost)
        : device_(device), vk_(vk), device_lost_(device_lost) {
        VkSemaphoreTypeCreateInfo type{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
        type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        type.initialValue = 0;
        VkSemaphoreCreateInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &type};
        if (vk_.vkCreateSemaphore(device_, &info, nullptr, &wakeup_sem_) != VK_SUCCESS)
            wakeup_sem_ = VK_NULL_HANDLE;
        semaphores_.reserve(64);
        values_.reserve(64);
    }

    ~FenceWaitDispatcher() {
        stop();
        if (wakeup_sem_) vk_.vkDestroySemaphore(device_, wakeup_sem_, nullptr);
    }

    FenceWaitDispatcher(const FenceWaitDispatcher&) = delete;
    FenceWaitDispatcher& operator=(const FenceWaitDispatcher&) = delete;

    void set_device_fault(DeviceFaultDiagnostics* diagnostics) { diagnostics_ = diagnostics; }

    void set_signal_event_fn(SignalEventFn fn) {
        std::lock_guard lock(mutex_);
        signal_event_ = fn;
    }

    void enqueue(VkSemaphore sem, std::uint64_t value, void* event_handle) {
        if (!sem || !event_handle) return;
        SignalEventFn immediate_fn = nullptr;
        {
            std::lock_guard lock(mutex_);
            if (stop_ || device_lost_.load(std::memory_order_acquire)) {
                immediate_fn = signal_event_;
            } else {
                pending_.push_back({sem, value, event_handle, {}});
                pending_count_.store(pending_.size(), std::memory_order_release);
                if (!thread_.joinable() && wakeup_sem_) thread_ = std::thread([this] { run(); });
                signal_wakeup_locked();
            }
        }
        if (immediate_fn) {
            immediate_fn(event_handle);
            return;
        }
        cv_.notify_one();
    }

    void enqueue_callback(VkSemaphore sem, std::uint64_t value, std::function<void()> cb) {
        if (!sem || !cb) return;
        std::function<void()> immediate_cb;
        {
            std::lock_guard lock(mutex_);
            if (stop_ || device_lost_.load(std::memory_order_acquire)) {
                immediate_cb = std::move(cb);
            } else {
                pending_.push_back({sem, value, nullptr, std::move(cb)});
                pending_count_.store(pending_.size(), std::memory_order_release);
                if (!thread_.joinable() && wakeup_sem_) thread_ = std::thread([this] { run(); });
                signal_wakeup_locked();
            }
        }
        if (immediate_cb) {
            immediate_cb();
            return;
        }
        cv_.notify_one();
    }

    // Returns the largest pending waited value <= `value` for `sem`, or 0 when no CPU waiter awaits <= `value`.
    std::uint64_t max_wait_value_le(VkSemaphore sem, std::uint64_t value) const {
        if (!sem || pending_count_.load(std::memory_order_acquire) == 0) return 0;
        std::lock_guard lock(mutex_);
        std::uint64_t best = 0;
        for (const Entry& e : pending_) {
            if (e.semaphore == sem && e.value <= value && e.value > best) best = e.value;
        }
        return best;
    }

    // Wakes run() when another path marks the device lost so pending waits complete and stop()/cancel() never hang.
    void notify_device_lost() {
        {
            std::lock_guard lock(mutex_);
            signal_wakeup_locked();
        }
        cv_.notify_all();
    }

    // Completes the pending waits for `sem` (a released D3D12 fence counts as reached, as on the PC runtime) and
    // blocks until both a vkWaitSemaphores on `sem` and any callback already dispatched for it have finished, before
    // the caller destroys the VkSemaphore.
    void cancel(VkSemaphore sem) {
        if (!sem) return;
        std::vector<Entry> cancelled;
        SignalEventFn fn = nullptr;
        {
            std::unique_lock lock(mutex_);
            fn = signal_event_;
            std::size_t write = 0;
            for (std::size_t i = 0; i < pending_.size(); ++i) {
                if (pending_[i].semaphore == sem) {
                    cancelled.push_back(std::move(pending_[i]));
                } else {
                    if (write != i) pending_[write] = std::move(pending_[i]);
                    ++write;
                }
            }
            pending_.resize(write);
            pending_count_.store(pending_.size(), std::memory_order_release);
            // Entries already taken out of pending_ but not yet started belong to a live ready batch. Mark them
            // cancelled so dispatch skips them; this is how a re-entrant cancel() from another callback in the
            // same batch is completed without waiting on the callback that is already on this thread's stack.
            for (const auto& slot : dispatching_) {
                if (slot->semaphore == sem && !slot->active) slot->cancelled = true;
            }
            if (thread_.joinable() && (waiting_ || callback_on_other_thread(sem))) {
                const std::uint64_t target_epoch = epoch_ + 1;
                signal_wakeup_locked();
                epoch_cv_.wait(lock, [this, sem, target_epoch] {
                    return (!waiting_ || epoch_ >= target_epoch) && !callback_on_other_thread(sem);
                });
            }
        }
        dispatch_entries(cancelled, fn);
    }

    void stop() {
        {
            std::lock_guard lock(mutex_);
            if (stop_) return;
            stop_ = true;
            signal_wakeup_locked();
        }
        cv_.notify_all();
        if (thread_.joinable()) thread_.join();
    }

private:
    static void dispatch_entries(std::vector<Entry>& entries, SignalEventFn signal_event) {
        for (Entry& e : entries) {
            if (e.callback) e.callback();
            else if (signal_event && e.event_handle) signal_event(e.event_handle);
        }
    }

    // True only for a callback already running on another thread. A slot that has not started yet is cancelled by
    // cancel() and skipped, so it never uses a destroyed semaphore and must not hold up the cancelling caller.
    bool callback_on_other_thread(VkSemaphore sem) const {
        const std::thread::id me = std::this_thread::get_id();
        return std::any_of(dispatching_.begin(), dispatching_.end(), [&](const auto& d) {
            return d->semaphore == sem && d->active && d->owner != me;
        });
    }

    // mutex_ is held on entry and again on return. Recording every entry before unlocking closes the window where
    // run() has taken them out of pending_ but has not started the callback, yet cancel() can return and let the
    // caller destroy the semaphore. Each entry has its own slot so a cancel() for a later entry in the same batch
    // skips that entry instead of being mistaken for the callback already on the cancelling thread's stack.
    void dispatch_outside_lock(std::unique_lock<std::mutex>& lock, std::vector<Entry>& entries, SignalEventFn fn) {
        const std::thread::id me = std::this_thread::get_id();
        std::vector<std::shared_ptr<DispatchSlot>> slots;
        slots.reserve(entries.size());
        for (const Entry& e : entries) {
            auto slot = std::make_shared<DispatchSlot>();
            slot->semaphore = e.semaphore;
            slot->owner = me;
            slots.push_back(slot);
            dispatching_.push_back(slot);
        }
        lock.unlock();
        const submit_phases::WorkerScope busy(submit_phases::worker_fence_thread);  // the "submit frame" line
        for (std::size_t i = 0; i < entries.size(); ++i) {
            Entry& e = entries[i];
            DispatchSlot& slot = *slots[i];
            bool skip = false;
            {
                std::lock_guard guard(mutex_);
                skip = slot.cancelled;
                if (!skip) slot.active = true;
            }
            if (skip) {
                // A re-entrant cancel() removed this wait; still complete its event so no waiter is left behind.
                if (fn && e.event_handle) fn(e.event_handle);
                continue;
            }
            if (e.callback) e.callback();
            else if (fn && e.event_handle) fn(e.event_handle);
            {
                std::lock_guard guard(mutex_);
                slot.active = false;
                const auto it = std::find(dispatching_.begin(), dispatching_.end(), slots[i]);
                if (it != dispatching_.end()) dispatching_.erase(it);
            }
            epoch_cv_.notify_all();
        }
        lock.lock();
        for (const auto& slot : slots) {
            const auto it = std::find(dispatching_.begin(), dispatching_.end(), slot);
            if (it != dispatching_.end()) dispatching_.erase(it);
        }
        epoch_cv_.notify_all();
    }

    // Only signals wakeup_sem_ when run() is actively waiting (waiting_ == true) on wakeup_val_ + 1,
    // eliminating any lost-wakeup race between snapshot construction and vkWaitSemaphores. Keep signaling even if
    // device_lost_ is already set so a blocked vkWaitSemaphores wakes cleanly for cancel()/stop().
    void signal_wakeup_locked() {
        if (!waiting_ || !wakeup_sem_) return;
        const std::uint64_t next = ++wakeup_val_;
        VkSemaphoreSignalInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO};
        si.semaphore = wakeup_sem_;
        si.value = next;
        DeviceFaultDiagnostics::observe(diagnostics_, vk_.vkSignalSemaphore(device_, &si),
                                       "vkSignalSemaphore (fence wakeup)");
    }

    void run() {
        std::unique_lock lock(mutex_);
        while (true) {
            cv_.wait(lock, [this] { return stop_ || !pending_.empty() || device_lost_.load(std::memory_order_acquire); });
            if (device_lost_.load(std::memory_order_acquire)) {
                std::vector<Entry> drained = std::move(pending_);
                pending_.clear();
                pending_count_.store(0, std::memory_order_release);
                dispatch_outside_lock(lock, drained, signal_event_);
                return;
            }
            if (stop_) return;

            semaphores_.clear();
            values_.clear();
            semaphores_.push_back(wakeup_sem_);
            values_.push_back(wakeup_val_ + 1);
            for (const Entry& e : pending_) {
                semaphores_.push_back(e.semaphore);
                values_.push_back(e.value);
            }

            waiting_ = true;
            lock.unlock();

            VkSemaphoreWaitInfo wi{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
            wi.flags = VK_SEMAPHORE_WAIT_ANY_BIT;
            wi.semaphoreCount = static_cast<std::uint32_t>(semaphores_.size());
            wi.pSemaphores = semaphores_.data();
            wi.pValues = values_.data();
            // A bounded slice, not UINT64_MAX. Under GPU paravirtualization (measured in Windows Sandbox on an
            // RTX 5060 Ti) this wait is a kernel wait the OS cannot interrupt: a process that exits while a wait
            // is pending (d3d12_queue_wait_timeout's _Exit, or quitting with a fence outstanding) never finishes
            // exiting, and TerminateProcess cannot end it either. A timeout returns here and the loop re-arms the
            // same wait; completions still wake it at once through the ANY wait.
            const VkResult res = DeviceFaultDiagnostics::observe(
                diagnostics_, vk_.vkWaitSemaphores(device_, &wi, kWaitSliceNs), "vkWaitSemaphores (fence dispatcher)");

            lock.lock();
            waiting_ = false;
            ++epoch_;
            epoch_cv_.notify_all();

            if (res == VK_ERROR_DEVICE_LOST || device_lost_.load(std::memory_order_acquire)) {
                set_device_lost(device_lost_, "vkWaitSemaphores in the fence dispatcher returned VK_ERROR_DEVICE_LOST");
                std::vector<Entry> drained = std::move(pending_);
                pending_.clear();
                pending_count_.store(0, std::memory_order_release);
                dispatch_outside_lock(lock, drained, signal_event_);
                return;
            }
            if (stop_) return;

            std::vector<Entry> ready;
            bool lost = false;
            std::size_t write = 0;
            for (std::size_t i = 0; i < pending_.size(); ++i) {
                std::uint64_t done = 0;
                const VkResult cr = DeviceFaultDiagnostics::observe(
                    diagnostics_, vk_.vkGetSemaphoreCounterValue(device_, pending_[i].semaphore, &done),
                    "vkGetSemaphoreCounterValue (fence dispatcher)");
                if (cr == VK_ERROR_DEVICE_LOST) {
                    lost = true;
                    for (std::size_t j = i; j < pending_.size(); ++j) ready.push_back(std::move(pending_[j]));
                    break;
                }
                if (cr == VK_SUCCESS && done >= pending_[i].value) {
                    ready.push_back(std::move(pending_[i]));
                } else {
                    if (write != i) pending_[write] = std::move(pending_[i]);
                    ++write;
                }
            }
            if (lost) {
                pending_.resize(write);
                for (Entry& e : pending_) ready.push_back(std::move(e));
                pending_.clear();
                pending_count_.store(0, std::memory_order_release);
                set_device_lost(device_lost_,
                                "vkGetSemaphoreCounterValue in the fence dispatcher returned VK_ERROR_DEVICE_LOST");
                dispatch_outside_lock(lock, ready, signal_event_);
                return;
            }
            pending_.resize(write);
            pending_count_.store(pending_.size(), std::memory_order_release);
            if (!ready.empty()) {
                dispatch_outside_lock(lock, ready, signal_event_);
                if (stop_) return;
            }
        }
    }

    VkDevice device_;
    const VulkanDeviceFunctions& vk_;
    std::atomic<bool>& device_lost_;
    DeviceFaultDiagnostics* diagnostics_ = nullptr;  // configured before the dispatcher starts
    SignalEventFn signal_event_ = nullptr;
    VkSemaphore wakeup_sem_ = VK_NULL_HANDLE;
    std::uint64_t wakeup_val_ = 0;

    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::condition_variable epoch_cv_;
    std::vector<Entry> pending_;
    std::atomic<std::size_t> pending_count_{0};
    std::vector<std::shared_ptr<DispatchSlot>> dispatching_;  // ready-batch callbacks outside mutex_
    std::vector<VkSemaphore> semaphores_;
    std::vector<std::uint64_t> values_;
    std::uint64_t epoch_ = 0;
    bool waiting_ = false;
    bool stop_ = false;
    std::thread thread_;
};

// A range of a D3D12 heap allocation. Mappings and in-flight unbinds keep the arena range alive.
struct TileHeapAllocation {
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize offset = 0;
    VkDeviceSize size = 0;
    std::uint32_t memory_type = 0;
};

struct TileMapping {
    std::shared_ptr<const TileHeapAllocation> heap;
    VkDeviceSize offset = 0;
    bool operator==(const TileMapping&) const = default;
};

struct SparseResource {
    SparseResource(VkDevice device, const VulkanDeviceFunctions& vk) : device(device), vk(vk) {}
    SparseResource(const SparseResource&) = delete;
    SparseResource& operator=(const SparseResource&) = delete;
    ~SparseResource() {
        if (image) vk.vkDestroyImage(device, image, nullptr);
        if (buffer) vk.vkDestroyBuffer(device, buffer, nullptr);
    }

    VkDevice device;
    const VulkanDeviceFunctions& vk;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkImage image = VK_NULL_HANDLE;
    VkExtent2D image_extent{};
    std::uint64_t tile_count = 0;
    std::uint32_t memory_types = 0;
    std::map<std::uint64_t, TileMapping> mapped;
};

struct TileRegion {
    std::uint64_t first = 0;
    std::uint64_t count = 0;
};

enum class TileRangeMode { sequential, unmap, skip, reuse };

struct TileRange {
    TileRangeMode mode;
    std::uint64_t first;
    std::uint64_t count;
};

// Per device: what the queues share. Resources are registered by their stable ID (the COM resource pointer).
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4324)  // intentional alignas(64) cache-line padding to prevent false sharing
#endif
struct GraphicsIndirectCapabilities {
    bool first_instance = false;
    bool multiple = false;
    bool count_buffer = false;
    std::uint32_t max_count = 1;
    bool robust_index_view = false;
    bool require_clamp = false;
};

class VulkanContext {
public:
    // mesh_queries: VK_EXT_mesh_shader's meshShaderQueries feature is enabled on `device`, so a mesh segment's
    // clipping invocations can come from a mesh-primitives query (D3D-MESH-CLIPPING-COUNTER). Set by the device
    // that created the context.
    bool mesh_queries = false;
    // conditional_rendering: VK_EXT_conditional_rendering is enabled on `device` (SetPredication).
    // ray_tracing: VK_KHR_acceleration_structure + VK_KHR_ray_query are enabled on `device`.
    // xdxr: acceleration-structure builds write XDXR BVH4 records (xdxr_build.h) instead; needs bufferDeviceAddress.
    VulkanContext(PFN_vkGetDeviceProcAddr get, VkDevice device, bool separate_depth_stencil_layouts,
                  bool conditional_rendering = false, bool ray_tracing = false, bool xdxr = false,
                  GraphicsIndirectCapabilities indirect = {})
        : device_(device), tracker_(separate_depth_stencil_layouts), ray_tracing_(ray_tracing) {
        loaded_ = vk.load(get, device);
        // VK_EXT_transform_feedback: enabled on the device when its indexed-query entry points resolved (D3D-302).
        vk.transform_feedback = vk.vkCmdBeginQueryIndexedEXT && vk.vkCmdEndQueryIndexedEXT;
        if (loaded_ && gpu_labels_.initialize(device, get)) {
            replay_.gpu_label = [this](VkCommandBuffer cb, const label::Write& write) { gpu_labels_.record(cb, write); };
        }
        if (loaded_ && gpu_profile::enabled()) gpu_profile_.configure(device, get);  // LEONIDA_GPU_PROFILE=1
        replay_.pipeline_barrier = vk.vkCmdPipelineBarrier;
        replay_.clear_color_image = vk.vkCmdClearColorImage;
        replay_.clear_depth_stencil_image = vk.vkCmdClearDepthStencilImage;
        replay_.pipeline_barrier2 = vk.vkCmdPipelineBarrier2;
        replay_.begin_rendering = vk.vkCmdBeginRendering;
        replay_.end_rendering = vk.vkCmdEndRendering;
        replay_.clear_attachments = vk.vkCmdClearAttachments;
        replay_.copy_buffer = vk.vkCmdCopyBuffer;
        replay_.copy_buffer_to_image = vk.vkCmdCopyBufferToImage;
        replay_.copy_image_to_buffer = vk.vkCmdCopyImageToBuffer;
        replay_.copy_image = vk.vkCmdCopyImage;
        replay_.blit_image = vk.vkCmdBlitImage;
        replay_.resolve_image = vk.vkCmdResolveImage;
        replay_.fill_buffer = vk.vkCmdFillBuffer;
        replay_.update_buffer = vk.vkCmdUpdateBuffer;
        replay_.draw_indirect = vk.vkCmdDrawIndirect;
        replay_.draw_indexed_indirect = vk.vkCmdDrawIndexedIndirect;
        replay_.draw_indirect_first_instance = indirect.first_instance;
        replay_.multi_draw_indirect = indirect.multiple;
        replay_.draw_indirect_count_enabled = indirect.count_buffer;
        replay_.max_draw_indirect_count = indirect.max_count;
        replay_.require_indirect_clamp = indirect.require_clamp;
        replay_.draw_indirect_count = vk.vkCmdDrawIndirectCount;
        replay_.draw_indexed_indirect_count = vk.vkCmdDrawIndexedIndirectCount;
        replay_.dispatch_indirect = vk.vkCmdDispatchIndirect;
        if (conditional_rendering) {  // a driver may hand out these pointers even when the extension is off
            replay_.begin_conditional_rendering = vk.vkCmdBeginConditionalRenderingEXT;
            replay_.end_conditional_rendering = vk.vkCmdEndConditionalRenderingEXT;
        }
        replay_.reset_query_pool = vk.vkCmdResetQueryPool;
        replay_.begin_query = vk.vkCmdBeginQuery;
        replay_.end_query = vk.vkCmdEndQuery;
        // VK_EXT_transform_feedback (D3D-302), loaded when the device enabled it: indexed SO_STATISTICS queries.
        if (vk.transform_feedback) {
            replay_.begin_query_indexed = vk.vkCmdBeginQueryIndexedEXT;
            replay_.end_query_indexed = vk.vkCmdEndQueryIndexedEXT;
        }
        replay_.write_timestamp = vk.vkCmdWriteTimestamp;
        replay_.copy_query_pool_results = vk.vkCmdCopyQueryPoolResults;
        replay_.bind_pipeline = vk.vkCmdBindPipeline;
        replay_.push_constants = vk.vkCmdPushConstants;
        replay_.bind_descriptor_sets = vk.vkCmdBindDescriptorSets;
        replay_.dispatch = vk.vkCmdDispatch;
        replay_.compute_pipeline = [this](std::uintptr_t id) -> const ComputePipeline* {
            const auto it = pipelines_.find(id);
            return it == pipelines_.end() ? nullptr : &it->second;
        };
        replay_.bind_vertex_buffers2 = vk.vkCmdBindVertexBuffers2;
        replay_.bind_index_buffer = vk.vkCmdBindIndexBuffer;
        if (indirect.robust_index_view)
            replay_.bind_robust_index_view = reinterpret_cast<decltype(replay_.bind_robust_index_view)>(
                get(device, "vkCmdBindIndexBuffer2KHR"));
        replay_.set_viewport = vk.vkCmdSetViewport;
        replay_.set_scissor = vk.vkCmdSetScissor;
        replay_.set_viewport_with_count = vk.vkCmdSetViewportWithCount;
        replay_.set_scissor_with_count = vk.vkCmdSetScissorWithCount;
        replay_.set_primitive_topology = vk.vkCmdSetPrimitiveTopology;
        replay_.set_primitive_restart_enable = vk.vkCmdSetPrimitiveRestartEnable;
        replay_.set_stencil_reference = vk.vkCmdSetStencilReference;
        replay_.set_blend_constants = vk.vkCmdSetBlendConstants;
        replay_.set_depth_bounds = vk.vkCmdSetDepthBounds;
        replay_.draw = vk.vkCmdDraw;
        replay_.draw_mesh_tasks = vk.vkCmdDrawMeshTasksEXT;  // null unless VK_EXT_mesh_shader is enabled
        replay_.draw_mesh_tasks_indirect = vk.vkCmdDrawMeshTasksIndirectEXT;
        replay_.draw_mesh_tasks_indirect_count = vk.vkCmdDrawMeshTasksIndirectCountEXT;
        replay_.draw_indexed = vk.vkCmdDrawIndexed;
        replay_.graphics_pipeline = [this](std::uintptr_t id) -> const GraphicsPipeline* {
            const auto it = graphics_pipelines_.find(id);
            return it == graphics_pipelines_.end() ? nullptr : &it->second;
        };
        // Called with resource_mutex_ held. A binary search of the buffers' VA ranges (buffer_va_index.h), kept in step
        // with resources_ by set_resource_locked and remove_resource.
        // A range that runs past its buffer is cut at the buffer's end (`size` is updated): D3D12 reads past a view's
        // buffer as out of bounds, which is no reason to drop the submission.
        replay_.buffer_at = [this](std::uint64_t va, std::uint64_t& size, VkBuffer& buffer, VkDeviceSize& offset) {
            BufferVaIndex<ResourceId>::Hit hit{};
            if (!buffer_vas_.find(va, hit, size)) {  // prefers an aliasing buffer that holds all of it
                // Title memory the page mirror holds (set_mirror_buffer): its chunk at heap_base + va.
                return mirror_buffer_ && mirror_buffer_(va, size, buffer, offset);
            }
            const ResourceBinding* b = resolve(hit.id);
            if (!b) return false;
            buffer = b->buffer;
            offset = b->buffer_offset + (va - b->va);
            size = std::min<std::uint64_t>(size, b->size - (va - b->va));
            return true;
        };
        // Host reads of small CPU-written inputs. Unlike buffer_at, the whole range must be in one CPU-writable
        // mapping: a partial read would see bytes the application has not written through the D3D12 API. The
        // binding's host pointer is null on DEFAULT and READBACK heaps, whose mapped bytes the GPU can write.
        replay_.host_at = [this](std::uint64_t va, std::uint64_t bytes, const std::uint8_t*& host) {
            BufferVaIndex<ResourceId>::Hit hit{};
            if (!buffer_vas_.find(va, hit, bytes)) return HostReadStatus::absent;
            const ResourceBinding* b = resolve(hit.id);
            if (!b) return HostReadStatus::absent;
            if (!b->host) return HostReadStatus::not_cpu_written;
            const std::uint64_t offset = va - b->va;
            if (offset > b->size || bytes > b->size - offset) return HostReadStatus::absent;
            host = b->host + offset;
            return HostReadStatus::ok;
        };
        if (xdxr && loaded_) {
            xdxr_ = std::make_unique<XdxrBuilder>(get, device);
            if (xdxr_->loaded()) {
                replay_.xdxr_check = [](const BuildAccelerationStructure& b) {
                    XdxrLayout layout;
                    // An update needs ALLOW_UPDATE (D3D12 requires it on the build and the update): a structure
                    // built without it may have paired leaves, which a refit would rewrite as wrong geometry.
                    return xdxr_inputs(b, nullptr, 0) && layout.compute(xdxr_elements(b), b.is_tlas, xdxr_box16(b)) &&
                           b.dst_va && b.scratch_va && (!b.is_tlas || !b.instance_count || b.instances_va) &&
                           (!b.update || (b.build_flags & VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR) != 0);
                };
                replay_.xdxr_record = [this](VkCommandBuffer cb, const BuildAccelerationStructure& b,
                                             const XdxrBlasTable& blases) {
                    // Every address is the VA shaders reach the memory by: an arena VA, or for the title's own memory
                    // the VA of its GPU copy, resolved when the build was recorded (Device::resolve_xdxr_build). The
                    // device address is heap_base + VA, and the BLAS addresses in instance descs, which are the
                    // title's, go through `blases` and get the same bias.
                    const std::uint64_t base = replay_.heap_base;
                    XdxrLayout layout;
                    layout.compute(xdxr_elements(b), b.is_tlas, xdxr_box16(b));
                    // PERFORM_UPDATE refits the source (the destination itself when no source is given, as D3D12
                    // allows for an in-place update); a full build otherwise.
                    const std::uint64_t from = b.update ? base + (b.src_va ? b.src_va : b.dst_va) : 0;
                    if (b.is_tlas) {
                        xdxr_->record_tlas(cb, layout, b.instances_va ? base + b.instances_va : 0, base + b.dst_va,
                                           base + b.scratch_va, base, from, &blases);
                    } else {
                        std::vector<XdxrTriangles> geometries;
                        xdxr_inputs(b, &geometries, base);
                        xdxr_->record_blas(cb, layout, geometries, base + b.dst_va, base + b.scratch_va, from,
                                           xdxr_pair(b));
                    }
                };
                replay_.xdxr_copy = [this](VkCommandBuffer cb, std::uint64_t src, std::uint64_t dst, std::uint64_t bytes) {
                    const std::uint64_t base = replay_.heap_base;
                    xdxr_->record_copy(cb, base + src, base + dst, bytes);
                };
                replay_.xdxr_postbuild = [this](VkCommandBuffer cb, std::uint64_t src, std::uint64_t dest) {
                    const std::uint64_t base = replay_.heap_base;
                    xdxr_->record_postbuild(cb, base + src, base + dest);
                };
            } else {
                xdxr_.reset();
            }
        }
        if (!xdxr_ && ray_tracing_ && vk.vkCmdBuildAccelerationStructuresKHR) {
            replay_.build_acceleration_structures = vk.vkCmdBuildAccelerationStructuresKHR;
            replay_.copy_acceleration_structure = vk.vkCmdCopyAccelerationStructureKHR;
            replay_.write_acceleration_structures_properties = vk.vkCmdWriteAccelerationStructuresPropertiesKHR;
            replay_.resolve_acceleration_structure = [this](std::uint64_t va) {
                const auto it = acceleration_structures_.find(va);
                return it == acceleration_structures_.end() ? VK_NULL_HANDLE : it->second;
            };
        }
        replay_.image_view = [this](const ResourceBinding& r, const VkImageSubresourceRange& range) {
            return view(r, range);
        };
        if (vk.vkCreateRenderPass2 && vk.vkDestroyRenderPass && vk.vkCreateFramebuffer && vk.vkDestroyFramebuffer &&
            vk.vkCmdBeginRenderPass && vk.vkCmdEndRenderPass) {
            replay_.begin_render_pass = vk.vkCmdBeginRenderPass;
            replay_.end_render_pass = vk.vkCmdEndRenderPass;
            replay_.depth_resolve_framebuffer = [this](const ResourceBinding& source, VkImageView from,
                                                       VkImageLayout from_layout, const ResourceBinding& target,
                                                       VkImageView to, VkImageLayout to_layout,
                                                       VkResolveModeFlagBits mode, VkExtent2D extent,
                                                       VkRenderPass& pass) {
                return depth_resolve_framebuffer(source, from, from_layout, target, to, to_layout, mode, extent, pass);
            };
        }
        replay_.query_pool = [this](QueryHeapId id) -> QueryPool* {  // called with resource_mutex_ held
            const auto it = query_pools_.find(id);
            return it == query_pools_.end() ? nullptr : &it->second;
        };
        replay_.separate_depth_stencil_layouts = separate_depth_stencil_layouts;
        if (loaded_) {
            fence_dispatcher_ = std::make_unique<FenceWaitDispatcher>(device_, vk, device_lost_);
        }
    }

    ~VulkanContext() {
        progress_watch_ = {};  // stop before queues, timelines or diagnostic state go away
        gpu_labels_.stop(device_);
        fence_dispatcher_.reset();
        gpu_profile_.stop();  // after the fence callbacks that hold its recordings
        deletion_queue_.drain(UINT64_MAX, device_, vk);
        for (const auto& [key, framebuffer] : resolve_framebuffers_)
            vk.vkDestroyFramebuffer(device_, framebuffer, nullptr);
        for (const auto& [key, pass] : resolve_passes_) vk.vkDestroyRenderPass(device_, pass, nullptr);
        for (const auto& [key, view] : views_) vk.vkDestroyImageView(device_, view, nullptr);
    }

    VulkanContext(const VulkanContext&) = delete;
    VulkanContext& operator=(const VulkanContext&) = delete;

    bool loaded() const { return loaded_; }

    // `memory`: the physical device's memory types, for the AMD marker buffer (enabled.markers).
    void configure_device_fault(PFN_vkGetDeviceProcAddr get, DeviceFaultExtensions enabled,
                                const VkPhysicalDeviceMemoryProperties* memory = nullptr) {
        device_fault_ = std::make_unique<DeviceFaultDiagnostics>(get, device_, enabled, memory,
                                                                 vk.vkCmdPipelineBarrier2 != nullptr);
        device_fault_->set_submit_lock(&submit_mutex_);
        if (fence_dispatcher_) fence_dispatcher_->set_device_fault(device_fault_.get());
        if (loaded_) progress_watch_ = std::jthread([this](std::stop_token stop) { watch_queue_progress(stop); });
    }

    void add_diagnostic_queue(VkQueue queue) {
        if (device_fault_) device_fault_->add_queue(queue);
    }

    DeviceFaultDiagnostics* device_fault() const { return device_fault_.get(); }

    VkResult observe_result(VkResult result, const char* operation) const noexcept {
        return DeviceFaultDiagnostics::observe(device_fault_.get(), result, operation);
    }
    void configure_predication(PFN_vkGetDeviceProcAddr get, const VkPhysicalDeviceMemoryProperties& memory,
                               VkDeviceSize max_range, std::uint32_t family) {
        std::unique_lock lock(resource_mutex_);
        predicate_family_ = family;
        predicate_reducer_ = std::make_unique<PredicateReducer>(get, device_, memory, max_range);
        bump_replay_version();
    }
    // Configure before submitting work, only when storageBuffer8BitAccess is enabled on the device.
    void configure_predicated_copy(PFN_vkGetDeviceProcAddr get, const VkPhysicalDeviceLimits& limits) {
        std::unique_lock lock(resource_mutex_);
        predicated_copier_ = std::make_unique<PredicatedBufferCopier>(get, device_, limits);
        bump_replay_version();
    }
    // Counts for DISPATCH-only ExecuteIndirect are read by a helper compute shader from GPU memory.
    // Requires Vulkan bufferDeviceAddress, already enabled for the D3D12 address arena.
    void configure_counted_dispatch(PFN_vkGetDeviceProcAddr get, const VkPhysicalDeviceMemoryProperties& memory,
                                    const VkPhysicalDeviceLimits& limits) {
        std::unique_lock lock(resource_mutex_);
        counted_dispatch_ = std::make_unique<IndirectDispatchCounter>(get, device_, memory, limits);
        bump_replay_version();
    }
    // Indexed quad lists are drawn from triangle indices a compute pass writes into the copy scratch
    // (quad_index_expand.h). Call only with bufferDeviceAddress enabled and after an enable_copy_scratch whose chunks
    // have INDEX_BUFFER and SHADER_DEVICE_ADDRESS usage on memory allocated with a device address.
    void configure_quad_indices(PFN_vkGetDeviceProcAddr get, const VkPhysicalDeviceLimits& limits) {
        std::unique_lock lock(resource_mutex_);
        quad_indices_ = std::make_unique<QuadIndexExpander>(get, device_, limits);
        bump_replay_version();
    }

    // Every ExecuteIndirect's arguments are clamped on the GPU into scratch before the indirect command reads them
    // (vulkan_indirect_clamp.h), so a stale or garbage record from a skipped compute pass cannot reset the display
    // driver. Needs bufferDeviceAddress and a compute-capable queue; `allocate` gives the record scratch
    // device-local memory, and must set VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT. LEONIDA_INDIRECT_CLAMP=0 turns the
    // pass off at replay time.
    void enable_indirect_clamp(PFN_vkGetDeviceProcAddr get, const VkPhysicalDeviceMemoryProperties& memory,
                               const VkPhysicalDeviceLimits& limits, ScratchBuffers::Allocate allocate,
                               std::vector<std::uint32_t> families = {}) {
        std::unique_lock lock(resource_mutex_);
        // An enabled clamp is required: a call whose scratch preparation fails is refused (unsupported) instead of
        // reading the application's argument buffer, which may hold a record no compute pass wrote. Off, or on a
        // device the pass cannot run on, the replay keeps the application's buffer as before.
        replay_.require_indirect_clamp = indirect_clamp_enabled();
        if (!replay_.require_indirect_clamp) {
            bump_replay_version();
            return;
        }
        indirect_clamp_ = std::make_unique<IndirectArgumentClamp>(get, device_, memory, limits, std::move(allocate),
                                                                 device_fault_.get(), std::move(families));
        if (!indirect_clamp_->loaded()) {
            indirect_clamp_.reset();
            replay_.require_indirect_clamp = false;
            bump_replay_version();
            return;
        }
        // The preparation callback needs this submission's counter-slot batch, so it is set per ExecuteCommandLists
        // (execute()); recording the pass has no per-submission state.
        replay_.emit_indirect_clamp = [this](VkCommandBuffer command_buffer, const ClampedIndirect& job) {
            indirect_clamp_->emit(command_buffer, job);
        };
        bump_replay_version();
    }
    VkDevice device() const { return device_; }

    // Debug markers become labels from here on; call before any queue or list uses the context. Incomplete sets
    // are ignored, so markers stay no-ops.
    void enable_debug_labels(const DebugLabelFunctions& labels) {
        if (!labels.complete()) return;
        labels_ = labels;
        replay_.insert_label = labels.cmd_insert;
        replay_.begin_label = labels.cmd_begin;
        replay_.end_label = labels.cmd_end;
        bump_replay_version();
    }
    bool debug_labels() const { return labels_.complete(); }

    // Registers a resource in its D3D12 initial state (a legacy state; images get the matching layout).
    // The image must already be in that layout, e.g. transitioned from UNDEFINED at creation.
    void add_resource(ResourceId id, const ResourceBinding& binding) {
        std::unique_lock lock(resource_mutex_);
        set_resource_locked(id, binding);
    }
    // Whether `va` is in a buffer this context knows by VA (the index buffer_at searches).
    bool knows_buffer_va(std::uint64_t va, std::uint64_t bytes = 1) {
        std::shared_lock lock(resource_mutex_);
        return buffer_vas_.covers(va, bytes);
    }
    void add_resource(ResourceId id, const ResourceBinding& binding, VkImageLayout current_layout) {
        std::unique_lock rlock(resource_mutex_);
        set_resource_locked(id, binding);
        if (binding.image) {
            std::lock_guard tlock(tracker_mutex_);
            tracker_.track(id, binding, current_layout);
        }
    }

    // A shader can access a storage image or an initially unbound sparse image before any command names it.
    // The first shader submission moves it to GENERAL, the descriptor's layout, ahead of its lists.
    void note_shader_image(ResourceId id) {
        std::lock_guard lock(storage_mutex_);
        storage_pending_.insert(id);
    }

    // A device-local buffer D3D12 says starts zeroed (D3D-71): the host cannot write it, so the next submission on
    // any queue fills it with zeros ahead of its lists. That submission publishes the same initialisation semaphore as
    // the storage images above, so a list on another queue that uses the buffer first waits for the fill.
    void zero_on_next_submit(ResourceId id, VkBuffer buffer) {
        std::lock_guard lock(storage_mutex_);
        zero_pending_.push_back({id, buffer});
    }

    void add_sparse_resource(ResourceId id, const ResourceBinding& binding, std::shared_ptr<SparseResource> sparse) {
        std::unique_lock lock(resource_mutex_);
        if (!binding.image) {
            auto [it, inserted] = sparse_resources_.try_emplace(id, std::move(sparse));
            try {
                set_resource_locked(id, binding);
            } catch (...) {
                if (inserted) sparse_resources_.erase(it);
                throw;
            }
            return;
        }
        set_resource_locked(id, binding);
        sparse_resources_.emplace(id, std::move(sparse));
        if (binding.image) {
            std::lock_guard tlock(tracker_mutex_);
            tracker_.track(id, binding, VK_IMAGE_LAYOUT_UNDEFINED);
        }
    }

    void remove_resource(ResourceId id) {
        VkImage img = VK_NULL_HANDLE;
        std::shared_ptr<SparseResource> sparse_resource;
        {
            std::unique_lock rlock(resource_mutex_);
            const auto it = resources_.find(id);
            if (it == resources_.end()) return;
            img = it->second.image;
            unindex_buffer(id, it->second);
            resources_.erase(it);
            if (auto sparse = sparse_resources_.find(id); sparse != sparse_resources_.end()) {
                sparse_resource = std::move(sparse->second);
                sparse_resources_.erase(sparse);
            }
        }
        {
            std::lock_guard lock(storage_mutex_);  // a buffer destroyed before any submission is not filled
            storage_pending_.erase(id);
            std::erase_if(zero_pending_, [id](const PendingZero& zero) { return zero.id == id; });
        }
        if (img) {
            std::lock_guard vlock(views_mutex_);
            for (auto f = resolve_framebuffers_.begin(); f != resolve_framebuffers_.end();) {
                if (std::get<0>(f->first) == img || std::get<1>(f->first) == img) {
                    defer_destroy(DeferredDeletionQueue::Kind::framebuffer, f->second);
                    f = resolve_framebuffers_.erase(f);
                } else {
                    ++f;
                }
            }
            for (auto v = views_.begin(); v != views_.end();) {
                if (std::get<0>(v->first) == img) {
                    defer_destroy(DeferredDeletionQueue::Kind::image_view, v->second);
                    v = views_.erase(v);
                } else {
                    ++v;
                }
            }
        }
        if (sparse_resource)
            defer_destroy(DeferredDeletionQueue::Kind::sparse_resource,
                          new std::shared_ptr<SparseResource>(std::move(sparse_resource)));
        std::lock_guard tlock(tracker_mutex_);
        tracker_.forget(id);
    }

    // A query heap, by its stable ID (the COM query heap pointer). The pool stays the caller's to destroy, after
    // remove_query_pool.
    void add_query_pool(QueryHeapId id, const QueryPool& pool) {
        std::unique_lock lock(resource_mutex_);
        query_pools_[id] = pool;
    }
    void remove_query_pool(QueryHeapId id) {
        // Lock hierarchy: resource_mutex_ -> tracker_mutex_. Holding tracker_mutex_ before erasing keeps the
        // QueryPool node alive until any in-flight execute() that unlocked resource_mutex_ finishes QueryReplay::commit().
        std::unique_lock lock(resource_mutex_);
        std::lock_guard tlock(tracker_mutex_);
        query_pools_.erase(id);
    }
    // The heap's d3d12_query_heap type, or ~0u for a heap that is not registered.
    std::uint32_t query_heap_type(QueryHeapId id) {
        std::shared_lock lock(resource_mutex_);   // query_pools_ is guarded by resource_mutex_, like add/remove
        const auto it = query_pools_.find(id);
        return it == query_pools_.end() ? ~0u : it->second.heap_type;
    }

    // VkPhysicalDeviceLimits::timestampPeriod (nanoseconds per tick), or 0 when the queue writes no timestamps
    // (timestampValidBits 0). Set by the device before any queue uses it.
    double timestamp_period() const { return timestamp_period_; }
    void set_timestamp_period(double period) { timestamp_period_ = period; }
    // timestampValidBits by queue family, 0 for one that cannot reset or resolve queries. Set by the device before any
    // queue uses it; LEONIDA_GPU_PROFILE profiles the queues of the families with bits.
    void set_timestamp_bits(std::vector<std::uint32_t> bits) { timestamp_bits_ = std::move(bits); }
    std::uint32_t timestamp_bits(std::uint32_t family) const {
        return family < timestamp_bits_.size() ? timestamp_bits_[family] : 0;
    }
    // LEONIDA_GPU_PROFILE=1: the GPU time per pipeline (vulkan_gpu_profile.h). Tests read its sums.
    gpu_profile::Profiler& gpu_profile() { return gpu_profile_; }
    // True where a timestamp resolve must be ordered explicitly after the timestamps it copies (MoltenVK on Apple
    // GPUs, docs/d3d12_vulkan.md section 4.6). Set by the device before any queue uses it.
    bool order_timestamp_copies() const { return order_timestamp_copies_; }
    void set_order_timestamp_copies(bool order) {
        order_timestamp_copies_ = order;
        bump_replay_version();
    }

    // The device address of the GPU address arena: console VAs are offsets from it (docs section 3.3).
    // execute() copies replay_ under a shared resource_mutex_, so writers hold it uniquely.
    // `heap_size`: the arena's bytes, below which a VA a GPU pass finds in memory (a TLAS instance's BLAS) is an arena
    // VA; 0 when unknown (such a VA is then taken as one whatever its value).
    void set_heap_base(std::uint64_t heap_base, std::uint64_t heap_size = 0) {
        std::unique_lock lock(resource_mutex_);
        replay_.heap_base = heap_base;
        replay_.heap_size = heap_size;
        bump_replay_version();
    }

    // Zeroed memory for unbound vertex and index buffers to read (ReplayFunctions::zero_buffer).
    void set_zero_page(VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size) {
        std::unique_lock lock(resource_mutex_);
        replay_.zero_buffer = buffer;
        replay_.zero_offset = offset;
        replay_.zero_size = size;
        bump_replay_version();
    }

    void set_depth_range_unrestricted(bool enabled) {
        std::unique_lock lock(resource_mutex_);
        replay_.depth_range_unrestricted = enabled;
        bump_replay_version();
    }

    struct GeometryUpload {
        std::uint64_t source{}, size{};
        VkBuffer buffer{};
        VkDeviceSize offset{};
        std::shared_ptr<void> owner;
    };
    using GeometryProvider = std::function<bool(std::uint64_t,std::uint64_t,GeometryUpload&)>;
    void set_geometry_provider(GeometryProvider provider) {
        std::unique_lock lock(resource_mutex_);
        geometry_provider_=std::move(provider);
        bump_replay_version();
    }

    // LOCAL: the copy-in fallback for vertex and index views in title memory (ReplayFunctions::title_memory).
    void set_title_memory(std::function<bool(std::uint64_t, std::uint64_t, VkBuffer&, VkDeviceSize&)> copy) {
        std::unique_lock lock(resource_mutex_);
        replay_.title_memory = std::move(copy);
        bump_replay_version();
    }

    // Pre-expanded 32-bit triangle indices for non-indexed quad lists (ReplayFunctions::quad_indices).
    void set_quad_indices(VkBuffer buffer, VkDeviceSize offset, std::uint32_t count) {
        std::unique_lock lock(resource_mutex_);
        replay_.quad_indices = buffer;
        replay_.quad_indices_offset = offset;
        replay_.quad_index_count = count;
        bump_replay_version();
    }

    // Physical-device COLOR_ATTACHMENT format check for reinterpreted UAV clears.
    void set_color_attachment_format(std::function<bool(VkFormat)> check) {
        std::unique_lock lock(resource_mutex_);
        replay_.color_attachment_format = std::move(check);
        bump_replay_version();
    }
    std::uint64_t heap_base() {
        std::shared_lock lock(resource_mutex_);
        return replay_.heap_base;
    }

    // Pipelines are registered by an opaque id (the PSO object) for as long as the PSO lives.
    // Guarded by pipeline_mutex_ (separate from resource_mutex_) so shader compilation on one queue does not block
    // resource registration or destruction on the render thread.
    void add_pipeline(std::uintptr_t id, const ComputePipeline& pipeline, const char* shader = nullptr) {
        std::unique_lock lock(pipeline_mutex_);
        pipelines_[id] = pipeline;
        if (!pipeline.pipeline && pipeline.build && pipeline.built && !pipeline.async_build && !pipeline.skipped &&
            !pipeline.built(pipeline.build_context))
            note_unbuilt_sync_pipeline();
        if (shader && device_fault_) device_fault_->pipeline_names[id] = {shader};
        if (shader && gpu_profile::enabled()) profile_names_[id] = {shader};
    }
    void remove_pipeline(std::uintptr_t id) {
        std::unique_lock lock(pipeline_mutex_);
        pipelines_.erase(id);
        graphics_pipelines_.erase(id);
        if (device_fault_) device_fault_->pipeline_names.erase(id);
        if (gpu_profile::enabled()) profile_names_.erase(id);
    }
    void add_graphics_pipeline(std::uintptr_t id, const GraphicsPipeline& pipeline,
                               const CheckpointShaders* shaders = nullptr) {
        std::unique_lock lock(pipeline_mutex_);
        graphics_pipelines_[id] = pipeline;
        if (!pipeline.async_build && pipeline.variant && pipeline.built && !pipeline.pipeline && !pipeline.built(0))
            note_unbuilt_sync_pipeline();
        if (shaders && device_fault_) device_fault_->pipeline_names[id] = *shaders;
        if (shaders && gpu_profile::enabled()) profile_names_[id] = *shaders;
    }
    // Tracks whether any registered synchronous pipeline may still need pre-pass compilation so warm runs skip
    // collect_unbuilt_pipelines() in O(1) (pass-0 prepare() remains as a safe fallback for late variants or
    // previously-unused PSOs).
    void note_unbuilt_sync_pipeline() noexcept {
        sync_pipeline_gen_.fetch_add(1, std::memory_order_acq_rel);
        warm_empty_pipeline_scans_.store(0, std::memory_order_relaxed);
        unbuilt_sync_pipelines_.fetch_add(1, std::memory_order_acq_rel);
    }
    std::uint64_t sync_pipeline_gen() const noexcept { return sync_pipeline_gen_.load(std::memory_order_acquire); }
    bool has_unbuilt_sync_pipelines() const noexcept {
        return unbuilt_sync_pipelines_.load(std::memory_order_acquire) != 0;
    }
    void note_sync_pipelines_built(std::size_t count) noexcept {
        warm_empty_pipeline_scans_.store(0, std::memory_order_relaxed);
        std::uint64_t cur = unbuilt_sync_pipelines_.load(std::memory_order_acquire);
        while (cur > 0) {
            const std::uint64_t next = cur > count ? cur - count : 0;
            if (unbuilt_sync_pipelines_.compare_exchange_weak(cur, next, std::memory_order_acq_rel)) break;
        }
    }
    void note_empty_pipeline_scan(std::uint64_t gen_before_scan) noexcept {
        if (sync_pipeline_gen_.load(std::memory_order_acquire) != gen_before_scan) return;
        constexpr std::uint32_t kWarmEmptyScans = 16;
        if (warm_empty_pipeline_scans_.fetch_add(1, std::memory_order_relaxed) + 1 >= kWarmEmptyScans) {
            if (sync_pipeline_gen_.load(std::memory_order_acquire) == gen_before_scan) {
                unbuilt_sync_pipelines_.store(0, std::memory_order_release);
                // Re-arm if a pipeline was registered concurrently between the gen check and store.
                if (sync_pipeline_gen_.load(std::memory_order_acquire) != gen_before_scan) {
                    unbuilt_sync_pipelines_.fetch_add(1, std::memory_order_acq_rel);
                }
            }
        }
    }
    // Workers for the pipelines a submission needs and that are not built yet. Null (the default): compiled serially on
    // the submitting thread once resource_mutex_ is released. Replacing it waits for submissions that are compiling, and
    // ends the workers it had.
    void set_pipeline_workers(std::unique_ptr<PipelineWorkers> workers) {
        std::unique_lock lock(pipeline_mutex_);
        pipeline_workers_.swap(workers);
    }
    // For the large pipelines of a submission: the memory they may use while they compile together
    // (PipelineCompiles::build), and what is kept while they do (LargeModuleGate::Demand). None (the default): they compile
    // one after another.
    void set_large_compiles(std::function<std::uint64_t()> budget, std::function<std::shared_ptr<void>()> demand) {
        std::unique_lock lock(pipeline_mutex_);
        large_compile_budget_ = std::move(budget);
        large_compile_demand_ = std::move(demand);
    }
    using BatchCompileFn = std::function<void(std::size_t, const std::function<void(std::size_t)>&)>;
    void set_batch_compile_runner(BatchCompileFn runner) {
        std::unique_lock lock(pipeline_mutex_);
        batch_compile_runner_ = std::move(runner);
    }
    // Tracks pending queue Wait(semaphore, value) calls across all queues on this context so same-queue coalescing
    // can merge [execute, signal(F, V), execute] when no waiter awaits (F, <= V), while splitting right after
    // Signal(F, V) whenever any queue or CPU wait awaits a value <= V that is still pending in QueueHolds.
    void note_wait_added(VkSemaphore semaphore, std::uint64_t value) {
        if (!semaphore) return;
        std::lock_guard lock(wait_tracker_mutex_);
        pending_waits_.push_back({semaphore, value});
        pending_wait_count_.store(pending_waits_.size(), std::memory_order_release);
    }
    void note_wait_removed(VkSemaphore semaphore, std::uint64_t value) {
        if (!semaphore) return;
        std::lock_guard lock(wait_tracker_mutex_);
        for (auto it = pending_waits_.begin(); it != pending_waits_.end(); ++it) {
            if (it->first == semaphore && it->second == value) {
                *it = pending_waits_.back();
                pending_waits_.pop_back();
                break;
            }
        }
        pending_wait_count_.store(pending_waits_.size(), std::memory_order_release);
    }
    void note_waits_removed(std::span<const VkSemaphore> semaphores, std::span<const std::uint64_t> values) {
        const std::size_t n = (std::min)(semaphores.size(), values.size());
        if (!n || pending_wait_count_.load(std::memory_order_acquire) == 0) return;
        std::lock_guard lock(wait_tracker_mutex_);
        for (std::size_t i = 0; i < n; ++i) {
            if (!semaphores[i]) continue;
            for (auto it = pending_waits_.begin(); it != pending_waits_.end(); ++it) {
                if (it->first == semaphores[i] && it->second == values[i]) {
                    *it = pending_waits_.back();
                    pending_waits_.pop_back();
                    break;
                }
            }
        }
        pending_wait_count_.store(pending_waits_.size(), std::memory_order_release);
    }
    bool has_unsatisfied_waiter(VkSemaphore semaphore, std::uint64_t value) const {
        if (!semaphore) return false;
        std::uint64_t max_waited = 0;
        if (pending_wait_count_.load(std::memory_order_acquire) != 0) {
            std::lock_guard lock(wait_tracker_mutex_);
            for (const auto& [sem, waited] : pending_waits_) {
                if (sem == semaphore && waited <= value && waited > max_waited) max_waited = waited;
            }
        }
        if (fence_dispatcher_) {
            const std::uint64_t cpu_max = fence_dispatcher_->max_wait_value_le(semaphore, value);
            if (cpu_max > max_waited) max_waited = cpu_max;
        }
        return max_waited != 0 && holds_.pending(semaphore, max_waited);
    }
    PipelineWorkers::Stats pipeline_worker_stats() {
        std::shared_lock lock(pipeline_mutex_);
        return pipeline_workers_ ? pipeline_workers_->stats() : PipelineWorkers::Stats{};
    }
    // Configure before queue creation, when the selected queue family supports compute.
    void set_occlusion_resolve_memory(const VkPhysicalDeviceMemoryProperties& memory) {
        occlusion_resolve_memory_ = memory;
        bump_replay_version();
    }

    // A complete replacement of these color subresources from a staging buffer (TitleImageUpload): the title's CPU
    // writes into its own texture memory, brought to the image. Serialized with replay and present while the old
    // layouts are read and the transfer is submitted. It never waits for the GPU on the CPU: `copy(replay, pieces,
    // waits)` records the transfer and submits it with a GPU wait on `waits`, every queue timeline value submitted so
    // far, so it runs after all earlier work that may still read the image, on any queue, and before every submission
    // made after it (the ones that sample the new texels). Until 2026-11 this waited on the CPU for that work for at
    // most 250 ms and skipped the write when it was not done: a world frame takes longer, so changed textures (the
    // planes of a video frame among them) arrived late or out of step, and the submitting thread stalled. A GPU wait
    // cannot deadlock: every value in `waits` was submitted already, and a console Wait whose signal is not submitted
    // yet is held in QueueHolds, outside the Vulkan queues, so the work behind these values completes without
    // anything submitted later.
    template<class Copy>
    VkResult write_staged_title_image(ResourceId id, const VkImageSubresourceRange& range, Copy&& copy) {
        std::unique_lock layout_lock(tracker_mutex_, std::defer_lock);
        submit_phases::lock_timed(layout_lock);
        std::unique_lock queue_lock(submit_mutex_, std::defer_lock);
        submit_phases::lock_timed(queue_lock);
        if (is_device_lost()) return VK_ERROR_DEVICE_LOST;
        LayoutPieces pieces;
        if (!tracker_.pieces(id, range, pieces) || pieces.empty()) return VK_ERROR_UNKNOWN;
        const auto waits = retire_tickets_.pending_through(submitted_ticket_.load(std::memory_order_acquire));
        const auto r = observe_result(copy(replay_, pieces, waits), "title staging upload");
        if (r == VK_SUCCESS) {
            for (const auto& p : pieces)
                if (p.layout == VK_IMAGE_LAYOUT_UNDEFINED) tracker_.set(id, p.range, VK_IMAGE_LAYOUT_GENERAL);
        }
        return r;
    }

    // Split record/submit variant: holds tracker_mutex_ during command recording while leaving submit_mutex_ unlocked
    // so presents, compiles, and cross-queue submits never stall on command recording. Acquires submit_mutex_ ONLY
    // around snapshotting cross-queue `waits` and calling `submit(waits)`.
    template<class Record, class Submit>
    VkResult write_staged_title_image(ResourceId id, const VkImageSubresourceRange& range, Record&& record,
                                      Submit&& submit) {
        std::unique_lock layout_lock(tracker_mutex_, std::defer_lock);
        submit_phases::lock_timed(layout_lock);
        if (is_device_lost()) return VK_ERROR_DEVICE_LOST;
        LayoutPieces pieces;
        if (!tracker_.pieces(id, range, pieces) || pieces.empty()) return VK_ERROR_UNKNOWN;
        VkResult r = observe_result(record(replay_, pieces), "title staging record");
        if (r == VK_SUCCESS) {
            std::unique_lock queue_lock(submit_mutex_, std::defer_lock);
            submit_phases::lock_timed(queue_lock);
            if (is_device_lost()) return VK_ERROR_DEVICE_LOST;
            const auto waits = retire_tickets_.pending_through(submitted_ticket_.load(std::memory_order_acquire));
            r = observe_result(submit(waits), "title staging upload");
        }
        if (r == VK_SUCCESS) {
            for (const auto& p : pieces)
                if (p.layout == VK_IMAGE_LAYOUT_UNDEFINED) tracker_.set(id, p.range, VK_IMAGE_LAYOUT_GENERAL);
        }
        return r;
    }

    // Atomic batched variant of write_staged_title_image: acquires tracker_mutex_ ONCE for all targets, records
    // command buffers with submit_mutex_ UNLOCKED, then acquires submit_mutex_ ONLY for snapshotting cross-queue
    // waits and submitting. Transitions UNDEFINED -> GENERAL in tracker_ ONLY if submit returns VK_SUCCESS.
    struct StagedImageTarget {
        ResourceId id{};
        VkImageSubresourceRange range{};
    };
    template<class Copy>
    VkResult write_staged_title_images(std::span<const StagedImageTarget> targets, Copy&& copy) {
        if (targets.empty()) return VK_SUCCESS;
        std::unique_lock layout_lock(tracker_mutex_, std::defer_lock);
        submit_phases::lock_timed(layout_lock);
        std::unique_lock queue_lock(submit_mutex_, std::defer_lock);
        submit_phases::lock_timed(queue_lock);
        if (is_device_lost()) return VK_ERROR_DEVICE_LOST;
        std::vector<LayoutPieces> all_pieces(targets.size());
        for (std::size_t i = 0; i < targets.size(); ++i) {
            if (!tracker_.pieces(targets[i].id, targets[i].range, all_pieces[i]) || all_pieces[i].empty())
                return VK_ERROR_UNKNOWN;
        }
        const auto waits = retire_tickets_.pending_through(submitted_ticket_.load(std::memory_order_acquire));
        const auto r = observe_result(copy(replay_, all_pieces, waits), "title staging upload batch");
        if (r == VK_SUCCESS) {
            for (std::size_t i = 0; i < targets.size(); ++i) {
                for (const auto& p : all_pieces[i])
                    if (p.layout == VK_IMAGE_LAYOUT_UNDEFINED)
                        tracker_.set(targets[i].id, p.range, VK_IMAGE_LAYOUT_GENERAL);
            }
        }
        return r;
    }

    template<class Record, class Submit>
    VkResult write_staged_title_images(std::span<const StagedImageTarget> targets, Record&& record, Submit&& submit) {
        if (targets.empty()) return VK_SUCCESS;
        std::unique_lock layout_lock(tracker_mutex_, std::defer_lock);
        submit_phases::lock_timed(layout_lock);
        if (is_device_lost()) return VK_ERROR_DEVICE_LOST;
        std::vector<LayoutPieces> all_pieces(targets.size());
        for (std::size_t i = 0; i < targets.size(); ++i) {
            if (!tracker_.pieces(targets[i].id, targets[i].range, all_pieces[i]) || all_pieces[i].empty())
                return VK_ERROR_UNKNOWN;
        }
        VkResult r = observe_result(record(replay_, all_pieces), "title staging record batch");
        if (r == VK_SUCCESS) {
            std::unique_lock queue_lock(submit_mutex_, std::defer_lock);
            submit_phases::lock_timed(queue_lock);
            if (is_device_lost()) return VK_ERROR_DEVICE_LOST;
            const auto waits = retire_tickets_.pending_through(submitted_ticket_.load(std::memory_order_acquire));
            r = observe_result(submit(waits), "title staging upload batch");
        }
        if (r == VK_SUCCESS) {
            for (std::size_t i = 0; i < targets.size(); ++i) {
                for (const auto& p : all_pieces[i])
                    if (p.layout == VK_IMAGE_LAYOUT_UNDEFINED)
                        tracker_.set(targets[i].id, p.range, VK_IMAGE_LAYOUT_GENERAL);
            }
        }
        return r;
    }

    // For tests and traces.
    VkImageLayout layout(ResourceId id, std::uint32_t subresource) {
        std::lock_guard lock(tracker_mutex_);
        return tracker_.layout(id, subresource);
    }

    bool has_resource(ResourceId id) {
        std::shared_lock lock(resource_mutex_);
        return resources_.find(id) != resources_.end();
    }

    // The registered buffer whose VA range holds `va`, preferring one that holds [va, va + bytes) as buffer_at does: its
    // ID, first VA and size in bytes. False when none does, or when `acquire(id)` returns false. `acquire` runs with
    // the registry locked, so the resource cannot finish being removed under it: it takes the caller's reference
    // (Resource's destructor removes its entry first, which waits for this lock). Used where a console call names a
    // buffer by VA only (Xbox device slot 70).
    template<class Acquire>
    bool buffer_resource_at(std::uint64_t va, std::uint64_t bytes, ResourceId& id, std::uint64_t& start,
                            std::uint64_t& size, Acquire&& acquire) {
        std::shared_lock lock(resource_mutex_);
        BufferVaIndex<ResourceId>::Hit hit{};
        if (!buffer_vas_.find(va, hit, bytes) || !acquire(hit.id)) return false;
        id = hit.id;
        start = hit.start;
        size = hit.size;
        return true;
    }

    bool ray_tracing_supported() const { return ray_tracing_ && vk.vkCmdBuildAccelerationStructuresKHR != nullptr; }
    bool xdxr_acceleration_structures() const { return xdxr_ != nullptr; }

    // Whether a build's inner nodes are BOX16 (64 bytes, half boxes rounded outward) instead of BOX32: bottom-level
    // builds with MINIMIZE_MEMORY (VK LOW_MEMORY). A top-level build keeps BOX32: it is in world space, where half
    // precision would make boxes metres too large a few kilometres from the origin. The flags of an update must match
    // its build's (D3D12), so prebuild info, the build and its refits agree. WARNING: an update that breaks that rule
    // (MINIMIZE_MEMORY on one and not the other) is refitted with the wrong layout and writes past the structure; it
    // cannot be detected when the command is recorded.
    static bool xdxr_box16(const BuildAccelerationStructure& b) {
        return !b.is_tlas && (b.build_flags & VK_BUILD_ACCELERATION_STRUCTURE_LOW_MEMORY_BIT_KHR) != 0;
    }
    // Whether a bottom-level build pairs sibling triangles into TRI0/TRI1 leaves: not when the structure may be
    // updated (ALLOW_UPDATE), since a refit rewrites the leaves one triangle each and would leave every TRI1 child
    // reading wrong vertices. xdxr_check refuses an update without ALLOW_UPDATE for that reason. Pairing changes
    // neither the layout nor the tree, so prebuild info does not depend on it.
    static bool xdxr_pair(const BuildAccelerationStructure& b) {
        return !b.is_tlas && !b.update && (b.build_flags & VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR) == 0;
    }
    // Triangles (BLAS) or instances (TLAS) a build has: what XdxrLayout is computed from, for the build and for
    // GetRaytracingAccelerationStructurePrebuildInfo.
    static std::uint32_t xdxr_elements(const BuildAccelerationStructure& b) {
        if (b.is_tlas) return b.instance_count;
        std::uint64_t n = 0;
        for (const auto& t : b.triangles) n += (t.index_type ? t.index_count : t.vertex_count) / 3;
        return n > 0xffffffffu ? 0xffffffffu : std::uint32_t(n);
    }
    // The geometries as the builder reads them, with `base` added to their VAs; false for an input it cannot read
    // (a vertex format other than R32G32B32_FLOAT or R32G32_FLOAT, an index type other than none/R16/R32).
    static bool xdxr_inputs(const BuildAccelerationStructure& b, std::vector<XdxrTriangles>* out, std::uint64_t base) {
        if (b.is_tlas) return true;
        for (const auto& t : b.triangles) {
            if (t.vertex_format != VK_FORMAT_R32G32B32_SFLOAT && t.vertex_format != VK_FORMAT_R32G32_SFLOAT) return false;
            if (t.index_type > 2) return false;
            const std::uint32_t count = (t.index_type ? t.index_count : t.vertex_count) / 3;
            if (count && !t.vertex_va) return false;
            if (count && t.index_type && !t.index_va) return false;
            if (!out) continue;
            XdxrTriangles g;
            g.vertices = t.vertex_va ? base + t.vertex_va : 0;
            g.vertex_stride = t.vertex_stride;
            g.vertex_count = t.vertex_count;
            g.xy_only = t.vertex_format == VK_FORMAT_R32G32_SFLOAT;
            g.indices = t.index_type && t.index_va ? base + t.index_va : 0;
            g.index_type = t.index_type;
            g.triangle_count = count;
            g.transform = t.transform_va ? base + t.transform_va : 0;
            g.flags = t.flags | (t.opaque ? 1u : 0u);
            out->push_back(g);
        }
        return true;
    }

    void register_acceleration_structure(std::uint64_t va, VkAccelerationStructureKHR as) {
        if (!va || !as) return;
        std::unique_lock lock(resource_mutex_);
        acceleration_structures_[va] = as;
    }

    void unregister_acceleration_structure(std::uint64_t va) {
        if (!va) return;
        std::unique_lock lock(resource_mutex_);
        acceleration_structures_.erase(va);
    }

    VkAccelerationStructureKHR resolve_acceleration_structure(std::uint64_t va) {
        std::shared_lock lock(resource_mutex_);
        const auto it = acceleration_structures_.find(va);
        return it == acceleration_structures_.end() ? VK_NULL_HANDLE : it->second;
    }

    // `prepare` is called at the start of the command buffer of every ExecuteCommandLists, before the lists are
    // written: the device brings the bindless set 0 of the translated shaders up to date there (vulkan_bindless.h)
    // and records what the set needs first. When it fails, nothing of that call is executed.
    void set_transfer_only_family(std::uint32_t family) {
        transfer_only_family_ = family;
        bump_replay_version();
    }
    void set_graphics_family(std::uint32_t family) {
        graphics_family_ = family;
        bump_replay_version();
    }
    void set_compute_copy_aspects(std::function<VkImageAspectFlags(VkFormat)> check) {
        replay_.compute_copy_aspects = std::move(check);
        bump_replay_version();
    }

    // `before` is called at the start of every submission of command lists (a held batch: when it is released),
    // before any lock here is taken: the device re-copies its slot-133 shadow buffers from the title's pages there.
    // Set once, before the first queue exists.

    // Table VAs are offsets in this device's arena. Install its scanner before creating any queues.
    // (The per-device scan is from the patch descriptor-table-refresh, part 2, of
    // Sigtryggur-Mani-Gudmundsson/graphics-runtime-patches at d9a0bc1b, GPL-3.0.)
    using TableScanner = std::function<void(std::uint64_t, std::uint32_t)>;
    void set_table_scanner(TableScanner scan) { table_scanner_ = std::move(scan); }
    void set_cpu_texture_publisher(std::function<bool()> f) { cpu_texture_publisher_ = std::move(f); }
    bool publish_cpu_textures() const { return !cpu_texture_publisher_ || cpu_texture_publisher_(); }
    void set_sample_table_resolver(std::function<bool(std::uint64_t,std::uint32_t)> f) { sample_table_resolver_ = std::move(f); }
    bool prepare_sample_table(std::uint64_t address, std::uint32_t count) const {
        return !sample_table_resolver_ || sample_table_resolver_(address,count);
    }
    // False when this context has no scanner of its own.
    bool scan_table(std::uint64_t va, std::uint32_t descriptors) const {
        if (!table_scanner_) return false;
        table_scanner_(va, descriptors);
        return true;
    }

    // `before` is called at the start of every submission of command lists (a held batch: when it is released),
    // before any lock here is taken: the device re-copies its slot-133 shadow buffers from the title's pages there.
    // Set once, before the first queue exists.
    using InputProbe = std::function<void(std::span<const CommandList* const>, std::uint64_t, const char*)>;
    void set_input_probe(InputProbe probe) { input_probe_ = std::move(probe); }
    bool diagnostic_buffer(std::uint64_t va, std::uint64_t& host, std::uint64_t& room) {
        std::shared_lock lock(resource_mutex_);
        BufferVaIndex<ResourceId>::Hit hit{};
        if (!buffer_vas_.find(va, hit)) return false;
        const auto* b = resolve(hit.id);
        if (!b || va - b->va >= b->size) return false;
        const auto offset = va - b->va;
        host = b->host ? reinterpret_cast<std::uint64_t>(b->host + offset) : 0;
        room = b->size - offset;
        return true;
    }
    const char* diagnostic_compute_name(std::uintptr_t id) {
        std::shared_lock lock(pipeline_mutex_);
        if (!device_fault_) return nullptr;
        const auto it = device_fault_->pipeline_names.find(id);
        return it == device_fault_->pipeline_names.end() ? nullptr : it->second[0];
    }
    void set_before_execute(std::function<void()> before) { before_execute_ = std::move(before); }
    // Monotonic visibility epoch bumped on ExecuteCommandLists and CPU Fence::Signal (tracked per-fence in
    // QueueHolds) so before_execute_() can coalesce concurrent or back-to-back cross-queue refreshes while
    // guaranteeing every CPU write before a batch's enqueue or a waited CPU Fence::Signal is visible.
    std::uint64_t next_visibility_seq() noexcept {
        return visibility_seq_.fetch_add(1, std::memory_order_acq_rel) + 1;
    }
    std::uint64_t current_visibility_seq() const noexcept {
        return visibility_seq_.load(std::memory_order_acquire);
    }
    // Minimum visibility epoch required by the execute_now() invoking before_execute_() on this thread (0 outside).
    static std::uint64_t current_before_execute_seq() noexcept { return t_before_execute_seq_; }
    // The page mirror's lookup for buffer_at (src/title_mirror.h): the buffer and offset that hold a title address no
    // buffer of ours holds. Set while no replay runs (device creation and destruction).
    using MirrorBuffer = std::function<bool(std::uint64_t va, std::uint64_t& size, VkBuffer& buffer, VkDeviceSize& offset)>;
    void set_mirror_buffer(MirrorBuffer lookup) { mirror_buffer_ = std::move(lookup); }
    // The page mirror bound pages in a batch that signals `value` on `semaphore`: every queue's next command-buffer
    // submission waits for it (checked under submit_mutex_ before vkQueueSubmit and record_submit_locked()). Returns
    // the retire ticket of the latest submission accepted before this bind was published, so freed homes stamped with
    // it cannot be reused while any submission that did not wait for this bind is still in flight.
    std::uint64_t note_mirror_bind(VkSemaphore semaphore, std::uint64_t value) {
        std::lock_guard lock(submit_mutex_);
        mirror_bind_semaphore_ = semaphore;
        mirror_bind_value_ = value;
        return submitted_retire_ticket();
    }
    // Reads the latest published page-mirror bind timeline semaphore and value (caller must hold submit_mutex_).
    void mirror_bind_locked(VkSemaphore& semaphore, std::uint64_t& value) const noexcept {
        semaphore = mirror_bind_semaphore_;
        value = mirror_bind_value_;
    }
    // Publishes the latest staged title-texture upload timeline (caller holds submit_mutex_): every queue's next
    // command-buffer submission waits lazily on `(semaphore, value)` in VulkanQueue::submit_multi() instead of
    // submitting an empty wait-only vkQueueSubmit per consumer queue.
    void note_title_upload_locked(VkSemaphore semaphore, std::uint64_t value) noexcept {
        title_upload_semaphore_ = semaphore;
        title_upload_value_ = value;
    }
    // Clears the published title-upload timeline before destroying `semaphore` (caller holds submit_mutex_).
    void clear_title_upload_locked(VkSemaphore semaphore) noexcept {
        if (title_upload_semaphore_ == semaphore) {
            title_upload_semaphore_ = VK_NULL_HANDLE;
            title_upload_value_ = 0;
        }
    }
    std::uint64_t replay_version() const noexcept { return replay_version_.load(std::memory_order_acquire); }
    // Called at the end of every ExecuteCommandLists replay that called before_execute_, however it returns: by then
    // the submissions it made are recorded, so submitted_retire_ticket() covers them.
    void set_after_execute(std::function<void()> after) { after_execute_ = std::move(after); }
    // Enqueue-time / finish-time serial hooks for GeometryViews::kAfterExecutes: `on_enqueue` runs synchronously inside
    // VulkanQueue::execute() BEFORE the batch enters an async worker inbox or wait-hold queue, and `on_finish` runs
    // after the batch's (or coalesced submission's) vkQueueSubmit has recorded its retire ticket (or when a queued/held
    // batch is dropped on device loss or teardown).
    void set_on_enqueue_execute(std::function<std::uint64_t()> on_enqueue) {
        on_enqueue_execute_ = std::move(on_enqueue);
    }
    void set_on_finish_execute(std::function<void(std::uint64_t)> on_finish) {
        on_finish_execute_ = std::move(on_finish);
    }
    // Called after a submission containing at least one emitted draw or dispatch with `writes_buffer == true` (or
    // legacy untyped indirect) succeeds on vkQueueSubmit, with the device's current submitted retire ticket.
    void set_on_buffer_write_submitted(std::function<void(std::uint64_t)> on_buffer_write) {
        on_buffer_write_submitted_ = std::move(on_buffer_write);
    }
    // GPU work the lists of a submission depend on, recorded at the start of its command buffer by a queue that can
    // run compute (the console heap translation pass, src/console_heap_gpu.h): `record` returns true when it recorded
    // something, and `commit` follows that submit (true: it reached the queue; the ticket is a retire ticket at or
    // after it). Every other queue's next submission then waits for that one, as for a bindless table upload: a
    // queue that records nothing itself may run lists that read what the pass wrote. Called with tracker_mutex_
    // held, so passes and those waits follow submission order. Set once, before the first queue exists.
    void set_prelude(std::function<bool(VkCommandBuffer)> record, std::function<void(bool, std::uint64_t)> commit) {
        prelude_record_ = std::move(record);
        prelude_commit_ = std::move(commit);
    }

    void set_bindless_prepare(std::function<bool(VkCommandBuffer)> prepare) {
        std::unique_lock lock(resource_mutex_);
        bindless_prepare_ = std::move(prepare);
    }

    // D24_UNORM_S8_UINT stored as D32: copies of its depth plane run a compute pass (vulkan_depth24.h). `allocate`
    // gives the scratch buffers device-local memory. Without this such a copy is refused as unsupported.
    void enable_depth24(PFN_vkGetDeviceProcAddr get, Depth24Converter::Allocate allocate,
                       std::uint64_t max_storage_range = 1u << 27) {
        std::unique_lock lock(resource_mutex_);
        depth24_ = std::make_unique<Depth24Converter>(get, device_, std::move(allocate), max_storage_range,
                                                      device_fault_.get());
        if (!depth24_->loaded()) {
            depth24_.reset();
            return;
        }
        replay_.depth24_prepare = [this](Depth24Job& job, std::vector<Depth24Job>& parts) { return depth24_->prepare(job, parts); };
        replay_.depth24_run = [this](VkCommandBuffer command_buffer, const Depth24Job& job) {
            depth24_->run(command_buffer, job);
        };
        bump_replay_version();
    }

    bool has_depth24() {
        std::shared_lock lock(resource_mutex_);
        return depth24_ != nullptr;
    }

    // Copies the replay stages through a buffer (depth <-> colour, vulkan_replay.h) take device-local scratch from
    // `allocate`. Without this such a copy is refused as unsupported. `extra_usage` is added to the chunks' transfer
    // usage, for configure_quad_indices.
    void enable_copy_scratch(PFN_vkGetDeviceProcAddr get, ScratchBuffers::Allocate allocate,
                             VkBufferUsageFlags extra_usage = 0, std::vector<std::uint32_t> families = {}) {
        std::unique_lock lock(resource_mutex_);
        scratch_ = std::make_unique<ScratchBuffers>(get, device_, std::move(allocate), extra_usage, std::move(families),
                                                    device_fault_.get());
        if (!scratch_->loaded()) scratch_.reset();
        bump_replay_version();
    }

    // Called after that command buffer's submit: true when it was accepted, false when vkQueueSubmit failed. The
    // set keeps its upload pending in the second case, because the copy never reached the GPU.
    void set_bindless_commit(std::function<void(bool)> commit) {
        std::unique_lock lock(resource_mutex_);
        bindless_commit_ = std::move(commit);
    }

    // The set 0 the commands of this ExecuteCommandLists bind, read after `prepare` succeeded and under the same
    // locks (a flush may have made a new set). Without it the pipelines' own set is bound.
    void set_bindless_current(std::function<VkDescriptorSet()> current) {
        std::unique_lock lock(resource_mutex_);
        bindless_current_ = std::move(current);
    }

    void set_bindless_release(DeferredDeletionQueue::BindlessReleaseFn fn, void* user) {
        deletion_queue_.set_bindless_release(fn, user);
    }

    // The ticket of the latest submission: everything submitted so far is at or below it, and a resource that any
    // of that work may use can be freed once completed_retire_ticket() has reached it. Tickets run over all queues,
    // each recorded with its queue's timeline value, so this waits for every queue's timeline to pass what was
    // submitted to it, and never makes a queue wait.
    std::uint64_t submitted_retire_ticket() const { return submitted_ticket_.load(std::memory_order_acquire); }

    template<typename VkHandle>
    void defer_destroy(DeferredDeletionQueue::Kind kind, VkHandle handle) {
        const auto raw = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(handle));
        if (!raw) return;
        std::lock_guard lock(submit_mutex_);
        const std::uint64_t safe_ticket = submitted_ticket_.load(std::memory_order_acquire) + 1;
        deletion_queue_.push(safe_ticket, kind, raw);
    }

    void defer_destroy(DeferredDeletionQueue::Kind kind, std::uint64_t raw_handle) {
        if (!raw_handle && kind != DeferredDeletionQueue::Kind::bindless_slot) return;
        std::lock_guard lock(submit_mutex_);
        const std::uint64_t safe_ticket = submitted_ticket_.load(std::memory_order_acquire) + 1;
        deletion_queue_.push(safe_ticket, kind, raw_handle);
    }

    // Waits (not for the device: for the queues' timelines) until every submission up to `ticket` is done, then
    // returns the completed ticket. For a caller whose list of retired objects has grown past what it will hold.
    std::uint64_t wait_retire_ticket(std::uint64_t ticket) {
        std::vector<std::pair<VkSemaphore, std::uint64_t>> pending;
        {
            std::lock_guard lock(submit_mutex_);
            if (is_device_lost()) return 0;
            pending = retire_tickets_.pending_through(ticket);
        }
        if (!pending.empty()) {
            std::vector<VkSemaphore> semaphores;
            std::vector<std::uint64_t> values;
            for (const auto& [semaphore, value] : pending) {
                semaphores.push_back(semaphore);
                values.push_back(value);
            }
            VkSemaphoreWaitInfo wait{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
            wait.semaphoreCount = std::uint32_t(semaphores.size());
            wait.pSemaphores = semaphores.data();
            wait.pValues = values.data();
            // Outside submit_mutex_: a signal may need it.
            observe_result(vk.vkWaitSemaphores(device_, &wait, UINT64_MAX), "vkWaitSemaphores (retirement)");
        }
        return completed_retire_ticket();
    }

    // The bindless flush of this ExecuteCommandLists started a new set generation, whose table upload is in its
    // command buffer: called from the `prepare` callback (tracker_mutex_ held), so execute() makes every other
    // queue's next submission wait for this one, which binds the new set.
    void note_bindless_upload() { bindless_upload_pending_ = true; }

    std::uint64_t completed_retire_ticket() {
        bool lost = false;
        std::uint64_t ticket = 0;
        {
            std::lock_guard lock(submit_mutex_);
            if (is_device_lost()) return 0;
            ticket = retire_tickets_.completed([this, &lost](VkSemaphore semaphore, std::uint64_t value) {
                std::uint64_t done = 0;
                const VkResult result = observe_result(vk.vkGetSemaphoreCounterValue(device_, semaphore, &done),
                                                       "vkGetSemaphoreCounterValue (retirement)");
                if (result == VK_ERROR_DEVICE_LOST) lost = true;
                return result == VK_SUCCESS && done >= value;
            });
        }
        // Outside submit_mutex_, as in submit(): notifying the fence dispatcher can run callbacks that submit.
        // A DEVICE_LOST seen here is the first Vulkan operation to discover it; it must not be mistaken for
        // "this ticket has not completed".
        if (lost) mark_device_lost();
        return lost ? 0 : ticket;
    }

    void drain_deleted_resources(std::uint64_t current_ticket) {
        deletion_queue_.drain(current_ticket, device_, vk);
    }

    // True when the VA names a sparse resource: the sparse pass binds its memory, not the arena's mapping, so the
    // host's bytes at that offset are not the resource's. The console's value writes skip those.
    bool is_sparse_va(std::uint64_t va) {
        const std::shared_lock lock(resource_mutex_);
        BufferVaIndex<ResourceId>::Hit hit{};
        if (!buffer_vas_.find(va, hit)) return false;
        const ResourceBinding* const b = resolve(hit.id);
        return b && b->sparse_buffer;
    }

    // Both methods are called with submit_mutex_ held. A queue's own progress timeline is
    // monotonic even when another queue completes first; global tickets are retired in order.
    void record_submit_locked(VkSemaphore progress, std::uint64_t value) {
        const auto ticket = submitted_ticket_.fetch_add(1, std::memory_order_acq_rel) + 1;
        retire_tickets_.record(ticket, progress, value);
    }
    void retire_queue_idle_locked(VkSemaphore progress) {
        retire_tickets_.queue_idle(progress);
    }

    VulkanDeviceFunctions vk;

    // Vulkan requires external synchronisation of a VkQueue; every D3D12 queue of the device may share one.
    SubmitMutex& submit_mutex() { return submit_mutex_; }

    // vkQueueSubmit through the compile gate (SubmitMutex::SubmitTurn): it waits here, not in the driver, while a
    // pipeline create is in the driver. Called with the submit lock held; the caller observes the result.
    VkResult queue_submit(VkQueue queue, std::uint32_t count, const VkSubmitInfo* submits, VkFence fence) {
        const SubmitMutex::SubmitTurn turn(submit_mutex_);
        return vk.vkQueueSubmit(queue, count, submits, fence);
    }

    void mark_device_lost(const char* where = "a Vulkan call returned VK_ERROR_DEVICE_LOST") noexcept {
        if (set_device_lost(device_lost_, where)) submit_mutex_.report_creates_on_loss();  // which creates were in it
        run_device_lost_hook();  // LEONIDA_AFTERMATH: the GPU crash dump is written before waiters wake
        if (fence_dispatcher_) {
            try {
                fence_dispatcher_->notify_device_lost();
            } catch (...) {
            }
        }
    }
    bool is_device_lost() const noexcept { return device_lost_.load(std::memory_order_acquire); }

    void set_signal_event_fn(FenceWaitDispatcher::SignalEventFn fn) {
        if (fence_dispatcher_) fence_dispatcher_->set_signal_event_fn(fn);
    }
    void enqueue_fence_wait(VkSemaphore sem, std::uint64_t value, void* event_handle) {
        if (fence_dispatcher_) fence_dispatcher_->enqueue(sem, value, event_handle);
    }
    void enqueue_fence_callback(VkSemaphore sem, std::uint64_t value, std::function<void()> cb) {
        if (fence_dispatcher_) fence_dispatcher_->enqueue_callback(sem, value, std::move(cb));
    }
    void cancel_fence_waits(VkSemaphore sem) {
        if (fence_dispatcher_) fence_dispatcher_->cancel(sem);
    }

    // Cross-queue Wait-before-Signal: per fence, the highest value a Signal was submitted for, and the queues that
    // hold batches until one is (queue_holds.h).
    QueueHolds<VkSemaphore>& holds() { return holds_; }

private:
    friend class VulkanQueue;
    friend class VulkanPresenter;

    // buffer_at finds a buffer by VA: one with no image, a VkBuffer, a VA and bytes (what the linear search it replaced
    // accepted). Both are called with resource_mutex_ held uniquely.
    static bool addressable(const ResourceBinding& b) { return !b.image && b.buffer && b.va && b.size; }
    void unindex_buffer(ResourceId id, const ResourceBinding& b) {
        if (addressable(b)) buffer_vas_.erase(b.va, id);
    }
    // resources_[id] = binding, with the VA index kept in step (a re-registered ID drops its old range first).
    void set_resource_locked(ResourceId id, const ResourceBinding& binding) {
        auto [it, inserted] = resources_.try_emplace(id, binding);
        if (addressable(binding)) {
            try {
                buffer_vas_.insert(binding.va, binding.size, id);
            } catch (...) {
                if (inserted) resources_.erase(it);
                throw;
            }
        }
        if (!inserted) {
            // A same-VA insert updates its size in place; removing the old key would remove the new range.
            if (addressable(it->second) && (!addressable(binding) || it->second.va != binding.va))
                unindex_buffer(id, it->second);
            it->second = binding;
        }
    }

    const ResourceBinding* resolve(ResourceId id) const {
        const auto it = resources_.find(id);
        if (it == resources_.end()) return nullptr;
        const auto& binding = it->second;
        if (auto* memory = binding.deferred_memory; memory && !memory->ensure()) return nullptr;
        // The caller holds resource_mutex_, so the registered owner's lifetime includes this bind.
        if (binding.ensure_residency && !binding.ensure_residency(binding.residency_owner)) return nullptr;
        if (auto* seen = binding.replay_seen; seen && !seen->load(std::memory_order_relaxed))
            seen->store(true, std::memory_order_relaxed);
        return &binding;
    }

    // Called from prepare; protected by views_mutex_ so concurrent shared_lock readers are safe.
    VkImageView view(const ResourceBinding& r, const VkImageSubresourceRange& range) {
        std::lock_guard lock(views_mutex_);
        const auto key = std::make_tuple(r.image, range.aspectMask, range.baseMipLevel, range.baseArrayLayer,
                                         range.layerCount, r.format);
        if (const auto it = views_.find(key); it != views_.end()) return it->second;
        VkImageViewCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        info.image = r.image;
        info.viewType = r.line ? VK_IMAGE_VIEW_TYPE_1D_ARRAY : VK_IMAGE_VIEW_TYPE_2D_ARRAY;  // 1D images take no 2D view
        info.format = r.format;
        info.subresourceRange = range;
        // Attachment views need only color-attachment usage. Inheriting STORAGE usage from a mutable/extended
        // resource (typeless views such as RGBA8 sRGB, or R32_UINT-backed R9G9B9E5 UAV images) would require unrelated usages.
        VkImageViewUsageCreateInfo usage{VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO};
        usage.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        if (r.typeless || (range.aspectMask == VK_IMAGE_ASPECT_COLOR_BIT && r.color_attachment))
            info.pNext = &usage;
        VkImageView view = VK_NULL_HANDLE;
        if (vk.vkCreateImageView(device_, &info, nullptr, &view) != VK_SUCCESS) return VK_NULL_HANDLE;
        views_[key] = view;
        return view;
    }

    // The render pass and framebuffer that resolve one layer of a depth-only multisampled image into one layer of
    // another: views `from` and `to` (from view()), in the layouts given. Used where the dynamic-rendering resolve
    // reads the wrong source layer (D3D-417; vulkan_replay.h says when). Both objects are cached: a pass by what it
    // describes, a framebuffer by its images and views, until remove_resource retires either image.
    VkFramebuffer depth_resolve_framebuffer(const ResourceBinding& source, VkImageView from, VkImageLayout from_layout,
                                            const ResourceBinding& target, VkImageView to, VkImageLayout to_layout,
                                            VkResolveModeFlagBits mode, VkExtent2D extent, VkRenderPass& pass) {
        std::lock_guard lock(views_mutex_);
        const auto pass_key = std::make_tuple(source.format, source.samples, from_layout, to_layout, mode);
        if (const auto it = resolve_passes_.find(pass_key); it != resolve_passes_.end()) {
            pass = it->second;
        } else {
            VkAttachmentDescription2 attachments[2]{};
            for (auto& attachment : attachments) {
                attachment.sType = VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_2;
                attachment.format = source.format;
                attachment.samples = VK_SAMPLE_COUNT_1_BIT;
                attachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;      // the source is read; the destination keeps
                attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;   // what is outside the render area
                attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;  // depth-only formats
                attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            }
            attachments[0].samples = static_cast<VkSampleCountFlagBits>(source.samples);
            attachments[0].initialLayout = attachments[0].finalLayout = from_layout;
            attachments[1].initialLayout = attachments[1].finalLayout = to_layout;
            const VkAttachmentReference2 from_reference{VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2, nullptr, 0,
                                                        from_layout, VK_IMAGE_ASPECT_DEPTH_BIT};
            const VkAttachmentReference2 to_reference{VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2, nullptr, 1, to_layout,
                                                      VK_IMAGE_ASPECT_DEPTH_BIT};
            VkSubpassDescriptionDepthStencilResolve resolve{
                VK_STRUCTURE_TYPE_SUBPASS_DESCRIPTION_DEPTH_STENCIL_RESOLVE};
            resolve.depthResolveMode = mode;
            resolve.stencilResolveMode = VK_RESOLVE_MODE_NONE;
            resolve.pDepthStencilResolveAttachment = &to_reference;
            VkSubpassDescription2 subpass{VK_STRUCTURE_TYPE_SUBPASS_DESCRIPTION_2};
            subpass.pNext = &resolve;
            subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
            subpass.pDepthStencilAttachment = &from_reference;
            VkRenderPassCreateInfo2 info{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO_2};
            info.attachmentCount = 2;
            info.pAttachments = attachments;
            info.subpassCount = 1;
            info.pSubpasses = &subpass;
            if (vk.vkCreateRenderPass2(device_, &info, nullptr, &pass) != VK_SUCCESS) return VK_NULL_HANDLE;
            resolve_passes_[pass_key] = pass;
        }
        const auto key = std::make_tuple(source.image, target.image, from, to, pass);
        if (const auto it = resolve_framebuffers_.find(key); it != resolve_framebuffers_.end()) return it->second;
        const VkImageView views[] = {from, to};
        VkFramebufferCreateInfo info{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        info.renderPass = pass;
        info.attachmentCount = 2;
        info.pAttachments = views;
        info.width = extent.width;
        info.height = extent.height;
        info.layers = 1;
        VkFramebuffer framebuffer = VK_NULL_HANDLE;
        if (vk.vkCreateFramebuffer(device_, &info, nullptr, &framebuffer) != VK_SUCCESS) return VK_NULL_HANDLE;
        resolve_framebuffers_[key] = framebuffer;
        return framebuffer;
    }

    std::uint64_t poll_retirement_locked(VkResult& result) {
        return retire_tickets_.completed([&](VkSemaphore semaphore, std::uint64_t value) {
            std::uint64_t done = 0;
            result = observe_result(vk.vkGetSemaphoreCounterValue(device_, semaphore, &done),
                                    "vkGetSemaphoreCounterValue (retirement)");
            return result == VK_SUCCESS && done >= value;
        });
    }

    void watch_queue_progress(std::stop_token stop) {
        using Watch = SubmissionProgressWatch;
        Watch watch;
        std::uint64_t retired = 0;
        std::int64_t reported_hold = -1;
        std::int64_t busy_since = -1;
        SubmitMutex::GateStats gate_logged;
        unsigned tick = 0;
        std::mutex timer_mutex;
        std::condition_variable_any timer;
        std::unique_lock timer_lock(timer_mutex);
        // LOCAL DIAGNOSTIC (post-load GPU hangs): with the device-fault report on, sample every 250 ms, so a loss the
        // driver has already reset is seen (and its checkpoints read) before a watchdog outside kills the process.
        const auto period = device_fault_ ? std::chrono::milliseconds(250) : std::chrono::milliseconds(1000);
        const unsigned per_10s = unsigned(std::chrono::seconds(10) / period);
        while (!stop.stop_requested()) {
            timer.wait_for(timer_lock, stop, period, [] { return false; });
            if (stop.stop_requested() || is_device_lost()) break;
            auto state = Watch::State::submit_lock_busy;
            VkResult result = VK_SUCCESS;
            std::uint64_t submitted;
            {
                // Keep queue timelines alive as in completed_retire_ticket(). A submit or idle wait
                // blocked inside the driver can hold this mutex indefinitely. Under heavy submission one try finds
                // it busy most of the time: a few more within a millisecond or two.
                std::unique_lock lock(submit_mutex_, std::try_to_lock);
                for (int retry = 0; !lock.owns_lock() && retry < 20; ++retry) {
                    std::this_thread::sleep_for(std::chrono::microseconds(100));
                    if (lock.try_lock()) break;
                }
                submitted = submitted_retire_ticket();
                if (lock.owns_lock()) {
                    if (is_device_lost()) break;
                    retired = poll_retirement_locked(result);
                    state = retired < submitted ? Watch::State::pending : Watch::State::idle;
                }
            }
            // LOCAL DIAGNOSTIC (early device loss): a submission unfinished for 4 s dumps the recent submissions with
            // their semaphore counters (unreadable once the device is lost); a hold of the submit lock of 3 s names
            // its thread and prints its call chain, once per hold.
            // A vkQueueSubmit stuck in the driver (measured: 10 s, then VK_ERROR_DEVICE_LOST) also blocks
            // vkGetSemaphoreCounterValue, so with the lock busy the watch stays out of the driver: the holder report
            // samples stacks only, and the counters are read once the lock is free again.
            if (state == Watch::State::submit_lock_busy) {
                const std::uint32_t owner = submit_mutex_.owner();
                const std::int64_t since = submit_mutex_.since_ms();
                const std::int64_t held = diagnostic_ms() - since;
                if (owner && held >= 3000 && since != reported_hold) {
                    reported_hold = since;
                    device_fault_->report_submit_lock_holder(owner, held);
                }
                // Busy at every sample but changing hands: contended, not stuck. Only a hold that lasts (the same
                // holder since the same time) counts toward the 15 s stall report.
                if (since != busy_since) {
                    busy_since = since;
                    state = Watch::State::idle;
                }
            } else {
                device_fault_->check_submit_progress();
            }
            if (result == VK_ERROR_DEVICE_LOST) {
                mark_device_lost();
                break;
            }
            // A failed counter query does not measure GPU progress.
            if (result != VK_SUCCESS) state = Watch::State::idle;
            // Every 10 s while it changes: what the compile gate has cost so far (SubmitMutex::CompileTurn).
            if (++tick % per_10s == 0) {
                const SubmitMutex::GateStats gate = submit_mutex_.gate_stats();
                if (gate.submits != gate_logged.submits || gate.creates != gate_logged.creates) {
                    gate_logged = gate;
                    std::fprintf(stderr, "[rt] compile-gate: t=%.0f s: %llu submissions waited for pipeline creates "
                                         "(%lld ms in all, longest %lld ms, %llu went ahead at the limit); %llu creates "
                                         "waited behind a submission (%lld ms in all, longest %lld ms)\n",
                                 double(diagnostic_ms()) / 1000.0, static_cast<unsigned long long>(gate.submits),
                                 static_cast<long long>(gate.submit_ms), static_cast<long long>(gate.submit_max_ms),
                                 static_cast<unsigned long long>(gate.ahead),
                                 static_cast<unsigned long long>(gate.creates), static_cast<long long>(gate.create_ms),
                                 static_cast<long long>(gate.create_max_ms));
                }
            }
            // LEONIDA_HANG_EXIT_S=<s> (for boots): work pending and retirement still for s seconds is a GPU hang
            // the driver may recover from without Vulkan ever reporting the device lost (Xid 109: the title then
            // waits forever, the window frozen). The stall is reported (with the checkpoints under
            // LEONIDA_STALL_CHECKPOINTS=1) and the process ends, so the boot does not sit to its time limit.
            {
                static const long hang_exit_s = [] {
                    const char* v = std::getenv("LEONIDA_HANG_EXIT_S");
                    return v && *v ? std::strtol(v, nullptr, 10) : 0l;
                }();
                static std::uint64_t last_retired = ~std::uint64_t(0);
                static auto still_since = Watch::Clock::now();
                const auto now = Watch::Clock::now();
                if (state != Watch::State::pending || retired != last_retired) {
                    last_retired = retired;
                    still_since = now;
                } else if (hang_exit_s > 0 && now - still_since >= std::chrono::seconds(hang_exit_s)) {
                    std::fprintf(stderr, "[rt] hang watchdog: queue retirement still for %ld s with work pending "
                                         "(submitted ticket %llu, retired %llu): ending the process\n",
                                 hang_exit_s, static_cast<unsigned long long>(submitted),
                                 static_cast<unsigned long long>(retired));
                    device_fault_->report_stall("hang watchdog", submitted, retired);
                    std::fflush(stderr);
                    std::_Exit(0x47);  // 'HANG' (portable immediate process termination)
                }
            }
            if (watch.sample(state, retired, Watch::Clock::now()))
                device_fault_->report_stall(state == Watch::State::pending ? "queue retirement has not advanced"
                                            : submit_mutex_.gate_waiter() && submit_mutex_.gate_waiter() == submit_mutex_.owner()
                                                ? "could not sample the submit lock (its holder waits at the compile gate)"
                                                : "could not sample the submit lock",
                                            submitted, retired);
        }
    }

    VkDevice device_;
    bool loaded_ = false;
    bool ray_tracing_ = false;
    std::unique_ptr<XdxrBuilder> xdxr_;   // set: acceleration structures are XDXR BVH4 records
    std::atomic<bool> device_lost_{false};
    alignas(64) std::shared_mutex pipeline_mutex_;  // shared for prepare/compile/emit pipeline lookups, unique for add/remove/workers
    alignas(64) std::shared_mutex resource_mutex_;  // shared for prepare lookups, unique for add/remove
    alignas(64) std::mutex tracker_mutex_;
    alignas(64) std::mutex views_mutex_;
    // Lock hierarchy (see VulkanQueue::execute): resource_mutex_ -> tracker_mutex_ -> submit_mutex_. A PSO build
    // (shims/d3d12_vulkan_device.cpp, PipelineState::build_compute / build) holds its variants_mutex_ and the
    // static big_compile mutex while it waits for the submit lock (SubmitMutex::let_submit_finish) or at the compile
    // gate: variants_mutex_ -> big_compile -> submit_mutex_, and nothing holding submit_mutex_ takes either. The
    // gate's own mutex is a leaf; a submission waits at the gate with submit_mutex_ held, for creates already in the
    // driver only (SubmitMutex::CompileTurn).
    alignas(64) SubmitMutex submit_mutex_;
    RetireTickets<VkSemaphore> retire_tickets_;
    std::atomic<std::uint64_t> submitted_ticket_{0};
    DeferredDeletionQueue deletion_queue_;
    std::unique_ptr<DeviceFaultDiagnostics> device_fault_;
    std::jthread progress_watch_;
    std::unique_ptr<FenceWaitDispatcher> fence_dispatcher_;
    QueueHolds<VkSemaphore> holds_;
    std::unordered_map<ResourceId, ResourceBinding> resources_;
    BufferVaIndex<ResourceId> buffer_vas_;  // the addressable buffers of resources_, by VA range (buffer_at)
    std::unordered_map<ResourceId, std::shared_ptr<SparseResource>> sparse_resources_;
    std::unordered_map<std::uintptr_t, ComputePipeline> pipelines_;
    std::unordered_map<std::uintptr_t, GraphicsPipeline> graphics_pipelines_;
    std::atomic<std::uint64_t> unbuilt_sync_pipelines_{0};
    std::atomic<std::uint64_t> sync_pipeline_gen_{0};
    std::atomic<std::uint32_t> warm_empty_pipeline_scans_{0};
    std::unique_ptr<PipelineWorkers> pipeline_workers_;
    BatchCompileFn batch_compile_runner_;
    std::function<std::uint64_t()> large_compile_budget_;
    std::function<std::shared_ptr<void>()> large_compile_demand_;
    mutable std::mutex wait_tracker_mutex_;
    std::vector<std::pair<VkSemaphore, std::uint64_t>> pending_waits_;
    std::atomic<std::size_t> pending_wait_count_{0};
    std::unordered_map<std::uint64_t, VkAccelerationStructureKHR> acceleration_structures_;
    LayoutTracker tracker_;
    std::mutex storage_mutex_;  // a leaf lock: guards storage_pending_ and zero_pending_ only
    std::unordered_set<ResourceId> storage_pending_;
    struct PendingZero {
        ResourceId id;
        VkBuffer buffer;
    };
    std::vector<PendingZero> zero_pending_;  // zero_on_next_submit
    // The last submission that initialized storage images (its queue's progress semaphore and ticket) and a counter
    // that says which one: every other queue's next submission waits for it, so no queue uses such an image before
    // the zero-fill that put it in GENERAL has run.
    VkSemaphore storage_init_semaphore_ = VK_NULL_HANDLE;
    std::uint64_t storage_init_value_ = 0;
    std::uint64_t storage_init_version_ = 0;
    std::map<std::tuple<VkImage, VkImageAspectFlags, std::uint32_t, std::uint32_t, std::uint32_t, VkFormat>, VkImageView> views_;
    // depth_resolve_framebuffer's caches, under views_mutex_ like views_.
    std::map<std::tuple<VkFormat, std::uint32_t, VkImageLayout, VkImageLayout, VkResolveModeFlagBits>, VkRenderPass>
        resolve_passes_;
    std::map<std::tuple<VkImage, VkImage, VkImageView, VkImageView, VkRenderPass>, VkFramebuffer> resolve_framebuffers_;

    std::function<void()> before_execute_;
    std::atomic<std::uint64_t> visibility_seq_{0};
    std::atomic<std::uint64_t> replay_version_{1};
    void bump_replay_version() noexcept { replay_version_.fetch_add(1, std::memory_order_release); }
    static inline thread_local std::uint64_t t_before_execute_seq_ = 0;  // trivially destructible (no TLS callback)
    std::function<void()> after_execute_;
    std::function<std::uint64_t()> on_enqueue_execute_;
    std::function<void(std::uint64_t)> on_finish_execute_;
    std::function<void(std::uint64_t)> on_buffer_write_submitted_;
    InputProbe input_probe_;
    TableScanner table_scanner_;
    GeometryProvider geometry_provider_;
    std::function<bool(std::uint64_t,std::uint32_t)> sample_table_resolver_;
    std::function<bool()> cpu_texture_publisher_;
    std::function<bool(VkCommandBuffer)> bindless_prepare_;
    std::function<VkDescriptorSet()> bindless_current_;
    bool bindless_upload_pending_ = false;  // tracker_mutex_
    // The last submission that carried a new generation's table upload (its queue's progress semaphore and value), and
    // a counter that says which one: every other queue's next submission waits for it (storage_mutex_).
    VkSemaphore bindless_upload_semaphore_ = VK_NULL_HANDLE;
    std::uint64_t bindless_upload_value_ = 0;
    // The page mirror's last bind batch (src/title_mirror.h): every queue's next command-buffer submission waits for
    // it under submit_mutex_, so none reads a page whose bind has not landed (submit_mutex_).
    MirrorBuffer mirror_buffer_;
    VkSemaphore mirror_bind_semaphore_ = VK_NULL_HANDLE;
    std::uint64_t mirror_bind_value_ = 0;
    // The latest staged title-texture upload timeline (d3d12_title_image_upload.h): every queue's next command-buffer
    // submission waits for it lazily under submit_mutex_ instead of queuing empty wait submits per consumer queue.
    VkSemaphore title_upload_semaphore_ = VK_NULL_HANDLE;
    std::uint64_t title_upload_value_ = 0;
    std::uint64_t bindless_upload_version_ = 0;
    std::function<bool(VkCommandBuffer)> prelude_record_;
    std::function<void(bool, std::uint64_t)> prelude_commit_;
    VkSemaphore prelude_semaphore_ = VK_NULL_HANDLE;  // the queue whose submission recorded the last prelude
    std::uint64_t prelude_value_ = 0;
    std::uint64_t prelude_version_ = 0;
    std::unique_ptr<Depth24Converter> depth24_;
    std::unique_ptr<ScratchBuffers> scratch_;  // copies staged through a buffer (depth <-> colour)
    std::function<void(bool)> bindless_commit_;
    std::unordered_map<QueryHeapId, QueryPool> query_pools_;  // element pointers stay valid across inserts
    double timestamp_period_ = 0;
    std::vector<std::uint32_t> timestamp_bits_;  // by queue family (set_timestamp_bits)
    gpu_profile::Profiler gpu_profile_;          // LEONIDA_GPU_PROFILE=1
    gpu_profile::NameMap profile_names_;         // its pipeline names, guarded by pipeline_mutex_
    bool order_timestamp_copies_ = false;
    VkPhysicalDeviceMemoryProperties occlusion_resolve_memory_{};
    label::Completions gpu_labels_;
    ReplayFunctions replay_;
    std::unique_ptr<PredicateReducer> predicate_reducer_;
    std::unique_ptr<PredicatedBufferCopier> predicated_copier_;
    std::unique_ptr<IndirectDispatchCounter> counted_dispatch_;
    std::unique_ptr<QuadIndexExpander> quad_indices_;
    std::unique_ptr<IndirectArgumentClamp> indirect_clamp_;
    std::uint32_t predicate_family_ = UINT32_MAX;
    std::uint32_t transfer_only_family_ = UINT32_MAX;
    std::uint32_t graphics_family_ = UINT32_MAX;
    DebugLabelFunctions labels_;
};
#ifdef _MSC_VER
#pragma warning(pop)
#endif

// A Vulkan timeline semaphore with D3D12 fence semantics for the host side.
class VulkanFence {
public:
    VulkanFence(VulkanContext& context, std::uint64_t initial) : context_(context) {
        VkSemaphoreTypeCreateInfo type{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
        type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        type.initialValue = initial;
        VkSemaphoreCreateInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &type};
        if (context.vk.vkCreateSemaphore(context.device(), &info, nullptr, &semaphore_) != VK_SUCCESS)
            semaphore_ = VK_NULL_HANDLE;
        if (semaphore_) context.holds().add_fence(semaphore_, initial);
    }
    ~VulkanFence() {
        if (!semaphore_) return;
        // LOCAL DIAGNOSTIC (early device loss): a fence released while a Signal the GPU has not run yet names it.
        if (!context_.is_device_lost()) {
            const std::uint64_t submitted = context_.holds().submitted_value(semaphore_);
            std::uint64_t reached = 0;
            if (submitted && context_.vk.vkGetSemaphoreCounterValue(context_.device(), semaphore_, &reached) == VK_SUCCESS &&
                reached < submitted)
                std::fprintf(stderr, "[rt] d3d12: fence semaphore %p destroyed with a Signal pending (submitted %llu, "
                                     "reached %llu) at t=%.3f s\n",
                             static_cast<void*>(semaphore_), static_cast<unsigned long long>(submitted),
                             static_cast<unsigned long long>(reached), double(diagnostic_ms()) / 1000.0);
        }
        context_.holds().remove_fence(semaphore_);  // batches held on it can never run: dropped and reported
        if (DeviceFaultDiagnostics* ring = context_.device_fault()) ring->forget(semaphore_);  // no report reads it
        context_.vk.vkDestroySemaphore(context_.device(), semaphore_, nullptr);
    }
    VulkanFence(const VulkanFence&) = delete;
    VulkanFence& operator=(const VulkanFence&) = delete;

    VkSemaphore semaphore() const { return semaphore_; }

    std::uint64_t completed() const {
        if (context_.is_device_lost()) return UINT64_MAX;
        std::uint64_t value = 0;
        const VkResult result = context_.observe_result(
            context_.vk.vkGetSemaphoreCounterValue(context_.device(), semaphore_, &value), "vkGetSemaphoreCounterValue");
        if (result == VK_ERROR_DEVICE_LOST) {
            context_.mark_device_lost();
            return UINT64_MAX;
        }
        if (result != VK_SUCCESS) return 0;
        if (value == UINT64_MAX) {
            // A lost device can show itself here first: measured on NVIDIA 621.24 and reported on 617.14, the counter
            // query succeeds with UINT64_MAX (and vkWaitSemaphores with a zero timeout succeeds too) before any call
            // has returned VK_ERROR_DEVICE_LOST. vkDeviceWaitIdle is what then says so, and it is also what a driver
            // needs to have returned before vkGetDeviceFaultInfoEXT answers (LEONIDA_DEVICE_FAULT=1). When
            // vkDeviceWaitIdle succeeds, nothing is recorded and the value is returned as the driver gave it.
            VkResult idle = VK_SUCCESS;
            {
                std::lock_guard lock(context_.submit_mutex());
                idle = context_.observe_result(context_.vk.vkDeviceWaitIdle(context_.device()),
                                               "vkDeviceWaitIdle (a fence counter of UINT64_MAX)");
            }
            if (idle == VK_ERROR_DEVICE_LOST)
                context_.mark_device_lost(
                    "a fence's counter read UINT64_MAX and vkDeviceWaitIdle then returned VK_ERROR_DEVICE_LOST");
        }
        return value;
    }

    // ID3D12Fence::Signal from the CPU. Vulkan timelines only increase: a value at or below the current one is
    // ignored (D3D12 would set it), and the caller is told.
    bool signal(std::uint64_t value) {
        // The dispatcher may be testing a failed counter query. Do not call completed() here:
        // it marks the context lost on a query error and prevents the dispatcher from waking.
        if (context_.is_device_lost()) return false;
        std::uint64_t current = 0;
        if (context_.observe_result(context_.vk.vkGetSemaphoreCounterValue(context_.device(), semaphore_, &current),
                                    "vkGetSemaphoreCounterValue") == VK_SUCCESS &&
            value <= current)
            return false;
        VkSemaphoreSignalInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO};
        info.semaphore = semaphore_;
        info.value = value;
        const VkResult result = context_.observe_result(context_.vk.vkSignalSemaphore(context_.device(), &info),
                                                        "vkSignalSemaphore");
        if (result == VK_ERROR_DEVICE_LOST) context_.mark_device_lost();
        if (result != VK_SUCCESS) return false;
        {
            QueueHolds<VkSemaphore>::WorkGuard active(context_.holds());
            context_.holds().signaled(semaphore_, value, context_.next_visibility_seq());  // queues held on this fence run with this CPU visibility epoch
            context_.holds().pump_ready();
        }
        return true;
    }

    // true when the value was reached within the timeout.
    bool wait(std::uint64_t value, std::uint64_t timeout_ns) const {
        VkSemaphoreWaitInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
        info.semaphoreCount = 1;
        info.pSemaphores = &semaphore_;
        info.pValues = &value;
        const VkResult result = context_.observe_result(
            context_.vk.vkWaitSemaphores(context_.device(), &info, timeout_ns), "vkWaitSemaphores");
        if (result == VK_ERROR_DEVICE_LOST) context_.mark_device_lost();
        return result == VK_SUCCESS;
    }

private:
    VulkanContext& context_;
    VkSemaphore semaphore_ = VK_NULL_HANDLE;
};

// One D3D12 command queue on one VkQueue. D3D12 lets one thread use a queue at a time per call and the COM wrapper
// serialises its calls; state_mutex_ protects only the queue's pending/held bookkeeping (microsecond hold time),
// while the executing_ flag serialises actual Vulkan replay, before_execute_(), and submission without holding
// state_mutex_ across them. Wait, Signal, Marker, and batches blocked on an unmet Wait append under state_mutex_
// without waiting on before_execute_() or replay; an unblocked synchronous ExecuteCommandLists waits on exec_cv_
// until executing_ clears so before_execute_() and replay still finish synchronously before returning.
//
// A Wait on a fence value no Signal has been submitted for yet holds the next batch (ExecuteCommandLists or Signal)
// on the CPU, with everything the queue is asked to do after it, instead of replaying it against layouts from before
// the work it waits for: Vulkan itself would accept the wait (timeline semaphores allow wait-before-signal), but the
// layout tracker follows replay order. Holding on the CPU also keeps a Wait and the Signal that satisfies it from
// landing on one VkQueue in the wrong order, which would never make progress on the GPU.
class VulkanQueue : private QueueHolds<VkSemaphore>::Holder {
    // RAII guard for executing_ = true. Re-acquires state_mutex_ on abnormal exit so executing_ can never stick true.
    struct ExecGuard {
        VulkanQueue* q;
        std::unique_lock<std::mutex>& state;
        bool active = true;
        void release() {
            if (!active) return;
            active = false;
            if (!state.owns_lock()) state.lock();
            q->executing_ = false;
            q->exec_cv_.notify_all();
        }
        ~ExecGuard() {
            if (!active) return;
            if (!state.owns_lock()) state.lock();
            q->restore_exec_waits_locked();
            q->set_active_seq_locked(0);
            q->executing_ = false;
            q->exec_cv_.notify_all();
        }
    };

public:
    enum class QueueKind : std::uint8_t { direct, compute, copy };

    // The most command buffers a queue holds at once. At that many, a caller that outruns the GPU waits for the oldest
    // submitted one to finish (before taking any context lock, see wait_for_room) instead of allocating without bound. The
    // cap holds while the GPU is slow too: a wait that times out is retried, in kQueuedWaitNs slices so a device lost
    // meanwhile ends it, and a device lost makes execute() submit nothing.
    static constexpr std::size_t kMaxQueuedCommandBuffers = 64;
    static constexpr std::uint64_t kQueuedWaitNs = 5'000'000'000;
    static constexpr std::uint64_t kUnsetEpoch = UINT64_MAX;

    static constexpr VkPipelineStageFlags2 queue_supported_stages(QueueKind kind) noexcept {
        switch (kind) {
        case QueueKind::copy:
            return VK_PIPELINE_STAGE_2_COPY_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        case QueueKind::compute:
            return VK_PIPELINE_STAGE_2_COPY_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT |
                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        case QueueKind::direct:
        default:
            return VK_PIPELINE_STAGE_2_COPY_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT |
                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT;
        }
    }

    VulkanQueue(VulkanContext& context, VkQueue queue, std::uint32_t family, QueueKind kind = QueueKind::direct)
        : context_(context), queue_(queue), kind_(kind), family_(family) {
        auto& vk = context.vk;
        VkCommandPoolCreateInfo pool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pool.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pool.queueFamilyIndex = family;
        ok_ = vk.vkCreateCommandPool(context.device(), &pool, nullptr, &pool_) == VK_SUCCESS;
        VkSemaphoreTypeCreateInfo type{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
        type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        VkSemaphoreCreateInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &type};
        ok_ = ok_ && vk.vkCreateSemaphore(context.device(), &info, nullptr, &progress_) == VK_SUCCESS;
        try {
            context_.holds().attach(this);
        } catch (...) {
            // A failed registration leaves no queue to run the destructor for these Vulkan handles.
            if (progress_) vk.vkDestroySemaphore(context.device(), progress_, nullptr);
            if (pool_) vk.vkDestroyCommandPool(context.device(), pool_, nullptr);
            throw;
        }
    }

    ~VulkanQueue() {
        auto& vk = context_.vk;
        stop_async();  // LEONIDA_SUBMIT_ASYNC: what is queued runs first
        std::fprintf(stderr, "[rt] d3d12: queue %p (progress %p) destroyed at t=%.3f s\n", static_cast<void*>(queue_),
                     static_cast<void*>(progress_), double(diagnostic_ms()) / 1000.0);
        context_.holds().detach(this);  // no thread releases into this queue from here on
        {
            std::unique_lock state(state_mutex_);
            exec_cv_.wait(state, [this] { return !executing_; });
        }
        report_held_at_teardown();
        {
            std::lock_guard state(state_mutex_);
            if (!waits_.empty()) {
                context_.note_waits_removed(waits_, wait_values_);
                waits_.clear();
                wait_values_.clear();
                wait_pins_.clear();
            }
        }
        context_.holds().pump_ready();  // presents queued behind what was just dropped can run
        if (queue_) {
            std::lock_guard lock(context_.submit_mutex());
            const SubmitMutex::SubmitTurn turn(context_.submit_mutex(), false);  // a queue operation: the compile gate
            for (; open_labels_; --open_labels_) context_.labels_.queue_end(queue_);
            const VkResult idle_result = context_.observe_result(vk.vkQueueWaitIdle(queue_), "vkQueueWaitIdle");
            if (idle_result == VK_SUCCESS) {
                if (progress_) context_.retire_queue_idle_locked(progress_);
            } else {
                // Any failure here makes every fence read ~0 from now on: say so when it is not a real loss.
                if (idle_result != VK_ERROR_DEVICE_LOST)
                    std::fprintf(stderr, "[rt] d3d12: vkQueueWaitIdle at queue teardown returned %d; the device is "
                                         "treated as lost\n", int(idle_result));
                context_.mark_device_lost();
            }
        }
        if (progress_) {
            std::lock_guard lock(context_.tracker_mutex_);
            if (context_.depth24_) context_.depth24_->forget(progress_);  // its scratch is idle now
            if (context_.scratch_) context_.scratch_->forget(progress_);
            if (context_.indirect_clamp_) context_.indirect_clamp_->forget(progress_);
        }
        if (progress_) {
            // This queue is idle, so whatever initialisation it published (storage images, zero-filled buffers) has
            // finished: the next submission on another queue must not wait on a semaphore that is about to go.
            std::lock_guard lock(context_.storage_mutex_);
            if (context_.storage_init_semaphore_ == progress_) context_.storage_init_semaphore_ = VK_NULL_HANDLE;
            // The same for the bindless table upload: a queue that had not seen its version yet would otherwise
            // wait on this destroyed semaphore in its next vkQueueSubmit.
            if (context_.bindless_upload_semaphore_ == progress_) context_.bindless_upload_semaphore_ = VK_NULL_HANDLE;
        }
        if (pool_) vk.vkDestroyCommandPool(context_.device(), pool_, nullptr);
        if (progress_) {
            if (DeviceFaultDiagnostics* ring = context_.device_fault()) ring->forget(progress_);  // no report reads it
            vk.vkDestroySemaphore(context_.device(), progress_, nullptr);
        }
        destroy_khr_stages(khr_stage_pool_);
        destroy_khr_as_queries(khr_as_query_pool_);
        for (auto& slot : buffers_) {
            destroy_khr_stages(slot.khr_instances);
            destroy_khr_as_queries(slot.khr_as_queries);
            free_events_.insert(free_events_.end(), slot.events.begin(), slot.events.end());
        }
        for (VkEvent event : free_events_) vk.vkDestroyEvent(context_.device(), event, nullptr);
    }

    VulkanQueue(const VulkanQueue&) = delete;
    VulkanQueue& operator=(const VulkanQueue&) = delete;

    bool ok() const { return ok_; }
    // A queue of the runtime's own whose caller waits for each call (a staging copy): never on a worker.
    void make_synchronous() { synchronous_ = true; }
    // Skip context_.before_execute_() (and its paired after_execute_(), on_enqueue_execute_ serials, non-shader
    // bindless/prelude prep, and submit_phases submission accounting) for internal runtime queues such as
    // WriteToSubresource/ReadFromSubresource staging copies that never read console heaps, shadows, or GeometryViews.
    void set_skip_before_execute(bool skip = true) { skip_before_execute_.store(skip, std::memory_order_release); }
    bool skip_before_execute() const { return skip_before_execute_.load(std::memory_order_acquire); }
    // Override LEONIDA_SUBMIT_ASYNC depth for this queue before its first queued call (used by unit tests).
    void set_async_depth(unsigned depth) {
        std::lock_guard lock(inbox_mutex_);
        if (!worker_.joinable()) async_depth_ = depth;
    }
    // Async worker telemetry snapshots for unit tests.
    std::uint64_t async_run_ns_for_test() {
        std::lock_guard lock(inbox_mutex_);
        return async_stats_.run_ns + async_pump_ns_.load(std::memory_order_relaxed);
    }
    std::uint64_t async_room_waits_for_test() {
        std::lock_guard lock(inbox_mutex_);
        return async_stats_.room_waits;
    }

    // Entries kept between executions. Execute only uses the first lists.size(); a larger count is retained storage,
    // not work the next submission will replay (vk_queue_test checks both halves of that contract).
    std::size_t prepared_entries() const { return prepared_.size(); }

    template<typename GetHeap>
    bool update_tiles(ResourceId id, std::span<const TileRegion> regions,
                      std::span<const TileRange> ranges, GetHeap&& get_heap) {
        flush_async();  // after the calls queued before it (LEONIDA_SUBMIT_ASYNC)
        bool ok = false;
        QueueHolds<VkSemaphore>::WorkGuard work_guard;
        std::vector<std::shared_ptr<void>> retired;
        {
            std::unique_lock state(state_mutex_);
            exec_cv_.wait(state, [this] { return !executing_; });
            work_guard.arm(context_.holds());
            executing_ = true;
            set_active_seq_locked(context_.holds().next_seq());
            ExecGuard exec_guard{this, state};
            take_waits_to_exec_locked();
            state.unlock();

            ok = [&]() -> bool {
                std::unique_lock lock(context_.resource_mutex_);
                const auto it = context_.sparse_resources_.find(id);
                if (it == context_.sparse_resources_.end()) return false;
                const auto& sparse = it->second;
                for (const auto& region : regions)
                    if (!valid_tiles(*sparse, region)) return false;
                const auto heap = get_heap(sparse->image != VK_NULL_HANDLE, sparse->memory_types);

                std::map<std::uint64_t, TileMapping> changes;
                std::size_t region_index = 0;
                std::uint64_t consumed = 0;
                for (const auto& range : ranges) {
                    const bool backed = range.mode == TileRangeMode::sequential || range.mode == TileRangeMode::reuse;
                    const std::uint64_t heap_count = heap ? heap->size / tile_size_in_bytes : 0;
                    const std::uint64_t required = range.mode == TileRangeMode::reuse ? 1 : range.count;
                    if (backed && (!heap || range.first > heap_count || required > heap_count - range.first)) return false;
                    for (std::uint64_t i = 0; i < range.count;) {
                        if (region_index == regions.size()) return false;
                        const auto& region = regions[region_index];
                        const auto count = std::min(range.count - i, region.count - consumed);
                        const auto first = region.first + consumed;
                        if (range.mode == TileRangeMode::unmap) {
                            const auto end = first + count;
                            for (auto old = sparse->mapped.lower_bound(first); old != sparse->mapped.end() && old->first < end; ++old)
                                changes[old->first] = {};
                            for (auto change = changes.lower_bound(first); change != changes.end() && change->first < end; ++change)
                                change->second = {};
                        } else if (backed) {
                            for (std::uint64_t j = 0; j < count; ++j) {
                                const auto page = range.first + (range.mode == TileRangeMode::sequential ? i + j : 0);
                                changes[first + j] = {heap, heap->offset + page * tile_size_in_bytes};
                            }
                        }
                        i += count;
                        consumed += count;
                        if (consumed == region.count) {
                            ++region_index;
                            consumed = 0;
                        }
                    }
                }
                return region_index == regions.size() && bind_tiles(id, sparse, changes);
            }();

            state.lock();
            restore_exec_waits_locked();
            flush_exec_pins_locked(retired);
            set_active_seq_locked(0);
            if (!async_started_.load(std::memory_order_acquire) || t_worker_of_ == this)
                drain_held_while_executing(state, retired);
            exec_guard.release();
            drain_pins(retired);
        }
        retired.clear();
        context_.holds().pump_ready();
        return ok;
    }

    bool copy_tile_mappings(ResourceId dst, std::span<const TileRegion> dst_regions, ResourceId src,
                            std::span<const TileRegion> src_regions) {
        flush_async();
        bool ok = false;
        QueueHolds<VkSemaphore>::WorkGuard work_guard;
        std::vector<std::shared_ptr<void>> retired;
        {
            std::unique_lock state(state_mutex_);
            exec_cv_.wait(state, [this] { return !executing_; });
            work_guard.arm(context_.holds());
            executing_ = true;
            set_active_seq_locked(context_.holds().next_seq());
            ExecGuard exec_guard{this, state};
            take_waits_to_exec_locked();
            state.unlock();

            ok = [&]() -> bool {
                std::unique_lock lock(context_.resource_mutex_);
                const auto source = context_.sparse_resources_.find(src), target = context_.sparse_resources_.find(dst);
                if (source == context_.sparse_resources_.end() || target == context_.sparse_resources_.end() ||
                    bool(source->second->image) != bool(target->second->image) || src_regions.size() != dst_regions.size())
                    return false;
                // Read the complete source mapping before changing any destination slot, including copies within one resource.
                std::map<std::uint64_t, TileMapping> changes;
                for (std::size_t row = 0; row < src_regions.size(); ++row) {
                    const auto from = src_regions[row], to = dst_regions[row];
                    if (!valid_tiles(*source->second, from) || !valid_tiles(*target->second, to) || from.count != to.count)
                        return false;
                    for (std::uint64_t i = 0; i < from.count; ++i) {
                        const auto page = source->second->mapped.find(from.first + i);
                        changes.emplace(to.first + i, page == source->second->mapped.end() ? TileMapping{} : page->second);
                    }
                }
                return bind_tiles(dst, target->second, changes);
            }();

            state.lock();
            restore_exec_waits_locked();
            flush_exec_pins_locked(retired);
            set_active_seq_locked(0);
            if (!async_started_.load(std::memory_order_acquire) || t_worker_of_ == this)
                drain_held_while_executing(state, retired);
            exec_guard.release();
            drain_pins(retired);
        }
        retired.clear();
        context_.holds().pump_ready();
        return ok;
    }

    // ExecuteCommandLists: every list is checked before any is written, so a failure submits nothing and leaves
    // the layout tracker as it was. Pending Wait()s are attached to this submission. When one of them (or an
    // earlier wait-blocked held operation) requires holding, the batch is appended to held_ under state_mutex_
    // without blocking on before_execute_() or replay, and the call succeeds: the batch is replayed in strict FIFO
    // order once the wait is satisfied, which is also where its errors are reported (stderr, and take_held_failure()).
    // Otherwise, if another thread is currently executing or draining ready held ops, this call waits on exec_cv_
    // (with state_mutex_ unlocked) and runs before_execute_() and execute_now() synchronously before returning.
    //
    // `pin`, called only when the batch is held or queued asynchronously, returns what has to stay alive until the
    // batch has run: the COM layer passes the pipeline states and root signatures the lists named (the application
    // may release them and Reset the lists as soon as the call returns).
    using PinFn = std::function<std::shared_ptr<void>()>;

private:
    // Move-only RAII token for an enqueue-time execute serial (GeometryViews::execute_enqueue / execute_end).
    // Lives across inbox_, held_, and coalesced execute_now() submissions, calling on_finish_execute_ only after the
    // submission's retire ticket has been recorded (or when a queued/held batch is dropped).
    struct EnqueueSerial {
        VulkanContext* context;
        std::uint64_t value;
        EnqueueSerial() noexcept : context(nullptr), value(0) {}
        EnqueueSerial(VulkanContext* ctx, std::uint64_t v) noexcept : context(ctx), value(v) {}
        EnqueueSerial(EnqueueSerial&& o) noexcept
            : context(std::exchange(o.context, nullptr)), value(std::exchange(o.value, 0)) {}
        EnqueueSerial& operator=(EnqueueSerial&& o) noexcept {
            if (this != &o) {
                finish();
                context = std::exchange(o.context, nullptr);
                value = std::exchange(o.value, 0);
            }
            return *this;
        }
        EnqueueSerial(const EnqueueSerial&) = delete;
        EnqueueSerial& operator=(const EnqueueSerial&) = delete;
        ~EnqueueSerial() { finish(); }
        void finish() noexcept {
            if (value && context && context->on_finish_execute_) {
                const std::uint64_t v = std::exchange(value, 0);
                context->on_finish_execute_(v);
            } else {
                value = 0;
            }
        }
    };

    EnqueueSerial make_enqueue_serial() {
        return (!skip_before_execute_.load(std::memory_order_acquire) && context_.on_enqueue_execute_)
                   ? EnqueueSerial(&context_, context_.on_enqueue_execute_())
                   : EnqueueSerial{};
    }

public:
    ReplayResult execute(std::span<const CommandList* const> lists, const PinFn& pin = {}) {
        bool all_empty = true;
        for (const CommandList* list : lists) {
            if (list && !list->commands().empty()) {
                all_empty = false;
                break;
            }
        }
        const bool skip_before = skip_before_execute_.load(std::memory_order_acquire);
        if (!skip_before) pipeline_wait::note_submission();  // big compute warm-ups are left to their first use while the title submits
        const std::uint64_t vis_seq = (skip_before || all_empty) ? 0 : context_.next_visibility_seq();
        auto* diagnostics = context_.device_fault();
        const auto execute_call = diagnostics ? diagnostics->next_execute_call() : 0;
        EnqueueSerial enqueue_serial = make_enqueue_serial();
        if (async_on()) {  // copied and queued: the worker replays it (a failure is reported as a held one's)
            Call call;
            call.kind = Call::Kind::execute;
            call.execute_call = execute_call;
            call.visibility_seq = vis_seq;
            call.enqueue_serial = std::move(enqueue_serial);
            call.lists.reserve(lists.size());
            for (const CommandList* list : lists) call.lists.push_back(std::make_shared<const CommandList>(*list));
            if (pin) call.pin = pin();
            enqueue(std::move(call));
            return ReplayResult::success;
        }
        return execute_sync(lists, pin, execute_call, std::move(enqueue_serial), vis_seq);
    }

    // Zero-copy ExecuteCommandLists for immutable closed CommandList snapshots (used by the COM layer so Reset()
    // allocates fresh storage only while a submitted snapshot is still in flight, avoiding deep copies on submission).
    ReplayResult execute(std::span<const std::shared_ptr<const CommandList>> lists, const PinFn& pin = {}) {
        bool all_empty = true;
        for (const auto& list : lists) {
            if (list && !list->commands().empty()) {
                all_empty = false;
                break;
            }
        }
        const bool skip_before = skip_before_execute_.load(std::memory_order_acquire);
        if (!skip_before) pipeline_wait::note_submission();
        const std::uint64_t vis_seq = (skip_before || all_empty) ? 0 : context_.next_visibility_seq();
        auto* diagnostics = context_.device_fault();
        const auto execute_call = diagnostics ? diagnostics->next_execute_call() : 0;
        EnqueueSerial enqueue_serial = make_enqueue_serial();
        if (async_on()) {
            Call call;
            call.kind = Call::Kind::execute;
            call.execute_call = execute_call;
            call.visibility_seq = vis_seq;
            call.enqueue_serial = std::move(enqueue_serial);
            call.lists.assign(lists.begin(), lists.end());
            if (pin) call.pin = pin();
            enqueue(std::move(call));
            return ReplayResult::success;
        }
        const CommandList* small_ptrs[16];
        std::vector<const CommandList*> large_ptrs;
        const CommandList* const* ptr_data = nullptr;
        if (lists.size() <= 16) {
            for (std::size_t i = 0; i < lists.size(); ++i) small_ptrs[i] = lists[i].get();
            ptr_data = small_ptrs;
        } else {
            large_ptrs.reserve(lists.size());
            for (const auto& list : lists) large_ptrs.push_back(list.get());
            ptr_data = large_ptrs.data();
        }
        return execute_sync_impl(
            {ptr_data, lists.size()}, pin, execute_call,
            [&lists] { return std::vector<std::shared_ptr<const CommandList>>(lists.begin(), lists.end()); },
            {}, std::move(enqueue_serial), false, kUnsetEpoch, vis_seq);
    }

    ReplayResult execute_sync(std::span<const CommandList* const> lists, const PinFn& pin, std::uint64_t execute_call,
                              EnqueueSerial enqueue_serial = {}, std::uint64_t vis_seq = 0) {
        return execute_sync_impl(
            lists, pin, execute_call,
            [&lists] {
                std::vector<std::shared_ptr<const CommandList>> copies;
                copies.reserve(lists.size());
                for (const CommandList* list : lists) copies.push_back(std::make_shared<const CommandList>(*list));
                return copies;
            },
            {}, std::move(enqueue_serial), false, kUnsetEpoch, vis_seq);
    }

    // ID3D12CommandQueue::Signal: the fence reaches `value` after all work submitted so far. Held behind earlier
    // held operations (or an active execution), so the fence does not run ahead of the work it follows. `pin`
    // keeps the fence object alive until the submission has completed, as D3D12 does.
    // GPU-only queue Signals do NOT bump visibility_seq_ (any CPU writes made prior to ExecuteCommandLists were
    // already covered by that execute's visibility_seq; CPU-to-GPU edges are tracked per-fence on Fence::Signal).
    bool signal(VkSemaphore fence, std::uint64_t value, const PinFn& pin = {}) {
        if (async_on()) {
            Call call;
            call.kind = Call::Kind::signal;
            call.fence = fence;
            call.value = value;
            call.visibility_seq = 0;
            if (pin) call.pin = pin();
            enqueue(std::move(call));
            return true;
        }
        return signal_sync(fence, value, pin, {}, kUnsetEpoch, 0);
    }

    bool signal_sync(VkSemaphore fence, std::uint64_t value, const PinFn& pin, std::shared_ptr<void> async_pin = {},
                     std::uint64_t async_present_epoch = kUnsetEpoch, std::uint64_t vis_seq = 0) {
        bool ok = true;
        QueueHolds<VkSemaphore>::WorkGuard work_guard;
        std::vector<std::shared_ptr<void>> retired;
        {
            std::unique_lock state(state_mutex_);
            if (executing_ || must_hold()) {
                Deferred op;
                op.kind = Deferred::Kind::signal;
                op.present_epoch = async_present_epoch;
                op.visibility_seq = vis_seq;
                op.fence = fence;
                op.value = value;
                if (async_pin) op.pins.push_back(std::move(async_pin));
                else if (pin) op.pins.push_back(pin());
                hold(std::move(op));
            } else {
                work_guard.arm(context_.holds());
                executing_ = true;
                set_active_seq_locked(async_seq_ ? async_seq_ : context_.holds().next_seq());
                ExecGuard exec_guard{this, state};
                take_waits_to_exec_locked();
                state.unlock();

                ok = submit(VK_NULL_HANDLE, fence, value, nullptr, vis_seq);
                // As for a held Signal (release_held): the fence object, and so its semaphore, stays alive until the
                // GPU has run the signal. Destroying a semaphore a pending submission signals is undefined
                // (VUID-vkDestroySemaphore-semaphore-05149), and a title may release a fence right after Signal.
                if (ok) {
                    std::shared_ptr<void> keep = async_pin ? std::move(async_pin) : (pin ? pin() : nullptr);
                    if (keep) exec_ticketed_pins_.push_back({submitted_, std::move(keep)});
                } else if (async_pin) {
                    exec_retired_pins_.push_back(std::move(async_pin));
                }

                state.lock();
                restore_exec_waits_locked();
                flush_exec_pins_locked(retired);
                set_active_seq_locked(0);
                drain_held_while_executing(state, retired);
                exec_guard.release();
            }
            drain_pins(retired);
        }
        retired.clear();
        context_.holds().pump_ready();  // this Signal, its active_seq_ completion, or a drain may unblock waiters
        return ok;
    }

    // ID3D12CommandQueue::Wait: later submissions wait for the fence. A wait in a Vulkan batch only holds back that
    // batch, so it is kept and attached to the next submission (held on the CPU if no Signal reaches it yet).
    // `pin` returns what keeps the fence object alive until then (a D3D12 queue holds its fences).
    void wait(VkSemaphore fence, std::uint64_t value, const PinFn& pin = {}) {
        if (async_on()) {
            Call call;
            call.kind = Call::Kind::wait;
            call.fence = fence;
            call.value = value;
            if (pin) call.pin = pin();
            context_.note_wait_added(fence, value);
            enqueue(std::move(call));
            return;
        }
        wait_sync(fence, value, pin);
    }

    void wait_sync(VkSemaphore fence, std::uint64_t value, const PinFn& pin, std::shared_ptr<void> async_pin = {},
                   bool already_tracked = false) {
        if (!already_tracked) context_.note_wait_added(fence, value);
        std::vector<std::shared_ptr<void>> retired;  // fences whose submission has completed: released unlocked
        {
            std::lock_guard state(state_mutex_);
            waits_.push_back(fence);
            wait_values_.push_back(value);
            if (async_pin) wait_pins_.push_back(std::move(async_pin));
            else if (pin) wait_pins_.push_back(pin());
            drain_pins(retired);
        }
    }

    // The first failure of a held ExecuteCommandLists since the last call (what the call would have returned had it
    // not been held), or success. A Signal behind a failed batch still signals, as when the failure is not held: the
    // work did not run, the application is told here and on stderr, and nothing waits forever.
    ReplayResult take_held_failure() {
        return static_cast<ReplayResult>(held_failure_.exchange(static_cast<int>(ReplayResult::success)));
    }

    // ID3D12CommandQueue::SetMarker/BeginEvent/EndEvent: queue labels, between the submissions around them. An end
    // with no begin open on this queue is dropped, as Vulkan requires a matching begin. No-ops without the extension.
    void marker(DebugMarker::Kind kind, const char* label) {
        if (!context_.debug_labels()) return;
        if (async_on()) {
            Call call;
            call.kind = Call::Kind::marker;
            call.marker_kind = kind;
            call.label = label ? label : "";
            enqueue(std::move(call));
            return;
        }
        marker_sync(kind, label);
    }

    void marker_sync(DebugMarker::Kind kind, const char* label, std::uint64_t async_present_epoch = kUnsetEpoch) {
        if (!context_.debug_labels()) return;
        bool ran = false;
        QueueHolds<VkSemaphore>::WorkGuard work_guard;
        std::vector<std::shared_ptr<void>> retired;
        {
            std::unique_lock state(state_mutex_);
            if (executing_ || !held_.empty()) {  // behind active/held work, so behind it in the queue's labels too
                Deferred op;
                op.kind = Deferred::Kind::marker;
                op.present_epoch = async_present_epoch;
                op.marker_kind = kind;
                op.label = label ? label : "";
                hold(std::move(op));
            } else {
                work_guard.arm(context_.holds());
                executing_ = true;
                set_active_seq_locked(async_seq_ ? async_seq_ : context_.holds().next_seq());
                ExecGuard exec_guard{this, state};
                state.unlock();

                marker_now(kind, label);

                state.lock();
                flush_exec_pins_locked(retired);
                set_active_seq_locked(0);
                drain_held_while_executing(state, retired);
                exec_guard.release();
                ran = true;
            }
            drain_pins(retired);
        }
        retired.clear();
        if (ran) context_.holds().pump_ready();
    }

    bool idle() {
        flush_async();
        {
            std::unique_lock state(state_mutex_);
            exec_cv_.wait(state, [this] { return !executing_; });
        }
        std::lock_guard lock(context_.submit_mutex());
        const SubmitMutex::SubmitTurn turn(context_.submit_mutex(), false);  // a queue operation: the compile gate
        return context_.observe_result(context_.vk.vkQueueWaitIdle(queue_), "vkQueueWaitIdle") == VK_SUCCESS;
    }

    // Waits on the CPU until what this queue has submitted so far is done: its own timeline, without the device's
    // submit lock, so other queues keep submitting, and without waiting for anything submitted after (review D9;
    // idle() waits for the whole VkQueue, which D3D12 queues may share, and blocks every submission meanwhile).
    bool wait_submitted() {
        const std::uint64_t value = submitted_gpu_.load(std::memory_order_acquire);
        if (!value) return true;
        VkSemaphoreWaitInfo wait{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
        wait.semaphoreCount = 1;
        wait.pSemaphores = &progress_;
        wait.pValues = &value;
        constexpr std::uint64_t slice_ns = 100'000'000;  // 100 ms
        for (;;) {  // bounded slices: a lost device wakes the wait
            if (context_.is_device_lost()) return false;
            const VkResult res = context_.vk.vkWaitSemaphores(context_.device(), &wait, slice_ns);
            if (res == VK_SUCCESS) return true;
            if (res == VK_TIMEOUT) continue;
            if (res == VK_ERROR_DEVICE_LOST) context_.mark_device_lost();
            return false;
        }
    }

private:
    // Completed stages, reused by later ARRAY_OF_POINTERS builds. Only execute_now() touches the pool, and it is
    // serialized by executing_ (without holding state_mutex_), so no extra lock is needed.
    std::vector<KhrInstanceStage> khr_stage_pool_;
    std::vector<KhrAsQueryPool> khr_as_query_pool_;
    std::vector<TitleVaAliases::Mapping> xdxr_ranges_;  // stage_xdxr_table's snapshot, kept for its capacity

    void destroy_khr_stages(std::vector<KhrInstanceStage>& stages) {
        for (const auto& stage : stages) {
            if (stage.memory) context_.vk.vkFreeMemory(context_.device(), stage.memory, nullptr);
            if (stage.buffer) context_.vk.vkDestroyBuffer(context_.device(), stage.buffer, nullptr);
        }
        stages.clear();
    }

    // A stage returns to the pool only after its submission has completed, so rewriting it cannot race the GPU.
    void recycle_khr_stages(std::vector<KhrInstanceStage>& stages) {
        for (auto& stage : stages) khr_stage_pool_.push_back(std::move(stage));
        stages.clear();
    }

    void destroy_khr_as_queries(std::vector<KhrAsQueryPool>& pools) {
        if (context_.vk.vkDestroyQueryPool) {
            for (const auto& entry : pools) {
                if (entry.pool) context_.vk.vkDestroyQueryPool(context_.device(), entry.pool, nullptr);
            }
        }
        pools.clear();
    }

    void recycle_khr_as_queries(std::vector<KhrAsQueryPool>& pools) {
        for (auto& entry : pools) {
            entry.used = 0;
            khr_as_query_pool_.push_back(entry);
        }
        pools.clear();
    }

    bool allocate_khr_as_queries(std::uint32_t count, std::vector<KhrAsQueryPool>& active, VkQueryPool& out_pool,
                                 std::uint32_t& out_first) {
        if (!count || !context_.vk.vkCreateQueryPool || !context_.vk.vkDestroyQueryPool) return false;
        if (!active.empty()) {
            auto& back = active.back();
            if (back.used <= back.capacity && count <= back.capacity - back.used) {
                out_pool = back.pool;
                out_first = back.used;
                back.used += count;
                return true;
            }
        }
        for (auto it = khr_as_query_pool_.begin(); it != khr_as_query_pool_.end(); ++it) {
            if (it->capacity >= count) {
                KhrAsQueryPool entry = *it;
                khr_as_query_pool_.erase(it);
                entry.used = count;
                active.push_back(entry);
                out_pool = entry.pool;
                out_first = 0;
                return true;
            }
        }
        const std::uint32_t capacity = std::max<std::uint32_t>(count, 16u);
        VkQueryPoolCreateInfo info{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        info.queryType = VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR;
        info.queryCount = capacity;
        VkQueryPool pool = VK_NULL_HANDLE;
        if (context_.vk.vkCreateQueryPool(context_.device(), &info, nullptr, &pool) != VK_SUCCESS || !pool)
            return false;
        active.push_back({pool, capacity, count});
        out_pool = pool;
        out_first = 0;
        return true;
    }

    void destroy_stage(KhrInstanceStage& stage) {
        if (stage.memory) context_.vk.vkFreeMemory(context_.device(), stage.memory, nullptr);
        if (stage.buffer) context_.vk.vkDestroyBuffer(context_.device(), stage.buffer, nullptr);
        stage = {};
    }

    // A host-visible, device-addressable stage of at least `bytes`: a completed one of the same usage from the pool
    // when one is large enough (a per-frame TLAS rebuild then reuses the same buffer and memory instead of making a
    // new vkAllocateMemory call every frame), a new one otherwise. False, with `stage` empty, when none can be made.
    bool acquire_stage(std::uint64_t bytes, VkBufferUsageFlags usage, KhrInstanceStage& stage) {
        const auto& funcs = context_.vk;
        stage = {};
        if (!funcs.vkCreateBuffer || !funcs.vkDestroyBuffer || !funcs.vkGetBufferMemoryRequirements ||
            !funcs.vkAllocateMemory || !funcs.vkFreeMemory || !funcs.vkBindBufferMemory || !bytes)
            return false;
        for (auto it = khr_stage_pool_.begin(); it != khr_stage_pool_.end(); ++it) {
            if (it->size >= bytes && it->usage == usage) {
                stage = std::move(*it);
                khr_stage_pool_.erase(it);
                return true;
            }
        }
        VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        info.size = bytes;
        info.usage = usage;
        if (funcs.vkCreateBuffer(context_.device(), &info, nullptr, &stage.buffer) != VK_SUCCESS) {
            stage = {};
            return false;
        }
        VkMemoryRequirements requirements{};
        funcs.vkGetBufferMemoryRequirements(context_.device(), stage.buffer, &requirements);
        const auto& memory = context_.occlusion_resolve_memory_;
        constexpr VkMemoryPropertyFlags host_flags =
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        std::uint32_t type = memory.memoryTypeCount;
        for (std::uint32_t i = 0; i < memory.memoryTypeCount; ++i) {
            if (!(requirements.memoryTypeBits & (1u << i))) continue;
            if ((memory.memoryTypes[i].propertyFlags & host_flags) == host_flags) {
                type = i;
                break;
            }
        }
        VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
        flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &flags};
        allocation.allocationSize = requirements.size;
        allocation.memoryTypeIndex = type;
        if (type == memory.memoryTypeCount ||
            funcs.vkAllocateMemory(context_.device(), &allocation, nullptr, &stage.memory) != VK_SUCCESS ||
            funcs.vkBindBufferMemory(context_.device(), stage.buffer, stage.memory, 0) != VK_SUCCESS) {
            destroy_stage(stage);
            return false;
        }
        stage.size = bytes;
        stage.usage = usage;
        return true;
    }

    // The table an XDXR TLAS build translates its instances' BLAS addresses with (xdxr_build.comp blas_address):
    // every range of the title's memory the layer reaches through a copy (src/title_va_alias.h), as 24-byte entries
    // {title VA, end, shader VA} in title-VA order, staged like the KHR pointer lists and kept until the submission
    // completes. `count` 0 and `address` 0 when there are no ranges, or the stage could not be made.
    ReplayResult stage_xdxr_table(std::vector<KhrInstanceStage>& stages, VkDeviceAddress& address,
                                  std::uint32_t& count) {
        const auto& funcs = context_.vk;
        address = 0;
        count = 0;
        // Without a table the build still runs: a BLAS in the title's memory is then unreachable (an instance with
        // no BLAS), never read through heap_base + its title address.
        const auto none = [] {
            static std::atomic<bool> logged{false};
            if (!logged.exchange(true))
                std::fprintf(stderr, "[rt] d3d12: could not stage the title-memory table of an XDXR TLAS build; its "
                                     "instances in the title's memory get no BLAS\n");
            return ReplayResult::success;
        };
        // A member, not thread_local: a C++ thread_local with a destructor in gta6rt.exe adds TLS callbacks to the
        // image the loader's TLS index-0 move must not have (pe_loader.cpp reserve_tls_index0).
        auto& ranges = xdxr_ranges_;
        title_va_aliases().mappings(ranges);
        if (ranges.empty()) return ReplayResult::success;
        if (ranges.size() > 0xffffffffull / 24 || !funcs.vkMapMemory || !funcs.vkUnmapMemory ||
            !funcs.vkGetBufferDeviceAddress)
            return none();
        const std::uint64_t bytes = 24ull * ranges.size();
        KhrInstanceStage stage{};
        if (!acquire_stage(bytes, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                           stage))
            return none();
        void* mapped = nullptr;
        if (funcs.vkMapMemory(context_.device(), stage.memory, 0, bytes, 0, &mapped) != VK_SUCCESS) {
            destroy_stage(stage);
            return none();
        }
        auto* out = static_cast<std::uint8_t*>(mapped);
        for (const auto& range : ranges) {
            const std::uint64_t entry[3] = {range.title_va, range.end, range.shader_va};
            std::memcpy(out, entry, sizeof(entry));
            out += sizeof(entry);
        }
        funcs.vkUnmapMemory(context_.device(), stage.memory);
        if (!stage.address) {
            VkBufferDeviceAddressInfo info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
            info.buffer = stage.buffer;
            stage.address = funcs.vkGetBufferDeviceAddress(context_.device(), &info);
        }
        if (!stage.address) {
            destroy_stage(stage);
            return none();
        }
        address = stage.address;
        count = std::uint32_t(ranges.size());
        stages.push_back(std::move(stage));
        return ReplayResult::success;
    }

    // A KHR TLAS built with D3D12_ELEMENTS_LAYOUT_ARRAY_OF_POINTERS holds console VAs in its pointer list; Vulkan
    // reads those 8-byte entries as device addresses. Stage a translated copy in our own host-visible buffer instead
    // of editing the guest's input. The caller keeps the stage until the submission that builds with it completes.
    ReplayResult stage_khr_instances(const ReplayFunctions& vk, const BuildAccelerationStructure& build,
                                     std::vector<KhrInstanceStage>& stages, VkDeviceAddress& address) {
        const auto& funcs = context_.vk;
        if (!funcs.vkCreateBuffer || !funcs.vkDestroyBuffer || !funcs.vkGetBufferMemoryRequirements ||
            !funcs.vkAllocateMemory || !funcs.vkFreeMemory || !funcs.vkBindBufferMemory || !funcs.vkMapMemory ||
            !funcs.vkUnmapMemory || !funcs.vkGetBufferDeviceAddress || !vk.host_at || !vk.buffer_at ||
            !build.instance_count)
            return queue_unsupported(__LINE__);
        const std::uint64_t bytes = 8ull * build.instance_count;
        const std::uint8_t* guest = nullptr;
        const HostReadStatus status = vk.host_at(build.instances_va, bytes, guest);
        if (status == HostReadStatus::not_cpu_written) return queue_unsupported(__LINE__);
        if (status != HostReadStatus::ok) return ReplayResult::invalid_copy;

        KhrInstanceStage stage{};
        const auto discard = [&] { destroy_stage(stage); };
        if (!acquire_stage(bytes,
                           VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                               VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                           stage))
            return queue_unsupported(__LINE__);

        void* mapped = nullptr;
        if (funcs.vkMapMemory(context_.device(), stage.memory, 0, bytes, 0, &mapped) != VK_SUCCESS) {
            discard();
            return queue_unsupported(__LINE__);
        }
        // Validate and translate in one pass straight into the staging buffer; no second vector of addresses.
        for (std::uint32_t i = 0; i < build.instance_count; ++i) {
            std::uint64_t va = 0;
            std::memcpy(&va, guest + 8ull * i, sizeof(va));
            VkBuffer buffer = VK_NULL_HANDLE;
            VkDeviceSize offset = 0;
            std::uint64_t room = 64;  // one D3D12_RAYTRACING_INSTANCE_DESC
            // No overflow check on heap_base + va: a buffer outside the arena whose device address lies below the
            // arena's (an UPLOAD heap in system memory without Resizable BAR) has a VA that wraps (address - base),
            // and the unsigned sum wraps back to its device address, as the replay's other VA translations do.
            if (!va || !vk.buffer_at(va, room, buffer, offset) || room < 64) {
                funcs.vkUnmapMemory(context_.device(), stage.memory);
                discard();
                return ReplayResult::invalid_copy;
            }
            const std::uint64_t device_address = vk.heap_base + va;
            std::memcpy(static_cast<std::uint8_t*>(mapped) + 8ull * i, &device_address, sizeof(device_address));
        }
        funcs.vkUnmapMemory(context_.device(), stage.memory);
        VkBufferDeviceAddressInfo device_address{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
        device_address.buffer = stage.buffer;
        stage.address = funcs.vkGetBufferDeviceAddress(context_.device(), &device_address);
        if (!stage.address) {
            discard();
            return queue_unsupported(__LINE__);
        }
        stages.push_back(std::move(stage));
        address = stages.back().address;
        return ReplayResult::success;
    }

    struct PendingSignal {
        VkSemaphore fence = VK_NULL_HANDLE;
        std::uint64_t value = 0;
        std::uint64_t visibility_seq = 0;
    };

    struct Depth24Guard {
        Depth24Converter* d24 = nullptr;
        VkSemaphore progress = VK_NULL_HANDLE;
        bool active = false;
        ~Depth24Guard() {
            if (active && d24) d24->retire(progress, 0);
        }
    };
    struct ScratchGuard {
        ScratchBuffers* scratch = nullptr;
        VkSemaphore progress = VK_NULL_HANDLE;
        bool active = false;
        ~ScratchGuard() {
            if (active && scratch) scratch->retire(progress, 0);
        }
    };

    // Maximum CPU Fence::Signal visibility epoch among this submission's explicit user waits (0 if none waited on a
    // CPU-signaled fence).
    std::uint64_t exec_waits_vis_seq() const {
        std::uint64_t max_seq = 0;
        const std::size_t n = (std::min)(exec_user_wait_count_, exec_waits_.size());
        for (std::size_t i = 0; i < n; ++i) {
            const std::uint64_t s = context_.holds().signaled_vis_seq(exec_waits_[i], exec_wait_values_[i]);
            if (s > max_seq) max_seq = s;
        }
        return max_seq;
    }

    // Rebuilds cached_replay_ from context_.replay_ only when context_.replay_version() changes (typically once per
    // queue lifetime), avoiding copying 27 std::functions and allocating helper lambdas on every execute_now().
    // Caller holds shared resource_mutex_.
    void ensure_cached_replay_locked(bool transfer_only, bool can_resolve_occlusion) {
        const std::uint64_t ver = context_.replay_version();
        if (cached_replay_version_ == ver) return;
        cached_replay_ = context_.replay_;
        cached_replay_on_buffer_write_ = cached_replay_.on_buffer_write;
        cached_replay_on_buffer_write_ctx_ = cached_replay_.on_buffer_write_context;
        if (context_.geometry_provider_) {
            cached_replay_.title_memory = [this](std::uint64_t va, std::uint64_t bytes, VkBuffer& buffer,
                                                 VkDeviceSize& offset) {
                if (!va || !bytes || bytes > UINT64_MAX - va) return false;
                for (const auto& upload : geometry_uploads_) {
                    if (va < upload.source || va - upload.source > upload.size ||
                        bytes > upload.size - (va - upload.source))
                        continue;
                    buffer = upload.buffer;
                    offset = upload.offset + (va - upload.source);
                    return true;
                }
                VulkanContext::GeometryUpload upload;
                if (!context_.geometry_provider_(va, bytes, upload)) return false;
                buffer = upload.buffer;
                offset = upload.offset;
                geometry_uploads_.push_back(std::move(upload));
                return true;
            };
        }
        cached_replay_.graphics_family = family_ == context_.graphics_family_;
        cached_replay_.queue_stages = kind_ == QueueKind::copy      ? detail::queue_stages_copy
                                      : kind_ == QueueKind::compute ? detail::queue_stages_compute
                                                                    : 0;
        cached_replay_.prepare_khr_instances = [this](const BuildAccelerationStructure& build,
                                                      VkDeviceAddress& address) {
            return stage_khr_instances(cached_replay_, build, *exec_replay_state_.khr_stages, address);
        };
        cached_replay_.prepare_xdxr_table = [this](VkDeviceAddress& address, std::uint32_t& count) {
            return stage_xdxr_table(*exec_replay_state_.khr_stages, address, count);
        };
        if (cached_replay_.write_acceleration_structures_properties) {
            cached_replay_.prepare_khr_postbuild = [this](std::uint32_t count, VkQueryPool& pool,
                                                          std::uint32_t& first) {
                return allocate_khr_as_queries(count, *exec_replay_state_.khr_as_queries, pool, first);
            };
        }
        if (kind_ != QueueKind::copy && context_.predicate_reducer_ && family_ == context_.predicate_family_) {
            cached_replay_.prepare_predicate = [this](VkBuffer buffer, VkDeviceSize offset, PredicateCapture& out) {
                return (*exec_replay_state_.pred_batch)->prepare(buffer, offset, out);
            };
            cached_replay_.capture_predicate = [this](VkCommandBuffer cb, const PredicateCapture& p) {
                context_.predicate_reducer_->capture(cb, p);
            };
            if (context_.predicated_copier_) {
                cached_replay_.prepare_predicated_copy = [this](VkBuffer source, VkBuffer destination, VkDeviceSize src,
                                                                VkDeviceSize dst, VkDeviceSize size,
                                                                std::vector<PredicatedCopyDispatch>& out) {
                    return (*exec_replay_state_.copy_batch)->prepare(source, destination, src, dst, size, out);
                };
                cached_replay_.emit_predicated_copy = [this](VkCommandBuffer cb,
                                                             const std::vector<PredicatedCopyDispatch>& jobs) {
                    context_.predicated_copier_->emit(cb, jobs);
                };
            }
        }
        if (context_.scratch_) {
            cached_replay_.copy_scratch = [this](VkDeviceSize bytes, VkBuffer& buffer, VkDeviceSize& offset) {
                auto& tlock = *exec_replay_state_.tracker_lock;
                if (!tlock.owns_lock()) submit_phases::lock_timed(tlock);
                auto& guard = *exec_replay_state_.scratch_guard;
                if (!guard.active) {
                    guard.scratch = context_.scratch_.get();
                    context_.scratch_->collect();
                    guard.active = true;
                }
                return context_.scratch_->get(bytes, buffer, offset);
            };
        }
        if (kind_ != QueueKind::copy && context_.indirect_clamp_) {
            // IndirectArgumentClamp manages its own internal locks and per-Batch scratch chunks: prepare() runs
            // concurrently across queues without acquiring context_.tracker_mutex_.
            cached_replay_.prepare_indirect_clamp = [this](VkBuffer arguments, VkDeviceSize argument_offset,
                                                           VkBuffer count, VkDeviceSize count_offset,
                                                           std::uint32_t kind, std::uint32_t stride,
                                                           std::uint32_t max_count, ClampedIndirect& out) {
                auto& cb = *exec_replay_state_.clamp_batch;
                if (!cb) cb = context_.indirect_clamp_->batch();
                return cb->prepare(arguments, argument_offset, count, count_offset, kind, stride, max_count, out);
            };
        }
        if (kind_ != QueueKind::copy && context_.counted_dispatch_) {
            cached_replay_.prepare_counted_dispatch = [this](VkBuffer args, VkDeviceSize args_offset, VkBuffer count,
                                                             VkDeviceSize count_offset, std::uint32_t stride,
                                                             std::uint32_t maximum, CountedDispatch& out) {
                auto& tlock = *exec_replay_state_.tracker_lock;
                if (!tlock.owns_lock()) submit_phases::lock_timed(tlock);
                auto& cb = *exec_replay_state_.counted_batch;
                if (!cb) cb = context_.counted_dispatch_->batch();
                return cb->prepare(args, args_offset, count, count_offset, stride, maximum, out);
            };
            cached_replay_.emit_counted_dispatch = [this](VkCommandBuffer cb, const CountedDispatch& job) {
                context_.counted_dispatch_->emit(cb, job);
            };
        }
        if (kind_ != QueueKind::copy && context_.quad_indices_ && context_.scratch_) {
            cached_replay_.prepare_quad_indices = [this](VkBuffer indices, VkDeviceSize offset, VkIndexType type,
                                                         std::uint64_t readable, std::uint32_t quads, VkBuffer scratch,
                                                         VkDeviceSize scratch_offset, QuadIndexExpansion& out) {
                auto& tlock = *exec_replay_state_.tracker_lock;
                if (!tlock.owns_lock()) submit_phases::lock_timed(tlock);
                return context_.quad_indices_->prepare(indices, offset, type, readable, quads, scratch, scratch_offset,
                                                       out);
            };
            cached_replay_.emit_quad_indices = [this](VkCommandBuffer cb, const QuadIndexExpansion& job) {
                context_.quad_indices_->emit(cb, job);
            };
        }
        if (context_.depth24_ && !transfer_only) {
            cached_replay_.depth24_prepare = [this](Depth24Job& job, std::vector<Depth24Job>& parts) {
                auto& tlock = *exec_replay_state_.tracker_lock;
                if (!tlock.owns_lock()) submit_phases::lock_timed(tlock);
                auto& guard = *exec_replay_state_.depth24_guard;
                if (!guard.active) {
                    guard.d24 = context_.depth24_.get();
                    context_.depth24_->collect();
                    guard.active = true;
                }
                return context_.depth24_->prepare(job, parts);
            };
        }
        if (can_resolve_occlusion) {
            cached_replay_.sum_occlusion = [this](VkCommandBuffer cb, VkQueryPool pool, std::uint32_t first,
                                                  std::uint32_t count, VkBuffer dst, VkDeviceSize offset) {
                occlusion_resolve_->sum(cb, pool, first, count, dst, offset);
            };
            cached_replay_.resolve_binary_occlusion = [this](VkCommandBuffer cb, VkQueryPool pool, std::uint32_t first,
                                                             std::uint32_t count, VkBuffer dst, VkDeviceSize offset) {
                occlusion_resolve_->normalize(cb, pool, first, count, dst, offset);
            };
            cached_replay_.sum_statistics = [this](VkCommandBuffer cb, VkQueryPool pool, VkQueryPool mesh_pool,
                                                   VkQueryPool primitives_pool, const std::uint8_t* mesh,
                                                   std::uint32_t first, std::uint32_t count, VkBuffer dst,
                                                   VkDeviceSize offset) {
                occlusion_resolve_->sum_statistics(cb, pool, mesh_pool, primitives_pool, mesh, first, count, dst,
                                                   offset);
            };
            cached_replay_.sum_binary_occlusion = [this](VkCommandBuffer cb, VkQueryPool pool, std::uint32_t first,
                                                         std::uint32_t count, VkBuffer dst, VkDeviceSize offset) {
                occlusion_resolve_->sum_binary(cb, pool, first, count, dst, offset);
            };
        }
        const auto& events = context_.vk;
        if (context_.order_timestamp_copies() && events.vkCreateEvent && events.vkDestroyEvent && events.vkResetEvent &&
            events.vkCmdSetEvent && events.vkCmdWaitEvents) {
            cached_replay_.order_timestamp_copy = [this](VkCommandBuffer cb) {
                const VkEvent event = take_event();
                if (!event) return;
                exec_replay_state_.slot->events.push_back(event);
                const auto& vk = context_.vk;
                vk.vkCmdSetEvent(cb, event, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
                vk.vkCmdWaitEvents(cb, 1, &event, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                   VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, nullptr, 0, nullptr, 0, nullptr);
                *exec_replay_state_.ordered_timestamps = true;
            };
        }
        cached_replay_version_ = ver;
    }

    ReplayResult execute_now(std::span<const CommandList* const> lists, std::uint64_t execute_call,
                             std::span<const std::size_t> batch_breaks = {},
                             std::span<const PendingSignal> trailing_signals = {},
                             std::uint64_t visibility_seq = 0, bool /*from_held*/ = false) {
        const bool skip_before = skip_before_execute_.load(std::memory_order_acquire);
        // The "submit frame" line and the floor: this submission from before its first phase to after its last.
        std::optional<submit_phases::Submission> submission;
        if (!skip_before) submission.emplace(int(kind_));
        {
            std::optional<submit_phases::Scope> waited;
            if (!skip_before) waited.emplace(submit_phases::wait_room);
            if (context_.is_device_lost() || !wait_for_room()) return queue_unsupported(__LINE__);
        }
        std::size_t total_commands = 0;
        std::uint64_t captures = 0;
        bool runs_shaders = false;
        bool has_rt_ds_clear = false;
        for (const auto* list : lists) {
            if (!list) continue;
            const auto& cmds = list->commands();
            total_commands += cmds.size();
            if (list->closed()) {
                captures += list->predication_captures();
                runs_shaders = runs_shaders || list->runs_shaders();
                has_rt_ds_clear = has_rt_ds_clear || list->has_rt_ds_clear();
            } else {
                for (const auto& command : cmds) {
                    if (std::holds_alternative<ClearRenderTarget>(command) ||
                        std::holds_alternative<ClearDepthStencil>(command))
                        has_rt_ds_clear = true;
                    if (const auto* p = std::get_if<Predication>(&command); p && p->buffer) ++captures;
                    runs_shaders |= std::holds_alternative<Dispatch>(command) || draw_command(command).state ||
                                    std::holds_alternative<ExecuteIndirect>(command);
                }
            }
        }
        const bool run_before_after = !skip_before && total_commands != 0;
        if (run_before_after && context_.before_execute_) {
            std::uint64_t req_seq = visibility_seq ? visibility_seq : context_.current_visibility_seq();
            const std::uint64_t wait_vis = (std::max)(pending_wait_vis_seq_, exec_waits_vis_seq());
            if (wait_vis > req_seq) req_seq = wait_vis;
            struct BeforeSeqGuard {
                std::uint64_t prev;
                explicit BeforeSeqGuard(std::uint64_t seq) noexcept : prev(VulkanContext::t_before_execute_seq_) {
                    VulkanContext::t_before_execute_seq_ = seq;
                }
                ~BeforeSeqGuard() { VulkanContext::t_before_execute_seq_ = prev; }
            } before_seq_guard{req_seq};
            context_.before_execute_();
            pending_wait_vis_seq_ = 0;
        } else if (!skip_before) {
            // Empty-command batch skipped before_execute_(): NEVER clear pending_wait_vis_seq_, and carry forward any
            // CPU Fence::Signal visibility edges from this batch's waits (including coalesced completed waits).
            const std::uint64_t wait_vis = (std::max)(visibility_seq, exec_waits_vis_seq());
            pending_wait_vis_seq_ = (std::max)(pending_wait_vis_seq_, wait_vis);
        }
        struct AfterExecute {
            VulkanContext& context;
            bool active;
            ~AfterExecute() {
                if (active && context.after_execute_) context.after_execute_();
            }
        } after_execute{context_, run_before_after};
        // The rest of this call is the replay phase; the submission is counted when it returns, however it returns.
        struct Replayed {
            bool active;
            std::int64_t start;
            explicit Replayed(bool a) : active(a), start(a ? submit_phases::now_ns() : 0) {}
            ~Replayed() {
                if (!active) return;
                submit_phases::add(submit_phases::replay, std::uint64_t(submit_phases::now_ns() - start));
                submit_phases::submitted();
            }
        } replayed{!skip_before};
        if (!skip_before && context_.input_probe_)
            context_.input_probe_(lists, execute_call, kind_ == QueueKind::direct ? "direct" :
                                                      kind_ == QueueKind::compute ? "compute" : "copy");
        if (kind_ != QueueKind::direct && has_rt_ds_clear) return queue_unsupported(__LINE__);
        const bool transfer_only = family_ == context_.transfer_only_family_;
        const bool skip_shader_prep = transfer_only || (skip_before && !runs_shaders);
        // Fast path for empty command-list batches (0 commands across all lists, no GPU profiler): skips copying the
        // 25-member std::function ReplayFunctions table, constructing per-pass helper lambdas, pipeline_mutex_,
        // prepare(), PipelineCompiles, and QueryReplay, while preserving the D3D12 boundary barrier, bindless flushes,
        // zero_pending_ fills, prelude recording, cross-queue semaphore waits, and timeline retirement.
        if (total_commands == 0 && !gpu_profile::enabled()) {
            std::shared_lock resource_lock(context_.resource_mutex_, std::defer_lock);
            if (!skip_shader_prep) submit_phases::lock_timed(resource_lock);
            std::function<void(bool)> empty_bindless_commit =
                !skip_shader_prep ? context_.bindless_commit_ : std::function<void(bool)>{};
            CommandSlot* slot = acquire();
            if (!slot) return queue_unsupported(__LINE__);
            VkCommandBuffer command_buffer = slot->buffer;
            const auto discard = [&] {
                if (context_.vk.vkResetCommandBuffer(command_buffer, 0) == VK_SUCCESS) slot->ticket = 0;
            };
            VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            context_.gpu_labels_.begin_record(command_buffer);
            if (context_.observe_result(context_.vk.vkBeginCommandBuffer(command_buffer, &begin),
                                        "vkBeginCommandBuffer") != VK_SUCCESS) {
                discard();
                return queue_unsupported(__LINE__);
            }
            if (context_.vk.vkCmdPipelineBarrier) {
                VkMemoryBarrier full{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
                full.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
                full.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
                context_.vk.vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                                 VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &full, 0, nullptr, 0,
                                                 nullptr);
            }
            std::unique_lock tracker_lock(context_.tracker_mutex_, std::defer_lock);
            submit_phases::lock_timed(tracker_lock);
            if (!skip_shader_prep && context_.bindless_prepare_ && !context_.bindless_prepare_(command_buffer)) {
                const bool ended = context_.observe_result(context_.vk.vkEndCommandBuffer(command_buffer),
                                                           "vkEndCommandBuffer") == VK_SUCCESS;
                if (resource_lock.owns_lock()) resource_lock.unlock();
                const bool submitted = ended && submit(command_buffer, VK_NULL_HANDLE, 0);
                if (!submitted) discard();
                if (empty_bindless_commit) empty_bindless_commit(submitted);
                tracker_lock.unlock();
                return queue_unsupported(__LINE__);
            }
            if (resource_lock.owns_lock()) resource_lock.unlock();
            const bool bindless_uploaded =
                !skip_shader_prep && std::exchange(context_.bindless_upload_pending_, false);
            auto& zero_buffers = zero_buffers_scratch_;
            zero_buffers.clear();
            {
                std::lock_guard lock(context_.storage_mutex_);
                if (context_.storage_init_semaphore_ && context_.storage_init_semaphore_ != progress_ &&
                    context_.storage_init_version_ != storage_init_seen_) {
                    exec_waits_.push_back(context_.storage_init_semaphore_);
                    exec_wait_values_.push_back(context_.storage_init_value_);
                    storage_init_seen_ = context_.storage_init_version_;
                }
                if (!skip_shader_prep) {
                    if (context_.bindless_upload_semaphore_ && context_.bindless_upload_semaphore_ != progress_ &&
                        context_.bindless_upload_version_ != bindless_upload_seen_) {
                        exec_waits_.push_back(context_.bindless_upload_semaphore_);
                        exec_wait_values_.push_back(context_.bindless_upload_value_);
                    }
                    bindless_upload_seen_ = context_.bindless_upload_version_;
                }
                zero_buffers.swap(context_.zero_pending_);
            }
            for (const auto& zero : zero_buffers)
                context_.vk.vkCmdFillBuffer(command_buffer, zero.buffer, 0, VK_WHOLE_SIZE, 0);
            if (!zero_buffers.empty()) {
                VkMemoryBarrier filled{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
                filled.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                filled.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
                context_.vk.vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                                 VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &filled, 0, nullptr, 0,
                                                 nullptr);
            }
            const bool prelude = !skip_shader_prep && context_.prelude_record_ && context_.prelude_record_(command_buffer);
            static const bool cross_wait_empty = [] {
                const char* v = std::getenv("LEONIDA_PRELUDE_QUEUE_WAIT");
                return !(v && v[0] == '0' && v[1] == 0);
            }();
            if (!skip_shader_prep) {
                std::lock_guard lock(context_.storage_mutex_);
                if (cross_wait_empty && context_.prelude_semaphore_ && context_.prelude_semaphore_ != progress_ &&
                    context_.prelude_version_ != prelude_seen_) {
                    exec_waits_.push_back(context_.prelude_semaphore_);
                    exec_wait_values_.push_back(context_.prelude_value_);
                }
                prelude_seen_ = context_.prelude_version_;
            }
            CheckpointSubmission diagnostics;
            if (execute_call) {
                const char* queue_kind = kind_ == QueueKind::direct ? "direct" :
                                         kind_ == QueueKind::compute ? "compute" : "copy";
                diagnostics = context_.device_fault()->begin_submission(execute_call, queue_, queue_kind, lists.size());
            }
            const bool ended = context_.observe_result(context_.vk.vkEndCommandBuffer(command_buffer),
                                                       "vkEndCommandBuffer") == VK_SUCCESS;
            const bool submitted = ended && submit_multi(command_buffer, trailing_signals,
                                                         execute_call ? &diagnostics : nullptr);
            if (!submitted && !zero_buffers.empty()) {
                std::lock_guard lock(context_.storage_mutex_);
                context_.zero_pending_.insert(context_.zero_pending_.end(), zero_buffers.begin(), zero_buffers.end());
            }
            if (submitted && !zero_buffers.empty()) {
                std::lock_guard lock(context_.storage_mutex_);
                context_.storage_init_semaphore_ = progress_;
                context_.storage_init_value_ = submitted_;
                ++context_.storage_init_version_;
                storage_init_seen_ = context_.storage_init_version_;
            }
            if (submitted && bindless_uploaded) {
                std::lock_guard lock(context_.storage_mutex_);
                context_.bindless_upload_semaphore_ = progress_;
                context_.bindless_upload_value_ = submitted_;
                ++context_.bindless_upload_version_;
                bindless_upload_seen_ = context_.bindless_upload_version_;
            }
            if (prelude) {
                if (submitted) {
                    std::lock_guard lock(context_.storage_mutex_);
                    context_.prelude_semaphore_ = progress_;
                    context_.prelude_value_ = submitted_;
                    ++context_.prelude_version_;
                    prelude_seen_ = context_.prelude_version_;
                }
                if (context_.prelude_commit_) context_.prelude_commit_(submitted, context_.submitted_retire_ticket());
            }
            if (!submitted) discard();
            if (empty_bindless_commit) empty_bindless_commit(submitted);
            tracker_lock.unlock();
            return submitted ? ReplayResult::success : queue_unsupported(__LINE__);
        }
        VkCommandBuffer command_buffer = VK_NULL_HANDLE;
        struct KhrGuard {
            VulkanQueue* queue;
            std::vector<KhrInstanceStage> stages;
            std::vector<KhrAsQueryPool> as_queries;
            ~KhrGuard() {
                queue->recycle_khr_stages(stages);
                queue->recycle_khr_as_queries(as_queries);
            }
        } khr_guard{this};
        std::function<void(bool)> bindless_commit;
        // Lock hierarchy: executing_ (serializes this queue without holding state_mutex_) -> pipeline_mutex_ ->
        // resource_mutex_ -> tracker_mutex_ -> submit_mutex_.
        // Shared pipeline_lock keeps registered pipelines, pipeline_names, and pipeline_workers_ alive across
        // preparation, compilation, and emit. Shared resource_lock allows concurrent prepare() across queues and is
        // released while unbuilt pipelines compile so add_resource()/remove_resource() never stall behind shader
        // compiles; tracker_mutex_ is acquired lazily during prepare() only if a list uses a shared helper, and
        // otherwise right before bindless_prepare_/emit() through submit() so bindless flushes, layout emits, and
        // submissions stay in order.
        std::shared_lock pipeline_lock(context_.pipeline_mutex_, std::defer_lock);
        submit_phases::lock_timed(pipeline_lock);
        std::shared_lock resource_lock(context_.resource_mutex_, std::defer_lock);
        std::unique_lock tracker_lock(context_.tracker_mutex_, std::defer_lock);
        // Declared after tracker_lock so any early return after depth24_->prepare() retires uncommitted scratch/pools
        // with ticket 0 while tracker_lock is still held.
        Depth24Guard depth24_guard{nullptr, progress_, false};
        ScratchGuard scratch_guard{nullptr, progress_, false};
        // Kept between executions: prepare refills each list's entries in the storage the last execution left, where
        // making them fresh for every command was about half the layer's CPU cost per command (replay_cpu_bench).
        auto& prepared = prepared_;
        // Keep the storage from a larger batch. resize() down would destroy the unused entries and their vectors, so
        // a queue that sees 8 lists and then 1 in every frame would rebuild them on each execution. The loops below
        // use only the first lists.size() entries, so stale entries are never prepared, queried or emitted.
        if (prepared.size() < lists.size()) prepared.resize(lists.size());
        unsigned mesh_draws = 0;
        auto& geometry = geometry_uploads_;
        geometry.clear();
        std::unique_ptr<PredicateReducer::Batch> batch;
        std::unique_ptr<PredicatedBufferCopier::Batch> copy_batch;
        std::unique_ptr<IndirectDispatchCounter::Batch> counted_batch;
        std::unique_ptr<IndirectArgumentClamp::Batch> clamp_batch;
        bool ordered_timestamps = false;
        exec_replay_state_ = {&tracker_lock, &khr_guard.stages, &khr_guard.as_queries, &batch,        &copy_batch,
                              &counted_batch, &clamp_batch,     &depth24_guard,        &scratch_guard, nullptr,
                              &ordered_timestamps};
        struct ExecReplayStateGuard {
            VulkanQueue* q;
            ~ExecReplayStateGuard() {
                q->exec_replay_state_ = {};
                q->cached_replay_.compiles = nullptr;
                q->cached_replay_.mesh_draw_count = nullptr;
                q->cached_replay_.bindless_set = VK_NULL_HANDLE;
                q->cached_replay_.gpu_profile = nullptr;
                q->cached_replay_.on_buffer_write = q->cached_replay_on_buffer_write_;
                q->cached_replay_.on_buffer_write_context = q->cached_replay_on_buffer_write_ctx_;
            }
        } exec_replay_guard{this};
        const auto& occlusion_memory = context_.occlusion_resolve_memory_;
        const bool can_resolve_occlusion = occlusion_memory.memoryTypeCount != 0;
        const auto resolve = [this](ResourceId id) { return context_.resolve(id); };
        std::uint32_t segments = 0, statistics_segments = 0;
        bool has_queries = false;
        const bool has_parallel_compiler = bool(context_.batch_compile_runner_) || bool(context_.pipeline_workers_);
        const auto run_compiles = [this](std::size_t count, const std::function<void(std::size_t)>& job) {
            if (context_.batch_compile_runner_) context_.batch_compile_runner_(count, job);
            else context_.pipeline_workers_->run(count, job);
        };
        PipelineCompiles prior_compiles;
        bool has_prior_compiles = false;
        // Fast pre-pass before acquiring resource_mutex_ or running prepare(): collect all unique unbuilt synchronous
        // graphics (pipeline, unbound_mask) and compute pipelines across the batch and compile them concurrently so
        // prepare() completes in one pass with zero wasted transient allocations or skipped draws. Skipped in O(1) on
        // warm runs when no unbuilt synchronous pipelines remain (pass-0 prepare() below remains as a safe fallback).
        if (has_parallel_compiler && runs_shaders && context_.has_unbuilt_sync_pipelines()) {
            const std::uint64_t gen_before = context_.sync_pipeline_gen();
            PipelineCompiles pre_compiles;
            collect_unbuilt_pipelines(lists, context_.replay_, pre_compiles);
            if (!pre_compiles.empty()) {
                submit_phases::note_change(submit_phases::change_pipelines);
                std::shared_ptr<void> demand;
                if (context_.large_compile_demand_ && pre_compiles.any_large()) demand = context_.large_compile_demand_();
                pre_compiles.build(run_compiles, context_.large_compile_budget_);
                demand.reset();
                context_.note_sync_pipelines_built(pre_compiles.size());
                prior_compiles = std::move(pre_compiles);
                has_prior_compiles = true;
            } else {
                context_.note_empty_pipeline_scan(gen_before);
            }
        }
        for (int pass = 0;; ++pass) {
            submit_phases::lock_timed(resource_lock);
            bindless_commit = !skip_shader_prep ? context_.bindless_commit_ : std::function<void(bool)>{};
            ensure_cached_replay_locked(transfer_only, can_resolve_occlusion);
            ReplayFunctions& replay = cached_replay_;
            mesh_draws = 0;
            replay.mesh_draw_count = pipeline_profile::enabled() ? &mesh_draws : nullptr;
            replay.bindless_set = VK_NULL_HANDLE;
            replay.gpu_profile = nullptr;
            replay.on_buffer_write = cached_replay_on_buffer_write_;
            replay.on_buffer_write_context = cached_replay_on_buffer_write_ctx_;
            if (captures) {
                if (kind_ == QueueKind::copy || !context_.predicate_reducer_ || family_ != context_.predicate_family_)
                    return queue_unsupported(__LINE__);
                // The helper's lazy pipeline is shared by queues. Serialize preparation with the tracker lock,
                // retaining pipeline -> resource -> tracker -> submit lock order through emit and submission.
                if (!tracker_lock.owns_lock()) submit_phases::lock_timed(tracker_lock);
                batch = context_.predicate_reducer_->batch(captures);
                if (!batch) return queue_unsupported(__LINE__);
                if (context_.predicated_copier_) copy_batch = context_.predicated_copier_->batch();
            }
            segments = 0;
            statistics_segments = 0;
            has_queries = false;
            // Collect unbuilt pipelines via PipelineCompiles (whether pipeline_workers_ is enabled or serial).
            // On pass 0, if any pipeline must be compiled, release pass-0 transient state along with tracker_lock
            // and resource_lock before invoking the compiler (keeping pipeline_lock so registered pipelines and
            // workers stay alive), then re-prepare on pass 1 against current resources.
            PipelineCompiles compiles;
            replay.compiles = &compiles;
            ReplayResult prepare_result = ReplayResult::success;
            for (std::size_t i = 0; i < lists.size(); ++i) {
                const auto result = prepare(*lists[i], resolve, replay, prepared[i], can_resolve_occlusion);
                if (result != ReplayResult::success) {
                    prepare_result = result;
                    break;
                }
                if (prepared[i].segment_count > UINT32_MAX - segments ||
                    prepared[i].statistics_segment_count > UINT32_MAX - statistics_segments)
                    return queue_unsupported(__LINE__);
                segments += prepared[i].segment_count;
                statistics_segments += prepared[i].statistics_segment_count;
                has_queries = has_queries || !prepared[i].queries.empty();
            }
            bool needs_unlock_compile = !compiles.empty();
            if (needs_unlock_compile && has_prior_compiles) {
                needs_unlock_compile = false;
                for (const PipelineCompiles::Use& u : compiles.uses()) {
                    if (!prior_compiles.contains_ticket(compiles, u.ticket)) {
                        needs_unlock_compile = true;
                        break;
                    }
                }
            }
            if (pass == 0 && needs_unlock_compile) {
                submit_phases::note_change(submit_phases::change_pipelines);  // not a floor submission
                if (depth24_guard.active) {
                    context_.depth24_->retire(progress_, 0);
                    depth24_guard.active = false;
                }
                if (scratch_guard.active) {
                    context_.scratch_->retire(progress_, 0);
                    scratch_guard.active = false;
                }
                batch.reset();
                copy_batch.reset();
                counted_batch.reset();
                clamp_batch.reset();
                if (tracker_lock.owns_lock()) tracker_lock.unlock();
                resource_lock.unlock();
                recycle_khr_stages(khr_guard.stages);
                recycle_khr_as_queries(khr_guard.as_queries);
                geometry.clear();

                std::shared_ptr<void> demand;
                if (context_.large_compile_demand_ && compiles.any_large()) demand = context_.large_compile_demand_();
                if (has_parallel_compiler)
                    compiles.build(run_compiles, context_.large_compile_budget_);
                else
                    compiles.build_serial();
                demand.reset();
                const ReplayResult compile_result = check_compiled_pipelines(compiles, prepare_result);
                if (compile_result != ReplayResult::success) return compile_result;
                if (!has_prior_compiles && compiles.size() >= 2) context_.note_unbuilt_sync_pipeline();
                prior_compiles = std::move(compiles);
                has_prior_compiles = true;
                continue;
            }
            const PipelineCompiles* prior = has_prior_compiles ? &prior_compiles : nullptr;
            prepare_result = has_parallel_compiler
                                 ? build_pipelines(compiles, std::span(prepared).first(lists.size()), prepare_result,
                                                   run_compiles, context_.large_compile_budget_, prior)
                                 : build_pipelines_serial(compiles, std::span(prepared).first(lists.size()),
                                                          prepare_result, prior);
            cached_replay_.compiles = nullptr;
            if (prepare_result != ReplayResult::success) return prepare_result;
            break;
        }
        ReplayFunctions& replay = cached_replay_;
        const auto segment_pool = [this](VkQueryType type, std::uint32_t count,
                                         std::shared_ptr<OcclusionSegmentPool>& out) {
            const auto& vk = context_.vk;
            if (!vk.vkCreateQueryPool || !vk.vkDestroyQueryPool) return false;
            out = std::make_shared<OcclusionSegmentPool>();
            out->device = context_.device();
            out->destroy = vk.vkDestroyQueryPool;
            VkQueryPoolCreateInfo info{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
            info.queryType = type;
            info.queryCount = count;
            // The counters of a D3D12 pipeline-statistics heap, in D3D12_QUERY_DATA_PIPELINE_STATISTICS order.
            if (type == VK_QUERY_TYPE_PIPELINE_STATISTICS) info.pipelineStatistics = 0x7ff;
            return vk.vkCreateQueryPool(out->device, &info, nullptr, &out->pool) == VK_SUCCESS;
        };
        std::shared_ptr<OcclusionSegmentPool> physical, statistics;
        if (segments && !segment_pool(VK_QUERY_TYPE_OCCLUSION, segments, physical)) return queue_unsupported(__LINE__);
        // Without the compute helper a split query could not be summed: leave statistics queries whole, as before
        // (QueryReplay::plan refuses a list whose statistics query holds a mesh draw then).
        if (statistics_segments && can_resolve_occlusion) {
            if (!segment_pool(VK_QUERY_TYPE_PIPELINE_STATISTICS, statistics_segments, statistics))
                return queue_unsupported(__LINE__);
            for (std::size_t i = 0; i < lists.size(); ++i)
                statistics->mesh.insert(statistics->mesh.end(), prepared[i].statistics_mesh.begin(),
                                        prepared[i].statistics_mesh.end());
            if (std::find(statistics->mesh.begin(), statistics->mesh.end(), 1) == statistics->mesh.end()) {
                statistics->mesh.clear();
            } else {
                // The counters a mesh draw allows (VUID-vkCmdDrawMeshTasksEXT-pipelineStatistics-07076): the
                // fragment and compute invocations, and no other (D3D-291).
                VkQueryPoolCreateInfo info{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
                info.queryType = VK_QUERY_TYPE_PIPELINE_STATISTICS;
                info.queryCount = statistics_segments;
                info.pipelineStatistics = kMeshStatisticsFlags;
                if (context_.vk.vkCreateQueryPool(statistics->device, &info, nullptr, &statistics->mesh_pool) !=
                    VK_SUCCESS)
                    return queue_unsupported(__LINE__);
                // The mesh-primitives count of a mesh segment (D3D-MESH-CLIPPING-COUNTER): a query of its own type,
                // which the pipeline-statistics pool cannot hold. The device needs the meshShaderQueries feature,
                // or the clipping field of a mesh segment stays 0.
                if (context_.mesh_queries) {
                    VkQueryPoolCreateInfo primitives{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
                    primitives.queryType = VK_QUERY_TYPE_MESH_PRIMITIVES_GENERATED_EXT;
                    primitives.queryCount = statistics_segments;
                    if (context_.vk.vkCreateQueryPool(statistics->device, &primitives, nullptr,
                                                      &statistics->primitives_pool) != VK_SUCCESS)
                        return queue_unsupported(__LINE__);
                }
            }
        }
        // QueryReplay::plan() reads QueryPool::ended / occlusion and QueryReplay::commit() writes them after
        // resource_lock is unlocked. Hold tracker_mutex_ across plan -> emit -> submit -> commit so concurrent queues
        // see submission-order query state and remove_query_pool() waits until commit finishes.
        if (has_queries && !tracker_lock.owns_lock()) submit_phases::lock_timed(tracker_lock);
        QueryReplay queries;
        std::uint32_t base = 0, statistics_base = 0;
        for (std::size_t i = 0; i < lists.size(); ++i) {
            auto& list = prepared[i];
            const auto result = queries.plan(list, physical, base, can_resolve_occlusion, statistics, statistics_base);
            if (result != ReplayResult::success) return result;
            base += list.segment_count;
            statistics_base += list.statistics_segment_count;
        }
        if (queries.needs_sum || queries.needs_binary || queries.needs_statistics_sum || queries.needs_sum_binary) {
            if (!occlusion_resolve_) {
                auto helper = std::make_unique<OcclusionResolve>(context_.vk, context_.device());
                if (!helper->initialize(occlusion_memory)) return queue_unsupported(__LINE__);
                occlusion_resolve_ = std::move(helper);
            }
            if (!occlusion_resolve_->prepare(queries.needs_sum, queries.needs_binary, queries.needs_statistics_sum,
                                             queries.needs_sum_binary))
                return queue_unsupported(__LINE__);
        }

        CommandSlot* slot = acquire();
        if (!slot) return queue_unsupported(__LINE__);
        slot->khr_instances = std::move(khr_guard.stages);  // freed when this slot is reused
        khr_guard.stages.clear();
        slot->khr_as_queries = std::move(khr_guard.as_queries);
        khr_guard.as_queries.clear();
        command_buffer = slot->buffer;
        exec_replay_state_.slot = slot;
        const auto discard = [&] {
            if (context_.vk.vkResetCommandBuffer(command_buffer, 0) == VK_SUCCESS) slot->ticket = 0;
        };
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        context_.gpu_labels_.begin_record(command_buffer);
        if (context_.observe_result(context_.vk.vkBeginCommandBuffer(command_buffer, &begin),
                                    "vkBeginCommandBuffer") != VK_SUCCESS) {
            discard();
            return queue_unsupported(__LINE__);
        }
        // D3D12 ends every ExecuteCommandLists with all its work done and its writes visible to the next one on the
        // queue (implicit state decay relies on it), and a list may then touch the same memory without a barrier:
        // two submissions copying into one buffer, say. Vulkan orders submissions on a queue only by what their
        // barriers say, so this one starts with a full memory barrier against everything submitted before it
        // (synchronization validation: WRITE_AFTER_WRITE between vk_queue_test's back-to-back copies).
        if (context_.vk.vkCmdPipelineBarrier) {  // a mocked context may leave it out
            VkMemoryBarrier full{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            full.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
            full.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
            context_.vk.vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                             VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &full, 0, nullptr, 0, nullptr);
        }
        if (!tracker_lock.owns_lock()) submit_phases::lock_timed(tracker_lock);
        if (!skip_shader_prep && context_.bindless_prepare_ && !context_.bindless_prepare_(command_buffer)) {
            const bool ended = context_.observe_result(context_.vk.vkEndCommandBuffer(command_buffer),
                                                      "vkEndCommandBuffer") == VK_SUCCESS;
            resource_lock.unlock();
            pipeline_lock.unlock();
            const bool submitted = ended && submit(command_buffer, VK_NULL_HANDLE, 0);
            if (!submitted) discard();
            if (depth24_guard.active) {
                context_.depth24_->retire(progress_, 0);
                depth24_guard.active = false;
            }
            if (scratch_guard.active) {
                context_.scratch_->retire(progress_, 0);
                scratch_guard.active = false;
            }
            if (bindless_commit) bindless_commit(submitted);
            tracker_lock.unlock();
            return queue_unsupported(__LINE__);
        }
        replay.bindless_set =
            (!skip_shader_prep && context_.bindless_current_) ? context_.bindless_current_() : VK_NULL_HANDLE;
        // LEONIDA_GPU_PROFILE=1: the submission's query pool is reset and its first timestamp written here, after the
        // failure path above (which submits the buffer without a recording); dropped unless the buffer is submitted.
        std::unique_ptr<gpu_profile::Recording> profile;
        if (gpu_profile::enabled()) {
            std::size_t commands = 0;
            for (const CommandList* list : lists) commands += list->commands().size();
            profile = context_.gpu_profile_.begin(command_buffer, int(kind_), context_.timestamp_period(),
                                                  context_.timestamp_bits(family_), commands, &context_.profile_names_);
            replay.gpu_profile = profile.get();
        }
        const bool bindless_uploaded =
            !skip_shader_prep && std::exchange(context_.bindless_upload_pending_, false);
        // Layout changes go into the tracker itself, journalled, and are undone unless the submission reaches the
        // queue (tracker_mutex_ is held from here to the commit, so nothing else sees them meanwhile).
        auto& next_tracker = context_.tracker_;
        next_tracker.begin();
        struct TrackerTransaction {
            LayoutTracker& tracker;
            bool done = false;
            ~TrackerTransaction() {
                if (!done) tracker.rollback();
            }
        } tracker_transaction{next_tracker};
        // Bindless images that may already be visible to shaders are initialized on a submission that runs shaders.
        // initialize() zeroes committed storage images and only transitions sparse images.
        auto& storage_images = storage_images_scratch_;
        storage_images.clear();
        auto& zero_buffers = zero_buffers_scratch_;
        zero_buffers.clear();
        {
            std::lock_guard lock(context_.storage_mutex_);
            if (context_.storage_init_semaphore_ && context_.storage_init_semaphore_ != progress_ &&
                context_.storage_init_version_ != storage_init_seen_) {
                exec_waits_.push_back(context_.storage_init_semaphore_);
                exec_wait_values_.push_back(context_.storage_init_value_);
                storage_init_seen_ = context_.storage_init_version_;
            }
            // The set this submission binds may be a new generation whose table another queue's submission uploads:
            // wait for that one, as for the storage images. The table and fallback images share the device's families.
            if (!skip_shader_prep) {
                if (context_.bindless_upload_semaphore_ && context_.bindless_upload_semaphore_ != progress_ &&
                    context_.bindless_upload_version_ != bindless_upload_seen_) {
                    exec_waits_.push_back(context_.bindless_upload_semaphore_);
                    exec_wait_values_.push_back(context_.bindless_upload_value_);
                }
                bindless_upload_seen_ = context_.bindless_upload_version_;
            }
            if (kind_ != QueueKind::copy && runs_shaders)
                storage_images.assign(context_.storage_pending_.begin(), context_.storage_pending_.end());
            if (!storage_images.empty()) context_.storage_pending_.clear();
            zero_buffers.swap(context_.zero_pending_);  // any queue can fill a buffer
        }
        for (const auto& zero : zero_buffers)
            context_.vk.vkCmdFillBuffer(command_buffer, zero.buffer, 0, VK_WHOLE_SIZE, 0);
        if (!zero_buffers.empty()) {
            VkMemoryBarrier filled{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            filled.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            filled.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
            context_.vk.vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                             VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &filled, 0, nullptr, 0, nullptr);
        }
        for (const ResourceId id : storage_images) {
            const ResourceBinding* image = context_.resolve(id);
            if (!image || !image->image) continue;
            const VkImageSubresourceRange range{image->aspects, 0, image->mip_levels, 0, image->array_layers};
            auto pieces = detail::current(&next_tracker, id, range, VK_IMAGE_LAYOUT_UNDEFINED);
            detail::initialize(replay, command_buffer, *image, id, pieces, VK_IMAGE_LAYOUT_GENERAL, &next_tracker);
        }
        // The prelude (set_prelude): recorded here, after the set's upload and before the lists, and in either case
        // after the last submission of another queue that recorded one (its pass writes what these lists read, and
        // a pass here must not overtake it).
        const bool prelude = !skip_shader_prep && context_.prelude_record_ && context_.prelude_record_(command_buffer);
        // LEONIDA_PRELUDE_QUEUE_WAIT=0 (an experiment): no wait for another queue's pass, so the queues do not
        // serialise on each other's whole submissions; a list may then read what that pass has not written yet.
        static const bool cross_wait = [] {
            const char* v = std::getenv("LEONIDA_PRELUDE_QUEUE_WAIT");
            return !(v && v[0] == '0' && v[1] == 0);
        }();
        if (!skip_shader_prep) {
            std::lock_guard lock(context_.storage_mutex_);
            if (cross_wait && context_.prelude_semaphore_ && context_.prelude_semaphore_ != progress_ &&
                context_.prelude_version_ != prelude_seen_) {
                exec_waits_.push_back(context_.prelude_semaphore_);
                exec_wait_values_.push_back(context_.prelude_value_);
            }
            prelude_seen_ = context_.prelude_version_;
        }
        bool wrote_readback = false;
        VkPipelineStageFlags2 readback_stages = 0;
        bool wrote_buffer = false;
        struct BufferWriteChain {
            bool* flag;
            void (*next)(void*);
            void* next_ctx;
        } buffer_write_chain{&wrote_buffer, replay.on_buffer_write, replay.on_buffer_write_context};
        replay.on_buffer_write = +[](void* ctx) {
            auto* c = static_cast<BufferWriteChain*>(ctx);
            *c->flag = true;
            if (c->next) c->next(c->next_ctx);
        };
        replay.on_buffer_write_context = &buffer_write_chain;
        std::uint32_t open_labels = 0;  // events may span the lists of one ExecuteCommandLists
        std::size_t break_cursor = 0;
        CheckpointSubmission diagnostics;
        PFN_vkCmdSetCheckpointNV checkpoint = nullptr;
        if (execute_call) {
            auto& fault = *context_.device_fault();
            checkpoint = fault.checkpoint_command();
            const char* queue_kind = kind_ == QueueKind::direct ? "direct" :
                                     kind_ == QueueKind::compute ? "compute" : "copy";
            diagnostics = fault.begin_submission(execute_call, queue_, queue_kind, lists.size());
        }
        ReplayStateCache replay_state;
        for (std::size_t i = 0; i < lists.size(); ++i) {
            // Between merged ExecuteCommandLists batches: close any open list debug labels and emit D3D12's implicit
            // ExecuteCommandLists boundary barrier so writes from the previous batch are visible to the next.
            while (break_cursor < batch_breaks.size() && batch_breaks[break_cursor] == i) {
                ++break_cursor;
                close_labels(command_buffer, replay, open_labels);
                open_labels = 0;
                if (context_.vk.vkCmdPipelineBarrier) {
                    VkMemoryBarrier between{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
                    between.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
                    between.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
                    context_.vk.vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                                     VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &between, 0, nullptr, 0,
                                                     nullptr);
                }
            }
            const auto& list = prepared[i];
            if (execute_call && i < diagnostics.lists.size()) {
                const CheckpointRecorder recorder{checkpoint, diagnostics, context_.device_fault()->pipeline_names, i};
                emit_commands<true>(list, command_buffer, replay, &next_tracker, &open_labels, &recorder,
                                    &replay_state, &emit_scratch_);
            } else {
                emit(list, command_buffer, replay, &next_tracker, &open_labels, &replay_state, &emit_scratch_);
            }
            wrote_readback |= list.wrote_readback;
            readback_stages |= list.readback_src_stages;
        }
        close_labels(command_buffer, replay, open_labels);
        if (profile) profile->finish(command_buffer);
        // Only flush GPU writes to the host domain when this batch actually wrote to a readback buffer.
        if (wrote_readback) {
            const VkPipelineStageFlags2 allowed = queue_supported_stages(kind_);
            VkPipelineStageFlags2 src_stages = readback_stages & allowed;
            if (!src_stages) src_stages = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            const bool copy_only =
                (src_stages & ~(VK_PIPELINE_STAGE_2_COPY_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT)) == 0;
            const VkAccessFlags2 src_access =
                copy_only ? VK_ACCESS_2_TRANSFER_WRITE_BIT : VK_ACCESS_2_MEMORY_WRITE_BIT;

            if (context_.vk.vkCmdPipelineBarrier2) {
                VkMemoryBarrier2 host{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
                host.srcStageMask = src_stages;
                host.srcAccessMask = src_access;
                host.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
                host.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
                VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
                dep.memoryBarrierCount = 1;
                dep.pMemoryBarriers = &host;
                context_.vk.vkCmdPipelineBarrier2(command_buffer, &dep);
            } else {
                const VkPipelineStageFlags legacy_src =
                    copy_only ? VK_PIPELINE_STAGE_TRANSFER_BIT : VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
                VkMemoryBarrier host{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
                host.srcAccessMask = copy_only ? VK_ACCESS_TRANSFER_WRITE_BIT : VK_ACCESS_MEMORY_WRITE_BIT;
                host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
                context_.vk.vkCmdPipelineBarrier(command_buffer, legacy_src, VK_PIPELINE_STAGE_HOST_BIT,
                                                 0, 1, &host, 0, nullptr, 0, nullptr);
            }
        }
        const bool ended = context_.observe_result(context_.vk.vkEndCommandBuffer(command_buffer),
                                                  "vkEndCommandBuffer") == VK_SUCCESS;
        resource_lock.unlock();  // release resource_mutex_ and pipeline_mutex_ before vkQueueSubmit while keeping tracker_lock held
        pipeline_lock.unlock();
        const bool wait_previous = ordered_timestamps && submitted_;
        if (wait_previous) {
            exec_waits_.push_back(progress_);
            exec_wait_values_.push_back(submitted_);
        }
        const bool submitted = ended && submit_multi(command_buffer, trailing_signals,
                                                     execute_call ? &diagnostics : nullptr);
        if (submitted && wrote_buffer && context_.on_buffer_write_submitted_)
            context_.on_buffer_write_submitted_(context_.submitted_retire_ticket());
        if (profile && submitted) {
            // Read once this submission (ticket submitted_ on progress_) is done; the pool is reused after that.
            profile->mark_submitted();
            std::shared_ptr<gpu_profile::Recording> recording(std::move(profile));
            context_.enqueue_fence_callback(progress_, submitted_, [profiler = &context_.gpu_profile_, recording] {
                profiler->resolve(*recording);
            });
        }
        if (mesh_draws)
            std::fprintf(stderr, "[rt] pso-mesh unix_ms=%lld draws=%u submitted=%d\n",
                         pipeline_profile::unix_ms(), mesh_draws, int(submitted));
        if (!submitted && (!storage_images.empty() || !zero_buffers.empty())) {
            // not initialized after all: try again with the next submission
            std::lock_guard lock(context_.storage_mutex_);
            context_.storage_pending_.insert(storage_images.begin(), storage_images.end());
            context_.zero_pending_.insert(context_.zero_pending_.end(), zero_buffers.begin(), zero_buffers.end());
        }
        if (submitted && (!storage_images.empty() || !zero_buffers.empty())) {
            std::lock_guard lock(context_.storage_mutex_);
            context_.storage_init_semaphore_ = progress_;
            context_.storage_init_value_ = submitted_;
            ++context_.storage_init_version_;
            storage_init_seen_ = context_.storage_init_version_;
        }
        if (submitted && bindless_uploaded) {
            std::lock_guard lock(context_.storage_mutex_);
            context_.bindless_upload_semaphore_ = progress_;
            context_.bindless_upload_value_ = submitted_;
            ++context_.bindless_upload_version_;
            bindless_upload_seen_ = context_.bindless_upload_version_;
        }
        if (prelude) {
            if (submitted) {
                std::lock_guard lock(context_.storage_mutex_);
                context_.prelude_semaphore_ = progress_;
                context_.prelude_value_ = submitted_;
                ++context_.prelude_version_;
                prelude_seen_ = context_.prelude_version_;
            }
            if (context_.prelude_commit_) context_.prelude_commit_(submitted, context_.submitted_retire_ticket());
        }
        if (wait_previous && !submitted && !exec_waits_.empty()) {  // a failed submit keeps the waits for the next one
            exec_waits_.pop_back();
            exec_wait_values_.pop_back();
        }
        if (submitted) {
            next_tracker.commit();
            tracker_transaction.done = true;
            slot->predicate = std::move(batch);
            slot->predicated_copy = std::move(copy_batch);
            slot->counted_dispatch = std::move(counted_batch);
            slot->indirect_clamp = std::move(clamp_batch);
            queries.commit();
            slot->segments = std::move(queries.retained);
            if (!geometry.empty()) {
                slot->geometry.clear();
                slot->geometry.swap(geometry);
            }
        } else {
            next_tracker.rollback();  // here, under tracker_mutex_ (unlocked below, before the guard's scope ends)
            tracker_transaction.done = true;
            discard();
        }
        if (depth24_guard.active) {
            context_.depth24_->retire(progress_, submitted ? submitted_ : 0);
            depth24_guard.active = false;
        }
        if (scratch_guard.active) {
            context_.scratch_->retire(progress_, submitted ? submitted_ : 0);
            scratch_guard.active = false;
        }
        // Under tracker_mutex_, so no other flush comes between this submit and the set's note of whether it reached
        // the queue (it keeps a single upload obligation).
        if (bindless_commit) bindless_commit(submitted);
        tracker_lock.unlock();
        return submitted ? ReplayResult::success : queue_unsupported(__LINE__);
    }

    void marker_now(DebugMarker::Kind kind, const char* label) {
        const DebugLabelFunctions& f = context_.labels_;
        if (!f.complete()) return;
        if (kind == DebugMarker::Kind::end && !open_labels_) return;
        VkDebugUtilsLabelEXT info{VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT};
        info.pLabelName = label ? label : "";
        std::lock_guard lock(context_.submit_mutex());
        if (kind == DebugMarker::Kind::set) {
            f.queue_insert(queue_, &info);
        } else if (kind == DebugMarker::Kind::begin) {
            f.queue_begin(queue_, &info);
            ++open_labels_;
        } else {
            f.queue_end(queue_);
            --open_labels_;
        }
    }

    // ---- Wait-before-Signal: operations held on the CPU (queue_holds.h) ----

    // One held operation, with the waits that were pending when it was asked for.
    struct Deferred {
        enum class Kind : std::uint8_t { batch, signal, marker };
        Kind kind = Kind::batch;
        bool barrier_after = false;  // Present / after_held() was queued right after this op: do not merge past it
        std::uint64_t seq = 0;  // QueueHolds::next_seq(): the order operations were held in, across queues
        std::uint64_t present_epoch = kUnsetEpoch;  // immutable submit_phases::present_calls at enqueue time
        std::uint64_t execute_call = 0;  // original ExecuteCommandLists call for a held batch
        std::uint64_t visibility_seq = 0;  // context_.next_visibility_seq() at call time
        EnqueueSerial enqueue_serial;
        std::vector<VkSemaphore> waits;
        std::vector<std::uint64_t> wait_values;
        // Immutable closed CommandList snapshots (zero-copy when submitted via shared_ptr<const CommandList>).
        std::vector<std::shared_ptr<const CommandList>> lists;
        VkSemaphore fence = VK_NULL_HANDLE;  // signal
        std::uint64_t value = 0;
        DebugMarker::Kind marker_kind = DebugMarker::Kind::set;  // marker
        std::string label;
        // Alive until the operation has run: the fences it waits on and signals, the pipeline states and root
        // signatures of the lists (the application may release them right after the call).
        std::vector<std::shared_ptr<void>> pins;
        std::vector<std::shared_ptr<void>> wait_pins;  // the fences of `waits`
        std::chrono::steady_clock::time_point held_at{};  // LEONIDA_GPU_LATENCY: when it was held
    };

    static bool crosses_barrier(std::uint64_t cur_seq, std::uint64_t next_seq, std::uint64_t cur_epoch,
                                std::uint64_t next_epoch, std::uint64_t bseq) noexcept {
        return cur_epoch != next_epoch || (bseq && cur_seq <= bseq && next_seq > bseq);
    }

    struct DrainInfo {
        bool released = false;
        bool stopped_for_barrier = false;
        std::uint64_t last_seq = 0;
        std::uint64_t last_epoch = kUnsetEpoch;
    };

    struct HeldCountGuard {
        QueueHolds<VkSemaphore>& holds;
        std::int64_t delta;
        void release() noexcept {
            if (delta) {
                holds.held_changed(delta);
                delta = 0;
            }
        }
        ~HeldCountGuard() { release(); }
    };

    void set_active_seq_locked(std::uint64_t first) {
        active_seq_ = first;
        update_held_front_seq_locked();
    }

    // Waits for any satisfied barriers registered strictly before `op_seq` to finish executing on QueueHolds' barrier
    // thread before running post-barrier GPU commands (strictly ordering Frame N Present blit before Frame N+1 writes).
    // Empty-command batches execute no GPU commands or resource transitions and do not block on unrelated barriers.
    // Called outside state_mutex_ by the thread owning executing_ == true.
    void wait_for_preceding_barriers(std::uint64_t op_seq, std::span<const CommandList* const> lists) {
        if (!op_seq) return;
        bool has_cmds = false;
        for (const CommandList* list : lists) {
            if (list && !list->commands().empty()) {
                has_cmds = true;
                break;
            }
        }
        if (has_cmds) {
            context_.holds().wait_barriers_drained(op_seq - 1);
        }
    }

    bool waits_pending(const std::vector<VkSemaphore>& semaphores, const std::vector<std::uint64_t>& values) const {
        for (std::size_t i = 0; i < semaphores.size(); ++i)
            if (context_.holds().pending(semaphores[i], values[i])) return true;
        return false;
    }

    // True when a Wait(fence, value) is already satisfied on the GPU (no pending QueueHolds signal AND the Vulkan
    // timeline semaphore counter has already reached `value`), so coalescing can safely merge past it while still
    // preserving its CPU Fence::Signal visibility edge via holds().signaled_vis_seq().
    bool wait_already_completed(VkSemaphore fence, std::uint64_t value) const {
        if (!fence || value == 0) return true;
        if (context_.is_device_lost()) return false;
        if (context_.holds().pending(fence, value)) return false;
        std::uint64_t done = 0;
        return context_.vk.vkGetSemaphoreCounterValue(context_.device(), fence, &done) == VK_SUCCESS && done >= value;
    }

    bool all_waits_already_completed(std::span<const VkSemaphore> semaphores,
                                     std::span<const std::uint64_t> values) const {
        const std::size_t n = (std::min)(semaphores.size(), values.size());
        for (std::size_t i = 0; i < n; ++i) {
            if (!wait_already_completed(semaphores[i], values[i])) return false;
        }
        return true;
    }

    // The next batch or Signal has to be held: something is held ahead of it (queue order), or a Wait it carries
    // has no submitted Signal yet. Called with state_mutex_ held.
    bool must_hold() const { return !held_.empty() || waits_pending(waits_, wait_values_); }

    // True when a synchronous caller cannot run even after the active execution finishes draining ready held ops:
    // either this call carries a Wait on an unsignaled fence, or the front of held_ is blocked on one.
    // Called with state_mutex_ held.
    bool blocked_by_wait_locked() const {
        return waits_pending(waits_, wait_values_) || (!held_.empty() && !ready(held_.front()));
    }

    // Called with state_mutex_ held whenever active_seq_ or held_ changes. Keeps held_front_seq_ covering both any
    // popped op currently executing and the front of held_ so Present barriers never overtake executing work.
    void update_held_front_seq_locked() {
        has_held_.store(!held_.empty(), std::memory_order_release);
        const std::uint64_t queued = held_.empty() ? 0 : held_.front().seq;
        const std::uint64_t oldest = (active_seq_ && queued) ? std::min(active_seq_, queued)
                                   : active_seq_             ? active_seq_
                                                             : queued;
        held_front_seq_.store(oldest, std::memory_order_release);
    }

    // Moves pending waits (state_mutex_-owned) into execution-local storage (owned by executing_). state_mutex_ held.
    void take_waits_to_exec_locked() {
        exec_user_wait_count_ += waits_.size();
        exec_waits_.insert(exec_waits_.end(), waits_.begin(), waits_.end());
        exec_wait_values_.insert(exec_wait_values_.end(), wait_values_.begin(), wait_values_.end());
        for (auto& p : wait_pins_) exec_wait_pins_.push_back(std::move(p));
        waits_.clear();
        wait_values_.clear();
        wait_pins_.clear();
    }

    // Restores unconsumed execution waits after a transient vkQueueSubmit failure onto the front of pending waits_,
    // matching run()'s semantics so a held Signal behind the failed batch keeps its own waits and the retry gets these.
    // Called with state_mutex_ held.
    void restore_exec_waits_locked() {
        if (exec_waits_.empty()) {
            exec_user_wait_count_ = 0;
            return;
        }
        waits_.insert(waits_.begin(), exec_waits_.begin(), exec_waits_.end());
        wait_values_.insert(wait_values_.begin(), exec_wait_values_.begin(), exec_wait_values_.end());
        wait_pins_.insert(wait_pins_.begin(), std::make_move_iterator(exec_wait_pins_.begin()),
                          std::make_move_iterator(exec_wait_pins_.end()));
        exec_waits_.clear();
        exec_wait_values_.clear();
        exec_wait_pins_.clear();
        exec_user_wait_count_ = 0;
    }

    // The waits are consumed: submitted as `ticket` (0: dropped on device loss). Called by the thread owning
    // executing_ (without state_mutex_ held). Retired pins are transferred back under state_mutex_ and destroyed
    // by the caller after state_mutex_ is released (drain_pins): the last release of a fence waits for a running
    // release, which must not be waiting for state_mutex_.
    void clear_exec_waits(std::uint64_t ticket = 0) {
        if (exec_user_wait_count_) {
            const std::size_t n = (std::min)(exec_user_wait_count_, exec_waits_.size());
            context_.note_waits_removed(std::span(exec_waits_).first(n), std::span(exec_wait_values_).first(n));
            exec_user_wait_count_ = 0;
        }
        exec_waits_.clear();
        exec_wait_values_.clear();
        retire_exec_pins(exec_wait_pins_, ticket);
    }

    void retire_exec_pins(std::vector<std::shared_ptr<void>>& pins, std::uint64_t ticket) {
        for (auto& pin : pins) {
            if (ticket) exec_ticketed_pins_.push_back({ticket, std::move(pin)});
            else exec_retired_pins_.push_back(std::move(pin));
        }
        pins.clear();
    }

    // Transfers pins accumulated while executing_ was held without state_mutex_. state_mutex_ held.
    void flush_exec_pins_locked(std::vector<std::shared_ptr<void>>& retired) {
        for (auto& tp : exec_ticketed_pins_) pin_queue_.push_back(std::move(tp));
        exec_ticketed_pins_.clear();
        for (auto& p : exec_retired_pins_) retired.push_back(std::move(p));
        exec_retired_pins_.clear();
    }

    // Moves the pins whose submission has completed (and the ones retired outright) to `out`. state_mutex_ held.
    void drain_pins(std::vector<std::shared_ptr<void>>& out) {
        if (!retired_pins_.empty()) {
            for (auto& p : retired_pins_) out.push_back(std::move(p));
            retired_pins_.clear();
        }
        if (pin_queue_.empty()) return;
        const std::uint64_t done = completed();
        while (!pin_queue_.empty() && pin_queue_.front().ticket <= done) {
            out.push_back(std::move(pin_queue_.front().pin));
            pin_queue_.pop_front();
        }
    }

    // Appends `op`, which takes the pending waits with it (a marker has none). Called with state_mutex_ held.
    void hold(Deferred&& op) {
        if (op.present_epoch == kUnsetEpoch)
            op.present_epoch = submit_phases::totals().present_calls.load(std::memory_order_acquire);
        // A call the worker runs (LEONIDA_SUBMIT_ASYNC) keeps the number it was queued with: what was registered
        // behind it (a Present) still waits for it.
        op.seq = async_seq_ ? async_seq_ : context_.holds().next_seq();
        if (op.kind != Deferred::Kind::marker) {
            op.waits = std::move(waits_);
            op.wait_values = std::move(wait_values_);
            op.wait_pins = std::move(wait_pins_);
            waits_.clear();
            wait_values_.clear();
            wait_pins_.clear();
        }
        if (gpu_latency::enabled()) op.held_at = std::chrono::steady_clock::now();
        held_.push_back(std::move(op));
        update_held_front_seq_locked();
        context_.holds().held_changed(1);
    }

    template<typename MakeHeldLists>
    ReplayResult execute_sync_impl(std::span<const CommandList* const> lists, const PinFn& pin,
                                   std::uint64_t execute_call, MakeHeldLists&& make_held_lists,
                                   std::shared_ptr<void> async_pin = {}, EnqueueSerial enqueue_serial = {},
                                   bool async_barrier_after = false,
                                   std::uint64_t async_present_epoch = kUnsetEpoch,
                                   std::uint64_t vis_seq = 0) {
        ReplayResult result = ReplayResult::success;
        bool ran = false;
        bool held = false;
        bool pre_drained = false;
        QueueHolds<VkSemaphore>::WorkGuard work_guard;
        std::vector<std::shared_ptr<void>> retired;  // destroyed after state_mutex_ is released
        EnqueueSerial dropped_serial;                // finished after state_mutex_ is released if lost
        auto hold_batch_locked = [&] {
            Deferred op;
            op.kind = Deferred::Kind::batch;
            op.barrier_after = async_barrier_after;
            op.present_epoch = async_present_epoch;
            op.execute_call = execute_call;
            op.visibility_seq = vis_seq;
            op.enqueue_serial = std::move(enqueue_serial);
            op.lists = make_held_lists();
            if (async_pin) op.pins.push_back(std::move(async_pin));
            else if (pin) op.pins.push_back(pin());
            hold(std::move(op));
            held = true;
        };
        {
            std::unique_lock state(state_mutex_, std::defer_lock);
            submit_phases::lock_timed(state);
            // If another thread is executing or draining ready held ops and this batch is not blocked by an unmet
            // Wait, wait on exec_cv_ (with state_mutex_ unlocked so Wait/Signal/wait-blocked callers never stall on
            // before_execute_()) so this batch still runs before_execute_() and execute_now() synchronously.
            if (executing_ && !blocked_by_wait_locked() && !context_.is_device_lost()) {
                exec_cv_.wait(state, [this] {
                    return !executing_ || blocked_by_wait_locked() || context_.is_device_lost();
                });
            }
            if (context_.is_device_lost()) {
                result = queue_unsupported(__LINE__);
                if (async_pin) retired.push_back(std::move(async_pin));
                dropped_serial = std::move(enqueue_serial);
            } else if (!executing_ && !blocked_by_wait_locked()) {
                work_guard.arm(context_.holds());
                executing_ = true;
                ExecGuard exec_guard{this, state};
                // Drain any ready held ops ahead of us first (strict FIFO, e.g. a gate just satisfied by
                // QueueHolds::signaled() before pump_ready() reached this queue).
                DrainInfo pre_info{};
                if (!held_.empty()) pre_drained = drain_held_while_executing(state, retired, false, &pre_info);
                const std::uint64_t call_epoch =
                    async_present_epoch != kUnsetEpoch
                        ? async_present_epoch
                        : submit_phases::totals().present_calls.load(std::memory_order_acquire);
                const std::uint64_t call_seq_hint = async_seq_ ? async_seq_ : context_.holds().current_seq() + 1;
                const std::uint64_t bseq_now = barrier_seq_.load(std::memory_order_acquire);
                const bool barrier_before_me =
                    pre_info.released &&
                    (pre_info.stopped_for_barrier ||
                     crosses_barrier(pre_info.last_seq, call_seq_hint, pre_info.last_epoch, call_epoch, bseq_now));
                if (context_.is_device_lost()) {
                    result = queue_unsupported(__LINE__);
                    if (async_pin) retired.push_back(std::move(async_pin));
                    dropped_serial = std::move(enqueue_serial);
                } else if (must_hold() || barrier_before_me) {
                    hold_batch_locked();
                    if (barrier_before_me) pause_release_for_barrier_.store(true, std::memory_order_release);
                } else {
                    const std::uint64_t my_seq = async_seq_ ? async_seq_ : context_.holds().next_seq();
                    set_active_seq_locked(my_seq);
                    take_waits_to_exec_locked();
                    state.unlock();

                    wait_for_preceding_barriers(my_seq, lists);
                    result = execute_now(lists, execute_call, {}, {}, vis_seq, false);
                    note_coalesced_submit(1);
                    enqueue_serial.finish();
                    if (async_pin) {
                        if (result == ReplayResult::success && submitted_)
                            exec_ticketed_pins_.push_back({submitted_, std::move(async_pin)});
                        else
                            exec_retired_pins_.push_back(std::move(async_pin));
                    }

                    state.lock();
                    restore_exec_waits_locked();
                    flush_exec_pins_locked(retired);
                    set_active_seq_locked(0);
                    drain_held_while_executing(state, retired);
                    ran = true;
                }
                exec_guard.release();
            } else {
                hold_batch_locked();
            }
            drain_pins(retired);
        }
        dropped_serial.finish();
        retired.clear();
        if (ran || held || pre_drained) context_.holds().pump_ready();
        return result;
    }

    // Drains all ready held operations in strict FIFO order while executing_ is owned by the caller, coalescing
    // consecutive ready command-list batches that have no intervening Signal, Wait, Marker, or Present barrier into a
    // single before_execute_() + single vkQueueSubmit (with inter-batch D3D12 memory barriers and per-batch fallback
    // isolation on error). Releases state_mutex_ around every execute_now() / submit() / marker_now().
    // Entry & exit: state_mutex_ held, executing_ == true.
    bool drain_held_while_executing(std::unique_lock<std::mutex>& state,
                                    std::vector<std::shared_ptr<void>>& retired, bool from_pump = false,
                                    DrainInfo* info = nullptr) {
        bool released = false;
        while (!held_.empty() && (context_.is_device_lost() || ready(held_.front()))) {
            if (context_.is_device_lost()) {
                Deferred op = std::move(held_.front());
                held_.pop_front();
                set_active_seq_locked(op.seq);
                if (!op.waits.empty()) context_.note_waits_removed(op.waits, op.wait_values);
                for (auto& p : op.pins) retired.push_back(std::move(p));
                for (auto& p : op.wait_pins) retired.push_back(std::move(p));
                EnqueueSerial dropped = std::move(op.enqueue_serial);
                context_.holds().held_changed(-1);
                set_active_seq_locked(0);
                if (dropped.value) {
                    state.unlock();
                    dropped.finish();
                    state.lock();
                }
                released = true;
                continue;
            }

            if (held_.front().kind == Deferred::Kind::batch) {
                const std::uint64_t bseq = barrier_seq_.load(std::memory_order_acquire);
                std::vector<Deferred> group_ops;
                group_ops.push_back(std::move(held_.front()));
                held_.pop_front();
                std::size_t batch_count = 1;
                bool stop_after_signal = false;
                while (!group_ops.back().barrier_after && !held_.empty()) {
                    const Deferred& next = held_.front();
                    if (crosses_barrier(group_ops.back().seq, next.seq, group_ops.back().present_epoch,
                                        next.present_epoch, bseq))
                        break;
                    if (!next.waits.empty() && !all_waits_already_completed(next.waits, next.wait_values)) break;
                    if (next.kind == Deferred::Kind::batch) {
                        if (stop_after_signal) break;
                        group_ops.push_back(std::move(held_.front()));
                        held_.pop_front();
                        ++batch_count;
                    } else if (next.kind == Deferred::Kind::signal) {
                        group_ops.push_back(std::move(held_.front()));
                        held_.pop_front();
                        if (context_.has_unsatisfied_waiter(group_ops.back().fence, group_ops.back().value))
                            stop_after_signal = true;
                    } else {
                        break;
                    }
                }

                HeldCountGuard held_guard{context_.holds(), -static_cast<std::int64_t>(group_ops.size())};

                const std::uint64_t last_seq = group_ops.back().seq;
                const std::uint64_t last_epoch = group_ops.back().present_epoch;
                bool stop_for_barrier =
                    group_ops.back().barrier_after ||
                    (!held_.empty() &&
                     crosses_barrier(last_seq, held_.front().seq, last_epoch, held_.front().present_epoch, bseq));

                Deferred& first = group_ops.front();
                set_active_seq_locked(first.seq);

                const bool log_latency = gpu_latency::enabled();
                std::uint64_t latency_fence = 0;
                std::uint64_t latency_value = 0;
                if (log_latency) {
                    for (std::size_t i = 0; i < first.waits.size(); ++i)
                        if (first.wait_values[i] > latency_value) {
                            latency_fence = std::uint64_t(reinterpret_cast<std::uintptr_t>(first.waits[i]));
                            latency_value = first.wait_values[i];
                        }
                }

                exec_user_wait_count_ += first.waits.size();
                exec_waits_.insert(exec_waits_.end(), first.waits.begin(), first.waits.end());
                exec_wait_values_.insert(exec_wait_values_.end(), first.wait_values.begin(), first.wait_values.end());
                for (auto& p : first.wait_pins) exec_wait_pins_.push_back(std::move(p));
                first.waits.clear();
                first.wait_values.clear();
                first.wait_pins.clear();

                state.unlock();

                if (log_latency) {
                    for (const Deferred& op : group_ops)
                        gpu_latency::held(int(kind_), int(op.kind), op.held_at, latency_fence, latency_value);
                }

                exec_list_ptrs_.clear();
                exec_batch_breaks_.clear();
                exec_pending_signals_.clear();
                std::uint64_t group_vis_seq = 0;
                for (std::size_t gi = 1; gi < group_ops.size(); ++gi) {
                    Deferred& op = group_ops[gi];
                    if (!op.waits.empty()) {
                        const std::size_t wn = (std::min)(op.waits.size(), op.wait_values.size());
                        for (std::size_t wi = 0; wi < wn; ++wi) {
                            const std::uint64_t w_vis =
                                context_.holds().signaled_vis_seq(op.waits[wi], op.wait_values[wi]);
                            op.visibility_seq = (std::max)(op.visibility_seq, w_vis);
                        }
                        context_.note_waits_removed(op.waits, op.wait_values);
                        retire_exec_pins(op.wait_pins, 0);
                        op.waits.clear();
                        op.wait_values.clear();
                    }
                }
                for (const Deferred& op : group_ops) {
                    group_vis_seq = (std::max)(group_vis_seq, op.visibility_seq);
                    if (op.kind == Deferred::Kind::batch) {
                        if (!exec_list_ptrs_.empty() && !op.lists.empty())
                            exec_batch_breaks_.push_back(exec_list_ptrs_.size());
                        for (const auto& list : op.lists) exec_list_ptrs_.push_back(list.get());
                    } else if (op.kind == Deferred::Kind::signal) {
                        exec_pending_signals_.push_back({op.fence, op.value, op.visibility_seq});
                    }
                }

                wait_for_preceding_barriers(first.seq, exec_list_ptrs_);
                const ReplayResult result = execute_now(exec_list_ptrs_, first.execute_call, exec_batch_breaks_,
                                                        exec_pending_signals_, group_vis_seq, /*from_held=*/true);
                if (result == ReplayResult::success) {
                    note_coalesced_submit(batch_count);
                    for (Deferred& op : group_ops) {
                        retire_exec_pins(op.pins, submitted_);
                        op.enqueue_serial.finish();
                        op.lists.clear();
                    }
                } else if (batch_count > 1 && !context_.is_device_lost()) {
                    // Isolate the failing batch so valid peers and trailing Signals still submit in FIFO order.
                    for (Deferred& op : group_ops) {
                        if (op.kind == Deferred::Kind::batch) {
                            exec_list_ptrs_.clear();
                            for (const auto& list : op.lists) exec_list_ptrs_.push_back(list.get());
                            const ReplayResult sub =
                                execute_now(exec_list_ptrs_, op.execute_call, {}, {}, op.visibility_seq, true);
                            note_coalesced_submit(1);
                            if (sub != ReplayResult::success) {
                                int none = static_cast<int>(ReplayResult::success);
                                held_failure_.compare_exchange_strong(none, static_cast<int>(sub));
                                std::fprintf(stderr,
                                             "[rt] d3d12: a held ExecuteCommandLists failed when it was released "
                                             "(result %d); nothing was submitted for it, and a Signal behind it still "
                                             "signals\n",
                                             static_cast<int>(sub));
                            }
                            retire_exec_pins(op.pins, sub == ReplayResult::success ? submitted_ : 0);
                            op.enqueue_serial.finish();
                            op.lists.clear();
                        } else if (op.kind == Deferred::Kind::signal) {
                            // Preserve any unconsumed waits from a preceding failed batch rather than letting this
                            // trailing Signal (which carried no waits of its own) consume them.
                            auto saved_waits = std::move(exec_waits_);
                            auto saved_values = std::move(exec_wait_values_);
                            auto saved_pins = std::move(exec_wait_pins_);
                            const std::size_t saved_user = std::exchange(exec_user_wait_count_, 0);
                            exec_waits_.clear();
                            exec_wait_values_.clear();
                            exec_wait_pins_.clear();
                            if (op.visibility_seq > pending_wait_vis_seq_) pending_wait_vis_seq_ = op.visibility_seq;
                            const bool ok = submit(VK_NULL_HANDLE, op.fence, op.value, nullptr, op.visibility_seq);
                            if (!ok) std::fprintf(stderr, "[rt] d3d12: a held Signal failed when it was released\n");
                            retire_exec_pins(op.pins, ok ? submitted_ : 0);
                            exec_waits_ = std::move(saved_waits);
                            exec_wait_values_ = std::move(saved_values);
                            exec_wait_pins_ = std::move(saved_pins);
                            exec_user_wait_count_ = saved_user;
                        }
                    }
                } else {
                    note_coalesced_submit(batch_count);
                    int none = static_cast<int>(ReplayResult::success);
                    held_failure_.compare_exchange_strong(none, static_cast<int>(result));
                    std::fprintf(stderr,
                                 "[rt] d3d12: a held ExecuteCommandLists failed when it was released "
                                 "(result %d); nothing was submitted for it, and a Signal behind it still signals\n",
                                 static_cast<int>(result));
                    // Single batch failed (do NOT re-run it); preserve its unconsumed waits for retry, and still submit
                    // any trailing Signals that rode along with it (unless device is lost).
                    auto saved_waits = std::move(exec_waits_);
                    auto saved_values = std::move(exec_wait_values_);
                    auto saved_pins = std::move(exec_wait_pins_);
                    const std::size_t saved_user = std::exchange(exec_user_wait_count_, 0);
                    exec_waits_.clear();
                    exec_wait_values_.clear();
                    exec_wait_pins_.clear();
                    for (Deferred& op : group_ops) {
                        if (op.kind == Deferred::Kind::signal && !context_.is_device_lost()) {
                            if (op.visibility_seq > pending_wait_vis_seq_) pending_wait_vis_seq_ = op.visibility_seq;
                            const bool ok = submit(VK_NULL_HANDLE, op.fence, op.value, nullptr, op.visibility_seq);
                            if (!ok) std::fprintf(stderr, "[rt] d3d12: a held Signal failed when it was released\n");
                            retire_exec_pins(op.pins, ok ? submitted_ : 0);
                        } else {
                            retire_exec_pins(op.pins, 0);
                            op.enqueue_serial.finish();
                            op.lists.clear();
                        }
                    }
                    exec_waits_ = std::move(saved_waits);
                    exec_wait_values_ = std::move(saved_values);
                    exec_wait_pins_ = std::move(saved_pins);
                    exec_user_wait_count_ = saved_user;
                }

                state.lock();
                restore_exec_waits_locked();
                flush_exec_pins_locked(retired);
                held_guard.release();
                set_active_seq_locked(0);
                released = true;
                const std::uint64_t bseq_after = barrier_seq_.load(std::memory_order_acquire);
                if (!stop_for_barrier && !held_.empty()) {
                    stop_for_barrier =
                        crosses_barrier(last_seq, held_.front().seq, last_epoch, held_.front().present_epoch, bseq_after);
                }
                if (info) {
                    info->released = true;
                    info->last_seq = last_seq;
                    info->last_epoch = last_epoch;
                    info->stopped_for_barrier = stop_for_barrier;
                }
                if (stop_for_barrier) {
                    if (!from_pump && !held_.empty()) pause_release_for_barrier_.store(true, std::memory_order_release);
                    break;
                }
                continue;
            }

            Deferred op = std::move(held_.front());
            held_.pop_front();
            HeldCountGuard held_guard{context_.holds(), -1};
            const std::uint64_t op_seq = op.seq;
            const std::uint64_t op_epoch = op.present_epoch;
            const bool op_barrier = op.barrier_after;
            set_active_seq_locked(op_seq);

            // Capture latency-log metadata before moving waits out.
            const bool log_latency = gpu_latency::enabled();
            const int latency_kind = int(op.kind);
            const auto latency_held_at = op.held_at;
            std::uint64_t latency_fence = 0;
            std::uint64_t latency_value = 0;
            if (log_latency) {
                for (std::size_t i = 0; i < op.waits.size(); ++i)
                    if (op.wait_values[i] > latency_value) {
                        latency_fence = std::uint64_t(reinterpret_cast<std::uintptr_t>(op.waits[i]));
                        latency_value = op.wait_values[i];
                    }
            }

            // Move this op's waits into execution-local storage.
            exec_user_wait_count_ += op.waits.size();
            exec_waits_.insert(exec_waits_.end(), op.waits.begin(), op.waits.end());
            exec_wait_values_.insert(exec_wait_values_.end(), op.wait_values.begin(), op.wait_values.end());
            for (auto& p : op.wait_pins) exec_wait_pins_.push_back(std::move(p));
            op.waits.clear();
            op.wait_values.clear();
            op.wait_pins.clear();

            state.unlock();

            if (log_latency) gpu_latency::held(int(kind_), latency_kind, latency_held_at, latency_fence, latency_value);

            switch (op.kind) {
            case Deferred::Kind::batch:
                break;
            case Deferred::Kind::signal: {
                const bool ok = submit(VK_NULL_HANDLE, op.fence, op.value, nullptr, op.visibility_seq);
                if (!ok) std::fprintf(stderr, "[rt] d3d12: a held Signal failed when it was released\n");
                retire_exec_pins(op.pins, ok ? submitted_ : 0);
                break;
            }
            case Deferred::Kind::marker:
                marker_now(op.marker_kind, op.label.c_str());
                retire_exec_pins(op.pins, 0);
                break;
            }
            op.enqueue_serial.finish();
            op.lists.clear();  // drop CommandList refs before re-acquiring state_mutex_

            state.lock();
            restore_exec_waits_locked();
            flush_exec_pins_locked(retired);
            held_guard.release();
            set_active_seq_locked(0);
            released = true;
            const std::uint64_t bseq_after = barrier_seq_.load(std::memory_order_acquire);
            const bool hit_barrier =
                op_barrier ||
                (!held_.empty() &&
                 crosses_barrier(op_seq, held_.front().seq, op_epoch, held_.front().present_epoch, bseq_after));
            if (info) {
                info->released = true;
                info->last_seq = op_seq;
                info->last_epoch = op_epoch;
                info->stopped_for_barrier = hit_barrier;
            }
            if (hit_barrier) {
                if (!from_pump && !held_.empty()) pause_release_for_barrier_.store(true, std::memory_order_release);
                break;
            }
        }
        return released;
    }

    bool ready(const Deferred& op) const { return !waits_pending(op.waits, op.wait_values); }

    // Holder: called by the thread whose Signal made a held operation ready (never with a queue lock held).
    // With LEONIDA_SUBMIT_ASYNC, never run release_held() inline under QueueHolds::pump_mutex_ (even when called from
    // our own async_worker): schedule a worker turn so pump_ready() collects barriers and releases pump_mutex_ before
    // post-barrier work waits in wait_barriers_drained(), keeping cross-queue pump_ready() unblocked during Present.
    bool release_ready() override {
        if (async_started_.load(std::memory_order_acquire)) {
            const bool paused_for_barrier = pause_release_for_barrier_.exchange(false, std::memory_order_acq_rel);
            // Fast path: do not touch inbox_mutex_ or stamp barrier_after when this queue has no held operations.
            // barrier_after is stamped ONLY by real after_held() barriers (oldest_held_seq() / present_epoch),
            // never by cross-queue Signal() pump_ready() calls.
            if (!has_held_.load(std::memory_order_acquire)) return false;
            // On our own worker's trailing pump_ready(), avoid scheduling a redundant wakeup if the front held op is
            // still blocked on an unmet wait (we only need another turn when paused at a barrier or when front became
            // ready).
            if (t_worker_of_ == this && !paused_for_barrier) {
                std::lock_guard state(state_mutex_);
                if (held_.empty() || (!context_.is_device_lost() && !ready(held_.front()))) return false;
            }
            bool wake = false;
            {
                std::lock_guard lock(inbox_mutex_);
                if (async_stop_) return false;  // worker is stopping/joined during teardown
                // Only ask the worker to drain held_ when this queue actually has held work; avoids idle wakeups
                // and cross-queue ping-pong while another queue holds an unsatisfied wait.
                if (has_held_.load(std::memory_order_acquire) && !release_requested_) {
                    release_requested_ = true;
                    context_.holds().work_changed(1);
                    wake = true;
                }
            }
            if (wake) inbox_cv_.notify_one();
            return false;
        }
        // If an earlier drain stopped at a barrier boundary outside pump_ready(), yield once so pump_ready()
        // runs collect_ready_barriers() (Present / after_held()) BEFORE we drain post-barrier held work on the next pass.
        if (pause_release_for_barrier_.exchange(false, std::memory_order_acq_rel)) return true;
        return release_held(/*from_pump=*/true);
    }

    bool release_held(bool from_pump = false, DrainInfo* info = nullptr) {
        std::vector<std::shared_ptr<void>> retired;
        bool released = false;
        {
            std::unique_lock state(state_mutex_);
            if (!executing_ && !held_.empty() && (context_.is_device_lost() || ready(held_.front()))) {
                executing_ = true;
                ExecGuard exec_guard{this, state};
                released = drain_held_while_executing(state, retired, from_pump, info);
                exec_guard.release();
            }
            drain_pins(retired);
        }
        retired.clear();
        return released;
    }

    // Holder: the fence is released, so what waits on it can never run.
    void discard_waiting_on(VkSemaphore semaphore) override {
        std::vector<Deferred> dropped;
        {
            std::lock_guard state(state_mutex_);
            for (auto it = held_.begin(); it != held_.end();) {
                if (std::find(it->waits.begin(), it->waits.end(), semaphore) == it->waits.end() &&
                    !(it->kind == Deferred::Kind::signal && it->fence == semaphore)) {
                    ++it;
                    continue;
                }
                std::fprintf(stderr, "[rt] d3d12: a fence was released while a queue still held work that names it; "
                                     "that work is dropped\n");
                if (!it->waits.empty()) context_.note_waits_removed(it->waits, it->wait_values);
                dropped.push_back(std::move(*it));
                it = held_.erase(it);
                context_.holds().held_changed(-1);
            }
            update_held_front_seq_locked();
        }
    }

    // Holder: records a real after_held() barrier at `seq` so batches at or before `seq` never coalesce with batches
    // enqueued after `seq` (including batches in transit between inbox_ and held_). Never called by run_barriers().
    void note_barrier(std::uint64_t seq) override {
        if (!seq) return;
        std::uint64_t prev = barrier_seq_.load(std::memory_order_relaxed);
        while (seq > prev && !barrier_seq_.compare_exchange_weak(prev, seq, std::memory_order_acq_rel)) {}
        {
            std::lock_guard state(state_mutex_);
            for (auto it = held_.rbegin(); it != held_.rend(); ++it) {
                if (it->seq <= seq) {
                    it->barrier_after = true;
                    break;
                }
            }
        }
        if (async_started_.load(std::memory_order_acquire)) {
            std::lock_guard lock(inbox_mutex_);
            for (auto it = inbox_.rbegin(); it != inbox_.rend(); ++it) {
                if (it->seq <= seq) {
                    it->barrier_after = true;
                    break;
                }
            }
        }
    }

    // Holder: the sequence number of the oldest active, held, or queued operation (read-only query).
    std::uint64_t oldest_held_seq() override {
        // Read inbox_mutex_ (running_seq_ / inbox_) before state_mutex_ (active_seq_ / held_): items only move
        // inbox -> running -> held/active -> done, so reading upstream first cannot miss an item in transit.
        std::uint64_t oldest = 0;
        if (async_started_.load(std::memory_order_acquire)) {
            std::lock_guard lock(inbox_mutex_);
            oldest = running_seq_ ? running_seq_ : inbox_.empty() ? 0 : inbox_.front().seq;
        }
        {
            std::lock_guard state(state_mutex_);
            const std::uint64_t queued = held_.empty() ? 0 : held_.front().seq;
            const std::uint64_t active = (active_seq_ && queued) ? std::min(active_seq_, queued)
                                       : active_seq_             ? active_seq_
                                                                 : queued;
            if (active && (!oldest || active < oldest)) oldest = active;
        }
        return oldest;
    }

    // Holder: true while this queue has submitted GPU work that has not retired yet. Lock-free so
    // QueueHolds::has_progress() can query it safely without taking queue or submit locks.
    bool has_pending_gpu_work() const override {
        if (context_.is_device_lost()) return false;
        const std::uint64_t sub = submitted_gpu_.load(std::memory_order_acquire);
        if (!sub) return false;
        std::uint64_t done = 0;
        const VkResult result = context_.vk.vkGetSemaphoreCounterValue(context_.device(), progress_, &done);
        return result == VK_SUCCESS && done < sub;
    }

    // A queue destroyed with held work. Reported, not waited for: nothing was submitted for it, so the VkQueue is
    // idle and the destructor goes on. The fences of held Signals will never reach their values, so what is waiting
    // for them on the CPU (SetEventOnCompletion, as a released fence completes them) is woken, and the batches
    // that wait for them on other queues stay held (and are reported when those queues are destroyed).
    void report_held_at_teardown() {
        std::deque<Deferred> dropped;
        {
            std::lock_guard state(state_mutex_);
            if (held_.empty()) return;
            std::uint64_t value = 0;
            for (const Deferred& op : held_)
                for (std::size_t i = 0; i < op.waits.size() && !value; ++i)
                    if (context_.holds().pending(op.waits[i], op.wait_values[i])) value = op.wait_values[i];
            std::size_t signals = 0;
            for (const Deferred& op : held_) signals += op.kind == Deferred::Kind::signal;
            std::fprintf(stderr, "[rt] d3d12: queue destroyed with %zu held operation(s) never submitted: waits for a "
                                 "fence value (%llu) that no Signal reached; the fence was never signaled%s\n",
                         held_.size(), static_cast<unsigned long long>(value),
                         signals ? "; CPU waits on the fences of its held Signals are completed" : "");
            for (const Deferred& op : held_) {
                if (op.kind == Deferred::Kind::signal) context_.cancel_fence_waits(op.fence);
                if (!op.waits.empty()) context_.note_waits_removed(op.waits, op.wait_values);
            }
            context_.holds().held_changed(-static_cast<std::int64_t>(held_.size()));
            dropped.swap(held_);
            set_active_seq_locked(0);
        }
        // The pins (fences, pipeline states) are released here, outside the lock.
    }

    struct CommandSlot {
        VkCommandBuffer buffer;
        std::uint64_t ticket;
        std::vector<std::shared_ptr<const OcclusionSegmentPool>> segments;
        std::vector<VkEvent> events;  // order_timestamp_copy's, reset and reused once the slot completes
        std::unique_ptr<PredicatedBufferCopier::Batch> predicated_copy;
        std::unique_ptr<IndirectDispatchCounter::Batch> counted_dispatch;
        std::unique_ptr<IndirectArgumentClamp::Batch> indirect_clamp;  // counter slots, read once this slot completes
        std::unique_ptr<PredicateReducer::Batch> predicate; // retained until this slot completes
        std::vector<KhrInstanceStage> khr_instances;  // KHR ARRAY_OF_POINTERS staging, same lifetime
        std::vector<KhrAsQueryPool> khr_as_queries;   // KHR postbuild COMPACTED_SIZE query pools, same lifetime
        std::vector<VulkanContext::GeometryUpload> geometry;
    };

    struct PendingBindings {
        std::uint64_t ticket = 0;
        // Vulkan needs the resource until the bind completes and the previous allocation until it is unmapped.
        std::vector<std::shared_ptr<const TileHeapAllocation>> old_heaps;
        std::shared_ptr<SparseResource> resource;
    };

    static bool valid_tiles(const SparseResource& resource, TileRegion region) {
        return region.count && region.first < resource.tile_count && region.count <= resource.tile_count - region.first;
    }

    bool initialize_sparse_image_layout(ResourceId id) {
        std::lock_guard lock(context_.tracker_mutex_);
        if (context_.tracker_.layout(id, 0) != VK_IMAGE_LAYOUT_UNDEFINED) return true;
        auto* slot = acquire();
        if (!slot) return false;
        const VkCommandBuffer command_buffer = slot->buffer;
        auto& vk = context_.vk;
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        context_.gpu_labels_.begin_record(command_buffer);
        if (vk.vkBeginCommandBuffer(command_buffer, &begin) != VK_SUCCESS) {
            slot->ticket = 0;
            return false;
        }
        // Transition while the image is still unbound. Doing this after binding could discard another image's
        // contents in an aliased page. Subsequent uses keep the image in GENERAL.
        constexpr VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        detail::image_barrier(context_.replay_, command_buffer, context_.resources_.at(id), range,
                              VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, 0,
                              VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
                              VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
        if (vk.vkEndCommandBuffer(command_buffer) != VK_SUCCESS || !submit(command_buffer, VK_NULL_HANDLE, 0)) {
            slot->ticket = 0;
            return false;
        }
        context_.tracker_.set(id, range, VK_IMAGE_LAYOUT_GENERAL);
        return true;
    }

    bool bind_tiles(ResourceId id, const std::shared_ptr<SparseResource>& resource,
                    std::map<std::uint64_t, TileMapping>& changes) {
        auto& vk = context_.vk;
        if (!vk.vkQueueBindSparse || context_.is_device_lost()) return false;
        PendingBindings pending{0, {}, resource};
        std::vector<VkSparseMemoryBind> binds;
        std::vector<VkSparseImageMemoryBind> image_binds;
        for (const auto& [index, page] : changes) {
            const auto old = resource->mapped.find(index);
            if (old != resource->mapped.end()) {
                if (old->second == page) continue;
                pending.old_heaps.push_back(old->second.heap);
            } else if (!page.heap) {
                continue;
            }
            if (page.heap && !(resource->memory_types & (1u << page.heap->memory_type))) return false;
            const VkDeviceMemory memory = page.heap ? page.heap->memory : VK_NULL_HANDLE;
            if (resource->image) {
                constexpr auto shape = element_tile_shape(false, 4);
                const auto extent = resource->image_extent;
                const auto columns = 1 + (extent.width - 1) / shape.width;
                const auto x = std::uint32_t(index % columns) * shape.width;
                const auto y = std::uint32_t(index / columns) * shape.height;
                image_binds.push_back({
                    {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0}, {std::int32_t(x), std::int32_t(y), 0},
                    {std::min(shape.width, extent.width - x), std::min(shape.height, extent.height - y), 1},
                    memory, page.offset, 0});
                continue;
            }
            const VkDeviceSize resource_offset = index * tile_size_in_bytes;
            if (!binds.empty()) {
                auto& previous = binds.back();
                if (previous.resourceOffset + previous.size == resource_offset && previous.memory == memory &&
                    (!memory || previous.memoryOffset + previous.size == page.offset)) {
                    previous.size += tile_size_in_bytes;
                    continue;
                }
            }
            binds.push_back({resource_offset, tile_size_in_bytes, memory, page.offset, 0});
        }
        if (binds.empty() && image_binds.empty()) return true;
        if (!image_binds.empty() && !initialize_sparse_image_layout(id)) return false;
        const auto ticket = submitted_ + 1;
        pending.ticket = ticket;
        collect_bindings(completed());
        auto semaphores = exec_waits_;
        auto values = exec_wait_values_;
        // Sparse binds are not ordered against work on the same Vulkan queue by submission order alone.
        if (submitted_) {
            semaphores.push_back(progress_);
            values.push_back(submitted_);
        }
        VkTimelineSemaphoreSubmitInfo tickets{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
        tickets.waitSemaphoreValueCount = std::uint32_t(values.size());
        tickets.pWaitSemaphoreValues = values.data();
        tickets.signalSemaphoreValueCount = 1;
        tickets.pSignalSemaphoreValues = &ticket;
        const VkSparseBufferMemoryBindInfo buffer_bind{resource->buffer, std::uint32_t(binds.size()), binds.data()};
        const VkSparseImageMemoryBindInfo image_bind{
            resource->image, std::uint32_t(image_binds.size()), image_binds.data()};
        VkBindSparseInfo info{VK_STRUCTURE_TYPE_BIND_SPARSE_INFO, &tickets};
        info.waitSemaphoreCount = std::uint32_t(semaphores.size());
        info.pWaitSemaphores = semaphores.data();
        if (!binds.empty()) {
            info.bufferBindCount = 1;
            info.pBufferBinds = &buffer_bind;
        }
        if (!image_binds.empty()) {
            info.imageBindCount = 1;
            info.pImageBinds = &image_bind;
        }
        info.signalSemaphoreCount = 1;
        info.pSignalSemaphores = &progress_;
        pending_bindings_.push_back(std::move(pending));
        {
            std::lock_guard lock(context_.submit_mutex());
            VkResult bound;
            {
                const SubmitMutex::SubmitTurn turn(context_.submit_mutex());  // a queue operation: the compile gate
                bound = vk.vkQueueBindSparse(queue_, 1, &info, VK_NULL_HANDLE);
            }
            const auto result = context_.observe_result(bound, "vkQueueBindSparse");
            if (result != VK_SUCCESS) {
                if (result == VK_ERROR_DEVICE_LOST) context_.mark_device_lost();
                pending_bindings_.pop_back();
                return false;
            }
            context_.record_submit_locked(progress_, ticket);
        }
        submitted_ = last_sparse_ = ticket;
        submitted_gpu_.store(ticket, std::memory_order_release);
        clear_exec_waits(ticket);
        for (auto it = changes.begin(); it != changes.end();) {
            resource->mapped.erase(it->first);
            if (it->second.heap) ++it;
            else it = changes.erase(it);
        }
        resource->mapped.merge(changes);
        return true;
    }

    std::uint64_t completed() const {
        if (context_.is_device_lost()) return 0;
        std::uint64_t done = 0;
        const VkResult result = context_.observe_result(
            context_.vk.vkGetSemaphoreCounterValue(context_.device(), progress_, &done), "vkGetSemaphoreCounterValue");
        if (result == VK_ERROR_DEVICE_LOST) context_.mark_device_lost();
        // A failed timeline query cannot be used as a resource-retirement or command-buffer reuse ticket.
        return result == VK_SUCCESS ? done : 0;
    }

    void collect_bindings(std::uint64_t done) {
        while (!pending_bindings_.empty() && pending_bindings_.front().ticket <= done)
            pending_bindings_.pop_front();
        if (last_sparse_ <= done) last_sparse_ = 0;
    }

    // Releases what every finished command buffer holds; the first of them, or null when none has finished.
    CommandSlot* recycle_finished() {
        const auto done = completed();
        if (context_.is_device_lost()) return nullptr;
        collect_bindings(done);
        CommandSlot* reuse = nullptr;
        for (auto& slot : buffers_) {
            if (slot.ticket > done) continue;
            slot.segments.clear();
            slot.geometry.clear();
            recycle_khr_stages(slot.khr_instances);
            recycle_khr_as_queries(slot.khr_as_queries);
            for (VkEvent event : slot.events)
                if (context_.vk.vkResetEvent(context_.device(), event) == VK_SUCCESS) free_events_.push_back(event);
                else context_.vk.vkDestroyEvent(context_.device(), event, nullptr);
            slot.events.clear();
            if (!reuse) reuse = &slot;
        }
        return reuse;
    }

    // A queue at the cap with nothing finished waits here for its oldest submission. execute_now() calls it before it
    // takes resource_mutex_ or tracker_mutex_: the submission waited for may itself wait on another queue's work, and
    // that queue needs those locks to submit it. One execute runs at a time on a queue, so buffers_ is this caller's.
    // A timeout only means the GPU is slow: wait again rather than allocate past the cap. False when the device is lost
    // (now or while waiting), so the caller submits nothing.
    bool wait_for_room() {
        for (;;) {
            if (context_.is_device_lost()) return false;
            if (buffers_.size() < kMaxQueuedCommandBuffers) return true;
            const auto done = completed();
            for (const auto& slot : buffers_)
                if (slot.ticket <= done) return true;
            switch (wait_oldest_submitted()) {
            case VK_SUCCESS:
                return true;
            case VK_TIMEOUT:
                continue;
            case VK_ERROR_DEVICE_LOST:
                context_.mark_device_lost();
                return false;
            default:
                return true;  // nothing submitted to wait for (or a host error): no wait can make room, allocate
            }
        }
    }

    // Waits up to kQueuedWaitNs for the oldest submitted command buffer. VK_INCOMPLETE when none is submitted yet.
    VkResult wait_oldest_submitted() {
        std::uint64_t oldest = UINT64_MAX;
        for (const auto& slot : buffers_)
            if (slot.ticket < oldest) oldest = slot.ticket;
        if (oldest == UINT64_MAX) return VK_INCOMPLETE;  // all still being recorded
        VkSemaphoreWaitInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
        info.semaphoreCount = 1;
        info.pSemaphores = &progress_;
        info.pValues = &oldest;
        return context_.observe_result(context_.vk.vkWaitSemaphores(context_.device(), &info, kQueuedWaitNs),
                                       "vkWaitSemaphores (queue progress)");
    }

    CommandSlot* acquire() {
        if (context_.is_device_lost()) return nullptr;
        if (context_.deletion_queue_.has_pending())
            context_.drain_deleted_resources(context_.completed_retire_ticket());
        CommandSlot* reuse = recycle_finished();
        if (reuse) {
            if (context_.vk.vkResetCommandBuffer(reuse->buffer, 0) != VK_SUCCESS) return nullptr;
            reuse->predicate.reset();
            reuse->predicated_copy.reset();
            reuse->counted_dispatch.reset();
            reuse->indirect_clamp.reset();
            reuse->ticket = UINT64_MAX;  // in use until submitted
            return reuse;
        }
        VkCommandBufferAllocateInfo info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        info.commandPool = pool_;
        info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        info.commandBufferCount = 1;
        VkCommandBuffer command_buffer = VK_NULL_HANDLE;
        if (context_.vk.vkAllocateCommandBuffers(context_.device(), &info, &command_buffer) != VK_SUCCESS)
            return nullptr;
        buffers_.push_back({command_buffer, UINT64_MAX});
        return &buffers_.back();
    }

    // An unsignaled event for order_timestamp_copy, or null. MoltenVK decides when the command buffer is submitted
    // whether a set or wait does anything, from the event's state then, so each use gets an event of its own.
    VkEvent take_event() {
        if (!free_events_.empty()) {
            const VkEvent event = free_events_.back();
            free_events_.pop_back();
            return event;
        }
        VkEventCreateInfo info{VK_STRUCTURE_TYPE_EVENT_CREATE_INFO};
        VkEvent event = VK_NULL_HANDLE;
        if (context_.vk.vkCreateEvent(context_.device(), &info, nullptr, &event) != VK_SUCCESS) return VK_NULL_HANDLE;
        return event;
    }

    static void pass_trace(const CheckpointSubmission& d) {
        static const std::uint64_t every = [] {
            const char* v = std::getenv("LEONIDA_PASS_TRACE");
            return v && *v ? std::strtoull(v, nullptr, 10) : 0ull;
        }();
        if (!every) return;
        const std::uint64_t present = submit_phases::totals().present_serial.load(std::memory_order_relaxed);
        if (present % every != 1 % every) return;
        std::string line;
        const char* last = nullptr;
        unsigned run = 0;
        const auto flush = [&] {
            if (!last) return;
            char text[96];
            std::snprintf(text, sizeof(text), run > 1 ? " %s x%u" : " %s", last, run);
            line += text;
        };
        for (const auto& x : d.dispatches) {
            const char* name = x.shader ? x.shader : "-";
            if (const char* slash = std::strrchr(name, '/')) name = slash + 1;
            if (last && std::strcmp(last, name) == 0) {
                ++run;
                continue;
            }
            flush();
            last = name;
            run = 1;
        }
        flush();
        std::fprintf(stderr, "[rt] passes: present %llu execute %llu (%s) %zu dispatches%s:%s\n",
                     static_cast<unsigned long long>(present), static_cast<unsigned long long>(d.execute_call),
                     d.queue_kind ? d.queue_kind : "?", d.dispatch_count,
                     d.dispatch_count > d.dispatches.size() ? " (last 64)" : "", line.empty() ? " none" : line.c_str());
    }

    bool submit(VkCommandBuffer command_buffer, VkSemaphore fence, std::uint64_t value,
                CheckpointSubmission* diagnostics = nullptr, std::uint64_t visibility_seq = 0) {
        if (fence) {
            const PendingSignal sig{fence, value, visibility_seq};
            return submit_multi(command_buffer, {&sig, 1}, diagnostics);
        }
        return submit_multi(command_buffer, {}, diagnostics);
    }

    bool submit_multi(VkCommandBuffer command_buffer, std::span<const PendingSignal> extra_signals,
                      CheckpointSubmission* diagnostics = nullptr) {
        if (context_.is_device_lost()) return false;
        const std::uint64_t ticket = submitted_ + 1;
        if (last_sparse_) {
            collect_bindings(completed());
            if (context_.is_device_lost()) return false;
        }
        std::span<const VkSemaphore> semaphores = exec_waits_;
        std::span<const std::uint64_t> values = exec_wait_values_;
        bool using_sparse = false;
        if (last_sparse_) {
            sparse_waits_ = exec_waits_;
            sparse_values_ = exec_wait_values_;
            sparse_waits_.push_back(progress_);
            sparse_values_.push_back(last_sparse_);
            semaphores = sparse_waits_;
            values = sparse_values_;
            using_sparse = true;
        }
        constexpr VkPipelineStageFlags kAllCmds = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        VkPipelineStageFlags stack_stages[16] = {
            kAllCmds, kAllCmds, kAllCmds, kAllCmds, kAllCmds, kAllCmds, kAllCmds, kAllCmds,
            kAllCmds, kAllCmds, kAllCmds, kAllCmds, kAllCmds, kAllCmds, kAllCmds, kAllCmds,
        };
        const VkPipelineStageFlags* stages_ptr = stack_stages;
        if (semaphores.size() > 16) {
            submit_stages_.assign(semaphores.size(), kAllCmds);
            stages_ptr = submit_stages_.data();
        }
        // Deduplicate extra_signals by VkSemaphore, keeping the highest timeline value per fence so VkSubmitInfo never
        // names the same timeline semaphore twice (VUID-VkSubmitInfo-pSignalSemaphores-03242).
        PendingSignal stack_dedup[16];
        std::size_t dedup_count = 0;
        std::vector<PendingSignal> heap_dedup;
        auto add_dedup = [&](VkSemaphore f, std::uint64_t v) {
            if (!f || f == progress_) return;
            PendingSignal* data = heap_dedup.empty() ? stack_dedup : heap_dedup.data();
            const std::size_t count = heap_dedup.empty() ? dedup_count : heap_dedup.size();
            for (std::size_t i = 0; i < count; ++i) {
                if (data[i].fence == f) {
                    if (v > data[i].value) data[i].value = v;
                    return;
                }
            }
            if (heap_dedup.empty()) {
                if (dedup_count < 16) {
                    stack_dedup[dedup_count++] = {f, v, 0};
                    return;
                }
                heap_dedup.assign(stack_dedup, stack_dedup + dedup_count);
            }
            heap_dedup.push_back({f, v, 0});
        };
        for (const PendingSignal& sig : extra_signals) {
            add_dedup(sig.fence, sig.value);
        }
        const PendingSignal* const dedup_ptr = heap_dedup.empty() ? stack_dedup : heap_dedup.data();
        const std::size_t dedup_size = heap_dedup.empty() ? dedup_count : heap_dedup.size();

        // A signal-only submit (no command buffer) does not run before_execute_(): carry any CPU-signaled fence
        // visibility epoch from its consumed user waits to this queue's next execute_now().
        const std::uint64_t consumed_wait_vis = command_buffer ? 0 : exec_waits_vis_seq();
        std::uint64_t mirror_bind_waited = 0;
        VkSemaphore title_upload_waited_sem = VK_NULL_HANDLE;
        std::uint64_t title_upload_waited_val = 0;
        VkResult res = VK_SUCCESS;
        {
            std::unique_lock lock(context_.submit_mutex(), std::defer_lock);
            submit_phases::lock_timed(lock);  // a wait for another queue's submit is counted (the frame line)
            auto add_lazy_wait = [&](VkSemaphore sem, std::uint64_t val) {
                if (!using_sparse) {
                    sparse_waits_ = exec_waits_;
                    sparse_values_ = exec_wait_values_;
                    using_sparse = true;
                }
                bool merged = false;
                for (std::size_t i = 0; i < sparse_waits_.size(); ++i) {
                    if (sparse_waits_[i] == sem) {
                        if (val > sparse_values_[i]) sparse_values_[i] = val;
                        merged = true;
                        break;
                    }
                }
                if (!merged) {
                    sparse_waits_.push_back(sem);
                    sparse_values_.push_back(val);
                }
                semaphores = sparse_waits_;
                values = sparse_values_;
                if (semaphores.size() > 16) {
                    submit_stages_.assign(semaphores.size(), kAllCmds);
                    stages_ptr = submit_stages_.data();
                }
            };
            // Check the page mirror's latest bind under submit_mutex_ (serialized with note_mirror_bind() and
            // record_submit_locked()), so every command-buffer submission stamped after note_mirror_bind() waits on
            // the sparse bind before reading any mirrored page.
            if (command_buffer && context_.mirror_bind_semaphore_ && context_.mirror_bind_value_ > mirror_bind_seen_) {
                add_lazy_wait(context_.mirror_bind_semaphore_, context_.mirror_bind_value_);
                mirror_bind_waited = context_.mirror_bind_value_;
            }
            // Check the latest staged title-texture upload timeline under submit_mutex_ (serialized with
            // note_title_upload_locked()), so every queue waits lazily on its next command-buffer submit without
            // empty wait-only vkQueueSubmits per consumer queue.
            if (command_buffer && context_.title_upload_semaphore_ &&
                (context_.title_upload_semaphore_ != title_upload_sem_seen_ ||
                 context_.title_upload_value_ > title_upload_seen_)) {
                add_lazy_wait(context_.title_upload_semaphore_, context_.title_upload_value_);
                title_upload_waited_sem = context_.title_upload_semaphore_;
                title_upload_waited_val = context_.title_upload_value_;
            }
            VkSemaphore stack_signals[17]{progress_};
            std::uint64_t stack_signal_values[17]{ticket};
            VkSemaphore* signals_ptr = stack_signals;
            std::uint64_t* signal_values_ptr = stack_signal_values;
            std::uint32_t signal_count = 1;
            if (dedup_size + 1 > 17) {
                submit_signals_.clear();
                submit_signal_values_.clear();
                submit_signals_.reserve(dedup_size + 1);
                submit_signal_values_.reserve(dedup_size + 1);
                submit_signals_.push_back(progress_);
                submit_signal_values_.push_back(ticket);
            }
            // Decided under the submit lock, so the order of the checks is the order of the submissions. A Signal
            // that is not above what was already submitted for this fence (a held Signal(F, 2) released after an
            // unheld Signal(F, 3)) leaves the Vulkan signal out: the timeline cannot go down, and waits for 2 pass.
            // The value counts as submitted only after the submit succeeded (below).
            for (std::size_t i = 0; i < dedup_size; ++i) {
                const VkSemaphore f = dedup_ptr[i].fence;
                const std::uint64_t v = dedup_ptr[i].value;
                if (!context_.holds().would_signal(f, v)) continue;
                if (dedup_size + 1 > 17) {
                    submit_signals_.push_back(f);
                    submit_signal_values_.push_back(v);
                } else {
                    stack_signals[signal_count] = f;
                    stack_signal_values[signal_count] = v;
                }
                ++signal_count;
            }
            if (dedup_size + 1 > 17) {
                signals_ptr = submit_signals_.data();
                signal_values_ptr = submit_signal_values_.data();
            }
            VkTimelineSemaphoreSubmitInfo timeline{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
            timeline.waitSemaphoreValueCount = std::uint32_t(values.size());
            timeline.pWaitSemaphoreValues = values.data();
            timeline.signalSemaphoreValueCount = signal_count;
            timeline.pSignalSemaphoreValues = signal_values_ptr;
            VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO, &timeline};
            submit.waitSemaphoreCount = std::uint32_t(semaphores.size());
            submit.pWaitSemaphores = semaphores.data();
            submit.pWaitDstStageMask = stages_ptr;
            submit.commandBufferCount = command_buffer ? 1 : 0;
            submit.pCommandBuffers = &command_buffer;
            submit.signalSemaphoreCount = signal_count;
            submit.pSignalSemaphores = signals_ptr;
            // LEONIDA_PASS_TRACE=<n> (with LEONIDA_DEVICE_FAULT=1): in the frame after every n-th present, each
            // submission logs its compute programs in order (repeats folded), for the frame's pass order.
            if (diagnostics) pass_trace(*diagnostics);
            // Publish before the driver can execute a marker or a concurrent fence wait can see the loss.
            if (diagnostics) context_.device_fault()->publish(std::move(*diagnostics));
            DeviceFaultDiagnostics* const ring = context_.device_fault();
            std::size_t ring_at = 0;
            const std::int64_t submit_start = diagnostic_ms();
            if (ring) {
                SubmitRecord record;
                record.at_ms = submit_start;
                record.queue = queue_;
                record.kind = kind_ == QueueKind::direct ? "direct" : kind_ == QueueKind::compute ? "compute" : "copy";
                record.progress = progress_;
                record.ticket = ticket;
                record.buffers = submit.commandBufferCount;
                if (signal_count > 1) {
                    record.fence = signals_ptr[signal_count - 1];
                    record.fence_value = signal_values_ptr[signal_count - 1];
                }
                record.waits = std::uint32_t(semaphores.size());
                for (std::size_t i = 0; i < semaphores.size() && i < SubmitRecord::max_waits; ++i) {
                    record.wait_semaphores[i] = semaphores[i];
                    record.wait_values[i] = values[i];
                }
                ring_at = ring->begin_submit(record);
            }
            const std::int64_t call_start = submit_phases::now_ns();
            res = context_.observe_result(context_.queue_submit(queue_, 1, &submit, context_.gpu_labels_.submit_fence(command_buffer)),
                                          "vkQueueSubmit");
            submit_phases::add_part(submit_phases::part_submit_call, std::uint64_t(submit_phases::now_ns() - call_start));
            context_.gpu_labels_.submitted(command_buffer, res == VK_SUCCESS);
            if (ring) ring->end_submit(ring_at, diagnostic_ms() - submit_start, res);
            if (res == VK_SUCCESS) {
                context_.record_submit_locked(progress_, ticket);
                for (std::uint32_t i = 1; i < signal_count; ++i)
                    context_.holds().signaled(signals_ptr[i], signal_values_ptr[i]);
            }
        }
        if (res == VK_ERROR_DEVICE_LOST) {
            context_.mark_device_lost();
            clear_exec_waits();
            return false;
        }
        const bool ok = (res == VK_SUCCESS);
        if (ok && gpu_latency::enabled()) {
            // LEONIDA_GPU_LATENCY=1: when the GPU finished this submission, measured from the submit.
            const auto kind = int(kind_);
            const auto at = std::chrono::steady_clock::now();
            context_.enqueue_fence_callback(progress_, ticket, [kind, at] { gpu_latency::done(kind, at); });
        }
        if (ok) {
            if (mirror_bind_waited > mirror_bind_seen_) mirror_bind_seen_ = mirror_bind_waited;
            if (title_upload_waited_sem) {
                title_upload_sem_seen_ = title_upload_waited_sem;
                title_upload_seen_ = title_upload_waited_val;
            }
            if (consumed_wait_vis > pending_wait_vis_seq_) pending_wait_vis_seq_ = consumed_wait_vis;
            submitted_ = ticket;
            submitted_gpu_.store(ticket, std::memory_order_release);
            clear_exec_waits(ticket);
            for (auto& slot : buffers_)
                if (slot.buffer == command_buffer) slot.ticket = ticket;
        }
        return ok;
    }

    struct ExecReplayState {
        std::unique_lock<std::mutex>* tracker_lock = nullptr;
        std::vector<KhrInstanceStage>* khr_stages = nullptr;
        std::vector<KhrAsQueryPool>* khr_as_queries = nullptr;
        std::unique_ptr<PredicateReducer::Batch>* pred_batch = nullptr;
        std::unique_ptr<PredicatedBufferCopier::Batch>* copy_batch = nullptr;
        std::unique_ptr<IndirectDispatchCounter::Batch>* counted_batch = nullptr;
        std::unique_ptr<IndirectArgumentClamp::Batch>* clamp_batch = nullptr;
        Depth24Guard* depth24_guard = nullptr;
        ScratchGuard* scratch_guard = nullptr;
        CommandSlot* slot = nullptr;
        bool* ordered_timestamps = nullptr;
    };

    VulkanContext& context_;
    VkQueue queue_;
    QueueKind kind_ = QueueKind::direct;
    std::uint32_t family_;
    VkCommandPool pool_ = VK_NULL_HANDLE;
    VkSemaphore progress_ = VK_NULL_HANDLE;  // our own timeline: which command buffers are done
    std::uint64_t submitted_ = 0;
    std::atomic<std::uint64_t> submitted_gpu_{0};  // mirror of submitted_, read lock-free by has_pending_gpu_work()
    std::vector<CommandSlot> buffers_;
    std::vector<VkEvent> free_events_;  // unsignaled, for order_timestamp_copy
    std::unique_ptr<OcclusionResolve> occlusion_resolve_;
    std::uint64_t storage_init_seen_ = 0;  // the storage_init_version_ this queue has waited for
    std::uint64_t bindless_upload_seen_ = 0;  // the bindless_upload_version_ this queue has waited for
    std::uint64_t mirror_bind_seen_ = 0;      // the page mirror's bind value this queue has waited for
    VkSemaphore title_upload_sem_seen_ = VK_NULL_HANDLE;
    std::uint64_t title_upload_seen_ = 0;     // the staged title-upload timeline value this queue has waited for
    std::uint64_t pending_wait_vis_seq_ = 0;  // CPU Fence::Signal visibility epoch carried from signal-only wait
    std::uint64_t prelude_seen_ = 0;  // the prelude_version_ this queue has waited for (or recorded)
    ReplayFunctions cached_replay_{};
    std::uint64_t cached_replay_version_ = 0;
    void (*cached_replay_on_buffer_write_)(void*) = nullptr;
    void* cached_replay_on_buffer_write_ctx_ = nullptr;
    ExecReplayState exec_replay_state_{};
    std::uint64_t last_sparse_ = 0;
    std::deque<PendingBindings> pending_bindings_;
    std::vector<VkSemaphore> waits_;
    std::vector<std::uint64_t> wait_values_;
    // execute's working storage. Per queue like buffers_: one execute at a time on a queue, as acquire() and
    // submit() already assume; other queues prepare concurrently with their own.
    std::vector<PreparedList> prepared_;
    std::uint32_t open_labels_ = 0;  // queue labels begun and not ended
    bool ok_ = false;
    // ---- LEONIDA_SUBMIT_ASYNC=<n>: the queue's calls run on a worker of its own -----------------------------------
    // ExecuteCommandLists, Signal, Wait and the markers are queued in call order and return at once; one worker per
    // queue runs each through the synchronous path (execute_sync & co.), so the replay, the before-submission
    // refreshes and vkQueueSubmit leave the caller's thread and overlap with its next frame. At most n batches wait per
    // queue; the caller of the next one waits for room (the in-flight bound). A queued call counts as held work
    // (QueueHolds), numbered when it was called, so a Present behind it still waits for it and cross-queue order is
    // the call order. Calls that use the queue's state at once (tile mappings, idle, the destructor) first wait for
    // what is queued. A batch that fails is reported as a held one is (stderr, take_held_failure()). Off (0) by
    // default: every call runs on the caller's thread, as before.
    static unsigned async_depth() {
        static const unsigned depth = [] {
            const char* v = std::getenv("LEONIDA_SUBMIT_ASYNC");
            const int n = v && *v ? std::atoi(v) : 0;
            return n > 0 ? unsigned(n) : 0u;
        }();
        return depth;
    }

    bool async_on() const { return async_depth_ && !synchronous_; }

    struct Call {
        enum class Kind : std::uint8_t { execute, signal, wait, marker };
        Kind kind = Kind::execute;
        bool barrier_after = false;  // Present / after_held() arrived after this call: do not merge past it
        std::uint64_t seq = 0, present_epoch = kUnsetEpoch, execute_call = 0, visibility_seq = 0, value = 0;
        EnqueueSerial enqueue_serial;
        // Immutable closed CommandList snapshots (zero-copy when submitted via shared_ptr<const CommandList>).
        std::vector<std::shared_ptr<const CommandList>> lists;
        std::shared_ptr<void> pin;       // the fence, or the lists' pipeline states and root signatures
        VkSemaphore fence = VK_NULL_HANDLE;
        DebugMarker::Kind marker_kind = DebugMarker::Kind::set;
        std::string label;
        std::chrono::steady_clock::time_point queued_at{};
    };

    struct AsyncStats {  // under inbox_mutex_, since the last line
        std::uint64_t calls = 0, batches = 0, room_waits = 0, room_wait_ns = 0, run_ns = 0, lag_ns = 0, lag_max_ns = 0;
        std::size_t depth_max = 0;
        std::chrono::steady_clock::time_point since = std::chrono::steady_clock::now();
    };

    // Counts distinct frame / barrier groups pending in inbox_ (plus 1 when the worker is busy): intra-frame
    // cross-queue Wait/Signal splits within the same present_epoch stay in ONE frame group so a barrier-heavy frame
    // (25+ Direct batches separated by cross-queue Waits) does not stall the title thread on room_cv_ mid-frame
    // (which would also delay posting cross-queue Signals needed by earlier queued Waits), while async_depth_ still
    // bounds distinct frames (Present / after_held barriers) and max_batches bounds total queued batch memory.
    // inbox_mutex_ held.
    std::size_t pending_frame_groups_locked() const {
        std::size_t groups = worker_busy_ ? 1 : 0;
        bool in_group = false;
        std::uint64_t prev_epoch = kUnsetEpoch;
        bool prev_barrier = false;
        for (const Call& c : inbox_) {
            if (c.kind == Call::Kind::execute) {
                if (!in_group || prev_barrier || c.present_epoch != prev_epoch) {
                    ++groups;
                    in_group = true;
                }
                prev_epoch = c.present_epoch;
                prev_barrier = c.barrier_after;
            } else if (in_group && (prev_barrier || c.present_epoch != prev_epoch)) {
                in_group = false;
                prev_barrier = false;
            } else if (in_group && c.barrier_after) {
                prev_barrier = true;
            }
        }
        return groups;
    }

    void enqueue(Call&& call) {
        const bool batch = call.kind == Call::Kind::execute;
        std::unique_lock lock(inbox_mutex_);
        if (!worker_.joinable()) {
            worker_ = std::thread([this] { async_worker(); });
            async_started_.store(true, std::memory_order_release);
        }
        // n frame groups, with bounded batch headroom for multi-batch frames (max(16n, 64) batches, 4x calls when
        // n >= 2; n = 1 keeps strict single-batch backpressure).
        const std::size_t max_batches =
            async_depth_ <= 1 ? std::size_t(async_depth_)
                              : (std::max)(std::size_t(async_depth_) * 16, std::size_t(64));
        const std::size_t max_calls = max_batches * 4;
        const auto room = [this, batch, max_batches, max_calls] {
            if (async_stop_) return true;
            if (inbox_.size() >= max_calls) return false;
            if (!batch) return true;
            if (inbox_batches_ < async_depth_) return true;
            if (async_depth_ <= 1 || inbox_batches_ >= max_batches) return false;
            return pending_frame_groups_locked() < async_depth_;
        };
        if (!room()) {
            const auto start = std::chrono::steady_clock::now();
            ++room_waiters_;
            room_cv_.wait(lock, room);
            --room_waiters_;
            ++async_stats_.room_waits;
            async_stats_.room_wait_ns += std::uint64_t(
                std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count());
        }
        if (batch) {
            ++inbox_batches_;
            ++async_stats_.batches;
        }
        ++async_stats_.calls;
        call.present_epoch = submit_phases::totals().present_calls.load(std::memory_order_acquire);
        call.seq = context_.holds().next_seq();
        call.queued_at = std::chrono::steady_clock::now();
        context_.holds().held_changed(1);
        context_.holds().work_changed(1);
        inbox_.push_back(std::move(call));
        async_stats_.depth_max = (std::max)(async_stats_.depth_max, inbox_.size());
        lock.unlock();
        inbox_cv_.notify_one();
    }

    void async_worker() {
        t_worker_of_ = this;
        std::vector<Call> group_calls;
        group_calls.reserve(16);
        for (;;) {
            Call single_call;
            group_calls.clear();
            bool have_single = false;
            bool release = false;
            std::size_t exec_count = 0;
            std::int64_t consumed_work = 0;
            {
                std::unique_lock lock(inbox_mutex_);
                inbox_cv_.wait(lock, [this] { return !inbox_.empty() || release_requested_ || async_stop_; });
                release = std::exchange(release_requested_, false);
                if (release) ++consumed_work;
                if (!inbox_.empty()) {
                    if (inbox_.front().kind == Call::Kind::execute) {
                        const std::uint64_t bseq = barrier_seq_.load(std::memory_order_acquire);
                        group_calls.push_back(std::move(inbox_.front()));
                        inbox_.pop_front();
                        exec_count = 1;
                        ++consumed_work;
                        bool stop_after_signal = false;
                        while (!group_calls.back().barrier_after && !inbox_.empty()) {
                            const Call& next = inbox_.front();
                            if (crosses_barrier(group_calls.back().seq, next.seq, group_calls.back().present_epoch,
                                                next.present_epoch, bseq))
                                break;
                            if (next.kind == Call::Kind::execute) {
                                if (stop_after_signal) break;
                                group_calls.push_back(std::move(inbox_.front()));
                                inbox_.pop_front();
                                ++exec_count;
                                ++consumed_work;
                            } else if (next.kind == Call::Kind::signal) {
                                group_calls.push_back(std::move(inbox_.front()));
                                inbox_.pop_front();
                                ++consumed_work;
                                if (context_.has_unsatisfied_waiter(group_calls.back().fence, group_calls.back().value))
                                    stop_after_signal = true;
                            } else if (next.kind == Call::Kind::wait && !stop_after_signal &&
                                       wait_already_completed(next.fence, next.value)) {
                                group_calls.push_back(std::move(inbox_.front()));
                                inbox_.pop_front();
                                ++consumed_work;
                            } else {
                                break;
                            }
                        }
                        running_seq_ = group_calls.front().seq;
                        // A coalesced group executes as ONE in-flight submission; release the (exec_count - 1) merged
                        // batch slots immediately so enqueue() can accumulate the next coalescing window without stalling.
                        if (exec_count > 1) inbox_batches_ -= (exec_count - 1);
                        if (room_waiters_ > 0) room_cv_.notify_all();
                    } else {
                        single_call = std::move(inbox_.front());
                        inbox_.pop_front();
                        running_seq_ = single_call.seq;
                        have_single = true;
                        ++consumed_work;
                        if (room_waiters_ > 0) room_cv_.notify_all();
                    }
                } else if (!release) {
                    return;  // stopping, nothing queued
                }
                worker_busy_ = true;
                // Transfer consumed inbox/release work into this worker turn: keep +1 active until pump_ready() ends.
                if (consumed_work != 1) context_.holds().work_changed(1 - consumed_work);
            }
            struct WorkerTurnGuard {
                QueueHolds<VkSemaphore>& holds;
                ~WorkerTurnGuard() { holds.work_changed(-1); }
            } turn_guard{context_.holds()};
            const std::size_t group_count = group_calls.size();
            const bool have = have_single || group_count != 0;
            HeldCountGuard inbox_held_guard{
                context_.holds(), have ? -static_cast<std::int64_t>(group_count ? group_count : 1) : 0};
            // Wrap the entire worker turn (release_held, call execution, and pump_ready) so gate-released batches and
            // deferred presents running on async_worker are attributed to worker_submit, never to title threads.
            const submit_phases::WorkerScope busy(submit_phases::worker_submit);
            const auto turn_start = std::chrono::steady_clock::now();
            DrainInfo rel_info{};
            const bool released = release && release_held(/*from_pump=*/false, &rel_info);
            if (released && have) {
                const std::uint64_t next_seq = group_count ? group_calls.front().seq : single_call.seq;
                const std::uint64_t next_epoch =
                    group_count ? group_calls.front().present_epoch : single_call.present_epoch;
                const std::uint64_t bseq = barrier_seq_.load(std::memory_order_acquire);
                if (rel_info.stopped_for_barrier ||
                    crosses_barrier(rel_info.last_seq, next_seq, rel_info.last_epoch, next_epoch, bseq)) {
                    // Pre-barrier held work finished; pump now and wait for satisfied pre-barrier Presents /
                    // after_held() callbacks to finish on the barrier thread before post-barrier inbox calls execute.
                    context_.holds().pump_ready();
                    if (next_seq) context_.holds().wait_barriers_drained(next_seq - 1);
                }
            }
            auto have_start = turn_start;
            if (have) {
                have_start = std::chrono::steady_clock::now();
                if (group_count == 1) {
                    run_call(group_calls[0]);
                } else if (group_count > 1) {
                    run_coalesced_calls(group_calls, exec_count);
                } else {
                    run_call(single_call);
                }
            }
            const auto turn_end = std::chrono::steady_clock::now();
            const std::uint64_t ran_ns = (have || released)
                ? std::uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(turn_end - turn_start).count())
                : 0;
            {
                std::lock_guard lock(inbox_mutex_);
                async_stats_.run_ns += ran_ns + async_pump_ns_.exchange(0, std::memory_order_relaxed);
                if (have) {
                    if (group_count != 0) {
                        for (const Call& c : group_calls) {
                            const auto lag = std::uint64_t(
                                std::chrono::duration_cast<std::chrono::nanoseconds>(have_start - c.queued_at).count());
                            async_stats_.lag_ns += lag;
                            async_stats_.lag_max_ns = (std::max)(async_stats_.lag_max_ns, lag);
                        }
                    } else {
                        const auto lag = std::uint64_t(
                            std::chrono::duration_cast<std::chrono::nanoseconds>(have_start - single_call.queued_at)
                                .count());
                        async_stats_.lag_ns += lag;
                        async_stats_.lag_max_ns = (std::max)(async_stats_.lag_max_ns, lag);
                    }
                }
                running_seq_ = 0;
                worker_busy_ = false;
                if (exec_count != 0) --inbox_batches_;
                if (room_waiters_ > 0) room_cv_.notify_all();
                idle_cv_.notify_all();
                report_async_locked();
            }
            inbox_held_guard.release();
            single_call = Call{};
            group_calls.clear();  // the snapshots, pins, and serials go before the pump, outside the lock
            if (have || released) {
                const auto pump_start = std::chrono::steady_clock::now();
                context_.holds().pump_ready();
                const auto pump_ns = std::uint64_t(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - pump_start)
                        .count());
                async_pump_ns_.fetch_add(pump_ns, std::memory_order_relaxed);
            }
        }
    }

    // Coalesces consecutive execute and Signal items from async_worker's inbox into ONE before_execute_() + ONE
    // vkQueueSubmit (with all Signals attached to that submit's timeline signaling, deduplicated to the highest value
    // per fence), or moves them into held_ in exact FIFO order when blocked by an unmet Wait so they still coalesce
    // when released. Falls back per-call in exact FIFO order on non-device-lost replay failure so valid peers and
    // trailing Signals are never dropped.
    void run_coalesced_calls(std::vector<Call>& calls, std::size_t exec_count) {
        try {
            run_coalesced_calls_unguarded(calls, exec_count);
        } catch (const std::exception& e) {
            worker_threw(calls.front(), e.what());
        } catch (...) {
            worker_threw(calls.front(), "not a std::exception");
        }
        async_seq_ = 0;
    }

    void run_coalesced_calls_unguarded(std::vector<Call>& calls, std::size_t exec_count) {
        QueueHolds<VkSemaphore>::WorkGuard work_guard;
        std::vector<std::shared_ptr<void>> retired;
        std::vector<EnqueueSerial> dropped_serials;
        auto hold_all_locked = [&] {
            for (Call& call : calls) {
                if (call.kind == Call::Kind::wait) {
                    waits_.push_back(call.fence);
                    wait_values_.push_back(call.value);
                    if (call.pin) wait_pins_.push_back(std::move(call.pin));
                    continue;
                }
                async_seq_ = call.seq;
                Deferred op;
                op.barrier_after = call.barrier_after;
                op.present_epoch = call.present_epoch;
                op.visibility_seq = call.visibility_seq;
                if (call.kind == Call::Kind::signal) {
                    op.kind = Deferred::Kind::signal;
                    op.fence = call.fence;
                    op.value = call.value;
                } else {
                    op.kind = Deferred::Kind::batch;
                    op.execute_call = call.execute_call;
                    op.enqueue_serial = std::move(call.enqueue_serial);
                    op.lists = std::move(call.lists);
                }
                if (call.pin) op.pins.push_back(std::move(call.pin));
                hold(std::move(op));
            }
            async_seq_ = 0;
        };
        {
            std::unique_lock state(state_mutex_, std::defer_lock);
            submit_phases::lock_timed(state);
            if (executing_ && !blocked_by_wait_locked() && !context_.is_device_lost()) {
                exec_cv_.wait(state, [this] {
                    return !executing_ || blocked_by_wait_locked() || context_.is_device_lost();
                });
            }
            if (context_.is_device_lost()) {
                for (Call& call : calls) {
                    if (call.kind == Call::Kind::wait) context_.note_wait_removed(call.fence, call.value);
                    if (call.pin) retired.push_back(std::move(call.pin));
                    if (call.enqueue_serial.value) dropped_serials.push_back(std::move(call.enqueue_serial));
                }
            } else if (!executing_ && !blocked_by_wait_locked()) {
                work_guard.arm(context_.holds());
                executing_ = true;
                ExecGuard exec_guard{this, state};
                DrainInfo pre_info{};
                if (!held_.empty()) drain_held_while_executing(state, retired, false, &pre_info);
                const std::uint64_t bseq_now = barrier_seq_.load(std::memory_order_acquire);
                const bool barrier_before_me =
                    pre_info.released &&
                    (pre_info.stopped_for_barrier ||
                     crosses_barrier(pre_info.last_seq, calls.front().seq, pre_info.last_epoch,
                                     calls.front().present_epoch, bseq_now));
                if (context_.is_device_lost()) {
                    for (Call& call : calls) {
                        if (call.kind == Call::Kind::wait) context_.note_wait_removed(call.fence, call.value);
                        if (call.pin) retired.push_back(std::move(call.pin));
                        if (call.enqueue_serial.value) dropped_serials.push_back(std::move(call.enqueue_serial));
                    }
                } else if (must_hold() || barrier_before_me) {
                    hold_all_locked();
                    if (barrier_before_me) pause_release_for_barrier_.store(true, std::memory_order_release);
                } else {
                    set_active_seq_locked(calls.front().seq);
                    take_waits_to_exec_locked();
                    state.unlock();

                    exec_list_ptrs_.clear();
                    exec_batch_breaks_.clear();
                    exec_pending_signals_.clear();
                    std::uint64_t group_vis_seq = 0;
                    for (const Call& call : calls) {
                        if (call.kind == Call::Kind::execute) {
                            if (!exec_list_ptrs_.empty() && !call.lists.empty())
                                exec_batch_breaks_.push_back(exec_list_ptrs_.size());
                            for (const auto& list : call.lists) exec_list_ptrs_.push_back(list.get());
                            group_vis_seq = (std::max)(group_vis_seq, call.visibility_seq);
                        } else if (call.kind == Call::Kind::signal) {
                            exec_pending_signals_.push_back({call.fence, call.value, call.visibility_seq});
                        } else if (call.kind == Call::Kind::wait) {
                            // Preserve CPU Fence::Signal visibility edge across coalesced completed waits.
                            const std::uint64_t w_vis = context_.holds().signaled_vis_seq(call.fence, call.value);
                            group_vis_seq = (std::max)(group_vis_seq, w_vis);
                        }
                    }

                    wait_for_preceding_barriers(calls.front().seq, exec_list_ptrs_);
                    const ReplayResult result =
                        execute_now(exec_list_ptrs_, calls.front().execute_call, exec_batch_breaks_,
                                    exec_pending_signals_, group_vis_seq, /*from_held=*/false);
                    if (result == ReplayResult::success) {
                        note_coalesced_submit(exec_count);
                        for (Call& call : calls) {
                            if (call.kind == Call::Kind::wait) {
                                context_.note_wait_removed(call.fence, call.value);
                                if (call.pin) exec_retired_pins_.push_back(std::move(call.pin));
                                continue;
                            }
                            if (call.pin) {
                                if (submitted_) exec_ticketed_pins_.push_back({submitted_, std::move(call.pin)});
                                else exec_retired_pins_.push_back(std::move(call.pin));
                            }
                            call.enqueue_serial.finish();
                        }
                    } else if (exec_count > 1 && !context_.is_device_lost()) {
                        for (Call& call : calls) {
                            if (call.kind == Call::Kind::execute) {
                                exec_list_ptrs_.clear();
                                for (const auto& list : call.lists) exec_list_ptrs_.push_back(list.get());
                                const ReplayResult sub =
                                    execute_now(exec_list_ptrs_, call.execute_call, {}, {}, call.visibility_seq, false);
                                note_coalesced_submit(1);
                                if (sub != ReplayResult::success) {
                                    int none = static_cast<int>(ReplayResult::success);
                                    held_failure_.compare_exchange_strong(none, static_cast<int>(sub));
                                    std::fprintf(stderr,
                                                 "[rt] d3d12: a queued ExecuteCommandLists failed on the queue's worker "
                                                 "(result %d); nothing was submitted for it\n",
                                                 static_cast<int>(sub));
                                }
                                if (call.pin) {
                                    if (sub == ReplayResult::success && submitted_)
                                        exec_ticketed_pins_.push_back({submitted_, std::move(call.pin)});
                                    else
                                        exec_retired_pins_.push_back(std::move(call.pin));
                                }
                                call.enqueue_serial.finish();
                            } else if (call.kind == Call::Kind::signal) {
                                auto saved_waits = std::move(exec_waits_);
                                auto saved_values = std::move(exec_wait_values_);
                                auto saved_pins = std::move(exec_wait_pins_);
                                const std::size_t saved_user = std::exchange(exec_user_wait_count_, 0);
                                exec_waits_.clear();
                                exec_wait_values_.clear();
                                exec_wait_pins_.clear();
                                const bool ok =
                                    submit(VK_NULL_HANDLE, call.fence, call.value, nullptr, call.visibility_seq);
                                if (!ok) {
                                    int none = static_cast<int>(ReplayResult::success);
                                    held_failure_.compare_exchange_strong(none, static_cast<int>(ReplayResult::unsupported));
                                    std::fprintf(stderr, "[rt] d3d12: a queued Signal failed on the queue's worker\n");
                                }
                                if (call.pin) {
                                    if (ok && submitted_)
                                        exec_ticketed_pins_.push_back({submitted_, std::move(call.pin)});
                                    else
                                        exec_retired_pins_.push_back(std::move(call.pin));
                                }
                                exec_waits_ = std::move(saved_waits);
                                exec_wait_values_ = std::move(saved_values);
                                exec_wait_pins_ = std::move(saved_pins);
                                exec_user_wait_count_ = saved_user;
                            } else if (call.kind == Call::Kind::wait) {
                                const std::uint64_t w_vis = context_.holds().signaled_vis_seq(call.fence, call.value);
                                pending_wait_vis_seq_ = (std::max)(pending_wait_vis_seq_, w_vis);
                                context_.note_wait_removed(call.fence, call.value);
                                if (call.pin) exec_retired_pins_.push_back(std::move(call.pin));
                            }
                        }
                    } else {
                        note_coalesced_submit(exec_count);
                        int none = static_cast<int>(ReplayResult::success);
                        held_failure_.compare_exchange_strong(none, static_cast<int>(result));
                        auto saved_waits = std::move(exec_waits_);
                        auto saved_values = std::move(exec_wait_values_);
                        auto saved_pins = std::move(exec_wait_pins_);
                        const std::size_t saved_user = std::exchange(exec_user_wait_count_, 0);
                        exec_waits_.clear();
                        exec_wait_values_.clear();
                        exec_wait_pins_.clear();
                        for (Call& call : calls) {
                            if (call.kind == Call::Kind::signal && !context_.is_device_lost()) {
                                const bool ok =
                                    submit(VK_NULL_HANDLE, call.fence, call.value, nullptr, call.visibility_seq);
                                if (!ok) {
                                    int none_s = static_cast<int>(ReplayResult::success);
                                    held_failure_.compare_exchange_strong(none_s, static_cast<int>(ReplayResult::unsupported));
                                    std::fprintf(stderr, "[rt] d3d12: a queued Signal failed on the queue's worker\n");
                                }
                                if (call.pin) {
                                    if (ok && submitted_)
                                        exec_ticketed_pins_.push_back({submitted_, std::move(call.pin)});
                                    else
                                        exec_retired_pins_.push_back(std::move(call.pin));
                                }
                            } else {
                                if (call.kind == Call::Kind::wait) context_.note_wait_removed(call.fence, call.value);
                                if (call.pin) exec_retired_pins_.push_back(std::move(call.pin));
                                call.enqueue_serial.finish();
                            }
                        }
                        exec_waits_ = std::move(saved_waits);
                        exec_wait_values_ = std::move(saved_values);
                        exec_wait_pins_ = std::move(saved_pins);
                        exec_user_wait_count_ = saved_user;
                    }

                    state.lock();
                    restore_exec_waits_locked();
                    flush_exec_pins_locked(retired);
                    set_active_seq_locked(0);
                    const std::uint64_t bseq_after = barrier_seq_.load(std::memory_order_acquire);
                    const bool barrier_after_group =
                        calls.back().barrier_after ||
                        (!held_.empty() &&
                         crosses_barrier(calls.back().seq, held_.front().seq, calls.back().present_epoch,
                                         held_.front().present_epoch, bseq_after));
                    if (barrier_after_group) {
                        if (!held_.empty()) pause_release_for_barrier_.store(true, std::memory_order_release);
                    } else {
                        drain_held_while_executing(state, retired);
                    }
                }
                exec_guard.release();
            } else {
                hold_all_locked();
            }
            drain_pins(retired);
        }
        for (EnqueueSerial& s : dropped_serials) s.finish();
        retired.clear();
    }

    // An exception on the worker would end the process (std::terminate) and leave the queue's callers waiting: it is
    // turned into a held failure and a lost device instead, as a failed submission would be, and the worker goes on
    // (each later call fails at once on the lost device).
    void run_call(Call& call) {
        try {
            run_call_unguarded(call);
        } catch (const std::exception& e) {
            worker_threw(call, e.what());
        } catch (...) {
            worker_threw(call, "not a std::exception");
        }
        async_seq_ = 0;
    }

    void worker_threw(const Call& call, const char* what) {
        int none = static_cast<int>(ReplayResult::success);
        held_failure_.compare_exchange_strong(none, static_cast<int>(ReplayResult::unsupported));
        std::fprintf(stderr, "[rt] d3d12: the queue's worker caught an exception running a queued %s (%s); the device is "
                             "marked lost\n",
                     call.kind == Call::Kind::execute ? "ExecuteCommandLists"
                     : call.kind == Call::Kind::signal ? "Signal"
                     : call.kind == Call::Kind::wait   ? "Wait"
                                                       : "marker",
                     what);
        context_.mark_device_lost();
    }

    void run_call_unguarded(Call& call) {
        async_seq_ = call.seq;
        switch (call.kind) {
        case Call::Kind::execute: {
            const CommandList* small_ptrs[16];
            std::vector<const CommandList*> large_ptrs;
            const CommandList* const* ptr_data = nullptr;
            if (call.lists.size() <= 16) {
                for (std::size_t i = 0; i < call.lists.size(); ++i) small_ptrs[i] = call.lists[i].get();
                ptr_data = small_ptrs;
            } else {
                large_ptrs.reserve(call.lists.size());
                for (const auto& list : call.lists) large_ptrs.push_back(list.get());
                ptr_data = large_ptrs.data();
            }
            const std::size_t count = call.lists.size();
            const ReplayResult result = execute_sync_impl(
                {ptr_data, count}, {}, call.execute_call, [&call] { return std::move(call.lists); },
                std::move(call.pin), std::move(call.enqueue_serial), call.barrier_after, call.present_epoch,
                call.visibility_seq);
            if (result != ReplayResult::success) {
                int none = static_cast<int>(ReplayResult::success);
                held_failure_.compare_exchange_strong(none, static_cast<int>(result));
                std::fprintf(stderr, "[rt] d3d12: a queued ExecuteCommandLists failed on the queue's worker (result "
                                     "%d); nothing was submitted for it\n",
                             static_cast<int>(result));
            }
            break;
        }
        case Call::Kind::signal:
            if (!signal_sync(call.fence, call.value, {}, std::move(call.pin), call.present_epoch,
                             call.visibility_seq)) {
                int none = static_cast<int>(ReplayResult::success);
                held_failure_.compare_exchange_strong(none, static_cast<int>(ReplayResult::unsupported));
                std::fprintf(stderr, "[rt] d3d12: a queued Signal failed on the queue's worker\n");
            }
            break;
        case Call::Kind::wait: wait_sync(call.fence, call.value, {}, std::move(call.pin), /*already_tracked=*/true); break;
        case Call::Kind::marker: marker_sync(call.marker_kind, call.label.c_str(), call.present_epoch); break;
        }
    }

    void note_coalesced_submit(std::size_t batches) noexcept {
        coalesced_batches_.fetch_add(batches, std::memory_order_relaxed);
        coalesced_submits_.fetch_add(1, std::memory_order_relaxed);
    }

    // Every 10 s, per queue: what the worker did. Called with inbox_mutex_ held.
    void report_async_locked() {
        const auto now = std::chrono::steady_clock::now();
        const double s = std::chrono::duration<double>(now - async_stats_.since).count();
        if (s < 10.0) return;
        static constexpr const char* kinds[] = {"direct", "compute", "copy"};
        const AsyncStats& a = async_stats_;
        const std::uint64_t merged_batches = coalesced_batches_.exchange(0, std::memory_order_relaxed);
        const std::uint64_t merged_submits = coalesced_submits_.exchange(0, std::memory_order_relaxed);
        std::fprintf(stderr,
                     "[rt] submit-async: %s queue %p: %.1f calls/s (%.1f batches/s, merged %llu batches into %llu "
                     "submits), queued now %zu (max %zu), worker busy %.0f%%, call waited for the worker avg %.2f ms "
                     "max %.2f ms, caller waited for room %llu times %.1f ms\n",
                     kinds[int(kind_) < 3 ? int(kind_) : 0], static_cast<const void*>(this), double(a.calls) / s,
                     double(a.batches) / s, static_cast<unsigned long long>(merged_batches),
                     static_cast<unsigned long long>(merged_submits), inbox_.size(), a.depth_max,
                     100.0 * double(a.run_ns) / (s * 1e9),
                     a.calls ? double(a.lag_ns) / double(a.calls) / 1e6 : 0.0, double(a.lag_max_ns) / 1e6,
                     static_cast<unsigned long long>(a.room_waits), double(a.room_wait_ns) / 1e6);
        async_stats_ = AsyncStats{};
    }

    // What was queued before this call has run (held work may stay held, as without the worker). A worker that does
    // not come back (a GPU hang inside wait_for_room, say) is reported every 10 s, and after 60 s the caller goes on
    // without it: false.
    bool flush_async() {
        if (!async_started_.load(std::memory_order_acquire) || t_worker_of_ == this) return true;
        std::unique_lock lock(inbox_mutex_);
        const auto idle = [this] { return inbox_.empty() && !worker_busy_ && !release_requested_; };
        for (int slice = 1; slice <= 6; ++slice) {
            if (idle_cv_.wait_for(lock, std::chrono::seconds(10), idle)) return true;
            std::fprintf(stderr, "[rt] d3d12: waited %d s for a queue's worker to finish what is queued (%zu calls, "
                                 "running call %llu)%s\n",
                         slice * 10, inbox_.size(), static_cast<unsigned long long>(running_seq_),
                         slice == 6 ? "; going on without it" : "");
        }
        return false;
    }

    void stop_async() {
        if (!async_started_.load(std::memory_order_acquire)) return;
        const bool drained = flush_async();
        {
            std::lock_guard lock(inbox_mutex_);
            async_stop_ = true;
        }
        inbox_cv_.notify_all();
        room_cv_.notify_all();
        if (!worker_.joinable()) return;
        if (drained) {
            worker_.join();
            return;
        }
        // A worker stuck for a minute holds this queue's objects: it is left running rather than joined forever
        // (only at teardown; the process is ending or the device is gone).
        std::fprintf(stderr, "[rt] d3d12: a queue's worker did not finish; it is detached at the queue's destruction\n");
        worker_.detach();
    }

    std::mutex inbox_mutex_;
    std::condition_variable inbox_cv_, room_cv_, idle_cv_;
    std::deque<Call> inbox_;
    std::size_t inbox_batches_ = 0;
    std::size_t room_waiters_ = 0;
    std::uint64_t running_seq_ = 0;      // oldest seq of the call(s) the worker runs now
    bool worker_busy_ = false, release_requested_ = false, async_stop_ = false;
    AsyncStats async_stats_;
    std::atomic<std::uint64_t> async_pump_ns_{0};  // pump_ready() ns folded into async_stats_.run_ns on next lock
    unsigned async_depth_ = async_depth();
    std::atomic<bool> async_started_{false};
    bool synchronous_ = false;  // make_synchronous()
    std::atomic<bool> skip_before_execute_{false};  // set_skip_before_execute()
    std::atomic<std::uint64_t> coalesced_batches_{0};
    std::atomic<std::uint64_t> coalesced_submits_{0};
    std::thread worker_;
    static inline thread_local VulkanQueue* t_worker_of_ = nullptr;  // the queue whose worker this thread is
    static inline thread_local std::uint64_t async_seq_ = 0;  // the sequence number of the call being run
    std::atomic<std::uint64_t> held_front_seq_{0};  // min_nonzero(active_seq_, held_.front().seq); see release_held()
    std::atomic<bool> has_held_{false};  // !held_.empty(), lock-free check for cross-queue release_ready()
    std::atomic<std::uint64_t> barrier_seq_{0};  // monotonic newest seq seen by oldest_held_seq() (after_held barrier)
    std::atomic<bool> pause_release_for_barrier_{false};  // yield once to pump_ready() so run_barriers() fires first

    // Wait-before-Signal & queue state. state_mutex_ is held ONLY for short bookkeeping (never across
    // before_execute_(), execute_now(), submit(), bind_tiles(), or pin destruction). Execution on this queue is
    // serialized by executing_ (true while one thread runs/drains work): Wait/Signal/Marker and wait-blocked
    // batches append under state_mutex_ without blocking, while unblocked sync execute() waits on exec_cv_.
    std::mutex state_mutex_;
    std::condition_variable exec_cv_;
    bool executing_ = false;
    std::uint64_t active_seq_ = 0;
    std::deque<Deferred> held_;
    std::vector<std::shared_ptr<void>> wait_pins_;     // keep the fences of waits_ alive
    std::vector<std::shared_ptr<void>> retired_pins_;  // released by the caller once state_mutex_ is dropped
    struct TicketedPin {
        std::uint64_t ticket;
        std::shared_ptr<void> pin;
    };
    std::deque<TicketedPin> pin_queue_;  // released once progress_ reaches the ticket
    // Execution-owned wait, pin & scratch staging (touched only by the single thread owning executing_ == true).
    std::vector<VkSemaphore> exec_waits_;
    std::vector<std::uint64_t> exec_wait_values_;
    std::size_t exec_user_wait_count_ = 0;
    std::vector<std::shared_ptr<void>> exec_wait_pins_;
    std::vector<std::shared_ptr<void>> exec_retired_pins_;
    std::vector<TicketedPin> exec_ticketed_pins_;
    std::vector<const CommandList*> exec_list_ptrs_;
    std::vector<std::size_t> exec_batch_breaks_;
    std::vector<PendingSignal> exec_pending_signals_;
    std::vector<VulkanContext::GeometryUpload> geometry_uploads_;
    std::vector<ResourceId> storage_images_scratch_;
    std::vector<VulkanContext::PendingZero> zero_buffers_scratch_;
    std::vector<VkSemaphore> sparse_waits_;
    std::vector<std::uint64_t> sparse_values_;
    std::vector<VkPipelineStageFlags> submit_stages_;
    std::vector<VkSemaphore> submit_signals_;
    std::vector<std::uint64_t> submit_signal_values_;
    EmitScratch emit_scratch_;
    std::atomic<int> held_failure_{0};                 // a ReplayResult: the first held batch that failed
};

} // namespace leonida
