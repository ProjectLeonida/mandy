// Pure-CPU tests for runtime/src/vulkan_predicate.h:
//   - skips undersized memory heaps (VUID-vkAllocateMemory-pAllocateInfo-01713) via memory_type_order;
//   - falls back to subsequent compatible memory types on VK_ERROR_OUT_OF_DEVICE_MEMORY;
//   - fails cleanly without calling vkAllocateMemory when no compatible heap fits;
//   - cleans up handles cleanly when every compatible memory type returns OOM.
// No GPU, Vulkan loader, or game files required.
#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include "../src/vulkan_predicate.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <utility>
#include <vector>

#define CHECK(cond) do { \
    if (!(cond)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #cond); return 1; } \
} while (0)

namespace {

struct MockState {
    std::uint32_t memory_type_bits = 0x3u;
    std::uint32_t fail_oom_mask = 0u;
    std::uint64_t next_handle = 1;
    int destroy_buffer_calls = 0;
    int free_memory_calls = 0;
    int destroy_pool_calls = 0;
    int update_desc_calls = 0;
    std::vector<std::pair<std::uint32_t, VkDeviceSize>> alloc_attempts;
};

MockState* g_mock = nullptr;

template<typename Handle>
Handle make_handle(std::uint64_t id) {
    return reinterpret_cast<Handle>(static_cast<std::uintptr_t>(id));
}

VKAPI_ATTR VkResult VKAPI_CALL mock_create_buffer(VkDevice, const VkBufferCreateInfo* info,
                                                  const VkAllocationCallbacks*, VkBuffer* out) {
    *out = make_handle<VkBuffer>(info->size);  // encode byte size for requirements lookup
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL mock_destroy_buffer(VkDevice, VkBuffer, const VkAllocationCallbacks*) {
    ++g_mock->destroy_buffer_calls;
}

VKAPI_ATTR void VKAPI_CALL mock_get_buffer_memory_requirements(VkDevice, VkBuffer buffer,
                                                               VkMemoryRequirements* req) {
    req->size = static_cast<VkDeviceSize>(reinterpret_cast<std::uintptr_t>(buffer));
    req->alignment = 256;
    req->memoryTypeBits = g_mock->memory_type_bits;
}

VKAPI_ATTR VkResult VKAPI_CALL mock_allocate_memory(VkDevice, const VkMemoryAllocateInfo* info,
                                                    const VkAllocationCallbacks*, VkDeviceMemory* out) {
    g_mock->alloc_attempts.push_back({info->memoryTypeIndex, info->allocationSize});
    if (info->memoryTypeIndex < 32 && (g_mock->fail_oom_mask & (1u << info->memoryTypeIndex))) {
        *out = VK_NULL_HANDLE;
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    }
    *out = make_handle<VkDeviceMemory>(++g_mock->next_handle);
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL mock_free_memory(VkDevice, VkDeviceMemory, const VkAllocationCallbacks*) {
    ++g_mock->free_memory_calls;
}

VKAPI_ATTR VkResult VKAPI_CALL mock_bind_buffer_memory(VkDevice, VkBuffer, VkDeviceMemory, VkDeviceSize) {
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL mock_create_descriptor_pool(VkDevice, const VkDescriptorPoolCreateInfo*,
                                                           const VkAllocationCallbacks*, VkDescriptorPool* out) {
    *out = make_handle<VkDescriptorPool>(++g_mock->next_handle);
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL mock_destroy_descriptor_pool(VkDevice, VkDescriptorPool, const VkAllocationCallbacks*) {
    ++g_mock->destroy_pool_calls;
}

VKAPI_ATTR VkResult VKAPI_CALL mock_allocate_descriptor_sets(VkDevice, const VkDescriptorSetAllocateInfo*,
                                                             VkDescriptorSet* out) {
    *out = make_handle<VkDescriptorSet>(++g_mock->next_handle);
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL mock_update_descriptor_sets(VkDevice, std::uint32_t, const VkWriteDescriptorSet*,
                                                       std::uint32_t, const VkCopyDescriptorSet*) {
    ++g_mock->update_desc_calls;
}

VKAPI_ATTR VkResult VKAPI_CALL mock_create_shader_module(VkDevice, const VkShaderModuleCreateInfo*,
                                                         const VkAllocationCallbacks*, VkShaderModule* out) {
    *out = make_handle<VkShaderModule>(++g_mock->next_handle);
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL mock_destroy_shader_module(VkDevice, VkShaderModule, const VkAllocationCallbacks*) {}

VKAPI_ATTR VkResult VKAPI_CALL mock_create_descriptor_set_layout(VkDevice, const VkDescriptorSetLayoutCreateInfo*,
                                                                 const VkAllocationCallbacks*,
                                                                 VkDescriptorSetLayout* out) {
    *out = make_handle<VkDescriptorSetLayout>(++g_mock->next_handle);
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL mock_destroy_descriptor_set_layout(VkDevice, VkDescriptorSetLayout,
                                                              const VkAllocationCallbacks*) {}

VKAPI_ATTR VkResult VKAPI_CALL mock_create_pipeline_layout(VkDevice, const VkPipelineLayoutCreateInfo*,
                                                           const VkAllocationCallbacks*, VkPipelineLayout* out) {
    *out = make_handle<VkPipelineLayout>(++g_mock->next_handle);
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL mock_destroy_pipeline_layout(VkDevice, VkPipelineLayout, const VkAllocationCallbacks*) {}

VKAPI_ATTR VkResult VKAPI_CALL mock_create_compute_pipelines(VkDevice, VkPipelineCache, std::uint32_t,
                                                             const VkComputePipelineCreateInfo*,
                                                             const VkAllocationCallbacks*, VkPipeline* out) {
    *out = make_handle<VkPipeline>(++g_mock->next_handle);
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL mock_destroy_pipeline(VkDevice, VkPipeline, const VkAllocationCallbacks*) {}

VKAPI_ATTR void VKAPI_CALL mock_cmd_copy_buffer(VkCommandBuffer, VkBuffer, VkBuffer, std::uint32_t,
                                                const VkBufferCopy*) {}

VKAPI_ATTR void VKAPI_CALL mock_cmd_pipeline_barrier(VkCommandBuffer, VkPipelineStageFlags, VkPipelineStageFlags,
                                                     VkDependencyFlags, std::uint32_t, const VkMemoryBarrier*,
                                                     std::uint32_t, const VkBufferMemoryBarrier*,
                                                     std::uint32_t, const VkImageMemoryBarrier*) {}

VKAPI_ATTR void VKAPI_CALL mock_cmd_bind_pipeline(VkCommandBuffer, VkPipelineBindPoint, VkPipeline) {}

VKAPI_ATTR void VKAPI_CALL mock_cmd_bind_descriptor_sets(VkCommandBuffer, VkPipelineBindPoint, VkPipelineLayout,
                                                         std::uint32_t, std::uint32_t, const VkDescriptorSet*,
                                                         std::uint32_t, const std::uint32_t*) {}

VKAPI_ATTR void VKAPI_CALL mock_cmd_push_constants(VkCommandBuffer, VkPipelineLayout, VkShaderStageFlags,
                                                   std::uint32_t, std::uint32_t, const void*) {}

VKAPI_ATTR void VKAPI_CALL mock_cmd_dispatch(VkCommandBuffer, std::uint32_t, std::uint32_t, std::uint32_t) {}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL mock_get_device_proc_addr(VkDevice, const char* name) {
    if (std::strcmp(name, "vkCreateBuffer") == 0) return reinterpret_cast<PFN_vkVoidFunction>(mock_create_buffer);
    if (std::strcmp(name, "vkDestroyBuffer") == 0) return reinterpret_cast<PFN_vkVoidFunction>(mock_destroy_buffer);
    if (std::strcmp(name, "vkGetBufferMemoryRequirements") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(mock_get_buffer_memory_requirements);
    if (std::strcmp(name, "vkAllocateMemory") == 0) return reinterpret_cast<PFN_vkVoidFunction>(mock_allocate_memory);
    if (std::strcmp(name, "vkFreeMemory") == 0) return reinterpret_cast<PFN_vkVoidFunction>(mock_free_memory);
    if (std::strcmp(name, "vkBindBufferMemory") == 0) return reinterpret_cast<PFN_vkVoidFunction>(mock_bind_buffer_memory);
    if (std::strcmp(name, "vkCreateDescriptorPool") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(mock_create_descriptor_pool);
    if (std::strcmp(name, "vkDestroyDescriptorPool") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(mock_destroy_descriptor_pool);
    if (std::strcmp(name, "vkAllocateDescriptorSets") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(mock_allocate_descriptor_sets);
    if (std::strcmp(name, "vkUpdateDescriptorSets") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(mock_update_descriptor_sets);
    if (std::strcmp(name, "vkCreateShaderModule") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(mock_create_shader_module);
    if (std::strcmp(name, "vkDestroyShaderModule") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(mock_destroy_shader_module);
    if (std::strcmp(name, "vkCreateDescriptorSetLayout") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(mock_create_descriptor_set_layout);
    if (std::strcmp(name, "vkDestroyDescriptorSetLayout") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(mock_destroy_descriptor_set_layout);
    if (std::strcmp(name, "vkCreatePipelineLayout") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(mock_create_pipeline_layout);
    if (std::strcmp(name, "vkDestroyPipelineLayout") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(mock_destroy_pipeline_layout);
    if (std::strcmp(name, "vkCreateComputePipelines") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(mock_create_compute_pipelines);
    if (std::strcmp(name, "vkDestroyPipeline") == 0) return reinterpret_cast<PFN_vkVoidFunction>(mock_destroy_pipeline);
    if (std::strcmp(name, "vkCmdCopyBuffer") == 0) return reinterpret_cast<PFN_vkVoidFunction>(mock_cmd_copy_buffer);
    if (std::strcmp(name, "vkCmdPipelineBarrier") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(mock_cmd_pipeline_barrier);
    if (std::strcmp(name, "vkCmdBindPipeline") == 0) return reinterpret_cast<PFN_vkVoidFunction>(mock_cmd_bind_pipeline);
    if (std::strcmp(name, "vkCmdBindDescriptorSets") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(mock_cmd_bind_descriptor_sets);
    if (std::strcmp(name, "vkCmdPushConstants") == 0) return reinterpret_cast<PFN_vkVoidFunction>(mock_cmd_push_constants);
    if (std::strcmp(name, "vkCmdDispatch") == 0) return reinterpret_cast<PFN_vkVoidFunction>(mock_cmd_dispatch);
    return nullptr;
}

}  // namespace

int main() {
    using namespace leonida;
    const VkDevice dummy_device = make_handle<VkDevice>(1);
    const VkDeviceSize kMaxRange = VkDeviceSize(1) << 30;

    // 1. Undersized first DEVICE_LOCAL heap must be skipped (VUID-vkAllocateMemory-pAllocateInfo-01713)
    //    and allocation must succeed on the larger fitting DEVICE_LOCAL heap (Type 1).
    {
        MockState state;
        g_mock = &state;
        VkPhysicalDeviceMemoryProperties mem{};
        mem.memoryHeapCount = 2;
        mem.memoryHeaps[0] = {1024, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT};
        mem.memoryHeaps[1] = {VkDeviceSize(1) << 30, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT};
        mem.memoryTypeCount = 2;
        mem.memoryTypes[0] = {VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, 0};
        mem.memoryTypes[1] = {VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 1};

        PredicateReducer reducer(mock_get_device_proc_addr, dummy_device, mem, kMaxRange);
        // 128 * 16 = 2048 bytes (> 1024-byte heap 0, fits in 1 GiB heap 1).
        auto batch = reducer.batch(128);
        CHECK(batch != nullptr);
        CHECK(state.alloc_attempts.size() == 1);
        CHECK(state.alloc_attempts[0].first == 1u);
        CHECK(state.update_desc_calls == 1);
        batch.reset();
        CHECK(state.destroy_pool_calls == 1 && state.destroy_buffer_calls == 1 && state.free_memory_calls == 1);
    }

    // 2. OOM on the first DEVICE_LOCAL type falls back to the next compatible memory type.
    {
        MockState state;
        state.fail_oom_mask = 0x1u;
        g_mock = &state;
        VkPhysicalDeviceMemoryProperties mem{};
        mem.memoryHeapCount = 2;
        mem.memoryHeaps[0] = {VkDeviceSize(8) << 30, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT};
        mem.memoryHeaps[1] = {VkDeviceSize(16) << 30, 0};
        mem.memoryTypeCount = 2;
        mem.memoryTypes[0] = {VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0};
        mem.memoryTypes[1] = {VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 1};

        PredicateReducer reducer(mock_get_device_proc_addr, dummy_device, mem, kMaxRange);
        auto batch = reducer.batch(16);
        CHECK(batch != nullptr);
        CHECK(state.alloc_attempts.size() == 2);
        CHECK(state.alloc_attempts[0].first == 0u);
        CHECK(state.alloc_attempts[1].first == 1u);
        batch.reset();
        CHECK(state.destroy_pool_calls == 1 && state.destroy_buffer_calls == 1 && state.free_memory_calls == 1);
    }

    // 3. When every compatible heap is smaller than requirements.size, batch() fails without calling vkAllocateMemory.
    {
        MockState state;
        state.memory_type_bits = 0x1u;
        g_mock = &state;
        VkPhysicalDeviceMemoryProperties mem{};
        mem.memoryHeapCount = 1;
        mem.memoryHeaps[0] = {64, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT};
        mem.memoryTypeCount = 1;
        mem.memoryTypes[0] = {VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0};

        PredicateReducer reducer(mock_get_device_proc_addr, dummy_device, mem, kMaxRange);
        auto batch = reducer.batch(16);  // 256 bytes > 64 bytes
        CHECK(batch == nullptr);
        CHECK(state.alloc_attempts.empty());
        CHECK(state.destroy_buffer_calls == 1 && state.free_memory_calls == 0);
    }

    // 4. When every compatible memory type fails with OOM, batch() returns nullptr and cleans up without freeing null.
    {
        MockState state;
        state.fail_oom_mask = 0x3u;
        g_mock = &state;
        VkPhysicalDeviceMemoryProperties mem{};
        mem.memoryHeapCount = 2;
        mem.memoryHeaps[0] = {VkDeviceSize(1) << 30, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT};
        mem.memoryHeaps[1] = {VkDeviceSize(1) << 30, 0};
        mem.memoryTypeCount = 2;
        mem.memoryTypes[0] = {VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0};
        mem.memoryTypes[1] = {VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, 1};

        PredicateReducer reducer(mock_get_device_proc_addr, dummy_device, mem, kMaxRange);
        auto batch = reducer.batch(16);
        CHECK(batch == nullptr);
        CHECK(state.alloc_attempts.size() == 2);
        CHECK(state.destroy_buffer_calls == 1 && state.free_memory_calls == 0 && state.destroy_pool_calls == 0);
    }

    std::printf("predicate_alloc_test: all checks passed\n");
    return 0;
}
