// Portable CPU-only tests for cached Vulkan device proc pointers, BAR-aware memory fallback,
// immediate failure cleanup, and uncounted count_offset validation across the four Vulkan compute helpers:
//   - runtime/src/vulkan_indirect_dispatch_count.h (IndirectDispatchCounter)
//   - runtime/src/vulkan_predicate.h               (PredicateReducer)
//   - runtime/src/vulkan_predicated_copy.h         (PredicatedBufferCopier)
//   - runtime/src/quad_index_expand.h              (QuadIndexExpander)
// No GPU, Vulkan loader, or game data required.
#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include "../src/quad_index_expand.h"
#include "../src/vulkan_indirect_dispatch_count.h"
#include "../src/vulkan_predicate.h"
#include "../src/vulkan_predicated_copy.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(cond)                                                                                           \
    do {                                                                                                      \
        if (!(cond)) {                                                                                        \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);                              \
            std::exit(1);                                                                                     \
        }                                                                                                     \
    } while (0)

namespace {

template <typename H>
H handle(std::uintptr_t id) {
    return reinterpret_cast<H>(id);
}

struct MockVk {
    std::uintptr_t next_id = 10;
    int gdpa_calls = 0;
    int create_buffer_calls = 0;
    int destroy_buffer_calls = 0;
    int free_memory_calls = 0;
    int create_pool_calls = 0;
    int destroy_pool_calls = 0;
    int dispatch_calls = 0;

    std::uint32_t memory_type_bits = 0x7u;
    VkDeviceSize req_size = 256;
    std::uint32_t oom_type = UINT32_MAX;
    VkResult bind_result = VK_SUCCESS;
    VkResult alloc_sets_result = VK_SUCCESS;
    bool zero_output_va = false;
    std::vector<std::uint32_t> alloc_types;
};

MockVk* g_vk = nullptr;

VKAPI_ATTR VkResult VKAPI_CALL m_create_buffer(VkDevice, const VkBufferCreateInfo*, const VkAllocationCallbacks*,
                                               VkBuffer* out) {
    ++g_vk->create_buffer_calls;
    *out = handle<VkBuffer>(++g_vk->next_id);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL m_destroy_buffer(VkDevice, VkBuffer, const VkAllocationCallbacks*) {
    ++g_vk->destroy_buffer_calls;
}
VKAPI_ATTR void VKAPI_CALL m_get_reqs(VkDevice, VkBuffer, VkMemoryRequirements* r) {
    r->size = g_vk->req_size;
    r->alignment = 256;
    r->memoryTypeBits = g_vk->memory_type_bits;
}
VKAPI_ATTR VkResult VKAPI_CALL m_alloc_mem(VkDevice, const VkMemoryAllocateInfo* info, const VkAllocationCallbacks*,
                                           VkDeviceMemory* out) {
    g_vk->alloc_types.push_back(info->memoryTypeIndex);
    if (info->memoryTypeIndex == g_vk->oom_type) {
        *out = VK_NULL_HANDLE;
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    }
    *out = handle<VkDeviceMemory>(++g_vk->next_id);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL m_free_mem(VkDevice, VkDeviceMemory, const VkAllocationCallbacks*) {
    ++g_vk->free_memory_calls;
}
VKAPI_ATTR VkResult VKAPI_CALL m_bind_buf(VkDevice, VkBuffer, VkDeviceMemory, VkDeviceSize) {
    return g_vk->bind_result;
}
VKAPI_ATTR VkDeviceAddress VKAPI_CALL m_get_addr(VkDevice, const VkBufferDeviceAddressInfo* info) {
    if (g_vk->zero_output_va && reinterpret_cast<std::uintptr_t>(info->buffer) >= 11) return 0;
    return static_cast<VkDeviceAddress>(0x10000000ull + reinterpret_cast<std::uintptr_t>(info->buffer) * 0x1000ull);
}
VKAPI_ATTR VkResult VKAPI_CALL m_create_pool(VkDevice, const VkDescriptorPoolCreateInfo*, const VkAllocationCallbacks*,
                                             VkDescriptorPool* out) {
    ++g_vk->create_pool_calls;
    *out = handle<VkDescriptorPool>(++g_vk->next_id);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL m_destroy_pool(VkDevice, VkDescriptorPool, const VkAllocationCallbacks*) {
    ++g_vk->destroy_pool_calls;
}
VKAPI_ATTR VkResult VKAPI_CALL m_alloc_sets(VkDevice, const VkDescriptorSetAllocateInfo* info, VkDescriptorSet* out) {
    if (g_vk->alloc_sets_result != VK_SUCCESS) return g_vk->alloc_sets_result;
    for (std::uint32_t i = 0; i < info->descriptorSetCount; ++i) out[i] = handle<VkDescriptorSet>(++g_vk->next_id);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL m_update_sets(VkDevice, std::uint32_t, const VkWriteDescriptorSet*, std::uint32_t,
                                         const VkCopyDescriptorSet*) {}
VKAPI_ATTR VkResult VKAPI_CALL m_create_shader(VkDevice, const VkShaderModuleCreateInfo*, const VkAllocationCallbacks*,
                                               VkShaderModule* out) {
    *out = handle<VkShaderModule>(++g_vk->next_id);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL m_destroy_shader(VkDevice, VkShaderModule, const VkAllocationCallbacks*) {}
VKAPI_ATTR VkResult VKAPI_CALL m_create_set_layout(VkDevice, const VkDescriptorSetLayoutCreateInfo*,
                                                   const VkAllocationCallbacks*, VkDescriptorSetLayout* out) {
    *out = handle<VkDescriptorSetLayout>(++g_vk->next_id);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL m_destroy_set_layout(VkDevice, VkDescriptorSetLayout, const VkAllocationCallbacks*) {}
VKAPI_ATTR VkResult VKAPI_CALL m_create_layout(VkDevice, const VkPipelineLayoutCreateInfo*,
                                               const VkAllocationCallbacks*, VkPipelineLayout* out) {
    *out = handle<VkPipelineLayout>(++g_vk->next_id);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL m_destroy_layout(VkDevice, VkPipelineLayout, const VkAllocationCallbacks*) {}
VKAPI_ATTR VkResult VKAPI_CALL m_create_pipelines(VkDevice, VkPipelineCache, std::uint32_t,
                                                  const VkComputePipelineCreateInfo*, const VkAllocationCallbacks*,
                                                  VkPipeline* out) {
    *out = handle<VkPipeline>(++g_vk->next_id);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL m_destroy_pipeline(VkDevice, VkPipeline, const VkAllocationCallbacks*) {}
VKAPI_ATTR void VKAPI_CALL m_cmd_copy(VkCommandBuffer, VkBuffer, VkBuffer, std::uint32_t, const VkBufferCopy*) {}
VKAPI_ATTR void VKAPI_CALL m_cmd_barrier(VkCommandBuffer, VkPipelineStageFlags, VkPipelineStageFlags, VkDependencyFlags,
                                         std::uint32_t, const VkMemoryBarrier*, std::uint32_t,
                                         const VkBufferMemoryBarrier*, std::uint32_t, const VkImageMemoryBarrier*) {}
VKAPI_ATTR void VKAPI_CALL m_cmd_bind_pipe(VkCommandBuffer, VkPipelineBindPoint, VkPipeline) {}
VKAPI_ATTR void VKAPI_CALL m_cmd_bind_sets(VkCommandBuffer, VkPipelineBindPoint, VkPipelineLayout, std::uint32_t,
                                           std::uint32_t, const VkDescriptorSet*, std::uint32_t,
                                           const std::uint32_t*) {}
VKAPI_ATTR void VKAPI_CALL m_cmd_push(VkCommandBuffer, VkPipelineLayout, VkShaderStageFlags, std::uint32_t,
                                      std::uint32_t, const void*) {}
VKAPI_ATTR void VKAPI_CALL m_cmd_dispatch(VkCommandBuffer, std::uint32_t, std::uint32_t, std::uint32_t) {
    ++g_vk->dispatch_calls;
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL m_gdpa(VkDevice, const char* name) {
    ++g_vk->gdpa_calls;
    if (std::strcmp(name, "vkCreateBuffer") == 0) return reinterpret_cast<PFN_vkVoidFunction>(m_create_buffer);
    if (std::strcmp(name, "vkDestroyBuffer") == 0) return reinterpret_cast<PFN_vkVoidFunction>(m_destroy_buffer);
    if (std::strcmp(name, "vkGetBufferMemoryRequirements") == 0) return reinterpret_cast<PFN_vkVoidFunction>(m_get_reqs);
    if (std::strcmp(name, "vkAllocateMemory") == 0) return reinterpret_cast<PFN_vkVoidFunction>(m_alloc_mem);
    if (std::strcmp(name, "vkFreeMemory") == 0) return reinterpret_cast<PFN_vkVoidFunction>(m_free_mem);
    if (std::strcmp(name, "vkBindBufferMemory") == 0) return reinterpret_cast<PFN_vkVoidFunction>(m_bind_buf);
    if (std::strcmp(name, "vkGetBufferDeviceAddress") == 0) return reinterpret_cast<PFN_vkVoidFunction>(m_get_addr);
    if (std::strcmp(name, "vkCreateDescriptorPool") == 0) return reinterpret_cast<PFN_vkVoidFunction>(m_create_pool);
    if (std::strcmp(name, "vkDestroyDescriptorPool") == 0) return reinterpret_cast<PFN_vkVoidFunction>(m_destroy_pool);
    if (std::strcmp(name, "vkAllocateDescriptorSets") == 0) return reinterpret_cast<PFN_vkVoidFunction>(m_alloc_sets);
    if (std::strcmp(name, "vkUpdateDescriptorSets") == 0) return reinterpret_cast<PFN_vkVoidFunction>(m_update_sets);
    if (std::strcmp(name, "vkCreateShaderModule") == 0) return reinterpret_cast<PFN_vkVoidFunction>(m_create_shader);
    if (std::strcmp(name, "vkDestroyShaderModule") == 0) return reinterpret_cast<PFN_vkVoidFunction>(m_destroy_shader);
    if (std::strcmp(name, "vkCreateDescriptorSetLayout") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(m_create_set_layout);
    if (std::strcmp(name, "vkDestroyDescriptorSetLayout") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(m_destroy_set_layout);
    if (std::strcmp(name, "vkCreatePipelineLayout") == 0) return reinterpret_cast<PFN_vkVoidFunction>(m_create_layout);
    if (std::strcmp(name, "vkDestroyPipelineLayout") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(m_destroy_layout);
    if (std::strcmp(name, "vkCreateComputePipelines") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(m_create_pipelines);
    if (std::strcmp(name, "vkDestroyPipeline") == 0) return reinterpret_cast<PFN_vkVoidFunction>(m_destroy_pipeline);
    if (std::strcmp(name, "vkCmdCopyBuffer") == 0) return reinterpret_cast<PFN_vkVoidFunction>(m_cmd_copy);
    if (std::strcmp(name, "vkCmdPipelineBarrier") == 0) return reinterpret_cast<PFN_vkVoidFunction>(m_cmd_barrier);
    if (std::strcmp(name, "vkCmdBindPipeline") == 0) return reinterpret_cast<PFN_vkVoidFunction>(m_cmd_bind_pipe);
    if (std::strcmp(name, "vkCmdBindDescriptorSets") == 0) return reinterpret_cast<PFN_vkVoidFunction>(m_cmd_bind_sets);
    if (std::strcmp(name, "vkCmdPushConstants") == 0) return reinterpret_cast<PFN_vkVoidFunction>(m_cmd_push);
    if (std::strcmp(name, "vkCmdDispatch") == 0) return reinterpret_cast<PFN_vkVoidFunction>(m_cmd_dispatch);
    return nullptr;
}

VkPhysicalDeviceMemoryProperties non_rebar_memory() {
    VkPhysicalDeviceMemoryProperties m{};
    m.memoryHeapCount = 3;
    m.memoryHeaps[0] = {VkDeviceSize(256) << 20, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT};  // 256 MiB BAR
    m.memoryHeaps[1] = {VkDeviceSize(8) << 30, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT};    // 8 GiB VRAM
    m.memoryHeaps[2] = {VkDeviceSize(16) << 30, 0};                                 // 16 GiB system RAM
    m.memoryTypeCount = 3;
    m.memoryTypes[0] = {VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, 0};
    m.memoryTypes[1] = {VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 1};
    m.memoryTypes[2] = {VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 2};
    return m;
}

VkPhysicalDeviceLimits default_limits() {
    VkPhysicalDeviceLimits l{};
    l.maxComputeWorkGroupCount[0] = 65535;
    l.minStorageBufferOffsetAlignment = 16;
    l.maxStorageBufferRange = 1u << 28;
    return l;
}

void test_indirect_dispatch_counter() {
    const VkDevice dev = handle<VkDevice>(1);
    const VkCommandBuffer cb = handle<VkCommandBuffer>(2);
    const VkBuffer args = handle<VkBuffer>(3);
    const VkBuffer cnt = handle<VkBuffer>(4);

    // 1. Uncounted call with non-zero count_offset must be rejected (even with LEONIDA_INDIRECT_GROUP_BUDGET set).
    {
        MockVk vk;
        g_vk = &vk;
        leonida::IndirectDispatchCounter counter(m_gdpa, dev, non_rebar_memory(), default_limits());
        auto batch = counter.batch();
        leonida::CountedDispatch out{};
        CHECK(!batch->prepare(args, 0, VK_NULL_HANDLE, 16, 12, 4, out));
        CHECK(vk.create_buffer_calls == 0);
        CHECK(out.count == 0);
    }

    // 2. Partial allocations are destroyed IMMEDIATELY on bind / BDA failure (before batch.reset()).
    {
        MockVk vk;
        vk.bind_result = VK_ERROR_OUT_OF_DEVICE_MEMORY;
        g_vk = &vk;
        leonida::IndirectDispatchCounter counter(m_gdpa, dev, non_rebar_memory(), default_limits());
        auto batch = counter.batch();
        leonida::CountedDispatch out{};
        CHECK(!batch->prepare(args, 0, cnt, 0, 12, 4, out));
        CHECK(vk.create_buffer_calls == 1 && vk.destroy_buffer_calls == 1 && vk.free_memory_calls == 1);
        CHECK(batch->allocations.empty());

        vk.bind_result = VK_SUCCESS;
        vk.zero_output_va = true;
        CHECK(!batch->prepare(args, 0, cnt, 0, 12, 4, out));
        CHECK(vk.create_buffer_calls == 2 && vk.destroy_buffer_calls == 2 && vk.free_memory_calls == 2);
        CHECK(batch->allocations.empty());
    }

    // 3. Cached device procs: zero vkGetDeviceProcAddr calls after initial make_pipeline().
    {
        MockVk vk;
        g_vk = &vk;
        leonida::IndirectDispatchCounter counter(m_gdpa, dev, non_rebar_memory(), default_limits());
        auto batch = counter.batch();
        leonida::CountedDispatch out{};
        CHECK(batch->prepare(args, 0, cnt, 0, 12, 4, out));
        const int after_init = vk.gdpa_calls;
        CHECK(after_init > 0);
        for (int i = 0; i < 32; ++i) {
            CHECK(batch->prepare(args, 0, cnt, 0, 12, 8, out));
            counter.emit(cb, out);
        }
        batch.reset();
        CHECK(vk.gdpa_calls == after_init);
        CHECK(vk.dispatch_calls == 32);
    }
}

void test_predicate_reducer() {
    const VkDevice dev = handle<VkDevice>(1);
    const VkCommandBuffer cb = handle<VkCommandBuffer>(2);
    const VkBuffer src = handle<VkBuffer>(5);

    // 1. Non-ReBAR: skips 256 MiB BAR Type 0 and picks 8 GiB pure-VRAM Type 1; falls back on OOM.
    {
        MockVk vk;
        g_vk = &vk;
        leonida::PredicateReducer reducer(m_gdpa, dev, non_rebar_memory(), 1u << 20);
        auto b1 = reducer.batch(4);
        CHECK(b1 != nullptr);
        CHECK(vk.alloc_types.size() == 1 && vk.alloc_types[0] == 1u);

        vk.alloc_types.clear();
        vk.oom_type = 1u;  // Pure VRAM OOM -> retries next compatible type cleanly.
        auto b2 = reducer.batch(4);
        CHECK(b2 != nullptr);
        CHECK(vk.alloc_types.size() == 2 && vk.alloc_types[0] == 1u && vk.alloc_types[1] != 1u);

        // Undersized heap filtering: only Type 0 (256 MiB) compatible, request 512 MiB -> rejected without alloc.
        vk.alloc_types.clear();
        vk.oom_type = UINT32_MAX;
        vk.memory_type_bits = 0x1u;
        vk.req_size = VkDeviceSize(512) << 20;
        CHECK(reducer.batch(4) == nullptr);
        CHECK(vk.alloc_types.empty());
    }

    // 2. Cached device procs: zero vkGetDeviceProcAddr calls after make_pipeline().
    {
        MockVk vk;
        g_vk = &vk;
        leonida::PredicateReducer reducer(m_gdpa, dev, non_rebar_memory(), 1u << 20);
        auto b = reducer.batch(64);
        CHECK(b != nullptr);
        const int after_init = vk.gdpa_calls;
        CHECK(after_init > 0);
        for (int i = 0; i < 32; ++i) {
            leonida::PredicateCapture cap{};
            CHECK(b->prepare(src, 0, cap));
            reducer.capture(cb, cap);
        }
        b.reset();
        CHECK(vk.gdpa_calls == after_init);
        CHECK(vk.dispatch_calls == 32);
    }
}

void test_predicated_buffer_copier() {
    const VkDevice dev = handle<VkDevice>(1);
    const VkCommandBuffer cb = handle<VkCommandBuffer>(2);
    const VkBuffer src = handle<VkBuffer>(5);
    const VkBuffer dst = handle<VkBuffer>(6);

    // 1. Failed vkAllocateDescriptorSets destroys the newly created pool immediately (not retained in batch).
    {
        MockVk vk;
        vk.alloc_sets_result = VK_ERROR_OUT_OF_POOL_MEMORY;
        g_vk = &vk;
        leonida::PredicatedBufferCopier copier(m_gdpa, dev, default_limits());
        auto batch = copier.batch();
        std::vector<leonida::PredicatedCopyDispatch> jobs;
        CHECK(!batch->prepare(src, dst, 0, 0, 128, jobs));
        CHECK(vk.create_pool_calls == 1 && vk.destroy_pool_calls == 1);
        CHECK(batch->pools.empty());
    }

    // 2. Cached device procs: zero vkGetDeviceProcAddr calls after make_pipeline().
    {
        MockVk vk;
        g_vk = &vk;
        leonida::PredicatedBufferCopier copier(m_gdpa, dev, default_limits());
        auto batch = copier.batch();
        std::vector<leonida::PredicatedCopyDispatch> jobs;
        CHECK(batch->prepare(src, dst, 0, 0, 128, jobs));
        const int after_init = vk.gdpa_calls;
        CHECK(after_init > 0);
        for (int i = 0; i < 32; ++i) {
            CHECK(batch->prepare(src, dst, 0, 0, 256, jobs));
            copier.emit(cb, jobs);
        }
        batch.reset();
        CHECK(vk.gdpa_calls == after_init);
        CHECK(vk.dispatch_calls == 32);
    }
}

void test_quad_index_expander() {
    const VkDevice dev = handle<VkDevice>(1);
    const VkCommandBuffer cb = handle<VkCommandBuffer>(2);
    const VkBuffer idx = handle<VkBuffer>(5);
    const VkBuffer scratch = handle<VkBuffer>(6);

    MockVk vk;
    g_vk = &vk;
    leonida::QuadIndexExpander expander(m_gdpa, dev, default_limits());
    leonida::QuadIndexExpansion job{};
    CHECK(expander.prepare(idx, 0, VK_INDEX_TYPE_UINT16, 16, 4, scratch, 0, job));
    const int after_init = vk.gdpa_calls;
    CHECK(after_init > 0);
    for (int i = 0; i < 32; ++i) {
        CHECK(expander.prepare(idx, 4, VK_INDEX_TYPE_UINT32, 32, 8, scratch, 64, job));
        expander.emit(cb, job);
    }
    CHECK(vk.gdpa_calls == after_init);
    CHECK(vk.dispatch_calls == 32);
}

}  // namespace

int main() {
    test_indirect_dispatch_counter();
    test_predicate_reducer();
    test_predicated_buffer_copier();
    test_quad_index_expander();
    std::printf("vk_helper_proc_cache_test: PASS\n");
    return 0;
}
