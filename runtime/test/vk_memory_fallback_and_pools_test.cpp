// Unit tests for BAR-aware memory fallback in PresentComposer::ensure_target,
// OcclusionResolve::initialize, and VulkanQueue::acquire_stage, plus bounded
// pool caps and best-fit reuse in VulkanQueue's KHR stage and postbuild query pools.
// Uses mock Vulkan device procedures: no Vulkan loader, GPU, or game data needed.

#include "../src/vulkan_backend.h"
#include "../src/vulkan_present_compose.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <set>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

int g_failures = 0;

void expect(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_failures;
    }
}

template<typename Handle>
Handle make_handle(std::uintptr_t value) {
    if constexpr (std::is_pointer_v<Handle>)
        return reinterpret_cast<Handle>(value);
    else
        return static_cast<Handle>(value);
}

struct AllocAttempt {
    std::uint32_t type = 0;
    VkDeviceSize size = 0;
    VkResult result = VK_SUCCESS;
};

struct MockDriver {
    std::uintptr_t next_id = 1;
    std::uint32_t memory_type_bits = 0xFFFFFFFFu;
    VkDeviceSize forced_image_req_size = 0;
    VkDeviceSize forced_buffer_req_size = 0;
    std::set<std::uint32_t> fail_memory_types;
    bool write_garbage_on_alloc_fail = false;
    bool fail_bind_image = false;
    bool fail_bind_buffer = false;

    std::vector<AllocAttempt> alloc_attempts;
    std::set<VkBuffer> live_buffers;
    std::unordered_map<VkBuffer, VkDeviceSize> buffer_sizes;
    std::set<VkDeviceMemory> live_memories;
    std::set<VkImage> live_images;
    std::set<VkImageView> live_views;
    std::set<VkQueryPool> live_query_pools;
    std::unordered_map<VkQueryPool, std::uint32_t> query_pool_caps;
    std::set<VkCommandPool> live_command_pools;
    std::set<VkSemaphore> live_semaphores;
    std::set<VkDescriptorSetLayout> live_set_layouts;
    std::set<VkDescriptorPool> live_desc_pools;
    std::set<VkPipelineLayout> live_pipe_layouts;
    std::set<VkShaderModule> live_modules;
    std::set<VkPipeline> live_pipelines;

    int image_creates = 0;
    int buffer_creates = 0;
    int buffer_destroys = 0;
    int memory_allocs = 0;
    int memory_frees = 0;
    int query_pool_creates = 0;
    int query_pool_destroys = 0;
    int bad_frees = 0;
    int mapped_count = 0;
    int as_builds = 0;
    int write_props_calls = 0;
    VkQueryPool last_postbuild_pool = VK_NULL_HANDLE;
    VkDeviceAddress last_staged_bda = 0;
    std::vector<VkDeviceSize> xdxr_staged_sizes;
    std::vector<std::uint8_t> map_backing = std::vector<std::uint8_t>(1 << 16, 0);

    void reset_counters() {
        alloc_attempts.clear();
        image_creates = 0;
        buffer_creates = 0;
        buffer_destroys = 0;
        memory_allocs = 0;
        memory_frees = 0;
        query_pool_creates = 0;
        query_pool_destroys = 0;
        as_builds = 0;
        write_props_calls = 0;
        last_postbuild_pool = VK_NULL_HANDLE;
        last_staged_bda = 0;
        xdxr_staged_sizes.clear();
    }

    bool all_clean() const {
        return live_buffers.empty() && live_memories.empty() && live_images.empty() && live_views.empty() &&
               live_query_pools.empty() && live_command_pools.empty() && live_semaphores.empty() &&
               live_set_layouts.empty() && live_desc_pools.empty() && live_pipe_layouts.empty() &&
               live_modules.empty() && live_pipelines.empty() && bad_frees == 0 && mapped_count == 0;
    }
};

MockDriver* g_drv = nullptr;

VKAPI_ATTR VkResult VKAPI_CALL m_create_cmd_pool(VkDevice, const VkCommandPoolCreateInfo*, const VkAllocationCallbacks*,
                                                 VkCommandPool* out) {
    *out = make_handle<VkCommandPool>(g_drv->next_id++);
    g_drv->live_command_pools.insert(*out);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL m_destroy_cmd_pool(VkDevice, VkCommandPool p, const VkAllocationCallbacks*) {
    if (!g_drv->live_command_pools.erase(p)) ++g_drv->bad_frees;
}
VKAPI_ATTR VkResult VKAPI_CALL m_alloc_cmd_bufs(VkDevice, const VkCommandBufferAllocateInfo* info,
                                                VkCommandBuffer* out) {
    for (std::uint32_t i = 0; i < info->commandBufferCount; ++i)
        out[i] = make_handle<VkCommandBuffer>(g_drv->next_id++);
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL m_reset_cmd_buf(VkCommandBuffer, VkCommandBufferResetFlags) { return VK_SUCCESS; }
VKAPI_ATTR VkResult VKAPI_CALL m_begin_cmd_buf(VkCommandBuffer, const VkCommandBufferBeginInfo*) { return VK_SUCCESS; }
VKAPI_ATTR VkResult VKAPI_CALL m_end_cmd_buf(VkCommandBuffer) { return VK_SUCCESS; }
VKAPI_ATTR VkResult VKAPI_CALL m_queue_submit(VkQueue, std::uint32_t, const VkSubmitInfo*, VkFence) {
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL m_queue_wait_idle(VkQueue) { return VK_SUCCESS; }
VKAPI_ATTR VkResult VKAPI_CALL m_device_wait_idle(VkDevice) { return VK_SUCCESS; }
VKAPI_ATTR VkResult VKAPI_CALL m_create_semaphore(VkDevice, const VkSemaphoreCreateInfo*, const VkAllocationCallbacks*,
                                                  VkSemaphore* out) {
    *out = make_handle<VkSemaphore>(g_drv->next_id++);
    g_drv->live_semaphores.insert(*out);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL m_destroy_semaphore(VkDevice, VkSemaphore s, const VkAllocationCallbacks*) {
    if (!g_drv->live_semaphores.erase(s)) ++g_drv->bad_frees;
}
VKAPI_ATTR VkResult VKAPI_CALL m_sem_counter(VkDevice, VkSemaphore, std::uint64_t* out) {
    *out = UINT64_MAX;
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL m_wait_sems(VkDevice, const VkSemaphoreWaitInfo*, std::uint64_t) { return VK_SUCCESS; }
VKAPI_ATTR VkResult VKAPI_CALL m_signal_sem(VkDevice, const VkSemaphoreSignalInfo*) { return VK_SUCCESS; }
VKAPI_ATTR VkResult VKAPI_CALL m_create_image(VkDevice, const VkImageCreateInfo* info, const VkAllocationCallbacks*,
                                              VkImage* out) {
    ++g_drv->image_creates;
    if (info->extent.width == 0 || info->extent.height == 0) return VK_ERROR_UNKNOWN;
    *out = make_handle<VkImage>(g_drv->next_id++);
    g_drv->live_images.insert(*out);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL m_destroy_image(VkDevice, VkImage img, const VkAllocationCallbacks*) {
    if (!g_drv->live_images.erase(img)) ++g_drv->bad_frees;
}
VKAPI_ATTR void VKAPI_CALL m_get_image_reqs(VkDevice, VkImage, VkMemoryRequirements* req) {
    req->size = g_drv->forced_image_req_size ? g_drv->forced_image_req_size : (VkDeviceSize{16} << 20);
    req->alignment = 4096;
    req->memoryTypeBits = g_drv->memory_type_bits;
}
VKAPI_ATTR VkResult VKAPI_CALL m_bind_image_mem(VkDevice, VkImage, VkDeviceMemory, VkDeviceSize) {
    return g_drv->fail_bind_image ? VK_ERROR_OUT_OF_DEVICE_MEMORY : VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL m_create_image_view(VkDevice, const VkImageViewCreateInfo*, const VkAllocationCallbacks*,
                                                   VkImageView* out) {
    *out = make_handle<VkImageView>(g_drv->next_id++);
    g_drv->live_views.insert(*out);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL m_destroy_image_view(VkDevice, VkImageView v, const VkAllocationCallbacks*) {
    if (!g_drv->live_views.erase(v)) ++g_drv->bad_frees;
}
VKAPI_ATTR VkResult VKAPI_CALL m_create_buffer(VkDevice, const VkBufferCreateInfo* info, const VkAllocationCallbacks*,
                                               VkBuffer* out) {
    ++g_drv->buffer_creates;
    *out = make_handle<VkBuffer>(g_drv->next_id++);
    g_drv->live_buffers.insert(*out);
    g_drv->buffer_sizes[*out] = info->size;
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL m_destroy_buffer(VkDevice, VkBuffer b, const VkAllocationCallbacks*) {
    ++g_drv->buffer_destroys;
    if (!g_drv->live_buffers.erase(b)) ++g_drv->bad_frees;
    g_drv->buffer_sizes.erase(b);
}
VKAPI_ATTR void VKAPI_CALL m_get_buffer_reqs(VkDevice, VkBuffer b, VkMemoryRequirements* req) {
    const auto it = g_drv->buffer_sizes.find(b);
    const VkDeviceSize base = it != g_drv->buffer_sizes.end() ? it->second : 256;
    req->size = g_drv->forced_buffer_req_size ? g_drv->forced_buffer_req_size : std::max<VkDeviceSize>(base, 256);
    req->alignment = 256;
    req->memoryTypeBits = g_drv->memory_type_bits;
}
VKAPI_ATTR VkResult VKAPI_CALL m_allocate_memory(VkDevice, const VkMemoryAllocateInfo* info,
                                                 const VkAllocationCallbacks*, VkDeviceMemory* out) {
    const bool fail = g_drv->fail_memory_types.count(info->memoryTypeIndex) != 0;
    const VkResult res = fail ? VK_ERROR_OUT_OF_DEVICE_MEMORY : VK_SUCCESS;
    g_drv->alloc_attempts.push_back({info->memoryTypeIndex, info->allocationSize, res});
    if (fail) {
        if (g_drv->write_garbage_on_alloc_fail) *out = make_handle<VkDeviceMemory>(0xdeadbeefull);
        return res;
    }
    ++g_drv->memory_allocs;
    *out = make_handle<VkDeviceMemory>(g_drv->next_id++);
    g_drv->live_memories.insert(*out);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL m_free_memory(VkDevice, VkDeviceMemory m, const VkAllocationCallbacks*) {
    ++g_drv->memory_frees;
    if (!g_drv->live_memories.erase(m)) ++g_drv->bad_frees;
}
VKAPI_ATTR VkResult VKAPI_CALL m_bind_buffer_mem(VkDevice, VkBuffer, VkDeviceMemory, VkDeviceSize) {
    return g_drv->fail_bind_buffer ? VK_ERROR_OUT_OF_DEVICE_MEMORY : VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL m_map_memory(VkDevice, VkDeviceMemory, VkDeviceSize offset, VkDeviceSize size,
                                            VkMemoryMapFlags, void** data) {
    const std::size_t need = static_cast<std::size_t>(offset + (size == VK_WHOLE_SIZE ? 64 : size));
    if (g_drv->map_backing.size() < need) g_drv->map_backing.resize(need, 0);
    ++g_drv->mapped_count;
    *data = g_drv->map_backing.data() + offset;
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL m_unmap_memory(VkDevice, VkDeviceMemory) { --g_drv->mapped_count; }
VKAPI_ATTR VkDeviceAddress VKAPI_CALL m_get_bda(VkDevice, const VkBufferDeviceAddressInfo* info) {
    const auto raw = reinterpret_cast<std::uintptr_t>(info->buffer);
    const VkDeviceAddress addr = 0x10000000ull + static_cast<VkDeviceAddress>(raw) * 0x1000ull;
    g_drv->last_staged_bda = addr;
    return addr;
}
VKAPI_ATTR VkResult VKAPI_CALL m_create_set_layout(VkDevice, const VkDescriptorSetLayoutCreateInfo*,
                                                   const VkAllocationCallbacks*, VkDescriptorSetLayout* out) {
    *out = make_handle<VkDescriptorSetLayout>(g_drv->next_id++);
    g_drv->live_set_layouts.insert(*out);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL m_destroy_set_layout(VkDevice, VkDescriptorSetLayout l, const VkAllocationCallbacks*) {
    if (!g_drv->live_set_layouts.erase(l)) ++g_drv->bad_frees;
}
VKAPI_ATTR VkResult VKAPI_CALL m_create_desc_pool(VkDevice, const VkDescriptorPoolCreateInfo*,
                                                  const VkAllocationCallbacks*, VkDescriptorPool* out) {
    *out = make_handle<VkDescriptorPool>(g_drv->next_id++);
    g_drv->live_desc_pools.insert(*out);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL m_destroy_desc_pool(VkDevice, VkDescriptorPool p, const VkAllocationCallbacks*) {
    if (!g_drv->live_desc_pools.erase(p)) ++g_drv->bad_frees;
}
VKAPI_ATTR VkResult VKAPI_CALL m_alloc_desc_sets(VkDevice, const VkDescriptorSetAllocateInfo* info,
                                                 VkDescriptorSet* out) {
    for (std::uint32_t i = 0; i < info->descriptorSetCount; ++i)
        out[i] = make_handle<VkDescriptorSet>(g_drv->next_id++);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL m_update_desc_sets(VkDevice, std::uint32_t, const VkWriteDescriptorSet*, std::uint32_t,
                                              const VkCopyDescriptorSet*) {}
VKAPI_ATTR VkResult VKAPI_CALL m_create_pipe_layout(VkDevice, const VkPipelineLayoutCreateInfo*,
                                                    const VkAllocationCallbacks*, VkPipelineLayout* out) {
    *out = make_handle<VkPipelineLayout>(g_drv->next_id++);
    g_drv->live_pipe_layouts.insert(*out);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL m_destroy_pipe_layout(VkDevice, VkPipelineLayout l, const VkAllocationCallbacks*) {
    if (!g_drv->live_pipe_layouts.erase(l)) ++g_drv->bad_frees;
}
VKAPI_ATTR VkResult VKAPI_CALL m_create_shader_mod(VkDevice, const VkShaderModuleCreateInfo*,
                                                   const VkAllocationCallbacks*, VkShaderModule* out) {
    *out = make_handle<VkShaderModule>(g_drv->next_id++);
    g_drv->live_modules.insert(*out);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL m_destroy_shader_mod(VkDevice, VkShaderModule m, const VkAllocationCallbacks*) {
    if (!g_drv->live_modules.erase(m)) ++g_drv->bad_frees;
}
VKAPI_ATTR VkResult VKAPI_CALL m_create_compute_pipes(VkDevice, VkPipelineCache, std::uint32_t count,
                                                      const VkComputePipelineCreateInfo*, const VkAllocationCallbacks*,
                                                      VkPipeline* out) {
    for (std::uint32_t i = 0; i < count; ++i) {
        out[i] = make_handle<VkPipeline>(g_drv->next_id++);
        g_drv->live_pipelines.insert(out[i]);
    }
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL m_create_gfx_pipes(VkDevice, VkPipelineCache, std::uint32_t count,
                                                  const VkGraphicsPipelineCreateInfo*, const VkAllocationCallbacks*,
                                                  VkPipeline* out) {
    for (std::uint32_t i = 0; i < count; ++i) {
        out[i] = make_handle<VkPipeline>(g_drv->next_id++);
        g_drv->live_pipelines.insert(out[i]);
    }
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL m_destroy_pipe(VkDevice, VkPipeline p, const VkAllocationCallbacks*) {
    if (!g_drv->live_pipelines.erase(p)) ++g_drv->bad_frees;
}
VKAPI_ATTR VkResult VKAPI_CALL m_create_query_pool(VkDevice, const VkQueryPoolCreateInfo* info,
                                                   const VkAllocationCallbacks*, VkQueryPool* out) {
    ++g_drv->query_pool_creates;
    *out = make_handle<VkQueryPool>(g_drv->next_id++);
    g_drv->live_query_pools.insert(*out);
    g_drv->query_pool_caps[*out] = info->queryCount;
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL m_destroy_query_pool(VkDevice, VkQueryPool p, const VkAllocationCallbacks*) {
    ++g_drv->query_pool_destroys;
    if (!g_drv->live_query_pools.erase(p)) ++g_drv->bad_frees;
    g_drv->query_pool_caps.erase(p);
}
VKAPI_ATTR void VKAPI_CALL m_cmd_reset_query_pool(VkCommandBuffer, VkQueryPool, std::uint32_t, std::uint32_t) {}
VKAPI_ATTR void VKAPI_CALL m_cmd_copy_query_results(VkCommandBuffer, VkQueryPool, std::uint32_t, std::uint32_t,
                                                    VkBuffer, VkDeviceSize, VkDeviceSize, VkQueryResultFlags) {}
VKAPI_ATTR void VKAPI_CALL m_cmd_write_as_props(VkCommandBuffer, std::uint32_t, const VkAccelerationStructureKHR*,
                                                VkQueryType, VkQueryPool pool, std::uint32_t) {
    ++g_drv->write_props_calls;
    g_drv->last_postbuild_pool = pool;
}
VKAPI_ATTR void VKAPI_CALL m_cmd_build_as(VkCommandBuffer, std::uint32_t,
                                          const VkAccelerationStructureBuildGeometryInfoKHR*,
                                          const VkAccelerationStructureBuildRangeInfoKHR* const*) {
    ++g_drv->as_builds;
}
VKAPI_ATTR void VKAPI_CALL m_cmd_barrier(VkCommandBuffer, VkPipelineStageFlags, VkPipelineStageFlags,
                                         VkDependencyFlags, std::uint32_t, const VkMemoryBarrier*, std::uint32_t,
                                         const VkBufferMemoryBarrier*, std::uint32_t, const VkImageMemoryBarrier*) {}
VKAPI_ATTR void VKAPI_CALL m_cmd_clear_color(VkCommandBuffer, VkImage, VkImageLayout, const VkClearColorValue*,
                                             std::uint32_t, const VkImageSubresourceRange*) {}
VKAPI_ATTR void VKAPI_CALL m_cmd_clear_ds(VkCommandBuffer, VkImage, VkImageLayout, const VkClearDepthStencilValue*,
                                          std::uint32_t, const VkImageSubresourceRange*) {}
VKAPI_ATTR void VKAPI_CALL m_cmd_clear_att(VkCommandBuffer, std::uint32_t, const VkClearAttachment*, std::uint32_t,
                                           const VkClearRect*) {}
VKAPI_ATTR void VKAPI_CALL m_cmd_copy_buf(VkCommandBuffer, VkBuffer, VkBuffer, std::uint32_t, const VkBufferCopy*) {}
VKAPI_ATTR void VKAPI_CALL m_cmd_copy_b2i(VkCommandBuffer, VkBuffer, VkImage, VkImageLayout, std::uint32_t,
                                          const VkBufferImageCopy*) {}
VKAPI_ATTR void VKAPI_CALL m_cmd_copy_i2b(VkCommandBuffer, VkImage, VkImageLayout, VkBuffer, std::uint32_t,
                                          const VkBufferImageCopy*) {}
VKAPI_ATTR void VKAPI_CALL m_cmd_copy_img(VkCommandBuffer, VkImage, VkImageLayout, VkImage, VkImageLayout,
                                          std::uint32_t, const VkImageCopy*) {}
VKAPI_ATTR void VKAPI_CALL m_cmd_resolve(VkCommandBuffer, VkImage, VkImageLayout, VkImage, VkImageLayout,
                                         std::uint32_t, const VkImageResolve*) {}
VKAPI_ATTR void VKAPI_CALL m_cmd_fill(VkCommandBuffer, VkBuffer, VkDeviceSize, VkDeviceSize, std::uint32_t) {}
VKAPI_ATTR void VKAPI_CALL m_cmd_update(VkCommandBuffer, VkBuffer, VkDeviceSize, VkDeviceSize, const void*) {}
VKAPI_ATTR void VKAPI_CALL m_cmd_bind_pipe(VkCommandBuffer, VkPipelineBindPoint, VkPipeline) {}
VKAPI_ATTR void VKAPI_CALL m_cmd_bind_sets(VkCommandBuffer, VkPipelineBindPoint, VkPipelineLayout, std::uint32_t,
                                           std::uint32_t, const VkDescriptorSet*, std::uint32_t,
                                           const std::uint32_t*) {}
VKAPI_ATTR void VKAPI_CALL m_cmd_push(VkCommandBuffer, VkPipelineLayout, VkShaderStageFlags, std::uint32_t,
                                      std::uint32_t, const void*) {}
VKAPI_ATTR void VKAPI_CALL m_cmd_dispatch(VkCommandBuffer, std::uint32_t, std::uint32_t, std::uint32_t) {}
VKAPI_ATTR void VKAPI_CALL m_cmd_bind_ib(VkCommandBuffer, VkBuffer, VkDeviceSize, VkIndexType) {}
VKAPI_ATTR void VKAPI_CALL m_cmd_set_vp(VkCommandBuffer, std::uint32_t, std::uint32_t, const VkViewport*) {}
VKAPI_ATTR void VKAPI_CALL m_cmd_set_sc(VkCommandBuffer, std::uint32_t, std::uint32_t, const VkRect2D*) {}
VKAPI_ATTR void VKAPI_CALL m_cmd_set_st(VkCommandBuffer, VkStencilFaceFlags, std::uint32_t) {}
VKAPI_ATTR void VKAPI_CALL m_cmd_set_bc(VkCommandBuffer, const float[4]) {}
VKAPI_ATTR void VKAPI_CALL m_cmd_draw(VkCommandBuffer, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t) {}
VKAPI_ATTR void VKAPI_CALL m_cmd_draw_idx(VkCommandBuffer, std::uint32_t, std::uint32_t, std::uint32_t, std::int32_t,
                                          std::uint32_t) {}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL mock_get_proc(VkDevice, const char* name) {
#define PROC(vk_name, impl) \
    if (std::strcmp(name, #vk_name) == 0) return reinterpret_cast<PFN_vkVoidFunction>(impl)
    PROC(vkCreateCommandPool, m_create_cmd_pool);
    PROC(vkDestroyCommandPool, m_destroy_cmd_pool);
    PROC(vkAllocateCommandBuffers, m_alloc_cmd_bufs);
    PROC(vkResetCommandBuffer, m_reset_cmd_buf);
    PROC(vkBeginCommandBuffer, m_begin_cmd_buf);
    PROC(vkEndCommandBuffer, m_end_cmd_buf);
    PROC(vkQueueSubmit, m_queue_submit);
    PROC(vkQueueWaitIdle, m_queue_wait_idle);
    PROC(vkDeviceWaitIdle, m_device_wait_idle);
    PROC(vkCreateSemaphore, m_create_semaphore);
    PROC(vkDestroySemaphore, m_destroy_semaphore);
    PROC(vkGetSemaphoreCounterValue, m_sem_counter);
    PROC(vkWaitSemaphores, m_wait_sems);
    PROC(vkSignalSemaphore, m_signal_sem);
    PROC(vkCreateImage, m_create_image);
    PROC(vkDestroyImage, m_destroy_image);
    PROC(vkGetImageMemoryRequirements, m_get_image_reqs);
    PROC(vkBindImageMemory, m_bind_image_mem);
    PROC(vkCreateImageView, m_create_image_view);
    PROC(vkDestroyImageView, m_destroy_image_view);
    PROC(vkCreateBuffer, m_create_buffer);
    PROC(vkDestroyBuffer, m_destroy_buffer);
    PROC(vkGetBufferMemoryRequirements, m_get_buffer_reqs);
    PROC(vkAllocateMemory, m_allocate_memory);
    PROC(vkFreeMemory, m_free_memory);
    PROC(vkBindBufferMemory, m_bind_buffer_mem);
    PROC(vkMapMemory, m_map_memory);
    PROC(vkUnmapMemory, m_unmap_memory);
    PROC(vkGetBufferDeviceAddress, m_get_bda);
    PROC(vkCreateDescriptorSetLayout, m_create_set_layout);
    PROC(vkDestroyDescriptorSetLayout, m_destroy_set_layout);
    PROC(vkCreateDescriptorPool, m_create_desc_pool);
    PROC(vkDestroyDescriptorPool, m_destroy_desc_pool);
    PROC(vkAllocateDescriptorSets, m_alloc_desc_sets);
    PROC(vkUpdateDescriptorSets, m_update_desc_sets);
    PROC(vkCreatePipelineLayout, m_create_pipe_layout);
    PROC(vkDestroyPipelineLayout, m_destroy_pipe_layout);
    PROC(vkCreateShaderModule, m_create_shader_mod);
    PROC(vkDestroyShaderModule, m_destroy_shader_mod);
    PROC(vkCreateComputePipelines, m_create_compute_pipes);
    PROC(vkCreateGraphicsPipelines, m_create_gfx_pipes);
    PROC(vkDestroyPipeline, m_destroy_pipe);
    PROC(vkCreateQueryPool, m_create_query_pool);
    PROC(vkDestroyQueryPool, m_destroy_query_pool);
    PROC(vkCmdResetQueryPool, m_cmd_reset_query_pool);
    PROC(vkCmdCopyQueryPoolResults, m_cmd_copy_query_results);
    PROC(vkCmdWriteAccelerationStructuresPropertiesKHR, m_cmd_write_as_props);
    PROC(vkCmdBuildAccelerationStructuresKHR, m_cmd_build_as);
    PROC(vkCmdPipelineBarrier, m_cmd_barrier);
    PROC(vkCmdClearColorImage, m_cmd_clear_color);
    PROC(vkCmdClearDepthStencilImage, m_cmd_clear_ds);
    PROC(vkCmdClearAttachments, m_cmd_clear_att);
    PROC(vkCmdCopyBuffer, m_cmd_copy_buf);
    PROC(vkCmdCopyBufferToImage, m_cmd_copy_b2i);
    PROC(vkCmdCopyImageToBuffer, m_cmd_copy_i2b);
    PROC(vkCmdCopyImage, m_cmd_copy_img);
    PROC(vkCmdResolveImage, m_cmd_resolve);
    PROC(vkCmdFillBuffer, m_cmd_fill);
    PROC(vkCmdUpdateBuffer, m_cmd_update);
    PROC(vkCmdBindPipeline, m_cmd_bind_pipe);
    PROC(vkCmdBindDescriptorSets, m_cmd_bind_sets);
    PROC(vkCmdPushConstants, m_cmd_push);
    PROC(vkCmdDispatch, m_cmd_dispatch);
    PROC(vkCmdBindIndexBuffer, m_cmd_bind_ib);
    PROC(vkCmdSetViewport, m_cmd_set_vp);
    PROC(vkCmdSetScissor, m_cmd_set_sc);
    PROC(vkCmdSetStencilReference, m_cmd_set_st);
    PROC(vkCmdSetBlendConstants, m_cmd_set_bc);
    PROC(vkCmdDraw, m_cmd_draw);
    PROC(vkCmdDrawIndexed, m_cmd_draw_idx);
#undef PROC
    return nullptr;
}

// Non-ReBAR GPU with 256 MiB BAR listed BEFORE main 8 GiB VRAM and 32 GiB system RAM.
VkPhysicalDeviceMemoryProperties make_non_rebar_props() {
    VkPhysicalDeviceMemoryProperties m{};
    m.memoryHeapCount = 3;
    m.memoryHeaps[0] = {VkDeviceSize{256} << 20, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT};   // 256 MiB BAR
    m.memoryHeaps[1] = {VkDeviceSize{8} << 30, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT};     // 8 GiB VRAM
    m.memoryHeaps[2] = {VkDeviceSize{32} << 30, 0};                                  // 32 GiB system RAM
    m.memoryTypeCount = 3;
    m.memoryTypes[0] = {VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                        0};
    m.memoryTypes[1] = {VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 1};
    m.memoryTypes[2] = {VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                            VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
                        2};
    return m;
}

// ReBAR GPU: 16 GiB CPU-visible VRAM + 32 GiB system RAM.
VkPhysicalDeviceMemoryProperties make_rebar_props() {
    VkPhysicalDeviceMemoryProperties m{};
    m.memoryHeapCount = 2;
    m.memoryHeaps[0] = {VkDeviceSize{16} << 30, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT};
    m.memoryHeaps[1] = {VkDeviceSize{32} << 30, 0};
    m.memoryTypeCount = 2;
    m.memoryTypes[0] = {VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                        0};
    m.memoryTypes[1] = {VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 1};
    return m;
}

void test_present_composer_memory_and_extent() {
    MockDriver drv;
    g_drv = &drv;
    {
        leonida::VulkanContext ctx(mock_get_proc, make_handle<VkDevice>(1), false);
        expect(ctx.loaded(), "context loaded for PresentComposer test");

        // 1. Non-ReBAR: pure VRAM (type 1) preferred over 256 MiB BAR (type 0), and zero-extent rejected.
        {
            const auto mem = make_non_rebar_props();
            leonida::PresentComposer composer(ctx, mem);
            expect(composer.ready(), "PresentComposer ready");

            bool changed = true;
            expect(!composer.ensure_target({0, 1080}, changed) && !changed && drv.image_creates == 0,
                   "zero width rejected before vkCreateImage");
            expect(!composer.ensure_target({1920, 0}, changed) && !changed && drv.image_creates == 0,
                   "zero height rejected before vkCreateImage");

            expect(composer.ensure_target({1920, 1080}, changed) && changed,
                   "PresentComposer allocates 1080p target on non-ReBAR GPU");
            expect(drv.alloc_attempts.size() == 1 && drv.alloc_attempts[0].type == 1,
                   "PresentComposer picks pure VRAM (type 1) before 256 MiB BAR (type 0)");
        }

        // 2. OOM cascade on non-ReBAR: type 1 (VRAM) fails, type 0 (BAR) fails -> falls back to type 2 (RAM).
        //    Also write garbage handle on failed vkAllocateMemory to prove failed handles are cleared before retry.
        {
            drv.reset_counters();
            drv.fail_memory_types = {1, 0};
            drv.write_garbage_on_alloc_fail = true;
            const auto mem = make_non_rebar_props();
            leonida::PresentComposer composer(ctx, mem);
            expect(composer.ready(), "PresentComposer ready for OOM fallback");
            bool changed = false;
            expect(composer.ensure_target({1280, 720}, changed) && changed,
                   "PresentComposer falls back to host RAM when VRAM and BAR OOM");
            expect(drv.alloc_attempts.size() == 3 && drv.alloc_attempts[0].type == 1 &&
                       drv.alloc_attempts[1].type == 0 && drv.alloc_attempts[2].type == 2,
                   "PresentComposer tries VRAM (1) -> BAR (0) -> RAM (2) in order");
            drv.fail_memory_types.clear();
            drv.write_garbage_on_alloc_fail = false;
        }

        // 3. Heap-size filter (VUID-vkAllocateMemory-pAllocateInfo-01713): skip undersized VRAM heap.
        {
            drv.reset_counters();
            drv.forced_image_req_size = VkDeviceSize{32} << 20;  // 32 MiB
            VkPhysicalDeviceMemoryProperties mem = make_non_rebar_props();
            mem.memoryHeaps[0].size = VkDeviceSize{16} << 20;  // too small
            mem.memoryHeaps[1].size = VkDeviceSize{16} << 20;  // too small
            leonida::PresentComposer composer(ctx, mem);
            expect(composer.ready(), "PresentComposer ready for undersized heap test");
            bool changed = false;
            expect(composer.ensure_target({2560, 1440}, changed) && changed,
                   "PresentComposer skips undersized heaps and succeeds on heap 2");
            expect(drv.alloc_attempts.size() == 1 && drv.alloc_attempts[0].type == 2,
                   "PresentComposer never calls vkAllocateMemory on heaps smaller than requirements.size");
            drv.forced_image_req_size = 0;
        }

        // 4. Software/host-only GPU with zero DEVICE_LOCAL types still succeeds.
        {
            drv.reset_counters();
            VkPhysicalDeviceMemoryProperties mem{};
            mem.memoryHeapCount = 1;
            mem.memoryHeaps[0] = {VkDeviceSize{4} << 30, 0};
            mem.memoryTypeCount = 1;
            mem.memoryTypes[0] = {VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0};
            leonida::PresentComposer composer(ctx, mem);
            expect(composer.ready(), "PresentComposer ready on host-only GPU");
            bool changed = false;
            expect(composer.ensure_target({640, 480}, changed) && changed,
                   "PresentComposer succeeds without DEVICE_LOCAL memory type");
            expect(drv.alloc_attempts.size() == 1 && drv.alloc_attempts[0].type == 0,
                   "PresentComposer uses host-only type 0");
        }
    }
    expect(drv.all_clean(), "PresentComposer test leaked or double-freed Vulkan handles");
}

void test_occlusion_resolve_memory_fallback() {
    MockDriver drv;
    g_drv = &drv;
    leonida::VulkanDeviceFunctions vk{};
    expect(vk.load(mock_get_proc, make_handle<VkDevice>(1)), "VulkanDeviceFunctions loaded");

    // 1. Non-ReBAR: picks pure VRAM (1) over 256 MiB BAR (0).
    {
        const auto mem = make_non_rebar_props();
        leonida::OcclusionResolve helper(vk, make_handle<VkDevice>(1));
        expect(helper.initialize(mem), "OcclusionResolve initializes on non-ReBAR GPU");
        expect(drv.alloc_attempts.size() == 1 && drv.alloc_attempts[0].type == 1,
               "OcclusionResolve picks pure VRAM (1) before BAR (0)");
    }

    // 2. Non-ReBAR OOM cascade: VRAM (1) and BAR (0) fail -> succeeds on RAM (2).
    {
        drv.reset_counters();
        drv.fail_memory_types = {1, 0};
        drv.write_garbage_on_alloc_fail = true;
        const auto mem = make_non_rebar_props();
        leonida::OcclusionResolve helper(vk, make_handle<VkDevice>(1));
        expect(helper.initialize(mem), "OcclusionResolve falls back to RAM on VRAM+BAR OOM");
        expect(drv.alloc_attempts.size() == 3 && drv.alloc_attempts[0].type == 1 &&
                   drv.alloc_attempts[1].type == 0 && drv.alloc_attempts[2].type == 2,
               "OcclusionResolve tries VRAM (1) -> BAR (0) -> RAM (2)");
        drv.fail_memory_types.clear();
        drv.write_garbage_on_alloc_fail = false;
    }

    // 3. Undersized heap skipped (VUID-vkAllocateMemory-pAllocateInfo-01713).
    {
        drv.reset_counters();
        drv.forced_buffer_req_size = VkDeviceSize{8} << 20;
        auto mem = make_non_rebar_props();
        mem.memoryHeaps[0].size = VkDeviceSize{4} << 20;
        mem.memoryHeaps[1].size = VkDeviceSize{4} << 20;
        leonida::OcclusionResolve helper(vk, make_handle<VkDevice>(1));
        expect(helper.initialize(mem), "OcclusionResolve skips undersized heaps");
        expect(drv.alloc_attempts.size() == 1 && drv.alloc_attempts[0].type == 2,
               "OcclusionResolve avoids calling vkAllocateMemory on undersized heaps");
        drv.forced_buffer_req_size = 0;
    }

    // 4. Synthetic zero-heap mock compatibility (like replay_query_edges_test.cpp).
    {
        drv.reset_counters();
        VkPhysicalDeviceMemoryProperties mem{};
        mem.memoryTypeCount = 1;
        leonida::OcclusionResolve helper(vk, make_handle<VkDevice>(1));
        expect(helper.initialize(mem), "OcclusionResolve still accepts synthetic zero-heap mock memory properties");
    }

    expect(drv.all_clean(), "OcclusionResolve test leaked or double-freed Vulkan handles");
}

leonida::ResourceBinding make_host_buffer(std::uintptr_t id, std::uint64_t va, std::uint64_t size,
                                          const std::uint8_t* host) {
    leonida::ResourceBinding b{};
    b.buffer = make_handle<VkBuffer>(id);
    b.va = va;
    b.size = size;
    b.host = host;
    return b;
}

void test_acquire_stage_memory_policy_and_fallback() {
    constexpr std::uint64_t kHeapBase = 0x1000000ull;
    constexpr std::uint64_t kDstVa = 0x10000ull;
    constexpr std::uint64_t kInstVa = 0x4000ull;
    constexpr std::uint64_t kPtrVa = 0x8000ull;

    std::array<std::uint8_t, 128> inst_storage{};
    std::array<std::uint64_t, 4> ptrs{kInstVa, kInstVa + 64, kInstVa, kInstVa + 64};

    const auto run_once = [&](const VkPhysicalDeviceMemoryProperties& mem, const std::set<std::uint32_t>& fail_types,
                              VkDeviceSize forced_req = 0) {
        MockDriver drv;
        g_drv = &drv;
        drv.fail_memory_types = fail_types;
        drv.forced_buffer_req_size = forced_req;
        drv.write_garbage_on_alloc_fail = true;
        std::vector<AllocAttempt> attempts;
        leonida::ReplayResult result = leonida::ReplayResult::unsupported;
        {
            leonida::VulkanContext ctx(mock_get_proc, make_handle<VkDevice>(1), false, false, true, false);
            ctx.set_occlusion_resolve_memory(mem);
            ctx.set_heap_base(kHeapBase);
            ctx.register_acceleration_structure(kDstVa, make_handle<VkAccelerationStructureKHR>(0x500));
            ctx.add_resource(1, make_host_buffer(1, kInstVa, inst_storage.size(), inst_storage.data()));
            ctx.add_resource(2, make_host_buffer(2, kPtrVa, ptrs.size() * sizeof(std::uint64_t),
                                                 reinterpret_cast<const std::uint8_t*>(ptrs.data())));
            leonida::VulkanQueue queue(ctx, make_handle<VkQueue>(1), 0);
            expect(queue.ok(), "acquire_stage mock queue initialized");

            leonida::BuildAccelerationStructure b{};
            b.is_tlas = true;
            b.dst_va = kDstVa;
            b.scratch_va = 0x20000;
            b.instances_va = kPtrVa;
            b.instance_count = 2;
            b.instance_pointers = true;
            leonida::CommandList list;
            expect(list.build_acceleration_structure(std::move(b)) && list.close(), "record KHR TLAS list");
            const leonida::CommandList* lists[] = {&list};
            result = queue.execute(lists);
            attempts = drv.alloc_attempts;
        }
        expect(drv.all_clean(), "acquire_stage test leaked or double-freed Vulkan handles");
        return std::pair{result, attempts};
    };

    // 1. Non-ReBAR: prefers system RAM (type 2) before 256 MiB BAR (type 0).
    {
        const auto [res, attempts] = run_once(make_non_rebar_props(), {});
        expect(res == leonida::ReplayResult::success, "non-ReBAR acquire_stage succeeds");
        expect(attempts.size() == 1 && attempts[0].type == 2,
               "non-ReBAR acquire_stage picks system RAM (2) before 256 MiB BAR (0)");
    }

    // 2. Non-ReBAR OOM on system RAM (type 2): falls back to BAR (type 0) and succeeds.
    {
        const auto [res, attempts] = run_once(make_non_rebar_props(), {2});
        expect(res == leonida::ReplayResult::success, "non-ReBAR acquire_stage falls back from RAM to BAR on OOM");
        expect(attempts.size() == 2 && attempts[0].type == 2 && attempts[1].type == 0,
               "non-ReBAR acquire_stage tries RAM (2) then BAR (0)");
    }

    // 3. ReBAR: prefers DEVICE_LOCAL | HOST_VISIBLE (type 0) first; falls back to RAM (type 1) on OOM.
    {
        const auto [res1, a1] = run_once(make_rebar_props(), {});
        expect(res1 == leonida::ReplayResult::success && a1.size() == 1 && a1[0].type == 0,
               "ReBAR acquire_stage prefers ReBAR VRAM (0)");

        const auto [res2, a2] = run_once(make_rebar_props(), {0});
        expect(res2 == leonida::ReplayResult::success && a2.size() == 2 && a2[0].type == 0 && a2[1].type == 1,
               "ReBAR acquire_stage falls back to system RAM (1) on VRAM OOM");
    }

    // 4. Heap smaller than requirements.size is skipped (VUID-vkAllocateMemory-pAllocateInfo-01713).
    {
        auto mem = make_rebar_props();
        mem.memoryHeaps[0].size = VkDeviceSize{1} << 20;  // 1 MiB < forced 4 MiB
        const auto [res, attempts] = run_once(mem, {}, VkDeviceSize{4} << 20);
        expect(res == leonida::ReplayResult::success && attempts.size() == 1 && attempts[0].type == 1,
               "acquire_stage skips undersized host-visible heap without calling vkAllocateMemory on it");
    }
}

void test_khr_stage_pool_cap_and_best_fit() {
    MockDriver drv;
    g_drv = &drv;

    constexpr std::uint64_t kHeapBase = 0x1000000ull;
    constexpr std::uint64_t kDstVa = 0x10000ull;
    constexpr std::uint64_t kInstVa = 0x4000ull;
    constexpr std::uint64_t kPtrVa = 0x8000ull;

    std::array<std::uint8_t, 128> inst_storage{};
    std::array<std::uint64_t, 128> ptrs{};
    ptrs.fill(kInstVa);

    // Track every alias we add so we leave the process-global title_va_aliases() clean.
    std::vector<std::pair<std::uint64_t, std::uint64_t>> added_aliases;
    const auto clear_aliases = [&] {
        for (const auto& [tva, sva] : added_aliases) leonida::title_va_aliases().remove(tva, sva);
        added_aliases.clear();
    };

    {
        leonida::VulkanContext ctx(mock_get_proc, make_handle<VkDevice>(1), false, false, false, true);
        expect(ctx.loaded(), "XDXR context loaded");
        ctx.set_occlusion_resolve_memory(make_non_rebar_props());
        ctx.set_heap_base(kHeapBase);
        ctx.add_resource(1, make_host_buffer(1, kInstVa, inst_storage.size(), inst_storage.data()));
        ctx.add_resource(2, make_host_buffer(2, kPtrVa, ptrs.size() * sizeof(std::uint64_t),
                                             reinterpret_cast<const std::uint8_t*>(ptrs.data())));

        leonida::VulkanQueue queue(ctx, make_handle<VkQueue>(1), 0);
        expect(queue.ok(), "stage pool cap mock queue initialized");

        // 1. Growing XDXR alias table across 32 sequential completed submissions (sizes 24..768 bytes).
        //    Each submission forces a new larger allocation; recycled smaller stages must be evicted once
        //    the pool reaches its cap (8), rather than leaking all 32 VkDeviceMemory allocations forever.
        for (std::uint32_t n = 1; n <= 32; ++n) {
            const std::uint64_t tva = 0x7000000000ull + std::uint64_t(n) * 0x10000ull;
            const std::uint64_t sva = 0x200000ull + std::uint64_t(n) * 0x10000ull;
            leonida::title_va_aliases().add(tva, 0x1000, sva);
            added_aliases.push_back({tva, sva});

            leonida::BuildAccelerationStructure b{};
            b.is_tlas = true;
            b.dst_va = kDstVa;
            b.scratch_va = 0x20000;
            b.instances_va = kInstVa;
            b.instance_count = 1;
            leonida::CommandList list;
            expect(list.build_acceleration_structure(std::move(b)) && list.close(), "record XDXR TLAS list");
            const leonida::CommandList* lists[] = {&list};
            expect(queue.execute(lists) == leonida::ReplayResult::success, "growing XDXR TLAS submission succeeds");
        }
        // Flush completed slots back into khr_stage_pool_ with one empty submission.
        {
            leonida::CommandList empty;
            expect(empty.close(), "close empty flush list");
            const leonida::CommandList* lists[] = {&empty};
            expect(queue.execute(lists) == leonida::ReplayResult::success, "empty flush list succeeds");
        }
        expect(drv.live_buffers.size() <= 8 && drv.live_memories.size() <= 8,
               "khr_stage_pool_ must cap retained stages at <= 8 after 32 growing submissions");
        expect(drv.buffer_destroys >= 24 && drv.memory_frees >= 24,
               "excess obsolete stages must be destroyed instead of leaking until queue teardown");

        // 2. Subsequent submission needing a smaller stage (e.g. 5 aliases = 120 B) reuses a pooled stage (0 allocs).
        clear_aliases();
        for (std::uint32_t n = 1; n <= 5; ++n) {
            const std::uint64_t tva = 0x7100000000ull + std::uint64_t(n) * 0x10000ull;
            const std::uint64_t sva = 0x300000ull + std::uint64_t(n) * 0x10000ull;
            leonida::title_va_aliases().add(tva, 0x1000, sva);
            added_aliases.push_back({tva, sva});
        }
        const int allocs_before_reuse = drv.memory_allocs;
        {
            leonida::BuildAccelerationStructure b{};
            b.is_tlas = true;
            b.dst_va = kDstVa;
            b.scratch_va = 0x20000;
            b.instances_va = kInstVa;
            b.instance_count = 1;
            leonida::CommandList list;
            expect(list.build_acceleration_structure(std::move(b)) && list.close(), "record reuse XDXR list");
            const leonida::CommandList* lists[] = {&list};
            expect(queue.execute(lists) == leonida::ReplayResult::success, "reuse XDXR submission succeeds");
        }
        expect(drv.memory_allocs == allocs_before_reuse,
               "largest retained stages satisfy subsequent smaller XDXR table without allocating");
        clear_aliases();
    }
    expect(drv.all_clean(), "growing XDXR stage pool test leaked or double-freed Vulkan handles");

    // 3. Best-fit reuse + single-submission burst cap on KHR ARRAY_OF_POINTERS staging.
    {
        drv.reset_counters();
        leonida::VulkanContext ctx(mock_get_proc, make_handle<VkDevice>(1), false, false, true, false);
        ctx.set_occlusion_resolve_memory(make_non_rebar_props());
        ctx.set_heap_base(kHeapBase);
        ctx.register_acceleration_structure(kDstVa, make_handle<VkAccelerationStructureKHR>(0x500));
        ctx.add_resource(1, make_host_buffer(1, kInstVa, inst_storage.size(), inst_storage.data()));
        ctx.add_resource(2, make_host_buffer(2, kPtrVa, ptrs.size() * sizeof(std::uint64_t),
                                             reinterpret_cast<const std::uint8_t*>(ptrs.data())));
        leonida::VulkanQueue queue(ctx, make_handle<VkQueue>(1), 0);
        expect(queue.ok(), "KHR burst/best-fit queue initialized");

        // Submit ONE list containing 1 big TLAS (64 instances = 512 B) followed by 1 small TLAS (4 instances = 32 B).
        // Pool receives [512 B, 32 B] in that order (big first!).
        {
            leonida::CommandList list;
            for (std::uint32_t cnt : {64u, 4u}) {
                leonida::BuildAccelerationStructure b{};
                b.is_tlas = true;
                b.dst_va = kDstVa;
                b.scratch_va = 0x20000;
                b.instances_va = kPtrVa;
                b.instance_count = cnt;
                b.instance_pointers = true;
                expect(list.build_acceleration_structure(std::move(b)), "record best-fit seed build");
            }
            expect(list.close(), "close best-fit seed list");
            const leonida::CommandList* lists[] = {&list};
            expect(queue.execute(lists) == leonida::ReplayResult::success, "seed best-fit pool");
        }
        // Next submission asks for small (4 instances = 32 B) FIRST, then big (64 instances = 512 B).
        // First-fit would grab the 512 B stage for the 32 B request and force a brand-new 512 B allocation!
        // Best-fit must take the 32 B stage for the 32 B request and the 512 B stage for the 512 B request (0 allocs).
        const int allocs_before_best_fit = drv.memory_allocs;
        {
            leonida::CommandList list;
            for (std::uint32_t cnt : {4u, 64u}) {
                leonida::BuildAccelerationStructure b{};
                b.is_tlas = true;
                b.dst_va = kDstVa;
                b.scratch_va = 0x20000;
                b.instances_va = kPtrVa;
                b.instance_count = cnt;
                b.instance_pointers = true;
                expect(list.build_acceleration_structure(std::move(b)), "record best-fit check build");
            }
            expect(list.close(), "close best-fit check list");
            const leonida::CommandList* lists[] = {&list};
            expect(queue.execute(lists) == leonida::ReplayResult::success, "best-fit submission succeeds");
        }
        expect(drv.memory_allocs == allocs_before_best_fit,
               "acquire_stage best-fit reuses 32 B stage for small build so 512 B stage remains for large build");

        // Burst of 16 builds in one submission -> after completion + flush, pool is trimmed to <= 8 live allocations.
        {
            leonida::CommandList burst;
            for (int k = 0; k < 16; ++k) {
                leonida::BuildAccelerationStructure b{};
                b.is_tlas = true;
                b.dst_va = kDstVa;
                b.scratch_va = 0x20000;
                b.instances_va = kPtrVa;
                b.instance_count = 2;
                b.instance_pointers = true;
                expect(burst.build_acceleration_structure(std::move(b)), "record burst build");
            }
            expect(burst.close(), "close burst list");
            const leonida::CommandList* lists[] = {&burst};
            expect(queue.execute(lists) == leonida::ReplayResult::success, "burst submission succeeds");
        }
        {
            leonida::CommandList empty;
            expect(empty.close(), "close post-burst flush");
            const leonida::CommandList* lists[] = {&empty};
            expect(queue.execute(lists) == leonida::ReplayResult::success, "post-burst flush succeeds");
        }
        expect(drv.live_buffers.size() <= 8 && drv.live_memories.size() <= 8,
               "single-submission burst of 16 KHR stages is trimmed to <= 8 pooled stages on recycle");
    }
    expect(drv.all_clean(), "KHR stage burst/best-fit test leaked or double-freed Vulkan handles");
}

void test_khr_as_query_pool_cap_and_best_fit() {
    MockDriver drv;
    g_drv = &drv;

    constexpr std::uint64_t kDstBufVa = 0x50000ull;
    constexpr std::uint64_t kAsBaseVa = 0x10000ull;

    {
        leonida::VulkanContext ctx(mock_get_proc, make_handle<VkDevice>(1), false, false, true, false);
        ctx.set_occlusion_resolve_memory(make_non_rebar_props());
        ctx.add_resource(1, make_host_buffer(1, kDstBufVa, 4096, nullptr));
        for (std::uint32_t i = 0; i < 64; ++i) {
            ctx.register_acceleration_structure(kAsBaseVa + i * 0x100ull,
                                                make_handle<VkAccelerationStructureKHR>(0x600 + i));
        }

        leonida::VulkanQueue queue(ctx, make_handle<VkQueue>(1), 0);
        expect(queue.ok(), "KHR query pool mock queue initialized");

        // 1. Best-fit reuse: seed pool with a 64-entry pool FIRST, then a 16-entry pool in one submission
        //    (two postbuilds in one list: first 64 fills a 64-cap pool; second 16 needs a new 16-cap pool).
        const auto make_sources = [](std::uint32_t count) {
            std::vector<std::uint64_t> src(count);
            for (std::uint32_t i = 0; i < count; ++i) src[i] = kAsBaseVa + (i % 64u) * 0x100ull;
            return src;
        };
        {
            leonida::CommandList list;
            expect(list.acceleration_structure_postbuild({kDstBufVa, 0, make_sources(64)}), "record 64-src postbuild");
            expect(list.acceleration_structure_postbuild({kDstBufVa, 0, make_sources(16)}), "record 16-src postbuild");
            expect(list.close(), "close query best-fit seed list");
            const leonida::CommandList* lists[] = {&list};
            expect(queue.execute(lists) == leonida::ReplayResult::success, "seed query pool with 64 and 16");
        }
        const int pools_before_best_fit = drv.query_pool_creates;
        {
            leonida::CommandList list;
            // Request 16 FIRST, then 48: best-fit picks the 16-cap pool first, leaving the 64-cap pool for 48!
            expect(list.acceleration_structure_postbuild({kDstBufVa, 0, make_sources(16)}), "record 16-src check");
            expect(list.acceleration_structure_postbuild({kDstBufVa, 0, make_sources(48)}), "record 48-src check");
            expect(list.close(), "close query best-fit check list");
            const leonida::CommandList* lists[] = {&list};
            expect(queue.execute(lists) == leonida::ReplayResult::success, "best-fit query submission succeeds");
        }
        expect(drv.query_pool_creates == pools_before_best_fit,
               "allocate_khr_as_queries best-fit reuses 16-cap pool first so 64-cap pool satisfies 48-src postbuild");

        // 2. Growing counts across 24 sequential submissions (17..40) must cap pooled VkQueryPools at <= 8.
        //    First grow past 64 so every step requires a strictly larger pool.
        for (std::uint32_t n = 65; n <= 88; ++n) {
            leonida::CommandList list;
            expect(list.acceleration_structure_postbuild({kDstBufVa, 0, make_sources(n)}), "record growing postbuild");
            expect(list.close(), "close growing postbuild list");
            const leonida::CommandList* lists[] = {&list};
            expect(queue.execute(lists) == leonida::ReplayResult::success, "growing postbuild succeeds");
        }
        {
            leonida::CommandList empty;
            expect(empty.close(), "close query flush list");
            const leonida::CommandList* lists[] = {&empty};
            expect(queue.execute(lists) == leonida::ReplayResult::success, "query flush succeeds");
        }
        expect(drv.live_query_pools.size() <= 8,
               "khr_as_query_pool_ must cap retained VkQueryPools at <= 8 after 24 growing submissions");
        expect(drv.query_pool_destroys >= 16,
               "evicted VkQueryPools must be destroyed via vkDestroyQueryPool");
    }
    expect(drv.all_clean(), "KHR AS query pool test leaked or double-freed Vulkan handles");
}

}  // namespace

int main() {
    test_present_composer_memory_and_extent();
    test_occlusion_resolve_memory_fallback();
    test_acquire_stage_memory_policy_and_fallback();
    test_khr_stage_pool_cap_and_best_fit();
    test_khr_as_query_pool_cap_and_best_fit();
    if (g_failures) {
        std::fprintf(stderr, "vk_memory_fallback_and_pools: %d failure(s)\n", g_failures);
        return 1;
    }
    std::puts("vk_memory_fallback_and_pools: ok");
    return 0;
}
