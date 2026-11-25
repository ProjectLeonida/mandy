// CPU-only unit test for BindlessSet memory-type ordering, heap-size filtering
// (VUID-vkAllocateMemory-pAllocateInfo-01713), BAR avoidance on non-ReBAR GPUs,
// multi-type OOM retry fallback, and failed-handle cleanup (runtime/src/vulkan_bindless.h).
#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include "../src/vulkan_bindless.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#define EXPECT(condition)                                                  \
    do {                                                                   \
        if (!(condition)) {                                                \
            std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #condition); \
            return 1;                                                      \
        }                                                                  \
    } while (0)

namespace {

using namespace leonida;

struct MockVulkan {
    VkPhysicalDeviceMemoryProperties memory{};
    std::uint32_t fail_type_mask = 0;  // memory type indices that fail vkAllocateMemory
    VkResult fail_result = VK_ERROR_OUT_OF_DEVICE_MEMORY;
    bool write_garbage_on_fail = false;
    bool fail_create_sampler = false;
    VkDeviceSize min_buffer_req_size = 0;
    VkDeviceSize image_req_size = 4096;

    std::uint64_t next_handle = 1;
    std::unordered_set<std::uint64_t> live_layouts;
    std::unordered_set<std::uint64_t> live_pools;
    std::unordered_set<std::uint64_t> live_buffers;
    std::unordered_set<std::uint64_t> live_images;
    std::unordered_set<std::uint64_t> live_views;
    std::unordered_set<std::uint64_t> live_samplers;
    std::unordered_set<std::uint64_t> live_memories;
    std::unordered_map<std::uint64_t, VkBufferUsageFlags> buffer_usage;
    std::unordered_map<std::uint64_t, VkDeviceSize> buffer_size;
    std::unordered_map<std::uint64_t, std::uint32_t> memory_type_of;
    std::unordered_map<std::uint64_t, std::vector<std::uint8_t>> mapped_backing;

    struct AllocAttempt {
        std::uint32_t type_index;
        VkDeviceSize size;
        VkResult result;
    };
    std::vector<AllocAttempt> alloc_attempts;
    std::vector<std::uint32_t> table_types_bound;
    std::vector<std::uint32_t> staging_types_bound;
    std::vector<std::uint32_t> image_types_bound;
    unsigned invalid_frees = 0;
    unsigned undersized_alloc_attempts = 0;

    static constexpr std::uint64_t kGarbageHandle = 0xdeadbeef00001234ull;

    template <typename Handle>
    Handle make_handle() {
        return reinterpret_cast<Handle>(static_cast<std::uintptr_t>(next_handle++));
    }

    template <typename Handle>
    static std::uint64_t handle_key(Handle h) {
        return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(h));
    }

    bool clean() const {
        return invalid_frees == 0 && live_layouts.empty() && live_pools.empty() && live_buffers.empty() &&
               live_images.empty() && live_views.empty() && live_samplers.empty() && live_memories.empty();
    }
};

MockVulkan* g_mock = nullptr;

VKAPI_ATTR VkResult VKAPI_CALL mock_create_descriptor_set_layout(VkDevice, const VkDescriptorSetLayoutCreateInfo*,
                                                                 const VkAllocationCallbacks*,
                                                                 VkDescriptorSetLayout* out) {
    *out = g_mock->make_handle<VkDescriptorSetLayout>();
    g_mock->live_layouts.insert(MockVulkan::handle_key(*out));
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL mock_destroy_descriptor_set_layout(VkDevice, VkDescriptorSetLayout layout,
                                                              const VkAllocationCallbacks*) {
    if (!g_mock->live_layouts.erase(MockVulkan::handle_key(layout))) ++g_mock->invalid_frees;
}

VKAPI_ATTR VkResult VKAPI_CALL mock_create_descriptor_pool(VkDevice, const VkDescriptorPoolCreateInfo*,
                                                           const VkAllocationCallbacks*, VkDescriptorPool* out) {
    *out = g_mock->make_handle<VkDescriptorPool>();
    g_mock->live_pools.insert(MockVulkan::handle_key(*out));
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL mock_destroy_descriptor_pool(VkDevice, VkDescriptorPool pool, const VkAllocationCallbacks*) {
    if (!g_mock->live_pools.erase(MockVulkan::handle_key(pool))) ++g_mock->invalid_frees;
}

VKAPI_ATTR VkResult VKAPI_CALL mock_reset_descriptor_pool(VkDevice, VkDescriptorPool, VkDescriptorPoolResetFlags) {
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL mock_allocate_descriptor_sets(VkDevice, const VkDescriptorSetAllocateInfo* info,
                                                             VkDescriptorSet* out) {
    for (std::uint32_t i = 0; i < info->descriptorSetCount; ++i) out[i] = g_mock->make_handle<VkDescriptorSet>();
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL mock_update_descriptor_sets(VkDevice, std::uint32_t, const VkWriteDescriptorSet*,
                                                       std::uint32_t, const VkCopyDescriptorSet*) {}

VKAPI_ATTR VkResult VKAPI_CALL mock_create_buffer(VkDevice, const VkBufferCreateInfo* info,
                                                  const VkAllocationCallbacks*, VkBuffer* out) {
    *out = g_mock->make_handle<VkBuffer>();
    const auto key = MockVulkan::handle_key(*out);
    g_mock->live_buffers.insert(key);
    g_mock->buffer_usage[key] = info->usage;
    g_mock->buffer_size[key] = info->size;
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL mock_destroy_buffer(VkDevice, VkBuffer buffer, const VkAllocationCallbacks*) {
    const auto key = MockVulkan::handle_key(buffer);
    if (!g_mock->live_buffers.erase(key)) ++g_mock->invalid_frees;
    g_mock->buffer_usage.erase(key);
    g_mock->buffer_size.erase(key);
}

VKAPI_ATTR void VKAPI_CALL mock_get_buffer_memory_requirements(VkDevice, VkBuffer buffer,
                                                               VkMemoryRequirements* req) {
    const auto key = MockVulkan::handle_key(buffer);
    const VkDeviceSize size = g_mock->buffer_size[key];
    req->size = std::max(size, g_mock->min_buffer_req_size);
    req->alignment = 256;
    req->memoryTypeBits = g_mock->memory.memoryTypeCount >= 32
                              ? 0xffffffffu
                              : ((1u << g_mock->memory.memoryTypeCount) - 1u);
}

VKAPI_ATTR VkResult VKAPI_CALL mock_bind_buffer_memory(VkDevice, VkBuffer buffer, VkDeviceMemory memory,
                                                       VkDeviceSize) {
    const auto bkey = MockVulkan::handle_key(buffer);
    const auto mkey = MockVulkan::handle_key(memory);
    const std::uint32_t type = g_mock->memory_type_of[mkey];
    if (g_mock->buffer_usage[bkey] & VK_BUFFER_USAGE_TRANSFER_DST_BIT)
        g_mock->table_types_bound.push_back(type);
    else
        g_mock->staging_types_bound.push_back(type);
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL mock_allocate_memory(VkDevice, const VkMemoryAllocateInfo* info,
                                                    const VkAllocationCallbacks*, VkDeviceMemory* out) {
    const std::uint32_t type = info->memoryTypeIndex;
    if (g_mock->memory.memoryHeapCount > 0 && type < g_mock->memory.memoryTypeCount) {
        const std::uint32_t heap = g_mock->memory.memoryTypes[type].heapIndex;
        if (heap >= g_mock->memory.memoryHeapCount || g_mock->memory.memoryHeaps[heap].size < info->allocationSize) {
            ++g_mock->undersized_alloc_attempts;
            g_mock->alloc_attempts.push_back({type, info->allocationSize, VK_ERROR_OUT_OF_DEVICE_MEMORY});
            if (g_mock->write_garbage_on_fail)
                *out = reinterpret_cast<VkDeviceMemory>(static_cast<std::uintptr_t>(MockVulkan::kGarbageHandle));
            return VK_ERROR_OUT_OF_DEVICE_MEMORY;
        }
    }
    if (type < 32 && (g_mock->fail_type_mask & (1u << type))) {
        g_mock->alloc_attempts.push_back({type, info->allocationSize, g_mock->fail_result});
        if (g_mock->write_garbage_on_fail)
            *out = reinterpret_cast<VkDeviceMemory>(static_cast<std::uintptr_t>(MockVulkan::kGarbageHandle));
        return g_mock->fail_result;
    }
    *out = g_mock->make_handle<VkDeviceMemory>();
    const auto key = MockVulkan::handle_key(*out);
    g_mock->live_memories.insert(key);
    g_mock->memory_type_of[key] = type;
    g_mock->mapped_backing[key].resize(static_cast<std::size_t>(std::max<VkDeviceSize>(info->allocationSize, 1)), 0);
    g_mock->alloc_attempts.push_back({type, info->allocationSize, VK_SUCCESS});
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL mock_free_memory(VkDevice, VkDeviceMemory memory, const VkAllocationCallbacks*) {
    const auto key = MockVulkan::handle_key(memory);
    if (!g_mock->live_memories.erase(key)) ++g_mock->invalid_frees;
    g_mock->memory_type_of.erase(key);
    g_mock->mapped_backing.erase(key);
}

VKAPI_ATTR VkResult VKAPI_CALL mock_map_memory(VkDevice, VkDeviceMemory memory, VkDeviceSize offset, VkDeviceSize,
                                               VkMemoryMapFlags, void** data) {
    auto& buf = g_mock->mapped_backing[MockVulkan::handle_key(memory)];
    *data = buf.data() + offset;
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL mock_create_image(VkDevice, const VkImageCreateInfo*, const VkAllocationCallbacks*,
                                                 VkImage* out) {
    *out = g_mock->make_handle<VkImage>();
    g_mock->live_images.insert(MockVulkan::handle_key(*out));
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL mock_destroy_image(VkDevice, VkImage image, const VkAllocationCallbacks*) {
    if (!g_mock->live_images.erase(MockVulkan::handle_key(image))) ++g_mock->invalid_frees;
}

VKAPI_ATTR void VKAPI_CALL mock_get_image_memory_requirements(VkDevice, VkImage, VkMemoryRequirements* req) {
    req->size = g_mock->image_req_size;
    req->alignment = 256;
    req->memoryTypeBits = g_mock->memory.memoryTypeCount >= 32
                              ? 0xffffffffu
                              : ((1u << g_mock->memory.memoryTypeCount) - 1u);
}

VKAPI_ATTR VkResult VKAPI_CALL mock_bind_image_memory(VkDevice, VkImage, VkDeviceMemory memory, VkDeviceSize) {
    g_mock->image_types_bound.push_back(g_mock->memory_type_of[MockVulkan::handle_key(memory)]);
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL mock_create_image_view(VkDevice, const VkImageViewCreateInfo*,
                                                      const VkAllocationCallbacks*, VkImageView* out) {
    *out = g_mock->make_handle<VkImageView>();
    g_mock->live_views.insert(MockVulkan::handle_key(*out));
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL mock_destroy_image_view(VkDevice, VkImageView view, const VkAllocationCallbacks*) {
    if (!g_mock->live_views.erase(MockVulkan::handle_key(view))) ++g_mock->invalid_frees;
}

VKAPI_ATTR VkResult VKAPI_CALL mock_create_sampler(VkDevice, const VkSamplerCreateInfo*, const VkAllocationCallbacks*,
                                                   VkSampler* out) {
    if (g_mock->fail_create_sampler) {
        if (g_mock->write_garbage_on_fail)
            *out = reinterpret_cast<VkSampler>(static_cast<std::uintptr_t>(MockVulkan::kGarbageHandle));
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    *out = g_mock->make_handle<VkSampler>();
    g_mock->live_samplers.insert(MockVulkan::handle_key(*out));
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL mock_destroy_sampler(VkDevice, VkSampler sampler, const VkAllocationCallbacks*) {
    if (!g_mock->live_samplers.erase(MockVulkan::handle_key(sampler))) ++g_mock->invalid_frees;
}

VKAPI_ATTR void VKAPI_CALL mock_cmd_pipeline_barrier(VkCommandBuffer, VkPipelineStageFlags, VkPipelineStageFlags,
                                                     VkDependencyFlags, std::uint32_t, const VkMemoryBarrier*,
                                                     std::uint32_t, const VkBufferMemoryBarrier*, std::uint32_t,
                                                     const VkImageMemoryBarrier*) {}

VKAPI_ATTR void VKAPI_CALL mock_cmd_copy_buffer(VkCommandBuffer, VkBuffer, VkBuffer, std::uint32_t,
                                                const VkBufferCopy*) {}

VKAPI_ATTR void VKAPI_CALL mock_cmd_update_buffer(VkCommandBuffer, VkBuffer, VkDeviceSize, VkDeviceSize, const void*) {}

VKAPI_ATTR void VKAPI_CALL mock_cmd_clear_color_image(VkCommandBuffer, VkImage, VkImageLayout, const VkClearColorValue*,
                                                      std::uint32_t, const VkImageSubresourceRange*) {}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL mock_get_device_proc_addr(VkDevice, const char* name) {
#define MATCH(fn, target) \
    if (std::strcmp(name, #fn) == 0) return reinterpret_cast<PFN_vkVoidFunction>(target)
    MATCH(vkCreateDescriptorSetLayout, mock_create_descriptor_set_layout);
    MATCH(vkDestroyDescriptorSetLayout, mock_destroy_descriptor_set_layout);
    MATCH(vkCreateDescriptorPool, mock_create_descriptor_pool);
    MATCH(vkDestroyDescriptorPool, mock_destroy_descriptor_pool);
    MATCH(vkResetDescriptorPool, mock_reset_descriptor_pool);
    MATCH(vkAllocateDescriptorSets, mock_allocate_descriptor_sets);
    MATCH(vkUpdateDescriptorSets, mock_update_descriptor_sets);
    MATCH(vkCreateBuffer, mock_create_buffer);
    MATCH(vkDestroyBuffer, mock_destroy_buffer);
    MATCH(vkGetBufferMemoryRequirements, mock_get_buffer_memory_requirements);
    MATCH(vkBindBufferMemory, mock_bind_buffer_memory);
    MATCH(vkAllocateMemory, mock_allocate_memory);
    MATCH(vkFreeMemory, mock_free_memory);
    MATCH(vkMapMemory, mock_map_memory);
    MATCH(vkCreateImage, mock_create_image);
    MATCH(vkDestroyImage, mock_destroy_image);
    MATCH(vkGetImageMemoryRequirements, mock_get_image_memory_requirements);
    MATCH(vkBindImageMemory, mock_bind_image_memory);
    MATCH(vkCreateImageView, mock_create_image_view);
    MATCH(vkDestroyImageView, mock_destroy_image_view);
    MATCH(vkCreateSampler, mock_create_sampler);
    MATCH(vkDestroySampler, mock_destroy_sampler);
    MATCH(vkCmdPipelineBarrier, mock_cmd_pipeline_barrier);
    MATCH(vkCmdCopyBuffer, mock_cmd_copy_buffer);
    MATCH(vkCmdUpdateBuffer, mock_cmd_update_buffer);
    MATCH(vkCmdClearColorImage, mock_cmd_clear_color_image);
#undef MATCH
    return nullptr;
}

VkPhysicalDeviceMemoryProperties make_non_rebar_bar_first() {
    VkPhysicalDeviceMemoryProperties m{};
    m.memoryHeapCount = 3;
    m.memoryHeaps[0] = {VkDeviceSize(256) << 20, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT};   // 256 MiB BAR
    m.memoryHeaps[1] = {VkDeviceSize(8) << 30, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT};     // 8 GiB VRAM
    m.memoryHeaps[2] = {VkDeviceSize(16) << 30, 0};                                  // 16 GiB RAM
    m.memoryTypeCount = 3;
    m.memoryTypes[0] = {VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                        0};  // BAR
    m.memoryTypes[1] = {VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 1};  // VRAM
    m.memoryTypes[2] = {VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 2};  // RAM
    return m;
}

VkPhysicalDeviceMemoryProperties make_rebar_layout() {
    VkPhysicalDeviceMemoryProperties m{};
    m.memoryHeapCount = 2;
    m.memoryHeaps[0] = {VkDeviceSize(16) << 30, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT};  // 16 GiB ReBAR VRAM
    m.memoryHeaps[1] = {VkDeviceSize(32) << 30, 0};                                // 32 GiB RAM
    m.memoryTypeCount = 3;
    m.memoryTypes[0] = {VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0};
    m.memoryTypes[1] = {VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                        0};  // ReBAR
    m.memoryTypes[2] = {VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 1};
    return m;
}

DescriptorKey make_key(std::uint32_t n) {
    return {n, n * 0x9e3779b9u, n ^ 0xa5a5a5a5u, n + 17u, 1u, n * 3u, ~n, n << 1};
}

void force_growth(BindlessSet& set, std::uint32_t base_key) {
    const std::size_t before = set.table().capacity();
    for (std::uint32_t k = 0; set.table().capacity() == before; ++k) set.insert_key(make_key(base_key + k), 0);
}

int test_non_rebar_bar_avoidance_and_local_eq_fallback_oom() {
    MockVulkan mock;
    mock.memory = make_non_rebar_bar_first();
    g_mock = &mock;
    const auto fake_device = reinterpret_cast<VkDevice>(std::uintptr_t(0x100));
    const auto fake_cmd = reinterpret_cast<VkCommandBuffer>(std::uintptr_t(0x200));
    {
        BindlessSet set(mock_get_device_proc_addr, fake_device, mock.memory);
        EXPECT(set.ok());
        // Initial table, staging, and fallback images all stay off the 256 MiB BAR window (type 0).
        EXPECT(!mock.table_types_bound.empty() && mock.table_types_bound.back() == 1);
        EXPECT(!mock.staging_types_bound.empty() && mock.staging_types_bound.back() == 2);
        EXPECT(!mock.image_types_bound.empty());
        for (std::uint32_t t : mock.image_types_bound) EXPECT(t == 1);
        EXPECT(set.table_in_device_local_memory());
        EXPECT(set.table_host_fallbacks() == 0);

        EXPECT(set.flush(fake_cmd, 1, [] { return std::uint64_t(0); }));
        set.table_upload_result(true);

        // Fail BOTH type 0 (BAR) and type 1 (VRAM): table must fall back to type 2 (RAM) instead of
        // hitting local == fallback == 0 and degrading.
        mock.fail_type_mask = (1u << 0) | (1u << 1);
        force_growth(set, 1000);
        EXPECT(set.flush(fake_cmd, 2, [] { return std::uint64_t(0); }));
        set.table_upload_result(true);
        EXPECT(!set.last_flush_degraded());
        EXPECT(set.degraded_flushes() == 0);
        EXPECT(!set.table_in_device_local_memory());
        EXPECT(set.table_host_fallbacks() == 1);
        EXPECT(mock.table_types_bound.back() == 2);
        EXPECT(mock.staging_types_bound.back() == 2);

        // Fail ONLY main VRAM (type 1): table falls back to secondary DEVICE_LOCAL (BAR type 0),
        // staging still uses RAM (type 2), and table_host_fallbacks stays 1.
        mock.fail_type_mask = (1u << 1);
        force_growth(set, 100000);
        EXPECT(set.flush(fake_cmd, 3, [] { return std::uint64_t(0); }));
        set.table_upload_result(true);
        EXPECT(!set.last_flush_degraded());
        EXPECT(set.table_in_device_local_memory());
        EXPECT(set.table_host_fallbacks() == 1);
        EXPECT(mock.table_types_bound.back() == 0);
        EXPECT(mock.staging_types_bound.back() == 2);

        // Fail RAM (type 2): staging falls back to BAR (type 0) instead of degrading.
        mock.fail_type_mask = (1u << 2);
        force_growth(set, 200000);
        EXPECT(set.flush(fake_cmd, 4, [] { return std::uint64_t(0); }));
        set.table_upload_result(true);
        EXPECT(!set.last_flush_degraded());
        EXPECT(mock.table_types_bound.back() == 1);
        EXPECT(mock.staging_types_bound.back() == 0);
    }
    EXPECT(mock.clean());
    return 0;
}

int test_heap_size_filtering_vuid_01713() {
    MockVulkan mock;
    mock.memory = make_non_rebar_bar_first();
    // Shrink heap 0 (BAR) below every allocation size so VUID-01713 forbids choosing it.
    mock.memory.memoryHeaps[0].size = 512;
    mock.min_buffer_req_size = 4096;
    mock.image_req_size = 4096;
    g_mock = &mock;
    const auto fake_device = reinterpret_cast<VkDevice>(std::uintptr_t(0x100));
    const auto fake_cmd = reinterpret_cast<VkCommandBuffer>(std::uintptr_t(0x200));
    {
        BindlessSet set(mock_get_device_proc_addr, fake_device, mock.memory);
        EXPECT(set.ok());
        EXPECT(mock.undersized_alloc_attempts == 0);
        EXPECT(mock.table_types_bound.back() == 1);
        EXPECT(mock.staging_types_bound.back() == 2);

        EXPECT(set.flush(fake_cmd, 1, [] { return std::uint64_t(0); }));
        set.table_upload_result(true);

        // Even when VRAM (type 1) is OOM, undersized BAR (type 0) is never passed to vkAllocateMemory.
        mock.fail_type_mask = (1u << 1);
        force_growth(set, 300000);
        EXPECT(set.flush(fake_cmd, 2, [] { return std::uint64_t(0); }));
        set.table_upload_result(true);
        EXPECT(mock.undersized_alloc_attempts == 0);
        EXPECT(mock.table_types_bound.back() == 2);
        EXPECT(set.table_host_fallbacks() == 1);
    }
    EXPECT(mock.clean());

    // Also test when the ONLY DEVICE_LOCAL heap is undersized: skips straight to RAM with no VUID-01713 violation.
    MockVulkan mock2;
    mock2.memory.memoryHeapCount = 2;
    mock2.memory.memoryHeaps[0] = {512, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT};
    mock2.memory.memoryHeaps[1] = {VkDeviceSize(4) << 30, 0};
    mock2.memory.memoryTypeCount = 2;
    mock2.memory.memoryTypes[0] = {VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                   0};
    mock2.memory.memoryTypes[1] = {VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 1};
    mock2.min_buffer_req_size = 4096;
    mock2.image_req_size = 4096;
    g_mock = &mock2;
    {
        BindlessSet set(mock_get_device_proc_addr, fake_device, mock2.memory);
        EXPECT(set.ok());
        EXPECT(mock2.undersized_alloc_attempts == 0);
        EXPECT(!set.table_in_device_local_memory());
        EXPECT(set.table_host_fallbacks() == 1);
        EXPECT(mock2.table_types_bound.back() == 1);
        EXPECT(mock2.staging_types_bound.back() == 1);
        for (std::uint32_t t : mock2.image_types_bound) EXPECT(t == 1);
    }
    EXPECT(mock2.clean());
    return 0;
}

int test_fallback_image_oom_retry_and_cleanup() {
    MockVulkan mock;
    mock.memory = make_non_rebar_bar_first();
    mock.fail_type_mask = (1u << 1);  // VRAM full during BindlessSet construction
    mock.write_garbage_on_fail = true;
    g_mock = &mock;
    const auto fake_device = reinterpret_cast<VkDevice>(std::uintptr_t(0x100));
    {
        BindlessSet set(mock_get_device_proc_addr, fake_device, mock.memory);
        EXPECT(set.ok());
        // Fallback images retried off VRAM into system RAM (avoiding small BAR).
        EXPECT(!mock.image_types_bound.empty());
        for (std::uint32_t t : mock.image_types_bound) EXPECT(t == 2);
    }
    EXPECT(mock.clean());

    // All types fail during fallback image creation with garbage out-handles: clean teardown, zero invalid frees.
    MockVulkan mock_all_fail;
    mock_all_fail.memory = make_non_rebar_bar_first();
    mock_all_fail.fail_type_mask = 0x7u;
    mock_all_fail.write_garbage_on_fail = true;
    g_mock = &mock_all_fail;
    {
        BindlessSet set(mock_get_device_proc_addr, fake_device, mock_all_fail.memory);
        EXPECT(!set.ok());
        EXPECT(set.last_failure() != nullptr && std::strcmp(set.last_failure(), "fallback image") == 0);
        EXPECT(set.last_failure_result() == VK_ERROR_OUT_OF_DEVICE_MEMORY);
    }
    EXPECT(mock_all_fail.clean());

    // Sampler creation failure writing garbage handle: clean teardown, zero invalid frees.
    MockVulkan mock_sampler_fail;
    mock_sampler_fail.memory = make_non_rebar_bar_first();
    mock_sampler_fail.fail_create_sampler = true;
    mock_sampler_fail.write_garbage_on_fail = true;
    g_mock = &mock_sampler_fail;
    {
        BindlessSet set(mock_get_device_proc_addr, fake_device, mock_sampler_fail.memory);
        EXPECT(!set.ok());
    }
    EXPECT(mock_sampler_fail.clean());
    return 0;
}

int test_rebar_prefers_rebar_staging_with_ram_fallback() {
    MockVulkan mock;
    mock.memory = make_rebar_layout();
    g_mock = &mock;
    const auto fake_device = reinterpret_cast<VkDevice>(std::uintptr_t(0x100));
    const auto fake_cmd = reinterpret_cast<VkCommandBuffer>(std::uintptr_t(0x200));
    {
        BindlessSet set(mock_get_device_proc_addr, fake_device, mock.memory);
        EXPECT(set.ok());
        EXPECT(mock.table_types_bound.back() == 0);
        EXPECT(mock.staging_types_bound.back() == 1);  // ReBAR staging

        EXPECT(set.flush(fake_cmd, 1, [] { return std::uint64_t(0); }));
        set.table_upload_result(true);

        // If ReBAR VRAM (type 1) fails OOM, staging retries system RAM (type 2).
        mock.fail_type_mask = (1u << 1);
        force_growth(set, 400000);
        EXPECT(set.flush(fake_cmd, 2, [] { return std::uint64_t(0); }));
        set.table_upload_result(true);
        EXPECT(!set.last_flush_degraded());
        EXPECT(mock.table_types_bound.back() == 0);
        EXPECT(mock.staging_types_bound.back() == 2);
    }
    EXPECT(mock.clean());
    return 0;
}

int test_synthetic_zero_heap_mock_compatibility() {
    MockVulkan mock;
    mock.memory.memoryHeapCount = 0;  // synthetic mock with types populated but no heaps
    mock.memory.memoryTypeCount = 2;
    mock.memory.memoryTypes[0] = {VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0};
    mock.memory.memoryTypes[1] = {VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0};
    g_mock = &mock;
    const auto fake_device = reinterpret_cast<VkDevice>(std::uintptr_t(0x100));
    const auto fake_cmd = reinterpret_cast<VkCommandBuffer>(std::uintptr_t(0x200));
    {
        BindlessSet set(mock_get_device_proc_addr, fake_device, mock.memory);
        EXPECT(set.ok());
        EXPECT(mock.table_types_bound.back() == 0);
        EXPECT(mock.staging_types_bound.back() == 1);
        EXPECT(set.flush(fake_cmd, 1, [] { return std::uint64_t(0); }));
        set.table_upload_result(true);

        mock.fail_type_mask = (1u << 0);
        force_growth(set, 500000);
        EXPECT(set.flush(fake_cmd, 2, [] { return std::uint64_t(0); }));
        set.table_upload_result(true);
        EXPECT(!set.last_flush_degraded());
        EXPECT(mock.table_types_bound.back() == 1);
        EXPECT(set.table_host_fallbacks() == 1);
    }
    EXPECT(mock.clean());
    return 0;
}

}  // namespace

int main() {
    if (test_non_rebar_bar_avoidance_and_local_eq_fallback_oom()) return 1;
    if (test_heap_size_filtering_vuid_01713()) return 1;
    if (test_fallback_image_oom_retry_and_cleanup()) return 1;
    if (test_rebar_prefers_rebar_staging_with_ram_fallback()) return 1;
    if (test_synthetic_zero_heap_mock_compatibility()) return 1;
    std::printf("vk_bindless_memory_fallback_test: ok\n");
    return 0;
}
