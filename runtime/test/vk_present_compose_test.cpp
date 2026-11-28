// Portable CPU-only tests for leonida::PresentComposer (runtime/src/vulkan_present_compose.h):
//   - BAR-aware pure-VRAM selection on non-ReBAR GPUs, heap-size filtering, and OOM retry fallback;
//   - Complete Vulkan device proc validation in ready() (no null-proc dereference on partial devices);
//   - Immediate cleanup of partial pipeline objects on ready() failure;
//   - Zero-extent refusal without destroying an existing valid compose target.
// No GPU, Vulkan loader, or game data required.
#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include "../src/vulkan_present_compose.h"

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
    std::uintptr_t next_id = 100;
    const char* omit_proc = nullptr;
    bool fail_create_pipeline_layout = false;
    bool fail_create_image = false;
    bool garbage_on_image_error = false;
    std::uint32_t oom_type = UINT32_MAX;
    std::uint32_t memory_type_bits = 0x7u;
    VkDeviceSize req_size = VkDeviceSize(64) << 20;  // 64 MiB (4K RGBA16F)

    int live_set_layouts = 0;
    int live_pools = 0;
    int live_pipeline_layouts = 0;
    int live_pipelines = 0;
    int live_shaders = 0;
    int live_images = 0;
    int live_views = 0;
    int live_memory = 0;
    int create_image_calls = 0;
    std::vector<std::uint32_t> alloc_types;
};

MockVk* g_vk = nullptr;

VKAPI_ATTR VkResult VKAPI_CALL m_create_set_layout(VkDevice, const VkDescriptorSetLayoutCreateInfo*,
                                                   const VkAllocationCallbacks*, VkDescriptorSetLayout* out) {
    ++g_vk->live_set_layouts;
    *out = handle<VkDescriptorSetLayout>(++g_vk->next_id);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL m_destroy_set_layout(VkDevice, VkDescriptorSetLayout, const VkAllocationCallbacks*) {
    --g_vk->live_set_layouts;
}
VKAPI_ATTR VkResult VKAPI_CALL m_create_pool(VkDevice, const VkDescriptorPoolCreateInfo*, const VkAllocationCallbacks*,
                                             VkDescriptorPool* out) {
    ++g_vk->live_pools;
    *out = handle<VkDescriptorPool>(++g_vk->next_id);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL m_destroy_pool(VkDevice, VkDescriptorPool, const VkAllocationCallbacks*) {
    --g_vk->live_pools;
}
VKAPI_ATTR VkResult VKAPI_CALL m_alloc_sets(VkDevice, const VkDescriptorSetAllocateInfo* info, VkDescriptorSet* out) {
    for (std::uint32_t i = 0; i < info->descriptorSetCount; ++i) out[i] = handle<VkDescriptorSet>(++g_vk->next_id);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL m_update_sets(VkDevice, std::uint32_t, const VkWriteDescriptorSet*, std::uint32_t,
                                         const VkCopyDescriptorSet*) {}
VKAPI_ATTR VkResult VKAPI_CALL m_create_layout(VkDevice, const VkPipelineLayoutCreateInfo*,
                                               const VkAllocationCallbacks*, VkPipelineLayout* out) {
    if (g_vk->fail_create_pipeline_layout) {
        *out = handle<VkPipelineLayout>(0xbad0);
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    ++g_vk->live_pipeline_layouts;
    *out = handle<VkPipelineLayout>(++g_vk->next_id);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL m_destroy_layout(VkDevice, VkPipelineLayout, const VkAllocationCallbacks*) {
    --g_vk->live_pipeline_layouts;
}
VKAPI_ATTR VkResult VKAPI_CALL m_create_shader(VkDevice, const VkShaderModuleCreateInfo*, const VkAllocationCallbacks*,
                                               VkShaderModule* out) {
    ++g_vk->live_shaders;
    *out = handle<VkShaderModule>(++g_vk->next_id);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL m_destroy_shader(VkDevice, VkShaderModule, const VkAllocationCallbacks*) {
    --g_vk->live_shaders;
}
VKAPI_ATTR VkResult VKAPI_CALL m_create_pipelines(VkDevice, VkPipelineCache, std::uint32_t,
                                                  const VkComputePipelineCreateInfo*, const VkAllocationCallbacks*,
                                                  VkPipeline* out) {
    ++g_vk->live_pipelines;
    *out = handle<VkPipeline>(++g_vk->next_id);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL m_destroy_pipeline(VkDevice, VkPipeline, const VkAllocationCallbacks*) {
    --g_vk->live_pipelines;
}
VKAPI_ATTR VkResult VKAPI_CALL m_create_image(VkDevice, const VkImageCreateInfo* info, const VkAllocationCallbacks*,
                                              VkImage* out) {
    ++g_vk->create_image_calls;
    CHECK(info->extent.width > 0 && info->extent.height > 0 && info->extent.depth > 0);
    if (g_vk->fail_create_image) {
        if (g_vk->garbage_on_image_error) *out = handle<VkImage>(0xdeadbeef);
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    }
    ++g_vk->live_images;
    *out = handle<VkImage>(++g_vk->next_id);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL m_destroy_image(VkDevice, VkImage, const VkAllocationCallbacks*) { --g_vk->live_images; }
VKAPI_ATTR void VKAPI_CALL m_get_image_reqs(VkDevice, VkImage, VkMemoryRequirements* r) {
    r->size = g_vk->req_size;
    r->alignment = 65536;
    r->memoryTypeBits = g_vk->memory_type_bits;
}
VKAPI_ATTR VkResult VKAPI_CALL m_alloc_mem(VkDevice, const VkMemoryAllocateInfo* info, const VkAllocationCallbacks*,
                                           VkDeviceMemory* out) {
    g_vk->alloc_types.push_back(info->memoryTypeIndex);
    if (info->memoryTypeIndex == g_vk->oom_type) {
        *out = VK_NULL_HANDLE;
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    }
    ++g_vk->live_memory;
    *out = handle<VkDeviceMemory>(++g_vk->next_id);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL m_free_mem(VkDevice, VkDeviceMemory, const VkAllocationCallbacks*) { --g_vk->live_memory; }
VKAPI_ATTR VkResult VKAPI_CALL m_bind_image(VkDevice, VkImage, VkDeviceMemory, VkDeviceSize) { return VK_SUCCESS; }
VKAPI_ATTR VkResult VKAPI_CALL m_create_view(VkDevice, const VkImageViewCreateInfo*, const VkAllocationCallbacks*,
                                             VkImageView* out) {
    ++g_vk->live_views;
    *out = handle<VkImageView>(++g_vk->next_id);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL m_destroy_view(VkDevice, VkImageView, const VkAllocationCallbacks*) { --g_vk->live_views; }
VKAPI_ATTR void VKAPI_CALL m_cmd_bind_pipe(VkCommandBuffer, VkPipelineBindPoint, VkPipeline) {}
VKAPI_ATTR void VKAPI_CALL m_cmd_bind_sets(VkCommandBuffer, VkPipelineBindPoint, VkPipelineLayout, std::uint32_t,
                                           std::uint32_t, const VkDescriptorSet*, std::uint32_t,
                                           const std::uint32_t*) {}
VKAPI_ATTR void VKAPI_CALL m_cmd_push(VkCommandBuffer, VkPipelineLayout, VkShaderStageFlags, std::uint32_t,
                                      std::uint32_t, const void*) {}
VKAPI_ATTR void VKAPI_CALL m_cmd_dispatch(VkCommandBuffer, std::uint32_t, std::uint32_t, std::uint32_t) {}
VKAPI_ATTR void VKAPI_CALL m_cmd_barrier(VkCommandBuffer, VkPipelineStageFlags, VkPipelineStageFlags, VkDependencyFlags,
                                         std::uint32_t, const VkMemoryBarrier*, std::uint32_t,
                                         const VkBufferMemoryBarrier*, std::uint32_t, const VkImageMemoryBarrier*) {}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL m_gdpa(VkDevice, const char* name) {
    if (g_vk->omit_proc && std::strcmp(name, g_vk->omit_proc) == 0) return nullptr;
    if (std::strcmp(name, "vkCreateDescriptorSetLayout") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(m_create_set_layout);
    if (std::strcmp(name, "vkDestroyDescriptorSetLayout") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(m_destroy_set_layout);
    if (std::strcmp(name, "vkCreateDescriptorPool") == 0) return reinterpret_cast<PFN_vkVoidFunction>(m_create_pool);
    if (std::strcmp(name, "vkDestroyDescriptorPool") == 0) return reinterpret_cast<PFN_vkVoidFunction>(m_destroy_pool);
    if (std::strcmp(name, "vkAllocateDescriptorSets") == 0) return reinterpret_cast<PFN_vkVoidFunction>(m_alloc_sets);
    if (std::strcmp(name, "vkUpdateDescriptorSets") == 0) return reinterpret_cast<PFN_vkVoidFunction>(m_update_sets);
    if (std::strcmp(name, "vkCreatePipelineLayout") == 0) return reinterpret_cast<PFN_vkVoidFunction>(m_create_layout);
    if (std::strcmp(name, "vkDestroyPipelineLayout") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(m_destroy_layout);
    if (std::strcmp(name, "vkCreateShaderModule") == 0) return reinterpret_cast<PFN_vkVoidFunction>(m_create_shader);
    if (std::strcmp(name, "vkDestroyShaderModule") == 0) return reinterpret_cast<PFN_vkVoidFunction>(m_destroy_shader);
    if (std::strcmp(name, "vkCreateComputePipelines") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(m_create_pipelines);
    if (std::strcmp(name, "vkDestroyPipeline") == 0) return reinterpret_cast<PFN_vkVoidFunction>(m_destroy_pipeline);
    if (std::strcmp(name, "vkCreateImage") == 0) return reinterpret_cast<PFN_vkVoidFunction>(m_create_image);
    if (std::strcmp(name, "vkDestroyImage") == 0) return reinterpret_cast<PFN_vkVoidFunction>(m_destroy_image);
    if (std::strcmp(name, "vkGetImageMemoryRequirements") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(m_get_image_reqs);
    if (std::strcmp(name, "vkAllocateMemory") == 0) return reinterpret_cast<PFN_vkVoidFunction>(m_alloc_mem);
    if (std::strcmp(name, "vkFreeMemory") == 0) return reinterpret_cast<PFN_vkVoidFunction>(m_free_mem);
    if (std::strcmp(name, "vkBindImageMemory") == 0) return reinterpret_cast<PFN_vkVoidFunction>(m_bind_image);
    if (std::strcmp(name, "vkCreateImageView") == 0) return reinterpret_cast<PFN_vkVoidFunction>(m_create_view);
    if (std::strcmp(name, "vkDestroyImageView") == 0) return reinterpret_cast<PFN_vkVoidFunction>(m_destroy_view);
    if (std::strcmp(name, "vkCmdBindPipeline") == 0) return reinterpret_cast<PFN_vkVoidFunction>(m_cmd_bind_pipe);
    if (std::strcmp(name, "vkCmdBindDescriptorSets") == 0) return reinterpret_cast<PFN_vkVoidFunction>(m_cmd_bind_sets);
    if (std::strcmp(name, "vkCmdPushConstants") == 0) return reinterpret_cast<PFN_vkVoidFunction>(m_cmd_push);
    if (std::strcmp(name, "vkCmdDispatch") == 0) return reinterpret_cast<PFN_vkVoidFunction>(m_cmd_dispatch);
    if (std::strcmp(name, "vkCmdPipelineBarrier") == 0) return reinterpret_cast<PFN_vkVoidFunction>(m_cmd_barrier);
    return nullptr;
}

VkPhysicalDeviceMemoryProperties non_rebar_memory() {
    VkPhysicalDeviceMemoryProperties m{};
    m.memoryHeapCount = 3;
    m.memoryHeaps[0] = {VkDeviceSize(256) << 20, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT};  // 256 MiB BAR
    m.memoryHeaps[1] = {VkDeviceSize(8) << 30, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT};    // 8 GiB VRAM
    m.memoryHeaps[2] = {VkDeviceSize(16) << 30, 0};                                 // 16 GiB RAM
    m.memoryTypeCount = 3;
    m.memoryTypes[0] = {VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, 0};
    m.memoryTypes[1] = {VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 1};
    m.memoryTypes[2] = {VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 2};
    return m;
}

void assert_all_freed(const MockVk& vk) {
    CHECK(vk.live_set_layouts == 0 && vk.live_pools == 0 && vk.live_pipeline_layouts == 0 && vk.live_pipelines == 0 &&
          vk.live_shaders == 0 && vk.live_images == 0 && vk.live_views == 0 && vk.live_memory == 0);
}

}  // namespace

int main() {
    const VkDevice dev = handle<VkDevice>(1);

    // 1. Missing any required proc (including get_image_reqs, bind_image, barrier, or destroy procs) makes ready()
    //    fail cleanly without crashing.
    for (const char* missing :
         {"vkGetImageMemoryRequirements", "vkBindImageMemory", "vkCmdPipelineBarrier", "vkDestroyImage",
          "vkDestroyImageView", "vkDestroyDescriptorPool", "vkDestroyShaderModule", "vkFreeMemory"}) {
        MockVk vk;
        vk.omit_proc = missing;
        g_vk = &vk;
        leonida::VulkanContext ctx(m_gdpa, dev, false);
        leonida::PresentComposer composer(ctx, non_rebar_memory());
        CHECK(!composer.ready());
        bool changed = true;
        CHECK(!composer.ensure_target({1920, 1080}, changed));
        CHECK(!changed);
        assert_all_freed(vk);
    }

    // 2. Partial pipeline creation failure cleans up already-created set layout and pool immediately.
    {
        MockVk vk;
        vk.fail_create_pipeline_layout = true;
        g_vk = &vk;
        leonida::VulkanContext ctx(m_gdpa, dev, false);
        leonida::PresentComposer composer(ctx, non_rebar_memory());
        CHECK(!composer.ready());
        assert_all_freed(vk);
    }

    // 3. Non-ReBAR: picks pure-VRAM Type 1 (not 256 MiB BAR Type 0); retries on OOM; filters undersized heaps;
    //    refuses 0x0 extent without destroying existing valid target; handles garbage handle on vkCreateImage error.
    {
        MockVk vk;
        g_vk = &vk;
        {
            leonida::VulkanContext ctx(m_gdpa, dev, false);
            leonida::PresentComposer composer(ctx, non_rebar_memory());
            CHECK(composer.ready());

            bool changed = false;
            CHECK(composer.ensure_target({3840, 2160}, changed));
            CHECK(changed && composer.target() != VK_NULL_HANDLE);
            CHECK(vk.alloc_types.size() == 1 && vk.alloc_types[0] == 1u);

            // 0x0 extent must fail without destroying the existing 4K target or invoking vkCreateImage.
            const VkImage saved = composer.target();
            const int image_calls = vk.create_image_calls;
            CHECK(!composer.ensure_target({0, 1080}, changed) && !changed);
            CHECK(!composer.ensure_target({1920, 0}, changed) && !changed);
            CHECK(composer.target() == saved && vk.create_image_calls == image_calls);

            // Pure-VRAM OOM on resize falls back cleanly to the next compatible memory type.
            vk.alloc_types.clear();
            vk.oom_type = 1u;
            CHECK(composer.ensure_target({2560, 1440}, changed) && changed);
            CHECK(vk.alloc_types.size() == 2 && vk.alloc_types[0] == 1u && vk.alloc_types[1] != 1u);

            // Undersized heap filtering: only 256 MiB BAR compatible, 512 MiB requested -> fails with 0 alloc calls.
            vk.alloc_types.clear();
            vk.oom_type = UINT32_MAX;
            vk.memory_type_bits = 0x1u;
            vk.req_size = VkDeviceSize(512) << 20;
            CHECK(!composer.ensure_target({1280, 720}, changed));
            CHECK(vk.alloc_types.empty() && composer.target() == VK_NULL_HANDLE);

            // Garbage handle written on vkCreateImage error must not trigger vkDestroyImage on invalid handle.
            vk.fail_create_image = true;
            vk.garbage_on_image_error = true;
            CHECK(!composer.ensure_target({1920, 1080}, changed));
            CHECK(composer.target() == VK_NULL_HANDLE);
        }
        assert_all_freed(vk);
    }

    std::printf("vk_present_compose_test: PASS\n");
    return 0;
}
