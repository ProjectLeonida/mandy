// CPU-only unit tests for rt::d3d12::TitleImageUpload (runtime/shims/d3d12_title_image_upload.h, no Vulkan GPU, no
// game files). Covers 64-byte unrolled has_nonzero_bytes() parity, null/overflow guards in write(), real multi-mip
// staging upload with mock Vulkan functions, null vkWaitSemaphores handling in wait_slot(), and full state reset in
// release().
#include "../shims/d3d12_title_image_upload.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <type_traits>
#include <vector>

#define CHECK(cond) do {                                                        \
    if (!(cond)) {                                                              \
        std::fprintf(stderr, "line %d: CHECK failed: %s\n", __LINE__, #cond);   \
        return 1;                                                               \
    }                                                                           \
} while (0)

namespace {

template<typename Handle>
Handle fake_handle(std::uintptr_t v) {
    if constexpr (std::is_pointer_v<Handle>)
        return reinterpret_cast<Handle>(v);
    else
        return static_cast<Handle>(v);
}

template<typename Handle>
std::uintptr_t handle_value(Handle h) {
    if constexpr (std::is_pointer_v<Handle>)
        return reinterpret_cast<std::uintptr_t>(h);
    else
        return static_cast<std::uintptr_t>(h);
}

struct MockState {
    std::vector<std::vector<std::uint8_t>> mapped_buffers;
    std::vector<VkBufferImageCopy> copied_regions;
    std::vector<VkQueue> submitted_queues;
    std::uint64_t sem_counter = 0;
    std::uint64_t last_signaled = 0;
    unsigned pools_created = 0;
    unsigned pools_destroyed = 0;
    unsigned sems_created = 0;
    unsigned sems_destroyed = 0;
    unsigned buffers_created = 0;
    unsigned buffers_destroyed = 0;
    unsigned mems_freed = 0;
    unsigned waits = 0;
    bool auto_complete_sem = true;
};

MockState* g_state = nullptr;

VKAPI_ATTR VkResult VKAPI_CALL m_create_pool(VkDevice, const VkCommandPoolCreateInfo*, const VkAllocationCallbacks*,
                                             VkCommandPool* out) {
    ++g_state->pools_created;
    *out = fake_handle<VkCommandPool>(0x100 + g_state->pools_created);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL m_destroy_pool(VkDevice, VkCommandPool, const VkAllocationCallbacks*) {
    ++g_state->pools_destroyed;
}
VKAPI_ATTR VkResult VKAPI_CALL m_alloc_cb(VkDevice, const VkCommandBufferAllocateInfo* info, VkCommandBuffer* out) {
    for (std::uint32_t i = 0; i < info->commandBufferCount; ++i) out[i] = fake_handle<VkCommandBuffer>(0x200 + i);
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL m_reset_cb(VkCommandBuffer, VkCommandBufferResetFlags) { return VK_SUCCESS; }
VKAPI_ATTR VkResult VKAPI_CALL m_begin_cb(VkCommandBuffer, const VkCommandBufferBeginInfo*) { return VK_SUCCESS; }
VKAPI_ATTR VkResult VKAPI_CALL m_end_cb(VkCommandBuffer) { return VK_SUCCESS; }
VKAPI_ATTR void VKAPI_CALL m_barrier(VkCommandBuffer, VkPipelineStageFlags, VkPipelineStageFlags, VkDependencyFlags,
                                     std::uint32_t, const VkMemoryBarrier*, std::uint32_t,
                                     const VkBufferMemoryBarrier*, std::uint32_t, const VkImageMemoryBarrier*) {}
VKAPI_ATTR void VKAPI_CALL m_copy_buf_img(VkCommandBuffer, VkBuffer, VkImage, VkImageLayout, std::uint32_t count,
                                          const VkBufferImageCopy* regions) {
    g_state->copied_regions.assign(regions, regions + count);
}
VKAPI_ATTR VkResult VKAPI_CALL m_create_sem(VkDevice, const VkSemaphoreCreateInfo*, const VkAllocationCallbacks*,
                                            VkSemaphore* out) {
    ++g_state->sems_created;
    *out = fake_handle<VkSemaphore>(0x300 + g_state->sems_created);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL m_destroy_sem(VkDevice, VkSemaphore, const VkAllocationCallbacks*) {
    ++g_state->sems_destroyed;
}
VKAPI_ATTR VkResult VKAPI_CALL m_get_sem(VkDevice, VkSemaphore, std::uint64_t* value) {
    *value = g_state->sem_counter;
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL m_wait_sem(VkDevice, const VkSemaphoreWaitInfo* info, std::uint64_t) {
    ++g_state->waits;
    if (info && info->semaphoreCount && info->pValues && *info->pValues > g_state->sem_counter)
        g_state->sem_counter = *info->pValues;
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL m_create_buf(VkDevice, const VkBufferCreateInfo* info, const VkAllocationCallbacks*,
                                            VkBuffer* out) {
    ++g_state->buffers_created;
    g_state->mapped_buffers.emplace_back(static_cast<std::size_t>(info->size), std::uint8_t{0});
    *out = fake_handle<VkBuffer>(g_state->mapped_buffers.size());
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL m_destroy_buf(VkDevice, VkBuffer, const VkAllocationCallbacks*) {
    ++g_state->buffers_destroyed;
}
VKAPI_ATTR void VKAPI_CALL m_buf_reqs(VkDevice, VkBuffer, VkMemoryRequirements* req) {
    req->size = 4096;
    req->alignment = 256;
    req->memoryTypeBits = 1;
}
VKAPI_ATTR VkResult VKAPI_CALL m_bind_buf(VkDevice, VkBuffer, VkDeviceMemory, VkDeviceSize) { return VK_SUCCESS; }
VKAPI_ATTR VkResult VKAPI_CALL m_map_mem(VkDevice, VkDeviceMemory mem, VkDeviceSize, VkDeviceSize, VkMemoryMapFlags,
                                         void** data) {
    const std::size_t idx = static_cast<std::size_t>(handle_value(mem)) - 1u;
    *data = g_state->mapped_buffers[idx].data();
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL m_free_mem(VkDevice, VkDeviceMemory, const VkAllocationCallbacks*) { ++g_state->mems_freed; }
VKAPI_ATTR VkResult VKAPI_CALL m_queue_submit(VkQueue q, std::uint32_t count, const VkSubmitInfo* submits, VkFence) {
    g_state->submitted_queues.push_back(q);
    for (std::uint32_t i = 0; i < count; ++i) {
        if (submits[i].signalSemaphoreCount && submits[i].pNext) {
            const auto* tl = static_cast<const VkTimelineSemaphoreSubmitInfo*>(submits[i].pNext);
            if (tl->signalSemaphoreValueCount && tl->pSignalSemaphoreValues) {
                g_state->last_signaled = tl->pSignalSemaphoreValues[0];
                if (g_state->auto_complete_sem) g_state->sem_counter = g_state->last_signaled;
            }
        }
    }
    return VK_SUCCESS;
}

struct MockVk {
    PFN_vkCreateCommandPool vkCreateCommandPool = m_create_pool;
    PFN_vkDestroyCommandPool vkDestroyCommandPool = m_destroy_pool;
    PFN_vkAllocateCommandBuffers vkAllocateCommandBuffers = m_alloc_cb;
    PFN_vkResetCommandBuffer vkResetCommandBuffer = m_reset_cb;
    PFN_vkBeginCommandBuffer vkBeginCommandBuffer = m_begin_cb;
    PFN_vkEndCommandBuffer vkEndCommandBuffer = m_end_cb;
    PFN_vkCmdCopyBufferToImage vkCmdCopyBufferToImage = m_copy_buf_img;
    PFN_vkCreateSemaphore vkCreateSemaphore = m_create_sem;
    PFN_vkDestroySemaphore vkDestroySemaphore = m_destroy_sem;
    PFN_vkGetSemaphoreCounterValue vkGetSemaphoreCounterValue = m_get_sem;
    PFN_vkWaitSemaphores vkWaitSemaphores = m_wait_sem;
    PFN_vkCreateBuffer vkCreateBuffer = m_create_buf;
    PFN_vkDestroyBuffer vkDestroyBuffer = m_destroy_buf;
    PFN_vkGetBufferMemoryRequirements vkGetBufferMemoryRequirements = m_buf_reqs;
    PFN_vkBindBufferMemory vkBindBufferMemory = m_bind_buf;
    PFN_vkMapMemory vkMapMemory = m_map_mem;
    PFN_vkFreeMemory vkFreeMemory = m_free_mem;
    PFN_vkQueueSubmit vkQueueSubmit = m_queue_submit;
};

struct MockContext {
    MockVk vk;
    template<class Fn>
    VkResult write_staged_title_image(leonida::ResourceId, const VkImageSubresourceRange& range, Fn&& fn) {
        leonida::ReplayFunctions replay{};
        replay.pipeline_barrier = m_barrier;
        leonida::LayoutPieces pieces;
        pieces.push_back({range, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL});
        return fn(replay, pieces);
    }
};

struct MockDeviceInfo {
    VkDevice device = fake_handle<VkDevice>(0x10);
    std::uint32_t queue_family = 0;
    VkQueue queue = fake_handle<VkQueue>(0x20);
    std::vector<VkQueue> command_queues{fake_handle<VkQueue>(0x20), fake_handle<VkQueue>(0x21)};
    std::vector<VkQueue> compute_queues{fake_handle<VkQueue>(0x22)};
    std::vector<VkQueue> copy_queues{fake_handle<VkQueue>(0x21)};  // duplicate queue deduped in consumers_
};

struct MockDevice {
    MockContext ctx;
    MockContext* context = &ctx;
    MockDeviceInfo info;
    VkDeviceMemory allocate(const VkMemoryRequirements&, VkMemoryPropertyFlags, VkMemoryAllocateFlags) {
        return fake_handle<VkDeviceMemory>(g_state->mapped_buffers.size());
    }
};

}  // namespace

int main() {
    using namespace rt::d3d12;

    // 1. Exhaustive parity of 64-byte unrolled detail::has_nonzero_bytes() across lengths 0..192 and every offset.
    {
        CHECK(!detail::has_nonzero_bytes(nullptr, 1024));
        for (std::size_t len = 0; len <= 192; ++len) {
            std::vector<std::uint8_t> buf(len, 0);
            CHECK(!detail::has_nonzero_bytes(buf.data(), len));
            for (std::size_t pos = 0; pos < len; ++pos) {
                buf[pos] = 1;
                CHECK(detail::has_nonzero_bytes(buf.data(), len));
                buf[pos] = 0;
            }
        }
    }

    // Build a valid 2-mip linear RGBA8 texture layout (16x16 + 8x8).
    rt::xg_xs::ResourceDesc desc{};
    desc.dimension = rt::xg_xs::kTexture2D;
    desc.width = 16;
    desc.height = 16;
    desc.depth_or_array = 1;
    desc.mips = 2;
    desc.format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.samples = 1;
    desc.layout = rt::xg_xs::kLinear;
    rt::xg_xs::Texture tex{};
    CHECK(rt::xg_xs::build_texture(desc, tex));

    leonida::ResourceBinding img{};
    img.image = fake_handle<VkImage>(0x50);
    img.format = VK_FORMAT_R8G8B8A8_UNORM;
    img.width = 16;
    img.height = 16;
    img.depth = 1;
    img.mip_levels = 2;
    img.array_layers = 1;

    MockState state;
    g_state = &state;
    MockDevice dev;
    TitleImageUpload upload;
    std::vector<std::uint64_t> columns;
    std::vector<std::uint8_t> tiled(static_cast<std::size_t>(tex.layout.size), 0);

    // 2. Null tiled source and oversized slice fail cleanly without allocating Vulkan resources.
    {
        bool nonzero = false;
        CHECK(!upload.write(dev, 1, img, tex, nullptr, columns, true, nonzero));
        CHECK(state.pools_created == 0 && state.buffers_created == 0);

        leonida::ResourceBinding huge_img = img;
        huge_img.width = 0x10000u;
        huge_img.height = 0x10000u;  // 65536 * 4 * 65536 = 16 GiB > UINT32_MAX
        huge_img.mip_levels = 1;
        rt::xg_xs::Texture huge_tex = tex;
        huge_tex.layout.mips = 1;
        huge_tex.layout.plane[0].mips[0].width_elements = 0x10000u;
        huge_tex.layout.plane[0].mips[0].height_elements = 0x10000u;
        CHECK(!upload.write(dev, 1, huge_img, huge_tex, tiled.data(), columns, true, nonzero));
        CHECK(state.pools_created == 0 && state.buffers_created == 0);
    }

    // 3. Valid multi-mip upload detects all-zero vs non-zero tail byte, aligns offsets, and notifies consumer queues.
    {
        bool nonzero = false;
        CHECK(upload.write(dev, 1, img, tex, tiled.data(), columns, true, nonzero));
        CHECK(!nonzero);
        CHECK(state.copied_regions.size() == 2);
        CHECK(state.copied_regions[0].bufferOffset == 0);
        CHECK(state.copied_regions[1].bufferOffset % 16 == 0);
        // Primary queue (0x20) + 2 distinct consumer queues (0x21, 0x22)
        CHECK(state.submitted_queues.size() == 3);

        // Set the very last byte of mip 1 (8x8x4 = 256 bytes into mip 1's linear region) non-zero.
        const std::size_t mip1_off = static_cast<std::size_t>(tex.layout.plane[0].mips[1].offset);
        const std::size_t mip1_pitch = static_cast<std::size_t>(tex.layout.plane[0].mips[1].pitch_bytes);
        tiled[mip1_off + 7 * mip1_pitch + 7 * 4 + 3] = 0x7fu;
        nonzero = false;
        CHECK(upload.write(dev, 1, img, tex, tiled.data(), columns, true, nonzero));
        CHECK(nonzero);
    }

    // 4. Missing vkWaitSemaphores fails cleanly instead of dereferencing null when slot reuse must wait.
    {
        state.auto_complete_sem = false;
        bool nonzero = false;
        // Fill remaining 2 slots so next write wraps back to slot 0 (which will hold an incomplete timeline value).
        CHECK(upload.write(dev, 1, img, tex, tiled.data(), columns, false, nonzero));
        CHECK(upload.write(dev, 1, img, tex, tiled.data(), columns, false, nonzero));
        // Slot 0 now gets a fresh incomplete submission; wrap around 4 slots to hit it again.
        CHECK(upload.write(dev, 1, img, tex, tiled.data(), columns, false, nonzero));
        CHECK(upload.write(dev, 1, img, tex, tiled.data(), columns, false, nonzero));
        CHECK(upload.write(dev, 1, img, tex, tiled.data(), columns, false, nonzero));
        CHECK(upload.write(dev, 1, img, tex, tiled.data(), columns, false, nonzero));
        dev.ctx.vk.vkWaitSemaphores = nullptr;
        CHECK(!upload.write(dev, 1, img, tex, tiled.data(), columns, false, nonzero));
        dev.ctx.vk.vkWaitSemaphores = m_wait_sem;
        state.auto_complete_sem = true;
    }

    // 5. release() destroys all buffers/pool/semaphore and resets internal state so a fresh write() works cleanly.
    {
        upload.release(dev);
        CHECK(upload.capacity() == 0);
        CHECK(state.pools_destroyed == state.pools_created);
        CHECK(state.sems_destroyed == state.sems_created);
        CHECK(state.buffers_destroyed == state.buffers_created);
        CHECK(state.mems_freed == state.buffers_created);

        state.submitted_queues.clear();
        state.sem_counter = 0;
        bool nonzero = false;
        CHECK(upload.write(dev, 1, img, tex, tiled.data(), columns, true, nonzero));
        CHECK(nonzero);
        // consumers_ was cleared on release(), so no duplicate queue waits (still 1 primary + 2 consumers = 3).
        CHECK(state.submitted_queues.size() == 3);
        upload.release(dev);
        CHECK(upload.capacity() == 0);
    }

    std::printf("title_image_upload_test: all checks passed\n");
    return 0;
}
