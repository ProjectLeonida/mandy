// Real Vulkan GPU regression test for D3D12 Enhanced TextureBarrier with
// layout_before == UNDEFINED and layout_after == UNDEFINED (no-layout-change sync barrier)
// (runtime/src/vulkan_replay.h):
//   In D3D12 Enhanced Barriers, UNDEFINED -> UNDEFINED on a texture barrier requests an execution/memory
//   dependency without changing the texture's layout (accepted by CommandList::barrier()).
//   Previously, emit() treated it as a discard transition to newLayout = VK_IMAGE_LAYOUT_UNDEFINED, which:
//     1. Violated VUID-VkImageMemoryBarrier2-newLayout-01198 (newLayout must not be UNDEFINED),
//     2. Overwrote LayoutTracker to VK_IMAGE_LAYOUT_UNDEFINED, causing the next copy/draw to re-run
//        detail::initialize() and wipe the texture's contents to zeros on the GPU.
// Skips with exit code 77 when no Vulkan 1.3 device is available. No game data.
#include "vk_test_device.h"
#include "../src/vulkan_backend.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string_view>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); return 1; } \
} while (0)

namespace {

namespace driver {
PFN_vkGetDeviceProcAddr real_get_device = nullptr;
PFN_vkCmdPipelineBarrier2 real_barrier2 = nullptr;
std::atomic<std::uint32_t> illegal_undef_new_layout{0};

VKAPI_ATTR void VKAPI_CALL cmd_barrier2(VkCommandBuffer cb, const VkDependencyInfo* dep) {
    if (dep) {
        for (std::uint32_t i = 0; i < dep->imageMemoryBarrierCount; ++i) {
            if (dep->pImageMemoryBarriers[i].newLayout == VK_IMAGE_LAYOUT_UNDEFINED) {
                illegal_undef_new_layout.fetch_add(1, std::memory_order_relaxed);
                // Avoid passing an illegal newLayout=UNDEFINED barrier into the ICD.
                return;
            }
        }
    }
    real_barrier2(cb, dep);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL get_device(VkDevice device, const char* name) {
    if (std::string_view(name) == "vkCmdPipelineBarrier2")
        return reinterpret_cast<PFN_vkVoidFunction>(cmd_barrier2);
    return real_get_device(device, name);
}

void install(PFN_vkGetDeviceProcAddr device_proc, VkDevice device) {
    real_get_device = device_proc;
    real_barrier2 = reinterpret_cast<PFN_vkCmdPipelineBarrier2>(device_proc(device, "vkCmdPipelineBarrier2"));
}
}  // namespace driver

}  // namespace

int main() {
    using namespace leonida;

    vktest::Device d;
    if (!d.create(false)) {
        std::fprintf(stderr, "vk_texture_barrier_undef_noop_test: no Vulkan 1.3 device, skipped\n");
        return 77;
    }
    driver::install(d.device_proc, d.device);
    EXPECT(driver::real_barrier2 != nullptr);

    constexpr std::uint32_t kW = 4, kH = 4;
    constexpr std::uint32_t kRowPitch = 256;
    const VkImage tex = d.image(VK_FORMAT_R8G8B8A8_UNORM, kW, kH, 1, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
    void* readback_ptr = nullptr;
    const VkBuffer readback_buf = d.buffer(kRowPitch * kH, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &readback_ptr);
    EXPECT(tex && readback_buf && readback_ptr);
    std::memset(readback_ptr, 0x5A, kRowPitch * kH);

    {
        VulkanContext context(driver::get_device, d.device, true);
        EXPECT(context.loaded());
        VulkanQueue queue(context, d.queue, d.family);
        EXPECT(queue.ok());

        ResourceBinding tex_res{};
        tex_res.image = tex;
        tex_res.format = VK_FORMAT_R8G8B8A8_UNORM;
        tex_res.aspects = VK_IMAGE_ASPECT_COLOR_BIT;
        tex_res.width = kW;
        tex_res.height = kH;
        tex_res.color_attachment = true;
        context.add_resource(1, tex_res, VK_IMAGE_LAYOUT_UNDEFINED);

        ResourceBinding buf_res{};
        buf_res.buffer = readback_buf;
        buf_res.size = kRowPitch * kH;
        buf_res.host_readable = true;
        context.add_resource(2, buf_res);

        // 1. Clear the texture to bright magenta (R=255, G=0, B=255, A=255 -> 0xFFFF00FF in little-endian RGBA8).
        // 2. Issue an Enhanced TextureBarrier with UNDEFINED -> UNDEFINED (sync-only, no layout change).
        // 3. Copy the texture to the readback buffer and verify every pixel is still magenta (not wiped to 0).
        CommandList list;
        EXPECT(list.clear_render_target({1, 0, 0, 1}, {1.0f, 0.0f, 1.0f, 1.0f}));

        TextureBarrier tb{};
        tb.resource = 1;
        tb.sync_before = d3d12_sync::render_target;
        tb.sync_after = d3d12_sync::copy;
        tb.access_before = d3d12_access::render_target;
        tb.access_after = d3d12_access::copy_source;
        tb.layout_before = d3d12_layout::undefined;
        tb.layout_after = d3d12_layout::undefined;
        tb.range = {all_subresources, 0, 0, 0, 0, 0};
        EXPECT(list.barrier(tb));

        CopyTexture copy{};
        copy.src = {1, false, 0};
        copy.dst = {2, true, 0, 0, kW, kH, 1, kRowPitch};
        EXPECT(list.copy_texture(copy));
        EXPECT(list.close());

        const CommandList* lists[] = {&list};
        driver::illegal_undef_new_layout.store(0);
        EXPECT(queue.execute(lists) == ReplayResult::success);
        EXPECT(context.vk.vkQueueWaitIdle(d.queue) == VK_SUCCESS);

        EXPECT(driver::illegal_undef_new_layout.load() == 0);
        EXPECT(context.layout(1, 0) != VK_IMAGE_LAYOUT_UNDEFINED);

        const auto* bytes = static_cast<const std::uint8_t*>(readback_ptr);
        for (std::uint32_t y = 0; y < kH; ++y) {
            for (std::uint32_t x = 0; x < kW; ++x) {
                const std::uint8_t* px = bytes + y * kRowPitch + x * 4;
                EXPECT(px[0] == 255 && px[1] == 0 && px[2] == 255 && px[3] == 255);
            }
        }
    }

    std::printf("vk_texture_barrier_undef_noop_test: PASS (0 illegal UNDEFINED newLayout, 16/16 magenta pixels preserved)\n");
    return 0;
}
