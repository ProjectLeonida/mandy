// A title texture upload (VulkanContext::write_staged_title_image) never waits for the GPU on the CPU: it hands the
// transfer every queue timeline value submitted so far, to wait for on the GPU, and keeps other submissions out while
// it reads the layouts and submits. Until 2026-11 it waited on the CPU for that work for at most 250 ms and skipped the
// write when the work was not done.
#include "../src/vulkan_backend.h"

#include <cstdio>
#include <thread>
#include <utility>
#include <vector>

namespace {
unsigned waits = 0;
VkResult VKAPI_CALL wait_semaphores(VkDevice, const VkSemaphoreWaitInfo*, std::uint64_t) {
    ++waits;
    return VK_TIMEOUT;
}
VkResult VKAPI_CALL device_idle(VkDevice) { return VK_ERROR_UNKNOWN; }  // must not be called
PFN_vkVoidFunction VKAPI_CALL no_proc(VkDevice, const char*) { return nullptr; }
}

int main() {
    leonida::VulkanContext ctx(no_proc, VK_NULL_HANDLE, false);
    ctx.vk.vkWaitSemaphores = wait_semaphores;
    ctx.vk.vkDeviceWaitIdle = device_idle;
    leonida::ResourceBinding binding{};
    binding.image = reinterpret_cast<VkImage>(1);
    binding.mip_levels = 1;
    binding.array_layers = 1;
    ctx.add_resource(1, binding, VK_IMAGE_LAYOUT_GENERAL);
    const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    unsigned copies = 0;
    std::vector<std::pair<VkSemaphore, std::uint64_t>> seen;
    const auto copy = [&](const leonida::ReplayFunctions&, const leonida::LayoutPieces& pieces, const auto& handed) {
        ++copies;
        seen.assign(handed.begin(), handed.end());
        bool raced = false;
        std::thread contender([&] {
            if (ctx.submit_mutex().try_lock()) {
                raced = true;
                ctx.submit_mutex().unlock();
            }
        });
        contender.join();
        if (raced || pieces.empty() || pieces[0].layout != VK_IMAGE_LAYOUT_GENERAL) return VK_ERROR_UNKNOWN;
        return VK_SUCCESS;
    };
    // Nothing submitted: nothing to wait for.
    if (ctx.write_staged_title_image(1, range, copy) != VK_SUCCESS || copies != 1 || !seen.empty()) return 1;
    const auto graphics = reinterpret_cast<VkSemaphore>(std::uintptr_t{0x10});
    const auto compute = reinterpret_cast<VkSemaphore>(std::uintptr_t{0x20});
    {  // work in flight on two queues, which a CPU wait would time out on (wait_semaphores says VK_TIMEOUT)
        std::lock_guard lock(ctx.submit_mutex());
        ctx.record_submit_locked(graphics, 7);
        ctx.record_submit_locked(compute, 3);
    }
    if (ctx.write_staged_title_image(1, range, copy) != VK_SUCCESS || copies != 2) return 2;  // not skipped
    if (waits) return 3;  // no CPU wait
    bool g = false, c = false;
    for (const auto& [semaphore, value] : seen) {
        g |= semaphore == graphics && value == 7;
        c |= semaphore == compute && value == 3;
    }
    if (!g || !c) return 4;  // the transfer waits for both queues' work on the GPU
    if (ctx.write_staged_title_image(2, range, copy) != VK_ERROR_UNKNOWN || copies != 2) return 5;  // unknown image
    std::puts("title texture upload: no CPU wait, no skip, GPU waits for every queue's submitted work, submit exclusion");
}
