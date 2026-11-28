// The context's deferred deletion queue after a device loss (runtime/src/vulkan_backend.h,
// VulkanContext::defer_destroy). Once the device is lost no retire ticket completes again and acquire() returns before
// its drain, so an object queued then would stay until the context goes, and the queue would grow with every object
// the title releases.
//   1. Buffers deferred before the loss wait for a submission that never comes.
//   2. After the loss, the next deferred destruction frees them and every later one is freed at once: all 1000 buffers
//      are gone while the context lives, and the context's own teardown destroys none of them twice.
//   3. A bindless slot, queued before the loss or after it, still waits in the queue (its release takes the device's
//      descriptor lock, which the caller holds) and is released once, at teardown.
// vkDestroyBuffer is interposed on a real device's table to count the destructions. Skips with exit code 77 when no
// Vulkan 1.3 device is available; LEONIDA_VK_VALIDATION=1 runs it under the validation layer. No game data.
#include "vk_test_device.h"
#include "../src/vulkan_backend.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string_view>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); std::_Exit(1); } \
} while (0)

namespace {

namespace driver {
PFN_vkGetDeviceProcAddr real_get_device = nullptr;
PFN_vkDestroyBuffer real_destroy_buffer = nullptr;

std::atomic<unsigned> destroyed_buffers{0};

VKAPI_ATTR void VKAPI_CALL destroy_buffer(VkDevice device, VkBuffer buffer, const VkAllocationCallbacks* allocator) {
    if (buffer) destroyed_buffers.fetch_add(1, std::memory_order_relaxed);
    real_destroy_buffer(device, buffer, allocator);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL get_device(VkDevice device, const char* name) {
    if (std::string_view(name) == "vkDestroyBuffer") return reinterpret_cast<PFN_vkVoidFunction>(destroy_buffer);
    return real_get_device(device, name);
}

void install(PFN_vkGetDeviceProcAddr device_proc, VkDevice device) {
    real_get_device = device_proc;
    real_destroy_buffer = reinterpret_cast<PFN_vkDestroyBuffer>(device_proc(device, "vkDestroyBuffer"));
}
}  // namespace driver

}  // namespace

int main() {
    using namespace leonida;

    vktest::Device d;
    if (!d.create(false)) {
        std::fprintf(stderr, "vk_deferred_destroy_lost_test: no Vulkan 1.3 device, skipped\n");
        return 77;
    }
    driver::install(d.device_proc, d.device);
    const auto create_buffer = d.fn<PFN_vkCreateBuffer>("vkCreateBuffer");
    EXPECT(create_buffer);
    const auto make_buffer = [&] {
        VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        info.size = 256;
        info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VkBuffer buffer = VK_NULL_HANDLE;
        EXPECT(create_buffer(d.device, &info, nullptr, &buffer) == VK_SUCCESS && buffer);
        return buffer;
    };

    constexpr unsigned before_loss = 500;
    constexpr unsigned total = 1000;
    unsigned released_slots = 0;
    {
        VulkanContext context(driver::get_device, d.device, true);
        EXPECT(context.loaded());
        context.set_bindless_release(
            [](void* user, std::uint32_t, std::uint32_t) { ++*static_cast<unsigned*>(user); }, &released_slots);

        // 1, with a bindless slot among the buffers
        for (unsigned i = 0; i < before_loss; ++i) {
            context.defer_destroy(DeferredDeletionQueue::Kind::buffer, make_buffer());
            if (i == before_loss / 2)
                context.defer_destroy(DeferredDeletionQueue::Kind::bindless_slot, (std::uint64_t(1) << 32) | 3u);
        }
        EXPECT(driver::destroyed_buffers.load() == 0);

        // 2
        context.mark_device_lost("the test marks the device lost");
        for (unsigned i = before_loss; i < total; ++i)
            context.defer_destroy(DeferredDeletionQueue::Kind::buffer, make_buffer());
        EXPECT(driver::destroyed_buffers.load() == total);

        // 3
        context.defer_destroy(DeferredDeletionQueue::Kind::bindless_slot, (std::uint64_t(1) << 32) | 7u);
        EXPECT(released_slots == 0);
    }
    EXPECT(driver::destroyed_buffers.load() == total);
    EXPECT(released_slots == 2);
    EXPECT(d.validation_errors() == 0);
    std::puts("vk_deferred_destroy_lost: ok");
    return 0;
}
