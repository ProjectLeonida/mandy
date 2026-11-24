// Real Vulkan GPU regression test for cross-queue bindless_upload_semaphore_ lifetime in ~VulkanQueue()
// (runtime/src/vulkan_backend.h):
//   1. Queue A executes a submission that publishes a new bindless table generation via context.note_bindless_upload(),
//      setting context_.bindless_upload_semaphore_ = queue_a.progress_.
//   2. Queue A is destroyed (~VulkanQueue waits idle and destroys queue_a.progress_).
//   3. Queue B executes a real GPU CopyBufferRegion and reads back the destination buffer on the host.
//   Without clearing context_.bindless_upload_semaphore_ in ~VulkanQueue(), Queue B passes Queue A's destroyed
//   VkSemaphore handle to vkQueueSubmit() in pWaitSemaphores (use-after-free / VUID-VkSubmitInfo-pWaitSemaphores-parameter).
// Skips with exit code 77 when no Vulkan 1.3 device is available. No game data.
#include "vk_test_device.h"
#include "../src/vulkan_backend.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <string_view>
#include <unordered_set>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); return 1; } \
} while (0)

namespace {

namespace driver {
PFN_vkGetDeviceProcAddr real_get_device = nullptr;
PFN_vkQueueSubmit real_submit = nullptr;
PFN_vkDestroySemaphore real_destroy_semaphore = nullptr;

std::mutex state_mutex;
std::unordered_set<std::uintptr_t> destroyed_semaphores;
std::atomic<std::uint32_t> dangling_wait_hits{0};

VKAPI_ATTR void VKAPI_CALL destroy_semaphore(VkDevice device, VkSemaphore semaphore,
                                             const VkAllocationCallbacks* alloc) {
    if (semaphore) {
        std::lock_guard lock(state_mutex);
        destroyed_semaphores.insert(reinterpret_cast<std::uintptr_t>(semaphore));
    }
    real_destroy_semaphore(device, semaphore, alloc);
}

VKAPI_ATTR VkResult VKAPI_CALL queue_submit(VkQueue queue, std::uint32_t count, const VkSubmitInfo* submits,
                                            VkFence fence) {
    for (std::uint32_t i = 0; i < count; ++i) {
        const VkSubmitInfo& s = submits[i];
        std::lock_guard lock(state_mutex);
        for (std::uint32_t w = 0; w < s.waitSemaphoreCount; ++w) {
            const auto raw = reinterpret_cast<std::uintptr_t>(s.pWaitSemaphores[w]);
            if (destroyed_semaphores.count(raw)) {
                dangling_wait_hits.fetch_add(1, std::memory_order_relaxed);
                // Do not forward a destroyed handle to the ICD, which can crash or hang the driver.
                return VK_ERROR_VALIDATION_FAILED_EXT;
            }
        }
    }
    return real_submit(queue, count, submits, fence);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL get_device(VkDevice device, const char* name) {
    const std::string_view n = name;
    if (n == "vkQueueSubmit") return reinterpret_cast<PFN_vkVoidFunction>(queue_submit);
    if (n == "vkDestroySemaphore") return reinterpret_cast<PFN_vkVoidFunction>(destroy_semaphore);
    return real_get_device(device, name);
}

void install(PFN_vkGetDeviceProcAddr device_proc, VkDevice device) {
    real_get_device = device_proc;
    real_submit = reinterpret_cast<PFN_vkQueueSubmit>(device_proc(device, "vkQueueSubmit"));
    real_destroy_semaphore = reinterpret_cast<PFN_vkDestroySemaphore>(device_proc(device, "vkDestroySemaphore"));
}
}  // namespace driver

}  // namespace

int main() {
    using namespace leonida;

    vktest::Device d;
    if (!d.create(false)) {
        std::fprintf(stderr, "vk_queue_bindless_uaf_test: no Vulkan 1.3 device, skipped\n");
        return 77;
    }
    driver::install(d.device_proc, d.device);

    void* src_ptr = nullptr;
    void* dst_ptr = nullptr;
    const VkBuffer src_buf = d.buffer(64, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &src_ptr);
    const VkBuffer dst_buf = d.buffer(64, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &dst_ptr);
    EXPECT(src_buf && dst_buf && src_ptr && dst_ptr);

    constexpr std::uint32_t kPatternA = 0x11223344u;
    constexpr std::uint32_t kPatternB = 0xDEADBEEFu;
    std::uint32_t init_a[16];
    std::uint32_t init_b[16];
    for (int i = 0; i < 16; ++i) {
        init_a[i] = kPatternA + std::uint32_t(i);
        init_b[i] = kPatternB ^ std::uint32_t(i);
    }
    std::memcpy(src_ptr, init_a, sizeof(init_a));
    std::memset(dst_ptr, 0, sizeof(init_a));

    {
        VulkanContext context(driver::get_device, d.device, true);
        EXPECT(context.loaded());

        ResourceBinding src_res{};
        src_res.buffer = src_buf;
        src_res.size = 64;
        ResourceBinding dst_res{};
        dst_res.buffer = dst_buf;
        dst_res.size = 64;
        dst_res.host_readable = true;
        context.add_resource(1, src_res);
        context.add_resource(2, dst_res);

        bool publish_bindless = true;
        context.set_bindless_prepare([&](VkCommandBuffer) {
            if (publish_bindless) {
                context.note_bindless_upload();
                publish_bindless = false;
            }
            return true;
        });

        auto queue_a = std::make_unique<VulkanQueue>(context, d.queue, d.family);
        VulkanQueue queue_b(context, d.queue, d.family);
        EXPECT(queue_a->ok() && queue_b.ok());

        CommandList copy_list;
        EXPECT(copy_list.copy_buffer({2, 0, 1, 0, 64}));
        EXPECT(copy_list.close());
        const CommandList* lists[] = {&copy_list};

        // 1. Queue A executes with a bindless table upload, publishing queue_a->progress_ to context_.bindless_upload_semaphore_.
        EXPECT(queue_a->execute(lists) == ReplayResult::success);
        EXPECT(context.vk.vkQueueWaitIdle(d.queue) == VK_SUCCESS);
        EXPECT(std::memcmp(dst_ptr, init_a, sizeof(init_a)) == 0);

        // 2. Destroy Queue A (~VulkanQueue waits idle and destroys queue_a->progress_).
        queue_a.reset();

        // 3. Queue B executes a new GPU copy and host readback. It must NOT wait on Queue A's destroyed semaphore.
        std::memcpy(src_ptr, init_b, sizeof(init_b));
        std::memset(dst_ptr, 0x5A, sizeof(init_b));
        const ReplayResult res = queue_b.execute(lists);
        EXPECT(driver::dangling_wait_hits.load() == 0);
        EXPECT(res == ReplayResult::success);
        EXPECT(context.vk.vkQueueWaitIdle(d.queue) == VK_SUCCESS);
        EXPECT(std::memcmp(dst_ptr, init_b, sizeof(init_b)) == 0);
    }

    std::printf("vk_queue_bindless_uaf_test: PASS (0 dangling waits, GPU readback verified)\n");
    return 0;
}
