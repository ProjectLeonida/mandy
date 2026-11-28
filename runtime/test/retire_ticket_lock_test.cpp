// VulkanContext::completed_retire_ticket() (src/vulkan_backend.h) reads the retirement records under their own
// lock (retire_mutex_), so a caller that only asks what retired does not wait behind a submission that holds the
// submit lock (a queue submission, or a vkDeviceWaitIdle under it). CPU only: a fake device whose device-function
// lookup answers null is enough, and with no recorded submission the retirement query calls no Vulkan entry point.
#include "../src/vulkan_backend.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <thread>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

using namespace leonida;
using namespace std::chrono_literals;

template<class T>
T handle(std::uintptr_t n) {
    return reinterpret_cast<T>(n);
}

// No device: every function resolves to null, which is enough for a context with no resources or submissions.
PFN_vkVoidFunction VKAPI_CALL no_device_functions(VkDevice, const char*) { return nullptr; }

std::int64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Another thread holds the submit lock for 500 ms; the retirement read must return at once with ticket 0.
int completed_ticket_does_not_wait_for_the_submit_lock() {
    VulkanContext context(no_device_functions, handle<VkDevice>(1), false);
    std::atomic<bool> holding{false};
    std::thread holder([&] {
        std::lock_guard lock(context.submit_mutex());
        holding.store(true, std::memory_order_release);
        std::this_thread::sleep_for(500ms);
    });
    while (!holding.load(std::memory_order_acquire)) std::this_thread::yield();

    const std::int64_t start = now_ms();
    const std::uint64_t ticket = context.completed_retire_ticket();
    const std::int64_t waited = now_ms() - start;
    holder.join();

    std::printf("completed_retire_ticket returned %llums after a submit lock was held (ticket %llu)\n",
                static_cast<long long>(waited), static_cast<unsigned long long>(ticket));
    std::fflush(stdout);      // before a diagnostic the EXPECT may print to stderr
    EXPECT(ticket == 0);
    EXPECT(waited < 200);
    return 0;
}

}  // namespace

int main() {
    if (completed_ticket_does_not_wait_for_the_submit_lock() != 0) return 1;
    std::printf("retire_ticket_lock_test: passed\n");
    return 0;
}
