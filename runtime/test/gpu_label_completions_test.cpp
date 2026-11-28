// Pure-CPU mock-Vulkan tests for leonida::label::Completions (runtime/src/vulkan_gpu_labels.h):
//   - publish() runs without Completions::mutex_ held (no AB-BA deadlock with value_sink::arena -> resource_mutex_);
//   - timestamp labels (clock == 1) request VK_QUERY_RESULT_WAIT_BIT so deferred writeback is never dropped;
//   - immediate labels (clock == 0) skip query pool creation and vkCmdWriteTimestamp;
//   - mixed clock=0 / clock=1 labels spanning > 64 timestamp queries map dense query slots accurately;
//   - stop() drains batches signaled during vkDeviceWaitIdle and is safe under concurrent stop() + record() (TSan).
// No GPU, Vulkan loader, or game files required.
#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include "../src/vulkan_gpu_labels.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#define CHECK(cond)                                                                       \
    do {                                                                                  \
        if (!(cond)) {                                                                    \
            std::fprintf(stderr, "line %d: CHECK failed: %s\n", __LINE__, #cond);         \
            return 1;                                                                     \
        }                                                                                 \
    } while (0)

namespace {

struct MockVk {
    std::mutex mutex;
    std::uint64_t next_handle = 1;
    int create_pool_calls = 0;
    int destroy_pool_calls = 0;
    int reset_query_calls = 0;
    int write_timestamp_calls = 0;
    int create_fence_calls = 0;
    int destroy_fence_calls = 0;
    int device_idle_calls = 0;
    bool require_wait_bit = true;
    bool signal_all_on_device_idle = false;
    std::unordered_set<std::uintptr_t> live_pools;
    std::unordered_set<std::uintptr_t> live_fences;
    std::unordered_set<std::uintptr_t> signaled_fences;
    // Key: (pool_handle << 16) | query_index -> timestamp written on command buffer.
    std::unordered_map<std::uint64_t, std::uint64_t> query_values;
    std::uint64_t next_timestamp = 0x1000000ull;

    void signal_fence(VkFence fence) {
        std::lock_guard lock(mutex);
        signaled_fences.insert(reinterpret_cast<std::uintptr_t>(fence));
    }
};

MockVk* g_vk = nullptr;

template <typename Handle>
Handle make_handle(std::uint64_t id) {
    return reinterpret_cast<Handle>(static_cast<std::uintptr_t>(id));
}

VKAPI_ATTR VkResult VKAPI_CALL mock_create_query_pool(VkDevice, const VkQueryPoolCreateInfo*,
                                                      const VkAllocationCallbacks*, VkQueryPool* out) {
    std::lock_guard lock(g_vk->mutex);
    ++g_vk->create_pool_calls;
    const auto id = ++g_vk->next_handle;
    g_vk->live_pools.insert(static_cast<std::uintptr_t>(id));
    *out = make_handle<VkQueryPool>(id);
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL mock_destroy_query_pool(VkDevice, VkQueryPool pool, const VkAllocationCallbacks*) {
    std::lock_guard lock(g_vk->mutex);
    ++g_vk->destroy_pool_calls;
    g_vk->live_pools.erase(reinterpret_cast<std::uintptr_t>(pool));
}

VKAPI_ATTR void VKAPI_CALL mock_cmd_reset_query_pool(VkCommandBuffer, VkQueryPool, std::uint32_t, std::uint32_t) {
    std::lock_guard lock(g_vk->mutex);
    ++g_vk->reset_query_calls;
}

VKAPI_ATTR void VKAPI_CALL mock_cmd_write_timestamp(VkCommandBuffer, VkPipelineStageFlagBits, VkQueryPool pool,
                                                    std::uint32_t query) {
    std::lock_guard lock(g_vk->mutex);
    ++g_vk->write_timestamp_calls;
    const std::uint64_t key = (static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(pool)) << 16) | query;
    g_vk->query_values[key] = ++g_vk->next_timestamp;
}

VKAPI_ATTR VkResult VKAPI_CALL mock_get_query_pool_results(VkDevice, VkQueryPool pool, std::uint32_t first_query,
                                                           std::uint32_t query_count, std::size_t data_size, void* data,
                                                           VkDeviceSize, VkQueryResultFlags flags) {
    std::lock_guard lock(g_vk->mutex);
    if (g_vk->require_wait_bit && (flags & VK_QUERY_RESULT_WAIT_BIT) == 0) return VK_NOT_READY;
    if ((flags & VK_QUERY_RESULT_64_BIT) == 0 || query_count != 1 || data_size < sizeof(std::uint64_t))
        return VK_ERROR_UNKNOWN;
    const std::uint64_t key = (static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(pool)) << 16) | first_query;
    const auto it = g_vk->query_values.find(key);
    if (it == g_vk->query_values.end()) return VK_NOT_READY;
    const std::uint64_t val = it->second;
    std::memcpy(data, &val, sizeof(val));
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL mock_create_fence(VkDevice, const VkFenceCreateInfo*, const VkAllocationCallbacks*,
                                                 VkFence* out) {
    std::lock_guard lock(g_vk->mutex);
    ++g_vk->create_fence_calls;
    const auto id = ++g_vk->next_handle;
    g_vk->live_fences.insert(static_cast<std::uintptr_t>(id));
    *out = make_handle<VkFence>(id);
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL mock_destroy_fence(VkDevice, VkFence fence, const VkAllocationCallbacks*) {
    std::lock_guard lock(g_vk->mutex);
    ++g_vk->destroy_fence_calls;
    const auto key = reinterpret_cast<std::uintptr_t>(fence);
    g_vk->live_fences.erase(key);
    g_vk->signaled_fences.erase(key);
}

VKAPI_ATTR VkResult VKAPI_CALL mock_get_fence_status(VkDevice, VkFence fence) {
    std::lock_guard lock(g_vk->mutex);
    return g_vk->signaled_fences.count(reinterpret_cast<std::uintptr_t>(fence)) ? VK_SUCCESS : VK_NOT_READY;
}

VKAPI_ATTR VkResult VKAPI_CALL mock_reset_fences(VkDevice, std::uint32_t count, const VkFence* fences) {
    std::lock_guard lock(g_vk->mutex);
    for (std::uint32_t i = 0; i < count; ++i)
        g_vk->signaled_fences.erase(reinterpret_cast<std::uintptr_t>(fences[i]));
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL mock_device_wait_idle(VkDevice) {
    std::lock_guard lock(g_vk->mutex);
    ++g_vk->device_idle_calls;
    if (g_vk->signal_all_on_device_idle) {
        for (const auto f : g_vk->live_fences) g_vk->signaled_fences.insert(f);
    }
    return VK_SUCCESS;
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL mock_gdpa(VkDevice, const char* name) {
#define ENTRY(fn_name, impl) \
    if (std::strcmp(name, #fn_name) == 0) return reinterpret_cast<PFN_vkVoidFunction>(impl)
    ENTRY(vkCreateQueryPool, mock_create_query_pool);
    ENTRY(vkDestroyQueryPool, mock_destroy_query_pool);
    ENTRY(vkCmdResetQueryPool, mock_cmd_reset_query_pool);
    ENTRY(vkCmdWriteTimestamp, mock_cmd_write_timestamp);
    ENTRY(vkGetQueryPoolResults, mock_get_query_pool_results);
    ENTRY(vkCreateFence, mock_create_fence);
    ENTRY(vkDestroyFence, mock_destroy_fence);
    ENTRY(vkGetFenceStatus, mock_get_fence_status);
    ENTRY(vkResetFences, mock_reset_fences);
    ENTRY(vkDeviceWaitIdle, mock_device_wait_idle);
#undef ENTRY
    return nullptr;
}

void enable_gpu_labels_env() {
#ifdef _WIN32
    _putenv_s("LEONIDA_GPU_LABELS", "1");
#else
    setenv("LEONIDA_GPU_LABELS", "1", 1);
#endif
}

struct ArenaSinkState {
    std::mutex mutex;
    std::condition_variable cv;
    std::unordered_map<std::uint64_t, std::uint64_t> values;
    int writes = 0;
    // Optional lock-inversion hook for Test 1.
    std::shared_mutex* resource_mutex = nullptr;
    std::atomic<bool>* publish_entered = nullptr;
    std::atomic<bool>* release_publish = nullptr;
};

leonida::value_sink::Landing test_arena_sink(void* ctx, std::uint64_t va, const void* bytes, unsigned size) {
    auto* st = static_cast<ArenaSinkState*>(ctx);
    if (st->publish_entered) {
        st->publish_entered->store(true, std::memory_order_release);
        while (st->release_publish && !st->release_publish->load(std::memory_order_acquire))
            std::this_thread::yield();
    }
    // Mirrors VulkanContext::is_sparse_va(): acquires resource_mutex_ shared inside write_arena_value().
    std::shared_lock<std::shared_mutex> resource_guard;
    if (st->resource_mutex) resource_guard = std::shared_lock(*st->resource_mutex);
    std::uint64_t val = 0;
    std::memcpy(&val, bytes, size);
    {
        std::lock_guard lock(st->mutex);
        st->values[va] = val;
        ++st->writes;
    }
    st->cv.notify_all();
    return leonida::value_sink::Landing::landed;
}

bool wait_for_writes(ArenaSinkState& st, int target_writes, int timeout_ms = 3000) {
    std::unique_lock lock(st.mutex);
    return st.cv.wait_for(lock, std::chrono::milliseconds(timeout_ms), [&] { return st.writes >= target_writes; });
}

}  // namespace

int main() {
    enable_gpu_labels_env();
    using namespace leonida;
    const VkDevice dummy_dev = make_handle<VkDevice>(1);
    constexpr std::uintptr_t kArenaBase = 0x10000;
    constexpr std::uintptr_t kArenaSize = 0x10000;
    label::memory().track(kArenaBase, kArenaSize, kArenaBase, false);

    // 1. Lock-order deadlock regression: publish() must NOT hold Completions::mutex_ across store_value().
    //    Reproduces the production 3-thread cycle:
    //      - Worker thread runs publish() -> value_sink::arena() -> shared_lock(resource_mutex)
    //      - Writer thread requests unique_lock(resource_mutex) (blocks new readers)
    //      - Submitter thread holds shared_lock(resource_mutex) and calls completions.record() (takes mutex_)
    {
        MockVk vk;
        g_vk = &vk;
        ArenaSinkState arena;
        std::shared_mutex resource_mutex;
        std::atomic<bool> publish_entered{false};
        std::atomic<bool> release_publish{false};
        arena.resource_mutex = &resource_mutex;
        arena.publish_entered = &publish_entered;
        arena.release_publish = &release_publish;
        value_sink::set_arena(&test_arena_sink, &arena);

        label::Completions completions;
        CHECK(completions.initialize(dummy_dev, mock_gdpa));

        const VkCommandBuffer cb1 = make_handle<VkCommandBuffer>(101);
        const VkCommandBuffer cb2 = make_handle<VkCommandBuffer>(102);
        const auto t1 = label::memory().capture(kArenaBase + 0x100, 8);
        const auto t2 = label::memory().capture(kArenaBase + 0x108, 8);
        CHECK(t1.bytes == 8 && t2.bytes == 8);

        completions.begin_record(cb1);
        completions.record(cb1, label::Write{t1, 0xCAFEBABE12345678ull, 0u});
        const VkFence f1 = completions.submit_fence(cb1);
        CHECK(f1 != VK_NULL_HANDLE);
        vk.signal_fence(f1);
        completions.submitted(cb1, true);

        // Wait until the worker thread is inside store_value() -> test_arena_sink() before acquiring resource_mutex.
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
        while (!publish_entered.load(std::memory_order_acquire)) {
            CHECK(std::chrono::steady_clock::now() < deadline);
            std::this_thread::yield();
        }

        // Hold resource_mutex shared (like VulkanQueue::execute()) and start a blocked unique_lock writer
        // (like resource creation), then call completions.record() while still holding shared_lock.
        std::atomic<bool> writer_done{false};
        std::thread writer;
        {
            std::shared_lock<std::shared_mutex> exec_lock(resource_mutex);
            writer = std::thread([&] {
                std::unique_lock<std::shared_mutex> create_lock(resource_mutex);
                writer_done.store(true, std::memory_order_release);
            });
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            // Let the worker proceed to acquire shared_lock(resource_mutex) (blocks behind writer on writer-preferring
            // shared_mutex implementations). On main, worker still held Completions::mutex_, so record() deadlocked.
            release_publish.store(true, std::memory_order_release);
            std::this_thread::sleep_for(std::chrono::milliseconds(10));

            completions.begin_record(cb2);
            completions.record(cb2, label::Write{t2, 0x0102030405060708ull, 0u});
            const VkFence f2 = completions.submit_fence(cb2);
            CHECK(f2 != VK_NULL_HANDLE);
            vk.signal_fence(f2);
            completions.submitted(cb2, true);
        }
        writer.join();
        CHECK(writer_done.load(std::memory_order_acquire));
        CHECK(wait_for_writes(arena, 2));
        CHECK(arena.values.at(kArenaBase + 0x100) == 0xCAFEBABE12345678ull);
        CHECK(arena.values.at(kArenaBase + 0x108) == 0x0102030405060708ull);

        completions.stop(dummy_dev);
        value_sink::clear_arena(&arena);
        CHECK(vk.live_fences.empty() && vk.live_pools.empty());
        CHECK(vk.create_fence_calls == vk.destroy_fence_calls);
        CHECK(vk.create_pool_calls == 0 && vk.write_timestamp_calls == 0);
    }

    // 2. Immediate labels (clock == 0) skip query pools & timestamps; GPU timestamp labels (clock == 1) pass
    //    VK_QUERY_RESULT_WAIT_BIT and index dense query slots cleanly across multiple 64-query pools.
    {
        MockVk vk;
        vk.require_wait_bit = true;
        g_vk = &vk;
        ArenaSinkState arena;
        value_sink::set_arena(&test_arena_sink, &arena);

        label::Completions completions;
        CHECK(completions.initialize(dummy_dev, mock_gdpa));
        const VkCommandBuffer cb = make_handle<VkCommandBuffer>(201);
        completions.begin_record(cb);

        // Interleave 70 clock=0 immediate writes and 70 clock=1 timestamp writes (spans 2 query pools of 64).
        for (std::uint32_t i = 0; i < 70; ++i) {
            const auto imm_target = label::memory().capture(kArenaBase + 0x1000 + i * 8, 8);
            const auto ts_target = label::memory().capture(kArenaBase + 0x2000 + i * 8, 8);
            completions.record(cb, label::Write{imm_target, 0xA000ull + i, 0u});
            completions.record(cb, label::Write{ts_target, 0u, 1u});
        }
        CHECK(vk.create_pool_calls == 2);       // 70 timestamp queries -> ceil(70/64) = 2 pools (not 3 for 140 labels!)
        CHECK(vk.reset_query_calls == 70);
        CHECK(vk.write_timestamp_calls == 70);

        const VkFence f = completions.submit_fence(cb);
        CHECK(f != VK_NULL_HANDLE);
        vk.signal_fence(f);
        completions.submitted(cb, true);

        CHECK(wait_for_writes(arena, 140));
        for (std::uint32_t i = 0; i < 70; ++i) {
            CHECK(arena.values.at(kArenaBase + 0x1000 + i * 8) == 0xA000ull + i);
            CHECK(arena.values.at(kArenaBase + 0x2000 + i * 8) == 0x1000001ull + i);
        }

        completions.stop(dummy_dev);
        value_sink::clear_arena(&arena);
        CHECK(vk.live_fences.empty() && vk.live_pools.empty());
        CHECK(vk.create_pool_calls == vk.destroy_pool_calls);
        CHECK(vk.create_fence_calls == vk.destroy_fence_calls);
    }

    // 3. Shutdown drain & concurrent stop() safety under TSan:
    //    - A pending batch whose fence only signals during vkDeviceWaitIdle() is still published before stop() returns.
    //    - Concurrent stop(dev) callers wait cleanly without double-join or double-destroy.
    {
        MockVk vk;
        vk.signal_all_on_device_idle = true;
        g_vk = &vk;
        ArenaSinkState arena;
        value_sink::set_arena(&test_arena_sink, &arena);

        label::Completions completions;
        CHECK(completions.initialize(dummy_dev, mock_gdpa));
        const VkCommandBuffer cb1 = make_handle<VkCommandBuffer>(301);
        const VkCommandBuffer cb2 = make_handle<VkCommandBuffer>(302);
        const auto t1 = label::memory().capture(kArenaBase + 0x3000, 8);
        const auto t2 = label::memory().capture(kArenaBase + 0x3008, 8);

        completions.begin_record(cb1);
        completions.record(cb1, label::Write{t1, 0x55AA55AA55AA55AAull, 0u});
        CHECK(completions.submit_fence(cb1) != VK_NULL_HANDLE);
        // Leave cb1's fence unsignaled until stop() calls vkDeviceWaitIdle().
        completions.submitted(cb1, true);

        // Give the worker a chance to observe VK_NOT_READY once before stop() runs.
        std::this_thread::sleep_for(std::chrono::milliseconds(5));

        std::atomic<bool> stop_started{false};
        std::thread stopper1([&] {
            stop_started.store(true, std::memory_order_release);
            completions.stop(dummy_dev);
        });
        std::thread stopper2([&] {
            while (!stop_started.load(std::memory_order_acquire)) std::this_thread::yield();
            completions.stop(dummy_dev);
        });
        std::thread late_recorder([&] {
            while (!stop_started.load(std::memory_order_acquire)) std::this_thread::yield();
            completions.begin_record(cb2);
            completions.record(cb2, label::Write{t2, 0x99u, 0u});
            (void)completions.submit_fence(cb2);
            completions.submitted(cb2, false);
        });

        stopper1.join();
        stopper2.join();
        late_recorder.join();

        // Batch 1 signaled during vkDeviceWaitIdle() must have been drained and published.
        CHECK(arena.values.count(kArenaBase + 0x3000) == 1);
        CHECK(arena.values.at(kArenaBase + 0x3000) == 0x55AA55AA55AA55AAull);
        CHECK(vk.device_idle_calls == 1);
        CHECK(vk.live_fences.empty() && vk.live_pools.empty());
        CHECK(vk.create_fence_calls == vk.destroy_fence_calls);
        CHECK(vk.create_pool_calls == vk.destroy_pool_calls);

        value_sink::clear_arena(&arena);
    }

    std::printf("gpu_label_completions_test: all checks passed\n");
    return 0;
}
