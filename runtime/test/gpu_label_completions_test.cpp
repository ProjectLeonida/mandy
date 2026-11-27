// leonida::label::Completions (runtime/src/vulkan_gpu_labels.h): what a GPU label records into a command buffer and
// how it is stored once the command buffer's fence signals. The Vulkan device is a mock reached through the
// PFN_vkGetDeviceProcAddr the class is given; the title's memory is a tracked range whose stores land in a stand-in for
// xmem's title-pages writer (value_sink::set_title_pages). CPU only: no Vulkan device, no GPU. No game data.
#include "../src/vulkan_gpu_labels.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace {

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const char* what) {
    ++g_checks;
    std::printf("%s %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition) ++g_failures;
}

namespace label = leonida::label;

// The mock device. Handles are counters; nothing is dereferenced.
struct MockDevice {
    std::mutex mutex;
    std::uintptr_t next_handle = 0x1000;
    std::map<VkFence, bool> signaled;
    int pools_created = 0, resets = 0, timestamps = 0, results_calls = 0;
    std::vector<std::uint32_t> timestamp_queries;  // the query index of each vkCmdWriteTimestamp
    VkQueryResultFlags last_flags = 0;
    bool not_ready_without_wait = true;  // a driver whose query results are not yet available when the fence signals
} g_dev;

const VkDevice kDevice = reinterpret_cast<VkDevice>(static_cast<std::uintptr_t>(0xD0));

template <class Handle> Handle make_handle() {
    std::lock_guard lock(g_dev.mutex);
    return reinterpret_cast<Handle>(g_dev.next_handle++);
}

VKAPI_ATTR VkResult VKAPI_CALL create_pool(VkDevice, const VkQueryPoolCreateInfo* info, const VkAllocationCallbacks*,
                                           VkQueryPool* pool) {
    if (info->queryType != VK_QUERY_TYPE_TIMESTAMP) return VK_ERROR_INITIALIZATION_FAILED;
    *pool = make_handle<VkQueryPool>();
    std::lock_guard lock(g_dev.mutex);
    ++g_dev.pools_created;
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL destroy_pool(VkDevice, VkQueryPool, const VkAllocationCallbacks*) {}
VKAPI_ATTR void VKAPI_CALL reset_query(VkCommandBuffer, VkQueryPool, std::uint32_t, std::uint32_t) {
    std::lock_guard lock(g_dev.mutex);
    ++g_dev.resets;
}
VKAPI_ATTR void VKAPI_CALL write_timestamp(VkCommandBuffer, VkPipelineStageFlagBits, VkQueryPool, std::uint32_t query) {
    std::lock_guard lock(g_dev.mutex);
    ++g_dev.timestamps;
    g_dev.timestamp_queries.push_back(query);
}
// A timestamp reads back as 0x7000 plus its query index, so the test sees which query a label read.
VKAPI_ATTR VkResult VKAPI_CALL query_results(VkDevice, VkQueryPool, std::uint32_t first, std::uint32_t count,
                                             std::size_t size, void* data, VkDeviceSize, VkQueryResultFlags flags) {
    std::lock_guard lock(g_dev.mutex);
    ++g_dev.results_calls;
    g_dev.last_flags = flags;
    if (count != 1 || size < 8 || !(flags & VK_QUERY_RESULT_64_BIT)) return VK_ERROR_UNKNOWN;
    if (g_dev.not_ready_without_wait && !(flags & VK_QUERY_RESULT_WAIT_BIT)) return VK_NOT_READY;
    const std::uint64_t value = 0x7000 + first;
    std::memcpy(data, &value, sizeof(value));
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL create_fence(VkDevice, const VkFenceCreateInfo*, const VkAllocationCallbacks*,
                                            VkFence* fence) {
    *fence = make_handle<VkFence>();
    std::lock_guard lock(g_dev.mutex);
    g_dev.signaled[*fence] = false;
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL destroy_fence(VkDevice, VkFence, const VkAllocationCallbacks*) {}
VKAPI_ATTR VkResult VKAPI_CALL fence_status(VkDevice, VkFence fence) {
    std::lock_guard lock(g_dev.mutex);
    return g_dev.signaled[fence] ? VK_SUCCESS : VK_NOT_READY;
}
VKAPI_ATTR VkResult VKAPI_CALL reset_fences(VkDevice, std::uint32_t count, const VkFence* fences) {
    std::lock_guard lock(g_dev.mutex);
    for (std::uint32_t i = 0; i < count; ++i)
        g_dev.signaled[fences[i]] = false;
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL device_idle(VkDevice) { return VK_SUCCESS; }

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL get_proc(VkDevice, const char* name) {
#define MOCK(fn, impl) \
    if (!std::strcmp(name, #fn)) return reinterpret_cast<PFN_vkVoidFunction>(&impl);
    MOCK(vkCreateQueryPool, create_pool)
    MOCK(vkDestroyQueryPool, destroy_pool)
    MOCK(vkCmdResetQueryPool, reset_query)
    MOCK(vkCmdWriteTimestamp, write_timestamp)
    MOCK(vkGetQueryPoolResults, query_results)
    MOCK(vkCreateFence, create_fence)
    MOCK(vkDestroyFence, destroy_fence)
    MOCK(vkGetFenceStatus, fence_status)
    MOCK(vkResetFences, reset_fences)
    MOCK(vkDeviceWaitIdle, device_idle)
#undef MOCK
    return nullptr;
}

// The title's memory: a tracked range that is never dereferenced; its stores land in the map below, through the
// stand-in for xmem's title-pages writer.
constexpr std::uintptr_t kTitleBase = 0x40000000;
std::mutex g_sink_mutex;
std::map<std::uintptr_t, std::uint64_t> g_sink;

// Set while a test wants to know whether a title thread can use the Completions while a label is being stored:
// the real writer takes xmem's lock (and kernel_writes the layer's), which a title thread may hold while it records.
label::Completions* g_probe = nullptr;
std::atomic<int> g_probe_tries{0}, g_probe_blocked{0};

bool title_pages(const void* address, const void* bytes, std::size_t size) {
    if (label::Completions* const c = g_probe) {
        ++g_probe_tries;
        // The writer waits for the title thread as xmem's lock would; if the Completions lock is held across the store,
        // that thread cannot finish while we wait (a deadlock in the runtime), so it is let go and finishes later.
        auto done = std::make_shared<std::atomic<bool>>(false);
        std::thread title([c, done] {
            (void)c->submit_fence(reinterpret_cast<VkCommandBuffer>(static_cast<std::uintptr_t>(0xEE)));
            *done = true;
        });
        const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(1);
        while (!*done && std::chrono::steady_clock::now() < until)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        if (*done) {
            title.join();
        } else {
            ++g_probe_blocked;
            title.detach();
        }
    }
    std::uint64_t value = 0;
    std::memcpy(&value, bytes, size);
    std::lock_guard lock(g_sink_mutex);
    g_sink[reinterpret_cast<std::uintptr_t>(address)] = value;
    return true;
}

label::Write write_at(std::uintptr_t offset, unsigned clock, std::uint64_t value = 0) {
    label::Write w;
    w.target = label::memory().capture(kTitleBase + offset, 8);
    w.value = value;
    w.clock = clock;
    return w;
}

bool sink_has(std::uintptr_t offset, std::uint64_t* value) {
    std::lock_guard lock(g_sink_mutex);
    const auto i = g_sink.find(kTitleBase + offset);
    if (i == g_sink.end()) return false;
    if (value) *value = i->second;
    return true;
}

// Signals the command buffer's fence, hands it to the worker and waits until all of `offsets` are stored (or 5 s).
bool complete(label::Completions& c, VkCommandBuffer cb, std::initializer_list<std::uintptr_t> offsets) {
    const VkFence fence = c.submit_fence(cb);
    if (!fence) return false;
    {
        std::lock_guard lock(g_dev.mutex);
        g_dev.signaled[fence] = true;
    }
    c.submitted(cb, true);
    const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    for (;;) {
        bool all = true;
        for (auto offset : offsets)
            all = all && sink_has(offset, nullptr);
        if (all) return true;
        if (std::chrono::steady_clock::now() >= until) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void reset_counts() {
    std::lock_guard lock(g_dev.mutex);
    g_dev.resets = g_dev.timestamps = g_dev.results_calls = 0;
    g_dev.timestamp_queries.clear();
    g_dev.last_flags = 0;
}

}  // namespace

int main() {
    _putenv_s("LEONIDA_GPU_LABELS", "1");
    check(label::enabled(), "LEONIDA_GPU_LABELS=1 turns the labels on");
    label::memory().track(kTitleBase, 0x10000, kTitleBase, false);
    leonida::value_sink::set_title_pages(&title_pages);

    label::Completions c;
    check(c.initialize(kDevice, &get_proc), "the completions start on the mock device");
    const auto cb = reinterpret_cast<VkCommandBuffer>(static_cast<std::uintptr_t>(0xC0));

    // 1. A supplied value (clock 0) and a system timestamp (clock 2) record nothing into the command buffer: no query
    //    reset, no vkCmdWriteTimestamp (which a queue without timestamp bits must not record), no query pool.
    c.begin_record(cb);
    c.record(cb, write_at(0x00, 0, 0x1234));
    c.record(cb, write_at(0x08, 2));
    {
        std::lock_guard lock(g_dev.mutex);
        check(g_dev.resets == 0 && g_dev.timestamps == 0,
              "a supplied value and a system timestamp record no query reset and no timestamp write");
        check(g_dev.pools_created == 0, "and take no query pool");
    }
    std::uint64_t value = 0;
    check(complete(c, cb, {0x00, 0x08}), "both labels are stored once the fence signals");
    check(sink_has(0x00, &value) && value == 0x1234, "the supplied value is the one stored");
    check(sink_has(0x08, &value) && value != 0, "the system timestamp is a nonzero counter");
    {
        std::lock_guard lock(g_dev.mutex);
        check(g_dev.results_calls == 0, "no query result is read for them");
    }

    // 2. GPU timestamps between supplied values: each takes the next query of the batch, and the result is read with
    //    WAIT_BIT, so a driver that has not made it available yet when the fence signals does not drop the label.
    reset_counts();
    c.begin_record(cb);
    c.record(cb, write_at(0x10, 1));
    c.record(cb, write_at(0x18, 0, 0x55));
    c.record(cb, write_at(0x20, 1));
    {
        std::lock_guard lock(g_dev.mutex);
        check(g_dev.resets == 2 && g_dev.timestamps == 2, "two GPU timestamps record two resets and two writes");
        check(g_dev.timestamp_queries == std::vector<std::uint32_t>{0, 1},
              "they use queries 0 and 1; the supplied value between them takes none");
    }
    check(complete(c, cb, {0x10, 0x18, 0x20}), "a GPU timestamp whose result is not ready without WAIT_BIT is stored");
    {
        std::lock_guard lock(g_dev.mutex);
        check((g_dev.last_flags & VK_QUERY_RESULT_WAIT_BIT) && (g_dev.last_flags & VK_QUERY_RESULT_64_BIT),
              "the query result is read with WAIT_BIT and 64_BIT");
    }
    check(sink_has(0x10, &value) && value == 0x7000, "the first timestamp label holds query 0's result");
    check(sink_has(0x18, &value) && value == 0x55, "the supplied value between them is stored");
    check(sink_has(0x20, &value) && value == 0x7001, "the second timestamp label holds query 1's result");

    // 3. 70 supplied values and one GPU timestamp: one query, one pool, still read correctly.
    reset_counts();
    const int pools_before = g_dev.pools_created;
    c.begin_record(cb);
    for (std::uintptr_t i = 0; i < 70; ++i)
        c.record(cb, write_at(0x100 + 8 * i, 0, i + 1));
    c.record(cb, write_at(0x800, 1));
    {
        std::lock_guard lock(g_dev.mutex);
        check(g_dev.timestamps == 1 && g_dev.timestamp_queries == std::vector<std::uint32_t>{0},
              "70 supplied values and one GPU timestamp record one timestamp, at query 0");
        check(g_dev.pools_created - pools_before <= 1, "and need at most one query pool");
    }
    check(complete(c, cb, {0x100, 0x100 + 8 * 69, 0x800}), "all 71 labels are stored");
    check(sink_has(0x100 + 8 * 69, &value) && value == 70, "the 70th supplied value is stored");
    check(sink_has(0x800, &value) && value == 0x7000, "the timestamp after them holds query 0's result");

    // 4. While a label is being stored the Completions lock is free: the store takes other locks (xmem's, the
    //    layer's shadow openers), and a title thread holding one of them may be recording into the Completions.
    g_probe_tries = g_probe_blocked = 0;
    g_probe = &c;
    c.begin_record(cb);
    c.record(cb, write_at(0x900, 0, 9));
    const bool stored = complete(c, cb, {0x900});
    g_probe = nullptr;
    check(stored && g_probe_tries == 1, "the label is stored through the title-pages writer");
    check(g_probe_blocked == 0, "a title thread can use the Completions while a label is stored (no lock held)");

    c.stop(kDevice);
    leonida::value_sink::set_title_pages(nullptr);
    std::printf("gpu_label_completions_test: %s (%d checks, %d failures)\n", g_failures ? "FAILED" : "ok", g_checks,
                g_failures);
    return g_failures ? 1 : 0;
}
