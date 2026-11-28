// Unit and concurrency tests for GraphicsShaderModules (src/vulkan_graphics_shader_modules.h), no GPU needed.
#define VK_NO_PROTOTYPES
#include "../src/vulkan_graphics_shader_modules.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <thread>
#include <type_traits>
#include <vector>

#define CHECK(cond) do {                                             \
    if (!(cond)) {                                                   \
        std::fprintf(stderr, "failed at line %d: %s\n", __LINE__, #cond); \
        return 1;                                                    \
    }                                                                \
} while (false)

namespace {

template <typename Handle>
Handle fake_handle(std::uintptr_t value) {
    if constexpr (std::is_pointer_v<Handle>)
        return reinterpret_cast<Handle>(value);
    else
        return static_cast<Handle>(value);
}

std::atomic<std::uintptr_t> g_next_handle{1};
std::atomic<unsigned> g_created{0};
std::atomic<unsigned> g_destroyed{0};
std::atomic<bool> g_fail_create{false};
std::atomic<bool> g_return_null_handle{false};

// Concurrency hooks for duplicate-create race and destroy-outside-lock proof.
std::mutex g_race_mutex;
std::condition_variable g_race_cv;
std::atomic<bool> g_barrier_create{false};
unsigned g_in_create_count = 0;
bool g_release_create = false;

std::atomic<bool> g_pause_destroy{false};
bool g_in_destroy = false;
bool g_release_destroy = false;

VKAPI_ATTR VkResult VKAPI_CALL mock_create(VkDevice, const VkShaderModuleCreateInfo* info,
                                           const VkAllocationCallbacks*, VkShaderModule* out) {
    if (!info || !info->pCode || !info->codeSize || !out) return VK_ERROR_INITIALIZATION_FAILED;
    if (g_barrier_create.load(std::memory_order_acquire)) {
        std::unique_lock lock(g_race_mutex);
        ++g_in_create_count;
        g_race_cv.notify_all();
        g_race_cv.wait_for(lock, std::chrono::seconds(2), [] { return g_release_create; });
    }
    if (g_fail_create.load(std::memory_order_acquire)) {
        *out = fake_handle<VkShaderModule>(0xdead);  // test that failure ignores driver garbage
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    if (g_return_null_handle.load(std::memory_order_acquire)) {
        *out = VK_NULL_HANDLE;
        return VK_SUCCESS;
    }
    ++g_created;
    *out = fake_handle<VkShaderModule>(g_next_handle.fetch_add(1, std::memory_order_relaxed));
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL mock_destroy(VkDevice, VkShaderModule, const VkAllocationCallbacks*) {
    if (g_pause_destroy.load(std::memory_order_acquire)) {
        std::unique_lock lock(g_race_mutex);
        if (!g_in_destroy) {
            g_in_destroy = true;
            g_race_cv.notify_all();
            g_race_cv.wait_for(lock, std::chrono::seconds(2), [] { return g_release_destroy; });
        }
    }
    ++g_destroyed;
}

}  // namespace

int main() {
    leonida::GraphicsShaderModules cache;
    const VkDevice dev_a = fake_handle<VkDevice>(0x100);
    const VkDevice dev_b = fake_handle<VkDevice>(0x200);
    const std::vector<std::uint32_t> spv_a{0x07230203u, 0x00010000u, 0, 4, 0, 111};
    const std::vector<std::uint32_t> spv_b{0x07230203u, 0x00010000u, 0, 4, 0, 222};

    // 1. Guard empty input, null procs, create errors, and VK_SUCCESS with VK_NULL_HANDLE.
    {
        const unsigned created_before = g_created.load();
        const unsigned destroyed_before = g_destroyed.load();
        CHECK(!cache.acquire(dev_a, mock_create, mock_destroy, {}));
        CHECK(!cache.acquire(dev_a, nullptr, mock_destroy, spv_a));
        CHECK(!cache.acquire(dev_a, mock_create, nullptr, spv_a));
        g_fail_create.store(true, std::memory_order_release);
        CHECK(!cache.acquire(dev_a, mock_create, mock_destroy, spv_a));
        g_fail_create.store(false, std::memory_order_release);
        g_return_null_handle.store(true, std::memory_order_release);
        CHECK(!cache.acquire(dev_a, mock_create, mock_destroy, spv_a));
        g_return_null_handle.store(false, std::memory_order_release);
        CHECK(g_created.load() == created_before);
        CHECK(g_destroyed.load() == destroyed_before);
    }

    // 2. Deduplication on the same device, distinct modules for distinct SPIR-V, and expiration/re-creation.
    {
        auto m1 = cache.acquire(dev_a, mock_create, mock_destroy, spv_a);
        auto m2 = cache.acquire(dev_a, mock_create, mock_destroy, spv_a);
        auto m3 = cache.acquire(dev_a, mock_create, mock_destroy, spv_b);
        CHECK(m1 && m2 && m3);
        CHECK(m1.get() == m2.get() && m1->handle == m2->handle);
        CHECK(m1->handle != m3->handle);
        CHECK(g_created.load() == 2 && g_destroyed.load() == 0);

        // 3. Cross-device isolation: identical SPIR-V on a different VkDevice must not alias dev_a's handle.
        auto m_other = cache.acquire(dev_b, mock_create, mock_destroy, spv_a);
        CHECK(m_other && m_other->device == dev_b && m_other->handle != m1->handle);
        CHECK(g_created.load() == 3);

        m1.reset();
        m2.reset();
        CHECK(g_destroyed.load() == 1);
        auto m1_rebuilt = cache.acquire(dev_a, mock_create, mock_destroy, spv_a);
        CHECK(m1_rebuilt && m1_rebuilt->device == dev_a && m1_rebuilt->handle != m_other->handle);
        CHECK(g_created.load() == 4);
        m1_rebuilt.reset();
        m3.reset();
        m_other.reset();
        CHECK(g_destroyed.load() == 4);
    }

    // 4. Concurrency proof: losing duplicate module's vkDestroyShaderModule must run OUTSIDE the cache lock.
    // On unpatched code, returning `found` while holding `lock_guard` invokes `mock_destroy` under `mutex_`,
    // blocking the concurrent `cache.acquire(..., spv_b)` call below until timeout.
    {
        const unsigned created_before = g_created.load();
        const unsigned destroyed_before = g_destroyed.load();
        {
            std::lock_guard lk(g_race_mutex);
            g_in_create_count = 0;
            g_release_create = false;
            g_in_destroy = false;
            g_release_destroy = false;
        }
        g_barrier_create.store(true, std::memory_order_release);
        g_pause_destroy.store(true, std::memory_order_release);

        leonida::GraphicsShaderModules::Reference r1, r2;
        std::thread t1([&] { r1 = cache.acquire(dev_a, mock_create, mock_destroy, spv_a); });
        std::thread t2([&] { r2 = cache.acquire(dev_a, mock_create, mock_destroy, spv_a); });
        {
            std::unique_lock lk(g_race_mutex);
            CHECK(g_race_cv.wait_for(lk, std::chrono::seconds(2), [] { return g_in_create_count == 2; }));
            g_release_create = true;
        }
        g_race_cv.notify_all();

        // Wait until the losing racer enters mock_destroy for its redundant VkShaderModule.
        {
            std::unique_lock lk(g_race_mutex);
            CHECK(g_race_cv.wait_for(lk, std::chrono::seconds(2), [] { return g_in_destroy; }));
        }
        // Turn off create barrier so a third acquire can run immediately if the cache lock is not held!
        g_barrier_create.store(false, std::memory_order_release);
        std::atomic<bool> probe_finished{false};
        leonida::GraphicsShaderModules::Reference probe_ref;
        std::thread probe([&] {
            probe_ref = cache.acquire(dev_a, mock_create, mock_destroy, spv_b);
            probe_finished.store(true, std::memory_order_release);
        });
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
        while (!probe_finished.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        const bool unlocked_during_destroy = probe_finished.load(std::memory_order_acquire);
        {
            std::lock_guard lk(g_race_mutex);
            g_release_destroy = true;
        }
        g_race_cv.notify_all();
        t1.join();
        t2.join();
        probe.join();
        g_pause_destroy.store(false, std::memory_order_release);

        CHECK(unlocked_during_destroy);
        CHECK(r1 && r2 && r1.get() == r2.get() && probe_ref);
        CHECK(g_created.load() == created_before + 3);    // 2 racers + 1 probe
        CHECK(g_destroyed.load() == destroyed_before + 1); // losing duplicate destroyed
        r1.reset();
        r2.reset();
        probe_ref.reset();
        CHECK(g_destroyed.load() == destroyed_before + 3);
    }

    // 5. Multi-threaded TSan stress across shared SPIR-V modules, two devices, and churn.
    {
        const unsigned created_before = g_created.load();
        const unsigned destroyed_before = g_destroyed.load();
        std::vector<std::vector<std::uint32_t>> corpus;
        for (std::uint32_t i = 0; i < 8; ++i)
            corpus.push_back({0x07230203u, 0x00010000u, 0, 8, 0, i, i * 17u + 3u});
        std::vector<std::thread> workers;
        for (int t = 0; t < 4; ++t) {
            workers.emplace_back([&, t] {
                std::vector<leonida::GraphicsShaderModules::Reference> held;
                for (int i = 0; i < 200; ++i) {
                    const VkDevice dev = ((i + t) & 1) ? dev_b : dev_a;
                    const auto& spv = corpus[static_cast<std::size_t>((i * 5 + t) % corpus.size())];
                    auto mod = cache.acquire(dev, mock_create, mock_destroy, spv);
                    if (!mod || mod->device != dev || mod->handle == VK_NULL_HANDLE) std::abort();
                    held.push_back(std::move(mod));
                    if (held.size() > 6) held.erase(held.begin());
                }
            });
        }
        for (auto& w : workers) w.join();
        CHECK(g_created.load() > created_before);
        CHECK(g_destroyed.load() - destroyed_before == g_created.load() - created_before);
    }

    std::puts("vulkan_graphics_shader_modules: dedup, cross-device isolation, guards, unlock-on-destroy and TSan stress OK");
    return 0;
}
