// Small optimal images share memory until their deferred image destruction has run.
#define VK_NO_PROTOTYPES
#include "../src/vulkan_title_image_pool.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <mutex>
#include <thread>
#include <vector>

namespace {
std::atomic<unsigned> allocated{0}, freed{0}, destroyed{0};
unsigned next_type;
bool fail_alloc, fail_bind;
VkDeviceSize last_offset;

// Optional hook invoked during bind_image for concurrency testing.
std::mutex hook_mutex;
std::condition_variable hook_cv;
std::atomic<bool> pause_in_bind{false};
bool entered_bind = false;
bool resume_bind = false;

VKAPI_ATTR VkResult VKAPI_CALL bind_image(VkDevice, VkImage, VkDeviceMemory, VkDeviceSize offset) {
    last_offset = offset;
    if (pause_in_bind.load(std::memory_order_acquire)) {
        std::unique_lock lock(hook_mutex);
        if (pause_in_bind.load(std::memory_order_relaxed) && !entered_bind) {
            entered_bind = true;
            hook_cv.notify_all();
            hook_cv.wait_for(lock, std::chrono::seconds(2), [] { return resume_bind; });
        }
    }
    return fail_bind ? VK_ERROR_OUT_OF_DEVICE_MEMORY : VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL noop_bind_image(VkDevice, VkImage, VkDeviceMemory, VkDeviceSize) { return VK_SUCCESS; }
VKAPI_ATTR void VKAPI_CALL destroy_image(VkDevice, VkImage, const VkAllocationCallbacks*) { ++destroyed; }
VKAPI_ATTR void VKAPI_CALL free_memory(VkDevice, VkDeviceMemory, const VkAllocationCallbacks*) { ++freed; }

VKAPI_ATTR VkResult VKAPI_CALL stress_bind_image(VkDevice, VkImage, VkDeviceMemory, VkDeviceSize offset) {
    // Reject every 17th bind to exercise slot return and fresh-page cleanup under contention.
    return (offset / 4096) % 17 == 16 ? VK_ERROR_OUT_OF_DEVICE_MEMORY : VK_SUCCESS;
}
#define CHECK(cond) do { if (!(cond)) { std::fprintf(stderr, "failed at line %d: %s\n", __LINE__, #cond); return 1; } } while (false)
}

int main() {
    leonida::TitleImagePool pool;
    VkMemoryRequirements req{800, 2048, 1};
    unsigned next_image = 0;
    const auto make = [&] {
        const auto image = (VkImage)(std::uintptr_t(++next_image));
        return pool.bind(VK_NULL_HANDLE, image, req, bind_image, destroy_image, free_memory,
                         [](VkDeviceSize, std::uint32_t& type) {
                             type = next_type;
                             return fail_alloc ? VkDeviceMemory{} : (VkDeviceMemory)(std::uintptr_t(++allocated));
                         });
    };
    auto a = make();
    CHECK(a && last_offset == 0);
    auto b = make();
    CHECK(b && last_offset == 2048 && allocated == 1 && b->page == a->page);
    // Simulate removing the first image from replay, then draining its deferred deletion later.
    auto retired = std::move(a);
    auto c = make();
    CHECK(c && last_offset == 4096 && destroyed == 0);
    retired.reset();
    CHECK(destroyed == 1 && freed == 0);
    fail_bind = true;
    CHECK(!make() && last_offset == 0 && destroyed == 1);
    fail_bind = false;
    a = make();
    CHECK(a && last_offset == 0 && allocated == 1);
    req.memoryTypeBits = 2;
    next_type = 1;
    auto other = make();
    CHECK(other && other->page != a->page && allocated == 2);
    b.reset(); c.reset(); a.reset(); other.reset();
    CHECK(freed == 2 && destroyed == 5);

    req.size = 2u << 20;
    std::vector<std::unique_ptr<leonida::PooledTitleImage>> live;
    for (int i = 0; i < 5; ++i) live.push_back(make());
    CHECK(live.front() && live.back() && allocated == 4 && last_offset == 0);
    CHECK(live[0]->page == live[3]->page && live[4]->page != live[3]->page);
    live.clear();
    CHECK(freed == 4 && destroyed == 10);
    fail_alloc = true;
    CHECK(!make() && allocated == 4 && freed == 4);
    fail_alloc = false;
    req.size++;
    CHECK(!make() && allocated == 4);
    req = {701440, 512, 2};
    a = make(); b = make();
    CHECK(a && b && a->page == b->page && last_offset % req.alignment == 0);
    CHECK(last_offset >= req.size && last_offset <= req.size + req.size / 8);
    a.reset(); b.reset();
    CHECK(freed == 5 && destroyed == 12);

    // Guard null handles/procs and non-power-of-two alignments without calling allocate.
    {
        const unsigned alloc_before = allocated.load();
        const VkImage valid_img = (VkImage)(std::uintptr_t(++next_image));
        const VkMemoryRequirements good_req{4096, 4096, 1};
        const auto no_alloc = [](VkDeviceSize, std::uint32_t& type) -> VkDeviceMemory {
            type = 0;
            return (VkDeviceMemory)(std::uintptr_t(++allocated));
        };
        CHECK(!pool.bind(VK_NULL_HANDLE, VK_NULL_HANDLE, good_req, bind_image, destroy_image, free_memory, no_alloc));
        CHECK(!pool.bind(VK_NULL_HANDLE, valid_img, good_req, nullptr, destroy_image, free_memory, no_alloc));
        CHECK(!pool.bind(VK_NULL_HANDLE, valid_img, good_req, bind_image, nullptr, free_memory, no_alloc));
        CHECK(!pool.bind(VK_NULL_HANDLE, valid_img, good_req, bind_image, destroy_image, nullptr, no_alloc));
        CHECK(!pool.bind(VK_NULL_HANDLE, valid_img, VkMemoryRequirements{4096, 3, 1},
                         bind_image, destroy_image, free_memory, no_alloc));
        CHECK(!pool.bind(VK_NULL_HANDLE, valid_img, VkMemoryRequirements{4096, 1000, 1},
                         bind_image, destroy_image, free_memory, no_alloc));
        CHECK(!pool.bind(VK_NULL_HANDLE, valid_img, VkMemoryRequirements{0, 0, 1},
                         bind_image, destroy_image, free_memory, no_alloc));
        CHECK(!pool.bind(VK_NULL_HANDLE, valid_img, VkMemoryRequirements{4096, 4096, 0},
                         bind_image, destroy_image, free_memory, no_alloc));
        CHECK(allocated.load() == alloc_before);
    }

    // Out-of-range (>= 32) or incompatible memory type returned by allocate() must fail and free memory (no UB/leak).
    {
        const unsigned alloc_before = allocated.load();
        const unsigned freed_before = freed.load();
        const unsigned destroyed_before = destroyed.load();
        req = {4096, 4096, 2u};  // only memory type 1 allowed
        next_type = 32;          // out-of-range: would shift UB on (1u << 32)
        CHECK(!make());
        CHECK(allocated.load() == alloc_before + 1 && freed.load() == freed_before + 1);
        next_type = 0;           // incompatible with memoryTypeBits == 2
        CHECK(!make());
        CHECK(allocated.load() == alloc_before + 2 && freed.load() == freed_before + 2);
        CHECK(destroyed.load() == destroyed_before);
        // Subsequent valid allocation succeeds on a fresh page (neither bad page was published).
        next_type = 1;
        auto ok = make();
        CHECK(ok && ok->page->type == 1 && allocated.load() == alloc_before + 3);
        ok.reset();
        CHECK(freed.load() == freed_before + 3 && destroyed.load() == destroyed_before + 1);
    }

    // Failed bind_image on a fresh page must immediately free the page without publishing it.
    {
        const unsigned alloc_before = allocated.load();
        const unsigned freed_before = freed.load();
        req = {8192, 4096, 1u};
        next_type = 0;
        fail_bind = true;
        CHECK(!make());
        CHECK(allocated.load() == alloc_before + 1 && freed.load() == freed_before + 1);
        fail_bind = false;
    }

    // Concurrency proof 1: mutex_ is NOT held across allocate() — a second thread completes bind() while
    // thread 1 is paused inside allocate(). (Holding mutex_ across allocate() would stall/deadlock here.)
    {
        std::mutex m;
        std::condition_variable cv;
        bool in_allocate = false;
        bool other_finished = false;
        std::unique_ptr<leonida::PooledTitleImage> first_img;
        std::thread t1([&] {
            const VkMemoryRequirements r1{16384, 4096, 1u};
            const auto img1 = (VkImage)(std::uintptr_t(10001));
            first_img = pool.bind(VK_NULL_HANDLE, img1, r1, noop_bind_image, destroy_image, free_memory,
                                  [&](VkDeviceSize, std::uint32_t& type) {
                                      type = 0;
                                      std::unique_lock lk(m);
                                      in_allocate = true;
                                      cv.notify_all();
                                      cv.wait_for(lk, std::chrono::seconds(2), [&] { return other_finished; });
                                      return (VkDeviceMemory)(std::uintptr_t(++allocated));
                                  });
        });
        {
            std::unique_lock lk(m);
            CHECK(cv.wait_for(lk, std::chrono::seconds(2), [&] { return in_allocate; }));
        }
        const VkMemoryRequirements r2{32768, 4096, 1u};
        const auto img2 = (VkImage)(std::uintptr_t(10002));
        auto second_img = pool.bind(VK_NULL_HANDLE, img2, r2, noop_bind_image, destroy_image, free_memory,
                                    [](VkDeviceSize, std::uint32_t& type) {
                                        type = 0;
                                        return (VkDeviceMemory)(std::uintptr_t(++allocated));
                                    });
        CHECK(second_img != nullptr);
        {
            std::lock_guard lk(m);
            other_finished = true;
        }
        cv.notify_all();
        t1.join();
        CHECK(first_img != nullptr);
        first_img.reset();
        second_img.reset();
    }

    // Concurrency proof 2: mutex_ is NOT held across bind_image() — a second thread completes bind() while
    // thread 1 is paused inside bind_image().
    {
        {
            std::lock_guard lk(hook_mutex);
            entered_bind = false;
            resume_bind = false;
        }
        pause_in_bind.store(true, std::memory_order_release);
        std::unique_ptr<leonida::PooledTitleImage> first_img;
        std::thread t1([&] {
            const VkMemoryRequirements r1{65536, 4096, 1u};
            const auto img1 = (VkImage)(std::uintptr_t(20001));
            first_img = pool.bind(VK_NULL_HANDLE, img1, r1, bind_image, destroy_image, free_memory,
                                  [](VkDeviceSize, std::uint32_t& type) {
                                      type = 0;
                                      return (VkDeviceMemory)(std::uintptr_t(++allocated));
                                  });
        });
        {
            std::unique_lock lk(hook_mutex);
            CHECK(hook_cv.wait_for(lk, std::chrono::seconds(2), [] { return entered_bind; }));
        }
        const VkMemoryRequirements r2{65536, 4096, 1u};
        const auto img2 = (VkImage)(std::uintptr_t(20002));
        auto second_img = pool.bind(VK_NULL_HANDLE, img2, r2, noop_bind_image, destroy_image, free_memory,
                                    [](VkDeviceSize, std::uint32_t& type) {
                                        type = 0;
                                        return (VkDeviceMemory)(std::uintptr_t(++allocated));
                                    });
        CHECK(second_img != nullptr);
        {
            std::lock_guard lk(hook_mutex);
            resume_bind = true;
        }
        hook_cv.notify_all();
        t1.join();
        pause_in_bind.store(false, std::memory_order_release);
        CHECK(first_img != nullptr);
        first_img.reset();
        second_img.reset();
    }

    // Multi-threaded TSan stress: concurrent binds, full-page lock-free skips, retirements, and bind failures.
    {
        const unsigned alloc_before = allocated.load();
        const unsigned freed_before = freed.load();
        const unsigned destroyed_before = destroyed.load();
        std::atomic<unsigned> succeeded{0};
        std::atomic<std::uintptr_t> img_id{30000};
        std::vector<std::thread> workers;
        for (int t = 0; t < 4; ++t) {
            workers.emplace_back([&, t] {
                std::vector<std::unique_ptr<leonida::PooledTitleImage>> held;
                held.reserve(32);
                const VkDeviceSize sizes[3] = {4096, 65536, 512u << 10};
                for (int i = 0; i < 120; ++i) {
                    const std::uint32_t mtype = static_cast<std::uint32_t>((i + t) & 1);
                    const VkMemoryRequirements r{sizes[(i + t) % 3], 4096, 1u << mtype};
                    const auto img = (VkImage)(img_id.fetch_add(1, std::memory_order_relaxed));
                    auto p = pool.bind(VK_NULL_HANDLE, img, r, stress_bind_image, destroy_image, free_memory,
                                       [mtype](VkDeviceSize, std::uint32_t& type) {
                                           type = mtype;
                                           return (VkDeviceMemory)(std::uintptr_t(++allocated));
                                       });
                    if (p) {
                        ++succeeded;
                        held.push_back(std::move(p));
                        if (held.size() > 16) {
                            held.erase(held.begin());
                        }
                    }
                }
            });
        }
        for (auto& w : workers) w.join();
        CHECK(allocated.load() - alloc_before > 0);
        CHECK(freed.load() - freed_before == allocated.load() - alloc_before);
        CHECK(destroyed.load() - destroyed_before == succeeded.load());
    }

    std::puts("title image pool alignment, type separation, retirement, reuse, concurrency and guards OK");
}
