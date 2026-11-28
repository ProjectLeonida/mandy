#pragma once
#include "gpu_label_memory.h"
#include "value_write_sink.h"
#include <vulkan/vulkan.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include "title_page_watch.h"
#endif

namespace leonida::label {
// LEONIDA_GPU_LABELS=1 records list slots 71 and 72 here, for the lists the list values (shims/d3d12_xs_list_values.h,
// LEONIDA_LIST_VALUES) do not track: those land their writes themselves, and one address has one writer.
inline bool enabled() {
    static const bool active = [] {
        const char *v = std::getenv("LEONIDA_GPU_LABELS");
        return v && v[0] == '1';
    }();
    return active;
}
// Stores a completed label into the title's memory: only into a committed, private, writable page of the allocation
// the label was captured in, and only while that allocation's generation is the captured one (Memory::publish).
//
// A page the D3D12 layer watches for the title's writes is read-only while it waits (title_page_watch.h), and it is
// writable as far as the title knows. Such a page is announced and opened the way a kernel write is
// (kernel_writes::begin_write), so the label reaches it and the layer copies the page again; a page that stays
// read-only after that is the title's own doing and the label is dropped, as before. Called with no shadow lock held.
inline bool store_value(const Target &target, std::uint64_t value) {
    // The runtime's own memory first (src/value_write_sink.h): an arena offset is not a host address, so it goes to the
    // arena and never to the store below; a page xmem maps PAGE_NOACCESS is written through xmem's own mapping. The
    // value is written little-endian, the way the console's x64 writes it: the low 'width' bytes of the word. A
    // target neither takes - the title's own private pages - is left to the path below, which opens a read-only page
    // the way a kernel write does.
    {
        const std::uint64_t word = value;
        bool refused = false;
        if (memory().publish(target, [&](std::uintptr_t address, unsigned width, std::uintptr_t) {
                const value_sink::Landing landing = value_sink::arena(address, &word, width);
                if (landing != value_sink::Landing::elsewhere) {
                    refused = landing == value_sink::Landing::refused;
                    return !refused;
                }
                return value_sink::title_pages(reinterpret_cast<const void *>(address), &word, width);
            }))
            return true;
        if (refused) return false;
    }
#ifdef _WIN32
    return memory().publish(target, [&](std::uintptr_t address, unsigned width, std::uintptr_t allocation_base) {
        MEMORY_BASIC_INFORMATION mbi{};
        const auto in_allocation = [&] {
            return VirtualQuery(reinterpret_cast<const void *>(address), &mbi, sizeof(mbi)) && mbi.State == MEM_COMMIT &&
                   mbi.Type == MEM_PRIVATE && reinterpret_cast<std::uintptr_t>(mbi.AllocationBase) == allocation_base &&
                   !(mbi.Protect & PAGE_GUARD);
        };
        if (!in_allocation()) return false;
        std::uint64_t pin = 0;
        if ((mbi.Protect & 0xff) == PAGE_READONLY) {
            pin = kernel_writes::begin_write(reinterpret_cast<const void *>(address), width, nullptr);
            if (!pin || !in_allocation()) {
                kernel_writes::end_write(pin, false);
                return false;
            }
        }
        const auto begin = reinterpret_cast<std::uintptr_t>(mbi.BaseAddress);
        const bool writable = (mbi.Protect & 0xff) == PAGE_READWRITE || (mbi.Protect & 0xff) == PAGE_WRITECOPY;
        const bool inside = address >= begin && address - begin <= mbi.RegionSize &&
                            mbi.RegionSize - (address - begin) >= width;
        if (writable && inside) {
            if (width == 4)
                InterlockedExchange(reinterpret_cast<volatile LONG *>(address), static_cast<LONG>(value));
            else
                InterlockedExchange64(reinterpret_cast<volatile LONG64 *>(address), static_cast<LONG64>(value));
        }
        kernel_writes::end_write(pin, false);
        return writable && inside;
    });
#else
    (void)target;
    (void)value;
    return false;
#endif
}

// Each submitted Vulkan command buffer owns a fence and its query slots until the GPU finishes it.
// The CPU destination is a generation-checked title data allocation. No guest pointer is a GPU command destination.
class Completions {
    static constexpr unsigned per_pool = 64;
    static constexpr std::uint32_t no_query = (std::numeric_limits<std::uint32_t>::max)();
    struct Entry {
        Write op;
        std::uint32_t query = no_query;
    };
    struct Batch {
        VkFence fence{};
        std::vector<VkQueryPool> pools;
        std::vector<Entry> labels;
        std::uint32_t query_count = 0;
    };
    struct PublishCounts {
        std::uint64_t errors = 0;
        std::uint64_t expired = 0;
    };
    std::mutex mutex_;
    std::condition_variable changed_;
    bool stopping_ = false;
    bool closing_ = false;
    std::thread worker_;
    VkDevice device_{};
    std::unordered_map<VkCommandBuffer, std::unique_ptr<Batch>> recording_;
    std::vector<std::unique_ptr<Batch>> pending_;
    std::vector<VkQueryPool> idle_pools_;
    std::vector<VkFence> idle_fences_;
    PFN_vkCreateQueryPool create_pool_{};
    PFN_vkDestroyQueryPool destroy_pool_{};
    PFN_vkCmdResetQueryPool reset_query_{};
    PFN_vkCmdWriteTimestamp timestamp_{};
    PFN_vkGetQueryPoolResults results_{};
    PFN_vkCreateFence create_fence_{};
    PFN_vkDestroyFence destroy_fence_{};
    PFN_vkGetFenceStatus fence_status_{};
    PFN_vkResetFences reset_fences_{};
    PFN_vkDeviceWaitIdle device_idle_{};
    std::uint64_t expired_ = 0, errors_ = 0;
    VkQueryPool pool() {
        if (!idle_pools_.empty()) {
            auto h = idle_pools_.back();
            idle_pools_.pop_back();
            return h;
        }
        VkQueryPoolCreateInfo c{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        c.queryType = VK_QUERY_TYPE_TIMESTAMP;
        c.queryCount = per_pool;
        VkQueryPool p{};
        return create_pool_(device_, &c, nullptr, &p) == VK_SUCCESS ? p : VK_NULL_HANDLE;
    }
    VkFence fence() {
        if (!idle_fences_.empty()) {
            auto f = idle_fences_.back();
            idle_fences_.pop_back();
            if (reset_fences_(device_, 1, &f) == VK_SUCCESS) return f;
            destroy_fence_(device_, f, nullptr);
        }
        VkFenceCreateInfo c{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        VkFence f{};
        return create_fence_(device_, &c, nullptr, &f) == VK_SUCCESS ? f : VK_NULL_HANDLE;
    }
    void recycle(Batch &b) {
        for (auto p : b.pools)
            idle_pools_.push_back(p);
        b.pools.clear();
        if (b.fence) {
            idle_fences_.push_back(b.fence);
            b.fence = VK_NULL_HANDLE;
        }
    }
    static bool store(const Target &target, std::uint64_t value) { return store_value(target, value); }
    static PublishCounts publish(const Batch &batch, VkDevice dev, PFN_vkGetQueryPoolResults results) {
        PublishCounts counts{};
        for (const Entry &entry : batch.labels) {
            const Write &op = entry.op;
            std::uint64_t value = op.value;
            if (op.clock == 1) {
                const auto pool_index = entry.query / per_pool;
                // The batch's fence has already signaled VK_SUCCESS; pass WAIT_BIT so drivers that defer timestamp
                // writeback cannot drop a completed label with VK_NOT_READY.
                if (entry.query == no_query || pool_index >= batch.pools.size() ||
                    results(dev, batch.pools[pool_index], static_cast<unsigned>(entry.query % per_pool), 1,
                            sizeof(value), &value, sizeof(value),
                            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) != VK_SUCCESS) {
                    ++counts.errors;
                    continue;
                }
            } else if (op.clock == 2) {
#ifdef _WIN32
                LARGE_INTEGER qpc{};
                if (!QueryPerformanceCounter(&qpc)) {
                    ++counts.errors;
                    continue;
                }
                value = static_cast<std::uint64_t>(qpc.QuadPart);
#else
                ++counts.errors;
                continue;
#endif
            }
            if (!store(op.target, value)) ++counts.expired;
        }
        return counts;
    }
    void run() {
        std::unique_lock lock(mutex_);
        const VkDevice dev = device_;
        const auto fence_status = fence_status_;
        const auto results = results_;
        for (;;) {
            if (pending_.empty() && !closing_) changed_.wait(lock, [&] { return closing_ || !pending_.empty(); });
            if (pending_.empty()) return;
            const bool closing_now = closing_;
            std::vector<std::unique_ptr<Batch>> checking;
            checking.swap(pending_);
            // Release mutex_ before polling fences or publishing labels: store_value() invokes value_sink::arena()
            // (which takes VulkanContext::resource_mutex_) and kernel_writes::begin_write(), while threads holding
            // resource_mutex_ call begin_record()/record() and acquire mutex_.
            lock.unlock();

            std::vector<std::unique_ptr<Batch>> still_pending;
            std::vector<std::unique_ptr<Batch>> done;
            PublishCounts totals{};
            for (auto &batch : checking) {
                const auto status = fence_status(dev, batch->fence);
                if (status == VK_NOT_READY && !closing_now) {
                    still_pending.push_back(std::move(batch));
                    continue;
                }
                if (status == VK_SUCCESS) {
                    const PublishCounts c = publish(*batch, dev, results);
                    totals.errors += c.errors;
                    totals.expired += c.expired;
                } else {
                    ++totals.errors;
                }
                done.push_back(std::move(batch));
            }

            lock.lock();
            errors_ += totals.errors;
            expired_ += totals.expired;
            for (auto &b : done)
                recycle(*b);
            const bool arrived = !pending_.empty();
            if (!still_pending.empty()) {
                if (pending_.empty()) {
                    pending_.swap(still_pending);
                } else {
                    still_pending.insert(still_pending.end(), std::make_move_iterator(pending_.begin()),
                                         std::make_move_iterator(pending_.end()));
                    pending_.swap(still_pending);
                }
            }
            if (closing_) {
                if (pending_.empty()) return;
                continue;
            }
            if (!pending_.empty() && !arrived) changed_.wait_for(lock, std::chrono::milliseconds(1));
        }
    }

  public:
    ~Completions() {
        VkDevice dev = VK_NULL_HANDLE;
        {
            std::lock_guard lock(mutex_);
            dev = device_;
        }
        if (dev) stop(dev);
    }

    bool initialize(VkDevice dev, PFN_vkGetDeviceProcAddr get) {
        if (!enabled() || !dev || !get) return false;
        std::lock_guard lock(mutex_);
        if (device_) return device_ == dev && !stopping_;
#define LABEL_FN(name) reinterpret_cast<PFN_##name>(get(dev, #name))
        const auto create_pool = LABEL_FN(vkCreateQueryPool);
        const auto destroy_pool = LABEL_FN(vkDestroyQueryPool);
        const auto reset_query = LABEL_FN(vkCmdResetQueryPool);
        const auto timestamp = LABEL_FN(vkCmdWriteTimestamp);
        const auto results = LABEL_FN(vkGetQueryPoolResults);
        const auto create_fence = LABEL_FN(vkCreateFence);
        const auto destroy_fence = LABEL_FN(vkDestroyFence);
        const auto fence_status = LABEL_FN(vkGetFenceStatus);
        const auto reset_fences = LABEL_FN(vkResetFences);
        const auto device_idle = LABEL_FN(vkDeviceWaitIdle);
#undef LABEL_FN
        if (!create_pool || !destroy_pool || !reset_query || !timestamp || !results || !create_fence ||
            !destroy_fence || !fence_status || !reset_fences || !device_idle)
            return false;
        create_pool_ = create_pool;
        destroy_pool_ = destroy_pool;
        reset_query_ = reset_query;
        timestamp_ = timestamp;
        results_ = results;
        create_fence_ = create_fence;
        destroy_fence_ = destroy_fence;
        fence_status_ = fence_status;
        reset_fences_ = reset_fences;
        device_idle_ = device_idle;
        device_ = dev;
        stopping_ = false;
        closing_ = false;
        try {
            worker_ = std::thread([this] { run(); });
        } catch (...) {
            device_ = VK_NULL_HANDLE;
            throw;
        }
        return true;
    }
    void begin_record(VkCommandBuffer cb) {
        if (!enabled()) return;
        std::lock_guard lock(mutex_);
        auto i = recording_.find(cb);
        if (i != recording_.end()) {
            recycle(*i->second);
            recording_.erase(i);
        }
    }
    void record(VkCommandBuffer cb, const Write &op) {
        if (!enabled()) return;
        std::lock_guard lock(mutex_);
        if (!device_ || stopping_ || closing_) {
            ++errors_;
            return;
        }
        auto &slot = recording_[cb];
        if (!slot) {
            slot = std::make_unique<Batch>();
            slot->fence = fence();
        }
        if (!slot->fence) {
            ++errors_;
            return;
        }
        if (op.clock == 1) {
            const auto q = slot->query_count;
            const auto pool_index = q / per_pool;
            const auto j = static_cast<unsigned>(q % per_pool);
            if (pool_index >= slot->pools.size()) {
                auto p = pool();
                if (!p) {
                    ++errors_;
                    return;
                }
                try {
                    slot->pools.push_back(p);
                } catch (...) {
                    idle_pools_.push_back(p);
                    throw;
                }
            }
            auto p = slot->pools[pool_index];
            slot->labels.push_back(Entry{op, q});
            ++slot->query_count;
            reset_query_(cb, p, j, 1);
            timestamp_(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, p, j);
        } else {
            slot->labels.push_back(Entry{op, no_query});
        }
    }
    VkFence submit_fence(VkCommandBuffer cb) {
        if (!enabled() || !cb) return VK_NULL_HANDLE;
        std::lock_guard lock(mutex_);
        if (stopping_ || closing_) return VK_NULL_HANDLE;
        auto i = recording_.find(cb);
        return i == recording_.end() ? VK_NULL_HANDLE : i->second->fence;
    }
    void submitted(VkCommandBuffer cb, bool ok) {
        if (!enabled() || !cb) return;
        std::lock_guard lock(mutex_);
        auto i = recording_.find(cb);
        if (i == recording_.end()) return;
        if (ok && i->second->fence && !closing_) {
            pending_.push_back(std::move(i->second));
            changed_.notify_one();
        } else {
            if (ok && i->second->fence) ++errors_;
            recycle(*i->second);
        }
        recording_.erase(i);
    }
    void stop(VkDevice dev) {
        PFN_vkDeviceWaitIdle device_idle = nullptr;
        {
            std::unique_lock lock(mutex_);
            if (!dev) return;
            if (stopping_) {
                if (device_ == dev) changed_.wait(lock, [&] { return !stopping_; });
                return;
            }
            if (device_ != dev) return;
            stopping_ = true;
            device_idle = device_idle_;
        }
        if (device_idle) device_idle(dev);
        std::thread worker;
        {
            std::lock_guard lock(mutex_);
            closing_ = true;
            changed_.notify_all();
            worker = std::move(worker_);
        }
        if (worker.joinable()) worker.join();
        std::lock_guard lock(mutex_);
        for (auto &pair : recording_)
            recycle(*pair.second);
        recording_.clear();
        for (auto &b : pending_)
            recycle(*b);
        pending_.clear();
        for (auto f : idle_fences_)
            destroy_fence_(dev, f, nullptr);
        idle_fences_.clear();
        for (auto p : idle_pools_)
            destroy_pool_(dev, p, nullptr);
        idle_pools_.clear();
        if (errors_) {
            std::fprintf(stderr, "[rt] d3d12: %llu GPU label operations failed, %llu destinations expired\n",
                         static_cast<unsigned long long>(errors_), static_cast<unsigned long long>(expired_));
        }
        expired_ = errors_ = 0;
        device_ = VK_NULL_HANDLE;
        stopping_ = false;
        closing_ = false;
        changed_.notify_all();
    }
};
} // namespace leonida::label
