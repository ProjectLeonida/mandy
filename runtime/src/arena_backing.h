#pragma once
// The GPU address arena's memory (docs/d3d12_vulkan.md section 3.3, "Backing blocks"): which arena offsets have
// memory behind them, where the CPU reaches it, and when it is bound and unbound.
//
// With sparse binding the arena is a sparse VkBuffer that reserves the address range only. Its memory comes in
// backing units: a unit is one VkDeviceMemory, mapped once, bound over one or more whole blocks of the arena. A unit
// of one block holds any number of allocations no larger than a block; an allocation larger than a block gets a
// unit of its own, sized to whole blocks. No allocation crosses a unit, so every allocation is one contiguous run
// of one mapping, and the CPU reaches it through host(): an offset resolves to its unit through the block table in
// constant time.
//
// Without sparse binding (or when the sparse address space is smaller than the arena) there is one permanent unit
// over the whole arena: the arena's one allocation, as before. Every caller goes through the same accessors.
//
// Lifetime of a unit:
//   - bound when the first allocation needs a block it covers (acquire), before the allocation is handed out;
//   - when its last allocation is released it becomes pending, with the retire ticket that every submission which
//     could still name it is at or below (the caller passes the submitted ticket + 1, as defer_destroy does);
//   - an allocation that lands in a pending unit's block revives it (no rebinding);
//   - reclaim(completed) unbinds pending units whose ticket the GPU has passed, then frees their memory. A unit's
//     table entries are cleared under the exclusive lock before its memory is freed, so with_host() and
//     offset_of() never see freed memory.
//
// Threading: acquire, release, reclaim and keep_out run under the device's arena mutex. host() takes no lock: it is
// for an offset inside an allocation the caller holds, whose unit cannot be freed while the allocation lives.
// with_host() and offset_of() take the shared lock, for callers that name an offset or a pointer they do not own.
//
// The class makes no Vulkan call itself: the device (or a test) passes the hooks that allocate, bind and free.

#include <vulkan/vulkan.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <vector>

namespace leonida {

// Chosen from the arena's measured use (a world run: about 5,000 live allocations averaging 160 KiB, 790 MiB live
// over a 990 MiB peak end) and the device: a multiple of the 64 KiB sparse page, few enough units that a 4 GiB arena
// stays within the 4,096 memory objects some Windows drivers allow (2,048 at most, about 500 at that peak), and the
// size of amdgpu's large page.
constexpr VkDeviceSize kArenaBlockBytes = VkDeviceSize(2) << 20;

// The queue the arena binds on: one nothing else submits to, so a bind (and the wait for it) never sits behind a
// submission that waits for a fence the CPU signals later. A family with the sparse binding bit and a queue to spare,
// a sparse-only family first (RADV has one). `used[f]` is how many queues of family f the device already takes;
// `cap` the most it may take of one. No family: {UINT32_MAX, 0}, and the arena stays one allocation.
struct ArenaBindQueue {
    std::uint32_t family = UINT32_MAX;
    std::uint32_t index = 0;
};
inline ArenaBindQueue pick_arena_bind_queue(const VkQueueFamilyProperties* families, std::uint32_t count,
                                            const std::uint32_t* used, std::uint32_t cap) {
    constexpr VkQueueFlags work = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT;
    for (const bool sparse_only : {true, false})
        for (std::uint32_t f = 0; f < count; ++f) {
            const VkQueueFlags flags = families[f].queueFlags;
            if (!(flags & VK_QUEUE_SPARSE_BINDING_BIT) || (sparse_only && (flags & work))) continue;
            if (used[f] >= families[f].queueCount || used[f] >= cap) continue;
            return {f, used[f]};
        }
    return {};
}

struct ArenaBackingHooks {
    // A unit's memory, mapped: false when there is none (the allocation is refused).
    std::function<bool(VkDeviceSize bytes, VkDeviceMemory& memory, std::uint8_t*& host)> make;
    // Frees a unit's memory (which unmaps it). Called only after the unit is unbound.
    std::function<void(VkDeviceMemory memory)> destroy;
    // Binds (memory set) or unbinds (memory null) these ranges of the arena buffer, and returns once the binds have
    // completed. False when the bind failed.
    std::function<bool(const VkSparseMemoryBind* binds, std::uint32_t count)> bind;
};

class ArenaBacking {
public:
    struct Location {
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkDeviceSize memory_offset = 0;  // the offset's place in `memory`
        std::uint8_t* host = nullptr;
    };
    struct Stats {
        VkDeviceSize bound = 0, peak_bound = 0;
        std::uint64_t units = 0, pending = 0, binds = 0, unbinds = 0, bind_failures = 0;
    };

    ArenaBacking() = default;
    ArenaBacking(const ArenaBacking&) = delete;
    ArenaBacking& operator=(const ArenaBacking&) = delete;
    ~ArenaBacking() { clear(); }

    // One permanent unit over [0, size): the arena's single allocation (no sparse binding). `memory` stays the
    // caller's to free.
    void init_fixed(VkDeviceSize size, VkDeviceMemory memory, std::uint8_t* host) {
        clear();
        size_ = size;
        block_ = size;
        shift_ = 63;  // every offset below 2^63 is block 0
        slots_ = std::make_unique<std::atomic<Unit*>[]>(1);
        auto* unit = new Unit{0, size, memory, host, 0, 0, false, true, false};
        slots_[0].store(unit, std::memory_order_release);
        units_by_host_[reinterpret_cast<std::uintptr_t>(host)] = unit;
        stats_.bound = stats_.peak_bound = size;
        stats_.units = 1;
    }

    // A sparse arena of `size` bytes in blocks of `block` (a power of two). Nothing is bound yet.
    bool init_sparse(VkDeviceSize size, VkDeviceSize block, ArenaBackingHooks hooks) {
        clear();
        if (!size || !block || (block & (block - 1)) || !hooks.make || !hooks.destroy || !hooks.bind) return false;
        size_ = size;
        block_ = block;
        shift_ = 0;
        while ((VkDeviceSize(1) << shift_) < block) ++shift_;
        count_ = std::size_t((size + block - 1) >> shift_);
        slots_ = std::make_unique<std::atomic<Unit*>[]>(count_);
        for (std::size_t i = 0; i < count_; ++i) slots_[i].store(nullptr, std::memory_order_relaxed);
        hooks_ = std::move(hooks);
        sparse_ = true;
        return true;
    }

    bool ready() const { return size_ != 0; }
    bool sparse() const { return sparse_; }
    VkDeviceSize size() const { return size_; }
    VkDeviceSize block() const { return block_; }
    Stats stats() const { return stats_; }  // arena mutex held

    // The bytes an allocation of `bytes` takes in the free-block map: larger than a block, it is rounded up to whole
    // blocks, so no other allocation shares its unit.
    VkDeviceSize footprint(VkDeviceSize bytes) const {
        if (!sparse_ || bytes <= block_) return bytes;
        return (bytes + block_ - 1) & ~(block_ - 1);
    }
    // The alignment an allocation of `bytes` needs: a block's for one larger than a block.
    VkDeviceSize alignment(VkDeviceSize bytes, VkDeviceSize alignment) const {
        return sparse_ && bytes > block_ ? std::max(alignment, block_) : alignment;
    }

    // For ArenaBlocks::allocate's keep-out: the end of a range [lo, hi) may not take, or 0. An allocation no larger
    // than a block may not cross a block boundary, nor share a block with a pending unit of several blocks (it would
    // keep all of them bound). A larger one (block-aligned, whole blocks) may not cover a block that still has a unit.
    // Arena mutex held.
    std::uint64_t keep_out(std::uint64_t lo, std::uint64_t hi) const {
        if (!sparse_ || hi <= lo) return 0;
        const std::size_t first = std::size_t(lo >> shift_), last = std::size_t((hi - 1) >> shift_);
        if (hi - lo <= block_) {
            if (first != last) return VkDeviceSize(last) << shift_;  // move up to the next block
            const Unit* unit = slots_[first].load(std::memory_order_relaxed);
            if (unit && unit->pending && unit->bytes > block_) return unit->base + unit->bytes;
            return 0;
        }
        for (std::size_t i = last + 1; i-- > first;)
            if (const Unit* unit = slots_[i].load(std::memory_order_relaxed)) return unit->base + unit->bytes;
        return 0;
    }

    // Whether [offset, offset + bytes) can be acquired as placed (allocate_at): inside one unit that exists, or over
    // blocks that have none. Arena mutex held.
    bool placeable(std::uint64_t offset, std::uint64_t bytes) const {
        if (!sparse_) return true;
        if (!bytes || offset >= size_ || bytes > size_ - offset) return false;
        const std::size_t first = std::size_t(offset >> shift_), last = std::size_t((offset + bytes - 1) >> shift_);
        const Unit* unit = slots_[first].load(std::memory_order_relaxed);
        if (unit) return offset + bytes <= unit->base + unit->bytes;
        for (std::size_t i = first; i <= last; ++i)
            if (slots_[i].load(std::memory_order_relaxed)) return false;
        return true;
    }

    // Memory behind [offset, offset + bytes), bound before this returns. The range was just taken from the free-block
    // map under keep_out() (or checked with placeable()). False when no memory could be made or bound: the caller
    // gives the range back. `permanent`: never released (the reserved head of the arena). Arena mutex held.
    bool acquire(std::uint64_t offset, std::uint64_t bytes, bool permanent = false) {
        if (!sparse_) return true;
        if (!bytes || offset >= size_ || bytes > size_ - offset) return false;
        const std::size_t first = std::size_t(offset >> shift_), last = std::size_t((offset + bytes - 1) >> shift_);
        if (Unit* unit = slots_[first].load(std::memory_order_relaxed)) {
            if (offset + bytes > unit->base + unit->bytes) return false;  // would cross its unit
            if (unit->pending) revive(unit);
            ++unit->live;
            unit->permanent = unit->permanent || permanent;
            return true;
        }
        for (std::size_t i = first; i <= last; ++i)
            if (slots_[i].load(std::memory_order_relaxed)) return false;
        const VkDeviceSize base = VkDeviceSize(first) << shift_;
        const VkDeviceSize unit_bytes = std::min<VkDeviceSize>(VkDeviceSize(last - first + 1) << shift_, size_ - base);
        VkDeviceMemory memory = VK_NULL_HANDLE;
        std::uint8_t* host = nullptr;
        if (!hooks_.make(unit_bytes, memory, host) || !memory || !host) {
            ++stats_.bind_failures;
            return false;
        }
        const VkSparseMemoryBind bind{base, unit_bytes, memory, 0, 0};
        if (!hooks_.bind(&bind, 1)) {
            hooks_.destroy(memory);
            ++stats_.bind_failures;
            return false;
        }
        ++stats_.binds;
        auto* unit = new Unit{base, unit_bytes, memory, host, 1, 0, false, permanent, false};
        {
            std::unique_lock lock(table_mutex_);
            for (std::size_t i = first; i <= last; ++i) slots_[i].store(unit, std::memory_order_release);
            units_by_host_[reinterpret_cast<std::uintptr_t>(host)] = unit;
        }
        stats_.bound += unit_bytes;
        stats_.peak_bound = std::max(stats_.peak_bound, stats_.bound);
        ++stats_.units;
        return true;
    }

    // The allocation at [offset, offset + bytes) is gone. When it was its unit's last, the unit is pending until
    // `ticket` has retired (reclaim). Arena mutex held.
    void release(std::uint64_t offset, std::uint64_t bytes, std::uint64_t ticket) {
        if (!sparse_ || !bytes || offset >= size_) return;
        Unit* unit = slots_[std::size_t(offset >> shift_)].load(std::memory_order_relaxed);
        if (!unit || !unit->live) return;
        if (--unit->live || unit->permanent) return;
        unit->pending = true;
        unit->ticket = ticket;
        if (!unit->listed) {
            unit->listed = true;
            pending_.push_back(unit);
        }
        ++stats_.pending;
    }

    bool has_pending() const { return stats_.pending != 0; }  // arena mutex held

    // Unbinds and frees every pending unit whose ticket `completed` has reached; the rest stay pending. Returns the
    // bytes unbound. Arena mutex held.
    VkDeviceSize reclaim(std::uint64_t completed) {
        if (!sparse_ || pending_.empty()) return 0;
        std::vector<Unit*> retired;
        std::vector<VkSparseMemoryBind> unbinds;
        for (auto it = pending_.begin(); it != pending_.end();) {
            Unit* unit = *it;
            if (!unit->pending) {  // revived since
                unit->listed = false;
                it = pending_.erase(it);
                continue;
            }
            if (unit->ticket > completed) {
                ++it;
                continue;
            }
            retired.push_back(unit);
            unbinds.push_back({unit->base, unit->bytes, VK_NULL_HANDLE, 0, 0});
            it = pending_.erase(it);
        }
        if (retired.empty()) return 0;
        // The GPU has passed every submission that could name these units: unbind them, then let them go.
        if (!hooks_.bind(unbinds.data(), std::uint32_t(unbinds.size()))) {
            ++stats_.bind_failures;  // keep them bound and pending; the next reclaim tries again
            pending_.insert(pending_.end(), retired.begin(), retired.end());
            return 0;
        }
        ++stats_.unbinds;
        VkDeviceSize freed = 0;
        {
            std::unique_lock lock(table_mutex_);
            for (Unit* unit : retired) {
                const std::size_t first = std::size_t(unit->base >> shift_);
                const std::size_t last = std::size_t((unit->base + unit->bytes - 1) >> shift_);
                for (std::size_t i = first; i <= last; ++i) slots_[i].store(nullptr, std::memory_order_release);
                units_by_host_.erase(reinterpret_cast<std::uintptr_t>(unit->host));
            }
        }
        for (Unit* unit : retired) {
            hooks_.destroy(unit->memory);
            freed += unit->bytes;
            stats_.bound -= unit->bytes;
            --stats_.units;
            --stats_.pending;
            delete unit;
        }
        return freed;
    }

    // The CPU address of [offset, offset + bytes), or null when that range has no memory or crosses a unit. No lock:
    // for a range inside an allocation the caller holds. Constant time: one table read.
    std::uint8_t* host(std::uint64_t offset, std::uint64_t bytes) const {
        if (!bytes || offset >= size_ || bytes > size_ - offset) return nullptr;
        const Unit* unit = slots_[std::size_t(offset >> shift_)].load(std::memory_order_acquire);
        if (!unit || offset + bytes > unit->base + unit->bytes) return nullptr;
        return unit->host + (offset - unit->base);
    }

    // The bytes from `offset` to the end of the unit that holds it (0: none does). As host().
    VkDeviceSize room(std::uint64_t offset) const {
        if (offset >= size_) return 0;
        const Unit* unit = slots_[std::size_t(offset >> shift_)].load(std::memory_order_acquire);
        return unit && offset >= unit->base ? unit->base + unit->bytes - offset : 0;
    }

    // The memory behind [offset, offset + bytes) (a D3D12 heap's tiles bind it to reserved resources). As host().
    bool locate(std::uint64_t offset, std::uint64_t bytes, Location& out) const {
        if (!bytes || offset >= size_ || bytes > size_ - offset) return false;
        const Unit* unit = slots_[std::size_t(offset >> shift_)].load(std::memory_order_acquire);
        if (!unit || offset + bytes > unit->base + unit->bytes) return false;
        out = {unit->memory, offset - unit->base, unit->host + (offset - unit->base)};
        return true;
    }

    // Calls `f(pointer)` with the CPU address of [offset, offset + bytes) while no reclaim can free it; false (and no
    // call) when the range has no memory or crosses a unit. For a caller that names an offset it does not hold.
    template<class F>
    bool with_host(std::uint64_t offset, std::uint64_t bytes, F&& f) const {
        std::shared_lock lock(table_mutex_);
        std::uint8_t* p = host(offset, bytes);
        if (!p) return false;
        f(p);
        return true;
    }

    // The arena offset of the CPU range [p, p + n), if it lies inside one unit's mapping. Shared lock; O(log units).
    bool offset_of(const void* p, std::size_t n, std::uint64_t& offset) const {
        const auto address = reinterpret_cast<std::uintptr_t>(p);
        std::shared_lock lock(table_mutex_);
        auto it = units_by_host_.upper_bound(address);
        if (it == units_by_host_.begin()) return false;
        --it;
        const Unit* unit = it->second;
        const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(unit->host);
        if (address - begin > unit->bytes || n > unit->bytes - (address - begin)) return false;
        offset = unit->base + (address - begin);
        return true;
    }

    // Calls `f(base, bytes, host)` for every unit, in offset order, under the shared lock.
    template<class F>
    void for_each_unit(F&& f) const {
        std::shared_lock lock(table_mutex_);
        std::map<VkDeviceSize, const Unit*> order;
        for (const auto& [address, unit] : units_by_host_) order.emplace(unit->base, unit);
        for (const auto& [base, unit] : order) f(unit->base, unit->bytes, unit->host);
    }

    // Frees every unit of a sparse arena (the device is idle: it is being destroyed). A fixed arena's memory stays
    // the caller's.
    void clear() {
        std::vector<Unit*> all;
        {
            std::unique_lock lock(table_mutex_);
            for (const auto& [address, unit] : units_by_host_) all.push_back(unit);
            units_by_host_.clear();
            for (std::size_t i = 0; slots_ && i < count_; ++i) slots_[i].store(nullptr, std::memory_order_relaxed);
        }
        for (Unit* unit : all) {
            if (sparse_ && hooks_.destroy) hooks_.destroy(unit->memory);
            delete unit;
        }
        pending_.clear();
        slots_.reset();
        count_ = 1;
        size_ = block_ = 0;
        sparse_ = false;
        stats_ = {};
    }

private:
    struct Unit {
        VkDeviceSize base, bytes;
        VkDeviceMemory memory;
        std::uint8_t* host;
        std::uint64_t live;    // allocations in it
        std::uint64_t ticket;  // pending: unbind once this retire ticket is complete
        bool pending;
        bool permanent;
        bool listed;  // in pending_
    };

    void revive(Unit* unit) {
        unit->pending = false;  // stays in pending_ until the next reclaim drops it
        --stats_.pending;
    }

    VkDeviceSize size_ = 0, block_ = 0;
    unsigned shift_ = 63;
    std::size_t count_ = 1;
    bool sparse_ = false;
    std::unique_ptr<std::atomic<Unit*>[]> slots_;  // one per block: the unit over it
    mutable std::shared_mutex table_mutex_;          // slots_ stores and units_by_host_, against freeing
    std::map<std::uintptr_t, Unit*> units_by_host_;  // by mapping address, for offset_of
    std::vector<Unit*> pending_;
    ArenaBackingHooks hooks_;
    Stats stats_;
};

}  // namespace leonida
