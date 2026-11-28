#pragma once
// The page mirror (title_mirror): one GPU copy of the title's GPU-readable pages, at heap_base + the title's address.
//
// A V# or a 64-bit pointer the title writes into its own memory names a title address, and the translated shaders reach
// heap_base + sext48(address) (docs/design.md, "Heap-relative buffer device addresses"). The mirror makes that address
// hold the title's bytes:
//   - the address range: sparse buffers ("chunks", kChunk each) created at heap_base + the title's address with
//     bufferDeviceAddressCaptureReplay, over every reservation xmem placed in src/title_mirror_space.h. A page with
//     nothing bound reads zero and drops stores (residencyNonResidentStrict), so a title address in a reservation that
//     is not mirrored, or a page not mapped, never faults;
//   - one home per physical page: 64 KiB of a 32 MiB unit (one VkDeviceMemory, host-visible, mapped once). Every title
//     page that shows the physical page (a CPU view, a GPU-only alias, a CopyPageMappingsBatchX copy) is bound to that
//     one home, so aliases share the bytes and the GPU's writes, and a remap is a rebind, never a copy;
//   - the change feed: xmem reports every change of what a page of the mirror's range shows (src/title_frames.h,
//     Feed), in order; map() takes it;
//   - the dirty list: a home is filled from its physical page when it is made, and again when the caller says the page
//     was written (dirty(): the geometry watch's cells say so: a guard fault, a kernel write, a remap);
//   - flush(): the binds noted since the last one go in one batch, which signals a value the next submission of every
//     queue waits for (the ordering rule: a submission after a remap in call order never reads a page whose bind has
//     not landed). A home unbound everywhere is freed once the submissions that could read it have retired;
//   - aliases (alias()): a range of title pages also bound at an address of the caller's (a console descriptor heap's
//     table address below 4 GiB), each page there with a home of its own, the filtered copy, which hooks.filter fills
//     from the title's bytes whenever they change. The page's own home stays the title's bytes exactly: a V# or a
//     pointer of the title into those pages reads what the title wrote, and only the alias reads filtered bytes.
//
// No Vulkan call here: the device (or a test) passes hooks, as for src/arena_backing.h. Not thread-safe: the caller
// serialises (the device under its mirror mutex).
#include <vulkan/vulkan.h>

#include "shadow_keep.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace leonida {

struct TitleMirrorHooks {
    // A sparse buffer of `bytes` at device address heap_base + title_base; its handle. False: not placed there.
    std::function<bool(std::uint64_t title_base, std::uint64_t bytes, VkBuffer& out)> make_chunk;
    // One unit of home memory, mapped: kUnitBytes of device memory and its host address.
    std::function<bool(VkDeviceMemory& out, std::uint8_t*& host)> make_unit;
    std::function<void(VkDeviceMemory)> free_unit;
    // The binds in one batch (per chunk: VkSparseMemoryBind with resourceOffset inside it); signals `value`.
    std::function<bool(const std::vector<VkSparseBufferMemoryBindInfo>& binds, std::uint64_t value)> bind;
    // Copies a physical page's 64 KiB into its home; false when its bytes cannot be read (the home keeps zeros).
    // May be called concurrently from internal worker threads during batched upload(); must be thread-safe.
    std::function<bool(std::uint8_t* to, const std::uint8_t* from, std::size_t bytes)> copy;
    // Fills the filtered copy of a page in an aliased range (alias): the title's bytes at `from` (the page shown at
    // title address `va`) in, what the GPU reads at the alias into `to`. Called after the page's own home took the
    // title's bytes, when they changed. `fresh`: the copy holds nothing of an earlier fill (made now, or a failed fill
    // zeroed it), so every byte must be written; otherwise `to` still holds this filter's last output for the same page.
    // The title's bytes and the page's own home are never written. False as for `copy` (the copy is zeroed).
    std::function<bool(std::uint64_t va, std::uint8_t* to, const std::uint8_t* from, std::size_t bytes, bool fresh)> filter;
    // Optional fast path for `filter`: called whenever `filter_pieces` is set, passing the already-copied 64 KiB
    // `scratch` buffer (no duplicate guarded copy; synced with `home` and `home.sums[]`) and the 256-bit
    // `changed_pieces` mask (bit k set when piece [k*256, (k+1)*256) changed; all bits set when `fresh` or after a
    // write-back refilter). Falls back to `filter` (still receiving the same `scratch` snapshot) when `filter_pieces`
    // is not set.
    std::function<bool(std::uint64_t va, std::uint8_t* to, const std::uint8_t* scratch, std::size_t bytes, bool fresh,
                       const std::array<std::uint64_t, 4>& changed_pieces)>
        filter_pieces;
};

class TitleMirror {
public:
    static constexpr std::uint64_t kPage = 0x10000;
    static constexpr std::uint64_t kChunk = std::uint64_t(2) << 30;
    static constexpr std::uint64_t kUnitBytes = std::uint64_t(32) << 20;
    static constexpr std::uint32_t kUnitPages = std::uint32_t(kUnitBytes / kPage);
    static constexpr std::size_t kParallelThreshold = 16;
    static constexpr std::size_t kMaxWorkers = 6;
    // Sentinel ticket for freed homes waiting for flush() + note_mirror_bind() to publish the sparse unbind before
    // stamping their retire ticket via stamp_freed().
    static constexpr std::uint64_t kUnstamped = UINT64_MAX;

    struct Counters {
        std::uint64_t chunks = 0, chunk_failures = 0, homes = 0, units = 0, pages = 0;  // now
        std::uint64_t binds = 0, unbinds = 0, batches = 0, uploads = 0, upload_failures = 0, homes_freed = 0;  // so far
        std::uint64_t unmirrored_changes = 0;  // feed entries outside every chunk
        std::uint64_t filtered = 0;            // fills of an alias's filtered copy through hooks.filter
        // Bytes the uploads copied into homes, bytes write_back() copied to the title's pages, and 256-byte pieces
        // both sides had changed (the title's bytes kept), so far.
        std::uint64_t uploaded_bytes = 0, written_back_bytes = 0, conflicts = 0;
    };

    // Reusable thread pool for batched upload(), write_back_batch(), and lock-free readback pre-hashing. Synchronous
    // barrier: every worker finishes before run() returns, so callers' state remains single-owner with zero lock-order
    // issues.
    class WorkerPool {
    public:
        explicit WorkerPool(std::size_t bg_threads) : scratches_(bg_threads + 1) {
            threads_.reserve(bg_threads);
            try {
                for (std::size_t t = 0; t < bg_threads; ++t) {
                    threads_.emplace_back([this, tid = t + 1] { worker_loop(tid); });
                }
            } catch (...) {
                shutdown();
                throw;
            }
        }
        ~WorkerPool() { shutdown(); }
        WorkerPool(const WorkerPool&) = delete;
        WorkerPool& operator=(const WorkerPool&) = delete;

        std::size_t worker_count() const noexcept { return threads_.size() + 1; }
        std::uint8_t* scratch(std::size_t worker_id) noexcept { return scratches_[worker_id].bytes.data(); }

        template <class Fn>
        void run(std::size_t count, std::size_t chunk_size, Fn&& fn) {
            if (!count) return;
            if (threads_.empty()) {
                fn(0, count, 0);
                return;
            }
            auto thunk = [](void* ctx, std::size_t b, std::size_t e, std::size_t tid) {
                (*static_cast<Fn*>(ctx))(b, e, tid);
            };
            {
                std::lock_guard lock(mutex_);
                task_fn_ = thunk;
                task_ctx_ = &fn;
                task_count_ = count;
                task_chunk_ = chunk_size;
                next_index_.store(0, std::memory_order_relaxed);
                remaining_workers_.store(threads_.size(), std::memory_order_release);
                ++generation_;
            }
            work_cv_.notify_all();
            struct JoinGuard {
                WorkerPool& pool;
                ~JoinGuard() {
                    if (pool.remaining_workers_.load(std::memory_order_acquire) != 0) {
                        std::unique_lock lock(pool.mutex_);
                        pool.done_cv_.wait(lock, [this] {
                            return pool.remaining_workers_.load(std::memory_order_acquire) == 0;
                        });
                    }
                    pool.task_fn_ = nullptr;
                    pool.task_ctx_ = nullptr;
                }
            } join_guard{*this};
            drain_work(thunk, &fn, count, chunk_size, 0);
        }

    private:
        struct alignas(64) Scratch {
            std::array<std::uint8_t, kPage> bytes{};
        };
        using Thunk = void (*)(void*, std::size_t, std::size_t, std::size_t);

        void shutdown() noexcept {
            {
                std::lock_guard lock(mutex_);
                stop_ = true;
                ++generation_;
            }
            work_cv_.notify_all();
            for (std::thread& t : threads_)
                if (t.joinable()) t.join();
            threads_.clear();
        }

        void drain_work(Thunk fn, void* ctx, std::size_t count, std::size_t chunk, std::size_t tid) {
            for (;;) {
                const std::size_t b = next_index_.fetch_add(chunk, std::memory_order_relaxed);
                if (b >= count) break;
                const std::size_t e = (std::min)(b + chunk, count);
                fn(ctx, b, e, tid);
            }
        }

        void worker_loop(std::size_t tid) {
            std::uint64_t seen = 0;
            for (;;) {
                Thunk fn = nullptr;
                void* ctx = nullptr;
                std::size_t count = 0;
                std::size_t chunk = 0;
                {
                    std::unique_lock lock(mutex_);
                    work_cv_.wait(lock, [this, seen] { return generation_ != seen; });
                    seen = generation_;
                    if (stop_) return;
                    fn = task_fn_;
                    ctx = task_ctx_;
                    count = task_count_;
                    chunk = task_chunk_;
                }
                struct DoneGuard {
                    WorkerPool& pool;
                    ~DoneGuard() {
                        if (pool.remaining_workers_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                            std::lock_guard lock(pool.mutex_);
                            pool.done_cv_.notify_one();
                        }
                    }
                } done_guard{*this};
                drain_work(fn, ctx, count, chunk, tid);
            }
        }

        std::vector<Scratch> scratches_;
        std::vector<std::thread> threads_;
        std::mutex mutex_;
        std::condition_variable work_cv_;
        std::condition_variable done_cv_;
        Thunk task_fn_ = nullptr;
        void* task_ctx_ = nullptr;
        std::size_t task_count_ = 0;
        std::size_t task_chunk_ = 1;
        std::uint64_t generation_ = 0;
        bool stop_ = false;
        alignas(64) std::atomic<std::size_t> remaining_workers_{0};
        alignas(64) std::atomic<std::size_t> next_index_{0};
    };

    static std::unique_ptr<WorkerPool> make_worker_pool() {
        const unsigned hw = std::thread::hardware_concurrency();
        const std::size_t total = (hw >= 4) ? (std::min)(std::size_t(hw / 2), kMaxWorkers)
                                            : ((hw >= 2) ? std::size_t(hw) : std::size_t(1));
        if (total <= 1) return nullptr;
        try {
            return std::make_unique<WorkerPool>(total - 1);
        } catch (...) {
            return nullptr;
        }
    }

    explicit TitleMirror(TitleMirrorHooks hooks) : hooks_(std::move(hooks)) {}
    ~TitleMirror() {
        pool_.reset();
        for (const Unit& u : units_)
            if (u.memory && hooks_.free_unit) hooks_.free_unit(u.memory);
    }
    TitleMirror(const TitleMirror&) = delete;
    TitleMirror& operator=(const TitleMirror&) = delete;

    // Enables or disables internal worker-pool parallelisation for large batches (enabled by default; tests toggle it
    // to compare single-threaded vs parallel output and timings directly).
    void set_parallel(bool enabled) noexcept { parallel_enabled_ = enabled; }
    bool parallel_enabled() const noexcept { return parallel_enabled_; }

    // The chunks over [base, end) of the title's address space (a reservation xmem placed). False when one could not
    // be placed: its pages stay unmirrored (a title address there reads what heap_base + it names: the caller logs).
    bool cover(std::uint64_t base, std::uint64_t end) {
        bool ok = true;
        for (std::uint64_t c = base / kChunk; c * kChunk < end; ++c) {
            if (chunks_.count(c)) continue;
            VkBuffer buffer = VK_NULL_HANDLE;
            if (!hooks_.make_chunk || !hooks_.make_chunk(c * kChunk, kChunk, buffer)) {
                ++counters_.chunk_failures;
                ok = false;
                chunks_[c] = VK_NULL_HANDLE;  // tried once
                continue;
            }
            chunks_[c] = buffer;
            ++counters_.chunks;
        }
        return ok;
    }
    bool covers(std::uint64_t va) const {
        const auto it = chunks_.find(va / kChunk);
        return it != chunks_.end() && it->second != VK_NULL_HANDLE;
    }

    // The feed: the 64 KiB page at `va` shows the physical page whose 64 KiB are at `frame` (its section's scratch view,
    // kept alive by `keep`), or nothing (`frame` null). The bind is noted for the next flush(); a new home is filled
    // immediately unless `defer_fill` is true (and the page is not aliased), in which case it is queued in `dirty_` for
    // parallel batched filling by the upcoming upload().
    // Freed homes default to kUnstamped and wait for stamp_freed() after flush() publishes the sparse unbind.
    void map(std::uint64_t va, std::uint8_t* frame, std::shared_ptr<void> keep,
             std::uint64_t ticket = kUnstamped, bool defer_fill = false) {
        va &= ~(kPage - 1);
        if (!covers(va)) {
            ++counters_.unmirrored_changes;
            return;
        }
        const auto page = pages_.find(va);
        if (page != pages_.end()) {
            std::uint8_t* const old_frame = page->second;
            if (old_frame == frame) return;
            // A filtered copy made for this page: another aliased page showing the same home takes it over (filled
            // whole for that page), else it goes with this page. Bind `h` by reference and save `old_frame` by value
            // before sync_and_fill_copy(), which may re-entrantly insert into `pages_` / `homes_` and rehash them.
            if (const auto shown = homes_.find(old_frame);
                shown != homes_.end() && shown->second.copy && shown->second.filter_va == va) {
                std::uint64_t other_va = 0;
                for (const auto& [other, other_frame] : pages_)
                    if (other != va && other_frame == old_frame && aliased(other)) other_va = other;
                Home& h = shown->second;
                if (other_va) {
                    h.filter_va = other_va;
                    h.fresh = true;
                    sync_and_fill_copy(h, old_frame, true);
                } else {
                    drop_copy(h, ticket);
                }
            }
            release_home(old_frame, ticket);
            pages_.erase(va);
        }
        if (!frame) {
            note_bind(va, VK_NULL_HANDLE, 0);
            note_alias_bind(va, VK_NULL_HANDLE, 0);
            ++counters_.unbinds;
            return;
        }
        bool created = false;
        Home* home = home_of(frame, std::move(keep), created);
        if (!home) {
            note_bind(va, VK_NULL_HANDLE, 0);
            note_alias_bind(va, VK_NULL_HANDLE, 0);
            return;
        }
        ++home->refs;
        pages_[va] = frame;
        note_bind(va, units_[home->unit].memory, VkDeviceSize(home->slot) * kPage);
        const bool is_aliased = aliased(va);
        if (created) {
            if (is_aliased) {
                if (take_slot(home->copy_unit, home->copy_slot)) {
                    home->copy = true;
                    home->filter_va = va;
                    home->fresh = true;
                    note_alias_bind(va, units_[home->copy_unit].memory, VkDeviceSize(home->copy_slot) * kPage);
                } else {
                    ++counters_.upload_failures;
                    note_alias_bind(va, VK_NULL_HANDLE, 0);  // no room: the alias reads zero there
                }
                fill(*home, frame, true);
            } else if (defer_fill) {
                home->whole_fill = true;
                home->dirty = true;
                dirty_.push_back(frame);
                dirty_count_.store(dirty_.size(), std::memory_order_release);
            } else {
                fill(*home, frame, true);
            }
        } else if (is_aliased) {
            if (!home->copy) make_copy(*home, frame, va);
            if (home->copy) note_alias_bind(va, units_[home->copy_unit].memory, VkDeviceSize(home->copy_slot) * kPage);
            else note_alias_bind(va, VK_NULL_HANDLE, 0);  // no room: the alias reads zero there
        }
        ++counters_.binds;
    }

    // [lo, hi) of the title's address space (whole pages) is also bound into `buffer` at `offset` + (va - lo), each page
    // there with a home of its own, the filtered copy: hooks.filter fills it from the title's bytes now and whenever the
    // page is filled again, it follows the page's remaps and unmaps, and its binds go in the same batches as the chunk
    // binds, so the same wait orders a reader of either address. The page's own home, at the title's address, keeps
    // the title's bytes. A console descriptor heap uses it to be read filtered at an offset below 4 GiB of the arena
    // (32-bit descriptor-table pointers). Pages not mapped are left unbound there (they read zero).
    void alias(std::uint64_t lo, std::uint64_t hi, VkBuffer buffer, VkDeviceSize offset) {
        aliases_[lo] = {hi, buffer, offset};
        std::vector<std::pair<std::uint64_t, std::uint8_t*>> matched;
        for (const auto& [va, frame] : pages_)
            if (va >= lo && va < hi) matched.emplace_back(va, frame);
        for (const auto& [va, frame] : matched) {
            const auto home = homes_.find(frame);
            if (home == homes_.end()) continue;
            Home& h = home->second;  // reference survives re-entrant homes_ rehash inside make_copy()
            if (!h.copy) make_copy(h, frame, va);
            if (h.copy) note_alias_bind(va, units_[h.copy_unit].memory, VkDeviceSize(h.copy_slot) * kPage);
        }
    }
    // The alias at `lo` goes: its pages are unbound there, and their filtered copies are freed after `ticket`
    // (kUnstamped: wait for stamp_freed() after flush() publishes the sparse unbind).
    void unalias(std::uint64_t lo, std::uint64_t ticket = kUnstamped) {
        const auto it = aliases_.find(lo);
        if (it == aliases_.end()) return;
        const Alias alias = it->second;
        for (const auto& [va, frame] : pages_)
            if (va >= lo && va < alias.hi) {
                alias_pending_[alias.buffer][alias.offset + (va - lo)] =
                    {alias.offset + (va - lo), kPage, VK_NULL_HANDLE, 0, 0};
                has_pending_binds_.store(true, std::memory_order_release);
                const auto home = homes_.find(frame);
                if (home != homes_.end() && home->second.copy && home->second.filter_va == va)
                    drop_copy(home->second, ticket);
            }
        aliases_.erase(it);
    }
    // The filtered copy of the aliased page at `va`, mapped (its host bytes): what a filter's caller writes one
    // changed slot into between fills. Null when the page is not mapped or has no copy.
    std::uint8_t* copy_bytes(std::uint64_t va) {
        const auto page = pages_.find(va & ~(kPage - 1));
        if (page == pages_.end()) return nullptr;
        const auto home = homes_.find(page->second);
        if (home == homes_.end() || !home->second.copy) return nullptr;
        return units_[home->second.copy_unit].host + std::size_t(home->second.copy_slot) * kPage;
    }

    // The page at `va` was written (its physical page's bytes changed): its home is filled again, once however many
    // pages show it. True when `va` is a mirrored page.
    bool dirty(std::uint64_t va) {
        const auto page = pages_.find(va & ~(kPage - 1));
        if (page == pages_.end()) return false;
        const auto home = homes_.find(page->second);
        if (home == homes_.end()) return false;
        if (!home->second.dirty) {
            home->second.dirty = true;
            dirty_.push_back(page->second);
            dirty_count_.store(dirty_.size(), std::memory_order_release);
        }
        return true;
    }
    // Fills the homes dirty() (and deferred map()) named. Returns how many.
    // For batches >= kParallelThreshold, fans out per-page guarded copy + 256-byte piece hashing + coalesced home
    // writes across the internal worker pool (each worker owns disjoint homes and scratch slots), then accumulates
    // counters, notifies `on_changed(frame, changed_pieces)` for each home that actually modified >= 1 piece, and runs
    // filtered-heap callbacks sequentially on the calling thread in dirty_ order.
    template <class OnChanged>
    std::size_t upload(OnChanged&& on_changed) {
        const std::uint64_t uploaded_before = counters_.uploaded_bytes;
        if (dirty_.empty()) {
            last_upload_bytes_ = 0;
            return 0;
        }
        std::vector<std::uint8_t*> batch_frames;
        batch_frames.swap(dirty_);
        dirty_count_.store(0, std::memory_order_release);

        std::size_t n = 0;
        if (!parallel_enabled_ || batch_frames.size() < kParallelThreshold) {
            for (std::uint8_t* frame : batch_frames) {
                const auto home = homes_.find(frame);
                if (home == homes_.end() || !home->second.dirty) continue;
                home->second.dirty = false;
                const bool whole = std::exchange(home->second.whole_fill, false);
                std::array<std::uint64_t, 4> changed_pieces{};
                std::size_t changed = 0;
                fill_with_mask(home->second, frame, whole, changed_pieces, changed);
                if (changed) on_changed(frame, changed_pieces);
                ++n;
            }
        } else {
            upload_jobs_.clear();
            upload_jobs_.reserve(batch_frames.size());
            std::size_t aliased_count = 0;
            for (std::uint8_t* frame : batch_frames) {
                const auto home = homes_.find(frame);
                if (home == homes_.end() || !home->second.dirty) continue;
                home->second.dirty = false;
                UploadJob job;
                job.frame = frame;
                job.to = units_[home->second.unit].host + std::size_t(home->second.slot) * kPage;
                job.sums = home->second.sums.get();
                job.whole = std::exchange(home->second.whole_fill, false);
                if (home->second.copy) job.aliased_slot = aliased_count++;
                upload_jobs_.push_back(job);
            }
            n = upload_jobs_.size();
            if (n < kParallelThreshold || !ensure_pool()) {
                for (const UploadJob& job : upload_jobs_) {
                    const auto it = homes_.find(job.frame);
                    if (it == homes_.end()) continue;
                    std::array<std::uint64_t, 4> changed_pieces{};
                    std::size_t changed = 0;
                    fill_with_mask(it->second, job.frame, job.whole, changed_pieces, changed);
                    if (changed) on_changed(job.frame, changed_pieces);
                }
            } else {
                if (aliased_count > 0) {
                    const std::size_t need_bytes = aliased_count * kPage;
                    if (aliased_scratch_.size() < need_bytes) aliased_scratch_.resize(need_bytes);
                }
                const std::size_t workers = pool_->worker_count();
                const std::size_t chunk = (std::max)(std::size_t(1), (std::min)(std::size_t(4), n / (workers * 2)));
                pool_->run(n, chunk, [this](std::size_t b, std::size_t e, std::size_t tid) {
                    std::uint8_t* const worker_scratch = pool_->scratch(tid);
                    for (std::size_t i = b; i < e; ++i) {
                        UploadJob& job = upload_jobs_[i];
                        std::uint8_t* const scratch =
                            (job.aliased_slot != kNoAliasedSlot)
                                ? (aliased_scratch_.data() + job.aliased_slot * kPage)
                                : worker_scratch;
                        job.failed = sync_page_into_scratch(job.to, scratch, job.sums, job.frame, job.whole,
                                                            job.changed_pieces, job.changed);
                    }
                });
                for (const UploadJob& job : upload_jobs_) {
                    counters_.uploaded_bytes += std::uint64_t(job.changed) * kPiece;
                    if (job.failed) ++counters_.upload_failures;
                    ++counters_.uploads;
                    if (job.changed) on_changed(job.frame, job.changed_pieces);
                    if (job.aliased_slot != kNoAliasedSlot) {
                        // Re-lookup by frame key: filter_pieces may re-entrantly map() and rehash homes_.
                        const auto it = homes_.find(job.frame);
                        if (it != homes_.end() && it->second.copy &&
                            (job.changed || it->second.fresh || it->second.refilter)) {
                            fill_copy(it->second, aliased_scratch_.data() + job.aliased_slot * kPage, false,
                                      &job.changed_pieces);
                        }
                    }
                }
            }
        }
        if (dirty_.empty()) {
            batch_frames.clear();
            dirty_.swap(batch_frames);  // keep vector capacity for the next refresh
        }
        last_upload_bytes_ = counters_.uploaded_bytes - uploaded_before;
        return n;
    }
    std::size_t upload() {
        return upload([](std::uint8_t*, const std::array<std::uint64_t, 4>&) {});
    }

    // A mirrored page snapshotted for GPU write-back readback: carries its chunk location and the physical frame it
    // showed at snapshot time, so the caller can release the mirror mutex during the GPU copy and write_back_batch()
    // can skip any page remapped or given a pending bind meanwhile.
    struct HomeRef {
        std::uint64_t page = 0;
        VkBuffer chunk = VK_NULL_HANDLE;
        VkDeviceSize offset = 0;
        std::uint8_t* frame = nullptr;
    };

    // True when the 64 KiB page at `va` has an unflushed bind/unbind in pending_: the sparse chunk on the GPU does not
    // yet reflect pages_[va], so reading back from chunk_at(va) would read the previous binding (or zeros).
    bool has_pending(std::uint64_t va) const {
        const std::uint64_t page = va & ~(kPage - 1);
        const std::uint64_t chunk = page / kChunk;
        const auto it = pending_.find(chunk);
        return it != pending_.end() && it->second.count(page - chunk * kChunk) != 0;
    }

    // The mirrored pages of [begin, end) that have a flushed home, in address order: what write_back reads. Pages with
    // an unflushed bind in pending_ are skipped so the caller never copies a stale chunk binding.
    void homes_in(std::uint64_t begin, std::uint64_t end, std::vector<std::uint64_t>& out) const {
        if (begin >= end) return;
        for (std::uint64_t page = begin & ~(kPage - 1); page < end; page += kPage) {
            if (!has_pending(page)) {
                const auto mapped = pages_.find(page);
                if (mapped != pages_.end() && homes_.count(mapped->second)) out.push_back(page);
            }
            if (page > UINT64_MAX - kPage) break;
        }
    }

    // Deduplicated, page-sorted HomeRefs for every flushed mirrored page overlapping any [begin, end) in `ranges`.
    std::vector<HomeRef> homes_in_ranges(const std::vector<std::pair<std::uint64_t, std::uint64_t>>& ranges) const {
        std::vector<HomeRef> out;
        if (ranges.empty()) return out;
        std::vector<std::pair<std::uint64_t, std::uint64_t>> owned;
        const auto& merged = is_sorted_disjoint_ranges(ranges) ? ranges : (owned = merge_ranges(ranges));
        for (const auto& [begin, end] : merged) {
            for (std::uint64_t page = begin & ~(kPage - 1); page < end; page += kPage) {
                if ((out.empty() || page > out.back().page) && !has_pending(page)) {
                    const auto mapped = pages_.find(page);
                    VkBuffer chunk = VK_NULL_HANDLE;
                    VkDeviceSize offset = 0;
                    if (mapped != pages_.end() && homes_.count(mapped->second) && chunk_at(page, chunk, offset))
                        out.push_back({page, chunk, offset, mapped->second});
                }
                if (page > UINT64_MAX - kPage) break;
            }
        }
        return out;
    }

    // Lock-free Phase 1 output for async readback: per-page scan mask and precomputed 256-byte GPU piece hashes over
    // the readback staging buffer. Computed entirely outside `page_mirror.mutex`, then consumed in O(pages) under the
    // mirror lock by write_back_prehashed().
    struct PrehashedReadback {
        struct PagePlan {
            std::array<std::uint64_t, 4> scan_mask{};
            bool full_page = false;
            bool any_piece = false;
        };
        std::vector<PagePlan> plans;
        std::vector<std::uint64_t> sums;  // count * kPieces
    };

    static void prehash_readback_slice(const HomeRef* slice, std::size_t count,
                                       const std::vector<std::pair<std::uint64_t, std::uint64_t>>& ranges,
                                       const std::uint8_t* staged, PrehashedReadback& out,
                                       WorkerPool* pool = nullptr) {
        out.plans.clear();
        if (!slice || !count || !staged || ranges.empty()) return;
        std::vector<std::pair<std::uint64_t, std::uint64_t>> owned;
        const auto& merged = is_sorted_disjoint_ranges(ranges) ? ranges : (owned = merge_ranges(ranges));
        if (merged.empty()) return;

        out.plans.assign(count, PrehashedReadback::PagePlan{});
        std::size_t active_pages = 0;
        std::size_t cursor = 0;
        for (std::size_t i = 0; i < count; ++i) {
            const HomeRef& ref = slice[i];
            if (i > 0 && ref.page < slice[i - 1].page) cursor = 0;
            while (cursor < merged.size() && merged[cursor].second <= ref.page) ++cursor;
            if (cursor >= merged.size()) break;
            if (merged[cursor].first >= ref.page + kPage) continue;
            auto& plan = out.plans[i];
            std::size_t next_k = 0;
            std::size_t piece_count = 0;
            for (std::size_t r = cursor; r < merged.size() && merged[r].first < ref.page + kPage; ++r) {
                const std::uint64_t lo = (std::max)(merged[r].first, ref.page);
                const std::uint64_t hi = (std::min)(merged[r].second, ref.page + kPage);
                if (lo >= hi) continue;
                const std::size_t k0 = (std::max)(next_k, std::size_t((lo - ref.page) / kPiece));
                const std::size_t k1 = std::size_t((hi - ref.page + kPiece - 1) / kPiece);
                if (k0 < k1) {
                    set_piece_range(plan.scan_mask, k0, k1);
                    piece_count += k1 - k0;
                    next_k = k1;
                }
            }
            if (piece_count) {
                plan.any_piece = true;
                plan.full_page = (piece_count == kPieces);
                ++active_pages;
            }
        }
        if (!active_pages) return;

        const std::size_t need_sums = count * kPieces;
        if (out.sums.size() < need_sums) out.sums.resize(need_sums);

        const auto hash_range = [&](std::size_t b, std::size_t e) {
            for (std::size_t i = b; i < e; ++i) {
                const auto& plan = out.plans[i];
                if (!plan.any_piece) continue;
                const std::uint8_t* const gpu = staged + i * kPage;
                std::uint64_t* const dst = out.sums.data() + i * kPieces;
                if (plan.full_page) {
                    for (std::size_t k = 0; k < kPieces; ++k) dst[k] = shadow_keep::sum_of(gpu + k * kPiece, kPiece);
                } else {
                    for (std::size_t w = 0; w < 4; ++w) {
                        std::uint64_t m = plan.scan_mask[w];
                        while (m) {
                            const unsigned bit = static_cast<unsigned>(std::countr_zero(m));
                            m &= m - 1;
                            const std::size_t k = w * 64 + bit;
                            dst[k] = shadow_keep::sum_of(gpu + k * kPiece, kPiece);
                        }
                    }
                }
            }
        };

        if (pool && active_pages >= kParallelThreshold) {
            const std::size_t workers = pool->worker_count();
            const std::size_t chunk = (std::max)(std::size_t(1), (std::min)(std::size_t(4), count / (workers * 2)));
            pool->run(count, chunk, [&](std::size_t b, std::size_t e, std::size_t) { hash_range(b, e); });
        } else {
            hash_range(0, count);
        }
    }

    // Phase 2 of lock-free readback: compares precomputed GPU piece sums against live `home.sums[]` under the mirror
    // lock (O(pieces) uint64 compares, zero GPU-buffer hashing) and writes back coalesced non-conflicting runs.
    // `skip_masks` (optional, size `count`): per-slice 256-bit masks of pieces concurrently overwritten by upload()
    // while the GPU copy was in flight (CPU wins under the conflict rule; skipped without retrying the page).
    // `skip_page` (optional, size `count`): non-zero entries are skipped whole (e.g. externally invalidated frames).
    template <class Write>
    std::uint64_t write_back_prehashed(const HomeRef* slice, std::size_t count, const PrehashedReadback& prehashed,
                                       const std::uint8_t* staged, const std::array<std::uint64_t, 4>* skip_masks,
                                       const std::uint8_t* skip_page, Write&& write) {
        if (!slice || !count || !staged || prehashed.plans.size() < count) return 0;
        std::uint64_t written = 0;
        std::vector<std::uint8_t*> refilter_frames;
        for (std::size_t i = 0; i < count; ++i) {
            if (skip_page && skip_page[i]) continue;
            const auto& plan = prehashed.plans[i];
            if (!plan.any_piece) continue;
            const HomeRef& ref = slice[i];
            if (has_pending(ref.page)) continue;
            const auto mapped = pages_.find(ref.page);
            if (mapped == pages_.end() || mapped->second != ref.frame) continue;
            const auto found = homes_.find(mapped->second);
            if (found == homes_.end()) continue;

            const std::uint64_t* const gpu_sums = prehashed.sums.data() + i * kPieces;
            const std::uint64_t* const agreed = found->second.sums.get();
            const std::array<std::uint64_t, 4>* const sm = skip_masks ? &skip_masks[i] : nullptr;

            std::array<std::uint64_t, 4> diff_mask{};
            std::size_t diff_count = 0;
            if (plan.full_page && (!sm || ((*sm)[0] | (*sm)[1] | (*sm)[2] | (*sm)[3]) == 0)) {
                for (std::size_t k = 0; k < kPieces; ++k) {
                    if (gpu_sums[k] != agreed[k]) {
                        diff_mask[k >> 6] |= std::uint64_t(1) << (k & 63);
                        ++diff_count;
                    }
                }
            } else {
                for (std::size_t w = 0; w < 4; ++w) {
                    std::uint64_t m = plan.scan_mask[w] & ~(sm ? (*sm)[w] : 0ull);
                    while (m) {
                        const unsigned bit = static_cast<unsigned>(std::countr_zero(m));
                        m &= m - 1;
                        const std::size_t k = w * 64 + bit;
                        if (gpu_sums[k] != agreed[k]) {
                            diff_mask[w] |= std::uint64_t(1) << bit;
                            ++diff_count;
                        }
                    }
                }
            }
            if (!diff_count) continue;
            written += write_back_diff_coalesced(found->second, mapped->second, staged + i * kPage, gpu_sums,
                                                 diff_mask, diff_count, write, refilter_frames);
        }
        finish_write_back(written, refilter_frames);
        return written;
    }

    // The home of the page at `va`, mapped (its host bytes): the title's bytes as the GPU reads them at the title's
    // address. Null when the page is not mapped.
    std::uint8_t* home_bytes(std::uint64_t va) {
        const auto page = pages_.find(va & ~(kPage - 1));
        if (page == pages_.end()) return nullptr;
        const auto home = homes_.find(page->second);
        if (home == homes_.end()) return nullptr;
        return units_[home->second.unit].host + std::size_t(home->second.slot) * kPage;
    }

    // What the GPU wrote into the homes of the mirrored pages of [begin, end) goes back to each page's physical frame,
    // merged at the 256-byte piece against the sums both sides last agreed on: a piece only the GPU changed is written
    // back with `write(frame_dst, bytes, n)` (and agreed on again); a piece only the title changed waits for the
    // upload; a piece both changed keeps the title's bytes and is counted. `write` receives the physical frame address
    // (`mapped->second + k * kPiece`, identical to the title VA for ordinary memory and the writable scratch view for
    // xmem physical pages), so a concurrent CPU remap of `va` never receives the previous mapping's bytes. Pages with
    // an unflushed bind in pending_ are skipped. `staged(page)` gives the page's home as the GPU left it, or nullptr to
    // skip the page. Returns the bytes written back.
    // The same, reading each home where it lies (host memory: the tests' homes; the device copies its homes out first).
    template <class Write>
    std::uint64_t write_back(std::uint64_t begin, std::uint64_t end, Write&& write) {
        return write_back(begin, end, [this](std::uint64_t page) -> const std::uint8_t* { return home_bytes(page); },
                          std::forward<Write>(write));
    }
    template <class Staged, class Write>
    std::uint64_t write_back(std::uint64_t begin, std::uint64_t end, Staged&& staged, Write&& write) {
        if (begin >= end) return 0;
        std::uint64_t written = 0;
        std::vector<std::uint8_t*> refilter_frames;
        for (std::uint64_t page = begin & ~(kPage - 1); page < end; page += kPage) {
            if (!has_pending(page)) {
                const auto mapped = pages_.find(page);
                if (mapped != pages_.end()) {
                    const auto found = homes_.find(mapped->second);
                    if (found != homes_.end()) {
                        if (const std::uint8_t* gpu = staged(page)) {
                            const std::size_t k0 = std::size_t(((std::max)(begin, page) - page) / kPiece);
                            const std::size_t k1 = std::size_t(((std::min)(end, page + kPage) - page + kPiece - 1) / kPiece);
                            written += write_back_pieces(found->second, mapped->second, gpu, k0, k1, write,
                                                         refilter_frames);
                        }
                    }
                }
            }
            if (page > UINT64_MAX - kPage) break;
        }
        finish_write_back(written, refilter_frames);
        return written;
    }

    // Batched multi-range write-back over `count` snapshotted HomeRefs whose GPU copies lie contiguously at
    // `staged + i * kPage`. Skips any page that now has a pending bind or was remapped to a different physical frame
    // since `homes_in_ranges()` snapshotted it.
    // For batches >= kParallelThreshold, scans 256-byte piece hashes read-only across the internal worker pool, then
    // applies only changed pieces (conflict checks, write() callbacks, stats, and refilters) sequentially in exact
    // batch order on the calling thread.
    template <class Write>
    std::uint64_t write_back_batch(const HomeRef* batch, std::size_t count,
                                   const std::vector<std::pair<std::uint64_t, std::uint64_t>>& ranges,
                                   const std::uint8_t* staged, Write&& write) {
        if (!batch || !count || !staged || ranges.empty()) return 0;
        std::vector<std::pair<std::uint64_t, std::uint64_t>> owned;
        const auto& merged = is_sorted_disjoint_ranges(ranges) ? ranges : (owned = merge_ranges(ranges));
        if (merged.empty()) return 0;
        std::uint64_t written = 0;
        std::vector<std::uint8_t*> refilter_frames;
        if (!parallel_enabled_ || count < kParallelThreshold) {
            std::size_t cursor = 0;
            for (std::size_t i = 0; i < count; ++i) {
                const HomeRef& ref = batch[i];
                if (i > 0 && ref.page < batch[i - 1].page) cursor = 0;
                while (cursor < merged.size() && merged[cursor].second <= ref.page) ++cursor;
                if (cursor >= merged.size()) break;
                if (merged[cursor].first >= ref.page + kPage) continue;
                if (has_pending(ref.page)) continue;
                const auto mapped = pages_.find(ref.page);
                if (mapped == pages_.end() || mapped->second != ref.frame) continue;
                const auto found = homes_.find(mapped->second);
                if (found == homes_.end()) continue;
                const std::uint8_t* gpu = staged + i * kPage;
                std::size_t next_k = 0;
                for (std::size_t r = cursor; r < merged.size() && merged[r].first < ref.page + kPage; ++r) {
                    const std::uint64_t lo = (std::max)(merged[r].first, ref.page);
                    const std::uint64_t hi = (std::min)(merged[r].second, ref.page + kPage);
                    if (lo >= hi) continue;
                    const std::size_t k0 = (std::max)(next_k, std::size_t((lo - ref.page) / kPiece));
                    const std::size_t k1 = std::size_t((hi - ref.page + kPiece - 1) / kPiece);
                    if (k0 < k1) {
                        written += write_back_pieces(found->second, mapped->second, gpu, k0, k1, write, refilter_frames);
                        next_k = k1;
                    }
                }
            }
            finish_write_back(written, refilter_frames);
            return written;
        }

        wb_jobs_.clear();
        wb_jobs_.reserve(count);
        bool any_shared_home = false;
        std::size_t cursor = 0;
        for (std::size_t i = 0; i < count; ++i) {
            const HomeRef& ref = batch[i];
            if (i > 0 && ref.page < batch[i - 1].page) cursor = 0;
            while (cursor < merged.size() && merged[cursor].second <= ref.page) ++cursor;
            if (cursor >= merged.size()) break;
            if (merged[cursor].first >= ref.page + kPage) continue;
            if (has_pending(ref.page)) continue;
            const auto mapped = pages_.find(ref.page);
            if (mapped == pages_.end() || mapped->second != ref.frame) continue;
            const auto found = homes_.find(mapped->second);
            if (found == homes_.end()) continue;

            WriteBackJob job;
            job.home = &found->second;
            job.frame = mapped->second;
            job.gpu = staged + i * kPage;
            job.sums = found->second.sums.get();
            if (found->second.refs > 1) any_shared_home = true;

            std::size_t next_k = 0;
            std::size_t piece_count = 0;
            for (std::size_t r = cursor; r < merged.size() && merged[r].first < ref.page + kPage; ++r) {
                const std::uint64_t lo = (std::max)(merged[r].first, ref.page);
                const std::uint64_t hi = (std::min)(merged[r].second, ref.page + kPage);
                if (lo >= hi) continue;
                const std::size_t k0 = (std::max)(next_k, std::size_t((lo - ref.page) / kPiece));
                const std::size_t k1 = std::size_t((hi - ref.page + kPiece - 1) / kPiece);
                if (k0 < k1) {
                    set_piece_range(job.scan_mask, k0, k1);
                    piece_count += k1 - k0;
                    next_k = k1;
                }
            }
            if (!piece_count) continue;
            job.full_page = (piece_count == kPieces);
            wb_jobs_.push_back(job);
        }

        const std::size_t n_jobs = wb_jobs_.size();
        if (!n_jobs) return 0;

        if (n_jobs < kParallelThreshold || !ensure_pool()) {
            for (const WriteBackJob& job : wb_jobs_) {
                written += write_back_masked(*job.home, job.frame, job.gpu, job.scan_mask, job.full_page, write,
                                             refilter_frames);
            }
            finish_write_back(written, refilter_frames);
            return written;
        }

        // Phase 1 (parallel, read-only over staged and home.sums[]): compute 256-bit diff_mask per page.
        const std::size_t workers = pool_->worker_count();
        const std::size_t chunk = (std::max)(std::size_t(1), (std::min)(std::size_t(4), n_jobs / (workers * 2)));
        pool_->run(n_jobs, chunk, [this](std::size_t b, std::size_t e, std::size_t) {
            for (std::size_t i = b; i < e; ++i) {
                WriteBackJob& job = wb_jobs_[i];
                std::uint16_t diffs = 0;
                if (job.full_page) {
                    for (std::size_t k = 0; k < kPieces; ++k) {
                        if (shadow_keep::sum_of(job.gpu + k * kPiece, kPiece) != job.sums[k]) {
                            job.diff_mask[k >> 6] |= std::uint64_t(1) << (k & 63);
                            ++diffs;
                        }
                    }
                } else {
                    for (std::size_t w = 0; w < 4; ++w) {
                        std::uint64_t m = job.scan_mask[w];
                        while (m) {
                            const unsigned bit = static_cast<unsigned>(std::countr_zero(m));
                            m &= m - 1;
                            const std::size_t k = w * 64 + bit;
                            if (shadow_keep::sum_of(job.gpu + k * kPiece, kPiece) != job.sums[k]) {
                                job.diff_mask[w] |= std::uint64_t(1) << bit;
                                ++diffs;
                            }
                        }
                    }
                }
                job.diff_count = diffs;
            }
        });

        // Phase 2 (sequential on calling thread in exact batch order): apply only changed pieces.
        // If two HomeRefs in the batch alias the same physical frame and an earlier one updated home.sums[], fall back
        // to scanning the later alias's scan_mask sequentially so conflict resolution matches single-threaded order.
        std::vector<std::uint8_t*> modified_shared_frames;
        for (const WriteBackJob& job : wb_jobs_) {
            const bool frame_modified_earlier =
                any_shared_home && !modified_shared_frames.empty() &&
                std::find(modified_shared_frames.begin(), modified_shared_frames.end(), job.frame) !=
                    modified_shared_frames.end();
            if (!job.diff_count && !frame_modified_earlier) continue;
            const std::uint64_t page_written =
                frame_modified_earlier
                    ? write_back_masked(*job.home, job.frame, job.gpu, job.scan_mask, job.full_page, write,
                                        refilter_frames)
                    : write_back_masked(*job.home, job.frame, job.gpu, job.diff_mask, false, write, refilter_frames);
            if (page_written) {
                written += page_written;
                if (any_shared_home && job.home->refs > 1 && !frame_modified_earlier)
                    modified_shared_frames.push_back(job.frame);
            }
        }
        finish_write_back(written, refilter_frames);
        return written;
    }
    template <class Write>
    std::uint64_t write_back_batch(const std::vector<HomeRef>& batch,
                                   const std::vector<std::pair<std::uint64_t, std::uint64_t>>& ranges,
                                   const std::uint8_t* staged, Write&& write) {
        return write_back_batch(batch.data(), batch.size(), ranges, staged, std::forward<Write>(write));
    }

    // The binds noted since the last flush, as one batch signalling `value`. False when there were none (nothing to
    // wait for) or the bind failed (counted; the caller logs).
    bool flush(std::uint64_t value) {
        if (pending_.empty() && alias_pending_.empty()) return false;
        std::size_t binds = 0;
        for (const auto& [chunk, by_offset] : pending_) binds += by_offset.size();
        for (const auto& [buffer, by_offset] : alias_pending_) binds += by_offset.size();
        last_batch_ = binds;
        // The last bind of a page in this batch is the one that counts: binds of one range in one batch are not
        // ordered by the API.
        std::vector<std::vector<VkSparseMemoryBind>> lists;
        std::vector<VkSparseBufferMemoryBindInfo> infos;
        lists.reserve(pending_.size() + alias_pending_.size());
        for (auto& [chunk, by_offset] : pending_) {
            auto& list = lists.emplace_back();
            list.reserve(by_offset.size());
            for (const auto& [offset, bind] : by_offset) list.push_back(bind);
            infos.push_back({chunks_[chunk], std::uint32_t(list.size()), list.data()});
        }
        // The aliases' binds in the same batch, so the one wait orders a reader of either address.
        for (auto& [buffer, by_offset] : alias_pending_) {
            auto& list = lists.emplace_back();
            list.reserve(by_offset.size());
            for (const auto& [offset, bind] : by_offset) list.push_back(bind);
            infos.push_back({buffer, std::uint32_t(list.size()), list.data()});
        }
        const bool ok = hooks_.bind && hooks_.bind(infos, value);
        ++counters_.batches;
        pending_.clear();
        alias_pending_.clear();
        has_pending_binds_.store(false, std::memory_order_release);
        return ok;
    }

    // Stamps every unstamped freed home (queued by map()/unalias() with kUnstamped) with `ticket`, taken after
    // flush() published the sparse unbind via note_mirror_bind(). Unstamped homes are never retired until stamped.
    void stamp_freed(std::uint64_t ticket) {
        if (unstamped_freed_.empty()) return;
        for (const auto& [unit, slot] : unstamped_freed_) freed_.push_back({unit, slot, ticket});
        unstamped_freed_.clear();
        sync_freed_atomics();
    }
    std::size_t unstamped_freed_count() const noexcept { return unstamped_freed_.size(); }

    // Frees the homes no page binds any more whose last reader has retired (`completed` ticket).
    void retire(std::uint64_t completed) {
        if (freed_.empty()) return;
        bool popped = false;
        while (!freed_.empty() && freed_.front().ticket <= completed) {
            const auto [unit, slot, ticket] = freed_.front();
            freed_.pop_front();
            units_[unit].free.push_back(slot);
            ++counters_.homes_freed;
            popped = true;
        }
        if (popped) sync_freed_atomics();
    }

    // Lock-free O(1) clean fast-path queries so before_execute_() / flush_page_mirror() skip taking page_mirror.mutex
    // and completed_retire_ticket() when nothing is pending.
    bool has_pending_binds() const noexcept { return has_pending_binds_.load(std::memory_order_acquire); }
    bool has_dirty() const noexcept { return dirty_count_.load(std::memory_order_acquire) != 0; }
    bool has_pending_work() const noexcept { return has_pending_binds() || has_dirty(); }
    bool has_retired_homes() const noexcept { return freed_count_.load(std::memory_order_acquire) != 0; }
    bool has_retired_homes(std::uint64_t completed) const noexcept {
        return freed_count_.load(std::memory_order_acquire) != 0 &&
               oldest_freed_ticket_.load(std::memory_order_acquire) <= completed;
    }
    std::uint64_t last_upload_bytes() const noexcept { return last_upload_bytes_; }

    // The home's memory and offset of the mirrored page at `va` (tests and diagnostics).
    bool home_at(std::uint64_t va, VkDeviceMemory& memory, VkDeviceSize& offset, std::uint8_t** host = nullptr) const {
        const auto page = pages_.find(va & ~(kPage - 1));
        if (page == pages_.end()) return false;
        const auto home = homes_.find(page->second);
        if (home == homes_.end()) return false;
        memory = units_[home->second.unit].memory;
        offset = VkDeviceSize(home->second.slot) * kPage;
        if (host) *host = units_[home->second.unit].host + offset;
        return true;
    }
    bool mirrored(std::uint64_t va) const { return pages_.count(va & ~(kPage - 1)) != 0; }
    // Mixes the cached 256-byte piece hashes (Home::sums[]) for every 256-byte piece overlapping [va, va + bytes).
    // Returns false when bytes == 0, the range overflows, or any 64 KiB page in [va, va + bytes) is not mirrored or
    // has no home. Zero reads of title or device memory: O(overlapping 256-byte pieces).
    bool range_sum(std::uint64_t va, std::uint64_t bytes, std::uint64_t& out_hash) const {
        if (!bytes || va + bytes < va) return false;
        const std::uint64_t end = va + bytes;
        std::uint64_t h = 14695981039346656037ull ^ bytes;
        for (std::uint64_t page = va & ~(kPage - 1); page < end; page += kPage) {
            const auto mapped = pages_.find(page);
            if (mapped == pages_.end()) return false;
            const auto found = homes_.find(mapped->second);
            if (found == homes_.end() || found->second.whole_fill) return false;
            const std::uint64_t page_end = (page <= UINT64_MAX - kPage) ? (page + kPage) : UINT64_MAX;
            const std::uint64_t lo = (std::max)(va, page);
            const std::uint64_t hi = (std::min)(end, page_end);
            const std::size_t k0 = std::size_t((lo - page) / kPiece);
            const std::size_t k1 = std::size_t((hi - page + kPiece - 1) / kPiece);
            const std::uint64_t* const sums = found->second.sums.get();
            for (std::size_t k = k0; k < k1; ++k) {
                h = (h ^ sums[k]) * 1099511628211ull;
                h ^= h >> 29;
            }
            if (page > UINT64_MAX - kPage) break;
        }
        out_hash = h;
        return true;
    }
    // The physical page's bytes the mirrored page at `va` shows (its frame; for ordinary memory the page itself).
    std::uint8_t* frame_at(std::uint64_t va) const {
        const auto page = pages_.find(va & ~(kPage - 1));
        return page == pages_.end() ? nullptr : page->second;
    }
    // The chunk over `va` and the offset of `va` in it (tests and diagnostics).
    bool chunk_at(std::uint64_t va, VkBuffer& buffer, VkDeviceSize& offset) const {
        const auto it = chunks_.find(va / kChunk);
        if (it == chunks_.end() || it->second == VK_NULL_HANDLE) return false;
        buffer = it->second;
        offset = va - it->first * kChunk;
        return true;
    }

    Counters counters() const {
        Counters c = counters_;
        c.homes = homes_.size();
        c.units = units_.size();
        c.pages = pages_.size();
        return c;
    }

private:
    // A home and the sum of each of its 256-byte pieces as both sides last agreed on it (src/shadow_keep.h): what was
    // filled from the title's page, or written back to it. A piece whose home differs from its sum was written by the
    // GPU, one whose title page differs by the title.
    static constexpr std::size_t kPiece = shadow_keep::kPiece, kPieces = kPage / kPiece;
    struct Home {
        std::uint32_t unit = 0, slot = 0, refs = 0;
        std::shared_ptr<void> keep;
        bool dirty = false;
        bool whole_fill = false;  // deferred initial fill from map(..., defer_fill=true)
        std::unique_ptr<std::uint64_t[]> sums = std::make_unique<std::uint64_t[]>(kPieces);
        // The filtered copy (alias): its own slot, bound at the alias of `filter_va`, the aliased page it was made for.
        bool copy = false;
        std::uint32_t copy_unit = 0, copy_slot = 0;
        std::uint64_t filter_va = 0;
        bool fresh = true;       // the copy's next fill writes every byte (hooks.filter's `fresh`)
        bool refilter = false;   // the title's bytes changed outside an upload (write_back): fill the copy
    };
    struct Alias {
        std::uint64_t hi;
        VkBuffer buffer;
        VkDeviceSize offset;
    };
    struct Unit {
        VkDeviceMemory memory = VK_NULL_HANDLE;
        std::uint8_t* host = nullptr;
        std::vector<std::uint32_t> free;
    };
    struct Freed {
        std::uint32_t unit, slot;
        std::uint64_t ticket;
    };

    static bool is_sorted_disjoint_ranges(const std::vector<std::pair<std::uint64_t, std::uint64_t>>& ranges) noexcept {
        for (std::size_t i = 0; i < ranges.size(); ++i) {
            if (ranges[i].first >= ranges[i].second) return false;
            if (i > 0 && ranges[i].first <= ranges[i - 1].second) return false;
        }
        return true;
    }

    static std::vector<std::pair<std::uint64_t, std::uint64_t>> merge_ranges(
        const std::vector<std::pair<std::uint64_t, std::uint64_t>>& ranges) {
        std::vector<std::pair<std::uint64_t, std::uint64_t>> sorted;
        sorted.reserve(ranges.size());
        for (const auto& [b, e] : ranges)
            if (b < e) sorted.emplace_back(b, e);
        if (sorted.size() <= 1) return sorted;
        std::sort(sorted.begin(), sorted.end());
        std::vector<std::pair<std::uint64_t, std::uint64_t>> merged;
        merged.reserve(sorted.size());
        for (const auto& [b, e] : sorted) {
            if (!merged.empty() && b <= merged.back().second) {
                if (e > merged.back().second) merged.back().second = e;
            } else {
                merged.emplace_back(b, e);
            }
        }
        return merged;
    }

    static void set_piece_range(std::array<std::uint64_t, 4>& mask, std::size_t k0, std::size_t k1) noexcept {
        if (k0 >= k1) return;
        const std::size_t w0 = k0 >> 6;
        const std::size_t w1 = (k1 - 1) >> 6;
        const std::uint64_t lo_mask = ~std::uint64_t(0) << (k0 & 63);
        const std::size_t hi_bits = k1 & 63;
        const std::uint64_t hi_mask = hi_bits ? ((std::uint64_t(1) << hi_bits) - 1) : ~std::uint64_t(0);
        if (w0 == w1) {
            mask[w0] |= lo_mask & hi_mask;
        } else {
            mask[w0] |= lo_mask;
            for (std::size_t w = w0 + 1; w < w1; ++w) mask[w] = ~std::uint64_t(0);
            mask[w1] |= hi_mask;
        }
    }

    // Conflict-checks the GPU-differing pieces in `diff_mask` against the title's physical frame (using a single
    // whole-page guarded copy when >= 16 pieces differ on the page) and coalesces contiguous non-conflicting pieces
    // into single `write(frame_dst, gpu_src, run_bytes)` calls (falling back piece-by-piece if a multi-piece write
    // fails). `pre_sums` holds the already-computed GPU piece sums for every bit set in `diff_mask`.
    template <class Write>
    std::uint64_t write_back_diff_coalesced(Home& home, std::uint8_t* frame, const std::uint8_t* gpu,
                                            const std::uint64_t* pre_sums,
                                            const std::array<std::uint64_t, 4>& diff_mask, std::size_t diff_count,
                                            Write&& write, std::vector<std::uint8_t*>& refilter_frames) {
        if (!diff_count) return 0;
        bool whole_title_ok = false;
        if (diff_count >= 16) {
            whole_title_ok = hooks_.copy ? hooks_.copy(fill_scratch_.data(), frame, kPage)
                                         : (std::memcpy(fill_scratch_.data(), frame, kPage), true);
        }
        std::array<std::uint64_t, 4> write_mask{};
        std::size_t write_count = 0;
        alignas(64) std::uint8_t title_piece[kPiece];
        for (std::size_t w = 0; w < 4; ++w) {
            std::uint64_t m = diff_mask[w];
            while (m) {
                const unsigned bit = static_cast<unsigned>(std::countr_zero(m));
                m &= m - 1;
                const std::size_t k = w * 64 + bit;
                const std::uint64_t agreed = home.sums[k];
                if (pre_sums[k] == agreed) continue;
                const std::uint8_t* title_ptr = nullptr;
                if (whole_title_ok) {
                    title_ptr = fill_scratch_.data() + k * kPiece;
                } else {
                    const std::uint8_t* from = frame + k * kPiece;
                    if (!(hooks_.copy ? hooks_.copy(title_piece, from, kPiece)
                                      : (std::memcpy(title_piece, from, kPiece), true)))
                        continue;  // the title's page went away
                    title_ptr = title_piece;
                }
                if (shadow_keep::sum_of(title_ptr, kPiece) != agreed) {
                    ++counters_.conflicts;  // both changed it: the title's bytes win at the next upload
                    continue;
                }
                write_mask[w] |= std::uint64_t(1) << bit;
                ++write_count;
            }
        }
        if (!write_count) return 0;

        std::uint64_t written = 0;
        const auto frame_base = reinterpret_cast<std::uintptr_t>(frame);
        const auto flush_run = [&](std::size_t run_start, std::size_t run_end) {
            const std::size_t run_pieces = run_end - run_start;
            if (write(static_cast<std::uint64_t>(frame_base + run_start * kPiece), gpu + run_start * kPiece,
                      run_pieces * kPiece)) {
                for (std::size_t k = run_start; k < run_end; ++k) home.sums[k] = pre_sums[k];
                written += std::uint64_t(run_pieces) * kPiece;
                return;
            }
            if (run_pieces == 1) return;
            for (std::size_t k = run_start; k < run_end; ++k) {
                if (write(static_cast<std::uint64_t>(frame_base + k * kPiece), gpu + k * kPiece, kPiece)) {
                    home.sums[k] = pre_sums[k];
                    written += kPiece;
                }
            }
        };

        std::size_t run_start = kPieces;
        for (std::size_t w = 0; w < 4; ++w) {
            const std::uint64_t word = write_mask[w];
            if (!word) {
                if (run_start < kPieces) {
                    flush_run(run_start, w * 64);
                    run_start = kPieces;
                }
                continue;
            }
            if (word == ~std::uint64_t(0)) {
                if (run_start == kPieces) run_start = w * 64;
                continue;
            }
            std::size_t bit = 0;
            while (bit < 64) {
                const std::uint64_t rem = word >> bit;
                if (!rem) {
                    if (run_start < kPieces) {
                        flush_run(run_start, w * 64 + bit);
                        run_start = kPieces;
                    }
                    break;
                }
                const std::size_t zeros = std::size_t(std::countr_zero(rem));
                if (zeros > 0) {
                    if (run_start < kPieces) {
                        flush_run(run_start, w * 64 + bit);
                        run_start = kPieces;
                    }
                    bit += zeros;
                }
                const std::size_t ones = std::size_t(std::countr_one(word >> bit));
                if (run_start == kPieces) run_start = w * 64 + bit;
                bit += ones;
            }
        }
        if (run_start < kPieces) flush_run(run_start, kPieces);

        if (written && home.copy && !home.refilter) {
            home.refilter = true;  // the title's bytes changed: its filtered copy follows
            refilter_frames.push_back(frame);
        }
        return written;
    }

    template <class Write>
    std::uint64_t write_back_pieces(Home& home, std::uint8_t* frame, const std::uint8_t* gpu, std::size_t k0,
                                    std::size_t k1, Write&& write, std::vector<std::uint8_t*>& refilter_frames) {
        std::array<std::uint64_t, 4> diff_mask{};
        std::uint64_t gpu_sums[kPieces];
        std::size_t diff_count = 0;
        const std::uint64_t* const agreed = home.sums.get();
        for (std::size_t k = k0; k < k1; ++k) {
            const std::uint64_t now = shadow_keep::sum_of(gpu + k * kPiece, kPiece);
            if (now != agreed[k]) {
                gpu_sums[k] = now;
                diff_mask[k >> 6] |= std::uint64_t(1) << (k & 63);
                ++diff_count;
            }
        }
        if (!diff_count) return 0;
        return write_back_diff_coalesced(home, frame, gpu, gpu_sums, diff_mask, diff_count, write, refilter_frames);
    }

    template <class Write>
    std::uint64_t write_back_masked(Home& home, std::uint8_t* frame, const std::uint8_t* gpu,
                                    const std::array<std::uint64_t, 4>& mask, bool full_page, Write&& write,
                                    std::vector<std::uint8_t*>& refilter_frames) {
        if (full_page) return write_back_pieces(home, frame, gpu, 0, kPieces, write, refilter_frames);
        std::array<std::uint64_t, 4> diff_mask{};
        std::uint64_t gpu_sums[kPieces];
        std::size_t diff_count = 0;
        const std::uint64_t* const agreed = home.sums.get();
        for (std::size_t w = 0; w < 4; ++w) {
            std::uint64_t m = mask[w];
            while (m) {
                const unsigned bit = static_cast<unsigned>(std::countr_zero(m));
                m &= m - 1;
                const std::size_t k = w * 64 + bit;
                const std::uint64_t now = shadow_keep::sum_of(gpu + k * kPiece, kPiece);
                if (now != agreed[k]) {
                    gpu_sums[k] = now;
                    diff_mask[w] |= std::uint64_t(1) << bit;
                    ++diff_count;
                }
            }
        }
        if (!diff_count) return 0;
        return write_back_diff_coalesced(home, frame, gpu, gpu_sums, diff_mask, diff_count, write, refilter_frames);
    }

    void finish_write_back(std::uint64_t written, const std::vector<std::uint8_t*>& refilter_frames) {
        counters_.written_back_bytes += written;
        for (std::uint8_t* frame : refilter_frames) {
            const auto it = homes_.find(frame);
            if (it != homes_.end() && it->second.refilter) sync_and_fill_copy(it->second, frame, false);
        }
    }

    // Coalesces contiguous changed pieces across the 4 x 64-bit mask words into single memcpy bursts (optimal for
    // write-combined device-local homes). Never reads from `to`.
    static void flush_changed_runs(std::uint8_t* to, const std::uint8_t* scratch,
                                   const std::array<std::uint64_t, 4>& changed_pieces, std::size_t changed) noexcept {
        if (changed == kPieces) {
            std::memcpy(to, scratch, kPage);
            return;
        }
        std::size_t run_start = kPieces;
        for (std::size_t w = 0; w < 4; ++w) {
            const std::uint64_t word = changed_pieces[w];
            if (!word) {
                if (run_start < kPieces) {
                    const std::size_t run_end = w * 64;
                    std::memcpy(to + run_start * kPiece, scratch + run_start * kPiece,
                                (run_end - run_start) * kPiece);
                    run_start = kPieces;
                }
                continue;
            }
            if (word == ~std::uint64_t(0)) {
                if (run_start == kPieces) run_start = w * 64;
                continue;
            }
            std::size_t bit = 0;
            while (bit < 64) {
                const std::uint64_t rem = word >> bit;
                if (!rem) {
                    if (run_start < kPieces) {
                        const std::size_t run_end = w * 64 + bit;
                        std::memcpy(to + run_start * kPiece, scratch + run_start * kPiece,
                                    (run_end - run_start) * kPiece);
                        run_start = kPieces;
                    }
                    break;
                }
                const std::size_t zeros = std::size_t(std::countr_zero(rem));
                if (zeros > 0) {
                    if (run_start < kPieces) {
                        const std::size_t run_end = w * 64 + bit;
                        std::memcpy(to + run_start * kPiece, scratch + run_start * kPiece,
                                    (run_end - run_start) * kPiece);
                        run_start = kPieces;
                    }
                    bit += zeros;
                }
                const std::size_t ones = std::size_t(std::countr_one(word >> bit));
                if (run_start == kPieces) run_start = w * 64 + bit;
                bit += ones;
            }
        }
        if (run_start < kPieces) {
            std::memcpy(to + run_start * kPiece, scratch + run_start * kPiece, (kPieces - run_start) * kPiece);
        }
    }

    // Stateless per-page copy + 256-byte piece hash + coalesced WC home flush (safe to run across worker threads on
    // disjoint homes and scratch buffers).
    bool sync_page_into_scratch(std::uint8_t* to, std::uint8_t* scratch, std::uint64_t* sums,
                                const std::uint8_t* frame, bool whole,
                                std::array<std::uint64_t, 4>& changed_pieces, std::size_t& changed) const {
        bool failed = false;
        const bool whole_ok = hooks_.copy ? hooks_.copy(scratch, frame, kPage)
                                          : (std::memcpy(scratch, frame, kPage), true);
        if (!whole_ok) {
            for (std::size_t k = 0; k < kPieces; ++k) {
                std::uint8_t* piece = scratch + k * kPiece;
                if (!hooks_.copy(piece, frame + k * kPiece, kPiece)) {
                    std::memset(piece, 0, kPiece);
                    failed = true;
                }
            }
        }

        changed_pieces = {};
        changed = 0;
        if (whole) {
            for (std::size_t k = 0; k < kPieces; ++k) sums[k] = shadow_keep::sum_of(scratch + k * kPiece, kPiece);
            std::memcpy(to, scratch, kPage);
            changed = kPieces;
            changed_pieces = kAllPieces;
        } else {
            for (std::size_t k = 0; k < kPieces; ++k) {
                const std::uint64_t sum = shadow_keep::sum_of(scratch + k * kPiece, kPiece);
                if (sum == sums[k]) continue;
                sums[k] = sum;
                changed_pieces[k >> 6] |= std::uint64_t(1) << (k & 63);
                ++changed;
            }
            if (changed) flush_changed_runs(to, scratch, changed_pieces, changed);
        }
        return failed;
    }

    // Brings the home up to date with the title's page, piece by piece, into `fill_scratch_`: `whole` (a new home)
    // copies every piece, and otherwise only a piece the title changed since both sides last agreed, so a piece only
    // the GPU wrote keeps the GPU's bytes (and goes back with write_back). A piece both changed takes the title's
    // bytes. Copies the whole 64 KiB page in one hooks_.copy call, falling back to per-piece copies only when the
    // whole-page copy fails, then updates home.sums in a tight pass and flushes changed pieces to `to` in coalesced
    // contiguous runs (optimal for write-combined device-local homes). Never reads from `to`. Returns whether any
    // piece copy failed.
    bool sync_home_from_frame(Home& home, const std::uint8_t* frame, bool whole,
                              std::array<std::uint64_t, 4>& changed_pieces, std::size_t& changed) {
        std::uint8_t* const to = units_[home.unit].host + std::size_t(home.slot) * kPage;
        const bool failed =
            sync_page_into_scratch(to, fill_scratch_.data(), home.sums.get(), frame, whole, changed_pieces, changed);
        counters_.uploaded_bytes += std::uint64_t(changed) * kPiece;
        return failed;
    }

    void fill_with_mask(Home& home, const std::uint8_t* frame, bool whole,
                        std::array<std::uint64_t, 4>& changed_pieces, std::size_t& changed) {
        const bool failed = sync_home_from_frame(home, frame, whole, changed_pieces, changed);
        // The filtered copy follows the title's bytes from the exact same `fill_scratch_` snapshot so `home.sums[]`
        // and the filtered copy never diverge.
        if (home.copy && (changed || home.fresh || home.refilter))
            fill_copy(home, fill_scratch_.data(), false, &changed_pieces);
        if (failed) ++counters_.upload_failures;
        ++counters_.uploads;
    }

    void fill(Home& home, const std::uint8_t* frame, bool whole) {
        std::array<std::uint64_t, 4> changed_pieces{};
        std::size_t changed = 0;
        fill_with_mask(home, frame, whole, changed_pieces, changed);
    }

    // Refreshes `home` and `home.sums[]` for any CPU-changed pieces (preserving GPU-only pieces) and filters the
    // copy from that exact same snapshot (used by make_copy, alias handoff, and finish_write_back).
    void sync_and_fill_copy(Home& home, const std::uint8_t* frame, bool whole_copy) {
        const bool whole_home = std::exchange(home.whole_fill, false);
        if (whole_home) home.dirty = false;
        std::array<std::uint64_t, 4> changed_pieces{};
        std::size_t changed = 0;
        const bool failed = sync_home_from_frame(home, frame, whole_home, changed_pieces, changed);
        if (failed) ++counters_.upload_failures;
        if (whole_home) ++counters_.uploads;
        fill_copy(home, fill_scratch_.data(), whole_copy, &changed_pieces);
    }

    Home* home_of(std::uint8_t* frame, std::shared_ptr<void> keep, bool& created) {
        if (const auto it = homes_.find(frame); it != homes_.end()) {
            created = false;
            return &it->second;
        }
        std::uint32_t unit = 0, slot = 0;
        if (!take_slot(unit, slot)) return nullptr;
        Home& home = homes_[frame];
        home.unit = unit;
        home.slot = slot;
        home.keep = std::move(keep);
        created = true;
        return &home;
    }

    bool aliased(std::uint64_t va) const {
        auto it = aliases_.upper_bound(va);
        if (it == aliases_.begin()) return false;
        --it;
        return va < it->second.hi;
    }
    // A filtered copy for an existing `home`, made for the aliased page at `va` and synced/filled whole now from one
    // snapshot so `home`, `home.sums[]`, and the filtered copy stay in sync. No room: none (counted).
    void make_copy(Home& home, const std::uint8_t* frame, std::uint64_t va) {
        if (!take_slot(home.copy_unit, home.copy_slot)) {
            ++counters_.upload_failures;
            return;
        }
        home.copy = true;
        home.filter_va = va;
        home.fresh = true;
        sync_and_fill_copy(home, frame, true);
    }
    static constexpr std::array<std::uint64_t, 4> kAllPieces = {~std::uint64_t(0), ~std::uint64_t(0),
                                                                ~std::uint64_t(0), ~std::uint64_t(0)};
    // Fills the filtered copy from the synced `scratch` snapshot through hooks.filter_pieces (preferred) or
    // hooks.filter (`whole`: every byte).
    void fill_copy(Home& home, const std::uint8_t* scratch, bool whole,
                   const std::array<std::uint64_t, 4>* changed_pieces = nullptr) {
        const bool had_refilter = home.refilter;
        home.refilter = false;
        if (!home.copy) return;
        std::uint8_t* to = units_[home.copy_unit].host + std::size_t(home.copy_slot) * kPage;
        const bool fresh = whole || home.fresh;
        home.fresh = false;
        bool ok = false;
        if (hooks_.filter_pieces) {
            const auto& mask = (fresh || had_refilter || !changed_pieces) ? kAllPieces : *changed_pieces;
            ok = hooks_.filter_pieces(home.filter_va, to, scratch, kPage, fresh, mask);
        } else if (hooks_.filter) {
            ok = hooks_.filter(home.filter_va, to, scratch, kPage, fresh);
        } else {
            std::memcpy(to, scratch, kPage);
            ok = true;
        }
        if (!ok) {
            std::memset(to, 0, kPage);
            home.fresh = true;  // zeros now: the next fill writes all again
            ++counters_.upload_failures;
        }
        ++counters_.filtered;
    }
    // The filtered copy goes, after `ticket` (the submissions that could read it at the alias).
    void drop_copy(Home& home, std::uint64_t ticket) {
        if (!home.copy) return;
        push_freed(home.copy_unit, home.copy_slot, ticket);
        home.copy = false;
        home.filter_va = 0;
        home.fresh = true;
        home.refilter = false;
    }
    // The bind of the page at `va` into every alias over it, into the next batch.
    void note_alias_bind(std::uint64_t va, VkDeviceMemory memory, VkDeviceSize offset) {
        if (aliases_.empty()) return;
        auto it = aliases_.upper_bound(va);
        if (it == aliases_.begin()) return;
        --it;
        if (va >= it->second.hi) return;
        const VkDeviceSize at = it->second.offset + (va - it->first);
        alias_pending_[it->second.buffer][at] = {at, kPage, memory, offset, 0};
        has_pending_binds_.store(true, std::memory_order_release);
    }
    bool take_slot(std::uint32_t& unit, std::uint32_t& slot) {
        for (std::uint32_t u = 0; u < units_.size(); ++u)
            if (!units_[u].free.empty()) {
                unit = u;
                slot = units_[u].free.back();
                units_[u].free.pop_back();
                return true;
            }
        Unit made;
        if (!hooks_.make_unit || !hooks_.make_unit(made.memory, made.host)) return false;
        for (std::uint32_t s = kUnitPages; s-- > 0;) made.free.push_back(s);
        units_.push_back(std::move(made));
        unit = std::uint32_t(units_.size() - 1);
        slot = units_.back().free.back();
        units_.back().free.pop_back();
        return true;
    }
    // A page no longer shows `frame`: the home goes once no page shows it, after `ticket` (the submissions that could
    // read it) has retired.
    void release_home(std::uint8_t* frame, std::uint64_t ticket) {
        const auto it = homes_.find(frame);
        if (it == homes_.end() || --it->second.refs) return;
        drop_copy(it->second, ticket);
        push_freed(it->second.unit, it->second.slot, ticket);
        homes_.erase(it);
    }
    // One bind (memory null: unbind) of the page at `va`, in every chunk it lies in, into the next batch.
    void note_bind(std::uint64_t va, VkDeviceMemory memory, VkDeviceSize offset) {
        const std::uint64_t chunk = va / kChunk;
        pending_[chunk][va - chunk * kChunk] = {va - chunk * kChunk, kPage, memory, offset, 0};
        has_pending_binds_.store(true, std::memory_order_release);
    }
    void push_freed(std::uint32_t unit, std::uint32_t slot, std::uint64_t ticket) {
        if (ticket == kUnstamped) {
            unstamped_freed_.push_back({unit, slot});
            return;
        }
        freed_.push_back({unit, slot, ticket});
        if (ticket < oldest_freed_ticket_.load(std::memory_order_relaxed))
            oldest_freed_ticket_.store(ticket, std::memory_order_release);
        freed_count_.store(freed_.size(), std::memory_order_release);
    }
    void sync_freed_atomics() noexcept {
        if (freed_.empty()) {
            oldest_freed_ticket_.store(UINT64_MAX, std::memory_order_release);
            freed_count_.store(0, std::memory_order_release);
            return;
        }
        std::uint64_t oldest = freed_.front().ticket;
        for (const Freed& f : freed_)
            if (f.ticket < oldest) oldest = f.ticket;
        oldest_freed_ticket_.store(oldest, std::memory_order_release);
        freed_count_.store(freed_.size(), std::memory_order_release);
    }

    static constexpr std::size_t kNoAliasedSlot = ~std::size_t(0);
    struct UploadJob {
        std::uint8_t* frame = nullptr;
        std::uint8_t* to = nullptr;
        std::uint64_t* sums = nullptr;
        std::size_t aliased_slot = kNoAliasedSlot;
        std::array<std::uint64_t, 4> changed_pieces{};
        std::size_t changed = 0;
        bool whole = false;
        bool failed = false;
    };
    struct WriteBackJob {
        Home* home = nullptr;
        std::uint8_t* frame = nullptr;
        const std::uint8_t* gpu = nullptr;
        const std::uint64_t* sums = nullptr;
        std::array<std::uint64_t, 4> scan_mask{};
        std::array<std::uint64_t, 4> diff_mask{};
        std::uint16_t diff_count = 0;
        bool full_page = false;
    };

    bool ensure_pool() {
        if (pool_) return true;
        pool_ = make_worker_pool();
        return pool_ != nullptr;
    }

    TitleMirrorHooks hooks_;
    std::map<std::uint64_t, VkBuffer> chunks_;                 // chunk index -> buffer (null: could not be placed)
    std::unordered_map<std::uint64_t, std::uint8_t*> pages_;   // mirrored page -> its physical page's bytes
    std::unordered_map<std::uint8_t*, Home> homes_;            // physical page (its scratch bytes) -> home
    std::vector<Unit> units_;
    std::vector<std::uint8_t*> dirty_;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> unstamped_freed_;
    std::deque<Freed> freed_;
    std::map<std::uint64_t, std::map<VkDeviceSize, VkSparseMemoryBind>> pending_;  // chunk -> offset -> next bind
    std::map<std::uint64_t, Alias> aliases_;                    // title address -> [it, hi) also bound there, filtered
    std::map<VkBuffer, std::map<VkDeviceSize, VkSparseMemoryBind>> alias_pending_;  // buffer -> offset -> next bind
    Counters counters_;
    std::size_t last_batch_ = 0;
    std::uint64_t last_upload_bytes_ = 0;
    std::atomic<bool> has_pending_binds_{false};
    std::atomic<std::size_t> dirty_count_{0};
    std::atomic<std::size_t> freed_count_{0};
    std::atomic<std::uint64_t> oldest_freed_ticket_{UINT64_MAX};
    alignas(64) std::array<std::uint8_t, kPage> fill_scratch_{};
    std::vector<UploadJob> upload_jobs_;
    std::vector<std::uint8_t> aliased_scratch_;
    std::vector<WriteBackJob> wb_jobs_;
    std::unique_ptr<WorkerPool> pool_;
    bool parallel_enabled_ = true;

public:
    std::size_t last_batch() const { return last_batch_; }  // binds in the last flush()
};

}  // namespace leonida
