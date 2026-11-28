#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <span>
#include <thread>
#include <unordered_map>
#include <vector>

namespace leonida {

namespace queue_holds {

// Process-wide registry of live QueueHolds instances and their active CPU/async queue work count.
// Queried by the console list-value progress pump (shims/d3d12_xs_list_values.h) so held waits are never
// released unmet while any queue has queued async inbox calls, an active worker/executing submission, a pending
// held-release request, an active pump_ready() pass, or unretired GPU work.
struct Registry {
    struct Entry {
        const void* owner = nullptr;
        bool (*has_progress)(const void* owner) = nullptr;
    };

    std::atomic<std::int64_t> active_work{0};
    mutable std::mutex mutex;
    std::vector<Entry> entries;

    static Registry& get() {
        static Registry* r = new Registry;  // immortal: may be queried during teardown
        return *r;
    }

    void add(const void* owner, bool (*fn)(const void*)) {
        std::lock_guard lock(mutex);
        entries.push_back({owner, fn});
    }

    void remove(const void* owner) {
        std::lock_guard lock(mutex);
        for (auto it = entries.begin(); it != entries.end(); ++it)
            if (it->owner == owner) {
                entries.erase(it);
                return;
            }
    }

    bool any_progress() const {
        if (active_work.load(std::memory_order_acquire) > 0) return true;
        std::lock_guard lock(mutex);
        for (const Entry& e : entries)
            if (e.has_progress && e.has_progress(e.owner)) return true;
        return false;
    }
};

inline bool any_progress() { return Registry::get().any_progress(); }

}  // namespace queue_holds

// Cross-queue Wait-before-Signal (docs section 4.4, "Wait before Signal across queues").
//
// A queue replays its ExecuteCommandLists against one device-wide layout tracker, in the order the calls reach the
// layer. When queue B calls Wait(fence, v) and the Signal(fence, v) on queue A is only submitted afterwards, B's
// lists would be replayed against the layouts from before A's work. So a queue holds a batch (and everything it
// is asked to do after it) while any fence it waits on has no submitted Signal of at least that value, and
// releases it, in order, once one has been submitted: the layouts then follow the dependency order, while Vulkan's
// own timeline semaphores still order the GPU work.
//
// This class is the device-wide half: for every fence, the highest value a Signal has been submitted for (or the
// CPU has set it to), which is not the completed value; the held queues, which it asks to release whatever became
// ready; and "after held work" callbacks (the presenter), which run once every operation held when they were
// registered has been released. It owns no Vulkan objects. Semaphores it was never told about count as satisfied,
// so a queue used with a plain VkSemaphore behaves as before.
//
// Threads. No GPU wait happens here, but the thread that makes a wait satisfiable (a queue's Signal, a CPU Signal)
// does the release itself, after it has dropped its own queue's state lock: it records and submits the held
// batches (a held ExecuteCommandLists costs what it would have cost when called, which includes whatever
// execute_now does: the bindless upload's wait for the device is one) and runs the callbacks. The COM wrapper
// still holds the mutex of the queue whose call that thread is in, so that queue's other calls wait meanwhile.
// One thread releases at a time; a thread that finds another one releasing leaves a note and that one takes
// another pass. detach() and cancel_after() (and remove_fence() from outside a release) wait for a running
// release.
template<class Semaphore>
class QueueHolds {
public:
    // Replacement wait target for retroactive GPU-to-GPU wait relays. `pin` keeps the underlying fence object alive
    // until the substituted submission retires on the GPU.
    struct Wait {
        Semaphore semaphore{};
        std::uint64_t value = 0;
        std::shared_ptr<void> pin;
    };

    // A queue that can hold batches. The calls take the queue's own lock; none is made with it held.
    struct Holder {
        // Releases the held operations at the front of the queue whose waits are satisfied. True if any was.
        virtual bool release_ready() = 0;
        // Drops the held operations that wait on `semaphore` (its fence is gone), reporting each.
        virtual void discard_waiting_on(Semaphore semaphore) = 0;
        // Sequence number (next_seq()) of the oldest held operation, 0 when nothing is held. Read-only query.
        virtual std::uint64_t oldest_held_seq() = 0;
        // Notifies the holder that a real after_held() barrier was registered at `seq` so batches at or before `seq`
        // will not coalesce with batches enqueued after `seq`. Called with no queue lock held.
        virtual void note_barrier(std::uint64_t /*seq*/) {}
        // Substitutes any queued/pending/held waits matching `(gate_sem, gate_val)` with `replacements`.
        // Sets `any_substituted = true` when at least one matching wait was replaced; returns true when at least one
        // held operation was updated (and may now be ready to release). Called with no QueueHolds lock held.
        virtual bool apply_wait_relay(Semaphore /*gate_sem*/, std::uint64_t /*gate_val*/,
                                      std::span<const Wait> /*replacements*/, bool& /*any_substituted*/) {
            return false;
        }
        // True when this queue has submitted GPU work that has not retired yet. Must not take any QueueHolds or
        // queue lock (called under holders_use_mutex_ shared lock from any_progress()).
        virtual bool has_pending_gpu_work() const { return false; }

    protected:
        ~Holder() = default;
    };

    QueueHolds() { queue_holds::Registry::get().add(this, &has_progress_thunk); }
    ~QueueHolds() {
        queue_holds::Registry::get().remove(this);
        {
            std::lock_guard bl(barrier_mutex_);
            barrier_stop_ = true;
        }
        barrier_cv_.notify_all();
        if (barrier_thread_.joinable()) barrier_thread_.join();
    }
    QueueHolds(const QueueHolds&) = delete;
    QueueHolds& operator=(const QueueHolds&) = delete;

    // Tracks active CPU-side queue progress (async inbox items, busy worker turns, pending async release requests,
    // actively executing submissions, and active pump_ready() passes). Balanced in pairs / via WorkGuard.
    void work_changed(std::int64_t delta) {
        active_work_.fetch_add(delta, std::memory_order_acq_rel);
        queue_holds::Registry::get().active_work.fetch_add(delta, std::memory_order_acq_rel);
    }

    class WorkGuard {
    public:
        WorkGuard() = default;
        explicit WorkGuard(QueueHolds& holds) { arm(holds); }
        ~WorkGuard() { release(); }
        void arm(QueueHolds& holds) {
            if (holds_) return;
            holds_ = &holds;
            holds_->work_changed(1);
        }
        void release() {
            if (!holds_) return;
            QueueHolds* h = holds_;
            holds_ = nullptr;
            h->work_changed(-1);
        }
        WorkGuard(const WorkGuard&) = delete;
        WorkGuard& operator=(const WorkGuard&) = delete;

    private:
        QueueHolds* holds_ = nullptr;
    };

    // True when this device has active CPU/async queue work, an unpumped signal for held work, or unretired GPU
    // work on any attached queue. Lock order: Registry::mutex (optional) -> holders_use_mutex_ (shared) -> mutex_
    // (leaf, released before querying each holder's lock-free GPU progress).
    bool has_progress() const {
        if (active_work_.load(std::memory_order_acquire) > 0) return true;
        if (any_held() && signaled_unpumped_.load(std::memory_order_acquire)) return true;
        std::shared_lock use(holders_use_mutex_);
        const auto list = snapshot();
        for (const Holder* holder : list)
            if (holder->has_pending_gpu_work()) return true;
        return false;
    }

    void add_fence(Semaphore semaphore, std::uint64_t initial) {
        std::lock_guard lock(mutex_);
        fences_[semaphore] = FenceState{initial};
    }

    // The fence is gone. Operations that still reference it are dropped and reported (the COM layer keeps a fence
    // alive while a held operation names it, so this is a safety net for direct users of VulkanFence). Not done
    // from inside a release: the releasing thread already holds the lock, and what it is releasing cannot
    // reference a fence whose last reference it is dropping, since that reference is the one it holds.
    void remove_fence(Semaphore semaphore) {
        std::vector<std::shared_ptr<void>> dropped_pins;
        {
            std::lock_guard lock(mutex_);
            fences_.erase(semaphore);
            if (!relays_.empty()) {
                for (auto it = relays_.begin(); it != relays_.end();) {
                    bool match = (it->gate_sem == semaphore);
                    if (!match) {
                        for (const Wait& w : it->replacements) {
                            if (w.semaphore == semaphore) {
                                match = true;
                                break;
                            }
                        }
                    }
                    if (match) {
                        for (Wait& w : it->replacements)
                            if (w.pin) dropped_pins.push_back(std::move(w.pin));
                        it = relays_.erase(it);
                    } else {
                        ++it;
                    }
                }
                relay_count_.store(relays_.size(), std::memory_order_release);
            }
        }
        dropped_pins.clear();
        if (held_.load(std::memory_order_acquire) == 0 || in_pump()) return;
        {
            std::lock_guard pump(pump_mutex_);
            in_pump() = true;  // what is dropped may release other fences: they must not wait for this lock
            for (Holder* holder : snapshot()) holder->discard_waiting_on(semaphore);
            in_pump() = false;
        }
        pump_ready();
    }

    // A Signal of `value` was submitted to the GPU (or the CPU set the fence). Call after the submission, with no
    // queue lock held; then pump_ready() to release what it satisfied.
    // `vis_seq` is non-zero for CPU-side Fence::Signal edges (establishing a happens-before edge for CPU memory
    // written before the signal) and 0 for GPU-only queue Signals, tracked per `(semaphore, value)` so waiters on
    // unrelated fences or GPU-only signals never force redundant global visibility refreshes.
    void signaled(Semaphore semaphore, std::uint64_t value, std::uint64_t vis_seq = 0) {
        bool advanced = false;
        std::vector<std::shared_ptr<void>> dropped_pins;
        {
            std::lock_guard lock(mutex_);
            const auto it = fences_.find(semaphore);
            if (it != fences_.end()) {
                FenceState& fs = it->second;
                if (fs.value < value) {
                    fs.value = value;
                    advanced = true;
                }
                if (vis_seq) {
                    if (vis_seq > fs.max_cpu_vis_seq) fs.max_cpu_vis_seq = vis_seq;
                    if (!fs.cpu_vis.empty() && fs.cpu_vis.back().first == value) {
                        if (vis_seq > fs.cpu_vis.back().second) fs.cpu_vis.back().second = vis_seq;
                    } else {
                        if (fs.cpu_vis.size() >= kMaxFenceVisHistory) fs.cpu_vis.pop_front();
                        fs.cpu_vis.emplace_back(value, vis_seq);
                    }
                }
            }
            if (!relays_.empty()) {
                for (auto rit = relays_.begin(); rit != relays_.end();) {
                    if (rit->gate_sem == semaphore && rit->gate_val <= value) {
                        for (Wait& w : rit->replacements)
                            if (w.pin) dropped_pins.push_back(std::move(w.pin));
                        rit = relays_.erase(rit);
                    } else {
                        ++rit;
                    }
                }
                relay_count_.store(relays_.size(), std::memory_order_release);
            }
        }
        dropped_pins.clear();
        if (advanced && any_held()) signaled_unpumped_.store(true, std::memory_order_release);
    }

    // Retroactively redirects any queued, pending, or held wait on `(gate_sem, gate_val)` to `replacements`
    // (GPU-to-GPU timeline waits, never GPU-signaling `gate_sem`), then calls pump_ready() so newly-unblocked held
    // ops release immediately. When `consume_on_substitute` is true (single-owner gate fences), the redirect entry is
    // pruned immediately once substituted (or once consumed by an in-transit lookup_relay). Returns true iff at least
    // one existing wait was substituted; when no wait matched yet, the redirect entry remains registered for an
    // in-transit lookup_relay() and this call returns false so callers preserve the fallback gate.
    bool relay_wait(Semaphore gate_sem, std::uint64_t gate_val, std::span<const Wait> replacements,
                    bool consume_on_substitute = false) {
        if (!gate_sem || !gate_val || replacements.empty()) return false;
        for (const Wait& w : replacements) {
            if (!w.semaphore || w.semaphore == gate_sem) return false;
        }
        std::vector<std::shared_ptr<void>> dropped_pins;
        {
            std::lock_guard lock(mutex_);
            const auto fit = fences_.find(gate_sem);
            if (fit != fences_.end() && fit->second.value >= gate_val) return false;
            for (const WaitRelay& existing : relays_) {
                if (existing.gate_sem == gate_sem && existing.gate_val == gate_val) return false;
            }
            while (relays_.size() >= kMaxRelays) {
                for (Wait& w : relays_.front().replacements)
                    if (w.pin) dropped_pins.push_back(std::move(w.pin));
                relays_.pop_front();
            }
            relays_.push_back(WaitRelay{gate_sem, gate_val,
                                        std::vector<Wait>(replacements.begin(), replacements.end()),
                                        consume_on_substitute});
            relay_count_.store(relays_.size(), std::memory_order_release);
        }
        dropped_pins.clear();

        WorkGuard relaying(*this);
        bool any_held_updated = false;
        bool any_substituted = false;
        {
            std::shared_lock use(holders_use_mutex_);
            for (Holder* holder : snapshot()) {
                bool sub = false;
                if (holder->apply_wait_relay(gate_sem, gate_val, replacements, sub)) any_held_updated = true;
                if (sub) any_substituted = true;
            }
        }
        if (consume_on_substitute && any_substituted) {
            {
                std::lock_guard lock(mutex_);
                for (auto it = relays_.begin(); it != relays_.end(); ++it) {
                    if (it->gate_sem == gate_sem && it->gate_val == gate_val) {
                        for (Wait& w : it->replacements)
                            if (w.pin) dropped_pins.push_back(std::move(w.pin));
                        relays_.erase(it);
                        relay_count_.store(relays_.size(), std::memory_order_release);
                        break;
                    }
                }
            }
            dropped_pins.clear();
        }
        if (any_held_updated || (any_substituted && any_held())) {
            if (any_held()) signaled_unpumped_.store(true, std::memory_order_release);
            pump_ready();
        }
        return any_substituted;
    }

    // Checks whether `(gate_sem, gate_val)` has an active wait redirect (catching waits in transit between an async
    // inbox pop and state_mutex_). Lock-free fast path when no redirects are active.
    bool lookup_relay(Semaphore gate_sem, std::uint64_t gate_val, std::vector<Wait>& out) {
        if (!gate_sem || !gate_val || relay_count_.load(std::memory_order_acquire) == 0) return false;
        std::lock_guard lock(mutex_);
        for (auto it = relays_.begin(); it != relays_.end(); ++it) {
            if (it->gate_sem == gate_sem && it->gate_val == gate_val) {
                if (it->consume_on_substitute) {
                    out = std::move(it->replacements);
                    relays_.erase(it);
                    relay_count_.store(relays_.size(), std::memory_order_release);
                } else {
                    out = it->replacements;
                }
                return true;
            }
        }
        return false;
    }

    std::size_t active_relay_count() const noexcept { return relay_count_.load(std::memory_order_acquire); }

    // Minimum CPU visibility epoch established by a CPU Fence::Signal that reached `wait_value` on `semaphore`
    // (0 when `semaphore` has never received a CPU signal).
    std::uint64_t signaled_vis_seq(Semaphore semaphore, std::uint64_t wait_value) const {
        std::lock_guard lock(mutex_);
        const auto it = fences_.find(semaphore);
        if (it == fences_.end() || it->second.max_cpu_vis_seq == 0) return 0;
        const FenceState& fs = it->second;
        for (const auto& [sig_val, seq] : fs.cpu_vis) {
            if (sig_val >= wait_value) return seq;
        }
        return fs.max_cpu_vis_seq;
    }

    // A queue is about to submit a Signal of `value`; call inside the device submit lock, and record the value with
    // signaled() only once the vkQueueSubmit succeeded (a failed submit signals nothing: waiters must stay held and a
    // retry must go through). False when a Signal of at least `value` was already submitted: the timeline cannot go
    // down or stay (VUID-VkSubmitInfo-pSignalSemaphores-03242 and the timeline rules), and a wait for `value` is
    // already satisfied, so the Vulkan signal is left out. This happens when a held Signal(F, 2) is released after an
    // unheld Signal(F, 3). Unknown semaphores are always signaled.
    bool would_signal(Semaphore semaphore, std::uint64_t value) const {
        std::lock_guard lock(mutex_);
        const auto it = fences_.find(semaphore);
        return it == fences_.end() || it->second.value < value;
    }

    // The highest Signal submitted for the fence (or set by the CPU); 0 for an unknown semaphore.
    std::uint64_t submitted_value(Semaphore semaphore) const {
        std::lock_guard lock(mutex_);
        const auto it = fences_.find(semaphore);
        return it == fences_.end() ? 0 : it->second.value;
    }

    // true when no submitted Signal reaches `value` yet. Unknown semaphores are satisfied.
    bool pending(Semaphore semaphore, std::uint64_t value) const {
        std::lock_guard lock(mutex_);
        const auto it = fences_.find(semaphore);
        return it != fences_.end() && it->second.value < value;
    }

    void attach(Holder* holder) {
        std::lock_guard lock(mutex_);
        holders_.push_back(holder);
    }

    // Returns once no thread is using `holder`.
    void detach(Holder* holder) {
        {
            std::lock_guard pump(pump_mutex_);
            {
                std::lock_guard lock(mutex_);
                for (auto it = holders_.begin(); it != holders_.end(); ++it)
                    if (*it == holder) {
                        holders_.erase(it);
                        break;
                    }
            }
            // Wait for any concurrent has_progress() reader holding a snapshot that still names `holder`.
            std::unique_lock use(holders_use_mutex_);
        }
        pump_ready();
    }

    // Sequence numbers of held operations, in the order they were held (across queues).
    std::uint64_t next_seq() { return seq_.fetch_add(1, std::memory_order_acq_rel) + 1; }
    std::uint64_t current_seq() const { return seq_.load(std::memory_order_acquire); }

    // Runs `fn` once every operation held up to now has been released (or dropped), on the thread that releases the
    // last of them, or right here if none is held. Callbacks of one owner run in registration order. Used by the
    // presenter: a Present queued behind held work shows what that work draws.
    void after_held(const void* owner, std::function<void()> fn) {
        const std::uint64_t seq = current_seq();
        {
            std::lock_guard lock(mutex_);
            barriers_.push_back({owner, seq, std::move(fn)});
            barrier_count_.fetch_add(1, std::memory_order_acq_rel);
        }
        if (seq) {
            std::shared_lock use(holders_use_mutex_);
            for (Holder* holder : snapshot()) holder->note_barrier(seq);
        }
        pump_ready();
    }

    // Drops the callbacks of `owner` that have not run, and waits for any in-flight callback on the barrier thread.
    // Lock order: pump_mutex_ -> barrier_mutex_ -> mutex_ (mutex_ is always leaf; barrier callbacks run unlocked).
    void cancel_after(const void* owner) {
        std::int64_t removed_ready = 0;
        {
            std::lock_guard bl(barrier_mutex_);
            std::lock_guard lock(mutex_);
            auto erase_from = [&](std::deque<Barrier>& q, bool is_ready) {
                for (auto it = q.begin(); it != q.end();) {
                    if (it->owner == owner) {
                        it = q.erase(it);
                        barrier_count_.fetch_sub(1, std::memory_order_acq_rel);
                        if (is_ready) ++removed_ready;
                    } else {
                        ++it;
                    }
                }
            };
            erase_from(barriers_, false);
            erase_from(ready_barriers_, true);
            if (removed_ready) update_active_barrier_min_seq_locked();
        }
        if (removed_ready) {
            barrier_drained_cv_.notify_all();
            work_changed(-removed_ready);
        }
        // Wait for any already-popped callback currently running on the dedicated barrier thread.
        if (!in_barrier()) {
            std::unique_lock bl(barrier_mutex_);
            barrier_drained_cv_.wait(bl, [this] { return !barrier_running_; });
        }
        pump_ready();
    }

    // Waits until every satisfied barrier with `barrier.seq <= seq` that is queued in ready_barriers_ or currently
    // running on the dedicated barrier thread has finished executing. Lock-free fast path when no matching barrier is
    // active. Never waits on unsatisfied barriers still in barriers_.
    void wait_barriers_drained(std::uint64_t seq) {
        if (in_barrier()) return;
        const std::uint64_t min_p1 = active_barrier_min_seq_.load(std::memory_order_acquire);
        if (min_p1 == 0 || (min_p1 - 1) > seq) return;
        std::unique_lock bl(barrier_mutex_);
        barrier_drained_cv_.wait(bl, [this, seq] {
            std::lock_guard lock(mutex_);
            if (barrier_running_ && running_barrier_seq_ <= seq) return false;
            if (!ready_barriers_.empty() && ready_barriers_.front().seq <= seq) return false;
            return true;
        });
    }

    // Held operations over all queues, for the cheap "is there anything to do" test.
    void held_changed(std::int64_t delta) { held_.fetch_add(delta, std::memory_order_acq_rel); }
    bool any_held() const { return held_.load(std::memory_order_acquire) > 0; }

    // Releases every held operation that has become ready, in dependency order: a release can submit a Signal
    // that makes another queue's front ready, so passes repeat until one finds nothing new. Satisfied barriers
    // (Present / after_held()) are handed off in strict FIFO order to the dedicated barrier-drain thread (keeping
    // active_work_ +1 until executed) so callers such as the list-value releaser never stall on Present.
    void pump_ready() {
        if (!any_held() && barrier_count_.load(std::memory_order_acquire) == 0) {
            signaled_unpumped_.store(false, std::memory_order_release);
            return;
        }
        WorkGuard pumping(*this);
        dirty_.store(true, std::memory_order_release);
        if (in_pump()) return;  // a release on this thread submitted a Signal: the running pass will go again
        for (;;) {
            std::unique_lock pump(pump_mutex_, std::try_to_lock);
            if (!pump) return;  // another thread is releasing and will see dirty_
            in_pump() = true;
            while (dirty_.exchange(false, std::memory_order_acq_rel)) {
                signaled_unpumped_.store(false, std::memory_order_release);
                bool progress = true;
                while (progress) {
                    progress = false;
                    for (Holder* holder : snapshot()) {
                        if (holder->release_ready()) {
                            progress = true;
                            if (collect_ready_barriers()) {
                                // Satisfied barriers were handed to the barrier thread; yield to let post-barrier
                                // work drain on the next pass (which waits via wait_barriers_drained()).
                                dirty_.store(true, std::memory_order_release);
                                break;
                            }
                        }
                    }
                }
                (void)collect_ready_barriers();
            }
            in_pump() = false;
            pump.unlock();
            if (!dirty_.load(std::memory_order_acquire)) return;
        }
    }

private:
    static constexpr std::size_t kMaxFenceVisHistory = 32;
    static constexpr std::size_t kMaxRelays = 256;

    struct WaitRelay {
        Semaphore gate_sem{};
        std::uint64_t gate_val = 0;
        std::vector<Wait> replacements;
        bool consume_on_substitute = false;
    };

    struct FenceState {
        std::uint64_t value = 0;
        std::uint64_t max_cpu_vis_seq = 0;
        std::deque<std::pair<std::uint64_t, std::uint64_t>> cpu_vis;  // (signaled_value, vis_seq)
    };

    struct Barrier {
        const void* owner = nullptr;
        std::uint64_t seq = 0;
        std::function<void()> fn;
    };

    static bool& in_pump() {
        static thread_local bool pumping = false;
        return pumping;
    }

    static bool& in_barrier() {
        static thread_local bool running = false;
        return running;
    }

    std::vector<Holder*> snapshot() const {
        std::lock_guard lock(mutex_);
        return holders_;
    }

    // Called with both barrier_mutex_ and mutex_ held. Stores (min_seq + 1) of any running/ready barrier, or 0 if none.
    void update_active_barrier_min_seq_locked() noexcept {
        bool any = false;
        std::uint64_t min_seq = UINT64_MAX;
        if (barrier_running_) {
            any = true;
            min_seq = running_barrier_seq_;
        }
        if (!ready_barriers_.empty()) {
            if (!any || ready_barriers_.front().seq < min_seq) min_seq = ready_barriers_.front().seq;
            any = true;
        }
        active_barrier_min_seq_.store(any ? (min_seq == UINT64_MAX ? UINT64_MAX : min_seq + 1) : 0,
                                      std::memory_order_release);
    }

    // Called with pump_mutex_ held. Moves every satisfied barrier from barriers_ to ready_barriers_ in strict FIFO
    // order, increments active_work_ (+1 per ready barrier so has_progress() stays true until executed), and wakes the
    // dedicated barrier thread. True when at least one barrier became ready.
    bool collect_ready_barriers() {
        bool moved = false;
        for (;;) {
            const void* owner = nullptr;
            std::uint64_t seq = 0;
            {
                std::lock_guard lock(mutex_);
                if (barriers_.empty()) break;
                owner = barriers_.front().owner;
                seq = barriers_.front().seq;
            }
            bool blocked = false;
            for (Holder* holder : snapshot()) {
                const std::uint64_t oldest = holder->oldest_held_seq();
                if (oldest && oldest <= seq) {
                    blocked = true;
                    break;
                }
            }
            if (blocked) break;
            {
                std::lock_guard bl(barrier_mutex_);
                std::lock_guard lock(mutex_);
                if (barriers_.empty() || barriers_.front().seq != seq || barriers_.front().owner != owner)
                    continue;  // cancelled meanwhile
                ready_barriers_.push_back(std::move(barriers_.front()));
                barriers_.pop_front();
                work_changed(1);
                update_active_barrier_min_seq_locked();
                if (!barrier_thread_.joinable() && !barrier_stop_) {
                    barrier_thread_ = std::thread([this] { barrier_worker(); });
                }
                moved = true;
            }
        }
        if (moved) barrier_cv_.notify_one();
        return moved;
    }

    // Dedicated thread that executes satisfied ready_barriers_ in strict FIFO order without blocking pump_ready() or
    // the list-value releaser. Keeps active_work_ +1 across each barrier and signals barrier_drained_cv_ on completion.
    void barrier_worker() {
        in_barrier() = true;
        struct InBarrierGuard {
            ~InBarrierGuard() { in_barrier() = false; }
        } guard;
        for (;;) {
            Barrier next;
            {
                std::unique_lock bl(barrier_mutex_);
                barrier_cv_.wait(bl, [this] {
                    std::lock_guard lock(mutex_);
                    return !ready_barriers_.empty() || barrier_stop_;
                });
                std::lock_guard lock(mutex_);
                if (ready_barriers_.empty()) {
                    if (barrier_stop_) return;
                    continue;
                }
                next = std::move(ready_barriers_.front());
                ready_barriers_.pop_front();
                barrier_count_.fetch_sub(1, std::memory_order_acq_rel);
                barrier_running_ = true;
                running_barrier_seq_ = next.seq;
                update_active_barrier_min_seq_locked();
            }
            try {
                if (next.fn) next.fn();
            } catch (...) {
            }
            next.fn = {};  // release captured objects outside locks
            {
                std::lock_guard bl(barrier_mutex_);
                std::lock_guard lock(mutex_);
                barrier_running_ = false;
                running_barrier_seq_ = 0;
                update_active_barrier_min_seq_locked();
            }
            barrier_drained_cv_.notify_all();
            work_changed(-1);
        }
    }

    static bool has_progress_thunk(const void* owner) {
        return static_cast<const QueueHolds*>(owner)->has_progress();
    }

    mutable std::mutex mutex_;  // leaf: fences_, holders_, barriers_, ready_barriers_
    std::mutex pump_mutex_;     // one releasing thread; held across release_ready(), never across barrier callbacks
    std::mutex barrier_mutex_;  // guards barrier_thread_, barrier_running_, barrier_stop_, and barrier CVs
    std::condition_variable barrier_cv_;
    std::condition_variable barrier_drained_cv_;
    std::thread barrier_thread_;
    bool barrier_running_ = false;
    bool barrier_stop_ = false;
    std::uint64_t running_barrier_seq_ = 0;
    std::atomic<std::uint64_t> active_barrier_min_seq_{0};  // 0 if none, else min(seq)+1 of running/ready barriers
    mutable std::shared_mutex holders_use_mutex_;  // shared by has_progress(), unique in detach() after erasing holder
    std::unordered_map<Semaphore, FenceState> fences_;
    std::deque<WaitRelay> relays_;
    std::atomic<std::size_t> relay_count_{0};
    std::vector<Holder*> holders_;
    std::deque<Barrier> barriers_;
    std::deque<Barrier> ready_barriers_;
    std::atomic<std::int64_t> held_{0};
    std::atomic<std::int64_t> barrier_count_{0};
    std::atomic<std::uint64_t> seq_{0};
    std::atomic<bool> dirty_{false};
    std::atomic<std::int64_t> active_work_{0};
    std::atomic<bool> signaled_unpumped_{false};
};

}  // namespace leonida
