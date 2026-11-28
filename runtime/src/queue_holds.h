#pragma once

#include <atomic>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <shared_mutex>
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
        // True when this queue has submitted GPU work that has not retired yet. Must not take any QueueHolds or
        // queue lock (called under holders_use_mutex_ shared lock from any_progress()).
        virtual bool has_pending_gpu_work() const { return false; }

    protected:
        ~Holder() = default;
    };

    QueueHolds() { queue_holds::Registry::get().add(this, &has_progress_thunk); }
    ~QueueHolds() { queue_holds::Registry::get().remove(this); }
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
        {
            std::lock_guard lock(mutex_);
            fences_.erase(semaphore);
        }
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
        }
        if (advanced && any_held()) signaled_unpumped_.store(true, std::memory_order_release);
    }

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

    // Drops the callbacks of `owner` that have not run, and waits for one that is running.
    // Lock order: never hold pump_mutex_ and barrier_mutex_ together (mutex_ is always leaf).
    void cancel_after(const void* owner) {
        {
            std::lock_guard lock(mutex_);
            auto erase_from = [&](std::deque<Barrier>& q) {
                for (auto it = q.begin(); it != q.end();) {
                    if (it->owner == owner) {
                        it = q.erase(it);
                        barrier_count_.fetch_sub(1, std::memory_order_acq_rel);
                    } else {
                        ++it;
                    }
                }
            };
            erase_from(barriers_);
            erase_from(ready_barriers_);
        }
        // Wait for any already-popped callback currently running in drain_ready_barriers().
        if (!in_barrier()) {
            std::lock_guard bl(barrier_mutex_);
        }
        pump_ready();
    }

    // Held operations over all queues, for the cheap "is there anything to do" test.
    void held_changed(std::int64_t delta) { held_.fetch_add(delta, std::memory_order_acq_rel); }
    bool any_held() const { return held_.load(std::memory_order_acquire) > 0; }

    // Releases every held operation that has become ready, in dependency order: a release can submit a Signal
    // that makes another queue's front ready, so passes repeat until one finds nothing new. Satisfied barriers
    // (Present / after_held()) are popped in FIFO order under pump_mutex_ into ready_barriers_ and executed OUTSIDE
    // pump_mutex_ under barrier_mutex_ so Presents stay strictly FIFO while never stalling cross-queue pump_ready().
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
            bool collected_barriers = false;
            while (dirty_.exchange(false, std::memory_order_acq_rel)) {
                signaled_unpumped_.store(false, std::memory_order_release);
                bool progress = true;
                while (progress) {
                    progress = false;
                    for (Holder* holder : snapshot()) {
                        if (holder->release_ready()) {
                            progress = true;
                            if (collect_ready_barriers()) {
                                collected_barriers = true;
                                // Pause pumping to run satisfied barriers outside pump_mutex_ before draining
                                // post-barrier work on the next pass.
                                dirty_.store(true, std::memory_order_release);
                                break;
                            }
                        }
                    }
                    if (collected_barriers) break;
                }
                if (collected_barriers) break;
                if (collect_ready_barriers()) {
                    collected_barriers = true;
                    break;
                }
            }
            in_pump() = false;
            pump.unlock();
            if (collected_barriers) drain_ready_barriers();
            if (!dirty_.load(std::memory_order_acquire)) return;
        }
    }

private:
    static constexpr std::size_t kMaxFenceVisHistory = 32;

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

    // Called with pump_mutex_ held. Moves every satisfied barrier from barriers_ to ready_barriers_ in strict FIFO
    // order without invoking any callback under pump_mutex_. True when at least one barrier became ready.
    bool collect_ready_barriers() {
        bool moved = false;
        for (;;) {
            const void* owner = nullptr;
            std::uint64_t seq = 0;
            {
                std::lock_guard lock(mutex_);
                if (barriers_.empty()) return moved;
                owner = barriers_.front().owner;
                seq = barriers_.front().seq;
            }
            for (Holder* holder : snapshot()) {
                const std::uint64_t oldest = holder->oldest_held_seq();
                if (oldest && oldest <= seq) return moved;
            }
            {
                std::lock_guard lock(mutex_);
                if (barriers_.empty() || barriers_.front().seq != seq || barriers_.front().owner != owner)
                    continue;  // cancelled meanwhile
                ready_barriers_.push_back(std::move(barriers_.front()));
                barriers_.pop_front();
                moved = true;
            }
        }
    }

    // Called OUTSIDE pump_mutex_. Executes all queued ready_barriers_ in strict FIFO order under barrier_mutex_.
    void drain_ready_barriers() {
        if (in_barrier()) return;
        struct InBarrierGuard {
            bool& flag;
            explicit InBarrierGuard(bool& f) noexcept : flag(f) { flag = true; }
            ~InBarrierGuard() { flag = false; }
        };
        for (;;) {
            std::unique_lock bl(barrier_mutex_);
            InBarrierGuard running(in_barrier());
            for (;;) {
                Barrier next;
                {
                    std::lock_guard lock(mutex_);
                    if (ready_barriers_.empty()) break;
                    next = std::move(ready_barriers_.front());
                    ready_barriers_.pop_front();
                    barrier_count_.fetch_sub(1, std::memory_order_acq_rel);
                }
                next.fn();
            }
            bl.unlock();
            bool more = false;
            {
                std::lock_guard lock(mutex_);
                more = !ready_barriers_.empty();
            }
            if (!more) return;
        }
    }

    static bool has_progress_thunk(const void* owner) {
        return static_cast<const QueueHolds*>(owner)->has_progress();
    }

    mutable std::mutex mutex_;  // leaf: fences_, holders_, barriers_, ready_barriers_
    std::mutex pump_mutex_;     // one releasing thread; held across release_ready(), never across barrier callbacks
    std::mutex barrier_mutex_;  // serializes ready_barriers_ execution in strict FIFO order outside pump_mutex_
    mutable std::shared_mutex holders_use_mutex_;  // shared by has_progress(), unique in detach() after erasing holder
    std::unordered_map<Semaphore, FenceState> fences_;
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
