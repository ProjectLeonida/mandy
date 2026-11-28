// The "submit frame" and "submit floor" lines (runtime/src/submit_phases.h): their sums against the per-submission
// phases the "submit phases" line adds up.
//   1. A scripted run on the CPU, with known phase times: two frames, a submission on each of two queue kinds, a phase
//      on one of our threads (a WorkerScope), a title call (TitleCall) and a contended lock. The frame sums, the
//      split between title and own threads, the worst frame and the floor must come out as scripted.
//   2. Real submissions through VulkanQueue (empty lists; 77-free: without a Vulkan device this part is skipped and
//      the test still passes): the floor's phase sums must equal what the phase totals grew by, submission for
//      submission, and every one of them is a floor submission.
#include "vk_test_device.h"
#include "../src/list_value_progress.h"
#include "../src/vulkan_backend.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

using namespace leonida;
namespace sp = leonida::submit_phases;
constexpr std::uint64_t ms = 1000000;  // ns

std::uint64_t load(const std::atomic<std::uint64_t>& a) { return a.load(std::memory_order_relaxed); }

std::uint64_t phase_total() {
    std::uint64_t n = 0;
    for (const auto& p : sp::totals().ns) n += load(p);
    return n;
}

int scripted() {
    sp::FrameTotals& f = sp::frame_totals();
    // Frame 1: a direct submission with heaps 1 ms and replay 2 ms, nothing changed (a floor submission).
    {
        const sp::Submission s(0);
        sp::add(sp::heaps, 1 * ms);
        sp::add(sp::replay, 2 * ms);
    }
    // One of our threads: its 5 ms of textures are the phase line's, not the title threads'.
    std::thread([] {
        const sp::WorkerScope busy(sp::worker_texture_reads);
        sp::add(sp::textures, 5 * ms);
    }).join();
    // A title call that spends about 3 ms, 1 ms of it in a phase: about 2 ms of fences.
    const auto call_start = std::chrono::steady_clock::now();
    {
        const sp::TitleCall call(sp::part_fences);
        sp::add(sp::replay, 1 * ms);
        std::this_thread::sleep_for(std::chrono::milliseconds(3));
    }
    const double call_ns = double(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                      std::chrono::steady_clock::now() - call_start).count());
    const std::uint64_t fences = load(f.part_ns[sp::part_fences]);
    EXPECT(fences > 1 * ms && double(fences) <= call_ns - 1e6);
    // A lock another thread holds for 4 ms: counted as a lock wait, inside the other parts (not in the frame's sum).
    std::mutex m;
    std::atomic<bool> taken{false};
    std::thread holder([&] {  // takes the lock and lets it go on its own thread, as a mutex must be
        const std::lock_guard held(m);
        taken.store(true);
        std::this_thread::sleep_for(std::chrono::milliseconds(4));
    });
    while (!taken.load()) std::this_thread::yield();
    {
        std::unique_lock lock(m, std::defer_lock);
        sp::lock_timed(lock);
    }
    holder.join();
    EXPECT(load(f.part_ns[sp::part_lock_wait]) >= 2 * ms);
    const std::uint64_t frame1 = 3 * ms + 1 * ms + fences;  // heaps, replay, replay in the call, fences
    EXPECT(load(f.current_ns) == frame1);
    sp::frame_presented();
    // Frame 2: a compute submission with geometry 4 ms that uploaded geometry: not a floor submission.
    {
        const sp::Submission s(1);
        sp::add(sp::geometry, 4 * ms);
        sp::note_change(sp::change_geometry);
    }
    sp::frame_presented();

    EXPECT(load(f.frames) == 2);
    EXPECT(load(f.submissions[0]) == 1 && load(f.submissions[1]) == 1 && load(f.submissions[2]) == 0);
    EXPECT(load(f.phase_ns[sp::heaps]) == 1 * ms && load(f.phase_ns[sp::replay]) == 3 * ms);
    EXPECT(load(f.phase_ns[sp::geometry]) == 4 * ms && load(f.phase_ns[sp::textures]) == 0);
    EXPECT(load(f.worker_ns[sp::worker_texture_reads]) > 0);
    // The phase line counts every thread: the title threads' phases and the worker's add up to its total.
    std::uint64_t title = 0;
    for (const auto& p : f.phase_ns) title += load(p);
    EXPECT(phase_total() == title + 5 * ms);
    EXPECT(load(f.worst_ns) == (frame1 > 4 * ms ? frame1 : 4 * ms));
    EXPECT(load(f.all_count) == 2 && load(f.all_ns) == 7 * ms);
    EXPECT(load(f.floor_count) == 1 && load(f.floor_min_ns) == 3 * ms);
    EXPECT(load(f.floor_ns[sp::heaps]) == 1 * ms && load(f.floor_ns[sp::replay]) == 2 * ms &&
           load(f.floor_ns[sp::geometry]) == 0);
    sp::frame_lines(1.0);  // the two lines' format, on stderr; a new window
    EXPECT(load(f.frames) == 0 && load(f.all_count) == 0 && load(f.floor_count) == 0 && load(f.worst_ns) == 0);
    for (const auto& p : f.phase_ns) EXPECT(load(p) == 0);
    // Mirror refresh/write-back inside a list submission: time inside Scope(shadows) is billed to shadows, not lists,
    // and contended lock_timed inside it is billed to lock waits.
    {
        const sp::TitleCall call(sp::part_lists);
        const sp::Scope shadows_scope(sp::shadows);
        std::mutex mirror_m;
        std::atomic<bool> held_flag{false};
        std::thread holder([&] {
            const std::lock_guard held(mirror_m);
            held_flag.store(true);
            std::this_thread::sleep_for(std::chrono::milliseconds(4));
        });
        while (!held_flag.load()) std::this_thread::yield();
        {
            std::unique_lock lock(mirror_m, std::defer_lock);
            sp::lock_timed(lock);
        }
        holder.join();
    }
    EXPECT(load(f.phase_ns[sp::shadows]) >= 2 * ms);
    EXPECT(load(f.part_ns[sp::part_lock_wait]) >= 2 * ms);
    EXPECT(load(f.part_ns[sp::part_lists]) < load(f.phase_ns[sp::shadows]));
    // Contended lock_timed inside a WorkerScope bills worker_lock_wait_ns without polluting title-thread lock waits.
    {
        const std::uint64_t title_waits_before = load(f.part_ns[sp::part_lock_wait]);
        std::mutex worker_m;
        std::atomic<bool> held_flag{false};
        std::thread holder([&] {
            const std::lock_guard held(worker_m);
            held_flag.store(true, std::memory_order_release);
            std::this_thread::sleep_for(std::chrono::milliseconds(4));
        });
        while (!held_flag.load(std::memory_order_acquire)) std::this_thread::yield();
        std::thread worker([&] {
            const sp::WorkerScope busy(sp::worker_submit);
            std::unique_lock lock(worker_m, std::defer_lock);
            sp::lock_timed(lock);
        });
        worker.join();
        holder.join();
        EXPECT(load(f.worker_lock_wait_ns) >= 2 * ms);
        EXPECT(load(f.part_ns[sp::part_lock_wait]) == title_waits_before);
    }
    sp::frame_lines(1.0);
    EXPECT(load(f.worker_lock_wait_ns) == 0);
    std::printf("scripted: frame sums, title and own threads, worker lock waits, worst frame and floor as scripted\n");
    return 0;
}

int submissions() {
    vktest::Device d;
    if (!d.create()) {
        std::printf("no usable Vulkan device: the submissions part is skipped\n");
        return 0;
    }
    VulkanContext context(d.device_proc, d.device, true);
    EXPECT(context.loaded());
    VulkanQueue direct(context, d.queue, d.family, VulkanQueue::QueueKind::direct);
    EXPECT(direct.ok());
    CommandList list;
    EXPECT(list.close());
    const CommandList* lists[] = {&list};
    sp::FrameTotals& f = sp::frame_totals();
    const std::uint64_t phases_before = phase_total(), submitted_before = load(sp::totals().submissions);
    const std::uint64_t all_before = load(f.all_count), all_ns_before = load(f.all_ns);
    const std::uint64_t floor_before = load(f.floor_count), direct_before = load(f.submissions[0]);
    std::uint64_t floor_ns_before = 0;
    for (const auto& p : f.floor_ns) floor_ns_before += load(p);
    constexpr std::uint64_t kSubmissions = 20;
    for (std::uint64_t i = 0; i < kSubmissions; ++i) EXPECT(direct.execute(lists) == ReplayResult::success);
    EXPECT(direct.idle());
    std::uint64_t floor_ns = 0;
    for (const auto& p : f.floor_ns) floor_ns += load(p);
    const std::uint64_t phases = phase_total() - phases_before;
    std::printf("submissions: %llu, phases %.3f ms in all, floor %llu of them, %.3f ms\n",
                static_cast<unsigned long long>(load(f.all_count) - all_before), double(phases) * 1e-6,
                static_cast<unsigned long long>(load(f.floor_count) - floor_before),
                double(floor_ns - floor_ns_before) * 1e-6);
    EXPECT(load(sp::totals().submissions) - submitted_before == kSubmissions);
    EXPECT(load(f.submissions[0]) - direct_before == kSubmissions);
    EXPECT(load(f.all_count) - all_before == kSubmissions);
    EXPECT(load(f.floor_count) - floor_before == kSubmissions);  // empty lists: nothing to bring up to date
    // Every phase of these submissions ran on this thread inside them: the sums agree to the nanosecond.
    EXPECT(load(f.all_ns) - all_ns_before == phases);
    EXPECT(floor_ns - floor_ns_before == phases);
    EXPECT(phases > 0);
    return 0;
}

// Queue sync contract, unlocked state_mutex_, per-batch FIFO drain, active_seq_ / after_held(),
// zero-copy CommandList snapshot, async worker, and QueueHolds progress tracking tests.
int queue_sync_and_async() {
    vktest::Device d;
    if (!d.create()) {
        std::printf("no usable Vulkan device: the queue_sync_and_async part is skipped\n");
        return 0;
    }

    constexpr VkDeviceSize kBufBytes = 256;
    const VkBufferUsageFlags usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    auto make_copy_list = [](ResourceId dst, std::uint64_t dst_off, ResourceId src, std::uint64_t src_off,
                             std::uint64_t bytes) -> std::shared_ptr<CommandList> {
        auto list = std::make_shared<CommandList>();
        if (!list->copy_buffer({dst, dst_off, src, src_off, bytes}) || !list->close()) return {};
        return list;
    };

    // 1. Synchronous execute() arriving while executing_ == true:
    //    (a) With !must_hold(), waits on exec_cv_ and runs before_execute_() synchronously before returning.
    //    (b) With must_hold() (unmet Wait) or Signal, enqueues immediately under state_mutex_ without blocking on
    //        before_execute_(), preserves zero-copy CommandList snapshots across Reset(), and runs before_execute_()
    //        once per batch on drain.
    //    (c) Arriving while a releaser thread is draining multiple ready held batches waits until the ready held
    //        batches finish and then runs before_execute_() synchronously before returning (strict FIFO).
    {
        VulkanContext context(d.device_proc, d.device, true);
        EXPECT(context.loaded());
        VulkanQueue queue(context, d.queue, d.family, VulkanQueue::QueueKind::direct);
        EXPECT(queue.ok());
        queue.set_async_depth(0);

        void* src_map = nullptr;
        void* dst_map = nullptr;
        VkBuffer src = d.buffer(kBufBytes, usage, &src_map);
        VkBuffer dst = d.buffer(kBufBytes, usage, &dst_map);
        EXPECT(src && dst);
        auto* sb = static_cast<std::uint8_t*>(src_map);
        auto* db = static_cast<std::uint8_t*>(dst_map);
        std::memset(sb, 0, kBufBytes);
        std::memset(db, 0, kBufBytes);
        sb[0] = 0x11;
        sb[16] = 0x22;
        sb[32] = 0x33;

        ResourceBinding r_src{}, r_dst{};
        r_src.buffer = src;
        r_src.size = kBufBytes;
        r_dst.buffer = dst;
        r_dst.size = kBufBytes;
        r_dst.host_readable = true;
        context.add_resource(1, r_src);
        context.add_resource(2, r_dst);

        std::atomic<unsigned> before_calls{0};
        std::atomic<bool> in_first_before{false};
        std::atomic<bool> release_first_before{false};
        context.set_before_execute([&] {
            const unsigned n = before_calls.fetch_add(1, std::memory_order_acq_rel) + 1;
            if (n == 1) {
                in_first_before.store(true, std::memory_order_release);
                while (!release_first_before.load(std::memory_order_acquire)) std::this_thread::yield();
            }
        });

        // Each batch writes its marker to its own slot AND overwrites slot 0 (final value proves strict FIFO order).
        auto b1 = std::make_shared<CommandList>();
        EXPECT(b1->copy_buffer({2, 16, 1, 0, 1}) && b1->copy_buffer({2, 0, 1, 0, 1}) && b1->close());
        auto b2 = std::make_shared<CommandList>();
        EXPECT(b2->copy_buffer({2, 32, 1, 16, 1}) && b2->copy_buffer({2, 0, 1, 16, 1}) && b2->close());
        auto b3 = std::make_shared<CommandList>();
        EXPECT(b3->copy_buffer({2, 48, 1, 32, 1}) && b3->copy_buffer({2, 0, 1, 32, 1}) && b3->close());

        // (a) Unheld sync execute() while executing_ == true must wait on exec_cv_ and run before_execute_().
        ReplayResult t1_res = ReplayResult::unsupported, t2_res = ReplayResult::unsupported;
        std::atomic<bool> t2_done{false};
        std::atomic<unsigned> t2_before_at_return{0};
        std::thread worker1([&] {
            std::shared_ptr<const CommandList> s1 = b1;
            t1_res = queue.execute({&s1, 1});
        });
        while (!in_first_before.load(std::memory_order_acquire)) std::this_thread::yield();

        std::thread worker2([&] {
            std::shared_ptr<const CommandList> s2 = b2;
            t2_res = queue.execute({&s2, 1});
            t2_before_at_return.store(before_calls.load(std::memory_order_acquire), std::memory_order_release);
            t2_done.store(true, std::memory_order_release);
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        EXPECT(!t2_done.load(std::memory_order_acquire));
        EXPECT(before_calls.load(std::memory_order_acquire) == 1);

        release_first_before.store(true, std::memory_order_release);
        worker1.join();
        worker2.join();
        EXPECT(t1_res == ReplayResult::success && t2_res == ReplayResult::success);
        EXPECT(t2_before_at_return.load(std::memory_order_acquire) == 2);
        EXPECT(queue.idle());
        EXPECT(db[16] == 0x11 && db[32] == 0x22 && db[0] == 0x22);

        // (b) must_hold() ops (behind an unmet Wait) and Signal enqueue immediately without blocking on executing_.
        std::memset(db, 0, kBufBytes);
        before_calls.store(0, std::memory_order_release);
        in_first_before.store(false, std::memory_order_release);
        release_first_before.store(false, std::memory_order_release);
        VulkanFence gate(context, 0), done(context, 0);
        EXPECT(gate.semaphore() != VK_NULL_HANDLE && done.semaphore() != VK_NULL_HANDLE);

        t1_res = ReplayResult::unsupported;
        std::thread worker_b([&] {
            std::shared_ptr<const CommandList> s1 = b1;
            t1_res = queue.execute({&s1, 1});
        });
        while (!in_first_before.load(std::memory_order_acquire)) std::this_thread::yield();

        // While worker_b is blocked inside before_execute_(), this thread queues wait(gate), b2, b3, and Signal(done).
        // If state_mutex_ were held across before_execute_() or wait-blocked ops waited on exec_cv_, this would
        // deadlock before release_first_before is set!
        queue.wait(gate.semaphore(), 1);
        std::shared_ptr<const CommandList> s2 = b2;
        std::shared_ptr<const CommandList> s3 = b3;
        EXPECT(queue.execute({&s2, 1}) == ReplayResult::success);
        EXPECT(queue.execute({&s3, 1}) == ReplayResult::success);
        EXPECT(queue.signal(done.semaphore(), 1));

        // Zero-copy proof: queue holds references to b2 and b3 without copying; simulating Reset() when use_count > 1
        // allocates fresh storage and never mutates the in-flight snapshot.
        EXPECT(b2.use_count() > 1 && b3.use_count() > 1);
        b2 = std::make_shared<CommandList>();
        EXPECT(b2->copy_buffer({2, 0, 1, 0, 1}) && b2->close());  // distinct content (0x11), must not affect in-flight

        release_first_before.store(true, std::memory_order_release);
        worker_b.join();
        EXPECT(t1_res == ReplayResult::success);
        EXPECT(before_calls.load(std::memory_order_acquire) == 1);  // b2 & b3 stay held behind gate

        EXPECT(gate.signal(1));
        EXPECT(done.wait(1, 5'000'000'000ull));
        EXPECT(queue.idle());
        // b1 ran first (1), then held {b2, b3} coalesced into one before_execute_() + one submit (2).
        EXPECT(before_calls.load(std::memory_order_acquire) == 2);
        EXPECT(db[16] == 0x11 && db[32] == 0x22 && db[48] == 0x33 && db[0] == 0x33);
        s2.reset();
        s3.reset();
        EXPECT(b1.use_count() == 1 && b3.use_count() == 1);
        EXPECT(b3->reset());  // steady-state reuse with zero allocation

        // (c) Sync execute() arriving while a releaser drains [b1, b2 (coalesced in held_)] must wait for the held
        //     drain to finish and run before_execute_() for b3 synchronously before returning.
        auto b2_fresh = std::make_shared<CommandList>();
        EXPECT(b2_fresh->copy_buffer({2, 32, 1, 16, 1}) && b2_fresh->copy_buffer({2, 0, 1, 16, 1}) &&
               b2_fresh->close());
        EXPECT(b3->copy_buffer({2, 48, 1, 32, 1}) && b3->copy_buffer({2, 0, 1, 32, 1}) && b3->close());
        std::memset(db, 0, kBufBytes);
        before_calls.store(0, std::memory_order_release);
        in_first_before.store(false, std::memory_order_release);
        release_first_before.store(false, std::memory_order_release);
        VulkanFence gate_c(context, 0);
        EXPECT(gate_c.semaphore() != VK_NULL_HANDLE);

        queue.wait(gate_c.semaphore(), 1);
        std::shared_ptr<const CommandList> c1 = b1, c2 = b2_fresh, c3 = b3;
        EXPECT(queue.execute({&c1, 1}) == ReplayResult::success);
        EXPECT(queue.execute({&c2, 1}) == ReplayResult::success);
        EXPECT(before_calls.load(std::memory_order_acquire) == 0);

        std::thread releaser([&] { (void)gate_c.signal(1); });
        while (!in_first_before.load(std::memory_order_acquire)) std::this_thread::yield();

        ReplayResult title_res = ReplayResult::unsupported;
        std::atomic<bool> title_done{false};
        std::atomic<unsigned> title_before_at_return{0};
        std::thread title([&] {
            title_res = queue.execute({&c3, 1});
            title_before_at_return.store(before_calls.load(std::memory_order_acquire), std::memory_order_release);
            title_done.store(true, std::memory_order_release);
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        EXPECT(!title_done.load(std::memory_order_acquire));
        EXPECT(before_calls.load(std::memory_order_acquire) == 1);

        release_first_before.store(true, std::memory_order_release);
        releaser.join();
        title.join();
        EXPECT(title_res == ReplayResult::success);
        EXPECT(title_before_at_return.load(std::memory_order_acquire) == 2);
        EXPECT(queue.idle());
        EXPECT(db[16] == 0x11 && db[32] == 0x22 && db[48] == 0x33 && db[0] == 0x33);

        // (d) Gate front held on `queue`, signaled while another thread holds pump_mutex_ (so pump_ready() has not
        //     reached `queue` yet and executing_ == false with !held_.empty() && ready(held_.front())):
        //     sync execute() must drain the ready front and run before_execute_() for itself before returning!
        VulkanQueue other_q(context, d.queue, d.family, VulkanQueue::QueueKind::compute);
        EXPECT(other_q.ok());
        other_q.set_async_depth(0);
        VulkanFence gate_other(context, 0), gate_d(context, 0);
        EXPECT(gate_other.semaphore() && gate_d.semaphore());
        auto empty_other = std::make_shared<CommandList>();
        EXPECT(empty_other->uav_barrier({}) && empty_other->close());
        std::shared_ptr<const CommandList> s_other = empty_other;

        std::memset(db, 0, kBufBytes);
        std::atomic<bool> in_pumper_before{false};
        std::atomic<bool> release_pumper{false};
        std::atomic<unsigned> main_before_calls{0};
        const std::thread::id main_tid = std::this_thread::get_id();
        context.set_before_execute([&] {
            if (std::this_thread::get_id() == main_tid) {
                main_before_calls.fetch_add(1, std::memory_order_acq_rel);
            } else {
                in_pumper_before.store(true, std::memory_order_release);
                while (!release_pumper.load(std::memory_order_acquire)) std::this_thread::yield();
            }
        });

        other_q.wait(gate_other.semaphore(), 1);
        EXPECT(other_q.execute({&s_other, 1}) == ReplayResult::success);
        queue.wait(gate_d.semaphore(), 1);
        EXPECT(queue.execute({&c1, 1}) == ReplayResult::success);

        // Pumper thread holds QueueHolds::pump_mutex_ while blocked inside other_q's before_execute_().
        std::thread pumper([&] { (void)gate_other.signal(1); });
        while (!in_pumper_before.load(std::memory_order_acquire)) std::this_thread::yield();

        // Signal gate_d: QueueHolds::signaled() marks c1 ready, while pump_ready() returns immediately because
        // pumper holds pump_mutex_. Thus queue has executing_ == false, !held_.empty(), and ready(held_.front()).
        EXPECT(gate_d.signal(1));
        EXPECT(main_before_calls.load(std::memory_order_acquire) == 0);

        // Sync execute(c2) must drain c1 and execute c2 synchronously before returning (while pumper still holds pump).
        EXPECT(queue.execute({&c2, 1}) == ReplayResult::success);
        EXPECT(main_before_calls.load(std::memory_order_acquire) == 2);
        EXPECT(queue.idle());
        EXPECT(db[16] == 0x11 && db[32] == 0x22 && db[0] == 0x22);

        release_pumper.store(true, std::memory_order_release);
        pumper.join();
        EXPECT(other_q.idle());
        std::printf("queue sync: exec_cv_ sync wait, unlocked must_hold, releaser-drain sync, ready-front drain, zero-copy Reset ok\n");
    }

    // 2. after_held() must wait behind an active unheld synchronous execute() and an active unheld synchronous
    //    signal() (active_seq_ tracking via oldest_held_seq()).
    {
        VulkanContext context(d.device_proc, d.device, true);
        EXPECT(context.loaded());
        VulkanQueue queue(context, d.queue, d.family, VulkanQueue::QueueKind::direct);
        EXPECT(queue.ok());
        queue.set_async_depth(0);

        auto empty = std::make_shared<CommandList>();
        EXPECT(empty->uav_barrier({}) && empty->close());
        std::shared_ptr<const CommandList> snap = empty;

        // (a) Active unheld sync execute().
        std::atomic<bool> in_before{false}, release_before{false};
        std::atomic<unsigned> after_calls{0};
        context.set_before_execute([&] {
            in_before.store(true, std::memory_order_release);
            while (!release_before.load(std::memory_order_acquire)) std::this_thread::yield();
        });
        context.set_after_execute([&] { after_calls.fetch_add(1, std::memory_order_acq_rel); });

        ReplayResult res = ReplayResult::unsupported;
        std::thread exec_t([&] { res = queue.execute({&snap, 1}); });
        while (!in_before.load(std::memory_order_acquire)) std::this_thread::yield();

        std::atomic<bool> barrier_ran{false};
        std::atomic<unsigned> barrier_saw_after{0};
        int owner_a = 0;
        context.holds().after_held(&owner_a, [&] {
            barrier_saw_after.store(after_calls.load(std::memory_order_acquire), std::memory_order_release);
            barrier_ran.store(true, std::memory_order_release);
        });
        EXPECT(!barrier_ran.load(std::memory_order_acquire));

        release_before.store(true, std::memory_order_release);
        exec_t.join();
        context.holds().wait_barriers_drained(UINT64_MAX);
        EXPECT(res == ReplayResult::success);
        EXPECT(barrier_ran.load(std::memory_order_acquire));
        EXPECT(barrier_saw_after.load(std::memory_order_acquire) == 1);

        // (b) Active unheld sync signal().
        context.set_before_execute({});
        context.set_after_execute({});
        VulkanFence sig_fence(context, 0);
        EXPECT(sig_fence.semaphore() != VK_NULL_HANDLE);

        std::atomic<bool> sig_ok{false};
        std::atomic<bool> sig_barrier_ran{false};
        std::atomic<bool> sig_barrier_saw_signaled{false};
        int owner_b = 0;
        const std::uint64_t seq_before = context.holds().current_seq();
        std::thread sig_t;
        {
            std::lock_guard hold_submit(context.submit_mutex());
            sig_t = std::thread([&] { sig_ok.store(queue.signal(sig_fence.semaphore(), 1), std::memory_order_release); });
            // Wait until signal_sync() has acquired active_seq_ under state_mutex_ and is blocked in submit().
            while (context.holds().current_seq() == seq_before) std::this_thread::yield();

            context.holds().after_held(&owner_b, [&] {
                sig_barrier_saw_signaled.store(!context.holds().pending(sig_fence.semaphore(), 1),
                                               std::memory_order_release);
                sig_barrier_ran.store(true, std::memory_order_release);
            });
            EXPECT(!sig_barrier_ran.load(std::memory_order_acquire));
        }
        sig_t.join();
        context.holds().wait_barriers_drained(UINT64_MAX);
        EXPECT(sig_ok.load(std::memory_order_acquire));
        EXPECT(sig_barrier_ran.load(std::memory_order_acquire));
        EXPECT(sig_barrier_saw_signaled.load(std::memory_order_acquire));
        EXPECT(sig_fence.wait(1, 5'000'000'000ull));
        EXPECT(queue.idle());
        std::printf("queue active_seq_: after_held waits behind active unheld execute() and signal() ok\n");
    }

    // 3. Held batches drain in strict FIFO order with before_execute_() called once per batch (no cross-batch
    //    coalescing), and an invalid held batch reports its failure without dropping valid peers.
    {
        VulkanContext context(d.device_proc, d.device, true);
        EXPECT(context.loaded());
        VulkanQueue queue(context, d.queue, d.family, VulkanQueue::QueueKind::direct);
        VulkanQueue copy_q(context, d.queue, d.family, VulkanQueue::QueueKind::copy);
        EXPECT(queue.ok() && copy_q.ok());
        queue.set_async_depth(0);
        copy_q.set_async_depth(0);

        void* src_map = nullptr;
        void* dst_map = nullptr;
        VkBuffer src = d.buffer(kBufBytes, usage, &src_map);
        VkBuffer dst = d.buffer(kBufBytes, usage, &dst_map);
        EXPECT(src && dst);
        auto* sb = static_cast<std::uint8_t*>(src_map);
        auto* db = static_cast<std::uint8_t*>(dst_map);
        std::memset(db, 0, kBufBytes);
        for (int i = 0; i < 6; ++i) sb[i * 16] = std::uint8_t(0xA1 + i);

        ResourceBinding r_src{}, r_dst{};
        r_src.buffer = src;
        r_src.size = kBufBytes;
        r_dst.buffer = dst;
        r_dst.size = kBufBytes;
        r_dst.host_readable = true;
        context.add_resource(1, r_src);
        context.add_resource(2, r_dst);

        std::atomic<unsigned> before_calls{0};
        context.set_before_execute([&] { before_calls.fetch_add(1, std::memory_order_relaxed); });

        VulkanFence gate(context, 0), mid(context, 0), done(context, 0);
        EXPECT(gate.semaphore() && mid.semaphore() && done.semaphore());

        auto make_step = [&](int idx) {
            auto l = std::make_shared<CommandList>();
            if (!l->copy_buffer({2, std::uint64_t((idx + 1) * 16), 1, std::uint64_t(idx * 16), 1}) ||
                !l->copy_buffer({2, 0, 1, std::uint64_t(idx * 16), 1}) || !l->close())
                return std::shared_ptr<const CommandList>{};
            return std::shared_ptr<const CommandList>(std::move(l));
        };
        std::shared_ptr<const CommandList> steps[6];
        for (int i = 0; i < 6; ++i) {
            steps[i] = make_step(i);
            EXPECT(steps[i] != nullptr);
        }

        queue.wait(gate.semaphore(), 1);
        EXPECT(queue.execute({&steps[0], 1}) == ReplayResult::success);
        EXPECT(queue.execute({&steps[1], 1}) == ReplayResult::success);
        EXPECT(queue.execute({&steps[2], 1}) == ReplayResult::success);
        EXPECT(queue.signal(mid.semaphore(), 1));
        EXPECT(queue.execute({&steps[3], 1}) == ReplayResult::success);
        queue.wait(mid.semaphore(), 1);
        EXPECT(queue.execute({&steps[4], 1}) == ReplayResult::success);
        EXPECT(queue.execute({&steps[5], 1}) == ReplayResult::success);
        EXPECT(queue.signal(done.semaphore(), 1));

        EXPECT(before_calls.load() == 0);
        EXPECT(gate.signal(1));
        EXPECT(done.wait(1, 5'000'000'000ull));
        EXPECT(queue.idle());

        // Held consecutive batches coalesce across {steps[0..2]} (1), split at Signal(mid), run {steps[3]} (2),
        // and either split at Wait(mid) (3) or coalesce {steps[3..5]} if mid already completed on the GPU (2).
        EXPECT(before_calls.load() == 2 || before_calls.load() == 3);
        for (int i = 0; i < 6; ++i) EXPECT(db[(i + 1) * 16] == std::uint8_t(0xA1 + i));
        EXPECT(db[0] == std::uint8_t(0xA6));  // strict FIFO through all 6 batches

        // Invalid held batch [valid, invalid ClearRenderTarget on copy queue, valid] must not drop valid peers.
        VulkanFence gate2(context, 0), done2(context, 0);
        EXPECT(gate2.semaphore() && done2.semaphore());
        std::memset(db, 0, kBufBytes);
        auto bad_clear = std::make_shared<CommandList>();
        TargetView tv{};
        tv.resource = 2;
        tv.layer_count = 1;
        EXPECT(bad_clear->clear_render_target(tv, ClearColor{}) && bad_clear->close());
        std::shared_ptr<const CommandList> bad_snap = bad_clear;
        copy_q.wait(gate2.semaphore(), 1);
        EXPECT(copy_q.execute({&steps[0], 1}) == ReplayResult::success);
        EXPECT(copy_q.execute({&bad_snap, 1}) == ReplayResult::success);
        EXPECT(copy_q.execute({&steps[1], 1}) == ReplayResult::success);
        EXPECT(copy_q.signal(done2.semaphore(), 1));
        EXPECT(gate2.signal(1));
        EXPECT(done2.wait(1, 5'000'000'000ull));
        EXPECT(copy_q.idle());
        EXPECT(copy_q.take_held_failure() == ReplayResult::unsupported);
        EXPECT(db[16] == 0xA1 && db[32] == 0xA2 && db[0] == 0xA2);
        std::printf("queue held FIFO drain: 6 batches -> 3 merged before_execute_ calls, invalid peer isolation ok\n");
    }

    // 4. LEONIDA_SUBMIT_ASYNC (per-queue async worker depth=4): same-queue inbox coalescing, Present barrier split &
    //    ordering, held-batch release coalescing through worker, and zero-copy CommandList Reset() while queued.
    {
        VulkanContext context(d.device_proc, d.device, true);
        EXPECT(context.loaded());
        VulkanQueue queue(context, d.queue, d.family, VulkanQueue::QueueKind::direct);
        EXPECT(queue.ok());
        queue.set_async_depth(4);

        void* src_map = nullptr;
        void* dst_map = nullptr;
        VkBuffer src = d.buffer(kBufBytes, usage, &src_map);
        VkBuffer dst = d.buffer(kBufBytes, usage, &dst_map);
        EXPECT(src && dst);
        auto* sb = static_cast<std::uint8_t*>(src_map);
        auto* db = static_cast<std::uint8_t*>(dst_map);
        std::memset(db, 0, kBufBytes);
        for (int i = 0; i < 4; ++i) sb[i * 16] = std::uint8_t(0xC1 + i);

        ResourceBinding r_src{}, r_dst{};
        r_src.buffer = src;
        r_src.size = kBufBytes;
        r_dst.buffer = dst;
        r_dst.size = kBufBytes;
        r_dst.host_readable = true;
        context.add_resource(1, r_src);
        context.add_resource(2, r_dst);

        std::atomic<unsigned> before_calls{0};
        std::atomic<bool> in_first{false};
        std::atomic<bool> go_first{false};
        context.set_before_execute([&] {
            const unsigned n = before_calls.fetch_add(1, std::memory_order_acq_rel) + 1;
            if (n == 1) {
                in_first.store(true, std::memory_order_release);
                while (!go_first.load(std::memory_order_acquire)) std::this_thread::yield();
            }
        });

        auto c0 = make_copy_list(2, 16, 1, 0, 1);
        auto c1 = make_copy_list(2, 32, 1, 16, 1);
        auto c2 = make_copy_list(2, 48, 1, 32, 1);
        auto c3 = make_copy_list(2, 64, 1, 48, 1);
        EXPECT(c0 && c1 && c2 && c3);

        std::shared_ptr<const CommandList> s0 = c0, s1 = c1, s2 = c2, s3 = c3;
        EXPECT(queue.execute({&s0, 1}) == ReplayResult::success);
        while (!in_first.load(std::memory_order_acquire)) std::this_thread::yield();

        // Enqueue c1, c2, then a Present after_held barrier, then c3.
        EXPECT(queue.execute({&s1, 1}) == ReplayResult::success);
        EXPECT(queue.execute({&s2, 1}) == ReplayResult::success);
        EXPECT(c1.use_count() > 1 && c2.use_count() > 1);  // queued zero-copy

        std::atomic<unsigned> barrier_seen_before_calls{0};
        std::atomic<bool> barrier_ran{false};
        int barrier_owner = 0;
        context.holds().after_held(&barrier_owner, [&] {
            barrier_seen_before_calls.store(before_calls.load(std::memory_order_acquire), std::memory_order_release);
            barrier_ran.store(true, std::memory_order_release);
        });

        EXPECT(queue.execute({&s3, 1}) == ReplayResult::success);
        VulkanFence done(context, 0);
        EXPECT(done.semaphore() != VK_NULL_HANDLE);
        EXPECT(queue.signal(done.semaphore(), 1));

        // Simulate Reset() of c1 while still in inbox: allocating new storage leaves in-flight snapshot intact.
        c1 = make_copy_list(2, 32, 1, 0, 1);
        EXPECT(c1 != nullptr);

        go_first.store(true, std::memory_order_release);
        EXPECT(done.wait(1, 5'000'000'000ull));
        EXPECT(queue.idle());
        EXPECT(barrier_ran.load(std::memory_order_acquire));

        // 3 before_execute_() calls total: {c0}, coalesced {c1, c2}, {c3}; barrier ran after {c1, c2} (count == 2)!
        EXPECT(before_calls.load() == 3);
        EXPECT(barrier_seen_before_calls.load() == 2);
        EXPECT(db[16] == 0xC1 && db[32] == 0xC2 && db[48] == 0xC3 && db[64] == 0xC4);

        // Held batches behind a gate in async mode: release_ready() wakes worker and coalesces {s0, s2} into 1 submit.
        before_calls.store(0);
        std::memset(db, 0, kBufBytes);
        VulkanFence gate(context, 0), done2(context, 0);
        EXPECT(gate.semaphore() && done2.semaphore());
        const std::uint64_t title_replay_before = load(sp::frame_totals().phase_ns[sp::replay]);
        queue.wait(gate.semaphore(), 1);
        EXPECT(queue.execute({&s0, 1}) == ReplayResult::success);
        EXPECT(queue.execute({&s2, 1}) == ReplayResult::success);
        EXPECT(queue.signal(done2.semaphore(), 1));
        EXPECT(queue.idle());  // flushes inbox into held_ while gate is still closed
        EXPECT(before_calls.load() == 0);
        EXPECT(gate.signal(1));
        EXPECT(done2.wait(1, 5'000'000'000ull));
        EXPECT(queue.idle());
        EXPECT(before_calls.load() == 1);
        EXPECT(db[16] == 0xC1 && db[48] == 0xC3);
        // WorkerScope(worker_submit) covers release_held() on async_worker, so title-thread replay_ns stays unchanged!
        EXPECT(load(sp::frame_totals().phase_ns[sp::replay]) == title_replay_before);
        std::printf("queue async (depth=4): inbox coalescing, Present barrier split, gate release & WorkerScope ok\n");
    }

    // 5. Async queue progress tracking (LEONIDA_SUBMIT_ASYNC=4 + queued Present + busy producer worker + genuine
    //    deadlock):
    //    (a) While the direct producer async worker is busy inside before_execute_() with a WriteGroup Signal
    //        still queued in its inbox_, a consumer compute gate with a Present queued behind it must NOT be
    //        released unmet (queue_holds::any_progress() == true -> list_value_progress::Reason::hold).
    //    (b) Once all async workers finish draining into held_ and the GPU is idle, two async queues holding
    //        unsatisfied waits must NOT ping-pong release_ready(), queue_holds::any_progress() settles to false,
    //        and a genuine Present deadlock breaks cleanly.
    //    (c) In synchronous mode, executing_ == true (unlocked before_execute_()) also keeps any_progress() true.
    {
        namespace lvp = leonida::list_value_progress;
        const auto met_ge = [](std::uint32_t, std::uint64_t v, std::uint64_t ref, bool) { return v >= ref; };

        VulkanContext context(d.device_proc, d.device, true);
        EXPECT(context.loaded());
        VulkanQueue direct(context, d.queue, d.family, VulkanQueue::QueueKind::direct);
        VulkanQueue compute(context, d.queue, d.family, VulkanQueue::QueueKind::compute);
        EXPECT(direct.ok() && compute.ok());
        direct.set_async_depth(4);
        compute.set_async_depth(4);

        auto empty_list = std::make_shared<CommandList>();
        EXPECT(empty_list->uav_barrier({}) && empty_list->close());
        std::shared_ptr<const CommandList> snap = empty_list;

        std::atomic<bool> hold_before{true};
        std::atomic<bool> in_before{false};
        context.set_before_execute([&] {
            if (hold_before.load(std::memory_order_acquire)) {
                in_before.store(true, std::memory_order_release);
                while (hold_before.load(std::memory_order_acquire)) std::this_thread::yield();
            }
        });

        VulkanFence producer_written(context, 0), consumer_gate(context, 0), consumer_done(context, 0);
        EXPECT(producer_written.semaphore() && consumer_gate.semaphore() && consumer_done.semaphore());

        // (a) Direct worker starts batch 0 and blocks inside before_execute_().
        EXPECT(direct.execute({&snap, 1}) == ReplayResult::success);
        while (!in_before.load(std::memory_order_acquire)) std::this_thread::yield();

        // Compute queue holds a batch behind consumer_gate (simulating wait64 on title memory).
        compute.wait(consumer_gate.semaphore(), 1);
        EXPECT(compute.execute({&snap, 1}) == ReplayResult::success);
        EXPECT(compute.signal(consumer_done.semaphore(), 1));

        // Direct queue queues the producer batch + Signal(producer_written, 1) behind the blocked batch in inbox_.
        EXPECT(direct.execute({&snap, 1}) == ReplayResult::success);
        EXPECT(direct.signal(producer_written.semaphore(), 1));

        // Title queues Present (after_held) and returns to idle (title_calls == 0).
        std::atomic<bool> present_ran{false};
        int present_owner = 0;
        context.holds().after_held(&present_owner, [&] { present_ran.store(true, std::memory_order_release); });

        // Wait briefly for compute's worker to move its batch into held_, while direct's worker stays busy.
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        // Producer's Signal has NOT reached Vulkan yet (still in direct.inbox_!), so pending() is true — the exact
        // condition where Pump::choose_release() sees wg.running == false and would falsely break at Present
        // without QueueHolds progress tracking.
        EXPECT(context.holds().pending(producer_written.semaphore(), 1));
        EXPECT(queue_holds::any_progress());

        const lvp::WaitPredicate wait_val[] = {{0x1000, 1, 7, true}};
        const lvp::PendingWrite write_val[] = {{0x1000, 1, 8, false}};
        const lvp::PendingGate gates[] = {
            {&compute, 1, false, false, wait_val, /*presents_at_hold=*/0, /*held_ms=*/0}};
        const lvp::PendingWriteGroup writes[] = {{&direct, 0, /*running=*/false, write_val}};
        lvp::Activity at_present;
        at_present.presents = 1;
        at_present.now_ms = 250;
        at_present.title_call_ms = 10;
        at_present.title_submitting = queue_holds::any_progress();
        EXPECT(lvp::decide(writes, gates, at_present, met_ge).reason == lvp::Reason::hold);
        EXPECT(!present_ran.load(std::memory_order_acquire));

        // Unblock direct's worker: producer executes and submits Signal(producer_written, 1) cleanly.
        hold_before.store(false, std::memory_order_release);
        EXPECT(producer_written.wait(1, 5'000'000'000ull));
        EXPECT(direct.idle());
        EXPECT(!context.holds().pending(producer_written.semaphore(), 1));

        // Now consumer_gate is met: signaling it wakes compute's worker, which drains consumer_batch and Present.
        EXPECT(consumer_gate.signal(1));
        EXPECT(consumer_done.wait(1, 5'000'000'000ull));
        EXPECT(compute.idle());
        context.holds().wait_barriers_drained(UINT64_MAX);
        EXPECT(present_ran.load(std::memory_order_acquire));

        // (b) Genuine deadlock across BOTH async queues (no producer can ever signal): any_progress() must settle
        //     to false (zero cross-queue release_ready() ping-pong) so Present breaks the deadlock cleanly.
        VulkanFence dead_gate_a(context, 0), dead_gate_b(context, 0), dead_done_a(context, 0), dead_done_b(context, 0);
        EXPECT(dead_gate_a.semaphore() && dead_gate_b.semaphore() && dead_done_a.semaphore() &&
               dead_done_b.semaphore());
        direct.wait(dead_gate_a.semaphore(), 1);
        EXPECT(direct.execute({&snap, 1}) == ReplayResult::success);
        EXPECT(direct.signal(dead_done_a.semaphore(), 1));
        compute.wait(dead_gate_b.semaphore(), 1);
        EXPECT(compute.execute({&snap, 1}) == ReplayResult::success);
        EXPECT(compute.signal(dead_done_b.semaphore(), 1));

        EXPECT(direct.idle());
        EXPECT(compute.idle());
        // Wait up to 500 ms for any trailing pump_ready() to finish; must settle to false and stay false.
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
        while (queue_holds::any_progress() && std::chrono::steady_clock::now() < deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        EXPECT(!queue_holds::any_progress());
        at_present.title_submitting = queue_holds::any_progress();
        const lvp::Decision broken = lvp::decide({}, gates, at_present, met_ge);
        EXPECT(broken.reason == lvp::Reason::present && broken.gate == 0);

        // Clean teardown of held deadlock ops.
        EXPECT(dead_gate_a.signal(1));
        EXPECT(dead_gate_b.signal(1));
        EXPECT(dead_done_a.wait(1, 5'000'000'000ull));
        EXPECT(dead_done_b.wait(1, 5'000'000'000ull));
        EXPECT(direct.idle() && compute.idle());

        // (c) Synchronous executing_ == true (unlocked before_execute_()) also keeps any_progress() true.
        VulkanQueue sync_q(context, d.queue, d.family, VulkanQueue::QueueKind::direct);
        EXPECT(sync_q.ok());
        sync_q.set_async_depth(0);
        hold_before.store(true, std::memory_order_release);
        in_before.store(false, std::memory_order_release);
        ReplayResult sync_res = ReplayResult::unsupported;
        std::thread sync_t([&] { sync_res = sync_q.execute({&snap, 1}); });
        while (!in_before.load(std::memory_order_acquire)) std::this_thread::yield();
        EXPECT(queue_holds::any_progress());
        hold_before.store(false, std::memory_order_release);
        sync_t.join();
        EXPECT(sync_res == ReplayResult::success);
        EXPECT(sync_q.idle());
        EXPECT(!queue_holds::any_progress());
        std::printf("queue async progress: held gate protected during busy worker/executing_, genuine deadlock breaks ok\n");
    }

    // 6. Explicit async inbox submit merging + vkQueueSubmit count + Signal / Wait / Present splits:
    //    Consecutive Call::Kind::execute items in async_worker() coalesce into ONE before_execute_() + ONE
    //    command-buffer vkQueueSubmit, while intervening Signal, Wait, and Present (after_held) each split cleanly.
    {
        static PFN_vkQueueSubmit s_real_submit = nullptr;
        static std::atomic<unsigned> s_cb_submits{0};
        struct HookSubmit {
            static VKAPI_ATTR VkResult VKAPI_CALL call(VkQueue q, std::uint32_t count, const VkSubmitInfo* infos,
                                                       VkFence fence) {
                for (std::uint32_t i = 0; i < count; ++i)
                    if (infos[i].commandBufferCount > 0) s_cb_submits.fetch_add(1, std::memory_order_relaxed);
                return s_real_submit(q, count, infos, fence);
            }
        };

        VulkanContext context(d.device_proc, d.device, true);
        EXPECT(context.loaded());
        s_real_submit = context.vk.vkQueueSubmit;
        s_cb_submits.store(0, std::memory_order_relaxed);
        context.vk.vkQueueSubmit = &HookSubmit::call;

        VulkanQueue queue(context, d.queue, d.family, VulkanQueue::QueueKind::direct);
        EXPECT(queue.ok());
        queue.set_async_depth(12);

        void* src_map = nullptr;
        void* dst_map = nullptr;
        VkBuffer src = d.buffer(kBufBytes, usage, &src_map);
        VkBuffer dst = d.buffer(kBufBytes, usage, &dst_map);
        EXPECT(src && dst);
        auto* sb = static_cast<std::uint8_t*>(src_map);
        auto* db = static_cast<std::uint8_t*>(dst_map);
        std::memset(db, 0, kBufBytes);
        for (int i = 0; i < 9; ++i) sb[i * 16] = std::uint8_t(0xD0 + i);

        ResourceBinding r_src{}, r_dst{};
        r_src.buffer = src;
        r_src.size = kBufBytes;
        r_dst.buffer = dst;
        r_dst.size = kBufBytes;
        r_dst.host_readable = true;
        context.add_resource(1, r_src);
        context.add_resource(2, r_dst);

        std::atomic<unsigned> before_calls{0};
        std::atomic<bool> in_first{false};
        std::atomic<bool> go_first{false};
        context.set_before_execute([&] {
            const unsigned n = before_calls.fetch_add(1, std::memory_order_acq_rel) + 1;
            if (n == 1) {
                in_first.store(true, std::memory_order_release);
                while (!go_first.load(std::memory_order_acquire)) std::this_thread::yield();
            }
        });

        std::shared_ptr<const CommandList> steps[9];
        for (int i = 0; i < 9; ++i) {
            auto l = std::make_shared<CommandList>();
            EXPECT(l->copy_buffer({2, std::uint64_t((i + 1) * 16), 1, std::uint64_t(i * 16), 1}) &&
                   l->copy_buffer({2, 0, 1, std::uint64_t(i * 16), 1}) && l->close());
            steps[i] = std::move(l);
        }

        VulkanFence mid(context, 0), done(context, 0);
        EXPECT(mid.semaphore() && done.semaphore());

        // Batch 0 starts on worker and holds in before_execute_() while the remaining 8 batches queue in inbox_.
        EXPECT(queue.execute({&steps[0], 1}) == ReplayResult::success);
        while (!in_first.load(std::memory_order_acquire)) std::this_thread::yield();

        // Group A: 3 consecutive execute batches -> coalesces into 1 submit (#2).
        EXPECT(queue.execute({&steps[1], 1}) == ReplayResult::success);
        EXPECT(queue.execute({&steps[2], 1}) == ReplayResult::success);
        EXPECT(queue.execute({&steps[3], 1}) == ReplayResult::success);
        // Split 1: Signal(mid, 1).
        EXPECT(queue.signal(mid.semaphore(), 1));
        // Group B: 2 consecutive execute batches -> coalesces into 1 submit (#3).
        EXPECT(queue.execute({&steps[4], 1}) == ReplayResult::success);
        EXPECT(queue.execute({&steps[5], 1}) == ReplayResult::success);
        // Split 2: Wait(mid, 1).
        queue.wait(mid.semaphore(), 1);
        // Group C: 2 consecutive execute batches -> coalesces into 1 submit (#4).
        EXPECT(queue.execute({&steps[6], 1}) == ReplayResult::success);
        EXPECT(queue.execute({&steps[7], 1}) == ReplayResult::success);
        // Split 3: Present barrier (after_held).
        std::atomic<unsigned> present_saw_submits{0};
        int present_owner = 0;
        context.holds().after_held(&present_owner, [&] {
            present_saw_submits.store(s_cb_submits.load(std::memory_order_acquire), std::memory_order_release);
        });
        // Group D: 1 execute batch after Present -> 1 submit (#5).
        EXPECT(queue.execute({&steps[8], 1}) == ReplayResult::success);
        EXPECT(queue.signal(done.semaphore(), 1));

        go_first.store(true, std::memory_order_release);
        EXPECT(done.wait(1, 5'000'000'000ull));
        EXPECT(queue.idle());

        // 9 execute calls coalesced into 4 (when mid already GPU-completed before Wait(mid)) or 5 submits!
        const unsigned bc = before_calls.load(std::memory_order_acquire);
        const unsigned sc = s_cb_submits.load(std::memory_order_acquire);
        const unsigned ps = present_saw_submits.load(std::memory_order_acquire);
        EXPECT((bc == 4 || bc == 5) && sc == bc && ps == bc - 1);
        for (int i = 0; i < 9; ++i) EXPECT(db[(i + 1) * 16] == std::uint8_t(0xD0 + i));
        EXPECT(db[0] == std::uint8_t(0xD8));  // exact FIFO overwrite order preserved across inter-batch barriers
        std::printf("queue async submit merging: 9 batches -> 5 submits, split cleanly on Signal/Wait/Present ok\n");
    }

    // 7. ASYNC enqueue-serial retirement epoch contract (set_on_enqueue_execute / set_on_finish_execute):
    //    A view retired with kAfterExecutes during batch 1's before_execute_() must NOT be stamped or reaped while
    //    an already-enqueued batch 2 is still waiting in async inbox_, even after batch 1's GPU work completes!
    {
        // Mirrors GeometryViews' exact execute_enqueue / retire(kAfterExecutes) / execute_end / reap state machine.
        struct EpochTracker {
            enum : std::uint64_t { kAfterExecutes = UINT64_MAX };
            struct Retired {
                std::uint64_t id = 0;
                std::uint64_t ticket = kAfterExecutes;
                std::uint64_t epoch = 0;
            };
            std::mutex mutex, execute_mutex;
            std::vector<std::uint64_t> executing;
            std::vector<Retired> retired;
            std::uint64_t execute_serial = 0;
            std::atomic<bool> unstamped{false};
            std::uint64_t reaped = 0;

            std::uint64_t execute_enqueue() {
                std::lock_guard lock(execute_mutex);
                executing.push_back(++execute_serial);
                return execute_serial;
            }
            void retire(std::uint64_t id) {
                std::lock_guard lock(mutex);
                std::uint64_t epoch = 0;
                {
                    std::lock_guard exec(execute_mutex);
                    epoch = execute_serial;
                }
                unstamped.store(true, std::memory_order_release);
                retired.push_back({id, kAfterExecutes, epoch});
            }
            void execute_end(std::uint64_t serial, const std::function<std::uint64_t()>& submitted) {
                if (!serial) return;
                {
                    std::lock_guard exec(execute_mutex);
                    std::erase(executing, serial);
                }
                if (!unstamped.load(std::memory_order_acquire)) return;
                std::lock_guard lock(mutex);
                std::uint64_t oldest = kAfterExecutes;
                {
                    std::lock_guard exec(execute_mutex);
                    for (const auto s : executing) oldest = std::min(oldest, s);
                }
                const std::uint64_t ticket = submitted();
                bool left = false;
                for (auto& r : retired) {
                    if (r.ticket != kAfterExecutes) continue;
                    if (r.epoch < oldest) r.ticket = ticket;
                    else left = true;
                }
                unstamped.store(left, std::memory_order_release);
            }
            std::size_t reap(std::uint64_t completed) {
                std::lock_guard lock(mutex);
                std::size_t freed = 0, keep = 0;
                for (std::size_t i = 0; i < retired.size(); ++i) {
                    if (retired[i].ticket != kAfterExecutes && retired[i].ticket <= completed) {
                        ++reaped;
                        ++freed;
                    } else {
                        if (keep != i) retired[keep] = retired[i];
                        ++keep;
                    }
                }
                retired.resize(keep);
                return freed;
            }
        } tracker;

        VulkanContext context(d.device_proc, d.device, true);
        EXPECT(context.loaded());
        context.set_on_enqueue_execute([&] { return tracker.execute_enqueue(); });
        context.set_on_finish_execute(
            [&](std::uint64_t serial) { tracker.execute_end(serial, [&] { return context.submitted_retire_ticket(); }); });

        VulkanQueue queue(context, d.queue, d.family, VulkanQueue::QueueKind::direct);
        EXPECT(queue.ok());
        queue.set_async_depth(4);

        auto empty_list = std::make_shared<CommandList>();
        EXPECT(empty_list->uav_barrier({}) && empty_list->close());
        std::shared_ptr<const CommandList> snap = empty_list;

        std::atomic<unsigned> before_n{0};
        std::atomic<bool> b1_entered{false}, b1_go{false}, b2_entered{false}, b2_go{false};
        context.set_before_execute([&] {
            const unsigned n = before_n.fetch_add(1, std::memory_order_acq_rel) + 1;
            if (n == 1) {
                b1_entered.store(true, std::memory_order_release);
                while (!b1_go.load(std::memory_order_acquire)) std::this_thread::yield();
                // Batch 1 retires view #1 AFTER batch 2 has already been enqueued in inbox_!
                tracker.retire(1);
            } else if (n == 2) {
                b2_entered.store(true, std::memory_order_release);
                while (!b2_go.load(std::memory_order_acquire)) std::this_thread::yield();
            }
        });

        // 1. Enqueue batch 1 (serial 1); wait until worker enters batch 1's before_execute_().
        EXPECT(queue.execute({&snap, 1}) == ReplayResult::success);
        while (!b1_entered.load(std::memory_order_acquire)) std::this_thread::yield();

        // 2. Title thread enqueues batch 2 (serial 2) + batch 3 (serial 3) into inbox_ while batch 1 is in flight!
        VulkanFence done(context, 0);
        EXPECT(done.semaphore() != VK_NULL_HANDLE);
        EXPECT(queue.execute({&snap, 1}) == ReplayResult::success);
        EXPECT(queue.execute({&snap, 1}) == ReplayResult::success);
        EXPECT(queue.signal(done.semaphore(), 1));

        // 3. Let batch 1 retire view #1, submit to Vulkan, and finish; worker then coalesces {batch 2, batch 3} and
        //    pauses at n == 2 inside before_execute_().
        b1_go.store(true, std::memory_order_release);
        while (!b2_entered.load(std::memory_order_acquire)) std::this_thread::yield();

        // Wait until batch 1's GPU submission has completely retired on the GPU!
        const std::uint64_t b1_ticket = context.submitted_retire_ticket();
        EXPECT(b1_ticket >= 1);
        EXPECT(context.wait_retire_ticket(b1_ticket) >= b1_ticket);

        // CRITICAL INVARIANT: Even though batch 1's GPU work is 100% done, view #1 MUST NOT be reaped because
        // {batch 2, batch 3} (enqueued BEFORE view #1 retired!) have not submitted yet!
        EXPECT(tracker.reap(context.completed_retire_ticket()) == 0);
        EXPECT(tracker.reaped == 0);

        // 4. Release {batch 2, batch 3}: both serials 2 and 3 finish after the coalesced vkQueueSubmit, stamping
        //    view #1 with the coalesced submit ticket; once GPU completes, view #1 reaps cleanly!
        b2_go.store(true, std::memory_order_release);
        EXPECT(done.wait(1, 5'000'000'000ull));
        EXPECT(queue.idle());
        EXPECT(tracker.reap(context.completed_retire_ticket()) == 1);
        EXPECT(tracker.reaped == 1);
        std::printf("queue async retirement epoch: view deferred across queued & coalesced inbox batches ok\n");
    }

    // 8. set_skip_before_execute(true) (internal staging_queue path): skips before_execute_(), after_execute_(),
    //    on_enqueue_execute_ / on_finish_execute_, non-shader bindless/prelude prep, and submit_phases accounting,
    //    while normal queues (and toggling back to false) still invoke all of them.
    {
        VulkanContext context(d.device_proc, d.device, true);
        EXPECT(context.loaded());

        void* src_map = nullptr;
        void* dst_map = nullptr;
        VkBuffer src = d.buffer(kBufBytes, usage, &src_map);
        VkBuffer dst = d.buffer(kBufBytes, usage, &dst_map);
        EXPECT(src && dst);
        auto* sb = static_cast<std::uint8_t*>(src_map);
        auto* db = static_cast<std::uint8_t*>(dst_map);
        std::memset(sb, 0, kBufBytes);
        std::memset(db, 0, kBufBytes);
        sb[0] = 0x5A;
        sb[16] = 0x6B;
        sb[32] = 0x7C;

        ResourceBinding r_src{}, r_dst{};
        r_src.buffer = src;
        r_src.size = kBufBytes;
        r_dst.buffer = dst;
        r_dst.size = kBufBytes;
        r_dst.host_readable = true;
        context.add_resource(1, r_src);
        context.add_resource(2, r_dst);

        std::atomic<unsigned> before_calls{0}, after_calls{0}, enqueue_calls{0}, finish_calls{0};
        std::atomic<unsigned> bindless_calls{0}, prelude_calls{0};
        std::atomic<std::uint64_t> next_serial{0};
        context.set_before_execute([&] { before_calls.fetch_add(1, std::memory_order_acq_rel); });
        context.set_after_execute([&] { after_calls.fetch_add(1, std::memory_order_acq_rel); });
        context.set_on_enqueue_execute([&] {
            enqueue_calls.fetch_add(1, std::memory_order_acq_rel);
            return next_serial.fetch_add(1, std::memory_order_acq_rel) + 1;
        });
        context.set_on_finish_execute([&](std::uint64_t) { finish_calls.fetch_add(1, std::memory_order_acq_rel); });
        context.set_bindless_prepare([&](VkCommandBuffer) {
            bindless_calls.fetch_add(1, std::memory_order_acq_rel);
            return true;
        });
        context.set_prelude(
            [&](VkCommandBuffer) {
                prelude_calls.fetch_add(1, std::memory_order_acq_rel);
                return false;
            },
            {});

        VulkanQueue staging_q(context, d.queue, d.family, VulkanQueue::QueueKind::direct);
        VulkanQueue normal_q(context, d.queue, d.family, VulkanQueue::QueueKind::direct);
        EXPECT(staging_q.ok() && normal_q.ok());
        staging_q.make_synchronous();
        staging_q.set_skip_before_execute(true);
        EXPECT(staging_q.skip_before_execute());
        EXPECT(!normal_q.skip_before_execute());
        normal_q.set_async_depth(0);

        auto copy0 = make_copy_list(2, 0, 1, 0, 1);
        auto copy1 = make_copy_list(2, 16, 1, 16, 1);
        auto copy2 = make_copy_list(2, 32, 1, 32, 1);
        EXPECT(copy0 && copy1 && copy2);

        const std::uint64_t subs_before = load(sp::totals().submissions);
        const std::uint64_t direct_before = load(sp::frame_totals().submissions[0]);
        const CommandList* raw0[] = {copy0.get()};
        EXPECT(staging_q.execute(raw0) == ReplayResult::success);
        EXPECT(staging_q.idle());
        EXPECT(db[0] == 0x5A);
        EXPECT(before_calls.load(std::memory_order_acquire) == 0);
        EXPECT(after_calls.load(std::memory_order_acquire) == 0);
        EXPECT(enqueue_calls.load(std::memory_order_acquire) == 0);
        EXPECT(finish_calls.load(std::memory_order_acquire) == 0);
        EXPECT(bindless_calls.load(std::memory_order_acquire) == 0);
        EXPECT(prelude_calls.load(std::memory_order_acquire) == 0);
        EXPECT(load(sp::totals().submissions) == subs_before);
        EXPECT(load(sp::frame_totals().submissions[0]) == direct_before);

        const CommandList* raw1[] = {copy1.get()};
        EXPECT(normal_q.execute(raw1) == ReplayResult::success);
        EXPECT(normal_q.idle());
        EXPECT(db[16] == 0x6B);
        EXPECT(before_calls.load(std::memory_order_acquire) == 1);
        EXPECT(after_calls.load(std::memory_order_acquire) == 1);
        EXPECT(enqueue_calls.load(std::memory_order_acquire) == 1);
        EXPECT(finish_calls.load(std::memory_order_acquire) == 1);
        EXPECT(bindless_calls.load(std::memory_order_acquire) == 1);
        EXPECT(prelude_calls.load(std::memory_order_acquire) == 1);
        EXPECT(load(sp::totals().submissions) == subs_before + 1);
        EXPECT(load(sp::frame_totals().submissions[0]) == direct_before + 1);

        // Toggling set_skip_before_execute(false) restores normal callbacks on the same queue.
        staging_q.set_skip_before_execute(false);
        EXPECT(!staging_q.skip_before_execute());
        const CommandList* raw2[] = {copy2.get()};
        EXPECT(staging_q.execute(raw2) == ReplayResult::success);
        EXPECT(staging_q.idle());
        EXPECT(db[32] == 0x7C);
        EXPECT(before_calls.load(std::memory_order_acquire) == 2);
        EXPECT(after_calls.load(std::memory_order_acquire) == 2);
        EXPECT(enqueue_calls.load(std::memory_order_acquire) == 2);
        EXPECT(finish_calls.load(std::memory_order_acquire) == 2);
        std::printf("queue set_skip_before_execute: staging skips hooks/phases while normal queue invokes all ok\n");
    }

    // 9. Cross-queue Signal() / pump_ready() -> release_ready() and no-op debug markers must NOT split coalescing
    //    (in either inbox_ or held_), while a real after_held() (Present) barrier DOES split coalescing and runs
    //    strictly between pre-barrier and post-barrier batches; coalesced batches release (K - 1) inbox_batches_
    //    slots immediately so async_depth_=4 does not stall enqueue() while a coalesced batch runs.
    {
        static PFN_vkQueueSubmit s_real_submit = nullptr;
        static std::atomic<unsigned> s_cb_submits{0};
        struct HookSubmit {
            static VKAPI_ATTR VkResult VKAPI_CALL call(VkQueue q, std::uint32_t count, const VkSubmitInfo* infos,
                                                       VkFence fence) {
                for (std::uint32_t i = 0; i < count; ++i)
                    if (infos[i].commandBufferCount > 0) s_cb_submits.fetch_add(1, std::memory_order_relaxed);
                return s_real_submit(q, count, infos, fence);
            }
        };

        VulkanContext context(d.device_proc, d.device, true);
        EXPECT(context.loaded());
        s_real_submit = context.vk.vkQueueSubmit;
        s_cb_submits.store(0, std::memory_order_relaxed);
        context.vk.vkQueueSubmit = &HookSubmit::call;

        VulkanQueue direct(context, d.queue, d.family, VulkanQueue::QueueKind::direct);
        VulkanQueue compute(context, d.queue, d.family, VulkanQueue::QueueKind::compute);
        EXPECT(direct.ok() && compute.ok());
        direct.set_async_depth(4);
        compute.set_async_depth(4);

        void* src_map = nullptr;
        void* dst_map = nullptr;
        VkBuffer src = d.buffer(kBufBytes, usage, &src_map);
        VkBuffer dst = d.buffer(kBufBytes, usage, &dst_map);
        EXPECT(src && dst);
        auto* sb = static_cast<std::uint8_t*>(src_map);
        auto* db = static_cast<std::uint8_t*>(dst_map);
        std::memset(db, 0, kBufBytes);
        for (int i = 0; i < 7; ++i) sb[i * 16] = std::uint8_t(0xE0 + i);

        ResourceBinding r_src{}, r_dst{};
        r_src.buffer = src;
        r_src.size = kBufBytes;
        r_dst.buffer = dst;
        r_dst.size = kBufBytes;
        r_dst.host_readable = true;
        context.add_resource(1, r_src);
        context.add_resource(2, r_dst);

        std::shared_ptr<const CommandList> steps[7];
        for (int i = 0; i < 7; ++i) {
            auto l = std::make_shared<CommandList>();
            EXPECT(l->copy_buffer({2, std::uint64_t((i + 1) * 16), 1, std::uint64_t(i * 16), 1}) &&
                   l->copy_buffer({2, 0, 1, std::uint64_t(i * 16), 1}) && l->close());
            steps[i] = std::move(l);
        }

        // (a) Cross-queue Signals (pump_ready() -> release_ready()) and no-op markers do NOT split inbox coalescing,
        //     while a real after_held() barrier DOES split coalescing strictly between {B1, B2} and {B3}.
        std::atomic<unsigned> before_calls{0};
        std::atomic<bool> in_first{false}, go_first{false};
        context.set_before_execute([&] {
            const unsigned n = before_calls.fetch_add(1, std::memory_order_acq_rel) + 1;
            if (n == 1) {
                in_first.store(true, std::memory_order_release);
                while (!go_first.load(std::memory_order_acquire)) std::this_thread::yield();
            }
        });

        VulkanFence cross_fence(context, 0), done_a(context, 0);
        EXPECT(cross_fence.semaphore() && done_a.semaphore());

        EXPECT(direct.execute({&steps[0], 1}) == ReplayResult::success);
        while (!in_first.load(std::memory_order_acquire)) std::this_thread::yield();

        EXPECT(direct.execute({&steps[1], 1}) == ReplayResult::success);
        // Cross-queue Signal triggers QueueHolds::pump_ready() -> direct.release_ready() while B1 sits in direct.inbox_.
        EXPECT(compute.signal(cross_fence.semaphore(), 1));
        EXPECT(cross_fence.wait(1, 5'000'000'000ull));
        EXPECT(compute.idle());
        // No-op debug marker (debug_labels() == false) must also not split inbox coalescing.
        direct.marker(DebugMarker::Kind::set, "noop-marker");
        EXPECT(direct.execute({&steps[2], 1}) == ReplayResult::success);
        EXPECT(compute.signal(cross_fence.semaphore(), 2));
        EXPECT(cross_fence.wait(2, 5'000'000'000ull));
        EXPECT(compute.idle());

        // Real after_held() (Present) barrier MUST split coalescing between {B1, B2} and {B3}.
        std::atomic<bool> barrier_ran{false};
        std::atomic<unsigned> barrier_seen_before{0}, barrier_seen_submits{0};
        int barrier_owner = 0;
        context.holds().after_held(&barrier_owner, [&] {
            barrier_seen_before.store(before_calls.load(std::memory_order_acquire), std::memory_order_release);
            barrier_seen_submits.store(s_cb_submits.load(std::memory_order_acquire), std::memory_order_release);
            barrier_ran.store(true, std::memory_order_release);
        });

        EXPECT(direct.execute({&steps[3], 1}) == ReplayResult::success);
        EXPECT(direct.signal(done_a.semaphore(), 1));

        go_first.store(true, std::memory_order_release);
        EXPECT(done_a.wait(1, 5'000'000'000ull));
        EXPECT(direct.idle());

        EXPECT(barrier_ran.load(std::memory_order_acquire));
        EXPECT(before_calls.load(std::memory_order_acquire) == 3);  // {B0}, coalesced {B1, B2}, {B3}
        EXPECT(s_cb_submits.load(std::memory_order_acquire) == 3);
        EXPECT(barrier_seen_before.load(std::memory_order_acquire) == 2);
        EXPECT(barrier_seen_submits.load(std::memory_order_acquire) == 2);
        for (int i = 0; i < 4; ++i) EXPECT(db[(i + 1) * 16] == std::uint8_t(0xE0 + i));
        EXPECT(db[0] == 0xE3);

        // (b) Cross-queue Signals calling direct.release_ready() while batches sit in direct.held_ must NOT split
        //     held-batch coalescing either.
        before_calls.store(0, std::memory_order_release);
        s_cb_submits.store(0, std::memory_order_release);
        std::memset(db, 0, kBufBytes);
        context.set_before_execute([&] { before_calls.fetch_add(1, std::memory_order_acq_rel); });

        VulkanFence gate_b(context, 0), done_b(context, 0);
        EXPECT(gate_b.semaphore() && done_b.semaphore());
        direct.wait(gate_b.semaphore(), 1);
        EXPECT(direct.execute({&steps[0], 1}) == ReplayResult::success);
        EXPECT(direct.idle());  // steps[0] is now in direct.held_
        EXPECT(compute.signal(cross_fence.semaphore(), 3));
        EXPECT(cross_fence.wait(3, 5'000'000'000ull));
        EXPECT(compute.idle());
        EXPECT(direct.execute({&steps[1], 1}) == ReplayResult::success);
        EXPECT(direct.signal(done_b.semaphore(), 1));
        EXPECT(direct.idle());  // steps[1] + signal are now in direct.held_ behind steps[0]
        EXPECT(compute.signal(cross_fence.semaphore(), 4));
        EXPECT(cross_fence.wait(4, 5'000'000'000ull));
        EXPECT(compute.idle());
        EXPECT(before_calls.load(std::memory_order_acquire) == 0);

        EXPECT(gate_b.signal(1));
        EXPECT(done_b.wait(1, 5'000'000'000ull));
        EXPECT(direct.idle());
        EXPECT(before_calls.load(std::memory_order_acquire) == 1);  // {steps[0], steps[1]} coalesced into 1 submit
        EXPECT(s_cb_submits.load(std::memory_order_acquire) == 1);
        EXPECT(db[16] == 0xE0 && db[32] == 0xE1 && db[0] == 0xE1);

        // (c) async_depth_=4 backpressure: while coalesced {B1, B2, B3} is running as 1 in-flight submission,
        //     enqueuing 3 more batches {B4, B5, B6} must NOT stall on room_cv_ and must coalesce into 1 submit.
        before_calls.store(0, std::memory_order_release);
        s_cb_submits.store(0, std::memory_order_release);
        std::memset(db, 0, kBufBytes);
        std::atomic<bool> p1_in{false}, p1_go{false}, p2_in{false}, p2_go{false};
        context.set_before_execute([&] {
            const unsigned n = before_calls.fetch_add(1, std::memory_order_acq_rel) + 1;
            if (n == 1) {
                p1_in.store(true, std::memory_order_release);
                while (!p1_go.load(std::memory_order_acquire)) std::this_thread::yield();
            } else if (n == 2) {
                p2_in.store(true, std::memory_order_release);
                while (!p2_go.load(std::memory_order_acquire)) std::this_thread::yield();
            }
        });

        VulkanFence done_c(context, 0);
        EXPECT(done_c.semaphore());
        EXPECT(direct.execute({&steps[0], 1}) == ReplayResult::success);
        while (!p1_in.load(std::memory_order_acquire)) std::this_thread::yield();
        EXPECT(direct.execute({&steps[1], 1}) == ReplayResult::success);
        EXPECT(direct.execute({&steps[2], 1}) == ReplayResult::success);
        EXPECT(direct.execute({&steps[3], 1}) == ReplayResult::success);

        // Release B0 -> worker pops coalesced {B1, B2, B3}, releases 2 batch slots immediately, and pauses at n == 2.
        p1_go.store(true, std::memory_order_release);
        while (!p2_in.load(std::memory_order_acquire)) std::this_thread::yield();

        // Enqueue 3 more batches on a separate thread while {B1, B2, B3} is paused: must finish without stalling!
        std::atomic<bool> queued_next_three{false};
        std::thread producer([&] {
            const bool ok4 = direct.execute({&steps[4], 1}) == ReplayResult::success;
            const bool ok5 = direct.execute({&steps[5], 1}) == ReplayResult::success;
            const bool ok6 = direct.execute({&steps[6], 1}) == ReplayResult::success;
            const bool ok_sig = direct.signal(done_c.semaphore(), 1);
            if (ok4 && ok5 && ok6 && ok_sig) queued_next_three.store(true, std::memory_order_release);
        });
        const auto wait_deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
        while (!queued_next_three.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < wait_deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        const bool non_stalling = queued_next_three.load(std::memory_order_acquire);
        p2_go.store(true, std::memory_order_release);
        producer.join();
        EXPECT(non_stalling);
        EXPECT(done_c.wait(1, 5'000'000'000ull));
        EXPECT(direct.idle());
        EXPECT(before_calls.load(std::memory_order_acquire) == 3);  // {B0}, {B1..B3}, {B4..B6}
        EXPECT(s_cb_submits.load(std::memory_order_acquire) == 3);
        for (int i = 0; i < 7; ++i) EXPECT(db[(i + 1) * 16] == std::uint8_t(0xE0 + i));
        EXPECT(db[0] == 0xE6);
        std::printf("queue cross-queue Signal & depth=4 coalescing: inbox/held merge, Present split & non-stalling room ok\n");
    }

    // 10. Present / after_held() barrier ordering across held batches (P0 regressions):
    //     (a) Race A: PresentX (TitleCall(part_present) + after_held()) when another queue short-circuits
    //         QueueHolds::run_barriers() so direct->oldest_held_seq() is NOT called before post-Present B2 & B3
    //         enter direct.held_. Immutable per-op present_epoch MUST split B1 from {B2, B3}, fire Present0
    //         strictly between B1 and {B2, B3}, and still coalesce {B2, B3} into 1 submit.
    //     (b) Race B: Generic after_held() (no present_calls bump) racing with B1 moving inbox_ -> running_seq_ ->
    //         held_ before post-barrier B2 & B3 enqueue. Monotonic barrier_seq_ + pause_release_for_barrier_ MUST
    //         split B1 from {B2, B3} and fire barrier strictly between B1 and {B2, B3} on every iteration.
    {
        static PFN_vkQueueSubmit s_real_submit = nullptr;
        static std::atomic<unsigned> s_cb_submits{0};
        struct HookSubmit {
            static VKAPI_ATTR VkResult VKAPI_CALL call(VkQueue q, std::uint32_t count, const VkSubmitInfo* infos,
                                                       VkFence fence) {
                for (std::uint32_t i = 0; i < count; ++i)
                    if (infos[i].commandBufferCount > 0) s_cb_submits.fetch_add(1, std::memory_order_relaxed);
                return s_real_submit(q, count, infos, fence);
            }
        };

        VulkanContext context(d.device_proc, d.device, true);
        EXPECT(context.loaded());
        s_real_submit = context.vk.vkQueueSubmit;
        s_cb_submits.store(0, std::memory_order_relaxed);
        context.vk.vkQueueSubmit = &HookSubmit::call;

        // `blocker` is attached BEFORE `direct` so QueueHolds::run_barriers() inspects `blocker` first and
        // short-circuits before calling `direct.oldest_held_seq()` while `blocker` holds an older seq.
        VulkanQueue blocker(context, d.queue, d.family, VulkanQueue::QueueKind::compute);
        VulkanQueue direct(context, d.queue, d.family, VulkanQueue::QueueKind::direct);
        EXPECT(blocker.ok() && direct.ok());
        blocker.set_async_depth(4);
        direct.set_async_depth(4);

        void* src_map = nullptr;
        void* dst_map = nullptr;
        VkBuffer src = d.buffer(kBufBytes, usage, &src_map);
        VkBuffer dst = d.buffer(kBufBytes, usage, &dst_map);
        EXPECT(src && dst);
        auto* sb = static_cast<std::uint8_t*>(src_map);
        auto* db = static_cast<std::uint8_t*>(dst_map);
        std::memset(db, 0, kBufBytes);
        sb[0] = 0xF1;
        sb[16] = 0xF2;
        sb[32] = 0xF3;

        ResourceBinding r_src{}, r_dst{};
        r_src.buffer = src;
        r_src.size = kBufBytes;
        r_dst.buffer = dst;
        r_dst.size = kBufBytes;
        r_dst.host_readable = true;
        context.add_resource(1, r_src);
        context.add_resource(2, r_dst);

        std::shared_ptr<const CommandList> steps[3];
        for (int i = 0; i < 3; ++i) {
            auto l = std::make_shared<CommandList>();
            EXPECT(l->copy_buffer({2, std::uint64_t((i + 1) * 16), 1, std::uint64_t(i * 16), 1}) &&
                   l->copy_buffer({2, 0, 1, std::uint64_t(i * 16), 1}) && l->close());
            steps[i] = std::move(l);
        }

        // (a) Race A: PresentX with short-circuited run_barriers() -> immutable present_epoch splits B1 from {B2, B3}.
        std::atomic<unsigned> before_calls{0};
        context.set_before_execute([&] { before_calls.fetch_add(1, std::memory_order_acq_rel); });

        VulkanFence blk_gate(context, 0), blk_done(context, 0), dir_gate(context, 0), dir_done(context, 0);
        EXPECT(blk_gate.semaphore() && blk_done.semaphore() && dir_gate.semaphore() && dir_done.semaphore());

        // Hold a Signal on `blocker` (holder 0, seq=1) so run_barriers() returns early without touching `direct`.
        blocker.wait(blk_gate.semaphore(), 1);
        EXPECT(blocker.signal(blk_done.semaphore(), 1));
        EXPECT(blocker.idle());

        // Frame 0: hold B1 on `direct` behind dir_gate (epoch = E0).
        direct.wait(dir_gate.semaphore(), 1);
        EXPECT(direct.execute({&steps[0], 1}) == ReplayResult::success);
        EXPECT(direct.idle());  // B1 is now in direct.held_ with epoch = E0

        // Frame 0 PresentX: bumps present_calls to E0 + 1 and registers after_held(Present0).
        // Because `blocker` holds seq=1, run_barriers() short-circuits and NEVER calls direct.oldest_held_seq()!
        std::atomic<bool> present_ran{false};
        std::atomic<unsigned> present_seen_before{0}, present_seen_submits{0};
        int present_owner = 0;
        {
            const sp::TitleCall present_call(sp::part_present);
            context.holds().after_held(&present_owner, [&] {
                present_seen_before.store(before_calls.load(std::memory_order_acquire), std::memory_order_release);
                present_seen_submits.store(s_cb_submits.load(std::memory_order_acquire), std::memory_order_release);
                present_ran.store(true, std::memory_order_release);
            });
        }

        // Frame 1: enqueue post-Present B2 & B3 onto `direct` (epoch = E0 + 1), both landing in direct.held_.
        EXPECT(direct.execute({&steps[1], 1}) == ReplayResult::success);
        EXPECT(direct.execute({&steps[2], 1}) == ReplayResult::success);
        EXPECT(direct.signal(dir_done.semaphore(), 1));
        EXPECT(direct.idle());
        EXPECT(!present_ran.load(std::memory_order_acquire));
        EXPECT(before_calls.load(std::memory_order_acquire) == 0);

        // Release `blocker` first, then `direct`: B1 must run alone, Present0 must fire at count==1, then {B2, B3}
        // must coalesce into submit #2!
        EXPECT(blk_gate.signal(1));
        EXPECT(blk_done.wait(1, 5'000'000'000ull));
        EXPECT(blocker.idle());
        EXPECT(!present_ran.load(std::memory_order_acquire));

        EXPECT(dir_gate.signal(1));
        EXPECT(dir_done.wait(1, 5'000'000'000ull));
        EXPECT(direct.idle());

        EXPECT(present_ran.load(std::memory_order_acquire));
        EXPECT(present_seen_before.load(std::memory_order_acquire) == 1);
        EXPECT(present_seen_submits.load(std::memory_order_acquire) == 1);
        EXPECT(before_calls.load(std::memory_order_acquire) == 2);  // {B1}, then coalesced {B2, B3}
        EXPECT(s_cb_submits.load(std::memory_order_acquire) == 2);
        EXPECT(db[16] == 0xF1 && db[32] == 0xF2 && db[48] == 0xF3 && db[0] == 0xF3);

        // (b) Race B: Generic after_held() (NO present_calls bump) racing with in-transit B1 (inbox_ ->
        //     running_seq_ -> held_) followed by B2 & B3, repeated 20 times to cover every transit state.
        for (int iter = 0; iter < 20; ++iter) {
            before_calls.store(0, std::memory_order_release);
            s_cb_submits.store(0, std::memory_order_release);
            std::memset(db, 0, kBufBytes);
            VulkanFence g(context, 0), done(context, 0);
            EXPECT(g.semaphore() && done.semaphore());

            direct.wait(g.semaphore(), 1);
            EXPECT(direct.execute({&steps[0], 1}) == ReplayResult::success);
            // Do NOT flush direct before after_held(): B1 may be in inbox_, in-transit (running_seq_), or in held_.
            std::atomic<bool> b_ran{false};
            std::atomic<unsigned> b_seen_before{0}, b_seen_submits{0};
            int b_owner = 0;
            context.holds().after_held(&b_owner, [&] {
                b_seen_before.store(before_calls.load(std::memory_order_acquire), std::memory_order_release);
                b_seen_submits.store(s_cb_submits.load(std::memory_order_acquire), std::memory_order_release);
                b_ran.store(true, std::memory_order_release);
            });
            EXPECT(direct.execute({&steps[1], 1}) == ReplayResult::success);
            EXPECT(direct.execute({&steps[2], 1}) == ReplayResult::success);
            EXPECT(direct.signal(done.semaphore(), 1));
            EXPECT(direct.idle());
            EXPECT(!b_ran.load(std::memory_order_acquire));

            EXPECT(g.signal(1));
            EXPECT(done.wait(1, 5'000'000'000ull));
            EXPECT(direct.idle());

            EXPECT(b_ran.load(std::memory_order_acquire));
            EXPECT(b_seen_before.load(std::memory_order_acquire) == 1);
            EXPECT(b_seen_submits.load(std::memory_order_acquire) == 1);
            EXPECT(before_calls.load(std::memory_order_acquire) == 2);  // {B1}, then coalesced {B2, B3}
            EXPECT(s_cb_submits.load(std::memory_order_acquire) == 2);
            EXPECT(db[16] == 0xF1 && db[32] == 0xF2 && db[48] == 0xF3 && db[0] == 0xF3);
        }
        std::printf("queue held Present/after_held barrier ordering: present_epoch & in-transit barrier_seq_ ok\n");
    }

    // 10. note_mirror_bind() returns the latest submitted retire ticket under submit_mutex_, and every subsequent
    //     command-buffer submission on VulkanQueue waits on the newly published mirror bind timeline semaphore value.
    {
        VulkanContext context(d.device_proc, d.device, true);
        EXPECT(context.loaded());
        VulkanQueue queue(context, d.queue, d.family, VulkanQueue::QueueKind::direct);
        EXPECT(queue.ok());

        auto empty = std::make_shared<CommandList>();
        EXPECT(empty->close());
        std::shared_ptr<const CommandList> snap = empty;

        EXPECT(queue.execute({&snap, 1}) == ReplayResult::success);
        EXPECT(queue.idle());
        const std::uint64_t t_before = context.submitted_retire_ticket();
        EXPECT(t_before >= 1);

        VulkanFence mirror_tl(context, 0), done(context, 0);
        EXPECT(mirror_tl.semaphore() != VK_NULL_HANDLE && done.semaphore() != VK_NULL_HANDLE);
        const std::uint64_t bind_ticket = context.note_mirror_bind(mirror_tl.semaphore(), 1);
        EXPECT(bind_ticket == t_before);
        {
            std::lock_guard lock(context.submit_mutex());
            VkSemaphore seen_sem = VK_NULL_HANDLE;
            std::uint64_t seen_val = 0;
            context.mirror_bind_locked(seen_sem, seen_val);
            EXPECT(seen_sem == mirror_tl.semaphore() && seen_val == 1);
        }

        // Next submission MUST wait on (mirror_tl, 1) before completing on the GPU.
        EXPECT(queue.execute({&snap, 1}) == ReplayResult::success);
        EXPECT(queue.signal(done.semaphore(), 1));
        EXPECT(!done.wait(1, 20'000'000ull));  // 20 ms: still blocked waiting for mirror_tl == 1
        EXPECT(mirror_tl.signal(1));
        EXPECT(done.wait(1, 5'000'000'000ull));
        EXPECT(queue.idle());
        EXPECT(context.submitted_retire_ticket() > bind_ticket);
        std::printf("note_mirror_bind ticket & submit wait ordering: ok\n");
    }

    // 11. Round 5 same-queue execute+signal coalescing, depth=4 frame-burst room gating, cross-queue Wait split,
    //     Present barrier split, and coalesced error fallback isolation (LEONIDA_VK_VALIDATION=1 clean).
    {
        static PFN_vkQueueSubmit s_real_submit = nullptr;
        static std::atomic<unsigned> s_cb_submits{0};
        static std::atomic<unsigned> s_total_submits{0};
        struct HookSubmit {
            static VKAPI_ATTR VkResult VKAPI_CALL call(VkQueue q, std::uint32_t count, const VkSubmitInfo* infos,
                                                       VkFence fence) {
                for (std::uint32_t i = 0; i < count; ++i) {
                    s_total_submits.fetch_add(1, std::memory_order_relaxed);
                    if (infos[i].commandBufferCount > 0) s_cb_submits.fetch_add(1, std::memory_order_relaxed);
                }
                return s_real_submit(q, count, infos, fence);
            }
        };

        VulkanContext context(d.device_proc, d.device, true);
        EXPECT(context.loaded());
        s_real_submit = context.vk.vkQueueSubmit;
        s_cb_submits.store(0, std::memory_order_relaxed);
        s_total_submits.store(0, std::memory_order_relaxed);
        context.vk.vkQueueSubmit = &HookSubmit::call;

        VulkanQueue direct(context, d.queue, d.family, VulkanQueue::QueueKind::direct);
        VulkanQueue compute(context, d.queue, d.family, VulkanQueue::QueueKind::compute);
        VulkanQueue copy_q(context, d.queue, d.family, VulkanQueue::QueueKind::copy);
        EXPECT(direct.ok() && compute.ok() && copy_q.ok());
        direct.set_async_depth(4);
        compute.set_async_depth(4);
        copy_q.set_async_depth(4);

        constexpr VkDeviceSize kBurstBufBytes = 512;
        void* src_map = nullptr;
        void* dst_map = nullptr;
        VkBuffer src = d.buffer(kBurstBufBytes, usage, &src_map);
        VkBuffer dst = d.buffer(kBurstBufBytes, usage, &dst_map);
        EXPECT(src && dst);
        auto* sb = static_cast<std::uint8_t*>(src_map);
        auto* db = static_cast<std::uint8_t*>(dst_map);
        std::memset(sb, 0, kBurstBufBytes);
        std::memset(db, 0, kBurstBufBytes);
        constexpr int kSteps = 18;
        for (int i = 0; i < kSteps; ++i) sb[i * 16] = std::uint8_t(0x40 + i);

        ResourceBinding r_src{}, r_dst{};
        r_src.buffer = src;
        r_src.size = kBurstBufBytes;
        r_dst.buffer = dst;
        r_dst.size = kBurstBufBytes;
        r_dst.host_readable = true;
        context.add_resource(1, r_src);
        context.add_resource(2, r_dst);

        std::shared_ptr<const CommandList> steps[kSteps];
        for (int i = 0; i < kSteps; ++i) {
            auto l = std::make_shared<CommandList>();
            EXPECT(l->copy_buffer({2, std::uint64_t((i + 1) * 16), 1, std::uint64_t(i * 16), 1}) &&
                   l->copy_buffer({2, 0, 1, std::uint64_t(i * 16), 1}) && l->close());
            steps[i] = std::move(l);
        }

        // (a) Unwaited [execute, signal(F, 1), execute, signal(F, 2), execute, signal(F, 3), execute, signal(F, 4)]
        //     plus trailing signal(F2, 7) coalesces into ONE before_execute_() + ONE vkQueueSubmit (deduping F -> 4).
        std::atomic<unsigned> before_calls{0};
        std::atomic<bool> in_first{false}, go_first{false};
        context.set_before_execute([&] {
            const unsigned n = before_calls.fetch_add(1, std::memory_order_acq_rel) + 1;
            if (n == 1) {
                in_first.store(true, std::memory_order_release);
                while (!go_first.load(std::memory_order_acquire)) std::this_thread::yield();
            }
        });

        VulkanFence f_a(context, 0), f_a2(context, 0);
        EXPECT(f_a.semaphore() && f_a2.semaphore());

        EXPECT(direct.execute({&steps[0], 1}) == ReplayResult::success);
        while (!in_first.load(std::memory_order_acquire)) std::this_thread::yield();

        for (int i = 1; i <= 4; ++i) {
            EXPECT(direct.execute({&steps[i], 1}) == ReplayResult::success);
            EXPECT(direct.signal(f_a.semaphore(), std::uint64_t(i)));
        }
        EXPECT(direct.signal(f_a2.semaphore(), 7));

        go_first.store(true, std::memory_order_release);
        EXPECT(f_a.wait(4, 5'000'000'000ull));
        EXPECT(f_a2.wait(7, 5'000'000'000ull));
        EXPECT(direct.idle());

        EXPECT(before_calls.load(std::memory_order_acquire) == 2);
        EXPECT(s_cb_submits.load(std::memory_order_acquire) == 2);
        EXPECT(s_total_submits.load(std::memory_order_acquire) == 2);  // zero separate signal-only vkQueueSubmits!
        for (int i = 0; i < 5; ++i) EXPECT(db[(i + 1) * 16] == std::uint8_t(0x40 + i));
        EXPECT(db[0] == std::uint8_t(0x44));

        // (b) 17-batch + 17-signal single-frame burst on depth=4 completes without stalling on room_cv_ and merges
        //     all 17 batches + 17 signals into ONE vkQueueSubmit.
        before_calls.store(0, std::memory_order_release);
        s_cb_submits.store(0, std::memory_order_release);
        s_total_submits.store(0, std::memory_order_release);
        in_first.store(false, std::memory_order_release);
        go_first.store(false, std::memory_order_release);
        std::memset(db, 0, kBurstBufBytes);

        VulkanFence f_burst(context, 0);
        EXPECT(f_burst.semaphore());
        EXPECT(direct.execute({&steps[0], 1}) == ReplayResult::success);
        while (!in_first.load(std::memory_order_acquire)) std::this_thread::yield();

        std::atomic<bool> burst_queued{false};
        std::thread burst_producer([&] {
            bool ok = true;
            for (int i = 1; i <= 17 && ok; ++i) {
                ok = (direct.execute({&steps[i], 1}) == ReplayResult::success) &&
                     direct.signal(f_burst.semaphore(), std::uint64_t(i));
            }
            if (ok) burst_queued.store(true, std::memory_order_release);
        });
        const auto burst_deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
        while (!burst_queued.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < burst_deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        const bool burst_no_stall = burst_queued.load(std::memory_order_acquire);
        go_first.store(true, std::memory_order_release);
        burst_producer.join();
        EXPECT(burst_no_stall);
        EXPECT(f_burst.wait(17, 5'000'000'000ull));
        EXPECT(direct.idle());

        EXPECT(before_calls.load(std::memory_order_acquire) == 2);
        EXPECT(s_cb_submits.load(std::memory_order_acquire) == 2);
        EXPECT(s_total_submits.load(std::memory_order_acquire) == 2);
        for (int i = 0; i < 18; ++i) EXPECT(db[(i + 1) * 16] == std::uint8_t(0x40 + i));
        EXPECT(db[0] == std::uint8_t(0x40 + 17));

        // (c) Cross-queue Wait(cross_f, 1) on `compute` splits `direct` right after Signal(cross_f, 1) (attaching
        //     Signal(cross_f, 1) to B1's submit rather than merging B2 ahead of it) and unblocks `compute` cleanly.
        before_calls.store(0, std::memory_order_release);
        s_cb_submits.store(0, std::memory_order_release);
        s_total_submits.store(0, std::memory_order_release);
        in_first.store(false, std::memory_order_release);
        go_first.store(false, std::memory_order_release);
        std::memset(db, 0, kBurstBufBytes);

        VulkanFence cross_f(context, 0), compute_done(context, 0), direct_done(context, 0);
        EXPECT(cross_f.semaphore() && compute_done.semaphore() && direct_done.semaphore());

        auto compute_step = make_copy_list(2, 320, 1, 16, 1);  // copies 0x41 -> db[320]
        EXPECT(compute_step != nullptr);
        std::shared_ptr<const CommandList> s_comp = compute_step;

        EXPECT(direct.execute({&steps[0], 1}) == ReplayResult::success);
        while (!in_first.load(std::memory_order_acquire)) std::this_thread::yield();

        // Register cross-queue Wait(cross_f, 1) on compute queue and flush into held_.
        compute.wait(cross_f.semaphore(), 1);
        EXPECT(compute.execute({&s_comp, 1}) == ReplayResult::success);
        EXPECT(compute.signal(compute_done.semaphore(), 1));
        EXPECT(compute.idle());

        // Queue B1 + Signal(cross_f, 1) + B2 + Signal(direct_done, 1) on direct.
        EXPECT(direct.execute({&steps[1], 1}) == ReplayResult::success);
        EXPECT(direct.signal(cross_f.semaphore(), 1));
        EXPECT(direct.execute({&steps[2], 1}) == ReplayResult::success);
        EXPECT(direct.signal(direct_done.semaphore(), 1));

        go_first.store(true, std::memory_order_release);
        EXPECT(compute_done.wait(1, 5'000'000'000ull));
        EXPECT(direct_done.wait(1, 5'000'000'000ull));
        EXPECT(direct.idle() && compute.idle());

        // Direct split into {B0}, {B1 + sig(cross_f)}, {B2 + sig(direct_done)} (3 submits) + compute {s_comp + sig} (1) = 4.
        EXPECT(before_calls.load(std::memory_order_acquire) == 4);
        EXPECT(s_cb_submits.load(std::memory_order_acquire) == 4);
        EXPECT(s_total_submits.load(std::memory_order_acquire) == 4);
        EXPECT(db[16] == 0x40 && db[32] == 0x41 && db[48] == 0x42 && db[0] == 0x42 && db[320] == 0x41);

        // (d) Present (after_held) barrier strictly splits [B1, signal(F, 1)] from [B2, signal(F, 2)].
        before_calls.store(0, std::memory_order_release);
        s_cb_submits.store(0, std::memory_order_release);
        s_total_submits.store(0, std::memory_order_release);
        in_first.store(false, std::memory_order_release);
        go_first.store(false, std::memory_order_release);
        std::memset(db, 0, kBurstBufBytes);

        VulkanFence f_pres(context, 0);
        EXPECT(f_pres.semaphore());
        EXPECT(direct.execute({&steps[0], 1}) == ReplayResult::success);
        while (!in_first.load(std::memory_order_acquire)) std::this_thread::yield();

        EXPECT(direct.execute({&steps[1], 1}) == ReplayResult::success);
        EXPECT(direct.signal(f_pres.semaphore(), 1));

        std::atomic<bool> pres_ran{false};
        std::atomic<unsigned> pres_seen_before{0}, pres_seen_total{0};
        int pres_owner = 0;
        context.holds().after_held(&pres_owner, [&] {
            pres_seen_before.store(before_calls.load(std::memory_order_acquire), std::memory_order_release);
            pres_seen_total.store(s_total_submits.load(std::memory_order_acquire), std::memory_order_release);
            pres_ran.store(true, std::memory_order_release);
        });

        EXPECT(direct.execute({&steps[2], 1}) == ReplayResult::success);
        EXPECT(direct.signal(f_pres.semaphore(), 2));

        go_first.store(true, std::memory_order_release);
        EXPECT(f_pres.wait(2, 5'000'000'000ull));
        EXPECT(direct.idle());

        EXPECT(pres_ran.load(std::memory_order_acquire));
        EXPECT(pres_seen_before.load(std::memory_order_acquire) == 2);
        EXPECT(pres_seen_total.load(std::memory_order_acquire) == 2);
        EXPECT(before_calls.load(std::memory_order_acquire) == 3);
        EXPECT(s_cb_submits.load(std::memory_order_acquire) == 3);
        EXPECT(s_total_submits.load(std::memory_order_acquire) == 3);
        EXPECT(db[16] == 0x40 && db[32] == 0x41 && db[48] == 0x42 && db[0] == 0x42);

        // (e) Coalesced [valid1, signal(s1, 1), bad_clear, valid2, signal(s2, 1)] on async copy queue falls back
        //     per-item in exact FIFO order without dropping valid peers or signals.
        before_calls.store(0, std::memory_order_release);
        in_first.store(false, std::memory_order_release);
        go_first.store(false, std::memory_order_release);
        std::memset(db, 0, kBurstBufBytes);

        auto bad_clear = std::make_shared<CommandList>();
        TargetView bad_tv{};
        bad_tv.resource = 2;
        bad_tv.layer_count = 1;
        EXPECT(bad_clear->clear_render_target(bad_tv, ClearColor{}) && bad_clear->close());
        std::shared_ptr<const CommandList> bad_snap = bad_clear;

        VulkanFence err_s1(context, 0), err_s2(context, 0);
        EXPECT(err_s1.semaphore() && err_s2.semaphore());

        EXPECT(copy_q.execute({&steps[0], 1}) == ReplayResult::success);
        while (!in_first.load(std::memory_order_acquire)) std::this_thread::yield();

        EXPECT(copy_q.execute({&steps[1], 1}) == ReplayResult::success);
        EXPECT(copy_q.signal(err_s1.semaphore(), 1));
        EXPECT(copy_q.execute({&bad_snap, 1}) == ReplayResult::success);
        EXPECT(copy_q.execute({&steps[2], 1}) == ReplayResult::success);
        EXPECT(copy_q.signal(err_s2.semaphore(), 1));

        go_first.store(true, std::memory_order_release);
        EXPECT(err_s1.wait(1, 5'000'000'000ull));
        EXPECT(err_s2.wait(1, 5'000'000'000ull));
        EXPECT(copy_q.idle());
        EXPECT(copy_q.take_held_failure() == ReplayResult::unsupported);
        EXPECT(db[16] == 0x40 && db[32] == 0x41 && db[48] == 0x42 && db[0] == 0x42);
        std::printf("queue execute+signal coalescing: unwaited merge, 17-batch burst, cross-queue Wait split, Present split & error fallback ok\n");
    }

    // 12. Round 6 queue contention fixes:
    //     (a) 24 Wait-separated batches within one frame on depth=4 enqueue with zero room_cv_ stalls;
    //     (b) Distinct Present-separated frames still throttle cleanly at depth=4 (bounded backpressure);
    //     (c) Held-only Compute queue executions via release_held() are counted in async_run_ns (worker busy %);
    //     (d) Visibility epochs (current_before_execute_seq / signaled_visibility_seq) allow safe single-flight
    //         before_execute_() coalescing across concurrent/back-to-back queues while NEVER skipping a refresh
    //         when a Wait-before-Signal CPU Fence::Signal lands after the previous refresh started.
    {
        VulkanContext context(d.device_proc, d.device, true);
        EXPECT(context.loaded());
        VulkanQueue direct(context, d.queue, d.family, VulkanQueue::QueueKind::direct);
        VulkanQueue compute(context, d.queue, d.family, VulkanQueue::QueueKind::compute);
        EXPECT(direct.ok() && compute.ok());
        direct.set_async_depth(4);
        compute.set_async_depth(4);

        auto empty = std::make_shared<CommandList>();
        EXPECT(empty->uav_barrier({}) && empty->close());
        std::shared_ptr<const CommandList> snap = empty;

        // (a) 24 Wait-separated batches within a single present_epoch on depth=4: zero room_cv_ stalls!
        std::atomic<bool> block_first{true}, in_first{false};
        context.set_before_execute([&] {
            if (block_first.exchange(false, std::memory_order_acq_rel)) {
                in_first.store(true, std::memory_order_release);
                while (in_first.load(std::memory_order_acquire)) std::this_thread::yield();
            }
        });
        VulkanFence intra_f(context, 0), done_a(context, 0);
        EXPECT(intra_f.semaphore() && done_a.semaphore());
        EXPECT(direct.execute({&snap, 1}) == ReplayResult::success);
        while (!in_first.load(std::memory_order_acquire)) std::this_thread::yield();

        std::atomic<bool> burst24_done{false};
        std::thread producer24([&] {
            bool ok = true;
            for (int i = 1; i <= 24 && ok; ++i) {
                direct.wait(intra_f.semaphore(), std::uint64_t(i - 1));
                ok = (direct.execute({&snap, 1}) == ReplayResult::success) &&
                     direct.signal(intra_f.semaphore(), std::uint64_t(i));
            }
            ok = ok && direct.signal(done_a.semaphore(), 1);
            if (ok) burst24_done.store(true, std::memory_order_release);
        });
        const auto d24 = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
        while (!burst24_done.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < d24)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        const bool non_stalling_24 = burst24_done.load(std::memory_order_acquire);
        in_first.store(false, std::memory_order_release);
        producer24.join();
        EXPECT(non_stalling_24);
        EXPECT(done_a.wait(1, 5'000'000'000ull));
        EXPECT(direct.idle());
        EXPECT(direct.async_room_waits_for_test() == 0);

        // (b) Distinct Present frames on depth=4 still throttle when 4 frames are already queued/running.
        block_first.store(true, std::memory_order_release);
        in_first.store(false, std::memory_order_release);
        VulkanFence done_b(context, 0);
        EXPECT(done_b.semaphore());
        EXPECT(direct.execute({&snap, 1}) == ReplayResult::success);
        while (!in_first.load(std::memory_order_acquire)) std::this_thread::yield();

        std::atomic<unsigned> frames_queued{0};
        std::thread frame_producer([&] {
            for (unsigned f = 1; f <= 5; ++f) {
                {
                    const sp::TitleCall pres(sp::part_present);
                }
                (void)direct.execute({&snap, 1});
                frames_queued.store(f, std::memory_order_release);
            }
            (void)direct.signal(done_b.semaphore(), 1);
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        // Worker busy (1) + 3 distinct present_epoch frames in inbox_ = 4 == depth: 4th queued frame must wait!
        EXPECT(frames_queued.load(std::memory_order_acquire) == 3);
        in_first.store(false, std::memory_order_release);
        frame_producer.join();
        EXPECT(done_b.wait(1, 5'000'000'000ull));
        EXPECT(direct.idle());
        EXPECT(direct.async_room_waits_for_test() >= 1);

        // (c) Held-only Compute queue execution via release_held() is measured in async_run_ns (worker busy %).
        context.set_before_execute([] { std::this_thread::sleep_for(std::chrono::milliseconds(10)); });
        VulkanFence gate_c(context, 0), done_c(context, 0);
        EXPECT(gate_c.semaphore() && done_c.semaphore());
        compute.wait(gate_c.semaphore(), 1);
        EXPECT(compute.execute({&snap, 1}) == ReplayResult::success);
        EXPECT(compute.signal(done_c.semaphore(), 1));
        EXPECT(compute.idle());  // flushed into held_ while gate_c is unmet
        const std::uint64_t run_before = compute.async_run_ns_for_test();
        EXPECT(gate_c.signal(1));
        EXPECT(done_c.wait(1, 5'000'000'000ull));
        EXPECT(compute.idle());
        const std::uint64_t run_after = compute.async_run_ns_for_test();
        EXPECT(run_after - run_before >= 5 * ms);

        // (d) Visibility epoch contract for single-flight before_execute_() coalescing + Wait-before-Signal safety.
        std::mutex coal_m;
        std::uint64_t coal_completed = 0;
        std::atomic<unsigned> coal_ran{0}, coal_skipped{0};
        std::atomic<bool> hold_lead{false}, in_lead{false}, coal_bad_req{false};
        std::atomic<int> cpu_data{0}, observed_data{0};
        context.set_before_execute([&] {
            const std::uint64_t req = VulkanContext::current_before_execute_seq();
            if (req == 0) {
                coal_bad_req.store(true, std::memory_order_relaxed);
                return;
            }
            std::unique_lock lock(coal_m);
            if (coal_completed >= req) {
                coal_skipped.fetch_add(1, std::memory_order_relaxed);
                return;
            }
            const std::uint64_t pass = (std::max)(req, context.current_visibility_seq());
            lock.unlock();
            if (hold_lead.exchange(false, std::memory_order_acq_rel)) {
                in_lead.store(true, std::memory_order_release);
                while (in_lead.load(std::memory_order_acquire)) std::this_thread::yield();
            }
            observed_data.store(cpu_data.load(std::memory_order_acquire), std::memory_order_release);
            coal_ran.fetch_add(1, std::memory_order_relaxed);
            lock.lock();
            if (pass > coal_completed) coal_completed = pass;
        });

        // Case d1: Back-to-back direct + compute enqueued before direct's pass starts -> compute coalesces cleanly!
        VulkanFence cross_d(context, 0), comp_done_d(context, 0);
        EXPECT(cross_d.semaphore() && comp_done_d.semaphore());
        hold_lead.store(true, std::memory_order_release);
        EXPECT(direct.execute({&snap, 1}) == ReplayResult::success);  // batch 0 holds lead
        while (!in_lead.load(std::memory_order_acquire)) std::this_thread::yield();

        cpu_data.store(42, std::memory_order_release);
        EXPECT(direct.execute({&snap, 1}) == ReplayResult::success);
        EXPECT(direct.signal(cross_d.semaphore(), 1));
        compute.wait(cross_d.semaphore(), 1);
        EXPECT(compute.execute({&snap, 1}) == ReplayResult::success);
        EXPECT(compute.signal(comp_done_d.semaphore(), 1));
        EXPECT(compute.idle());  // compute batch now held waiting for cross_d

        in_lead.store(false, std::memory_order_release);
        EXPECT(comp_done_d.wait(1, 5'000'000'000ull));
        EXPECT(direct.idle() && compute.idle());
        EXPECT(observed_data.load(std::memory_order_acquire) == 42);
        EXPECT(coal_skipped.load(std::memory_order_acquire) >= 1);

        // Case d2: Wait-before-Signal with CPU write + CPU Fence::Signal AFTER prior pass completed MUST NOT skip!
        VulkanFence cpu_gate(context, 0), comp_done_d2(context, 0);
        EXPECT(cpu_gate.semaphore() && comp_done_d2.semaphore());
        compute.wait(cpu_gate.semaphore(), 1);
        EXPECT(compute.execute({&snap, 1}) == ReplayResult::success);
        EXPECT(compute.signal(comp_done_d2.semaphore(), 1));
        EXPECT(compute.idle());  // held in compute.held_

        // Run a direct pass now: coal_completed advances past compute's enqueue visibility_seq!
        EXPECT(direct.execute({&snap, 1}) == ReplayResult::success);
        EXPECT(direct.idle());
        const unsigned ran_before_cpu_sig = coal_ran.load(std::memory_order_acquire);

        // Title writes new CPU data and signals cpu_gate from CPU: compute MUST run a fresh refresh and see 99!
        cpu_data.store(99, std::memory_order_release);
        EXPECT(cpu_gate.signal(1));
        EXPECT(comp_done_d2.wait(1, 5'000'000'000ull));
        EXPECT(compute.idle());
        EXPECT(!coal_bad_req.load(std::memory_order_acquire));
        EXPECT(coal_ran.load(std::memory_order_acquire) == ran_before_cpu_sig + 1);
        EXPECT(observed_data.load(std::memory_order_acquire) == 99);
        std::printf("queue r6 contention: 24-wait frame 0 room stalls, frame-group bound, held run_ns & visibility coalescing ok\n");
    }

    // 13. Round 7 sync & texture upload optimizations:
    //     (a) Per-fence vis_seq in QueueHolds + GPU-only VulkanQueue::signal() does NOT bump visibility_seq_
    //         (stopping cross-fence & GPU-signal false invalidations while preserving CPU Fence::Signal visibility,
    //         including across Wait -> Signal -> Execute);
    //     (b) Unlocked QueueHolds::run_barriers() under barrier_mutex_: strict FIFO Present ordering while
    //         concurrent cross-queue pump_ready() never stalls behind a blocked barrier, plus cancel_after() wait;
    //     (c) Lazy title-upload semaphore in VulkanQueue::submit_multi() + submit_mutex_ unlocked during
    //         write_staged_title_images() recording.
    {
        // (a) Per-fence vis_seq & GPU-only signal visibility isolation.
        VulkanContext context(d.device_proc, d.device, true);
        EXPECT(context.loaded());
        VulkanQueue direct(context, d.queue, d.family, VulkanQueue::QueueKind::direct);
        VulkanQueue compute(context, d.queue, d.family, VulkanQueue::QueueKind::compute);
        EXPECT(direct.ok() && compute.ok());
        direct.set_async_depth(4);
        compute.set_async_depth(4);

        auto empty = std::make_shared<CommandList>();
        EXPECT(empty->uav_barrier({}) && empty->close());
        std::shared_ptr<const CommandList> snap = empty;

        // GPU-only queue Signal must NOT bump visibility_seq_.
        VulkanFence gpu_f(context, 0);
        EXPECT(gpu_f.semaphore());
        const std::uint64_t vis_before_gpu_sig = context.current_visibility_seq();
        EXPECT(direct.signal(gpu_f.semaphore(), 1));
        EXPECT(gpu_f.wait(1, 5'000'000'000ull));
        EXPECT(direct.idle());
        EXPECT(context.current_visibility_seq() == vis_before_gpu_sig);
        EXPECT(context.holds().signaled_vis_seq(gpu_f.semaphore(), 1) == 0);

        // Unrelated CPU Fence::Signal on fence_a MUST NOT invalidate a held batch waiting on GPU-signaled fence_b.
        std::mutex coal_m;
        std::uint64_t coal_completed = 0;
        std::atomic<unsigned> coal_ran{0}, coal_skipped{0};
        std::atomic<int> cpu_val{0}, seen_val{0};
        context.set_before_execute([&] {
            const std::uint64_t req = VulkanContext::current_before_execute_seq();
            std::unique_lock lock(coal_m);
            if (req && coal_completed >= req) {
                coal_skipped.fetch_add(1, std::memory_order_relaxed);
                return;
            }
            const std::uint64_t pass = (std::max)(req, context.current_visibility_seq());
            lock.unlock();
            seen_val.store(cpu_val.load(std::memory_order_acquire), std::memory_order_release);
            coal_ran.fetch_add(1, std::memory_order_relaxed);
            lock.lock();
            if (pass > coal_completed) coal_completed = pass;
        });

        VulkanFence fence_a(context, 0), fence_b(context, 0), comp_done(context, 0);
        EXPECT(fence_a.semaphore() && fence_b.semaphore() && comp_done.semaphore());

        // Hold compute batch on GPU fence_b.
        compute.wait(fence_b.semaphore(), 1);
        EXPECT(compute.execute({&snap, 1}) == ReplayResult::success);
        EXPECT(compute.signal(comp_done.semaphore(), 1));
        EXPECT(compute.idle());

        // Direct runs a pass advancing coal_completed past compute's enqueue visibility_seq.
        EXPECT(direct.execute({&snap, 1}) == ReplayResult::success);
        EXPECT(direct.idle());
        const unsigned ran_before = coal_ran.load(std::memory_order_acquire);
        const unsigned skipped_before = coal_skipped.load(std::memory_order_acquire);

        // Unrelated CPU signal on fence_a bumps visibility_seq_ and fence_a's per-fence vis_seq ONLY.
        EXPECT(fence_a.signal(1));
        EXPECT(context.holds().signaled_vis_seq(fence_a.semaphore(), 1) > 0);
        EXPECT(context.holds().signaled_vis_seq(fence_b.semaphore(), 1) == 0);

        // GPU-only signal on fence_b releases compute: must COALESCE (not re-run) because fence_b has vis_seq == 0!
        EXPECT(direct.signal(fence_b.semaphore(), 1));
        EXPECT(comp_done.wait(1, 5'000'000'000ull));
        EXPECT(direct.idle() && compute.idle());
        EXPECT(coal_ran.load(std::memory_order_acquire) == ran_before);
        EXPECT(coal_skipped.load(std::memory_order_acquire) == skipped_before + 1);

        // Wait(cpu_f, 1) -> Signal(mid, 1) -> Execute(B2): CPU Fence::Signal on cpu_f MUST still lift B2's req_seq!
        VulkanFence cpu_f(context, 0), mid_f(context, 0), done_f(context, 0);
        EXPECT(cpu_f.semaphore() && mid_f.semaphore() && done_f.semaphore());
        compute.wait(cpu_f.semaphore(), 1);
        EXPECT(compute.signal(mid_f.semaphore(), 1));
        EXPECT(compute.execute({&snap, 1}) == ReplayResult::success);
        EXPECT(compute.signal(done_f.semaphore(), 1));
        EXPECT(compute.idle());

        EXPECT(direct.execute({&snap, 1}) == ReplayResult::success);
        EXPECT(direct.idle());
        const unsigned ran_before_cpu = coal_ran.load(std::memory_order_acquire);
        cpu_val.store(777, std::memory_order_release);
        EXPECT(cpu_f.signal(1));
        EXPECT(done_f.wait(1, 5'000'000'000ull));
        EXPECT(compute.idle());
        EXPECT(coal_ran.load(std::memory_order_acquire) == ran_before_cpu + 1);
        EXPECT(seen_val.load(std::memory_order_acquire) == 777);

        // (b) Unlocked QueueHolds::run_barriers(): strict FIFO order + concurrent pump_ready() unblocked + cancel_after.
        context.set_before_execute({});
        auto empty_13b = std::make_shared<CommandList>();
        EXPECT(empty_13b->close());
        std::shared_ptr<const CommandList> snap_13b = empty_13b;
        VulkanFence hold_b1(context, 0), hold_comp(context, 0), comp_b_done(context, 0);
        EXPECT(hold_b1.semaphore() && hold_comp.semaphore() && comp_b_done.semaphore());

        direct.wait(hold_b1.semaphore(), 1);
        EXPECT(direct.execute({&snap_13b, 1}) == ReplayResult::success);
        EXPECT(direct.idle());

        std::mutex fifo_m;
        std::vector<int> fifo_order;
        std::atomic<bool> in_b1{false}, release_b1{false}, b3_ran{false};
        int owner_fifo = 0, owner_cancel = 0;
        context.holds().after_held(&owner_fifo, [&] {
            in_b1.store(true, std::memory_order_release);
            while (!release_b1.load(std::memory_order_acquire)) std::this_thread::yield();
            std::lock_guard lk(fifo_m);
            fifo_order.push_back(1);
        });
        context.holds().after_held(&owner_fifo, [&] {
            std::lock_guard lk(fifo_m);
            fifo_order.push_back(2);
        });
        context.holds().after_held(&owner_cancel, [&] { b3_ran.store(true, std::memory_order_release); });

        // Hold an independent compute batch behind hold_comp.
        compute.wait(hold_comp.semaphore(), 1);
        EXPECT(compute.execute({&snap_13b, 1}) == ReplayResult::success);
        EXPECT(compute.signal(comp_b_done.semaphore(), 1));
        EXPECT(compute.idle());

        // Release direct -> starts barrier 1 (blocks on release_b1).
        std::thread sig_b1([&] { (void)hold_b1.signal(1); });
        while (!in_b1.load(std::memory_order_acquire)) std::this_thread::yield();

        // While barrier 1 is blocked inside barrier_mutex_, pump_mutex_ MUST be free so hold_comp.signal(1)
        // releases compute and completes comp_b_done immediately without stalling!
        EXPECT(hold_comp.signal(1));
        EXPECT(comp_b_done.wait(1, 2'000'000'000ull));
        EXPECT(compute.idle());

        // Cancel owner_cancel while barrier 1 is running: b3 must be dropped and never run!
        std::atomic<bool> cancel_done{false};
        std::thread canceller([&] {
            context.holds().cancel_after(&owner_cancel);
            cancel_done.store(true, std::memory_order_release);
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        // cancel_after waits for barrier_mutex_ (running barrier) before returning!
        EXPECT(!cancel_done.load(std::memory_order_acquire));

        release_b1.store(true, std::memory_order_release);
        sig_b1.join();
        canceller.join();
        context.holds().wait_barriers_drained(UINT64_MAX);
        EXPECT(direct.idle());
        EXPECT(cancel_done.load(std::memory_order_acquire));
        EXPECT(!b3_ran.load(std::memory_order_acquire));
        EXPECT((fifo_order == std::vector<int>{1, 2}));

        // (c) Lazy title-upload semaphore in submit_multi() + submit_mutex_ unlocked during write_staged_title_images()
        //     recording.
        static PFN_vkQueueSubmit s_r7_real_submit = nullptr;
        static VkSemaphore s_r7_watch_sem = VK_NULL_HANDLE;
        static std::atomic<unsigned> s_r7_wait_hits{0};
        static std::atomic<std::uint64_t> s_r7_last_wait_val{0};
        struct R7HookSubmit {
            static VKAPI_ATTR VkResult VKAPI_CALL call(VkQueue q, std::uint32_t count, const VkSubmitInfo* infos,
                                                       VkFence fence) {
                for (std::uint32_t i = 0; i < count; ++i) {
                    const auto* tl = static_cast<const VkTimelineSemaphoreSubmitInfo*>(infos[i].pNext);
                    for (std::uint32_t w = 0; w < infos[i].waitSemaphoreCount; ++w) {
                        if (infos[i].pWaitSemaphores[w] == s_r7_watch_sem) {
                            s_r7_wait_hits.fetch_add(1, std::memory_order_relaxed);
                            if (tl && tl->sType == VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO &&
                                w < tl->waitSemaphoreValueCount) {
                                s_r7_last_wait_val.store(tl->pWaitSemaphoreValues[w], std::memory_order_relaxed);
                            }
                        }
                    }
                }
                return s_r7_real_submit(q, count, infos, fence);
            }
        };

        VulkanContext ctx_up(d.device_proc, d.device, true);
        EXPECT(ctx_up.loaded());
        VulkanFence upload_tl(ctx_up, 0);
        EXPECT(upload_tl.semaphore());
        s_r7_real_submit = ctx_up.vk.vkQueueSubmit;
        s_r7_watch_sem = upload_tl.semaphore();
        s_r7_wait_hits.store(0, std::memory_order_relaxed);
        s_r7_last_wait_val.store(0, std::memory_order_relaxed);
        ctx_up.vk.vkQueueSubmit = &R7HookSubmit::call;

        VulkanQueue q_dir(ctx_up, d.queue, d.family, VulkanQueue::QueueKind::direct);
        VulkanQueue q_cmp(ctx_up, d.queue, d.family, VulkanQueue::QueueKind::compute);
        EXPECT(q_dir.ok() && q_cmp.ok());
        q_dir.set_async_depth(0);
        q_cmp.set_async_depth(0);

        // Register a dummy image so write_staged_title_images() exercises its split record/submit path.
        ResourceBinding img_b{};
        img_b.image = reinterpret_cast<VkImage>(0x1234);
        img_b.aspects = VK_IMAGE_ASPECT_COLOR_BIT;
        img_b.mip_levels = 1;
        img_b.array_layers = 1;
        ctx_up.add_resource(99, img_b, VK_IMAGE_LAYOUT_UNDEFINED);

        bool submit_unlocked_in_record = false;
        bool submit_locked_in_submit = false;
        const VulkanContext::StagedImageTarget target{99, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        const VkResult staged_r = ctx_up.write_staged_title_images(
            {&target, 1},
            [&](const ReplayFunctions&, const std::vector<LayoutPieces>& pieces) -> VkResult {
                if (pieces.size() != 1 || pieces[0].empty()) return VK_ERROR_UNKNOWN;
                // submit_mutex_ MUST be unlocked during command recording!
                if (ctx_up.submit_mutex().try_lock()) {
                    submit_unlocked_in_record = true;
                    ctx_up.submit_mutex().unlock();
                }
                return VK_SUCCESS;
            },
            [&](const auto&) -> VkResult {
                // submit_mutex_ MUST be held during submit callback!
                submit_locked_in_submit = !ctx_up.submit_mutex().try_lock();
                ctx_up.note_title_upload_locked(upload_tl.semaphore(), 1);
                return VK_SUCCESS;
            });
        EXPECT(staged_r == VK_SUCCESS);
        EXPECT(submit_unlocked_in_record && submit_locked_in_submit);
        EXPECT(ctx_up.layout(99, 0) == VK_IMAGE_LAYOUT_GENERAL);

        // Signal upload_tl to 1 so GPU waits complete immediately.
        EXPECT(upload_tl.signal(1));

        // A signal-only submit on q_dir must NOT consume title_upload_seen_.
        VulkanFence sig_only(ctx_up, 0);
        EXPECT(sig_only.semaphore());
        EXPECT(q_dir.signal(sig_only.semaphore(), 1));
        EXPECT(sig_only.wait(1, 5'000'000'000ull));
        EXPECT(s_r7_wait_hits.load(std::memory_order_acquire) == 0);

        // First command-buffer submit on q_dir MUST wait on (upload_tl, 1); second MUST NOT wait redundantly!
        EXPECT(q_dir.execute({&snap, 1}) == ReplayResult::success);
        EXPECT(s_r7_wait_hits.load(std::memory_order_acquire) == 1);
        EXPECT(s_r7_last_wait_val.load(std::memory_order_acquire) == 1);
        EXPECT(q_dir.execute({&snap, 1}) == ReplayResult::success);
        EXPECT(s_r7_wait_hits.load(std::memory_order_acquire) == 1);

        // First command-buffer submit on q_cmp MUST ALSO wait on (upload_tl, 1) independently!
        EXPECT(q_cmp.execute({&snap, 1}) == ReplayResult::success);
        EXPECT(s_r7_wait_hits.load(std::memory_order_acquire) == 2);
        EXPECT(q_cmp.execute({&snap, 1}) == ReplayResult::success);
        EXPECT(s_r7_wait_hits.load(std::memory_order_acquire) == 2);

        {
            std::lock_guard lk(ctx_up.submit_mutex());
            ctx_up.clear_title_upload_locked(upload_tl.semaphore());
        }
        EXPECT(q_dir.idle() && q_cmp.idle());
        std::printf("queue r7 sync-tex: per-fence vis_seq, unlocked run_barriers FIFO, lazy title_upload wait & unlocked record ok\n");
    }

    // 14. Round 8 WS3:
    //     (a) Empty-batch before_execute_() / after_execute_() / visibility_seq_ skip + mandatory guardrail that
    //         pending_wait_vis_seq_ is NEVER cleared and carries exec_waits_vis_seq() forward to the next non-empty
    //         execute_now();
    //     (b) Coalescing past already-GPU-completed Waits in BOTH async_worker inbox_ and drain_held_while_executing(),
    //         merging holds().signaled_vis_seq() into group_vis_seq so CPU Fence::Signal visibility edges are preserved;
    //     (c) Dedicated QueueHolds barrier-drain thread: gate.signal() returns immediately without blocking on Present,
    //         active_work_ / any_progress() stays true while barrier runs, and post-barrier Frame N+1 GPU commands wait
    //         via wait_barriers_drained() before running.
    {
        static PFN_vkQueueSubmit s_r8_real_submit = nullptr;
        static std::atomic<unsigned> s_r8_cb_submits{0};
        struct R8HookSubmit {
            static VKAPI_ATTR VkResult VKAPI_CALL call(VkQueue q, std::uint32_t count, const VkSubmitInfo* infos,
                                                       VkFence fence) {
                for (std::uint32_t i = 0; i < count; ++i)
                    if (infos[i].commandBufferCount > 0) s_r8_cb_submits.fetch_add(1, std::memory_order_relaxed);
                return s_r8_real_submit(q, count, infos, fence);
            }
        };

        VulkanContext context(d.device_proc, d.device, true);
        EXPECT(context.loaded());
        s_r8_real_submit = context.vk.vkQueueSubmit;
        s_r8_cb_submits.store(0, std::memory_order_relaxed);
        context.vk.vkQueueSubmit = &R8HookSubmit::call;

        VulkanQueue direct(context, d.queue, d.family, VulkanQueue::QueueKind::direct);
        EXPECT(direct.ok());
        direct.set_async_depth(4);

        void* src_map = nullptr;
        void* dst_map = nullptr;
        VkBuffer src = d.buffer(kBufBytes, usage, &src_map);
        VkBuffer dst = d.buffer(kBufBytes, usage, &dst_map);
        EXPECT(src && dst);
        auto* sb = static_cast<std::uint8_t*>(src_map);
        auto* db = static_cast<std::uint8_t*>(dst_map);
        std::memset(sb, 0, kBufBytes);
        std::memset(db, 0, kBufBytes);
        sb[0] = 0x81;
        sb[16] = 0x82;
        sb[32] = 0x83;
        ResourceBinding r_src{}, r_dst{};
        r_src.buffer = src;
        r_src.size = kBufBytes;
        r_dst.buffer = dst;
        r_dst.size = kBufBytes;
        r_dst.host_readable = true;
        context.add_resource(1, r_src);
        context.add_resource(2, r_dst);

        auto empty_list = std::make_shared<CommandList>();
        EXPECT(empty_list->close());
        std::shared_ptr<const CommandList> empty_snap = empty_list;

        auto make_step = [&](int idx) -> std::shared_ptr<const CommandList> {
            auto l = std::make_shared<CommandList>();
            if (!l->copy_buffer({2, std::uint64_t((idx + 1) * 16), 1, std::uint64_t(idx * 16), 1}) ||
                !l->copy_buffer({2, 0, 1, std::uint64_t(idx * 16), 1}) || !l->close())
                return {};
            return l;
        };
        std::shared_ptr<const CommandList> b0 = make_step(0), b1 = make_step(1), b2 = make_step(2);
        EXPECT(b0 && b1 && b2);

        // (a) Empty-command ExecuteCommandLists MUST still run before_execute_() / after_execute_() (titles and
        //     d3d12_existing_heap rely on empty submissions to flush pending streamed views & title textures).
        std::atomic<unsigned> before_calls{0}, after_calls{0};
        std::atomic<std::uint64_t> last_req_seq{0};
        context.set_before_execute([&] {
            before_calls.fetch_add(1, std::memory_order_acq_rel);
            last_req_seq.store(VulkanContext::current_before_execute_seq(), std::memory_order_release);
        });
        context.set_after_execute([&] { after_calls.fetch_add(1, std::memory_order_acq_rel); });

        const std::uint64_t vis0 = context.current_visibility_seq();
        EXPECT(direct.execute({&empty_snap, 1}) == ReplayResult::success);
        EXPECT(direct.idle());
        EXPECT(context.current_visibility_seq() > vis0);
        EXPECT(before_calls.load(std::memory_order_acquire) == 1);
        EXPECT(after_calls.load(std::memory_order_acquire) == 1);

        // Wait on a CPU-signaled fence (vis_seq V), consume it with an empty list + non-empty list: before_execute_()
        // MUST observe req_seq >= V!
        before_calls.store(0, std::memory_order_release);
        after_calls.store(0, std::memory_order_release);
        VulkanFence cpu_f2(context, 0), done_a(context, 0);
        EXPECT(cpu_f2.semaphore() && done_a.semaphore());
        direct.wait(cpu_f2.semaphore(), 1);
        EXPECT(direct.execute({&empty_snap, 1}) == ReplayResult::success);
        EXPECT(direct.execute({&b0, 1}) == ReplayResult::success);
        EXPECT(direct.signal(done_a.semaphore(), 1));
        EXPECT(direct.idle());
        EXPECT(before_calls.load(std::memory_order_acquire) == 0);

        EXPECT(cpu_f2.signal(1));
        const std::uint64_t cpu_vis2 = context.holds().signaled_vis_seq(cpu_f2.semaphore(), 1);
        EXPECT(done_a.wait(1, 5'000'000'000ull));
        EXPECT(direct.idle());
        EXPECT(before_calls.load(std::memory_order_acquire) == 1);
        EXPECT(after_calls.load(std::memory_order_acquire) == 1);
        EXPECT(last_req_seq.load(std::memory_order_acquire) >= cpu_vis2);
        EXPECT(db[16] == 0x81);

        // (b1) Coalescing past already-GPU-completed Wait in async_worker inbox_ + CPU Fence::Signal vis_seq merge.
        before_calls.store(0, std::memory_order_release);
        s_r8_cb_submits.store(0, std::memory_order_release);
        last_req_seq.store(0, std::memory_order_release);
        std::memset(db, 0, kBufBytes);
        std::atomic<bool> in_lead{false}, go_lead{false};
        context.set_before_execute([&] {
            const unsigned n = before_calls.fetch_add(1, std::memory_order_acq_rel) + 1;
            last_req_seq.store(VulkanContext::current_before_execute_seq(), std::memory_order_release);
            if (n == 1) {
                in_lead.store(true, std::memory_order_release);
                while (!go_lead.load(std::memory_order_acquire)) std::this_thread::yield();
            }
        });

        VulkanFence done_wait_f(context, 0), done_b1(context, 0);
        EXPECT(done_wait_f.semaphore() && done_b1.semaphore());

        EXPECT(direct.execute({&b0, 1}) == ReplayResult::success);  // pauses in before_execute_()
        while (!in_lead.load(std::memory_order_acquire)) std::this_thread::yield();

        // Enqueue b1, then Wait(done_wait_f, 1) (already CPU-signaled with a HIGHER vis_seq!), then b2.
        EXPECT(direct.execute({&b1, 1}) == ReplayResult::success);
        EXPECT(done_wait_f.signal(1));  // GPU counter is already 1 AND carries a fresh CPU vis_seq!
        const std::uint64_t done_wait_vis = context.holds().signaled_vis_seq(done_wait_f.semaphore(), 1);
        direct.wait(done_wait_f.semaphore(), 1);
        EXPECT(direct.execute({&b2, 1}) == ReplayResult::success);
        EXPECT(direct.signal(done_b1.semaphore(), 1));

        go_lead.store(true, std::memory_order_release);
        EXPECT(done_b1.wait(1, 5'000'000'000ull));
        EXPECT(direct.idle());

        // {b0} (1) + coalesced {b1, completed Wait, b2} (2) = exactly 2 submits, and req_seq >= done_wait_vis!
        EXPECT(before_calls.load(std::memory_order_acquire) == 2);
        EXPECT(s_r8_cb_submits.load(std::memory_order_acquire) == 2);
        EXPECT(last_req_seq.load(std::memory_order_acquire) >= done_wait_vis);
        EXPECT(db[16] == 0x81 && db[32] == 0x82 && db[48] == 0x83 && db[0] == 0x83);

        // (b2) Coalescing past already-GPU-completed Wait in drain_held_while_executing() + CPU vis_seq merge.
        before_calls.store(0, std::memory_order_release);
        s_r8_cb_submits.store(0, std::memory_order_release);
        last_req_seq.store(0, std::memory_order_release);
        std::memset(db, 0, kBufBytes);
        context.set_before_execute([&] {
            before_calls.fetch_add(1, std::memory_order_acq_rel);
            last_req_seq.store(VulkanContext::current_before_execute_seq(), std::memory_order_release);
        });

        VulkanFence hold_gate(context, 0), held_cpu_f(context, 0), done_b2(context, 0);
        EXPECT(hold_gate.semaphore() && held_cpu_f.semaphore() && done_b2.semaphore());

        direct.wait(hold_gate.semaphore(), 1);
        EXPECT(direct.execute({&b1, 1}) == ReplayResult::success);
        direct.wait(held_cpu_f.semaphore(), 1);  // unsignaled when queued -> lands in b2.waits in held_
        EXPECT(direct.execute({&b2, 1}) == ReplayResult::success);
        EXPECT(direct.signal(done_b2.semaphore(), 1));
        EXPECT(direct.idle());  // both b1 and b2 are now in held_
        EXPECT(before_calls.load(std::memory_order_acquire) == 0);

        // Signal held_cpu_f from CPU first (now GPU-completed with a newer vis_seq), THEN release hold_gate!
        EXPECT(held_cpu_f.signal(1));
        const std::uint64_t held_cpu_vis = context.holds().signaled_vis_seq(held_cpu_f.semaphore(), 1);
        EXPECT(hold_gate.signal(1));
        EXPECT(done_b2.wait(1, 5'000'000'000ull));
        EXPECT(direct.idle());

        // Held {b1, completed Wait(held_cpu_f), b2} coalesced into ONE submit & preserved held_cpu_vis!
        EXPECT(before_calls.load(std::memory_order_acquire) == 1);
        EXPECT(s_r8_cb_submits.load(std::memory_order_acquire) == 1);
        EXPECT(last_req_seq.load(std::memory_order_acquire) >= held_cpu_vis);
        EXPECT(db[32] == 0x82 && db[48] == 0x83 && db[0] == 0x83);

        // (c) Dedicated QueueHolds barrier-drain thread: releaser returns immediately while barrier blocks,
        //     any_progress() stays true, post-barrier Frame N+1 strictly waits for barrier completion OUTSIDE
        //     pump_mutex_, and an independent compute queue's held work still pumps and completes while direct's
        //     Frame N+1 is waiting on blocked Present N!
        VulkanQueue compute_c(context, d.queue, d.family, VulkanQueue::QueueKind::compute);
        EXPECT(compute_c.ok());
        compute_c.set_async_depth(4);

        before_calls.store(0, std::memory_order_release);
        std::memset(db, 0, kBufBytes);
        context.set_before_execute([&] { before_calls.fetch_add(1, std::memory_order_acq_rel); });

        VulkanFence f_gate(context, 0), f_done_c(context, 0), comp_gate_c(context, 0), comp_done_c(context, 0);
        EXPECT(f_gate.semaphore() && f_done_c.semaphore() && comp_gate_c.semaphore() && comp_done_c.semaphore());

        direct.wait(f_gate.semaphore(), 1);
        EXPECT(direct.execute({&b0, 1}) == ReplayResult::success);  // Frame N
        EXPECT(direct.idle());

        std::atomic<bool> in_present{false}, release_present{false};
        std::atomic<unsigned> present_saw_before{0};
        int present_owner = 0;
        context.holds().after_held(&present_owner, [&] {
            present_saw_before.store(before_calls.load(std::memory_order_acquire), std::memory_order_release);
            in_present.store(true, std::memory_order_release);
            while (!release_present.load(std::memory_order_acquire)) std::this_thread::yield();
        });

        EXPECT(direct.execute({&b1, 1}) == ReplayResult::success);  // Frame N+1 (must wait for Present N!)
        EXPECT(direct.signal(f_done_c.semaphore(), 1));
        EXPECT(direct.idle());

        // Hold independent work on compute_c behind comp_gate_c (skip before_execute so before_calls tracks direct).
        compute_c.set_skip_before_execute(true);
        compute_c.wait(comp_gate_c.semaphore(), 1);
        EXPECT(compute_c.execute({&empty_snap, 1}) == ReplayResult::success);
        EXPECT(compute_c.signal(comp_done_c.semaphore(), 1));
        EXPECT(compute_c.idle());

        // Releaser signals f_gate: MUST return immediately without stalling on blocked Present N!
        EXPECT(f_gate.signal(1));
        while (!in_present.load(std::memory_order_acquire)) std::this_thread::yield();

        // While Present N is blocked on the barrier thread AND direct's async_worker is waiting in
        // wait_barriers_drained() for Frame N+1:
        //   - any_progress() == true;
        //   - Frame N+1 (b1) has NOT run yet;
        //   - pump_mutex_ MUST be free so comp_gate_c.signal(1) pumps compute_c and completes comp_done_c immediately!
        std::this_thread::sleep_for(std::chrono::milliseconds(15));
        EXPECT(queue_holds::any_progress());
        EXPECT(before_calls.load(std::memory_order_acquire) == 1);  // only Frame N (b0) ran; b1 waits for Present N!
        EXPECT(present_saw_before.load(std::memory_order_acquire) == 1);
        EXPECT(comp_gate_c.signal(1));
        EXPECT(comp_done_c.wait(1, 2'000'000'000ull));
        EXPECT(compute_c.idle());
        EXPECT(before_calls.load(std::memory_order_acquire) == 1);  // b1 STILL waits for Present N!

        release_present.store(true, std::memory_order_release);
        EXPECT(f_done_c.wait(1, 5'000'000'000ull));
        EXPECT(direct.idle());
        EXPECT(before_calls.load(std::memory_order_acquire) == 2);
        EXPECT(db[16] == 0x81 && db[32] == 0x82 && db[0] == 0x82);
        std::printf("queue r8 ws3: empty-batch skip+vis carry, completed-wait coalescing, dedicated barrier thread ok\n");
    }

    return 0;
}

}  // namespace

int main() {
    if (scripted() || submissions() || queue_sync_and_async()) return 1;
    std::printf("submit_frame_test: OK\n");
    return 0;
}
