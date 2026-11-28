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
    sp::frame_lines(1.0);
    std::printf("scripted: frame sums, title and own threads, worst frame and floor as scripted\n");
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
        EXPECT(empty_other->close());
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
        EXPECT(empty->close());
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
        // split at Wait(mid), and coalesce {steps[4..5]} (3): 6 batches -> 3 before_execute_() calls!
        EXPECT(before_calls.load() == 3);
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
        EXPECT(empty_list->close());
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

        // 9 execute calls coalesced into exactly 5 before_execute_() calls AND 5 command-buffer vkQueueSubmits!
        EXPECT(before_calls.load(std::memory_order_acquire) == 5);
        EXPECT(s_cb_submits.load(std::memory_order_acquire) == 5);
        EXPECT(present_saw_submits.load(std::memory_order_acquire) == 4);
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
        EXPECT(empty_list->close());
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

    return 0;
}

}  // namespace

int main() {
    if (scripted() || submissions() || queue_sync_and_async()) return 1;
    std::printf("submit_frame_test: OK\n");
    return 0;
}
