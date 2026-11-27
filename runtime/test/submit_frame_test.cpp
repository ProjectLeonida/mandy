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

// Round 1D queue concurrency, batch coalescing, zero-copy CommandList snapshot, and async worker tests.
int queue_async_and_coalescing() {
    vktest::Device d;
    if (!d.create()) {
        std::printf("no usable Vulkan device: the queue_async_and_coalescing part is skipped\n");
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

    // 1. Concurrent submission while before_execute_() is running must not block on state_mutex_, must preserve
    //    strict FIFO execution order, and must coalesce consecutive queued batches on drain.
    {
        VulkanContext context(d.device_proc, d.device, true);
        EXPECT(context.loaded());
        VulkanQueue queue(context, d.queue, d.family, VulkanQueue::QueueKind::direct);
        EXPECT(queue.ok());

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

        VulkanFence done(context, 0);
        EXPECT(done.semaphore() != VK_NULL_HANDLE);

        ReplayResult t_res = ReplayResult::unsupported;
        std::thread worker([&] {
            std::shared_ptr<const CommandList> s1 = b1;
            t_res = queue.execute({&s1, 1});
        });

        while (!in_first_before.load(std::memory_order_acquire)) std::this_thread::yield();

        // While worker is blocked inside before_execute_(), this thread submits b2, b3, and Signal(done, 1).
        // If state_mutex_ were held across before_execute_(), this would deadlock before release_first_before is set!
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
        worker.join();
        EXPECT(t_res == ReplayResult::success);
        EXPECT(done.wait(1, 5'000'000'000ull));
        EXPECT(queue.idle());

        // b1 ran first (before_calls #1), then b2+b3 coalesced into ONE execute_now() (before_calls #2).
        EXPECT(before_calls.load(std::memory_order_acquire) == 2);
        EXPECT(db[16] == 0x11 && db[32] == 0x22 && db[48] == 0x33);
        EXPECT(db[0] == 0x33);  // strict FIFO: b1 -> b2 -> b3
        // After completion, s2/s3 snapshots are released (only local s3 holds b3's old CommandList).
        s2.reset();
        s3.reset();
        EXPECT(b1.use_count() == 1 && b3.use_count() == 1);
        EXPECT(b3->reset());  // steady-state reuse with zero allocation
        std::printf("queue sync: unlocked before_execute_, strict FIFO, coalesced drain, zero-copy Reset ok\n");
    }

    // 2. Held batches behind a fence gate coalesce consecutive ready batches into ONE before_execute_() call,
    //    while an intervening Signal or Wait stops coalescing. Also verify invalid-batch fallback inside a
    //    coalesced group does not drop valid peers.
    {
        VulkanContext context(d.device_proc, d.device, true);
        EXPECT(context.loaded());
        VulkanQueue queue(context, d.queue, d.family, VulkanQueue::QueueKind::direct);
        VulkanQueue copy_q(context, d.queue, d.family, VulkanQueue::QueueKind::copy);
        EXPECT(queue.ok() && copy_q.ok());

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
        EXPECT(queue.execute({&steps[0], 1}) == ReplayResult::success);  // group 1: steps 0, 1, 2
        EXPECT(queue.execute({&steps[1], 1}) == ReplayResult::success);
        EXPECT(queue.execute({&steps[2], 1}) == ReplayResult::success);
        EXPECT(queue.signal(mid.semaphore(), 1));                        // Signal stops coalescing
        EXPECT(queue.execute({&steps[3], 1}) == ReplayResult::success);  // group 2: step 3 alone
        queue.wait(mid.semaphore(), 1);                                  // Wait stops coalescing
        EXPECT(queue.execute({&steps[4], 1}) == ReplayResult::success);  // group 3: steps 4, 5
        EXPECT(queue.execute({&steps[5], 1}) == ReplayResult::success);
        EXPECT(queue.signal(done.semaphore(), 1));

        EXPECT(before_calls.load() == 0);
        EXPECT(gate.signal(1));
        EXPECT(done.wait(1, 5'000'000'000ull));
        EXPECT(queue.idle());

        // Exactly 3 before_execute_() calls: {0,1,2}, {3}, {4,5}.
        EXPECT(before_calls.load() == 3);
        for (int i = 0; i < 6; ++i) EXPECT(db[(i + 1) * 16] == std::uint8_t(0xA1 + i));
        EXPECT(db[0] == std::uint8_t(0xA6));  // strict FIFO through all 6 batches

        // Coalesced fallback: [valid, invalid ClearRenderTarget on copy queue, valid] held behind gate2.
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
        std::printf("queue held coalescing: 6 batches -> 3 submits, Signal/Wait split, invalid fallback ok\n");
    }

    // 3. LEONIDA_SUBMIT_ASYNC (per-queue async worker depth=4): inbox coalescing, Present barrier split,
    //    held-batch release through worker, and zero-copy CommandList Reset() while queued.
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

        // Enqueue c1, c2 (will coalesce), then a Present after_held barrier, then c3 (must NOT coalesce past barrier).
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

        // 3 before_execute_() calls total: {c0}, {c1, c2}, {c3}; barrier ran after {c1, c2} (count == 2), before {c3}!
        EXPECT(before_calls.load() == 3);
        EXPECT(barrier_seen_before_calls.load() == 2);
        EXPECT(db[16] == 0xC1 && db[32] == 0xC2 && db[48] == 0xC3 && db[64] == 0xC4);

        // Held batches behind a gate in async mode: release_ready() wakes worker and coalesces them.
        before_calls.store(0);
        std::memset(db, 0, kBufBytes);
        VulkanFence gate(context, 0), done2(context, 0);
        EXPECT(gate.semaphore() && done2.semaphore());
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
        std::printf("queue async (depth=4): inbox coalescing, Present barrier split, gate release ok\n");
    }

    // 4. Async queue progress tracking (LEONIDA_SUBMIT_ASYNC=4 + queued Present + busy producer worker + genuine
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

    return 0;
}

}  // namespace

int main() {
    if (scripted() || submissions() || queue_async_and_coalescing()) return 1;
    std::printf("submit_frame_test: OK\n");
    return 0;
}
