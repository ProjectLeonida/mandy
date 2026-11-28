// Portable CPU-only tests for runtime/src/list_value_progress.h: when a held console memory wait may go without its
// value. Nothing can make progress first; then a cycle goes at once, a root (a wait no held write meets) goes at a
// present queued after it or when the title has stopped calling, and a gate whose writer is held behind another
// queue's root waits for that root. No time alone ever releases a gate. No Windows, D3D12 or Vulkan headers.
#include "../src/list_value_progress.h"
#include "../src/queue_holds.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <mutex>
#include <thread>
#include <vector>

#define CHECK(cond)                                                                \
    do {                                                                           \
        if (!(cond)) {                                                             \
            std::fprintf(stderr, "line %d: CHECK failed: %s\n", __LINE__, #cond); \
            return 1;                                                              \
        }                                                                          \
    } while (0)

namespace {

using namespace leonida::list_value_progress;

constexpr bool met(std::uint32_t comparison, std::uint64_t value, std::uint64_t reference, bool wide) {
    if (!wide) {
        value &= 0xffffffffu;
        reference &= 0xffffffffu;
    }
    switch (comparison) {
    case 2: return value < reference;
    case 3: return value == reference;
    case 4: return value <= reference;
    case 5: return value > reference;
    case 6: return value != reference;
    case 7: return value >= reference;
    default: return true;
    }
}

int qa_, qb_, qc_;
const void* const qa = &qa_;
const void* const qb = &qb_;
const void* const qc = &qc_;

PendingGate gate(const void* queue, std::uint64_t value, std::span<const WaitPredicate> waits,
                 std::uint64_t presents_at_hold = 0, std::uint64_t held_ms = 0) {
    PendingGate g;
    g.queue = queue;
    g.value = value;
    g.unmet_waits = waits;
    g.presents_at_hold = presents_at_hold;
    g.held_ms = held_ms;
    return g;
}

PendingWriteGroup group(const void* queue, std::uint64_t after_gate, std::span<const PendingWrite> writes,
                        bool running = false) {
    PendingWriteGroup w;
    w.queue = queue;
    w.after_gate = after_gate;
    w.running = running;
    w.writes = writes;
    return w;
}

Decision run(std::span<const PendingWriteGroup> writes, std::span<const PendingGate> gates, const Activity& activity) {
    return decide(writes, gates, activity, met);
}

}  // namespace

int main() {
    const WaitPredicate wait_x[] = {{0x1000, 5, 7, true}};  // x >= 5
    const WaitPredicate wait_y[] = {{0x2000, 9, 7, true}};  // y >= 9
    const WaitPredicate wait_z[] = {{0x3000, 1, 3, false}}; // z == 1 (32-bit)
    const PendingWrite write_x[] = {{0x1000, 5, 8, false}};
    const PendingWrite write_y[] = {{0x2000, 9, 8, false}};
    const PendingWrite write_x_low[] = {{0x1000, 4, 8, false}};

    // 1. One gate, nothing in flight, no present, the title calling: held, however long it has stood.
    {
        const PendingGate gates[] = {gate(qa, 1, wait_x, 0, 0)};
        Activity a;
        a.now_ms = 3'600'000;      // an hour after it was held
        a.title_call_ms = a.now_ms;  // the title called just now
        CHECK(run({}, gates, a).reason == Reason::hold);
    }

    // 2. Something can still make progress: no release, whatever else holds (a present queued, the title quiet).
    {
        const PendingGate gates[] = {gate(qa, 1, wait_x, 0, 0)};
        Activity quiet;
        quiet.presents = 5;
        quiet.now_ms = 100'000;
        Activity a = quiet;
        a.releasing = true;
        CHECK(run({}, gates, a).reason == Reason::hold);
        a = quiet;
        a.title_submitting = true;
        CHECK(run({}, gates, a).reason == Reason::hold);
        // A running write group, even one that does not meet the wait: it may unblock what does.
        const PendingWriteGroup running[] = {group(qb, 0, write_y, true)};
        CHECK(run(running, gates, quiet).reason == Reason::hold);
        // A running action on another queue.
        PendingGate action = gate(qc, 1, {});
        action.is_action = true;
        action.action_running = true;
        const PendingGate with_action[] = {gates[0], action};
        CHECK(run({}, with_action, quiet).reason == Reason::hold);
        CHECK(can_progress(running, gates, quiet) && !can_progress({}, gates, quiet));
        // The same with nothing running: the present releases it.
        CHECK(run({}, gates, quiet).reason == Reason::present && run({}, gates, quiet).gate == 0);
    }

    // 3. Present rule: only a present made after the gate was held counts.
    {
        const PendingGate gates[] = {gate(qa, 1, wait_x, 3, 0)};
        Activity a;
        a.now_ms = 10;
        a.title_call_ms = 10;
        a.presents = 3;  // the present before the hold
        CHECK(run({}, gates, a).reason == Reason::hold);
        a.presents = 4;
        const Decision d = run({}, gates, a);
        CHECK(d.reason == Reason::present && d.gate == 0);
    }

    // 4. Stalled rule: the title made no queue call for kStalledMs while nothing ran, and the gate stood that long.
    {
        const PendingGate gates[] = {gate(qa, 1, wait_x, 0, 1000)};
        Activity a;
        a.title_call_ms = 1000;
        a.now_ms = 1000 + kStalledMs - 1;
        CHECK(run({}, gates, a).reason == Reason::hold);
        a.now_ms = 1000 + kStalledMs;
        CHECK(run({}, gates, a).reason == Reason::stalled);
        a.title_call_ms = a.now_ms - 1;  // a call just now: held again
        CHECK(run({}, gates, a).reason == Reason::hold);
        // A gate held after the title's last call needs to have stood kStalledMs too.
        const PendingGate young[] = {gate(qa, 1, wait_x, 0, 1000 + kStalledMs)};
        a.title_call_ms = 1000;
        CHECK(run({}, young, a).reason == Reason::hold);
    }

    // 5. A cycle goes at once, without a present: A waits for x, written by B's group held behind B's gate, which
    //    waits for y, written by A's group held behind A's gate.
    {
        const PendingGate gates[] = {gate(qa, 1, wait_x), gate(qb, 1, wait_y)};
        const PendingWriteGroup writes[] = {group(qb, 1, write_x), group(qa, 1, write_y)};
        Activity a;
        a.now_ms = 5;
        a.title_call_ms = 5;
        const Decision d = run(writes, gates, a);
        CHECK(d.reason == Reason::cycle && d.gate == 0);
        // A queue's own later write behind its gate: a cycle of one queue.
        const PendingGate self[] = {gate(qa, 1, wait_x)};
        const PendingWriteGroup own[] = {group(qa, 1, write_x)};
        CHECK(run(own, self, a).reason == Reason::cycle);
        // But while either writer runs, nothing goes.
        const PendingWriteGroup running[] = {group(qb, 1, write_x), group(qa, 0, write_y, true)};
        CHECK(run(running, gates, a).reason == Reason::hold);
    }

    // 6. A chain unwinds from its root: A waits for x, which B's group writes behind B's gate; B's gate waits for z,
    //    which nothing writes. At a present, B's gate goes first (A then waits for B's write), though A is older.
    {
        const PendingGate gates[] = {gate(qa, 1, wait_x), gate(qb, 1, wait_z)};
        const PendingWriteGroup writes[] = {group(qb, 1, write_x)};
        Activity a;
        a.now_ms = 5;
        a.title_call_ms = 5;
        CHECK(run(writes, gates, a).reason == Reason::hold);  // no present, the title calling: held
        a.presents = 1;
        const Decision d = run(writes, gates, a);
        CHECK(d.reason == Reason::present && d.gate == 1);
    }

    // 7. A write that does not meet the wait (too low), or one overwritten later in its group, is no writer: the gate
    //    is a root. A timestamp write to the address counts as a writer.
    {
        const PendingGate gates[] = {gate(qa, 1, wait_x)};
        const PendingWriteGroup low[] = {group(qb, 1, write_x_low)};
        const PendingGate b_gate[] = {gates[0], gate(qb, 1, wait_z)};
        Activity a;
        a.presents = 1;
        a.now_ms = 5;
        a.title_call_ms = 5;
        Decision d = run(low, b_gate, a);
        CHECK(d.reason == Reason::present && d.gate == 0);  // both roots: the older goes
        const PendingWrite overwritten[] = {{0x1000, 5, 8, false}, {0x1000, 4, 8, false}};
        const PendingWriteGroup over[] = {group(qb, 1, overwritten)};
        d = run(over, b_gate, a);
        CHECK(d.reason == Reason::present && d.gate == 0);
        const PendingWrite stamp[] = {{0x1000, 0, 8, true}};
        const PendingWriteGroup stamped[] = {group(qb, 1, stamp)};
        d = run(stamped, b_gate, a);
        CHECK(d.reason == Reason::present && d.gate == 1);  // A waits for B's timestamp: B's root goes first
    }

    // 8. Only a queue's first gate is a candidate; a queue's later gate never goes before its first.
    {
        const PendingGate gates[] = {gate(qa, 1, wait_x, 1), gate(qa, 2, wait_y, 0)};
        Activity a;
        a.presents = 1;  // after the second gate's hold, not after the first's
        a.now_ms = 5;
        a.title_call_ms = 5;
        CHECK(run({}, gates, a).reason == Reason::hold);
        a.presents = 2;
        const Decision d = run({}, gates, a);
        CHECK(d.reason == Reason::present && d.gate == 0);
    }

    // 9. A writer held by something not modelled here (a relayed wait or a title fence; no gate of ours before it):
    //    the gate counts as a root, so the present releases it rather than waiting for ever.
    {
        const PendingGate gates[] = {gate(qa, 1, wait_x)};
        const PendingWriteGroup writes[] = {group(qb, 0, write_x, false)};
        Activity a;
        a.presents = 1;
        a.now_ms = 5;
        a.title_call_ms = 5;
        const Decision d = run(writes, gates, a);
        CHECK(d.reason == Reason::present && d.gate == 0);
    }

    // 10. Action gates are never released by these rules (they open when their point of the queue is reached).
    {
        PendingGate action = gate(qa, 1, {});
        action.is_action = true;
        const PendingGate gates[] = {action};
        Activity a;
        a.presents = 9;
        a.now_ms = 100'000;
        CHECK(run({}, gates, a).reason == Reason::hold);
    }

    // 11. QueueHolds progress integration (LEONIDA_SUBMIT_ASYNC + busy worker + unretired GPU work):
    //     a held gate with a queued Present must NOT be released unmet while any queue has active CPU/async work,
    //     an unpumped signal for held work, or unretired GPU work; once all queues and the GPU are truly idle,
    //     genuine Present and cycle deadlocks still break cleanly.
    {
        struct MockHolder final : leonida::QueueHolds<std::uint64_t>::Holder {
            std::atomic<bool> gpu_busy{false};
            bool release_ready() override { return false; }
            void discard_waiting_on(std::uint64_t) override {}
            std::uint64_t oldest_held_seq() override { return 0; }
            bool has_pending_gpu_work() const override { return gpu_busy.load(std::memory_order_acquire); }
        };

        leonida::QueueHolds<std::uint64_t> holds;
        MockHolder holder;
        holds.attach(&holder);
        holds.add_fence(1, 0);
        CHECK(!leonida::queue_holds::any_progress());

        const PendingGate gates[] = {gate(qa, 1, wait_x, 0, 0)};
        // Producer's write group has running == false because its Signal is still sitting in an async queue inbox.
        const PendingWriteGroup queued_write[] = {group(qb, 0, write_x, false)};
        Activity at_present;
        at_present.presents = 1;
        at_present.now_ms = 250;
        at_present.title_call_ms = 10;

        // 11a. Active async inbox / busy worker keeps any_progress() true -> gate stays held at Present.
        {
            leonida::QueueHolds<std::uint64_t>::WorkGuard busy(holds);
            CHECK(leonida::queue_holds::any_progress());
            Activity a = at_present;
            a.title_submitting = leonida::queue_holds::any_progress();
            CHECK(run(queued_write, gates, a).reason == Reason::hold);
        }
        CHECK(!leonida::queue_holds::any_progress());

        // 11b. Unpumped Signal while held work exists keeps any_progress() true until pump_ready() runs.
        holds.held_changed(1);
        holds.signaled(1, 1);
        CHECK(leonida::queue_holds::any_progress());
        {
            Activity a = at_present;
            a.title_submitting = leonida::queue_holds::any_progress();
            CHECK(run(queued_write, gates, a).reason == Reason::hold);
        }
        holds.pump_ready();
        holds.held_changed(-1);
        CHECK(!leonida::queue_holds::any_progress());

        // 11c. Unretired GPU submission on any queue keeps any_progress() true -> gate stays held at Present.
        holder.gpu_busy.store(true, std::memory_order_release);
        CHECK(leonida::queue_holds::any_progress());
        {
            Activity a = at_present;
            a.title_submitting = leonida::queue_holds::any_progress();
            CHECK(run({}, gates, a).reason == Reason::hold);
        }
        // Once the GPU retires and all queues are idle, a genuine Present deadlock breaks cleanly.
        holder.gpu_busy.store(false, std::memory_order_release);
        CHECK(!leonida::queue_holds::any_progress());
        {
            Activity a = at_present;
            a.title_submitting = leonida::queue_holds::any_progress();
            const Decision d = run({}, gates, a);
            CHECK(d.reason == Reason::present && d.gate == 0);
        }

        // 11d. Genuine cycle deadlock also breaks cleanly once queues and GPU are idle.
        {
            const PendingGate cycle_gates[] = {gate(qa, 1, wait_x), gate(qb, 1, wait_y)};
            const PendingWriteGroup cycle_writes[] = {group(qb, 1, write_x), group(qa, 1, write_y)};
            Activity a;
            a.now_ms = 5;
            a.title_call_ms = 5;
            a.title_submitting = leonida::queue_holds::any_progress();
            const Decision d = run(cycle_writes, cycle_gates, a);
            CHECK(d.reason == Reason::cycle && d.gate == 0);
        }

        holds.detach(&holder);
    }

    // 12. Round 8 P0 TOCTOU + Present quiet-window debounce invariants:
    //     (a) A write group or action whose GPU fence already completed (done >= target, awaiting take_landed()) MUST
    //         count as active progress (running == true -> Reason::hold even at Present), never stuck!
    //     (b) A 10 ms quiet debounce on Reason::present (matching Pump::choose_release()) defers unmet Present release
    //         while progress occurred within the last 10 ms, while leaving Reason::cycle immediate and decide() pure.
    {
        const PendingGate gates[] = {gate(qa, 1, wait_x, 0, 0)};
        Activity at_present;
        at_present.presents = 1;
        at_present.now_ms = 100;
        at_present.title_call_ms = 95;

        // Simulate write group whose GPU fence already retired (done == target, done != UINT64_MAX):
        // must be classified as active progress (never passed to decide() as running == false).
        const auto write_progress = [](std::uint64_t done, std::uint64_t target, bool submitted) {
            return done != UINT64_MAX && (done >= target || submitted);
        };
        CHECK(write_progress(/*done=*/5, /*target=*/5, /*submitted=*/false));   // retired between take_landed & check
        CHECK(write_progress(/*done=*/6, /*target=*/5, /*submitted=*/false));   // retired past target
        CHECK(write_progress(/*done=*/4, /*target=*/5, /*submitted=*/true));    // in flight on GPU
        CHECK(!write_progress(/*done=*/4, /*target=*/5, /*submitted=*/false));  // held in layer, not submitted
        CHECK(!write_progress(UINT64_MAX, /*target=*/5, /*submitted=*/true));   // device lost

        // Present quiet window helper matching Pump::choose_release(): gates ONLY Reason::present.
        constexpr std::uint64_t kPresentQuietMs = 10;
        const auto apply_present_quiet = [&](Decision d, std::uint64_t now_ms, std::uint64_t last_progress_ms) {
            if (d.reason == Reason::present && now_ms >= last_progress_ms &&
                now_ms - last_progress_ms < kPresentQuietMs)
                return Decision{};
            return d;
        };
        const Decision raw_present = run({}, gates, at_present);
        CHECK(raw_present.reason == Reason::present && raw_present.gate == 0);  // decide() itself remains pure
        CHECK(apply_present_quiet(raw_present, /*now_ms=*/100, /*last_progress_ms=*/95).reason == Reason::hold);
        CHECK(apply_present_quiet(raw_present, /*now_ms=*/104, /*last_progress_ms=*/95).reason == Reason::hold);
        CHECK(apply_present_quiet(raw_present, /*now_ms=*/105, /*last_progress_ms=*/95).reason == Reason::present);

        // Cycle release is NOT delayed by the Present quiet window.
        const PendingGate cycle_gates[] = {gate(qa, 1, wait_x), gate(qb, 1, wait_y)};
        const PendingWriteGroup cycle_writes[] = {group(qb, 1, write_x), group(qa, 1, write_y)};
        const Decision raw_cycle = run(cycle_writes, cycle_gates, at_present);
        CHECK(raw_cycle.reason == Reason::cycle && raw_cycle.gate == 0);
        CHECK(apply_present_quiet(raw_cycle, /*now_ms=*/100, /*last_progress_ms=*/100).reason == Reason::cycle);
    }

    // 13. Round 8 P2 QueueHolds dedicated barrier-drain thread guardrails (a), (b), (c):
    //     (a) active_work_ / has_progress() stays true while barriers are queued or running on the barrier thread;
    //     (b) wait_barriers_drained(seq) waits for ready/running barriers <= seq without blocking on barriers > seq;
    //     (c) cancel_after(owner) cancels queued barriers (balancing active_work_) and waits for any running barrier;
    //         ~QueueHolds() cleanly shuts down and joins the barrier thread.
    {
        struct SeqHolder final : leonida::QueueHolds<std::uint64_t>::Holder {
            std::atomic<std::uint64_t> oldest{0};
            bool release_ready() override { return false; }
            void discard_waiting_on(std::uint64_t) override {}
            std::uint64_t oldest_held_seq() override { return oldest.load(std::memory_order_acquire); }
        };

        {
            leonida::QueueHolds<std::uint64_t> holds;
            SeqHolder holder;
            holds.attach(&holder);

            // Hold seq 1 so barriers wait until released.
            const std::uint64_t s1 = holds.next_seq();
            holder.oldest.store(s1, std::memory_order_release);
            holds.held_changed(1);

            std::atomic<bool> in_b1{false}, go_b1{false}, b2_ran{false}, b3_ran{false};
            std::mutex order_m;
            std::vector<int> order;
            int owner_a = 0, owner_cancel = 0;

            holds.after_held(&owner_a, [&] {
                in_b1.store(true, std::memory_order_release);
                while (!go_b1.load(std::memory_order_acquire)) std::this_thread::yield();
                std::lock_guard lk(order_m);
                order.push_back(1);
            });
            holds.after_held(&owner_a, [&] {
                b2_ran.store(true, std::memory_order_release);
                std::lock_guard lk(order_m);
                order.push_back(2);
            });
            holds.after_held(&owner_cancel, [&] { b3_ran.store(true, std::memory_order_release); });

            // Release held op: pump_ready() hands barriers to the dedicated barrier thread and returns IMMEDIATELY!
            holder.oldest.store(0, std::memory_order_release);
            holds.held_changed(-1);
            holds.pump_ready();

            while (!in_b1.load(std::memory_order_acquire)) std::this_thread::yield();

            // Guardrail (a): active_work_ / has_progress() MUST be true while b1 is running & b2/b3 are queued.
            CHECK(holds.has_progress());
            CHECK(leonida::queue_holds::any_progress());

            // Guardrail (b): wait_barriers_drained(s1) MUST block until b1 & b2 finish; wait_barriers_drained(s1 - 1)
            // returns immediately when s1 > 0.
            if (s1 > 0) holds.wait_barriers_drained(s1 - 1);
            std::atomic<bool> drained_s1{false};
            std::thread waiter([&] {
                holds.wait_barriers_drained(s1);
                drained_s1.store(true, std::memory_order_release);
            });
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            CHECK(!drained_s1.load(std::memory_order_acquire));

            // Guardrail (c): cancel_after(&owner_cancel) drops queued b3 and waits for running b1 before returning.
            std::atomic<bool> cancel_done{false};
            std::thread canceller([&] {
                holds.cancel_after(&owner_cancel);
                cancel_done.store(true, std::memory_order_release);
            });
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            CHECK(!cancel_done.load(std::memory_order_acquire));

            go_b1.store(true, std::memory_order_release);
            waiter.join();
            canceller.join();

            CHECK(drained_s1.load(std::memory_order_acquire));
            CHECK(cancel_done.load(std::memory_order_acquire));
            CHECK(b2_ran.load(std::memory_order_acquire));
            CHECK(!b3_ran.load(std::memory_order_acquire));
            CHECK((order == std::vector<int>{1, 2}));

            // Once all barriers finished (and cancelled b3 decremented active_work_), has_progress() settles to false!
            const auto settle = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
            while (holds.has_progress() && std::chrono::steady_clock::now() < settle)
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            CHECK(!holds.has_progress());
            CHECK(!leonida::queue_holds::any_progress());

            holds.detach(&holder);
        }  // ~QueueHolds() cleanly stops and joins barrier_thread_
    }

    // 14. Round 10 WS3 QueueHolds::relay_wait redirect table, pruning, multi-wait substitution, and in-transit lookup:
    {
        struct RelayHolder final : leonida::QueueHolds<std::uint64_t>::Holder {
            using Wait = leonida::QueueHolds<std::uint64_t>::Wait;
            struct Op {
                std::vector<std::uint64_t> sems;
                std::vector<std::uint64_t> vals;
            };
            std::vector<Op> held;
            std::atomic<unsigned> release_calls{0};
            bool release_ready() override {
                release_calls.fetch_add(1, std::memory_order_relaxed);
                return false;
            }
            void discard_waiting_on(std::uint64_t) override {}
            std::uint64_t oldest_held_seq() override { return 0; }
            bool apply_wait_relay(std::uint64_t gate_sem, std::uint64_t gate_val, std::span<const Wait> replacements,
                                  bool& any_substituted) override {
                bool updated = false;
                for (Op& op : held) {
                    const std::size_t n = op.sems.size();
                    for (std::size_t i = 0; i < n; ++i) {
                        if (op.sems[i] != gate_sem || op.vals[i] != gate_val) continue;
                        any_substituted = true;
                        updated = true;
                        op.sems[i] = replacements[0].semaphore;
                        op.vals[i] = replacements[0].value;
                        for (std::size_t k = 1; k < replacements.size(); ++k) {
                            op.sems.push_back(replacements[k].semaphore);
                            op.vals.push_back(replacements[k].value);
                        }
                    }
                }
                return updated;
            }
        };

        leonida::QueueHolds<std::uint64_t> holds;
        RelayHolder holder;
        holds.attach(&holder);
        holds.add_fence(10, 0);  // gate fence (never signaled!)
        holds.add_fence(20, 0);  // writer A
        holds.add_fence(30, 0);  // writer B
        holds.add_fence(40, 0);  // already-signaled gate
        holds.signaled(40, 5, 0);

        using Wait = leonida::QueueHolds<std::uint64_t>::Wait;
        // Reject invalid relays (self-relay, empty, already-passed gate).
        const Wait self_rep[] = {{10, 1, {}}};
        CHECK(!holds.relay_wait(10, 1, self_rep));
        CHECK(!holds.relay_wait(10, 1, {}));
        const Wait ok_rep[] = {{20, 2, {}}};
        CHECK(!holds.relay_wait(40, 5, ok_rep));  // gate 40 already at 5
        CHECK(holds.active_relay_count() == 0);

        // Multi-wait substitution in held op + consume_on_substitute immediate pruning + pump_ready() trigger.
        holder.held.push_back({{10}, {1}});
        holds.held_changed(1);
        const Wait multi_rep[] = {{20, 2, {}}, {30, 3, {}}};
        CHECK(holds.relay_wait(10, 1, multi_rep, /*consume_on_substitute=*/true));
        CHECK(holds.active_relay_count() == 0);  // pruned immediately on substitute!
        CHECK(holder.release_calls.load() >= 1);
        CHECK((holder.held[0].sems == std::vector<std::uint64_t>{20, 30}));
        CHECK((holder.held[0].vals == std::vector<std::uint64_t>{2, 3}));
        CHECK(holds.pending(10, 1));  // gate 10 was NEVER signaled!
        holds.held_changed(-1);

        // In-transit lookup_relay() when apply_wait_relay finds nothing yet (race ordering): relay_wait() returns false
        // (nothing substituted yet) while keeping the redirect registered for lookup_relay() and consume pruning.
        const Wait single_rep[] = {{20, 4, {}}};
        CHECK(!holds.relay_wait(10, 2, single_rep, /*consume_on_substitute=*/true));
        CHECK(holds.active_relay_count() == 1);  // retained for in-transit waiter
        CHECK(!holds.relay_wait(10, 2, single_rep, /*consume_on_substitute=*/true));  // duplicate rejected cleanly
        CHECK(holds.active_relay_count() == 1);
        std::vector<Wait> found;
        CHECK(holds.lookup_relay(10, 2, found));
        CHECK(found.size() == 1 && found[0].semaphore == 20 && found[0].value == 4);
        CHECK(holds.active_relay_count() == 0);  // consumed and pruned!
        CHECK(!holds.lookup_relay(10, 2, found));

        // Non-consuming relays (with no matching waiter yet -> returns false, still registered) prune cleanly when
        // gate is later signaled past gate_val or replacement fence is removed.
        CHECK(!holds.relay_wait(10, 3, single_rep, /*consume_on_substitute=*/false));
        CHECK(!holds.relay_wait(10, 7, single_rep, /*consume_on_substitute=*/false));
        CHECK(holds.active_relay_count() == 2);
        holds.signaled(10, 5, 0);
        CHECK(holds.active_relay_count() == 1);  // val 3 pruned, val 7 kept
        holds.remove_fence(20);
        CHECK(holds.active_relay_count() == 0);  // replacement fence removed -> pruned

        holds.detach(&holder);
    }

    std::printf("list_value_progress_test: all checks passed\n");
    return 0;
}
