// Portable CPU-only tests for runtime/src/list_value_progress.h: when a held console memory wait may go without its
// value. Nothing can make progress first; then a cycle goes at once, a root (a wait no held write meets) goes at a
// present queued after it or when the title has stopped calling, and a gate whose writer is held behind another
// queue's root waits for that root. No time alone ever releases a gate. No Windows, D3D12 or Vulkan headers.
#include "../src/list_value_progress.h"

#include <cstdio>
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

    std::printf("list_value_progress_test: all checks passed\n");
    return 0;
}
