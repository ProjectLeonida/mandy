#pragma once
// When a held console memory wait may go without its value (runtime/shims/d3d12_xs_list_values.h,
// docs/d3d12_xs_console_abi.md §5.5).
//
// A queue that stands at a wait for a value no write has met yet is held until the value is in memory: the console's
// command processor would stand there as long, and work that reads what the writer produces must not run before it.
// There is no timeout. A held wait goes without its value only when nothing else can make progress, and then only by
// one of three rules, each about progress, never about how long the wait has stood:
// - cycle: the wait's value is written by work that is itself held behind a wait that depends, through further held
//   writes, on the waiting queue. Nothing modelled can ever meet it.
// - present: the title has presented since the wait was held. A wait is not kept across frames: the present stands
//   behind the held work (src/queue_holds.h, after_held), and the title's next frame would otherwise never be shown.
// - stalled: the title has made no queue call at all for kStalledMs while nothing ran. Its threads wait for something
//   behind the held work (a fence the held part signals, say): a deadlock through the title's CPU, which no write
//   can end. Not a timeout of the wait: one held for a minute while the title keeps calling stays held.
// "Nothing else can make progress" is: the releaser has no gate to signal and is not inside a Signal, nothing landed
// or opened in this pass, no title ExecuteCommandLists is in progress, and no write group or action is running on the
// GPU (its Signal submitted and its fence not reached: work held in the layer, behind a gate, a relayed wait or a
// title fence, is not running).
// The candidate is a queue's first held gate (each queue's gates open in order). With several, the rules go to the
// oldest gate whose wait no held write could meet ("root"); a gate whose writer is held behind another queue's root
// waits for that root, so the chain unwinds in order. Device loss and the end of the process are handled by the pump.
//
// The release rule follows VIctory-Belobug-AMD (GPL-3.0-or-later): crates/victory-gpu/src/queue/behind.rs, `step`
// (a held wait is forced only when a present is in the line and nothing else can run) and
// crates/victory-gpu/src/queue.rs, `release_held` ("a wait is not kept across frames"). Written anew for this
// runtime's queues; no code is copied.
//
// Pure C++20 standard library only, so the policy is tested portably (test/list_value_progress_test.cpp).

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace leonida::list_value_progress {

// The title's queues called nothing for this long while nothing ran: its threads wait for held work.
inline constexpr std::uint64_t kStalledMs = 2000;

struct WaitPredicate {
    std::uint64_t address = 0;
    std::uint64_t reference = 0;
    std::uint32_t comparison = 0;
    bool wide = false;
};

struct PendingWrite {
    std::uint64_t address = 0;
    std::uint64_t value = 0;
    std::uint8_t bytes = 0;
    bool timestamp = false;  // GPU/system timestamp: the value is known only at landing
};

struct PendingWriteGroup {
    const void* queue = nullptr;
    std::uint64_t after_gate = 0;  // highest gate value pending on `queue` when the group was submitted (0: none)
    bool running = false;          // its Signal reached the GPU and its fence is not reached yet
    std::span<const PendingWrite> writes;
};

struct PendingGate {
    const void* queue = nullptr;
    std::uint64_t value = 0;
    bool is_action = false;       // a CPU action at a point of the queue's timeline, not a memory wait
    bool action_running = false;  // action: the work before it is on the GPU and not done yet
    std::span<const WaitPredicate> unmet_waits;
    std::uint64_t presents_at_hold = 0;  // the title's present count when the gate was held
    std::uint64_t held_ms = 0;           // when it was held
};

struct Activity {
    bool releasing = false;         // the releaser has gates to signal or is inside a Signal, or this pass moved
    bool title_submitting = false;  // a title ExecuteCommandLists is in progress
    std::uint64_t presents = 0;     // the title's presents so far
    std::uint64_t now_ms = 0;
    std::uint64_t title_call_ms = 0;  // when the title's latest queue call began or ended
};

enum class Reason : std::uint8_t { hold, cycle, present, stalled };

inline const char* reason_text(Reason reason) {
    switch (reason) {
    case Reason::cycle: return "its writer is held behind a wait on this queue (a cycle)";
    case Reason::present: return "a present is queued behind it and nothing else can run";
    case Reason::stalled: return "the title made no queue call while nothing ran (it waits for held work)";
    default: return "held";
    }
}

struct Decision {
    std::size_t gate = SIZE_MAX;  // index into the gates given to decide()
    Reason reason = Reason::hold;
};

namespace detail {

// True when `group`'s last write touching [wait.address, wait.address + need) lands a value that meets `wait` (or is a
// timestamp written to that address, whose value is not known before it lands).
template <class MetFn>
bool group_meets(const PendingWriteGroup& group, const WaitPredicate& wait, MetFn&& met) {
    const std::uint64_t need = wait.wide ? 8u : 4u;
    if (!wait.address || wait.address > UINT64_MAX - need) return false;
    const std::uint64_t wait_end = wait.address + need;
    for (std::size_t i = group.writes.size(); i-- > 0;) {
        const PendingWrite& w = group.writes[i];
        if (!w.bytes || w.address > UINT64_MAX - w.bytes) continue;
        if (w.address >= wait_end || wait.address >= w.address + w.bytes) continue;
        return w.address == wait.address && w.bytes >= need &&
               (w.timestamp || met(wait.comparison, w.value, wait.reference, wait.wide));
    }
    return false;
}

// True when `gates[from]` waits, directly or through the writers of what it waits for, on a gate of `target`.
template <class MetFn>
bool reaches(std::size_t from, const void* target, std::span<const PendingWriteGroup> writes,
             std::span<const PendingGate> gates, MetFn&& met, std::vector<std::size_t>& visiting) {
    const PendingGate& g = gates[from];
    if (g.queue == target) return true;
    if (g.is_action || std::find(visiting.begin(), visiting.end(), from) != visiting.end()) return false;
    visiting.push_back(from);
    for (const WaitPredicate& wait : g.unmet_waits)
        for (const PendingWriteGroup& w : writes) {
            if (!w.after_gate || !group_meets(w, wait, met)) continue;
            for (std::size_t b = 0; b < gates.size(); ++b)
                if (gates[b].queue == w.queue && gates[b].value <= w.after_gate &&
                    reaches(b, target, writes, gates, met, visiting)) {
                    visiting.pop_back();
                    return true;
                }
        }
    visiting.pop_back();
    return false;
}

}  // namespace detail

// Something can still make progress: what it does may meet a held wait, or show that it cannot be met.
inline bool can_progress(std::span<const PendingWriteGroup> writes, std::span<const PendingGate> gates,
                         const Activity& activity) {
    if (activity.releasing || activity.title_submitting) return true;
    for (const PendingWriteGroup& w : writes)
        if (w.running) return true;
    for (const PendingGate& g : gates)
        if (g.is_action && g.action_running) return true;
    return false;
}

// Which held gate goes now without its value, if any. `gates` are the pending gates in the order they were held
// (each queue's in rising value); `met(comparison, value, reference, wide)` is console_value_met.
template <class MetFn>
Decision decide(std::span<const PendingWriteGroup> writes, std::span<const PendingGate> gates,
                const Activity& activity, MetFn&& met) {
    if (can_progress(writes, gates, activity)) return {};
    std::vector<const void*> seen;
    std::vector<std::size_t> roots, chained, cycles;
    std::vector<std::size_t> visiting;
    for (std::size_t i = 0; i < gates.size(); ++i) {
        const PendingGate& g = gates[i];
        if (std::find(seen.begin(), seen.end(), g.queue) != seen.end()) continue;
        seen.push_back(g.queue);  // only a queue's first gate can open
        if (g.is_action) continue;
        bool matched = false, behind_gate = false, cyclic = false;
        for (const WaitPredicate& wait : g.unmet_waits)
            for (const PendingWriteGroup& w : writes) {
                if (!detail::group_meets(w, wait, met)) continue;
                matched = true;
                if (!w.after_gate) continue;
                for (std::size_t b = 0; b < gates.size() && !cyclic; ++b) {
                    if (gates[b].queue != w.queue || gates[b].value > w.after_gate) continue;
                    behind_gate = true;
                    visiting.clear();
                    cyclic = detail::reaches(b, g.queue, writes, gates, met, visiting);
                }
            }
        if (cyclic)
            cycles.push_back(i);
        else if (matched && behind_gate)
            chained.push_back(i);  // waits for another queue's held gate, which goes first
        else
            roots.push_back(i);  // no held write meets it, or its writer is held behind something not modelled here
    }
    if (!cycles.empty()) return {cycles.front(), Reason::cycle};
    const std::vector<std::size_t>& candidates = roots.empty() ? chained : roots;
    for (std::size_t i : candidates)
        if (activity.presents > gates[i].presents_at_hold) return {i, Reason::present};
    const auto quiet = [&](std::uint64_t since) {
        return activity.now_ms >= since && activity.now_ms - since >= kStalledMs;
    };
    if (quiet(activity.title_call_ms))
        for (std::size_t i : candidates)
            if (quiet(gates[i].held_ms)) return {i, Reason::stalled};
    return {};
}

}  // namespace leonida::list_value_progress
