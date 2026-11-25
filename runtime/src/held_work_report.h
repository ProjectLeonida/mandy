#pragma once
// Stall reports across layers: who holds work, asked by whoever sees nothing move.
//
// QueueHolds (queue_holds.h) registers each instance here; the console list-value pump (shims/d3d12_xs_list_values.h)
// calls report_held_work() when its counters have stood still while writes are still waiting to land. The two live
// in different layers of one process and know nothing else of each other. Diagnostics only: nothing here changes
// what runs.

#include <cstdio>
#include <mutex>
#include <utility>
#include <vector>

namespace leonida {

using HeldWorkReport = void (*)(const void* owner, std::FILE* out);

struct HeldWorkRegistry {
    std::mutex mutex;
    std::vector<std::pair<const void*, HeldWorkReport>> entries;
};

inline HeldWorkRegistry& held_work_registry() {
    static HeldWorkRegistry* registry = new HeldWorkRegistry;  // immortal: owners may unregister during teardown
    return *registry;
}

inline void register_held_work(const void* owner, HeldWorkReport report) {
    HeldWorkRegistry& registry = held_work_registry();
    std::lock_guard lock(registry.mutex);
    registry.entries.emplace_back(owner, report);
}

// Returns once no report of `owner` is running: an owner unregisters before it is destroyed.
inline void unregister_held_work(const void* owner) {
    HeldWorkRegistry& registry = held_work_registry();
    std::lock_guard lock(registry.mutex);
    for (auto it = registry.entries.begin(); it != registry.entries.end(); ++it)
        if (it->first == owner) {
            registry.entries.erase(it);
            return;
        }
}

// Asks every registered owner to describe what it holds. The registry stays locked meanwhile, so an owner cannot
// go away during its own report; a report must not register or unregister anything.
inline void report_held_work(std::FILE* out) {
    HeldWorkRegistry& registry = held_work_registry();
    std::lock_guard lock(registry.mutex);
    for (const auto& [owner, report] : registry.entries) report(owner, out);
}

}  // namespace leonida
