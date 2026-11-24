// Module base/end ranges for diagnostics, published by normal code and read without a lock by a crash handler.
//
// The crash log cannot use shims::module_offset: it takes the shim registry's lookup lock and calls into the Loader,
// so a fault while either lock is held would deadlock the handler instead of letting the process terminate. This
// header keeps a small table of image ranges instead. Writers (a Loader after each map, install() for the host
// modules) publish under an ordinary mutex; readers see a per-slot seqlock, so finding an address is a handful of
// atomic loads and plain reads of fixed buffers: no mutex, no Loader call, no allocation.
//
// There are two tables:
//   * mapped images, replaced by the Loader that mapped them (gta6rt has one; a second concurrent Loader would
//     replace the first one's table, which is why the owner token exists for the clear path);
//   * host modules, enumerated once when install() runs, before the handler is enabled. A host DLL loaded later
//     (the game's LoadLibrary, the Vulkan loader) is not in the table; the handler falls back to the raw address.
//
// Names are copied into fixed slots, so a Loader's later destruction cannot leave the handler reading freed memory.
#pragma once

#include <atomic>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace rt::module_ranges {

inline constexpr int kMaxModules = 256;
inline constexpr size_t kNameBytes = 96;

struct Entry {
    uint64_t base = 0;
    uint64_t size = 0;
    std::string name;
};

namespace detail {

struct Slot {
    std::atomic<uint32_t> sequence{0};  // odd while a writer owns the slot
    std::atomic<uint64_t> base{0};
    std::atomic<uint64_t> size{0};
    char name[kNameBytes] = {};
};

inline Slot loader_slots[kMaxModules];
inline Slot host_slots[kMaxModules];
inline std::atomic<int> loader_count{0};
inline std::atomic<int> host_count{0};
inline const void* loader_owner = nullptr;  // under writers()

inline std::mutex& writers() {
    static auto* m = new std::mutex;
    return *m;
}

inline void copy_name(char* to, const std::string& from) {
    const size_t n = from.size() < kNameBytes - 1 ? from.size() : kNameBytes - 1;
    if (n) std::memcpy(to, from.data(), n);
    to[n] = '\0';
}

inline void retire(Slot& slot) {
    const uint32_t sequence = slot.sequence.load(std::memory_order_relaxed);
    slot.sequence.store(sequence + 1, std::memory_order_release);
    slot.base.store(0, std::memory_order_relaxed);
    slot.sequence.store(sequence + 2, std::memory_order_release);
}

// Caller holds writers().
inline void write_table(Slot* slots, std::atomic<int>& count, const std::vector<Entry>& entries) {
    int n = static_cast<int>(entries.size());
    if (n > kMaxModules) n = kMaxModules;
    const int old = count.load(std::memory_order_relaxed);
    for (int i = 0; i < n; ++i) {
        Slot& slot = slots[i];
        const uint32_t sequence = slot.sequence.load(std::memory_order_relaxed);
        slot.sequence.store(sequence + 1, std::memory_order_release);  // odd: being written
        slot.base.store(0, std::memory_order_relaxed);
        copy_name(slot.name, entries[i].name);
        slot.size.store(entries[i].size, std::memory_order_relaxed);
        slot.base.store(entries[i].base, std::memory_order_relaxed);
        slot.sequence.store(sequence + 2, std::memory_order_release);
    }
    // A reader can hold a stale, larger count for one load: clear the slots it may still reach.
    for (int i = n; i < old && i < kMaxModules; ++i) retire(slots[i]);
    count.store(n, std::memory_order_release);
}

inline bool find_in(const Slot* slots, const std::atomic<int>& count, uint64_t address, char* name,
                    size_t name_size, uint32_t* rva) {
    const int n = count.load(std::memory_order_acquire);
    for (int i = 0; i < n && i < kMaxModules; ++i) {
        const Slot& slot = slots[i];
        uint64_t base = 0, size = 0;
        char candidate[kNameBytes] = {};
        bool stable = false;
        for (int attempt = 0; attempt < 4 && !stable; ++attempt) {
            const uint32_t before = slot.sequence.load(std::memory_order_acquire);
            if (before & 1u) continue;  // a writer owns the slot
            base = slot.base.load(std::memory_order_relaxed);
            size = slot.size.load(std::memory_order_relaxed);
            std::memcpy(candidate, slot.name, kNameBytes);
            const uint32_t after = slot.sequence.load(std::memory_order_acquire);
            stable = before == after;
        }
        if (!stable || !base || address < base || address - base >= size) continue;
        const size_t bytes = name_size ? name_size - 1 : 0;
        if (bytes) {
            std::memcpy(name, candidate, bytes);
            name[bytes] = '\0';
        }
        *rva = static_cast<uint32_t>(address - base);
        return true;
    }
    return false;
}

}  // namespace detail

// Host images seen at install(). Normal code path: may lock and allocate.
inline void publish_host(const std::vector<Entry>& entries) {
    const std::lock_guard lock(detail::writers());
    detail::write_table(detail::host_slots, detail::host_count, entries);
}

// The Loader's mapped images, replacing its previous table (or another Loader's). Normal code path.
inline void publish_loader(const void* owner, const std::vector<Entry>& entries) {
    const std::lock_guard lock(detail::writers());
    detail::loader_owner = owner;
    detail::write_table(detail::loader_slots, detail::loader_count, entries);
}

// Called by ~Loader before the mappings are freed; only the owner that published the table can clear it.
inline void clear_loader(const void* owner) {
    const std::lock_guard lock(detail::writers());
    if (detail::loader_owner != owner) return;
    detail::loader_owner = nullptr;
    detail::write_table(detail::loader_slots, detail::loader_count, {});
}

// Lock-free: atomic loads and reads of the published buffers only. False (and *rva untouched) for an address no
// published range contains. `*host` is set when the match came from the host-image table.
inline bool find(uint64_t address, char* name, size_t name_size, uint32_t* rva, bool* host = nullptr) {
    if (detail::find_in(detail::loader_slots, detail::loader_count, address, name, name_size, rva)) {
        if (host) *host = false;
        return true;
    }
    if (!detail::find_in(detail::host_slots, detail::host_count, address, name, name_size, rva)) return false;
    if (host) *host = true;
    return true;
}

}  // namespace rt::module_ranges
