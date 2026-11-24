// Two tables of the bindless set's bookkeeping in DeviceState (runtime/shims/d3d12_vulkan_device.cpp), kept here
// so that a CPU-only test can check them against the linear scans they replace (runtime/test/descriptor_tables_test.cpp).
//
// BorderColorTable: the custom sampler border colours an S# names by index. CreateSampler (and every static
// sampler of a root signature) with a colour other than the three named ones looks its colour up here. It used to
// be a linear scan of up to 4000 entries under descriptor_mutex. Past kScanMax entries a hash index answers instead
// (O(1)); below it the scan stays, as it is faster on a handful of colours. Both give the same index: entries are
// never removed or reordered, and a colour is only added when it is not already there, so each colour has exactly
// one index, the one the scan's first match gives.
//
// ReplacedImages: the views a T# had before an unordered-access view of the same bytes replaced them, kept until
// their texture goes. Destroying any texture scanned every replaced view of every texture. Past kScanMax views
// they move into a multimap by the texture's address, and a destruction touches only that texture's. Views of one
// address leave in the order they came either way (std::multimap keeps equal keys in insertion order, and the move
// inserts them in vector order), as std::erase_if over the vector released them.
//
// Standard library only: no Vulkan types, so the test needs neither the headers nor a device.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <unordered_map>
#include <utility>
#include <vector>

namespace leonida {

class BorderColorTable {
public:
    static constexpr std::uint32_t kCapacity = 4000;  // the S#'s border colour index field holds 0 to 4095
    static constexpr std::size_t kScanMax = 16;        // the scan is the faster up to here (descriptor_tables_test)

    struct Entry {
        bool is_int = false;
        std::array<std::uint32_t, 4> bits{};  // the VkClearColorValue's 16 bytes, float or integer
    };

    struct Result {
        std::uint32_t index;
        bool full;  // the colour is new and the table has no room: index is kCapacity - 1
    };

    // The index of the colour (`is_int`, the 16 bytes at `components`), adding it at the end if it is new.
    // Colours are equal when is_int and all 16 bytes are (memcmp, so -0.0 and 0.0 differ, as before).
    Result find_or_add(bool is_int, const void* components) {
        Entry entry{};
        entry.is_int = is_int;
        std::memcpy(entry.bits.data(), components, sizeof(entry.bits));
        if (index_.empty()) {  // no more than kScanMax entries
            for (std::size_t i = 0; i < entries_.size(); ++i)
                if (Equal{}(entries_[i], entry)) return {static_cast<std::uint32_t>(i), false};
        } else {
            const auto found = index_.find(entry);
            if (found != index_.end()) return {found->second, false};
        }
        if (entries_.size() >= kCapacity) return {kCapacity - 1, true};
        const auto index = static_cast<std::uint32_t>(entries_.size());
        entries_.push_back(entry);
        if (!index_.empty()) {
            index_.emplace(entry, index);
        } else if (entries_.size() > kScanMax) {
            index_.reserve(entries_.size() * 2);
            for (std::size_t i = 0; i < entries_.size(); ++i)
                index_.emplace(entries_[i], static_cast<std::uint32_t>(i));
        }
        return {index, false};
    }

    bool hashed() const { return !index_.empty(); }

    std::size_t size() const { return entries_.size(); }
    const Entry& operator[](std::size_t index) const { return entries_[index]; }

private:
    struct Hash {
        std::size_t operator()(const Entry& e) const {
            // FNV-1a over the flag and the four words.
            std::uint64_t h = 0xcbf29ce484222325ull ^ static_cast<std::uint64_t>(e.is_int);
            h *= 0x100000001b3ull;
            for (std::uint32_t w : e.bits) {
                h ^= w;
                h *= 0x100000001b3ull;
            }
            return static_cast<std::size_t>(h ^ (h >> 32));
        }
    };
    struct Equal {
        bool operator()(const Entry& a, const Entry& b) const {
            return a.is_int == b.is_int && std::memcmp(a.bits.data(), b.bits.data(), sizeof(a.bits)) == 0;
        }
    };
    std::vector<Entry> entries_;
    std::unordered_map<Entry, std::uint32_t, Hash, Equal> index_;
};

template <class Entry>
class ReplacedImages {
public:
    static constexpr std::size_t kScanMax = 384;  // the vector is the faster up to here (descriptor_tables_test)

    void add(std::uint64_t address, const Entry& entry) {
        if (!by_address_.empty() || list_.size() >= kScanMax) {
            for (const auto& [a, e] : list_) by_address_.emplace(a, e);  // in order: equal keys keep it
            list_.clear();
            list_.shrink_to_fit();
            by_address_.emplace(address, entry);
        } else {
            list_.push_back({address, entry});
        }
    }

    // Calls release(entry) on each entry of `address`, oldest first, and drops them.
    template <class Release>
    void take(std::uint64_t address, Release&& release) {
        if (!list_.empty()) {
            std::erase_if(list_, [&](const std::pair<std::uint64_t, Entry>& item) {
                if (item.first != address) return false;
                release(item.second);
                return true;
            });
            return;
        }
        const auto [first, last] = by_address_.equal_range(address);
        for (auto it = first; it != last; ++it) release(it->second);
        by_address_.erase(first, last);
    }

    std::size_t size() const { return list_.size() + by_address_.size(); }
    bool keyed() const { return !by_address_.empty(); }

private:
    // At most one of the two holds entries.
    std::vector<std::pair<std::uint64_t, Entry>> list_;
    std::multimap<std::uint64_t, Entry> by_address_;
};

}  // namespace leonida
