#pragma once
#include "shadow_keep.h"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <deque>
#include <map>
#include <utility>
#include <vector>

namespace leonida {

using TitleWriteRanges = std::vector<std::pair<std::uintptr_t, std::uintptr_t>>;

// Computes the exact 64-bit piece-sum digest matching TitleMirror::range_sum(va, bytes, out) whenever
// `(va & (shadow_keep::kPiece - 1)) == 0 && (bytes & (shadow_keep::kPiece - 1)) == 0` (256-byte aligned, true for
// virtually all console textures and every 64 KiB tile boundary). Also works for arbitrary byte lengths (hashing any
// trailing partial piece with shadow_keep::sum_of) for fast 4-lane CPU piece hashing (~18 GB/s vs ~3.6 GB/s).
inline std::uint64_t mirror_range_sum_bytes(const std::uint8_t* data, std::uint64_t bytes) {
    if (!bytes) return 0;
    constexpr std::uint64_t kPiece = shadow_keep::kPiece;
    std::uint64_t h = 14695981039346656037ull ^ bytes;
    for (std::uint64_t off = 0; off < bytes; off += kPiece) {
        const std::size_t n = static_cast<std::size_t>((std::min)(kPiece, bytes - off));
        const std::uint64_t s = shadow_keep::sum_of(data + off, n);
        h = (h ^ s) * 1099511628211ull;
        h ^= h >> 29;
    }
    return h;
}

// Same digest built from precomputed 256-byte piece sums (avoids a second pass over texture bytes when both per-tile
// TilePieces hashes and the full-texture mirror_hash are needed).
inline std::uint64_t mirror_range_sum_from_pieces(const std::uint64_t* piece_sums, std::size_t piece_count,
                                                  std::uint64_t bytes) {
    if (!bytes || !piece_count) return 0;
    std::uint64_t h = 14695981039346656037ull ^ bytes;
    for (std::size_t i = 0; i < piece_count; ++i) {
        h = (h ^ piece_sums[i]) * 1099511628211ull;
        h ^= h >> 29;
    }
    return h;
}

// Per-entry retire-watermarked passive write-back set for console-heap V# ranges awaiting a `writes_buffer == true`
// submission. Replaces the coarse `in_flight_executes > 1` non-clearing guard:
//   - Every noted [begin, end) is stamped with a monotonic sequence number `seq` and `retire_id = 0` (unpromoted).
//   - Overlapping older ranges are split around [begin, end) so a fresh small note never extends an older range's seq
//     or clears its retire watermark, and adjacent ranges are kept separate so stale ranges never snowball.
//   - `begin_execute()` allocates a monotonic `exec_id`; `end_execute(id)` removes it and prunes any already-promoted
//     entry (`retire_id != 0`) whose `retire_id` is strictly below the oldest remaining active execute (or all
//     already-promoted entries when no execute remains active).
//   - `promote(my_exec_id, emit)` emits entries where `retire_id == 0 || my_exec_id == 0 || my_exec_id <= retire_id`:
//       * An unpromoted entry (`retire_id == 0`) gets `retire_id = max_other_active_id` (or is erased immediately when
//         no other execute is in flight), guaranteeing every concurrent queue already in flight (even one that started
//         before or after `note()`) still promotes it at its own later ticket while newer queues ignore it.
//       * Any promoted entry with no remaining other active execute having `id <= retire_id` is erased immediately.
class PassiveWriteBackTracker {
public:
    static constexpr std::size_t kDefaultMaxRanges = 8192;
    static constexpr std::uint64_t kDefaultMaxBytes = std::uint64_t(64) << 20;

    explicit PassiveWriteBackTracker(std::size_t max_ranges = kDefaultMaxRanges,
                                     std::uint64_t max_bytes = kDefaultMaxBytes)
        : max_ranges_(max_ranges), max_bytes_(max_bytes) {}

    bool empty() const noexcept { return ranges_.empty(); }
    std::size_t size() const noexcept { return ranges_.size(); }
    std::uint64_t bytes() const noexcept { return total_bytes_; }
    std::size_t active_executes() const noexcept { return active_executes_.size(); }
    std::uint64_t current_seq() const noexcept { return next_seq_; }

    std::uint64_t begin_execute() {
        const std::uint64_t id = ++next_exec_id_;
        active_executes_.push_back(id);
        return id;
    }

    void end_execute(std::uint64_t id) {
        if (!id) return;
        for (std::size_t i = 0; i < active_executes_.size(); ++i) {
            if (active_executes_[i] == id) {
                active_executes_[i] = active_executes_.back();
                active_executes_.pop_back();
                break;
            }
        }
        std::uint64_t min_active_id = UINT64_MAX;
        for (const std::uint64_t active_id : active_executes_) {
            if (active_id < min_active_id) min_active_id = active_id;
        }
        for (auto it = ranges_.begin(); it != ranges_.end();) {
            if (it->second.retire_id != 0 && it->second.retire_id < min_active_id) {
                const std::uint64_t span = it->second.end - it->first;
                total_bytes_ = total_bytes_ >= span ? total_bytes_ - span : 0;
                it = ranges_.erase(it);
            } else {
                ++it;
            }
        }
        if (ranges_.empty()) order_.clear();
    }

    void note(std::uint64_t begin, std::uint64_t end) {
        if (begin >= end) return;
        const std::uint64_t seq = ++next_seq_;

        // Find all strictly overlapping intervals [scan->first, scan->second.end).
        // Adjacent intervals (scan->second.end == begin or scan->first == end) are intentionally NOT merged so a new
        // note never extends an older interval's lifetime or resets its retire_id.
        auto it = ranges_.lower_bound(begin);
        if (it != ranges_.begin() && std::prev(it)->second.end > begin) --it;

        // Fast-path exact match: refresh seq and reset retire_id = 0 (unpromoted) without mutating map nodes.
        if (it != ranges_.end() && it->first == begin && it->second.end == end) {
            it->second.seq = seq;
            it->second.retire_id = 0;
            order_.emplace_back(seq, begin);
            trim_order();
            return;
        }

        bool has_left = false;
        std::uint64_t left_begin = 0, left_seq = 0, left_retire = 0;
        bool has_right = false;
        std::uint64_t right_end = 0, right_seq = 0, right_retire = 0;
        std::uint64_t removed_bytes = 0;

        for (auto scan = it; scan != ranges_.end() && scan->first < end;) {
            if (scan->first < begin) {
                has_left = true;
                left_begin = scan->first;
                left_seq = scan->second.seq;
                left_retire = scan->second.retire_id;
            }
            if (scan->second.end > end) {
                has_right = true;
                right_end = scan->second.end;
                right_seq = scan->second.seq;
                right_retire = scan->second.retire_id;
            }
            removed_bytes += scan->second.end - scan->first;
            scan = ranges_.erase(scan);
        }
        total_bytes_ = total_bytes_ >= removed_bytes ? total_bytes_ - removed_bytes : 0;

        if (has_left) {
            ranges_.emplace(left_begin, Entry{begin, left_seq, left_retire});
            total_bytes_ += begin - left_begin;
        }
        if (has_right) {
            ranges_.emplace(end, Entry{right_end, right_seq, right_retire});
            total_bytes_ += right_end - end;
        }

        const std::uint64_t new_bytes = end - begin;
        evict_to_fit(new_bytes);

        ranges_.emplace(begin, Entry{end, seq, 0});
        total_bytes_ += new_bytes;
        order_.emplace_back(seq, begin);
        trim_order();
    }

    // Emits entries relevant to `my_exec_id` (`retire_id == 0 || my_exec_id == 0 || my_exec_id <= retire_id`), stamps
    // newly promoted entries with `max_other_active_id` when concurrent executes are in flight, and immediately erases
    // every entry whose `retire_id` has no remaining other active execute (`min_other_active_id > retire_id`).
    // Returns true if at least one range was emitted.
    template<class Emit>
    bool promote(std::uint64_t my_exec_id, Emit&& emit) {
        if (ranges_.empty()) return false;
        std::uint64_t min_other_active_id = UINT64_MAX;
        std::uint64_t max_other_active_id = 0;
        for (const std::uint64_t active_id : active_executes_) {
            if (active_id == my_exec_id) continue;
            if (active_id < min_other_active_id) min_other_active_id = active_id;
            if (active_id > max_other_active_id) max_other_active_id = active_id;
        }
        bool emitted = false;
        for (auto it = ranges_.begin(); it != ranges_.end();) {
            Entry& entry = it->second;
            if (entry.retire_id == 0) {
                emit(it->first, entry.end);
                emitted = true;
                if (max_other_active_id == 0) {
                    const std::uint64_t span = entry.end - it->first;
                    total_bytes_ = total_bytes_ >= span ? total_bytes_ - span : 0;
                    it = ranges_.erase(it);
                    continue;
                }
                entry.retire_id = max_other_active_id;
            } else if (my_exec_id == 0 || my_exec_id <= entry.retire_id) {
                emit(it->first, entry.end);
                emitted = true;
            }
            if (entry.retire_id != 0 && min_other_active_id > entry.retire_id) {
                const std::uint64_t span = entry.end - it->first;
                total_bytes_ = total_bytes_ >= span ? total_bytes_ - span : 0;
                it = ranges_.erase(it);
            } else {
                ++it;
            }
        }
        if (ranges_.empty()) order_.clear();
        return emitted;
    }

    void clear() {
        ranges_.clear();
        order_.clear();
        total_bytes_ = 0;
    }

private:
    struct Entry {
        std::uint64_t end = 0;
        std::uint64_t seq = 0;
        std::uint64_t retire_id = 0;  // 0 = unpromoted; > 0 = promoted while executes <= retire_id were in flight
    };

    void evict_to_fit(std::uint64_t incoming_bytes) {
        while (!ranges_.empty() &&
               (ranges_.size() >= max_ranges_ || total_bytes_ + incoming_bytes > max_bytes_)) {
            auto victim = ranges_.end();
            while (!order_.empty()) {
                const auto [seq, key] = order_.front();
                order_.pop_front();
                const auto found = ranges_.find(key);
                if (found != ranges_.end() && found->second.seq == seq) {
                    victim = found;
                    break;
                }
            }
            if (victim == ranges_.end()) {
                // Rare fallback (e.g. split right remainder whose key shifted): evict strictly oldest seq.
                victim = ranges_.begin();
                for (auto it = std::next(victim); it != ranges_.end(); ++it) {
                    if (it->second.seq < victim->second.seq) victim = it;
                }
            }
            const std::uint64_t victim_bytes = victim->second.end - victim->first;
            total_bytes_ = total_bytes_ >= victim_bytes ? total_bytes_ - victim_bytes : 0;
            ranges_.erase(victim);
        }
    }

    void trim_order() {
        if (order_.size() <= 2 * max_ranges_) return;
        while (!order_.empty()) {
            const auto [seq, key] = order_.front();
            const auto found = ranges_.find(key);
            if (found != ranges_.end() && found->second.seq == seq) break;
            order_.pop_front();
        }
    }

    std::size_t max_ranges_ = kDefaultMaxRanges;
    std::uint64_t max_bytes_ = kDefaultMaxBytes;
    std::map<std::uint64_t, Entry> ranges_;
    std::deque<std::pair<std::uint64_t, std::uint64_t>> order_;
    std::uint64_t total_bytes_ = 0;
    std::uint64_t next_seq_ = 0;
    std::uint64_t next_exec_id_ = 0;
    std::vector<std::uint64_t> active_executes_;
};

// Store each texture once, in the smallest aligned power-of-two page containing its CPU range.
// A journal interval needs only the overlapping pages at each occupied size. Aliases keep distinct keys.
class TitleTextureRangeIndex {
    using Address = std::uint64_t;
    struct Range { Address first, last; };
    using PageKey = std::pair<Address, Address>;  // page, registry key
    std::map<unsigned, std::map<PageKey, Range>> pages_;
    std::map<Address, std::pair<unsigned, Address>> location_;
    static Address page(Address at, unsigned shift) { return shift == 64 ? 0 : at >> shift; }

public:
    void erase(Address key) {
        const auto where = location_.find(key);
        if (where == location_.end()) return;
        const auto [shift, number] = where->second;
        auto level = pages_.find(shift);
        level->second.erase({number, key});
        if (level->second.empty()) pages_.erase(level);
        location_.erase(where);
    }
    void set(Address key, Address first, Address bytes) {
        erase(key);
        if (!bytes) return;
        const Address last = first + std::min(bytes - 1, UINT64_MAX - first);
        const unsigned shift = unsigned(std::bit_width(first ^ last));
        const Address number = page(first, shift);
        pages_[shift][{number, key}] = {first, last};
        location_[key] = {shift, number};
    }
    template<class Visit>
    void overlapping(Address from, Address to, Visit visit) const {
        if (from >= to) return;
        for (const auto& [shift, entries] : pages_) {
            const Address last = page(to - 1, shift);
            for (auto it = entries.lower_bound({page(from, shift), 0});
                 it != entries.end() && it->first.first <= last; ++it) {
                if (it->second.first < to && from <= it->second.last) visit(it->first.second);
            }
        }
    }
};

// Sorted, disjoint half-open intervals let each texture find its first possible writer by end address.
inline void merge_title_writes(TitleWriteRanges& writes) {
    if (writes.empty()) return;
    if (writes.size() == 1) {
        if (writes[0].first >= writes[0].second) writes.clear();
        return;
    }
    if (!std::is_sorted(writes.begin(), writes.end())) std::sort(writes.begin(), writes.end());
    std::size_t count = 0;
    for (const auto [begin, end] : writes) {
        if (begin >= end) continue;
        if (count && begin <= writes[count - 1].second)
            writes[count - 1].second = std::max(writes[count - 1].second, end);
        else
            writes[count++] = {begin, end};
    }
    writes.resize(count);
}

inline bool title_write_overlaps(const TitleWriteRanges& writes, std::uintptr_t begin, std::uintptr_t end) {
    if (begin >= end || writes.empty()) return false;
    if (end <= writes.front().first || begin >= writes.back().second) return false;
    if (writes.size() == 1) return true;
    const auto it = std::upper_bound(writes.begin(), writes.end(), begin,
                                     [](auto address, const auto& range) { return address < range.second; });
    return it != writes.end() && it->first < end;
}

// The registry's entries as one pass will visit them, in placement order. A pass cannot walk the registry itself: a
// fill it makes can submit, a submission destroys a retired image, and that image's destructor erases its entry
// (DeviceState::title_texture_gone). Windows' mutex is recursive for its owner, so the erase lands inside the pass and
// the walk would step through a freed node. The list holds a reference to each entry, so an entry erased during the
// pass is still alive when its turn comes.
template<class Registry>
std::vector<std::pair<std::uint64_t, typename Registry::mapped_type>> title_texture_snapshot(
    const Registry& textures, std::uint64_t start = 0, std::size_t limit = SIZE_MAX) {
    std::vector<std::pair<std::uint64_t, typename Registry::mapped_type>> entries;
    const auto count = std::min(textures.size(), limit);
    entries.reserve(count);
    for (auto it = textures.lower_bound(start); it != textures.end() && entries.size() < count; ++it)
        entries.emplace_back(it->first, it->second);
    return entries;
}

// Poll unmarked textures of a snapshot in placement order. A time or byte limit resumes the next pass at
// the following address. The cursor also survives insertion or removal between passes.
template<class Entry, class Visit, class Exhausted>
void poll_title_textures(const std::vector<Entry>& entries, std::uint64_t& cursor, Visit visit, Exhausted exhausted) {
    if (entries.empty()) return;
    // A snapshot of a std::map comes out sorted by address.
    const auto first = std::lower_bound(entries.begin(), entries.end(), cursor,
                                        [](const Entry& entry, std::uint64_t at) { return entry.first < at; });
    auto index = std::size_t(first - entries.begin());
    if (index == entries.size()) index = 0;  // the cursor is past the last address: resume at the first
    for (auto left = entries.size(); left; --left) {
        const auto& [address, texture] = entries[index];
        cursor = address == UINT64_MAX ? 0u : address + 1u;
        if (!texture->mapped && !texture->stale) visit(*texture);
        if (exhausted()) break;  // advance at least one registry entry even if a deadline already passed
        index = index + 1 == entries.size() ? 0 : index + 1;
    }
}

// Visit the marked textures of a snapshot (mapped by the title, or known stale) under a time limit.
// Those `urgent` answers true for (a mapped texture never filled yet) are all
// visited first, in placement order, whatever the limit. The others are visited
// from `cursor`, as poll_title_textures does for the unmarked ones, until `exhausted()` answers true after a visit;
// the next pass resumes at the entry after the last one visited, so every marked texture gets its turn however many
// there are. An entry skipped as unmarked costs no check.
template<class Entry, class Visit, class Urgent, class Exhausted>
void visit_marked_title_textures(const std::vector<Entry>& entries, std::uint64_t& cursor, Visit visit, Urgent urgent,
                                 Exhausted exhausted) {
    if (entries.empty()) return;
    // Which entries are urgent is decided before any visit: a visit fills the texture, and it must not come round
    // again in the rotation of the same pass.
    std::vector<char> was_urgent(entries.size());
    for (std::size_t i = 0; i < entries.size(); ++i) {
        const auto& texture = entries[i].second;
        was_urgent[i] = (texture->mapped || texture->stale) && urgent(*texture);
    }
    for (std::size_t i = 0; i < entries.size(); ++i)
        if (was_urgent[i]) visit(*entries[i].second);
    const auto first = std::lower_bound(entries.begin(), entries.end(), cursor,
                                        [](const Entry& entry, std::uint64_t at) { return entry.first < at; });
    auto index = std::size_t(first - entries.begin());
    if (index == entries.size()) index = 0;
    for (auto left = entries.size(); left; --left) {
        const auto& [address, texture] = entries[index];
        cursor = address + 1;
        if ((texture->mapped || texture->stale) && !was_urgent[index]) {
            visit(*texture);
            if (exhausted()) break;
        }
        index = index + 1 == entries.size() ? 0 : index + 1;
    }
}

}  // namespace leonida
