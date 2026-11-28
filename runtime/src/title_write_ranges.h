// The title-memory ranges a submission reached, kept for the copy back after it (d3d12_vulkan_device.cpp,
// DeviceState::note_title_write and write_back_titles).
//
// The map holds disjoint ranges, start -> end. A new range joins every range it overlaps or touches, so the map stays
// disjoint, no byte of the new range is dropped, and `bytes` stays the size of the union (what the per-submission byte
// cap compares). Kept apart from DeviceState so the merge can be tested without a device (title_write_ranges_test).
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <map>

namespace leonida {

// Adds [start, end) to `ranges`. False when the caps refuse it: a range that adds a new entry past `max_ranges`, or
// adds bytes once `bytes` has reached `max_bytes`. A range already covered always fits.
inline bool add_reached_range(std::map<std::uint64_t, std::uint64_t>& ranges, std::uint64_t& bytes, std::uint64_t start,
                              std::uint64_t end, std::size_t max_ranges, std::uint64_t max_bytes) {
    if (start >= end) return true;
    // The first range that overlaps or touches [start, end): the one before `start`, if it reaches it.
    auto first = ranges.upper_bound(start);
    if (first != ranges.begin() && std::prev(first)->second >= start) --first;
    std::uint64_t lo = start, hi = end, covered = 0;
    std::size_t joined = 0;
    auto last = first;
    for (; last != ranges.end() && last->first <= end; ++last, ++joined) {
        lo = std::min(lo, last->first);
        hi = std::max(hi, last->second);
        covered += last->second - last->first;
    }
    const std::uint64_t added = (hi - lo) - covered;
    if ((!joined && ranges.size() >= max_ranges) || (added && bytes >= max_bytes)) return false;
    ranges.erase(first, last);
    ranges.emplace(lo, hi);
    bytes += added;
    return true;
}

}  // namespace leonida
