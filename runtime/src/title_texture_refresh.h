#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace leonida {

using TitleWriteRanges = std::vector<std::pair<std::uintptr_t, std::uintptr_t>>;

// Sorted, disjoint half-open intervals let each texture find its first possible writer by end address.
inline void merge_title_writes(TitleWriteRanges& writes) {
    std::sort(writes.begin(), writes.end());
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
    const auto it = std::upper_bound(writes.begin(), writes.end(), begin,
                                     [](auto address, const auto& range) { return address < range.second; });
    return begin < end && it != writes.end() && it->first < end;
}

// Each digest describes a chunk of the last successful upload, never a subsequently observed CPU snapshot.
class TitleTextureChunkHashes {
public:
    static constexpr std::size_t chunk_bytes = 256u << 10;

    struct Chunk {
        std::size_t offset, bytes;
        std::uint64_t hash;
    };

    bool ready() const { return !hashes_.empty(); }

    template<class Hash>
    void snapshot(std::size_t bytes, Hash hash) {
        bytes_ = bytes;
        next_ = 0;
        hashes_.resize(bytes / chunk_bytes + (bytes % chunk_bytes != 0));
        for (std::size_t i = 0; i < hashes_.size(); ++i) {
            const auto offset = i * chunk_bytes;
            hashes_[i] = hash(offset, std::min(chunk_bytes, bytes - offset));
        }
    }

    // The caller checks ready() first. A failed read still advances so later chunks get their turn.
    Chunk next() {
        const auto index = next_;
        next_ = index + 1 == hashes_.size() ? 0 : index + 1;
        const auto offset = index * chunk_bytes;
        return {offset, std::min(chunk_bytes, bytes_ - offset), hashes_[index]};
    }

private:
    std::vector<std::uint64_t> hashes_;
    std::size_t bytes_ = 0, next_ = 0;
};

// The registry's entries as one pass will visit them, in placement order. A pass cannot walk the registry itself: a
// fill it makes can submit, a submission destroys a retired image, and that image's destructor erases its entry
// (DeviceState::title_texture_gone). Windows' mutex is recursive for its owner, so the erase lands inside the pass and
// the walk would step through a freed node. The list holds a reference to each entry, so an entry erased during the
// pass is still alive when its turn comes.
template<class Registry>
std::vector<std::pair<std::uint64_t, typename Registry::mapped_type>> title_texture_snapshot(const Registry& textures) {
    std::vector<std::pair<std::uint64_t, typename Registry::mapped_type>> entries;
    entries.reserve(textures.size());
    for (const auto& [address, texture] : textures) entries.emplace_back(address, texture);
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
        cursor = address + 1;
        if (!texture->mapped && !texture->stale) visit(*texture);
        if (exhausted()) break;  // advance at least one registry entry even if a deadline already passed
        index = index + 1 == entries.size() ? 0 : index + 1;
    }
}

// Visit the marked textures of a snapshot (mapped by the title, or known stale) under a time limit
// (LEONIDA_TITLE_EVERY_BUDGET_US). Those `urgent` answers true for (a mapped texture never filled yet) are all
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
