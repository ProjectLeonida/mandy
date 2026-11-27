#pragma once
// The page mirror (title_mirror): one GPU copy of the title's GPU-readable pages, at heap_base + the title's address.
//
// A V# or a 64-bit pointer the title writes into its own memory names a title address, and the translated shaders reach
// heap_base + sext48(address) (docs/design.md, "Heap-relative buffer device addresses"). The mirror makes that address
// hold the title's bytes:
//   - the address range: sparse buffers ("chunks", kChunk each) created at heap_base + the title's address with
//     bufferDeviceAddressCaptureReplay, over every reservation xmem placed in src/title_mirror_space.h. A page with
//     nothing bound reads zero and drops stores (residencyNonResidentStrict), so a title address in a reservation that
//     is not mirrored, or a page not mapped, never faults;
//   - one home per physical page: 64 KiB of a 32 MiB unit (one VkDeviceMemory, host-visible, mapped once). Every title
//     page that shows the physical page (a CPU view, a GPU-only alias, a CopyPageMappingsBatchX copy) is bound to that
//     one home, so aliases share the bytes and the GPU's writes, and a remap is a rebind, never a copy;
//   - the change feed: xmem reports every change of what a page of the mirror's range shows (src/title_frames.h,
//     Feed), in order; map() takes it;
//   - the dirty list: a home is filled from its physical page when it is made, and again when the caller says the page
//     was written (dirty(): the geometry watch's cells say so: a guard fault, a kernel write, a remap);
//   - flush(): the binds noted since the last one go in one batch, which signals a value the next submission of every
//     queue waits for (the ordering rule: a submission after a remap in call order never reads a page whose bind has
//     not landed). A home unbound everywhere is freed once the submissions that could read it have retired.
//
// No Vulkan call here: the device (or a test) passes hooks, as for src/arena_backing.h. Not thread-safe: the caller
// serialises (the device under its mirror mutex).
#include <vulkan/vulkan.h>

#include "shadow_keep.h"

#include <cstdint>
#include <cstring>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

namespace leonida {

struct TitleMirrorHooks {
    // A sparse buffer of `bytes` at device address heap_base + title_base; its handle. False: not placed there.
    std::function<bool(std::uint64_t title_base, std::uint64_t bytes, VkBuffer& out)> make_chunk;
    // One unit of home memory, mapped: kUnitBytes of device memory and its host address.
    std::function<bool(VkDeviceMemory& out, std::uint8_t*& host)> make_unit;
    std::function<void(VkDeviceMemory)> free_unit;
    // The binds in one batch (per chunk: VkSparseMemoryBind with resourceOffset inside it); signals `value`.
    std::function<bool(const std::vector<VkSparseBufferMemoryBindInfo>& binds, std::uint64_t value)> bind;
    // Copies a physical page's 64 KiB into its home; false when its bytes cannot be read (the home keeps zeros).
    std::function<bool(std::uint8_t* to, const std::uint8_t* from, std::size_t bytes)> copy;
};

class TitleMirror {
public:
    static constexpr std::uint64_t kPage = 0x10000;
    static constexpr std::uint64_t kChunk = std::uint64_t(2) << 30;
    static constexpr std::uint64_t kUnitBytes = std::uint64_t(32) << 20;
    static constexpr std::uint32_t kUnitPages = std::uint32_t(kUnitBytes / kPage);

    struct Counters {
        std::uint64_t chunks = 0, chunk_failures = 0, homes = 0, units = 0, pages = 0;  // now
        std::uint64_t binds = 0, unbinds = 0, batches = 0, uploads = 0, upload_failures = 0, homes_freed = 0;  // so far
        std::uint64_t unmirrored_changes = 0;  // feed entries outside every chunk
        // Bytes the uploads copied into homes, bytes write_back() copied to the title's pages, and 256-byte pieces
        // both sides had changed (the title's bytes kept), so far.
        std::uint64_t uploaded_bytes = 0, written_back_bytes = 0, conflicts = 0;
    };

    explicit TitleMirror(TitleMirrorHooks hooks) : hooks_(std::move(hooks)) {}
    ~TitleMirror() {
        for (const Unit& u : units_)
            if (u.memory && hooks_.free_unit) hooks_.free_unit(u.memory);
    }
    TitleMirror(const TitleMirror&) = delete;
    TitleMirror& operator=(const TitleMirror&) = delete;

    // The chunks over [base, end) of the title's address space (a reservation xmem placed). False when one could not
    // be placed: its pages stay unmirrored (a title address there reads what heap_base + it names: the caller logs).
    bool cover(std::uint64_t base, std::uint64_t end) {
        bool ok = true;
        for (std::uint64_t c = base / kChunk; c * kChunk < end; ++c) {
            if (chunks_.count(c)) continue;
            VkBuffer buffer = VK_NULL_HANDLE;
            if (!hooks_.make_chunk || !hooks_.make_chunk(c * kChunk, kChunk, buffer)) {
                ++counters_.chunk_failures;
                ok = false;
                chunks_[c] = VK_NULL_HANDLE;  // tried once
                continue;
            }
            chunks_[c] = buffer;
            ++counters_.chunks;
        }
        return ok;
    }
    bool covers(std::uint64_t va) const {
        const auto it = chunks_.find(va / kChunk);
        return it != chunks_.end() && it->second != VK_NULL_HANDLE;
    }

    // The feed: the 64 KiB page at `va` shows the physical page whose 64 KiB are at `frame` (its section's scratch view,
    // kept alive by `keep`), or nothing (`frame` null). The bind is noted for the next flush(); a new home is filled.
    void map(std::uint64_t va, std::uint8_t* frame, std::shared_ptr<void> keep, std::uint64_t ticket) {
        va &= ~(kPage - 1);
        if (!covers(va)) {
            ++counters_.unmirrored_changes;
            return;
        }
        const auto page = pages_.find(va);
        if (page != pages_.end()) {
            if (page->second == frame) return;
            release_home(page->second, ticket);
            pages_.erase(page);
        }
        if (!frame) {
            note_bind(va, VK_NULL_HANDLE, 0);
            ++counters_.unbinds;
            return;
        }
        Home* home = home_of(frame, std::move(keep));
        if (!home) {
            note_bind(va, VK_NULL_HANDLE, 0);
            return;
        }
        ++home->refs;
        pages_[va] = frame;
        note_bind(va, units_[home->unit].memory, VkDeviceSize(home->slot) * kPage);
        ++counters_.binds;
    }

    // The page at `va` was written (its physical page's bytes changed): its home is filled again, once however many
    // pages show it. True when `va` is a mirrored page.
    bool dirty(std::uint64_t va) {
        const auto page = pages_.find(va & ~(kPage - 1));
        if (page == pages_.end()) return false;
        const auto home = homes_.find(page->second);
        if (home == homes_.end()) return false;
        if (!home->second.dirty) {
            home->second.dirty = true;
            dirty_.push_back(page->second);
        }
        return true;
    }
    // Fills the homes dirty() named. Returns how many.
    std::size_t upload() {
        std::size_t n = 0;
        for (std::uint8_t* frame : dirty_) {
            const auto home = homes_.find(frame);
            if (home == homes_.end() || !home->second.dirty) continue;
            home->second.dirty = false;
            fill(home->second, frame, false);
            ++n;
        }
        dirty_.clear();
        return n;
    }

    // What the GPU wrote into the homes of the mirrored pages of [begin, end) goes to the title's pages, merged at the
    // 256-byte piece against the sums both sides last agreed on: a piece only the GPU changed is written back with
    // `write(va, bytes, n)` (and agreed on again); a piece only the title changed waits for the upload; a piece both
    // changed keeps the title's bytes and is counted. Call it once the submissions that could write the range have
    // completed. Returns the bytes written back.
    template <class Write>
    std::uint64_t write_back(std::uint64_t begin, std::uint64_t end, Write&& write) {
        std::uint64_t written = 0;
        alignas(64) std::uint8_t title[kPiece];
        for (std::uint64_t page = begin & ~(kPage - 1); page < end; page += kPage) {
            const auto mapped = pages_.find(page);
            if (mapped == pages_.end()) continue;
            const auto found = homes_.find(mapped->second);
            if (found == homes_.end()) continue;
            Home& home = found->second;
            const std::uint8_t* gpu = units_[home.unit].host + std::size_t(home.slot) * kPage;
            const std::size_t k0 = std::size_t((std::max(begin, page) - page) / kPiece);
            const std::size_t k1 = std::size_t((std::min(end, page + kPage) - page + kPiece - 1) / kPiece);
            for (std::size_t k = k0; k < k1; ++k) {
                const std::uint64_t agreed = home.sums[k];
                const std::uint64_t now = shadow_keep::sum_of(gpu + k * kPiece, kPiece);
                if (now == agreed) continue;  // the GPU did not change it
                const std::uint8_t* from = mapped->second + k * kPiece;
                if (!(hooks_.copy ? hooks_.copy(title, from, kPiece) : (std::memcpy(title, from, kPiece), true)))
                    continue;  // the title's page went away
                if (shadow_keep::sum_of(title, kPiece) != agreed) {
                    ++counters_.conflicts;  // both changed it: the title's bytes win at the next upload
                    continue;
                }
                if (!write(page + k * kPiece, gpu + k * kPiece, kPiece)) continue;
                home.sums[k] = now;
                written += kPiece;
            }
        }
        counters_.written_back_bytes += written;
        return written;
    }

    // The binds noted since the last flush, as one batch signalling `value`. False when there were none (nothing to
    // wait for) or the bind failed (counted; the caller logs).
    bool flush(std::uint64_t value) {
        if (pending_.empty()) return false;
        std::size_t binds = 0;
        for (const auto& [chunk, by_offset] : pending_) binds += by_offset.size();
        last_batch_ = binds;
        // The last bind of a page in this batch is the one that counts: binds of one range in one batch are not
        // ordered by the API.
        std::vector<std::vector<VkSparseMemoryBind>> lists;
        std::vector<VkSparseBufferMemoryBindInfo> infos;
        lists.reserve(pending_.size());
        for (auto& [chunk, by_offset] : pending_) {
            auto& list = lists.emplace_back();
            list.reserve(by_offset.size());
            for (const auto& [offset, bind] : by_offset) list.push_back(bind);
            infos.push_back({chunks_[chunk], std::uint32_t(list.size()), list.data()});
        }
        const bool ok = hooks_.bind && hooks_.bind(infos, value);
        ++counters_.batches;
        pending_.clear();
        return ok;
    }

    // Frees the homes no page binds any more whose last reader has retired (`completed` ticket).
    void retire(std::uint64_t completed) {
        while (!freed_.empty() && freed_.front().ticket <= completed) {
            const auto [unit, slot, ticket] = freed_.front();
            freed_.pop_front();
            units_[unit].free.push_back(slot);
            ++counters_.homes_freed;
        }
    }

    // The home's memory and offset of the mirrored page at `va` (tests and diagnostics).
    bool home_at(std::uint64_t va, VkDeviceMemory& memory, VkDeviceSize& offset, std::uint8_t** host = nullptr) const {
        const auto page = pages_.find(va & ~(kPage - 1));
        if (page == pages_.end()) return false;
        const auto home = homes_.find(page->second);
        if (home == homes_.end()) return false;
        memory = units_[home->second.unit].memory;
        offset = VkDeviceSize(home->second.slot) * kPage;
        if (host) *host = units_[home->second.unit].host + offset;
        return true;
    }
    bool mirrored(std::uint64_t va) const { return pages_.count(va & ~(kPage - 1)) != 0; }
    // The physical page's bytes the mirrored page at `va` shows (its frame; for ordinary memory the page itself).
    std::uint8_t* frame_at(std::uint64_t va) const {
        const auto page = pages_.find(va & ~(kPage - 1));
        return page == pages_.end() ? nullptr : page->second;
    }
    // The chunk over `va` and the offset of `va` in it (tests and diagnostics).
    bool chunk_at(std::uint64_t va, VkBuffer& buffer, VkDeviceSize& offset) const {
        const auto it = chunks_.find(va / kChunk);
        if (it == chunks_.end() || it->second == VK_NULL_HANDLE) return false;
        buffer = it->second;
        offset = va - it->first * kChunk;
        return true;
    }

    Counters counters() const {
        Counters c = counters_;
        c.homes = homes_.size();
        c.units = units_.size();
        c.pages = pages_.size();
        return c;
    }

private:
    // A home and the sum of each of its 256-byte pieces as both sides last agreed on it (src/shadow_keep.h): what was
    // filled from the title's page, or written back to it. A piece whose home differs from its sum was written by the
    // GPU, one whose title page differs by the title.
    static constexpr std::size_t kPiece = shadow_keep::kPiece, kPieces = kPage / kPiece;
    struct Home {
        std::uint32_t unit = 0, slot = 0, refs = 0;
        std::shared_ptr<void> keep;
        bool dirty = false;
        std::unique_ptr<std::uint64_t[]> sums = std::make_unique<std::uint64_t[]>(kPieces);
    };
    struct Unit {
        VkDeviceMemory memory = VK_NULL_HANDLE;
        std::uint8_t* host = nullptr;
        std::vector<std::uint32_t> free;
    };
    struct Freed {
        std::uint32_t unit, slot;
        std::uint64_t ticket;
    };

    // Brings the home up to date with the title's page, piece by piece: `whole` (a new home) copies every piece, and
    // otherwise only a piece the title changed since both sides last agreed, so a piece only the GPU wrote keeps the
    // GPU's bytes (and goes back with write_back). A piece both changed takes the title's bytes.
    void fill(Home& home, const std::uint8_t* frame, bool whole) {
        std::uint8_t* to = units_[home.unit].host + std::size_t(home.slot) * kPage;
        alignas(64) std::uint8_t title[kPiece];
        bool failed = false;
        for (std::size_t k = 0; k < kPieces; ++k) {
            const std::uint8_t* from = frame + k * kPiece;
            if (!(hooks_.copy ? hooks_.copy(title, from, kPiece) : (std::memcpy(title, from, kPiece), true))) {
                std::memset(title, 0, kPiece);
                failed = true;
            }
            const std::uint64_t sum = shadow_keep::sum_of(title, kPiece);
            if (!whole && sum == home.sums[k]) continue;
            std::memcpy(to + k * kPiece, title, kPiece);
            home.sums[k] = sum;
            counters_.uploaded_bytes += kPiece;
        }
        if (failed) ++counters_.upload_failures;
        ++counters_.uploads;
    }

    Home* home_of(std::uint8_t* frame, std::shared_ptr<void> keep) {
        if (const auto it = homes_.find(frame); it != homes_.end()) return &it->second;
        std::uint32_t unit = 0, slot = 0;
        if (!take_slot(unit, slot)) return nullptr;
        Home& home = homes_[frame];
        home.unit = unit;
        home.slot = slot;
        home.keep = std::move(keep);
        fill(home, frame, true);
        return &home;
    }
    bool take_slot(std::uint32_t& unit, std::uint32_t& slot) {
        for (std::uint32_t u = 0; u < units_.size(); ++u)
            if (!units_[u].free.empty()) {
                unit = u;
                slot = units_[u].free.back();
                units_[u].free.pop_back();
                return true;
            }
        Unit made;
        if (!hooks_.make_unit || !hooks_.make_unit(made.memory, made.host)) return false;
        for (std::uint32_t s = kUnitPages; s-- > 0;) made.free.push_back(s);
        units_.push_back(std::move(made));
        unit = std::uint32_t(units_.size() - 1);
        slot = units_.back().free.back();
        units_.back().free.pop_back();
        return true;
    }
    // A page no longer shows `frame`: the home goes once no page shows it, after `ticket` (the submissions that could
    // read it) has retired.
    void release_home(std::uint8_t* frame, std::uint64_t ticket) {
        const auto it = homes_.find(frame);
        if (it == homes_.end() || --it->second.refs) return;
        freed_.push_back({it->second.unit, it->second.slot, ticket});
        homes_.erase(it);
    }
    // One bind (memory null: unbind) of the page at `va`, in every chunk it lies in, into the next batch.
    void note_bind(std::uint64_t va, VkDeviceMemory memory, VkDeviceSize offset) {
        const std::uint64_t chunk = va / kChunk;
        pending_[chunk][va - chunk * kChunk] = {va - chunk * kChunk, kPage, memory, offset, 0};
    }

    TitleMirrorHooks hooks_;
    std::map<std::uint64_t, VkBuffer> chunks_;                 // chunk index -> buffer (null: could not be placed)
    std::unordered_map<std::uint64_t, std::uint8_t*> pages_;   // mirrored page -> its physical page's bytes
    std::unordered_map<std::uint8_t*, Home> homes_;            // physical page (its scratch bytes) -> home
    std::vector<Unit> units_;
    std::vector<std::uint8_t*> dirty_;
    std::deque<Freed> freed_;
    std::map<std::uint64_t, std::map<VkDeviceSize, VkSparseMemoryBind>> pending_;  // chunk -> offset -> next bind
    Counters counters_;
    std::size_t last_batch_ = 0;

public:
    std::size_t last_batch() const { return last_batch_; }  // binds in the last flush()
};

}  // namespace leonida
