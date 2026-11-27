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
//     not landed). A home unbound everywhere is freed once the submissions that could read it have retired;
//   - aliases (alias()): a range of title pages also bound at an address of the caller's (a console descriptor heap's
//     table address below 4 GiB), each page there with a home of its own, the filtered copy, which hooks.filter fills
//     from the title's bytes whenever they change. The page's own home stays the title's bytes exactly: a V# or a
//     pointer of the title into those pages reads what the title wrote, and only the alias reads filtered bytes.
//
// No Vulkan call here: the device (or a test) passes hooks, as for src/arena_backing.h. Not thread-safe: the caller
// serialises (the device under its mirror mutex).
#include <vulkan/vulkan.h>

#include "shadow_keep.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <unordered_map>
#include <utility>
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
    // Fills the filtered copy of a page in an aliased range (alias): the title's bytes at `from` (the page shown at
    // title address `va`) in, what the GPU reads at the alias into `to`. Called after the page's own home took the
    // title's bytes, when they changed. `fresh`: the copy holds nothing of an earlier fill (made now, or a failed fill
    // zeroed it), so every byte must be written; otherwise `to` still holds this filter's last output for the same page.
    // The title's bytes and the page's own home are never written. False as for `copy` (the copy is zeroed).
    std::function<bool(std::uint64_t va, std::uint8_t* to, const std::uint8_t* from, std::size_t bytes, bool fresh)> filter;
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
        std::uint64_t filtered = 0;            // fills of an alias's filtered copy through hooks.filter
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
            // A filtered copy made for this page: another aliased page showing the same home takes it over (filled
            // whole for that page), else it goes with this page.
            if (const auto shown = homes_.find(page->second);
                shown != homes_.end() && shown->second.copy && shown->second.filter_va == va) {
                std::uint64_t other_va = 0;
                for (const auto& [other, other_frame] : pages_)
                    if (other != va && other_frame == page->second && aliased(other)) other_va = other;
                if (other_va) {
                    shown->second.filter_va = other_va;
                    fill_copy(shown->second, page->second, true);
                } else {
                    drop_copy(shown->second, ticket);
                }
            }
            release_home(page->second, ticket);
            pages_.erase(page);
        }
        if (!frame) {
            note_bind(va, VK_NULL_HANDLE, 0);
            note_alias_bind(va, VK_NULL_HANDLE, 0);
            ++counters_.unbinds;
            return;
        }
        Home* home = home_of(frame, std::move(keep), va);
        if (!home) {
            note_bind(va, VK_NULL_HANDLE, 0);
            note_alias_bind(va, VK_NULL_HANDLE, 0);
            return;
        }
        ++home->refs;
        pages_[va] = frame;
        note_bind(va, units_[home->unit].memory, VkDeviceSize(home->slot) * kPage);
        if (aliased(va)) {
            if (!home->copy) make_copy(*home, frame, va);
            if (home->copy) note_alias_bind(va, units_[home->copy_unit].memory, VkDeviceSize(home->copy_slot) * kPage);
            else note_alias_bind(va, VK_NULL_HANDLE, 0);  // no room: the alias reads zero there
        }
        ++counters_.binds;
    }

    // [lo, hi) of the title's address space (whole pages) is also bound into `buffer` at `offset` + (va - lo), each page
    // there with a home of its own, the filtered copy: hooks.filter fills it from the title's bytes now and whenever the
    // page is filled again, it follows the page's remaps and unmaps, and its binds go in the same batches as the chunk
    // binds, so the same wait orders a reader of either address. The page's own home, at the title's address, keeps
    // the title's bytes. A console descriptor heap uses it to be read filtered at an offset below 4 GiB of the arena
    // (32-bit descriptor-table pointers). Pages not mapped are left unbound there (they read zero).
    void alias(std::uint64_t lo, std::uint64_t hi, VkBuffer buffer, VkDeviceSize offset) {
        aliases_[lo] = {hi, buffer, offset};
        for (const auto& [va, frame] : pages_) {
            if (va < lo || va >= hi) continue;
            const auto home = homes_.find(frame);
            if (home == homes_.end()) continue;
            if (!home->second.copy) make_copy(home->second, frame, va);
            if (home->second.copy)
                note_alias_bind(va, units_[home->second.copy_unit].memory, VkDeviceSize(home->second.copy_slot) * kPage);
        }
    }
    // The alias at `lo` goes: its pages are unbound there, and their filtered copies are freed after `ticket`.
    void unalias(std::uint64_t lo, std::uint64_t ticket) {
        const auto it = aliases_.find(lo);
        if (it == aliases_.end()) return;
        const Alias alias = it->second;
        for (const auto& [va, frame] : pages_)
            if (va >= lo && va < alias.hi) {
                alias_pending_[alias.buffer][alias.offset + (va - lo)] =
                    {alias.offset + (va - lo), kPage, VK_NULL_HANDLE, 0, 0};
                const auto home = homes_.find(frame);
                if (home != homes_.end() && home->second.copy && home->second.filter_va == va)
                    drop_copy(home->second, ticket);
            }
        aliases_.erase(it);
    }
    // The filtered copy of the aliased page at `va`, mapped (its host bytes): what a filter's caller writes one
    // changed slot into between fills. Null when the page is not mapped or has no copy.
    std::uint8_t* copy_bytes(std::uint64_t va) {
        const auto page = pages_.find(va & ~(kPage - 1));
        if (page == pages_.end()) return nullptr;
        const auto home = homes_.find(page->second);
        if (home == homes_.end() || !home->second.copy) return nullptr;
        return units_[home->second.copy_unit].host + std::size_t(home->second.copy_slot) * kPage;
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

    // A mirrored page snapshotted for GPU write-back readback: carries its chunk location and the physical frame it
    // showed at snapshot time, so the caller can release the mirror mutex during the GPU copy and write_back_batch()
    // can skip any page remapped or given a pending bind meanwhile.
    struct HomeRef {
        std::uint64_t page = 0;
        VkBuffer chunk = VK_NULL_HANDLE;
        VkDeviceSize offset = 0;
        std::uint8_t* frame = nullptr;
    };

    // True when the 64 KiB page at `va` has an unflushed bind/unbind in pending_: the sparse chunk on the GPU does not
    // yet reflect pages_[va], so reading back from chunk_at(va) would read the previous binding (or zeros).
    bool has_pending(std::uint64_t va) const {
        const std::uint64_t page = va & ~(kPage - 1);
        const std::uint64_t chunk = page / kChunk;
        const auto it = pending_.find(chunk);
        return it != pending_.end() && it->second.count(page - chunk * kChunk) != 0;
    }

    // The mirrored pages of [begin, end) that have a flushed home, in address order: what write_back reads. Pages with
    // an unflushed bind in pending_ are skipped so the caller never copies a stale chunk binding.
    void homes_in(std::uint64_t begin, std::uint64_t end, std::vector<std::uint64_t>& out) const {
        if (begin >= end) return;
        for (std::uint64_t page = begin & ~(kPage - 1); page < end; page += kPage) {
            if (!has_pending(page)) {
                const auto mapped = pages_.find(page);
                if (mapped != pages_.end() && homes_.count(mapped->second)) out.push_back(page);
            }
            if (page > UINT64_MAX - kPage) break;
        }
    }

    // Deduplicated, page-sorted HomeRefs for every flushed mirrored page overlapping any [begin, end) in `ranges`.
    std::vector<HomeRef> homes_in_ranges(const std::vector<std::pair<std::uint64_t, std::uint64_t>>& ranges) const {
        std::vector<HomeRef> out;
        if (ranges.empty()) return out;
        const auto merged = merge_ranges(ranges);
        for (const auto& [begin, end] : merged) {
            for (std::uint64_t page = begin & ~(kPage - 1); page < end; page += kPage) {
                if ((out.empty() || page > out.back().page) && !has_pending(page)) {
                    const auto mapped = pages_.find(page);
                    VkBuffer chunk = VK_NULL_HANDLE;
                    VkDeviceSize offset = 0;
                    if (mapped != pages_.end() && homes_.count(mapped->second) && chunk_at(page, chunk, offset))
                        out.push_back({page, chunk, offset, mapped->second});
                }
                if (page > UINT64_MAX - kPage) break;
            }
        }
        return out;
    }

    // The home of the page at `va`, mapped (its host bytes): the title's bytes as the GPU reads them at the title's
    // address. Null when the page is not mapped.
    std::uint8_t* home_bytes(std::uint64_t va) {
        const auto page = pages_.find(va & ~(kPage - 1));
        if (page == pages_.end()) return nullptr;
        const auto home = homes_.find(page->second);
        if (home == homes_.end()) return nullptr;
        return units_[home->second.unit].host + std::size_t(home->second.slot) * kPage;
    }

    // What the GPU wrote into the homes of the mirrored pages of [begin, end) goes back to each page's physical frame,
    // merged at the 256-byte piece against the sums both sides last agreed on: a piece only the GPU changed is written
    // back with `write(frame_dst, bytes, n)` (and agreed on again); a piece only the title changed waits for the
    // upload; a piece both changed keeps the title's bytes and is counted. `write` receives the physical frame address
    // (`mapped->second + k * kPiece`, identical to the title VA for ordinary memory and the writable scratch view for
    // xmem physical pages), so a concurrent CPU remap of `va` never receives the previous mapping's bytes. Pages with
    // an unflushed bind in pending_ are skipped. `staged(page)` gives the page's home as the GPU left it, or nullptr to
    // skip the page. Returns the bytes written back.
    // The same, reading each home where it lies (host memory: the tests' homes; the device copies its homes out first).
    template <class Write>
    std::uint64_t write_back(std::uint64_t begin, std::uint64_t end, Write&& write) {
        return write_back(begin, end, [this](std::uint64_t page) -> const std::uint8_t* { return home_bytes(page); },
                          std::forward<Write>(write));
    }
    template <class Staged, class Write>
    std::uint64_t write_back(std::uint64_t begin, std::uint64_t end, Staged&& staged, Write&& write) {
        if (begin >= end) return 0;
        std::uint64_t written = 0;
        std::vector<std::uint8_t*> refilter_frames;
        for (std::uint64_t page = begin & ~(kPage - 1); page < end; page += kPage) {
            if (!has_pending(page)) {
                const auto mapped = pages_.find(page);
                if (mapped != pages_.end()) {
                    const auto found = homes_.find(mapped->second);
                    if (found != homes_.end()) {
                        if (const std::uint8_t* gpu = staged(page)) {
                            const std::size_t k0 = std::size_t((std::max(begin, page) - page) / kPiece);
                            const std::size_t k1 = std::size_t((std::min(end, page + kPage) - page + kPiece - 1) / kPiece);
                            written += write_back_pieces(found->second, mapped->second, gpu, k0, k1, write,
                                                         refilter_frames);
                        }
                    }
                }
            }
            if (page > UINT64_MAX - kPage) break;
        }
        finish_write_back(written, refilter_frames);
        return written;
    }

    // Batched multi-range write-back over `count` snapshotted HomeRefs whose GPU copies lie contiguously at
    // `staged + i * kPage`. Skips any page that now has a pending bind or was remapped to a different physical frame
    // since `homes_in_ranges()` snapshotted it.
    template <class Write>
    std::uint64_t write_back_batch(const HomeRef* batch, std::size_t count,
                                   const std::vector<std::pair<std::uint64_t, std::uint64_t>>& ranges,
                                   const std::uint8_t* staged, Write&& write) {
        if (!batch || !count || !staged || ranges.empty()) return 0;
        const auto merged = merge_ranges(ranges);
        if (merged.empty()) return 0;
        std::uint64_t written = 0;
        std::vector<std::uint8_t*> refilter_frames;
        std::size_t cursor = 0;
        for (std::size_t i = 0; i < count; ++i) {
            const HomeRef& ref = batch[i];
            if (i > 0 && ref.page < batch[i - 1].page) cursor = 0;
            while (cursor < merged.size() && merged[cursor].second <= ref.page) ++cursor;
            if (cursor >= merged.size()) break;
            if (merged[cursor].first >= ref.page + kPage) continue;
            if (has_pending(ref.page)) continue;
            const auto mapped = pages_.find(ref.page);
            if (mapped == pages_.end() || mapped->second != ref.frame) continue;
            const auto found = homes_.find(mapped->second);
            if (found == homes_.end()) continue;
            const std::uint8_t* gpu = staged + i * kPage;
            std::size_t next_k = 0;
            for (std::size_t r = cursor; r < merged.size() && merged[r].first < ref.page + kPage; ++r) {
                const std::uint64_t lo = std::max(merged[r].first, ref.page);
                const std::uint64_t hi = std::min(merged[r].second, ref.page + kPage);
                if (lo >= hi) continue;
                const std::size_t k0 = std::max(next_k, std::size_t((lo - ref.page) / kPiece));
                const std::size_t k1 = std::size_t((hi - ref.page + kPiece - 1) / kPiece);
                if (k0 < k1) {
                    written += write_back_pieces(found->second, mapped->second, gpu, k0, k1, write, refilter_frames);
                    next_k = k1;
                }
            }
        }
        finish_write_back(written, refilter_frames);
        return written;
    }
    template <class Write>
    std::uint64_t write_back_batch(const std::vector<HomeRef>& batch,
                                   const std::vector<std::pair<std::uint64_t, std::uint64_t>>& ranges,
                                   const std::uint8_t* staged, Write&& write) {
        return write_back_batch(batch.data(), batch.size(), ranges, staged, std::forward<Write>(write));
    }

    // The binds noted since the last flush, as one batch signalling `value`. False when there were none (nothing to
    // wait for) or the bind failed (counted; the caller logs).
    bool flush(std::uint64_t value) {
        if (pending_.empty() && alias_pending_.empty()) return false;
        std::size_t binds = 0;
        for (const auto& [chunk, by_offset] : pending_) binds += by_offset.size();
        for (const auto& [buffer, by_offset] : alias_pending_) binds += by_offset.size();
        last_batch_ = binds;
        // The last bind of a page in this batch is the one that counts: binds of one range in one batch are not
        // ordered by the API.
        std::vector<std::vector<VkSparseMemoryBind>> lists;
        std::vector<VkSparseBufferMemoryBindInfo> infos;
        lists.reserve(pending_.size() + alias_pending_.size());
        for (auto& [chunk, by_offset] : pending_) {
            auto& list = lists.emplace_back();
            list.reserve(by_offset.size());
            for (const auto& [offset, bind] : by_offset) list.push_back(bind);
            infos.push_back({chunks_[chunk], std::uint32_t(list.size()), list.data()});
        }
        // The aliases' binds in the same batch, so the one wait orders a reader of either address.
        for (auto& [buffer, by_offset] : alias_pending_) {
            auto& list = lists.emplace_back();
            list.reserve(by_offset.size());
            for (const auto& [offset, bind] : by_offset) list.push_back(bind);
            infos.push_back({buffer, std::uint32_t(list.size()), list.data()});
        }
        const bool ok = hooks_.bind && hooks_.bind(infos, value);
        ++counters_.batches;
        pending_.clear();
        alias_pending_.clear();
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
        // The filtered copy (alias): its own slot, bound at the alias of `filter_va`, the aliased page it was made for.
        bool copy = false;
        std::uint32_t copy_unit = 0, copy_slot = 0;
        std::uint64_t filter_va = 0;
        bool fresh = true;       // the copy's next fill writes every byte (hooks.filter's `fresh`)
        bool refilter = false;   // the title's bytes changed outside an upload (write_back): fill the copy
    };
    struct Alias {
        std::uint64_t hi;
        VkBuffer buffer;
        VkDeviceSize offset;
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

    static std::vector<std::pair<std::uint64_t, std::uint64_t>> merge_ranges(
        const std::vector<std::pair<std::uint64_t, std::uint64_t>>& ranges) {
        std::vector<std::pair<std::uint64_t, std::uint64_t>> sorted;
        sorted.reserve(ranges.size());
        for (const auto& [b, e] : ranges)
            if (b < e) sorted.emplace_back(b, e);
        if (sorted.size() <= 1) return sorted;
        std::sort(sorted.begin(), sorted.end());
        std::vector<std::pair<std::uint64_t, std::uint64_t>> merged;
        merged.reserve(sorted.size());
        for (const auto& [b, e] : sorted) {
            if (!merged.empty() && b <= merged.back().second) {
                if (e > merged.back().second) merged.back().second = e;
            } else {
                merged.emplace_back(b, e);
            }
        }
        return merged;
    }

    template <class Write>
    std::uint64_t write_back_pieces(Home& home, std::uint8_t* frame, const std::uint8_t* gpu, std::size_t k0,
                                    std::size_t k1, Write&& write, std::vector<std::uint8_t*>& refilter_frames) {
        std::uint64_t written = 0;
        alignas(64) std::uint8_t title[kPiece];
        const auto frame_base = reinterpret_cast<std::uintptr_t>(frame);
        for (std::size_t k = k0; k < k1; ++k) {
            const std::uint64_t agreed = home.sums[k];
            const std::uint64_t now = shadow_keep::sum_of(gpu + k * kPiece, kPiece);
            if (now == agreed) continue;  // the GPU did not change it
            const std::uint8_t* from = frame + k * kPiece;
            if (!(hooks_.copy ? hooks_.copy(title, from, kPiece) : (std::memcpy(title, from, kPiece), true)))
                continue;  // the title's page went away
            if (shadow_keep::sum_of(title, kPiece) != agreed) {
                ++counters_.conflicts;  // both changed it: the title's bytes win at the next upload
                continue;
            }
            if (!write(static_cast<std::uint64_t>(frame_base + k * kPiece), gpu + k * kPiece, kPiece)) continue;
            home.sums[k] = now;
            written += kPiece;
            if (home.copy && !home.refilter) {
                home.refilter = true;  // the title's bytes changed: its filtered copy follows
                refilter_frames.push_back(frame);
            }
        }
        return written;
    }

    void finish_write_back(std::uint64_t written, const std::vector<std::uint8_t*>& refilter_frames) {
        counters_.written_back_bytes += written;
        for (std::uint8_t* frame : refilter_frames) {
            const auto it = homes_.find(frame);
            if (it != homes_.end() && it->second.refilter) fill_copy(it->second, frame, false);
        }
    }

    // Brings the home up to date with the title's page, piece by piece: `whole` (a new home) copies every piece, and
    // otherwise only a piece the title changed since both sides last agreed, so a piece only the GPU wrote keeps the
    // GPU's bytes (and goes back with write_back). A piece both changed takes the title's bytes. Copies the whole 64 KiB
    // page in one hooks_.copy call, falling back to per-piece copies only when the whole-page copy fails.
    void fill(Home& home, const std::uint8_t* frame, bool whole) {
        std::uint8_t* to = units_[home.unit].host + std::size_t(home.slot) * kPage;
        bool failed = false;
        std::size_t changed = 0;
        const bool whole_ok = hooks_.copy ? hooks_.copy(fill_scratch_.data(), frame, kPage)
                                          : (std::memcpy(fill_scratch_.data(), frame, kPage), true);
        for (std::size_t k = 0; k < kPieces; ++k) {
            std::uint8_t* title = fill_scratch_.data() + k * kPiece;
            if (!whole_ok && !hooks_.copy(title, frame + k * kPiece, kPiece)) {
                std::memset(title, 0, kPiece);
                failed = true;
            }
            const std::uint64_t sum = shadow_keep::sum_of(title, kPiece);
            if (!whole && sum == home.sums[k]) continue;
            ++changed;
            std::memcpy(to + k * kPiece, title, kPiece);
            home.sums[k] = sum;
            counters_.uploaded_bytes += kPiece;
        }
        // The filtered copy follows the title's bytes: refilled when a piece of them changed (the filter itself finds
        // which slots), and whole when it holds nothing yet.
        if (home.copy && (changed || home.fresh || home.refilter)) fill_copy(home, frame, false);
        if (failed) ++counters_.upload_failures;
        ++counters_.uploads;
    }

    Home* home_of(std::uint8_t* frame, std::shared_ptr<void> keep, std::uint64_t va) {
        if (const auto it = homes_.find(frame); it != homes_.end()) return &it->second;
        std::uint32_t unit = 0, slot = 0;
        if (!take_slot(unit, slot)) return nullptr;
        Home& home = homes_[frame];
        home.unit = unit;
        home.slot = slot;
        home.keep = std::move(keep);
        fill(home, frame, true);
        (void)va;
        return &home;
    }

    bool aliased(std::uint64_t va) const {
        auto it = aliases_.upper_bound(va);
        if (it == aliases_.begin()) return false;
        --it;
        return va < it->second.hi;
    }
    // A filtered copy for `home`, made for the aliased page at `va` and filled whole now. No room: none (counted).
    void make_copy(Home& home, const std::uint8_t* frame, std::uint64_t va) {
        if (!take_slot(home.copy_unit, home.copy_slot)) {
            ++counters_.upload_failures;
            return;
        }
        home.copy = true;
        home.filter_va = va;
        home.fresh = true;
        fill_copy(home, frame, true);
    }
    // Fills the filtered copy from the title's page through hooks.filter (`whole`: every byte).
    void fill_copy(Home& home, const std::uint8_t* frame, bool whole) {
        home.refilter = false;
        if (!home.copy) return;
        std::uint8_t* to = units_[home.copy_unit].host + std::size_t(home.copy_slot) * kPage;
        const bool fresh = whole || home.fresh;
        home.fresh = false;
        const bool ok = hooks_.filter ? hooks_.filter(home.filter_va, to, frame, kPage, fresh)
                        : hooks_.copy  ? hooks_.copy(to, frame, kPage)
                                       : (std::memcpy(to, frame, kPage), true);
        if (!ok) {
            std::memset(to, 0, kPage);
            home.fresh = true;  // zeros now: the next fill writes all again
            ++counters_.upload_failures;
        }
        ++counters_.filtered;
    }
    // The filtered copy goes, after `ticket` (the submissions that could read it at the alias).
    void drop_copy(Home& home, std::uint64_t ticket) {
        if (!home.copy) return;
        freed_.push_back({home.copy_unit, home.copy_slot, ticket});
        home.copy = false;
        home.filter_va = 0;
        home.fresh = true;
        home.refilter = false;
    }
    // The bind of the page at `va` into every alias over it, into the next batch.
    void note_alias_bind(std::uint64_t va, VkDeviceMemory memory, VkDeviceSize offset) {
        if (aliases_.empty()) return;
        auto it = aliases_.upper_bound(va);
        if (it == aliases_.begin()) return;
        --it;
        if (va >= it->second.hi) return;
        const VkDeviceSize at = it->second.offset + (va - it->first);
        alias_pending_[it->second.buffer][at] = {at, kPage, memory, offset, 0};
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
        drop_copy(it->second, ticket);
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
    std::map<std::uint64_t, Alias> aliases_;                    // title address -> [it, hi) also bound there, filtered
    std::map<VkBuffer, std::map<VkDeviceSize, VkSparseMemoryBind>> alias_pending_;  // buffer -> offset -> next bind
    Counters counters_;
    std::size_t last_batch_ = 0;
    alignas(64) std::array<std::uint8_t, kPage> fill_scratch_{};

public:
    std::size_t last_batch() const { return last_batch_; }  // binds in the last flush()
};

}  // namespace leonida
