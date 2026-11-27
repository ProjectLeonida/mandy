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
    // Fills the home of a page inside a filtered range (filter_range) in place of `copy`: the title's bytes at `from`
    // (the page shown at title address `va`) in, what the GPU reads into `to`. `fresh`: the home holds nothing of an
    // earlier fill (made now, or the page's filter changed), so every byte must be written; otherwise `to` still holds
    // this filter's last output for the same page. The title's bytes are never written. False as for `copy`.
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
        std::uint64_t filtered = 0;            // fills through hooks.filter (filter_range), of the uploads
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
            // A home filtered for this page that other pages still show is filled as they need it.
            if (const auto shown = homes_.find(page->second);
                shown != homes_.end() && shown->second.filter_va == va && shown->second.refs > 1) {
                shown->second.filter_va = 0;
                for (const auto& [other, other_frame] : pages_)
                    if (other != va && other_frame == page->second && filtered(other)) shown->second.filter_va = other;
                mark_fresh(shown->second, page->second);
            }
            release_home(page->second, ticket);
            pages_.erase(page);
        }
        if (!frame) {
            note_bind(va, VK_NULL_HANDLE, 0);
            ++counters_.unbinds;
            return;
        }
        Home* home = home_of(frame, std::move(keep), va);
        if (!home) {
            note_bind(va, VK_NULL_HANDLE, 0);
            return;
        }
        ++home->refs;
        pages_[va] = frame;
        note_bind(va, units_[home->unit].memory, VkDeviceSize(home->slot) * kPage);
        ++counters_.binds;
    }

    // [lo, hi) of the title's address space (whole pages) is also bound into `buffer` at `offset` + (va - lo): each page
    // of it shows its home there too, follows its remaps and unmaps, and goes in the same batches as the chunk binds, so
    // the same wait orders a reader of either address. A console descriptor heap uses it to be read at an offset below
    // 4 GiB of the arena (32-bit descriptor-table pointers). Pages not mapped are left unbound there (they read zero).
    void alias(std::uint64_t lo, std::uint64_t hi, VkBuffer buffer, VkDeviceSize offset) {
        aliases_[lo] = {hi, buffer, offset};
        for (const auto& [va, frame] : pages_)
            if (va >= lo && va < hi) {
                const auto home = homes_.find(frame);
                if (home != homes_.end())
                    note_alias_bind(va, units_[home->second.unit].memory, VkDeviceSize(home->second.slot) * kPage);
            }
    }
    // The alias at `lo` goes: its pages are unbound there.
    void unalias(std::uint64_t lo) {
        const auto it = aliases_.find(lo);
        if (it == aliases_.end()) return;
        const Alias alias = it->second;
        for (const auto& [va, frame] : pages_)
            if (va >= lo && va < alias.hi) alias_pending_[alias.buffer][alias.offset + (va - lo)] =
                {alias.offset + (va - lo), kPage, VK_NULL_HANDLE, 0, 0};
        aliases_.erase(it);
    }

    // [lo, hi) of the title's address space (whole pages) is filtered: the home of a page there is filled through
    // hooks.filter instead of a plain copy, now (fresh) and whenever the page is filled again. A home that pages at two
    // addresses show is filtered when one of them is in a filtered range (a page's home holds one set of bytes).
    void filter_range(std::uint64_t lo, std::uint64_t hi) {
        filters_[lo] = hi;
        for (const auto& [va, frame] : pages_)
            if (va >= lo && va < hi) refilter(frame, va);
    }
    // Filtered no more: the homes there are filled again as plain copies.
    void unfilter_range(std::uint64_t lo) {
        const auto it = filters_.find(lo);
        if (it == filters_.end()) return;
        const std::uint64_t hi = it->second;
        filters_.erase(it);
        for (auto& [frame, home] : homes_)
            if (home.filter_va >= lo && home.filter_va < hi) {
                home.filter_va = 0;
                mark_fresh(home, frame);
            }
    }
    // The home of the page at `va`, mapped (its host bytes): what a filter's caller writes one changed slot into
    // between fills. Null when the page is not mapped.
    std::uint8_t* home_bytes(std::uint64_t va) {
        const auto page = pages_.find(va & ~(kPage - 1));
        if (page == pages_.end()) return nullptr;
        const auto home = homes_.find(page->second);
        if (home == homes_.end()) return nullptr;
        return units_[home->second.unit].host + std::size_t(home->second.slot) * kPage;
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
            fill(home->second, frame);
            ++n;
        }
        dirty_.clear();
        return n;
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
    struct Home {
        std::uint32_t unit = 0, slot = 0, refs = 0;
        std::shared_ptr<void> keep;
        bool dirty = false;
        std::uint64_t filter_va = 0;  // a filtered page that shows it (filter_range), 0: filled by copy
        bool fresh = true;            // the next fill writes every byte (hooks.filter's `fresh`)
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

    void fill(Home& home, const std::uint8_t* frame) {
        std::uint8_t* to = units_[home.unit].host + std::size_t(home.slot) * kPage;
        const bool fresh = home.fresh;
        home.fresh = false;
        const bool ok = home.filter_va && hooks_.filter ? hooks_.filter(home.filter_va, to, frame, kPage, fresh)
                        : hooks_.copy                  ? hooks_.copy(to, frame, kPage)
                                                       : (std::memcpy(to, frame, kPage), true);
        if (!ok) {
            std::memset(to, 0, kPage);
            home.fresh = true;  // zeros now: the next fill writes all again
            ++counters_.upload_failures;
        }
        ++counters_.uploads;
        if (home.filter_va) ++counters_.filtered;
    }

    Home* home_of(std::uint8_t* frame, std::shared_ptr<void> keep, std::uint64_t va) {
        if (const auto it = homes_.find(frame); it != homes_.end()) {
            if (!it->second.filter_va && filtered(va)) refilter(frame, va);
            return &it->second;
        }
        std::uint32_t unit = 0, slot = 0;
        if (!take_slot(unit, slot)) return nullptr;
        Home& home = homes_[frame];
        home.unit = unit;
        home.slot = slot;
        home.keep = std::move(keep);
        home.filter_va = filtered(va) ? va : 0;
        fill(home, frame);
        return &home;
    }

    bool filtered(std::uint64_t va) const {
        auto it = filters_.upper_bound(va);
        if (it == filters_.begin()) return false;
        --it;
        return va < it->second;
    }
    // The home of `frame` is filtered for the page at `va` from its next fill, which writes every byte.
    void refilter(std::uint8_t* frame, std::uint64_t va) {
        const auto home = homes_.find(frame);
        if (home == homes_.end()) return;
        home->second.filter_va = va;
        mark_fresh(home->second, frame);
    }
    void mark_fresh(Home& home, std::uint8_t* frame) {
        home.fresh = true;
        if (!home.dirty) {
            home.dirty = true;
            dirty_.push_back(frame);
        }
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
        freed_.push_back({it->second.unit, it->second.slot, ticket});
        homes_.erase(it);
    }
    // One bind (memory null: unbind) of the page at `va`, in every chunk it lies in, into the next batch.
    void note_bind(std::uint64_t va, VkDeviceMemory memory, VkDeviceSize offset) {
        const std::uint64_t chunk = va / kChunk;
        pending_[chunk][va - chunk * kChunk] = {va - chunk * kChunk, kPage, memory, offset, 0};
        note_alias_bind(va, memory, offset);
    }

    TitleMirrorHooks hooks_;
    std::map<std::uint64_t, VkBuffer> chunks_;                 // chunk index -> buffer (null: could not be placed)
    std::unordered_map<std::uint64_t, std::uint8_t*> pages_;   // mirrored page -> its physical page's bytes
    std::unordered_map<std::uint8_t*, Home> homes_;            // physical page (its scratch bytes) -> home
    std::vector<Unit> units_;
    std::vector<std::uint8_t*> dirty_;
    std::deque<Freed> freed_;
    std::map<std::uint64_t, std::map<VkDeviceSize, VkSparseMemoryBind>> pending_;  // chunk -> offset -> next bind
    std::map<std::uint64_t, Alias> aliases_;                    // title address -> [it, hi) also bound there
    std::map<VkBuffer, std::map<VkDeviceSize, VkSparseMemoryBind>> alias_pending_;  // buffer -> offset -> next bind
    std::map<std::uint64_t, std::uint64_t> filters_;            // filtered ranges: lo -> hi
    Counters counters_;
    std::size_t last_batch_ = 0;

public:
    std::size_t last_batch() const { return last_batch_; }  // binds in the last flush()
};

}  // namespace leonida
