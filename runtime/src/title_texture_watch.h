#pragma once
// Which title textures may have changed since they were last uploaded (the D3D12 layer's title texture refresh,
// DeviceState / Device::refresh_title_textures in d3d12_vulkan_device.cpp).
//
// A texture the title keeps in its own memory is filled from that memory, and filled again when the title writes it.
// The writes are found where they happen, not by reading the bytes back: each texture's 64 KiB tiles (at its CPU view,
// and at every other address xmem maps the same physical pages at) are cells of the geometry watch
// (title_geometry_watch.h), armed read-only. A write faults once per tile, and the fault handler, xmem's mapping
// changes and a kernel write's opening bump the tile's version and add it to this watch's write queue. collect()
// takes that queue and names the textures over the tiles in it, and nothing else: a texture nobody wrote is not
// looked at. Before a texture is read for its upload, examine() arms its changed tiles again, so a write that lands
// during or after the read faults again and is in the next collect().
//
// Memory not committed yet is armed like the rest: a page that arrives through xmem bumps the tile's mapping, so the
// tile is in the queue then. A tile whose arming fails stays `unarmed`, and the textures over it are named at every
// collect() until it is armed, but what examine() says decides whether they are read:
//   - busy: a kernel write into the tile is in progress (a ReadFile or a DirectStorage request pinned it). Reading now
//     would upload a half-written texture, again at every pass until the write ends; the texture is read once the pin
//     is gone and the tile could be armed.
//   - unwatched: the guard cannot be set (VirtualProtect refused, no write-fault handler): read, and uploaded only when
//     the bytes differ from the last upload's. The one fallback, for memory a guard cannot watch.
//
// Complexity: O(dirty tiles + unarmed tiles) per collect() and O(dirty tiles of key) per examine() using hash tables
// and per-pass generation stamps, with an O(1) lock-free may_have_work() fast-path that decouples buffer-only hot
// pages in TitleMirror from waking or scanning watched textures.
//
// Watch is the geometry watch in the runtime and a fake in the tests:
//   using Cell = ...;                            // has std::atomic<std::uint64_t> version, mapping
//   Cell* acquire(std::uintptr_t tile);          // a reference to the tile's cell
//   void release(std::uintptr_t tile);           // drops it (and the tile's guard with the last one)
//   int arm(std::uintptr_t tile, Cell&);         // makes the tile read-only: 1 done, 0 busy, -1 cannot be guarded
//   void aliases(std::uintptr_t tile, std::vector<std::uintptr_t>&);  // the tile's other addresses
//   void take_written(std::vector<std::uintptr_t>& tiles, bool& overflow);  // the queue since the last call
//   bool enabled();                              // a write-fault handler is installed: arming is allowed
//   void hot_tiles(std::vector<std::uintptr_t>&); // tiles another owner leaves writable: written at every look
//   Optional lock-free hints (detected via requires):
//   bool has_written() const;                    // true if take_written() has queued tiles
//   std::uint64_t hot_generation() const;        // monotonic counter bumped on every hot-set change (preferred)
//   std::size_t hot_count() const;               // fallback when hot_generation() is not provided
//
// Not thread-safe (except may_have_work()): the caller holds the title texture registry's lock.
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace leonida {

// A texture's bytes in each 64 KiB tile it covers, hashed as of its last read. When some of its tiles were written
// (TitleTextureWatch::examine's `written`), only its bytes in those tiles are hashed again: equal, and the texture is
// unchanged without being read whole. A write to a neighbour in the same tile costs that texture its share of the
// tile, not a read of all of it. The faults that report writes are per tile (the geometry watch's cells reopen a
// whole tile, which the geometry copies sharing them depend on), so this is the finest the watch can tell apart.
struct TilePieces {
    static constexpr std::uintptr_t kTile = 0x10000;
    std::uintptr_t cpu = 0;
    std::uint64_t span = 0;
    std::vector<std::uint64_t> hashes;  // per tile from cpu's tile on

    // The texture's bytes [cpu, cpu + span) inside `tile`, as an offset from cpu and a length.
    static std::pair<std::uint64_t, std::uint64_t> part(std::uintptr_t cpu, std::uint64_t span, std::uintptr_t tile) {
        const std::uintptr_t from = (std::max)(cpu, tile), to = (std::min)(std::uintptr_t(cpu + span), tile + kTile);
        return from < to ? std::pair<std::uint64_t, std::uint64_t>{from - cpu, to - from}
                         : std::pair<std::uint64_t, std::uint64_t>{0, 0};
    }
    bool valid_for(std::uintptr_t at, std::uint64_t bytes) const {
        return cpu == at && span == bytes && !hashes.empty();
    }
    // The tiles [at, at + bytes) touches.
    static std::size_t tiles(std::uintptr_t at, std::uint64_t bytes) {
        return std::size_t((((at + bytes + kTile - 1) & ~(kTile - 1)) - (at & ~(kTile - 1))) / kTile);
    }
    // For a texture at `at`: one hash a tile, to be set by part (set_part, from any thread for distinct parts).
    void reset(std::uintptr_t at, std::uint64_t bytes) {
        cpu = at;
        span = bytes;
        hashes.assign(tiles(at, bytes), 0);
    }
    // The texture's hash: its parts' hashes (and its size) mixed in order, so a texture read in parts needs no second
    // pass over its bytes.
    std::uint64_t combined() const {
        std::uint64_t h = 1469598103934665603ull ^ span;
        for (const std::uint64_t part : hashes) {
            h = (h ^ part) * 1099511628211ull;
            h ^= h >> 29;
        }
        return h;
    }
    // From a whole copy of the texture's bytes: hash(pointer, length) of each tile's part.
    template<class Hash>
    void compute(std::uintptr_t at, std::uint64_t bytes, const std::uint8_t* copy, Hash&& hash) {
        cpu = at;
        span = bytes;
        hashes.clear();
        for (std::uintptr_t tile = at & ~(kTile - 1); tile < at + bytes; tile += kTile) {
            const auto [offset, length] = part(at, bytes, tile);
            hashes.push_back(hash(copy + offset, length));
        }
    }
    // Whether the texture's parts in `written` hash as before: hash(title address, length) reads them where they are.
    // False when a part differs, cannot be read (hash returns nullopt) or is not known.
    template<class Hash>
    bool unchanged(const std::vector<std::uintptr_t>& written, Hash&& hash) const {
        for (const std::uintptr_t tile : written) {
            const std::size_t i = (tile - (cpu & ~(kTile - 1))) / kTile;
            if (tile < (cpu & ~(kTile - 1)) || i >= hashes.size()) return false;
            const auto [offset, length] = part(cpu, span, tile);
            const auto h = hash(cpu + offset, length);
            if (!h || *h != hashes[i]) return false;
        }
        return true;
    }
};

template<class Watch, class Key = std::uint64_t>
class TitleTextureWatch {
public:
    static constexpr std::uintptr_t kTile = 0x10000;
    using Cell = typename Watch::Cell;

    explicit TitleTextureWatch(Watch& watch) : watch_(watch) {}
    ~TitleTextureWatch() {
        for (auto& [address, tile] : tiles_) drop(address, tile);
    }
    TitleTextureWatch(const TitleTextureWatch&) = delete;
    TitleTextureWatch& operator=(const TitleTextureWatch&) = delete;

    struct Stats {
        std::uint64_t collects = 0, named = 0, unarmed_named = 0, arms = 0, arm_failures = 0, arm_busy = 0,
                      written_tiles = 0, overflows = 0, hot_skips = 0;
    };
    const Stats& stats() const { return stats_; }
    std::size_t textures() const { return textures_.size(); }
    std::size_t tiles() const { return tiles_.size(); }
    std::size_t unarmed() const { return unarmed_list_.size(); }
    std::size_t hot_overlap() const { return active_hot_tiles_.size(); }

    // O(1) lock-free fast-path check: false guarantees collect() would emit nothing (no pending dirty/unarmed/noted
    // work, no queued write notifications, and no watched tiles overlapping hot mirror pages).
    bool may_have_work() const noexcept {
        if (has_pending_.load(std::memory_order_acquire)) return true;
        if (hot_overlap_.load(std::memory_order_acquire) != 0) return true;
        if constexpr (requires(const Watch& w) { w.has_written(); }) {
            if (watch_.has_written()) return true;
        } else {
            return true;
        }
        if constexpr (has_hot_stamp()) {
            if (hot_stamp() != last_hot_stamp_atomic_.load(std::memory_order_acquire)) return true;
        }
        return false;
    }

    // Recomputes the lock-free fast-path hint after examine() finishes arming tiles for the pass.
    void update_idle_hint() noexcept {
        const bool pending = !unarmed_list_.empty() || !pending_dirty_keys_.empty() ||
                             !pending_noted_tiles_.empty() || hot_sync_needed_;
        hot_overlap_.store(active_hot_tiles_.size(), std::memory_order_release);
        last_hot_stamp_atomic_.store(last_hot_stamp_, std::memory_order_release);
        has_pending_.store(pending, std::memory_order_release);
    }

    // The texture `key` reads [cpu, cpu + bytes) from now on (a new texture, or one whose range changed): it is named
    // by the next collect() whatever its tiles say. Tracking it again over the same tiles changes nothing (its tiles stay
    // armed and it is not named: the title's Map, a view, a layout found for the same bytes are not writes; touch()
    // names it when its bytes must be read again); over other tiles it replaces its range.
    void track(Key key, std::uintptr_t cpu, std::uint64_t bytes) {
        const std::uintptr_t first = cpu & ~(kTile - 1);
        const std::uintptr_t last = (cpu + bytes + kTile - 1) & ~(kTile - 1);
        if (const auto it = textures_.find(key); it != textures_.end() && bytes &&
                                                 it->second.first == first && it->second.last == last)
            return;
        untrack(key);
        if (!bytes) return;
        Texture& t = textures_[key];
        t.first = first;
        t.last = last;
        bool added_tile = false;
        for (std::uintptr_t a = t.first; a < t.last; a += kTile) {
            auto [it, fresh] = tiles_.try_emplace(a);
            Tile& tile = it->second;
            if (fresh) {
                tile.primary = watch_.acquire(a);
                add_unarmed(a, tile);  // not armed for anyone yet
                added_tile = true;
            }
            tile.textures.push_back(key);
        }
        if (added_tile) {
            sorted_tiles_dirty_ = true;
            hot_sync_needed_ = true;
        }
        pending_dirty_keys_.push_back(key);
        has_pending_.store(true, std::memory_order_release);
    }

    // The texture is gone, or no longer filled from memory: its tiles are released when no other texture reads them.
    void untrack(Key key) {
        const auto it = textures_.find(key);
        if (it == textures_.end()) return;
        bool removed_tile = false;
        for (std::uintptr_t a = it->second.first; a < it->second.last; a += kTile) {
            const auto t = tiles_.find(a);
            if (t == tiles_.end()) continue;
            auto& vec = t->second.textures;
            if (const auto pos = std::find(vec.begin(), vec.end(), key); pos != vec.end()) {
                *pos = vec.back();
                vec.pop_back();
            }
            if (vec.empty()) {
                drop(a, t->second);
                remove_unarmed(t->second);
                if (t->second.hot_index != kNoIndex) remove_active_hot(t->second);
                tiles_.erase(t);
                removed_tile = true;
            }
        }
        if (removed_tile) sorted_tiles_dirty_ = true;
        textures_.erase(it);
        hot_overlap_.store(active_hot_tiles_.size(), std::memory_order_release);
    }

    bool tracked(Key key) const { return textures_.count(key) != 0; }

    // Something other than a fault says [from, to) changed (a write noted by its writer, a mapping change seen by
    // the caller): the textures over it are named by the next collect().
    void note_changed(std::uintptr_t from, std::uintptr_t to) {
        if (from >= to || tiles_.empty()) return;
        for_each_tile_in(from, to, [&](std::uintptr_t addr, Tile& tile) {
            tile.stale = true;
            if (!tile.pending_note) {
                tile.pending_note = true;
                pending_noted_tiles_.push_back(addr);
            }
        });
        if (!pending_noted_tiles_.empty()) has_pending_.store(true, std::memory_order_release);
    }

    // The texture is named by the next collect() (its last fill did not complete: no image backing yet, a failed
    // upload, memory not readable yet).
    void touch(Key key) {
        if (textures_.count(key)) {
            pending_dirty_keys_.push_back(key);
            has_pending_.store(true, std::memory_order_release);
        }
    }

    // Each texture over [from, to), once.
    template<class F>
    void for_each_over(std::uintptr_t from, std::uintptr_t to, F&& f) {
        if (from >= to || tiles_.empty()) return;
        const std::uint64_t stamp = ++over_gen_;
        for_each_tile_in(from, to, [&](std::uintptr_t, Tile& tile) {
            for (const Key k : tile.textures) {
                if (const auto tex = textures_.find(k); tex != textures_.end() && tex->second.over_gen != stamp) {
                    tex->second.over_gen = stamp;
                    f(k);
                }
            }
        });
    }

    // The textures that may have changed since their last examine(), each once, in key order.
    // Cost: strictly O(dirty tiles + unarmed tiles + overlapping hot tiles), never O(total watched textures/tiles).
    template<class F>
    void collect(F&& f) {
        ++stats_.collects;
        const std::uint64_t gen = ++pass_gen_;
        named_.clear();

        // 1. Tiles noted via note_changed() since the last collect().
        for (const std::uintptr_t a : pending_noted_tiles_) {
            const auto it = tiles_.find(a);
            if (it == tiles_.end()) continue;
            it->second.pending_note = false;
            mark_tile_pass(a, it->second, gen, false);
        }
        pending_noted_tiles_.clear();

        // 2. Keys dirtied via track() or touch() since the last collect().
        for (const Key k : pending_dirty_keys_) {
            if (const auto it = textures_.find(k); it != textures_.end()) mark_key_pass(k, it->second, gen, false);
        }
        pending_dirty_keys_.clear();

        // 3. Tiles from the write-fault / kernel-write / remap queue.
        bool overflow = false;
        watch_.take_written(written_, overflow);
        stats_.written_tiles += written_.size();
        if (overflow) {
            ++stats_.overflows;
            for (auto& [address, tile] : tiles_) changed(address, tile, gen);
        } else {
            for (const std::uintptr_t a : written_) {
                if (const auto it = tiles_.find(a); it != tiles_.end()) changed(a, it->second, gen);
                if (!by_peer_.empty()) {
                    if (const auto peer = by_peer_.find(a); peer != by_peer_.end())
                        for (const std::uintptr_t owner : peer->second)
                            if (const auto it = tiles_.find(owner); it != tiles_.end()) changed(owner, it->second, gen);
                }
            }
        }

        // 4. Tiles another owner keeps writable (TitleMirror hot pages): decouple buffer-only hot pages so
        //    non-overlapping hot pages neither lock geometry_writes on quiet passes nor scan all watched tiles.
        bool need_hot_poll = true;
        std::uint64_t cur_hot_stamp = 0;
        if constexpr (has_hot_stamp()) {
            cur_hot_stamp = hot_stamp();
            need_hot_poll = cur_hot_stamp != last_hot_stamp_ || !written_.empty() || overflow || hot_sync_needed_ ||
                            !active_hot_tiles_.empty();
            if (!need_hot_poll) ++stats_.hot_skips;
        }
        if (need_hot_poll) {
            clear_active_hot();
            watch_.hot_tiles(hot_);
            for (const std::uintptr_t a : hot_) {
                if (const auto it = tiles_.find(a); it != tiles_.end()) {
                    it->second.hot_gen = gen;
                    add_active_hot(a, it->second);
                    mark_tile_pass(a, it->second, gen, false);
                }
                if (!by_peer_.empty()) {
                    if (const auto peer = by_peer_.find(a); peer != by_peer_.end()) {
                        for (const std::uintptr_t owner : peer->second) {
                            if (const auto it = tiles_.find(owner); it != tiles_.end()) {
                                it->second.hot_gen = gen;
                                add_active_hot(owner, it->second);
                                mark_tile_pass(owner, it->second, gen, false);
                            }
                        }
                    }
                }
            }
            last_hot_stamp_ = cur_hot_stamp;
            hot_sync_needed_ = false;
        }

        // 5. Tiles not armed: whatever their version says, a write there need not have faulted.
        for (const std::uintptr_t a : unarmed_list_) {
            const auto it = tiles_.find(a);
            if (it == tiles_.end()) continue;
            it->second.stale = true;
            mark_tile_pass(a, it->second, gen, true);
        }

        std::sort(named_.begin(), named_.end());
        stats_.named += named_.size();
        for (const Key k : named_) f(k);
        // Keep fast-path hint accurate after callback (e.g. if callback examined/armed tiles).
        update_idle_hint();
    }

    // What examine() found: every tile armed (read it now; it is known unchanged afterwards until a tile is written),
    // a kernel write in progress over it (do not read it yet), or a tile that cannot be guarded (read it every time).
    enum class Look { armed, busy, unwatched };

    // Before the texture's bytes are read for an upload: its tiles that changed (or were never armed) are armed
    // again, so a write from now on is seen by the next collect(). Tiles not armed are tried at the next collect().
    // `written` (optional) gets those tiles: the only ones whose bytes can differ from the texture's last read.
    // Cost: O(dirty tiles of key), not O(total tiles of key).
    Look examine(Key key, std::vector<std::uintptr_t>* written = nullptr) {
        if (written) written->clear();
        const auto it = textures_.find(key);
        if (it == textures_.end()) return Look::unwatched;
        Texture& tex = it->second;
        Look look = Look::armed;
        const std::uint64_t gen = pass_gen_;
        if (tex.dirty_gen == gen) {
            if (tex.dirty_tiles_gen == gen) {
                if (tex.dirty_tiles.size() > 1) std::sort(tex.dirty_tiles.begin(), tex.dirty_tiles.end());
                for (const std::uintptr_t a : tex.dirty_tiles) {
                    const auto t = tiles_.find(a);
                    if (t == tiles_.end()) continue;
                    if (written && (t->second.pass_written_gen == gen || t->second.stale || t->second.unarmed_index != kNoIndex))
                        written->push_back(a);
                    const int armed = arm(a, t->second);
                    if (!armed) look = Look::busy;
                    else if (armed < 0 && look == Look::armed) look = Look::unwatched;
                }
            }
            return look;
        }
        // Direct examine() outside collect(): scan the texture's tiles.
        for (std::uintptr_t a = tex.first; a < tex.last; a += kTile) {
            const auto t = tiles_.find(a);
            if (t == tiles_.end()) continue;
            if (written && (t->second.pass_written_gen == gen || t->second.stale || t->second.unarmed_index != kNoIndex))
                written->push_back(a);
            const int armed = arm(a, t->second);
            if (!armed) look = Look::busy;
            else if (armed < 0 && look == Look::armed) look = Look::unwatched;
        }
        return look;
    }

private:
    static constexpr std::uint32_t kNoIndex = ~std::uint32_t(0);

    struct Tile {
        Cell* primary = nullptr;
        std::uint64_t version = 0, mapping = 0;  // as last armed; mapping 0: peers not looked up yet
        std::vector<std::pair<std::uintptr_t, Cell*>> peers;
        std::vector<Key> textures;
        std::uint64_t pass_written_gen = 0;
        std::uint64_t hot_gen = 0;
        std::uint32_t unarmed_index = kNoIndex;
        std::uint32_t hot_index = kNoIndex;
        bool stale = true;  // changed (or never armed) since it was last armed
        bool pending_note = false;
    };
    struct Texture {
        std::uintptr_t first = 0, last = 0;  // its tiles, [first, last)
        std::uint64_t dirty_gen = 0;
        std::uint64_t dirty_tiles_gen = 0;
        std::uint64_t over_gen = 0;
        std::vector<std::uintptr_t> dirty_tiles;
    };

    void add_unarmed(std::uintptr_t address, Tile& tile) {
        if (tile.unarmed_index != kNoIndex) return;
        tile.unarmed_index = static_cast<std::uint32_t>(unarmed_list_.size());
        unarmed_list_.push_back(address);
    }
    void remove_unarmed(Tile& tile) {
        if (tile.unarmed_index == kNoIndex) return;
        const std::uint32_t idx = tile.unarmed_index;
        const std::uintptr_t last = unarmed_list_.back();
        unarmed_list_[idx] = last;
        if (idx + 1 < unarmed_list_.size()) tiles_.find(last)->second.unarmed_index = idx;
        unarmed_list_.pop_back();
        tile.unarmed_index = kNoIndex;
    }

    void add_active_hot(std::uintptr_t address, Tile& tile) {
        if (tile.hot_index != kNoIndex) return;
        tile.hot_index = static_cast<std::uint32_t>(active_hot_tiles_.size());
        active_hot_tiles_.push_back(address);
    }
    void remove_active_hot(Tile& tile) {
        if (tile.hot_index == kNoIndex) return;
        const std::uint32_t idx = tile.hot_index;
        const std::uintptr_t last = active_hot_tiles_.back();
        active_hot_tiles_[idx] = last;
        if (idx + 1 < active_hot_tiles_.size()) tiles_.find(last)->second.hot_index = idx;
        active_hot_tiles_.pop_back();
        tile.hot_index = kNoIndex;
    }
    void clear_active_hot() {
        for (const std::uintptr_t a : active_hot_tiles_) {
            if (const auto it = tiles_.find(a); it != tiles_.end()) it->second.hot_index = kNoIndex;
        }
        active_hot_tiles_.clear();
    }

    template<class F>
    void for_each_tile_in(std::uintptr_t from, std::uintptr_t to, F&& f) {
        if (from == 0 && to == UINTPTR_MAX) {
            for (auto& [addr, tile] : tiles_) f(addr, tile);
            return;
        }
        const std::uintptr_t first = from & ~(kTile - 1);
        const std::uintptr_t span = to > first ? to - first : 0;
        if (span <= 64u * kTile) {
            for (std::uintptr_t a = first; a < to;) {
                if (const auto it = tiles_.find(a); it != tiles_.end()) f(a, it->second);
                if (to - a <= kTile) break;
                a += kTile;
            }
            return;
        }
        if (sorted_tiles_dirty_) {
            sorted_tiles_.clear();
            sorted_tiles_.reserve(tiles_.size());
            for (const auto& [addr, tile] : tiles_) sorted_tiles_.push_back(addr);
            std::sort(sorted_tiles_.begin(), sorted_tiles_.end());
            sorted_tiles_dirty_ = false;
        }
        for (auto it = std::lower_bound(sorted_tiles_.begin(), sorted_tiles_.end(), first);
             it != sorted_tiles_.end() && *it < to; ++it) {
            if (const auto found = tiles_.find(*it); found != tiles_.end()) f(*it, found->second);
        }
    }

    void drop(std::uintptr_t address, Tile& tile) {
        for (const auto& [peer, cell] : tile.peers) {
            unlink_peer(peer, address);
            watch_.release(peer);
        }
        tile.peers.clear();
        if (tile.primary) watch_.release(address);
        tile.primary = nullptr;
    }
    void unlink_peer(std::uintptr_t peer, std::uintptr_t owner) {
        const auto it = by_peer_.find(peer);
        if (it == by_peer_.end()) return;
        std::erase(it->second, owner);
        if (it->second.empty()) by_peer_.erase(it);
    }

    void mark_key_pass(Key k, Texture& tex, std::uint64_t gen, bool from_unarmed) {
        if (tex.dirty_gen != gen) {
            tex.dirty_gen = gen;
            named_.push_back(k);
            if (from_unarmed) ++stats_.unarmed_named;
        }
    }

    void mark_tile_pass(std::uintptr_t address, Tile& tile, std::uint64_t gen, bool from_unarmed) {
        tile.stale = true;
        const bool first_this_pass = tile.pass_written_gen != gen;
        if (first_this_pass) tile.pass_written_gen = gen;
        for (const Key k : tile.textures) {
            const auto it = textures_.find(k);
            if (it == textures_.end()) continue;
            Texture& tex = it->second;
            mark_key_pass(k, tex, gen, from_unarmed);
            if (first_this_pass) {
                if (tex.dirty_tiles_gen != gen) {
                    tex.dirty_tiles_gen = gen;
                    tex.dirty_tiles.clear();
                }
                tex.dirty_tiles.push_back(address);
            }
        }
    }

    // A tile the queue named: changed when its version (or a peer's) moved since it was armed.
    void changed(std::uintptr_t address, Tile& tile, std::uint64_t gen) {
        if (tile.stale) {  // known already
            mark_tile_pass(address, tile, gen, false);
            return;
        }
        std::uint64_t version = tile.primary->version.load(std::memory_order_acquire);
        for (const auto& [peer, cell] : tile.peers)
            version = (std::max)(version, cell->version.load(std::memory_order_acquire));
        if (version != tile.version || tile.primary->mapping.load(std::memory_order_acquire) != tile.mapping) {
            mark_tile_pass(address, tile, gen, false);
            add_unarmed(address, tile);  // until examine() arms it again
        }
    }

    int arm(std::uintptr_t address, Tile& tile) {
        if (!tile.stale && tile.unarmed_index == kNoIndex) return 1;
        if (!watch_.enabled()) {
            add_unarmed(address, tile);
            tile.stale = true;
            return -1;
        }
        // The tile's other addresses follow its mapping: looked up again when xmem remapped it.
        const std::uint64_t mapping = tile.primary->mapping.load(std::memory_order_acquire);
        if (mapping != tile.mapping) {
            aliases_.clear();
            watch_.aliases(address, aliases_);
            auto old = std::move(tile.peers);
            tile.peers.clear();
            for (const std::uintptr_t peer : aliases_) {
                if (peer == address) continue;
                tile.peers.emplace_back(peer, watch_.acquire(peer));
                by_peer_[peer].push_back(address);
                hot_sync_needed_ = true;
            }
            for (const auto& [peer, cell] : old) {
                unlink_peer(peer, address);
                watch_.release(peer);
            }
            tile.mapping = mapping;
        }
        // A tile currently held hot by TitleMirror is left writable in geometry_writes (arm() is a no-op under its
        // mutex). Skip the redundant arm call while it has no un-hot peers; when it cools down, set_hot(false) bumps
        // its version and queues it so the next collect() re-arms it cleanly.
        if (tile.hot_gen == pass_gen_ && tile.peers.empty()) {
            tile.version = tile.primary->version.load(std::memory_order_acquire);
            tile.stale = false;
            remove_unarmed(tile);
            return 1;
        }
        // The version is read before arming: a write between the two moves it, and the next collect() sees that.
        std::uint64_t version = tile.primary->version.load(std::memory_order_acquire);
        for (const auto& [peer, cell] : tile.peers)
            version = (std::max)(version, cell->version.load(std::memory_order_acquire));
        ++stats_.arms;
        int result = watch_.arm(address, *tile.primary);
        for (const auto& [peer, cell] : tile.peers) result = (std::min)(result, watch_.arm(peer, *cell) + 0);
        if (result <= 0) {
            ++(result ? stats_.arm_failures : stats_.arm_busy);
            add_unarmed(address, tile);
            tile.stale = true;
            return result;
        }
        tile.version = version;
        tile.stale = false;
        remove_unarmed(tile);
        return 1;
    }

    static constexpr bool has_hot_stamp() noexcept {
        return requires(const Watch& w) { w.hot_generation(); } || requires(const Watch& w) { w.hot_count(); };
    }
    std::uint64_t hot_stamp() const noexcept {
        if constexpr (requires(const Watch& w) { w.hot_generation(); }) {
            return static_cast<std::uint64_t>(watch_.hot_generation());
        } else if constexpr (requires(const Watch& w) { w.hot_count(); }) {
            return static_cast<std::uint64_t>(watch_.hot_count());
        } else {
            return 0;
        }
    }

    Watch& watch_;
    std::unordered_map<std::uintptr_t, Tile> tiles_;
    std::unordered_map<std::uintptr_t, std::vector<std::uintptr_t>> by_peer_;
    std::unordered_map<Key, Texture> textures_;
    std::vector<std::uintptr_t> unarmed_list_;
    std::vector<std::uintptr_t> active_hot_tiles_;
    std::vector<std::uintptr_t> sorted_tiles_;
    bool sorted_tiles_dirty_ = false;
    std::vector<std::uintptr_t> pending_noted_tiles_;
    std::vector<Key> pending_dirty_keys_;
    std::vector<Key> named_;
    std::vector<std::uintptr_t> written_, aliases_, hot_;
    std::uint64_t pass_gen_ = 0;
    std::uint64_t over_gen_ = 0;
    std::uint64_t last_hot_stamp_ = 0;
    bool hot_sync_needed_ = false;
    std::atomic<bool> has_pending_{false};
    std::atomic<std::size_t> hot_overlap_{0};
    std::atomic<std::uint64_t> last_hot_stamp_atomic_{0};
    Stats stats_;
};

}  // namespace leonida
