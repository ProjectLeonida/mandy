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
//     would upload a half-written texture, again at every pass until the write ends (in run 57, ~1.5 GiB of such
//     uploads per 10 s); the texture is read once the pin is gone and the tile could be armed.
//   - unwatched: the guard cannot be set (VirtualProtect refused, no write-fault handler): read, and uploaded only when
//     the bytes differ from the last upload's. The one fallback, for memory a guard cannot watch.
//
// A texture the title writes every frame (a video plane, a dynamic atlas) is not treated apart: its tiles fault once
// each per frame the title writes them, which costs far less than reading the texture back at every submission
// (write-combined memory reads at a few hundred MB/s).
//
// Watch is the geometry watch in the runtime and a fake in the tests:
//   using Cell = ...;                            // has std::atomic<std::uint64_t> version, mapping
//   Cell* acquire(std::uintptr_t tile);          // a reference to the tile's cell
//   void release(std::uintptr_t tile);           // drops it (and the tile's guard with the last one)
//   int arm(std::uintptr_t tile, Cell&);         // makes the tile read-only: 1 done, 0 busy (a kernel write in
//                                                // progress), -1 cannot be guarded
//   void aliases(std::uintptr_t tile, std::vector<std::uintptr_t>&);  // the tile's other addresses
//   void take_written(std::vector<std::uintptr_t>& tiles, bool& overflow);  // the queue since the last call
//   bool enabled();                              // a write-fault handler is installed: arming is allowed
//
// Not thread-safe: the caller holds the title texture registry's lock.
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
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
                      written_tiles = 0, overflows = 0;
    };
    const Stats& stats() const { return stats_; }
    std::size_t textures() const { return textures_.size(); }
    std::size_t tiles() const { return tiles_.size(); }
    std::size_t unarmed() const { return unarmed_.size(); }

    // The texture `key` reads [cpu, cpu + bytes) from now on (a new texture, or one whose range changed): it is named
    // by the next collect() whatever its tiles say. Tracking it again over the same tiles changes nothing (its tiles stay
    // armed and it is not named: the title's Map, a view, a layout found for the same bytes are not writes; touch()
    // names it when its bytes must be read again); over other tiles it replaces its range.
    void track(Key key, std::uintptr_t cpu, std::uint64_t bytes) {
        if (const auto it = textures_.find(key); it != textures_.end() && bytes &&
                                                 it->second.first == (cpu & ~(kTile - 1)) &&
                                                 it->second.last == ((cpu + bytes + kTile - 1) & ~(kTile - 1)))
            return;
        untrack(key);
        if (!bytes) return;
        Texture& t = textures_[key];
        t.first = cpu & ~(kTile - 1);
        t.last = (cpu + bytes + kTile - 1) & ~(kTile - 1);
        for (std::uintptr_t a = t.first; a < t.last; a += kTile) {
            auto [it, fresh] = tiles_.try_emplace(a);
            Tile& tile = it->second;
            if (fresh) {
                tile.primary = watch_.acquire(a);
                unarmed_.insert(a);  // not armed for anyone yet
            }
            tile.textures.push_back(key);
        }
        dirty_.insert(key);
    }

    // The texture is gone, or no longer filled from memory: its tiles are released when no other texture reads them.
    void untrack(Key key) {
        const auto it = textures_.find(key);
        if (it == textures_.end()) return;
        for (std::uintptr_t a = it->second.first; a < it->second.last; a += kTile) {
            const auto t = tiles_.find(a);
            if (t == tiles_.end()) continue;
            std::erase(t->second.textures, key);
            if (t->second.textures.empty()) {
                drop(a, t->second);
                unarmed_.erase(a);
                tiles_.erase(t);
            }
        }
        dirty_.erase(key);
        textures_.erase(it);
    }

    bool tracked(Key key) const { return textures_.count(key) != 0; }

    // Something other than a fault says [from, to) changed (a write noted by its writer, a mapping change seen by
    // the caller): the textures over it are named by the next collect().
    void note_changed(std::uintptr_t from, std::uintptr_t to) {
        if (from >= to) return;
        for (auto it = tiles_.lower_bound(from & ~(kTile - 1)); it != tiles_.end() && it->first < to; ++it)
            mark(it->second);
    }

    // The texture is named by the next collect() (its last fill did not complete: no image backing yet, a failed
    // upload, memory not readable yet).
    void touch(Key key) {
        if (textures_.count(key)) dirty_.insert(key);
    }

    // Each texture over [from, to), once.
    template<class F>
    void for_each_over(std::uintptr_t from, std::uintptr_t to, F&& f) {
        over_.clear();
        for (auto it = tiles_.lower_bound(from & ~(kTile - 1)); it != tiles_.end() && it->first < to; ++it)
            for (const Key k : it->second.textures)
                if (over_.insert(k).second) f(k);
    }

    // The textures that may have changed since their last examine(), each once, in key order. Cost: the tiles written
    // since the last call, plus the tiles that could not be armed.
    template<class F>
    void collect(F&& f) {
        ++stats_.collects;
        bool overflow = false;
        watch_.take_written(written_, overflow);
        stats_.written_tiles += written_.size();
        if (overflow) {
            ++stats_.overflows;
            for (auto& [address, tile] : tiles_) changed(address, tile);
        } else {
            for (const std::uintptr_t a : written_) {
                if (const auto it = tiles_.find(a); it != tiles_.end()) changed(a, it->second);
                if (const auto peer = by_peer_.find(a); peer != by_peer_.end())
                    for (const std::uintptr_t owner : peer->second)
                        if (const auto it = tiles_.find(owner); it != tiles_.end()) changed(owner, it->second);
            }
        }
        // Tiles not armed: whatever their version says, a write there need not have faulted.
        for (const std::uintptr_t a : unarmed_)
            if (const auto it = tiles_.find(a); it != tiles_.end()) {
                it->second.stale = true;
                for (const Key k : it->second.textures)
                    if (dirty_.insert(k).second) ++stats_.unarmed_named;
            }
        // The tiles written as of now, for examine()'s `written`: the first texture examined over a tile arms it again,
        // and the others over it must still see it written.
        pass_written_.clear();
        for (const Key k : dirty_) {
            const auto it = textures_.find(k);
            if (it == textures_.end()) continue;
            for (std::uintptr_t a = it->second.first; a < it->second.last; a += kTile)
                if (const auto t = tiles_.find(a); t != tiles_.end() && (t->second.stale || unarmed_.count(a)))
                    pass_written_.insert(a);
        }
        named_.assign(dirty_.begin(), dirty_.end());
        dirty_.clear();
        stats_.named += named_.size();
        for (const Key k : named_) f(k);
    }

    // What examine() found: every tile armed (read it now; it is known unchanged afterwards until a tile is written),
    // a kernel write in progress over it (do not read it yet), or a tile that cannot be guarded (read it every time).
    enum class Look { armed, busy, unwatched };

    // Before the texture's bytes are read for an upload: its tiles that changed (or were never armed) are armed
    // again, so a write from now on is seen by the next collect(). Tiles not armed are tried at the next collect().
    // `written` (optional) gets those tiles: the only ones whose bytes can differ from the texture's last read.
    Look examine(Key key, std::vector<std::uintptr_t>* written = nullptr) {
        if (written) written->clear();
        const auto it = textures_.find(key);
        if (it == textures_.end()) return Look::unwatched;
        Look look = Look::armed;
        for (std::uintptr_t a = it->second.first; a < it->second.last; a += kTile) {
            const auto t = tiles_.find(a);
            if (t == tiles_.end()) continue;
            if (written && (pass_written_.count(a) || t->second.stale || unarmed_.count(a))) written->push_back(a);
            const int armed = arm(a, t->second);
            if (!armed) look = Look::busy;
            else if (armed < 0 && look == Look::armed) look = Look::unwatched;
        }
        return look;
    }

private:
    struct Tile {
        Cell* primary = nullptr;
        std::uint64_t version = 0, mapping = 0;  // as last armed; mapping 0: peers not looked up yet
        std::vector<std::pair<std::uintptr_t, Cell*>> peers;
        std::vector<Key> textures;
        bool stale = true;  // changed (or never armed) since it was last armed
    };
    struct Texture {
        std::uintptr_t first = 0, last = 0;  // its tiles, [first, last)
    };

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

    void mark(Tile& tile) {
        tile.stale = true;
        for (const Key k : tile.textures) dirty_.insert(k);
    }
    // A tile the queue named: changed when its version (or a peer's) moved since it was armed.
    void changed(std::uintptr_t address, Tile& tile) {
        if (tile.stale) {  // known already
            for (const Key k : tile.textures) dirty_.insert(k);
            return;
        }
        std::uint64_t version = tile.primary->version.load(std::memory_order_acquire);
        for (const auto& [peer, cell] : tile.peers) version = (std::max)(version, cell->version.load(std::memory_order_acquire));
        if (version != tile.version || tile.primary->mapping.load(std::memory_order_acquire) != tile.mapping) {
            mark(tile);
            unarmed_.insert(address);  // until examine() arms it again
        }
    }

    int arm(std::uintptr_t address, Tile& tile) {
        if (!tile.stale && !unarmed_.count(address)) return 1;
        if (!watch_.enabled()) {
            unarmed_.insert(address);
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
            }
            for (const auto& [peer, cell] : old) {
                unlink_peer(peer, address);
                watch_.release(peer);
            }
            tile.mapping = mapping;
        }
        // The version is read before arming: a write between the two moves it, and the next collect() sees that.
        std::uint64_t version = tile.primary->version.load(std::memory_order_acquire);
        for (const auto& [peer, cell] : tile.peers) version = (std::max)(version, cell->version.load(std::memory_order_acquire));
        ++stats_.arms;
        int result = watch_.arm(address, *tile.primary);
        for (const auto& [peer, cell] : tile.peers) result = (std::min)(result, watch_.arm(peer, *cell) + 0);
        if (result <= 0) {
            ++(result ? stats_.arm_failures : stats_.arm_busy);
            unarmed_.insert(address);
            tile.stale = true;
            return result;
        }
        tile.version = version;
        tile.stale = false;
        unarmed_.erase(address);
        return 1;
    }

    Watch& watch_;
    std::map<std::uintptr_t, Tile> tiles_;
    std::map<std::uintptr_t, std::vector<std::uintptr_t>> by_peer_;  // another address -> the tiles mapped there too
    std::map<Key, Texture> textures_;
    std::set<std::uintptr_t> unarmed_;
    std::set<Key> dirty_, over_;
    std::set<std::uintptr_t> pass_written_;  // the tiles written as the last collect() found them
    std::vector<Key> named_;
    std::vector<std::uintptr_t> written_, aliases_;
    Stats stats_;
};

}  // namespace leonida
