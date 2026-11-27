#pragma once
// Where xmem puts the title's GPU-readable reservations (title_mirror_space): a fixed range of the title's address space
// that the D3D12 layer's page mirror can cover (src/title_mirror.h).
//
// The translated shaders read a V# base or a 64-bit pointer as heap_base + sext48(address) (docs/design.md,
// "Heap-relative buffer device addresses"). A V# or pointer the title wrote into its own memory names a title address,
// so it reaches the title's bytes only if the GPU memory at heap_base + that address holds them. The mirror puts sparse
// buffers exactly there (bufferDeviceAddressCaptureReplay) and binds each page to its physical page's GPU home. RADV
// honours a chosen address only inside its sparse range, the top 64 TiB of the address space (vk_sparse_bind_bench), and
// heap_base, the sparse arena's address, lies at its bottom: so heap_base + address stays inside it for addresses below
// about 64 TiB. Under Wine and on Windows the host puts large reservations near the top of the 128 TiB user space
// (0x7fb3... in a world run), out of reach. So xmem picks the address of every XMemVirtualAlloc reservation the title lets
// it place (address null) from [kLow, kHigh), which the host leaves free, and the title only ever sees that address.
//
// The range is handed out in order and not reused: a released reservation's addresses are never given out again
// (32 TiB lasts any run measured; a reservation that does not fit is placed by the host, as before, and counted).
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <utility>
#include <vector>

namespace leonida::title_mirror_space {

constexpr std::uint64_t kLow = 0x100000000000ull;   // 16 TiB
constexpr std::uint64_t kHigh = 0x300000000000ull;  // 48 TiB
constexpr std::uint64_t kChunk = std::uint64_t(2) << 30;  // the mirror's sparse buffers: 2 GiB each, aligned

struct State {
    std::mutex mutex;
    std::uint64_t next = kLow;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> taken;  // [base, end), in order
    std::vector<bool> plain;  // per taken range: ordinary memory (not physical-page windows or placeholders)
    std::atomic<std::uint64_t> generation{0};
    std::atomic<std::uint64_t> placed_bytes{0}, refused{0};
};
inline State& state() {
    static State s;
    return s;
}

// LEONIDA_PAGE_MIRROR=0: the host places every reservation, as before.
inline bool enabled() {
    static const bool on = [] {
        const char* v = std::getenv("LEONIDA_PAGE_MIRROR");
        return !(v && v[0] == '0');
    }();
    return on;
}

inline bool contains(std::uint64_t va) { return va >= kLow && va < kHigh; }

// The next free address for `size` bytes aligned to `alignment` (a power of two), or 0 when the range is spent. `reserve`
// tries the host's reservation there: false (the host has something there) moves on by one chunk. A reservation of at
// most one sparse chunk (kChunk, 2 GiB) that would cross a chunk boundary advances to the next chunk boundary so placed
// buffers inside it never straddle two chunks. Under the state's lock:
// `reserve` must not take a lock a reserving thread holds (xmem calls this before taking any of its own).
// `plain`: ordinary memory the title's CPU and GPU read at this address (an XMemVirtualAlloc region of committed or
// reserved pages), mirrored page by page as readers name it; otherwise physical-page windows or placeholders, whose
// pages xmem's feed reports as they are mapped.
template <class Reserve>
std::uint64_t take(std::uint64_t size, std::uint64_t alignment, bool plain, Reserve&& reserve) {
    State& s = state();
    if (!enabled() || !size) return 0;
    if (alignment < 0x10000) alignment = 0x10000;
    std::lock_guard lock(s.mutex);
    for (int tries = 0; tries < 64; ++tries) {
        std::uint64_t at = (s.next + alignment - 1) & ~(alignment - 1);
        if (at >= kHigh) break;
        if (size <= kChunk) {
            const std::uint64_t chunk_end = (at & ~(kChunk - 1)) + kChunk;
            if (size > chunk_end - at) at = (chunk_end + alignment - 1) & ~(alignment - 1);
        }
        if (at >= kHigh || size > kHigh - at) break;
        if (reserve(at)) {
            s.next = at + ((size + 0xffff) & ~std::uint64_t(0xffff));
            s.taken.emplace_back(at, at + size);
            s.plain.push_back(plain);
            s.placed_bytes.fetch_add(size, std::memory_order_relaxed);
            s.generation.fetch_add(1, std::memory_order_release);
            return at;
        }
        s.next = ((at + kChunk) & ~(kChunk - 1));  // something of the host's is there: past it
    }
    s.refused.fetch_add(1, std::memory_order_relaxed);
    return 0;
}

inline std::uint64_t generation() { return state().generation.load(std::memory_order_acquire); }

// The reservations made so far: f(base, end, plain) for each.
template <class F>
void for_each_taken(F&& f) {
    State& s = state();
    std::vector<std::pair<std::uint64_t, std::uint64_t>> copy;
    std::vector<bool> plain;
    {
        std::lock_guard lock(s.mutex);
        copy = s.taken;
        plain = s.plain;
    }
    for (std::size_t i = 0; i < copy.size(); ++i) f(copy[i].first, copy[i].second, bool(plain[i]));
}

}  // namespace leonida::title_mirror_space
