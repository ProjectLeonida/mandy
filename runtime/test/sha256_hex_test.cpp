// leonida::sha256_hex (src/shader_library.h) against the implementation it replaced, which copied the whole input into
// a vector, padded it a byte at a time and grew the result a character at a time. Checks the FIPS 180-4 vectors,
// every length from 0 to 1100 bytes (so every tail size and both padding cases, one and two blocks), random inputs
// up to 70 000 bytes, and that a hash now allocates once (its 64-character result) instead of 3 to 7 times. No game
// data: every input is built here.
//
// `sha256_hex_test --bench` also times old and new on the input sizes the runtime hashes (a pipeline key of a few
// hundred bytes, a shader of a few to tens of kilobytes) and prints the ratio.
#include "../src/shader_library.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>
#include <utility>
#include <vector>

#define CHECK(condition) do {                                    \
    if (!(condition)) {                                          \
        std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); \
        return 1;                                                \
    }                                                            \
} while (0)

namespace {
std::atomic<std::size_t> g_allocations{0};
}

void* operator new(std::size_t size) {
    g_allocations.fetch_add(1, std::memory_order_relaxed);
    if (void* p = std::malloc(size ? size : 1)) return p;
    throw std::bad_alloc();
}
void* operator new[](std::size_t size) { return operator new(size); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }

namespace {

// The implementation before this change, kept verbatim as the reference.
std::string reference_sha256_hex(std::span<const std::uint8_t> data) {
    static constexpr std::uint32_t k[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
    std::uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                          0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    const auto rotr = [](std::uint32_t x, int n) { return (x >> n) | (x << (32 - n)); };
    std::vector<std::uint8_t> message(data.begin(), data.end());
    const std::uint64_t bits = std::uint64_t(data.size()) * 8;
    message.push_back(0x80);
    while (message.size() % 64 != 56) message.push_back(0);
    for (int i = 7; i >= 0; --i) message.push_back(std::uint8_t(bits >> (8 * i)));
    for (std::size_t block = 0; block < message.size(); block += 64) {
        std::uint32_t w[64];
        for (int i = 0; i < 16; ++i)
            w[i] = (std::uint32_t(message[block + 4 * i]) << 24) | (std::uint32_t(message[block + 4 * i + 1]) << 16) |
                   (std::uint32_t(message[block + 4 * i + 2]) << 8) | message[block + 4 * i + 3];
        for (int i = 16; i < 64; ++i) {
            const std::uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const std::uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        std::uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; ++i) {
            const std::uint32_t t1 = hh + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + k[i] + w[i];
            const std::uint32_t t2 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
            hh = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }
    static constexpr char hex[] = "0123456789abcdef";
    std::string out;
    for (const std::uint32_t word : h)
        for (int shift = 28; shift >= 0; shift -= 4) out.push_back(hex[(word >> shift) & 0xf]);
    return out;
}

struct Rng {
    std::uint64_t s;
    std::uint64_t next() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        return s;
    }
};

std::vector<std::uint8_t> random_bytes(Rng& rng, std::size_t n) {
    std::vector<std::uint8_t> out(n);
    for (auto& b : out) b = std::uint8_t(rng.next() >> 56);
    return out;
}

std::span<const std::uint8_t> text(const char* s) {
    return {reinterpret_cast<const std::uint8_t*>(s), std::strlen(s)};
}

template<typename F>
std::size_t allocations_of(F&& f) {
    const std::size_t before = g_allocations.load(std::memory_order_relaxed);
    f();
    return g_allocations.load(std::memory_order_relaxed) - before;
}

template<typename F>
double ns_per_call(F&& f, std::size_t iterations) {
    for (std::size_t i = 0; i < iterations / 8 + 1; ++i) f();  // warm up
    const auto start = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < iterations; ++i) f();
    const auto stop = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::nano>(stop - start).count() / double(iterations);
}

std::string portable_sha256_hex(std::span<const std::uint8_t> data) {
    return leonida::detail::sha256_hex_with(data, [](std::uint32_t (&h)[8], const std::uint8_t* p, std::size_t n) {
        leonida::detail::sha256_blocks_portable(h, p, n);
    });
}

#if LEONIDA_SHA256_X86
std::string x86_sha256_hex(std::span<const std::uint8_t> data) {
    return leonida::detail::sha256_hex_with(data, [](std::uint32_t (&h)[8], const std::uint8_t* p, std::size_t n) {
        leonida::detail::sha256_blocks_x86(h, p, n);
    });
}
#endif

bool have_x86() {
#if LEONIDA_SHA256_X86
    return leonida::detail::sha256_x86_supported();
#else
    return false;
#endif
}

using Hash = std::string (*)(std::span<const std::uint8_t>);

// The implementations to check against the reference: the portable compression, the SHA extensions one when this
// processor has them, and sha256_hex itself (whichever of the two it picks).
std::vector<std::pair<const char*, Hash>> implementations() {
    std::vector<std::pair<const char*, Hash>> out{{"portable", portable_sha256_hex},
                                                  {"sha256_hex", leonida::sha256_hex}};
#if LEONIDA_SHA256_X86
    if (have_x86()) out.push_back({"x86 SHA extensions", x86_sha256_hex});
#endif
    return out;
}

int bench() {
    Rng rng{0x5eed5eed5eedull};
    const std::size_t sizes[] = {64, 200, 400, 1024, 4096, 16384, 65536};
    std::size_t sink = 0;
    std::printf("sha256_hex uses the %s compression\n", have_x86() ? "x86 SHA extensions" : "portable");
    std::printf("%8s %10s %10s %8s %10s %8s %7s %7s\n", "bytes", "old ns", "portable", "speedup", "sha256_hex",
                "speedup", "old new", "allocs");
    for (const std::size_t n : sizes) {
        const auto data = random_bytes(rng, n);
        const std::size_t iterations = n <= 1024 ? 200000 : 2000000 / (n / 64);
        // Best of five runs each, interleaved, so a frequency step or a context switch hurts no side alone.
        double old_ns = 1e30, portable_ns = 1e30, new_ns = 1e30;
        for (int run = 0; run < 5; ++run) {
            old_ns = std::min(old_ns, ns_per_call([&] { sink += reference_sha256_hex(data)[0]; }, iterations));
            portable_ns = std::min(portable_ns, ns_per_call([&] { sink += portable_sha256_hex(data)[0]; }, iterations));
            new_ns = std::min(new_ns, ns_per_call([&] { sink += leonida::sha256_hex(data)[0]; }, iterations));
        }
        const std::size_t old_allocs = allocations_of([&] { sink += reference_sha256_hex(data)[0]; });
        const std::size_t new_allocs = allocations_of([&] { sink += leonida::sha256_hex(data)[0]; });
        std::printf("%8zu %10.1f %10.1f %7.2fx %10.1f %7.2fx %3zu %3zu\n", n, old_ns, portable_ns, old_ns / portable_ns,
                    new_ns, old_ns / new_ns, old_allocs, new_allocs);
    }
    std::printf("(checksum %zu)\n", sink);
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    // FIPS 180-4 / NIST CAVS vectors, each implementation.
    const std::vector<std::uint8_t> million(1000000, 'a');
    for (const auto& [name, hash] : implementations()) {
        std::printf("checking %s\n", name);
        CHECK(hash({}) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
        CHECK(hash(text("abc")) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
        CHECK(hash(text("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq")) ==
              "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
        CHECK(hash(text("abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnopjklmnopqklmnopqrlmnopq"
                        "rsmnopqrstnopqrstu")) ==
              "cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1");
        CHECK(hash(million) == "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
        // A span over no storage at all (data() == nullptr) hashes as the empty string.
        CHECK(hash(std::span<const std::uint8_t>(static_cast<const std::uint8_t*>(nullptr), 0)) == hash(text("")));

        // Every length through 1100 bytes: every tail size 0..63, the one-block (tail < 56) and two-block
        // (tail >= 56) padding, after 0..17 whole blocks. Then random lengths up to 70 000 bytes.
        Rng rng{0x9e3779b97f4a7c15ull};
        for (std::size_t n = 0; n <= 1100; ++n) {
            const auto data = random_bytes(rng, n);
            const std::string expected = reference_sha256_hex(data);
            CHECK(hash(data) == expected);
            // The same bytes at an odd address: whole blocks are read where they lie, not from an aligned copy.
            std::vector<std::uint8_t> shifted(n + 1);
            if (n) std::memcpy(shifted.data() + 1, data.data(), n);
            CHECK(hash(std::span<const std::uint8_t>(shifted.data() + 1, n)) == expected);
        }
        for (int i = 0; i < 300; ++i) {
            const auto data = random_bytes(rng, std::size_t(rng.next() % 70000));
            CHECK(hash(data) == reference_sha256_hex(data));
        }
        // Constant fills, which a schedule or padding mistake would hash alike: all zero, all 0x80, all 0xff.
        for (const std::uint8_t fill : {std::uint8_t(0), std::uint8_t(0x80), std::uint8_t(0xff)})
            for (const std::size_t n : {0u, 55u, 56u, 63u, 64u, 119u, 120u, 127u, 128u, 4096u}) {
                const std::vector<std::uint8_t> data(n, fill);
                CHECK(hash(data) == reference_sha256_hex(data));
            }
    }

    // Allocations: the new hash allocates its result and nothing else, at every padding case and size; the old one
    // allocated more (2 to 7 depending on the library and the size).
    Rng rng{0x1234567887654321ull};
    for (const std::size_t n : {0u, 1u, 55u, 56u, 63u, 64u, 300u, 4096u, 65536u}) {
        const auto data = random_bytes(rng, n);
        std::string hash;
        const std::size_t allocs = allocations_of([&] { hash = leonida::sha256_hex(data); });
        CHECK(allocs == 1);
        CHECK(hash.size() == 64);
        const std::size_t old_allocs = allocations_of([&] { hash = reference_sha256_hex(data); });
        CHECK(old_allocs > allocs);
    }

    if (argc > 1 && std::strcmp(argv[1], "--bench") == 0) return bench();
    std::printf("sha256_hex: ok (%s compression)\n", have_x86() ? "x86 SHA extensions" : "portable");
    return 0;
}
