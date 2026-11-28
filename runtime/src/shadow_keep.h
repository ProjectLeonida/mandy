// What a shadow copy of the title's pages keeps of the GPU's writes (LEONIDA_SHADOW_KEEP, shims/
// d3d12_vulkan_device.cpp ShadowPages). Some pages are copied at every refresh without a sign that the title wrote
// them (mapped at two addresses, unguarded, reopened with the 64 KiB block around a written page, write-watched or
// pinned): copied whole, their unchanged bytes overwrote what a shader had stored in the copy since. Here the title's
// bytes are compared, per 256-byte piece of the title's address space, with a 64-bit sum of those copied last, and
// only a piece that differs is copied. Portable: no Windows headers, no Vulkan.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#if defined(__AVX2__)
#include <immintrin.h>
#elif defined(__x86_64__) || defined(_M_X64)
#include <emmintrin.h>
#endif

namespace leonida::shadow_keep {

inline constexpr std::size_t kPiece = 256;

// The sum of `n` bytes, never 0 (0 is "not known": the piece is copied at its next look).
// On AVX2 (x86-64-v3): 4 x 256-bit accumulators (128 bytes/step, XXH3-style vpmuludq + vpaddq with distinct evolving
// per-lane keys — 2 iterations per 256-byte piece).
// On baseline x86-64 (SSE2): 4 x 128-bit accumulators (64 bytes/step).
// Portable fallback: four independent 64-bit lanes over 32 bytes a step.
inline std::uint64_t sum_of(const std::uint8_t* bytes, std::size_t n) {
    constexpr std::uint64_t k = 0xff51afd7ed558ccdull;
    const auto rotl = [](std::uint64_t x, int r) { return (x << r) | (x >> (64 - r)); };
#if defined(__AVX2__)
    const __m256i k_step = _mm256_set_epi32(
        static_cast<int>(0x9e3779b1u), static_cast<int>(0x85ebca77u), static_cast<int>(0xc2b2ae3du),
        static_cast<int>(0x27d4eb2fu), static_cast<int>(0x165667b1u), static_cast<int>(0x9e3779f9u),
        static_cast<int>(0x85ebca87u), static_cast<int>(0xc2b2ae63u));
    __m256i k0 = _mm256_set_epi32(
        static_cast<int>(0x165667b1u), static_cast<int>(0x9e3779f9u), static_cast<int>(0x85ebca87u),
        static_cast<int>(0xc2b2ae63u), static_cast<int>(0x27d4eb4fu), static_cast<int>(0x165667c5u),
        static_cast<int>(0xff51afd7u), static_cast<int>(0xed558ccdu));
    __m256i k1 = _mm256_set_epi32(
        static_cast<int>(0xc4ceb9feu), static_cast<int>(0x1a85ec53u), static_cast<int>(0x94d049bbu),
        static_cast<int>(0x133111ebu), static_cast<int>(0xbf58476du), static_cast<int>(0x1ce4e5b9u),
        static_cast<int>(0x9e3779b9u), static_cast<int>(0x7f4a7c15u));
    __m256i k2 = _mm256_set_epi32(
        static_cast<int>(0xa0761d65u), static_cast<int>(0xe7037ed1u), static_cast<int>(0x8ebc6af1u),
        static_cast<int>(0x589965cbu), static_cast<int>(0x1d8e4e27u), static_cast<int>(0xeb44accbu),
        static_cast<int>(0x2d358dcdu), static_cast<int>(0x75567b7bu));
    __m256i k3 = _mm256_set_epi32(
        static_cast<int>(0xd6e8feb9u), static_cast<int>(0x6659fd93u), static_cast<int>(0xb492b66fu),
        static_cast<int>(0x3c79ac49u), static_cast<int>(0xf1357aeau), static_cast<int>(0x60642e2bu),
        static_cast<int>(0x49d4f7b3u), static_cast<int>(0xc8b86a9du));

    __m256i s0 = _mm256_set_epi64x(
        static_cast<long long>(0x27d4eb2f165667c5ull), static_cast<long long>(0x165667b19e3779f9ull),
        static_cast<long long>(0xc2b2ae3d27d4eb4full), static_cast<long long>(0x9e3779b97f4a7c15ull ^ n));
    __m256i s1 = _mm256_set_epi64x(
        static_cast<long long>(0x94d049bb133111ebull), static_cast<long long>(0xc4ceb9fe1a85ec53ull),
        static_cast<long long>(0xff51afd7ed558ccdull), static_cast<long long>(0x85ebca77c2b2ae63ull));
    __m256i s2 = _mm256_set_epi64x(
        static_cast<long long>(0xbf58476d1ce4e5b9ull), static_cast<long long>(0xa0761d6478bd642full),
        static_cast<long long>(0xe7037ed1a0b428dbull), static_cast<long long>(0x8ebc6af09c88c6e3ull));
    __m256i s3 = _mm256_set_epi64x(
        static_cast<long long>(0x589965cc75374cc3ull), static_cast<long long>(0x1d8e4e27c47d124full),
        static_cast<long long>(0xeb44accab455d165ull), static_cast<long long>(0xd6e8feb86659fd93ull));

    const auto lane = [](__m256i acc, __m256i data, __m256i key) -> __m256i {
        const __m256i x = _mm256_xor_si256(data, key);
        const __m256i x_hi = _mm256_srli_epi64(x, 32);
        const __m256i prod = _mm256_mul_epu32(x, x_hi);
        const __m256i d_swap = _mm256_shuffle_epi32(data, _MM_SHUFFLE(1, 0, 3, 2));
        return _mm256_add_epi64(_mm256_add_epi64(acc, d_swap), prod);
    };

    std::size_t i = 0;
    for (; i + 128 <= n; i += 128) {
        const auto* p = reinterpret_cast<const __m256i*>(bytes + i);
        s0 = lane(s0, _mm256_loadu_si256(p + 0), k0);
        s1 = lane(s1, _mm256_loadu_si256(p + 1), k1);
        s2 = lane(s2, _mm256_loadu_si256(p + 2), k2);
        s3 = lane(s3, _mm256_loadu_si256(p + 3), k3);
        k0 = _mm256_add_epi32(k0, k_step);
        k1 = _mm256_add_epi32(k1, k_step);
        k2 = _mm256_add_epi32(k2, k_step);
        k3 = _mm256_add_epi32(k3, k_step);
    }

    const __m256i m0 = _mm256_xor_si256(s0, _mm256_shuffle_epi32(s2, _MM_SHUFFLE(1, 0, 3, 2)));
    const __m256i m1 = _mm256_xor_si256(s1, _mm256_shuffle_epi32(s3, _MM_SHUFFLE(1, 0, 3, 2)));
    const __m128i f0 = _mm_xor_si128(_mm256_castsi256_si128(m0),
                                     _mm_shuffle_epi32(_mm256_extracti128_si256(m0, 1), _MM_SHUFFLE(1, 0, 3, 2)));
    const __m128i f1 = _mm_xor_si128(_mm256_castsi256_si128(m1),
                                     _mm_shuffle_epi32(_mm256_extracti128_si256(m1, 1), _MM_SHUFFLE(1, 0, 3, 2)));
    std::uint64_t a[4];
    _mm_storeu_si128(reinterpret_cast<__m128i*>(a), f0);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(a + 2), f1);

    for (; i < n; i += 8) {
        std::uint64_t v = 0;
        std::memcpy(&v, bytes + i, n - i < 8 ? n - i : 8);
        a[0] = rotl(a[0] ^ v, 31) * k;
    }
    std::uint64_t h = (a[0] * k) ^ rotl(a[1] * 0xc4ceb9fe1a85ec53ull, 17) ^
                      rotl(a[2] * 0x94d049bb133111ebull, 29) ^ rotl(a[3] * 0xbf58476d1ce4e5b9ull, 47);
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ull;
    h ^= h >> 29;
    return h | 1;  // 0: not known
#elif defined(__x86_64__) || defined(_M_X64)
    const __m128i k_step =
        _mm_set_epi32(static_cast<int>(0x9e3779b1u), static_cast<int>(0x85ebca77u), static_cast<int>(0xc2b2ae3du),
                      static_cast<int>(0x27d4eb2fu));
    __m128i k0 =
        _mm_set_epi32(static_cast<int>(0x165667b1u), static_cast<int>(0x9e3779f9u), static_cast<int>(0x85ebca87u),
                      static_cast<int>(0xc2b2ae63u));
    __m128i k1 =
        _mm_set_epi32(static_cast<int>(0x27d4eb4fu), static_cast<int>(0x165667c5u), static_cast<int>(0xff51afd7u),
                      static_cast<int>(0xed558ccdu));
    __m128i k2 =
        _mm_set_epi32(static_cast<int>(0xc4ceb9feu), static_cast<int>(0x1a85ec53u), static_cast<int>(0x94d049bbu),
                      static_cast<int>(0x133111ebu));
    __m128i k3 =
        _mm_set_epi32(static_cast<int>(0xbf58476du), static_cast<int>(0x1ce4e5b9u), static_cast<int>(0x9e3779b9u),
                      static_cast<int>(0x7f4a7c15u));

    __m128i s0 = _mm_set_epi64x(static_cast<long long>(0xc2b2ae3d27d4eb4full),
                                static_cast<long long>(0x9e3779b97f4a7c15ull ^ n));
    __m128i s1 = _mm_set_epi64x(static_cast<long long>(0x165667b19e3779f9ull),
                                static_cast<long long>(0x27d4eb2f165667c5ull));
    __m128i s2 = _mm_set_epi64x(static_cast<long long>(0x85ebca77c2b2ae63ull),
                                static_cast<long long>(0xff51afd7ed558ccdull));
    __m128i s3 = _mm_set_epi64x(static_cast<long long>(0xc4ceb9fe1a85ec53ull),
                                static_cast<long long>(0x94d049bb133111ebull));

    const auto lane = [](__m128i acc, __m128i data, __m128i key) -> __m128i {
        const __m128i x = _mm_xor_si128(data, key);
        const __m128i x_hi = _mm_srli_epi64(x, 32);
        const __m128i prod = _mm_mul_epu32(x, x_hi);
        const __m128i d_swap = _mm_shuffle_epi32(data, _MM_SHUFFLE(1, 0, 3, 2));
        return _mm_add_epi64(_mm_add_epi64(acc, d_swap), prod);
    };

    std::size_t i = 0;
    for (; i + 64 <= n; i += 64) {
        const auto* p = reinterpret_cast<const __m128i*>(bytes + i);
        s0 = lane(s0, _mm_loadu_si128(p + 0), k0);
        s1 = lane(s1, _mm_loadu_si128(p + 1), k1);
        s2 = lane(s2, _mm_loadu_si128(p + 2), k2);
        s3 = lane(s3, _mm_loadu_si128(p + 3), k3);
        k0 = _mm_add_epi32(k0, k_step);
        k1 = _mm_add_epi32(k1, k_step);
        k2 = _mm_add_epi32(k2, k_step);
        k3 = _mm_add_epi32(k3, k_step);
    }

    const __m128i m0 = _mm_xor_si128(s0, _mm_shuffle_epi32(s2, _MM_SHUFFLE(1, 0, 3, 2)));
    const __m128i m1 = _mm_xor_si128(s1, _mm_shuffle_epi32(s3, _MM_SHUFFLE(1, 0, 3, 2)));
    std::uint64_t a[4];
    _mm_storeu_si128(reinterpret_cast<__m128i*>(a), m0);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(a + 2), m1);

    for (; i < n; i += 8) {
        std::uint64_t v = 0;
        std::memcpy(&v, bytes + i, n - i < 8 ? n - i : 8);
        a[0] = rotl(a[0] ^ v, 31) * k;
    }
    std::uint64_t h = (a[0] * k) ^ rotl(a[1] * 0xc4ceb9fe1a85ec53ull, 17) ^
                      rotl(a[2] * 0x94d049bb133111ebull, 29) ^ rotl(a[3] * 0xbf58476d1ce4e5b9ull, 47);
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ull;
    h ^= h >> 29;
    return h | 1;  // 0: not known
#else
    std::uint64_t a0 = 0x9e3779b97f4a7c15ull ^ n, a1 = 0xc2b2ae3d27d4eb4full, a2 = 0x165667b19e3779f9ull,
                  a3 = 0x27d4eb2f165667c5ull;
    std::size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        std::uint64_t v[4];
        std::memcpy(v, bytes + i, 32);
        a0 = rotl(a0 ^ v[0], 31) * k;
        a1 = rotl(a1 ^ v[1], 31) * k;
        a2 = rotl(a2 ^ v[2], 31) * k;
        a3 = rotl(a3 ^ v[3], 31) * k;
    }
    for (; i < n; i += 8) {
        std::uint64_t v = 0;
        std::memcpy(&v, bytes + i, n - i < 8 ? n - i : 8);
        a0 = rotl(a0 ^ v, 31) * k;
    }
    std::uint64_t h = a0 ^ rotl(a1, 17) ^ rotl(a2, 29) ^ rotl(a3, 47);
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ull;
    h ^= h >> 29;
    return h | 1;  // 0: not known
#endif
}

// Computes `count` consecutive 256-byte piece hashes `out[i] = sum_of(base + i * kPiece, kPiece)` with hoisted SIMD
// constants. Guaranteed bit-for-bit identical to calling `sum_of(base + i * kPiece, kPiece)` in a loop.
inline void sum_of_batch_256(const std::uint8_t* base, std::size_t count, std::uint64_t* out) {
    if (!count) return;
    constexpr std::uint64_t k = 0xff51afd7ed558ccdull;
    const auto rotl = [](std::uint64_t x, int r) { return (x << r) | (x >> (64 - r)); };
#if defined(__AVX2__)
    const __m256i k_step = _mm256_set_epi32(
        static_cast<int>(0x9e3779b1u), static_cast<int>(0x85ebca77u), static_cast<int>(0xc2b2ae3du),
        static_cast<int>(0x27d4eb2fu), static_cast<int>(0x165667b1u), static_cast<int>(0x9e3779f9u),
        static_cast<int>(0x85ebca87u), static_cast<int>(0xc2b2ae63u));
    const __m256i k0_0 = _mm256_set_epi32(
        static_cast<int>(0x165667b1u), static_cast<int>(0x9e3779f9u), static_cast<int>(0x85ebca87u),
        static_cast<int>(0xc2b2ae63u), static_cast<int>(0x27d4eb4fu), static_cast<int>(0x165667c5u),
        static_cast<int>(0xff51afd7u), static_cast<int>(0xed558ccdu));
    const __m256i k1_0 = _mm256_set_epi32(
        static_cast<int>(0xc4ceb9feu), static_cast<int>(0x1a85ec53u), static_cast<int>(0x94d049bbu),
        static_cast<int>(0x133111ebu), static_cast<int>(0xbf58476du), static_cast<int>(0x1ce4e5b9u),
        static_cast<int>(0x9e3779b9u), static_cast<int>(0x7f4a7c15u));
    const __m256i k2_0 = _mm256_set_epi32(
        static_cast<int>(0xa0761d65u), static_cast<int>(0xe7037ed1u), static_cast<int>(0x8ebc6af1u),
        static_cast<int>(0x589965cbu), static_cast<int>(0x1d8e4e27u), static_cast<int>(0xeb44accbu),
        static_cast<int>(0x2d358dcdu), static_cast<int>(0x75567b7bu));
    const __m256i k3_0 = _mm256_set_epi32(
        static_cast<int>(0xd6e8feb9u), static_cast<int>(0x6659fd93u), static_cast<int>(0xb492b66fu),
        static_cast<int>(0x3c79ac49u), static_cast<int>(0xf1357aeau), static_cast<int>(0x60642e2bu),
        static_cast<int>(0x49d4f7b3u), static_cast<int>(0xc8b86a9du));
    const __m256i k0_1 = _mm256_add_epi32(k0_0, k_step);
    const __m256i k1_1 = _mm256_add_epi32(k1_0, k_step);
    const __m256i k2_1 = _mm256_add_epi32(k2_0, k_step);
    const __m256i k3_1 = _mm256_add_epi32(k3_0, k_step);

    const __m256i s0_init = _mm256_set_epi64x(
        static_cast<long long>(0x27d4eb2f165667c5ull), static_cast<long long>(0x165667b19e3779f9ull),
        static_cast<long long>(0xc2b2ae3d27d4eb4full), static_cast<long long>(0x9e3779b97f4a7c15ull ^ kPiece));
    const __m256i s1_init = _mm256_set_epi64x(
        static_cast<long long>(0x94d049bb133111ebull), static_cast<long long>(0xc4ceb9fe1a85ec53ull),
        static_cast<long long>(0xff51afd7ed558ccdull), static_cast<long long>(0x85ebca77c2b2ae63ull));
    const __m256i s2_init = _mm256_set_epi64x(
        static_cast<long long>(0xbf58476d1ce4e5b9ull), static_cast<long long>(0xa0761d6478bd642full),
        static_cast<long long>(0xe7037ed1a0b428dbull), static_cast<long long>(0x8ebc6af09c88c6e3ull));
    const __m256i s3_init = _mm256_set_epi64x(
        static_cast<long long>(0x589965cc75374cc3ull), static_cast<long long>(0x1d8e4e27c47d124full),
        static_cast<long long>(0xeb44accab455d165ull), static_cast<long long>(0xd6e8feb86659fd93ull));

    const auto lane = [](__m256i acc, __m256i data, __m256i key) -> __m256i {
        const __m256i x = _mm256_xor_si256(data, key);
        const __m256i x_hi = _mm256_srli_epi64(x, 32);
        const __m256i prod = _mm256_mul_epu32(x, x_hi);
        const __m256i d_swap = _mm256_shuffle_epi32(data, _MM_SHUFFLE(1, 0, 3, 2));
        return _mm256_add_epi64(_mm256_add_epi64(acc, d_swap), prod);
    };

    for (std::size_t idx = 0; idx < count; ++idx) {
        const auto* p = reinterpret_cast<const __m256i*>(base + idx * kPiece);
        __m256i s0 = lane(s0_init, _mm256_loadu_si256(p + 0), k0_0);
        __m256i s1 = lane(s1_init, _mm256_loadu_si256(p + 1), k1_0);
        __m256i s2 = lane(s2_init, _mm256_loadu_si256(p + 2), k2_0);
        __m256i s3 = lane(s3_init, _mm256_loadu_si256(p + 3), k3_0);

        s0 = lane(s0, _mm256_loadu_si256(p + 4), k0_1);
        s1 = lane(s1, _mm256_loadu_si256(p + 5), k1_1);
        s2 = lane(s2, _mm256_loadu_si256(p + 6), k2_1);
        s3 = lane(s3, _mm256_loadu_si256(p + 7), k3_1);

        const __m256i m0 = _mm256_xor_si256(s0, _mm256_shuffle_epi32(s2, _MM_SHUFFLE(1, 0, 3, 2)));
        const __m256i m1 = _mm256_xor_si256(s1, _mm256_shuffle_epi32(s3, _MM_SHUFFLE(1, 0, 3, 2)));
        const __m128i f0 = _mm_xor_si128(
            _mm256_castsi256_si128(m0),
            _mm_shuffle_epi32(_mm256_extracti128_si256(m0, 1), _MM_SHUFFLE(1, 0, 3, 2)));
        const __m128i f1 = _mm_xor_si128(
            _mm256_castsi256_si128(m1),
            _mm_shuffle_epi32(_mm256_extracti128_si256(m1, 1), _MM_SHUFFLE(1, 0, 3, 2)));
        std::uint64_t a[4];
        _mm_storeu_si128(reinterpret_cast<__m128i*>(a), f0);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(a + 2), f1);

        std::uint64_t h = (a[0] * k) ^ rotl(a[1] * 0xc4ceb9fe1a85ec53ull, 17) ^
                          rotl(a[2] * 0x94d049bb133111ebull, 29) ^ rotl(a[3] * 0xbf58476d1ce4e5b9ull, 47);
        h ^= h >> 33;
        h *= 0xc4ceb9fe1a85ec53ull;
        h ^= h >> 29;
        out[idx] = h | 1;
    }
#elif defined(__x86_64__) || defined(_M_X64)
    const __m128i k_step =
        _mm_set_epi32(static_cast<int>(0x9e3779b1u), static_cast<int>(0x85ebca77u), static_cast<int>(0xc2b2ae3du),
                      static_cast<int>(0x27d4eb2fu));
    const __m128i k0_init =
        _mm_set_epi32(static_cast<int>(0x165667b1u), static_cast<int>(0x9e3779f9u), static_cast<int>(0x85ebca87u),
                      static_cast<int>(0xc2b2ae63u));
    const __m128i k1_init =
        _mm_set_epi32(static_cast<int>(0x27d4eb4fu), static_cast<int>(0x165667c5u), static_cast<int>(0xff51afd7u),
                      static_cast<int>(0xed558ccdu));
    const __m128i k2_init =
        _mm_set_epi32(static_cast<int>(0xc4ceb9feu), static_cast<int>(0x1a85ec53u), static_cast<int>(0x94d049bbu),
                      static_cast<int>(0x133111ebu));
    const __m128i k3_init =
        _mm_set_epi32(static_cast<int>(0xbf58476du), static_cast<int>(0x1ce4e5b9u), static_cast<int>(0x9e3779b9u),
                      static_cast<int>(0x7f4a7c15u));

    const __m128i s0_init = _mm_set_epi64x(static_cast<long long>(0xc2b2ae3d27d4eb4full),
                                           static_cast<long long>(0x9e3779b97f4a7c15ull ^ kPiece));
    const __m128i s1_init = _mm_set_epi64x(static_cast<long long>(0x165667b19e3779f9ull),
                                           static_cast<long long>(0x27d4eb2f165667c5ull));
    const __m128i s2_init = _mm_set_epi64x(static_cast<long long>(0x85ebca77c2b2ae63ull),
                                           static_cast<long long>(0xff51afd7ed558ccdull));
    const __m128i s3_init = _mm_set_epi64x(static_cast<long long>(0xc4ceb9fe1a85ec53ull),
                                           static_cast<long long>(0x94d049bb133111ebull));

    const auto lane = [](__m128i acc, __m128i data, __m128i key) -> __m128i {
        const __m128i x = _mm_xor_si128(data, key);
        const __m128i x_hi = _mm_srli_epi64(x, 32);
        const __m128i prod = _mm_mul_epu32(x, x_hi);
        const __m128i d_swap = _mm_shuffle_epi32(data, _MM_SHUFFLE(1, 0, 3, 2));
        return _mm_add_epi64(_mm_add_epi64(acc, d_swap), prod);
    };

    for (std::size_t idx = 0; idx < count; ++idx) {
        const auto* p = reinterpret_cast<const __m128i*>(base + idx * kPiece);
        __m128i k0 = k0_init, k1 = k1_init, k2 = k2_init, k3 = k3_init;
        __m128i s0 = s0_init, s1 = s1_init, s2 = s2_init, s3 = s3_init;
        for (std::size_t step = 0; step < 4; ++step, p += 4) {
            s0 = lane(s0, _mm_loadu_si128(p + 0), k0);
            s1 = lane(s1, _mm_loadu_si128(p + 1), k1);
            s2 = lane(s2, _mm_loadu_si128(p + 2), k2);
            s3 = lane(s3, _mm_loadu_si128(p + 3), k3);
            k0 = _mm_add_epi32(k0, k_step);
            k1 = _mm_add_epi32(k1, k_step);
            k2 = _mm_add_epi32(k2, k_step);
            k3 = _mm_add_epi32(k3, k_step);
        }
        const __m128i m0 = _mm_xor_si128(s0, _mm_shuffle_epi32(s2, _MM_SHUFFLE(1, 0, 3, 2)));
        const __m128i m1 = _mm_xor_si128(s1, _mm_shuffle_epi32(s3, _MM_SHUFFLE(1, 0, 3, 2)));
        std::uint64_t a[4];
        _mm_storeu_si128(reinterpret_cast<__m128i*>(a), m0);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(a + 2), m1);

        std::uint64_t h = (a[0] * k) ^ rotl(a[1] * 0xc4ceb9fe1a85ec53ull, 17) ^
                          rotl(a[2] * 0x94d049bb133111ebull, 29) ^ rotl(a[3] * 0xbf58476d1ce4e5b9ull, 47);
        h ^= h >> 33;
        h *= 0xc4ceb9fe1a85ec53ull;
        h ^= h >> 29;
        out[idx] = h | 1;
    }
#else
    constexpr std::uint64_t a0_init = 0x9e3779b97f4a7c15ull ^ kPiece;
    constexpr std::uint64_t a1_init = 0xc2b2ae3d27d4eb4full;
    constexpr std::uint64_t a2_init = 0x165667b19e3779f9ull;
    constexpr std::uint64_t a3_init = 0x27d4eb2f165667c5ull;
    for (std::size_t idx = 0; idx < count; ++idx) {
        const std::uint8_t* bytes = base + idx * kPiece;
        std::uint64_t a0 = a0_init, a1 = a1_init, a2 = a2_init, a3 = a3_init;
        for (std::size_t i = 0; i < kPiece; i += 32) {
            std::uint64_t v[4];
            std::memcpy(v, bytes + i, 32);
            a0 = rotl(a0 ^ v[0], 31) * k;
            a1 = rotl(a1 ^ v[1], 31) * k;
            a2 = rotl(a2 ^ v[2], 31) * k;
            a3 = rotl(a3 ^ v[3], 31) * k;
        }
        std::uint64_t h = a0 ^ rotl(a1, 17) ^ rotl(a2, 29) ^ rotl(a3, 47);
        h ^= h >> 33;
        h *= 0xc4ceb9fe1a85ec53ull;
        h ^= h >> 29;
        out[idx] = h | 1;
    }
#endif
}

// Fused copy + 256-byte piece hashing: copies `count * kPiece` bytes from `from` to `to` while computing
// `out[i] = sum_of(from + i * kPiece, kPiece)` in registers (one read pass over `from`, zero second pass over `to`).
// Guaranteed bit-for-bit identical to `std::memcpy(to, from, count * kPiece)` + `sum_of_batch_256(to, count, out)`.
// Intentionally NOT noexcept so SEH wrappers under /EHsc can invoke it inside a __try frame.
inline void copy_and_sum_256(std::uint8_t* to, const std::uint8_t* from, std::size_t count, std::uint64_t* out) {
    if (!count) return;
    constexpr std::uint64_t k = 0xff51afd7ed558ccdull;
    const auto rotl = [](std::uint64_t x, int r) { return (x << r) | (x >> (64 - r)); };
#if defined(__AVX2__)
    const __m256i k_step = _mm256_set_epi32(
        static_cast<int>(0x9e3779b1u), static_cast<int>(0x85ebca77u), static_cast<int>(0xc2b2ae3du),
        static_cast<int>(0x27d4eb2fu), static_cast<int>(0x165667b1u), static_cast<int>(0x9e3779f9u),
        static_cast<int>(0x85ebca87u), static_cast<int>(0xc2b2ae63u));
    const __m256i k0_0 = _mm256_set_epi32(
        static_cast<int>(0x165667b1u), static_cast<int>(0x9e3779f9u), static_cast<int>(0x85ebca87u),
        static_cast<int>(0xc2b2ae63u), static_cast<int>(0x27d4eb4fu), static_cast<int>(0x165667c5u),
        static_cast<int>(0xff51afd7u), static_cast<int>(0xed558ccdu));
    const __m256i k1_0 = _mm256_set_epi32(
        static_cast<int>(0xc4ceb9feu), static_cast<int>(0x1a85ec53u), static_cast<int>(0x94d049bbu),
        static_cast<int>(0x133111ebu), static_cast<int>(0xbf58476du), static_cast<int>(0x1ce4e5b9u),
        static_cast<int>(0x9e3779b9u), static_cast<int>(0x7f4a7c15u));
    const __m256i k2_0 = _mm256_set_epi32(
        static_cast<int>(0xa0761d65u), static_cast<int>(0xe7037ed1u), static_cast<int>(0x8ebc6af1u),
        static_cast<int>(0x589965cbu), static_cast<int>(0x1d8e4e27u), static_cast<int>(0xeb44accbu),
        static_cast<int>(0x2d358dcdu), static_cast<int>(0x75567b7bu));
    const __m256i k3_0 = _mm256_set_epi32(
        static_cast<int>(0xd6e8feb9u), static_cast<int>(0x6659fd93u), static_cast<int>(0xb492b66fu),
        static_cast<int>(0x3c79ac49u), static_cast<int>(0xf1357aeau), static_cast<int>(0x60642e2bu),
        static_cast<int>(0x49d4f7b3u), static_cast<int>(0xc8b86a9du));
    const __m256i k0_1 = _mm256_add_epi32(k0_0, k_step);
    const __m256i k1_1 = _mm256_add_epi32(k1_0, k_step);
    const __m256i k2_1 = _mm256_add_epi32(k2_0, k_step);
    const __m256i k3_1 = _mm256_add_epi32(k3_0, k_step);

    const __m256i s0_init = _mm256_set_epi64x(
        static_cast<long long>(0x27d4eb2f165667c5ull), static_cast<long long>(0x165667b19e3779f9ull),
        static_cast<long long>(0xc2b2ae3d27d4eb4full), static_cast<long long>(0x9e3779b97f4a7c15ull ^ kPiece));
    const __m256i s1_init = _mm256_set_epi64x(
        static_cast<long long>(0x94d049bb133111ebull), static_cast<long long>(0xc4ceb9fe1a85ec53ull),
        static_cast<long long>(0xff51afd7ed558ccdull), static_cast<long long>(0x85ebca77c2b2ae63ull));
    const __m256i s2_init = _mm256_set_epi64x(
        static_cast<long long>(0xbf58476d1ce4e5b9ull), static_cast<long long>(0xa0761d6478bd642full),
        static_cast<long long>(0xe7037ed1a0b428dbull), static_cast<long long>(0x8ebc6af09c88c6e3ull));
    const __m256i s3_init = _mm256_set_epi64x(
        static_cast<long long>(0x589965cc75374cc3ull), static_cast<long long>(0x1d8e4e27c47d124full),
        static_cast<long long>(0xeb44accab455d165ull), static_cast<long long>(0xd6e8feb86659fd93ull));

    const auto lane = [](__m256i acc, __m256i data, __m256i key) -> __m256i {
        const __m256i x = _mm256_xor_si256(data, key);
        const __m256i x_hi = _mm256_srli_epi64(x, 32);
        const __m256i prod = _mm256_mul_epu32(x, x_hi);
        const __m256i d_swap = _mm256_shuffle_epi32(data, _MM_SHUFFLE(1, 0, 3, 2));
        return _mm256_add_epi64(_mm256_add_epi64(acc, d_swap), prod);
    };

    for (std::size_t idx = 0; idx < count; ++idx) {
        const auto* p = reinterpret_cast<const __m256i*>(from + idx * kPiece);
        auto* d = reinterpret_cast<__m256i*>(to + idx * kPiece);
        const __m256i v0 = _mm256_loadu_si256(p + 0);
        const __m256i v1 = _mm256_loadu_si256(p + 1);
        const __m256i v2 = _mm256_loadu_si256(p + 2);
        const __m256i v3 = _mm256_loadu_si256(p + 3);
        _mm256_storeu_si256(d + 0, v0);
        _mm256_storeu_si256(d + 1, v1);
        _mm256_storeu_si256(d + 2, v2);
        _mm256_storeu_si256(d + 3, v3);
        __m256i s0 = lane(s0_init, v0, k0_0);
        __m256i s1 = lane(s1_init, v1, k1_0);
        __m256i s2 = lane(s2_init, v2, k2_0);
        __m256i s3 = lane(s3_init, v3, k3_0);

        const __m256i v4 = _mm256_loadu_si256(p + 4);
        const __m256i v5 = _mm256_loadu_si256(p + 5);
        const __m256i v6 = _mm256_loadu_si256(p + 6);
        const __m256i v7 = _mm256_loadu_si256(p + 7);
        _mm256_storeu_si256(d + 4, v4);
        _mm256_storeu_si256(d + 5, v5);
        _mm256_storeu_si256(d + 6, v6);
        _mm256_storeu_si256(d + 7, v7);
        s0 = lane(s0, v4, k0_1);
        s1 = lane(s1, v5, k1_1);
        s2 = lane(s2, v6, k2_1);
        s3 = lane(s3, v7, k3_1);

        const __m256i m0 = _mm256_xor_si256(s0, _mm256_shuffle_epi32(s2, _MM_SHUFFLE(1, 0, 3, 2)));
        const __m256i m1 = _mm256_xor_si256(s1, _mm256_shuffle_epi32(s3, _MM_SHUFFLE(1, 0, 3, 2)));
        const __m128i f0 = _mm_xor_si128(
            _mm256_castsi256_si128(m0),
            _mm_shuffle_epi32(_mm256_extracti128_si256(m0, 1), _MM_SHUFFLE(1, 0, 3, 2)));
        const __m128i f1 = _mm_xor_si128(
            _mm256_castsi256_si128(m1),
            _mm_shuffle_epi32(_mm256_extracti128_si256(m1, 1), _MM_SHUFFLE(1, 0, 3, 2)));
        std::uint64_t a[4];
        _mm_storeu_si128(reinterpret_cast<__m128i*>(a), f0);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(a + 2), f1);

        std::uint64_t h = (a[0] * k) ^ rotl(a[1] * 0xc4ceb9fe1a85ec53ull, 17) ^
                          rotl(a[2] * 0x94d049bb133111ebull, 29) ^ rotl(a[3] * 0xbf58476d1ce4e5b9ull, 47);
        h ^= h >> 33;
        h *= 0xc4ceb9fe1a85ec53ull;
        h ^= h >> 29;
        out[idx] = h | 1;
    }
#elif defined(__x86_64__) || defined(_M_X64)
    const __m128i k_step =
        _mm_set_epi32(static_cast<int>(0x9e3779b1u), static_cast<int>(0x85ebca77u), static_cast<int>(0xc2b2ae3du),
                      static_cast<int>(0x27d4eb2fu));
    const __m128i k0_init =
        _mm_set_epi32(static_cast<int>(0x165667b1u), static_cast<int>(0x9e3779f9u), static_cast<int>(0x85ebca87u),
                      static_cast<int>(0xc2b2ae63u));
    const __m128i k1_init =
        _mm_set_epi32(static_cast<int>(0x27d4eb4fu), static_cast<int>(0x165667c5u), static_cast<int>(0xff51afd7u),
                      static_cast<int>(0xed558ccdu));
    const __m128i k2_init =
        _mm_set_epi32(static_cast<int>(0xc4ceb9feu), static_cast<int>(0x1a85ec53u), static_cast<int>(0x94d049bbu),
                      static_cast<int>(0x133111ebu));
    const __m128i k3_init =
        _mm_set_epi32(static_cast<int>(0xbf58476du), static_cast<int>(0x1ce4e5b9u), static_cast<int>(0x9e3779b9u),
                      static_cast<int>(0x7f4a7c15u));

    const __m128i s0_init = _mm_set_epi64x(static_cast<long long>(0xc2b2ae3d27d4eb4full),
                                           static_cast<long long>(0x9e3779b97f4a7c15ull ^ kPiece));
    const __m128i s1_init = _mm_set_epi64x(static_cast<long long>(0x165667b19e3779f9ull),
                                           static_cast<long long>(0x27d4eb2f165667c5ull));
    const __m128i s2_init = _mm_set_epi64x(static_cast<long long>(0x85ebca77c2b2ae63ull),
                                           static_cast<long long>(0xff51afd7ed558ccdull));
    const __m128i s3_init = _mm_set_epi64x(static_cast<long long>(0xc4ceb9fe1a85ec53ull),
                                           static_cast<long long>(0x94d049bb133111ebull));

    const auto lane = [](__m128i acc, __m128i data, __m128i key) -> __m128i {
        const __m128i x = _mm_xor_si128(data, key);
        const __m128i x_hi = _mm_srli_epi64(x, 32);
        const __m128i prod = _mm_mul_epu32(x, x_hi);
        const __m128i d_swap = _mm_shuffle_epi32(data, _MM_SHUFFLE(1, 0, 3, 2));
        return _mm_add_epi64(_mm_add_epi64(acc, d_swap), prod);
    };

    for (std::size_t idx = 0; idx < count; ++idx) {
        const auto* p = reinterpret_cast<const __m128i*>(from + idx * kPiece);
        auto* d = reinterpret_cast<__m128i*>(to + idx * kPiece);
        __m128i k0 = k0_init, k1 = k1_init, k2 = k2_init, k3 = k3_init;
        __m128i s0 = s0_init, s1 = s1_init, s2 = s2_init, s3 = s3_init;
        for (std::size_t step = 0; step < 4; ++step, p += 4, d += 4) {
            const __m128i v0 = _mm_loadu_si128(p + 0);
            const __m128i v1 = _mm_loadu_si128(p + 1);
            const __m128i v2 = _mm_loadu_si128(p + 2);
            const __m128i v3 = _mm_loadu_si128(p + 3);
            _mm_storeu_si128(d + 0, v0);
            _mm_storeu_si128(d + 1, v1);
            _mm_storeu_si128(d + 2, v2);
            _mm_storeu_si128(d + 3, v3);
            s0 = lane(s0, v0, k0);
            s1 = lane(s1, v1, k1);
            s2 = lane(s2, v2, k2);
            s3 = lane(s3, v3, k3);
            k0 = _mm_add_epi32(k0, k_step);
            k1 = _mm_add_epi32(k1, k_step);
            k2 = _mm_add_epi32(k2, k_step);
            k3 = _mm_add_epi32(k3, k_step);
        }
        const __m128i m0 = _mm_xor_si128(s0, _mm_shuffle_epi32(s2, _MM_SHUFFLE(1, 0, 3, 2)));
        const __m128i m1 = _mm_xor_si128(s1, _mm_shuffle_epi32(s3, _MM_SHUFFLE(1, 0, 3, 2)));
        std::uint64_t a[4];
        _mm_storeu_si128(reinterpret_cast<__m128i*>(a), m0);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(a + 2), m1);

        std::uint64_t h = (a[0] * k) ^ rotl(a[1] * 0xc4ceb9fe1a85ec53ull, 17) ^
                          rotl(a[2] * 0x94d049bb133111ebull, 29) ^ rotl(a[3] * 0xbf58476d1ce4e5b9ull, 47);
        h ^= h >> 33;
        h *= 0xc4ceb9fe1a85ec53ull;
        h ^= h >> 29;
        out[idx] = h | 1;
    }
#else
    constexpr std::uint64_t a0_init = 0x9e3779b97f4a7c15ull ^ kPiece;
    constexpr std::uint64_t a1_init = 0xc2b2ae3d27d4eb4full;
    constexpr std::uint64_t a2_init = 0x165667b19e3779f9ull;
    constexpr std::uint64_t a3_init = 0x27d4eb2f165667c5ull;
    for (std::size_t idx = 0; idx < count; ++idx) {
        const std::uint8_t* bytes = from + idx * kPiece;
        std::uint8_t* dst = to + idx * kPiece;
        std::uint64_t a0 = a0_init, a1 = a1_init, a2 = a2_init, a3 = a3_init;
        for (std::size_t i = 0; i < kPiece; i += 32) {
            std::uint64_t v[4];
            std::memcpy(v, bytes + i, 32);
            std::memcpy(dst + i, v, 32);
            a0 = rotl(a0 ^ v[0], 31) * k;
            a1 = rotl(a1 ^ v[1], 31) * k;
            a2 = rotl(a2 ^ v[2], 31) * k;
            a3 = rotl(a3 ^ v[3], 31) * k;
        }
        std::uint64_t h = a0 ^ rotl(a1, 17) ^ rotl(a2, 29) ^ rotl(a3, 47);
        h ^= h >> 33;
        h *= 0xc4ceb9fe1a85ec53ull;
        h ^= h >> 29;
        out[idx] = h | 1;
    }
#endif
}

// `bytes` holds the title's bytes [va, va + n) as read now. sums[k] is the sum of the piece at base + k * kPiece
// (base kPiece-aligned) as last copied. Each piece, cut at kPiece boundaries of the address (the first and last may be
// partial), whose bytes differ from its sum takes the new sum and is handed to copy(from, to, source).
template <class Copy>
void copy_changed(const std::uint8_t* bytes, std::uintptr_t va, std::size_t n, std::uint64_t* sums,
                  std::uintptr_t base, Copy&& copy) {
    for (std::size_t off = 0; off < n;) {
        const std::size_t piece = (std::min)(n - off, std::size_t(kPiece - ((va + off) & (kPiece - 1))));
        std::uint64_t& known = sums[(va + off - base) / kPiece];
        const std::uint64_t sum = sum_of(bytes + off, piece);
        if (sum != known) {
            known = sum;
            copy(va + off, va + off + piece, bytes + off);
        }
        off += piece;
    }
}

// The pieces over [from, to) are not known any more (the copies were changed another way, as zeroed): each is copied
// at its next look.
inline void forget(std::uint64_t* sums, std::uintptr_t base, std::uintptr_t from, std::uintptr_t to) {
    if (from >= to) return;
    std::fill(sums + (from - base) / kPiece, sums + (to - base + kPiece - 1) / kPiece, std::uint64_t(0));
}

}  // namespace leonida::shadow_keep
