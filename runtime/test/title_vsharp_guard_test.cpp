// Deterministic unit tests for leonida::title_vsharp_guard::redirect_untranslated_vsharp (no GPU or game files).
#include "../src/title_vsharp_guard.h"

#include <cstdio>
#include <cstdint>
#include <cstring>

namespace {

int g_checks = 0;
int g_failures = 0;

void check(bool cond, const char* what) {
    ++g_checks;
    if (!cond) {
        ++g_failures;
        std::fprintf(stderr, "FAIL: %s\n", what);
    }
}

leonida::gfx10::BufferWords make_buffer_vsharp(std::uint64_t base, std::uint32_t num_records,
                                               std::uint32_t stride = 16) {
    leonida::gfx10::Buffer b{};
    b.address = base;
    b.stride = stride;
    b.num_records = num_records;
    b.format = 4;  // R32_FLOAT
    return leonida::gfx10::encode(b);
}

}  // namespace

int main() {
    constexpr std::uint64_t kArenaSize = 0x40000000ull;  // 1 GiB
    constexpr std::uint64_t kScratchVa = 0x00020000ull;
    constexpr std::uint64_t kTitleBase = 0x0000015EB72CE000ull;
    constexpr std::uint64_t kGpuViewBase = 0x0000020010000000ull;
    constexpr std::uint64_t kKnownImportVa = 0x0000000180000000ull;

    const auto to_cpu = [&](std::uint64_t va, std::uint64_t) -> std::uint64_t {
        if (va == kGpuViewBase) return kTitleBase;
        return va;
    };
    const auto is_known = [&](std::uint64_t va) -> bool { return va == kKnownImportVa; };
    const auto is_title = [&](std::uint64_t va) -> bool {
        return va >= kTitleBase && va < kTitleBase + 0x1000000ull;
    };
    const auto get_scratch = [&]() -> std::uint64_t { return kScratchVa; };

    // 1. Null descriptor is untouched.
    {
        std::uint32_t w[4] = {0, 0, 0, 0};
        check(!leonida::title_vsharp_guard::redirect_untranslated_vsharp(
                  w, kArenaSize, to_cpu, is_known, is_title, get_scratch) &&
                  w[0] == 0 && w[1] == 0 && w[2] == 0 && w[3] == 0,
              "null descriptor untouched");
    }

    // 2. Image T# (w[3] >> 30 != 0) is untouched even if low bits match a title VA.
    {
        std::uint32_t w[4] = {std::uint32_t(kTitleBase), std::uint32_t(kTitleBase >> 32), 64, 0x80000000u};
        const std::uint32_t orig[4] = {w[0], w[1], w[2], w[3]};
        check(!leonida::title_vsharp_guard::redirect_untranslated_vsharp(
                  w, kArenaSize, to_cpu, is_known, is_title, get_scratch) &&
                  std::memcmp(w, orig, sizeof(w)) == 0,
              "image T# untouched");
    }

    // 3. Arena-resident buffer V# (base < arena_size) is untouched.
    {
        auto words = make_buffer_vsharp(0x00100000ull, 256, 16);
        const auto orig = words;
        check(!leonida::title_vsharp_guard::redirect_untranslated_vsharp(
                  words.data(), kArenaSize, to_cpu, is_known, is_title, get_scratch) &&
                  words == orig,
              "arena V# untouched");
    }

    // 4. Known GPU VA (host-imported buffer or geometry view) is untouched.
    {
        auto words = make_buffer_vsharp(kKnownImportVa, 256, 16);
        const auto orig = words;
        check(!leonida::title_vsharp_guard::redirect_untranslated_vsharp(
                  words.data(), kArenaSize, to_cpu, is_known, is_title, get_scratch) &&
                  words == orig,
              "known GPU VA untouched");
    }

    // 5. Address outside title memory (e.g. arbitrary non-title pointer) is untouched.
    {
        auto words = make_buffer_vsharp(0x0000030000000000ull, 64, 16);
        const auto orig = words;
        check(!leonida::title_vsharp_guard::redirect_untranslated_vsharp(
                  words.data(), kArenaSize, to_cpu, is_known, is_title, get_scratch) &&
                  words == orig,
              "non-title address untouched");
    }

    // 6. Unrebased raw title-memory V# is redirected to scratch with num_records=0, preserving stride & w[3].
    {
        auto words = make_buffer_vsharp(kTitleBase, 1024, 32);
        const std::uint32_t orig_w1_hi = words[1] & 0xFFFF0000u;
        const std::uint32_t orig_w3 = words[3];
        std::uint64_t raw_va = 0, span = 0;
        const bool guarded = leonida::title_vsharp_guard::redirect_untranslated_vsharp(
            words.data(), kArenaSize, to_cpu, is_known, is_title, get_scratch, &raw_va, &span);
        const std::uint64_t new_base = std::uint64_t(words[0]) | (std::uint64_t(words[1] & 0xFFFFu) << 32);
        check(guarded && new_base == kScratchVa && words[2] == 0 && (words[1] & 0xFFFF0000u) == orig_w1_hi &&
                  words[3] == orig_w3 && raw_va == kTitleBase && span == 1024ull * 32ull,
              "unrebased raw title V# redirected to zero-extent scratch");
    }

    // 7. Unrebased GPU-only title view (mapped via to_cpu into title memory) is also redirected safely.
    {
        auto words = make_buffer_vsharp(kGpuViewBase, 512, 16);
        std::uint64_t raw_va = 0, span = 0;
        const bool guarded = leonida::title_vsharp_guard::redirect_untranslated_vsharp(
            words.data(), kArenaSize, to_cpu, is_known, is_title, get_scratch, &raw_va, &span);
        const std::uint64_t new_base = std::uint64_t(words[0]) | (std::uint64_t(words[1] & 0xFFFFu) << 32);
        check(guarded && new_base == kScratchVa && words[2] == 0 && raw_va == kGpuViewBase && span == 512ull * 16ull,
              "unrebased GPU-only title view redirected to zero-extent scratch");
    }

    // 8. When no scratch slot is available (UINT64_MAX), returns false and leaves descriptor untouched.
    {
        auto words = make_buffer_vsharp(kTitleBase, 128, 16);
        const auto orig = words;
        check(!leonida::title_vsharp_guard::redirect_untranslated_vsharp(
                  words.data(), kArenaSize, to_cpu, is_known, is_title, [] { return UINT64_MAX; }) &&
                  words == orig,
              "unavailable scratch leaves V# untouched");
    }

    std::printf("title_vsharp_guard_test: %d checks, %d failures\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
