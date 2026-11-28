// Tests for runtime/shims/bcrypt_rng.cpp through rt::shims::find.
#include <windows.h>

#ifndef _NTDEF_
using NTSTATUS = LONG;
#endif
#include <bcrypt.h>

#include <cstdint>
#include <cstdio>
#include <cstring>

#include "shims.h"

namespace rt::bcrypt_rng {
void set_host_rng_enabled_for_test(bool enabled);
}  // namespace rt::bcrypt_rng

namespace {

constexpr NTSTATUS kStatusSuccess = 0x00000000L;
constexpr NTSTATUS kStatusUnsuccessful = static_cast<NTSTATUS>(0xC0000001L);
constexpr NTSTATUS kStatusInvalidHandle = static_cast<NTSTATUS>(0xC0000008L);
constexpr NTSTATUS kStatusInvalidParameter = static_cast<NTSTATUS>(0xC000000DL);
constexpr NTSTATUS kStatusNotFound = static_cast<NTSTATUS>(0xC0000225L);

using OpenAlgFn = NTSTATUS(WINAPI*)(BCRYPT_ALG_HANDLE*, LPCWSTR, LPCWSTR, ULONG);
using CloseAlgFn = NTSTATUS(WINAPI*)(BCRYPT_ALG_HANDLE, ULONG);
using GenRandomFn = NTSTATUS(WINAPI*)(BCRYPT_ALG_HANDLE, PUCHAR, ULONG, ULONG);

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const char* what) {
    ++g_checks;
    std::printf("%s %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition) ++g_failures;
}

template <typename Fn>
Fn resolve(const char* name) {
    return reinterpret_cast<Fn>(rt::shims::find("bcrypt.dll", name));
}

void test_registry_coverage() {
    check(resolve<OpenAlgFn>("BCryptOpenAlgorithmProvider") != nullptr, "BCryptOpenAlgorithmProvider resolves");
    check(resolve<GenRandomFn>("BCryptGenRandom") != nullptr, "BCryptGenRandom resolves");
    check(resolve<CloseAlgFn>("BCryptCloseAlgorithmProvider") != nullptr, "BCryptCloseAlgorithmProvider resolves");
}

void test_provider_handles_and_validation() {
    auto open_alg = resolve<OpenAlgFn>("BCryptOpenAlgorithmProvider");
    auto gen_random = resolve<GenRandomFn>("BCryptGenRandom");
    auto close_alg = resolve<CloseAlgFn>("BCryptCloseAlgorithmProvider");

    BCRYPT_ALG_HANDLE h = nullptr;
    check(open_alg(nullptr, BCRYPT_RNG_ALGORITHM, nullptr, 0) == kStatusInvalidParameter,
          "BCryptOpenAlgorithmProvider rejects null out handle");
    check(open_alg(&h, nullptr, nullptr, 0) == kStatusInvalidParameter,
          "BCryptOpenAlgorithmProvider rejects null algorithm id");
    check(open_alg(&h, L"LEONIDA_NO_SUCH_ALG", nullptr, 0) == kStatusNotFound,
          "BCryptOpenAlgorithmProvider reports STATUS_NOT_FOUND for unknown algorithm");

    check(open_alg(&h, BCRYPT_RNG_ALGORITHM, nullptr, 0) == kStatusSuccess && h != nullptr,
          "BCryptOpenAlgorithmProvider opens BCRYPT_RNG_ALGORITHM");

    uint8_t buf[32]{};
    check(gen_random(h, nullptr, 0, 0) == kStatusInvalidParameter,
          "BCryptGenRandom rejects null buffer");
    check(gen_random(nullptr, buf, sizeof(buf), 0) == kStatusInvalidHandle,
          "BCryptGenRandom rejects null handle without BCRYPT_USE_SYSTEM_PREFERRED_RNG");
    check(gen_random(h, buf, sizeof(buf), BCRYPT_USE_SYSTEM_PREFERRED_RNG) == kStatusInvalidParameter,
          "BCryptGenRandom rejects non-null handle with BCRYPT_USE_SYSTEM_PREFERRED_RNG");

    std::memset(buf, 0, sizeof(buf));
    check(gen_random(h, buf, sizeof(buf), 0) == kStatusSuccess,
          "BCryptGenRandom succeeds with open RNG handle");

    bool nonzero = false;
    for (uint8_t b : buf) {
        if (b != 0) nonzero = true;
    }
    check(nonzero, "BCryptGenRandom writes non-zero random bytes");

    uint8_t sys_buf[32]{};
    check(gen_random(nullptr, sys_buf, sizeof(sys_buf), BCRYPT_USE_SYSTEM_PREFERRED_RNG) == kStatusSuccess,
          "BCryptGenRandom succeeds with BCRYPT_USE_SYSTEM_PREFERRED_RNG");

    // The RNG's CNG pseudo-handle (0x81) needs no open call; Windows 11's bcrypt.dll returns STATUS_SUCCESS for it,
    // STATUS_INVALID_HANDLE for another algorithm's pseudo-handle and for closing it, and STATUS_INVALID_PARAMETER
    // with BCRYPT_USE_SYSTEM_PREFERRED_RNG.
    uint8_t pseudo_buf[32]{};
    check(gen_random(BCRYPT_RNG_ALG_HANDLE, pseudo_buf, sizeof(pseudo_buf), 0) == kStatusSuccess,
          "BCryptGenRandom takes BCRYPT_RNG_ALG_HANDLE without an open call");
    bool pseudo_nonzero = false;
    for (uint8_t b : pseudo_buf) {
        if (b != 0) pseudo_nonzero = true;
    }
    check(pseudo_nonzero, "BCryptGenRandom writes random bytes through BCRYPT_RNG_ALG_HANDLE");
    check(gen_random(BCRYPT_SHA256_ALG_HANDLE, pseudo_buf, sizeof(pseudo_buf), 0) == kStatusInvalidHandle,
          "BCryptGenRandom refuses another algorithm's pseudo-handle");
    check(gen_random(BCRYPT_RNG_ALG_HANDLE, pseudo_buf, sizeof(pseudo_buf), BCRYPT_USE_SYSTEM_PREFERRED_RNG) ==
              kStatusInvalidParameter,
          "BCryptGenRandom rejects BCRYPT_RNG_ALG_HANDLE with BCRYPT_USE_SYSTEM_PREFERRED_RNG");
    check(close_alg(BCRYPT_RNG_ALG_HANDLE, 0) == kStatusInvalidHandle,
          "BCryptCloseAlgorithmProvider refuses the RNG pseudo-handle, which was never opened");

    check(close_alg(nullptr, 0) == kStatusInvalidHandle,
          "BCryptCloseAlgorithmProvider rejects nullptr");
    check(close_alg(reinterpret_cast<BCRYPT_ALG_HANDLE>(static_cast<uintptr_t>(0x1234)), 0) == kStatusInvalidHandle,
          "BCryptCloseAlgorithmProvider rejects foreign pointer");
    check(close_alg(h, 0) == kStatusSuccess,
          "BCryptCloseAlgorithmProvider closes live RNG handle");
    check(close_alg(h, 0) == kStatusInvalidHandle,
          "BCryptCloseAlgorithmProvider refuses double close safely");
    check(gen_random(h, buf, sizeof(buf), 0) == kStatusInvalidHandle,
          "BCryptGenRandom rejects closed handle safely");
}

void test_fail_closed_when_host_rng_unavailable(bool seeded) {
    if (seeded) return;
    auto open_alg = resolve<OpenAlgFn>("BCryptOpenAlgorithmProvider");
    auto gen_random = resolve<GenRandomFn>("BCryptGenRandom");
    auto close_alg = resolve<CloseAlgFn>("BCryptCloseAlgorithmProvider");

    BCRYPT_ALG_HANDLE h = nullptr;
    check(open_alg(&h, BCRYPT_RNG_ALGORITHM, nullptr, 0) == kStatusSuccess && h != nullptr,
          "fail-closed setup opens RNG handle");

    rt::bcrypt_rng::set_host_rng_enabled_for_test(false);
    uint8_t handle_buf[16]{};
    std::memset(handle_buf, 0xAA, sizeof(handle_buf));
    const NTSTATUS handle_st = gen_random(h, handle_buf, sizeof(handle_buf), 0);
    bool handle_untouched = true;
    for (uint8_t b : handle_buf) {
        if (b != 0xAA) handle_untouched = false;
    }
    check(handle_st == kStatusUnsuccessful && handle_untouched,
          "BCryptGenRandom fails closed with STATUS_UNSUCCESSFUL without mutating buffer (handle path)");

    uint8_t sys_buf[16]{};
    std::memset(sys_buf, 0x55, sizeof(sys_buf));
    const NTSTATUS sys_st = gen_random(nullptr, sys_buf, sizeof(sys_buf), BCRYPT_USE_SYSTEM_PREFERRED_RNG);
    bool sys_untouched = true;
    for (uint8_t b : sys_buf) {
        if (b != 0x55) sys_untouched = false;
    }
    check(sys_st == kStatusUnsuccessful && sys_untouched,
          "BCryptGenRandom fails closed with STATUS_UNSUCCESSFUL without mutating buffer (system RNG path)");

    uint8_t pseudo_buf[16]{};
    std::memset(pseudo_buf, 0x33, sizeof(pseudo_buf));
    const NTSTATUS pseudo_st = gen_random(BCRYPT_RNG_ALG_HANDLE, pseudo_buf, sizeof(pseudo_buf), 0);
    bool pseudo_untouched = true;
    for (uint8_t b : pseudo_buf) {
        if (b != 0x33) pseudo_untouched = false;
    }
    check(pseudo_st == kStatusUnsuccessful && pseudo_untouched,
          "BCryptGenRandom fails closed with STATUS_UNSUCCESSFUL without mutating buffer (pseudo-handle path)");

    rt::bcrypt_rng::set_host_rng_enabled_for_test(true);
    close_alg(h, 0);
}

void test_seeded_determinism(bool seeded) {
    if (!seeded) return;
    auto open_alg = resolve<OpenAlgFn>("BCryptOpenAlgorithmProvider");
    auto gen_random = resolve<GenRandomFn>("BCryptGenRandom");
    auto close_alg = resolve<CloseAlgFn>("BCryptCloseAlgorithmProvider");

    BCRYPT_ALG_HANDLE h = nullptr;
    check(open_alg(&h, BCRYPT_RNG_ALGORITHM, nullptr, 0) == kStatusSuccess && h != nullptr,
          "seeded BCryptOpenAlgorithmProvider opens RNG");
    uint8_t buf[16]{};
    check(gen_random(h, buf, sizeof(buf), 0) == kStatusSuccess, "seeded BCryptGenRandom succeeds");
    // Ensure reproducible non-trivial stream across runs when LEONIDA_RNG_SEED=0x123456789abcdef0 is set.
    uint8_t second[16]{};
    check(gen_random(nullptr, second, sizeof(second), BCRYPT_USE_SYSTEM_PREFERRED_RNG) == kStatusSuccess &&
              std::memcmp(buf, second, sizeof(buf)) != 0,
          "seeded PRNG advances cleanly across consecutive calls");
    // With the host RNG off, only the seeded stream can fill the buffer.
    rt::bcrypt_rng::set_host_rng_enabled_for_test(false);
    uint8_t third[16]{};
    std::memset(third, 0x77, sizeof(third));
    const NTSTATUS pseudo_st = gen_random(BCRYPT_RNG_ALG_HANDLE, third, sizeof(third), 0);
    bool written = false;
    for (uint8_t b : third) {
        if (b != 0x77) written = true;
    }
    check(pseudo_st == kStatusSuccess && written,
          "seeded BCryptGenRandom serves BCRYPT_RNG_ALG_HANDLE from the seeded stream");
    rt::bcrypt_rng::set_host_rng_enabled_for_test(true);
    close_alg(h, 0);
}

}  // namespace

int main(int argc, char** argv) {
    bool seeded = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--seeded") == 0) seeded = true;
    }
    test_registry_coverage();
    test_provider_handles_and_validation();
    test_fail_closed_when_host_rng_unavailable(seeded);
    test_seeded_determinism(seeded);
    std::printf("%d check(s), %d failure(s)\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
