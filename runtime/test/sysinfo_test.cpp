#include <windows.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "shims.h"

namespace {

int g_fail = 0;

#define CHECK(cond, ...)                                                       \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::fprintf(stderr, "FAIL (%d): ", __LINE__);                     \
            std::fprintf(stderr, __VA_ARGS__);                                 \
            std::fprintf(stderr, "\n");                                        \
            ++g_fail;                                                          \
        }                                                                      \
    } while (0)

DWORD count_cores_and_logical(const std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION>& recs,
                              DWORD& logical_out) {
    DWORD cores = 0;
    DWORD logical = 0;
    for (const auto& r : recs) {
        if (r.Relationship == RelationProcessorCore) {
            ++cores;
            ULONG_PTR m = r.ProcessorMask;
            while (m) {
                logical += static_cast<DWORD>(m & 1u);
                m >>= 1;
            }
        }
    }
    logical_out = logical;
    return cores;
}

}  // namespace

int main() {
    using GetLpiFn = BOOL(WINAPI*)(PSYSTEM_LOGICAL_PROCESSOR_INFORMATION, PDWORD);
    using GlobalMemFn = BOOL(WINAPI*)(LPMEMORYSTATUSEX);

    auto* get_lpi_api = reinterpret_cast<GetLpiFn>(
        rt::shims::find("api-ms-win-core-sysinfo-l1-1-0.dll", "GetLogicalProcessorInformation"));
    auto* get_lpi_k32 = reinterpret_cast<GetLpiFn>(
        rt::shims::find("kernel32.dll", "GetLogicalProcessorInformation"));
    auto* get_lpi_kb = reinterpret_cast<GetLpiFn>(
        rt::shims::find("kernelbase.dll", "GetLogicalProcessorInformation"));
    auto* mem_api = reinterpret_cast<GlobalMemFn>(
        rt::shims::find("api-ms-win-core-sysinfo-l1-1-0.dll", "GlobalMemoryStatusEx"));
    auto* mem_k32 = reinterpret_cast<GlobalMemFn>(
        rt::shims::find("kernel32.dll", "GlobalMemoryStatusEx"));

    CHECK(get_lpi_api != nullptr && get_lpi_k32 == get_lpi_api && get_lpi_kb == get_lpi_api,
          "GetLogicalProcessorInformation shim missing on expected DLL keys");
    CHECK(mem_api != nullptr && mem_k32 == mem_api,
          "GlobalMemoryStatusEx shim missing on expected DLL keys");
    CHECK(!rt::shims::owns_dll("api-ms-win-core-sysinfo-l1-1-0.dll"),
          "owns_dll must remain false for api-ms sysinfo contract");
    CHECK(!rt::shims::owns_dll("kernel32.dll"), "owns_dll must remain false for kernel32.dll");
    CHECK(!rt::shims::owns_dll("kernelbase.dll"), "owns_dll must remain false for kernelbase.dll");
    if (g_fail) return 1;

    // 1. Validation & error contracts
    SetLastError(0);
    CHECK(get_lpi_api(nullptr, nullptr) == FALSE && GetLastError() == ERROR_INVALID_PARAMETER,
          "null ReturnLength must fail with ERROR_INVALID_PARAMETER");
    CHECK(mem_api(nullptr) == FALSE && GetLastError() == ERROR_INVALID_PARAMETER,
          "null MEMORYSTATUSEX must fail with ERROR_INVALID_PARAMETER");
    MEMORYSTATUSEX bad_ms{};
    bad_ms.dwLength = 4;
    CHECK(mem_api(&bad_ms) == FALSE && GetLastError() == ERROR_INVALID_PARAMETER,
          "bad MEMORYSTATUSEX dwLength must fail with ERROR_INVALID_PARAMETER");

    // 2. Host passthrough mode: two-step size probe contract must hold (retail exe executes int3 otherwise).
    SetEnvironmentVariableA("LEONIDA_CPU_TOPOLOGY", nullptr);
    SetEnvironmentVariableA("LEONIDA_SYSTEM_MEMORY_MB", nullptr);
    DWORD len = 0;
    SetLastError(0);
    BOOL ok = get_lpi_api(nullptr, &len);
    DWORD err = GetLastError();
    CHECK(ok == FALSE && err == ERROR_INSUFFICIENT_BUFFER,
          "host size query must return FALSE + ERROR_INSUFFICIENT_BUFFER (ok=%d err=%lu)", ok, err);
    CHECK(len >= sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION) &&
              (len % sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION)) == 0,
          "host ReturnLength invalid: %lu", len);
    std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> recs(
        len / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION));
    CHECK(get_lpi_api(recs.data(), &len) == TRUE, "host populated GetLogicalProcessorInformation failed");
    DWORD host_logical = 0;
    DWORD host_cores = count_cores_and_logical(recs, host_logical);
    CHECK(host_cores >= 1 && host_logical >= host_cores,
          "host topology core/logical count invalid: %lu cores / %lu logical", host_cores, host_logical);

    MEMORYSTATUSEX host_ms{};
    host_ms.dwLength = sizeof(host_ms);
    CHECK(mem_api(&host_ms) == TRUE && host_ms.ullTotalPhys > 0, "host GlobalMemoryStatusEx failed");

    // 3. Xbox Series X console topology preset ("xbox" -> 8 physical cores / 16 logical SMT threads).
    SetEnvironmentVariableA("LEONIDA_CPU_TOPOLOGY", "xbox");
    len = 0;
    SetLastError(0);
    ok = get_lpi_api(nullptr, &len);
    err = GetLastError();
    CHECK(ok == FALSE && err == ERROR_INSUFFICIENT_BUFFER && len >= 32,
          "xbox preset probe failed: ok=%d err=%lu len=%lu", ok, err, len);
    DWORD short_len = 32;
    SYSTEM_LOGICAL_PROCESSOR_INFORMATION tiny{};
    SetLastError(0);
    CHECK(get_lpi_api(&tiny, &short_len) == FALSE &&
              GetLastError() == ERROR_INSUFFICIENT_BUFFER && short_len == len,
          "undersized buffer must update ReturnLength=%lu (got %lu)", len, short_len);

    recs.assign(len / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION), {});
    CHECK(get_lpi_api(recs.data(), &len) == TRUE, "xbox preset query failed");
    DWORD xbox_logical = 0;
    DWORD xbox_cores = count_cores_and_logical(recs, xbox_logical);
    CHECK(xbox_cores == 8 && xbox_logical == 16,
          "xbox topology expected 8 cores / 16 logical, got %lu / %lu", xbox_cores, xbox_logical);
    bool has_smt = false;
    bool has_l3 = false;
    for (const auto& r : recs) {
        if (r.Relationship == RelationProcessorCore && (r.ProcessorCore.Flags & LTP_PC_SMT)) {
            has_smt = true;
        }
        if (r.Relationship == RelationCache && r.Cache.Level == 3) {
            has_l3 = true;
        }
    }
    CHECK(has_smt && has_l3, "xbox topology missing SMT flags or L3 cache records");

    // 4. Custom asymmetric NxM topology ("6x10") and invalid fallback to host.
    SetEnvironmentVariableA("LEONIDA_CPU_TOPOLOGY", "6x10");
    len = 0;
    CHECK(get_lpi_api(nullptr, &len) == FALSE && GetLastError() == ERROR_INSUFFICIENT_BUFFER,
          "6x10 probe failed");
    recs.assign(len / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION), {});
    CHECK(get_lpi_api(recs.data(), &len) == TRUE, "6x10 query failed");
    DWORD custom_logical = 0;
    DWORD custom_cores = count_cores_and_logical(recs, custom_logical);
    CHECK(custom_cores == 6 && custom_logical == 10,
          "6x10 expected 6 cores / 10 logical, got %lu / %lu", custom_cores, custom_logical);

    SetEnvironmentVariableA("LEONIDA_CPU_TOPOLOGY", "not_valid_999");
    len = 0;
    CHECK(get_lpi_api(nullptr, &len) == FALSE && GetLastError() == ERROR_INSUFFICIENT_BUFFER,
          "invalid env must still satisfy probe contract via host");

    // 5. Console memory override ("xbox" = 16 GiB, explicit "8192" = 8 GiB).
    SetEnvironmentVariableA("LEONIDA_SYSTEM_MEMORY_MB", "xbox");
    MEMORYSTATUSEX xbox_ms{};
    xbox_ms.dwLength = sizeof(xbox_ms);
    CHECK(mem_api(&xbox_ms) == TRUE &&
              xbox_ms.ullTotalPhys == 16384ULL * 1024ULL * 1024ULL &&
              xbox_ms.ullAvailPhys > 0 && xbox_ms.ullAvailPhys <= xbox_ms.ullTotalPhys &&
              xbox_ms.dwMemoryLoad <= 100,
          "xbox memory override mismatch (total=%llu avail=%llu)", xbox_ms.ullTotalPhys, xbox_ms.ullAvailPhys);

    SetEnvironmentVariableA("LEONIDA_SYSTEM_MEMORY_MB", "8192");
    MEMORYSTATUSEX custom_ms{};
    custom_ms.dwLength = sizeof(custom_ms);
    CHECK(mem_api(&custom_ms) == TRUE &&
              custom_ms.ullTotalPhys == 8192ULL * 1024ULL * 1024ULL &&
              custom_ms.ullAvailPhys <= custom_ms.ullTotalPhys,
          "8192 MB memory override mismatch");

    SetEnvironmentVariableA("LEONIDA_CPU_TOPOLOGY", nullptr);
    SetEnvironmentVariableA("LEONIDA_SYSTEM_MEMORY_MB", nullptr);

    std::printf("sysinfo_test: %s (%d failures)\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
