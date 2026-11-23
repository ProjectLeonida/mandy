// System topology and memory shims (api-ms-win-core-sysinfo-l1-1-0.dll, kernel32.dll, kernelbase.dll).
// Honors LEONIDA_CPU_TOPOLOGY ("xbox"/"scarlett" for 8c/16t, "NxM" or "N") and
// LEONIDA_SYSTEM_MEMORY_MB ("xbox"/"scarlett" for 16384 MB, or explicit MiB) for deterministic console parity.

#include <windows.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "shims.h"

namespace rt::sysinfo {
namespace {

static_assert(sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION) == 32,
              "SYSTEM_LOGICAL_PROCESSOR_INFORMATION must match x64 ABI (32 bytes)");

constexpr DWORD kDefaultXboxCores = 8;
constexpr DWORD kDefaultXboxLogical = 16;
constexpr uint64_t kXboxMemoryMb = 16384ULL;
constexpr uint64_t kMinMemoryMb = 256ULL;
constexpr uint64_t kMaxMemoryMb = 1048576ULL;
constexpr uint64_t kFallbackUserVaBytes = (128ULL << 40) - (64ULL << 10);

struct TopologySpec {
    DWORD physical_cores = 0;
    DWORD logical_processors = 0;
};

std::string read_env_trimmed_lower(const char* name) {
    char buf[128]{};
    DWORD n = GetEnvironmentVariableA(name, buf, static_cast<DWORD>(sizeof(buf)));
    if (n == 0 || n >= sizeof(buf)) return {};
    std::string s(buf, n);
    size_t start = 0;
    while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start]))) ++start;
    size_t end = s.size();
    while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1]))) --end;
    std::string out = s.substr(start, end - start);
    for (char& ch : out) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return out;
}

bool parse_u32_decimal(const char* begin, const char* end, DWORD& out) {
    if (begin >= end) return false;
    uint64_t v = 0;
    for (const char* p = begin; p < end; ++p) {
        if (*p < '0' || *p > '9') return false;
        v = v * 10ULL + static_cast<uint64_t>(*p - '0');
        if (v > 0xFFFFFFFFULL) return false;
    }
    out = static_cast<DWORD>(v);
    return true;
}

bool parse_topology_spec(const std::string& raw, TopologySpec& out) {
    if (raw.empty() || raw == "host") return false;
    if (raw == "xbox" || raw == "scarlett") {
        out = {kDefaultXboxCores, kDefaultXboxLogical};
        return true;
    }
    size_t sep = raw.find('x');
    if (sep == std::string::npos) sep = raw.find('/');
    if (sep == std::string::npos) {
        DWORD total = 0;
        if (!parse_u32_decimal(raw.data(), raw.data() + raw.size(), total)) return false;
        if (total < 1 || total > 64) return false;
        out = {total, total};
        return true;
    }
    DWORD cores = 0;
    DWORD logical = 0;
    if (!parse_u32_decimal(raw.data(), raw.data() + sep, cores)) return false;
    if (!parse_u32_decimal(raw.data() + sep + 1, raw.data() + raw.size(), logical)) return false;
    if (cores < 1 || logical < cores || logical > 64) return false;
    out = {cores, logical};
    return true;
}

bool parse_memory_mb_spec(const std::string& raw, uint64_t& mb_out) {
    if (raw.empty() || raw == "host") return false;
    if (raw == "xbox" || raw == "scarlett") {
        mb_out = kXboxMemoryMb;
        return true;
    }
    uint64_t v = 0;
    for (char ch : raw) {
        if (ch < '0' || ch > '9') return false;
        if (v > (kMaxMemoryMb / 10ULL)) return false;
        v = v * 10ULL + static_cast<uint64_t>(ch - '0');
    }
    if (v == 0) return false;
    mb_out = std::clamp(v, kMinMemoryMb, kMaxMemoryMb);
    return true;
}

SYSTEM_LOGICAL_PROCESSOR_INFORMATION make_core_record(ULONG_PTR mask, BYTE smt_flag) {
    SYSTEM_LOGICAL_PROCESSOR_INFORMATION rec{};
    rec.ProcessorMask = mask;
    rec.Relationship = RelationProcessorCore;
    rec.ProcessorCore.Flags = smt_flag;
    return rec;
}

SYSTEM_LOGICAL_PROCESSOR_INFORMATION make_cache_record(ULONG_PTR mask, BYTE level, BYTE assoc,
                                                       WORD line_size, DWORD size_bytes,
                                                       PROCESSOR_CACHE_TYPE type) {
    SYSTEM_LOGICAL_PROCESSOR_INFORMATION rec{};
    rec.ProcessorMask = mask;
    rec.Relationship = RelationCache;
    rec.Cache.Level = level;
    rec.Cache.Associativity = assoc;
    rec.Cache.LineSize = line_size;
    rec.Cache.Size = size_bytes;
    rec.Cache.Type = type;
    return rec;
}

std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> synthesize_topology(const TopologySpec& spec) {
    const DWORD cores = std::clamp(spec.physical_cores, 1ul, 64ul);
    const DWORD logical = std::clamp(spec.logical_processors, cores, 64ul);

    std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> records;
    records.reserve(static_cast<size_t>(cores) * 4U + 4U);

    const DWORD base_threads = logical / cores;
    const DWORD extra_threads = logical % cores;
    DWORD bit = 0;
    ULONG_PTR all_mask = 0;
    std::vector<ULONG_PTR> core_masks(cores, 0);

    for (DWORD c = 0; c < cores; ++c) {
        const DWORD threads_here = base_threads + (c < extra_threads ? 1ul : 0ul);
        ULONG_PTR mask = 0;
        for (DWORD t = 0; t < threads_here && bit < 64; ++t, ++bit) {
            mask |= (static_cast<ULONG_PTR>(1) << bit);
        }
        core_masks[c] = mask;
        all_mask |= mask;
        records.push_back(make_core_record(mask, threads_here > 1 ? LTP_PC_SMT : 0));
    }

    // Standard per-core L1/L2 and shared 4-core cluster L3 cache hierarchy.
    for (DWORD c = 0; c < cores; ++c) {
        const ULONG_PTR m = core_masks[c];
        records.push_back(make_cache_record(m, 1, 8, 64, 32u * 1024u, CacheInstruction));
        records.push_back(make_cache_record(m, 1, 8, 64, 32u * 1024u, CacheData));
        records.push_back(make_cache_record(m, 2, 8, 64, 512u * 1024u, CacheUnified));
    }
    for (DWORD cluster = 0; cluster < cores; cluster += 4) {
        ULONG_PTR cluster_mask = 0;
        const DWORD end = (std::min)(cores, cluster + 4);
        for (DWORD c = cluster; c < end; ++c) cluster_mask |= core_masks[c];
        records.push_back(make_cache_record(cluster_mask, 3, 16, 64, 4u * 1024u * 1024u, CacheUnified));
    }

    SYSTEM_LOGICAL_PROCESSOR_INFORMATION numa{};
    numa.ProcessorMask = all_mask;
    numa.Relationship = RelationNumaNode;
    numa.NumaNode.NodeNumber = 0;
    records.push_back(numa);

    SYSTEM_LOGICAL_PROCESSOR_INFORMATION pkg{};
    pkg.ProcessorMask = all_mask;
    pkg.Relationship = RelationProcessorPackage;
    records.push_back(pkg);

    return records;
}

BOOL write_topology_records(const std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION>& records,
                            PSYSTEM_LOGICAL_PROCESSOR_INFORMATION buffer, PDWORD return_length) {
    const DWORD needed =
        static_cast<DWORD>(records.size() * sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION));
    if (!buffer || *return_length < needed) {
        *return_length = needed;
        SetLastError(ERROR_INSUFFICIENT_BUFFER);
        return FALSE;
    }
    std::memcpy(buffer, records.data(), needed);
    *return_length = needed;
    return TRUE;
}

TopologySpec host_fallback_spec() {
    SYSTEM_INFO si{};
    GetSystemInfo(&si);
    DWORD logical = std::clamp(si.dwNumberOfProcessors, 1ul, 64ul);
    DWORD cores = (logical >= 2 && (logical % 2) == 0) ? (logical / 2) : logical;
    return {cores, logical};
}

}  // namespace

BOOL WINAPI shim_GetLogicalProcessorInformation(PSYSTEM_LOGICAL_PROCESSOR_INFORMATION Buffer,
                                                PDWORD ReturnLength) {
    if (!ReturnLength) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return FALSE;
    }

    const std::string env = read_env_trimmed_lower("LEONIDA_CPU_TOPOLOGY");
    TopologySpec spec{};
    if (parse_topology_spec(env, spec)) {
        return write_topology_records(synthesize_topology(spec), Buffer, ReturnLength);
    }

    // Host pass-through with fallback protection if host fails the mandatory size-query contract
    // or returns an empty topology buffer (which triggers fail-fast int3 in the retail executable).
    const DWORD caller_bytes = *ReturnLength;
    if (!Buffer || caller_bytes == 0) {
        DWORD host_len = 0;
        const BOOL ok = ::GetLogicalProcessorInformation(nullptr, &host_len);
        const DWORD err = GetLastError();
        if (!ok && err == ERROR_INSUFFICIENT_BUFFER &&
            host_len >= sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION)) {
            *ReturnLength = host_len;
            SetLastError(ERROR_INSUFFICIENT_BUFFER);
            return FALSE;
        }
        return write_topology_records(synthesize_topology(host_fallback_spec()), nullptr, ReturnLength);
    }

    DWORD host_len = caller_bytes;
    if (::GetLogicalProcessorInformation(Buffer, &host_len)) {
        if (host_len >= sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION)) {
            *ReturnLength = host_len;
            return TRUE;
        }
        return write_topology_records(synthesize_topology(host_fallback_spec()), Buffer, ReturnLength);
    }

    const DWORD err = GetLastError();
    if (err == ERROR_INSUFFICIENT_BUFFER &&
        host_len >= sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION)) {
        *ReturnLength = host_len;
        SetLastError(ERROR_INSUFFICIENT_BUFFER);
        return FALSE;
    }

    return write_topology_records(synthesize_topology(host_fallback_spec()), Buffer, ReturnLength);
}

BOOL WINAPI shim_GlobalMemoryStatusEx(LPMEMORYSTATUSEX lpBuffer) {
    if (!lpBuffer || lpBuffer->dwLength != sizeof(MEMORYSTATUSEX)) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return FALSE;
    }

    uint64_t override_mb = 0;
    const std::string env = read_env_trimmed_lower("LEONIDA_SYSTEM_MEMORY_MB");
    if (!parse_memory_mb_spec(env, override_mb)) {
        return ::GlobalMemoryStatusEx(lpBuffer);
    }

    MEMORYSTATUSEX host{};
    host.dwLength = sizeof(host);
    const bool have_host = ::GlobalMemoryStatusEx(&host) != FALSE;

    const DWORDLONG total_phys = static_cast<DWORDLONG>(override_mb) * 1024ULL * 1024ULL;
    const DWORDLONG default_avail = (total_phys * 75ULL) / 100ULL;
    const DWORDLONG avail_phys =
        (have_host && host.ullAvailPhys > 0 && host.ullAvailPhys <= total_phys)
            ? host.ullAvailPhys
            : default_avail;

    lpBuffer->dwLength = sizeof(MEMORYSTATUSEX);
    lpBuffer->dwMemoryLoad =
        static_cast<DWORD>(((total_phys - avail_phys) * 100ULL) / total_phys);
    lpBuffer->ullTotalPhys = total_phys;
    lpBuffer->ullAvailPhys = avail_phys;
    lpBuffer->ullTotalPageFile = total_phys;
    lpBuffer->ullAvailPageFile = avail_phys;
    lpBuffer->ullTotalVirtual = have_host ? host.ullTotalVirtual : kFallbackUserVaBytes;
    lpBuffer->ullAvailVirtual = have_host ? host.ullAvailVirtual : kFallbackUserVaBytes;
    lpBuffer->ullAvailExtendedVirtual = 0;
    return TRUE;
}

}  // namespace rt::sysinfo

#define SYSINFO_API(dll)                                                                                 \
    {dll, "GetLogicalProcessorInformation",                                                              \
     reinterpret_cast<void*>(&::rt::sysinfo::shim_GetLogicalProcessorInformation)},                      \
        {dll, "GlobalMemoryStatusEx", reinterpret_cast<void*>(&::rt::sysinfo::shim_GlobalMemoryStatusEx)}

SHIM_TABLE(sysinfo,
    SYSINFO_API("kernel32.dll"),
    SYSINFO_API("kernelbase.dll"),
    SYSINFO_API("api-ms-win-core-sysinfo-l1-1-0.dll"),
);
