#pragma once

// How much of a range of this process's memory is resident in RAM, sampled with QueryWorkingSetEx (psapi): one 4 KiB
// page in every `step`, one call per range. Used by the memory probe to show how much of the title memory behind a
// GPU copy is also held in RAM, which is the duplicate docs/d3d12_vulkan.md and #1702 describe. An estimate, not an
// exact count: a range smaller than `step` pages is read from its first page only. Pages that are not committed, or
// not mapped at all, read as not resident.

#include <algorithm>
#include <cstdint>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <psapi.h>
#endif

namespace leonida::working_set {

constexpr std::uint64_t kPage = 4096;

struct CpuRange {
    std::uint64_t begin;
    std::uint64_t end;
};

// Texture aliases and overlapping views can refer to the same CPU bytes. Count their union, by virtual address: two
// distinct virtual views of one physical backing (a placeholder mapped twice, a GPU-only alias paired with a CPU view)
// do not overlap here, so their bytes count twice, and so does their residency.
inline void merge_cpu_ranges(std::vector<CpuRange>& ranges) {
    std::sort(ranges.begin(), ranges.end(), [](const CpuRange& a, const CpuRange& b) { return a.begin < b.begin; });
    std::size_t count = 0;
    for (const CpuRange range : ranges) {
        if (range.begin >= range.end) continue;
        if (count && range.begin <= ranges[count - 1].end) {
            ranges[count - 1].end = std::max(ranges[count - 1].end, range.end);
        } else {
            ranges[count++] = range;
        }
    }
    ranges.resize(count);
}

// Estimated resident bytes of [base, base + size): the share of sampled pages that are valid in the working set,
// times size. 0 for an empty range or where the query is not available.
inline std::uint64_t resident_bytes(std::uint64_t base, std::uint64_t size, unsigned step = 16) {
#ifdef _WIN32
    if (!size || !step) return 0;
    const std::uint64_t first = base & ~(kPage - 1);
    const std::uint64_t pages = (base + size - first + kPage - 1) / kPage;
    const std::uint64_t samples = (pages + step - 1) / step;
    std::vector<PSAPI_WORKING_SET_EX_INFORMATION> info(static_cast<std::size_t>(samples));
    for (std::uint64_t i = 0; i < samples; ++i)
        info[static_cast<std::size_t>(i)].VirtualAddress = reinterpret_cast<void*>(first + i * step * kPage);
    if (!QueryWorkingSetEx(GetCurrentProcess(), info.data(),
                           static_cast<DWORD>(info.size() * sizeof(PSAPI_WORKING_SET_EX_INFORMATION))))
        return 0;
    std::uint64_t valid = 0;
    for (const auto& e : info) valid += e.VirtualAttributes.Valid;
    return size * valid / samples;
#else
    (void)base;
    (void)size;
    (void)step;
    return 0;
#endif
}

}  // namespace leonida::working_set
