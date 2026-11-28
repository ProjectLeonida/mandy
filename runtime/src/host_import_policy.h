#pragma once

// Host-memory import eligibility and memory-type ordering for VK_EXT_external_memory_host.
//
// Pure stdlib policy (no Windows or Vulkan headers) so every rule is checked on the CPU by
// runtime/test/host_import_policy_test.cpp before touching a driver.
//
// Under Wine on Linux, nvidia.ko's long-term user-page pinning (pin_user_pages / FOLL_LONGTERM)
// rejects file-backed MAP_SHARED views (such as LEONIDA_XMEM_BACKING_DIR's sparse temp files)
// with kernel "Cannot map memory with base addr ..." errors and also refuses ranges spanning
// multiple VMAs (distinct VirtualQuery AllocationBase values). On NVIDIA 620.73 under Wine,
// bursts of those rejected host imports coincide with transient VK_ERROR_OUT_OF_HOST_MEMORY on
// concurrent normal allocations. Anonymous VirtualAlloc ranges and anonymous pagefile sections
// (CreateFileMappingW(INVALID_HANDLE_VALUE, ...)) stay importable.

#include <cstdint>
#include <vector>

namespace leonida {

struct HostRegion {
    std::uintptr_t allocation_base = 0;
    bool committed = false;
    bool writable = false;
    bool cache_modified = false;  // PAGE_WRITECOMBINE | PAGE_NOCACHE | PAGE_GUARD
    bool mapped_file = false;     // MEM_MAPPED with a backing file name (K32GetMappedFileNameW > 0)
};

struct HostImportRangeCheck {
    bool wine = false;
    bool require_writable = true;
    bool has_first = false;
    std::uintptr_t first_allocation_base = 0;

    bool accept(const HostRegion& region) {
        if (!region.committed || region.cache_modified) return false;
        if (require_writable && !region.writable) return false;
        if (wine) {
            if (region.mapped_file) return false;
            if (has_first && region.allocation_base != first_allocation_base) return false;
        }
        if (!has_first) {
            first_allocation_base = region.allocation_base;
            has_first = true;
        }
        return true;
    }
};

// Memory types allowed for a host-pointer import, preferring types that satisfy `preferred_flags`
// (e.g. HOST_VISIBLE | HOST_COHERENT) first, then any remaining allowed types. Every type index is
// emitted at most once so a failing import never probes the same Vulkan memory type twice.
inline std::vector<std::uint32_t> host_import_memory_types(std::uint32_t allowed_mask,
                                                           const std::uint32_t* property_flags,
                                                           std::uint32_t type_count,
                                                           std::uint32_t preferred_flags) {
    std::vector<std::uint32_t> order;
    if (!property_flags || !type_count || !allowed_mask) return order;
    const std::uint32_t count = type_count > 32 ? 32 : type_count;
    std::uint32_t emitted = 0;
    for (const std::uint32_t want : {preferred_flags, std::uint32_t{0}}) {
        for (std::uint32_t i = 0; i < count; ++i) {
            const std::uint32_t bit = std::uint32_t{1} << i;
            if (!(allowed_mask & bit) || (emitted & bit)) continue;
            if ((property_flags[i] & want) != want) continue;
            order.push_back(i);
            emitted |= bit;
        }
    }
    return order;
}

}  // namespace leonida
