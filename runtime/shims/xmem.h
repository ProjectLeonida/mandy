#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace rt::xmem {

void geometry_aliases(std::uintptr_t va, std::vector<std::uintptr_t>& addresses);
bool commit_physical_range(const void* address, std::size_t size);
bool commit_physical_pages(const std::uint64_t* page_array, std::size_t count);

// Copy a validated DirectStorage staging buffer through the page table. The page list has
// one frame number per 64 KiB page. All pages are resolved before a write starts.
bool transfer_physical(std::span<const std::uint64_t> pages, std::uint16_t offset,
                       std::span<std::uint8_t> staging, bool write);

}
