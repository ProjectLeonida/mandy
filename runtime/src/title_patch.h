#pragma once
// LEONIDA_TITLE_PATCH: checked byte patches of the title's image, named in the environment.
//
//   LEONIDA_TITLE_PATCH="<rva>:<expected hex>:<new hex>[,<rva>:<expected hex>:<new hex>...]"   ('+' works as well as
//   ',' between patches, for launchers whose own lists are comma-separated)
//
// Before the title runs, each place is rewritten only when the bytes there are exactly the expected ones, the two
// strings have the same length (1-64 bytes) and the place lies inside the image. A place that does not match is left
// alone. Like LEONIDA_TITLE_WATCHDOG_KICK, the RVAs and bytes are measured locally and are not kept in the runtime:
// this is the mechanism, not a list of patches. An example is the title's choice of upscaler, see docs/running.md.
#include <windows.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace rt::title_patch {

struct Patch {
    std::uint64_t rva = 0;
    std::vector<std::uint8_t> expect, replace;
};

enum class Result { applied, mismatch, outside, unreadable, protect_failed };

inline bool parse_hex(const char* text, std::size_t len, std::vector<std::uint8_t>& out) {
    out.clear();
    if (len == 0 || len % 2 || len > 128) return false;
    auto nibble = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
    };
    for (std::size_t i = 0; i < len; i += 2) {
        const int hi = nibble(text[i]), lo = nibble(text[i + 1]);
        if (hi < 0 || lo < 0) return false;
        out.push_back(static_cast<std::uint8_t>(hi << 4 | lo));
    }
    return true;
}

// The patches of a specification, in order; each item that is not <rva>:<hex>:<hex> of equal lengths goes to `bad`.
inline std::vector<Patch> parse(const char* spec, std::vector<std::string>* bad = nullptr) {
    std::vector<Patch> out;
    for (const char* at = spec ? spec : ""; *at;) {
        const char* end = at + std::strcspn(at, ",+");
        const std::string item(at, end);
        at = *end ? end + 1 : end;
        if (item.empty()) continue;
        const std::size_t c1 = item.find(':');
        const std::size_t c2 = c1 == std::string::npos ? c1 : item.find(':', c1 + 1);
        Patch p;
        char* rva_end = nullptr;
        p.rva = std::strtoull(item.c_str(), &rva_end, 16);
        if (c2 == std::string::npos || rva_end != item.c_str() + c1 || c1 == 0 ||
            !parse_hex(item.c_str() + c1 + 1, c2 - c1 - 1, p.expect) ||
            !parse_hex(item.c_str() + c2 + 1, item.size() - c2 - 1, p.replace) || p.expect.size() != p.replace.size()) {
            if (bad) bad->push_back(item);
            continue;
        }
        out.push_back(std::move(p));
    }
    return out;
}

// Writes one patch into the image at `base` (`size` bytes) when the expected bytes are there. Every page the patch
// covers is checked first (committed, readable, not a guard page), region by region, so a patch that runs into an
// uncommitted or no-access region is refused before anything is read; each region keeps its own protection.
inline Result apply(std::uint8_t* base, std::size_t size, const Patch& p) {
    if (p.expect.empty() || p.rva > size || p.expect.size() > size - p.rva) return Result::outside;
    std::uint8_t* const where = base + p.rva;
    std::uint8_t* const end = where + p.expect.size();
    struct Piece {
        std::uint8_t* at;
        std::size_t len;
        DWORD old;
    };
    Piece pieces[8]{};  // a patch is at most 64 bytes: it spans at most two pages, so few regions
    std::size_t count = 0;
    for (std::uint8_t* at = where; at < end;) {
        MEMORY_BASIC_INFORMATION mbi{};
        if (count == std::size(pieces) || !VirtualQuery(at, &mbi, sizeof(mbi)) || mbi.State != MEM_COMMIT ||
            !mbi.Protect || (mbi.Protect & (PAGE_NOACCESS | PAGE_GUARD)))
            return Result::unreadable;
        std::uint8_t* const region_end = static_cast<std::uint8_t*>(mbi.BaseAddress) + mbi.RegionSize;
        std::uint8_t* const stop = region_end < end ? region_end : end;
        pieces[count++] = {at, static_cast<std::size_t>(stop - at), 0};
        at = stop;
    }
    if (std::memcmp(where, p.expect.data(), p.expect.size()) != 0) return Result::mismatch;
    for (std::size_t i = 0; i < count; ++i) {
        if (!VirtualProtect(pieces[i].at, pieces[i].len, PAGE_EXECUTE_READWRITE, &pieces[i].old)) {
            DWORD ignored = 0;
            while (i-- > 0) VirtualProtect(pieces[i].at, pieces[i].len, pieces[i].old, &ignored);
            return Result::protect_failed;
        }
    }
    std::memcpy(where, p.replace.data(), p.replace.size());
    for (std::size_t i = 0; i < count; ++i) {
        DWORD ignored = 0;
        VirtualProtect(pieces[i].at, pieces[i].len, pieces[i].old, &ignored);
    }
    FlushInstructionCache(GetCurrentProcess(), where, p.replace.size());
    return Result::applied;
}

}  // namespace rt::title_patch
