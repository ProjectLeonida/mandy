#pragma once

// Detiling and tiling for AMD GFX10 / RDNA2 console-tiled resources (docs/d3d12_vulkan.md section 4.2).
//
// Resources created through Xbox slot 76 carry layout 0x101 (kXboxTiledLayout).
// The data layout inside each 64 KiB tile is GFX10 SW_64KB_D (Mesa addrlib GFX10_SW_64K_D_RBPLUS_PATINFO):
// 64 KiB macro-blocks are laid out in row-major order across the surface, and intra-tile coordinates (x, y)
// are swizzled into byte offsets via bit-permutation equations matching RDNA2 / Xbox Series hardware.
//
// Independent X and Y coordinate bit mappings permit O(1) intra-tile offset resolution via compile-time
// lookup tables (lut_x[x % tile_w] | lut_y[y % tile_h]). No Windows or Vulkan dependency.

#include "resource_tiling.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <span>

namespace leonida {

// The Xbox tiled layout word (device slot 76, docs/d3d12_vulkan.md section 4.2).
inline constexpr std::uint32_t xbox_tiled_layout = 0x101;

constexpr bool is_xbox_tiled(std::uint32_t layout) {
    return layout == xbox_tiled_layout;
}

// Intra-tile bit permutation equations for 64 KiB tiles (Mesa addrlib GFX10_SW_64K_D_RBPLUS_PATINFO).
// For each bit 0..15:
//   s > 0: bit is ((x >> (s - 1)) & 1)
//   s < 0: bit is ((y >> (-s - 1)) & 1)
//   s == 0: intra-element byte bit (element_bytes > 1)
template <std::uint32_t ElementBytes>
struct SwizzleLut64K {
    static constexpr std::uint32_t bpe = ElementBytes;
    static constexpr TileShape shape = element_tile_shape(false, bpe);
    static constexpr std::uint32_t tile_w = shape.width;
    static constexpr std::uint32_t tile_h = shape.height;

    static constexpr std::uint32_t run_elems = []() constexpr -> std::uint32_t {
        if constexpr (bpe == 1) return 8u;
        else if constexpr (bpe == 2) return 8u;
        else if constexpr (bpe == 4) return 4u;
        else if constexpr (bpe == 8) return 2u;
        else if constexpr (bpe == 16) return 2u;
        else return 1u;
    }();
    static constexpr std::uint32_t run_bytes = run_elems * bpe;
    static constexpr std::uint32_t runs_per_tile = tile_w / run_elems;

    std::array<std::uint16_t, tile_w> lut_x{};
    std::array<std::uint16_t, tile_h> lut_y{};
    std::array<std::uint16_t, runs_per_tile> lut_run_x{};

    constexpr SwizzleLut64K() {
        constexpr std::array<int8_t, 16> pat = []() {
            if constexpr (bpe == 1) {
                return std::array<int8_t, 16>{1, 2, 3, -2, -1, -3, 4, -4, -5, 5, -6, 6, -7, 7, -8, 8};
            } else if constexpr (bpe == 2) {
                return std::array<int8_t, 16>{0, 1, 2, 3, -1, -2, -3, 4, -4, 5, -5, 6, -6, 7, -7, 8};
            } else if constexpr (bpe == 4) {
                // RDNA2/RbPlus SW_64KB_D: bits 6..7 swapped vs Navi 1x ((3, -3) instead of (-3, 3))
                return std::array<int8_t, 16>{0, 0, 1, 2, -1, -2, 3, -3, -4, 4, -5, 5, -6, 6, -7, 7};
            } else if constexpr (bpe == 8) {
                return std::array<int8_t, 16>{0, 0, 0, 1, -1, 2, 3, -2, -3, 4, -4, 5, -5, 6, -6, 7};
            } else if constexpr (bpe == 16) {
                return std::array<int8_t, 16>{0, 0, 0, 0, 1, -1, 2, -2, -3, 3, -4, 4, -5, 5, -6, 6};
            } else {
                return std::array<int8_t, 16>{};
            }
        }();

        for (std::uint32_t x = 0; x < tile_w; ++x) {
            std::uint16_t val = 0;
            for (std::uint32_t bit = 0; bit < 16; ++bit) {
                const int8_t s = pat[bit];
                if (s > 0) {
                    val |= static_cast<std::uint16_t>(((x >> (s - 1)) & 1) << bit);
                }
            }
            lut_x[x] = val;
        }

        for (std::uint32_t y = 0; y < tile_h; ++y) {
            std::uint16_t val = 0;
            for (std::uint32_t bit = 0; bit < 16; ++bit) {
                const int8_t s = pat[bit];
                if (s < 0) {
                    val |= static_cast<std::uint16_t>(((y >> (-s - 1)) & 1) << bit);
                }
            }
            lut_y[y] = val;
        }

        for (std::uint32_t r = 0; r < runs_per_tile; ++r) {
            lut_run_x[r] = lut_x[r * run_elems];
        }
    }

    constexpr bool runs_valid() const {
        if (tile_w % run_elems != 0) return false;
        for (std::uint32_t r = 0; r < runs_per_tile; ++r) {
            if ((lut_run_x[r] & (run_bytes - 1u)) != 0u) return false;
            for (std::uint32_t k = 0; k < run_elems; ++k) {
                if (lut_x[r * run_elems + k] != static_cast<std::uint16_t>(lut_run_x[r] + k * bpe)) return false;
            }
        }
        for (std::uint32_t y = 0; y < tile_h; ++y) {
            if ((lut_y[y] & (run_bytes - 1u)) != 0u) return false;
            for (std::uint32_t r = 0; r < runs_per_tile; ++r) {
                if ((lut_run_x[r] & lut_y[y]) != 0u) return false;
            }
        }
        return true;
    }
};

inline constexpr SwizzleLut64K<1> swizzle_lut_64k_1bpe{};
inline constexpr SwizzleLut64K<2> swizzle_lut_64k_2bpe{};
inline constexpr SwizzleLut64K<4> swizzle_lut_64k_4bpe{};
inline constexpr SwizzleLut64K<8> swizzle_lut_64k_8bpe{};
inline constexpr SwizzleLut64K<16> swizzle_lut_64k_16bpe{};

static_assert(swizzle_lut_64k_1bpe.runs_valid());
static_assert(swizzle_lut_64k_2bpe.runs_valid());
static_assert(swizzle_lut_64k_4bpe.runs_valid());
static_assert(swizzle_lut_64k_8bpe.runs_valid());
static_assert(swizzle_lut_64k_16bpe.runs_valid());

namespace detail {

constexpr bool checked_mul(std::size_t a, std::size_t b, std::size_t& out) {
    if (a != 0 && b > (std::numeric_limits<std::size_t>::max)() / a) return false;
    out = a * b;
    return true;
}

struct SurfaceLayout64K {
    std::uint32_t blocks_w = 0;
    std::uint32_t blocks_h = 0;
    std::uint32_t tiles_x = 0;
    std::uint32_t tiles_y = 0;
    std::size_t linear_bytes = 0;
    std::size_t tiled_bytes = 0;
};

constexpr bool compute_surface_layout_64k(std::uint32_t width, std::uint32_t height,
                                          std::uint32_t element_bytes, std::uint32_t block_dim,
                                          SurfaceLayout64K& out) {
    if (!width || !height || (block_dim != 1 && block_dim != 4)) return false;
    const TileShape shape = element_tile_shape(false, element_bytes);
    if (!shape.width || !shape.height) return false;

    // Overflow-free ceiling division for positive uint32_t operands: 1 + (n - 1) / d.
    const std::uint32_t blocks_w = 1u + (width - 1u) / block_dim;
    const std::uint32_t blocks_h = 1u + (height - 1u) / block_dim;
    const std::uint32_t tiles_x = 1u + (blocks_w - 1u) / shape.width;
    const std::uint32_t tiles_y = 1u + (blocks_h - 1u) / shape.height;

    std::size_t block_count = 0;
    std::size_t linear_bytes = 0;
    std::size_t tile_count = 0;
    std::size_t tiled_bytes = 0;
    if (!checked_mul(blocks_w, blocks_h, block_count) ||
        !checked_mul(block_count, element_bytes, linear_bytes) ||
        !checked_mul(tiles_x, tiles_y, tile_count) ||
        !checked_mul(tile_count, tile_size_in_bytes, tiled_bytes)) {
        return false;
    }

    out = {blocks_w, blocks_h, tiles_x, tiles_y, linear_bytes, tiled_bytes};
    return true;
}

inline bool spans_overlap(const std::byte* a, std::size_t a_len,
                          const std::byte* b, std::size_t b_len) {
    if (!a_len || !b_len) return false;
    const std::less<const std::byte*> lt{};
    return lt(a, b + b_len) && lt(b, a + a_len);
}

}  // namespace detail

// Compute required tiled byte size for a 2D surface of the given dimensions (0 on invalid input or overflow).
constexpr std::size_t tiled_surface_size_64k(std::uint32_t width, std::uint32_t height,
                                            std::uint32_t element_bytes, std::uint32_t block_dim = 1) {
    detail::SurfaceLayout64K layout{};
    return detail::compute_surface_layout_64k(width, height, element_bytes, block_dim, layout)
               ? layout.tiled_bytes
               : 0;
}

// Compute required linear byte size for a 2D surface of the given dimensions (0 on invalid input or overflow).
constexpr std::size_t linear_surface_size(std::uint32_t width, std::uint32_t height,
                                         std::uint32_t element_bytes, std::uint32_t block_dim = 1) {
    detail::SurfaceLayout64K layout{};
    return detail::compute_surface_layout_64k(width, height, element_bytes, block_dim, layout)
               ? layout.linear_bytes
               : 0;
}

namespace detail {

template <std::uint32_t Bpe, bool ToTiled>
void transfer_typed(const std::byte* src, std::byte* dst,
                    std::uint32_t blocks_w, std::uint32_t blocks_h, std::uint32_t tiles_x) {
    constexpr const auto& lut = []() constexpr -> const auto& {
        if constexpr (Bpe == 1) return swizzle_lut_64k_1bpe;
        else if constexpr (Bpe == 2) return swizzle_lut_64k_2bpe;
        else if constexpr (Bpe == 4) return swizzle_lut_64k_4bpe;
        else if constexpr (Bpe == 8) return swizzle_lut_64k_8bpe;
        else if constexpr (Bpe == 16) return swizzle_lut_64k_16bpe;
    }();

    constexpr std::uint32_t tile_w = lut.tile_w;
    constexpr std::uint32_t tile_h = lut.tile_h;
    constexpr std::uint32_t run_elems = lut.run_elems;
    constexpr std::size_t run_bytes = lut.run_bytes;
    constexpr std::uint32_t runs_per_tile = lut.runs_per_tile;
    constexpr std::size_t tile_row_bytes = static_cast<std::size_t>(tile_w) * Bpe;

    const std::size_t linear_stride = static_cast<std::size_t>(blocks_w) * Bpe;
    const std::uint32_t full_tiles_x = blocks_w / tile_w;
    const std::uint32_t rem_x = blocks_w % tile_w;
    const std::uint32_t rem_full_runs = rem_x / run_elems;
    const std::size_t rem_tail_bytes = static_cast<std::size_t>(rem_x % run_elems) * Bpe;

    // Walk one 64 KiB macro-tile at a time so the swizzled side stays within a single 64 KiB cache footprint
    // while copying compile-time contiguous micro-runs (8/16/32 bytes) with zero inner-loop division/modulo.
    const std::uint32_t tiles_y = 1u + (blocks_h - 1u) / tile_h;
    for (std::uint32_t ty = 0; ty < tiles_y; ++ty) {
        const std::uint32_t y_base = ty * tile_h;
        const std::uint32_t y_end = std::min(tile_h, blocks_h - y_base);
        const std::size_t macro_y_base = static_cast<std::size_t>(ty) * tiles_x * tile_size_in_bytes;
        const std::size_t linear_y_base = static_cast<std::size_t>(y_base) * linear_stride;

        for (std::uint32_t tx = 0; tx < full_tiles_x; ++tx) {
            const std::size_t tile_base = macro_y_base + static_cast<std::size_t>(tx) * tile_size_in_bytes;
            std::size_t lin_row = linear_y_base + static_cast<std::size_t>(tx) * tile_row_bytes;
            for (std::uint32_t ly = 0; ly < y_end; ++ly, lin_row += linear_stride) {
                const std::size_t tile_row = tile_base + lut.lut_y[ly];
                for (std::uint32_t r = 0; r < runs_per_tile; ++r) {
                    const std::size_t t_off = tile_row + lut.lut_run_x[r];
                    const std::size_t l_off = lin_row + static_cast<std::size_t>(r) * run_bytes;
                    if constexpr (ToTiled) {
                        std::memcpy(dst + t_off, src + l_off, run_bytes);
                    } else {
                        std::memcpy(dst + l_off, src + t_off, run_bytes);
                    }
                }
            }
        }

        if (rem_x != 0u) {
            const std::size_t tile_base = macro_y_base + static_cast<std::size_t>(full_tiles_x) * tile_size_in_bytes;
            std::size_t lin_row = linear_y_base + static_cast<std::size_t>(full_tiles_x) * tile_row_bytes;
            const std::size_t tail_lin_rel = static_cast<std::size_t>(rem_full_runs) * run_bytes;
            for (std::uint32_t ly = 0; ly < y_end; ++ly, lin_row += linear_stride) {
                const std::size_t tile_row = tile_base + lut.lut_y[ly];
                for (std::uint32_t r = 0; r < rem_full_runs; ++r) {
                    const std::size_t t_off = tile_row + lut.lut_run_x[r];
                    const std::size_t l_off = lin_row + static_cast<std::size_t>(r) * run_bytes;
                    if constexpr (ToTiled) {
                        std::memcpy(dst + t_off, src + l_off, run_bytes);
                    } else {
                        std::memcpy(dst + l_off, src + t_off, run_bytes);
                    }
                }
                if (rem_tail_bytes != 0u) {
                    const std::size_t t_off = tile_row + lut.lut_run_x[rem_full_runs];
                    const std::size_t l_off = lin_row + tail_lin_rel;
                    if constexpr (ToTiled) {
                        std::memcpy(dst + t_off, src + l_off, rem_tail_bytes);
                    } else {
                        std::memcpy(dst + l_off, src + t_off, rem_tail_bytes);
                    }
                }
            }
        }
    }
}

template <std::uint32_t Bpe>
void detile_typed(std::span<const std::byte> tiled, std::span<std::byte> linear,
                  std::uint32_t blocks_w, std::uint32_t blocks_h, std::uint32_t tiles_x) {
    transfer_typed<Bpe, false>(tiled.data(), linear.data(), blocks_w, blocks_h, tiles_x);
}

template <std::uint32_t Bpe>
void tile_typed(std::span<const std::byte> linear, std::span<std::byte> tiled,
                std::uint32_t blocks_w, std::uint32_t blocks_h, std::uint32_t tiles_x) {
    transfer_typed<Bpe, true>(linear.data(), tiled.data(), blocks_w, blocks_h, tiles_x);
}

}  // namespace detail

// Detile a 2D surface from console-tiled 64 KiB layout into linear row-major order.
inline bool detile_surface_64k(std::span<const std::byte> tiled, std::span<std::byte> linear,
                               std::uint32_t width, std::uint32_t height,
                               std::uint32_t element_bytes, std::uint32_t block_dim = 1) {
    detail::SurfaceLayout64K layout{};
    if (!detail::compute_surface_layout_64k(width, height, element_bytes, block_dim, layout)) return false;
    if (linear.size() < layout.linear_bytes || tiled.size() < layout.tiled_bytes) return false;
    if (detail::spans_overlap(tiled.data(), layout.tiled_bytes, linear.data(), layout.linear_bytes)) return false;

    switch (element_bytes) {
    case 1:
        detail::detile_typed<1>(tiled, linear, layout.blocks_w, layout.blocks_h, layout.tiles_x);
        return true;
    case 2:
        detail::detile_typed<2>(tiled, linear, layout.blocks_w, layout.blocks_h, layout.tiles_x);
        return true;
    case 4:
        detail::detile_typed<4>(tiled, linear, layout.blocks_w, layout.blocks_h, layout.tiles_x);
        return true;
    case 8:
        detail::detile_typed<8>(tiled, linear, layout.blocks_w, layout.blocks_h, layout.tiles_x);
        return true;
    case 16:
        detail::detile_typed<16>(tiled, linear, layout.blocks_w, layout.blocks_h, layout.tiles_x);
        return true;
    default:
        return false;
    }
}

// Tile a 2D surface from linear row-major order into console-tiled 64 KiB layout.
inline bool tile_surface_64k(std::span<const std::byte> linear, std::span<std::byte> tiled,
                             std::uint32_t width, std::uint32_t height,
                             std::uint32_t element_bytes, std::uint32_t block_dim = 1) {
    detail::SurfaceLayout64K layout{};
    if (!detail::compute_surface_layout_64k(width, height, element_bytes, block_dim, layout)) return false;
    if (linear.size() < layout.linear_bytes || tiled.size() < layout.tiled_bytes) return false;
    if (detail::spans_overlap(linear.data(), layout.linear_bytes, tiled.data(), layout.tiled_bytes)) return false;

    switch (element_bytes) {
    case 1:
        detail::tile_typed<1>(linear, tiled, layout.blocks_w, layout.blocks_h, layout.tiles_x);
        return true;
    case 2:
        detail::tile_typed<2>(linear, tiled, layout.blocks_w, layout.blocks_h, layout.tiles_x);
        return true;
    case 4:
        detail::tile_typed<4>(linear, tiled, layout.blocks_w, layout.blocks_h, layout.tiles_x);
        return true;
    case 8:
        detail::tile_typed<8>(linear, tiled, layout.blocks_w, layout.blocks_h, layout.tiles_x);
        return true;
    case 16:
        detail::tile_typed<16>(linear, tiled, layout.blocks_w, layout.blocks_h, layout.tiles_x);
        return true;
    default:
        return false;
    }
}

}  // namespace leonida
