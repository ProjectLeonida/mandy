#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <type_traits>

namespace leonida::gfx10 {

enum class TileMode { sw_4kb_s, sw_64kb_d };

namespace tiling_detail {

// Mesa addrlib GFX10_SW_4K_S_RBPLUS_PATINFO / GFX10_SW_64K_D_RBPLUS_PATINFO;
// tools/ttpf_mips.py patterns, corpus validation in docs/textures.md. Zero bits address bytes within an element.
inline constexpr int patterns_4k[5][12] = {
    {1, 2, 3, 4, -1, -2, -3, -4, -5, 5, -6, 6},
    {0, 1, 2, 3, -1, -2, -3, 4, -4, 5, -5, 6},
    {0, 0, 1, 2, -1, -2, -3, 3, -4, 4, -5, 5},
    {0, 0, 0, 1, -1, -2, 2, 3, -3, 4, -4, 5},
    {0, 0, 0, 0, -1, -2, 1, 2, -3, 3, -4, 4}};
inline constexpr int patterns_64k[5][16] = {
    {1, 2, 3, -2, -1, -3, 4, -4, -5, 5, -6, 6, -7, 7, -8, 8},
    {0, 1, 2, 3, -1, -2, -3, 4, -4, 5, -5, 6, -6, 7, -7, 8},
    {0, 0, 1, 2, -1, -2, 3, -3, -4, 4, -5, 5, -6, 6, -7, 7},
    {0, 0, 0, 1, -1, 2, 3, -2, -3, 4, -4, 5, -5, 6, -6, 7},
    {0, 0, 0, 0, 1, -1, 2, -2, -3, 3, -4, 4, -5, 5, -6, 6}};

inline bool overlap(std::span<const std::byte> source, std::span<std::byte> destination) {
    const auto a = reinterpret_cast<std::uintptr_t>(source.data());
    const auto b = reinterpret_cast<std::uintptr_t>(destination.data());
    return a <= b ? b - a < source.size() : a - b < destination.size();
}

}  // namespace tiling_detail

// One standalone 2D level, no mip tail or tile XOR. BC formats use a compressed block as one element.
class TiledSurface2D {
public:
    static bool make(TileMode mode, std::size_t element_bytes, std::size_t width, std::size_t height,
                     TiledSurface2D& out) {
        if (!width || !height) return false;
        unsigned index = 0;
        switch (element_bytes) {
        case 1: break;
        case 2: index = 1; break;
        case 4: index = 2; break;
        case 8: index = 3; break;
        case 16: index = 4; break;
        default: return false;
        }
        TiledSurface2D next;
        switch (mode) {
        case TileMode::sw_4kb_s:
            next.pattern_ = tiling_detail::patterns_4k[index];
            next.bits_ = 12;
            break;
        case TileMode::sw_64kb_d:
            next.pattern_ = tiling_detail::patterns_64k[index];
            next.bits_ = 16;
            break;
        default: return false;
        }
        unsigned x_bits = 0, y_bits = 0;
        for (unsigned bit = 0; bit < next.bits_; ++bit) {
            x_bits += next.pattern_[bit] > 0;
            y_bits += next.pattern_[bit] < 0;
        }
        next.tile_width_ = std::size_t{1} << x_bits;
        next.tile_height_ = std::size_t{1} << y_bits;
        next.tiles_x_ = (width - 1) / next.tile_width_ + 1;
        const std::size_t tiles_y = (height - 1) / next.tile_height_ + 1;
        const std::size_t tile_bytes = std::size_t{1} << next.bits_;
        constexpr auto max = std::numeric_limits<std::size_t>::max();
        if (next.tiles_x_ > max / tiles_y || next.tiles_x_ * tiles_y > max / tile_bytes) return false;
        next.size_ = next.tiles_x_ * tiles_y * tile_bytes;
        next.width_ = width;
        next.height_ = height;
        next.element_bytes_ = element_bytes;
        const unsigned bpe_bits = static_cast<unsigned>(std::countr_zero(element_bytes));
        unsigned contig_x_bits = 0;
        while (bpe_bits + contig_x_bits < next.bits_ &&
               next.pattern_[bpe_bits + contig_x_bits] == static_cast<int>(contig_x_bits + 1)) {
            ++contig_x_bits;
        }
        next.run_elements_ = std::size_t{1} << contig_x_bits;
        next.run_bytes_ = next.run_elements_ * element_bytes;
        out = next;
        return true;
    }

    std::size_t tiled_size() const { return size_; }
    std::size_t tile_width() const { return tile_width_; }
    std::size_t tile_height() const { return tile_height_; }

    bool element_offset(std::size_t x, std::size_t y, std::size_t& offset) const {
        if (!size_ || x >= width_ || y >= height_) return false;
        offset = address(x, y);
        return true;
    }

    // Buffers must not overlap. Padding in either destination is left untouched; failure writes nothing.
    bool detile(std::span<const std::byte> tiled, std::span<std::byte> linear, std::size_t row_pitch) const {
        return copy(tiled, linear, row_pitch, false);
    }

    bool tile(std::span<const std::byte> linear, std::span<std::byte> tiled, std::size_t row_pitch) const {
        return copy(linear, tiled, row_pitch, true);
    }

private:
    std::size_t address(std::size_t x, std::size_t y) const {
        std::size_t offset = ((y / tile_height_) * tiles_x_ + x / tile_width_) << bits_;
        for (unsigned bit = 0; bit < bits_; ++bit) {
            const int source = pattern_[bit];
            if (source > 0) offset |= ((x >> (source - 1)) & 1) << bit;
            else if (source < 0) offset |= ((y >> (-source - 1)) & 1) << bit;
        }
        return offset;
    }

    template<bool ToTiled, std::size_t RunBytes>
    void copy_runs(const std::byte* src, std::byte* dst, std::size_t pitch,
                   const std::uint32_t* x_offsets, const std::uint32_t* y_offsets) const {
        const auto tile_y_bits = std::countr_zero(tile_height_);
        const std::size_t tile_y_mask = tile_height_ - 1;
        const std::size_t runs_per_tile = tile_width_ / run_elements_;
        const std::size_t full_tiles = width_ / tile_width_;
        const std::size_t rem_elems = width_ - full_tiles * tile_width_;
        const std::size_t rem_full_runs = rem_elems / run_elements_;
        const std::size_t rem_tail_bytes = (rem_elems - rem_full_runs * run_elements_) * element_bytes_;
        for (std::size_t y = 0; y < height_; ++y) {
            const std::size_t row_tile_base = ((y >> tile_y_bits) * tiles_x_) << bits_;
            const std::size_t y_off = y_offsets[y & tile_y_mask];
            std::size_t lin = y * pitch;
            for (std::size_t tx = 0; tx < full_tiles; ++tx) {
                const std::size_t tile_base = row_tile_base + (tx << bits_) + y_off;
                for (std::size_t r = 0; r < runs_per_tile; ++r) {
                    const std::size_t tiled = tile_base + x_offsets[r];
                    if constexpr (ToTiled) std::memcpy(dst + tiled, src + lin, RunBytes);
                    else std::memcpy(dst + lin, src + tiled, RunBytes);
                    lin += RunBytes;
                }
            }
            if (rem_elems) {
                const std::size_t tile_base = row_tile_base + (full_tiles << bits_) + y_off;
                for (std::size_t r = 0; r < rem_full_runs; ++r) {
                    const std::size_t tiled = tile_base + x_offsets[r];
                    if constexpr (ToTiled) std::memcpy(dst + tiled, src + lin, RunBytes);
                    else std::memcpy(dst + lin, src + tiled, RunBytes);
                    lin += RunBytes;
                }
                if (rem_tail_bytes) {
                    const std::size_t tiled = tile_base + x_offsets[rem_full_runs];
                    if constexpr (ToTiled) std::memcpy(dst + tiled, src + lin, rem_tail_bytes);
                    else std::memcpy(dst + lin, src + tiled, rem_tail_bytes);
                }
            }
        }
    }

    bool copy(std::span<const std::byte> source, std::span<std::byte> destination, std::size_t pitch,
              bool to_tiled) const {
        if (!size_) return false;
        const std::size_t row_bytes = width_ * element_bytes_;
        if (pitch < row_bytes || (height_ - 1 &&
            pitch > (std::numeric_limits<std::size_t>::max() - row_bytes) / (height_ - 1))) return false;
        const std::size_t linear_size = (height_ - 1) * pitch + row_bytes;
        if (source.size() < (to_tiled ? linear_size : size_) ||
            destination.size() < (to_tiled ? size_ : linear_size) ||
            tiling_detail::overlap(source, destination)) return false;
        const std::size_t runs_per_tile = tile_width_ / run_elements_;
        std::uint32_t x_offsets[256]{};
        for (std::size_t r = 0; r < runs_per_tile; ++r) {
            const std::size_t x = r * run_elements_;
            std::uint32_t off = 0;
            for (unsigned bit = 0; bit < bits_; ++bit) {
                const int source_bit = pattern_[bit];
                if (source_bit > 0) off |= static_cast<std::uint32_t>(((x >> (source_bit - 1)) & 1) << bit);
            }
            x_offsets[r] = off;
        }
        const std::size_t y_entries = height_ < tile_height_ ? height_ : tile_height_;
        std::uint32_t y_offsets[256]{};
        for (std::size_t y = 0; y < y_entries; ++y) {
            std::uint32_t off = 0;
            for (unsigned bit = 0; bit < bits_; ++bit) {
                const int source_bit = pattern_[bit];
                if (source_bit < 0) off |= static_cast<std::uint32_t>(((y >> (-source_bit - 1)) & 1) << bit);
            }
            y_offsets[y] = off;
        }
        const auto dispatch = [&](auto tag) {
            constexpr bool kToTiled = decltype(tag)::value;
            switch (run_bytes_) {
            case 8: copy_runs<kToTiled, 8>(source.data(), destination.data(), pitch, x_offsets, y_offsets); break;
            case 16: copy_runs<kToTiled, 16>(source.data(), destination.data(), pitch, x_offsets, y_offsets); break;
            case 32: copy_runs<kToTiled, 32>(source.data(), destination.data(), pitch, x_offsets, y_offsets); break;
            default: {
                const auto tile_x_bits = std::countr_zero(tile_width_);
                const auto tile_y_bits = std::countr_zero(tile_height_);
                for (std::size_t y = 0; y < height_; ++y) {
                    const std::size_t tile_row = (y >> tile_y_bits) * tiles_x_;
                    const std::size_t y_off = y_offsets[y & (tile_height_ - 1)];
                    for (std::size_t x = 0; x < width_; ++x) {
                        const std::size_t lin = y * pitch + x * element_bytes_;
                        const std::size_t tiled = ((tile_row + (x >> tile_x_bits)) << bits_) + y_off +
                                                  x_offsets[(x & (tile_width_ - 1)) / run_elements_] +
                                                  (x & (run_elements_ - 1)) * element_bytes_;
                        if constexpr (kToTiled)
                            std::memcpy(destination.data() + tiled, source.data() + lin, element_bytes_);
                        else
                            std::memcpy(destination.data() + lin, source.data() + tiled, element_bytes_);
                    }
                }
                break;
            }
            }
        };
        if (to_tiled) dispatch(std::true_type{});
        else dispatch(std::false_type{});
        return true;
    }

    const int* pattern_ = nullptr;
    unsigned bits_ = 0;
    std::size_t width_ = 0, height_ = 0, element_bytes_ = 0;
    std::size_t tile_width_ = 0, tile_height_ = 0, tiles_x_ = 0, size_ = 0;
    std::size_t run_elements_ = 1, run_bytes_ = 1;
};

}  // namespace leonida::gfx10
