// BCPack decoder for the DirectStorage worker: a port of tools/bcpack.py (docs/textures.md, "What BCPack is").
//
// BCPack is the console's entropy coder for BCn texture data: DirectStorage inflates a request (zlib), then decodes
// BCPack into BCn blocks, then swizzles (docs/directstorage.md). This file is the middle step only, portable (no
// windows.h), so it builds and tests on every OS. The lane layer is a line-by-line port of
// tools/bcpack_calibration/src/model.rs (itself a port of `_decode_lanes_raw`); the block rebuilds follow
// tools/bcpack_calibration/src/blocks.rs and `_bc1_blocks` .. `_bc5_blocks`. Strict like `bcpack.decode` by default:
// a block that refers to a built-in history entry that is not confirmed is refused, never guessed. The tables come
// from tools/bcpack_tables.py through tools/bcpack_header.py (bcpack_tables.h).
//
// Done: BCPack modes 1 (BC1), 3 (BC3), 4 (BC4) and 5 (BC5). Not yet: 6 (BC6H) and 7 (BC7), which return
// Status::unsupported_codec.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace leonida::bcpack {

enum class Status {
    ok,
    unsupported_codec,    // not a BCPack mode this decoder rebuilds
    bad_size,             // the output is not a whole number of blocks of the mode
    bad_stream,           // the lane layer refuses the stream (corrupt, cut short, a missing model, ...)
    trailing_data,        // non-zero bytes after the last lane finished
    lane1_nonzero,        // lane 1 carries data (unexpected for BC1..BC5)
    bad_symbol,           // a symbol outside its lookup table
    unconfirmed_history,  // a block refers to a built-in history entry that is not confirmed
};

const char* status_name(Status status);

// The lane widths of a BCPack mode (1, 3, 4, 5, 6 or 7); empty for another value.
std::span<const std::uint8_t> lane_widths(std::uint32_t mode);
// Bytes per block of a BCPack mode: 8 for BC1 and BC4, 16 for the others; 0 for another value.
std::uint32_t block_bytes(std::uint32_t mode);

// Entropy layer: the symbols of every lane, lane-major (`count` per lane), and the bytes of `in` it used.
struct Lanes {
    std::vector<std::uint16_t> symbols;
    std::size_t count = 0;
    std::size_t used = 0;
    std::uint16_t at(std::size_t lane, std::size_t i) const { return symbols[lane * count + i]; }
};
// `count` symbols per lane for `widths`; Status::bad_stream where tools/bcpack.py raises.
Status decode_lanes(std::span<const std::uint8_t> in, std::span<const std::uint8_t> widths, std::size_t count,
                    Lanes& out);

// Confirmed history entries from the user's calibrated tables (tools/bcpack_history_calib.py), to use besides the
// built-in ones: entry e of mode m at values[m][e * K .. e * K + K) with K = 6, 8, 2, 4 for modes 1, 3, 4, 5; -1 or a
// short vector means "not given". An entry given here wins over the built-in one (`_resolve_bc_history`'s merge).
struct History {
    std::array<std::vector<std::int16_t>, 8> values;
};

// Decodes one BCPack chunk of `mode` into `out`, whose size is the decoded size (a whole number of blocks). On any
// status but ok, `out` is left in an unspecified state.
Status decode(std::span<const std::uint8_t> in, std::uint32_t mode, std::span<std::uint8_t> out,
              const History* history = nullptr);

}  // namespace leonida::bcpack
