//! BC7 block decoder for `tools/dds_png.py`.
//!
//! A line-by-line port of `_decode_bc7_block` and `_decode_bc7` in `tools/dds_png.py`: the same
//! field order, endpoint unquantisation, weights, rounding, rotation and reserved-mode result,
//! so the RGBA8 bytes are identical. The partition masks and anchors are not copied here; the
//! caller passes the ones in `tools/bcpack_tables_bptc.py`. `tests/test_bc7_native.py` checks
//! that the two decoders agree.

use std::slice;

/// (subsets, partition bits, rotation bits, index-selector bits, colour bits, alpha bits,
///  p-bit per endpoint, p-bit per subset, index bits, second index bits), as `_BC7_MODES`.
const MODES: [[u32; 10]; 8] = [
    [3, 4, 0, 0, 4, 0, 1, 0, 3, 0],
    [2, 6, 0, 0, 6, 0, 0, 1, 3, 0],
    [3, 6, 0, 0, 5, 0, 0, 0, 2, 0],
    [2, 6, 0, 0, 7, 0, 1, 0, 2, 0],
    [1, 0, 2, 1, 5, 6, 0, 0, 2, 3],
    [1, 0, 2, 0, 7, 8, 0, 0, 2, 2],
    [1, 0, 0, 0, 7, 7, 1, 0, 4, 0],
    [2, 6, 0, 0, 5, 5, 1, 0, 2, 0],
];

const WEIGHTS2: [u32; 4] = [0, 21, 43, 64];
const WEIGHTS3: [u32; 8] = [0, 9, 18, 27, 37, 46, 55, 64];
const WEIGHTS4: [u32; 16] = [0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64];

/// Output byte of interpolated R, G, B, A under rotation 0-3, as `_BC7_ROTATIONS`.
const ROTATIONS: [[usize; 4]; 4] = [[0, 1, 2, 3], [3, 1, 2, 0], [0, 3, 2, 1], [0, 1, 3, 2]];

fn weights(index_bits: u32) -> &'static [u32] {
    match index_bits {
        2 => &WEIGHTS2,
        3 => &WEIGHTS3,
        _ => &WEIGHTS4,
    }
}

/// Endpoint precision (5..8 bits, p-bit included) to 8 bits by MSB replication.
fn unquantize(v: u32, prec: u32) -> u32 {
    (v << (8 - prec)) | (v >> (2 * prec - 8))
}

/// Per pixel: the subset it belongs to and whether it is an anchor (index MSB omitted).
#[derive(Clone, Copy)]
struct Layout {
    subset_of: [u8; 16],
    is_anchor: [u8; 16],
}

/// The partition layouts of one, two and three subsets, built from the caller's tables.
pub struct Layouts {
    one: Layout,
    two: [Layout; 64],
    three: [Layout; 64],
}

impl Layouts {
    /// `part2`/`part3` are `BC7_PART2_PACKED`/`BC7_PART3_PACKED`; the anchors are
    /// `BC7_ANCHOR2`, `BC7_ANCHOR3A` and `BC7_ANCHOR3B`.
    pub fn new(part2: &[u32; 64], part3: &[u32; 64], anchor2: &[u8; 64], anchor3a: &[u8; 64],
               anchor3b: &[u8; 64]) -> Layouts {
        let empty = Layout { subset_of: [0; 16], is_anchor: [0; 16] };
        let mut one = empty;
        one.is_anchor[0] = 1;
        let mut two = [empty; 64];
        let mut three = [empty; 64];
        for p in 0..64 {
            for i in 0..16 {
                two[p].subset_of[i] = ((part2[p] >> i) & 1) as u8;
                three[p].subset_of[i] = ((part3[p] >> (2 * i)) & 3) as u8;
            }
            two[p].is_anchor[0] = 1;
            three[p].is_anchor[0] = 1;
            // An anchor index outside 0..16 marks no pixel, as `p in anchors` does in Python.
            if (anchor2[p] as usize) < 16 {
                two[p].is_anchor[anchor2[p] as usize] = 1;
            }
            for &a in &[anchor3a[p], anchor3b[p]] {
                if (a as usize) < 16 {
                    three[p].is_anchor[a as usize] = 1;
                }
            }
        }
        Layouts { one, two, three }
    }

    fn get(&self, subsets: u32, partition: usize) -> &Layout {
        match subsets {
            1 => &self.one,
            2 => &self.two[partition],
            _ => &self.three[partition],
        }
    }
}

fn take(bits: &mut u128, n: u32) -> u32 {
    let v = (*bits & ((1u128 << n) - 1)) as u32;
    *bits >>= n;
    v
}

fn read_indices(bits: &mut u128, index_bits: u32, is_anchor: &[u8; 16]) -> [u32; 16] {
    let mut out = [0u32; 16];
    for (o, &a) in out.iter_mut().zip(is_anchor.iter()) {
        *o = take(bits, index_bits - a as u32);
    }
    out
}

/// Decode one little-endian 128-bit BC7 block to 64 bytes of row-major RGBA8.
pub fn decode_block(block: &[u8; 16], layouts: &Layouts) -> [u8; 64] {
    let block = u128::from_le_bytes(*block);
    if block & 0xFF == 0 {
        return [0; 64];
    }
    let mode = block.trailing_zeros();
    let [subsets, partition_bits, rotation_bits, selector_bits, color_bits, alpha_bits,
         endpoint_pbits, shared_pbits, index_bits, index2_bits] = MODES[mode as usize];

    let mut bits = block >> (mode + 1);
    let partition = take(&mut bits, partition_bits) as usize;
    let rotation = take(&mut bits, rotation_bits) as usize;
    let selector = take(&mut bits, selector_bits);

    // All R endpoints, then all G, B and A, each ordered subset 0 end 0, subset 0 end 1, ...
    let endpoint_count = (2 * subsets) as usize;
    let channels = if alpha_bits != 0 { 4 } else { 3 };
    let mut raw = [[0u32; 6]; 4];
    for (c, values) in raw.iter_mut().enumerate().take(channels) {
        let width = if c == 3 { alpha_bits } else { color_bits };
        for v in values.iter_mut().take(endpoint_count) {
            *v = take(&mut bits, width);
        }
    }

    let mut pbits = [0u32; 6];
    let has_pbits = endpoint_pbits != 0 || shared_pbits != 0;
    if endpoint_pbits != 0 {
        for (e, p) in pbits.iter_mut().enumerate().take(endpoint_count) {
            *p = ((bits >> e) & 1) as u32;
        }
        bits >>= endpoint_count;
    } else if shared_pbits != 0 {
        for (e, p) in pbits.iter_mut().enumerate().take(endpoint_count) {
            *p = ((bits >> (e >> 1)) & 1) as u32;
        }
        bits >>= subsets;
    }

    let mut endpoints = [[0u32; 6]; 4];
    for c in 0..channels {
        let mut width = if c == 3 { alpha_bits } else { color_bits };
        if has_pbits {
            width += 1;
        }
        for e in 0..endpoint_count {
            let mut v = raw[c][e];
            if has_pbits {
                v = (v << 1) | pbits[e];
            }
            endpoints[c][e] = unquantize(v, width);
        }
    }

    let layout = layouts.get(subsets, partition);
    let mut color_indices = read_indices(&mut bits, index_bits, &layout.is_anchor);
    let mut color_weights = weights(index_bits);
    let mut alpha_weights = color_weights;
    let mut alpha_indices = color_indices;
    // Subset of each pixel for the colour and alpha palettes; modes 4 and 5 have one subset.
    let subset_of = &layout.subset_of;
    if index2_bits != 0 {
        // Modes 4 and 5: the selector swaps which index set drives colour and alpha.
        alpha_indices = read_indices(&mut bits, index2_bits, &layout.is_anchor);
        alpha_weights = weights(index2_bits);
        if selector != 0 {
            std::mem::swap(&mut color_indices, &mut alpha_indices);
            std::mem::swap(&mut color_weights, &mut alpha_weights);
        }
    }

    // Alpha stays 255 in modes 0-3; rotation is applied after interpolation.
    let mut out = [0xFFu8; 64];
    let order = &ROTATIONS[rotation];
    for (c, ends) in endpoints.iter().enumerate().take(channels) {
        let (w, idx) = if c == 3 { (alpha_weights, &alpha_indices) } else { (color_weights, &color_indices) };
        let dst = order[c];
        for px in 0..16 {
            let s = subset_of[px] as usize;
            let e0 = ends[2 * s];
            let e1 = ends[2 * s + 1];
            let wt = w[idx[px] as usize];
            out[px * 4 + dst] = ((e0 * (64 - wt) + e1 * wt + 32) >> 6) as u8;
        }
    }
    out
}

/// Decode a `width` x `height` BC7 surface (rows of `(width + 3) / 4` blocks) to RGBA8, cropping
/// the edge blocks. `data` must hold every block and `out` exactly `width * height * 4` bytes.
pub fn decode_surface(data: &[u8], width: usize, height: usize, layouts: &Layouts, out: &mut [u8]) {
    let blocks_x = width.div_ceil(4);
    let blocks_y = height.div_ceil(4);
    let row_stride = width * 4;
    let mut pos = 0;
    // Consecutive equal blocks are common (flat regions); keep the previous result.
    let mut last_key = [0u8; 16];
    let mut last_rgba = decode_block(&last_key, layouts);
    for by in 0..blocks_y {
        let py0 = by * 4;
        let max_y = (height - py0).min(4);
        for bx in 0..blocks_x {
            let mut key = [0u8; 16];
            key.copy_from_slice(&data[pos..pos + 16]);
            pos += 16;
            if key != last_key {
                last_rgba = decode_block(&key, layouts);
                last_key = key;
            }
            let px0 = bx * 4;
            let span = (width - px0).min(4) * 4;
            let mut dst = py0 * row_stride + px0 * 4;
            for ty in 0..max_y {
                out[dst..dst + span].copy_from_slice(&last_rgba[ty * 16..ty * 16 + span]);
                dst += row_stride;
            }
        }
    }
}

/// C entry point used by `tools/dds_png.py` through ctypes.
///
/// `anchors` holds 192 bytes: `BC7_ANCHOR2`, `BC7_ANCHOR3A`, then `BC7_ANCHOR3B`. Returns 0, or
/// -1 when a pointer is null or a length does not match the surface (nothing is written then).
///
/// # Safety
/// Each pointer must be valid for the length given or implied: `data_len` bytes, 64 `u32` for
/// `part2` and `part3`, 192 bytes for `anchors` and `out_len` writable bytes for `out`.
#[no_mangle]
pub unsafe extern "C" fn bc7_decode(data: *const u8, data_len: usize, width: usize, height: usize,
                                    part2: *const u32, part3: *const u32, anchors: *const u8,
                                    out: *mut u8, out_len: usize) -> isize {
    if data.is_null() || part2.is_null() || part3.is_null() || anchors.is_null() || out.is_null() {
        return -1;
    }
    let blocks = match width.div_ceil(4).checked_mul(height.div_ceil(4)) {
        Some(b) => b,
        None => return -1,
    };
    let need_in = match blocks.checked_mul(16) {
        Some(n) => n,
        None => return -1,
    };
    let need_out = match width.checked_mul(height).and_then(|n| n.checked_mul(4)) {
        Some(n) => n,
        None => return -1,
    };
    if data_len < need_in || out_len != need_out {
        return -1;
    }
    // SAFETY: the caller guarantees the lengths above (see the function's safety section).
    let (data, part2, part3, anchors, out) = unsafe {
        (
            slice::from_raw_parts(data, data_len),
            &*(part2 as *const [u32; 64]),
            &*(part3 as *const [u32; 64]),
            slice::from_raw_parts(anchors, 192),
            slice::from_raw_parts_mut(out, out_len),
        )
    };
    let mut a2 = [0u8; 64];
    let mut a3a = [0u8; 64];
    let mut a3b = [0u8; 64];
    a2.copy_from_slice(&anchors[0..64]);
    a3a.copy_from_slice(&anchors[64..128]);
    a3b.copy_from_slice(&anchors[128..192]);
    let layouts = Layouts::new(part2, part3, &a2, &a3a, &a3b);
    decode_surface(data, width, height, &layouts, out);
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layouts() -> Layouts {
        // Partition 0 only matters for these tests; the rest stay subset 0.
        let mut part2 = [0u32; 64];
        let mut part3 = [0u32; 64];
        part2[0] = 0xCCCC;
        part3[0] = 0xAA50_5000;
        let a2 = [15u8; 64];
        let a3a = [3u8; 64];
        let a3b = [15u8; 64];
        Layouts::new(&part2, &part3, &a2, &a3a, &a3b)
    }

    fn pack(fields: &[(u128, u32)]) -> [u8; 16] {
        let mut v = 0u128;
        let mut at = 0;
        for &(value, width) in fields {
            v |= value << at;
            at += width;
        }
        assert_eq!(at, 128);
        v.to_le_bytes()
    }

    #[test]
    fn reserved_mode_is_transparent_black() {
        let mut block = [0xFFu8; 16];
        block[0] = 0;
        assert_eq!(decode_block(&block, &layouts()), [0u8; 64]);
    }

    #[test]
    fn mode6_endpoints_and_weights() {
        // Mode 6: R/G/B/A endpoints 7 bits, one p-bit per endpoint, 4-bit indices.
        // End 0 = 0 (p 0), end 1 = 127 (p 1) on every channel: 0 and 255 after unquantisation.
        let mut fields = vec![(1u128 << 6, 7)];
        for _ in 0..4 {
            fields.push((0, 7));
            fields.push((127, 7));
        }
        fields.push((0, 1));
        fields.push((1, 1));
        // Pixel 0 (anchor) index 7 in three bits, the rest 15.
        fields.push((7, 3));
        for _ in 1..16 {
            fields.push((15, 4));
        }
        let out = decode_block(&pack(&fields), &layouts());
        let mid = ((255 * 30 + 32) >> 6) as u8;
        assert_eq!(&out[0..4], &[mid; 4]);
        assert_eq!(&out[4..8], &[255; 4]);
    }

    #[test]
    fn surface_crops_edge_blocks() {
        let layouts = layouts();
        let mut block = [0u8; 16];
        block[0] = 0x40; // mode 6, everything else zero
        let one = decode_block(&block, &layouts);
        let mut data = Vec::new();
        for _ in 0..4 {
            data.extend_from_slice(&block);
        }
        let mut out = vec![0u8; 5 * 6 * 4];
        decode_surface(&data, 5, 6, &layouts, &mut out);
        for y in 0..6 {
            for x in 0..5 {
                let src = ((y % 4) * 4 + (x % 4)) * 4;
                assert_eq!(&out[(y * 5 + x) * 4..(y * 5 + x) * 4 + 4], &one[src..src + 4]);
            }
        }
    }

    #[test]
    fn ffi_rejects_bad_lengths() {
        let data = [0u8; 16];
        let part = [0u32; 64];
        let anchors = [15u8; 192];
        let mut out = [0u8; 64];
        // SAFETY: every pointer is valid for the lengths passed.
        unsafe {
            assert_eq!(bc7_decode(data.as_ptr(), 16, 4, 4, part.as_ptr(), part.as_ptr(), anchors.as_ptr(),
                                  out.as_mut_ptr(), 64), 0);
            assert_eq!(bc7_decode(data.as_ptr(), 15, 4, 4, part.as_ptr(), part.as_ptr(), anchors.as_ptr(),
                                  out.as_mut_ptr(), 64), -1);
            assert_eq!(bc7_decode(data.as_ptr(), 16, 4, 4, part.as_ptr(), part.as_ptr(), anchors.as_ptr(),
                                  out.as_mut_ptr(), 63), -1);
            assert_eq!(bc7_decode(std::ptr::null(), 16, 4, 4, part.as_ptr(), part.as_ptr(), anchors.as_ptr(),
                                  out.as_mut_ptr(), 64), -1);
        }
    }
}
