//! CPU rehydration of XDXR BVH4 structures, following tools/bvh_rehydrate.py and docs/bvh.md §4a.

use std::slice;

const TRI0: u32 = 0;
const TRI1: u32 = 1;
const BOX16: u32 = 4;
const BOX32: u32 = 5;
const INVALID: u32 = u32::MAX;
const FLAG_DEHYDRATED: u32 = 0x0100_0000;
const FLAG_UPDATABLE: u32 = 0x0004_0000;
const LEAVES_PER_CHUNK: usize = 2048;
const GROUP: usize = 64;
const ABSENT32: [f32; 6] = [
    f32::MAX,
    f32::MAX,
    f32::MAX,
    -f32::MAX,
    -f32::MAX,
    -f32::MAX,
];
const ABSENT16: [u16; 6] = [0x7c00, 0x7c00, 0x7c00, 0xfc00, 0xfc00, 0xfc00];

#[repr(C)]
pub struct Geometry {
    vertices: *const u8,
    vertices_len: usize,
    indices: *const u8,
    indices_len: usize,
    stride: usize,
    index_width: usize,
    transform: *const [f64; 12],
}

struct GeometryView<'a> {
    vertices: &'a [u8],
    indices: &'a [u8],
    stride: usize,
    index_width: usize,
    transform: Option<&'a [f64; 12]>,
}

fn read<const N: usize>(bytes: &[u8], at: usize) -> Option<[u8; N]> {
    bytes.get(at..at + N)?.try_into().ok()
}

fn u16_at(bytes: &[u8], at: usize) -> Option<u16> {
    Some(u16::from_le_bytes(read(bytes, at)?))
}

fn u24_at(bytes: &[u8], at: usize) -> Option<u32> {
    let [a, b, c] = read(bytes, at)?;
    Some(u32::from_le_bytes([a, b, c, 0]))
}

fn u32_at(bytes: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_le_bytes(read(bytes, at)?))
}

fn f32_at(bytes: &[u8], at: usize) -> Option<f32> {
    Some(f32::from_bits(u32_at(bytes, at)?))
}

fn store32(bytes: &mut [u8], at: usize, value: u32) {
    bytes[at..at + 4].copy_from_slice(&value.to_le_bytes());
}

// Python's operations use doubles until each explicit struct.pack("<f"). Keep that order,
// including the add in the second and third terms. A non-finite intermediate can raise
// OverflowError or produce a platform-dependent NaN, so Python handles it.
#[inline]
fn round32(value: f64) -> Option<f32> {
    let rounded = value as f32;
    rounded.is_finite().then_some(rounded)
}

#[inline]
fn dot(coefficients: [f64; 3], [x, y, z]: [f32; 3]) -> Option<f32> {
    let [a, b, c] = coefficients;
    let acc = round32(a * f64::from(x))?;
    let acc = round32(b * f64::from(y) + f64::from(acc))?;
    round32(c * f64::from(z) + f64::from(acc))
}

impl GeometryView<'_> {
    fn triangle(&self, prim: u32) -> Option<[usize; 3]> {
        let start = 3 * prim as usize;
        match self.index_width {
            0 => Some([start, start + 1, start + 2]),
            2 => {
                let at = start * 2;
                Some([
                    u16_at(self.indices, at)? as usize,
                    u16_at(self.indices, at + 2)? as usize,
                    u16_at(self.indices, at + 4)? as usize,
                ])
            }
            4 => {
                let at = start * 4;
                Some([
                    u32_at(self.indices, at)? as usize,
                    u32_at(self.indices, at + 4)? as usize,
                    u32_at(self.indices, at + 8)? as usize,
                ])
            }
            _ => None,
        }
    }

    fn vertex(&self, index: usize, frame: Option<&[f32; 15]>) -> Option<[f32; 3]> {
        let at = index.checked_mul(self.stride)?;
        let vertex = self.vertices.get(at..)?.get(..12)?;
        let mut point = [f32_at(vertex, 0)?, f32_at(vertex, 4)?, f32_at(vertex, 8)?];
        // Let struct handle NaN payloads and signaling NaNs from input float32 values.
        if point.iter().any(|x| x.is_nan()) {
            return None;
        }
        if let Some(matrix) = self.transform {
            let mut transformed = [0.0; 3];
            for (value, row) in transformed.iter_mut().zip(matrix.chunks_exact(4)) {
                let acc = dot([row[0], row[1], row[2]], point)?;
                *value = round32(f64::from(acc) + row[3])?;
            }
            point = transformed;
        }
        if let Some(frame) = frame {
            let mut transformed = [0.0; 3];
            for (axis, (value, row)) in transformed
                .iter_mut()
                .zip(frame[..9].chunks_exact(3))
                .enumerate()
            {
                let acc = dot([row[0].into(), row[1].into(), row[2].into()], point)?;
                let shifted = round32(f64::from(acc) - f64::from(frame[9 + axis]))?;
                *value = round32(f64::from(shifted) * f64::from(frame[12 + axis]))?;
            }
            point = transformed;
        }
        Some(point)
    }
}

struct Header {
    flags: u32,
    elements: usize,
    count: usize,
    nodes: usize,
    parents: u32,
    root: u32,
}

impl Header {
    fn read(src: &[u8]) -> Option<Self> {
        Some(Self {
            flags: u32_at(src, 0)?,
            elements: u32_at(src, 52)? as usize,
            count: u32_at(src, 12)? as usize,
            nodes: u32_at(src, 40)? as usize,
            parents: u32_at(src, 56)?,
            root: u32_at(src, 48)?,
        })
    }

    fn write_parent(&self, dst: &mut [u8], child: u32, parent: u32) {
        let at = node_offset(child) as u32;
        let unit = at.wrapping_sub(self.elements as u32) >> 6;
        let entry = self.parents.wrapping_add(4 * unit) as usize;
        if entry >= 64 && entry + 4 <= self.elements.min(dst.len()) {
            store32(dst, entry, parent);
        }
    }
}

fn node_offset(ptr: u32) -> usize {
    ((ptr >> 3) as usize) << 6
}

fn write_leaves(
    src: &[u8],
    dst: &mut [u8],
    header: &Header,
    geometries: &[GeometryView<'_>],
    frame: Option<&[f32; 15]>,
) -> Option<()> {
    let mut remaining = header.count;
    let mut base = header.elements;
    while remaining != 0 {
        let count = remaining.min(LEAVES_PER_CHUNK);
        let packed = src.get(base..base + 11 * count)?;
        let output = dst.get_mut(base..base + 64 * count)?;
        for (i, leaf) in output.chunks_exact_mut(64).enumerate() {
            let gi = u24_at(packed, 3 * i)?;
            let flags = u32::from(packed[3 * count + i]);
            let prim0 = u24_at(packed, 4 * count + 3 * i)?;
            let prim1 = u24_at(packed, 7 * count + 3 * i)?;
            let order = u32::from(packed[10 * count + i]);
            let a = (order & 3) as usize;
            let b = ((order >> 2) & 3) as usize;
            if a > 2 || b > 2 || a == b {
                return None;
            }

            let geometry = geometries.get(gi as usize)?;
            let slots = [2, 0, 1];
            let triangle = geometry.triangle(prim0)?;
            for (slot, index) in [slots[a], slots[b], slots[3 - a - b]]
                .into_iter()
                .zip(triangle)
            {
                let point = geometry.vertex(index, frame)?;
                for (axis, value) in point.into_iter().enumerate() {
                    store32(leaf, 12 * slot + 4 * axis, value.to_bits());
                }
            }

            let fourth = if prim1 == 0x00ff_ffff {
                [f32::from_bits(0x7fc0_0000); 3]
            } else {
                let triangle = geometry.triangle(prim1)?;
                let index = if order & 0x20 != 0 {
                    triangle[0]
                } else if order & 0x80 != 0 {
                    triangle[1]
                } else {
                    triangle[2]
                };
                geometry.vertex(index, frame)?
            };
            for (axis, value) in fourth.into_iter().enumerate() {
                store32(leaf, 36 + 4 * axis, value.to_bits());
            }
            store32(leaf, 48, gi << 8 | flags);
            store32(leaf, 52, prim0);
            store32(leaf, 56, if prim1 == 0x00ff_ffff { INVALID } else { prim1 });
            store32(leaf, 60, (order & 0xf) | ((order << 4) & 0xf00));
        }
        remaining -= count;
        base += LEAVES_PER_CHUNK * 64;
    }
    Some(())
}

// A half NaN is refused: struct.unpack("<e") canonicalizes its payload on CPython <= 3.12 but keeps
// it on 3.14, so Python decides and matches whichever interpreter is running.
fn half_value(half: u16) -> Option<f32> {
    let sign = u32::from(half & 0x8000) << 16;
    let fraction = u32::from(half & 0x03ff);
    let bits = match (half >> 10) & 31 {
        // The integer multiplication by a power of two is exact for every half subnormal.
        0 => (fraction as f32 * (1.0 / 16_777_216.0)).to_bits(),
        31 if fraction == 0 => f32::INFINITY.to_bits(),
        31 => return None,
        exp => (u32::from(exp) + 112) << 23 | fraction << 13,
    };
    Some(f32::from_bits(sign | bits))
}

fn half_bound(x: f32, upper: bool) -> Option<u16> {
    let bits = x.to_bits();
    let sign = ((bits >> 16) as u16) & 0x8000;
    if x.is_nan() {
        return Some(sign | 0x7e00);
    }
    let away = if upper { x > 0.0 } else { x < 0.0 };
    let magnitude = bits & 0x7fff_ffff;
    const MIN_NORMAL: u32 = 0x3880_0000;
    const MAX_HALF: u32 = 0x477f_e000;
    const INF: u32 = 0x7f80_0000;
    // Let Python decide how to handle finite inward bounds beyond the half range.
    if !away && (MAX_HALF + 1..INF).contains(&magnitude) {
        return None;
    }
    let rounded = if away {
        ((magnitude + 0x1fff) & !0x1fff).max(MIN_NORMAL)
    } else {
        magnitude & !0x1fff
    };
    if rounded < MIN_NORMAL {
        Some(0)
    } else if rounded > MAX_HALF {
        Some(sign | 0x7c00)
    } else {
        Some(sign | ((rounded - (112 << 23)) >> 13) as u16)
    }
}

#[inline]
fn prefer_candidate(first: f32, candidate: f32, upper: bool) -> bool {
    first.is_nan()
        || (!candidate.is_nan()
            && if upper {
                candidate > first
            } else {
                candidate < first
            })
}

fn node_bounds(dst: &[u8], ptr: u32) -> Option<[f32; 6]> {
    let at = node_offset(ptr);
    let kind = ptr & 7;
    let (start, count, stride) = match kind {
        TRI0 => {
            let flags = u32_at(dst, at + 48)?;
            (at, if flags & 0x80 == 0 { 3 } else { 4 }, 12)
        }
        TRI1 => (at + 12, 3, 12),
        BOX16 => (at + 16, 4, 12),
        BOX32 => (at + 16, 4, 24),
        _ => return Some(ABSENT32),
    };
    let points = dst.get(start..start + stride * count)?;
    let mut bounds = [0.0; 6];
    for (component, bound) in bounds.iter_mut().enumerate() {
        let read_value = |point: &[u8]| match kind {
            BOX16 => u16_at(point, 2 * component).and_then(half_value),
            BOX32 => f32_at(point, 4 * component),
            _ => f32_at(point, 4 * (component % 3)),
        };
        let mut values = points.chunks_exact(stride);
        *bound = read_value(values.next()?)?;
        for point in values {
            let candidate = read_value(point)?;
            if prefer_candidate(*bound, candidate, component >= 3) {
                *bound = candidate;
            }
        }
    }
    // A child can point back into copied header bytes on malformed input.
    // Let Python handle any NaN payload found there that the reduction kept.
    if bounds
        .iter()
        .any(|x| x.is_nan() && !matches!(x.to_bits(), 0x7fc0_0000 | 0xffc0_0000))
    {
        return None;
    }
    Some(bounds)
}

fn write_box_groups(
    src: &[u8],
    dst: &mut [u8],
    header: &Header,
    start: usize,
    count: usize,
    wide: bool,
) -> Option<()> {
    let (size, kind) = if wide { (128, BOX32) } else { (64, BOX16) };
    for group in (0..count).step_by(GROUP) {
        let n = (count - group).min(GROUP);
        let base = header.nodes + start + group * size;
        let words = src.get(base..base + 4 * n)?;
        let mut child_at = base + 4 * n;
        let mut value_at = child_at
            + 4 * words
                .chunks_exact(4)
                .map(|w| usize::from(w[3]))
                .sum::<usize>();
        for i in 0..n {
            let mut mask = u32_at(words, 4 * i)? & 0x00ff_ffff;
            let kids = usize::from(words[4 * i + 3]);
            let mut children = [INVALID; 4];
            for child in children.get_mut(..kids)? {
                *child = u32_at(src, child_at)?;
                child_at += 4;
            }
            let mut bounds32 = [ABSENT32; 4];
            let mut bounds16 = [ABSENT16; 4];
            for (slot, &child) in children[..kids].iter().enumerate() {
                let bounds = node_bounds(dst, child)?;
                if wide {
                    bounds32[slot] = bounds;
                } else {
                    for (c, value) in bounds.into_iter().enumerate() {
                        bounds16[slot][c] = half_bound(value, c >= 3)?;
                    }
                }
            }
            while mask != 0 {
                let bit = mask.trailing_zeros() as usize;
                mask &= mask - 1;
                if wide {
                    let value = f32_at(src, value_at)?;
                    if value.is_nan() {
                        return None;
                    }
                    bounds32[bit / 6][bit % 6] = value;
                    value_at += 4;
                } else {
                    bounds16[bit / 6][bit % 6] = u16_at(src, value_at)?;
                    value_at += 2;
                }
            }
            let at = base + i * size;
            let node = dst.get_mut(at..at + size)?;
            for (i, child) in children.into_iter().enumerate() {
                store32(node, i * 4, child);
            }
            if wide {
                for (i, value) in bounds32.into_iter().flatten().enumerate() {
                    store32(node, 16 + i * 4, value.to_bits());
                }
                node[112..].fill(0);
            } else {
                for (bytes, value) in node[16..]
                    .chunks_exact_mut(2)
                    .zip(bounds16.into_iter().flatten())
                {
                    bytes.copy_from_slice(&value.to_le_bytes());
                }
            }
            let parent = u32::try_from((at >> 6) << 3).ok()? | kind;
            for child in children {
                if child & 6 == BOX16 || (child & 7 == TRI0 && header.flags & FLAG_UPDATABLE != 0) {
                    header.write_parent(dst, child, parent);
                }
            }
        }
    }
    Some(())
}

fn rehydrate(
    src: &[u8],
    dst: &mut [u8],
    kept: usize,
    geometries: &[GeometryView<'_>],
    frame: Option<&[f32; 15]>,
) -> Option<()> {
    let header = Header::read(src)?;
    dst.get_mut(..kept)?.copy_from_slice(src.get(..kept)?);
    store32(dst, 0, header.flags & !FLAG_DEHYDRATED);
    header.write_parent(dst, header.root, INVALID);
    write_leaves(src, dst, &header, geometries, frame)?;

    if matches!(header.root & 7, BOX16 | BOX32) {
        let end = src.len() - 4;
        let count = u32_at(src, end)? as usize;
        let start = end.checked_sub(8 * count)?;
        if start < 64 {
            return None;
        }
        let mut prev = 0;
        for at in (start..end).step_by(8) {
            let box32_end = u32_at(src, at)? as usize;
            let box16_end = u32_at(src, at + 4)? as usize;
            let wide = box32_end.checked_sub(prev)?;
            let narrow = box16_end.checked_sub(box32_end)?;
            if wide % 128 != 0 || narrow % 64 != 0 {
                return None;
            }
            write_box_groups(src, dst, &header, prev, wide / 128, true)?;
            write_box_groups(src, dst, &header, box32_end, narrow / 64, false)?;
            prev = box16_end;
        }
    }
    Some(())
}

/// Rehydrate into a zeroed buffer of the same length as the source. Returns 0 on success or
/// -1 when the caller should discard the buffer and retry in Python.
///
/// # Safety
/// The caller must supply valid, non-overlapping source and output buffers of `len` bytes, an
/// aligned array of `geometry_count` descriptors and (unless null) a frame of 15 aligned floats.
/// Each descriptor must keep its referenced buffers and optional aligned matrix live for the call.
/// Only the output buffer may be written during the call.
#[no_mangle]
pub unsafe extern "C" fn bvh_rehydrate_v1(
    src: *const u8,
    len: usize,
    kept: usize,
    geometries: *const Geometry,
    geometry_count: usize,
    frame: *const [f32; 15],
    dst: *mut u8,
) -> i32 {
    // A panic must never unwind across the C ABI; treat it as a failure so the caller falls back to Python.
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> i32 {
        // Offsets on disk are u32, with a few scaled offsets that can exceed u32.
        if usize::BITS != 64
            || src.is_null()
            || dst.is_null()
            || len < 64
            || len > isize::MAX as usize
        {
            return -1;
        }
        if geometries.is_null()
            || geometry_count > isize::MAX as usize / std::mem::size_of::<Geometry>()
        {
            return -1;
        }
        let src = unsafe { slice::from_raw_parts(src, len) };
        let dst = unsafe { slice::from_raw_parts_mut(dst, len) };
        let Some(views) = unsafe { slice::from_raw_parts(geometries, geometry_count) }
            .iter()
            .map(|geometry| {
                if geometry.vertices.is_null()
                    || geometry.vertices_len > isize::MAX as usize
                    || geometry.indices_len > isize::MAX as usize
                    || geometry.stride < 12
                {
                    return None;
                }
                let indices = if geometry.index_width == 0 {
                    &[]
                } else if geometry.indices.is_null() || !matches!(geometry.index_width, 2 | 4) {
                    return None;
                } else {
                    unsafe { slice::from_raw_parts(geometry.indices, geometry.indices_len) }
                };
                Some(GeometryView {
                    vertices: unsafe {
                        slice::from_raw_parts(geometry.vertices, geometry.vertices_len)
                    },
                    indices,
                    stride: geometry.stride,
                    index_width: geometry.index_width,
                    transform: unsafe { geometry.transform.as_ref() },
                })
            })
            .collect::<Option<Vec<_>>>()
        else {
            return -1;
        };
        if rehydrate(src, dst, kept, &views, unsafe { frame.as_ref() }).is_some() {
            0
        } else {
            -1
        }
    }))
    .unwrap_or(-1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn outward_half_rounding() {
        assert_eq!(half_bound(1.0 + 1.0 / 4096.0, false), Some(0x3c00));
        assert_eq!(half_bound(1.0 + 1.0 / 4096.0, true), Some(0x3c01));
        assert_eq!(half_bound(-f32::from_bits(1), false), Some(0x8400));
        assert_eq!(half_bound(-f32::from_bits(1), true), Some(0));
        assert_eq!(half_bound(-0.0, false), Some(0));
        assert_eq!(half_bound(65504.0, false), Some(0x7bff));
        assert_eq!(half_bound(65504.0, true), Some(0x7bff));
        assert_eq!(half_bound(65520.0, false), None);
        assert_eq!(half_bound(65520.0, true), Some(0x7c00));
        assert_eq!(half_bound(-65520.0, false), Some(0xfc00));
        assert_eq!(half_bound(-65520.0, true), None);
        assert_eq!(half_bound(f32::NEG_INFINITY, true), Some(0xfc00));
    }

    #[test]
    fn half_unpack_matches_python_special_values() {
        assert_eq!(half_value(0x8000).map(f32::to_bits), Some((-0.0_f32).to_bits()));
        assert_eq!(half_value(0x8001), Some(-2.0_f32.powi(-24)));
        assert_eq!(half_value(0x7bff), Some(65504.0));
        assert_eq!(half_value(0x7c00), Some(f32::INFINITY));
        assert_eq!(half_value(0xfc00), Some(f32::NEG_INFINITY));
        // Every half NaN payload is left to Python.
        for nan in [0x7c01, 0x7e00, 0x7e01, 0xfc01, 0xfe01, 0xffff] {
            assert_eq!(half_value(nan), None);
        }
    }

    #[test]
    fn transform_add_uses_the_double_product_before_rounding() {
        // 4097 * 4097 differs by one from its rounded float32 value.
        assert_eq!(
            dot([1.0, 4097.0, 0.0], [-16_785_408.0, 4097.0, 0.0]),
            Some(1.0)
        );
        assert_eq!(round32(f64::from(f32::MAX) * 2.0), None);
    }

    #[test]
    fn truncated_leaf_chunk_is_refused() {
        let mut src = [0; 128];
        store32(&mut src, 0, FLAG_DEHYDRATED);
        store32(&mut src, 12, 1);
        store32(&mut src, 52, 120);
        let mut dst = [0; 128];
        assert_eq!(rehydrate(&src, &mut dst, 64, &[], None), None);
    }
}
