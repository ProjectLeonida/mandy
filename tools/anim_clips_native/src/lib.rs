//! Optional sequence decoder for anim_clips.py. See docs/animation.md, "Keyframe decoding".
use std::{panic::catch_unwind, slice};

const TRACK_LIMIT: usize = 1 << 14;
const HEADER_SIZE: usize = 32;
// Bound the scratch columns and packed output of pathological inputs. Python can still decode them.
const MAX_SAMPLES: usize = 8 * 1024 * 1024;

fn read<const N: usize>(data: &[u8], at: usize) -> Option<[u8; N]> {
    data.get(at..)?.get(..N)?.try_into().ok()
}

fn u16_at(data: &[u8], at: usize) -> Option<usize> {
    Some(u16::from_le_bytes(read(data, at)?) as usize)
}

fn u32_at(data: &[u8], at: usize) -> Option<usize> {
    Some(u32::from_le_bytes(read(data, at)?) as usize)
}

fn f32_at(data: &[u8], at: usize) -> Option<f64> {
    let value = f32::from_le_bytes(read(data, at)?) as f64;
    // Leave NaN payload propagation and exceptional arithmetic to the Python reference.
    value.is_finite().then_some(value)
}

#[derive(Clone, Copy)]
struct BitField {
    byte: usize,
    shift: usize,
    mask: u64,
    crosses_word: bool,
}

impl BitField {
    fn new(pos: usize, width: usize) -> Self {
        let (byte, shift) = (pos / 8, pos % 8);
        let mask = if width == 0 {
            0
        } else {
            u64::MAX >> (64 - width)
        };
        Self {
            byte,
            shift,
            mask,
            crosses_word: width + shift > 64,
        }
    }

    fn extract(self, data: &[u8]) -> u64 {
        if self.mask == 0 {
            return 0;
        }
        let word = if let Some(word) = read::<8>(data, self.byte) {
            word
        } else {
            let mut word = [0; 8];
            if let Some(tail) = data.get(self.byte..) {
                word[..tail.len()].copy_from_slice(tail);
            }
            word
        };
        self.extract_word(data, word)
    }

    fn extract_word(self, data: &[u8], word: [u8; 8]) -> u64 {
        let mut value = u64::from_le_bytes(word) >> self.shift;
        if self.crosses_word {
            let high = data.get(self.byte + 8).copied().unwrap_or(0) as u64;
            value |= high << (64 - self.shift);
        }
        value & self.mask
    }
}

fn bits(data: &[u8], pos: usize, width: usize) -> u64 {
    BitField::new(pos, width).extract(data)
}

fn delta_float(x: i128, quantum: f64, offset: f64) -> f64 {
    let float = if let Ok(value) = i64::try_from(x) {
        value as f64
    } else {
        x as f64
    };
    float * quantum + offset
}

fn next_one(data: &[u8], pos: usize) -> Option<usize> {
    let shift = pos % 8;
    let (&first, rest) = data.get(pos / 8..)?.split_first()?;
    let first = first >> shift;
    if first != 0 {
        return Some(first.trailing_zeros() as usize);
    }
    let (skip, byte) = rest.iter().enumerate().find(|(_, byte)| **byte != 0)?;
    Some(8 - shift + 8 * skip + byte.trailing_zeros() as usize)
}

#[derive(Default)]
struct DeltaStats {
    boundaries: u32,
    closed: u32,
}

fn decode_delta2(defs: &[u8], nf: usize, stats: &mut DeltaStats) -> Option<(Vec<f64>, usize)> {
    let size = u32_at(defs, 0)?.checked_mul(4)?;
    let packed = u32_at(defs, 4)?;
    let quantum = f32_at(defs, 8)?;
    let offset = f32_at(defs, 12)?;
    let blob = defs.get(..size)?.get(16..)?;
    let (c1, c2, c3) = (packed & 255, (packed >> 8) & 255, (packed >> 16) & 255);
    if c2 > 64 {
        return None;
    }
    let chunks = nf.div_ceil(64);
    let groups = nf.saturating_sub(1).div_ceil(8);
    let lane_bytes = 4 * c3;
    let lane_at = blob.len().checked_sub(8 * lane_bytes)?;
    let (front, interleaved) = blob.split_at(lane_at);
    let mut pos = chunks * (c1 + c2);
    if pos.div_ceil(8) > front.len() {
        return None;
    }

    let mut lanes = vec![0; interleaved.len()];
    for (word, row) in interleaved.chunks_exact(32).enumerate() {
        for (lane, src) in row.chunks_exact(4).enumerate() {
            let dst = lane * lane_bytes + word * 4;
            lanes[dst..dst + 4].copy_from_slice(src);
        }
    }

    let start = |chunk| bits(front, chunks * c1 + chunk * c2, c2) as i128;
    let mut x = start(0);
    let mut velocity = 0;
    let mut lp = 0;
    let mut values = Vec::with_capacity(nf);
    if nf > 0 {
        values.push(delta_float(x, quantum, offset));
    }
    for group in 0..groups {
        let width = next_one(front, pos)?;
        pos += width + 1;
        if width > 64 || lp + width > 8 * lane_bytes {
            return None;
        }
        let field = BitField::new(lp, width);
        for lane in 0..(nf - 1 - 8 * group).min(8) {
            let at = lane * lane_bytes;
            let raw = field.extract(&lanes[at..at + lane_bytes]);
            let delta = ((raw >> 1) as i128) ^ -((raw & 1) as i128);
            // At most 64 signed 64-bit deltas are integrated at a time.
            velocity += delta;
            x += velocity;
            let frame = group * 8 + lane + 1;
            if frame % 64 == 0 {
                let next = start(frame / 64);
                stats.boundaries += 1;
                stats.closed += u32::from(x == next);
                x = next;
                velocity = 0;
            }
            values.push(delta_float(x, quantum, offset));
        }
        lp += width;
    }
    Some((values, size))
}

enum Samples {
    Whole([f64; 3]),
    Scalar(f64),
    Frames(Vec<f64>),
    Marker,
}

impl Samples {
    fn at(&self, frame: usize) -> f64 {
        match self {
            Self::Scalar(value) => *value,
            Self::Frames(values) => values[frame],
            _ => unreachable!("only component channels have samples"),
        }
    }
}

struct Channel {
    kind: usize,
    track: usize,
    comp: usize,
    samples: Samples,
}

struct Sequence {
    frames: usize,
    channels: Vec<Channel>,
    counts: [u16; 11],
    quantize_bits_match: bool,
    delta: DeltaStats,
}

struct FrameFields<'a> {
    frame_data: &'a [u8],
    frames: usize,
    stride: usize,
    pos: usize,
    quantize_bits: usize,
}

impl FrameFields<'_> {
    fn read(
        &mut self,
        width: usize,
        mut convert: impl FnMut(u64) -> Option<f64>,
    ) -> Option<Samples> {
        if width > 64 || self.pos.checked_add(width)? > self.stride * 8 {
            return None;
        }
        if width == 0 {
            return Some(Samples::Frames(if self.frames == 0 {
                Vec::new()
            } else {
                vec![convert(0)?; self.frames]
            }));
        }
        let mut values = Vec::with_capacity(self.frames);
        let field = BitField::new(self.pos, width);
        for frame in 0..self.frames {
            let row = &self.frame_data[frame * self.stride..];
            // The sequence counts follow the last frame, so there are at least eight readable bytes.
            let word = read::<8>(row, field.byte)?;
            values.push(convert(field.extract_word(row, word))?);
        }
        self.pos += width;
        Some(Samples::Frames(values))
    }
}

fn parse_sequence(input: &[u8], ntracks: usize) -> Option<Sequence> {
    let payload_len = u32_at(input, 4)?;
    let frame_offset = u32_at(input, 12)?;
    let nf = u16_at(input, 22)?;
    let stride = u16_at(input, 24)?;
    let quantize_bits = u16_at(input, 28)?;
    let payload = input.get(HEADER_SIZE..)?.get(..payload_len)?;
    let lists_at = frame_offset.checked_add(stride.checked_mul(nf)?)?;
    let defs = payload.get(..frame_offset)?;
    let lists = payload.get(lists_at..)?;
    let mut counts = [0u16; 11];
    for (i, count) in counts.iter_mut().enumerate() {
        *count = u16_at(lists, 2 * i)? as u16;
    }
    let dynamic = [3, 4, 5, 10]
        .map(|kind| counts[kind] as usize)
        .iter()
        .sum::<usize>();
    if counts[6] != 0 || counts[9] != 0 || nf.checked_mul(dynamic)? > MAX_SAMPLES {
        return None;
    }
    let mut fields = FrameFields {
        frame_data: &payload[frame_offset..],
        frames: nf,
        stride,
        pos: 0,
        quantize_bits: 0,
    };
    let mut def_at = 0;
    let mut list_at = 22;
    let mut delta = DeltaStats::default();
    let mut channels = Vec::with_capacity(counts.iter().map(|&count| count as usize).sum());
    for (kind, &count) in counts.iter().enumerate() {
        let selectors = lists.get(list_at..)?.get(..2 * count as usize)?;
        for i in 0..count as usize {
            let selector = u16_at(selectors, 2 * i)?;
            if selector >> 2 >= ntracks {
                return None;
            }
            let def = defs.get(def_at..)?;
            let samples = match kind {
                0 | 1 => {
                    let value = [f32_at(def, 0)?, f32_at(def, 4)?, f32_at(def, 8)?];
                    def_at += 12;
                    Samples::Whole(value)
                }
                2 => {
                    let value = f32_at(def, 0)?;
                    def_at += 4;
                    Samples::Scalar(value)
                }
                3 => fields.read(32, |raw| {
                    let value = f32::from_bits(raw as u32) as f64;
                    value.is_finite().then_some(value)
                })?,
                4 => {
                    let width = u32_at(def, 0)?;
                    let q = f32_at(def, 4)?;
                    let o = f32_at(def, 8)?;
                    def_at += 12;
                    let samples = fields.read(width, |raw| Some(raw as f64 * q + o))?;
                    fields.quantize_bits += width;
                    samples
                }
                5 => {
                    let width = u32_at(def, 0)?;
                    let value_bits = u32_at(def, 4)?;
                    let words = u32_at(def, 8)?;
                    let q = f32_at(def, 12)?;
                    let o = f32_at(def, 16)?;
                    let table_size = words.checked_mul(4)?;
                    let table = def.get(20..)?.get(..table_size)?;
                    if value_bits > 64 {
                        return None;
                    }
                    def_at += 20 + table_size;
                    let entries = if value_bits == 0 {
                        1
                    } else {
                        words.checked_mul(32)? / value_bits
                    };
                    fields.read(width, |raw| {
                        let index = usize::try_from(raw).ok()?;
                        (index < entries)
                            .then(|| bits(table, index * value_bits, value_bits) as f64 * q + o)
                    })?
                }
                7 | 8 => Samples::Marker,
                10 => {
                    let (values, size) = decode_delta2(def, nf, &mut delta)?;
                    def_at += size;
                    Samples::Frames(values)
                }
                _ => return None,
            };
            channels.push(Channel {
                kind,
                track: selector >> 2,
                comp: selector & 3,
                samples,
            });
        }
        list_at += 8 * (count as usize).div_ceil(4);
    }
    if def_at != frame_offset {
        return None;
    }
    Some(Sequence {
        frames: nf,
        channels,
        counts,
        quantize_bits_match: fields.quantize_bits == quantize_bits,
        delta,
    })
}

#[derive(Default)]
struct Track<'a> {
    id: usize,
    whole: Option<(usize, [f64; 3])>,
    components: [Option<&'a Samples>; 4],
    rebuild: Option<usize>,
    normalize: bool,
}

#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq)]
enum TrackKind {
    Components,
    StaticQuaternion,
    StaticVector,
    Rebuilt,
    Normalized,
    FourStored,
}

impl Track<'_> {
    fn layout(&self) -> Option<(TrackKind, usize)> {
        if let Some((kind, _)) = self.whole {
            return Some(if kind == 0 {
                (TrackKind::StaticQuaternion, 4)
            } else {
                (TrackKind::StaticVector, 3)
            });
        }
        let width = self.components.iter().rposition(Option::is_some)? + 1;
        let filled = self.components.iter().filter(|c| c.is_some()).count();
        if self.rebuild.is_some() && (width != 3 || filled != 3) {
            return None;
        }
        if self.normalize && (width != 4 || filled != 4) {
            return None;
        }
        Some(if self.rebuild.is_some() {
            (TrackKind::Rebuilt, 4)
        } else if self.normalize {
            (TrackKind::Normalized, 4)
        } else if width == 4 {
            (TrackKind::FourStored, 4)
        } else {
            (TrackKind::Components, width)
        })
    }
}

fn sum_squares<const COMPENSATED: bool>(values: &[f64]) -> f64 {
    let mut sum = 0.0;
    let mut correction = 0.0;
    for &a in values {
        let x = a * a;
        let total = sum + x;
        // CPython 3.12+ sum uses Neumaier compensation for floats.
        if COMPENSATED {
            correction += if sum >= x {
                (sum - total) + x
            } else {
                (x - total) + sum
            };
        }
        sum = total;
    }
    sum + correction
}

fn put_values(out: &mut [u8], values: &[f64]) {
    for (dest, &value) in out.chunks_exact_mut(8).zip(values) {
        dest.copy_from_slice(&value.to_le_bytes());
    }
}

fn stored_row<const N: usize>(track: &Track<'_>, frame: usize) -> [f64; N] {
    std::array::from_fn(|comp| track.components[comp].map_or(0.0, |source| source.at(frame)))
}

fn normalize(row: &mut [f64; 4], length: f64) {
    let norm = if length == 0.0 { 1.0 } else { length };
    for value in row {
        *value /= norm;
    }
}

fn pack_components<const N: usize>(out: &mut [u8], track: &Track<'_>) {
    for (frame, output) in out.chunks_exact_mut(N * 8).enumerate() {
        put_values(output, &stored_row::<N>(track, frame));
    }
}

fn pack_four<const COMPENSATED: bool, const NORMALIZE: bool>(
    out: &mut [u8],
    track: &Track<'_>,
) -> f64 {
    let mut worst = 0.0_f64;
    for (frame, output) in out.chunks_exact_mut(32).enumerate() {
        let mut row = stored_row::<4>(track, frame);
        let length = sum_squares::<COMPENSATED>(&row).sqrt();
        worst = worst.max((length - 1.0).abs());
        if NORMALIZE {
            normalize(&mut row, length);
        }
        put_values(output, &row);
    }
    worst
}

fn pack_rebuilt<const COMPENSATED: bool>(out: &mut [u8], track: &Track<'_>, comp: usize) -> f64 {
    let mut worst = 0.0_f64;
    for (frame, output) in out.chunks_exact_mut(32).enumerate() {
        let stored = stored_row::<3>(track, frame);
        let remainder = 1.0 - sum_squares::<COMPENSATED>(&stored);
        worst = worst.max(-remainder);
        let missing = remainder.max(0.0).sqrt();
        let [x, y, z] = stored;
        let mut row = match comp {
            0 => [missing, x, y, z],
            1 => [x, missing, y, z],
            2 => [x, y, missing, z],
            _ => [x, y, z, missing],
        };
        let length = sum_squares::<COMPENSATED>(&row).sqrt();
        normalize(&mut row, length);
        put_values(output, &row);
    }
    worst
}

fn pack_track<const COMPENSATED: bool>(
    out: &mut Vec<u8>,
    track: &Track<'_>,
    kind: TrackKind,
    width: usize,
    nf: usize,
) {
    out.extend_from_slice(&(track.id as u16).to_le_bytes());
    out.extend_from_slice(&[kind as u8, width as u8]);
    let stats_at = out.len();
    let rows_at = stats_at + 8;
    let nrows = if track.whole.is_some() { 1 } else { nf };
    out.resize(rows_at + nrows * width * 8, 0);
    let rows = &mut out[rows_at..];
    let worst = if let Some((_, [x, y, z])) = track.whole {
        if kind == TrackKind::StaticQuaternion {
            let r = 1.0 - x * x - y * y - z * z;
            put_values(rows, &[x, y, z, r.max(0.0).sqrt()]);
            f64::from(r < -1e-3)
        } else {
            put_values(rows, &[x, y, z]);
            0.0
        }
    } else if let Some(comp) = track.rebuild {
        pack_rebuilt::<COMPENSATED>(rows, track, comp)
    } else if width == 4 {
        if track.normalize {
            pack_four::<COMPENSATED, true>(rows, track)
        } else {
            pack_four::<COMPENSATED, false>(rows, track)
        }
    } else {
        match width {
            1 => pack_components::<1>(rows, track),
            2 => pack_components::<2>(rows, track),
            3 => pack_components::<3>(rows, track),
            _ => unreachable!("track layout requires at least one component"),
        }
        0.0
    };
    out[stats_at..rows_at].copy_from_slice(&worst.to_le_bytes());
}

fn pack_sequence<const COMPENSATED: bool>(sequence: &Sequence, ntracks: usize) -> Option<Vec<u8>> {
    let mut tracks = Vec::<Track>::new();
    let mut plan_index = vec![None; ntracks.min(TRACK_LIMIT)];
    for channel in &sequence.channels {
        let i = *plan_index[channel.track].get_or_insert_with(|| {
            let index = tracks.len();
            tracks.push(Track {
                id: channel.track,
                ..Track::default()
            });
            index
        });
        let track = &mut tracks[i];
        match channel.samples {
            Samples::Whole(values) => {
                track.whole.get_or_insert((channel.kind, values));
            }
            Samples::Marker if channel.kind == 7 => track.rebuild = Some(channel.comp),
            Samples::Marker => track.normalize = true,
            _ => track.components[channel.comp] = Some(&channel.samples),
        }
    }
    let layouts = tracks
        .iter()
        .map(Track::layout)
        .collect::<Option<Vec<_>>>()?;
    let samples: usize = tracks
        .iter()
        .zip(&layouts)
        .map(|(track, &(_, width))| {
            width
                * if track.whole.is_some() {
                    1
                } else {
                    sequence.frames
                }
        })
        .sum();
    if samples > MAX_SAMPLES {
        return None;
    }

    // Little-endian packet: header "<HH?11H2I", then "<HBBd" and packed f64 rows for each track.
    let mut out = Vec::new();
    out.try_reserve_exact(35 + 12 * tracks.len() + 8 * samples)
        .ok()?;
    out.extend_from_slice(&(sequence.frames as u16).to_le_bytes());
    out.extend_from_slice(&(tracks.len() as u16).to_le_bytes());
    out.push(u8::from(sequence.quantize_bits_match));
    for count in sequence.counts {
        out.extend_from_slice(&count.to_le_bytes());
    }
    out.extend_from_slice(&sequence.delta.boundaries.to_le_bytes());
    out.extend_from_slice(&sequence.delta.closed.to_le_bytes());
    for (track, (kind, width)) in tracks.iter().zip(layouts) {
        pack_track::<COMPENSATED>(&mut out, track, kind, width, sequence.frames);
    }
    Some(out)
}

/// # Safety
/// Input must be readable for len bytes. out_len must point to a writable usize.
/// A non-null result must be released with anim_free_sequence_v1 and the returned length.
#[no_mangle]
pub unsafe extern "C" fn anim_decode_sequence_v1(
    input: *const u8,
    len: usize,
    at: usize,
    ntracks: usize,
    compensated: u8,
    out_len: *mut usize,
) -> *mut u8 {
    if input.is_null() || out_len.is_null() || len > isize::MAX as usize {
        return std::ptr::null_mut();
    }
    let result = catch_unwind(|| {
        let input = unsafe { slice::from_raw_parts(input, len) }.get(at..)?;
        let sequence = parse_sequence(input, ntracks)?;
        if compensated != 0 {
            pack_sequence::<true>(&sequence, ntracks)
        } else {
            pack_sequence::<false>(&sequence, ntracks)
        }
    });
    if let Ok(Some(out)) = result {
        unsafe { *out_len = out.len() };
        Box::into_raw(out.into_boxed_slice()).cast::<u8>()
    } else {
        std::ptr::null_mut()
    }
}

/// # Safety
/// ptr and len must come from a successful call to anim_decode_sequence_v1 and be freed once.
#[no_mangle]
pub unsafe extern "C" fn anim_free_sequence_v1(ptr: *mut u8, len: usize) {
    if !ptr.is_null() {
        let bytes = std::ptr::slice_from_raw_parts_mut(ptr, len);
        drop(unsafe { Box::from_raw(bytes) });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_unaligned_and_zero_width_fields() {
        let input = [0x80, 0x34, 0x12, 0, 0, 0, 0, 0, 0x02];
        assert_eq!(bits(&input, 7, 17), 0x2469);
        assert_eq!(bits(&input, 7, 64), 0x0400_0000_0000_2469);
        assert_eq!(bits(&input, 100, 0), 0);
        assert_eq!(next_one(&[0, 0, 8], 5), Some(14));
        assert_eq!(next_one(&[0, 0, 8], 20), None);
    }

    #[test]
    fn matches_both_python_sum_algorithms() {
        let values = [1.0, 2_f64.powi(-27), 2_f64.powi(-27), 2_f64.powi(-27)];
        assert_eq!(sum_squares::<false>(&values).to_bits(), 1.0_f64.to_bits());
        assert_eq!(
            sum_squares::<true>(&values).to_bits(),
            1.0_f64.to_bits() + 1
        );
    }

    #[test]
    fn refuses_incomplete_sequence() {
        let mut input = [0; 54];
        assert!(parse_sequence(&input, 1).is_none());
        input[4] = 22;
        assert!(parse_sequence(&input, 1).is_some());
        input[32 + 2 * 6] = 1;
        assert!(parse_sequence(&input, 1).is_none());
    }
}
