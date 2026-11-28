//! Optional kernels for meshlet_page.py. Exported calls return -1 on refusal;
//! Python then runs its reference path, including its original error reporting.

use std::collections::HashMap;
use std::slice;

type Vec3 = [f64; 3];

const FACES: u32 = 1;
const CENTRES: u32 = 2;
const NORMALS: u32 = 4;
const CONES: u32 = 8;
const RESULT_LEN: usize = 21;

fn u16_at(src: &[u8], at: usize) -> u16 {
    u16::from_le_bytes(src[at..at + 2].try_into().unwrap())
}

fn u32_at(src: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(src[at..at + 4].try_into().unwrap())
}

fn region(src: &[u8], at: usize, count: usize, stride: usize) -> Option<&[u8]> {
    src.get(at..at.checked_add(count.checked_mul(stride)?)?)
}

// CPython 3.12 and later compensate float sum(). Only calls that use Python
// sum() take this path; explicit additions in the reference stay left-to-right.
fn sum3(terms: Vec3, compensated: bool) -> f64 {
    if !compensated {
        return ((0.0 + terms[0]) + terms[1]) + terms[2];
    }
    let mut hi: f64 = 0.0;
    let mut lo = 0.0;
    for x in terms {
        let next = hi + x;
        lo += if hi.abs() >= x.abs() {
            (hi - next) + x
        } else {
            (x - next) + hi
        };
        hi = next;
    }
    if lo != 0.0 && lo.is_finite() {
        hi + lo
    } else {
        hi
    }
}

fn scale(v: Vec3, length: f64) -> Vec3 {
    [v[0] / length, v[1] / length, v[2] / length]
}

fn length(v: Vec3) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn cross([p, q, r]: [Vec3; 3]) -> Vec3 {
    let u = [q[0] - p[0], q[1] - p[1], q[2] - p[2]];
    let v = [r[0] - p[0], r[1] - p[1], r[2] - p[2]];
    [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ]
}

fn dot(a: Vec3, b: Vec3, compensated: bool) -> f64 {
    sum3([a[0] * b[0], a[1] * b[1], a[2] * b[2]], compensated)
}

fn octahedral(u: f64, v: f64) -> Vec3 {
    let z = 1.0 - u.abs() - v.abs();
    let n = if z < 0.0 {
        [(1.0 - v.abs()).copysign(u), (1.0 - u.abs()).copysign(v), z]
    } else {
        [u, v, z]
    };
    scale(n, length(n))
}

fn unorm10(x: u32) -> f64 {
    f64::from(x & 1023) / 511.5 - 1.0
}

fn decode_normal(word: u32, mode: u32, compensated: bool) -> Vec3 {
    if mode == 1 {
        return octahedral(unorm10(word), unorm10(word >> 10));
    }
    let n = [unorm10(word), unorm10(word >> 10), unorm10(word >> 20)];
    let norm = sum3(n.map(|c| c * c), compensated).sqrt();
    scale(n, if norm == 0.0 { 1.0 } else { norm })
}

struct PositionTransform {
    bias: Vec3,
    scale: Vec3,
    axis: u32,
}

impl PositionTransform {
    fn read(src: &[u8], at: usize) -> Option<Self> {
        let fields = region(src, at.checked_add(0x60)?, 7, 4)?;
        let f = |i| f64::from(f32::from_bits(u32_at(fields, i)));
        let bias = [f(0), f(4), f(8)];
        let axis = u32_at(fields, 12);
        let scale = [f(16), f(20), f(24)];
        // Pass NaN payload propagation and exceptional geometry to Python.
        if axis > 2 || !bias.into_iter().chain(scale).all(f64::is_finite) {
            return None;
        }
        Some(Self { bias, scale, axis })
    }

    fn decode(&self, word: &[u8]) -> Vec3 {
        let low = u32_at(word, 0);
        let a = self.bias[0] + self.scale[0] * f64::from(low >> 8);
        let b = self.bias[1]
            + self.scale[1] * f64::from((u32::from(u16_at(word, 4)) << 8) | (low & 255));
        let c = self.bias[2] + self.scale[2] * f64::from(u16_at(word, 6));
        match self.axis {
            0 => [c, b, a],
            1 => [b, c, a],
            _ => [a, b, c],
        }
    }
}

#[derive(Clone, Copy)]
struct Meshlet {
    vertex_base: usize,
    triangle_base: usize,
    vertices: usize,
    triangles: usize,
    radius: u16,
    centre_xy: u32,
    centre_z_cone: u32,
}

struct Group<'a> {
    records: &'a [u8],
    triangles: &'a [u8],
    refs: &'a [u8],
}

impl<'a> Group<'a> {
    fn read(src: &'a [u8], at: usize) -> Option<Self> {
        let header = region(src, at, 3, 4)?;
        let count = (u32_at(header, 0) ^ 0x8000_0000) as usize;
        let triangles = u32_at(header, 4) as usize;
        let refs = u32_at(header, 8) as usize;
        // Keep wrapped running sums, alternate layouts, and empty groups on the
        // reference path. The triangle word has 11 bits for the meshlet index.
        if !(1..=2048).contains(&count) || triangles > 65536 || refs > 65536 {
            return None;
        }
        let records = region(src, at + 12, count, 16)?;
        let tri_at = at + 12 + records.len();
        let triangles = region(src, tri_at, triangles, 4)?;
        let refs = region(src, tri_at + triangles.len(), refs, 2)?;
        Some(Self {
            records,
            triangles,
            refs,
        })
    }

    fn meshlets(&self) -> impl Iterator<Item = Option<Meshlet>> + '_ {
        self.records.chunks_exact(16).map(|rec| {
            let counts = u32_at(rec, 0);
            if counts & 0xc000 != 0 {
                return None; // Alternate 17-bit base layout.
            }
            Some(Meshlet {
                vertex_base: u16_at(rec, 4) as usize,
                triangle_base: u16_at(rec, 6) as usize,
                vertices: (counts as usize & 127) + 1,
                triangles: ((counts as usize >> 7) & 127) + 1,
                radius: (counts >> 16) as u16,
                centre_xy: u32_at(rec, 8),
                centre_z_cone: u32_at(rec, 12),
            })
        })
    }

    fn validate(&self) -> Option<[u32; 3]> {
        let (mut vbase, mut tbase, mut full) = (0, 0, 0);
        for (m, record) in self.meshlets().enumerate() {
            let meshlet = record?;
            if vbase != meshlet.vertex_base || tbase != meshlet.triangle_base {
                return None;
            }
            let words = region(self.triangles, 4 * tbase, meshlet.triangles, 4)?;
            for word in words.chunks_exact(4) {
                let t = u32_at(word, 0);
                if (t >> 21) as usize != m
                    || triangle_indices(t)
                        .into_iter()
                        .any(|i| i >= meshlet.vertices)
                {
                    return None;
                }
            }
            vbase += meshlet.vertices;
            tbase += meshlet.triangles;
            full += u32::from(meshlet.vertices == 128);
        }
        if vbase * 2 != self.refs.len() || tbase * 4 != self.triangles.len() {
            return None;
        }
        let mut seen = [0u64; 1024];
        let mut span = 0;
        for word in self.refs.chunks_exact(2) {
            let i = u16_at(word, 0) as usize;
            seen[i / 64] |= 1 << (i % 64);
            span = span.max(i + 1);
        }
        Some([
            full,
            span as u32,
            seen.iter().map(|word| word.count_ones()).sum(),
        ])
    }
}

fn triangle_indices(t: u32) -> [usize; 3] {
    [
        (t & 127) as usize,
        ((t >> 7) & 127) as usize,
        ((t >> 14) & 127) as usize,
    ]
}

struct Edge {
    first: u32,
    second: u32,
}

impl Edge {
    const ONE: u32 = u32::MAX;
    const MANY: u32 = u32::MAX - 1;
}

fn record_edges(edges: &mut HashMap<u32, Edge>, [a, b, c]: [usize; 3], face: u32) {
    use std::collections::hash_map::Entry;

    for (a, b) in [(a, b), (b, c), (c, a)] {
        let key = ((a.min(b) as u32) << 16) | a.max(b) as u32;
        match edges.entry(key) {
            Entry::Vacant(slot) => {
                slot.insert(Edge {
                    first: face,
                    second: Edge::ONE,
                });
            }
            Entry::Occupied(mut slot) => {
                let edge = slot.get_mut();
                edge.second = if edge.second == Edge::ONE {
                    face
                } else {
                    Edge::MANY
                };
            }
        }
    }
}

fn face_counts(edges: HashMap<u32, Edge>, normals: &[Option<Vec3>], compensated: bool) -> [u32; 3] {
    let mut result = [0; 3];
    for edge in edges.into_values().filter(|edge| edge.second < Edge::MANY) {
        if let (Some(a), Some(b)) = (normals[edge.first as usize], normals[edge.second as usize]) {
            let d = dot(a, b, compensated);
            result[0] += 1;
            result[1] += u32::from(d > 0.9);
            result[2] += u32::from(d < -0.9);
        }
    }
    result
}

fn normal_counts(
    stored: &[u8],
    ids: &[usize],
    mode: u32,
    accum: &[Vec3],
    compensated: bool,
) -> [u32; 4] {
    let mut result = [0; 4];
    for (s, &id) in accum.iter().zip(ids) {
        let word = &stored[4 * id..4 * id + 4];
        let norm = length(*s);
        if norm == 0.0 || !norm.is_finite() {
            continue;
        }
        let n = decode_normal(u32_at(word, 0), mode, compensated);
        let d = sum3(
            [n[0] * s[0] / norm, n[1] * s[1] / norm, n[2] * s[2] / norm],
            compensated,
        );
        result[0] += 1;
        result[1] += u32::from(d > 0.99);
        result[2] += u32::from(d > 0.9);
        result[3] += u32::from(d < 0.0);
    }
    result
}

struct Cone {
    axis: Vec3,
    cos_half: f64,
    total: Vec3,
    smallest: f64,
}

impl Cone {
    fn new(word: u32) -> Option<Self> {
        if word >> 12 == 0 {
            return None;
        }
        let field = |shift: u32| f64::from((word >> shift) & 63u32) / 31.5 - 1.0;
        let sin = f64::from(31 - ((word >> 26) & 31)) / 31.0;
        Some(Self {
            axis: octahedral(field(14), field(20)),
            cos_half: (1.0 - sin * sin).max(0.0).sqrt(),
            total: [0.0; 3],
            smallest: f64::INFINITY,
        })
    }

    fn add(&mut self, raw: Vec3, unit: Vec3, compensated: bool) {
        for (sum, term) in self.total.iter_mut().zip(raw) {
            *sum += term;
        }
        self.smallest = self.smallest.min(dot(self.axis, unit, compensated));
    }

    fn count(self, out: &mut [u32], compensated: bool) {
        let norm = sum3(self.total.map(|v| v * v), compensated).sqrt();
        if self.smallest == f64::INFINITY || norm == 0.0 {
            return;
        }
        let [a, b, c] = self.axis;
        let [x, y, z] = self.total;
        out[1] += 1;
        out[2] += u32::from(sum3([a * x / norm, b * y / norm, c * z / norm], compensated) > 0.9);
        out[3] += u32::from(self.smallest >= self.cos_half);
        out[4] += u32::from(self.smallest >= self.cos_half - 0.05);
    }
}

fn finite_half(bits: u16) -> Option<f64> {
    let exp = i32::from((bits >> 10) & 31);
    let fraction = f64::from(bits & 1023);
    let sign = if bits >> 15 == 0 { 1.0 } else { -1.0 };
    match exp {
        31 => None,
        0 => Some(sign * fraction * (1.0 / 16_777_216.0)),
        _ => Some(sign * (1024.0 + fraction) * f64::from_bits(((exp + 998) as u64) << 52)),
    }
}

fn centre_counts(meshlet: Meshlet, points: &[Vec3], out: &mut [u32]) -> Option<()> {
    let w2 = meshlet.centre_xy;
    let w3 = meshlet.centre_z_cone;
    let exp = w2 >> 28;
    let step = f64::from(1 << exp) / 16.0;
    let stored = [w2 & 4095, (w2 >> 12) & 4095, w3 & 4095].map(|v| (f64::from(v) - 2047.0) * step);
    out[1] += u32::from(w2 == 0x007f_f7ff && w3 == 0x7ff);
    out[2] += u32::from(exp != 0);
    let mut lo = points[0];
    let mut hi = points[0];
    let mut far: f64 = 0.0;
    for p in points {
        for i in 0..3 {
            lo[i] = lo[i].min(p[i]);
            hi[i] = hi[i].max(p[i]);
        }
        let [x, y, z] = [p[0] - stored[0], p[1] - stored[1], p[2] - stored[2]];
        far = far.max(x * x + y * y + z * z);
    }
    for i in 0..3 {
        let diff = (((lo[i] + hi[i]) / 2.0 - stored[i]) / step).abs();
        // Python round() ties to even: 0.5 rounds to zero, 1.5 rounds to two.
        out[4] += u32::from(diff <= 0.5);
        out[5] += u32::from(diff < 1.5);
    }
    let radius = finite_half(meshlet.radius)?;
    let far = far.sqrt();
    let slack = 3.0f64.sqrt() / 2.0 * step;
    let cover = far * (1.0 - 1.0 / 2048.0);
    let tight = 1.25 * far + slack;
    // The f32 header and integer fields bound all nonzero squared deltas to
    // normal f64s. The three squares, two sums, sqrt, and following arithmetic
    // round within a few ulps of the compensated math.dist comparisons.
    // Give Python the decision at any boundary within a conservative margin.
    let close = |a: f64, b: f64| (a - b).abs() <= 64.0 * f64::EPSILON * a.abs().max(b.abs());
    if far != 0.0 && (close(radius, cover) || close(radius + slack, far) || close(radius, tight)) {
        return None;
    }
    out[6] += u32::from(radius >= cover);
    out[7] += u32::from(radius + slack >= far);
    out[8] += u32::from(radius <= tight);
    Some(())
}

#[repr(C)]
struct Block {
    header: usize,
    vertices: usize,
    position_at: usize,
    normal_at: usize,
    normal_mode: u32,
}

fn allocated<T>(count: usize) -> Option<Vec<T>> {
    let mut out = Vec::new();
    out.try_reserve_exact(count).ok()?;
    Some(out)
}

fn geometry(
    src: &[u8],
    at: usize,
    block: &Block,
    flags: u32,
    compensated: bool,
) -> Option<[u32; RESULT_LEN]> {
    let group = Group::read(src, at)?;
    if !(1..=65536).contains(&block.vertices) {
        return None;
    }
    let positions = region(src, block.position_at, block.vertices, 8)?;
    let transform = PositionTransform::read(src, block.header)?;
    let stored_normals = if flags & NORMALS != 0 && matches!(block.normal_mode, 1 | 7) {
        region(src, block.normal_at, block.vertices, 4)?
    } else {
        &[]
    };
    // Decode only the vertices this group's meshlets reference: a block shared by many
    // sparse groups must not cost O(block vertices) per group.
    let mut unique = allocated(group.refs.len() / 2)?;
    for meshlet in group.meshlets() {
        let meshlet = meshlet?;
        let refs = region(group.refs, meshlet.vertex_base * 2, meshlet.vertices, 2)?;
        for word in refs.chunks_exact(2) {
            let index = u16_at(word, 0) as usize;
            if index >= block.vertices {
                return None;
            }
            unique.push(index);
        }
    }
    unique.sort_unstable();
    unique.dedup();
    let mut points = allocated(unique.len())?;
    points.extend(
        unique
            .iter()
            .map(|&i| transform.decode(&positions[8 * i..8 * i + 8])),
    );

    let face_check = flags & FACES != 0;
    let triangle_count = group.triangles.len() / 4;
    let mut edges = HashMap::new();
    let mut face_normals = allocated(if face_check { triangle_count } else { 0 })?;
    if face_check {
        edges.try_reserve(3 * triangle_count).ok()?;
    }
    let accum_len = if stored_normals.is_empty() { 0 } else { unique.len() };
    let mut accum = allocated(accum_len)?;
    accum.resize(accum_len, [0.0; 3]);
    let mut out = [0u32; RESULT_LEN];
    out[3] = (group.records.len() / 16) as u32;
    out[6] = 3 * out[3];
    out[12] = out[3];
    for meshlet in group.meshlets() {
        let meshlet = meshlet?;
        let refs = region(group.refs, meshlet.vertex_base * 2, meshlet.vertices, 2)?;
        let mut vertex = [0; 128];
        for (i, word) in refs.chunks_exact(2).enumerate() {
            vertex[i] = unique.binary_search(&(u16_at(word, 0) as usize)).ok()?;
        }
        if flags & CENTRES != 0 {
            let mut local = [[0.0; 3]; 128];
            for (dst, index) in local.iter_mut().zip(&vertex[..meshlet.vertices]) {
                *dst = points[*index];
            }
            centre_counts(meshlet, &local[..meshlet.vertices], &mut out[3..12])?;
        }

        let mut cone = if flags & CONES != 0 {
            Cone::new(meshlet.centre_z_cone)
        } else {
            None
        };
        if !face_check && accum.is_empty() && cone.is_none() {
            continue;
        }
        let words = region(
            group.triangles,
            meshlet.triangle_base * 4,
            meshlet.triangles,
            4,
        )?;
        for word in words.chunks_exact(4) {
            let local = triangle_indices(u32_at(word, 0));
            if local.into_iter().any(|i| i >= meshlet.vertices) {
                return None;
            }
            let indices = local.map(|i| vertex[i]);
            let raw = cross(indices.map(|i| points[i]));
            if !accum.is_empty() {
                for i in indices {
                    for (sum, term) in accum[i].iter_mut().zip(raw) {
                        *sum += term;
                    }
                }
            }
            if face_check || cone.is_some() {
                let norm = length(raw);
                let unit = (norm != 0.0 && norm.is_finite()).then(|| scale(raw, norm));
                if face_check {
                    record_edges(&mut edges, indices, face_normals.len() as u32);
                    face_normals.push(unit);
                }
                if let (Some(cone), Some(unit)) = (&mut cone, unit) {
                    cone.add(raw, unit, compensated);
                }
            }
        }
        if let Some(cone) = cone {
            cone.count(&mut out[12..17], compensated);
        }
    }
    if face_check {
        out[..3].copy_from_slice(&face_counts(edges, &face_normals, compensated));
    }
    if !accum.is_empty() {
        out[17..].copy_from_slice(&normal_counts(
            stored_normals,
            &unique,
            block.normal_mode,
            &accum,
            compensated,
        ));
    }
    Some(out)
}

fn block_counts(
    src: &[u8],
    block: &Block,
    tangents: &[usize],
    compensated: bool,
) -> Option<[usize; 7]> {
    let transform = PositionTransform::read(src, block.header)?;
    let positions = region(src, block.position_at, block.vertices, 8)?;
    let mut out = [0; 7];
    out[0] = block.vertices;
    out[1] = block.vertices; // A finite f32 times at most 24 bits stays finite as f64.
    let largest = [0xff_ffff, 0xff_ffff, 0xffff];
    let inside = (0..3).all(|i| {
        let bias = transform.bias[i];
        let end = bias + transform.scale[i] * f64::from(largest[i]);
        bias.abs() <= 300.0 && end.abs() <= 300.0
    });
    out[2] = if inside {
        block.vertices
    } else {
        positions
            .chunks_exact(8)
            .filter(|word| transform.decode(word).into_iter().all(|v| v.abs() <= 300.0))
            .count()
    };

    if !tangents.is_empty() {
        if !matches!(block.normal_mode, 1 | 7) {
            return None;
        }
        let normals = region(src, block.normal_at, block.vertices, 4)?;
        let mut streams = [&[][..]; 23];
        for (stream, at) in streams.iter_mut().zip(tangents) {
            *stream = region(src, *at, block.vertices, 4)?;
        }
        out[3] = block.vertices.checked_mul(tangents.len())?;
        for (i, word) in normals.chunks_exact(4).enumerate() {
            let [x, y, z] = decode_normal(u32_at(word, 0), block.normal_mode, compensated);
            for stream in &streams[..tangents.len()] {
                let word = u32_at(stream, 4 * i);
                let [a, b, c] = decode_normal(word, 7, compensated);
                out[4] += usize::from((x * a + y * b + z * c).abs() < 0.05);
                out[5] += usize::from(word >> 30 == 3);
                out[6] += usize::from(word >> 30 == 0);
            }
        }
    }
    Some(out)
}

fn decode_rows(
    src: &[u8],
    header: usize,
    start: usize,
    mode: u32,
    compensated: bool,
    output: &mut [f64],
) -> Option<()> {
    let width = if mode == 8 { 4 } else { 3 };
    let words = region(
        src,
        start,
        output.len() / width,
        if mode == 0 { 8 } else { 4 },
    )?;
    if mode == 0 {
        let transform = PositionTransform::read(src, header)?;
        for (word, row) in words.chunks_exact(8).zip(output.chunks_exact_mut(width)) {
            row.copy_from_slice(&transform.decode(word));
        }
    } else {
        for (word, row) in words.chunks_exact(4).zip(output.chunks_exact_mut(width)) {
            let word = u32_at(word, 0);
            row[..3].copy_from_slice(&decode_normal(word, mode, compensated));
            if mode == 8 {
                row[3] = [-1.0, 0.0, 0.0, 1.0][(word >> 30) as usize];
            }
        }
    }
    Some(())
}

// The Python binding owns the input bytes and output arrays for the duration
// of each C call. The source is immutable, and no output aliases it.

#[no_mangle]
unsafe extern "C" fn meshlet_validate_v1(
    src: *const u8,
    len: usize,
    at: usize,
    out: *mut [u32; 3],
) -> i32 {
    if src.is_null() || out.is_null() || len > isize::MAX as usize {
        return -1;
    }
    let src = unsafe { slice::from_raw_parts(src, len) };
    match Group::read(src, at).and_then(|group| group.validate()) {
        Some(result) => {
            unsafe { out.write(result) };
            0
        }
        None => -1,
    }
}

#[no_mangle]
unsafe extern "C" fn meshlet_geometry_v1(
    src: *const u8,
    len: usize,
    at: usize,
    block: *const Block,
    flags: u32,
    compensated: u32,
    out: *mut [u32; RESULT_LEN],
) -> i32 {
    if src.is_null()
        || block.is_null()
        || out.is_null()
        || len > isize::MAX as usize
        || flags & !15 != 0
    {
        return -1;
    }
    let src = unsafe { slice::from_raw_parts(src, len) };
    match geometry(src, at, unsafe { &*block }, flags, compensated != 0) {
        Some(result) => {
            unsafe { out.write(result) };
            0
        }
        None => -1,
    }
}

#[no_mangle]
unsafe extern "C" fn meshlet_block_v1(
    src: *const u8,
    len: usize,
    block: *const Block,
    tangents: *const usize,
    count: usize,
    compensated: u32,
    out: *mut [usize; 7],
) -> i32 {
    if src.is_null()
        || block.is_null()
        || tangents.is_null()
        || out.is_null()
        || len > isize::MAX as usize
        || count > 23
    {
        return -1;
    }
    let src = unsafe { slice::from_raw_parts(src, len) };
    let tangents = unsafe { slice::from_raw_parts(tangents, count) };
    match block_counts(src, unsafe { &*block }, tangents, compensated != 0) {
        Some(result) => {
            unsafe { out.write(result) };
            0
        }
        None => -1,
    }
}

#[no_mangle]
unsafe extern "C" fn meshlet_decode_v1(
    src: *const u8,
    len: usize,
    header: usize,
    start: usize,
    count: usize,
    mode: u32,
    compensated: u32,
    out: *mut f64,
) -> i32 {
    let width = if mode == 8 { 4 } else { 3 };
    if src.is_null()
        || out.is_null()
        || len > isize::MAX as usize
        || count > isize::MAX as usize / (8 * width)
        || !matches!(mode, 0 | 1 | 7 | 8)
    {
        return -1;
    }
    let src = unsafe { slice::from_raw_parts(src, len) };
    let output = unsafe { slice::from_raw_parts_mut(out, count * width) };
    if decode_rows(src, header, start, mode, compensated != 0, output).is_some() {
        0
    } else {
        -1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_values_are_exact() {
        assert_eq!(finite_half(0x8000).unwrap().to_bits(), (-0.0f64).to_bits());
        assert_eq!(finite_half(1), Some(2f64.powi(-24)));
        assert_eq!(finite_half(0x0400), Some(2f64.powi(-14)));
        assert_eq!(finite_half(0x3e00), Some(1.5));
        assert_eq!(finite_half(0x7bff), Some(65504.0));
        assert_eq!(finite_half(0x7c00), None);
    }

    #[test]
    fn packed_position_order_and_sign() {
        let transform = PositionTransform {
            bias: [-0.0, 0.0, 0.0],
            scale: [-0.0, 1.0, 1.0],
            axis: 1,
        };
        let word = [0x34, 0, 0, 0, 0x12, 0, 3, 0];
        let result = transform.decode(&word);
        assert_eq!([result[0], result[1]], [4660.0, 3.0]);
        assert_eq!(result[2].to_bits(), (-0.0f64).to_bits());
    }

    #[test]
    fn group_checks_layout_and_local_indices() {
        let mut src = Vec::new();
        for word in [0x8000_0001u32, 1, 1, 0, 0, 0, 0, 0] {
            src.extend(word.to_le_bytes());
        }
        src.extend(65535u16.to_le_bytes());
        assert_eq!(
            Group::read(&src, 0).unwrap().validate(),
            Some([0, 65536, 1])
        );
        assert!(Group::read(&src[..src.len() - 1], 0).is_none());
        src[28] = 1;
        assert_eq!(Group::read(&src, 0).unwrap().validate(), None);
    }

    #[test]
    fn sum_modes_match_reference_order() {
        let terms = [1e16, 1.0, -1e16];
        assert_eq!(sum3(terms, false), 0.0);
        assert_eq!(sum3(terms, true), 1.0);
    }
}
