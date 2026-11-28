//! Optional native kernels for `tools/glb_writer.py`.
//!
//! Every exported function either writes the exact bytes and values the Python reference produces
//! or returns a refusal code, in which case the caller discards the output and runs Python, which
//! then gives its own result or raises its own error. Nothing here raises or allocates on the
//! caller's behalf beyond the scratch arrays of the smooth-normal pass.

use std::ffi::c_void;
use std::slice;

/// The call wrote a complete result.
pub const OK: i32 = 0;
/// The input is outside what this kernel reproduces exactly; Python decides.
pub const DECLINE: i32 = 1;

/// Integers up to this magnitude give the same differences and cross products in CPython's exact
/// integer arithmetic and in binary64 (products stay below 2**51).
const EXACT_INT_LIMIT: f64 = 16_777_216.0;

#[inline(always)]
fn clean_zero(v: f64) -> f64 {
    if v == 0.0 {
        0.0
    } else {
        v
    }
}

/// `frexp` for a positive finite `x`: the exponent `e` with `x = m * 2**e`, `0.5 <= m < 1`.
#[inline(always)]
fn frexp_exponent(x: f64) -> i32 {
    let biased = ((x.to_bits() >> 52) & 0x7ff) as i32;
    if biased == 0 {
        // Subnormal: normalise by 2**54 first.
        let scaled = x * f64::from_bits(((1023 + 54) as u64) << 52);
        return ((scaled.to_bits() >> 52) & 0x7ff) as i32 - 1022 - 54;
    }
    biased - 1022
}

/// `ldexp(1.0, k)` for `-1074 <= k <= 1023`.
#[inline(always)]
fn pow2(k: i32) -> f64 {
    if k >= -1022 {
        f64::from_bits(((k + 1023) as u64) << 52)
    } else {
        f64::from_bits(1u64 << (k + 1074))
    }
}

#[inline(always)]
fn dl_mul(x: f64, y: f64) -> (f64, f64) {
    let z = x * y;
    (z, x.mul_add(y, -z))
}

#[inline(always)]
fn dl_fast_sum(a: f64, b: f64) -> (f64, f64) {
    let hi = a + b;
    (hi, (a - hi) + b)
}

/// The scaled, compensated norm CPython 3.12 uses for `math.hypot` with several arguments.
/// `vec` holds absolute values, `max` their maximum; all finite.
#[inline(always)]
fn vector_norm(vec: &mut [f64; 3], mut max: f64) -> f64 {
    if max == 0.0 {
        return max;
    }
    let mut outer = 1.0f64;
    let mut max_e = frexp_exponent(max);
    if max_e < -1023 {
        // CPython recurses once: a subnormal maximum divided by DBL_MIN is normal.
        for v in vec.iter_mut() {
            *v /= f64::MIN_POSITIVE;
        }
        max /= f64::MIN_POSITIVE;
        outer = f64::MIN_POSITIVE;
        max_e = frexp_exponent(max);
    }
    let scale = pow2(-max_e);
    let mut csum = 1.0f64;
    let mut frac1 = 0.0f64;
    let mut frac2 = 0.0f64;
    for &v in vec.iter() {
        let x = v * scale;
        let (hi, lo) = dl_mul(x, x);
        let (s_hi, s_lo) = dl_fast_sum(csum, hi);
        csum = s_hi;
        frac1 += lo;
        frac2 += s_lo;
    }
    let mut h = (csum - 1.0 + (frac1 + frac2)).sqrt();
    let (hi, lo) = dl_mul(-h, h);
    let (s_hi, s_lo) = dl_fast_sum(csum, hi);
    csum = s_hi;
    frac1 += lo;
    frac2 += s_lo;
    let x = csum - 1.0 + (frac1 + frac2);
    h += x / (2.0 * h);
    outer * (h / scale)
}

/// `math.hypot(x, y, z)` for finite arguments.
pub fn hypot3(x: f64, y: f64, z: f64) -> f64 {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("fma") {
        // SAFETY: the CPU has FMA.
        return unsafe { hypot3_fma(x, y, z) };
    }
    hypot3_body(x, y, z)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "fma")]
unsafe fn hypot3_fma(x: f64, y: f64, z: f64) -> f64 {
    hypot3_body(x, y, z)
}

/// `mul_add` is the exact fused multiply-add either way; with the `fma` target feature it is one
/// instruction instead of a library call.
#[inline(always)]
fn hypot3_body(x: f64, y: f64, z: f64) -> f64 {
    let mut vec = [x.abs(), y.abs(), z.abs()];
    let mut max = 0.0f64;
    for &v in vec.iter() {
        if v > max {
            max = v;
        }
    }
    vector_norm(&mut vec, max)
}

/// `glb_writer._normalize3`.
pub fn normalize3(nx: f64, ny: f64, nz: f64) -> (f64, f64, f64) {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("fma") {
        // SAFETY: the CPU has FMA.
        return unsafe { normalize3_fma(nx, ny, nz) };
    }
    normalize3_body(nx, ny, nz)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "fma")]
unsafe fn normalize3_fma(nx: f64, ny: f64, nz: f64) -> (f64, f64, f64) {
    normalize3_body(nx, ny, nz)
}

#[inline(always)]
fn normalize3_body(mut nx: f64, mut ny: f64, mut nz: f64) -> (f64, f64, f64) {
    if !(nx.is_finite() && ny.is_finite() && nz.is_finite()) {
        return (0.0, 1.0, 0.0);
    }
    let mut length = hypot3_body(nx, ny, nz);
    if length == 0.0 {
        return (0.0, 1.0, 0.0);
    }
    if length < f64::MIN_POSITIVE || !length.is_finite() {
        // Python's max(abs(nx), abs(ny), abs(nz)) on non-NaN floats.
        let mut scale = nx.abs();
        if ny.abs() > scale {
            scale = ny.abs();
        }
        if nz.abs() > scale {
            scale = nz.abs();
        }
        nx /= scale;
        ny /= scale;
        nz /= scale;
        length = hypot3_body(nx, ny, nz);
    }
    (clean_zero(nx / length), clean_zero(ny / length), clean_zero(nz / length))
}

#[derive(Clone, Copy, PartialEq)]
enum NormalSource {
    /// `values` are the primitive's own normals (3 per vertex), normalised one by one.
    Given,
    /// `values` are positions; `_compute_smooth_normals` over `indices`.
    Smooth,
}

/// The NORMAL buffer (float32 little-endian) into `out`, using hardware FMA when the CPU has it.
fn write_normals(values: &[f64], indices: &[i64], source: NormalSource, out: &mut [u8]) {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("fma") {
        // SAFETY: the CPU has FMA.
        return unsafe { write_normals_fma(values, indices, source, out) };
    }
    write_normals_body(values, indices, source, out)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "fma")]
unsafe fn write_normals_fma(values: &[f64], indices: &[i64], source: NormalSource, out: &mut [u8]) {
    write_normals_body(values, indices, source, out)
}

#[inline(always)]
fn write_normals_body(values: &[f64], indices: &[i64], source: NormalSource, out: &mut [u8]) {
    if source == NormalSource::Smooth {
        smooth_normals(values, indices, out);
        return;
    }
    for (src, dst) in values.chunks_exact(3).zip(out.chunks_exact_mut(12)) {
        write_normal(normalize3_body(src[0], src[1], src[2]), dst);
    }
}

/// `glb_writer._compute_smooth_normals`, written as float32 little-endian into `out`.
#[inline(always)]
fn smooth_normals(positions: &[f64], indices: &[i64], out: &mut [u8]) {
    let n = positions.len() / 3;
    let mut acc = vec![0.0f64; n * 3];
    for tri in indices.chunks_exact(3) {
        let (i0, i1, i2) = (tri[0] as usize * 3, tri[1] as usize * 3, tri[2] as usize * 3);
        let p0 = &positions[i0..i0 + 3];
        let p1 = &positions[i1..i1 + 3];
        let p2 = &positions[i2..i2 + 3];
        let e1x = p1[0] - p0[0];
        let e1y = p1[1] - p0[1];
        let e1z = p1[2] - p0[2];
        let e2x = p2[0] - p0[0];
        let e2y = p2[1] - p0[1];
        let e2z = p2[2] - p0[2];
        let cx = e1y * e2z - e1z * e2y;
        let cy = e1z * e2x - e1x * e2z;
        let cz = e1x * e2y - e1y * e2x;
        if !(cx.is_finite() && cy.is_finite() && cz.is_finite()) {
            continue;
        }
        for &i in &[i0, i1, i2] {
            acc[i] += cx;
            acc[i + 1] += cy;
            acc[i + 2] += cz;
        }
    }
    for (src, dst) in acc.chunks_exact(3).zip(out.chunks_exact_mut(12)) {
        write_normal(normalize3_body(src[0], src[1], src[2]), dst);
    }
}

#[inline(always)]
fn write_normal(n: (f64, f64, f64), dst: &mut [u8]) {
    dst[0..4].copy_from_slice(&(n.0 as f32).to_le_bytes());
    dst[4..8].copy_from_slice(&(n.1 as f32).to_le_bytes());
    dst[8..12].copy_from_slice(&(n.2 as f32).to_le_bytes());
}

/// The packed size `glb_pack` needs, or 0 when the counts overflow.
pub fn packed_len(vertices: usize, index_count: usize, has_uv: bool) -> usize {
    let index_width = if vertices <= 65535 { 2 } else { 4 };
    let per_vertex = if has_uv { 32 } else { 24 };
    vertices
        .checked_mul(per_vertex)
        .and_then(|v| index_count.checked_mul(index_width).and_then(|i| v.checked_add(i)))
        .unwrap_or(0)
}

/// `glb_writer.pack_primitive` on already-flattened inputs.
///
/// `positions` holds 3 binary64 values per vertex; `indices` the triangle list; `normals` (3 per
/// vertex) and `uvs` (2 per vertex) are optional. `out` receives positions, normals, UVs (when
/// given) and indices back to back. `bounds` receives the POSITION min xyz then max xyz.
/// `exact_ints` marks positions that held Python ints, whose arithmetic is only reproduced while
/// every magnitude stays at or below 2**24.
pub fn pack(
    positions: &[f64],
    indices: &[i64],
    normals: Option<&[f64]>,
    uvs: Option<&[f64]>,
    exact_ints: bool,
    out: &mut [u8],
    bounds: &mut [f64; 6],
) -> i32 {
    let n = positions.len() / 3;
    if n == 0 || positions.len() != n * 3 || indices.is_empty() || indices.len() % 3 != 0 {
        return DECLINE;
    }
    if normals.is_some_and(|v| v.len() != n * 3) || uvs.is_some_and(|v| v.len() != n * 2) {
        return DECLINE;
    }
    if out.len() != packed_len(n, indices.len(), uvs.is_some()) {
        return DECLINE;
    }
    if exact_ints && positions.iter().any(|v| !(v.abs() <= EXACT_INT_LIMIT)) {
        return DECLINE;
    }
    let limit = if n <= 65535 { n as i64 } else { (n as u64).min(1u64 << 32) as i64 };
    if indices.iter().any(|&i| i < 0 || i >= limit) {
        return DECLINE;
    }

    let (pos_out, rest) = out.split_at_mut(n * 12);
    let (nrm_out, rest) = rest.split_at_mut(n * 12);
    let (uv_out, idx_out) = rest.split_at_mut(if uvs.is_some() { n * 8 } else { 0 });

    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for (src, dst) in positions.chunks_exact(3).zip(pos_out.chunks_exact_mut(12)) {
        for k in 0..3 {
            let f = src[k] as f32;
            if !f.is_finite() {
                return DECLINE;
            }
            dst[k * 4..k * 4 + 4].copy_from_slice(&f.to_le_bytes());
            let w = f as f64;
            if w < lo[k] {
                lo[k] = w;
            }
            if w > hi[k] {
                hi[k] = w;
            }
        }
    }

    if let Some(uv) = uvs {
        for (src, dst) in uv.chunks_exact(2).zip(uv_out.chunks_exact_mut(8)) {
            for k in 0..2 {
                let v = if src[k].is_finite() { clean_zero(src[k]) } else { 0.0 };
                let f = v as f32;
                if !f.is_finite() {
                    return DECLINE;
                }
                dst[k * 4..k * 4 + 4].copy_from_slice(&f.to_le_bytes());
            }
        }
    }

    if n <= 65535 {
        for (&i, dst) in indices.iter().zip(idx_out.chunks_exact_mut(2)) {
            dst.copy_from_slice(&(i as u16).to_le_bytes());
        }
    } else {
        for (&i, dst) in indices.iter().zip(idx_out.chunks_exact_mut(4)) {
            dst.copy_from_slice(&(i as u32).to_le_bytes());
        }
    }

    match normals {
        Some(given) => write_normals(given, &[], NormalSource::Given, nrm_out),
        None => write_normals(positions, indices, NormalSource::Smooth, nrm_out),
    }

    for k in 0..3 {
        bounds[k] = clean_zero(lo[k]);
        bounds[k + 3] = clean_zero(hi[k]);
    }
    OK
}

/// One step of `glb_writer.compute_bounds`: folds the finite positions (3 binary64 values each)
/// into `acc` (min xyz, then max xyz) and reports whether there was one.
pub fn accumulate_bounds(positions: &[f64], acc: &mut [f64; 6]) -> bool {
    let mut found = false;
    for p in positions.chunks_exact(3) {
        if !(p[0].is_finite() && p[1].is_finite() && p[2].is_finite()) {
            continue;
        }
        found = true;
        for k in 0..3 {
            if p[k] < acc[k] {
                acc[k] = p[k];
            }
            if p[k] > acc[k + 3] {
                acc[k + 3] = p[k];
            }
        }
    }
    found
}

/// The CPython C-API functions and type objects the object readers use, filled in by the ctypes
/// binding from `ctypes.pythonapi`. Every reader runs with the GIL held (the binding calls through
/// `ctypes.PyDLL`) and only reads exact `float`, `int`, `list` and `tuple` objects, so no Python
/// code runs and nothing can change the containers while they are read.
#[repr(C)]
pub struct PyApi {
    list_size: unsafe extern "C" fn(*mut c_void) -> isize,
    list_get: unsafe extern "C" fn(*mut c_void, isize) -> *mut c_void,
    tuple_size: unsafe extern "C" fn(*mut c_void) -> isize,
    tuple_get: unsafe extern "C" fn(*mut c_void, isize) -> *mut c_void,
    float_as_double: unsafe extern "C" fn(*mut c_void) -> f64,
    long_as_double: unsafe extern "C" fn(*mut c_void) -> f64,
    long_as_long_long_and_overflow: unsafe extern "C" fn(*mut c_void, *mut i32) -> i64,
    err_occurred: unsafe extern "C" fn() -> *mut c_void,
    err_clear: unsafe extern "C" fn(),
    float_type: *const c_void,
    long_type: *const c_void,
    list_type: *const c_void,
    tuple_type: *const c_void,
    /// Nonzero once the binding has checked [`glb_layout_probe_v1`] on this interpreter: list and
    /// tuple items and float values are then read in place instead of through the functions above.
    direct: usize,
    /// Byte offset of a tuple's first item (`tuple.__basicsize__`).
    tuple_items: usize,
}

/// Offsets in a default 64-bit build: `ob_size` follows the object header, a list's item array
/// pointer follows `ob_size`, and a float's value follows the header.
const SIZE_OFFSET: usize = 16;
const LIST_ITEMS_OFFSET: usize = 24;
const FLOAT_VALUE_OFFSET: usize = 16;

/// `Py_TYPE(obj)`: in a default (GIL) build `ob_type` follows the reference count. The binding
/// checks this against `id(type(x))` before it uses any other reader.
#[inline]
unsafe fn type_of(obj: *mut c_void) -> *const c_void {
    *(obj as *const *const c_void).add(1)
}

#[derive(Clone, Copy)]
struct Seq {
    obj: *mut c_void,
    tuple: bool,
    len: usize,
}

impl PyApi {
    /// An exact list or tuple, with its length.
    unsafe fn seq(&self, obj: *mut c_void) -> Option<Seq> {
        if obj.is_null() {
            return None;
        }
        let t = type_of(obj);
        let tuple = if t == self.list_type {
            false
        } else if t == self.tuple_type {
            true
        } else {
            return None;
        };
        let len = if self.direct != 0 {
            *((obj as *const u8).add(SIZE_OFFSET) as *const isize)
        } else if tuple {
            (self.tuple_size)(obj)
        } else {
            (self.list_size)(obj)
        };
        usize::try_from(len).ok().map(|len| Seq { obj, tuple, len })
    }

    /// Item `i < seq.len` (a borrowed reference).
    #[inline]
    unsafe fn item(&self, seq: Seq, i: usize) -> *mut c_void {
        let base = seq.obj as *const u8;
        match (self.direct != 0, seq.tuple) {
            (true, true) => *(base.add(self.tuple_items) as *const *mut c_void).add(i),
            (true, false) => *(*(base.add(LIST_ITEMS_OFFSET) as *const *const *mut c_void)).add(i),
            (false, true) => (self.tuple_get)(seq.obj, i as isize),
            (false, false) => (self.list_get)(seq.obj, i as isize),
        }
    }

    /// `float(x)` for an exact float or int, and whether it was an int. A bool, a subclass, any
    /// other type, or an int too large for a double gives `None` and leaves no exception set.
    #[inline]
    unsafe fn number(&self, obj: *mut c_void) -> Option<(f64, bool)> {
        let t = type_of(obj);
        if t == self.float_type {
            let v = if self.direct != 0 {
                *((obj as *const u8).add(FLOAT_VALUE_OFFSET) as *const f64)
            } else {
                (self.float_as_double)(obj)
            };
            Some((v, false))
        } else if t == self.long_type {
            let v = (self.long_as_double)(obj);
            if v == -1.0 && !(self.err_occurred)().is_null() {
                (self.err_clear)();
                return None;
            }
            Some((v, true))
        } else {
            None
        }
    }

    /// The rows of `seq` (lists or tuples of exactly `width` exact floats or ints), appended to
    /// `out`. With `exact_ints`, an int above 2**24 in magnitude declines.
    unsafe fn rows(&self, seq: Seq, width: usize, exact_ints: bool, out: &mut Vec<f64>) -> Option<()> {
        out.reserve(seq.len.checked_mul(width)?);
        for i in 0..seq.len {
            let row = self.seq(self.item(seq, i))?;
            if row.len != width {
                return None;
            }
            for k in 0..width {
                let (v, is_int) = self.number(self.item(row, k))?;
                if exact_ints && is_int && !(v.abs() <= EXACT_INT_LIMIT) {
                    return None;
                }
                out.push(v);
            }
        }
        Some(())
    }

    /// A list or tuple of exact ints that fit in an `i64`.
    unsafe fn ints(&self, seq: Seq) -> Option<Vec<i64>> {
        let mut out = Vec::with_capacity(seq.len);
        for i in 0..seq.len {
            let obj = self.item(seq, i);
            if type_of(obj) != self.long_type {
                return None;
            }
            let mut overflow = 0i32;
            let v = (self.long_as_long_long_and_overflow)(obj, &mut overflow);
            if overflow != 0 || (v == -1 && !(self.err_occurred)().is_null()) {
                (self.err_clear)();
                return None;
            }
            out.push(v);
        }
        Some(out)
    }
}

/// The in-place reads of `seq` (an exact list or tuple), for the binding to compare with `len()`
/// and `id()` before it sets `direct`: writes the length, then each item's address, into `out`
/// (`1 + len` words, at most `cap`), and returns the number of words written, or -1. Nothing is
/// read through an item address, so a wrong layout gives a mismatch rather than a fault.
///
/// # Safety
/// Called with the GIL held on a live object; `out` addresses `cap` words; `api.tuple_items` is
/// `tuple.__basicsize__`.
#[no_mangle]
pub unsafe extern "C" fn glb_layout_probe_v1(api: *const PyApi, seq: *mut c_void, out: *mut u64, cap: usize) -> isize {
    if api.is_null() || out.is_null() {
        return -1;
    }
    let mut probe = std::ptr::read(api);
    probe.direct = 1;
    let Some(s) = probe.seq(seq) else {
        return -1;
    };
    if s.len >= cap {
        return -1;
    }
    let out = slice::from_raw_parts_mut(out, 1 + s.len);
    out[0] = s.len as u64;
    for i in 0..s.len {
        out[1 + i] = probe.item(s, i) as u64;
    }
    (1 + s.len) as isize
}

/// The raw bits at a float's value offset, for the binding to compare with the value.
///
/// # Safety
/// `obj` is a live exact float.
#[no_mangle]
pub unsafe extern "C" fn glb_float_probe_v1(obj: *mut c_void) -> u64 {
    (*((obj as *const u8).add(FLOAT_VALUE_OFFSET) as *const f64)).to_bits()
}

/// `Py_TYPE(obj)` as an address, for the binding's layout check.
///
/// # Safety
/// `obj` is a live Python object.
#[no_mangle]
pub unsafe extern "C" fn glb_type_of_v1(obj: *mut c_void) -> usize {
    type_of(obj) as usize
}

/// C entry point of [`hypot3`]; Python compares it with `math.hypot` before trusting the library.
#[no_mangle]
pub extern "C" fn glb_hypot3_v1(x: f64, y: f64, z: f64) -> f64 {
    hypot3(x, y, z)
}

/// [`pack`] reading `pack_primitive`'s lists directly. `normals` and `uvs` may be null (absent).
///
/// # Safety
/// Called with the GIL held; `api` is filled in from this interpreter; every object pointer is a
/// live object or null; `out` addresses `out_len` bytes and `bounds_out` six doubles.
#[no_mangle]
pub unsafe extern "C" fn glb_pack_obj_v1(
    api: *const PyApi,
    positions: *mut c_void,
    indices: *mut c_void,
    normals: *mut c_void,
    uvs: *mut c_void,
    out: *mut u8,
    out_len: usize,
    bounds_out: *mut f64,
) -> i32 {
    if api.is_null() || out.is_null() || bounds_out.is_null() {
        return DECLINE;
    }
    let api = &*api;
    let (Some(pos_seq), Some(idx_seq)) = (api.seq(positions), api.seq(indices)) else {
        return DECLINE;
    };
    let mut pos = Vec::new();
    if api.rows(pos_seq, 3, true, &mut pos).is_none() {
        return DECLINE;
    }
    let Some(idx) = api.ints(idx_seq) else {
        return DECLINE;
    };
    let mut nrm = Vec::new();
    if !normals.is_null() {
        let Some(seq) = api.seq(normals) else {
            return DECLINE;
        };
        if api.rows(seq, 3, false, &mut nrm).is_none() {
            return DECLINE;
        }
    }
    let mut uv = Vec::new();
    if !uvs.is_null() {
        let Some(seq) = api.seq(uvs) else {
            return DECLINE;
        };
        if api.rows(seq, 2, false, &mut uv).is_none() {
            return DECLINE;
        }
    }
    let out = slice::from_raw_parts_mut(out, out_len);
    let mut b = [0.0f64; 6];
    let rc = pack(
        &pos,
        &idx,
        (!normals.is_null()).then_some(&nrm[..]),
        (!uvs.is_null()).then_some(&uv[..]),
        false,
        out,
        &mut b,
    );
    if rc == OK {
        slice::from_raw_parts_mut(bounds_out, 6).copy_from_slice(&b);
    }
    rc
}

/// Folds one primitive's `positions` (a list or tuple of 3-number lists or tuples) into the
/// running `compute_bounds` state: `acc` (min xyz, max xyz) and `found`.
///
/// # Safety
/// As for [`glb_pack_obj_v1`]; `acc` addresses six doubles and `found` one int.
#[no_mangle]
pub unsafe extern "C" fn glb_bounds_obj_v1(
    api: *const PyApi,
    positions: *mut c_void,
    acc: *mut f64,
    found: *mut i32,
) -> i32 {
    if api.is_null() || acc.is_null() || found.is_null() {
        return DECLINE;
    }
    let api = &*api;
    let Some(seq) = api.seq(positions) else {
        return DECLINE;
    };
    let mut pos = Vec::new();
    if api.rows(seq, 3, false, &mut pos).is_none() {
        return DECLINE;
    }
    let acc = &mut *(acc as *mut [f64; 6]);
    if accumulate_bounds(&pos, acc) {
        *found = 1;
    }
    OK
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hypot_matches_simple_cases() {
        assert_eq!(hypot3(3.0, 4.0, 12.0), 13.0);
        assert_eq!(hypot3(0.0, 0.0, 0.0), 0.0);
        assert_eq!(hypot3(-2.0, 0.0, 0.0), 2.0);
        let tiny = f64::from_bits(1);
        assert_eq!(hypot3(tiny, 0.0, 0.0), tiny);
        assert!(hypot3(f64::MAX, f64::MAX, 0.0).is_infinite());
    }

    #[test]
    fn normalize_falls_back_to_up() {
        assert_eq!(normalize3(0.0, 0.0, 0.0), (0.0, 1.0, 0.0));
        assert_eq!(normalize3(f64::NAN, 0.0, 0.0), (0.0, 1.0, 0.0));
        assert_eq!(normalize3(-0.0, 0.0, -2.0), (0.0, 0.0, -1.0));
        let n = normalize3(f64::MAX, f64::MAX, 0.0);
        assert!(n.0 > 0.7 && n.0 == n.1 && n.2 == 0.0);
    }

    #[test]
    fn pack_one_triangle() {
        let pos = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let idx = [0i64, 1, 2];
        let mut out = vec![0u8; packed_len(3, 3, false)];
        let mut b = [0.0; 6];
        assert_eq!(pack(&pos, &idx, None, None, false, &mut out, &mut b), OK);
        assert_eq!(b, [0.0, 0.0, 0.0, 1.0, 1.0, 0.0]);
        // The normal of vertex 0 is +Z.
        assert_eq!(&out[36..48], &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x80, 0x3f]);
        assert_eq!(&out[72..78], &[0, 0, 1, 0, 2, 0]);
    }

    #[test]
    fn pack_declines_what_python_rejects() {
        let pos = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let mut out = vec![0u8; packed_len(3, 3, false)];
        let mut b = [0.0; 6];
        assert_eq!(pack(&pos, &[0, 1, 3], None, None, false, &mut out, &mut b), DECLINE);
        assert_eq!(pack(&pos, &[0, -1, 2], None, None, false, &mut out, &mut b), DECLINE);
        assert_eq!(pack(&pos, &[0, 1], None, None, false, &mut out, &mut b), DECLINE);
        let big = [1e39, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        assert_eq!(pack(&big, &[0, 1, 2], None, None, false, &mut out, &mut b), DECLINE);
        let ints = [2e7, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        assert_eq!(pack(&ints, &[0, 1, 2], None, None, true, &mut out, &mut b), DECLINE);
        assert_eq!(pack(&ints, &[0, 1, 2], None, None, false, &mut out, &mut b), OK);
        let mut uv_out = vec![0u8; packed_len(3, 3, true)];
        let huge_uv = [1e300, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert_eq!(pack(&pos, &[0, 1, 2], None, Some(&huge_uv), false, &mut uv_out, &mut b), DECLINE);
    }

    #[test]
    fn bounds_skip_nonfinite() {
        let inf = f64::INFINITY;
        let mut b = [inf, inf, inf, -inf, -inf, -inf];
        assert!(!accumulate_bounds(&[f64::NAN, 0.0, 0.0], &mut b));
        assert!(accumulate_bounds(&[inf, 0.0, 0.0, -0.0, 2.0, -3.0], &mut b));
        assert_eq!(b, [-0.0, 2.0, -3.0, -0.0, 2.0, -3.0]);
    }
}
