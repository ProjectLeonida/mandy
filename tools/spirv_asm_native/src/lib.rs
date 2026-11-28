//! SPIR-V text assembler for `tools/spirv_asm.py`.
//!
//! A port of `assemble()` in `tools/spirv_asm.py`: the same tokenizer, ID numbering, operand
//! grammar walk, literal rules and header, so a module it accepts comes out byte-identical. The
//! grammar tables are not copied here; `spirv_asm.py` serialises the ones in
//! `tools/spirv_asm_tables.py` and passes them to `spirv_asm_tables_new` once.
//!
//! The native path never reports an error of its own. Anything it does not reproduce exactly
//! (an error of any kind, non-ASCII text, control characters, integer spellings with `_`,
//! decimal floats spelled `inf`/`nan`, hex floats that need `float.fromhex` rounding, odd type
//! widths) makes `spirv_asm_assemble` return 1, and `spirv_asm.py` assembles the text in Python,
//! which then gives its usual result or `SpirvAsmError`. `tests/test_spirv_asm_native.py` checks
//! that the two agree.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};

/// A small multiply-rotate hasher (the symbol and opcode maps are the hot path; SipHash is not needed).
#[derive(Default)]
struct FastHasher(u64);

const HASH_K: u64 = 0x517c_c1b7_2722_0a95;

impl Hasher for FastHasher {
    fn finish(&self) -> u64 {
        // Spread the high bits (where the multiplies put each byte's influence) to the low ones.
        let h = (self.0 ^ (self.0 >> 32)).wrapping_mul(HASH_K);
        h ^ (h >> 29)
    }
    fn write(&mut self, bytes: &[u8]) {
        let mut h = self.0;
        let mut chunks = bytes.chunks_exact(8);
        for c in &mut chunks {
            let mut w = [0u8; 8];
            w.copy_from_slice(c);
            h = (h.rotate_left(5) ^ u64::from_le_bytes(w)).wrapping_mul(HASH_K);
        }
        let rest = chunks.remainder();
        if !rest.is_empty() {
            let mut w = [0u8; 8];
            w[..rest.len()].copy_from_slice(rest);
            h = (h.rotate_left(5) ^ u64::from_le_bytes(w)).wrapping_mul(HASH_K);
        }
        self.0 = h;
    }
    fn write_u128(&mut self, n: u128) {
        let h = (self.0.rotate_left(5) ^ n as u64).wrapping_mul(HASH_K);
        self.0 = (h.rotate_left(5) ^ (n >> 64) as u64).wrapping_mul(HASH_K);
    }
    fn write_u64(&mut self, n: u64) {
        self.0 = (self.0.rotate_left(5) ^ n).wrapping_mul(HASH_K);
    }
    fn write_usize(&mut self, n: usize) {
        self.0 = (self.0.rotate_left(5) ^ n as u64).wrapping_mul(HASH_K);
    }
}

type Map<K, V> = HashMap<K, V, BuildHasherDefault<FastHasher>>;
type Set<K> = HashSet<K, BuildHasherDefault<FastHasher>>;

/// Refusal: the Python assembler handles the text.
#[derive(Debug, PartialEq)]
struct Refuse;
type R<T> = Result<T, Refuse>;

#[derive(Clone, Copy, PartialEq, Debug)]
enum Kind {
    Id,
    LitInt,
    LitFloat,
    LitString,
    LitContextNumber,
    LitExtInst,
    PairLitId,
    PairIdId,
    PairIdLit,
    SpecConstantOp,
    Value(usize),
    Bits(usize),
    Unsupported,
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Quant {
    One,
    Optional,
    Many,
}

struct Insn {
    opcode: u32,
    has_rt: bool,
    has_res: bool,
    type_def: bool,
    specs: Vec<(Kind, Quant)>,
}

struct EnumEntry {
    value: u32,
    params: Vec<Kind>,
}

/// The grammar of `tools/spirv_asm_tables.py`, as `spirv_asm.py` serialises it.
pub struct Tables {
    /// Instructions named in up to 16 bytes, keyed by `pack(name)`; longer names in `long_insns`.
    insns: Map<u128, Insn>,
    long_insns: Map<Vec<u8>, Insn>,
    values: Vec<Map<Vec<u8>, EnumEntry>>,
    bits: Vec<Map<Vec<u8>, EnumEntry>>,
    ext_sets: Map<Vec<u8>, usize>,
    ext_tables: Vec<Map<Vec<u8>, u32>>,
}

impl Tables {
    fn insn(&self, name: &[u8]) -> Option<&Insn> {
        if name.len() <= 16 {
            if name.contains(&0) {
                return None;
            }
            self.insns.get(&pack(name))
        } else {
            self.long_insns.get(name)
        }
    }
}

fn parse_u32(s: &str) -> Option<u32> {
    s.parse::<u32>().ok()
}

impl Tables {
    /// Parses the table text: one record per line, fields separated by single spaces.
    ///
    /// `I <name> <opcode> <has result type 0|1> <has result 0|1> <kind>[?|*]...`,
    /// `V <kind> <name> <value> <param kind>...` (value enums), `B ...` (bit enums), and
    /// `X <set> <name> <opcode>` (extended instruction sets).
    pub fn parse(text: &str) -> Option<Tables> {
        let mut value_index: Map<&str, usize> = Map::default();
        let mut bits_index: Map<&str, usize> = Map::default();
        for line in text.lines() {
            let mut f = line.split(' ');
            match f.next() {
                Some("V") => {
                    let k = f.next()?;
                    let n = value_index.len();
                    value_index.entry(k).or_insert(n);
                }
                Some("B") => {
                    let k = f.next()?;
                    let n = bits_index.len();
                    bits_index.entry(k).or_insert(n);
                }
                _ => {}
            }
        }
        // Same order as `encode_operand`: ID kinds, literal kinds, value enums, then bit enums.
        let kind_of = |name: &str| -> Kind {
            match name {
                "IdRef" | "IdResult" | "IdResultType" | "IdScope" | "IdMemorySemantics" => Kind::Id,
                "LiteralInteger" => Kind::LitInt,
                "LiteralFloat" => Kind::LitFloat,
                "LiteralString" => Kind::LitString,
                "LiteralContextDependentNumber" => Kind::LitContextNumber,
                "LiteralExtInstInteger" => Kind::LitExtInst,
                "PairLiteralIntegerIdRef" => Kind::PairLitId,
                "PairIdRefIdRef" => Kind::PairIdId,
                "PairIdRefLiteralInteger" => Kind::PairIdLit,
                "LiteralSpecConstantOpInteger" => Kind::SpecConstantOp,
                _ => {
                    if let Some(&i) = value_index.get(name) {
                        Kind::Value(i)
                    } else if let Some(&i) = bits_index.get(name) {
                        Kind::Bits(i)
                    } else {
                        Kind::Unsupported
                    }
                }
            }
        };
        let mut t = Tables {
            insns: Map::default(),
            long_insns: Map::default(),
            values: (0..value_index.len()).map(|_| Map::default()).collect(),
            bits: (0..bits_index.len()).map(|_| Map::default()).collect(),
            ext_sets: Map::default(),
            ext_tables: Vec::new(),
        };
        for line in text.lines() {
            if line.is_empty() {
                continue;
            }
            let f: Vec<&str> = line.split(' ').collect();
            match f[0] {
                "I" if f.len() >= 5 => {
                    let mut specs = Vec::with_capacity(f.len() - 5);
                    for s in &f[5..] {
                        let (k, q) = if let Some(k) = s.strip_suffix('*') {
                            (k, Quant::Many)
                        } else if let Some(k) = s.strip_suffix('?') {
                            (k, Quant::Optional)
                        } else {
                            (*s, Quant::One)
                        };
                        specs.push((kind_of(k), q));
                    }
                    let has_res = f[4] == "1";
                    let insn = Insn {
                        opcode: parse_u32(f[2])?,
                        has_rt: f[3] == "1",
                        has_res,
                        type_def: has_res && f[1].starts_with("OpType"),
                        specs,
                    };
                    let name = f[1].as_bytes();
                    if name.len() <= 16 {
                        t.insns.insert(pack(name), insn);
                    } else {
                        t.long_insns.insert(name.to_vec(), insn);
                    }
                }
                "V" | "B" if f.len() >= 4 => {
                    let entry = EnumEntry {
                        value: parse_u32(f[3])?,
                        params: f[4..].iter().map(|k| kind_of(k)).collect(),
                    };
                    let table = if f[0] == "V" {
                        &mut t.values[value_index[f[1]]]
                    } else {
                        &mut t.bits[bits_index[f[1]]]
                    };
                    table.insert(f[2].as_bytes().to_vec(), entry);
                }
                "X" if f.len() == 4 => {
                    let n = t.ext_tables.len();
                    let i = *t.ext_sets.entry(f[1].as_bytes().to_vec()).or_insert(n);
                    if i == n {
                        t.ext_tables.push(Map::default());
                    }
                    t.ext_tables[i].insert(f[2].as_bytes().to_vec(), parse_u32(f[3])?);
                }
                _ => return None,
            }
        }
        Some(t)
    }
}

/// `type_info` of `assemble()`: ('int', width, signedness) or ('float', width).
#[derive(Clone, Copy)]
enum TypeInfo {
    None,
    Int(i64, bool),
    Float(i64),
}

#[derive(Clone, Copy)]
struct Ctx {
    res_type: Option<u32>,
    ext: Option<usize>,
    switch: Option<(i64, bool)>,
}

/// `int(tok, 0)` for the spellings that need no special casing: an optional sign, then
/// `0x`/`0o`/`0b` digits, a decimal without leading zeros, or zeros. Others (underscores) refuse.
fn parse_int(tok: &[u8]) -> R<i128> {
    let (neg, body) = match tok.first() {
        Some(b'-') => (true, &tok[1..]),
        Some(b'+') => (false, &tok[1..]),
        _ => (false, tok),
    };
    let (radix, digits) = if body.len() > 2 && body[0] == b'0' {
        match body[1] {
            b'x' | b'X' => (16, &body[2..]),
            b'o' | b'O' => (8, &body[2..]),
            b'b' | b'B' => (2, &body[2..]),
            _ => (10, body),
        }
    } else {
        (10, body)
    };
    if digits.is_empty() || digits.len() > 40 {
        return Err(Refuse);
    }
    if radix == 10 && digits[0] == b'0' && digits.iter().any(|&c| c != b'0') {
        return Err(Refuse);
    }
    let mut v: i128 = 0;
    for &c in digits {
        let d = (c as char).to_digit(radix).ok_or(Refuse)? as i128;
        v = v.checked_mul(radix as i128).and_then(|v| v.checked_add(d)).ok_or(Refuse)?;
    }
    Ok(if neg { -v } else { v })
}

/// `_int_literal`: the bits of an integer literal for a `width`-bit integer type.
fn int_literal(tok: &[u8], width: i64, signed: bool) -> R<u64> {
    if !matches!(width, 8 | 16 | 32 | 64) {
        return Err(Refuse);
    }
    let mut val = parse_int(tok)?;
    if !signed {
        if tok.first() == Some(&b'-') || (val >> width) != 0 {
            return Err(Refuse);
        }
    } else if tok.len() >= 2 && tok[0] == b'0' && (tok[1] == b'x' || tok[1] == b'X') && (val >> width) == 0 {
        if (val >> (width - 1)) != 0 {
            val -= 1i128 << width;
        }
    } else if !(-(1i128 << (width - 1)) <= val && val < (1i128 << (width - 1))) {
        return Err(Refuse);
    }
    let bits = width.max(32);
    Ok((val & ((1i128 << bits) - 1)) as u64)
}

fn u32_literal(tok: &[u8]) -> R<u32> {
    Ok(int_literal(tok, 32, false)? as u32)
}

/// `float(tok)` for `[+-]?(digits[.digits]|.digits)([eE][+-]?digits)?`; other spellings refuse.
fn parse_decimal(tok: &[u8]) -> R<f64> {
    let mut i = 0;
    let n = tok.len();
    if i < n && (tok[i] == b'+' || tok[i] == b'-') {
        i += 1;
    }
    let int_start = i;
    while i < n && tok[i].is_ascii_digit() {
        i += 1;
    }
    let mut mant_digits = i - int_start;
    if i < n && tok[i] == b'.' {
        i += 1;
        let frac_start = i;
        while i < n && tok[i].is_ascii_digit() {
            i += 1;
        }
        mant_digits += i - frac_start;
    }
    if mant_digits == 0 {
        return Err(Refuse);
    }
    if i < n && (tok[i] == b'e' || tok[i] == b'E') {
        i += 1;
        if i < n && (tok[i] == b'+' || tok[i] == b'-') {
            i += 1;
        }
        let exp_start = i;
        while i < n && tok[i].is_ascii_digit() {
            i += 1;
        }
        if i == exp_start {
            return Err(Refuse);
        }
    }
    if i != n {
        return Err(Refuse);
    }
    std::str::from_utf8(tok).map_err(|_| Refuse)?.parse::<f64>().map_err(|_| Refuse)
}

/// CPython's `PyFloat_Pack2` (`struct.pack('<e', x)`); `None` where it raises OverflowError.
fn pack_f16(x: f64) -> Option<u16> {
    let sign: u16 = if x.is_sign_negative() { 1 } else { 0 };
    if x == 0.0 {
        return Some(sign << 15);
    }
    if x.is_infinite() {
        return Some((sign << 15) | 0x7c00);
    }
    let x = x.abs();
    let raw = x.to_bits();
    let exp_field = ((raw >> 52) & 0x7ff) as i64;
    if exp_field == 0 {
        // A subnormal double is far below 2**-25: it underflows to zero.
        return Some(sign << 15);
    }
    // frexp, normalised to f in [1, 2).
    let mut e = exp_field - 1023;
    let mut f = f64::from_bits((raw & ((1u64 << 52) - 1)) | (1023u64 << 52));
    if e >= 16 {
        return None;
    } else if e < -25 {
        f = 0.0;
        e = 0;
    } else if e < -14 {
        f *= f64::from_bits(((1023 + 14 + e) as u64) << 52);
        e = 0;
    } else {
        e += 15;
        f -= 1.0;
    }
    f *= 1024.0;
    let mut bits = f as u16;
    let rest = f - bits as f64;
    if rest > 0.5 || (rest == 0.5 && bits % 2 == 1) {
        bits += 1;
        if bits == 1024 {
            bits = 0;
            e += 1;
            if e == 31 {
                return None;
            }
        }
    }
    Some(bits | ((e as u16) << 10) | (sign << 15))
}

/// `_parse_float_bits`: a float literal as IEEE-754 bits of `width`.
fn float_bits(tok: &[u8], width: i64) -> R<u64> {
    let (exp_bits, mant_bits, bias): (i64, i64, i64) = match width {
        16 => (5, 10, 15),
        32 => (8, 23, 127),
        64 => (11, 52, 1023),
        _ => return Err(Refuse),
    };
    let exp_max = (1i64 << exp_bits) - 1;
    if let Some(h) = hex_float(tok) {
        let (negative, ipart, fpart, exp_s) = h;
        if ipart.is_empty() && fpart.is_empty() {
            return Err(Refuse);
        }
        let sign = if negative { 1u64 } else { 0 };
        let sign_bit = sign << (exp_bits + mant_bits);
        let mut sig: u128 = 0;
        for &c in ipart.iter().chain(fpart.iter()) {
            let d = (c as char).to_digit(16).ok_or(Refuse)? as u128;
            if sig >> 120 != 0 {
                return Err(Refuse);
            }
            sig = (sig << 4) | d;
        }
        if sig == 0 {
            return Ok(sign_bit);
        }
        let exp_digits = exp_s.strip_prefix(b"+").or_else(|| exp_s.strip_prefix(b"-")).unwrap_or(exp_s);
        if exp_digits.len() > 15 {
            return Err(Refuse);
        }
        let exp_val = std::str::from_utf8(exp_s).map_err(|_| Refuse)?.parse::<i64>().map_err(|_| Refuse)?;
        let exp = exp_val - 4 * fpart.len() as i64;
        let top = 127 - sig.leading_zeros() as i64;
        let e_biased = exp + top + bias;
        if e_biased == exp_max {
            let frac = sig & ((1u128 << top) - 1);
            let mant = if top <= mant_bits { frac << (mant_bits - top) } else { frac >> (top - mant_bits) };
            return Ok(sign_bit | ((exp_max as u64) << mant_bits) | mant as u64);
        }
        if e_biased > exp_max {
            return Ok(sign_bit | ((exp_max as u64) << mant_bits));
        }
        if 0 < e_biased && e_biased < exp_max && top <= mant_bits {
            let frac = sig & ((1u128 << top) - 1);
            let mant = frac << (mant_bits - top);
            return Ok(sign_bit | ((e_biased as u64) << mant_bits) | mant as u64);
        }
        // Subnormal or excess precision: Python rounds through float.fromhex.
        return Err(Refuse);
    }
    let fv = parse_decimal(tok)?;
    Ok(match width {
        64 => fv.to_bits(),
        // (float)x rounds to nearest even; a finite value too large gives the infinity
        // that the OverflowError branch of `_parse_float_bits` writes.
        32 => (fv as f32).to_bits() as u64,
        _ => match pack_f16(fv) {
            Some(b) => b as u64,
            None => {
                let sign = if tok.first() == Some(&b'-') { 1u64 } else { 0 };
                (sign << 15) | 0x7c00
            }
        },
    })
}

/// `_HEX_FLOAT_RE`: `^([+-]?)0[xX]([0-9a-fA-F]*)(?:\.([0-9a-fA-F]*))?[pP]([+-]?\d+)$`.
fn hex_float(tok: &[u8]) -> Option<(bool, &[u8], &[u8], &[u8])> {
    let mut i = 0;
    let n = tok.len();
    let mut negative = false;
    if i < n && (tok[i] == b'+' || tok[i] == b'-') {
        negative = tok[i] == b'-';
        i += 1;
    }
    if i + 1 >= n || tok[i] != b'0' || (tok[i + 1] != b'x' && tok[i + 1] != b'X') {
        return None;
    }
    i += 2;
    let is = i;
    while i < n && tok[i].is_ascii_hexdigit() {
        i += 1;
    }
    let ipart = &tok[is..i];
    let mut fpart: &[u8] = &[];
    if i < n && tok[i] == b'.' {
        i += 1;
        let fs = i;
        while i < n && tok[i].is_ascii_hexdigit() {
            i += 1;
        }
        fpart = &tok[fs..i];
    }
    if i >= n || (tok[i] != b'p' && tok[i] != b'P') {
        return None;
    }
    i += 1;
    let es = i;
    if i < n && (tok[i] == b'+' || tok[i] == b'-') {
        i += 1;
    }
    let ds = i;
    while i < n && tok[i].is_ascii_digit() {
        i += 1;
    }
    if i == ds || i != n {
        return None;
    }
    Some((negative, ipart, fpart, &tok[es..]))
}

/// `_unquote_string`: the text of a string token, `\<ch>` read as `<ch>`.
fn unquote(tok: &[u8], out: &mut Vec<u8>) -> R<()> {
    if tok.len() < 2 || tok[0] != b'"' || tok[tok.len() - 1] != b'"' {
        return Err(Refuse);
    }
    let raw = &tok[1..tok.len() - 1];
    out.clear();
    let mut i = 0;
    while i < raw.len() {
        if raw[i] == b'\\' && i + 1 < raw.len() {
            out.push(raw[i + 1]);
            i += 2;
        } else {
            out.push(raw[i]);
            i += 1;
        }
    }
    Ok(())
}

fn push_string_words(s: &[u8], out: &mut Vec<u32>) {
    for chunk in s.chunks(4) {
        let mut w = [0u8; 4];
        w[..chunk.len()].copy_from_slice(chunk);
        out.push(u32::from_le_bytes(w));
    }
    if s.len() % 4 == 0 {
        out.push(0);
    }
}

fn is_eol(c: u8) -> bool {
    c == b'\n' || c == b'\r'
}

/// `_tokenize_line` on the line of ASCII `text` that starts at `pos`; returns where the next line starts.
///
/// Lines end at `\n`, `\r\n` or `\r`, as `str.splitlines` ends them. Any other control character but tab
/// refuses, in a token, between tokens or in a comment: Python splits lines and whitespace on more
/// characters (\v, \f, \x1c-\x1e) than this handles.
fn tokenize<'a>(text: &'a [u8], pos: usize, toks: &mut Vec<&'a [u8]>) -> R<usize> {
    toks.clear();
    let n = text.len();
    let mut i = pos;
    while i < n {
        let ch = text[i];
        if ch == b' ' || ch == b'\t' {
            i += 1;
            continue;
        }
        if ch < b' ' {
            if !is_eol(ch) {
                return Err(Refuse);
            }
            break;
        }
        if ch == b';' {
            while i < n && !is_eol(text[i]) {
                if text[i] < b' ' && text[i] != b'\t' {
                    return Err(Refuse);
                }
                i += 1;
            }
            break;
        }
        let mut j = i + 1;
        if ch == b'"' {
            let mut closed = false;
            while j < n {
                let c = text[j];
                if c < b' ' && c != b'\t' {
                    break; // end of line (unterminated) or a control character: both refuse
                }
                // An escaped end of line or control character is left to the check above.
                if c == b'\\' && j + 1 < n && (text[j + 1] >= b' ' || text[j + 1] == b'\t') {
                    j += 2;
                } else if c == b'"' {
                    j += 1;
                    closed = true;
                    break;
                } else {
                    j += 1;
                }
            }
            if !closed {
                return Err(Refuse);
            }
        } else {
            while j < n && text[j] > b' ' && text[j] != b';' {
                j += 1;
            }
        }
        toks.push(&text[i..j]);
        i = j;
    }
    if i + 1 < n && text[i] == b'\r' && text[i + 1] == b'\n' {
        i += 1;
    }
    Ok(i + 1)
}

/// Up to 16 bytes as one little-endian integer. Names never contain a zero byte, so the zero
/// padding keeps two names of different lengths apart.
fn pack(bytes: &[u8]) -> u128 {
    let mut w = [0u8; 16];
    w[..bytes.len()].copy_from_slice(bytes);
    u128::from_le_bytes(w)
}

struct Asm<'t> {
    tables: &'t Tables,
    syms: Map<&'t [u8], u32>,
    short_syms: Map<u128, u32>,
    prefix_keys: Vec<u64>,
    prefix_map: Map<u64, usize>,
    numbered: Vec<Vec<u32>>,
    last_prefix: usize,
    sparse: Map<u64, u32>,
    next_id: u32,
    defined_types: Set<&'t [u8]>,
    type_info: Vec<TypeInfo>,
    id_to_type: Vec<u32>,
    ext_imports: Vec<usize>,
    scratch: Vec<u8>,
    spec_depth: u32,
}

/// Nesting of `OpSpecConstantOp` operands past which the native assembler declines, so the Python
/// assembler (and its own `RecursionError` limit) handles the text; the native recursion stays shallow.
const MAX_SPEC_CONSTANT_OP_DEPTH: u32 = 64;

impl<'t> Asm<'t> {
    /// `get_id`: the number of `%name`, the next free one the first time the name is seen.
    ///
    /// Where a name is kept depends on its spelling alone, so the numbering is the one a single map
    /// would give: `%<prefix><n>`, with a prefix of up to 8 bytes and `<n>` up to five decimal digits
    /// without a leading zero, in a vector per prefix indexed by `n` (`%t123` is slot 123 of `t`) or,
    /// when `n` is far past the end of that vector, in a map by (prefix, `n`);
    /// other names of up to 16 bytes in a map keyed by the bytes as one integer; the rest by slice.
    fn get_id(&mut self, tok: &'t [u8]) -> R<u32> {
        if tok.first() != Some(&b'%') {
            return Err(Refuse);
        }
        let body = &tok[1..];
        let mut k = body.len();
        while k > 0 && body[k - 1].is_ascii_digit() {
            k -= 1;
        }
        let ndigits = body.len() - k;
        if (1..=8).contains(&k) && (1..=5).contains(&ndigits) && (body[k] != b'0' || ndigits == 1) {
            let n = body[k..].iter().fold(0usize, |v, &d| v * 10 + (d - b'0') as usize);
            let prefix = &body[..k];
            let key = pack(prefix) as u64;
            // Most shaders have a handful of prefixes: the last one, a scan, and a map past 16.
            let found = if self.prefix_keys.get(self.last_prefix) == Some(&key) {
                Some(self.last_prefix)
            } else if self.prefix_keys.len() <= 16 {
                self.prefix_keys.iter().position(|&p| p == key)
            } else {
                self.prefix_map.get(&key).copied()
            };
            let slot = match found {
                Some(s) => s,
                None => {
                    if !prefix.iter().all(|&c| c.is_ascii_alphanumeric() || c == b'_') {
                        return Err(Refuse);
                    }
                    self.prefix_map.insert(key, self.prefix_keys.len());
                    self.prefix_keys.push(key);
                    self.numbered.push(Vec::new());
                    self.numbered.len() - 1
                }
            };
            self.last_prefix = slot;
            let ids = &mut self.numbered[slot];
            if let Some(&v) = ids.get(n) {
                if v != 0 {
                    return Ok(v);
                }
            }
            // Numbers far past a prefix's vector (constants named by their bits, %c3fe12345) go to
            // a map instead of growing it; a name is looked up in both, and stored in one.
            let sparse_key = ((slot as u64) << 32) | n as u64;
            if !self.sparse.is_empty() {
                if let Some(&v) = self.sparse.get(&sparse_key) {
                    return Ok(v);
                }
            }
            let v = self.next_id;
            self.next_id = v.checked_add(1).ok_or(Refuse)?;
            let ids = &mut self.numbered[slot];
            if n < ids.len() {
                ids[n] = v;
            } else if n <= 2 * ids.len() + 16 {
                ids.resize(n + 1, 0);
                ids[n] = v;
            } else {
                self.sparse.insert(sparse_key, v);
            }
            return Ok(v);
        }
        if body.len() <= 16 {
            let key = pack(body);
            if let Some(&v) = self.short_syms.get(&key) {
                return Ok(v);
            }
            let v = self.new_id(body)?;
            self.short_syms.insert(key, v);
            return Ok(v);
        }
        if let Some(&v) = self.syms.get(tok) {
            return Ok(v);
        }
        let v = self.new_id(body)?;
        self.syms.insert(tok, v);
        Ok(v)
    }

    fn new_id(&mut self, body: &[u8]) -> R<u32> {
        if body.is_empty() || !body.iter().all(|&c| c.is_ascii_alphanumeric() || c == b'_') {
            return Err(Refuse);
        }
        let v = self.next_id;
        self.next_id = v.checked_add(1).ok_or(Refuse)?;
        Ok(v)
    }

    fn type_info_of(&self, id: Option<u32>) -> TypeInfo {
        id.and_then(|i| self.type_info.get(i as usize).copied()).unwrap_or(TypeInfo::None)
    }

    fn set_slot<T: Copy>(v: &mut Vec<T>, id: u32, val: T, fill: T) {
        let i = id as usize;
        if v.len() <= i {
            v.resize(i + 1, fill);
        }
        v[i] = val;
    }

    fn operands(&mut self, specs: &[(Kind, Quant)], toks: &[&'t [u8]], mut idx: usize, ctx: Ctx,
                out: &mut Vec<u32>) -> R<usize> {
        for &(kind, quant) in specs {
            match quant {
                Quant::Many => {
                    while idx < toks.len() {
                        idx = self.operand(kind, toks, idx, ctx, out)?;
                    }
                }
                Quant::Optional => {
                    if idx < toks.len() {
                        idx = self.operand(kind, toks, idx, ctx, out)?;
                    }
                }
                Quant::One => {
                    if idx >= toks.len() {
                        return Err(Refuse);
                    }
                    idx = self.operand(kind, toks, idx, ctx, out)?;
                }
            }
        }
        Ok(idx)
    }

    fn operand(&mut self, kind: Kind, toks: &[&'t [u8]], idx: usize, ctx: Ctx, out: &mut Vec<u32>) -> R<usize> {
        let tok = *toks.get(idx).ok_or(Refuse)?;
        match kind {
            Kind::Id => {
                out.push(self.get_id(tok)?);
                Ok(idx + 1)
            }
            Kind::LitInt => {
                out.push(u32_literal(tok)?);
                Ok(idx + 1)
            }
            Kind::LitFloat => {
                out.push(float_bits(tok, 32)? as u32);
                Ok(idx + 1)
            }
            Kind::LitString => {
                let mut s = std::mem::take(&mut self.scratch);
                let r = unquote(tok, &mut s);
                if r.is_ok() {
                    push_string_words(&s, out);
                }
                self.scratch = s;
                r?;
                Ok(idx + 1)
            }
            Kind::LitContextNumber => {
                match self.type_info_of(ctx.res_type) {
                    TypeInfo::Int(width, signed) => {
                        if width > 32 && width != 64 {
                            return Err(Refuse);
                        }
                        let v = int_literal(tok, width, signed)?;
                        out.push(v as u32);
                        if width > 32 {
                            out.push((v >> 32) as u32);
                        }
                    }
                    TypeInfo::Float(width) => {
                        let b = float_bits(tok, width)?;
                        out.push(b as u32);
                        if width == 64 {
                            out.push((b >> 32) as u32);
                        }
                    }
                    TypeInfo::None => return Err(Refuse),
                }
                Ok(idx + 1)
            }
            Kind::LitExtInst => {
                let named = ctx.ext.and_then(|e| self.tables.ext_tables[e].get(tok).copied());
                out.push(match named {
                    Some(v) => v,
                    None => u32_literal(tok)?,
                });
                Ok(idx + 1)
            }
            Kind::PairLitId => {
                let (width, signed) = ctx.switch.unwrap_or((32, tok.first() == Some(&b'-')));
                let v = int_literal(tok, width, signed)?;
                out.push(v as u32);
                if width == 64 {
                    out.push((v >> 32) as u32);
                }
                let label = *toks.get(idx + 1).ok_or(Refuse)?;
                out.push(self.get_id(label)?);
                Ok(idx + 2)
            }
            Kind::PairIdId => {
                let second = *toks.get(idx + 1).ok_or(Refuse)?;
                out.push(self.get_id(tok)?);
                out.push(self.get_id(second)?);
                Ok(idx + 2)
            }
            Kind::PairIdLit => {
                let second = *toks.get(idx + 1).ok_or(Refuse)?;
                out.push(self.get_id(tok)?);
                out.push(u32_literal(second)?);
                Ok(idx + 2)
            }
            Kind::SpecConstantOp => {
                let tables = self.tables;
                let sub = match tables.insn(tok) {
                    Some(s) => s,
                    None => {
                        let mut name = b"Op".to_vec();
                        name.extend_from_slice(tok);
                        tables.insn(&name).ok_or(Refuse)?
                    }
                };
                if self.spec_depth >= MAX_SPEC_CONSTANT_OP_DEPTH {
                    return Err(Refuse);
                }
                out.push(sub.opcode);
                self.spec_depth += 1;
                let r = self.operands(&sub.specs, toks, idx + 1, ctx, out);
                self.spec_depth -= 1;
                r
            }
            Kind::Value(e) => {
                let tables = self.tables;
                match tables.values[e].get(tok) {
                    None => {
                        if tok[0].is_ascii_digit() || tok.starts_with(b"0x") || tok.starts_with(b"0X") {
                            out.push(u32_literal(tok)?);
                            Ok(idx + 1)
                        } else {
                            Err(Refuse)
                        }
                    }
                    Some(entry) => {
                        out.push(entry.value);
                        let mut idx = idx + 1;
                        for &pk in &entry.params {
                            if idx >= toks.len() {
                                return Err(Refuse);
                            }
                            idx = self.operand(pk, toks, idx, ctx, out)?;
                        }
                        Ok(idx)
                    }
                }
            }
            Kind::Bits(e) => {
                let tables = self.tables;
                let mut mask = 0u32;
                let mut flag_params: Vec<(u32, &[Kind])> = Vec::new();
                for part in tok.split(|&c| c == b'|') {
                    match tables.bits[e].get(part) {
                        None => {
                            if part.first().is_some_and(|c| c.is_ascii_digit())
                                || part.starts_with(b"0x") || part.starts_with(b"0X") {
                                mask |= u32_literal(part)?;
                            } else {
                                return Err(Refuse);
                            }
                        }
                        Some(entry) => {
                            mask |= entry.value;
                            if !entry.params.is_empty() {
                                flag_params.push((entry.value, &entry.params));
                            }
                        }
                    }
                }
                out.push(mask);
                let mut idx = idx + 1;
                // Stable, like list.sort: parameters in ascending bit-value order.
                flag_params.sort_by_key(|p| p.0);
                for (_, params) in flag_params {
                    for &pk in params {
                        if idx >= toks.len() {
                            return Err(Refuse);
                        }
                        idx = self.operand(pk, toks, idx, ctx, out)?;
                    }
                }
                Ok(idx)
            }
            Kind::Unsupported => Err(Refuse),
        }
    }

    fn line(&mut self, toks: &[&'t [u8]], words: &mut Vec<u32>, out: &mut Vec<u32>) -> R<()> {
        let (res_tok, opname, mut arg_idx) = if toks.len() >= 3 && toks[1] == b"=" {
            (Some(toks[0]), toks[2], 3)
        } else {
            (None, toks[0], 1)
        };
        let tables = self.tables;
        let insn = tables.insn(opname).ok_or(Refuse)?;
        if let (Some(r), true) = (res_tok, insn.type_def) {
            if !self.defined_types.insert(r) {
                return Err(Refuse);
            }
        }
        out.clear();
        let mut res_type = None;
        let mut res_id = None;
        if insn.has_rt {
            let t = *toks.get(arg_idx).ok_or(Refuse)?;
            let id = self.get_id(t)?;
            res_type = Some(id);
            out.push(id);
            arg_idx += 1;
        }
        if insn.has_res {
            let id = self.get_id(res_tok.ok_or(Refuse)?)?;
            res_id = Some(id);
            out.push(id);
            if let Some(rt) = res_type {
                Self::set_slot(&mut self.id_to_type, id, rt, 0);
            }
        } else if res_tok.is_some() {
            return Err(Refuse);
        }

        if let Some(rid) = res_id {
            match opname {
                b"OpTypeInt" if arg_idx + 1 < toks.len() => {
                    let w = i64::try_from(parse_int(toks[arg_idx])?).map_err(|_| Refuse)?;
                    let s = parse_int(toks[arg_idx + 1])? != 0;
                    Self::set_slot(&mut self.type_info, rid, TypeInfo::Int(w, s), TypeInfo::None);
                }
                b"OpTypeFloat" if arg_idx < toks.len() => {
                    let w = i64::try_from(parse_int(toks[arg_idx])?).map_err(|_| Refuse)?;
                    Self::set_slot(&mut self.type_info, rid, TypeInfo::Float(w), TypeInfo::None);
                }
                b"OpExtInstImport" if arg_idx < toks.len() => {
                    let mut s = std::mem::take(&mut self.scratch);
                    let r = unquote(toks[arg_idx], &mut s);
                    let set = tables.ext_sets.get(&s[..]).copied();
                    self.scratch = s;
                    r?;
                    if let Some(set) = set {
                        Self::set_slot(&mut self.ext_imports, rid, set + 1, 0);
                    }
                }
                _ => {}
            }
        }

        let mut ctx = Ctx { res_type, ext: None, switch: None };
        if opname == b"OpExtInst" && arg_idx < toks.len() {
            let set = self.get_id(toks[arg_idx])?;
            ctx.ext = self.ext_imports.get(set as usize).copied().filter(|&e| e != 0).map(|e| e - 1);
        } else if opname == b"OpSwitch" && arg_idx < toks.len() {
            let sel = self.get_id(toks[arg_idx])?;
            let ty = self.id_to_type.get(sel as usize).copied().unwrap_or(0);
            if ty == 0 {
                return Err(Refuse);
            }
            if let TypeInfo::Int(w, s) = self.type_info_of(Some(ty)) {
                if w <= 32 || w == 64 {
                    ctx.switch = Some((w, s));
                }
            }
        }

        let end = self.operands(&insn.specs, toks, arg_idx, ctx, out)?;
        if end != toks.len() {
            return Err(Refuse);
        }
        let word_count = 1 + out.len();
        if word_count > 0xFFFF {
            return Err(Refuse);
        }
        words.push(((word_count as u32) << 16) | insn.opcode);
        words.extend_from_slice(out);
        Ok(())
    }
}

/// `assemble(text)` with `version` as the header version word, or `Err` to leave it to Python.
fn assemble_into(tables: &Tables, text: &[u8], version: u32, words: &mut Vec<u32>) -> R<()> {
    // Python also splits lines on \x85, U+2028 and U+2029: only ASCII text is handled here, and
    // `tokenize` refuses the ASCII control characters it does not split the same way.
    if !text.is_ascii() {
        return Err(Refuse);
    }
    words.clear();
    words.extend_from_slice(&[0x0723_0203, version, 0, 0, 0]);
    let mut asm = Asm {
        tables,
        syms: Map::default(),
        short_syms: Map::default(),
        prefix_keys: Vec::new(),
        prefix_map: Map::default(),
        numbered: Vec::new(),
        last_prefix: 0,
        sparse: Map::default(),
        next_id: 1,
        defined_types: Set::default(),
        type_info: Vec::new(),
        id_to_type: Vec::new(),
        ext_imports: Vec::new(),
        scratch: Vec::new(),
        spec_depth: 0,
    };
    asm.short_syms.reserve(text.len() / 128);
    let mut toks: Vec<&[u8]> = Vec::with_capacity(16);
    let mut out: Vec<u32> = Vec::with_capacity(16);
    let mut pos = 0;
    while pos < text.len() {
        pos = tokenize(text, pos, &mut toks)?;
        if !toks.is_empty() {
            asm.line(&toks, words, &mut out)?;
        }
    }
    words[3] = asm.next_id;
    Ok(())
}

/// Parses the table text from `spirv_asm.py` (UTF-8, `len` bytes). Null when it does not parse.
///
/// # Safety
/// `text` must be valid for `len` bytes.
#[no_mangle]
pub unsafe extern "C" fn spirv_asm_tables_new(text: *const u8, len: usize) -> *mut Tables {
    if text.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: the caller passes `len` readable bytes.
    let bytes = unsafe { std::slice::from_raw_parts(text, len) };
    let parsed = std::panic::catch_unwind(|| std::str::from_utf8(bytes).ok().and_then(Tables::parse));
    match parsed {
        Ok(Some(t)) => Box::into_raw(Box::new(t)),
        _ => std::ptr::null_mut(),
    }
}

/// Frees tables from `spirv_asm_tables_new`.
///
/// # Safety
/// `tables` must come from `spirv_asm_tables_new` and not be used afterwards.
#[no_mangle]
pub unsafe extern "C" fn spirv_asm_tables_free(tables: *mut Tables) {
    if !tables.is_null() {
        // SAFETY: the pointer came from Box::into_raw.
        drop(unsafe { Box::from_raw(tables) });
    }
}

thread_local! {
    static OUTPUT: RefCell<Vec<u32>> = const { RefCell::new(Vec::new()) };
}

/// Assembles `len` bytes of text. Returns 0 and sets `*out`/`*out_len` to the little-endian
/// module bytes (valid until the next call on the same thread), or 1 when the Python assembler
/// should handle the text (any error or unsupported spelling), or -1 on bad arguments.
///
/// # Safety
/// `tables` must come from `spirv_asm_tables_new`; `text` must be valid for `len` bytes; `out`
/// and `out_len` must be writable.
#[no_mangle]
pub unsafe extern "C" fn spirv_asm_assemble(tables: *const Tables, text: *const u8, len: usize, version: u32,
                                            out: *mut *const u8, out_len: *mut usize) -> i32 {
    if tables.is_null() || (text.is_null() && len != 0) || out.is_null() || out_len.is_null() {
        return -1;
    }
    // SAFETY: checked non-null above; the caller guarantees validity.
    let tables = unsafe { &*tables };
    let bytes: &[u8] = if len == 0 { &[] } else { unsafe { std::slice::from_raw_parts(text, len) } };
    let result = std::panic::catch_unwind(|| {
        OUTPUT.with(|cell| {
            let mut words = cell.borrow_mut();
            if assemble_into(tables, bytes, version, &mut words).is_err() {
                return None;
            }
            if cfg!(target_endian = "big") {
                for w in words.iter_mut() {
                    *w = w.swap_bytes();
                }
            }
            Some((words.as_ptr() as *const u8, words.len() * 4))
        })
    });
    match result {
        Ok(Some((ptr, n))) => {
            // SAFETY: checked non-null above.
            unsafe {
                *out = ptr;
                *out_len = n;
            }
            0
        }
        _ => 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TABLES: &str = "\
I OpCapability 17 0 0 Capability
I OpExtInstImport 11 0 1 LiteralString
I OpExtInst 12 1 1 IdRef LiteralExtInstInteger IdRef*
I OpMemoryModel 14 0 0 AddressingModel MemoryModel
I OpTypeVoid 19 0 1
I OpTypeInt 21 0 1 LiteralInteger LiteralInteger
I OpTypeFloat 22 0 1 LiteralInteger FPEncoding?
I OpConstant 43 1 1 LiteralContextDependentNumber
I OpIAdd 128 1 1 IdRef IdRef
I OpLoad 61 1 1 IdRef MemoryAccess?
I OpName 5 0 0 IdRef LiteralString
I OpSwitch 251 0 0 IdRef IdRef PairLiteralIntegerIdRef*
I OpSpecConstantOp 52 1 1 LiteralSpecConstantOpInteger
I OpExecutionMode 16 0 0 IdRef ExecutionMode
V Capability Shader 1
V AddressingModel Logical 0
V MemoryModel GLSL450 1
V FPEncoding BFloat16KHR 0
V ExecutionMode LocalSize 17 LiteralInteger LiteralInteger LiteralInteger
B MemoryAccess None 0
B MemoryAccess Volatile 1
B MemoryAccess Aligned 2 LiteralInteger
B MemoryAccess MakePointerAvailable 8 IdScope
X GLSL.std.450 FAbs 4
";

    fn tables() -> Tables {
        Tables::parse(TABLES).expect("test tables parse")
    }

    fn asm(text: &str) -> Option<Vec<u32>> {
        let mut words = Vec::new();
        assemble_into(&tables(), text.as_bytes(), 0x0001_0500, &mut words).ok().map(|_| words)
    }

    #[test]
    fn header_and_ids() {
        let w = asm("OpCapability Shader\n%int = OpTypeInt 32 1\n%a = OpConstant %int -2\n%b = OpIAdd %int %a %a\n")
            .unwrap();
        assert_eq!(&w[..5], &[0x0723_0203, 0x0001_0500, 0, 4, 0]);
        assert_eq!(&w[5..7], &[(2 << 16) | 17, 1]);
        assert_eq!(&w[7..11], &[(4 << 16) | 21, 1, 32, 1]);
        assert_eq!(&w[11..15], &[(4 << 16) | 43, 1, 2, 0xFFFF_FFFE]);
        assert_eq!(&w[15..], &[(5 << 16) | 128, 1, 3, 2, 2]);
    }

    #[test]
    fn empty_text_is_a_header() {
        assert_eq!(asm("").unwrap(), vec![0x0723_0203, 0x0001_0500, 0, 1, 0]);
        assert_eq!(asm("  ; only a comment\r\n\r\n").unwrap().len(), 5);
    }

    #[test]
    fn strings_and_escapes() {
        let w = asm("%x = OpTypeVoid\nOpName %x \"a\\\"b;c\" ; tail\n").unwrap();
        // "a\"b;c" reads as a"b;c: 5 bytes plus NUL, two words.
        assert_eq!(&w[7..], &[(4 << 16) | 5, 1, u32::from_le_bytes(*b"a\"b;"), u32::from_le_bytes(*b"c\0\0\0")]);
        let w = asm("%x = OpTypeVoid\nOpName %x \"abcd\"\n").unwrap();
        assert_eq!(&w[9..], &[u32::from_le_bytes(*b"abcd"), 0]);
        assert_eq!(asm("OpName %x \"abc\n"), None);
    }

    #[test]
    fn masks_sort_parameters_by_bit() {
        let w = asm("%u = OpTypeInt 32 0\n%p = OpTypeVoid\n%s = OpConstant %u 1\n\
                     %v = OpLoad %u %p MakePointerAvailable|Aligned 4 %s\n").unwrap();
        let load = &w[w.len() - 7..];
        assert_eq!(load, &[(7 << 16) | 61, 1, 4, 2, 10, 4, 3]);
        assert_eq!(asm("%u = OpTypeInt 32 0\n%v = OpLoad %u %p Bogus\n"), None);
    }

    #[test]
    fn enum_parameters_and_ext_inst() {
        let w = asm("OpExecutionMode %main LocalSize 8 4 1\n").unwrap();
        assert_eq!(&w[5..], &[(6 << 16) | 16, 1, 17, 8, 4, 1]);
        let w = asm("%g = OpExtInstImport \"GLSL.std.450\"\n%f = OpTypeFloat 32\n%r = OpExtInst %f %g FAbs %r\n")
            .unwrap();
        assert_eq!(&w[w.len() - 6..], &[(6 << 16) | 12, 2, 3, 1, 4, 3]);
    }

    #[test]
    fn switch_literal_width_follows_the_selector() {
        let w = asm("%l = OpTypeInt 64 0\n%s = OpConstant %l 5\nOpSwitch %s %d 0x100000000 %c\n").unwrap();
        assert_eq!(&w[w.len() - 6..], &[(6 << 16) | 251, 2, 3, 0, 1, 4]);
        assert_eq!(asm("OpSwitch %s %d 1 %c\n"), None);
    }

    #[test]
    fn spec_constant_op_names() {
        let w = asm("%u = OpTypeInt 32 0\n%a = OpConstant %u 1\n%b = OpSpecConstantOp %u IAdd %a %a\n").unwrap();
        assert_eq!(&w[w.len() - 6..], &[(6 << 16) | 52, 1, 3, 128, 2, 2]);
    }

    #[test]
    fn deeply_nested_spec_constant_op_is_declined() {
        let nested = |n: usize| format!("%u = OpTypeInt 32 0\n%a = OpConstant %u 1\n%b = OpSpecConstantOp %u {}IAdd %a %a\n", "SpecConstantOp ".repeat(n));
        assert!(asm(&nested(10)).is_some());
        assert_eq!(asm(&nested(100)), None);
        assert_eq!(asm(&nested(100_000)), None);
    }

    #[test]
    fn refusals() {
        for bad in [
            "%t = OpTypeVoid\n%t = OpTypeVoid\n",
            "OpBogus\n",
            "%u = OpTypeInt 32 0\n%c = OpConstant %u -1\n",
            "%u = OpTypeInt 32 0\n%c = OpConstant %u 1_0\n",
            "%u = OpTypeInt 32 0\n%c = OpConstant %u 010\n",
            "%u = OpTypeInt 16 1\n%c = OpConstant %u 0x10000\n",
            "%f = OpTypeFloat 32\n%c = OpConstant %f inf\n",
            "%f = OpTypeFloat 32\n%c = OpConstant %f 0x1.000001p-140\n",
            "OpCapability Shader extra\n",
            "OpName %bad-id \"x\"\n",
            "OpName %x \"caf\u{e9}\"\n",
            "OpCapability\x0bShader\n",
        ] {
            assert_eq!(asm(bad), None, "{bad:?}");
        }
    }

    #[test]
    fn numbered_and_sparse_names_get_one_id_each() {
        // %t5000 is past the end of the t vector when first seen (sparse map), then the vector grows
        // over its slot; both spellings must keep the ids of their first use.
        let mut text = String::from("%u = OpTypeInt 32 0
");
        let names = ["%t5000", "%t1", "%t01", "%t2", "%t5000", "%x", "%t3", "%t0", "%t1"];
        for n in names {
            text += &format!("OpName {n} \"\"
");
        }
        for i in 4..6000 {
            text += &format!("OpName %t{i} \"\"
");
        }
        let w = asm(&text).unwrap();
        let ids: Vec<u32> = w[9..].chunks(3).map(|c| c[1]).collect();
        assert_eq!(&ids[..9], &[2, 3, 4, 5, 2, 6, 7, 8, 3]);
        assert_eq!(ids[9 + 4996], 2); // %t5000 again, now inside the vector's range
        assert_eq!(w[3], 6000 - 4 + 8);
    }

    #[test]
    fn integer_literals() {
        assert_eq!(int_literal(b"0xFFFF", 16, true), Ok(0xFFFF_FFFF));
        assert_eq!(int_literal(b"0x8000", 16, true), Ok(0xFFFF_8000));
        assert_eq!(int_literal(b"+0xFFFF", 16, true), Err(Refuse));
        assert_eq!(int_literal(b"-0", 32, false), Err(Refuse));
        assert_eq!(int_literal(b"-0x8000000000000000", 64, true), Ok(0x8000_0000_0000_0000));
        assert_eq!(int_literal(b"0b101", 32, false), Ok(5));
        assert_eq!(int_literal(b"000", 32, false), Ok(0));
        assert_eq!(int_literal(b"4294967296", 32, false), Err(Refuse));
    }

    #[test]
    fn float_literals() {
        assert_eq!(float_bits(b"1.5", 32), Ok(0x3FC0_0000));
        assert_eq!(float_bits(b"-0.0", 16), Ok(0x8000));
        assert_eq!(float_bits(b"65504", 16), Ok(0x7BFF));
        assert_eq!(float_bits(b"65520", 16), Ok(0x7C00));
        assert_eq!(float_bits(b"-1e10", 16), Ok(0xFC00));
        assert_eq!(float_bits(b"5.960464477539063e-08", 16), Ok(0x0001));
        assert_eq!(float_bits(b"2.98023223876953125e-08", 16), Ok(0x0000));
        assert_eq!(float_bits(b"1e39", 32), Ok(0x7F80_0000));
        assert_eq!(float_bits(b"0x1p+128", 32), Ok(0x7F80_0000));
        assert_eq!(float_bits(b"-0x1.8p+128", 32), Ok(0xFFC0_0000));
        assert_eq!(float_bits(b"0x1.8p+1", 64), Ok(0x4008_0000_0000_0000));
        assert_eq!(float_bits(b"0x.8p1", 16), Ok(0x3C00));
        assert_eq!(float_bits(b"0x0p0", 32), Ok(0));
        assert_eq!(float_bits(b"0x.p0", 32), Err(Refuse));
        assert_eq!(float_bits(b"1e", 32), Err(Refuse));
    }

    #[test]
    fn ffi_rejects_bad_arguments() {
        let t = tables();
        let mut ptr = std::ptr::null();
        let mut n = 0usize;
        // SAFETY: every pointer is valid or null as tested.
        unsafe {
            assert_eq!(spirv_asm_assemble(std::ptr::null(), b"".as_ptr(), 0, 0, &mut ptr, &mut n), -1);
            assert_eq!(spirv_asm_assemble(&t, std::ptr::null(), 4, 0, &mut ptr, &mut n), -1);
            assert_eq!(spirv_asm_assemble(&t, b"OpBogus".as_ptr(), 7, 0, &mut ptr, &mut n), 1);
            assert_eq!(spirv_asm_assemble(&t, b"".as_ptr(), 0, 7, &mut ptr, &mut n), 0);
            assert_eq!(n, 20);
            assert_eq!(*ptr.add(4), 7);
            assert!(spirv_asm_tables_new(b"Q x".as_ptr(), 3).is_null());
            let blob = TABLES.as_bytes();
            let owned = spirv_asm_tables_new(blob.as_ptr(), blob.len());
            assert!(!owned.is_null());
            spirv_asm_tables_free(owned);
        }
    }
}
