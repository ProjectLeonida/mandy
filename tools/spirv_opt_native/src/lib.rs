//! Native port of `optimize()` in `tools/spirv_opt.py`.
//!
//! Every pass, table and iteration order follows the Python code line for line, so the output text is the same.
//! The port never reports an error: whenever the Python code would raise, loop forever, depend on set iteration
//! order, or read a spelling the port does not reproduce, it declines (`Decline`) and the Python code runs instead.

/// The operands of a `Parsed` or an `Inst`, which live in one flat arena.
macro_rules! sl {
    ($ar:expr, $e:expr) => {
        &$ar[$e.a0 as usize..($e.a0 + $e.an) as usize]
    };
}

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};

// ---------------------------------------------------------------------------------------------------------------
// Small fast hasher (the keys are interned integers and short integer vectors).

#[derive(Default, Clone, Copy)]
struct Fx(u64);

impl Fx {
    #[inline]
    fn add(&mut self, w: u64) {
        self.0 = (self.0.rotate_left(5) ^ w).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
    }
}

impl Hasher for Fx {
    #[inline]
    fn finish(&self) -> u64 {
        // fmix64: hashbrown picks buckets from the low bits, which a bare product leaves to the low input bytes.
        let mut h = self.0;
        h ^= h >> 33;
        h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
        h ^= h >> 33;
        h = h.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
        h ^ (h >> 33)
    }
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let mut chunks = bytes.chunks_exact(8);
        for c in &mut chunks {
            let mut a = [0u8; 8];
            a.copy_from_slice(c);
            self.add(u64::from_le_bytes(a));
        }
        let rest = chunks.remainder();
        if !rest.is_empty() {
            let mut a = [0u8; 8];
            a[..rest.len()].copy_from_slice(rest);
            self.add(u64::from_le_bytes(a) ^ ((rest.len() as u64) << 56));
        }
    }
    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.add(i as u64)
    }
    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.add(i as u64)
    }
    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.add(i)
    }
    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.add(i as u64)
    }
}

type Map<K, V> = HashMap<K, V, BuildHasherDefault<Fx>>;
type Set<K> = HashSet<K, BuildHasherDefault<Fx>>;

/// The Python code would raise, hang or give an order-dependent result, or the input has a spelling the port
/// leaves to Python.
#[derive(Debug, PartialEq, Eq)]
pub struct Decline;

type R<T> = Result<T, Decline>;

// ---------------------------------------------------------------------------------------------------------------
// Token kinds: every string the Python code compares a token against.

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum K {
    Other,
    Eq,
    Function,
    PushConstant,
    Input,
    OpLabel,
    OpFunction,
    OpFunctionEnd,
    OpVariable,
    OpAccessChain,
    OpLoad,
    OpStore,
    OpBranch,
    OpBranchConditional,
    OpSwitch,
    OpPhi,
    OpCopyObject,
    OpBitcast,
    OpCompositeConstruct,
    OpCompositeExtract,
    OpSelect,
    OpLogicalNot,
    OpLogicalAnd,
    OpLogicalOr,
    OpIEqual,
    OpINotEqual,
    OpExtInst,
    OpNot,
    OpUConvert,
    OpBitFieldUExtract,
    OpIAdd,
    OpISub,
    OpIMul,
    OpUDiv,
    OpUMod,
    OpBitwiseAnd,
    OpBitwiseOr,
    OpBitwiseXor,
    OpShiftLeftLogical,
    OpShiftRightLogical,
    OpShiftRightArithmetic,
    OpULessThan,
    OpULessThanEqual,
    OpUGreaterThan,
    OpUGreaterThanEqual,
    OpSLessThan,
    OpSLessThanEqual,
    OpSGreaterThan,
    OpSGreaterThanEqual,
    OpDecorate,
    OpDecorateId,
    OpDecorateString,
    OpDecorateStringGOOGLE,
    OpGroupDecorate,
    OpName,
    OpTypeInt,
    OpTypeArray,
    OpConstant,
    OpConstantTrue,
    OpConstantFalse,
    OpConstantComposite,
    OpConstantNull,
}

fn kind_of(s: &[u8]) -> K {
    match s {
        b"=" => K::Eq,
        b"Function" => K::Function,
        b"PushConstant" => K::PushConstant,
        b"Input" => K::Input,
        b"OpLabel" => K::OpLabel,
        b"OpFunction" => K::OpFunction,
        b"OpFunctionEnd" => K::OpFunctionEnd,
        b"OpVariable" => K::OpVariable,
        b"OpAccessChain" => K::OpAccessChain,
        b"OpLoad" => K::OpLoad,
        b"OpStore" => K::OpStore,
        b"OpBranch" => K::OpBranch,
        b"OpBranchConditional" => K::OpBranchConditional,
        b"OpSwitch" => K::OpSwitch,
        b"OpPhi" => K::OpPhi,
        b"OpCopyObject" => K::OpCopyObject,
        b"OpBitcast" => K::OpBitcast,
        b"OpCompositeConstruct" => K::OpCompositeConstruct,
        b"OpCompositeExtract" => K::OpCompositeExtract,
        b"OpSelect" => K::OpSelect,
        b"OpLogicalNot" => K::OpLogicalNot,
        b"OpLogicalAnd" => K::OpLogicalAnd,
        b"OpLogicalOr" => K::OpLogicalOr,
        b"OpIEqual" => K::OpIEqual,
        b"OpINotEqual" => K::OpINotEqual,
        b"OpExtInst" => K::OpExtInst,
        b"OpNot" => K::OpNot,
        b"OpUConvert" => K::OpUConvert,
        b"OpBitFieldUExtract" => K::OpBitFieldUExtract,
        b"OpIAdd" => K::OpIAdd,
        b"OpISub" => K::OpISub,
        b"OpIMul" => K::OpIMul,
        b"OpUDiv" => K::OpUDiv,
        b"OpUMod" => K::OpUMod,
        b"OpBitwiseAnd" => K::OpBitwiseAnd,
        b"OpBitwiseOr" => K::OpBitwiseOr,
        b"OpBitwiseXor" => K::OpBitwiseXor,
        b"OpShiftLeftLogical" => K::OpShiftLeftLogical,
        b"OpShiftRightLogical" => K::OpShiftRightLogical,
        b"OpShiftRightArithmetic" => K::OpShiftRightArithmetic,
        b"OpULessThan" => K::OpULessThan,
        b"OpULessThanEqual" => K::OpULessThanEqual,
        b"OpUGreaterThan" => K::OpUGreaterThan,
        b"OpUGreaterThanEqual" => K::OpUGreaterThanEqual,
        b"OpSLessThan" => K::OpSLessThan,
        b"OpSLessThanEqual" => K::OpSLessThanEqual,
        b"OpSGreaterThan" => K::OpSGreaterThan,
        b"OpSGreaterThanEqual" => K::OpSGreaterThanEqual,
        b"OpDecorate" => K::OpDecorate,
        b"OpDecorateId" => K::OpDecorateId,
        b"OpDecorateString" => K::OpDecorateString,
        b"OpDecorateStringGOOGLE" => K::OpDecorateStringGOOGLE,
        b"OpGroupDecorate" => K::OpGroupDecorate,
        b"OpName" => K::OpName,
        b"OpTypeInt" => K::OpTypeInt,
        b"OpTypeArray" => K::OpTypeArray,
        b"OpConstant" => K::OpConstant,
        b"OpConstantTrue" => K::OpConstantTrue,
        b"OpConstantFalse" => K::OpConstantFalse,
        b"OpConstantComposite" => K::OpConstantComposite,
        b"OpConstantNull" => K::OpConstantNull,
        _ => K::Other,
    }
}

/// `_PURE_OPS` in spirv_opt.py.
fn is_pure_op(s: &[u8]) -> bool {
    matches!(
        s,
        b"OpAccessChain" | b"OpCopyObject" | b"OpSelect" | b"OpBitcast" | b"OpIAdd" | b"OpISub" | b"OpIMul" |
        b"OpUDiv" | b"OpSDiv" | b"OpUMod" | b"OpSMod" | b"OpSRem" | b"OpFAdd" | b"OpFSub" | b"OpFMul" | b"OpFDiv" |
        b"OpFNegate" | b"OpFmaKHR" | b"OpBitwiseOr" | b"OpBitwiseAnd" | b"OpBitwiseXor" | b"OpNot" |
        b"OpShiftLeftLogical" | b"OpShiftRightLogical" | b"OpShiftRightArithmetic" | b"OpBitFieldUExtract" |
        b"OpBitFieldSExtract" | b"OpBitFieldInsert" | b"OpBitReverse" | b"OpBitCount" | b"OpIEqual" |
        b"OpINotEqual" | b"OpUGreaterThan" | b"OpUGreaterThanEqual" | b"OpULessThan" | b"OpULessThanEqual" |
        b"OpSGreaterThan" | b"OpSGreaterThanEqual" | b"OpSLessThan" | b"OpSLessThanEqual" | b"OpFOrdEqual" |
        b"OpFOrdNotEqual" | b"OpFOrdGreaterThan" | b"OpFOrdGreaterThanEqual" | b"OpFOrdLessThan" |
        b"OpFOrdLessThanEqual" | b"OpFUnordEqual" | b"OpFUnordNotEqual" | b"OpFUnordGreaterThan" |
        b"OpFUnordGreaterThanEqual" | b"OpFUnordLessThan" | b"OpFUnordLessThanEqual" | b"OpIsNan" | b"OpIsInf" |
        b"OpLogicalAnd" | b"OpLogicalOr" | b"OpLogicalNot" | b"OpLogicalEqual" | b"OpLogicalNotEqual" |
        b"OpCompositeConstruct" | b"OpCompositeExtract" | b"OpCompositeInsert" | b"OpVectorShuffle" |
        b"OpVectorExtractDynamic" | b"OpVectorTimesScalar" | b"OpConvertFToU" | b"OpConvertFToS" | b"OpConvertSToF" |
        b"OpConvertUToF" | b"OpUConvert" | b"OpSConvert" | b"OpFConvert" | b"OpQuantizeToF16" | b"OpIAddCarry" |
        b"OpISubBorrow" | b"OpUMulExtended" | b"OpSMulExtended"
    )
}

/// `_COMMUTATIVE_OPS` in spirv_opt.py.
fn is_comm_op(s: &[u8]) -> bool {
    matches!(
        s,
        b"OpIAdd" | b"OpIMul" | b"OpBitwiseOr" | b"OpBitwiseAnd" | b"OpBitwiseXor" | b"OpIEqual" | b"OpINotEqual" |
        b"OpLogicalAnd" | b"OpLogicalOr" | b"OpLogicalEqual" | b"OpLogicalNotEqual"
    )
}

/// `_PURE_GLSL_EXT` in spirv_opt.py.
fn is_pure_glsl(s: &[u8]) -> bool {
    matches!(
        s,
        b"FAbs" | b"SAbs" | b"FSign" | b"SSign" | b"Floor" | b"Trunc" | b"Round" | b"RoundEven" | b"Ceil" |
        b"Fract" | b"Sin" | b"Cos" | b"Tan" | b"Asin" | b"Acos" | b"Atan" | b"Sinh" | b"Cosh" | b"Tanh" | b"Pow" |
        b"Exp" | b"Log" | b"Exp2" | b"Log2" | b"Sqrt" | b"InverseSqrt" | b"FMin" | b"UMin" | b"SMin" | b"FMax" |
        b"UMax" | b"SMax" | b"FClamp" | b"UClamp" | b"SClamp" | b"NMin" | b"NMax" | b"NClamp" | b"FMix" | b"Step" |
        b"SmoothStep" | b"Fma" | b"PackHalf2x16" | b"UnpackHalf2x16" | b"PackSnorm4x8" | b"PackUnorm4x8" |
        b"PackSnorm2x16" | b"PackUnorm2x16" | b"UnpackSnorm4x8" | b"UnpackUnorm4x8" | b"UnpackSnorm2x16" |
        b"UnpackUnorm2x16" | b"FindILsb" | b"FindSMsb" | b"FindUMsb" | b"Ldexp" | b"FrexpStruct" | b"ModfStruct"
    )
}

// ---------------------------------------------------------------------------------------------------------------
// Interned tokens.

type Tok = u32;

struct TokInfo<'a> {
    s: &'a [u8],
    kind: K,
    pct: bool,       // t.startswith("%")
    volatile: bool,  // "Volatile" in t
    pure_op: bool,   // t in _PURE_OPS
    comm: bool,      // t in _COMMUTATIVE_OPS
    glsl_pure: bool, // t in _PURE_GLSL_EXT
}

struct Interner<'a> {
    table: Vec<u32>, // open addressing: token + 1, or 0 for empty
    hashes: Vec<u64>,
    info: Vec<TokInfo<'a>>,
}

#[inline]
fn hash_bytes(s: &[u8]) -> u64 {
    let mut h = (s.len() as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    let mut chunks = s.chunks_exact(8);
    for c in &mut chunks {
        let w = u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]);
        h = (h.rotate_left(23) ^ w).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
    }
    let mut w = 0u64;
    for (i, &b) in chunks.remainder().iter().enumerate() {
        w |= (b as u64) << (8 * i);
    }
    h = (h.rotate_left(23) ^ w).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
    h ^= h >> 32;
    h = h.wrapping_mul(0xd6e8_feb8_6659_fd93);
    h ^ (h >> 32)
}

impl<'a> Interner<'a> {
    fn new() -> Self {
        Interner {
            table: vec![0; 1024],
            hashes: Vec::new(),
            info: Vec::new(),
        }
    }

    #[inline]
    fn find(&self, s: &[u8], h: u64) -> Result<Tok, usize> {
        let mask = self.table.len() - 1;
        let mut i = h as usize & mask;
        loop {
            let e = self.table[i];
            if e == 0 {
                return Err(i);
            }
            let t = e - 1;
            if self.hashes[t as usize] == h && self.info[t as usize].s == s {
                return Ok(t);
            }
            i = (i + 1) & mask;
        }
    }

    fn grow(&mut self) {
        let size = self.table.len() * 2;
        let mut table = vec![0u32; size];
        let mask = size - 1;
        for (t, &h) in self.hashes.iter().enumerate() {
            let mut i = h as usize & mask;
            while table[i] != 0 {
                i = (i + 1) & mask;
            }
            table[i] = t as u32 + 1;
        }
        self.table = table;
    }

    fn intern(&mut self, s: &'a [u8]) -> Tok {
        let h = hash_bytes(s);
        let slot = match self.find(s, h) {
            Ok(t) => return t,
            Err(slot) => slot,
        };
        let t = self.info.len() as Tok;
        let pct = s.first() == Some(&b'%');
        let info = if pct {
            // No keyword starts with '%'.
            TokInfo { s, kind: K::Other, pct, volatile: contains(s, b"Volatile"), pure_op: false, comm: false,
                      glsl_pure: false }
        } else {
            TokInfo {
                s,
                kind: kind_of(s),
                pct,
                volatile: contains(s, b"Volatile"),
                pure_op: is_pure_op(s),
                comm: is_comm_op(s),
                glsl_pure: is_pure_glsl(s),
            }
        };
        self.info.push(info);
        self.hashes.push(h);
        self.table[slot] = t + 1;
        if self.info.len() * 2 > self.table.len() {
            self.grow();
        }
        t
    }

    #[inline]
    fn get(&self, s: &[u8]) -> Option<Tok> {
        self.find(s, hash_bytes(s)).ok()
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Python string and integer helpers (ASCII only; the caller has refused anything else).

#[inline]
fn is_ws(b: u8) -> bool {
    b == b' ' || b == b'\t'
}

/// `s.split()` for a line that holds no line breaks.
fn split_ws(s: &[u8]) -> impl Iterator<Item = &[u8]> + '_ {
    s.split(|&b| is_ws(b)).filter(|t| !t.is_empty())
}

fn lstrip(s: &[u8]) -> &[u8] {
    let n = s.iter().take_while(|&&b| is_ws(b)).count();
    &s[n..]
}

fn strip(s: &[u8]) -> &[u8] {
    let s = lstrip(s);
    let n = s.iter().rev().take_while(|&&b| is_ws(b)).count();
    &s[..s.len() - n]
}

fn contains(hay: &[u8], needle: &[u8]) -> bool {
    let Some((&first, rest)) = needle.split_first() else { return true };
    if hay.len() < needle.len() {
        return false;
    }
    let last = hay.len() - needle.len();
    let mut i = 0usize;
    while i <= last {
        match hay[i..=last].iter().position(|&b| b == first) {
            None => return false,
            Some(p) => {
                i += p;
                if &hay[i + 1..i + needle.len()] == rest {
                    return true;
                }
                i += 1;
            }
        }
    }
    false
}

/// Above CPython's int/str digit limit (4300 by default, configurable) `int()` raises; leave those to Python.
const MAX_INT_TOKEN: usize = 1000;

/// `int(s, 0)` reduced modulo 2**64, or None where Python raises ValueError.
fn py_int_base0(s: &[u8]) -> R<Option<u64>> {
    if s.len() > MAX_INT_TOKEN {
        return Err(Decline);
    }
    let (neg, body) = match s.first() {
        Some(b'+') => (false, &s[1..]),
        Some(b'-') => (true, &s[1..]),
        _ => (false, s),
    };
    if body.is_empty() {
        return Ok(None);
    }
    let (base, digits, prefixed) = if body.len() >= 2 && body[0] == b'0' {
        match body[1] {
            b'x' | b'X' => (16u64, &body[2..], true),
            b'o' | b'O' => (8, &body[2..], true),
            b'b' | b'B' => (2, &body[2..], true),
            _ => (10, body, false),
        }
    } else {
        (10, body, false)
    };
    let mut val: u64 = 0;
    let mut ndig = 0usize;
    let mut prev_us = false;
    let mut first_digit: Option<u8> = None;
    let mut all_zero = true;
    for (i, &c) in digits.iter().enumerate() {
        if c == b'_' {
            if prev_us || (i == 0 && !prefixed) {
                return Ok(None);
            }
            prev_us = true;
            continue;
        }
        let d = match c {
            b'0'..=b'9' => (c - b'0') as u64,
            b'a'..=b'f' => (c - b'a' + 10) as u64,
            b'A'..=b'F' => (c - b'A' + 10) as u64,
            _ => return Ok(None),
        };
        if d >= base {
            return Ok(None);
        }
        prev_us = false;
        if first_digit.is_none() {
            first_digit = Some(c);
        }
        if d != 0 {
            all_zero = false;
        }
        ndig += 1;
        val = val.wrapping_mul(base).wrapping_add(d);
    }
    if ndig == 0 || prev_us {
        return Ok(None);
    }
    if !prefixed && first_digit == Some(b'0') && !all_zero {
        return Ok(None); // int("010", 0) raises
    }
    Ok(Some(if neg { val.wrapping_neg() } else { val }))
}

/// `int(s)` (base 10): None where Python raises ValueError, else the value clamped to the i128 range.
fn py_int_base10(s: &[u8]) -> R<Option<i128>> {
    if s.len() > MAX_INT_TOKEN {
        return Err(Decline);
    }
    let (neg, body) = match s.first() {
        Some(b'+') => (false, &s[1..]),
        Some(b'-') => (true, &s[1..]),
        _ => (false, s),
    };
    let mut val: i128 = 0;
    let mut ndig = 0usize;
    let mut prev_us = false;
    for (i, &c) in body.iter().enumerate() {
        if c == b'_' {
            if prev_us || i == 0 {
                return Ok(None);
            }
            prev_us = true;
            continue;
        }
        if !c.is_ascii_digit() {
            return Ok(None);
        }
        prev_us = false;
        ndig += 1;
        val = val.saturating_mul(10).saturating_add((c - b'0') as i128);
    }
    if ndig == 0 || prev_us {
        return Ok(None);
    }
    Ok(Some(if neg { -val } else { val }))
}

#[inline]
fn mask(w: u32) -> u64 {
    if w >= 64 {
        u64::MAX
    } else {
        (1u64 << w) - 1
    }
}

#[inline]
fn to_signed(uval: u64, width: u32) -> i128 {
    let u = uval as i128;
    if width >= 128 {
        return u;
    }
    let sign_bit = 1i128 << (width - 1);
    if u >= sign_bit {
        u - (1i128 << width)
    } else {
        u
    }
}

/// A map keyed by interned tokens, stored as a flat array (the tokens are dense indices).
#[derive(Clone)]
struct Dense<T: Copy> {
    v: Vec<Option<T>>,
    n: usize,
    keys: Vec<Tok>,
}

impl<T: Copy> Dense<T> {
    fn new(size: usize) -> Self {
        Dense { v: vec![None; size], n: 0, keys: Vec::new() }
    }
    fn from_map(size: usize, m: &Map<Tok, T>) -> Self {
        let mut d = Dense::new(size);
        for (&k, &x) in m {
            d.insert(k, x);
        }
        d
    }
    #[inline]
    fn get(&self, k: &Tok) -> Option<&T> {
        self.v[*k as usize].as_ref()
    }
    #[inline]
    fn contains_key(&self, k: &Tok) -> bool {
        self.v[*k as usize].is_some()
    }
    #[inline]
    fn insert(&mut self, k: Tok, x: T) {
        let slot = &mut self.v[k as usize];
        if slot.is_none() {
            self.n += 1;
            self.keys.push(k);
        }
        *slot = Some(x);
    }
    #[inline]
    fn remove(&mut self, k: &Tok) -> Option<T> {
        let old = self.v[*k as usize].take();
        if old.is_some() {
            self.n -= 1;
        }
        old
    }
    #[inline]
    fn len(&self) -> usize {
        self.n
    }
    #[inline]
    fn is_empty(&self) -> bool {
        self.n == 0
    }
}

impl Dense<i64> {
    #[inline]
    fn add(&mut self, k: Tok, d: i64) {
        match self.v[k as usize].as_mut() {
            Some(x) => *x += d,
            None => self.insert(k, d),
        }
    }
}

/// A set of interned tokens as a flat array, remembering insertion order for iteration.
#[derive(Clone)]
struct DSet {
    v: Vec<bool>,
    items: Vec<Tok>,
}

impl DSet {
    fn new(size: usize) -> Self {
        DSet { v: vec![false; size], items: Vec::new() }
    }
    fn from_set(size: usize, s: &Set<Tok>) -> Self {
        let mut d = DSet::new(size);
        for &k in s {
            d.insert(k);
        }
        d
    }
    #[inline]
    fn contains(&self, k: &Tok) -> bool {
        self.v[*k as usize]
    }
    #[inline]
    fn insert(&mut self, k: Tok) {
        if !self.v[k as usize] {
            self.v[k as usize] = true;
            self.items.push(k);
        }
    }
    #[inline]
    fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
    fn iter(&self) -> std::slice::Iter<'_, Tok> {
        self.items.iter()
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Module-level facts (the header scan of optimize()).

#[derive(Clone)]
struct Parsed {
    res: Option<Tok>,
    op: Tok,
    a0: u32,
    an: u32,
}

struct Module<'a> {
    it: Interner<'a>,
    // module header facts
    const_ids: DSet,
    const_true: DSet,
    const_false: DSet,
    const_true_id: Option<Tok>,
    const_false_id: Option<Tok>,
    const_type: Dense<Tok>,
    int_type_info: Dense<u32>,
    const_int_val: Dense<(Tok, u64, u32)>,
    int_const_by_val: Map<(Tok, u64), Tok>,
    const_null_elem_zero: Map<Tok, Tok>,
    readonly_global_vars: DSet,
    glsl_sets: DSet,
    decorated_ids: DSet,
}

impl<'a> Module<'a> {
    #[inline]
    fn k(&self, t: Tok) -> K {
        self.it.info[t as usize].kind
    }
    #[inline]
    fn pct(&self, t: Tok) -> bool {
        self.it.info[t as usize].pct
    }
    #[inline]
    fn vol(&self, t: Tok) -> bool {
        self.it.info[t as usize].volatile
    }
    #[inline]
    fn s(&self, t: Tok) -> &[u8] {
        &self.it.info[t as usize].s
    }
}

/// `_parse_line`.
fn parse_line<'a>(it: &mut Interner<'a>, ar: &mut Vec<Tok>, line: &'a [u8]) -> Option<Parsed> {
    if line.contains(&b'"') {
        return None;
    }
    let code = match line.iter().position(|&b| b == b';') {
        Some(p) => &line[..p],
        None => line,
    };
    let mut toks = split_ws(code);
    let t0 = toks.next()?;
    let t1 = toks.next();
    let a0 = ar.len();
    if t1 == Some(&b"="[..]) {
        if let Some(t2) = toks.next() {
            let res = it.intern(t0);
            let op = it.intern(t2);
            ar.extend(toks.map(|t| it.intern(t)));
            return Some(Parsed { res: Some(res), op, a0: a0 as u32, an: (ar.len() - a0) as u32 });
        }
    }
    let op = it.intern(t0);
    if let Some(t1) = t1 {
        ar.push(it.intern(t1));
        ar.extend(toks.map(|t| it.intern(t)));
    }
    Some(Parsed { res: None, op, a0: a0 as u32, an: (ar.len() - a0) as u32 })
}

/// `str.splitlines()` for ASCII text without \x0b, \x0c, \x1c-\x1e (refused by the caller).
fn split_lines(text: &[u8]) -> Vec<&[u8]> {
    let mut out = Vec::new();
    let mut start = 0usize;
    let mut i = 0usize;
    let n = text.len();
    while i < n {
        let c = text[i];
        if c == b'\n' || c == b'\r' {
            out.push(&text[start..i]);
            if c == b'\r' && i + 1 < n && text[i + 1] == b'\n' {
                i += 1;
            }
            start = i + 1;
        }
        i += 1;
    }
    if start < n {
        out.push(&text[start..]);
    }
    out
}

// ---------------------------------------------------------------------------------------------------------------
// One function.

#[derive(Clone)]
struct Inst {
    res: Option<Tok>,
    op: Option<Tok>,
    a0: u32,
    an: u32,
    raw: usize, // line index in the module
}

type Slot = u32;

/// Interned slot tuples `(var_id, *const_indices)`; element 0 is the root token.
#[derive(Default)]
struct Slots {
    map: Map<Vec<u64>, Slot>,
    root: Vec<Tok>,
    len: Vec<usize>,
    data: Vec<Vec<u64>>,
}

impl Slots {
    fn intern(&mut self, v: Vec<u64>) -> Slot {
        if let Some(&s) = self.map.get(&v) {
            return s;
        }
        let s = self.root.len() as Slot;
        self.root.push(v[0] as Tok);
        self.len.push(v.len());
        self.data.push(v.clone());
        self.map.insert(v, s);
        s
    }
}

struct Cfg {
    blocks: Vec<Vec<usize>>,
    all_preds: Vec<Vec<usize>>, // a set: sorted, deduplicated
    rpo: Vec<usize>,
    idom: Vec<Option<usize>>,
    dom_children: Vec<Vec<usize>>,
}

/// `_build_cfg_and_dom`.
fn build_cfg(m: &Module<'_>, insts: &[Inst], ar: &[Tok]) -> Cfg {
    let mut blocks: Vec<Vec<usize>> = Vec::new();
    let mut label_to_block: Map<Tok, usize> = Map::default();
    let mut cur: Option<usize> = None;
    for (idx, e) in insts.iter().enumerate() {
        let op = e.op.map(|o| m.k(o));
        if op == Some(K::OpLabel) && e.res.is_some() {
            let b = blocks.len();
            label_to_block.insert(e.res.unwrap(), b);
            blocks.push(vec![idx]);
            cur = Some(b);
        } else if let Some(cb) = cur {
            if op == Some(K::OpFunctionEnd) {
                cur = None;
            } else {
                blocks[cb].push(idx);
            }
        }
    }
    let n = blocks.len();
    if n == 0 {
        return Cfg { blocks, all_preds: vec![], rpo: vec![], idom: vec![], dom_children: vec![] };
    }
    let mut succs: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut all_preds: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (b, b_insts) in blocks.iter().enumerate() {
        let mut seen: Vec<usize> = Vec::new();
        for &idx in b_insts {
            let e = &insts[idx];
            let Some(op) = e.op else { continue };
            let a = sl!(ar, e);
            let mut targets: Vec<Tok> = Vec::new();
            match m.k(op) {
                K::OpBranch if !a.is_empty() => targets.push(a[0]),
                K::OpBranchConditional if a.len() >= 3 => {
                    targets.push(a[1]);
                    targets.push(a[2]);
                }
                K::OpSwitch if a.len() >= 2 => {
                    targets.push(a[1]);
                    let mut k = 3;
                    while k < a.len() {
                        targets.push(a[k]);
                        k += 2;
                    }
                }
                _ => {}
            }
            for t in targets {
                if let Some(&tb) = label_to_block.get(&t) {
                    if !all_preds[tb].contains(&b) {
                        all_preds[tb].push(b);
                    }
                    if !seen.contains(&tb) {
                        seen.push(tb);
                        succs[b].push(tb);
                    }
                }
            }
        }
    }
    for p in all_preds.iter_mut() {
        p.sort_unstable();
    }

    let mut visited = vec![false; n];
    let mut postorder: Vec<usize> = Vec::new();
    let mut stack: Vec<(usize, usize)> = vec![(0, 0)];
    visited[0] = true;
    while let Some(&(b, si)) = stack.last() {
        if si < succs[b].len() {
            stack.last_mut().unwrap().1 = si + 1;
            let nb = succs[b][si];
            if !visited[nb] {
                visited[nb] = true;
                stack.push((nb, 0));
            }
        } else {
            stack.pop();
            postorder.push(b);
        }
    }
    let rpo: Vec<usize> = postorder.into_iter().rev().collect();
    let mut rpo_num = vec![usize::MAX; n];
    for (i, &b) in rpo.iter().enumerate() {
        rpo_num[b] = i;
    }
    let mut idom: Vec<Option<usize>> = vec![None; n];
    idom[0] = Some(0);
    let intersect = |idom: &Vec<Option<usize>>, mut b1: usize, mut b2: usize| -> usize {
        while b1 != b2 {
            while rpo_num[b1] > rpo_num[b2] {
                b1 = idom[b1].unwrap();
            }
            while rpo_num[b2] > rpo_num[b1] {
                b2 = idom[b2].unwrap();
            }
        }
        b1
    };
    let mut changed = true;
    while changed {
        changed = false;
        for &b in &rpo[1..] {
            let mut new_idom: Option<usize> = None;
            for &p in &all_preds[b] {
                if idom[p].is_some() {
                    new_idom = Some(match new_idom {
                        None => p,
                        Some(x) => intersect(&idom, p, x),
                    });
                }
            }
            if new_idom.is_some() && idom[b] != new_idom {
                idom[b] = new_idom;
                changed = true;
            }
        }
    }
    let mut dom_children: Vec<Vec<usize>> = vec![Vec::new(); n];
    for &b in &rpo[1..] {
        if let Some(p) = idom[b] {
            if p != b {
                dom_children[p].push(b);
            }
        }
    }
    Cfg { blocks, all_preds, rpo, idom, dom_children }
}

#[derive(Default)]
struct Undo {
    ac: Vec<Vec<Tok>>,
    val: Vec<Vec<Tok>>,
    ro: Vec<(Tok, Tok)>,
    bc: Vec<Tok>,
    bm: Vec<Tok>,
    ce: Vec<Tok>,
    ln: Vec<Tok>,
}

/// Per-pass state of `_optimize_function`.
struct Pass<'a, 'm> {
    m: &'a Module<'m>,
    slots: &'a Slots,
    ptr_root: Dense<Tok>,
    ptr_slot: Dense<Slot>,
    id_type: Dense<Tok>,
    trackable: DSet,
    static_null_zero: Map<Tok, Tok>,
    stored_slots: Set<Slot>,
    def_block: Dense<usize>,
    idom: Vec<Option<usize>>,
    ct: Option<Tok>,
    cf: Option<Tok>,
    alias: Dense<Tok>,
    dead_idx: Vec<bool>,
    changed: bool,
    ac_cse: Map<Vec<Tok>, Tok>,
    val_cse: Map<Vec<Tok>, Tok>,
    ro_load_cse: Map<(Tok, Tok), Tok>,
    bitcast_of: Dense<(Tok, Tok)>,
    bool_mask_of: Dense<(Tok, bool)>,
    ctor_elems: Map<Tok, Vec<Tok>>,
    logical_not_of: Dense<Tok>,
}

/// `resolve()` with its path compression; a cycle (where Python loops forever) declines.
fn resolve(alias: &mut Dense<Tok>, tok: Tok) -> R<Tok> {
    let mut cur = tok;
    let mut steps = 0usize;
    while let Some(&nx) = alias.get(&cur) {
        cur = nx;
        steps += 1;
        if steps > alias.len() {
            return Err(Decline);
        }
    }
    if cur != tok {
        alias.insert(tok, cur);
    }
    Ok(cur)
}

/// Python `del d[k]`: a missing key raises KeyError.
fn del<K2: std::hash::Hash + Eq, V>(map: &mut Map<K2, V>, k: &K2) -> R<()> {
    match map.remove(k) {
        Some(_) => Ok(()),
        None => Err(Decline),
    }
}

impl<'a, 'm> Pass<'a, 'm> {
    #[inline]
    fn id_type_get(&self, t: Tok) -> Option<Tok> {
        match self.id_type.get(&t) {
            Some(&x) => Some(x),
            None => self.m.const_type.get(&t).copied(),
        }
    }

    fn strictly_dominates(&self, a: usize, b: usize) -> bool {
        if a == b || self.idom[b].is_none() {
            return false;
        }
        let mut cur = self.idom[b].unwrap();
        while cur != a {
            match self.idom[cur] {
                None => return false,
                Some(nx) if nx == cur => return false,
                Some(nx) => cur = nx,
            }
        }
        true
    }

    fn kill(&mut self, res: Tok, to: Tok, idx: usize, deleted: &mut Set<Tok>) {
        self.alias.insert(res, to);
        self.dead_idx[idx] = true;
        deleted.insert(res);
        self.changed = true;
    }

    /// `_fold_int_op`.
    fn fold_int_op(&self, op: K, res_ty: Tok, args: &[Tok]) -> Option<Tok> {
        let m = self.m;
        let civ = &m.const_int_val;
        let by = |ty: Tok, v: u64| m.int_const_by_val.get(&(ty, v)).copied();
        let n = args.len();
        if op == K::OpNot && n == 2 {
            if let Some(&(_, v1, w)) = civ.get(&args[1]) {
                return by(res_ty, !v1 & mask(w));
            }
            return None;
        }
        if op == K::OpUConvert && n == 2 {
            let c1 = civ.get(&args[1]);
            let dst = m.int_type_info.get(&res_ty);
            if let (Some(&(_, v1, _)), Some(&dw)) = (c1, dst) {
                return by(res_ty, v1 & mask(dw));
            }
            return None;
        }
        if op == K::OpBitFieldUExtract && n == 4 {
            let (base, off_id, cnt_id) = (args[1], args[2], args[3]);
            let co = civ.get(&off_id);
            let cc = civ.get(&cnt_id);
            let dst = m.int_type_info.get(&res_ty);
            if let (Some(&(_, off, _)), Some(&(_, cnt, _)), Some(&w)) = (co, cc, dst) {
                let w = w as u64;
                if off == 0 && cnt == w {
                    return Some(base);
                }
                if let Some(&(_, bv, _)) = civ.get(&base) {
                    if off <= w && cnt <= w && off + cnt <= w {
                        let val = if cnt == 0 { 0 } else { ((bv as u128 >> off) as u64) & mask(cnt as u32) };
                        return by(res_ty, val);
                    }
                }
            }
            return None;
        }
        if n != 3 {
            return None;
        }
        let (a1, a2) = (args[1], args[2]);
        let c1 = civ.get(&a1).copied();
        let c2 = civ.get(&a2).copied();
        if let (Some((_, v1, w1)), Some((_, v2, w2))) = (c1, c2) {
            let mk = mask(w1);
            match op {
                K::OpIAdd => return by(res_ty, v1.wrapping_add(v2) & mk),
                K::OpISub => return by(res_ty, v1.wrapping_sub(v2) & mk),
                K::OpIMul => return by(res_ty, v1.wrapping_mul(v2) & mk),
                K::OpUDiv if v2 != 0 => return by(res_ty, (v1 / v2) & mk),
                K::OpUMod if v2 != 0 => return by(res_ty, (v1 % v2) & mk),
                K::OpBitwiseAnd => return by(res_ty, v1 & v2 & mk),
                K::OpBitwiseOr => return by(res_ty, (v1 | v2) & mk),
                K::OpBitwiseXor => return by(res_ty, (v1 ^ v2) & mk),
                K::OpShiftLeftLogical if v2 < w1 as u64 => return by(res_ty, (v1 << v2) & mk),
                K::OpShiftRightLogical if v2 < w1 as u64 => return by(res_ty, (v1 >> v2) & mk),
                K::OpShiftRightArithmetic if v2 < w1 as u64 => {
                    let sv = to_signed(v1, w1) >> v2;
                    return by(res_ty, ((sv as u128) & (mk as u128)) as u64);
                }
                _ => {}
            }
            if let (Some(t), Some(f)) = (self.ct, self.cf) {
                if w1 == w2 {
                    let b = |c: bool| Some(if c { t } else { f });
                    match op {
                        K::OpIEqual => return b(v1 == v2),
                        K::OpINotEqual => return b(v1 != v2),
                        K::OpULessThan => return b(v1 < v2),
                        K::OpULessThanEqual => return b(v1 <= v2),
                        K::OpUGreaterThan => return b(v1 > v2),
                        K::OpUGreaterThanEqual => return b(v1 >= v2),
                        _ => {}
                    }
                    let (s1, s2) = (to_signed(v1, w1), to_signed(v2, w2));
                    match op {
                        K::OpSLessThan => return b(s1 < s2),
                        K::OpSLessThanEqual => return b(s1 <= s2),
                        K::OpSGreaterThan => return b(s1 > s2),
                        K::OpSGreaterThanEqual => return b(s1 >= s2),
                        _ => {}
                    }
                }
            }
            return None;
        }
        let dst_mask = m.int_type_info.get(&res_ty).map(|&w| mask(w));
        let v1 = c1.map(|c| c.1);
        let v2 = c2.map(|c| c.1);
        match op {
            K::OpIAdd => {
                if v2 == Some(0) {
                    return Some(a1);
                }
                if v1 == Some(0) {
                    return Some(a2);
                }
            }
            K::OpISub => {
                if v2 == Some(0) {
                    return Some(a1);
                }
                if a1 == a2 {
                    return by(res_ty, 0);
                }
            }
            K::OpIMul => {
                if let Some(v) = v2 {
                    if v == 1 {
                        return Some(a1);
                    }
                    if v == 0 {
                        return Some(a2);
                    }
                }
                if let Some(v) = v1 {
                    if v == 1 {
                        return Some(a2);
                    }
                    if v == 0 {
                        return Some(a1);
                    }
                }
            }
            K::OpBitwiseAnd => {
                if a1 == a2 {
                    return Some(a1);
                }
                if let Some(v) = v2 {
                    if v == 0 {
                        return Some(a2);
                    }
                    if dst_mask == Some(v) {
                        return Some(a1);
                    }
                }
                if let Some(v) = v1 {
                    if v == 0 {
                        return Some(a1);
                    }
                    if dst_mask == Some(v) {
                        return Some(a2);
                    }
                }
            }
            K::OpBitwiseOr => {
                if a1 == a2 {
                    return Some(a1);
                }
                if let Some(v) = v2 {
                    if v == 0 {
                        return Some(a1);
                    }
                    if dst_mask == Some(v) {
                        return Some(a2);
                    }
                }
                if let Some(v) = v1 {
                    if v == 0 {
                        return Some(a2);
                    }
                    if dst_mask == Some(v) {
                        return Some(a1);
                    }
                }
            }
            K::OpBitwiseXor => {
                if a1 == a2 {
                    return by(res_ty, 0);
                }
                if v2 == Some(0) {
                    return Some(a1);
                }
                if v1 == Some(0) {
                    return Some(a2);
                }
            }
            K::OpShiftLeftLogical | K::OpShiftRightLogical | K::OpShiftRightArithmetic => {
                if v2 == Some(0) {
                    return Some(a1);
                }
                if op != K::OpShiftRightArithmetic && v1 == Some(0) {
                    return Some(a1);
                }
            }
            K::OpIEqual if a1 == a2 => return self.ct,
            K::OpINotEqual if a1 == a2 => return self.cf,
            _ => {}
        }
        None
    }

    #[inline]
    fn slot_len(&self, s: Slot) -> usize {
        self.slots.len[s as usize]
    }

    #[allow(clippy::too_many_arguments)]
    fn process_block(&mut self, insts: &[Inst], ar: &mut [Tok], block: &[usize], b: usize, slot_val: &mut Map<Slot, Tok>,
                     entry_null_zero: &mut Map<Tok, Tok>, reachable: bool, deleted: &mut Set<Tok>) -> R<Undo> {
        let m = self.m;
        let mut last_store: Map<Slot, usize> = Map::default();
        let mut u = Undo::default();

        for &idx in block {
            let Some(op_t) = insts[idx].op else { continue };
            let op = m.k(op_t);
            if op == K::OpLabel {
                continue;
            }
            let res = insts[idx].res;

            let (a0, an) = (insts[idx].a0 as usize, insts[idx].an as usize);
            if !self.alias.is_empty() && an > 0 {
                let mut diff = false;
                for t in ar[a0..a0 + an].iter_mut() {
                    if m.pct(*t) {
                        let r = resolve(&mut self.alias, *t)?;
                        if r != *t {
                            *t = r;
                            diff = true;
                        }
                    }
                }
                if diff {
                    self.changed = true;
                }
            }
            let args: &[Tok] = &ar[a0..a0 + an];
            let n = args.len();

            // Trivial OpPhi.
            if op == K::OpPhi && reachable && n >= 3 {
                if let Some(r) = res {
                    if !m.decorated_ids.contains(&r) {
                        let mut non_self: Vec<Tok> = Vec::new();
                        let mut k = 1;
                        while k < n {
                            let v = args[k];
                            if v != r && !non_self.contains(&v) {
                                non_self.push(v);
                            }
                            k += 2;
                        }
                        if non_self.len() == 1 {
                            let cand = non_self[0];
                            let ok = match self.def_block.get(&cand) {
                                None => true,
                                Some(&cb) => self.strictly_dominates(cb, b),
                            };
                            if ok {
                                self.kill(r, cand, idx, deleted);
                                continue;
                            }
                        }
                    }
                }
            }

            let res_ok = match res {
                Some(r) => !m.decorated_ids.contains(&r),
                None => false,
            };
            if res_ok {
                let r = res.unwrap();
                if op == K::OpCopyObject && n == 2 {
                    self.kill(r, args[1], idx, deleted);
                    continue;
                }
                if op == K::OpBitcast && n == 2 {
                    let (dst_ty, src) = (args[0], args[1]);
                    if let Some(&(psrc, pty)) = self.bitcast_of.get(&src) {
                        if pty == dst_ty {
                            self.kill(r, psrc, idx, deleted);
                            continue;
                        }
                    }
                    if let Some(src_ty) = self.id_type_get(src) {
                        self.bitcast_of.insert(r, (src, src_ty));
                        u.bc.push(r);
                    }
                }
                if op == K::OpCompositeConstruct && n >= 2 {
                    self.ctor_elems.insert(r, args[1..].to_vec());
                    u.ce.push(r);
                }
                if op == K::OpCompositeExtract && n == 3 {
                    let ks = m.s(args[2]);
                    if !ks.is_empty() && ks.iter().all(|c| c.is_ascii_digit()) {
                        if let Some(elems) = self.ctor_elems.get(&args[1]) {
                            if ks.len() > MAX_INT_TOKEN {
                                return Err(Decline);
                            }
                            let mut k: u64 = 0;
                            for &c in ks {
                                k = k.saturating_mul(10).saturating_add((c - b'0') as u64);
                            }
                            if k < elems.len() as u64 && elems.iter().all(|&e| self.id_type_get(e) == Some(args[0])) {
                                let to = elems[k as usize];
                                self.kill(r, to, idx, deleted);
                                continue;
                            }
                        }
                    }
                }
                if op == K::OpSelect && n == 4 {
                    let (cond, x, y) = (args[1], args[2], args[3]);
                    if m.const_true.contains(&cond) || x == y {
                        self.kill(r, x, idx, deleted);
                        continue;
                    }
                    if m.const_false.contains(&cond) {
                        self.kill(r, y, idx, deleted);
                        continue;
                    }
                    if m.const_true.contains(&x) && m.const_false.contains(&y) {
                        self.kill(r, cond, idx, deleted);
                        continue;
                    }
                    if let (Some(cx), Some(cy)) = (m.const_int_val.get(&x), m.const_int_val.get(&y)) {
                        if cx.1 != 0 && cy.1 == 0 {
                            self.bool_mask_of.insert(r, (cond, true));
                            u.bm.push(r);
                        } else if cx.1 == 0 && cy.1 != 0 {
                            self.bool_mask_of.insert(r, (cond, false));
                            u.bm.push(r);
                        }
                    }
                }
                if op == K::OpLogicalNot && n == 2 {
                    let inner = args[1];
                    if m.const_true.contains(&inner) {
                        if let Some(f) = self.cf {
                            self.kill(r, f, idx, deleted);
                            continue;
                        }
                    }
                    if m.const_false.contains(&inner) {
                        if let Some(t) = self.ct {
                            self.kill(r, t, idx, deleted);
                            continue;
                        }
                    }
                    if let Some(&pn) = self.logical_not_of.get(&inner) {
                        self.kill(r, pn, idx, deleted);
                        continue;
                    }
                    self.logical_not_of.insert(r, inner);
                    u.ln.push(r);
                }
                if op == K::OpLogicalAnd && n == 3 {
                    let (a1, a2) = (args[1], args[2]);
                    if a1 == a2 || m.const_true.contains(&a2) {
                        self.kill(r, a1, idx, deleted);
                        continue;
                    }
                    if m.const_true.contains(&a1) {
                        self.kill(r, a2, idx, deleted);
                        continue;
                    }
                    if m.const_false.contains(&a1) {
                        self.kill(r, a1, idx, deleted);
                        continue;
                    }
                    if m.const_false.contains(&a2) {
                        self.kill(r, a2, idx, deleted);
                        continue;
                    }
                }
                if op == K::OpLogicalOr && n == 3 {
                    let (a1, a2) = (args[1], args[2]);
                    if a1 == a2 || m.const_false.contains(&a2) {
                        self.kill(r, a1, idx, deleted);
                        continue;
                    }
                    if m.const_false.contains(&a1) {
                        self.kill(r, a2, idx, deleted);
                        continue;
                    }
                    if m.const_true.contains(&a1) {
                        self.kill(r, a1, idx, deleted);
                        continue;
                    }
                    if m.const_true.contains(&a2) {
                        self.kill(r, a2, idx, deleted);
                        continue;
                    }
                }
                if (op == K::OpINotEqual || op == K::OpIEqual) && n == 3 {
                    let (a1, a2) = (args[1], args[2]);
                    for (cand, other) in [(a1, a2), (a2, a1)] {
                        let bm = self.bool_mask_of.get(&cand).copied();
                        let co = m.const_int_val.get(&other);
                        if let (Some((cond_id, normal)), Some(co)) = (bm, co) {
                            if co.1 == 0
                                && ((op == K::OpINotEqual && normal) || (op == K::OpIEqual && !normal))
                            {
                                self.kill(r, cond_id, idx, deleted);
                                break;
                            }
                        }
                    }
                    if self.dead_idx[idx] {
                        continue;
                    }
                }
                // `_fold_int_op(op, args[0], ...)` raises IndexError on an instruction with no operands.
                if args.is_empty() {
                    return Err(Decline);
                }
                if let Some(f) = self.fold_int_op(op, args[0], args) {
                    self.kill(r, f, idx, deleted);
                    continue;
                }
            }

            // Dominator-scoped OpAccessChain CSE.
            if op == K::OpAccessChain && res_ok {
                let r = res.unwrap();
                if let Some(&prev) = self.ac_cse.get(args) {
                    self.kill(r, prev, idx, deleted);
                    continue;
                }
                self.ac_cse.insert(args.to_vec(), r);
                u.ac.push(args.to_vec());
                continue;
            }

            // Dominator-scoped pure-value CSE.
            let info = &m.it.info[op_t as usize];
            if res_ok
                && (info.pure_op
                    || (op == K::OpExtInst
                        && n >= 3
                        && m.glsl_sets.contains(&args[1])
                        && m.it.info[args[2] as usize].glsl_pure))
            {
                let r = res.unwrap();
                let mut key: Vec<Tok> = Vec::with_capacity(n + 1);
                key.push(op_t);
                if info.comm && n == 3 && m.s(args[1]) > m.s(args[2]) {
                    key.extend_from_slice(&[args[0], args[2], args[1]]);
                } else {
                    key.extend_from_slice(args);
                }
                if let Some(&prev) = self.val_cse.get(&key) {
                    self.kill(r, prev, idx, deleted);
                    continue;
                }
                self.val_cse.insert(key.clone(), r);
                u.val.push(key);
                continue;
            }

            // OpLoad forwarding & CSE.
            if op == K::OpLoad && res.is_some() && n >= 2 {
                let r = res.unwrap();
                let ptr = args[1];
                let volatile = args[2..].iter().any(|&t| m.vol(t));
                let root = self.ptr_root.get(&ptr).copied();
                let tracked = root.is_some_and(|x| self.trackable.contains(&x));
                if !volatile && tracked {
                    let root = root.unwrap();
                    if let Some(&slot) = self.ptr_slot.get(&ptr) {
                        let mut cached = slot_val.get(&slot).copied();
                        if cached.is_none() && self.slot_len(slot) == 2 {
                            if let Some(&z) = entry_null_zero.get(&root) {
                                cached = Some(z);
                            } else if let Some(&z) = self.static_null_zero.get(&root) {
                                if !self.stored_slots.contains(&slot) {
                                    cached = Some(z);
                                }
                            }
                        }
                        if let Some(c) = cached {
                            if !m.decorated_ids.contains(&r) {
                                self.kill(r, c, idx, deleted);
                                continue;
                            }
                        }
                        slot_val.insert(slot, r);
                        last_store.remove(&slot);
                    } else {
                        let slots = self.slots;
                        last_store.retain(|&k, _| slots.root[k as usize] != root);
                    }
                } else if !volatile
                    && (root.is_some_and(|x| m.readonly_global_vars.contains(&x))
                        || m.readonly_global_vars.contains(&ptr))
                    && !m.decorated_ids.contains(&r)
                {
                    let key = (args[0], ptr);
                    if let Some(&prev) = self.ro_load_cse.get(&key) {
                        self.kill(r, prev, idx, deleted);
                        continue;
                    }
                    self.ro_load_cse.insert(key, r);
                    u.ro.push(key);
                }
                continue;
            }

            // Function-scope OpStore tracking.
            if op == K::OpStore && n >= 2 {
                let (ptr, val) = (args[0], args[1]);
                let volatile = args[2..].iter().any(|&t| m.vol(t));
                let root = self.ptr_root.get(&ptr).copied();
                let tracked = root.is_some_and(|x| self.trackable.contains(&x));
                if !volatile && tracked {
                    let root = root.unwrap();
                    if let Some(&slot) = self.ptr_slot.get(&ptr) {
                        let mut cur_val = slot_val.get(&slot).copied();
                        if cur_val.is_none() && self.slot_len(slot) == 2 {
                            if let Some(&z) = entry_null_zero.get(&root) {
                                cur_val = Some(z);
                            }
                        }
                        if cur_val == Some(val) {
                            self.dead_idx[idx] = true;
                            self.changed = true;
                            continue;
                        }
                        if let Some(&prev) = last_store.get(&slot) {
                            self.dead_idx[prev] = true;
                            self.changed = true;
                        }
                        slot_val.insert(slot, val);
                        last_store.insert(slot, idx);
                    } else {
                        entry_null_zero.remove(&root);
                        let slots = self.slots;
                        slot_val.retain(|&k, _| slots.root[k as usize] != root);
                        last_store.retain(|&k, _| slots.root[k as usize] != root);
                    }
                }
                continue;
            }
        }
        Ok(u)
    }

    fn undo(&mut self, u: Undo) -> R<()> {
        for k in &u.ac {
            del(&mut self.ac_cse, k)?;
        }
        for k in &u.val {
            del(&mut self.val_cse, k)?;
        }
        for k in &u.ro {
            del(&mut self.ro_load_cse, k)?;
        }
        for k in &u.bc {
            self.bitcast_of.remove(k).ok_or(Decline)?;
        }
        for k in &u.bm {
            self.bool_mask_of.remove(k).ok_or(Decline)?;
        }
        for k in &u.ce {
            del(&mut self.ctor_elems, k)?;
        }
        for k in &u.ln {
            self.logical_not_of.remove(k).ok_or(Decline)?;
        }
        Ok(())
    }
}

enum Frame {
    Enter(usize, Map<Slot, Tok>, Map<Tok, Tok>),
    Exit(Undo),
}

/// `_optimize_function`: returns the optimized instructions.
fn optimize_function(m: &Module<'_>, parsed: &[Option<Parsed>], pargs: &[Tok], start: usize, end: usize, max_passes: i64,
                     deleted: &mut Set<Tok>) -> R<(Vec<Inst>, Vec<Tok>)> {
    let mut ar: Vec<Tok> = Vec::new();
    let mut insts: Vec<Inst> = (start..end)
        .map(|i| match &parsed[i] {
            None => Inst { res: None, op: None, a0: 0, an: 0, raw: i },
            Some(p) => {
                let a0 = ar.len() as u32;
                ar.extend_from_slice(sl!(pargs, p));
                Inst { res: p.res, op: Some(p.op), a0, an: p.an, raw: i }
            }
        })
        .collect();
    let ct = m.const_true_id;
    let cf = m.const_false_id;
    let nt = m.it.info.len();
    let mut slots = Slots::default();

    let mut pass = 0i64;
    while pass < max_passes {
        pass += 1;

        // 1. Function-scope variables, pointer slots, escapes, dynamic indices, result types, stores.
        let mut func_vars: Dense<Option<Tok>> = Dense::new(nt);
        let mut ptr_root: Dense<Tok> = Dense::new(nt);
        let mut ptr_slot: Dense<Slot> = Dense::new(nt);
        let mut escaped: DSet = DSet::new(nt);
        let mut dyn_indexed: DSet = DSet::new(nt);
        let mut id_type: Dense<Tok> = Dense::new(nt);

        for e in &insts {
            let Some(op_t) = e.op else { continue };
            let op = m.k(op_t);
            let a = sl!(ar, e);
            if let Some(r) = e.res {
                if !a.is_empty() && op != K::OpLabel {
                    id_type.insert(r, a[0]);
                }
            }
            if op == K::OpVariable && a.len() >= 2 && m.k(a[1]) == K::Function && e.res.is_some() {
                let r = e.res.unwrap();
                func_vars.insert(r, a.get(2).copied());
                ptr_root.insert(r, r);
                let s = slots.intern(vec![r as u64]);
                ptr_slot.insert(r, s);
            } else if op == K::OpAccessChain && e.res.is_some() && a.len() >= 2 {
                let r = e.res.unwrap();
                let base = a[1];
                let idxs = &a[2..];
                if let Some(&root) = ptr_root.get(&base) {
                    ptr_root.insert(r, root);
                    let base_slot = ptr_slot.get(&base).copied();
                    if base_slot.is_some() && idxs.iter().all(|i| m.const_int_val.contains_key(i)) {
                        let mut v = slots.data[base_slot.unwrap() as usize].clone();
                        v.extend(idxs.iter().map(|i| m.const_int_val.get(i).unwrap().1));
                        let s = slots.intern(v);
                        ptr_slot.insert(r, s);
                    } else {
                        ptr_slot.remove(&r);
                        if func_vars.contains_key(&root) {
                            dyn_indexed.insert(root);
                        }
                    }
                } else if m.readonly_global_vars.contains(&base) {
                    ptr_root.insert(r, base);
                }
            }
        }

        let fv_root = |t: &Tok| -> Option<Tok> {
            match ptr_root.get(t) {
                Some(r) if func_vars.contains_key(r) => Some(*r),
                _ => None,
            }
        };
        // root -> (first depth, mixed)
        let mut slot_depths: Map<Tok, (usize, bool)> = Map::default();
        let mut add_depth = |root: Tok, d: usize| {
            slot_depths
                .entry(root)
                .and_modify(|e| {
                    if e.0 != d {
                        e.1 = true;
                    }
                })
                .or_insert((d, false));
        };
        for e in &insts {
            let Some(op_t) = e.op else { continue };
            let a = sl!(ar, e);
            match m.k(op_t) {
                K::OpLoad => {
                    if a.len() >= 2 {
                        if let Some(root) = fv_root(&a[1]) {
                            if a[2..].iter().any(|&t| m.vol(t)) {
                                escaped.insert(root);
                            }
                            if let Some(&sl) = ptr_slot.get(&a[1]) {
                                add_depth(root, slots.len[sl as usize]);
                            }
                        }
                    }
                    for t in a.iter().skip(2) {
                        if let Some(root) = fv_root(t) {
                            escaped.insert(root);
                        }
                    }
                }
                K::OpStore => {
                    if a.len() >= 2 {
                        if let Some(root) = fv_root(&a[0]) {
                            if a[2..].iter().any(|&t| m.vol(t)) {
                                escaped.insert(root);
                            }
                            if let Some(&sl) = ptr_slot.get(&a[0]) {
                                add_depth(root, slots.len[sl as usize]);
                            }
                        }
                    }
                    for t in a.iter().skip(1) {
                        if let Some(root) = fv_root(t) {
                            escaped.insert(root);
                        }
                    }
                }
                K::OpAccessChain => {
                    for t in a.iter().skip(2) {
                        if let Some(root) = fv_root(t) {
                            escaped.insert(root);
                        }
                    }
                }
                K::OpVariable => {}
                _ => {
                    for t in a {
                        if let Some(root) = fv_root(t) {
                            escaped.insert(root);
                        }
                    }
                }
            }
        }
        for (root, (_, mixed)) in &slot_depths {
            if *mixed {
                escaped.insert(*root);
            }
        }

        let mut trackable = DSet::new(nt);
        for &v in &func_vars.keys {
            if func_vars.contains_key(&v) && !escaped.contains(&v) {
                trackable.insert(v);
            }
        }
        let mut static_trackable = DSet::new(nt);
        for &v in trackable.iter() {
            if !dyn_indexed.contains(&v) {
                static_trackable.insert(v);
            }
        }

        let mut stored_vars: DSet = DSet::new(nt);
        let mut stored_slots: Set<Slot> = Set::default();
        for e in &insts {
            let Some(op_t) = e.op else { continue };
            if m.k(op_t) == K::OpStore && e.an >= 2 {
                let ptr = ar[e.a0 as usize];
                if let Some(&root) = ptr_root.get(&ptr) {
                    if trackable.contains(&root) {
                        stored_vars.insert(root);
                        if let Some(&s) = ptr_slot.get(&ptr) {
                            stored_slots.insert(s);
                        }
                    }
                }
            }
        }

        let mut readonly_slot_val: Map<Slot, Tok> = Map::default();
        let mut readonly_null_zero: Map<Tok, Tok> = Map::default();
        for &v in trackable.iter() {
            if stored_vars.contains(&v) {
                continue;
            }
            let Some(init) = *func_vars.get(&v).unwrap() else { continue };
            if let Some(&z) = m.const_null_elem_zero.get(&init) {
                readonly_null_zero.insert(v, z);
            } else if m.const_ids.contains(&init) {
                readonly_slot_val.insert(slots.intern(vec![v as u64]), init);
            }
        }
        let mut static_null_zero: Map<Tok, Tok> = Map::default();
        for &v in static_trackable.iter() {
            if let Some(init) = *func_vars.get(&v).unwrap() {
                if let Some(&z) = m.const_null_elem_zero.get(&init) {
                    if !stored_slots.contains(&slots.intern(vec![v as u64])) {
                        static_null_zero.insert(v, z);
                    }
                }
            }
        }
        // Entry state needs slots for every trackable variable as well.
        let mut entry_slot_of: Map<Tok, Slot> = Map::default();
        for &v in trackable.iter() {
            entry_slot_of.insert(v, slots.intern(vec![v as u64]));
        }

        let cfg = build_cfg(m, &insts, &ar);
        let mut def_block: Dense<usize> = Dense::new(nt);
        for (bi, bl) in cfg.blocks.iter().enumerate() {
            for &ii in bl {
                if let Some(r) = insts[ii].res {
                    def_block.insert(r, bi);
                }
            }
        }

        let n_insts = insts.len();
        let mut p = Pass {
            m,
            slots: &slots,
            ptr_root,
            ptr_slot,
            id_type,
            trackable,
            static_null_zero,
            stored_slots,
            def_block,
            idom: cfg.idom.clone(),
            ct,
            cf,
            alias: Dense::new(nt),
            dead_idx: vec![false; n_insts],
            changed: false,
            ac_cse: Map::default(),
            val_cse: Map::default(),
            ro_load_cse: Map::default(),
            bitcast_of: Dense::new(nt),
            bool_mask_of: Dense::new(nt),
            ctor_elems: Map::default(),
            logical_not_of: Dense::new(nt),
        };

        if !cfg.blocks.is_empty() {
            let mut entry_sv = readonly_slot_val.clone();
            let mut entry_nz = readonly_null_zero.clone();
            if cfg.all_preds[0].is_empty() {
                for v in p.trackable.iter() {
                    let Some(init) = *func_vars.get(v).unwrap() else { continue };
                    if let Some(&z) = m.const_null_elem_zero.get(&init) {
                        entry_nz.insert(*v, z);
                    } else if m.const_ids.contains(&init) {
                        entry_sv.insert(entry_slot_of[v], init);
                    }
                }
            }
            let mut dfs: Vec<Frame> = vec![Frame::Enter(0, entry_sv, entry_nz)];
            while let Some(fr) = dfs.pop() {
                match fr {
                    Frame::Enter(b, mut sv, mut nz) => {
                        let u = p.process_block(&insts, &mut ar, &cfg.blocks[b], b, &mut sv, &mut nz, true, deleted)?;
                        dfs.push(Frame::Exit(u));
                        for &ch in cfg.dom_children[b].iter().rev() {
                            if cfg.all_preds[ch].len() == 1 && cfg.all_preds[ch][0] == b {
                                dfs.push(Frame::Enter(ch, sv.clone(), nz.clone()));
                            } else {
                                dfs.push(Frame::Enter(ch, readonly_slot_val.clone(), readonly_null_zero.clone()));
                            }
                        }
                    }
                    Frame::Exit(u) => p.undo(u)?,
                }
            }
            let mut reachable = vec![false; cfg.blocks.len()];
            for &b in &cfg.rpo {
                reachable[b] = true;
            }
            for b in 0..cfg.blocks.len() {
                if !reachable[b] {
                    let mut sv = readonly_slot_val.clone();
                    let mut nz = readonly_null_zero.clone();
                    let u = p.process_block(&insts, &mut ar, &cfg.blocks[b], b, &mut sv, &mut nz, false, deleted)?;
                    p.undo(u)?;
                }
            }
        }

        let mut changed = p.changed;
        if p.dead_idx.iter().any(|&d| d) {
            let dead = std::mem::take(&mut p.dead_idx);
            let mut i = 0usize;
            insts.retain(|_| {
                let keep = !dead[i];
                i += 1;
                keep
            });
        }

        // Post-walk alias resolution sweep.
        if !p.alias.is_empty() {
            for e in insts.iter() {
                if e.op.is_none() || e.an == 0 {
                    continue;
                }
                let mut diff = false;
                for t in ar[e.a0 as usize..(e.a0 + e.an) as usize].iter_mut() {
                    if m.pct(*t) {
                        let r = resolve(&mut p.alias, *t)?;
                        if r != *t {
                            *t = r;
                            diff = true;
                        }
                    }
                }
                if diff {
                    changed = true;
                }
            }
        }

        // 3. Whole-function dead-slot store elimination.
        if !static_trackable.is_empty() {
            let mut loaded: Set<Slot> = Set::default();
            for e in &insts {
                let Some(op_t) = e.op else { continue };
                if m.k(op_t) == K::OpLoad && e.an >= 2 {
                    if let Some(&s) = p.ptr_slot.get(&ar[e.a0 as usize + 1]) {
                        if static_trackable.contains(&slots.root[s as usize]) {
                            loaded.insert(s);
                        }
                    }
                }
            }
            let mut any = false;
            let dead: Vec<bool> = insts
                .iter()
                .map(|e| {
                    let Some(op_t) = e.op else { return false };
                    let a = sl!(ar, e);
                    if m.k(op_t) != K::OpStore || a.len() < 2 {
                        return false;
                    }
                    if a[2..].iter().any(|&t| m.vol(t)) {
                        return false;
                    }
                    match p.ptr_slot.get(&a[0]) {
                        Some(&s) if static_trackable.contains(&slots.root[s as usize]) && !loaded.contains(&s) => {
                            any = true;
                            true
                        }
                        _ => false,
                    }
                })
                .collect();
            if any {
                let mut i = 0usize;
                insts.retain(|_| {
                    let keep = !dead[i];
                    i += 1;
                    keep
                });
                changed = true;
            }
        }

        // 4. Backward pure DCE + dead Function-scope OpVariable removal.
        let mut use_count: Dense<i64> = Dense::new(nt);
        for e in &insts {
            if e.op.is_none() {
                continue;
            }
            for &t in sl!(ar, e) {
                if m.pct(t) {
                    use_count.add(t, 1);
                }
            }
        }
        let is_pure = |res: Tok, op_t: Tok, a: &[Tok]| -> bool {
            let info = &m.it.info[op_t as usize];
            let op = info.kind;
            if info.pure_op || op == K::OpPhi {
                return true;
            }
            if op == K::OpExtInst && a.len() >= 3 && m.glsl_sets.contains(&a[1]) && m.it.info[a[2] as usize].glsl_pure {
                return true;
            }
            if op == K::OpLoad && a.len() >= 2 && !a[2..].iter().any(|&t| m.vol(t)) {
                let ptr = a[1];
                let root = p.ptr_root.get(&ptr).copied();
                return root.is_some_and(|r| p.trackable.contains(&r) || m.readonly_global_vars.contains(&r))
                    || m.readonly_global_vars.contains(&ptr);
            }
            if op == K::OpVariable && a.len() >= 2 && m.k(a[1]) == K::Function && p.trackable.contains(&res) {
                return true;
            }
            false
        };
        let mut dce = vec![false; insts.len()];
        let mut any_dce = false;
        for idx in (0..insts.len()).rev() {
            let e = &insts[idx];
            let (Some(r), Some(op_t)) = (e.res, e.op) else { continue };
            if use_count.get(&r).copied().unwrap_or(0) == 0 && is_pure(r, op_t, sl!(ar, e)) {
                dce[idx] = true;
                any_dce = true;
                deleted.insert(r);
                for &t in sl!(ar, e) {
                    if m.pct(t) {
                        use_count.add(t, -1);
                    }
                }
            }
        }
        if any_dce {
            let mut i = 0usize;
            insts.retain(|_| {
                let keep = !dce[i];
                i += 1;
                keep
            });
            changed = true;
        }

        if !changed {
            break;
        }
    }
    Ok((insts, ar))
}

// ---------------------------------------------------------------------------------------------------------------
// optimize()

/// `optimize(spvasm, max_passes)` on the UTF-8 bytes of the text.
pub fn optimize(text: &[u8], max_passes: i64) -> R<Vec<u8>> {
    for &b in text {
        if b >= 0x80 || (b < 0x20 && b != b'\t' && b != b'\n' && b != b'\r') {
            return Err(Decline);
        }
    }
    let lines = split_lines(text);
    let mut it = Interner::new();
    let mut parsed: Vec<Option<Parsed>> = vec![None; lines.len()];
    let mut pargs: Vec<Tok> = Vec::new();

    let mut const_ids: Set<Tok> = Set::default();
    let mut const_true: Set<Tok> = Set::default();
    let mut const_false: Set<Tok> = Set::default();
    let mut const_type: Map<Tok, Tok> = Map::default();
    let mut int_type_info: Map<Tok, u32> = Map::default();
    let mut const_int_val: Map<Tok, (Tok, u64, u32)> = Map::default();
    let mut int_const_by_val: Map<(Tok, u64), Tok> = Map::default();
    let mut array_elem_type: Map<Tok, Tok> = Map::default();
    let mut zero_const_of_type: Map<Tok, Tok> = Map::default();
    let mut const_null_type: Vec<(Tok, Tok)> = Vec::new();
    let mut readonly_global_vars: Set<Tok> = Set::default();
    let mut glsl_sets: Set<Tok> = Set::default();
    let mut decorated_ids: Set<Tok> = Set::default();

    let mut func_ranges: Vec<(usize, usize)> = Vec::new();
    let mut fn_start: Option<usize> = None;
    for (i, &raw) in lines.iter().enumerate() {
        if fn_start.is_none() {
            if contains(raw, b"OpExtInstImport") && contains(raw, b"\"GLSL.std.450\"") {
                let lhs = match raw.iter().position(|&c| c == b'=') {
                    Some(p) => &raw[..p],
                    None => raw,
                };
                let lhs = strip(lhs);
                if lhs.first() == Some(&b'%') {
                    let t = it.intern(lhs);
                    glsl_sets.insert(t);
                }
                continue;
            }
            let stripped = lstrip(raw);
            let mut st = split_ws(stripped);
            let op_tok = st.next().unwrap_or(b"");
            if op_tok == b"OpDecorateString" || op_tok == b"OpDecorateStringGOOGLE" {
                if let Some(t1) = st.next() {
                    if t1.first() == Some(&b'%') {
                        let t = it.intern(t1);
                        decorated_ids.insert(t);
                    }
                }
                continue;
            }
            let Some(p) = parse_line(&mut it, &mut pargs, raw) else { continue };
            let op = it.info[p.op as usize].kind;
            let a = sl!(pargs, p);
            if op == K::OpFunction {
                fn_start = Some(i);
                parsed[i] = Some(p);
                continue;
            }
            match op {
                K::OpDecorate if !a.is_empty() => {
                    decorated_ids.insert(a[0]);
                }
                K::OpDecorateId if !a.is_empty() => {
                    decorated_ids.insert(a[0]);
                    for &t in a.iter().skip(2) {
                        if it.info[t as usize].pct {
                            decorated_ids.insert(t);
                        }
                    }
                }
                K::OpGroupDecorate if a.len() >= 2 => {
                    for &t in a.iter().skip(1) {
                        if it.info[t as usize].pct {
                            decorated_ids.insert(t);
                        }
                    }
                }
                K::OpTypeInt if p.res.is_some() && a.len() >= 2 => {
                    let w = py_int_base10(&it.info[a[0] as usize].s)?;
                    let s = py_int_base10(&it.info[a[1] as usize].s)?;
                    if let (Some(w), Some(_)) = (w, s) {
                        if w == 8 || w == 16 || w == 32 || w == 64 {
                            int_type_info.insert(p.res.unwrap(), w as u32);
                        }
                    }
                }
                K::OpConstant if p.res.is_some() && a.len() >= 2 => {
                    let r = p.res.unwrap();
                    const_ids.insert(r);
                    const_type.insert(r, a[0]);
                    let lit: &[u8] = &it.info[a[1] as usize].s;
                    if lit == b"0" || lit == b"0x0" || lit == b"0.0" {
                        zero_const_of_type.entry(a[0]).or_insert(r);
                    }
                    if let Some(&w) = int_type_info.get(&a[0]) {
                        if let Some(v) = py_int_base0(lit)? {
                            let uval = v & mask(w);
                            const_int_val.insert(r, (a[0], uval, w));
                            int_const_by_val.entry((a[0], uval)).or_insert(r);
                            if uval == 0 {
                                zero_const_of_type.entry(a[0]).or_insert(r);
                            }
                        }
                    }
                }
                K::OpConstantTrue if p.res.is_some() && !a.is_empty() => {
                    let r = p.res.unwrap();
                    const_ids.insert(r);
                    const_true.insert(r);
                    const_type.insert(r, a[0]);
                }
                K::OpConstantFalse if p.res.is_some() && !a.is_empty() => {
                    let r = p.res.unwrap();
                    const_ids.insert(r);
                    const_false.insert(r);
                    const_type.insert(r, a[0]);
                }
                K::OpConstantComposite if p.res.is_some() && !a.is_empty() => {
                    let r = p.res.unwrap();
                    const_ids.insert(r);
                    const_type.insert(r, a[0]);
                }
                K::OpConstantNull if p.res.is_some() && !a.is_empty() => {
                    let r = p.res.unwrap();
                    const_ids.insert(r);
                    const_type.insert(r, a[0]);
                    const_null_type.push((r, a[0]));
                }
                K::OpTypeArray if p.res.is_some() && !a.is_empty() => {
                    array_elem_type.insert(p.res.unwrap(), a[0]);
                }
                K::OpVariable
                    if p.res.is_some()
                        && a.len() >= 2
                        && matches!(it.info[a[1] as usize].kind, K::PushConstant | K::Input) =>
                {
                    readonly_global_vars.insert(p.res.unwrap());
                }
                _ => {}
            }
        } else {
            let p = parse_line(&mut it, &mut pargs, raw);
            let is_end = p.as_ref().is_some_and(|p| it.info[p.op as usize].kind == K::OpFunctionEnd);
            parsed[i] = p;
            if is_end {
                func_ranges.push((fn_start.unwrap(), i + 1));
                fn_start = None;
            }
        }
    }

    if func_ranges.is_empty() {
        return Ok(text.to_vec());
    }
    // `next(iter(set))` with more than one element depends on string hashing: leave it to Python.
    if const_true.len() > 1 || const_false.len() > 1 {
        return Err(Decline);
    }

    // dict semantics: the last OpConstantNull with a given id wins.
    let mut null_map: Map<Tok, Tok> = Map::default();
    for &(nid, ty) in &const_null_type {
        null_map.insert(nid, ty);
    }
    let mut const_null_elem_zero: Map<Tok, Tok> = Map::default();
    for (&nid, ty) in &null_map {
        if let Some(elem) = array_elem_type.get(ty) {
            if let Some(&z) = zero_const_of_type.get(elem) {
                const_null_elem_zero.insert(nid, z);
            }
        }
    }

    let nt = it.info.len();
    let m = Module {
        const_ids: DSet::from_set(nt, &const_ids),
        const_true: DSet::from_set(nt, &const_true),
        const_false: DSet::from_set(nt, &const_false),
        const_true_id: const_true.iter().next().copied(),
        const_false_id: const_false.iter().next().copied(),
        const_type: Dense::from_map(nt, &const_type),
        int_type_info: Dense::from_map(nt, &int_type_info),
        const_int_val: Dense::from_map(nt, &const_int_val),
        int_const_by_val,
        const_null_elem_zero,
        readonly_global_vars: DSet::from_set(nt, &readonly_global_vars),
        glsl_sets: DSet::from_set(nt, &glsl_sets),
        decorated_ids: DSet::from_set(nt, &decorated_ids),
        it,
    };

    let mut all_deleted: Set<Tok> = Set::default();
    let mut opt: Vec<(Vec<Inst>, Vec<Tok>)> = Vec::with_capacity(func_ranges.len());
    for &(s, e) in &func_ranges {
        opt.push(optimize_function(&m, &parsed, &pargs, s, e, max_passes, &mut all_deleted)?);
    }

    let deleted_str = |t: &[u8]| -> bool { m.it.get(t).is_some_and(|x| all_deleted.contains(&x)) };

    let mut out: Vec<u8> = Vec::new();
    out.try_reserve_exact(text.len()).map_err(|_| Decline)?;
    let mut first = true;
    let push_line = |out: &mut Vec<u8>, first: &mut bool, l: &[u8]| {
        if !*first {
            out.push(b'\n');
        }
        *first = false;
        out.extend_from_slice(l);
    };
    let mut fi = 0usize;
    let mut i = 0usize;
    let n = lines.len();
    let mut buf: Vec<u8> = Vec::new();
    while i < n {
        if fi < func_ranges.len() && func_ranges[fi].0 == i {
            let (fin, far) = &opt[fi];
            for e in fin {
                let keep_raw = match e.op {
                    None => true,
                    Some(o) => matches!(m.k(o), K::OpFunction | K::OpFunctionEnd | K::OpLabel | K::OpVariable),
                };
                if keep_raw {
                    push_line(&mut out, &mut first, lines[e.raw]);
                } else {
                    buf.clear();
                    buf.extend_from_slice(b"         ");
                    if let Some(r) = e.res {
                        buf.extend_from_slice(m.s(r));
                        buf.extend_from_slice(b" = ");
                    }
                    buf.extend_from_slice(m.s(e.op.unwrap()));
                    for &a in sl!(far, e) {
                        buf.push(b' ');
                        buf.extend_from_slice(m.s(a));
                    }
                    push_line(&mut out, &mut first, &buf);
                }
            }
            i = func_ranges[fi].1;
            fi += 1;
            continue;
        }
        let raw = lines[i];
        if !all_deleted.is_empty() {
            let stripped = lstrip(raw);
            let mut st = split_ws(stripped);
            let op_tok = st.next().unwrap_or(b"");
            match op_tok {
                b"OpDecorate" | b"OpName" | b"OpDecorateId" | b"OpDecorateString" | b"OpDecorateStringGOOGLE" => {
                    if let Some(t1) = st.next() {
                        if deleted_str(t1) {
                            i += 1;
                            continue;
                        }
                    }
                }
                b"OpGroupDecorate" => {
                    let semi = raw.iter().position(|&c| c == b';');
                    let (code, comment) = match semi {
                        Some(p) => (&raw[..p], &raw[p..]),
                        None => (raw, &raw[raw.len()..]),
                    };
                    let indent = &raw[..raw.len() - stripped.len()];
                    let toks: Vec<&[u8]> = split_ws(code).collect();
                    if toks.len() >= 3 {
                        let kept: Vec<&[u8]> = toks[2..].iter().copied().filter(|t| !deleted_str(t)).collect();
                        if kept.is_empty() {
                            i += 1;
                            continue;
                        }
                        if kept.len() != toks.len() - 2 {
                            buf.clear();
                            buf.extend_from_slice(indent);
                            buf.extend_from_slice(toks[0]);
                            buf.push(b' ');
                            buf.extend_from_slice(toks[1]);
                            buf.push(b' ');
                            for (k, t) in kept.iter().enumerate() {
                                if k > 0 {
                                    buf.push(b' ');
                                }
                                buf.extend_from_slice(t);
                            }
                            if !comment.is_empty() {
                                buf.push(b' ');
                                buf.extend_from_slice(comment);
                            }
                            push_line(&mut out, &mut first, &buf);
                            i += 1;
                            continue;
                        }
                    }
                }
                _ => {}
            }
        }
        push_line(&mut out, &mut first, raw);
        i += 1;
    }
    if text.last() == Some(&b'\n') {
        out.push(b'\n');
    }
    Ok(out)
}

// ---------------------------------------------------------------------------------------------------------------
// C ABI

thread_local! {
    static OUTPUT: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

/// Optimizes `len` bytes of SPIR-V assembly. Returns 0 and sets `*out`/`*out_len` to the optimized text (valid
/// until the next call on the same thread), 1 when the Python optimizer should handle the text, or -1 on bad
/// arguments.
///
/// # Safety
/// `text` must be valid for `len` bytes; `out` and `out_len` must be writable.
#[no_mangle]
pub unsafe extern "C" fn spirv_opt_optimize(text: *const u8, len: usize, max_passes: i64, out: *mut *const u8,
                                            out_len: *mut usize) -> i32 {
    if (text.is_null() && len != 0) || out.is_null() || out_len.is_null() {
        return -1;
    }
    // SAFETY: checked non-null above; the caller guarantees validity.
    let bytes: &[u8] = if len == 0 { &[] } else { unsafe { std::slice::from_raw_parts(text, len) } };
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        OUTPUT.with(|cell| {
            let mut buf = cell.borrow_mut();
            match optimize(bytes, max_passes) {
                Ok(v) => {
                    buf.clear();
                    if buf.try_reserve_exact(v.len()).is_err() {
                        return None;
                    }
                    buf.extend_from_slice(&v);
                    Some((buf.as_ptr(), buf.len()))
                }
                Err(Decline) => None,
            }
        })
    }));
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

    fn opt(s: &str) -> String {
        String::from_utf8(optimize(s.as_bytes(), 4).unwrap()).unwrap()
    }

    #[test]
    fn int_base0_matches_python_grammar() {
        let ok = |s: &str| py_int_base0(s.as_bytes()).unwrap();
        assert_eq!(ok("0"), Some(0));
        assert_eq!(ok("00"), Some(0));
        assert_eq!(ok("0_0"), Some(0));
        assert_eq!(ok("010"), None);
        assert_eq!(ok("0x_1f"), Some(31));
        assert_eq!(ok("0x1f_"), None);
        assert_eq!(ok("0x"), None);
        assert_eq!(ok("1__0"), None);
        assert_eq!(ok("1_0"), Some(10));
        assert_eq!(ok("_1"), None);
        assert_eq!(ok("-1"), Some(u64::MAX));
        assert_eq!(ok("+0b101"), Some(5));
        assert_eq!(ok("0o17"), Some(15));
        assert_eq!(ok("0o8"), None);
        assert_eq!(ok("1.0"), None);
        assert_eq!(ok("-"), None);
        assert_eq!(ok("0x10000000000000001"), Some(1));
        assert!(py_int_base0(&[b'1'; 2000]).is_err());
    }

    #[test]
    fn int_base10_matches_python_grammar() {
        let ok = |s: &str| py_int_base10(s.as_bytes()).unwrap();
        assert_eq!(ok("032"), Some(32));
        assert_eq!(ok("+32"), Some(32));
        assert_eq!(ok("-0"), Some(0));
        assert_eq!(ok("3_2"), Some(32));
        assert_eq!(ok("0x20"), None);
        assert_eq!(ok(""), None);
    }

    #[test]
    fn split_lines_matches_python() {
        let v: Vec<&[u8]> = split_lines(b"a\r\nb\rc\n\nd");
        assert_eq!(v, vec![&b"a"[..], b"b", b"c", b"", b"d"]);
        assert_eq!(split_lines(b"\n").len(), 1);
        assert!(split_lines(b"").is_empty());
    }

    #[test]
    fn no_function_returns_input() {
        let s = "OpCapability Shader\r\n; x\r";
        assert_eq!(opt(s), s);
    }

    #[test]
    fn copy_and_fold() {
        let s = "\
%uint = OpTypeInt 32 0
%u0 = OpConstant %uint 0
%u3 = OpConstant %uint 3
%u5 = OpConstant %uint 5
%u8 = OpConstant %uint 8
%main = OpFunction %void None %fn
%entry = OpLabel
%a = OpIAdd %uint %u3 %u5
%b = OpCopyObject %uint %a
OpStore %out %b
OpReturn
OpFunctionEnd
";
        let o = opt(s);
        assert!(o.contains("OpStore %out %u8"), "{o}");
        assert!(!o.contains("OpCopyObject"));
    }

    #[test]
    fn declines() {
        assert_eq!(optimize("caf\u{e9}".as_bytes(), 4), Err(Decline));
        assert_eq!(optimize(b"a\x0cb", 4), Err(Decline));
        let two_true = "%t1 = OpConstantTrue %bool\n%t2 = OpConstantTrue %bool\n%f = OpFunction %v None %fn\n\
                        %l = OpLabel\nOpReturn\nOpFunctionEnd\n";
        assert_eq!(optimize(two_true.as_bytes(), 4), Err(Decline));
        let cycle = "%f = OpFunction %v None %fn\n%l = OpLabel\n%a = OpCopyObject %t %a\nOpStore %p %a\n\
                     OpReturn\nOpFunctionEnd\n";
        assert_eq!(optimize(cycle.as_bytes(), 4), Err(Decline));
        let no_args = "%f = OpFunction %v None %fn\n%l = OpLabel\n%a = OpUndefined\nOpReturn\nOpFunctionEnd\n";
        assert_eq!(optimize(no_args.as_bytes(), 4), Err(Decline));
    }

    #[test]
    fn abi_round_trip() {
        let s = b"%f = OpFunction %v None %fn\n%l = OpLabel\nOpReturn\nOpFunctionEnd\n";
        let mut p: *const u8 = std::ptr::null();
        let mut n: usize = 0;
        // SAFETY: valid pointers to local storage.
        let rc = unsafe { spirv_opt_optimize(s.as_ptr(), s.len(), 4, &mut p, &mut n) };
        assert_eq!(rc, 0);
        // SAFETY: the library returned a buffer of n bytes.
        let got = unsafe { std::slice::from_raw_parts(p, n) };
        assert_eq!(got, b"%f = OpFunction %v None %fn\n%l = OpLabel\n         OpReturn\nOpFunctionEnd\n");
        // SAFETY: null out pointers are refused.
        assert_eq!(unsafe { spirv_opt_optimize(s.as_ptr(), s.len(), 4, std::ptr::null_mut(), &mut n) }, -1);
    }
}
