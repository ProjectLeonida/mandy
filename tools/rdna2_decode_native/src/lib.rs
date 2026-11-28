//! Native port of the RDNA2 disassembler in `tools/rdna2_decode.py` (`disasm_bytes`).
//!
//! `decode_one` follows `_decode_one` branch by branch and `format_insn` follows `_format_operand` and
//! `format_insn`, so the listing is byte-identical. The opcode tables and the name maps (`T.*`, `_HWREG_NAMES`,
//! `_SENDMSG_NAMES`, the MTBUF formats, ...) are not duplicated: `tools/rdna2_native.py` serialises the Python ones
//! into a text blob that `rdna2_native_init` parses. `FINGERPRINT` is the SHA-256 of the Python decoder functions
//! this file ports; the loader refuses the library when the Python source no longer hashes to it.
//!
//! A case where Python would raise (an opcode spec naming a source it does not count) makes `rdna2_native_disasm`
//! refuse the call, and the caller decodes in Python.

use std::collections::HashSet;
use std::fmt::Write as _;

/// SHA-256 of the Python decoder logic this file is a port of (see `tools/rdna2_native.py::fingerprint`).
macro_rules! fingerprint {
    () => {
        "2bc97d082e7c1cec462744fa4c042a153a3a9ab1ab6f682656cb283658bc24fb"
    };
}

pub const FINGERPRINT: &str = fingerprint!();

const FINGERPRINT_C: &str = concat!(fingerprint!(), "\0");

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum F {
    Vdst,
    Src0,
    Src1,
    Src2,
    Vsrc1,
    Vcc,
    Sdst,
    Ssrc0,
    Ssrc1,
    Simm16,
    Simm32,
    Literal,
    Addr,
    Data,
    Data0,
    Data1,
    Saddr,
    Vdata,
    Vaddr,
    Srsrc,
    Soffset,
    Ssamp,
    Sdata,
    Sbase,
    Offset,
    Other,
}

impl F {
    fn parse(s: &str) -> F {
        match s {
            "VDST" => F::Vdst,
            "SRC0" => F::Src0,
            "SRC1" => F::Src1,
            "SRC2" => F::Src2,
            "VSRC1" => F::Vsrc1,
            "VCC" => F::Vcc,
            "SDST" => F::Sdst,
            "SSRC0" => F::Ssrc0,
            "SSRC1" => F::Ssrc1,
            "SIMM16" => F::Simm16,
            "SIMM32" => F::Simm32,
            "LITERAL" => F::Literal,
            "ADDR" => F::Addr,
            "DATA" => F::Data,
            "DATA0" => F::Data0,
            "DATA1" => F::Data1,
            "SADDR" => F::Saddr,
            "VDATA" => F::Vdata,
            "VADDR" => F::Vaddr,
            "SRSRC" => F::Srsrc,
            "SOFFSET" => F::Soffset,
            "SSAMP" => F::Ssamp,
            "SDATA" => F::Sdata,
            "SBASE" => F::Sbase,
            "OFFSET" => F::Offset,
            _ => F::Other,
        }
    }
    fn is_src(self) -> bool {
        matches!(self, F::Src0 | F::Src1 | F::Src2)
    }
}

struct Entry {
    name: String,
    spec: Vec<(F, i32)>,
    /// SRC0/SRC1/SRC2 fields in `spec` (`nsrc` in `_decode_one`).
    nsrc: usize,
    /// The name is in `_SOPP_BRANCHES` / `_SOPP_NO_OPERAND`.
    branch: bool,
    no_operand: bool,
}

#[derive(Default)]
struct Table(Vec<Option<Entry>>);

impl Table {
    fn get(&self, opc: u32) -> Option<&Entry> {
        self.0.get(opc as usize).and_then(|e| e.as_ref())
    }
    fn contains(&self, opc: u32) -> bool {
        self.get(opc).is_some()
    }
}

#[derive(Default)]
struct IntMap<V>(Vec<Option<V>>);

impl<V> IntMap<V> {
    fn get(&self, k: u32) -> Option<&V> {
        self.0.get(k as usize).and_then(|e| e.as_ref())
    }
    fn insert(&mut self, k: usize, v: V) {
        if self.0.len() <= k {
            self.0.resize_with(k + 1, || None);
        }
        self.0[k] = Some(v);
    }
}

const TABLE_NAMES: [&str; 20] = [
    "DS", "FLAT", "GLOBAL", "MIMG", "MTBUF", "MUBUF", "SCRATCH", "SMEM", "SOP1", "SOP2", "SOPC",
    "SOPK", "SOPP", "VINTRP", "VOP1", "VOP2", "VOP3", "VOP3B", "VOP3P", "VOPC",
];
const DS: usize = 0;
const FLAT: usize = 1;
const GLOBAL: usize = 2;
const MIMG: usize = 3;
const MTBUF: usize = 4;
const MUBUF: usize = 5;
const SCRATCH: usize = 6;
const SMEM: usize = 7;
const SOP1: usize = 8;
const SOP2: usize = 9;
const SOPC: usize = 10;
const SOPK: usize = 11;
const SOPP: usize = 12;
const VINTRP: usize = 13;
const VOP1: usize = 14;
const VOP2: usize = 15;
const VOP3: usize = 16;
const VOP3B: usize = 17;
const VOP3P: usize = 18;
const VOPC: usize = 19;

/// Largest opcode or map key the blob may carry (VOP3 opcodes are 10 bits).
const MAX_KEY: usize = 4096;

/// The tables and name maps of `rdna2_decode.py`, parsed from the blob `rdna2_native.py` builds.
#[derive(Default)]
pub struct Ctx {
    tables: [Table; 20],
    hwreg: IntMap<String>,
    sendmsg: IntMap<String>,
    gs_ops: IntMap<String>,
    mimg_dims: IntMap<String>,
    sdwa_sels: IntMap<String>,
    sdwa_unused: IntMap<String>,
    omod: IntMap<String>,
    inline_float: IntMap<String>,
    formats: IntMap<String>,
    dim_coords: IntMap<i64>,
    grad_coords: IntMap<i64>,
    vop2_literal: IntMap<()>,
    sopk_literal: IntMap<()>,
    sopp_branches: HashSet<String>,
    sopp_no_operand: HashSet<String>,
}

impl Ctx {
    /// Parse the blob: one tab-separated record per line.
    ///
    /// `T <table> <opc> <name> <field>:<nregs>,...` an opcode entry; `M <map> <key> <text>` a name map entry;
    /// `I <map> <key> <int>` an integer map entry; `S <set> <item>` a set member.
    pub fn parse(blob: &str) -> Option<Ctx> {
        if !blob.is_ascii() {
            return None; // format_insn pads by bytes
        }
        let mut ctx = Ctx::default();
        for line in blob.split('\n') {
            if line.is_empty() {
                continue;
            }
            let parts: Vec<&str> = line.split('\t').collect();
            match parts[0] {
                "T" if parts.len() == 5 => {
                    let ti = TABLE_NAMES.iter().position(|n| *n == parts[1])?;
                    let opc: usize = parts[2].parse().ok()?;
                    if opc >= MAX_KEY {
                        return None;
                    }
                    let mut spec = Vec::new();
                    if !parts[4].is_empty() {
                        for field in parts[4].split(',') {
                            let (f, n) = field.split_once(':')?;
                            spec.push((F::parse(f), n.parse().ok()?));
                        }
                    }
                    let t = &mut ctx.tables[ti].0;
                    if t.len() <= opc {
                        t.resize_with(opc + 1, || None);
                    }
                    let nsrc = spec.iter().filter(|(f, _)| f.is_src()).count();
                    t[opc] = Some(Entry {
                        name: parts[3].to_string(),
                        spec,
                        nsrc,
                        branch: false,
                        no_operand: false,
                    });
                }
                "M" if parts.len() == 4 => {
                    let k: usize = parts[2].parse().ok()?;
                    if k >= MAX_KEY {
                        return None;
                    }
                    let v = parts[3].to_string();
                    match parts[1] {
                        "HWREG" => ctx.hwreg.insert(k, v),
                        "SENDMSG" => ctx.sendmsg.insert(k, v),
                        "GS_OPS" => ctx.gs_ops.insert(k, v),
                        "MIMG_DIMS" => ctx.mimg_dims.insert(k, v),
                        "SDWA_SELS" => ctx.sdwa_sels.insert(k, v),
                        "SDWA_UNUSED" => ctx.sdwa_unused.insert(k, v),
                        "OMOD" => ctx.omod.insert(k, v),
                        "INLINE_FLOAT" => ctx.inline_float.insert(k, v),
                        "FORMATS" => ctx.formats.insert(k, v),
                        _ => return None,
                    }
                }
                "I" if parts.len() == 4 => {
                    let k: usize = parts[2].parse().ok()?;
                    let v: i64 = parts[3].parse().ok()?;
                    if k >= MAX_KEY {
                        return None;
                    }
                    match parts[1] {
                        "DIM_COORDS" => ctx.dim_coords.insert(k, v),
                        "GRAD_COORDS" => ctx.grad_coords.insert(k, v),
                        _ => return None,
                    }
                }
                "S" if parts.len() == 3 => match parts[1] {
                    "VOP2_LITERAL" | "SOPK_LITERAL" => {
                        let k: usize = parts[2].parse().ok()?;
                        if k >= MAX_KEY {
                            return None;
                        }
                        if parts[1] == "VOP2_LITERAL" {
                            ctx.vop2_literal.insert(k, ());
                        } else {
                            ctx.sopk_literal.insert(k, ());
                        }
                    }
                    "SOPP_BRANCHES" => {
                        ctx.sopp_branches.insert(parts[2].to_string());
                    }
                    "SOPP_NO_OPERAND" => {
                        ctx.sopp_no_operand.insert(parts[2].to_string());
                    }
                    _ => return None,
                },
                _ => return None,
            }
        }
        for table in ctx.tables.iter_mut() {
            for e in table.0.iter_mut().flatten() {
                e.branch = ctx.sopp_branches.contains(&e.name);
                e.no_operand = ctx.sopp_no_operand.contains(&e.name);
            }
        }
        Some(ctx)
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Operands (`_sreg`, `_vreg`, `_src`, `_format_operand`)
//
// An instruction is decoded into a scratch buffer reused for every instruction: an operand's text is a byte range
// of `Sc::text`, and the modifiers are appended to `Sc::mods` already joined by spaces, as `format_insn` joins them.

#[derive(Clone, Copy)]
enum Base {
    Text(u32, u32),
    Static(&'static str),
    Imm(i64),
    Fimm(u32),
}

#[derive(Clone, Copy)]
struct Op {
    base: Base,
    sext: bool,
    abs: bool,
    neg: bool,
}

impl Op {
    fn new(base: Base) -> Op {
        Op {
            base,
            sext: false,
            abs: false,
            neg: false,
        }
    }
    fn st(s: &'static str) -> Op {
        Op::new(Base::Static(s))
    }
    fn imm(v: i64) -> Op {
        Op::new(Base::Imm(v))
    }
}

#[derive(Default)]
struct Sc {
    text: String,
    args: Vec<Op>,
    mods: String,
}

impl Sc {
    fn clear(&mut self) {
        self.text.clear();
        self.args.clear();
        self.mods.clear();
    }
    fn begin(&self) -> u32 {
        self.text.len() as u32
    }
    fn end(&self, start: u32) -> Op {
        Op::new(Base::Text(start, self.text.len() as u32))
    }
    /// An operand with formatted text.
    fn tf(&mut self, a: std::fmt::Arguments) -> Op {
        let start = self.begin();
        let _ = self.text.write_fmt(a);
        self.end(start)
    }
    fn arg(&mut self, op: Op) {
        self.args.push(op);
    }
    fn mod_sep(&mut self) {
        if !self.mods.is_empty() {
            self.mods.push(' ');
        }
    }
    /// A modifier with formatted text.
    fn mf(&mut self, a: std::fmt::Arguments) {
        self.mod_sep();
        let _ = self.mods.write_fmt(a);
    }
    fn ms(&mut self, s: &str) {
        self.mod_sep();
        self.mods.push_str(s);
    }
    fn range(&mut self, p: char, r: i64, n: i64) -> Op {
        let start = self.begin();
        if n == 1 {
            let _ = write!(self.text, "{p}{r}");
        } else {
            let _ = write!(self.text, "{p}[{r}:{}]", r + n - 1);
        }
        self.end(start)
    }
    /// `_vreg(reg, n)`.
    fn v(&mut self, reg: u32, n: i32) -> Op {
        self.range('v', (reg & 0xFF) as i64, n.max(1) as i64)
    }
    /// `_sreg(code, n)`.
    fn s(&mut self, code: u32, n: i32) -> Op {
        let c = code as i64;
        let n = n as i64;
        if code <= 105 {
            if n <= 1 {
                return self.range('s', c, 1);
            }
            if c + n <= 106 || c + n > 108 {
                return self.range('s', c, n);
            }
            // LLVM VCCTuple spanning SGPRs into vcc_lo (106) / vcc_hi (107).
            let start = self.begin();
            self.text.push('[');
            for r in c..c + n {
                if r > c {
                    self.text.push(',');
                }
                if r <= 105 {
                    let _ = write!(self.text, "s{r}");
                } else if r == 106 {
                    self.text.push_str("vcc_lo");
                } else {
                    self.text.push_str("vcc_hi");
                }
            }
            self.text.push(']');
            return self.end(start);
        }
        match code {
            106 => Op::st(if n == 2 { "vcc" } else { "vcc_lo" }),
            107 => Op::st("vcc_hi"),
            108..=123 => {
                let lo = c - 108;
                if n <= 1 {
                    self.tf(format_args!("ttmp{lo}"))
                } else {
                    self.tf(format_args!("ttmp[{lo}:{}]", lo + n - 1))
                }
            }
            124 => Op::st("m0"),
            125 => Op::st("null"),
            126 => Op::st(if n == 2 { "exec" } else { "exec_lo" }),
            127 => Op::st("exec_hi"),
            _ => self.tf(format_args!("sreg_{code}")),
        }
    }
}

fn vcc(wave32: bool) -> Op {
    Op::st(if wave32 { "vcc_lo" } else { "vcc" })
}

fn s16(v: u32) -> i64 {
    let v = (v & 0xFFFF) as i64;
    if v & 0x8000 != 0 {
        v - 0x10000
    } else {
        v
    }
}

fn s12(v: u32) -> i64 {
    let v = (v & 0xFFF) as i64;
    if v & 0x800 != 0 {
        v - 0x1000
    } else {
        v
    }
}

fn s21(v: u32) -> i64 {
    let v = (v & 0x1FFFFF) as i64;
    if v & 0x100000 != 0 {
        v - 0x200000
    } else {
        v
    }
}

fn width(nr: i32, wave32: bool) -> i32 {
    if nr == -1 {
        if wave32 {
            1
        } else {
            2
        }
    } else {
        nr
    }
}

impl Ctx {
    /// `_src(code, n, lit)`.
    fn src(&self, sc: &mut Sc, code: u32, n: i32, lit: Option<u32>) -> Op {
        match code {
            0..=127 => sc.s(code, n),
            128..=192 => Op::imm(code as i64 - 128),
            193..=208 => Op::imm(-(code as i64 - 192)),
            _ if self.inline_float.get(code).is_some() => Op::new(Base::Fimm(code)),
            251 => Op::st("src_vccz"),
            252 => Op::st("src_execz"),
            253 => Op::st("src_scc"),
            254 => Op::st("src_lds_direct"),
            255 => Op::imm(lit.unwrap_or(0) as i64),
            256..=511 => sc.v(code - 256, n),
            _ => sc.tf(format_args!("src_{code}")),
        }
    }

    fn src_mod(
        &self,
        sc: &mut Sc,
        code: u32,
        n: i32,
        lit: Option<u32>,
        neg: bool,
        absm: bool,
    ) -> Op {
        let mut op = self.src(sc, code, n, lit);
        if absm {
            op.abs = true;
        }
        if neg {
            op.neg = true;
        }
        op
    }

    fn sdwa_src(&self, sc: &mut Sc, reg: u32, is_sgpr: bool) -> Op {
        if is_sgpr {
            self.src(sc, reg, 1, None)
        } else {
            sc.v(reg, 1)
        }
    }

    /// `_format_operand(a, hex_imm)`, appended to `out`.
    fn put_operand(&self, text: &str, a: &Op, hex_imm: bool, out: &mut String) {
        let is_const = matches!(a.base, Base::Imm(_) | Base::Fimm(_));
        if a.neg {
            out.push_str(if is_const { "neg(" } else { "-" });
        }
        if a.abs {
            out.push('|');
        }
        if a.sext {
            out.push_str("sext(");
        }
        match a.base {
            Base::Text(s, e) => out.push_str(&text[s as usize..e as usize]),
            Base::Static(s) => out.push_str(s),
            Base::Imm(v) => {
                if hex_imm || v >= 100 {
                    if v < 0 {
                        let _ = write!(out, "0x-{:x}", -v);
                    } else {
                        let _ = write!(out, "0x{v:x}");
                    }
                } else {
                    let _ = write!(out, "{v}");
                }
            }
            Base::Fimm(c) => {
                if let Some(t) = self.inline_float.get(c) {
                    out.push_str(t);
                }
            }
        }
        if a.sext {
            out.push(')');
        }
        if a.abs {
            out.push('|');
        }
        if a.neg && is_const {
            out.push(')');
        }
    }

    fn sel(&self, v: u32) -> &str {
        self.sdwa_sels.get(v).map(|s| s.as_str()).unwrap_or("DWORD")
    }

    fn unused(&self, v: u32) -> &str {
        self.sdwa_unused
            .get(v)
            .map(|s| s.as_str())
            .unwrap_or("UNUSED_PAD")
    }

    fn omod_mod(&self, sc: &mut Sc, omod: u32) {
        if let Some(m) = self.omod.get(omod) {
            sc.ms(m);
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Modifier helpers (`_format_hwreg`, `_format_waitcnt`, `_format_sendmsg`, `_format_dpp16_ctrl`,
// `_format_ds_swizzle`, `_mimg_addr_count`, `_mimg_data_count`)

impl Ctx {
    fn hwreg(&self, sc: &mut Sc, simm16: u32) -> Op {
        let reg_id = simm16 & 0x3F;
        let offset = (simm16 >> 6) & 0x1F;
        let size = ((simm16 >> 11) & 0x1F) + 1;
        let start = sc.begin();
        sc.text.push_str("hwreg(");
        match self.hwreg.get(reg_id) {
            Some(n) => sc.text.push_str(n),
            None => {
                let _ = write!(sc.text, "{reg_id}");
            }
        }
        if !(offset == 0 && size == 32) {
            let _ = write!(sc.text, ", {offset}, {size}");
        }
        sc.text.push(')');
        sc.end(start)
    }

    fn sendmsg(&self, sc: &mut Sc, simm16: u32) -> Op {
        let msg_id = simm16 & 0xF;
        let op = (simm16 >> 4) & 0x7;
        let stream = (simm16 >> 8) & 0x3;
        let mname = match self.sendmsg.get(msg_id) {
            Some(n) => n,
            None => return sc.tf(format_args!("sendmsg({simm16})")),
        };
        if msg_id == 2 || msg_id == 3 {
            let start = sc.begin();
            let _ = write!(sc.text, "sendmsg({mname}|");
            match self.gs_ops.get(op) {
                Some(n) => sc.text.push_str(n),
                None => {
                    let _ = write!(sc.text, "{op}");
                }
            }
            if op != 0 {
                let _ = write!(sc.text, "|{stream}");
            }
            sc.text.push(')');
            return sc.end(start);
        }
        if op != 0 {
            return sc.tf(format_args!("sendmsg({mname}|{op})"));
        }
        sc.tf(format_args!("sendmsg({mname})"))
    }

    fn mimg_addr_count(&self, name: &str, dim: u32, a16: bool) -> i64 {
        if name == "image_bvh_intersect_ray" || name == "image_bvh64_intersect_ray" {
            return if name == "image_bvh_intersect_ray" {
                11
            } else {
                12
            };
        }
        let dimc = |d: u32| self.dim_coords.get(d).copied().unwrap_or(2);
        if name == "image_get_resinfo" {
            return 1;
        }
        if name == "image_get_lod" {
            return dimc(dim);
        }
        let base = dimc(dim);
        if name.starts_with("image_atomic_") {
            return if a16 { (base + 1).div_euclid(2) } else { base };
        }
        let has_o = name.ends_with("_o") || name.contains("_o_");
        let mut base_name = if name.ends_with("_o") {
            &name[..name.len() - 2]
        } else {
            name
        };
        let has_cl = base_name.ends_with("_cl");
        if has_cl {
            base_name = &base_name[..base_name.len() - 3];
        }
        let mut fixed32: i64 = if has_o { 1 } else { 0 };
        if base_name.contains("_b") && !base_name.starts_with("image_bvh") {
            fixed32 += 1;
        }
        let atomic_or_get =
            base_name.starts_with("image_atomic") || base_name.starts_with("image_get");
        if base_name.contains("_c") && !atomic_or_get {
            fixed32 += 1;
        }
        if base_name.contains("_d") && !atomic_or_get {
            fixed32 += 2 * self.grad_coords.get(dim).copied().unwrap_or(2);
        }
        let mut packable = base;
        if base_name.contains("_l")
            && !base_name.contains("_lz")
            && !(base_name.starts_with("image_load")
                || base_name.starts_with("image_store")
                || base_name.starts_with("image_atomic")
                || base_name.starts_with("image_get"))
        {
            packable += 1;
        }
        if has_cl {
            packable += 1;
        }
        if base_name == "image_load_mip" || base_name == "image_store_mip" {
            packable += 1;
        }
        if a16 {
            let packed = (base + 1).div_euclid(2) + (packable - base);
            return fixed32 + packed;
        }
        fixed32 + packable
    }
}

fn waitcnt(sc: &mut Sc, simm16: u32) {
    let vmcnt = (simm16 & 0xF) | (((simm16 >> 14) & 0x3) << 4);
    let expcnt = (simm16 >> 4) & 0x7;
    let lgkmcnt = (simm16 >> 8) & 0x3F;
    let before = sc.args.len();
    if vmcnt != 63 {
        let o = sc.tf(format_args!("vmcnt({vmcnt})"));
        sc.arg(o);
    }
    if expcnt != 7 {
        let o = sc.tf(format_args!("expcnt({expcnt})"));
        sc.arg(o);
    }
    if lgkmcnt != 63 {
        let o = sc.tf(format_args!("lgkmcnt({lgkmcnt})"));
        sc.arg(o);
    }
    if sc.args.len() == before {
        sc.arg(Op::imm(simm16 as i64));
    }
}

fn dpp16_ctrl(sc: &mut Sc, ctrl: u32) {
    if ctrl <= 0xFF {
        sc.mf(format_args!(
            "quad_perm:[{},{},{},{}]",
            ctrl & 3,
            (ctrl >> 2) & 3,
            (ctrl >> 4) & 3,
            (ctrl >> 6) & 3
        ));
        return;
    }
    match ctrl {
        0x101..=0x10F => sc.mf(format_args!("row_shl:{}", ctrl - 0x100)),
        0x111..=0x11F => sc.mf(format_args!("row_shr:{}", ctrl - 0x110)),
        0x121..=0x12F => sc.mf(format_args!("row_ror:{}", ctrl - 0x120)),
        0x140 => sc.ms("row_mirror"),
        0x141 => sc.ms("row_half_mirror"),
        0x150..=0x15F => sc.mf(format_args!("row_share:{}", ctrl - 0x150)),
        0x160..=0x16F => sc.mf(format_args!("row_xmask:{}", ctrl - 0x160)),
        _ => sc.mf(format_args!("dpp_ctrl:0x{ctrl:x}")),
    }
}

fn dpp8(sc: &mut Sc, w1: u32, src0: u32) {
    sc.ms("dpp8:[");
    for i in 0..8 {
        if i > 0 {
            sc.mods.push(',');
        }
        sc.mods
            .push((b'0' + ((w1 >> (8 + 3 * i)) & 7) as u8) as char);
    }
    sc.mods.push(']');
    if src0 == 234 {
        sc.ms("fi:1");
    }
}

fn ds_swizzle(sc: &mut Sc, offset16: u32) {
    if offset16 & 0x8000 == 0 {
        let and_m = offset16 & 0x1F;
        let or_m = (offset16 >> 5) & 0x1F;
        let xor_m = (offset16 >> 10) & 0x1F;
        if and_m == 0x1F && or_m == 0 {
            if matches!(xor_m, 1 | 2 | 4 | 8 | 16) {
                return sc.mf(format_args!("offset:swizzle(SWAP,{xor_m})"));
            }
            if matches!(xor_m, 3 | 7 | 15 | 31) {
                return sc.mf(format_args!("offset:swizzle(REVERSE,{})", xor_m + 1));
            }
        }
        if xor_m == 0 && (or_m & and_m) == 0 && matches!(0x1F ^ and_m, 1 | 3 | 7 | 15 | 31) {
            let grp = (0x1F ^ and_m) + 1;
            return sc.mf(format_args!("offset:swizzle(BROADCAST,{grp},{or_m})"));
        }
        let mut chars = [0u8; 5];
        for (i, bit) in (0..5).rev().enumerate() {
            chars[i] = match ((and_m >> bit) & 1, (or_m >> bit) & 1, (xor_m >> bit) & 1) {
                (0, 0, 0) => b'0',
                (0, 1, 0) => b'1',
                (1, 0, 0) => b'p',
                (1, 0, 1) => b'i',
                _ => return sc.mf(format_args!("offset:{offset16}")),
            };
        }
        let chars = std::str::from_utf8(&chars).unwrap_or("");
        return sc.mf(format_args!("offset:swizzle(BITMASK_PERM,\"{chars}\")"));
    }
    if offset16 & 0xFF00 == 0x8000 {
        return sc.mf(format_args!(
            "offset:swizzle(QUAD_PERM,{},{},{},{})",
            offset16 & 3,
            (offset16 >> 2) & 3,
            (offset16 >> 4) & 3,
            (offset16 >> 6) & 3
        ));
    }
    sc.mf(format_args!("offset:{offset16}"))
}

fn mimg_data_count(name: &str, dmask: u32, d16: bool, tfe: bool) -> i64 {
    if name == "image_bvh_intersect_ray" || name == "image_bvh64_intersect_ray" {
        return 4;
    }
    if name.starts_with("image_atomic_") {
        if matches!(dmask, 1 | 3 | 15) {
            return dmask.count_ones() as i64;
        }
        let mut n = if name.contains("cmpswap") { 2 } else { 1 };
        if name.ends_with("_x2") {
            n *= 2;
        }
        return n;
    }
    let mut n: i64 = if name.starts_with("image_gather4") {
        4
    } else {
        match (dmask & 0xF).count_ones() {
            0 => 1,
            c => c as i64,
        }
    };
    if d16 {
        n = (n + 1).div_euclid(2);
    }
    n + if tfe { 1 } else { 0 }
}

fn bits_mod(sc: &mut Sc, label: &str, v: u32, n: usize) {
    sc.mod_sep();
    sc.mods.push_str(label);
    sc.mods.push_str(":[");
    for i in 0..n {
        if i > 0 {
            sc.mods.push(',');
        }
        sc.mods.push(if (v >> i) & 1 != 0 { '1' } else { '0' });
    }
    sc.mods.push(']');
}

fn xyzw(c: u32) -> char {
    ['x', 'y', 'z', 'w'][(c & 3) as usize]
}

fn interp_param(sc: &mut Sc, v: u32) -> Op {
    match v & 3 {
        0 => Op::st("p10"),
        1 => Op::st("p20"),
        2 => Op::st("p0"),
        _ => sc.tf(format_args!("p{v}")),
    }
}

// ---------------------------------------------------------------------------------------------------------------
// One instruction (`_decode_one`)

/// The decoded op name (`base` + `suffix`) and byte length; operands and modifiers are in the scratch buffer.
struct Ins<'a> {
    base: &'a str,
    suffix: &'static str,
    nbytes: usize,
}

/// `Ok(None)`: an invalid word. `Err(())`: a case Python would raise on; the whole call is refused.
type Decoded<'a> = Result<Option<Ins<'a>>, ()>;

fn rd(code: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([code[at], code[at + 1], code[at + 2], code[at + 3]])
}

fn done<'a>(base: &'a str, suffix: &'static str, nbytes: usize) -> Decoded<'a> {
    Ok(Some(Ins {
        base,
        suffix,
        nbytes,
    }))
}

macro_rules! entry {
    ($self:ident, $t:expr, $opc:expr) => {
        match $self.tables[$t].get($opc) {
            Some(e) => e,
            None => return Ok(None),
        }
    };
}

impl Ctx {
    /// Decode at `pc` (at least 4 bytes remain) into `sc`.
    fn decode_one<'a>(
        &'a self,
        sc: &mut Sc,
        code: &[u8],
        pc: usize,
        wave32: bool,
        gfx101: bool,
    ) -> Decoded<'a> {
        let len = code.len();
        let need = |n: usize| pc + n <= len;
        let w0 = rd(code, pc);
        let word = |off: usize| rd(code, pc + off);
        let wv = if wave32 { 1 } else { 2 };

        // ---- 1. Bit 31 == 0: VOP1 / VOPC / VOP2 ----
        if w0 >> 31 == 0 {
            let hi7 = w0 >> 25;
            let src0 = w0 & 0x1FF;
            let is_dpp16 = src0 == 250;
            let is_dpp8 = src0 == 233 || src0 == 234;
            let is_sdwa = src0 == 249;

            if hi7 == 0x3F {
                // VOP1
                let opc = (w0 >> 9) & 0xFF;
                let vdst = (w0 >> 17) & 0xFF;
                let e = entry!(self, VOP1, opc);
                let name = e.name.as_str();
                if is_dpp16 || is_dpp8 || is_sdwa {
                    if !need(8) {
                        return Ok(None);
                    }
                    let w1 = word(4);
                    let src0_reg = w1 & 0xFF;
                    let suffix;
                    let mut s0;
                    if is_dpp16 {
                        suffix = "_dpp";
                        s0 = sc.v(src0_reg, 1);
                        s0.neg = (w1 >> 20) & 1 != 0;
                        s0.abs = (w1 >> 21) & 1 != 0;
                        dpp16_ctrl(sc, (w1 >> 8) & 0x1FF);
                        sc.mf(format_args!("row_mask:0x{:x}", (w1 >> 28) & 0xF));
                        sc.mf(format_args!("bank_mask:0x{:x}", (w1 >> 24) & 0xF));
                        if (w1 >> 19) & 1 != 0 {
                            sc.ms("bound_ctrl:1");
                        }
                        if (w1 >> 18) & 1 != 0 {
                            sc.ms("fi:1");
                        }
                    } else if is_dpp8 {
                        suffix = "_dpp";
                        s0 = sc.v(src0_reg, 1);
                        dpp8(sc, w1, src0);
                    } else {
                        suffix = "_sdwa";
                        s0 = self.sdwa_src(sc, src0_reg, (w1 >> 23) & 1 != 0);
                        if (w1 >> 19) & 1 != 0 {
                            s0.sext = true;
                        }
                        s0.neg = (w1 >> 20) & 1 != 0;
                        s0.abs = (w1 >> 21) & 1 != 0;
                        if (w1 >> 13) & 1 != 0 {
                            sc.ms("clamp");
                        }
                        self.omod_mod(sc, (w1 >> 14) & 3);
                        sc.mf(format_args!("dst_sel:{}", self.sel((w1 >> 8) & 7)));
                        sc.mf(format_args!("dst_unused:{}", self.unused((w1 >> 11) & 3)));
                        sc.mf(format_args!("src0_sel:{}", self.sel((w1 >> 16) & 7)));
                    }
                    for &(f, nr) in &e.spec {
                        let w = width(nr, wave32);
                        match f {
                            F::Vdst => {
                                let o = sc.v(vdst, w);
                                sc.arg(o);
                            }
                            F::Src0 => sc.arg(s0),
                            _ => {}
                        }
                    }
                    return done(name, suffix, 8);
                }
                let mut lit = None;
                let mut nbytes = 4;
                if src0 == 255 {
                    if !need(8) {
                        return Ok(None);
                    }
                    lit = Some(word(4));
                    nbytes = 8;
                }
                let suffix = if e.spec.is_empty()
                    || matches!(
                        name,
                        "v_nop" | "v_clrexcp" | "v_pipeflush" | "v_readfirstlane_b32"
                    ) {
                    ""
                } else {
                    "_e32"
                };
                for &(f, nr) in &e.spec {
                    let w = width(nr, wave32);
                    match f {
                        F::Vdst => {
                            let o = if name == "v_readfirstlane_b32" {
                                sc.s(vdst & 0x7F, 1)
                            } else {
                                sc.v(vdst, w)
                            };
                            sc.arg(o);
                        }
                        F::Src0 => {
                            let o = self.src(sc, src0, w, lit);
                            sc.arg(o);
                        }
                        _ => {}
                    }
                }
                return done(name, suffix, nbytes);
            }

            if hi7 == 0x3E {
                // VOPC
                let opc = (w0 >> 17) & 0xFF;
                let vsrc1 = (w0 >> 9) & 0xFF;
                let e = entry!(self, VOPC, opc);
                let name = e.name.as_str();
                let cmpx = name.starts_with("v_cmpx_");
                if is_sdwa {
                    if !need(8) {
                        return Ok(None);
                    }
                    let w1 = word(4);
                    let src0_reg = w1 & 0xFF;
                    let sdst = (w1 >> 8) & 0x7F;
                    let sd = (w1 >> 15) & 1 != 0;
                    let mut s0 = self.sdwa_src(sc, src0_reg, (w1 >> 23) & 1 != 0);
                    let mut s1 = self.sdwa_src(sc, vsrc1, (w1 >> 31) & 1 != 0);
                    if (w1 >> 19) & 1 != 0 {
                        s0.sext = true;
                    }
                    if (w1 >> 27) & 1 != 0 {
                        s1.sext = true;
                    }
                    s0.neg = (w1 >> 20) & 1 != 0;
                    s0.abs = (w1 >> 21) & 1 != 0;
                    s1.neg = (w1 >> 28) & 1 != 0;
                    s1.abs = (w1 >> 29) & 1 != 0;
                    sc.mf(format_args!("src0_sel:{}", self.sel((w1 >> 16) & 7)));
                    sc.mf(format_args!("src1_sel:{}", self.sel((w1 >> 24) & 7)));
                    let dst = if sd { sc.s(sdst, wv) } else { vcc(wave32) };
                    if !cmpx {
                        sc.arg(dst);
                    }
                    sc.arg(s0);
                    sc.arg(s1);
                    return done(name, "_sdwa", 8);
                }
                let mut lit = None;
                let mut nbytes = 4;
                if src0 == 255 {
                    if !need(8) {
                        return Ok(None);
                    }
                    lit = Some(word(4));
                    nbytes = 8;
                }
                for &(f, nr) in &e.spec {
                    let w = width(nr, wave32);
                    match f {
                        F::Vcc => {
                            if !cmpx {
                                sc.arg(vcc(wave32));
                            }
                        }
                        F::Src0 => {
                            let o = self.src(sc, src0, w, lit);
                            sc.arg(o);
                        }
                        F::Vsrc1 => {
                            let o = sc.v(vsrc1, w);
                            sc.arg(o);
                        }
                        _ => {}
                    }
                }
                return done(name, "_e32", nbytes);
            }

            // VOP2
            let opc = (w0 >> 25) & 0x3F;
            let vdst = (w0 >> 17) & 0xFF;
            let vsrc1 = (w0 >> 9) & 0xFF;
            let e = entry!(self, VOP2, opc);
            let name = e.name.as_str();
            if is_dpp16 || is_dpp8 || is_sdwa {
                if !need(8) {
                    return Ok(None);
                }
                let w1 = word(4);
                let src0_reg = w1 & 0xFF;
                let suffix;
                let mut s0;
                let mut s1;
                if is_dpp16 {
                    suffix = "_dpp";
                    s0 = sc.v(src0_reg, 1);
                    s0.neg = (w1 >> 20) & 1 != 0;
                    s0.abs = (w1 >> 21) & 1 != 0;
                    s1 = sc.v(vsrc1, 1);
                    s1.neg = (w1 >> 22) & 1 != 0;
                    s1.abs = (w1 >> 23) & 1 != 0;
                    dpp16_ctrl(sc, (w1 >> 8) & 0x1FF);
                    sc.mf(format_args!("row_mask:0x{:x}", (w1 >> 28) & 0xF));
                    sc.mf(format_args!("bank_mask:0x{:x}", (w1 >> 24) & 0xF));
                    if (w1 >> 19) & 1 != 0 {
                        sc.ms("bound_ctrl:1");
                    }
                    if (w1 >> 18) & 1 != 0 {
                        sc.ms("fi:1");
                    }
                } else if is_dpp8 {
                    suffix = "_dpp";
                    s0 = sc.v(src0_reg, 1);
                    s1 = sc.v(vsrc1, 1);
                    dpp8(sc, w1, src0);
                } else {
                    suffix = "_sdwa";
                    s0 = self.sdwa_src(sc, src0_reg, (w1 >> 23) & 1 != 0);
                    s1 = self.sdwa_src(sc, vsrc1, (w1 >> 31) & 1 != 0);
                    if (w1 >> 19) & 1 != 0 {
                        s0.sext = true;
                    }
                    if (w1 >> 27) & 1 != 0 {
                        s1.sext = true;
                    }
                    s0.neg = (w1 >> 20) & 1 != 0;
                    s0.abs = (w1 >> 21) & 1 != 0;
                    s1.neg = (w1 >> 28) & 1 != 0;
                    s1.abs = (w1 >> 29) & 1 != 0;
                    if (w1 >> 13) & 1 != 0 {
                        sc.ms("clamp");
                    }
                    self.omod_mod(sc, (w1 >> 14) & 3);
                    sc.mf(format_args!("dst_sel:{}", self.sel((w1 >> 8) & 7)));
                    sc.mf(format_args!("dst_unused:{}", self.unused((w1 >> 11) & 3)));
                    sc.mf(format_args!("src0_sel:{}", self.sel((w1 >> 16) & 7)));
                    sc.mf(format_args!("src1_sel:{}", self.sel((w1 >> 24) & 7)));
                }
                for &(f, nr) in &e.spec {
                    let w = width(nr, wave32);
                    match f {
                        F::Vdst => {
                            let o = sc.v(vdst, w);
                            sc.arg(o);
                        }
                        F::Vcc => sc.arg(vcc(wave32)),
                        F::Src0 => sc.arg(s0),
                        F::Vsrc1 => sc.arg(s1),
                        _ => {}
                    }
                }
                return done(name, suffix, 8);
            }
            let mandatory = self.vop2_literal.get(opc).is_some();
            let mut lit = None;
            let mut nbytes = 4;
            if src0 == 255 || mandatory {
                if !need(8) {
                    return Ok(None);
                }
                lit = Some(word(4));
                nbytes = 8;
            }
            for &(f, nr) in &e.spec {
                let w = width(nr, wave32);
                let o = match f {
                    F::Vdst => sc.v(vdst, w),
                    F::Vcc => vcc(wave32),
                    F::Src0 => self.src(sc, src0, w, lit),
                    F::Vsrc1 => sc.v(vsrc1, w),
                    F::Simm32 | F::Literal => Op::imm(lit.unwrap_or(0) as i64),
                    _ => continue,
                };
                sc.arg(o);
            }
            return done(name, if mandatory { "" } else { "_e32" }, nbytes);
        }

        // ---- 2. Bits [31:30] == 0b10: SOP1 / SOPC / SOPP / SOPK / SOP2 ----
        if w0 >> 30 == 0b10 {
            let hi9 = w0 >> 23;
            if hi9 == 0x17D || hi9 == 0x17E || (hi9 != 0x17F && w0 >> 28 != 0xB) {
                // SOP1 / SOPC / SOP2: up to two scalar sources sharing one literal
                let (t, opc, sdst, ssrc0, ssrc1) = match hi9 {
                    0x17D => (SOP1, (w0 >> 8) & 0xFF, (w0 >> 16) & 0x7F, w0 & 0xFF, 0),
                    0x17E => (SOPC, (w0 >> 16) & 0x7F, 0, w0 & 0xFF, (w0 >> 8) & 0xFF),
                    _ => (
                        SOP2,
                        (w0 >> 23) & 0x7F,
                        (w0 >> 16) & 0x7F,
                        w0 & 0xFF,
                        (w0 >> 8) & 0xFF,
                    ),
                };
                let e = entry!(self, t, opc);
                let mut lit = None;
                let mut nbytes = 4;
                if ssrc0 == 255 || (t != SOP1 && ssrc1 == 255) {
                    if !need(8) {
                        return Ok(None);
                    }
                    lit = Some(word(4));
                    nbytes = 8;
                }
                for &(f, nr) in &e.spec {
                    let w = width(nr, wave32);
                    let o = match f {
                        F::Sdst if t != SOPC => sc.s(sdst, w),
                        F::Ssrc0 => self.src(sc, ssrc0, w, lit),
                        F::Ssrc1 if t != SOP1 => self.src(sc, ssrc1, w, lit),
                        _ => continue,
                    };
                    sc.arg(o);
                }
                return done(&e.name, "", nbytes);
            }
            if hi9 == 0x17F {
                // SOPP
                let opc = (w0 >> 16) & 0x7F;
                let simm16 = w0 & 0xFFFF;
                let e = entry!(self, SOPP, opc);
                let name = e.name.as_str();
                if e.no_operand {
                } else if name == "s_waitcnt" {
                    waitcnt(sc, simm16);
                } else if name == "s_sendmsg" || name == "s_sendmsghalt" {
                    let o = self.sendmsg(sc, simm16);
                    sc.arg(o);
                } else if e.branch && name != "s_waitcnt_vscnt" {
                    sc.arg(Op::imm(s16(simm16)));
                } else {
                    sc.arg(Op::imm(simm16 as i64));
                }
                return done(name, "", 4);
            }
            // SOPK
            let opc = (w0 >> 23) & 0x1F;
            let sdst = (w0 >> 16) & 0x7F;
            let simm16 = w0 & 0xFFFF;
            let e = entry!(self, SOPK, opc);
            let name = e.name.as_str();
            let mut lit = None;
            let mut nbytes = 4;
            if self.sopk_literal.get(opc).is_some() {
                if !need(8) {
                    return Ok(None);
                }
                lit = Some(word(4));
                nbytes = 8;
            }
            if name == "s_setreg_b32" || name == "s_getreg_b32" {
                let hw = self.hwreg(sc, simm16);
                let reg = sc.s(sdst, 1);
                if name == "s_setreg_b32" {
                    sc.arg(hw);
                    sc.arg(reg);
                } else {
                    sc.arg(reg);
                    sc.arg(hw);
                }
                return done(name, "", 4);
            }
            if name == "s_setreg_imm32_b32" {
                let hw = self.hwreg(sc, simm16);
                sc.arg(hw);
                sc.arg(Op::imm(lit.unwrap_or(0) as i64));
                return done(name, "", nbytes);
            }
            if matches!(
                name,
                "s_waitcnt_vscnt" | "s_waitcnt_vmcnt" | "s_waitcnt_expcnt" | "s_waitcnt_lgkmcnt"
            ) {
                let o = sc.s(sdst, 1);
                sc.arg(o);
                sc.arg(Op::imm(simm16 as i64));
                return done(name, "", 4);
            }
            if name == "s_call_b64" {
                let o = sc.s(sdst, 2);
                sc.arg(o);
                sc.arg(Op::imm(s16(simm16)));
                return done(name, "", 4);
            }
            let signed = matches!(
                name,
                "s_movk_i32"
                    | "s_cmovk_i32"
                    | "s_cmpk_eq_i32"
                    | "s_cmpk_lg_i32"
                    | "s_cmpk_gt_i32"
                    | "s_cmpk_ge_i32"
                    | "s_cmpk_lt_i32"
                    | "s_cmpk_le_i32"
                    | "s_addk_i32"
                    | "s_mulk_i32"
            );
            for &(f, nr) in &e.spec {
                let w = width(nr, wave32);
                let o = match f {
                    F::Sdst => sc.s(sdst, w),
                    F::Simm16 => Op::imm(if signed { s16(simm16) } else { simm16 as i64 }),
                    _ => continue,
                };
                sc.arg(o);
            }
            return done(name, "", nbytes);
        }

        // ---- 3. Bits [31:26] == 0b11xxxx ----
        let hi6 = w0 >> 26;

        if hi6 == 0x32 {
            // VINTRP (32-bit)
            let opc = (w0 >> 16) & 3;
            let vdst = (w0 >> 18) & 0xFF;
            let attr = (w0 >> 10) & 0x3F;
            let chan = (w0 >> 8) & 3;
            let vsrc = w0 & 0xFF;
            let e = entry!(self, VINTRP, opc);
            let d = sc.v(vdst, 1);
            sc.arg(d);
            let p = if e.name == "v_interp_mov_f32" {
                interp_param(sc, vsrc)
            } else {
                sc.v(vsrc, 1)
            };
            sc.arg(p);
            let a = sc.tf(format_args!("attr{attr}.{}", xyzw(chan)));
            sc.arg(a);
            return done(&e.name, "", 4);
        }

        if !need(8) {
            return Ok(None);
        }
        let w1 = word(4);

        if hi6 == 0x33 {
            // VOP3P
            let opc = (w0 >> 16) & 0x7F;
            let vdst = w0 & 0xFF;
            let neg_hi = (w0 >> 8) & 7;
            let op_sel = (w0 >> 11) & 7;
            let op_sel_hi_2 = (w0 >> 14) & 1;
            let clamp = (w0 >> 15) & 1;
            let srcs = [w1 & 0x1FF, (w1 >> 9) & 0x1FF, (w1 >> 18) & 0x1FF];
            let op_sel_hi = ((w1 >> 27) & 3) | (op_sel_hi_2 << 2);
            let neg_lo = (w1 >> 29) & 7;
            let e = entry!(self, VOP3P, opc);
            let name = e.name.as_str();
            let nsrc = e.nsrc;
            if nsrc > 3 {
                return Err(());
            }
            let mut lit = None;
            let mut nbytes = 8;
            if srcs[..nsrc].contains(&255) {
                if !need(12) {
                    return Ok(None);
                }
                lit = Some(word(8));
                nbytes = 12;
            }
            let is_mix = name.starts_with("v_fma_mix");
            for &(f, nr) in &e.spec {
                let w = width(nr, wave32);
                let idx = match f {
                    F::Vdst => {
                        let o = sc.v(vdst, w);
                        sc.arg(o);
                        continue;
                    }
                    F::Src0 => 0,
                    F::Src1 => 1,
                    F::Src2 => 2,
                    _ => continue,
                };
                if idx >= nsrc {
                    return Err(()); // IndexError in Python
                }
                let o = self.src_mod(
                    sc,
                    srcs[idx],
                    w,
                    lit,
                    is_mix && neg_lo & (1 << idx) != 0,
                    is_mix && neg_hi & (1 << idx) != 0,
                );
                sc.arg(o);
            }
            let mask = (1u32 << nsrc) - 1;
            if op_sel & mask != 0 {
                bits_mod(sc, "op_sel", op_sel, nsrc);
            }
            let default_hi = if is_mix { 0 } else { mask };
            if op_sel_hi & mask != default_hi {
                bits_mod(sc, "op_sel_hi", op_sel_hi, nsrc);
            }
            if !is_mix && neg_lo & mask != 0 {
                bits_mod(sc, "neg_lo", neg_lo, nsrc);
            }
            if !is_mix && neg_hi & mask != 0 {
                bits_mod(sc, "neg_hi", neg_hi, nsrc);
            }
            if clamp != 0 {
                sc.ms("clamp");
            }
            return done(name, "", nbytes);
        }

        if hi6 == 0x35 {
            // VOP3 / VOP3B
            let opc = (w0 >> 16) & 0x3FF;
            let vdst = w0 & 0xFF;
            let clamp = (w0 >> 15) & 1;
            let src0 = w1 & 0x1FF;
            let src1 = (w1 >> 9) & 0x1FF;
            let src2 = (w1 >> 18) & 0x1FF;
            let omod = (w1 >> 27) & 3;
            let neg = (w1 >> 29) & 7;
            let is_vop3b = self.tables[VOP3B].contains(opc);
            let e = entry!(self, if is_vop3b { VOP3B } else { VOP3 }, opc);
            let name: &str = if opc == 0x140 && gfx101 {
                "v_mad_legacy_f32"
            } else {
                e.name.as_str()
            };
            let sdst = (w0 >> 8) & 0x7F;
            let abs_bits = if is_vop3b { 0 } else { (w0 >> 8) & 7 };
            let op_sel = if is_vop3b { 0 } else { (w0 >> 11) & 0xF };
            let nsrc = e.nsrc;
            let srcs = [src0, src1, src2];
            let mut lit = None;
            let mut nbytes = 8;
            if srcs[..nsrc.min(3)].contains(&255) {
                if !need(12) {
                    return Ok(None);
                }
                lit = Some(word(8));
                nbytes = 12;
            }
            let has_e32 = (opc < 0x100 && self.tables[VOPC].contains(opc))
                || ((0x100..0x140).contains(&opc) && self.tables[VOP2].contains(opc - 0x100))
                || ((0x180..0x200).contains(&opc) && self.tables[VOP1].contains(opc - 0x180));
            let suffix = if (has_e32 && !(name.starts_with("v_mad_") || name.starts_with("v_fma_")))
                || opc == 0x100 + 0x06
                || opc == 0x100 + 0x1F
            {
                "_e64"
            } else {
                ""
            };
            let is_interp_vop3 = matches!(opc, 0x200 | 0x201 | 0x202 | 0x342 | 0x343 | 0x35A);
            for &(f, nr) in &e.spec {
                let w = width(nr, wave32);
                let o = match f {
                    F::Vdst => {
                        if opc < 0x100 {
                            if name.starts_with("v_cmpx_") {
                                continue;
                            }
                            sc.s(vdst & 0x7F, wv)
                        } else if name == "v_readlane_b32" || name == "v_readfirstlane_b32" {
                            sc.s(vdst & 0x7F, 1)
                        } else {
                            sc.v(vdst, w)
                        }
                    }
                    F::Sdst => sc.s(sdst, wv),
                    F::Src0 => {
                        if is_interp_vop3 {
                            let a = sc.tf(format_args!(
                                "attr{}.{}",
                                src0 & 0x3F,
                                xyzw((src0 >> 6) & 3)
                            ));
                            if matches!(opc, 0x342 | 0x343 | 0x35A) && (src0 >> 8) & 1 != 0 {
                                sc.ms("high");
                            }
                            a
                        } else {
                            self.src_mod(sc, src0, w, lit, neg & 1 != 0, abs_bits & 1 != 0)
                        }
                    }
                    F::Src1 => {
                        if opc == 0x202 {
                            interp_param(sc, src1)
                        } else {
                            self.src_mod(sc, src1, w, lit, neg & 2 != 0, abs_bits & 2 != 0)
                        }
                    }
                    F::Src2 => self.src_mod(sc, src2, w, lit, neg & 4 != 0, abs_bits & 4 != 0),
                    _ => continue,
                };
                sc.arg(o);
            }
            if op_sel != 0 {
                if nsrc > 3 {
                    return Err(());
                }
                sc.ms("op_sel:[");
                for i in 0..nsrc {
                    if i > 0 {
                        sc.mods.push(',');
                    }
                    sc.mods.push(if (op_sel >> i) & 1 != 0 { '1' } else { '0' });
                }
                if opc >= 0x100 {
                    if nsrc > 0 {
                        sc.mods.push(',');
                    }
                    sc.mods.push(if (op_sel >> 3) & 1 != 0 { '1' } else { '0' });
                }
                sc.mods.push(']');
            }
            if clamp != 0 {
                sc.ms("clamp");
            }
            self.omod_mod(sc, omod);
            return done(name, suffix, nbytes);
        }

        if hi6 == 0x36 {
            // DS
            let opc = (w0 >> 18) & 0xFF;
            let offset0 = w0 & 0xFF;
            let offset1 = (w0 >> 8) & 0xFF;
            let gds = (w0 >> 17) & 1;
            let addr = w1 & 0xFF;
            let data0 = (w1 >> 8) & 0xFF;
            let data1 = (w1 >> 16) & 0xFF;
            let vdst = (w1 >> 24) & 0xFF;
            let e = entry!(self, DS, opc);
            let name = e.name.as_str();
            for &(f, nr) in &e.spec {
                let w = width(nr, wave32);
                let o = match f {
                    F::Vdst => sc.v(vdst, w),
                    F::Addr => sc.v(addr, w),
                    F::Data0 => sc.v(data0, w),
                    F::Data1 => sc.v(data1, w),
                    _ => continue,
                };
                sc.arg(o);
            }
            if name == "ds_swizzle_b32" {
                ds_swizzle(sc, offset0 | (offset1 << 8));
            } else if name.contains("2_") || name.contains("2st64_") {
                if offset0 != 0 {
                    sc.mf(format_args!("offset0:{offset0}"));
                }
                if offset1 != 0 {
                    sc.mf(format_args!("offset1:{offset1}"));
                }
            } else {
                let off16 = offset0 | (offset1 << 8);
                if off16 != 0 {
                    sc.mf(format_args!("offset:{off16}"));
                }
            }
            if gds != 0 {
                sc.ms("gds");
            }
            return done(name, "", 8);
        }

        if hi6 == 0x37 {
            // FLAT / SCRATCH / GLOBAL
            let seg = (w0 >> 14) & 3;
            let opc = (w0 >> 18) & 0x7F;
            let off12 = w0 & 0xFFF;
            let dlc = (w0 >> 12) & 1;
            let lds = (w0 >> 13) & 1;
            let glc = (w0 >> 16) & 1;
            let slc = (w0 >> 17) & 1;
            let addr = w1 & 0xFF;
            let data = (w1 >> 8) & 0xFF;
            let saddr = (w1 >> 16) & 0x7F;
            let vdst = (w1 >> 24) & 0xFF;
            let t = match seg {
                0 => FLAT,
                1 => SCRATCH,
                2 => GLOBAL,
                _ => return Ok(None),
            };
            let e = entry!(self, t, opc);
            let is_atomic = e.name.contains("_atomic_");
            let has_saddr = seg == 1 || seg == 2;
            let saddr_off = saddr == 0x7D || saddr == 0x7F;
            for &(f, nr) in &e.spec {
                let w = width(nr, wave32);
                let o = match f {
                    F::Vdst => {
                        // Atomics without glc return nothing; SCRATCH/GLOBAL direct-to-LDS loads write LDS.
                        if (is_atomic && glc == 0) || (lds != 0 && has_saddr) {
                            continue;
                        }
                        sc.v(vdst, w)
                    }
                    // SCRATCH addresses through saddr alone (vaddr "off") unless saddr is null.
                    F::Addr if seg == 1 && saddr != 0x7D => Op::st("off"),
                    F::Addr => sc.v(
                        addr,
                        if seg == 0 || (seg == 2 && saddr_off) {
                            2
                        } else {
                            1
                        },
                    ),
                    F::Data => sc.v(data, w),
                    F::Saddr if has_saddr => {
                        if saddr_off {
                            Op::st("off")
                        } else {
                            sc.s(saddr, if seg == 2 { 2 } else { 1 })
                        }
                    }
                    _ => continue,
                };
                sc.arg(o);
            }
            let off_s = if has_saddr { s12(off12) } else { off12 as i64 };
            if off_s != 0 {
                sc.mf(format_args!("offset:{off_s}"));
            }
            for (bit, m) in [(glc, "glc"), (slc, "slc"), (dlc, "dlc"), (lds, "lds")] {
                if bit != 0 {
                    sc.ms(m);
                }
            }
            return done(&e.name, "", 8);
        }

        if hi6 == 0x38 || hi6 == 0x3A {
            // MUBUF / MTBUF
            let is_mtbuf = hi6 == 0x3A;
            let opc = if is_mtbuf {
                ((w0 >> 16) & 0x7) | (((w1 >> 21) & 1) << 3)
            } else {
                (w0 >> 18) & 0xFF
            };
            let off12 = w0 & 0xFFF;
            let offen = (w0 >> 12) & 1;
            let idxen = (w0 >> 13) & 1;
            let glc = (w0 >> 14) & 1;
            let dlc = (w0 >> 15) & 1;
            let lds = (w0 >> 16) & 1;
            let fmt_code = (w0 >> 19) & 0x7F;
            let vaddr = w1 & 0xFF;
            let vdata = (w1 >> 8) & 0xFF;
            let srsrc = ((w1 >> 16) & 0x1F) * 4;
            let slc = (w1 >> 22) & 1;
            let tfe = (w1 >> 23) & 1;
            let soff = (w1 >> 24) & 0xFF;
            let e = entry!(self, if is_mtbuf { MTBUF } else { MUBUF }, opc);
            for &(f, nr) in &e.spec {
                let w = width(nr, wave32);
                let o = match f {
                    F::Vdata => sc.v(vdata, w),
                    F::Vaddr => {
                        if offen == 0 && idxen == 0 {
                            Op::st("off")
                        } else {
                            sc.v(vaddr, if offen != 0 && idxen != 0 { 2 } else { 1 })
                        }
                    }
                    F::Srsrc => sc.s(srsrc, 4),
                    F::Soffset => self.src(sc, soff, 1, None),
                    _ => continue,
                };
                sc.arg(o);
            }
            if is_mtbuf && fmt_code != 1 {
                match self.formats.get(fmt_code) {
                    Some(short) if !short.is_empty() => sc.mf(format_args!("format:[{short}]")),
                    _ => sc.mf(format_args!("format:{fmt_code}")),
                }
            }
            if idxen != 0 {
                sc.ms("idxen");
            }
            if offen != 0 {
                sc.ms("offen");
            }
            if off12 != 0 {
                sc.mf(format_args!("offset:{off12}"));
            }
            for (bit, m) in [(glc, "glc"), (slc, "slc"), (dlc, "dlc")] {
                if bit != 0 {
                    sc.ms(m);
                }
            }
            if !is_mtbuf && lds != 0 {
                sc.ms("lds");
            }
            if tfe != 0 {
                sc.ms("tfe");
            }
            return done(&e.name, "", 8);
        }

        if hi6 == 0x3C {
            // MIMG
            let nsa_dwords = ((w0 >> 1) & 3) as usize;
            let dim = (w0 >> 3) & 7;
            let dlc = (w0 >> 7) & 1;
            let dmask = (w0 >> 8) & 0xF;
            let unorm = (w0 >> 12) & 1;
            let glc = (w0 >> 13) & 1;
            let r128 = (w0 >> 15) & 1;
            let tfe = (w0 >> 16) & 1;
            let lwe = (w0 >> 17) & 1;
            let opc = ((w0 >> 18) & 0x7F) | ((w0 & 1) << 7);
            let vaddr0 = w1 & 0xFF;
            let vdata = (w1 >> 8) & 0xFF;
            let srsrc = ((w1 >> 16) & 0x1F) * 4;
            let ssamp = ((w1 >> 21) & 0x1F) * 4;
            let slc = (w0 >> 25) & 1;
            let a16 = (w1 >> 30) & 1;
            let d16 = (w1 >> 31) & 1;
            let e = entry!(self, MIMG, opc);
            let name = e.name.as_str();
            let nbytes = 8 + 4 * nsa_dwords;
            if !need(nbytes) {
                return Ok(None);
            }
            let n_addr = self.mimg_addr_count(name, dim, a16 != 0);
            let vaddr_op = if nsa_dwords != 0 {
                let take = (n_addr - 1).max(0) as usize;
                let start = sc.begin();
                let _ = write!(sc.text, "[v{vaddr0}");
                for &b in code[pc + 8..pc + nbytes].iter().take(take) {
                    let _ = write!(sc.text, ",v{b}");
                }
                sc.text.push(']');
                sc.end(start)
            } else {
                if n_addr > i32::MAX as i64 || n_addr < i32::MIN as i64 {
                    return Err(());
                }
                sc.v(vaddr0, n_addr as i32)
            };
            let n_data = mimg_data_count(name, dmask, d16 != 0, tfe != 0);
            let is_bvh = name.starts_with("image_bvh");
            let rsrc_w = if r128 != 0 || is_bvh { 4 } else { 8 };
            for &(f, _) in &e.spec {
                let o = match f {
                    F::Vdata => sc.v(vdata, n_data as i32),
                    F::Vaddr => vaddr_op,
                    F::Srsrc => sc.s(srsrc, rsrc_w),
                    F::Ssamp => sc.s(ssamp, 4),
                    _ => continue,
                };
                sc.arg(o);
            }
            if dmask != 0 && !is_bvh {
                sc.mf(format_args!("dmask:0x{dmask:x}"));
            }
            if !is_bvh {
                let d = self
                    .mimg_dims
                    .get(dim)
                    .map(|s| s.as_str())
                    .unwrap_or("SQ_RSRC_IMG_2D");
                sc.mf(format_args!("dim:{d}"));
                if unorm != 0 {
                    sc.ms("unorm");
                }
            }
            if glc != 0 {
                sc.ms("glc");
            }
            if slc != 0 {
                sc.ms("slc");
            }
            if r128 != 0 && !is_bvh {
                sc.ms("r128");
            }
            for (bit, m) in [
                (tfe, "tfe"),
                (lwe, "lwe"),
                (a16, "a16"),
                (d16, "d16"),
                (dlc, "dlc"),
            ] {
                if bit != 0 {
                    sc.ms(m);
                }
            }
            return done(name, "", nbytes);
        }

        if hi6 == 0x3D {
            // SMEM
            let sbase = (w0 & 0x3F) * 2;
            let sdata = (w0 >> 6) & 0x7F;
            let dlc = (w0 >> 14) & 1;
            let glc = (w0 >> 16) & 1;
            let opc = (w0 >> 18) & 0xFF;
            let off21 = w1 & 0x1FFFFF;
            let soff = (w1 >> 25) & 0x7F;
            let e = entry!(self, SMEM, opc);
            let mut has_offset = false;
            for &(f, nr) in &e.spec {
                let w = width(nr, wave32);
                let o = match f {
                    F::Sdata => sc.s(sdata, w),
                    F::Sbase => sc.s(sbase, w),
                    F::Offset | F::Soffset => {
                        has_offset = true;
                        continue;
                    }
                    _ => continue,
                };
                sc.arg(o);
            }
            if has_offset {
                let imm_val = if e.name.contains("buffer") {
                    off21 as i64
                } else {
                    s21(off21)
                };
                if soff == 125 {
                    sc.arg(if imm_val == 0 {
                        Op::st("null")
                    } else {
                        Op::imm(imm_val)
                    });
                } else {
                    let o = sc.s(soff, 1);
                    sc.arg(o);
                    if imm_val > 0 {
                        sc.mf(format_args!("offset:0x{imm_val:x}"));
                    } else if imm_val < 0 {
                        sc.mf(format_args!("offset:-0x{:x}", -imm_val));
                    }
                }
            }
            if glc != 0 {
                sc.ms("glc");
            }
            if dlc != 0 {
                sc.ms("dlc");
            }
            return done(&e.name, "", 8);
        }

        if hi6 == 0x3E {
            // EXP
            let en = w0 & 0xF;
            let tgt = (w0 >> 4) & 0x3F;
            let compr = (w0 >> 10) & 1;
            let done_bit = (w0 >> 11) & 1;
            let vm = (w0 >> 12) & 1;
            let vsrcs = [
                w1 & 0xFF,
                (w1 >> 8) & 0xFF,
                (w1 >> 16) & 0xFF,
                (w1 >> 24) & 0xFF,
            ];
            let t = match tgt {
                0..=7 => sc.tf(format_args!("mrt{tgt}")),
                8 => Op::st("mrtz"),
                9 => Op::st("null"),
                12..=16 => sc.tf(format_args!("pos{}", tgt - 12)),
                20 => Op::st("prim"),
                32..=63 => sc.tf(format_args!("param{}", tgt - 32)),
                _ => sc.tf(format_args!("tgt{tgt}")),
            };
            sc.arg(t);
            if compr != 0 {
                for (pair, &v) in vsrcs.iter().take(2).enumerate() {
                    let o = if (en >> (2 * pair)) & 3 != 0 {
                        sc.v(v, 1)
                    } else {
                        Op::st("off")
                    };
                    sc.arg(o);
                    sc.arg(o);
                }
            } else {
                for (i, &v) in vsrcs.iter().enumerate() {
                    let o = if (en >> i) & 1 != 0 {
                        sc.v(v, 1)
                    } else {
                        Op::st("off")
                    };
                    sc.arg(o);
                }
            }
            for (bit, m) in [(done_bit, "done"), (compr, "compr"), (vm, "vm")] {
                if bit != 0 {
                    sc.ms(m);
                }
            }
            return done("exp", "", 8);
        }

        Ok(None)
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Listing (`format_insn`, `_iter_insns`, `disasm_bytes`)

const HEX: &[u8; 16] = b"0123456789abcdef";

fn is_op(ins: &Ins, full: &str) -> bool {
    full.len() == ins.base.len() + ins.suffix.len()
        && full.starts_with(ins.base)
        && full.ends_with(ins.suffix)
}

impl Ctx {
    /// `format_insn(ins)` plus the newline `disasm_bytes` adds, appended to `out`.
    fn format_insn(&self, ins: &Ins, sc: &Sc, enc: &[u8], out: &mut String) {
        let hex_imm = is_op(ins, "s_ttracedata_imm");
        out.push('\t');
        let head_start = out.len();
        out.push_str(ins.base);
        out.push_str(ins.suffix);
        if !sc.args.is_empty() {
            out.push(' ');
            let exp = is_op(ins, "exp");
            for (i, a) in sc.args.iter().enumerate() {
                if i > 0 {
                    out.push_str(if exp && i == 1 { " " } else { ", " });
                }
                self.put_operand(&sc.text, a, hex_imm, out);
            }
            if exp && sc.args.len() == 1 {
                out.push(' '); // f"{op} {args[0]} " + ", ".join([])
            }
        }
        if !sc.mods.is_empty() {
            out.push(' ');
            out.push_str(&sc.mods);
        }
        // Every name and operand is ASCII (Ctx::parse refuses anything else), so bytes count characters.
        for _ in out.len() - head_start..40 {
            out.push(' ');
        }
        out.push_str("; encoding: [");
        for (i, b) in enc.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push_str("0x");
            out.push(HEX[(b >> 4) as usize] as char);
            out.push(HEX[(b & 15) as usize] as char);
        }
        out.push_str("]\n");
    }

    /// `disasm_bytes(code, mcpu=..., wave32=...)`: `Some((listing, invalid_words))`, or None to refuse.
    pub fn disasm(&self, code: &[u8], wave32: bool, gfx101: bool) -> Option<(String, usize)> {
        let mut out = String::with_capacity(code.len() * 20 + 64);
        let mut sc = Sc::default();
        let mut invalid = 0usize;
        let mut off = 0usize;
        while off < code.len() {
            if off + 4 > code.len() {
                invalid += 1;
                off += 4;
                continue;
            }
            sc.clear();
            match self.decode_one(&mut sc, code, off, wave32, gfx101) {
                Err(()) => return None,
                Ok(None) => {
                    invalid += 1;
                    off += 4;
                }
                Ok(Some(ins)) => {
                    self.format_insn(&ins, &sc, &code[off..off + ins.nbytes], &mut out);
                    off += ins.nbytes;
                }
            }
        }
        Some((out, invalid))
    }
}

// ---------------------------------------------------------------------------------------------------------------
// C ABI

/// NUL-terminated `FINGERPRINT`.
#[no_mangle]
pub extern "C" fn rdna2_native_fingerprint() -> *const u8 {
    FINGERPRINT_C.as_ptr()
}

/// Parse the table blob; null when it does not parse. The context lives until `rdna2_native_release`.
///
/// # Safety
/// `blob` must point to `len` readable bytes.
#[no_mangle]
pub unsafe extern "C" fn rdna2_native_init(blob: *const u8, len: usize) -> *mut Ctx {
    if blob.is_null() {
        return std::ptr::null_mut();
    }
    let bytes = std::slice::from_raw_parts(blob, len);
    match std::str::from_utf8(bytes).ok().and_then(Ctx::parse) {
        Some(ctx) => Box::into_raw(Box::new(ctx)),
        None => std::ptr::null_mut(),
    }
}

/// Free a context from `rdna2_native_init`.
///
/// # Safety
/// `ctx` must come from `rdna2_native_init` and not be used afterwards.
#[no_mangle]
pub unsafe extern "C" fn rdna2_native_release(ctx: *mut Ctx) {
    if !ctx.is_null() {
        drop(Box::from_raw(ctx));
    }
}

/// Disassemble `code`. On success (0) `*out` and `*out_len` hold a UTF-8 listing to pass to `rdna2_native_free`,
/// and `*invalid` the invalid word count; -1 means "decode in Python instead".
///
/// # Safety
/// `ctx` from `rdna2_native_init`; `code` readable for `len` bytes; the out pointers writable.
#[no_mangle]
pub unsafe extern "C" fn rdna2_native_disasm(
    ctx: *const Ctx,
    code: *const u8,
    len: usize,
    wave32: i32,
    gfx101: i32,
    out: *mut *mut u8,
    out_len: *mut usize,
    invalid: *mut usize,
) -> i32 {
    if ctx.is_null()
        || out.is_null()
        || out_len.is_null()
        || invalid.is_null()
        || (code.is_null() && len != 0)
    {
        return -1;
    }
    let code = if len == 0 {
        &[][..]
    } else {
        std::slice::from_raw_parts(code, len)
    };
    let result = std::panic::catch_unwind(|| (*ctx).disasm(code, wave32 != 0, gfx101 != 0));
    match result {
        Ok(Some((text, bad))) => {
            let boxed = text.into_bytes().into_boxed_slice();
            *out_len = boxed.len();
            *out = Box::into_raw(boxed) as *mut u8;
            *invalid = bad;
            0
        }
        _ => -1,
    }
}

/// Free a listing from `rdna2_native_disasm`.
///
/// # Safety
/// `ptr` and `len` must be exactly what `rdna2_native_disasm` returned.
#[no_mangle]
pub unsafe extern "C" fn rdna2_native_free(ptr: *mut u8, len: usize) {
    if !ptr.is_null() {
        drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(ptr, len)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> Ctx {
        let blob = "T\tSOPP\t1\ts_endpgm\t\n\
                    T\tSOPP\t0\ts_nop\tSIMM16:1\n\
                    T\tSOPP\t12\ts_waitcnt\tSIMM16:1\n\
                    T\tSOPP\t2\ts_branch\tSIMM16:1\n\
                    T\tSOP1\t3\ts_mov_b32\tSDST:1,SSRC0:1\n\
                    T\tVOP2\t3\tv_add_f32\tVDST:1,SRC0:1,VSRC1:1\n\
                    S\tSOPP_NO_OPERAND\ts_endpgm\n\
                    S\tSOPP_BRANCHES\ts_branch\n\
                    M\tINLINE_FLOAT\t242\t1.0\n";
        Ctx::parse(blob).expect("blob parses")
    }

    fn words(ws: &[u32]) -> Vec<u8> {
        ws.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    #[test]
    fn scalar_program() {
        let code = words(&[0xBF8C0070, 0xBE8403FF, 0x12345678, 0xBF82FFFC, 0xBF810000]);
        let (text, invalid) = ctx().disasm(&code, true, false).unwrap();
        assert_eq!(invalid, 0);
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(
            lines[0],
            format!(
                "\t{:<40}; encoding: [0x70,0x00,0x8c,0xbf]",
                "s_waitcnt vmcnt(0), lgkmcnt(0)"
            )
        );
        assert_eq!(
            lines[1],
            format!(
                "\t{:<40}; encoding: [0xff,0x03,0x84,0xbe,0x78,0x56,0x34,0x12]",
                "s_mov_b32 s4, 0x12345678"
            )
        );
        assert_eq!(
            lines[2],
            format!("\t{:<40}; encoding: [0xfc,0xff,0x82,0xbf]", "s_branch -4")
        );
        assert_eq!(
            lines[3],
            format!("\t{:<40}; encoding: [0x00,0x00,0x81,0xbf]", "s_endpgm")
        );
    }

    #[test]
    fn vop2_inline_float_and_vcc_tuple() {
        // v_add_f32_e32 v1, 1.0, v2
        let code = words(&[(3 << 25) | (1 << 17) | (2 << 9) | 242]);
        let (text, _) = ctx().disasm(&code, true, false).unwrap();
        assert!(text.starts_with("\tv_add_f32_e32 v1, 1.0, v2 "));
        let mut sc = Sc::default();
        let c = ctx();
        for (code, n, want) in [
            (104, 4, "[s104,s105,vcc_lo,vcc_hi]"),
            (110, 2, "ttmp[2:3]"),
            (106, 2, "vcc"),
        ] {
            let op = sc.s(code, n);
            let mut out = String::new();
            c.put_operand(&sc.text, &op, false, &mut out);
            assert_eq!(out, want);
        }
    }

    #[test]
    fn invalid_and_truncated_words() {
        let mut code = words(&[0xBF810000, 0xFFFFFFFF, 0xBE8403FF]);
        code.extend_from_slice(&[1, 2]);
        let (text, invalid) = ctx().disasm(&code, true, false).unwrap();
        // 0xFFFFFFFF is no encoding, the s_mov_b32 lacks its literal, and two bytes trail.
        assert_eq!(invalid, 3);
        assert_eq!(text.lines().count(), 1);
    }

    #[test]
    fn bad_blob_is_refused() {
        assert!(Ctx::parse("T\tNOPE\t1\tx\t\n").is_none());
        assert!(Ctx::parse("X\n").is_none());
        assert!(Ctx::parse("T\tSOPP\t99999\tx\t\n").is_none());
    }

    #[test]
    fn swizzle_and_dpp_text() {
        let mut sc = Sc::default();
        ds_swizzle(&mut sc, 0x041F);
        ds_swizzle(&mut sc, 0x80E4);
        dpp16_ctrl(&mut sc, 0x141);
        dpp16_ctrl(&mut sc, 0x1E4);
        assert_eq!(
            sc.mods,
            "offset:swizzle(SWAP,1) offset:swizzle(QUAD_PERM,0,1,2,3) row_half_mirror dpp_ctrl:0x1e4"
        );
    }
}
