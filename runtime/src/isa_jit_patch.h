// Ahead-of-time native AVX1 + SSE4.2 + x86-64 stub compiler for BMI1, BMI2, MOVBE, FMA3 and AVX2 sites on
// pre-Haswell AVX1 hosts (Intel Sandy Bridge and Ivy Bridge).
//
// Emulating every VEX BMI1/BMI2/AVX2/FMA3 instruction through #UD (STATUS_ILLEGAL_INSTRUCTION ->
// KiUserExceptionDispatcher) costs ~4,000-6,000 cycles per instruction, stalling hot decompression and SIMD loops.
// Before an image's sections are marked read/execute, patch_image() rewrites verified code sites in two ways:
//   1. Exact same-length in-place VEX rewrites (zero stub overhead) where an AVX1 instruction performs the identical
//      architectural bit operation with no SIMD FP exceptions:
//        - vpand / vpandn / vpor / vpxor (256b)  -> vandps / vandnps / vorps / vxorps (256b)
//        - vextracti128 / vinserti128 / vperm2i128 -> vextractf128 / vinsertf128 / vperm2f128
//        - vbroadcasti128 m128                   -> vbroadcastf128 m128
//        - vpblendd (128b/256b)                  -> vblendps (128b/256b)
//        - vpbroadcastd m32 (128b/256b)          -> vbroadcastss m32 (128b/256b)
//        - vpbroadcastq ymm, m64                 -> vbroadcastsd ymm, m64
//   2. Near `jmp rel32` trampolines to native machine-code stubs within +-2 GiB of the module:
//        - >=5-byte BMI1/BMI2/MOVBE sites -> native x86-64 GPR sequences preserving/setting exact architectural EFLAGS
//        - >=5-byte AVX2 & FMA3 sites     -> native AVX1 (YMM + 2x128-bit XMM) sequences preserving all GPRs & EFLAGS
//        - 4-byte 0xC5 AVX2 sites         -> fused with the immediately following instruction when no direct branch,
//                                            relocation, unwind scope, export or RVA table anywhere in the image
//                                            targets the interior boundary.
//
// Any unpatched site (e.g. F16C, vpmaskmov*, pdep/pext, addr32/FS/GS overrides, or unfusable 4-byte sites) remains
// untouched and continues to execute through bmi_emulation::handler / avx2_emulation::handler on #UD.
// Set LEONIDA_ISA_PATCH=0 to disable.
#pragma once

#ifdef _WIN32
#include <windows.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

#include "avx2_emulation.h"
#include "bmi_emulation.h"

namespace rt::isa_jit_patch {

inline bool enabled() {
    char v[8]{};
    std::size_t n = 0;
    return !(getenv_s(&n, v, "LEONIDA_ISA_PATCH") == 0 && n == 2 && v[0] == '0');
}

struct Stats {
    std::size_t inplace_sites = 0;
    std::size_t bmi_stubs = 0;
    std::size_t avx2_stubs = 0;
    std::size_t avx2_fused_4b = 0;
    std::size_t fma3_stubs = 0;
    uint8_t* stubs = nullptr;
    std::size_t stub_bytes = 0;
    std::size_t stub_reserved = 0;
};

namespace detail {

inline void emit_u8(std::vector<uint8_t>& o, uint8_t b) { o.push_back(b); }
inline void emit_i8(std::vector<uint8_t>& o, int8_t v) { o.push_back(uint8_t(v)); }
inline void emit_i32(std::vector<uint8_t>& o, int32_t v) {
    uint8_t b[4];
    std::memcpy(b, &v, 4);
    o.insert(o.end(), b, b + 4);
}
inline void emit_u32(std::vector<uint8_t>& o, uint32_t v) {
    uint8_t b[4];
    std::memcpy(b, &v, 4);
    o.insert(o.end(), b, b + 4);
}

inline void emit_jmp_rel32(std::vector<uint8_t>& o, uint64_t stub_base_va, uint64_t target_va) {
    emit_u8(o, 0xE9);
    const int64_t rel = int64_t(target_va) - int64_t(stub_base_va + o.size() + 4);
    emit_i32(o, int32_t(rel));
}

inline void emit_push_gpr(std::vector<uint8_t>& o, uint8_t r) {
    if (r >= 8) emit_u8(o, 0x41);
    emit_u8(o, uint8_t(0x50 + (r & 7)));
}

inline void emit_pop_gpr(std::vector<uint8_t>& o, uint8_t r) {
    if (r >= 8) emit_u8(o, 0x41);
    emit_u8(o, uint8_t(0x58 + (r & 7)));
}

inline void emit_rex(std::vector<uint8_t>& o, bool w, uint8_t reg, uint8_t idx, uint8_t rm_base, bool force = false) {
    const uint8_t rex =
        uint8_t(0x40 | (w ? 8 : 0) | ((reg >= 8) ? 4 : 0) | ((idx >= 8) ? 2 : 0) | ((rm_base >= 8) ? 1 : 0));
    if (rex != 0x40 || force) emit_u8(o, rex);
}

inline void emit_mov_rr(std::vector<uint8_t>& o, bool w64, uint8_t dst, uint8_t src) {
    // 89 /r: mov r/m, reg (dst = r/m, src = reg)
    emit_rex(o, w64, src, 0, dst);
    emit_u8(o, 0x89);
    emit_u8(o, uint8_t(0xC0 | ((src & 7) << 3) | (dst & 7)));
}

inline void emit_mov_imm32(std::vector<uint8_t>& o, uint8_t dst, uint32_t imm) {
    if (dst >= 8) emit_u8(o, 0x41);
    emit_u8(o, uint8_t(0xB8 + (dst & 7)));
    emit_u32(o, imm);
}

inline uint8_t scale_bits(uint8_t scale) {
    return scale == 8 ? 3 : scale == 4 ? 2 : scale == 2 ? 1 : 0;
}

// Emits ModRM + optional SIB + disp for a memory operand (already preceded by opcode byte), adjusting [rsp + disp] by
// `stack_adj` and re-anchoring [rip + disp] from `orig_next_va` to `stub_base_va + o.size() + 5 + trailing_imm_bytes`.
inline bool emit_mem_modrm(std::vector<uint8_t>& o, uint8_t reg_low3, int8_t base, int8_t index, uint8_t scale,
                           int32_t disp, int32_t stack_adj, uint64_t orig_next_va, uint64_t stub_base_va,
                           std::size_t trailing_imm_bytes = 0) {
    if (base == -2) {
        const uint64_t target_va = orig_next_va + uint64_t(int64_t(disp));
        const uint64_t stub_next_va = stub_base_va + o.size() + 1 + 4 + trailing_imm_bytes;
        const int64_t rel = int64_t(target_va) - int64_t(stub_next_va);
        if (rel < INT32_MIN || rel > INT32_MAX) return false;
        emit_u8(o, uint8_t(((reg_low3 & 7) << 3) | 5));
        emit_i32(o, int32_t(rel));
        return true;
    }

    const int64_t adj64 = int64_t(disp) + (base == 4 ? int64_t(stack_adj) : 0);
    if (adj64 < INT32_MIN || adj64 > INT32_MAX) return false;
    const int32_t adj = int32_t(adj64);

    if (base == -1) {
        emit_u8(o, uint8_t(((reg_low3 & 7) << 3) | 4));
        const uint8_t idx_bits = (index >= 0) ? uint8_t(index & 7) : 4u;
        emit_u8(o, uint8_t((scale_bits(scale) << 6) | (idx_bits << 3) | 5u));
        emit_i32(o, adj);
        return true;
    }

    const uint8_t base_low = uint8_t(base & 7);
    const uint8_t mod = (adj == 0 && base_low != 5) ? 0u : (adj >= -128 && adj <= 127) ? 1u : 2u;
    if (index >= 0 || base_low == 4) {
        emit_u8(o, uint8_t((mod << 6) | ((reg_low3 & 7) << 3) | 4u));
        const uint8_t idx_bits = (index >= 0) ? uint8_t(index & 7) : 4u;
        emit_u8(o, uint8_t((scale_bits(scale) << 6) | (idx_bits << 3) | base_low));
    } else {
        emit_u8(o, uint8_t((mod << 6) | ((reg_low3 & 7) << 3) | base_low));
    }
    if (mod == 1) emit_i8(o, int8_t(adj));
    else if (mod == 2) emit_i32(o, adj);
    return true;
}

inline uint8_t pick_scratch_gpr(uint32_t used_mask) {
    // Prefer volatile/common regs, never RSP (4).
    constexpr uint8_t order[] = {0, 1, 2, 8, 9, 10, 11, 3, 5, 6, 7, 12, 13, 14, 15};
    for (uint8_t r : order) {
        if (((used_mask >> r) & 1u) == 0) return r;
    }
    return 0xFF;
}

inline uint8_t pick_scratch_ymm(uint32_t used_mask) {
    for (uint8_t r = 0; r < 16; ++r) {
        if (((used_mask >> r) & 1u) == 0) return r;
    }
    return 0xFF;
}

// Emits `mov dst_gpr, r/m` (32-bit or 64-bit) for a BMI/MOVBE decoded operand.
inline bool emit_bmi_load_rm(std::vector<uint8_t>& o, const bmi_emulation::Decoded& d, uint8_t dst_gpr,
                             int32_t stack_adj, uint64_t orig_next_va, uint64_t stub_base_va) {
    const bool w64 = d.size == 8;
    if (!d.mem) {
        emit_mov_rr(o, w64, dst_gpr, d.rm);
        return true;
    }
    const uint8_t idx_r = d.index >= 0 ? uint8_t(d.index) : 0u;
    const uint8_t base_r = d.base >= 0 ? uint8_t(d.base) : 0u;
    emit_rex(o, w64, dst_gpr, idx_r, base_r);
    emit_u8(o, 0x8B);  // mov r, r/m
    return emit_mem_modrm(o, dst_gpr & 7, d.base, d.index, d.scale, d.disp, stack_adj, orig_next_va, stub_base_va);
}

// Emits a native x86-64 machine-code body (without trailing jmp) for a BMI1/BMI2/MOVBE instruction.
inline bool emit_bmi_body(std::vector<uint8_t>& o, const bmi_emulation::Decoded& d, uint64_t orig_va,
                          uint64_t stub_base_va, const bmi_emulation::Host& host) {
    using bmi_emulation::Op;
    if (d.addr32 || d.seg != 0) return false;
    if (d.size == 2 && d.op == Op::movbe_load) {
        if (host.movbe || d.dst == 4 || !d.mem) return false;
        const uint64_t orig_next_va = orig_va + d.length;
        const uint8_t idx_r = d.index >= 0 ? uint8_t(d.index) : 0u;
        const uint8_t base_r = d.base >= 0 ? uint8_t(d.base) : 0u;
        emit_u8(o, 0x66);
        emit_rex(o, false, d.dst, idx_r, base_r);
        emit_u8(o, 0x8B);  // mov r16, [mem] (preserves dst[63:16])
        if (!emit_mem_modrm(o, d.dst & 7, d.base, d.index, d.scale, d.disp, 0, orig_next_va, stub_base_va))
            return false;
        emit_u8(o, 0x9C);  // pushfq
        emit_u8(o, 0x66);
        emit_rex(o, false, 0, 0, d.dst);
        emit_u8(o, 0xC1);
        emit_u8(o, uint8_t(0xC0 | (d.dst & 7)));  // rol r16, 8 (byte-swaps 16-bit register)
        emit_u8(o, 8);
        emit_u8(o, 0x9D);  // popfq
        return true;
    }
    if (d.size != 4 && d.size != 8) return false;

    // Skip instructions the host CPU already executes natively without #UD.
    switch (d.op) {
    case Op::andn: case Op::bextr: case Op::blsi: case Op::blsmsk: case Op::blsr:
        if (host.bmi1) return false;
        break;
    case Op::bzhi: case Op::mulx: case Op::rorx: case Op::sarx: case Op::shlx: case Op::shrx:
        if (host.bmi2) return false;
        break;
    case Op::movbe_load: case Op::movbe_store:
        if (host.movbe) return false;
        break;
    default:
        return false;  // pdep, pext, tzcnt, lzcnt handled elsewhere
    }

    const bool w64 = d.size == 8;
    const uint8_t bits = w64 ? 64 : 32;
    const uint64_t orig_next_va = orig_va + d.length;

    uint32_t used = (1u << 4);  // never RSP
    used |= (1u << d.dst) | (1u << d.dst2) | (1u << d.src1);
    if (!d.mem) used |= (1u << d.rm);
    else {
        if (d.base >= 0) used |= (1u << d.base);
        if (d.index >= 0) used |= (1u << d.index);
    }

    if (d.op != Op::movbe_store && d.dst == 4) return false;
    if (d.op == Op::mulx && d.dst2 == 4) return false;

    switch (d.op) {
    case Op::movbe_load: {
        if (!emit_bmi_load_rm(o, d, d.dst, 0, orig_next_va, stub_base_va)) return false;
        emit_rex(o, w64, 0, 0, d.dst);
        emit_u8(o, 0x0F);
        emit_u8(o, uint8_t(0xC8 + (d.dst & 7)));  // bswap dst (touches no flags)
        return true;
    }
    case Op::movbe_store: {
        const uint8_t tmp = pick_scratch_gpr(used);
        if (tmp == 0xFF) return false;
        emit_push_gpr(o, tmp);
        emit_mov_rr(o, w64, tmp, d.src1);
        emit_rex(o, w64, 0, 0, tmp);
        emit_u8(o, 0x0F);
        emit_u8(o, uint8_t(0xC8 + (tmp & 7)));  // bswap tmp
        const uint8_t idx_r = d.index >= 0 ? uint8_t(d.index) : 0u;
        const uint8_t base_r = d.base >= 0 ? uint8_t(d.base) : 0u;
        emit_rex(o, w64, tmp, idx_r, base_r);
        emit_u8(o, 0x89);  // mov [mem], tmp
        if (!emit_mem_modrm(o, tmp & 7, d.base, d.index, d.scale, d.disp, 8, orig_next_va, stub_base_va)) return false;
        emit_pop_gpr(o, tmp);
        return true;
    }
    case Op::rorx: {
        const uint8_t cnt = d.imm & (bits - 1u);
        if (!emit_bmi_load_rm(o, d, d.dst, 0, orig_next_va, stub_base_va)) return false;
        if (cnt != 0) {
            emit_u8(o, 0x9C);  // pushfq
            emit_rex(o, w64, 0, 0, d.dst);
            emit_u8(o, 0xC1);
            emit_u8(o, uint8_t(0xC8 | (d.dst & 7)));  // ror dst, imm8
            emit_u8(o, cnt);
            emit_u8(o, 0x9D);  // popfq
        }
        return true;
    }
    case Op::shlx:
    case Op::shrx:
    case Op::sarx: {
        const uint8_t shift_sub = (d.op == Op::shlx) ? 4u : (d.op == Op::shrx) ? 5u : 7u;
        if (d.src1 == 1 && d.dst != 1) {
            // Fast path: count is already in RCX/CL and dst != RCX.
            if (!emit_bmi_load_rm(o, d, d.dst, 0, orig_next_va, stub_base_va)) return false;
            emit_u8(o, 0x9C);  // pushfq
            emit_rex(o, w64, 0, 0, d.dst);
            emit_u8(o, 0xD3);
            emit_u8(o, uint8_t(0xC0 | (shift_sub << 3) | (d.dst & 7)));
            emit_u8(o, 0x9D);  // popfq
            return true;
        }
        const uint8_t tmp = pick_scratch_gpr(used | (1u << 1));
        if (tmp == 0xFF) return false;
        emit_push_gpr(o, tmp);
        if (!emit_bmi_load_rm(o, d, tmp, 8, orig_next_va, stub_base_va)) return false;
        emit_u8(o, 0x51);  // push rcx
        emit_u8(o, 0x9C);  // pushfq
        if (d.src1 != 1) emit_mov_rr(o, false, 1, d.src1);  // mov ecx, src1_32
        emit_rex(o, w64, 0, 0, tmp);
        emit_u8(o, 0xD3);
        emit_u8(o, uint8_t(0xC0 | (shift_sub << 3) | (tmp & 7)));
        emit_u8(o, 0x9D);  // popfq
        emit_u8(o, 0x59);  // pop rcx
        emit_mov_rr(o, w64, d.dst, tmp);
        emit_pop_gpr(o, tmp);
        return true;
    }
    case Op::mulx: {
        const uint8_t tmp = pick_scratch_gpr(used | (1u << 0) | (1u << 2));
        if (tmp == 0xFF) return false;
        emit_push_gpr(o, tmp);
        if (!emit_bmi_load_rm(o, d, tmp, 8, orig_next_va, stub_base_va)) return false;
        emit_u8(o, 0x50);  // push rax ([rsp+16] after pushfq)
        emit_u8(o, 0x52);  // push rdx ([rsp+8] after pushfq)
        emit_u8(o, 0x9C);  // pushfq
        emit_mov_rr(o, w64, 0, 2);  // mov rax/eax, rdx/edx
        emit_rex(o, w64, 0, 0, tmp);
        emit_u8(o, 0xF7);
        emit_u8(o, uint8_t(0xE0 | (tmp & 7)));  // mul tmp -> rax=lo (zero-ext), rdx=hi (zero-ext)
        emit_u8(o, 0x9D);  // popfq -> now [rsp] = saved rdx, [rsp+8] = saved rax
        // Write low half (rax) to dst2 first, then high half (rdx) to dst (so dst wins when dst == dst2).
        const auto write_mulx_out = [&](uint8_t target_reg, uint8_t val_reg) {
            if (target_reg == 2) {
                // mov [rsp], val_reg (64-bit, already zero-extended when !w64)
                emit_u8(o, 0x48);
                emit_u8(o, 0x89);
                emit_u8(o, uint8_t(((val_reg & 7) << 3) | 4u));
                emit_u8(o, 0x24);
            } else if (target_reg == 0) {
                // mov [rsp+8], val_reg
                emit_u8(o, 0x48);
                emit_u8(o, 0x89);
                emit_u8(o, uint8_t(0x44 | ((val_reg & 7) << 3)));
                emit_u8(o, 0x24);
                emit_u8(o, 0x08);
            } else {
                emit_mov_rr(o, w64, target_reg, val_reg);
            }
        };
        write_mulx_out(d.dst2, 0);
        write_mulx_out(d.dst, 2);
        emit_u8(o, 0x5A);  // pop rdx
        emit_u8(o, 0x58);  // pop rax
        emit_pop_gpr(o, tmp);
        return true;
    }
    case Op::andn: {
        const uint8_t tmp = pick_scratch_gpr(used);
        if (tmp == 0xFF) return false;
        emit_push_gpr(o, tmp);
        if (!emit_bmi_load_rm(o, d, tmp, 8, orig_next_va, stub_base_va)) return false;
        emit_mov_rr(o, w64, d.dst, d.src1);
        emit_rex(o, w64, 0, 0, d.dst);
        emit_u8(o, 0xF7);
        emit_u8(o, uint8_t(0xD0 | (d.dst & 7)));  // not dst
        emit_rex(o, w64, tmp, 0, d.dst);
        emit_u8(o, 0x21);  // and dst, tmp (sets CF=0, OF=0, ZF, SF)
        emit_u8(o, uint8_t(0xC0 | ((tmp & 7) << 3) | (d.dst & 7)));
        emit_pop_gpr(o, tmp);
        return true;
    }
    case Op::blsi: {
        // res = (-s) & s; CF = (s != 0) == (res != 0), OF = 0, ZF = (res == 0), SF = msb(res)
        const uint8_t tmp = pick_scratch_gpr(used);
        if (tmp == 0xFF) return false;
        emit_push_gpr(o, tmp);
        if (!emit_bmi_load_rm(o, d, tmp, 8, orig_next_va, stub_base_va)) return false;
        emit_mov_rr(o, w64, d.dst, tmp);
        emit_rex(o, w64, 0, 0, d.dst);
        emit_u8(o, 0xF7);
        emit_u8(o, uint8_t(0xD8 | (d.dst & 7)));  // neg dst (sets CF = (s != 0), dst = -s)
        emit_rex(o, w64, tmp, 0, d.dst);
        emit_u8(o, 0x21);  // and dst, tmp (sets OF=0, ZF=(res==0), SF=msb(res), clears CF)
        emit_u8(o, uint8_t(0xC0 | ((tmp & 7) << 3) | (d.dst & 7)));
        emit_u8(o, 0x74);  // je +1 (skip stc when res == 0)
        emit_u8(o, 0x01);
        emit_u8(o, 0xF9);  // stc (CF = 1 when res != 0)
        emit_pop_gpr(o, tmp);
        return true;
    }
    case Op::blsr:
    case Op::blsmsk: {
        // BLSR:   res = (s - 1) & s; CF = (s == 0), OF = 0, ZF = (res == 0), SF = msb(res)
        // BLSMSK: res = (s - 1) ^ s; CF = (s == 0), OF = 0, ZF = (res == 0), SF = msb(res)
        const uint8_t tmp = pick_scratch_gpr(used);
        if (tmp == 0xFF) return false;
        emit_push_gpr(o, tmp);
        if (!emit_bmi_load_rm(o, d, tmp, 8, orig_next_va, stub_base_va)) return false;
        emit_rex(o, w64, tmp, 0, tmp);
        emit_u8(o, 0x85);
        emit_u8(o, uint8_t(0xC0 | ((tmp & 7) << 3) | (tmp & 7)));  // test tmp, tmp
        const std::size_t jz_patch = o.size();
        emit_u8(o, 0x74);  // jz is_zero
        emit_u8(o, 0x00);
        // Non-zero path (CF must be 0): lea dst, [tmp - 1]
        emit_rex(o, w64, d.dst, 0, tmp);
        emit_u8(o, 0x8D);
        if ((tmp & 7) == 4) {
            emit_u8(o, uint8_t(0x44 | ((d.dst & 7) << 3)));
            emit_u8(o, 0x24);
        } else {
            emit_u8(o, uint8_t(0x40 | ((d.dst & 7) << 3) | (tmp & 7)));
        }
        emit_i8(o, -1);
        // and / xor dst, tmp (sets CF=0, OF=0, ZF, SF)
        emit_rex(o, w64, tmp, 0, d.dst);
        emit_u8(o, d.op == Op::blsr ? 0x21 : 0x31);
        emit_u8(o, uint8_t(0xC0 | ((tmp & 7) << 3) | (d.dst & 7)));
        const std::size_t jmp_done = o.size();
        emit_u8(o, 0xEB);
        emit_u8(o, 0x00);
        // is_zero path (s == 0 -> CF = 1):
        o[jz_patch + 1] = uint8_t(o.size() - (jz_patch + 2));
        if (d.op == Op::blsr) {
            emit_rex(o, false, d.dst, 0, d.dst);
            emit_u8(o, 0x31);
            emit_u8(o, uint8_t(0xC0 | ((d.dst & 7) << 3) | (d.dst & 7)));  // xor dst32, dst32
        } else {
            emit_rex(o, w64, 0, 0, d.dst);
            emit_u8(o, 0x83);
            emit_u8(o, uint8_t(0xC8 | (d.dst & 7)));
            emit_u8(o, 0xFF);  // or dst, -1
        }
        emit_u8(o, 0xF9);  // stc
        o[jmp_done + 1] = uint8_t(o.size() - (jmp_done + 2));
        emit_pop_gpr(o, tmp);
        return true;
    }
    case Op::bzhi: {
        const uint8_t tmp = pick_scratch_gpr(used | (1u << 1));
        if (tmp == 0xFF) return false;
        const uint8_t tmp2 = pick_scratch_gpr(used | (1u << 1) | (1u << tmp));
        if (tmp2 == 0xFF) return false;
        emit_push_gpr(o, tmp);
        if (!emit_bmi_load_rm(o, d, tmp, 8, orig_next_va, stub_base_va)) return false;
        emit_push_gpr(o, tmp2);
        emit_u8(o, 0x51);  // push rcx
        if (d.src1 != 1) emit_mov_rr(o, false, 1, d.src1);  // mov ecx, src1_32
        emit_u8(o, 0x81); emit_u8(o, 0xE1); emit_u32(o, 0xFFu);  // and ecx, 0xFF
        emit_u8(o, 0x83); emit_u8(o, 0xF9); emit_u8(o, bits);    // cmp ecx, bits
        const std::size_t jae_ge = o.size();
        emit_u8(o, 0x73); emit_u8(o, 0x00);                      // jae ge_bits
        // < bits: tmp2 = (1 << cl) - 1; and tmp, tmp2 (sets CF=0, OF=0, ZF, SF)
        emit_mov_imm32(o, tmp2, 1);
        emit_rex(o, w64, 0, 0, tmp2);
        emit_u8(o, 0xD3);
        emit_u8(o, uint8_t(0xE0 | (tmp2 & 7)));  // shl tmp2, cl
        emit_rex(o, w64, 0, 0, tmp2);
        emit_u8(o, 0xFF);
        emit_u8(o, uint8_t(0xC8 | (tmp2 & 7)));  // dec tmp2
        emit_rex(o, w64, tmp2, 0, tmp);
        emit_u8(o, 0x21);
        emit_u8(o, uint8_t(0xC0 | ((tmp2 & 7) << 3) | (tmp & 7)));  // and tmp, tmp2
        const std::size_t jmp_done = o.size();
        emit_u8(o, 0xEB); emit_u8(o, 0x00);
        // ge_bits: res = tmp, CF = 1, OF = 0, ZF = (tmp == 0), SF = msb(tmp)
        o[jae_ge + 1] = uint8_t(o.size() - (jae_ge + 2));
        emit_rex(o, w64, tmp, 0, tmp);
        emit_u8(o, 0x85);
        emit_u8(o, uint8_t(0xC0 | ((tmp & 7) << 3) | (tmp & 7)));  // test tmp, tmp
        emit_u8(o, 0xF9);  // stc
        o[jmp_done + 1] = uint8_t(o.size() - (jmp_done + 2));
        emit_u8(o, 0x59);  // pop rcx
        emit_mov_rr(o, w64, d.dst, tmp);
        emit_pop_gpr(o, tmp2);
        emit_pop_gpr(o, tmp);
        return true;
    }
    case Op::bextr: {
        const uint8_t tmp = pick_scratch_gpr(used | (1u << 1));
        if (tmp == 0xFF) return false;
        const uint8_t tmp2 = pick_scratch_gpr(used | (1u << 1) | (1u << tmp));
        if (tmp2 == 0xFF) return false;
        emit_push_gpr(o, tmp);
        if (!emit_bmi_load_rm(o, d, tmp, 8, orig_next_va, stub_base_va)) return false;
        emit_push_gpr(o, tmp2);
        emit_u8(o, 0x51);  // push rcx
        emit_mov_rr(o, false, tmp2, d.src1);  // mov tmp2_32, src1_32 (ctl)
        emit_mov_rr(o, false, 1, tmp2);       // mov ecx, tmp2_32
        emit_u8(o, 0x81); emit_u8(o, 0xE1); emit_u32(o, 0xFFu);  // and ecx, 0xFF (start)
        emit_u8(o, 0x83); emit_u8(o, 0xF9); emit_u8(o, bits);    // cmp ecx, bits
        const std::size_t jae_zero = o.size();
        emit_u8(o, 0x73); emit_u8(o, 0x00);                      // jae zero_out
        // shr tmp, cl
        emit_rex(o, w64, 0, 0, tmp);
        emit_u8(o, 0xD3);
        emit_u8(o, uint8_t(0xE8 | (tmp & 7)));
        // ecx = (tmp2 >> 8) & 0xFF (len)
        emit_mov_rr(o, false, 1, tmp2);
        emit_u8(o, 0xC1); emit_u8(o, 0xE9); emit_u8(o, 0x08);    // shr ecx, 8
        emit_u8(o, 0x81); emit_u8(o, 0xE1); emit_u32(o, 0xFFu);  // and ecx, 0xFF (sets ZF if len == 0)
        const std::size_t jz_zero = o.size();
        emit_u8(o, 0x74); emit_u8(o, 0x00);                      // jz zero_out
        emit_u8(o, 0x83); emit_u8(o, 0xF9); emit_u8(o, bits);    // cmp ecx, bits
        const std::size_t jae_keep = o.size();
        emit_u8(o, 0x73); emit_u8(o, 0x00);                      // jae keep_shifted
        // 0 < len < bits: tmp2 = (1 << cl) - 1; and tmp, tmp2
        emit_mov_imm32(o, tmp2, 1);
        emit_rex(o, w64, 0, 0, tmp2);
        emit_u8(o, 0xD3);
        emit_u8(o, uint8_t(0xE0 | (tmp2 & 7)));  // shl tmp2, cl
        emit_rex(o, w64, 0, 0, tmp2);
        emit_u8(o, 0xFF);
        emit_u8(o, uint8_t(0xC8 | (tmp2 & 7)));  // dec tmp2
        emit_rex(o, w64, tmp2, 0, tmp);
        emit_u8(o, 0x21);
        emit_u8(o, uint8_t(0xC0 | ((tmp2 & 7) << 3) | (tmp & 7)));  // and tmp, tmp2 (CF=0, OF=0, ZF)
        const std::size_t jmp_done1 = o.size();
        emit_u8(o, 0xEB); emit_u8(o, 0x00);

        // zero_out:
        o[jae_zero + 1] = uint8_t(o.size() - (jae_zero + 2));
        o[jz_zero + 1] = uint8_t(o.size() - (jz_zero + 2));
        emit_rex(o, false, tmp, 0, tmp);
        emit_u8(o, 0x31);
        emit_u8(o, uint8_t(0xC0 | ((tmp & 7) << 3) | (tmp & 7)));  // xor tmp32, tmp32 (CF=0, OF=0, ZF=1, SF=0)
        const std::size_t jmp_done2 = o.size();
        emit_u8(o, 0xEB); emit_u8(o, 0x00);

        // keep_shifted:
        o[jae_keep + 1] = uint8_t(o.size() - (jae_keep + 2));
        emit_rex(o, w64, tmp, 0, tmp);
        emit_u8(o, 0x85);
        emit_u8(o, uint8_t(0xC0 | ((tmp & 7) << 3) | (tmp & 7)));  // test tmp, tmp (CF=0, OF=0, ZF)

        // done:
        o[jmp_done1 + 1] = uint8_t(o.size() - (jmp_done1 + 2));
        o[jmp_done2 + 1] = uint8_t(o.size() - (jmp_done2 + 2));
        emit_u8(o, 0x59);  // pop rcx
        emit_mov_rr(o, w64, d.dst, tmp);
        emit_pop_gpr(o, tmp2);
        emit_pop_gpr(o, tmp);
        return true;
    }
    default:
        return false;
    }
}

// 3-byte VEX helper (maps 1..3, all registers 0..15).
inline void emit_vex3(std::vector<uint8_t>& o, uint8_t vmap, uint8_t reg, uint8_t idx, uint8_t rm_base, uint8_t vex_w,
                      uint8_t vvvv, uint8_t vex_l, uint8_t pp, uint8_t op) {
    emit_u8(o, 0xC4);
    const uint8_t b1 = uint8_t(((reg >= 8 ? 0 : 1) << 7) | ((idx >= 8 ? 0 : 1) << 6) |
                               ((rm_base >= 8 ? 0 : 1) << 5) | (vmap & 0x1F));
    const uint8_t b2 = uint8_t(((vex_w & 1) << 7) | (((~vvvv) & 0xF) << 3) | ((vex_l & 1) << 2) | (pp & 3));
    emit_u8(o, b1);
    emit_u8(o, b2);
    emit_u8(o, op);
}

inline void emit_vex_rr(std::vector<uint8_t>& o, uint8_t vmap, uint8_t vex_w, uint8_t vex_l, uint8_t pp, uint8_t op,
                        uint8_t reg, uint8_t vvvv, uint8_t rm) {
    emit_vex3(o, vmap, reg, 0, rm, vex_w, vvvv, vex_l, pp, op);
    emit_u8(o, uint8_t(0xC0 | ((reg & 7) << 3) | (rm & 7)));
}

inline void emit_vex_rri(std::vector<uint8_t>& o, uint8_t vmap, uint8_t vex_w, uint8_t vex_l, uint8_t pp, uint8_t op,
                         uint8_t reg, uint8_t vvvv, uint8_t rm, uint8_t imm) {
    emit_vex_rr(o, vmap, vex_w, vex_l, pp, op, reg, vvvv, rm);
    emit_u8(o, imm);
}

// Flag-preserving stack frame helpers: `lea rsp, [rsp + delta]` touches zero flags and zero GPRs.
inline void emit_lea_rsp(std::vector<uint8_t>& o, int8_t delta) {
    emit_u8(o, 0x48); emit_u8(o, 0x8D); emit_u8(o, 0x64); emit_u8(o, 0x24); emit_i8(o, delta);
}

inline void emit_vmovdqu_rsp(std::vector<uint8_t>& o, bool store, uint8_t ymm, int8_t disp) {
    // vmovdqu ymm, [rsp + disp] (6F) or [rsp + disp], ymm (7F), VEX.256.F3.0F (pp=2, L=1)
    emit_vex3(o, 1, ymm, 0, 4, 0, 0, 1, 2, store ? 0x7F : 0x6F);
    if (disp == 0) {
        emit_u8(o, uint8_t(((ymm & 7) << 3) | 4u));
        emit_u8(o, 0x24);
    } else {
        emit_u8(o, uint8_t(0x40 | ((ymm & 7) << 3) | 4u));
        emit_u8(o, 0x24);
        emit_i8(o, disp);
    }
}

inline void emit_vextractf128_hi(std::vector<uint8_t>& o, uint8_t dst_xmm, uint8_t src_ymm) {
    // vextractf128 xmm_dst, ymm_src, 1: VEX.256.66.0F3A.W0 19 /r 01 (reg = src_ymm, rm = dst_xmm)
    emit_vex_rri(o, 3, 0, 1, 1, 0x19, src_ymm, 0, dst_xmm, 1);
}

inline void emit_vinsertf128_hi(std::vector<uint8_t>& o, uint8_t dst_ymm, uint8_t lo_ymm, uint8_t hi_xmm) {
    // vinsertf128 ymm_dst, ymm_lo, xmm_hi, 1: VEX.256.66.0F3A.W0 18 /r 01
    emit_vex_rri(o, 3, 0, 1, 1, 0x18, dst_ymm, lo_ymm, hi_xmm, 1);
}

// Loads `d.mem_bytes` (1, 2, 4, 8, 16, or 32) from `d`'s memory operand into `dst_vec` using AVX1 unaligned moves.
inline bool emit_avx1_load_mem(std::vector<uint8_t>& o, const avx2_emulation::Decoded& d, uint8_t dst_vec,
                               int32_t stack_adj, uint64_t orig_next_va, uint64_t stub_base_va) {
    const uint8_t idx_r = d.index >= 0 ? uint8_t(d.index) : 0u;
    const uint8_t base_r = d.base >= 0 ? uint8_t(d.base) : 0u;
    if (d.mem_bytes == 32) {
        emit_vex3(o, 1, dst_vec, idx_r, base_r, 0, 0, 1, 0, 0x10);  // vmovups ymm, m256
    } else if (d.mem_bytes == 16) {
        emit_vex3(o, 1, dst_vec, idx_r, base_r, 0, 0, 0, 0, 0x10);  // vmovups xmm, m128
    } else if (d.mem_bytes == 8) {
        emit_vex3(o, 1, dst_vec, idx_r, base_r, 0, 0, 0, 2, 0x7E);  // vmovq xmm, m64 (F3 0F 7E)
    } else if (d.mem_bytes == 4) {
        emit_vex3(o, 1, dst_vec, idx_r, base_r, 0, 0, 0, 1, 0x6E);  // vmovd xmm, m32 (66 0F 6E)
    } else if (d.mem_bytes == 2) {
        // vpinsrw xmm, xmm, m16, 0 (66 0F C4 /r 00) - reads exact 2 bytes
        emit_vex3(o, 1, dst_vec, idx_r, base_r, 0, dst_vec, 0, 1, 0xC4);
        if (!emit_mem_modrm(o, dst_vec & 7, d.base, d.index, d.scale, d.disp, stack_adj, orig_next_va, stub_base_va, 1))
            return false;
        emit_u8(o, 0x00);
        return true;
    } else if (d.mem_bytes == 1) {
        // vpinsrb xmm, xmm, m8, 0 (66 0F3A 20 /r 00) - reads exact 1 byte
        emit_vex3(o, 3, dst_vec, idx_r, base_r, 0, dst_vec, 0, 1, 0x20);
        if (!emit_mem_modrm(o, dst_vec & 7, d.base, d.index, d.scale, d.disp, stack_adj, orig_next_va, stub_base_va, 1))
            return false;
        emit_u8(o, 0x00);
        return true;
    } else {
        return false;
    }
    return emit_mem_modrm(o, dst_vec & 7, d.base, d.index, d.scale, d.disp, stack_adj, orig_next_va, stub_base_va);
}

// Returns true if `site[0..d.length)` can be rewritten in place into an exact same-length native AVX1 instruction.
// If `apply_change` is true, mutates `site` in place.
inline bool try_avx2_inplace(uint8_t* site, const avx2_emulation::Decoded& d, const avx2_emulation::Host& host,
                             bool apply_change) {
    if (d.cat != avx2_emulation::Category::avx2 || host.avx2) return false;
    // Memory-source vbroadcastss (0x18) and vbroadcastsd (0x19) are already native AVX1; leave them untouched.
    if (d.vmap == 2 && (d.op == 0x18 || d.op == 0x19) && d.mem) return false;

    std::size_t p = 0;
    while (p < d.length && site[p] != 0xC4 && site[p] != 0xC5) ++p;
    if (p >= d.length) return false;

    // 1. 256-bit vpand(DB)/vpandn(DF)/vpor(EB)/vpxor(EF) -> vandps(54)/vandnps(55)/vorps(56)/vxorps(57)
    if (d.vmap == 1 && d.pp == 1 && d.vex_l == 1 &&
        (d.op == 0xDB || d.op == 0xDF || d.op == 0xEB || d.op == 0xEF)) {
        const uint8_t new_op = d.op == 0xDB ? 0x54 : d.op == 0xDF ? 0x55 : d.op == 0xEB ? 0x56 : 0x57;
        if (apply_change) {
            if (site[p] == 0xC5) {
                site[p + 1] = uint8_t(site[p + 1] & ~3u);  // pp = 0
                site[p + 2] = new_op;
            } else {
                site[p + 2] = uint8_t(site[p + 2] & ~0x83u);  // vex_w = 0, pp = 0
                site[p + 3] = new_op;
            }
        }
        return true;
    }

    // 2. 3-byte VEX (0xC4, W=0, pp=1) exact bit-equivalent AVX1 opcodes:
    if (site[p] == 0xC4 && d.vex_w == 0 && d.pp == 1) {
        uint8_t new_op = 0;
        if (d.vmap == 3 && d.op == 0x39) new_op = 0x19;                  // vextracti128 -> vextractf128
        else if (d.vmap == 3 && d.op == 0x38) new_op = 0x18;             // vinserti128  -> vinsertf128
        else if (d.vmap == 3 && d.op == 0x46) new_op = 0x06;             // vperm2i128   -> vperm2f128
        else if (d.vmap == 3 && d.op == 0x02) new_op = 0x0C;             // vpblendd     -> vblendps
        else if (d.vmap == 2 && d.op == 0x5A && d.mem) new_op = 0x1A;    // vbroadcasti128 -> vbroadcastf128
        else if (d.vmap == 2 && d.op == 0x58 && d.mem) new_op = 0x18;    // vpbroadcastd m32 -> vbroadcastss m32
        else if (d.vmap == 2 && d.op == 0x59 && d.mem && d.vex_l == 1) new_op = 0x19;  // vpbroadcastq ymm,m64 -> vbroadcastsd
        if (new_op != 0) {
            if (apply_change) site[p + 3] = new_op;
            return true;
        }
    }
    return false;
}

// Emits a native AVX1 machine-code body (without trailing jmp) for an AVX2 or FMA3 instruction.
inline bool emit_avx2_fma_body(std::vector<uint8_t>& o, const avx2_emulation::Decoded& d, uint64_t orig_va,
                               uint64_t stub_base_va, const avx2_emulation::Host& host) {
    using avx2_emulation::Category;
    if (d.addr32 || d.seg != 0) return false;
    if (d.cat == Category::fma3 && host.fma) return false;
    if (d.cat == Category::avx2 && host.avx2) return false;
    if (d.cat != Category::avx2 && d.cat != Category::fma3) return false;
    // Skip native AVX1 memory broadcasts.
    if (d.cat == Category::avx2 && d.vmap == 2 && (d.op == 0x18 || d.op == 0x19) && d.mem) return false;

    const uint64_t orig_next_va = orig_va + d.length;
    uint32_t used_ymm = (1u << d.reg) | (1u << d.vvvv);
    if (!d.mem) used_ymm |= (1u << d.rm);
    if (d.vmap == 3 && d.op == 0x4C) used_ymm |= (1u << ((d.imm >> 4) & 0xFu));

    // ---- FMA3 ----
    if (d.cat == Category::fma3) {
        const uint8_t lo_nib = d.op & 0x0Fu;
        const uint8_t hi_nib = d.op & 0xF0u;
        // Support fmadd(8/9), fmsub(A/B), fnmadd(C/D), fnmsub(E/F), fmaddsub(6). Leave rare fmsubadd(7) to #UD.
        if (lo_nib == 7 || lo_nib < 6) return false;

        const uint8_t t0 = pick_scratch_ymm(used_ymm);
        const uint8_t t1 = pick_scratch_ymm(used_ymm | (1u << t0));
        if (t0 == 0xFF || t1 == 0xFF) return false;

        emit_lea_rsp(o, -64);
        emit_vmovdqu_rsp(o, true, t0, 0);
        emit_vmovdqu_rsp(o, true, t1, 32);

        uint8_t r3 = d.rm;
        if (d.mem) {
            if (!emit_avx1_load_mem(o, d, t1, 64, orig_next_va, stub_base_va)) return false;
            r3 = t1;
        }

        const uint8_t a_reg = (hi_nib == 0x90u) ? d.reg : d.vvvv;
        const uint8_t b_reg = (hi_nib == 0xA0u) ? d.reg : r3;
        const uint8_t c_reg = (hi_nib == 0x90u) ? d.vvvv : (hi_nib == 0xA0u) ? r3 : d.reg;
        const bool scalar = (lo_nib >= 8) && ((lo_nib & 1u) != 0);
        const bool is_f64 = d.vex_w != 0;
        const uint8_t pp_fp = scalar ? (is_f64 ? 3u : 2u) : (is_f64 ? 1u : 0u);  // F2/F3 for sd/ss, 66/none for pd/ps
        const uint8_t vl = scalar ? 0u : d.vex_l;

        // t0 = A * B
        emit_vex_rr(o, 1, 0, vl, pp_fp, 0x59, t0, a_reg, b_reg);  // vmulps/pd/ss/sd

        const uint8_t comb_dst = scalar ? t0 : d.reg;
        if (lo_nib == 8 || lo_nib == 9) {
            emit_vex_rr(o, 1, 0, vl, pp_fp, 0x58, comb_dst, t0, c_reg);  // vadd* (A*B + C)
        } else if (lo_nib == 0x0A || lo_nib == 0x0B) {
            emit_vex_rr(o, 1, 0, vl, pp_fp, 0x5C, comb_dst, t0, c_reg);  // vsub* (A*B - C)
        } else if (lo_nib == 0x0C || lo_nib == 0x0D) {
            emit_vex_rr(o, 1, 0, vl, pp_fp, 0x5C, comb_dst, c_reg, t0);  // vsub* (C - A*B)
        } else if (lo_nib == 6) {
            // vfmaddsub -> vaddsubps (F2 0F D0, pp=3) / vaddsubpd (66 0F D0, pp=1)
            emit_vex_rr(o, 1, 0, vl, is_f64 ? 1u : 3u, 0xD0, d.reg, t0, c_reg);
        } else {
            // fnmsub (0x0E / 0x0F): negate t0 via sign-bit mask in a third scratch register before subtracting c_reg.
            const uint8_t sm = pick_scratch_ymm(used_ymm | (1u << t0) | (1u << t1));
            if (sm == 0xFF) return false;
            emit_lea_rsp(o, -32);
            emit_vmovdqu_rsp(o, true, sm, 0);
            emit_vex_rr(o, 1, 0, 0, 1, 0x76, sm, sm, sm);  // vpcmpeqd xmm_sm, xmm_sm, xmm_sm
            if (!is_f64) {
                emit_vex_rri(o, 1, 0, 0, 1, 0x72, 6, sm, sm, 31);  // vpslld xmm_sm, xmm_sm, 31
            } else {
                emit_vex_rri(o, 1, 0, 0, 1, 0x73, 6, sm, sm, 63);  // vpsllq xmm_sm, xmm_sm, 63
            }
            if (vl == 1) emit_vinsertf128_hi(o, sm, sm, sm);
            emit_vex_rr(o, 1, 0, vl, is_f64 ? 1u : 0u, 0x57, t0, t0, sm);  // vxorps/pd t0, t0, sm
            emit_vmovdqu_rsp(o, false, sm, 0);
            emit_lea_rsp(o, 32);
            emit_vex_rr(o, 1, 0, vl, pp_fp, 0x5C, comb_dst, t0, c_reg);  // (-A*B) - C
        }

        if (scalar) {
            // Merge scalar lane t0[0] into d.reg[0], keeping d.reg[127:32/64] and zeroing d.reg[255:128].
            emit_vex_rr(o, 1, 0, 0, is_f64 ? 3u : 2u, 0x10, d.reg, d.reg, t0);  // vmovsd / vmovss d.reg, d.reg, t0
        }

        emit_vmovdqu_rsp(o, false, t1, 32);
        emit_vmovdqu_rsp(o, false, t0, 0);
        emit_lea_rsp(o, 64);
        return true;
    }

    // ---- AVX2 ----
    // 1. Register-source 32-bit/64-bit broadcasts (zero scratch registers, zero stack adjustment):
    if (d.vmap == 2 && !d.mem && (d.op == 0x18 || d.op == 0x58)) {
        emit_vex_rri(o, 1, 0, 0, 1, 0x70, d.reg, 0, d.rm, 0x00);  // vpshufd xmm(d.reg), xmm(d.rm), 0x00
        if (d.vex_l == 1) emit_vinsertf128_hi(o, d.reg, d.reg, d.reg);
        return true;
    }
    if (d.vmap == 2 && !d.mem && (d.op == 0x19 || d.op == 0x59)) {
        emit_vex_rr(o, 1, 0, 0, 3, 0x12, d.reg, 0, d.rm);  // vmovddup xmm(d.reg), xmm(d.rm)
        if (d.vex_l == 1) emit_vinsertf128_hi(o, d.reg, d.reg, d.reg);
        return true;
    }
    if (d.vmap == 2 && d.mem && d.op == 0x59 && d.vex_l == 0) {
        // vpbroadcastq xmm, m64 -> vmovddup xmm, m64
        const uint8_t idx_r = d.index >= 0 ? uint8_t(d.index) : 0u;
        const uint8_t base_r = d.base >= 0 ? uint8_t(d.base) : 0u;
        emit_vex3(o, 1, d.reg, idx_r, base_r, 0, 0, 0, 3, 0x12);
        return emit_mem_modrm(o, d.reg & 7, d.base, d.index, d.scale, d.disp, 0, orig_next_va, stub_base_va);
    }
    if (d.vmap == 2 && !d.mem && d.op == 0x79) {
        // vpbroadcastw reg-source: vpshuflw xmm, xmm_rm, 0 (pp=3); vpshufd xmm, xmm, 0 (pp=1); [vinsertf128]
        emit_vex_rri(o, 1, 0, 0, 3, 0x70, d.reg, 0, d.rm, 0x00);
        emit_vex_rri(o, 1, 0, 0, 1, 0x70, d.reg, 0, d.reg, 0x00);
        if (d.vex_l == 1) emit_vinsertf128_hi(o, d.reg, d.reg, d.reg);
        return true;
    }

    // Remaining AVX2 stubs allocate stack scratch for t0, t1 (flag-preserving via lea rsp).
    const uint8_t t0 = pick_scratch_ymm(used_ymm);
    const uint8_t t1 = pick_scratch_ymm(used_ymm | (1u << t0));
    if (t0 == 0xFF || t1 == 0xFF) return false;

    // 2. vpmovmskb r32/64, ymm (vmap=1, op=0xD7, !d.mem)
    if (d.vmap == 1 && d.op == 0xD7) {
        if (d.reg == 4) return false;
        const uint8_t tgpr = pick_scratch_gpr((1u << 4) | (1u << d.reg));
        if (tgpr == 0xFF) return false;
        emit_lea_rsp(o, -32);
        emit_vmovdqu_rsp(o, true, t0, 0);
        emit_push_gpr(o, tgpr);
        emit_u8(o, 0x9C);  // pushfq
        emit_vextractf128_hi(o, t0, d.rm);
        emit_vex_rr(o, 1, 0, 0, 1, 0xD7, tgpr, 0, t0);    // vpmovmskb tgpr32, xmm_t0
        emit_vex_rr(o, 1, 0, 0, 1, 0xD7, d.reg, 0, d.rm); // vpmovmskb reg32, xmm(d.rm) (zero-extends reg64)
        emit_rex(o, false, 0, 0, tgpr);
        emit_u8(o, 0xC1); emit_u8(o, uint8_t(0xE0 | (tgpr & 7))); emit_u8(o, 16);  // shl tgpr32, 16
        emit_rex(o, false, tgpr, 0, d.reg);
        emit_u8(o, 0x09); emit_u8(o, uint8_t(0xC0 | ((tgpr & 7) << 3) | (d.reg & 7)));  // or reg32, tgpr32
        emit_u8(o, 0x9D);  // popfq
        emit_pop_gpr(o, tgpr);
        emit_vmovdqu_rsp(o, false, t0, 0);
        emit_lea_rsp(o, 32);
        return true;
    }

    // Leave rare variable shifts (0x45..0x47), vpermps/vpermd (0x16/0x36), vpmaskmov (0x8C/0x8E), vpblendvb (0x4C)
    // to #UD emulation.
    if (d.vmap == 2 && (d.op == 0x16 || d.op == 0x36 || d.op == 0x45 || d.op == 0x46 || d.op == 0x47 ||
                        d.op == 0x8C || d.op == 0x8E))
        return false;
    if (d.vmap == 3 && d.op == 0x4C) return false;

    emit_lea_rsp(o, -64);
    emit_vmovdqu_rsp(o, true, t0, 0);
    emit_vmovdqu_rsp(o, true, t1, 32);

    uint8_t s2 = d.rm;
    if (d.mem) {
        if (!emit_avx1_load_mem(o, d, t1, 64, orig_next_va, stub_base_va)) return false;
        s2 = t1;
    }

    // 3. vpbroadcastb (0x78) & mem vpbroadcastw (0x79):
    if (d.vmap == 2 && d.op == 0x78) {
        emit_vex_rr(o, 1, 0, 0, 1, 0xEF, t0, t0, t0);         // vpxor xmm_t0, xmm_t0, xmm_t0
        emit_vex_rr(o, 2, 0, 0, 1, 0x00, d.reg, s2, t0);      // vpshufb xmm(d.reg), xmm(s2), xmm_t0
        if (d.vex_l == 1) emit_vinsertf128_hi(o, d.reg, d.reg, d.reg);
    } else if (d.vmap == 2 && d.op == 0x79) {
        emit_vex_rri(o, 1, 0, 0, 3, 0x70, d.reg, 0, s2, 0x00);
        emit_vex_rri(o, 1, 0, 0, 1, 0x70, d.reg, 0, d.reg, 0x00);
        if (d.vex_l == 1) emit_vinsertf128_hi(o, d.reg, d.reg, d.reg);
    }
    // 4. vpermq / vpermpd (vmap=3, op=0x00/0x01, vex_w=1, vex_l=1):
    else if (d.vmap == 3 && (d.op == 0x00 || d.op == 0x01)) {
        emit_vex_rr(o, 1, 0, 0, 0, 0x28, t0, 0, s2);  // vmovaps xmm_t0, xmm(s2) [low 128b]
        emit_vextractf128_hi(o, t1, s2);              // xmm_t1 = high 128b
        const uint8_t q0 = d.imm & 3u, q1 = (d.imm >> 2) & 3u, q2 = (d.imm >> 4) & 3u, q3 = (d.imm >> 6) & 3u;
        // vshufpd xmm(d.reg), srcA, srcB, imm2 (VEX.128.66.0F C6 /r ib)
        emit_vex_rri(o, 1, 0, 0, 1, 0xC6, d.reg, (q0 & 2u) ? t1 : t0, (q1 & 2u) ? t1 : t0,
                     uint8_t((q0 & 1u) | ((q1 & 1u) << 1)));
        emit_vex_rri(o, 1, 0, 0, 1, 0xC6, t0, (q2 & 2u) ? t1 : t0, (q3 & 2u) ? t1 : t0,
                     uint8_t((q2 & 1u) | ((q3 & 1u) << 1)));
        emit_vinsertf128_hi(o, d.reg, d.reg, t0);
    }
    // 5. vpmovsx* (0x20..0x25) & vpmovzx* (0x30..0x35), vex_l == 1:
    else if (d.vmap == 2 && ((d.op >= 0x20 && d.op <= 0x25) || (d.op >= 0x30 && d.op <= 0x35))) {
        const uint8_t k = d.op & 0x0Fu;
        const uint8_t half_bytes = (k == 0 || k == 3 || k == 5) ? 8u : (k == 1 || k == 4) ? 4u : 2u;
        emit_vex_rri(o, 1, 0, 0, 1, 0x73, 3, t0, s2, half_bytes);  // vpsrldq xmm_t0, xmm(s2), half_bytes
        emit_vex_rr(o, 2, 0, 0, 1, d.op, t0, 0, t0);               // high 128b
        emit_vex_rr(o, 2, 0, 0, 1, d.op, d.reg, 0, s2);            // low 128b
        emit_vinsertf128_hi(o, d.reg, d.reg, t0);
    }
    // 6. Map-1 immediate shifts (0x71, 0x72, 0x73: sub=d.reg, dst=d.vvvv, src=s2, imm=d.imm):
    else if (d.vmap == 1 && (d.op == 0x71 || d.op == 0x72 || d.op == 0x73)) {
        emit_vextractf128_hi(o, t0, s2);
        emit_vex_rri(o, 1, 0, 0, 1, d.op, d.reg, t0, t0, d.imm);
        emit_vex_rri(o, 1, 0, 0, 1, d.op, d.reg, d.vvvv, s2, d.imm);
        emit_vinsertf128_hi(o, d.vvvv, d.vvvv, t0);
    }
    // 7. Map-1 XMM-count shifts (0xD1..0xD3, 0xE1..0xE2, 0xF1..0xF3: count is low 64b of xmm(s2) for both lanes):
    else if (d.vmap == 1 && (d.op == 0xD1 || d.op == 0xD2 || d.op == 0xD3 || d.op == 0xE1 || d.op == 0xE2 ||
                             d.op == 0xF1 || d.op == 0xF2 || d.op == 0xF3)) {
        emit_vextractf128_hi(o, t0, d.vvvv);
        emit_vex_rr(o, 1, 0, 0, 1, d.op, t0, t0, s2);
        emit_vex_rr(o, 1, 0, 0, 1, d.op, d.reg, d.vvvv, s2);
        emit_vinsertf128_hi(o, d.reg, d.reg, t0);
    }
    // 8. Unary 256-bit lane-wise ops: vpshufd/hw/lw (vmap=1, op=0x70, imm8) & vpabsb/w/d (vmap=2, op=0x1C..0x1E):
    else if ((d.vmap == 1 && d.op == 0x70) || (d.vmap == 2 && (d.op == 0x1C || d.op == 0x1D || d.op == 0x1E))) {
        emit_vextractf128_hi(o, t0, s2);
        if (d.vmap == 1) {
            emit_vex_rri(o, 1, 0, 0, d.pp, 0x70, t0, 0, t0, d.imm);
            emit_vex_rri(o, 1, 0, 0, d.pp, 0x70, d.reg, 0, s2, d.imm);
        } else {
            emit_vex_rr(o, 2, 0, 0, 1, d.op, t0, 0, t0);
            emit_vex_rr(o, 2, 0, 0, 1, d.op, d.reg, 0, s2);
        }
        emit_vinsertf128_hi(o, d.reg, d.reg, t0);
    }
    // 9. All remaining binary 256-bit lane-wise ops (map 1, map 2, and map 3 vpblendw 0x0E / vpalignr 0x0F):
    else if (d.vex_l == 1 && (d.vmap == 1 || d.vmap == 2 || (d.vmap == 3 && (d.op == 0x0E || d.op == 0x0F)))) {
        emit_vextractf128_hi(o, t0, d.vvvv);  // save vvvv_hi before touching d.reg (in case d.reg == d.vvvv)
        if (d.mem) {
            // s2 == t1 holds [mem_lo, mem_hi] and d.reg != t1: compute low 128b into d.reg first while xmm_t1 == mem_lo.
            if (d.vmap == 3) {
                emit_vex_rri(o, 3, 0, 0, 1, d.op, d.reg, d.vvvv, t1, d.imm);
                emit_vextractf128_hi(o, t1, t1);
                emit_vex_rri(o, 3, 0, 0, 1, d.op, t0, t0, t1, d.imm);
            } else {
                emit_vex_rr(o, d.vmap, 0, 0, d.pp, d.op, d.reg, d.vvvv, t1);
                emit_vextractf128_hi(o, t1, t1);
                emit_vex_rr(o, d.vmap, 0, 0, d.pp, d.op, t0, t0, t1);
            }
        } else {
            emit_vextractf128_hi(o, t1, s2);
            if (d.vmap == 3) {
                emit_vex_rri(o, 3, 0, 0, 1, d.op, t0, t0, t1, d.imm);
                emit_vex_rri(o, 3, 0, 0, 1, d.op, d.reg, d.vvvv, s2, d.imm);
            } else {
                emit_vex_rr(o, d.vmap, 0, 0, d.pp, d.op, t0, t0, t1);
                emit_vex_rr(o, d.vmap, 0, 0, d.pp, d.op, d.reg, d.vvvv, s2);
            }
        }
        emit_vinsertf128_hi(o, d.reg, d.reg, t0);
    } else {
        return false;
    }

    emit_vmovdqu_rsp(o, false, t1, 32);
    emit_vmovdqu_rsp(o, false, t0, 0);
    emit_lea_rsp(o, 64);
    return true;
}

inline bool is_control_flow(const uint8_t* ins, std::size_t len, int64_t* rel_out = nullptr) {
    if (rel_out) *rel_out = 0;
    std::size_t p = 0;
    while (p + 1 < len) {
        const uint8_t b = ins[p];
        if ((b & 0xF0) == 0x40 || b == 0x66 || b == 0x67 || b == 0xF2 || b == 0xF3 ||
            b == 0x26 || b == 0x2E || b == 0x36 || b == 0x3E || b == 0x64 || b == 0x65)
            ++p;
        else
            break;
    }
    const uint8_t op = ins[p];
    if (((op >= 0x70 && op <= 0x7F) || (op >= 0xE0 && op <= 0xE3) || op == 0xEB) && p + 2 == len) {
        if (rel_out) *rel_out = int8_t(ins[p + 1]);
        return true;
    }
    if ((op == 0xE8 || op == 0xE9) && p + 5 == len) {
        int32_t d = 0;
        std::memcpy(&d, ins + p + 1, 4);
        if (rel_out) *rel_out = d;
        return true;
    }
    if (op == 0x0F && p + 1 < len) {
        const uint8_t op2 = ins[p + 1];
        if (op2 >= 0x80 && op2 <= 0x8F && p + 6 == len) {
            int32_t d = 0;
            std::memcpy(&d, ins + p + 2, 4);
            if (rel_out) *rel_out = d;
            return true;
        }
        if (op2 == 0x05 || op2 == 0x0B || op2 == 0x34) return true;
    }
    if (op == 0xC3 || op == 0xC2 || op == 0xCB || op == 0xCA || op == 0xCC || op == 0xCD || op == 0xCF || op == 0xF4)
        return true;
    if (op == 0xFF && p + 1 < len) {
        const uint8_t sub = (ins[p + 1] >> 3) & 7u;
        if (sub >= 2 && sub <= 5) return true;
    }
    return false;
}

// Finds the byte offset of the 4-byte RIP-relative displacement inside a non-control-flow instruction `ins[0..len)`,
// or returns 0 if `ins` is not RIP-relative. Returns SIZE_MAX if `ins` is ambiguous.
inline std::size_t find_rip_disp32_offset(const uint8_t* ins, std::size_t len) {
    std::size_t i = 0;
    bool p66 = false, pf2 = false, pf3 = false, any_rex = false;
    for (; i < len && i < 14; ++i) {
        const uint8_t b = ins[i];
        if ((b & 0xF0) == 0x40) { any_rex = true; continue; }
        if (b == 0x66) p66 = true;
        else if (b == 0xF2) pf2 = true;
        else if (b == 0xF3) pf3 = true;
        else if (b == 0x67) return SIZE_MAX;
        else if (b == 0x26 || b == 0x2E || b == 0x36 || b == 0x3E || b == 0x64 || b == 0x65) {}
        else break;
    }
    if (i >= len) return SIZE_MAX;
    const uint8_t op = ins[i++];
    if (op == 0xC4 || op == 0xC5) {
        if (p66 || pf2 || pf3 || any_rex) return SIZE_MAX;
        uint8_t vmap = 1, vop = 0;
        if (op == 0xC5) {
            if (i + 2 > len) return SIZE_MAX;
            vop = ins[i + 1];
            i += 2;
        } else {
            if (i + 3 > len) return SIZE_MAX;
            vmap = ins[i] & 0x1F;
            vop = ins[i + 2];
            i += 3;
        }
        if (vmap == 1 && vop == 0x77) return 0;
        if (i >= len) return SIZE_MAX;
        const uint8_t m = ins[i++];
        if ((m >> 6) == 0 && (m & 7) == 5) return (i + 4 <= len) ? i : SIZE_MAX;
        return 0;
    }
    if (op == 0x0F) {
        if (i >= len) return SIZE_MAX;
        const uint8_t op2 = ins[i++];
        if (op2 == 0x38 || op2 == 0x3A) {
            if (i >= len) return SIZE_MAX;
            ++i;
        } else {
            switch (op2) {
            case 0x05: case 0x06: case 0x07: case 0x08: case 0x09: case 0x0B: case 0x0E:
            case 0x30: case 0x31: case 0x32: case 0x33: case 0x34: case 0x35: case 0x37: case 0x77:
            case 0xA0: case 0xA1: case 0xA2: case 0xA8: case 0xA9: case 0xAA:
            case 0xC8: case 0xC9: case 0xCA: case 0xCB: case 0xCC: case 0xCD: case 0xCE: case 0xCF:
                return 0;
            default: break;
            }
            if (op2 >= 0x80 && op2 <= 0x8F) return 0;
        }
        if (i >= len) return SIZE_MAX;
        const uint8_t m = ins[i++];
        if ((m >> 6) == 0 && (m & 7) == 5) return (i + 4 <= len) ? i : SIZE_MAX;
        return 0;
    }
    // Single-byte non-ModRM opcodes:
    if ((op >= 0x50 && op <= 0x5F) || (op >= 0x90 && op <= 0x9F) || (op >= 0xA0 && op <= 0xAF) ||
        (op >= 0xB0 && op <= 0xBF) || (op & 0xC7) == 0x04 || (op & 0xC7) == 0x05 || op == 0x68 || op == 0x6A ||
        op == 0xC8 || op == 0xC9 || op == 0xD7 || (op >= 0xEC && op <= 0xEF) || (op >= 0xF8 && op <= 0xFD))
        return 0;
    if (i >= len) return SIZE_MAX;
    const uint8_t m = ins[i++];
    if ((m >> 6) == 0 && (m & 7) == 5) return (i + 4 <= len) ? i : SIZE_MAX;
    return 0;
}

// Emits `ins2` (the second instruction of a fused 4B+len2 pair) into `o`, either as a native stub body (if `ins2` is
// itself an emulated BMI/AVX2/FMA3/in-place op) or by relocating `ins2` verbatim.
inline bool emit_second_insn(std::vector<uint8_t>& o, const uint8_t* ins2, std::size_t len2, uint64_t orig_va2,
                             uint64_t stub_base_va, const bmi_emulation::Host& bmi_host,
                             const avx2_emulation::Host& avx_host) {
    // Check if ins2 is an AVX2/FMA3/F16C instruction:
    const auto ad2 = avx2_emulation::decode(ins2, len2);
    if (ad2.cat != avx2_emulation::Category::none &&
        !(ad2.vmap == 2 && (ad2.op == 0x18 || ad2.op == 0x19) && ad2.mem)) {
        uint8_t copy[15]{};
        std::memcpy(copy, ins2, len2);
        if (try_avx2_inplace(copy, ad2, avx_host, true)) {
            const std::size_t rip_off = find_rip_disp32_offset(copy, len2);
            if (rip_off == SIZE_MAX) return false;
            if (rip_off != 0) {
                int32_t old_d = 0;
                std::memcpy(&old_d, copy + rip_off, 4);
                const uint64_t target_va = orig_va2 + len2 + uint64_t(int64_t(old_d));
                const uint64_t stub_next_va = stub_base_va + o.size() + len2;
                const int64_t new_d = int64_t(target_va) - int64_t(stub_next_va);
                if (new_d < INT32_MIN || new_d > INT32_MAX) return false;
                const int32_t d32 = int32_t(new_d);
                std::memcpy(copy + rip_off, &d32, 4);
            }
            o.insert(o.end(), copy, copy + len2);
            return true;
        }
        return emit_avx2_fma_body(o, ad2, orig_va2, stub_base_va, avx_host);
    }

    // Check if ins2 is a BMI/MOVBE instruction:
    const auto bd2 = bmi_emulation::decode(ins2, len2);
    if (bd2.op != bmi_emulation::Op::none) {
        if (bd2.op == bmi_emulation::Op::tzcnt || bd2.op == bmi_emulation::Op::lzcnt) return false;
        return emit_bmi_body(o, bd2, orig_va2, stub_base_va, bmi_host);
    }

    // Skip rare 0F 01 / 0F 78 / 0F 79 / 0F AE encodings that may trap on non-Zen hosts.
    if (ins2[0] == 0x0F || ins2[0] == 0x66 || ins2[0] == 0xF2 || ins2[0] == 0xF3) {
        for (std::size_t k = 0; k + 1 < len2; ++k) {
            if (ins2[k] == 0x0F &&
                (ins2[k + 1] == 0x01 || ins2[k + 1] == 0x78 || ins2[k + 1] == 0x79 || ins2[k + 1] == 0xAE))
                return false;
        }
    }

    // Native x86-64 / AVX1 instruction: copy and relocate RIP-relative disp32 if present.
    const std::size_t rip_off = find_rip_disp32_offset(ins2, len2);
    if (rip_off == SIZE_MAX) return false;
    uint8_t copy[15]{};
    std::memcpy(copy, ins2, len2);
    if (rip_off != 0) {
        int32_t old_d = 0;
        std::memcpy(&old_d, copy + rip_off, 4);
        const uint64_t target_va = orig_va2 + len2 + uint64_t(int64_t(old_d));
        const uint64_t stub_next_va = stub_base_va + o.size() + len2;
        const int64_t new_d = int64_t(target_va) - int64_t(stub_next_va);
        if (new_d < INT32_MIN || new_d > INT32_MAX) return false;
        const int32_t d32 = int32_t(new_d);
        std::memcpy(copy + rip_off, &d32, 4);
    }
    o.insert(o.end(), copy, copy + len2);
    return true;
}

inline uint8_t* reserve_stub_region(uint64_t lo, uint64_t hi, std::size_t size) {
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    const uint64_t gran = si.dwAllocationGranularity;
    const uint64_t limit = uint64_t(1) << 31;
    for (uint64_t at = (hi + gran - 1) & ~(gran - 1); at + size < lo + limit - 0x10000; at += gran) {
        if (void* p = VirtualAlloc(reinterpret_cast<void*>(at), size, MEM_RESERVE, PAGE_NOACCESS))
            return static_cast<uint8_t*>(p);
    }
    for (uint64_t at = (lo > size ? (lo - size) & ~(gran - 1) : 0); at > gran && hi - at < limit - 0x10000; at -= gran) {
        if (void* p = VirtualAlloc(reinterpret_cast<void*>(at), size, MEM_RESERVE, PAGE_NOACCESS))
            return static_cast<uint8_t*>(p);
    }
    return nullptr;
}

}  // namespace detail

inline Stats patch_image(uint8_t* base, uint32_t image_size,
                         bmi_emulation::Host bmi_host = bmi_emulation::detect_host(),
                         avx2_emulation::Host avx_host = avx2_emulation::detect_host()) {
    Stats st{};
    if (!base || image_size < 0x200 || !enabled()) return st;

    if (bmi_host.bmi1 && bmi_host.bmi2 && bmi_host.movbe && avx_host.avx2 && avx_host.fma)
        return st;

    const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(base);
    if (dos->e_magic != IMAGE_DOS_SIGNATURE || dos->e_lfanew <= 0 ||
        uint64_t(dos->e_lfanew) + sizeof(IMAGE_NT_HEADERS64) > image_size)
        return st;
    const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(base + dos->e_lfanew);
    if (nt->Signature != IMAGE_NT_SIGNATURE || nt->OptionalHeader.Magic != IMAGE_NT_OPTIONAL_HDR64_MAGIC) return st;

    const uint64_t img_base = reinterpret_cast<uint64_t>(base);

    // Collect executable and non-executable section ranges.
    std::vector<std::pair<uint32_t, uint32_t>> exec_secs, data_secs;
    const IMAGE_SECTION_HEADER* sec = IMAGE_FIRST_SECTION(nt);
    for (unsigned i = 0; i < nt->FileHeader.NumberOfSections; ++i, ++sec) {
        if (sec->VirtualAddress >= image_size) continue;
        const uint32_t sz = uint32_t(std::min<uint64_t>(
            sec->Misc.VirtualSize ? sec->Misc.VirtualSize : sec->SizeOfRawData, image_size - sec->VirtualAddress));
        if (!sz) continue;
        if (sec->Characteristics & IMAGE_SCN_MEM_EXECUTE) exec_secs.push_back({sec->VirtualAddress, sz});
        else data_secs.push_back({sec->VirtualAddress, sz});
    }
    if (exec_secs.empty()) return st;

    // Build bitset of every RVA targeted by exports, relocations, .pdata unwind scopes, 32-bit RVA tables in data/gaps,
    // and direct intra-function branches (used to guarantee 4-byte instruction fusion never crosses a jump target).
    std::vector<uint8_t> target_bits((std::size_t(image_size) + 7u) >> 3, 0);
    const auto mark_rva = [&](uint64_t rva) {
        if (rva < image_size) target_bits[std::size_t(rva) >> 3] |= uint8_t(1u << (rva & 7u));
    };
    const auto is_marked_rva = [&](uint32_t rva) -> bool {
        return rva >= image_size || ((target_bits[rva >> 3] >> (rva & 7u)) & 1u) != 0;
    };

    if (nt->OptionalHeader.AddressOfEntryPoint) mark_rva(nt->OptionalHeader.AddressOfEntryPoint);

    const auto get_dir = [&](unsigned idx) -> IMAGE_DATA_DIRECTORY {
        return idx < nt->OptionalHeader.NumberOfRvaAndSizes ? nt->OptionalHeader.DataDirectory[idx]
                                                            : IMAGE_DATA_DIRECTORY{};
    };

    // 1. Exports
    const auto exp_dir = get_dir(IMAGE_DIRECTORY_ENTRY_EXPORT);
    if (exp_dir.VirtualAddress && exp_dir.Size >= sizeof(IMAGE_EXPORT_DIRECTORY) &&
        uint64_t(exp_dir.VirtualAddress) + sizeof(IMAGE_EXPORT_DIRECTORY) <= image_size) {
        const auto* ed = reinterpret_cast<const IMAGE_EXPORT_DIRECTORY*>(base + exp_dir.VirtualAddress);
        if (uint64_t(ed->AddressOfFunctions) + uint64_t(ed->NumberOfFunctions) * 4u <= image_size) {
            const auto* fns = reinterpret_cast<const uint32_t*>(base + ed->AddressOfFunctions);
            for (uint32_t i = 0; i < ed->NumberOfFunctions; ++i) mark_rva(fns[i]);
        }
    }

    // 2. Relocations (any absolute 64-bit pointer inside the image)
    const auto rel_dir = get_dir(IMAGE_DIRECTORY_ENTRY_BASERELOC);
    if (rel_dir.VirtualAddress && rel_dir.Size &&
        uint64_t(rel_dir.VirtualAddress) + rel_dir.Size <= image_size) {
        const uint8_t* p = base + rel_dir.VirtualAddress;
        const uint8_t* end = p + rel_dir.Size;
        while (p + sizeof(IMAGE_BASE_RELOCATION) <= end) {
            const auto* blk = reinterpret_cast<const IMAGE_BASE_RELOCATION*>(p);
            if (blk->SizeOfBlock < sizeof(IMAGE_BASE_RELOCATION) || p + blk->SizeOfBlock > end) break;
            const auto* ents = reinterpret_cast<const uint16_t*>(blk + 1);
            const std::size_t cnt = (blk->SizeOfBlock - sizeof(IMAGE_BASE_RELOCATION)) / 2;
            for (std::size_t i = 0; i < cnt; ++i) {
                if ((ents[i] >> 12) == IMAGE_REL_BASED_DIR64) {
                    const uint64_t rva = uint64_t(blk->VirtualAddress) + (ents[i] & 0xFFFu);
                    if (rva + 8 <= image_size) {
                        uint64_t va = 0;
                        std::memcpy(&va, base + rva, 8);
                        if (va >= img_base && va - img_base < image_size) mark_rva(va - img_base);
                    }
                }
            }
            p += blk->SizeOfBlock;
        }
    }

    // 3. Non-executable sections: mark every 4-byte uint32_t RVA candidate (1-byte stride) in [0, image_size) and 8-byte aligned VAs
    for (const auto& [drva, dsz] : data_secs) {
        const uint8_t* dp = base + drva;
        for (uint32_t off = 0; off + 4 <= dsz; ++off) {
            uint32_t w = 0;
            std::memcpy(&w, dp + off, 4);
            if (w < image_size) mark_rva(w);
            if ((off & 7u) == 0 && off + 8 <= dsz) {
                uint64_t q = 0;
                std::memcpy(&q, dp + off, 8);
                if (q >= img_base && q - img_base < image_size) mark_rva(q - img_base);
            }
        }
    }

    // 4. .pdata table
    const auto exc_dir = get_dir(IMAGE_DIRECTORY_ENTRY_EXCEPTION);
    const RUNTIME_FUNCTION* pdata = nullptr;
    uint32_t pdata_count = 0;
    if (exc_dir.VirtualAddress && exc_dir.Size >= sizeof(RUNTIME_FUNCTION) && exc_dir.VirtualAddress < image_size) {
        const uint64_t max_b = std::min<uint64_t>(exc_dir.Size, image_size - exc_dir.VirtualAddress);
        pdata_count = uint32_t(max_b / sizeof(RUNTIME_FUNCTION));
        pdata = reinterpret_cast<const RUNTIME_FUNCTION*>(base + exc_dir.VirtualAddress);
        for (uint32_t i = 0; i < pdata_count; ++i) {
            mark_rva(pdata[i].BeginAddress);
            mark_rva(pdata[i].EndAddress);
            const uint32_t uw_rva = pdata[i].UnwindData & ~1u;
            if (uw_rva + 4 <= image_size) {
                const uint8_t flags = base[uw_rva] >> 3;
                const uint8_t codes = base[uw_rva + 2];
                const uint32_t after_codes = uw_rva + 4u + uint32_t((codes + 1u) & ~1u) * 2u;
                if ((flags & 3u) && after_codes + 8 <= image_size) {
                    uint32_t handler_rva = 0, scope_cnt = 0;
                    std::memcpy(&handler_rva, base + after_codes, 4);
                    std::memcpy(&scope_cnt, base + after_codes + 4, 4);
                    mark_rva(handler_rva);
                    if (scope_cnt <= 4096 && after_codes + 8u + scope_cnt * 16u <= image_size) {
                        for (uint32_t s = 0; s < scope_cnt; ++s) {
                            uint32_t sb = 0, se = 0, sh = 0, sj = 0;
                            std::memcpy(&sb, base + after_codes + 8u + s * 16u + 0u, 4);
                            std::memcpy(&se, base + after_codes + 8u + s * 16u + 4u, 4);
                            std::memcpy(&sh, base + after_codes + 8u + s * 16u + 8u, 4);
                            std::memcpy(&sj, base + after_codes + 8u + s * 16u + 12u, 4);
                            mark_rva(sb); mark_rva(se); mark_rva(sh); mark_rva(sj);
                        }
                    }
                }
            }
        }
    }

    enum class SiteKind : uint8_t { bmi, avx2, fma3, fused_4b };
    struct PendingStub {
        SiteKind kind;
        uint32_t rva;
        uint8_t span_len;
        uint8_t ins2_len;  // non-zero only for fused_4b
        bmi_emulation::Decoded bd;
        avx2_emulation::Decoded ad;
    };
    std::vector<PendingStub> pending;

    const auto mark_block_targets = [&](uint32_t beg_rva, uint32_t end_rva, bool require_term) -> bool {
        const std::size_t len = end_rva - beg_rva;
        const uint8_t* block = base + beg_rva;
        std::size_t pos = 0, last_pos = 0, last_len = 0;
        while (pos < len) {
            const std::size_t ilen = bmi_emulation::detail::insn_len64(block + pos, len - pos, nullptr);
            if (!ilen) return false;
            int64_t rel = 0;
            if (detail::is_control_flow(block + pos, ilen, &rel) && rel != 0) {
                const int64_t tgt = int64_t(beg_rva) + int64_t(pos + ilen) + rel;
                if (tgt >= 0 && tgt < int64_t(image_size)) mark_rva(uint64_t(tgt));
            } else {
                const std::size_t rip_off = detail::find_rip_disp32_offset(block + pos, ilen);
                if (rip_off != 0 && rip_off != SIZE_MAX) {
                    int32_t disp = 0;
                    std::memcpy(&disp, block + pos + rip_off, 4);
                    const int64_t tgt = int64_t(beg_rva) + int64_t(pos + ilen) + disp;
                    if (tgt >= 0 && tgt < int64_t(image_size)) mark_rva(uint64_t(tgt));
                }
            }
            last_pos = pos;
            last_len = ilen;
            pos += ilen;
        }
        if (pos != len || !last_len) return false;
        if (require_term && !bmi_emulation::detail::is_block_terminator(block + last_pos, last_len)) return false;
        return true;
    };

    const auto emit_block_sites = [&](uint32_t beg_rva, uint32_t end_rva) {
        const std::size_t len = end_rva - beg_rva;
        uint8_t* block = base + beg_rva;

        struct Ins {
            uint32_t pos;
            uint8_t ilen;
            uint8_t tz_lz;
        };
        std::vector<Ins> insns;
        insns.reserve(len / 4 + 1);

        std::size_t pos = 0;
        while (pos < len) {
            int tz_lz = 0;
            const std::size_t ilen = bmi_emulation::detail::insn_len64(block + pos, len - pos, &tz_lz);
            if (!ilen) return;
            insns.push_back({uint32_t(pos), uint8_t(ilen), uint8_t(tz_lz)});
            pos += ilen;
        }
        if (pos != len || insns.empty()) return;

        for (std::size_t k = 0; k < insns.size(); ++k) {
            const uint32_t rva1 = beg_rva + insns[k].pos;
            uint8_t* p1 = base + rva1;
            const std::size_t len1 = insns[k].ilen;

            // Check AVX2 / FMA3 first:
            const auto ad = avx2_emulation::decode(p1, len1);
            if (ad.cat == avx2_emulation::Category::avx2 || ad.cat == avx2_emulation::Category::fma3) {
                if (ad.vmap == 2 && (ad.op == 0x18 || ad.op == 0x19) && ad.mem) continue;
                if (detail::try_avx2_inplace(p1, ad, avx_host, true)) {
                    ++st.inplace_sites;
                    continue;
                }
                if (ad.length >= 5) {
                    pending.push_back({ad.cat == avx2_emulation::Category::fma3 ? SiteKind::fma3 : SiteKind::avx2,
                                       rva1, ad.length, 0, {}, ad});
                    continue;
                }
                // 4-byte 0xC5 AVX2 site: try forward fusion with insns[k + 1] when rva2 is not a jump/table target.
                if (ad.length == 4 && k + 1 < insns.size()) {
                    const uint32_t rva2 = beg_rva + insns[k + 1].pos;
                    const uint8_t len2 = insns[k + 1].ilen;
                    const uint8_t* p2 = base + rva2;
                    if (insns[k + 1].tz_lz == 0 && !is_marked_rva(rva2) && !detail::is_control_flow(p2, len2)) {
                        pending.push_back({SiteKind::fused_4b, rva1, uint8_t(4 + len2), len2, {}, ad});
                        ++k;  // consume ins2
                    }
                }
                continue;
            }

            // Check BMI1 / BMI2 / MOVBE:
            if (insns[k].tz_lz == 0) {
                const auto bd = bmi_emulation::decode(p1, len1);
                if (bd.op != bmi_emulation::Op::none && bd.length >= 5) {
                    pending.push_back({SiteKind::bmi, rva1, bd.length, 0, bd, {}});
                }
            }
        }
    };

    // First pass over gaps in executable sections to mark any switch RVA tables sitting between .pdata functions.
    std::vector<std::pair<uint32_t, uint32_t>> all_funcs, all_gaps, valid_blocks;
    for (const auto& [sec_rva, sec_sz] : exec_secs) {
        const uint32_t sec_end = sec_rva + sec_sz;
        std::vector<std::pair<uint32_t, uint32_t>> funcs;
        if (pdata && pdata_count) {
            for (uint32_t i = 0; i < pdata_count; ++i) {
                const uint32_t b = pdata[i].BeginAddress, e = pdata[i].EndAddress;
                if (e > b && b >= sec_rva && e <= sec_end) funcs.push_back({b, e});
            }
            std::sort(funcs.begin(), funcs.end());
        }
        uint32_t cur = sec_rva;
        for (const auto& [fb, fe] : funcs) {
            if (fb > cur) all_gaps.push_back({cur, fb});
            if (fe > cur) {
                all_funcs.push_back({std::max(fb, cur), fe});
                cur = fe;
            }
        }
        if (cur < sec_end) all_gaps.push_back({cur, sec_end});
    }

    for (const auto& [gb, ge] : all_gaps) {
        const uint8_t* gp = base + gb;
        const uint32_t gsz = ge - gb;
        for (uint32_t off = 0; off + 4 <= gsz; ++off) {
            uint32_t w = 0;
            std::memcpy(&w, gp + off, 4);
            if (w < image_size) mark_rva(w);
        }
    }

    // Pass 1: validate blocks and mark all direct branch & RIP-relative targets across the entire module.
    valid_blocks.reserve(all_funcs.size() + 1024);
    for (const auto& [fb, fe] : all_funcs) {
        if (mark_block_targets(fb, fe, false)) valid_blocks.push_back({fb, fe});
    }
    for (const auto& [gb, ge] : all_gaps) {
        const uint8_t* g = base + gb;
        const std::size_t glen = ge - gb;
        std::size_t p = 0;
        while (p < glen) {
            while (p < glen && g[p] == 0xCC) ++p;
            if (p >= glen) break;
            std::size_t sub_end = p;
            while (sub_end < glen && !(sub_end + 1 < glen && g[sub_end] == 0xCC && g[sub_end + 1] == 0xCC)) ++sub_end;
            while (sub_end > p && g[sub_end - 1] == 0xCC) --sub_end;
            if (sub_end > p && mark_block_targets(gb + uint32_t(p), gb + uint32_t(sub_end), true))
                valid_blocks.push_back({gb + uint32_t(p), gb + uint32_t(sub_end)});
            p = sub_end + 1;
        }
    }

    // Pass 2: apply in-place rewrites and collect trampoline sites now that `target_bits` is complete.
    for (const auto& [bb, be] : valid_blocks) emit_block_sites(bb, be);

    // Free the temporary target bitset immediately before allocating stubs.
    {
        std::vector<uint8_t>().swap(target_bits);
    }

    if (pending.empty()) return st;

    const std::size_t reserve_bytes = (pending.size() * 96u + 0xFFFFu) & ~std::size_t(0xFFFFu);
    uint8_t* stubs = detail::reserve_stub_region(img_base, img_base + image_size, reserve_bytes);
    if (!stubs) return st;

    std::size_t used = 0;
    std::size_t committed = 0;
    std::vector<uint8_t> body;
    body.reserve(160);

    for (const auto& s : pending) {
        body.clear();
        uint8_t* site_ptr = base + s.rva;
        const uint64_t site_va = img_base + s.rva;
        const uint64_t stub_va = reinterpret_cast<uint64_t>(stubs + used);
        const int64_t jump_to_stub = int64_t(stub_va) - int64_t(site_va + 5);
        if (jump_to_stub < INT32_MIN || jump_to_stub > INT32_MAX) continue;

        bool ok = false;
        if (s.kind == SiteKind::bmi) {
            ok = detail::emit_bmi_body(body, s.bd, site_va, stub_va, bmi_host);
        } else if (s.kind == SiteKind::avx2 || s.kind == SiteKind::fma3) {
            ok = detail::emit_avx2_fma_body(body, s.ad, site_va, stub_va, avx_host);
        } else if (s.kind == SiteKind::fused_4b) {
            ok = detail::emit_avx2_fma_body(body, s.ad, site_va, stub_va, avx_host) &&
                 detail::emit_second_insn(body, site_ptr + 4, s.ins2_len, site_va + 4, stub_va, bmi_host, avx_host);
        }
        if (!ok) continue;

        const uint64_t resume_va = site_va + s.span_len;
        const int64_t jump_back = int64_t(resume_va) - int64_t(stub_va + body.size() + 5);
        if (jump_back < INT32_MIN || jump_back > INT32_MAX) continue;
        detail::emit_jmp_rel32(body, stub_va, resume_va);

        if (used + body.size() > reserve_bytes) break;
        if (used + body.size() > committed) {
            const std::size_t want_commit =
                std::min(reserve_bytes, (used + body.size() + 0xFFFFu) & ~std::size_t(0xFFFFu));
            if (!VirtualAlloc(stubs + committed, want_commit - committed, MEM_COMMIT, PAGE_READWRITE)) break;
            committed = want_commit;
        }

        std::memcpy(stubs + used, body.data(), body.size());
        used += (body.size() + 15u) & ~std::size_t(15u);

        const int32_t rel32 = int32_t(jump_to_stub);
        site_ptr[0] = 0xE9;
        std::memcpy(site_ptr + 1, &rel32, 4);
        for (uint8_t k = 5; k < s.span_len; ++k) site_ptr[k] = 0x90;

        if (s.kind == SiteKind::bmi) ++st.bmi_stubs;
        else if (s.kind == SiteKind::fma3) ++st.fma3_stubs;
        else if (s.kind == SiteKind::fused_4b) { ++st.avx2_stubs; ++st.avx2_fused_4b; }
        else ++st.avx2_stubs;
    }

    if (used > 0) {
        DWORD old = 0;
        VirtualProtect(stubs, committed, PAGE_EXECUTE_READ, &old);
        FlushInstructionCache(GetCurrentProcess(), stubs, used);
        st.stubs = stubs;
        st.stub_bytes = used;
        st.stub_reserved = reserve_bytes;
    } else {
        VirtualFree(stubs, 0, MEM_RELEASE);
    }
    return st;
}

}  // namespace rt::isa_jit_patch
#endif
