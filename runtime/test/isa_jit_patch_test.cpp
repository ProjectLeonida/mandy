// Tests for the ahead-of-time AVX1 + SSE4.2 stub compiler and in-place VEX rewriter (src/isa_jit_patch.h):
//   1. Live hardware execution of emitted BMI1, BMI2, MOVBE, AVX2 (in-place and out-of-line) and FMA3 stubs across
//      randomized GPR/YMM/RFLAGS states, verifying zero #UD traps and architectural parity against reference execution.
//   2. Forward fusion of 4-byte 0xC5 AVX2 instructions with 6-byte VEX and 8-byte RIP-relative instructions.
//   3. Full synthetic PE32+ image patching via patch_image(), verifying in-place rewrites, BMI/AVX2/FMA3 trampolines,
//      4-byte forward fusion when unmarked, fusion refusal across RVA-table / branch targets, and LEONIDA_ISA_PATCH=0.
// Synthetic machine-code buffers and PE headers only; no game files.
#include <windows.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "isa_jit_patch.h"

namespace {

struct alignas(64) CpuState {
    uint64_t gpr[16];  // gpr[4] carries guest [rsp+16] scratch payload across the harness
    uint64_t rflags;
    rt::avx2_emulation::Ymm ymm[16];
};

// Builds a small x64 machine-code harness that loads CpuState (RAX, RCX, RDX, RBX, RBP, RSI, RDI, R8..R15, RFLAGS,
// YMM0..YMM15), jumps to `entry_va`, and at `out_resume_va` stores the resulting state back into CpuState.
uint8_t* build_harness(uint8_t* page, uint64_t entry_va, uint64_t& out_resume_va) {
    using namespace rt::isa_jit_patch::detail;
    std::vector<uint8_t> c;
    const uint8_t saved_regs[] = {3, 5, 6, 7, 12, 13, 14, 15};
    for (uint8_t r : saved_regs) emit_push_gpr(c, r);
    // [rsp+0..63] = guest stack scratch, [rsp+64] = CpuState*, [rsp+80..239] = saved XMM6..XMM15
    emit_u8(c, 0x48); emit_u8(c, 0x81); emit_u8(c, 0xEC); emit_u32(c, 256);  // sub rsp, 256
    emit_u8(c, 0x48); emit_u8(c, 0x89); emit_u8(c, 0x4C); emit_u8(c, 0x24); emit_u8(c, 64);  // mov [rsp+64], rcx
    for (int i = 6; i < 16; ++i) {
        emit_vex3(c, 1, uint8_t(i), 0, 4, 0, 0, 0, 2, 0x7F);
        emit_u8(c, uint8_t(0x84 | ((i & 7) << 3)));
        emit_u8(c, 0x24);
        emit_i32(c, 80 + (i - 6) * 16);
    }
    emit_u8(c, 0x48); emit_u8(c, 0x89); emit_u8(c, 0xC8);  // mov rax, rcx
    const int32_t ymm_off = int32_t(offsetof(CpuState, ymm));
    for (int i = 0; i < 16; ++i) {
        emit_vex3(c, 1, uint8_t(i), 0, 0, 0, 0, 1, 2, 0x6F);  // vmovdqu ymm_i, [rax + disp32]
        emit_u8(c, uint8_t(0x80 | ((i & 7) << 3)));
        emit_i32(c, ymm_off + i * 32);
    }
    // Seed [rsp + 16] with CpuState->gpr[4] for [rsp + 16] memory-operand tests.
    emit_u8(c, 0x48); emit_u8(c, 0x8B); emit_u8(c, 0x48); emit_u8(c, int8_t(4 * 8));
    emit_u8(c, 0x48); emit_u8(c, 0x89); emit_u8(c, 0x4C); emit_u8(c, 0x24); emit_u8(c, 16);
    // Load RFLAGS
    emit_u8(c, 0xFF); emit_u8(c, 0xB0); emit_i32(c, int32_t(offsetof(CpuState, rflags)));
    emit_u8(c, 0x9D);
    // Load GPRs 1..15 (except RSP=4), then RAX last.
    for (int r = 1; r < 16; ++r) {
        if (r == 4) continue;
        emit_rex(c, true, uint8_t(r), 0, 0);
        emit_u8(c, 0x8B);
        emit_u8(c, uint8_t(0x80 | ((r & 7) << 3)));
        emit_i32(c, r * 8);
    }
    emit_u8(c, 0x48); emit_u8(c, 0x8B); emit_u8(c, 0x00);  // mov rax, [rax]
    emit_jmp_rel32(c, reinterpret_cast<uint64_t>(page), entry_va);

    out_resume_va = reinterpret_cast<uint64_t>(page) + c.size();
    emit_u8(c, 0x50);  // push rax
    emit_u8(c, 0x9C);  // pushfq
    emit_u8(c, 0x48); emit_u8(c, 0x8B); emit_u8(c, 0x44); emit_u8(c, 0x24); emit_u8(c, 80);  // mov rax, [rsp+80]
    for (int r = 1; r < 16; ++r) {
        if (r == 4) continue;
        emit_rex(c, true, uint8_t(r), 0, 0);
        emit_u8(c, 0x89);
        emit_u8(c, uint8_t(0x80 | ((r & 7) << 3)));
        emit_i32(c, r * 8);
    }
    emit_u8(c, 0x59);  // pop rcx (rflags)
    emit_u8(c, 0x48); emit_u8(c, 0x89); emit_u8(c, 0x88); emit_i32(c, int32_t(offsetof(CpuState, rflags)));
    emit_u8(c, 0x59);  // pop rcx (rax)
    emit_u8(c, 0x48); emit_u8(c, 0x89); emit_u8(c, 0x08);
    emit_u8(c, 0x48); emit_u8(c, 0x8B); emit_u8(c, 0x4C); emit_u8(c, 0x24); emit_u8(c, 16);
    emit_u8(c, 0x48); emit_u8(c, 0x89); emit_u8(c, 0x48); emit_u8(c, 32);
    for (int i = 0; i < 16; ++i) {
        emit_vex3(c, 1, uint8_t(i), 0, 0, 0, 0, 1, 2, 0x7F);
        emit_u8(c, uint8_t(0x80 | ((i & 7) << 3)));
        emit_i32(c, ymm_off + i * 32);
    }
    emit_u8(c, 0xC5); emit_u8(c, 0xF8); emit_u8(c, 0x77);  // vzeroupper
    for (int i = 6; i < 16; ++i) {
        emit_vex3(c, 1, uint8_t(i), 0, 4, 0, 0, 0, 2, 0x6F);
        emit_u8(c, uint8_t(0x84 | ((i & 7) << 3)));
        emit_u8(c, 0x24);
        emit_i32(c, 80 + (i - 6) * 16);
    }
    emit_u8(c, 0x48); emit_u8(c, 0x81); emit_u8(c, 0xC4); emit_u32(c, 256);  // add rsp, 256
    for (int k = 7; k >= 0; --k) emit_pop_gpr(c, saved_regs[k]);
    emit_u8(c, 0xC3);  // ret

    std::memcpy(page, c.data(), c.size());
    return page;
}

uint64_t splitmix64(uint64_t& x) {
    uint64_t z = (x += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

// Builds a synthetic in-memory PE32+ image (0x4000 bytes: headers, .text at 0x1000, .rdata at 0x2000, .pdata at 0x3000)
// to test patch_image() end-to-end without any external binaries.
bool test_synthetic_pe_patch() {
    constexpr uint32_t kImgSize = 0x4000;
    uint8_t* img = static_cast<uint8_t*>(
        VirtualAlloc(nullptr, kImgSize, MEM_RESERVE | MEM_COMMIT, PAGE_EXECUTE_READWRITE));
    if (!img) return false;
    std::memset(img, 0, kImgSize);
    std::memset(img + 0x1000, 0xCC, 0x1000);

    auto* dos = reinterpret_cast<IMAGE_DOS_HEADER*>(img);
    dos->e_magic = IMAGE_DOS_SIGNATURE;
    dos->e_lfanew = 0x80;
    auto* nt = reinterpret_cast<IMAGE_NT_HEADERS64*>(img + dos->e_lfanew);
    nt->Signature = IMAGE_NT_SIGNATURE;
    nt->FileHeader.Machine = IMAGE_FILE_MACHINE_AMD64;
    nt->FileHeader.NumberOfSections = 3;
    nt->FileHeader.SizeOfOptionalHeader = sizeof(IMAGE_OPTIONAL_HEADER64);
    nt->OptionalHeader.Magic = IMAGE_NT_OPTIONAL_HDR64_MAGIC;
    nt->OptionalHeader.SizeOfHeaders = 0x400;
    nt->OptionalHeader.SizeOfImage = kImgSize;
    nt->OptionalHeader.NumberOfRvaAndSizes = IMAGE_NUMBEROF_DIRECTORY_ENTRIES;
    nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_EXCEPTION] = {0x3000, sizeof(RUNTIME_FUNCTION)};

    auto* secs = IMAGE_FIRST_SECTION(nt);
    std::memcpy(secs[0].Name, ".text", 5);
    secs[0].VirtualAddress = 0x1000;
    secs[0].Misc.VirtualSize = 0x1000;
    secs[0].SizeOfRawData = 0x1000;
    secs[0].Characteristics = IMAGE_SCN_CNT_CODE | IMAGE_SCN_MEM_EXECUTE | IMAGE_SCN_MEM_READ;

    std::memcpy(secs[1].Name, ".rdata", 6);
    secs[1].VirtualAddress = 0x2000;
    secs[1].Misc.VirtualSize = 0x1000;
    secs[1].SizeOfRawData = 0x1000;
    secs[1].Characteristics = IMAGE_SCN_CNT_INITIALIZED_DATA | IMAGE_SCN_MEM_READ;

    std::memcpy(secs[2].Name, ".pdata", 6);
    secs[2].VirtualAddress = 0x3000;
    secs[2].Misc.VirtualSize = 0x1000;
    secs[2].SizeOfRawData = 0x1000;
    secs[2].Characteristics = IMAGE_SCN_CNT_INITIALIZED_DATA | IMAGE_SCN_MEM_READ;

    // Function at RVA 0x1000 (uint64_t fn(uint64_t rcx, uint64_t rdx)):
    //   +0x00 (5B): shlx rax, rcx, rdx          (C4 E2 E9 F7 C1)          -> BMI2 stub
    //   +0x05 (4B): vpxor ymm0, ymm0, ymm0      (C5 FD EF C0)             -> in-place rewrite to vxorps
    //   +0x09 (4B): vpaddd ymm1, ymm0, ymm0     (C5 FD FE C8)             \  fused 4B + 6B pair
    //   +0x0D (6B): vpermq ymm1, ymm1, 0xD8     (C4 E3 FD 00 C9 D8)       /  (interior 0x100D is unmarked)
    //   +0x13 (4B): vpaddd ymm2, ymm0, ymm0     (C5 FD FE D0)             <- NOT fused: 0x1017 is in .rdata RVA table!
    //   +0x17 (6B): vpermq ymm2, ymm2, 0x1B     (C4 E3 FD 00 D2 1B)       -> standalone >=5B AVX2 stub
    //   +0x1D (5B): vfmadd213ss xmm0, xmm0, xmm0 (C4 E2 79 A9 C0)         -> FMA3 stub
    //   +0x22 (3B): vzeroupper                  (C5 F8 77)
    //   +0x25 (1B): ret                         (C3)
    const uint8_t fn_bytes[] = {
        0xC4, 0xE2, 0xE9, 0xF7, 0xC1,
        0xC5, 0xFD, 0xEF, 0xC0,
        0xC5, 0xFD, 0xFE, 0xC8,
        0xC4, 0xE3, 0xFD, 0x00, 0xC9, 0xD8,
        0xC5, 0xFD, 0xFE, 0xD0,
        0xC4, 0xE3, 0xFD, 0x00, 0xD2, 0x1B,
        0xC4, 0xE2, 0x79, 0xA9, 0xC0,
        0xC5, 0xF8, 0x77,
        0xC3,
    };
    std::memcpy(img + 0x1000, fn_bytes, sizeof(fn_bytes));

    // Place interior RVA 0x1017 in .rdata jump table so +0x13 (4B vpaddd) must NOT fuse across 0x1017.
    const uint32_t jump_target_rva = 0x1017u;
    std::memcpy(img + 0x2010, &jump_target_rva, sizeof(jump_target_rva));

    // Minimal UNWIND_INFO at 0x2100 + RUNTIME_FUNCTION at 0x3000.
    img[0x2100] = 0x01;  // Version 1, no flags, 0 prolog, 0 unwind codes
    auto* rf = reinterpret_cast<RUNTIME_FUNCTION*>(img + 0x3000);
    rf->BeginAddress = 0x1000;
    rf->EndAddress = 0x1000 + uint32_t(sizeof(fn_bytes));
    rf->UnwindData = 0x2100;

    const rt::bmi_emulation::Host pre_haswell_bmi{false, false, false, false};
    const rt::avx2_emulation::Host pre_haswell_avx{false, false, false};
    const rt::bmi_emulation::Host full_bmi{true, true, true, true};
    const rt::avx2_emulation::Host full_avx{true, true, true};

    // 1. LEONIDA_ISA_PATCH=0 disables patching completely.
    _putenv_s("LEONIDA_ISA_PATCH", "0");
    const auto disabled_st = rt::isa_jit_patch::patch_image(img, kImgSize, pre_haswell_bmi, pre_haswell_avx);
    _putenv_s("LEONIDA_ISA_PATCH", "");
    if (disabled_st.inplace_sites != 0 || disabled_st.stubs != nullptr ||
        std::memcmp(img + 0x1000, fn_bytes, sizeof(fn_bytes)) != 0) {
        std::fprintf(stderr, "FAIL: LEONIDA_ISA_PATCH=0 modified image\n");
        VirtualFree(img, 0, MEM_RELEASE);
        return false;
    }

    // 2. Host with native BMI1/BMI2/MOVBE/AVX2/FMA3 skips patching completely.
    const auto native_st = rt::isa_jit_patch::patch_image(img, kImgSize, full_bmi, full_avx);
    if (native_st.inplace_sites != 0 || native_st.stubs != nullptr) {
        std::fprintf(stderr, "FAIL: full-ISA host should emit zero patches\n");
        VirtualFree(img, 0, MEM_RELEASE);
        return false;
    }

    // 3. Pre-Haswell host patches 1 in-place, 1 BMI, 2 AVX2 (1 fused 4B + 1 standalone 6B), 1 FMA3, leaving +0x13 untouched.
    const auto st = rt::isa_jit_patch::patch_image(img, kImgSize, pre_haswell_bmi, pre_haswell_avx);
    const bool counts_ok = st.inplace_sites == 1 && st.bmi_stubs == 1 && st.avx2_stubs == 2 &&
                           st.avx2_fused_4b == 1 && st.fma3_stubs == 1 && st.stubs != nullptr &&
                           img[0x1000 + 0x13] == 0xC5;  // unfused 4B site before marked RVA 0x1017 stays intact
    if (!counts_ok) {
        std::fprintf(stderr,
                     "FAIL: unexpected patch_image stats inplace=%zu bmi=%zu avx2=%zu fused=%zu fma3=%zu byte13=0x%02X\n",
                     st.inplace_sites, st.bmi_stubs, st.avx2_stubs, st.avx2_fused_4b, st.fma3_stubs,
                     unsigned(img[0x1000 + 0x13]));
        if (st.stubs) VirtualFree(st.stubs, 0, MEM_RELEASE);
        VirtualFree(img, 0, MEM_RELEASE);
        return false;
    }

    FlushInstructionCache(GetCurrentProcess(), img + 0x1000, sizeof(fn_bytes));
    auto fn = reinterpret_cast<uint64_t (*)(uint64_t, uint64_t)>(img + 0x1000);
    const uint64_t out = fn(0x123456789ABCDEF0ull, 8);
    const bool exec_ok = (out == (0x123456789ABCDEF0ull << 8));

    VirtualFree(st.stubs, 0, MEM_RELEASE);
    VirtualFree(img, 0, MEM_RELEASE);
    if (!exec_ok) {
        std::fprintf(stderr, "FAIL: patched synthetic PE function returned 0x%016llX\n",
                     static_cast<unsigned long long>(out));
        return false;
    }
    return true;
}

}  // namespace

int main() {
    if (!rt::bmi_emulation::install() || !rt::avx2_emulation::install()) {
        std::fprintf(stderr, "FAIL: could not install fallback VEH handlers\n");
        return 1;
    }

    uint8_t* arena = static_cast<uint8_t*>(
        VirtualAlloc(nullptr, 0x20000, MEM_RESERVE | MEM_COMMIT, PAGE_EXECUTE_READWRITE));
    if (!arena) return 1;
    uint8_t* harness_page = arena;
    uint8_t* site_page = arena + 0x4000;
    uint8_t* stub_page = arena + 0x8000;
    uint8_t* data_page = arena + 0xC000;
    uint8_t* ref_page = arena + 0x10000;

    uint64_t rng = 0x123456789ABCDEF0ull;
    for (int i = 0; i < 4096; ++i) data_page[i] = uint8_t(splitmix64(rng));

    const auto real_bmi = rt::bmi_emulation::detect_host();
    const auto real_avx = rt::avx2_emulation::detect_host();
    std::printf("isa_jit_patch_test host: bmi1=%d bmi2=%d movbe=%d avx2=%d fma=%d f16c=%d\n",
                real_bmi.bmi1, real_bmi.bmi2, real_bmi.movbe, real_avx.avx2, real_avx.fma, real_avx.f16c);

    // Force pre-Haswell target profile for the emitters so every AVX1/x64 stub is generated and executed on any x86-64
    // AVX+ host (comparing against #UD emulation on pre-Haswell CPUs and against hardware silicon on AVX2+ CPUs).
    const rt::bmi_emulation::Host pre_bmi{false, false, false, false};
    const rt::avx2_emulation::Host pre_avx{false, false, false};

    auto run_and_compare = [&](const char* label, const std::vector<uint8_t>& ins_bytes, uint32_t flag_mask,
                               bool allow_fma_ulp) -> bool {
        for (int trial = 0; trial < 8; ++trial) {
            CpuState init{};
            for (int r = 0; r < 16; ++r) init.gpr[r] = splitmix64(rng);
            init.gpr[8] = reinterpret_cast<uint64_t>(data_page + 128 + (trial & 1 ? 3 : 0));
            init.gpr[9] = uint64_t(trial % 4);
            if (trial == 0) init.gpr[1] = 0;
            if (trial == 1) init.gpr[1] = 32;
            if (trial == 2) init.gpr[1] = 64;
            if (trial == 3) init.gpr[1] = 0x0000ull | 5;
            if (trial == 4) init.gpr[1] = 0x4005ull;
            init.rflags = 0x202ull | ((trial & 1) ? 0x8D5ull : 0ull);
            for (int y = 0; y < 16; ++y) {
                for (int k = 0; k < 8; ++k) {
                    if (allow_fma_ulp) {
                        init.ymm[y].f32[k] = float(int(splitmix64(rng) % 2000) - 1000) * 0.125f;
                    } else {
                        init.ymm[y].u32[k] = uint32_t(splitmix64(rng));
                    }
                }
            }

            // 1. Reference execution (via #UD handler on pre-Haswell or native instruction on AVX2+ hosts)
            std::memcpy(ref_page, ins_bytes.data(), ins_bytes.size());
            uint64_t ref_resume = 0;
            build_harness(harness_page, reinterpret_cast<uint64_t>(ref_page), ref_resume);
            ref_page[ins_bytes.size()] = 0xE9;
            const int32_t rel_ref = int32_t(
                int64_t(ref_resume) - int64_t(reinterpret_cast<uint64_t>(ref_page) + ins_bytes.size() + 5));
            std::memcpy(ref_page + ins_bytes.size() + 1, &rel_ref, 4);
            FlushInstructionCache(GetCurrentProcess(), arena, 0x14000);

            CpuState ref_st = init;
            reinterpret_cast<void (*)(CpuState*)>(harness_page)(&ref_st);

            // 2. Ahead-of-time AVX1/x64 rewrite or stub execution (must take 0 #UDs on every host)
            const std::size_t slot_len = std::max<std::size_t>(ins_bytes.size(), 5);
            std::memset(site_page, 0x90, slot_len);
            std::memcpy(site_page, ins_bytes.data(), ins_bytes.size());
            uint64_t nat_resume = 0;
            build_harness(harness_page + 0x1000, reinterpret_cast<uint64_t>(site_page), nat_resume);
            site_page[slot_len] = 0xE9;
            const int32_t rel_nat_end =
                int32_t(int64_t(nat_resume) - int64_t(reinterpret_cast<uint64_t>(site_page) + slot_len + 5));
            std::memcpy(site_page + slot_len + 1, &rel_nat_end, 4);

            const uint64_t site_va = reinterpret_cast<uint64_t>(site_page);
            const uint64_t stub_va = reinterpret_cast<uint64_t>(stub_page);
            std::vector<uint8_t> body;
            const auto ad = rt::avx2_emulation::decode(site_page, ins_bytes.size());
            const auto bd = rt::bmi_emulation::decode(site_page, ins_bytes.size());
            bool patched = false;
            if (ad.cat != rt::avx2_emulation::Category::none && ad.length == ins_bytes.size()) {
                if (rt::isa_jit_patch::detail::try_avx2_inplace(site_page, ad, pre_avx, true)) {
                    patched = true;
                } else if (rt::isa_jit_patch::detail::emit_avx2_fma_body(body, ad, site_va, stub_va, pre_avx)) {
                    rt::isa_jit_patch::detail::emit_jmp_rel32(body, stub_va, site_va + slot_len);
                    std::memcpy(stub_page, body.data(), body.size());
                    site_page[0] = 0xE9;
                    const int32_t j = int32_t(int64_t(stub_va) - int64_t(site_va + 5));
                    std::memcpy(site_page + 1, &j, 4);
                    patched = true;
                }
            } else if (bd.op != rt::bmi_emulation::Op::none && bd.length == ins_bytes.size()) {
                if (rt::isa_jit_patch::detail::emit_bmi_body(body, bd, site_va, stub_va, pre_bmi)) {
                    rt::isa_jit_patch::detail::emit_jmp_rel32(body, stub_va, site_va + slot_len);
                    std::memcpy(stub_page, body.data(), body.size());
                    site_page[0] = 0xE9;
                    const int32_t j = int32_t(int64_t(stub_va) - int64_t(site_va + 5));
                    std::memcpy(site_page + 1, &j, 4);
                    patched = true;
                }
            }
            if (!patched) {
                std::fprintf(stderr, "FAIL [%s]: emitter rejected instruction\n", label);
                return false;
            }
            FlushInstructionCache(GetCurrentProcess(), arena, 0x14000);

            const uint64_t ud_before =
                rt::avx2_emulation::count(rt::avx2_emulation::Category::avx2).load() +
                rt::avx2_emulation::count(rt::avx2_emulation::Category::fma3).load() +
                rt::bmi_emulation::count(bd.op).load();

            CpuState nat_st = init;
            reinterpret_cast<void (*)(CpuState*)>(harness_page + 0x1000)(&nat_st);

            const uint64_t ud_after =
                rt::avx2_emulation::count(rt::avx2_emulation::Category::avx2).load() +
                rt::avx2_emulation::count(rt::avx2_emulation::Category::fma3).load() +
                rt::bmi_emulation::count(bd.op).load();

            if (ud_after != ud_before) {
                std::fprintf(stderr, "FAIL [%s]: native AVX1/x64 stub triggered #UD\n", label);
                return false;
            }

            for (int r = 0; r < 16; ++r) {
                if (nat_st.gpr[r] != ref_st.gpr[r]) {
                    std::fprintf(stderr, "FAIL [%s] trial %d: GPR[%d] nat=%016llx != ref=%016llx\n",
                                 label, trial, r, static_cast<unsigned long long>(nat_st.gpr[r]),
                                 static_cast<unsigned long long>(ref_st.gpr[r]));
                    return false;
                }
            }
            if ((nat_st.rflags ^ ref_st.rflags) & flag_mask) {
                std::fprintf(stderr, "FAIL [%s] trial %d: RFLAGS nat=%08llx != ref=%08llx (mask=%08x)\n",
                             label, trial, static_cast<unsigned long long>(nat_st.rflags),
                             static_cast<unsigned long long>(ref_st.rflags), flag_mask);
                return false;
            }
            for (int y = 0; y < 16; ++y) {
                if (!allow_fma_ulp) {
                    if (std::memcmp(&nat_st.ymm[y], &ref_st.ymm[y], 32) != 0) {
                        std::fprintf(stderr, "FAIL [%s] trial %d: YMM[%d] mismatch\n", label, trial, y);
                        return false;
                    }
                } else {
                    for (int k = 0; k < 8; ++k) {
                        if (std::fabs(nat_st.ymm[y].f32[k] - ref_st.ymm[y].f32[k]) > 1e-3f) {
                            std::fprintf(stderr, "FAIL [%s] trial %d: YMM[%d].f32[%d] nat=%f != ref=%f\n",
                                         label, trial, y, k, nat_st.ymm[y].f32[k], ref_st.ymm[y].f32[k]);
                            return false;
                        }
                    }
                }
            }
        }
        return true;
    };

    constexpr uint32_t kAllFlags = 0x8D5u;    // CF, PF, AF, ZF, SF, OF
    constexpr uint32_t kLogicFlags = 0x8C1u;  // CF, ZF, SF, OF
    constexpr uint32_t kBextrFlags = 0x841u;  // CF, ZF, OF

    struct Case {
        const char* name;
        std::vector<uint8_t> bytes;
        uint32_t fmask;
        bool fma;
    };
    const std::vector<Case> cases = {
        // BMI2 & MOVBE flag-preserving (including dst==rcx, fast-path rcx, mulx rax/rdx swap, [rsp+16], SIB)
        {"shlx_64_rcx_dst",      {0xC4, 0xE2, 0xF1, 0xF7, 0xCA}, kAllFlags, false},
        {"shrx_64_fast_rcx",     {0xC4, 0xE2, 0xF3, 0xF7, 0xD7}, kAllFlags, false},
        {"shlx_64_mem_r8",       {0xC4, 0x02, 0x91, 0xF7, 0x54, 0x88, 0x08}, kAllFlags, false},
        {"shlx_32_rsp16",        {0xC4, 0xE2, 0x71, 0xF7, 0x44, 0x24, 0x10}, kAllFlags, false},
        {"sarx_64_neg",          {0xC4, 0x62, 0xF2, 0xF7, 0xDE}, kAllFlags, false},
        {"rorx_64_imm13",        {0xC4, 0x63, 0xFB, 0xF0, 0xC3, 13}, kAllFlags, false},
        {"rorx_32_imm0",         {0xC4, 0xE3, 0x7B, 0xF0, 0xC0, 0}, kAllFlags, false},
        {"mulx_64_swap_rax_rdx", {0xC4, 0xE2, 0xFB, 0xF6, 0xD1}, kAllFlags, false},
        {"mulx_64_same_dst",     {0xC4, 0xE2, 0xFB, 0xF6, 0xC1}, kAllFlags, false},
        {"mulx_32_mem",          {0xC4, 0xC2, 0x63, 0xF6, 0x48, 0x10}, kAllFlags, false},
        {"movbe_load_64",        {0x49, 0x0F, 0x38, 0xF0, 0x40, 0x08}, kAllFlags, false},
        {"movbe_load_16",        {0x66, 0x41, 0x0F, 0x38, 0xF0, 0x10}, kAllFlags, false},
        // BMI1 & BMI2 flag-setting
        {"andn_64",              {0xC4, 0xE2, 0xE0, 0xF2, 0xC1}, kLogicFlags, false},
        {"blsi_64",              {0xC4, 0xE2, 0xE8, 0xF3, 0xD9}, kLogicFlags, false},
        {"blsr_64",              {0xC4, 0xE2, 0xE8, 0xF3, 0xC9}, kLogicFlags, false},
        {"blsmsk_64",            {0xC4, 0xE2, 0xE8, 0xF3, 0xD1}, kLogicFlags, false},
        {"bzhi_64_rcx_dst",      {0xC4, 0xE2, 0xF0, 0xF5, 0xCA}, kLogicFlags, false},
        {"bzhi_32",              {0xC4, 0xE2, 0x70, 0xF5, 0xC2}, kLogicFlags, false},
        {"bextr_64",             {0xC4, 0xE2, 0xF0, 0xF7, 0xC2}, kBextrFlags, false},
        {"bextr_32",             {0xC4, 0xE2, 0x70, 0xF7, 0xD3}, kBextrFlags, false},
        // AVX2 same-length in-place VEX rewrites
        {"vpxor_ymm_4b",         {0xC5, 0xAD, 0xEF, 0xF6}, kAllFlags, false},
        {"vpor_ymm_mem_unalign", {0xC4, 0xC1, 0x5D, 0xEB, 0x78, 0x00}, kAllFlags, false},
        {"vpandn_ymm",           {0xC5, 0xED, 0xDF, 0xC3}, kAllFlags, false},
        {"vextracti128_hi",      {0xC4, 0xE3, 0x7D, 0x39, 0xEA, 0x01}, kAllFlags, false},
        {"vinserti128_hi",       {0xC4, 0xE3, 0x55, 0x38, 0xEC, 0x01}, kAllFlags, false},
        {"vpblendd_256",         {0xC4, 0xE3, 0x6D, 0x02, 0xC3, 0xAA}, kAllFlags, false},
        {"vpbroadcastd_m32",     {0xC4, 0xC2, 0x7D, 0x58, 0x48, 0x04}, kAllFlags, false},
        // AVX2 out-of-line native AVX1 stubs
        {"vpackssdw_ymm",        {0xC5, 0xDD, 0x6B, 0xE7}, kAllFlags, false},
        {"vpcmpeqb_ymm_mem",     {0xC4, 0xC1, 0x5D, 0x74, 0x20}, kAllFlags, false},
        {"vpalignr_ymm_mem",     {0xC4, 0xC3, 0x5D, 0x0F, 0x20, 0x05}, kAllFlags, false},
        {"vpacksswb_ymm",        {0xC5, 0xD5, 0x63, 0xE4}, kAllFlags, false},
        {"vpcmpgtq_ymm",         {0xC4, 0xC2, 0x5D, 0x37, 0xE1}, kAllFlags, false},
        {"vpermq_216",           {0xC4, 0xE3, 0xFD, 0x00, 0xE4, 216}, kAllFlags, false},
        {"vpermq_mem_0x1B",      {0xC4, 0x43, 0xFD, 0x00, 0x60, 0x20, 0x1B}, kAllFlags, false},
        {"vbroadcastss_reg",     {0xC4, 0xE2, 0x7D, 0x18, 0xCA}, kAllFlags, false},
        {"vpbroadcastq_reg",     {0xC4, 0xE2, 0x7D, 0x59, 0xD3}, kAllFlags, false},
        {"vpbroadcastb_mem",     {0xC4, 0xC2, 0x7D, 0x78, 0x58, 0x05}, kAllFlags, false},
        {"vpbroadcastw_reg",     {0xC4, 0xE2, 0x7D, 0x79, 0xE4}, kAllFlags, false},
        {"vpmovsxbd_ymm_mem",    {0xC4, 0xC2, 0x7D, 0x21, 0x68, 0x08}, kAllFlags, false},
        {"vpmovzxwd_ymm_reg",    {0xC4, 0xE2, 0x7D, 0x33, 0xF7}, kAllFlags, false},
        {"vpmovmskb_ymm",        {0xC5, 0xFD, 0xD7, 0xC4}, kAllFlags, false},
        {"vpshufd_ymm",          {0xC5, 0xFD, 0x70, 0xC1, 0x93}, kAllFlags, false},
        {"vpslld_imm",           {0xC5, 0xDD, 0x72, 0xF4, 0x07}, kAllFlags, false},
        {"vpsrld_xmm_cnt",       {0xC5, 0xED, 0xD2, 0xC3}, kAllFlags, false},
        {"vpshufb_ymm",          {0xC4, 0xE2, 0x6D, 0x00, 0xC3}, kAllFlags, false},
        {"vpalignr_ymm",         {0xC4, 0xE3, 0x6D, 0x0F, 0xC3, 0x05}, kAllFlags, false},
        // FMA3 packed & scalar
        {"vfmadd213ps_256",      {0xC4, 0xE2, 0x6D, 0xA8, 0xC3}, kAllFlags, true},
        {"vfmsub132ps_128",      {0xC4, 0xE2, 0x69, 0x9A, 0xC3}, kAllFlags, true},
        {"vfnmadd231ps_mem",     {0xC4, 0xC2, 0x6D, 0xBC, 0x40, 0x00}, kAllFlags, true},
        {"vfnmsub213ss_scalar",  {0xC4, 0xE2, 0x69, 0xAF, 0xC3}, kAllFlags, true},
        {"vfmaddsub213ps_256",   {0xC4, 0xE2, 0x6D, 0xA6, 0xC3}, kAllFlags, true},
    };

    for (const auto& tc : cases) {
        if (!run_and_compare(tc.name, tc.bytes, tc.fmask, tc.fma)) {
            VirtualFree(arena, 0, MEM_RELEASE);
            return 1;
        }
    }

    // Test fused 4B + 6B VEX pair and fused 4B + 8B RIP-relative memory-operand pair.
    {
        const uint8_t pair_a[] = {0xC5, 0xDD, 0x6B, 0xE7, 0xC4, 0xE3, 0xFD, 0x00, 0xE4, 0xD8};
        uint8_t pair_b_ref[12] = {0xC5, 0xCD, 0x6B, 0xF7, 0xC5, 0xDD, 0xEB, 0x3D, 0, 0, 0, 0};
        uint8_t pair_b_nat[12] = {0xC5, 0xCD, 0x6B, 0xF7, 0xC5, 0xDD, 0xEB, 0x3D, 0, 0, 0, 0};
        const int32_t d_ref = int32_t(int64_t(data_page + 64) - int64_t(ref_page + 12));
        const int32_t d_nat = int32_t(int64_t(data_page + 64) - int64_t(site_page + 12));
        std::memcpy(pair_b_ref + 8, &d_ref, 4);
        std::memcpy(pair_b_nat + 8, &d_nat, 4);

        for (int which = 0; which < 2; ++which) {
            const uint8_t* p_ref = which == 0 ? pair_a : pair_b_ref;
            const uint8_t* p_nat = which == 0 ? pair_a : pair_b_nat;
            const std::size_t plen = which == 0 ? sizeof(pair_a) : sizeof(pair_b_nat);

            CpuState init{};
            for (int r = 0; r < 16; ++r) init.gpr[r] = splitmix64(rng);
            init.rflags = 0x202ull | 0x8D5ull;
            for (int y = 0; y < 16; ++y)
                for (int k = 0; k < 8; ++k) init.ymm[y].u32[k] = uint32_t(splitmix64(rng));

            std::memcpy(ref_page, p_ref, plen);
            uint64_t ref_res = 0;
            build_harness(harness_page, reinterpret_cast<uint64_t>(ref_page), ref_res);
            ref_page[plen] = 0xE9;
            const int32_t jr = int32_t(int64_t(ref_res) - int64_t(ref_page + plen + 5));
            std::memcpy(ref_page + plen + 1, &jr, 4);

            std::memcpy(site_page, p_nat, plen);
            uint64_t nat_res = 0;
            build_harness(harness_page + 0x1000, reinterpret_cast<uint64_t>(site_page), nat_res);
            site_page[plen] = 0xE9;
            const int32_t jn = int32_t(int64_t(nat_res) - int64_t(site_page + plen + 5));
            std::memcpy(site_page + plen + 1, &jn, 4);

            const auto ad1 = rt::avx2_emulation::decode(site_page, 4);
            std::vector<uint8_t> fbody;
            const uint64_t sva = reinterpret_cast<uint64_t>(site_page);
            const uint64_t stva = reinterpret_cast<uint64_t>(stub_page);
            if (!rt::isa_jit_patch::detail::emit_avx2_fma_body(fbody, ad1, sva, stva, pre_avx) ||
                !rt::isa_jit_patch::detail::emit_second_insn(fbody, site_page + 4, plen - 4, sva + 4, stva, pre_bmi,
                                                             pre_avx)) {
                std::fprintf(stderr, "FAIL: fused pair %d emit failed\n", which);
                VirtualFree(arena, 0, MEM_RELEASE);
                return 1;
            }
            rt::isa_jit_patch::detail::emit_jmp_rel32(fbody, stva, sva + plen);
            std::memcpy(stub_page, fbody.data(), fbody.size());
            site_page[0] = 0xE9;
            const int32_t js = int32_t(int64_t(stva) - int64_t(sva + 5));
            std::memcpy(site_page + 1, &js, 4);
            for (std::size_t k = 5; k < plen; ++k) site_page[k] = 0x90;
            FlushInstructionCache(GetCurrentProcess(), arena, 0x14000);

            CpuState rs = init, ns = init;
            reinterpret_cast<void (*)(CpuState*)>(harness_page)(&rs);
            const uint64_t udb = rt::avx2_emulation::count(rt::avx2_emulation::Category::avx2).load();
            reinterpret_cast<void (*)(CpuState*)>(harness_page + 0x1000)(&ns);
            const uint64_t uda = rt::avx2_emulation::count(rt::avx2_emulation::Category::avx2).load();
            if (uda != udb || std::memcmp(&rs, &ns, sizeof(CpuState)) != 0) {
                std::fprintf(stderr, "FAIL: fused pair %d mismatch (ud_delta=%llu)\n",
                             which, static_cast<unsigned long long>(uda - udb));
                VirtualFree(arena, 0, MEM_RELEASE);
                return 1;
            }
        }
    }

    VirtualFree(arena, 0, MEM_RELEASE);

    if (!test_synthetic_pe_patch()) return 1;

    std::printf("isa_jit_patch_test: OK (%zu single-insn families + 2 fused pairs + synthetic PE32+ image)\n",
                cases.size());
    return 0;
}
