; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; The scalar destination of a VALU carry/sign-out op under a partial EXEC (#218): every active lane writes its
; carry/borrow bit, every inactive lane writes 0 (RDNA2 ISA: SDST[lane] = EXEC[lane] ? bit : 0), as v_cmp does.
; The eight ops each run with EXEC = 0xAAAAAAAA (odd lanes on) and operands that carry/borrow/sign-out on every
; lane, so a correct sdst is exactly 0xAAAAAAAA; an sdst that ignores EXEC is 0xFFFFFFFF. Lane 0 (off in that
; mask) reads the result back with the four places below:
;   v_cndmask_b32_e64 v20, 0, 1, <sdst>   -> 1 on the lanes of the mask
;   v_cmp_ne_u32_e32 vcc_lo, 0, v20       -> the mask again, now under the whole wave
;   s_bcnt1 / s_ff1 of it                 -> lane count and lowest lane
; word = (set lanes) | (lowest lane) << 8, as cs_w32_mask_imm.asm derives: 16 | 1<<8 = 272; an unmasked sdst
; (all 32 bits) gives 32 | 0<<8 = 32, and a missing write 0 | -1<<8 = 0xFFFFFF00.
;   out[0] = v_add_co_u32       v3, vcc, 0xFFFFFFFF, 1       -> 0xAAAAAAAA  (carry on every active lane)
;   out[1] = v_add_co_ci_u32    v3, vcc, 0xFFFFFFFF, 0       -> 0xAAAAAAAA  (carry in from VCC = -1)
;   out[2] = v_sub_co_u32       v3, s10, 0, 1                -> 0xAAAAAAAA  (borrow on every active lane)
;   out[3] = v_sub_co_ci_u32    v3, s10, 0, 0, s12           -> 0xAAAAAAAA  (borrow in from s12 = -1)
;   out[4] = v_subrev_co_u32    v3, s10, 1, 0                -> 0xAAAAAAAA  (0 - 1)
;   out[5] = v_subrev_co_ci_u32 v3, s10, 0, 0, s12           -> 0xAAAAAAAA  (0 - 0 - 1)
;   out[6] = v_mad_u64_u32      v[24:25], s10, 0xFFFFFFFF, 1, 0xFFFFFFFF_FFFFFFFF -> 0xAAAAAAAA  (carry out)
;   out[7] = v_mad_i64_i32      v[24:25], s10, -1, 1, 0      -> 0xAAAAAAAA  (sum -1, sign bit 64)
; Until 2026-11-22 the six `*_co_*` ops and the two `v_mad_*64` ops did not AND the carry/sign bit with
; `active`, so inactive lanes leaked 1 bits into VCC/SGPR masks and s_bcnt1/s_cbranch_vccz/v_cndmask saw them.
; group: 32,1,1
; Run: VKC_SUBGROUP_SIZE=32 vk_compute_test cs_w32_sdst_exec.spv 1 dump -> out[0..7] = 272 272 272 272 272 272 272 272
; mattr: +wavefrontsize32,-wavefrontsize64
v_mov_b32_e32 v1, 0xffffffff
v_mov_b32_e32 v2, 1
s_mov_b32 exec_lo, 0xaaaaaaaa
v_add_co_u32 v3, vcc_lo, v1, v2
s_mov_b32 exec_lo, -1
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_cmp_ne_u32_e32 vcc_lo, 0, v20
s_bcnt1_i32_b32 s20, vcc_lo
s_ff1_i32_b32 s21, vcc_lo
s_lshl_b32 s21, s21, 8
s_or_b32 s20, s20, s21
s_mov_b32 vcc_lo, -1
v_mov_b32_e32 v2, 0
s_mov_b32 exec_lo, 0xaaaaaaaa
v_add_co_ci_u32_e32 v3, vcc_lo, v1, v2, vcc_lo
s_mov_b32 exec_lo, -1
v_cndmask_b32_e64 v20, 0, 1, vcc_lo
v_cmp_ne_u32_e32 vcc_lo, 0, v20
s_bcnt1_i32_b32 s22, vcc_lo
s_ff1_i32_b32 s23, vcc_lo
s_lshl_b32 s23, s23, 8
s_or_b32 s22, s22, s23
v_mov_b32_e32 v4, 0
v_mov_b32_e32 v5, 1
s_mov_b32 exec_lo, 0xaaaaaaaa
v_sub_co_u32 v3, s10, v4, v5
s_mov_b32 exec_lo, -1
v_cndmask_b32_e64 v20, 0, 1, s10
v_cmp_ne_u32_e32 vcc_lo, 0, v20
s_bcnt1_i32_b32 s24, vcc_lo
s_ff1_i32_b32 s25, vcc_lo
s_lshl_b32 s25, s25, 8
s_or_b32 s24, s24, s25
s_mov_b32 s12, -1
v_mov_b32_e32 v6, 0
s_mov_b32 exec_lo, 0xaaaaaaaa
v_sub_co_ci_u32_e64 v3, s10, v6, v6, s12
s_mov_b32 exec_lo, -1
v_cndmask_b32_e64 v20, 0, 1, s10
v_cmp_ne_u32_e32 vcc_lo, 0, v20
s_bcnt1_i32_b32 s26, vcc_lo
s_ff1_i32_b32 s27, vcc_lo
s_lshl_b32 s27, s27, 8
s_or_b32 s26, s26, s27
v_mov_b32_e32 v7, 1
v_mov_b32_e32 v8, 0
s_mov_b32 exec_lo, 0xaaaaaaaa
v_subrev_co_u32 v3, s10, v7, v8
s_mov_b32 exec_lo, -1
v_cndmask_b32_e64 v20, 0, 1, s10
v_cmp_ne_u32_e32 vcc_lo, 0, v20
s_bcnt1_i32_b32 s28, vcc_lo
s_ff1_i32_b32 s29, vcc_lo
s_lshl_b32 s29, s29, 8
s_or_b32 s28, s28, s29
s_mov_b32 exec_lo, 0xaaaaaaaa
v_subrev_co_ci_u32_e64 v3, s10, v6, v6, s12
s_mov_b32 exec_lo, -1
v_cndmask_b32_e64 v20, 0, 1, s10
v_cmp_ne_u32_e32 vcc_lo, 0, v20
s_bcnt1_i32_b32 s30, vcc_lo
s_ff1_i32_b32 s31, vcc_lo
s_lshl_b32 s31, s31, 8
s_or_b32 s30, s30, s31
v_mov_b32_e32 v9, 0xffffffff
v_mov_b32_e32 v10, 1
v_mov_b32_e32 v12, 0xffffffff
v_mov_b32_e32 v13, 0xffffffff
s_mov_b32 exec_lo, 0xaaaaaaaa
v_mad_u64_u32 v[24:25], s10, v9, v10, v[12:13]
s_mov_b32 exec_lo, -1
v_cndmask_b32_e64 v20, 0, 1, s10
v_cmp_ne_u32_e32 vcc_lo, 0, v20
s_bcnt1_i32_b32 s32, vcc_lo
s_ff1_i32_b32 s33, vcc_lo
s_lshl_b32 s33, s33, 8
s_or_b32 s32, s32, s33
v_mov_b32_e32 v14, -1
v_mov_b32_e32 v15, 1
v_mov_b32_e32 v16, 0
v_mov_b32_e32 v17, 0
s_mov_b32 exec_lo, 0xaaaaaaaa
v_mad_i64_i32 v[24:25], s10, v14, v15, v[16:17]
s_mov_b32 exec_lo, -1
v_cndmask_b32_e64 v20, 0, 1, s10
v_cmp_ne_u32_e32 vcc_lo, 0, v20
s_bcnt1_i32_b32 s34, vcc_lo
s_ff1_i32_b32 s35, vcc_lo
s_lshl_b32 s35, s35, 8
s_or_b32 s34, s34, s35
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v1, s20
v_mov_b32_e32 v2, 0
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32_e32 v1, s22
v_mov_b32_e32 v2, 4
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32_e32 v1, s24
v_mov_b32_e32 v2, 8
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32_e32 v1, s26
v_mov_b32_e32 v2, 12
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32_e32 v1, s28
v_mov_b32_e32 v2, 16
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32_e32 v1, s30
v_mov_b32_e32 v2, 20
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32_e32 v1, s32
v_mov_b32_e32 v2, 24
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32_e32 v1, s34
v_mov_b32_e32 v2, 28
buffer_store_dword v1, v2, s[0:3], 0 offen
s_endpgm
