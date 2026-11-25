; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; The 16-bit integer clamp bit and the five other carry-out ops with clamp (#265).
; Lane 0 of a wave32 group computes one result per form and stores eight dwords. The v_add_nc_u16/v_sub_nc_u16
; destinations are preloaded with 0xDEAD0000: the VOP3 16-bit forms are partial writes that keep the high half.
;   out[0] = v_add_nc_u16 0xFFFF + 1 clamp   = 0xDEADFFFF  (saturates to 0xFFFF, high half kept)
;   out[1] = v_sub_nc_u16 1 - 2 clamp        = 0xDEAD0000  (borrow saturates to 0)
;   out[2] = v_add_nc_u16 0xFFFF + 1         = 0xDEAD0000  (no clamp: wraps to 0)
;   out[3] = v_add_co_u32 0xFFFFFFFF + 1 clamp    = 0xFFFFFFFF  (carry saturates D; sdst still gets 1)
;   out[4] = v_add_co_ci_u32 0xFFFFFFFF + 0 + carry-in 1 clamp = 0xFFFFFFFF
;   out[5] = v_sub_co_u32 0 - 1 clamp             = 0x00000000  (borrow saturates D; sdst still gets 1)
;   out[6] = v_sub_co_ci_u32 0 - 0 - borrow-in 1 clamp = 0x00000000
;   out[7] = v_subrev_co_u32 1, 0 clamp           = 0x00000000  (S1 - S0 = 0 - 1 borrows)
; The sdst of the carry ops keeps the real carry/borrow (the CPU test checks that); v_subrev_co_ci_u32 clamp is
; out[7] of cs_w32_clamp32.asm. The unsigned 16-bit names on gfx10 are the _nc_ ones: v_add_u16/v_subrev_u16 do
; not assemble for gfx1013.
; group: 32,1,1
; Run: VKC_SUBGROUP_SIZE=32 vk_compute_test cs_w32_clamp16.spv 1 dump -> out[0..7] = 3735945215 3735879680 3735879680 4294967295 4294967295 0 0 0
; mattr: +wavefrontsize32,-wavefrontsize64
v_mov_b32_e32 v3, 0xdead0000
v_mov_b32_e32 v4, 0xdead0000
v_mov_b32_e32 v5, 0xdead0000
v_mov_b32_e32 v1, 0xffff
v_mov_b32_e32 v2, 1
v_add_nc_u16 v3, v1, v2 clamp
v_mov_b32_e32 v1, 1
v_mov_b32_e32 v2, 2
v_sub_nc_u16 v4, v1, v2 clamp
v_mov_b32_e32 v1, 0xffff
v_mov_b32_e32 v2, 1
v_add_nc_u16 v5, v1, v2
v_mov_b32_e32 v1, 0xffffffff
v_mov_b32_e32 v2, 1
v_add_co_u32 v6, vcc_lo, v1, v2 clamp
s_mov_b32 vcc_lo, -1
v_mov_b32_e32 v1, 0xffffffff
v_mov_b32_e32 v2, 0
v_add_co_ci_u32_e64 v7, vcc_lo, v1, v2, vcc_lo clamp
v_mov_b32_e32 v1, 0
v_mov_b32_e32 v2, 1
v_sub_co_u32 v8, vcc_lo, v1, v2 clamp
s_mov_b32 vcc_lo, -1
v_mov_b32_e32 v1, 0
v_mov_b32_e32 v2, 0
v_sub_co_ci_u32_e64 v9, vcc_lo, v1, v2, vcc_lo clamp
v_mov_b32_e32 v1, 1
v_mov_b32_e32 v2, 0
v_subrev_co_u32 v10, vcc_lo, v1, v2 clamp
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v20, 0
buffer_store_dword v3, v20, s[0:3], 0 offen
buffer_store_dword v4, v20, s[0:3], 0 offen offset:4
buffer_store_dword v5, v20, s[0:3], 0 offen offset:8
buffer_store_dword v6, v20, s[0:3], 0 offen offset:12
buffer_store_dword v7, v20, s[0:3], 0 offen offset:16
buffer_store_dword v8, v20, s[0:3], 0 offen offset:20
buffer_store_dword v9, v20, s[0:3], 0 offen offset:24
buffer_store_dword v10, v20, s[0:3], 0 offen offset:28
s_endpgm
