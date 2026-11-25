; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; The 32-bit integer clamp bit on the VOP3 add/sub forms (#265): v_add/sub/subrev_nc_u32, v_add/sub_nc_i32 and one
; of the carry-out ops. Lane 0 of a wave32 group computes one result per form and stores eight dwords:
;   out[0] = v_add_nc_u32    0xFFFFFFFE + 5 clamp   = 0xFFFFFFFF  (overflow saturates)
;   out[1] = v_sub_nc_u32    5 - 10 clamp           = 0x00000000  (borrow saturates to 0)
;   out[2] = v_subrev_nc_u32 10, 5 clamp            = 0x00000000  (S1 - S0 = 5 - 10; the operands are swapped)
;   out[3] = v_add_nc_i32    0x7FFFFFFF + 1 clamp   = 0x7FFFFFFF  (signed add saturates at INT32_MAX)
;   out[4] = v_sub_nc_i32    0x80000000 - 1 clamp   = 0x80000000  (signed sub saturates at INT32_MIN)
;   out[5] = v_add_nc_u32    0xFFFFFFFE + 5         = 0x00000003  (no clamp: wraps)
;   out[6] = v_add_nc_i32    5 + 7 clamp            = 0x0000000C  (no overflow: clamp keeps the sum)
;   out[7] = v_subrev_co_ci_u32 clamp, 0 - 0 - 1     = 0x00000000  (borrow-in saturates D to 0; sdst keeps the
;                                                      real borrow, checked by tests/semantic/test_alu_add_co_clamp.py)
; The 16-bit forms and the other five carry ops are in cs_w32_clamp16.asm.
; group: 32,1,1
; Run: VKC_SUBGROUP_SIZE=32 vk_compute_test cs_w32_clamp32.spv 1 dump -> out[0..7] = 4294967295 0 0 2147483647 2147483648 3 12 0
; mattr: +wavefrontsize32,-wavefrontsize64
v_mov_b32_e32 v1, 0xfffffffe
v_mov_b32_e32 v2, 5
v_add_nc_u32 v3, v1, v2 clamp
v_mov_b32_e32 v1, 5
v_mov_b32_e32 v2, 10
v_sub_nc_u32 v4, v1, v2 clamp
v_mov_b32_e32 v1, 10
v_mov_b32_e32 v2, 5
v_subrev_nc_u32 v5, v1, v2 clamp
v_mov_b32_e32 v1, 0x7fffffff
v_mov_b32_e32 v2, 1
v_add_nc_i32 v6, v1, v2 clamp
v_mov_b32_e32 v1, 0x80000000
v_sub_nc_i32 v7, v1, v2 clamp
v_mov_b32_e32 v1, 0xfffffffe
v_mov_b32_e32 v2, 5
v_add_nc_u32 v8, v1, v2
v_mov_b32_e32 v1, 5
v_mov_b32_e32 v2, 7
v_add_nc_i32 v9, v1, v2 clamp
s_mov_b32 vcc_lo, -1
v_mov_b32_e32 v1, 0
v_mov_b32_e32 v2, 0
v_subrev_co_ci_u32_e64 v10, vcc_lo, v1, v2, vcc_lo clamp
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
