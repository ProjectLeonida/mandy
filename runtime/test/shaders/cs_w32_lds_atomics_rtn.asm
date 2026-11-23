; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; ds_inc_rtn_u32 / ds_dec_rtn_u32 / ds_rsub_rtn_u32 (RDNA2 ISA, DS): the _rtn_ forms return the
; pre-op word in vdst. Only lane 0 runs the atomics, so each old value is exactly the seed.
;   ds_inc_rtn_u32:  vdst = M, M = (M >= DATA) ? 0 : M + 1
;   ds_dec_rtn_u32:  vdst = M, M = (M == 0 || M > DATA) ? DATA : M - 1
;   ds_rsub_rtn_u32: vdst = M, M = DATA - M (wrapping)
; DATA = 5.
;
;   M[0]  = 4 = DATA-1  inc_rtn -> old 4, new 5
;   M[4]  = 0           dec_rtn -> old 0, new 5 (M == 0 wraps to DATA)
;   M[8]  = 7           rsub_rtn -> old 7, new 5 - 7 = 0xFFFFFFFE (wraps)
;   M[12] = 5 = DATA    inc_rtn -> old 5, new 0 (M >= DATA)
;
; out[0..7] = 4 5 0 5 7 0xFFFFFFFE 5 0
;
; group: 32,1,1
; Run: vk_compute_test cs_w32_lds_atomics_rtn.spv 1 dump
;   -> out[0..7] = 0x00000004 0x00000005 0x00000000 0x00000005 0x00000007 0xFFFFFFFE 0x00000005 0x00000000
; mattr: +wavefrontsize32,-wavefrontsize64
v_mov_b32_e32 v1, 0
v_mov_b32_e32 v2, 4
ds_write_b32 v1, v2
v_mov_b32_e32 v1, 4
v_mov_b32_e32 v2, 0
ds_write_b32 v1, v2
v_mov_b32_e32 v1, 8
v_mov_b32_e32 v2, 7
ds_write_b32 v1, v2
v_mov_b32_e32 v1, 12
v_mov_b32_e32 v2, 5
ds_write_b32 v1, v2
s_waitcnt lgkmcnt(0)
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v3, 5
v_mov_b32_e32 v1, 0
ds_inc_rtn_u32 v5, v1, v3
v_mov_b32_e32 v1, 4
ds_dec_rtn_u32 v6, v1, v3
v_mov_b32_e32 v1, 8
ds_rsub_rtn_u32 v7, v1, v3
v_mov_b32_e32 v1, 12
ds_inc_rtn_u32 v8, v1, v3
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v1, 0
ds_read_b32 v10, v1
v_mov_b32_e32 v1, 4
ds_read_b32 v11, v1
v_mov_b32_e32 v1, 8
ds_read_b32 v12, v1
v_mov_b32_e32 v1, 12
ds_read_b32 v13, v1
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v21, 0
buffer_store_dword v5, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 4
buffer_store_dword v10, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 8
buffer_store_dword v6, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 12
buffer_store_dword v11, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 16
buffer_store_dword v7, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 20
buffer_store_dword v12, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 24
buffer_store_dword v8, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 28
buffer_store_dword v13, v21, s[0:3], 0 offen
s_endpgm
