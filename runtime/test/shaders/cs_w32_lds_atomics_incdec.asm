; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; ds_inc_u32 / ds_dec_u32 / ds_rsub_u32, the non-returning 32-bit LDS atomics (RDNA2 ISA, DS).
; Only lane 0 runs the atomics. DATA = 5.
;   ds_inc_u32:  M = (M >= DATA) ? 0 : M + 1                (unsigned)
;   ds_dec_u32:  M = (M == 0 || M > DATA) ? DATA : M - 1    (unsigned)
;   ds_rsub_u32: M = DATA - M                               (32-bit wrapping)
;
; Seeds and results (M before -> M after):
;   M[0]  = 4 = DATA-1  inc -> 5, the last value before the wrap
;   M[4]  = 5 = DATA    inc -> 0, M >= DATA wraps to 0
;   M[8]  = 6 > DATA    inc -> 0
;   M[12] = 0           dec -> 5, M == 0 wraps to DATA
;   M[16] = 1           dec -> 0
;   M[20] = 5 = DATA    dec -> 4, the last value before the wrap
;   M[24] = 6 > DATA    dec -> 5, M > DATA wraps to DATA
;   M[28] = 7           rsub -> 5 - 7 = 0xFFFFFFFE (wraps)
;
; group: 32,1,1
; Run: vk_compute_test cs_w32_lds_atomics_incdec.spv 1 dump
;   -> out[0..7] = 5 0 0 5 0 4 5 0xFFFFFFFE
; mattr: +wavefrontsize32,-wavefrontsize64
v_mov_b32_e32 v1, 0
v_mov_b32_e32 v2, 4
ds_write_b32 v1, v2
v_mov_b32_e32 v1, 4
v_mov_b32_e32 v2, 5
ds_write_b32 v1, v2
v_mov_b32_e32 v1, 8
v_mov_b32_e32 v2, 6
ds_write_b32 v1, v2
v_mov_b32_e32 v2, 0
v_mov_b32_e32 v1, 12
ds_write_b32 v1, v2
v_mov_b32_e32 v1, 16
v_mov_b32_e32 v2, 1
ds_write_b32 v1, v2
v_mov_b32_e32 v1, 20
v_mov_b32_e32 v2, 5
ds_write_b32 v1, v2
v_mov_b32_e32 v1, 24
v_mov_b32_e32 v2, 6
ds_write_b32 v1, v2
v_mov_b32_e32 v1, 28
v_mov_b32_e32 v2, 7
ds_write_b32 v1, v2
s_waitcnt lgkmcnt(0)
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v3, 5
v_mov_b32_e32 v1, 0
ds_inc_u32 v1, v3
v_mov_b32_e32 v1, 4
ds_inc_u32 v1, v3
v_mov_b32_e32 v1, 8
ds_inc_u32 v1, v3
v_mov_b32_e32 v1, 12
ds_dec_u32 v1, v3
v_mov_b32_e32 v1, 16
ds_dec_u32 v1, v3
v_mov_b32_e32 v1, 20
ds_dec_u32 v1, v3
v_mov_b32_e32 v1, 24
ds_dec_u32 v1, v3
v_mov_b32_e32 v1, 28
ds_rsub_u32 v1, v3
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v1, 0
ds_read_b32 v10, v1
v_mov_b32_e32 v1, 4
ds_read_b32 v11, v1
v_mov_b32_e32 v1, 8
ds_read_b32 v12, v1
v_mov_b32_e32 v1, 12
ds_read_b32 v13, v1
v_mov_b32_e32 v1, 16
ds_read_b32 v14, v1
v_mov_b32_e32 v1, 20
ds_read_b32 v15, v1
v_mov_b32_e32 v1, 24
ds_read_b32 v16, v1
v_mov_b32_e32 v1, 28
ds_read_b32 v17, v1
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v21, 0
buffer_store_dword v10, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 4
buffer_store_dword v11, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 8
buffer_store_dword v12, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 12
buffer_store_dword v13, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 16
buffer_store_dword v14, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 20
buffer_store_dword v15, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 24
buffer_store_dword v16, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 28
buffer_store_dword v17, v21, s[0:3], 0 offen
s_endpgm
