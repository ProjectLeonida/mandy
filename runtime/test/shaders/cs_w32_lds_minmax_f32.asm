; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; ds_min_f32 / ds_max_f32 / ds_min_rtn_f32 / ds_max_rtn_f32 (RDNA2 ISA, DS), 512 B LDS, only lane 0.
; GLSL.std.450 NMin / NMax: one NaN operand gives the other operand, -0.0 ranks below +0.0, and an
; infinity is an ordinary operand (the other one wins when it is finite and on the right side).
;
;   M[0]  = +0.0 (0x00000000)  min_f32      DATA -0.0 (0x80000000) -> -0.0
;   M[4]  = -0.0 (0x80000000)  max_f32      DATA +0.0 (0x00000000) -> +0.0
;   M[8]  = +inf (0x7F800000)  min_f32      DATA 1.0  (0x3F800000) -> 1.0
;   M[12] = -inf (0xFF800000)  max_f32      DATA 1.0  (0x3F800000) -> 1.0
;   M[16] = qNaN (0x7FC00000)  min_rtn_f32  DATA 1.0  -> old 0x7FC00000, new 1.0
;   M[20] = 1.0  (0x3F800000)  max_rtn_f32  DATA qNaN -> old 0x3F800000, new 1.0 (NaN loses)
;
; out[0] M[0] after 0x80000000   out[1] M[4] after  0x00000000
; out[2] M[8] after 0x3F800000   out[3] M[12] after 0x3F800000
; out[4] min_rtn old 0x7FC00000  out[5] M[16] after 0x3F800000
; out[6] max_rtn old 0x3F800000  out[7] M[20] after 0x3F800000
;
; group: 32,1,1
; Run: vk_compute_test cs_w32_lds_minmax_f32.spv 1 dump
;   -> out[0..7] = 0x80000000 0x00000000 0x3F800000 0x3F800000 0x7FC00000 0x3F800000 0x3F800000 0x3F800000
; mattr: +wavefrontsize32,-wavefrontsize64
v_mov_b32_e32 v1, 0
v_mov_b32_e32 v2, 0
ds_write_b32 v1, v2
v_mov_b32_e32 v1, 4
v_mov_b32_e32 v2, 0x80000000
ds_write_b32 v1, v2
v_mov_b32_e32 v1, 8
v_mov_b32_e32 v2, 0x7f800000
ds_write_b32 v1, v2
v_mov_b32_e32 v1, 12
v_mov_b32_e32 v2, 0xff800000
ds_write_b32 v1, v2
v_mov_b32_e32 v1, 16
v_mov_b32_e32 v2, 0x7fc00000
ds_write_b32 v1, v2
v_mov_b32_e32 v1, 20
v_mov_b32_e32 v2, 0x3f800000
ds_write_b32 v1, v2
s_waitcnt lgkmcnt(0)
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v1, 0
v_mov_b32_e32 v3, 0x80000000
ds_min_f32 v1, v3
v_mov_b32_e32 v1, 4
v_mov_b32_e32 v3, 0
ds_max_f32 v1, v3
v_mov_b32_e32 v1, 8
v_mov_b32_e32 v3, 0x3f800000
ds_min_f32 v1, v3
v_mov_b32_e32 v1, 12
ds_max_f32 v1, v3
v_mov_b32_e32 v1, 16
ds_min_rtn_f32 v5, v1, v3
v_mov_b32_e32 v1, 20
v_mov_b32_e32 v3, 0x7fc00000
ds_max_rtn_f32 v6, v1, v3
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
buffer_store_dword v5, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 20
buffer_store_dword v14, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 24
buffer_store_dword v6, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 28
buffer_store_dword v15, v21, s[0:3], 0 offen
s_endpgm
