; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; Packed 16-bit unsigned integer ops (VOP3P), the six u16 forms, run once on a GPU.
; Lane 0 of a wave32 group computes one result per op and stores eight dwords. Sources are
;   v1 = 0x00030001 (lo 1, hi 3), v2 = 0xFFFE0004 (lo 4, hi 0xFFFE), v3 = 0x00060009 (lo 9, hi 6).
; Result half 0 takes each source's half from op_sel[i], half 1 from op_sel_hi[i]; each half wraps to 16 bits.
;   out[0] = v_pk_add_u16 op_sel:[1,0] op_sel_hi:[0,1] = 0xFFFF0007
;            half 0: v1.hi 3 + v2.lo 4 = 7; half 1: v1.lo 1 + v2.hi 0xFFFE = 0xFFFF
;   out[1] = v_pk_sub_u16 v1, v2                       = 0x0005FFFD
;            half 0: 1 - 4 underflows to 0xFFFD; half 1: 3 - 0xFFFE = 0x10005 -> 5
;   out[2] = v_pk_mul_lo_u16 v1, v2                    = 0xFFFA0004
;            half 0: 1 * 4 = 4; half 1: 3 * 0xFFFE = 0x2FFFA -> low 16 bits 0xFFFA
;   out[3] = v_pk_mad_u16 v1, v2, v3                   = 0x0000000D
;            half 0: 1 * 4 + 9 = 13; half 1: 3 * 0xFFFE + 6 = 0x30000 -> 0
;   out[4] = v_pk_max_u16 v1, v2                       = 0xFFFE0004
;            half 0: max(1, 4) = 4; half 1: max(3, 0xFFFE) = 0xFFFE
;   out[5] = v_pk_min_u16 v1, v2                       = 0x00030001
;            half 0: min(1, 4) = 1; half 1: min(3, 0xFFFE) = 3
;   out[6] = v_pk_add_u16 v1, v2 clamp                 = 0xFFFF0005
;            half 0: 1 + 4 = 5; half 1: 3 + 0xFFFE = 0x10001 saturates to 0xFFFF
;   out[7] = v_pk_sub_u16 v1, v2 clamp                 = 0x00000000
;            both halves are below zero and saturate to 0 (unclamped: 0xFFFD and 5)
; group: 32,1,1
; Run: vk_compute_test cs_pk_int16_u.spv 1 dump -> out[0..7] = FFFF0007 0005FFFD FFFA0004 0000000D FFFE0004 00030001 FFFF0005 00000000 (hex)
; mattr: +wavefrontsize32,-wavefrontsize64
v_mov_b32_e32 v1, 0x00030001
v_mov_b32_e32 v2, 0xfffe0004
v_mov_b32_e32 v3, 0x00060009
v_pk_add_u16 v20, v1, v2 op_sel:[1,0] op_sel_hi:[0,1]
v_pk_sub_u16 v21, v1, v2
v_pk_mul_lo_u16 v22, v1, v2
v_pk_mad_u16 v23, v1, v2, v3
v_pk_max_u16 v24, v1, v2
v_pk_min_u16 v25, v1, v2
v_pk_add_u16 v26, v1, v2 clamp
v_pk_sub_u16 v27, v1, v2 clamp
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v10, 0
buffer_store_dword v20, v10, s[0:3], 0 offen offset:0
buffer_store_dword v21, v10, s[0:3], 0 offen offset:4
buffer_store_dword v22, v10, s[0:3], 0 offen offset:8
buffer_store_dword v23, v10, s[0:3], 0 offen offset:12
buffer_store_dword v24, v10, s[0:3], 0 offen offset:16
buffer_store_dword v25, v10, s[0:3], 0 offen offset:20
buffer_store_dword v26, v10, s[0:3], 0 offen offset:24
buffer_store_dword v27, v10, s[0:3], 0 offen offset:28
s_endpgm
