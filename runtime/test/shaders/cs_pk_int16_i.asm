; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; Packed 16-bit signed integer ops (VOP3P), the five i16 forms plus op_sel and clamp, run once on a GPU.
; Lane 0 of a wave32 group computes one result per op and stores eight dwords. Sources are
;   v1 = 0x00028001 (lo -32767, hi 2), v2 = 0xFFFFFFFF (lo -1, hi -1), v3 = 0x00030002 (lo 2, hi 3),
;   v4 = 0x80010002 (lo 2, hi -32767), v5 = 0x80018001 (both halves -32767).
; Result half 0 takes each source's half from op_sel[i], half 1 from op_sel_hi[i]; each half wraps to 16 bits.
;   out[0] = v_pk_add_i16 v1, v2                       = 0x00018000
;            half 0: -32767 - 1 = -32768; half 1: 2 - 1 = 1
;   out[1] = v_pk_sub_i16 v1, v2                       = 0x00038002
;            half 0: -32767 + 1 = -32766; half 1: 2 + 1 = 3
;   out[2] = v_pk_mad_i16 v1, v2, v3 op_sel:[1,0,1] op_sel_hi:[0,1,0] = 0x80010001
;            half 0: v1.hi 2 * v2.lo -1 + v3.hi 3 = 1; half 1: v1.lo -32767 * v2.hi -1 + v3.lo 2 = 32769 -> -32767
;   out[3] = v_pk_max_i16 v1, v2                        = 0x0002FFFF
;            half 0: max(-32767, -1) = -1; half 1: max(2, -1) = 2
;   out[4] = v_pk_min_i16 v1, v2                        = 0xFFFF8001
;            half 0: min(-32767, -1) = -32767; half 1: min(2, -1) = -1
;   out[5] = v_pk_add_i16 v5, v5 clamp                  = 0x80008000
;            both halves: -32767 + -32767 = -65534 saturates to -32768
;   out[6] = v_pk_sub_i16 v1, v4 clamp                  = 0x7FFF8000
;            half 0: -32767 - 2 = -32769 saturates to -32768; half 1: 2 - -32767 = 32769 saturates to 32767
;   out[7] = v_pk_mad_i16 v5, v2, v3 clamp              = 0x7FFF7FFF
;            half 0: -32767 * -1 + 2 = 32769 saturates to 32767; half 1: -32767 * -1 + 3 = 32770 saturates to 32767
; group: 32,1,1
; Run: vk_compute_test cs_pk_int16_i.spv 1 dump -> out[0..7] = 00018000 00038002 80010001 0002FFFF FFFF8001 80008000 7FFF8000 7FFF7FFF (hex)
; mattr: +wavefrontsize32,-wavefrontsize64
v_mov_b32_e32 v1, 0x00028001
v_mov_b32_e32 v2, 0xffffffff
v_mov_b32_e32 v3, 0x00030002
v_mov_b32_e32 v4, 0x80010002
v_mov_b32_e32 v5, 0x80018001
v_pk_add_i16 v20, v1, v2
v_pk_sub_i16 v21, v1, v2
v_pk_mad_i16 v22, v1, v2, v3 op_sel:[1,0,1] op_sel_hi:[0,1,0]
v_pk_max_i16 v23, v1, v2
v_pk_min_i16 v24, v1, v2
v_pk_add_i16 v25, v5, v5 clamp
v_pk_sub_i16 v26, v1, v4 clamp
v_pk_mad_i16 v27, v5, v2, v3 clamp
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
