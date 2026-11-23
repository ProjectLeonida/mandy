; Three-operand 16-bit min/max/median (VOP3): v_min3_f16, v_max3_f16, v_min3_i16, v_max3_i16,
; v_min3_u16, v_max3_u16, v_med3_i16, v_med3_u16 (docs/design.md contract row).
; Every op exercises op_sel source/destination half selection and preserves the unwritten destination half;
; f16 ops also check NaN ignoring, -0.0 < +0.0 ordering, source neg/abs and VOP3 clamp; i16/u16 ops check
; signed vs unsigned ordering (out[6] and out[7] take the exact same three 16-bit bit patterns).
; Lane 0 stores eight dwords:
;   out[0] = v_min3_f16(qNaN 0x7E01, -0.0 0x8000, +0.0 0x0000) op_sel:[1,0,1,0] over 0xCAFE1111 = 0xCAFE8000
;   out[1] = v_max3_f16(-2.0, |-3.0|, qNaN 0x7E00) op_sel:[0,1,0,1] clamp over 0x2222BEEF       = 0x3C00BEEF
;   out[2] = v_min3_i16(-32768, -1, 32767) op_sel:[1,0,1,0] over 0x12340000                      = 0x12348000
;   out[3] = v_max3_i16(-1, 5, -32768) op_sel:[0,1,0,1] over 0x00005678                          = 0x00055678
;   out[4] = v_min3_u16(32768, 65535, 32767) op_sel:[1,1,0,0] over 0x9ABC0000                    = 0x9ABC7FFF
;   out[5] = v_max3_u16(5, 32768, 65535) op_sel:[0,0,1,1] over 0x0000DEF0                        = 0xFFFFDEF0
;   out[6] = v_med3_i16(-10, 5, -32768) op_sel:[0,1,1,0] over 0x43210000                         = 0x4321FFF6
;   out[7] = v_med3_u16(65526, 5, 32768) op_sel:[0,1,1,1] over 0x00008765                        = 0x80008765
; group: 32,1,1
; Run: vk_compute_test cs_w32_min3_med3_16.spv 1 dump -> out[0..7] = 3405676544 1006681839 305430528 349816 2596044799 4294958832 1126301686 2147518309
; mattr: +wavefrontsize32,-wavefrontsize64
; regen_fixtures: wave32
v_mov_b32_e32 v10, 0xcafe1111
v_mov_b32_e32 v1, 0x7e013c00
v_mov_b32_e32 v2, 0x56788000
v_mov_b32_e32 v3, 0x00001234
v_min3_f16 v10, v1, v2, v3 op_sel:[1,0,1,0]

v_mov_b32_e32 v11, 0x2222beef
v_mov_b32_e32 v1, 0xaaaa4000
v_mov_b32_e32 v2, 0xc200bbbb
v_mov_b32_e32 v3, 0xcccc7e00
v_max3_f16 v11, -v1, |v2|, v3 op_sel:[0,1,0,1] clamp

v_mov_b32_e32 v12, 0x12340000
v_mov_b32_e32 v1, 0x80007fff
v_mov_b32_e32 v2, 0x0001ffff
v_mov_b32_e32 v3, 0x7fff8001
v_min3_i16 v12, v1, v2, v3 op_sel:[1,0,1,0]

v_mov_b32_e32 v13, 0x00005678
v_mov_b32_e32 v1, 0x0001ffff
v_mov_b32_e32 v2, 0x0005ffff
v_mov_b32_e32 v3, 0x00028000
v_max3_i16 v13, v1, v2, v3 op_sel:[0,1,0,1]

v_mov_b32_e32 v14, 0x9abc0000
v_mov_b32_e32 v1, 0x80000001
v_mov_b32_e32 v2, 0xffff0002
v_mov_b32_e32 v3, 0x00037fff
v_min3_u16 v14, v1, v2, v3 op_sel:[1,1,0,0]

v_mov_b32_e32 v15, 0x0000def0
v_mov_b32_e32 v1, 0x00000005
v_mov_b32_e32 v2, 0x00008000
v_mov_b32_e32 v3, 0xffff0001
v_max3_u16 v15, v1, v2, v3 op_sel:[0,0,1,1]

v_mov_b32_e32 v16, 0x43210000
v_mov_b32_e32 v17, 0x00008765
v_mov_b32_e32 v1, 0x0000fff6
v_mov_b32_e32 v2, 0x00050000
v_mov_b32_e32 v3, 0x80000000
v_med3_i16 v16, v1, v2, v3 op_sel:[0,1,1,0]
v_med3_u16 v17, v1, v2, v3 op_sel:[0,1,1,1]

v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v20, 0
buffer_store_dword v10, v20, s[0:3], 0 offen
v_mov_b32_e32 v20, 4
buffer_store_dword v11, v20, s[0:3], 0 offen
v_mov_b32_e32 v20, 8
buffer_store_dword v12, v20, s[0:3], 0 offen
v_mov_b32_e32 v20, 12
buffer_store_dword v13, v20, s[0:3], 0 offen
v_mov_b32_e32 v20, 16
buffer_store_dword v14, v20, s[0:3], 0 offen
v_mov_b32_e32 v20, 20
buffer_store_dword v15, v20, s[0:3], 0 offen
v_mov_b32_e32 v20, 24
buffer_store_dword v16, v20, s[0:3], 0 offen
v_mov_b32_e32 v20, 28
buffer_store_dword v17, v20, s[0:3], 0 offen
s_endpgm
