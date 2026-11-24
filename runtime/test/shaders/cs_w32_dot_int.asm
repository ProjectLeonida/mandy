; mcpu: gfx1030
; The packed integer dot products (VOP3P), run on a GPU; until now CPU-checked by the semantic tests and measured with
; tools/hw_oracle, but no fixture ran the translation (docs/design.md, "Packed integer dot products"). D = the sum of
; the element products + S2, each element sign- (i) or zero-extended (u), wrapping at 32 bits; with clamp the exact sum
; saturates to [INT32_MIN, INT32_MAX] (i32) or [0, UINT32_MAX] (u32). gfx1013 has none of these instructions, so this
; file is assembled for gfx1030 (the "; mcpu:" line above). Elements are numbered from bit 0.
;   out[0] v_dot2_i32_i16 0x0003fffe, 0x0005fff9, 100           (-2)(-7) + 3*5 + 100            = 129          0x00000081
;   out[1] v_dot2_u32_u16 0xffff8001, 0xffff0003, 0xffffff00     32769*3 + 65535^2 + 0xffffff00  wraps once     0xffff7f04
;   out[2] v_dot4_u32_u8  0x80ff0102, 0x7f02ff03, 5              2*3 + 1*255 + 255*2 + 128*127 + 5 = 17032      0x00004288
;   out[3] v_dot8_i32_i4  0x8f01f7a3, 0x12345678, -3             signed nibbles: -35 - 3        = -38          0xffffffda
;   out[4] v_dot8_u32_u4  0x8f01f7a3, 0x12345678, 7              unsigned nibbles: 253 + 7      = 260          0x00000104
;   out[5] v_dot2_u32_u16 0xffffffff, 0xffffffff, 0xffffffff clamp   2*65535^2 + 0xffffffff > UINT32_MAX     0xffffffff
;   out[6] v_dot2_i32_i16 0x80008000, 0x80008000, -2 clamp       2^31 - 2: the products overflow, the sum does not
;                                                                                                              0x7ffffffe
;   out[7] v_dot2_i32_i16 0x80008000, 0x80008000, 0 clamp        2^31 > INT32_MAX                              0x7fffffff
; group: 32,1,1
; Run: vk_compute_test cs_w32_dot_int.spv 1 dump -> out[0..7] = 129 4294934276 17032 4294967258 260 4294967295
; 2147483646 2147483647
; mattr: +wavefrontsize32,-wavefrontsize64
v_mov_b32_e32 v1, 0x0003fffe
v_mov_b32_e32 v2, 0x0005fff9
v_mov_b32_e32 v3, 0x64
v_dot2_i32_i16 v10, v1, v2, v3
v_mov_b32_e32 v1, 0xffff8001
v_mov_b32_e32 v2, 0xffff0003
v_mov_b32_e32 v3, 0xffffff00
v_dot2_u32_u16 v11, v1, v2, v3
v_mov_b32_e32 v1, 0x80ff0102
v_mov_b32_e32 v2, 0x7f02ff03
v_mov_b32_e32 v3, 5
v_dot4_u32_u8 v12, v1, v2, v3
v_mov_b32_e32 v1, 0x8f01f7a3
v_mov_b32_e32 v2, 0x12345678
v_mov_b32_e32 v3, -3
v_dot8_i32_i4 v13, v1, v2, v3
v_mov_b32_e32 v3, 7
v_dot8_u32_u4 v14, v1, v2, v3
v_mov_b32_e32 v1, -1
v_dot2_u32_u16 v15, v1, v1, v1 clamp
v_mov_b32_e32 v1, 0x80008000
v_mov_b32_e32 v3, -2
v_dot2_i32_i16 v16, v1, v1, v3 clamp
v_mov_b32_e32 v3, 0
v_dot2_i32_i16 v17, v1, v1, v3 clamp
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v25, 0
buffer_store_dword v10, v25, s[0:3], 0 offen
v_mov_b32_e32 v25, 4
buffer_store_dword v11, v25, s[0:3], 0 offen
v_mov_b32_e32 v25, 8
buffer_store_dword v12, v25, s[0:3], 0 offen
v_mov_b32_e32 v25, 12
buffer_store_dword v13, v25, s[0:3], 0 offen
v_mov_b32_e32 v25, 16
buffer_store_dword v14, v25, s[0:3], 0 offen
v_mov_b32_e32 v25, 20
buffer_store_dword v15, v25, s[0:3], 0 offen
v_mov_b32_e32 v25, 24
buffer_store_dword v16, v25, s[0:3], 0 offen
v_mov_b32_e32 v25, 28
buffer_store_dword v17, v25, s[0:3], 0 offen
s_endpgm
