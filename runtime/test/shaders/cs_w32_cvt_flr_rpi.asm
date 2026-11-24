; Float-to-signed-int floor and exact round-to-positive-infinity (VOP1/VOP3):
; v_cvt_flr_i32_f32 and v_cvt_rpi_i32_f32 (docs/design.md contract row).
; Checks floor vs truncate on negative half, exact floor(x + 0.5) without binary32 rounding
; of S0 + 0.5 (0x3EFFFFFF = 0.49999997 -> 0, 0x4B000001 = 8388609.0 -> 8388609), VOP3 source
; abs/neg modifiers, and sign-bit saturation for +NaN (0x7FFFFFFF) and -NaN (0x80000000).
; Lane 0 stores eight dwords:
;   out[0] = v_cvt_flr_i32_f32(-2.5)                   = 0xFFFFFFFD  (-3)
;   out[1] = v_cvt_rpi_i32_f32(-2.5)                   = 0xFFFFFFFE  (-2)
;   out[2] = v_cvt_rpi_i32_f32(0x3EFFFFFF = 0.49999997)= 0x00000000  (0; exact floor(x + 0.5))
;   out[3] = v_cvt_rpi_i32_f32(0x4B000001 = 8388609.0) = 0x00800001  (8388609; no f32 tie round)
;   out[4] = v_cvt_rpi_i32_f32_e64(|-2.5|)             = 0x00000003  (3; abs + half-up)
;   out[5] = v_cvt_flr_i32_f32_e64(-(3.25))            = 0xFFFFFFFC  (-4; neg + floor)
;   out[6] = v_cvt_flr_i32_f32(+qNaN 0x7FC00000)       = 0x7FFFFFFF  (INT_MAX by sign bit)
;   out[7] = v_cvt_rpi_i32_f32(-qNaN 0xFFC00000)       = 0x80000000  (INT_MIN by sign bit)
; group: 32,1,1
; Run: vk_compute_test cs_w32_cvt_flr_rpi.spv 1 dump -> out[0..7] = 4294967293 4294967294 0 8388609 3 4294967292 2147483647 2147483648
; mattr: +wavefrontsize32,-wavefrontsize64
v_mov_b32_e32 v1, 0xc0200000
v_cvt_flr_i32_f32_e32 v10, v1
v_cvt_rpi_i32_f32_e32 v11, v1
v_cvt_rpi_i32_f32_e64 v14, |v1|

v_mov_b32_e32 v2, 0x3effffff
v_cvt_rpi_i32_f32_e32 v12, v2

v_mov_b32_e32 v3, 0x4b000001
v_cvt_rpi_i32_f32_e32 v13, v3

v_mov_b32_e32 v4, 0x40500000
v_cvt_flr_i32_f32_e64 v15, -v4

v_mov_b32_e32 v5, 0x7fc00000
v_cvt_flr_i32_f32_e32 v16, v5

v_mov_b32_e32 v6, 0xffc00000
v_cvt_rpi_i32_f32_e32 v17, v6

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
