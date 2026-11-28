; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; Ops merged with CPU-only checks (#1954 ds_mskor, #1974 v_add/sub_nc_i16 clamp, #1948 s_buffer_load constant
; SGPR offset), run once on a GPU. Lane 0 of a wave32 group stores eight dwords:
;   out[0] = ds_mskor_rtn_b32 old value (LDS 0x12345678, DATA 0xFF00, DATA2 0xAB00)          = 0x12345678
;   out[1] = the LDS word after it: (0x12345678 & ~0xFF00) | 0xAB00                         = 0x1234AB78
;   out[2] = the LDS word after ds_mskor_b32 (LDS ~0, DATA 0xFFFF, DATA2 1)                 = 0xFFFF0001
;   out[3] = v_add_nc_i16 0x7FFF + 1 clamp into a zeroed register                            = 0x00007FFF
;   out[4] = v_sub_nc_i16 0x8000 - 1 clamp into a zeroed register                            = 0x00008000
;   out[5] = v_add_nc_i16 0x7FFF + 1 without clamp (wraps)                                   = 0x00008000
;   out[6] = s_buffer_load_dword with s12 = -4, offset:20 on the output V# (num_records 0xFFFFFFFF): the sum
;            0xFFFFFFFC + 20 is not reduced mod 2^32, so it is out of range and reads 0       = 0
;            (before #1948 the folded constant wrapped to 16 and read the 0xEE fill)
;   out[7] = the same load with s12 = 12 (byte 32, never written: the harness's 0xEE fill)   = 0xEEEEEEEE
; The scalar loads run before any store. EXEC is cut to lane 0 before the LDS ops (one LDS address, #1850).
; group: 32,1,1
; Run: vk_compute_test cs_w32_late_ext.spv 1 dump -> out[0..7] = 305419896 305441656 4294901761 32767 32768 32768 0 4008636142
; mattr: +wavefrontsize32,-wavefrontsize64
s_mov_b32 s12, -4
s_buffer_load_dword s20, s[0:3], s12 offset:20
s_mov_b32 s13, 12
s_buffer_load_dword s21, s[0:3], s13 offset:20
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v1, 0x7fff
v_mov_b32_e32 v2, 1
v_mov_b32_e32 v3, 0
v_add_nc_i16 v3, v1, v2 clamp
v_mov_b32_e32 v4, 0x8000
v_mov_b32_e32 v5, 0
v_sub_nc_i16 v5, v4, v2 clamp
v_mov_b32_e32 v6, 0
v_add_nc_i16 v6, v1, v2
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v10, 0x40
v_mov_b32_e32 v11, 0x12345678
ds_write_b32 v10, v11
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v12, 0xff00
v_mov_b32_e32 v13, 0xab00
ds_mskor_rtn_b32 v14, v10, v12, v13
s_waitcnt lgkmcnt(0)
ds_read_b32 v15, v10
v_mov_b32_e32 v16, 0x80
v_mov_b32_e32 v17, -1
ds_write_b32 v16, v17
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v12, 0xffff
v_mov_b32_e32 v13, 1
ds_mskor_b32 v16, v12, v13
s_waitcnt lgkmcnt(0)
ds_read_b32 v18, v16
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v20, 0
buffer_store_dword v14, v20, s[0:3], 0 offen
v_mov_b32_e32 v20, 4
buffer_store_dword v15, v20, s[0:3], 0 offen
v_mov_b32_e32 v20, 8
buffer_store_dword v18, v20, s[0:3], 0 offen
v_mov_b32_e32 v20, 12
buffer_store_dword v3, v20, s[0:3], 0 offen
v_mov_b32_e32 v20, 16
buffer_store_dword v5, v20, s[0:3], 0 offen
v_mov_b32_e32 v20, 20
buffer_store_dword v6, v20, s[0:3], 0 offen
v_mov_b32_e32 v20, 24
v_mov_b32_e32 v21, s20
buffer_store_dword v21, v20, s[0:3], 0 offen
v_mov_b32_e32 v20, 28
v_mov_b32_e32 v21, s21
buffer_store_dword v21, v20, s[0:3], 0 offen
s_endpgm
