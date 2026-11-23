; s_buffer_load out of range by more than 4 GB, with the offset in a register (#4573): the translator guards each dword
; with an OpSelectionMerge on the bound and gives 0 outside it, and the guarded load reads through
; OpSelect(in range, address, heap base), because llvmpipe runs it anyway (without the select this fixture segfaults
; vk_compute_test on llvmpipe, Mesa 26.0.8). The V# is vk_compute_test's (base 0, num_records 0xFFFFFFFF), so every
; out-of-range offset here is past 2^32; an in-range one 4 GB from a small buffer would be a real far read. All scalar
; loads run before any store. The ds_mskor_rtn_b32 in lane 0 after them is the shape that made llvmpipe hoist the
; load (reduced from #3039's cs_w32_late_ext); it touches LDS dword 0 only. Lane 0 stores eight dwords:
;   out[0] = s_buffer_load_dword, sOFF = -4 (from v_readfirstlane), offset:20: 0xFFFFFFFC + 20 past the bound = 0
;   out[1] = the same with sOFF = 12: byte 32, never written (the harness's 0xEE fill)        = 0xEEEEEEEE
;   out[2], out[3] = s_buffer_load_dwordx2, sOFF = -4, offset:0x10: both dwords past the bound  = 0 0
;   out[4] = s_buffer_load_dword, sOFF = -0x20, offset:0x40 (0x100000020)                       = 0
;   out[5] = s_buffer_load_dword, sOFF = 0, offset:0x24: byte 36, the fill                      = 0xEEEEEEEE
;   out[6] = sOFF = -4 from s_mov (a known constant), offset:20: still out of range             = 0
;   out[7] = sOFF = 12 from s_mov, offset:20: byte 32, the fill                                 = 0xEEEEEEEE
; group: 32,1,1
; Run: vk_compute_test cs_smem_oob_far.spv 1 dump -> out[0..7] = 0 4008636142 0 0 0 4008636142 0 4008636142
; mattr: +wavefrontsize32,-wavefrontsize64
v_mov_b32_e32 v1, -4
v_readfirstlane_b32 s12, v1
v_mov_b32_e32 v1, 12
v_readfirstlane_b32 s13, v1
v_mov_b32_e32 v1, 0xffffffe0
v_readfirstlane_b32 s14, v1
v_mov_b32_e32 v1, 0
v_readfirstlane_b32 s15, v1
s_buffer_load_dword s20, s[0:3], s12 offset:20
s_buffer_load_dword s21, s[0:3], s13 offset:20
s_buffer_load_dwordx2 s[22:23], s[0:3], s12 offset:0x10
s_buffer_load_dword s24, s[0:3], s14 offset:0x40
s_buffer_load_dword s25, s[0:3], s15 offset:0x24
s_mov_b32 s16, -4
s_buffer_load_dword s26, s[0:3], s16 offset:20
s_mov_b32 s17, 12
s_buffer_load_dword s27, s[0:3], s17 offset:20
s_waitcnt lgkmcnt(0)
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v10, 0
v_mov_b32_e32 v12, 0
v_mov_b32_e32 v13, 0
ds_mskor_rtn_b32 v14, v10, v12, v13
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v20, 0
v_mov_b32_e32 v21, s20
buffer_store_dword v21, v20, s[0:3], 0 offen
v_mov_b32_e32 v20, 4
v_mov_b32_e32 v21, s21
buffer_store_dword v21, v20, s[0:3], 0 offen
v_mov_b32_e32 v20, 8
v_mov_b32_e32 v21, s22
buffer_store_dword v21, v20, s[0:3], 0 offen
v_mov_b32_e32 v20, 12
v_mov_b32_e32 v21, s23
buffer_store_dword v21, v20, s[0:3], 0 offen
v_mov_b32_e32 v20, 16
v_mov_b32_e32 v21, s24
buffer_store_dword v21, v20, s[0:3], 0 offen
v_mov_b32_e32 v20, 20
v_mov_b32_e32 v21, s25
buffer_store_dword v21, v20, s[0:3], 0 offen
v_mov_b32_e32 v20, 24
v_mov_b32_e32 v21, s26
buffer_store_dword v21, v20, s[0:3], 0 offen
v_mov_b32_e32 v20, 28
v_mov_b32_e32 v21, s27
buffer_store_dword v21, v20, s[0:3], 0 offen
s_endpgm
