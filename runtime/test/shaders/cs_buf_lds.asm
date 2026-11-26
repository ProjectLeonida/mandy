; MUBUF loads with the `lds` modifier (RDNA2 ISA 8.1.9 "Memory Buffer Load to LDS"; docs/design.md). Lane t of the
; 32-thread group seeds heap dword t at 0x400 + 4t with 0x5EED0000 + t and byte 0x80 + t at 0x480 + t, then:
;   m0 = 64:  buffer_load_dword  v1, s[0:3], s8 offen lds   -> LDS[64  + 4t] = 0x5EED0000 + t
;   m0 = 192: buffer_load_ubyte  v1, s[0:3], s8 offen lds   -> LDS[192 + 4t] = 0x00000080 + t   (zero-extended)
;   m0 = 320: buffer_load_sbyte  v1, s[0:3], s8 offen lds   -> LDS[320 + 4t] = 0xFFFFFF80 + t   (sign-extended)
; (s8 = 0x400 is the SOFFSET; the ubyte/sbyte vaddr adds 0x80 so the address is 0x480 + t. LDS_ADDR = M0[15:0] +
; TIDinWave*4; TIDinWave is added on the LDS path even though the V#'s ADD_TID_ENABLE is clear.) After
; s_waitcnt + s_barrier each lane ds_read_b32 its own three slots and lane t^1's dword slot back. Lanes 0 and 1
; then store four words each through the output V# s[0:3] (EXEC is cut to them; the console dump judges out[0..7]):
;   out[0..3] = lane 0: 0x5EED0000, 0x00000080, 0xFFFFFF80, 0x5EED0001 (lane 1's slot)
;   out[4..7] = lane 1: 0x5EED0001, 0x00000081, 0xFFFFFF81, 0x5EED0000 (lane 0's slot)
; group: 32,1,1
; Run: vk_compute_test cs_buf_lds.spv 1 dump -> out[0..7] = 0x5EED0000 0x80 0xFFFFFF80 0x5EED0001 0x5EED0001 0x81 0xFFFFFF81 0x5EED0000
; mattr: +wavefrontsize32,-wavefrontsize64
v_lshlrev_b32_e32 v10, 2, v0
v_add_nc_u32_e32 v10, 0x400, v10
v_add_nc_u32_e32 v11, 0x5eed0000, v0
buffer_store_dword v11, v10, s[0:3], 0 offen
v_add_nc_u32_e32 v12, 0x480, v0
v_add_nc_u32_e32 v13, 0x80, v0
buffer_store_byte v13, v12, s[0:3], 0 offen
s_waitcnt vmcnt(0)
s_mov_b32 s8, 0x400
v_lshlrev_b32_e32 v1, 2, v0
s_mov_b32 m0, 64
buffer_load_dword v1, s[0:3], s8 offen lds
v_add_nc_u32_e32 v1, 0x80, v0
s_mov_b32 m0, 192
buffer_load_ubyte v1, s[0:3], s8 offen lds
s_mov_b32 m0, 320
buffer_load_sbyte v1, s[0:3], s8 offen lds
s_waitcnt vmcnt(0)
s_barrier
v_lshlrev_b32_e32 v20, 2, v0
v_add_nc_u32_e32 v21, 64, v20
ds_read_b32 v22, v21
v_add_nc_u32_e32 v23, 192, v20
ds_read_b32 v24, v23
v_add_nc_u32_e32 v25, 320, v20
ds_read_b32 v26, v25
v_xor_b32_e32 v27, 1, v0
v_lshlrev_b32_e32 v27, 2, v27
v_add_nc_u32_e32 v27, 64, v27
ds_read_b32 v28, v27
s_waitcnt lgkmcnt(0)
v_lshrrev_b32_e32 v31, 1, v0
v_cmpx_eq_u32_e32 0, v31
v_lshlrev_b32_e32 v30, 4, v0
buffer_store_dword v22, v30, s[0:3], 0 offen
buffer_store_dword v24, v30, s[0:3], 0 offen offset:4
buffer_store_dword v26, v30, s[0:3], 0 offen offset:8
buffer_store_dword v28, v30, s[0:3], 0 offen offset:12
s_endpgm
