; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; ds_wrxchg_rtn_b64 and ds_wrxchg2_rtn_b64 (RDNA2 ISA, DS), 512 B LDS, only lane 0 runs.
; A 64-bit element is two dwords, low first: M[A] = DATA and vdst gets the old 64 bits.
; ds_wrxchg_rtn_b64 has one element at vaddr + offset (bytes). ds_wrxchg2_rtn_b64 has two at
; vaddr + offset0/offset1 * 8, as ds_write2_b64 addresses them, and vdst is four VGPRs:
; vdst[0:1] = old at offset0, vdst[2:3] = old at offset1.
;
; Seeds:
;   M[0]  = 0x88888888:11111111   M[8]  = 0x44444444:22222222   M[16] = 0xCCCCCCCC:33333333
; Lane 0:
;   ds_wrxchg_rtn_b64 v[10:11], v1, v[4:5] at 0, DATA 0xAAAAAAAA:BBBBBBBB
;     -> v10/v11 = 0x11111111 / 0x88888888, M[0] = BBBBBBBB:AAAAAAAA
;   ds_wrxchg2_rtn_b64 v[12:15], v1, v[4:5], v[6:7] offset0:1 offset1:2, base 0
;     addresses 8 and 16, DATA 0xCCCCCCCC:DDDDDDDD and 0xFFFFFFFF:EEEEEEEE
;     -> v12/v13 = 0x22222222 / 0x44444444 (old at offset0)
;        v14/v15 = 0x33333333 / 0xCCCCCCCC (old at offset1)
;   The two readbacks take the low dword each exchange wrote: M[8] = 0xDDDDDDDD, M[16] = 0xEEEEEEEE.
;
; out[0..7] = 0x11111111 0x88888888 0x22222222 0x44444444 0x33333333 0xCCCCCCCC 0xDDDDDDDD 0xEEEEEEEE
;
; group: 32,1,1
; Run: vk_compute_test cs_w32_lds_wrxchg_b64.spv 1 dump
;   -> out[0..7] = 0x11111111 0x88888888 0x22222222 0x44444444 0x33333333 0xCCCCCCCC 0xDDDDDDDD 0xEEEEEEEE
; mattr: +wavefrontsize32,-wavefrontsize64
v_mov_b32_e32 v1, 0
v_mov_b32_e32 v2, 0x11111111
v_mov_b32_e32 v3, 0x88888888
ds_write_b64 v1, v[2:3]
v_mov_b32_e32 v1, 8
v_mov_b32_e32 v2, 0x22222222
v_mov_b32_e32 v3, 0x44444444
ds_write_b64 v1, v[2:3]
v_mov_b32_e32 v1, 16
v_mov_b32_e32 v2, 0x33333333
v_mov_b32_e32 v3, 0xcccccccc
ds_write_b64 v1, v[2:3]
s_waitcnt lgkmcnt(0)
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v1, 0
v_mov_b32_e32 v4, 0xbbbbbbbb
v_mov_b32_e32 v5, 0xaaaaaaaa
v_mov_b32_e32 v6, 0xdddddddd
v_mov_b32_e32 v7, 0xcccccccc
v_mov_b32_e32 v8, 0xeeeeeeee
v_mov_b32_e32 v9, 0xffffffff
ds_wrxchg_rtn_b64 v[10:11], v1, v[4:5]
ds_wrxchg2_rtn_b64 v[12:15], v1, v[6:7], v[8:9] offset0:1 offset1:2
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v1, 8
ds_read_b32 v16, v1
v_mov_b32_e32 v1, 16
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
