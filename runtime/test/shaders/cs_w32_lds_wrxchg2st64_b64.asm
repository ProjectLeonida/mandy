; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; ds_wrxchg2st64_rtn_b64 (RDNA2 ISA, DS), 2 KiB LDS (RSRC2 0x20088), only lane 0 runs.
; The st64 form addresses its two elements at vaddr + offset0/offset1 * 512 bytes (64 pairs per step),
; as ds_write2st64_b64 addresses them. vdst is four VGPRs: vdst[0:1] = old at offset0,
; vdst[2:3] = old at offset1; each exchanged dword is one 32-bit OpAtomicExchange.
;
; Seeds:
;   M[0x200] = 0x44444444:11111111   M[0x400] = 0x88888888:55555555
;   M[8]     = 0xDEADBEEF            M[16]    = 0xFEEDFACE   (markers: a *8 offset would hit these)
; Lane 0:
;   ds_wrxchg2st64_rtn_b64 v[10:13], v1, v[4:5], v[6:7] offset0:1 offset1:2, base 0
;     DATA 0x77777777:33333333 at 0x200, 0xEEEEEEEE:AAAAAAAA at 0x400
;     -> v10/v11 = 0x11111111 / 0x44444444, v12/v13 = 0x55555555 / 0x88888888
;     -> M[0x200] = 0x77777777:33333333, M[0x400] = 0xEEEEEEEE:AAAAAAAA
;
; out[0..7] = 0x11111111 0x44444444 0x55555555 0x88888888 0x33333333 0x77777777 0xAAAAAAAA 0xEEEEEEEE
;
; rsrc2: 0x20088
; group: 32,1,1
; Run: vk_compute_test cs_w32_lds_wrxchg2st64_b64.spv 1 dump
;   -> out[0..7] = 0x11111111 0x44444444 0x55555555 0x88888888 0x33333333 0x77777777 0xAAAAAAAA 0xEEEEEEEE
; mattr: +wavefrontsize32,-wavefrontsize64
v_mov_b32_e32 v1, 0x200
v_mov_b32_e32 v2, 0x11111111
v_mov_b32_e32 v3, 0x44444444
ds_write_b64 v1, v[2:3]
v_mov_b32_e32 v1, 0x400
v_mov_b32_e32 v2, 0x55555555
v_mov_b32_e32 v3, 0x88888888
ds_write_b64 v1, v[2:3]
v_mov_b32_e32 v1, 8
v_mov_b32_e32 v2, 0xdeadbeef
ds_write_b32 v1, v2
v_mov_b32_e32 v1, 16
v_mov_b32_e32 v2, 0xfeedface
ds_write_b32 v1, v2
s_waitcnt lgkmcnt(0)
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v1, 0
v_mov_b32_e32 v4, 0x33333333
v_mov_b32_e32 v5, 0x77777777
v_mov_b32_e32 v6, 0xaaaaaaaa
v_mov_b32_e32 v7, 0xeeeeeeee
ds_wrxchg2st64_rtn_b64 v[10:13], v1, v[4:5], v[6:7] offset0:1 offset1:2
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v1, 0x200
ds_read_b32 v14, v1
v_mov_b32_e32 v1, 0x204
ds_read_b32 v15, v1
v_mov_b32_e32 v1, 0x400
ds_read_b32 v16, v1
v_mov_b32_e32 v1, 0x404
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
