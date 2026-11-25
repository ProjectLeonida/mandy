; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; ds_mskor_b32 / ds_mskor_rtn_b32 (RDNA2 ISA, "Data Share Instructions"), LDS 512 B, only lane 0 runs.
;   tmp = M[A]; M[A] = (tmp & ~DATA) | DATA2; the _rtn_ form returns tmp in vdst.
; DATA is the mask, DATA2 is ORed in raw: bits outside DATA are ORed too, it is not a masked insert.
;
; Seeds (all lanes write the same words):
;   M[0]  = 0x11223344   M[4]  = 0xFFFF0000   M[8]  = 0x0F0F0F0F   M[12] = 0xCAFEBABE
; Lane 0 then runs:
;   ds_mskor_rtn_b32 v5, v1, v3, v4                 at 0, DATA 0x0000FFFF, DATA2 0x00000008
;     old = 0x11223344, new = (0x11223344 & 0xFFFF0000) | 0x00000008 = 0x11220008
;     (bit 3 is outside DATA, so it comes from DATA2 alone)
;   ds_mskor_rtn_b32 v6, v1, v3, v4 offset:4        at 4, DATA 0x00FF00FF, DATA2 0x00000001
;     old = 0xFFFF0000, new = (0xFFFF0000 & 0xFF00FF00) | 0x00000001 = 0xFF000001
;   ds_mskor_b32 v1, v3, v4                         at 8, DATA 0xF0F0F0F0, DATA2 0x50505050
;     new = (0x0F0F0F0F & 0x0F0F0F0F) | 0x50505050 = 0x5F5F5F5F
;     (DATA2 bits 4 and 6 are outside DATA and are ORed in)
;   ds_mskor_b32 v1, v3, v4                         at 12, DATA 0xFFFFFFFF, DATA2 0x12345678
;     new = (0xCAFEBABE & 0) | 0x12345678 = 0x12345678
; The non-returning form has no vdst: v7 (0xDEADBEEF) and v8 (0xFEEDFACE) keep their values.
; Under EXEC only lane 0 runs the four atomics, so the two atomics each one expands to cannot
; interleave with another lane's.
;
; out[0] rtn old 0        0x11223344   out[1] M[0] after   0x11220008
; out[2] rtn old 4        0xFFFF0000   out[3] M[4] after   0xFF000001
; out[4] v7 preserved     0xDEADBEEF   out[5] M[8] after   0x5F5F5F5F
; out[6] v8 preserved     0xFEEDFACE   out[7] M[12] after  0x12345678
;
; group: 32,1,1
; Run: vk_compute_test cs_w32_lds_mskor.spv 1 dump
;   -> out[0..7] = 0x11223344 0x11220008 0xFFFF0000 0xFF000001 0xDEADBEEF 0x5F5F5F5F 0xFEEDFACE 0x12345678
; mattr: +wavefrontsize32,-wavefrontsize64
v_mov_b32_e32 v1, 0
v_mov_b32_e32 v2, 0x11223344
ds_write_b32 v1, v2
v_mov_b32_e32 v1, 4
v_mov_b32_e32 v2, 0xffff0000
ds_write_b32 v1, v2
v_mov_b32_e32 v1, 8
v_mov_b32_e32 v2, 0x0f0f0f0f
ds_write_b32 v1, v2
v_mov_b32_e32 v1, 12
v_mov_b32_e32 v2, 0xcafebabe
ds_write_b32 v1, v2
s_waitcnt lgkmcnt(0)
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v1, 0
v_mov_b32_e32 v3, 0x0000ffff
v_mov_b32_e32 v4, 0x00000008
ds_mskor_rtn_b32 v5, v1, v3, v4
v_mov_b32_e32 v3, 0x00ff00ff
v_mov_b32_e32 v4, 0x00000001
ds_mskor_rtn_b32 v6, v1, v3, v4 offset:4
v_mov_b32_e32 v7, 0xdeadbeef
v_mov_b32_e32 v1, 8
v_mov_b32_e32 v3, 0xf0f0f0f0
v_mov_b32_e32 v4, 0x50505050
ds_mskor_b32 v1, v3, v4
v_mov_b32_e32 v8, 0xfeedface
v_mov_b32_e32 v1, 12
v_mov_b32_e32 v3, 0xffffffff
v_mov_b32_e32 v4, 0x12345678
ds_mskor_b32 v1, v3, v4
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v1, 0
ds_read_b32 v10, v1
v_mov_b32_e32 v1, 4
ds_read_b32 v11, v1
v_mov_b32_e32 v1, 8
ds_read_b32 v12, v1
v_mov_b32_e32 v1, 12
ds_read_b32 v13, v1
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v21, 0
buffer_store_dword v5, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 4
buffer_store_dword v10, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 8
buffer_store_dword v6, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 12
buffer_store_dword v11, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 16
buffer_store_dword v7, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 20
buffer_store_dword v12, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 24
buffer_store_dword v8, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 28
buffer_store_dword v13, v21, s[0:3], 0 offen
s_endpgm
