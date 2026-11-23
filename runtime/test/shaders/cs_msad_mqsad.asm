; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; v_msad_u8 and v_mqsad_u32_u8 (RDNA2 ISA VOP3), lane 0 of a wave32 group.
; Both are unsigned byte sums of absolute differences that wrap at 32 bits; a byte where S1's byte is 0
; adds nothing (masked SAD). v_mqsad_u32_u8 runs four of those over the sliding 32-bit windows
; S0[8k+31 : 8k] of the 64-bit S0 (k = 0..3) and adds each to its own 32-bit accumulator S2[k].
; v2:v3 = S0 = 0x0001020304050607, v4 = S1 = 0x00000201 (bytes 0x01, 0x02, 0, 0), S2 = v6..v9.
;   out[0] = v_msad_u8(0x10203040, 0x00100020, 0)        = 0x30       bytes 1 and 3 masked: 0x20 + 0x10
;   out[1] = v_msad_u8(0, 0x01010101, 0xFFFFFFF0)         = 0xFFFFFFF4 four 1s, wraps at 32 bits
;   out[2] = mqsad window 0 (0x04050607): |07-01| + |06-02|                     = 10
;            D[0] = 0xFFFFFFF0 + 10                       = 0xFFFFFFFA
;   out[3] = mqsad window 1 (0x03040506): |06-01| + |05-02|                     = 8
;            D[1] = 0xFFFFFFFF + 8                        = 0x00000007 (wraps)
;   out[4] = mqsad window 2 (0x02030405): |05-01| + |04-02|                     = 6
;            D[2] = 0x80000000 + 6                        = 0x80000006
;   out[5] = mqsad window 3 (0x01020304): |04-01| + |03-02|                     = 4
;            D[3] = 0x12345678 + 4                        = 0x1234567C
;   out[6] = v_msad_u8(0xFFFFFFFF, 0, 0xDEADBEEF)         = 0xDEADBEEF all four bytes masked, S2 unchanged
;   out[7] = v_msad_u8(1, 0x0000FF00, 0)                  = 0x000000FF only byte 1 counts, UMax - UMin
; The four mqsad results are read back from v10..v13.
; group: 32,1,1
; Run: vk_compute_test cs_msad_mqsad.spv 1 dump -> out[0..7] = 48 4294967284 4294967290 7 2147483654 305419900 3735928559 255
; mattr: +wavefrontsize32,-wavefrontsize64
v_mov_b32_e32 v2, 0x04050607
v_mov_b32_e32 v3, 0x00010203
v_mov_b32_e32 v4, 0x00000201
v_mov_b32_e32 v6, 0xfffffff0
v_mov_b32_e32 v7, 0xffffffff
v_mov_b32_e32 v8, 0x80000000
v_mov_b32_e32 v9, 0x12345678
v_mqsad_u32_u8 v[10:13], v[2:3], v4, v[6:9]
v_mov_b32_e32 v5, 0x10203040
v_mov_b32_e32 v14, 0x00100020
v_msad_u8 v15, v5, v14, 0
v_mov_b32_e32 v16, 0
v_mov_b32_e32 v17, 0x01010101
v_mov_b32_e32 v18, 0xfffffff0
v_msad_u8 v19, v16, v17, v18
v_mov_b32_e32 v20, 0xdeadbeef
v_mov_b32_e32 v22, 0xffffffff
v_mov_b32_e32 v23, 0
v_msad_u8 v24, v22, v23, v20
v_mov_b32_e32 v25, 1
v_mov_b32_e32 v26, 0x0000ff00
v_msad_u8 v27, v25, v26, 0
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v21, 0
buffer_store_dword v15, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 4
buffer_store_dword v19, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 8
buffer_store_dword v10, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 12
buffer_store_dword v11, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 16
buffer_store_dword v12, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 20
buffer_store_dword v13, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 24
buffer_store_dword v24, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 28
buffer_store_dword v27, v21, s[0:3], 0 offen
s_endpgm
