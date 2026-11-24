; v_mqsad_u32_u8 through the translated path (TEST-644). Every lane computes the same two results, lane 0 stores them.
; S0 = 0x88776655_44332211 (64-bit), S1 = 0x40005000 (bytes 0 and 2 are 0, so v_msad_u8 skips them).
; D[k] = S2[k] + sum over S1's non-zero bytes of |S0[8k+31:8k] byte - S1 byte|, wrapping at 32 bits:
;   window k=0 0x44332211: |0x22-0x50| + |0x44-0x40| = 46 + 4  = 0x32
;   window k=1 0x55443322: |0x33-0x50| + |0x55-0x40| = 29 + 21 = 0x32
;   window k=2 0x66554433: |0x44-0x50| + |0x66-0x40| = 12 + 38 = 0x32
;   window k=3 0x77665544: |0x55-0x50| + |0x77-0x40| = 5 + 55  = 0x3C
; A: D = S2 (in place), S2 = {0xFFFFFFF0, 1, 2, 3} -> out[0..3] = 0x22 (wraps), 0x33, 0x34, 0x3F
; B: D overlaps S0 (v[2:5] over v[2:3]), S2 = {0x100, 0x200, 0x300, 0x400}: every source is read before the first
;    write, so B uses the original S0 -> out[4..7] = 0x132, 0x232, 0x332, 0x43C
; group: 32,1,1
; Run: vk_compute_test cs_w32_mqsad_u32.spv 1 dump -> out[0..7] = 34 51 52 63 306 562 818 1084
; mattr: +wavefrontsize32,-wavefrontsize64
v_mov_b32_e32 v2, 0x44332211
v_mov_b32_e32 v3, 0x88776655
v_mov_b32_e32 v4, 0x40005000
v_mov_b32_e32 v10, 0xfffffff0
v_mov_b32_e32 v11, 1
v_mov_b32_e32 v12, 2
v_mov_b32_e32 v13, 3
v_mqsad_u32_u8 v[10:13], v[2:3], v4, v[10:13]
v_mov_b32_e32 v20, 0x100
v_mov_b32_e32 v21, 0x200
v_mov_b32_e32 v22, 0x300
v_mov_b32_e32 v23, 0x400
v_mqsad_u32_u8 v[2:5], v[2:3], v4, v[20:23]
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
buffer_store_dword v2, v25, s[0:3], 0 offen
v_mov_b32_e32 v25, 20
buffer_store_dword v3, v25, s[0:3], 0 offen
v_mov_b32_e32 v25, 24
buffer_store_dword v4, v25, s[0:3], 0 offen
v_mov_b32_e32 v25, 28
buffer_store_dword v5, v25, s[0:3], 0 offen
s_endpgm
