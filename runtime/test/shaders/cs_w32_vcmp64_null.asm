; Null sources in 64-bit integer VALU compares (docs/design.md contract row):
; v_cmp and v_cmpx relations eq, ne, lt, le, gt, ge on u64 and i64 with null (= 0) in src0, src1, or both.
; Wave32 (32 lanes): each lane i gets a 64-bit value V(i) by (i & 3):
;   r=0: 0x00000000_00000000 (0)
;   r=1: 0x00000000_00000001 (+1)
;   r=2: 0x00000001_00000000 (2^32, high dword non-zero, low dword 0)
;   r=3: 0x80000000_00000000 (2^63 unsigned / -2^63 signed)
; Each lane ORs (1 << lane) into LDS[k] when its bit in mask k is set, then lane 0 stores eight 32-bit words:
;   out[0] = v_cmp_eq_u64_e32 vcc_lo, null, v[2:3]       (0 == V; r=0 only)                  = 0x11111111
;   out[1] = v_cmp_ge_i64_e64 s8, v[2:3], null           (V >= 0 signed; r=0,1,2)            = 0x77777777
;   out[2] = v_cmp_lt_i64_e32 vcc_lo, null, v[2:3]       (0 < V signed; r=1,2)               = 0x66666666
;   out[3] = v_cmp_gt_i64_e64 s10, null, v[2:3]          (0 > V signed; r=3 only)            = 0x88888888
;   out[4] = v_cmp_gt_u64_e64 s11, v[2:3], null under EXEC=0x00FF00FF (V > 0 u64, masked)    = 0x00EE00EE
;   out[5] = (v_cmp_le_u64_e64 null, null | v_cmp_ne_i64_e64 null, null) under EXEC=0x5555AAAA = 0x5555AAAA
;   out[6] = EXEC after v_cmpx_lt_i64_e64 v[2:3], null under EXEC=0x0FFFFFFF (V < 0 signed)  = 0x08888888
;   out[7] = VCC after that v_cmpx (must keep sentinel 0x96A53C69 untouched)                 = 0x96A53C69
; group: 32,1,1
; Run: VKC_SUBGROUP_SIZE=32 vk_compute_test cs_w32_vcmp64_null.spv 1 dump -> out[0..7] = 286331153 2004318071 1717986918 2290649224 15597806 1431677610 143165576 2527411305
; mattr: +wavefrontsize32,-wavefrontsize64
; regen_fixtures: wave32
v_and_b32_e32 v4, 3, v0
v_cmp_eq_u32_e32 vcc_lo, 1, v4
v_cndmask_b32_e64 v2, 0, 1, vcc_lo
v_cmp_eq_u32_e32 vcc_lo, 2, v4
v_cndmask_b32_e64 v3, 0, 1, vcc_lo
v_cmp_eq_u32_e32 vcc_lo, 3, v4
v_cndmask_b32_e64 v3, v3, 0x80000000, vcc_lo

v_cmp_eq_u64_e32 vcc_lo, null, v[2:3]
s_mov_b32 s6, vcc_lo

v_cmp_ge_i64_e64 s8, v[2:3], null

v_cmp_lt_i64_e32 vcc_lo, null, v[2:3]
s_mov_b32 s9, vcc_lo

v_cmp_gt_i64_e64 s10, null, v[2:3]

s_mov_b32 exec_lo, 0x00ff00ff
v_cmp_gt_u64_e64 s11, v[2:3], null

s_mov_b32 exec_lo, 0x5555aaaa
v_cmp_le_u64_e64 s12, null, null
v_cmp_ne_i64_e64 s13, null, null
s_or_b32 s12, s12, s13

s_mov_b32 vcc_lo, 0x96a53c69
s_mov_b32 exec_lo, 0x0fffffff
v_cmpx_lt_i64_e64 v[2:3], null
s_mov_b32 s14, exec_lo
s_mov_b32 s15, vcc_lo

s_mov_b32 exec_lo, -1
v_mov_b32_e32 v6, 0
v_mov_b32_e32 v7, 0
v_cmpx_eq_u32_e32 0, v0
ds_write_b32 v6, v7
ds_write_b32 v6, v7 offset:4
ds_write_b32 v6, v7 offset:8
ds_write_b32 v6, v7 offset:12
ds_write_b32 v6, v7 offset:16
ds_write_b32 v6, v7 offset:20
ds_write_b32 v6, v7 offset:24
s_waitcnt lgkmcnt(0)
s_mov_b32 exec_lo, -1
s_barrier

v_mov_b32_e32 v5, 1
v_lshlrev_b32_e32 v5, v0, v5
v_cndmask_b32_e64 v7, 0, v5, s6
ds_or_b32 v6, v7
v_cndmask_b32_e64 v7, 0, v5, s8
ds_or_b32 v6, v7 offset:4
v_cndmask_b32_e64 v7, 0, v5, s9
ds_or_b32 v6, v7 offset:8
v_cndmask_b32_e64 v7, 0, v5, s10
ds_or_b32 v6, v7 offset:12
v_cndmask_b32_e64 v7, 0, v5, s11
ds_or_b32 v6, v7 offset:16
v_cndmask_b32_e64 v7, 0, v5, s12
ds_or_b32 v6, v7 offset:20
v_cndmask_b32_e64 v7, 0, v5, s14
ds_or_b32 v6, v7 offset:24
s_waitcnt lgkmcnt(0)
s_barrier

v_cmpx_eq_u32_e32 0, v0
ds_read_b32 v21, v6
ds_read_b32 v22, v6 offset:4
ds_read_b32 v23, v6 offset:8
ds_read_b32 v24, v6 offset:12
ds_read_b32 v25, v6 offset:16
ds_read_b32 v26, v6 offset:20
ds_read_b32 v27, v6 offset:24
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v20, 0
buffer_store_dword v21, v20, s[0:3], 0 offen
v_mov_b32_e32 v20, 4
buffer_store_dword v22, v20, s[0:3], 0 offen
v_mov_b32_e32 v20, 8
buffer_store_dword v23, v20, s[0:3], 0 offen
v_mov_b32_e32 v20, 12
buffer_store_dword v24, v20, s[0:3], 0 offen
v_mov_b32_e32 v20, 16
buffer_store_dword v25, v20, s[0:3], 0 offen
v_mov_b32_e32 v20, 20
buffer_store_dword v26, v20, s[0:3], 0 offen
v_mov_b32_e32 v20, 24
buffer_store_dword v27, v20, s[0:3], 0 offen
v_mov_b32_e32 v20, 28
v_mov_b32_e32 v28, s15
buffer_store_dword v28, v20, s[0:3], 0 offen
s_endpgm
