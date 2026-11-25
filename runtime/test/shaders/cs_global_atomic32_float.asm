; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; The 32-bit float and compare-swap GLOBAL atomics (#226): fmin, fmax and cmpswap, hit and miss.
; Lane 0 of a wave32 group seeds six dwords through the output V# (base 0), runs the atomics through a global
; pointer (s[20:21] = 0, vaddr v1 = 0, so the address is the instruction offset), waits and reads them back:
;   0x40 fmin     1.0, 2.0                    -> 1.0 (0x3F800000)
;   0x44 fmax     1.0, 2.0                    -> 2.0 (0x40000000)
;   0x48 fmin     qNaN (0x7FC00001), 3.0      -> 3.0 (NMin returns the non-NaN operand, no float-atomic extension)
;   0x4C fmax     -0.0, +0.0                  -> +0.0 (0x00000000; NMax ranks -0 below +0)
;   0x50 cmpswap  0x42 with DATA 0x43, CMP 0x42 -> hit: memory 0x43, vdst the old word 0x42
;   0x54 cmpswap  0x55 with DATA 0x43, CMP 0x44 -> miss: memory and vdst both 0x55
; The vdata pair is {DATA, CMP}: DATA in the low VGPR, the comparator in the next. Both cmpswap forms carry glc,
; which is where the returned old word comes from; the memory words are out[4] (hit) and out[6] (miss).
; out[0] = 0x40, out[1] = 0x44, out[2] = 0x48, out[3] = 0x4C, out[4] = 0x50, out[5] = 0x50's vdst,
; out[6] = 0x54, out[7] = 0x54's vdst.
; group: 32,1,1
; Run: VKC_SUBGROUP_SIZE=32 vk_compute_test cs_global_atomic32_float.spv 1 dump -> out[0..7] = 1065353216 1073741824 1077936128 0 67 66 85 85
; mattr: +wavefrontsize32,-wavefrontsize64
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v11, 0
v_mov_b32_e32 v10, 0x3f800000
buffer_store_dword v10, v11, s[0:3], 0 offen offset:0x40
buffer_store_dword v10, v11, s[0:3], 0 offen offset:0x44
v_mov_b32_e32 v10, 0x7fc00001
buffer_store_dword v10, v11, s[0:3], 0 offen offset:0x48
v_mov_b32_e32 v10, 0x80000000
buffer_store_dword v10, v11, s[0:3], 0 offen offset:0x4c
v_mov_b32_e32 v10, 0x42
buffer_store_dword v10, v11, s[0:3], 0 offen offset:0x50
v_mov_b32_e32 v10, 0x55
buffer_store_dword v10, v11, s[0:3], 0 offen offset:0x54
s_waitcnt vmcnt(0)
s_waitcnt_vscnt null, 0x0
s_mov_b32 s20, 0
s_mov_b32 s21, 0
v_mov_b32 v1, 0
v_mov_b32 v2, 0x40000000
global_atomic_fmin v1, v2, s[20:21] offset:0x40
global_atomic_fmax v1, v2, s[20:21] offset:0x44
v_mov_b32 v2, 0x40400000
global_atomic_fmin v1, v2, s[20:21] offset:0x48
v_mov_b32 v2, 0
global_atomic_fmax v1, v2, s[20:21] offset:0x4c
v_mov_b32 v2, 0x43
v_mov_b32 v3, 0x42
global_atomic_cmpswap v4, v1, v[2:3], s[20:21] offset:0x50 glc
v_mov_b32 v2, 0x43
v_mov_b32 v3, 0x44
global_atomic_cmpswap v5, v1, v[2:3], s[20:21] offset:0x54 glc
s_waitcnt vmcnt(0)
s_waitcnt_vscnt null, 0x0
buffer_load_dword v6, v11, s[0:3], 0 offen offset:0x40
buffer_load_dword v7, v11, s[0:3], 0 offen offset:0x44
buffer_load_dword v8, v11, s[0:3], 0 offen offset:0x48
buffer_load_dword v9, v11, s[0:3], 0 offen offset:0x4c
buffer_load_dword v10, v11, s[0:3], 0 offen offset:0x50
buffer_load_dword v12, v11, s[0:3], 0 offen offset:0x54
s_waitcnt vmcnt(0)
buffer_store_dword v6, v11, s[0:3], 0 offen
buffer_store_dword v7, v11, s[0:3], 0 offen offset:4
buffer_store_dword v8, v11, s[0:3], 0 offen offset:8
buffer_store_dword v9, v11, s[0:3], 0 offen offset:12
buffer_store_dword v10, v11, s[0:3], 0 offen offset:16
buffer_store_dword v4, v11, s[0:3], 0 offen offset:20
buffer_store_dword v12, v11, s[0:3], 0 offen offset:24
buffer_store_dword v5, v11, s[0:3], 0 offen offset:28
s_endpgm
