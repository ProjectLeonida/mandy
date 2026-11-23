; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; The ten integer 32-bit GLOBAL atomics, memory result (#226): add, sub, smin, umin, smax, umax, and, or.
; Lane 0 of a wave32 group seeds eight dwords through the output V# (base 0), runs one atomic on each through a
; global pointer (s[20:21] = 0, vaddr v1 = 0, so the address is the instruction offset), waits, reads each dword
; back and stores it:
;   0x40 add  5 + 0x7FFFFFFF          = 0x80000004
;   0x44 sub  5 - 3                   = 2
;   0x48 smin 0xFFFFFFFB (-5), 3      = 0xFFFFFFFB  (signed: -5 < 3)
;   0x4C umin 0xFFFFFFFB, 3           = 3           (unsigned: 3 < 0xFFFFFFFB)
;   0x50 smax 0xFFFFFFFB (-5), 3      = 3           (signed: 3 > -5)
;   0x54 umax 0xFFFFFFFB, 3           = 0xFFFFFFFB  (unsigned)
;   0x58 and  0xF0F0F0F0 & 0xFF00FF00 = 0xF000F000
;   0x5C or   0x0F0F0F0F | 0xFF00FF00 = 0xFF0FFF0F
; The pairs above are what tells the signed and unsigned forms apart: the same operand pair gives opposite results.
; The `glc` returns and the other seven ops are in cs_global_atomic32_rtn.asm / cs_global_atomic32_float.asm.
; group: 32,1,1
; Run: VKC_SUBGROUP_SIZE=32 vk_compute_test cs_global_atomic32_int.spv 1 dump -> out[0..7] = 2147483652 2 4294967291 3 3 4294967291 4026593280 4279238415
; mattr: +wavefrontsize32,-wavefrontsize64
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v11, 0
v_mov_b32_e32 v10, 5
buffer_store_dword v10, v11, s[0:3], 0 offen offset:0x40
buffer_store_dword v10, v11, s[0:3], 0 offen offset:0x44
v_mov_b32_e32 v10, 0xfffffffb
buffer_store_dword v10, v11, s[0:3], 0 offen offset:0x48
buffer_store_dword v10, v11, s[0:3], 0 offen offset:0x4c
buffer_store_dword v10, v11, s[0:3], 0 offen offset:0x50
buffer_store_dword v10, v11, s[0:3], 0 offen offset:0x54
v_mov_b32_e32 v10, 0xf0f0f0f0
buffer_store_dword v10, v11, s[0:3], 0 offen offset:0x58
v_mov_b32_e32 v10, 0x0f0f0f0f
buffer_store_dword v10, v11, s[0:3], 0 offen offset:0x5c
s_waitcnt vmcnt(0)
s_waitcnt_vscnt null, 0x0
s_mov_b32 s20, 0
s_mov_b32 s21, 0
v_mov_b32 v1, 0
v_mov_b32 v2, 0x7fffffff
global_atomic_add v1, v2, s[20:21] offset:0x40
v_mov_b32 v2, 3
global_atomic_sub v1, v2, s[20:21] offset:0x44
v_mov_b32 v2, 3
global_atomic_smin v1, v2, s[20:21] offset:0x48
global_atomic_umin v1, v2, s[20:21] offset:0x4c
global_atomic_smax v1, v2, s[20:21] offset:0x50
global_atomic_umax v1, v2, s[20:21] offset:0x54
v_mov_b32 v2, 0xff00ff00
global_atomic_and v1, v2, s[20:21] offset:0x58
global_atomic_or v1, v2, s[20:21] offset:0x5c
s_waitcnt vmcnt(0)
s_waitcnt_vscnt null, 0x0
buffer_load_dword v3, v11, s[0:3], 0 offen offset:0x40
buffer_load_dword v4, v11, s[0:3], 0 offen offset:0x44
buffer_load_dword v5, v11, s[0:3], 0 offen offset:0x48
buffer_load_dword v6, v11, s[0:3], 0 offen offset:0x4c
buffer_load_dword v7, v11, s[0:3], 0 offen offset:0x50
buffer_load_dword v8, v11, s[0:3], 0 offen offset:0x54
buffer_load_dword v9, v11, s[0:3], 0 offen offset:0x58
buffer_load_dword v10, v11, s[0:3], 0 offen offset:0x5c
s_waitcnt vmcnt(0)
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v11, 0
buffer_store_dword v3, v11, s[0:3], 0 offen
buffer_store_dword v4, v11, s[0:3], 0 offen offset:4
buffer_store_dword v5, v11, s[0:3], 0 offen offset:8
buffer_store_dword v6, v11, s[0:3], 0 offen offset:12
buffer_store_dword v7, v11, s[0:3], 0 offen offset:16
buffer_store_dword v8, v11, s[0:3], 0 offen offset:20
buffer_store_dword v9, v11, s[0:3], 0 offen offset:24
buffer_store_dword v10, v11, s[0:3], 0 offen offset:28
s_endpgm
