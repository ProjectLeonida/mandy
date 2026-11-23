; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; The other 32-bit GLOBAL atomics (#226): xor, swap, inc, dec, and add/sub/xor/swap again in their returning
; (glc) form, whose word is the pre-op value.
; Lane 0 of a wave32 group seeds eight dwords through the output V# (base 0), runs one atomic on each through a
; global pointer (s[20:21] = 0, vaddr v1 = 0, so the address is the instruction offset), waits, and stores:
;   0x40 xor  0xF0F0F0F0 ^ 0x0F0F0F0F = 0xFFFFFFFF                  -> out[0]
;   0x44 swap 0x12345678 -> 0xCAFEF00D                              -> out[1] (the old word is gone)
;   0x48 inc  5 with DATA 10: 5 < 10, so 6                          -> out[2]
;   0x4C dec  5 with DATA 10: 0 < 5 < 10, so 4                      -> out[3]
;   0x50 add  glc: 0x11111111 + 0x22222222; vdst = the old word      -> out[4] = 0x11111111
;   0x54 sub  glc: 0x100 - 1;         vdst = the old word            -> out[5] = 0x100
;   0x58 xor  glc: 0xF0F0 ^ 0x0FF0;   vdst = the old word            -> out[6] = 0xF0F0
;   0x5C swap glc: 0xAAAA <- 0x5555;  vdst = the old word            -> out[7] = 0xAAAA
; inc wraps to 0 when M >= DATA and dec returns DATA when M == 0 or M > DATA (cs_global_atomic32_float.asm keeps
; the fmin/fmax and cmpswap cases); the glc words here are the pre-op values, which the memory result of the same
; op cannot be confused with.
; group: 32,1,1
; Run: VKC_SUBGROUP_SIZE=32 vk_compute_test cs_global_atomic32_rtn.spv 1 dump -> out[0..7] = 4294967295 3405705229 6 4 286331153 256 61680 43690
; mattr: +wavefrontsize32,-wavefrontsize64
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v11, 0
v_mov_b32_e32 v10, 0xf0f0f0f0
buffer_store_dword v10, v11, s[0:3], 0 offen offset:0x40
v_mov_b32_e32 v10, 0x12345678
buffer_store_dword v10, v11, s[0:3], 0 offen offset:0x44
v_mov_b32_e32 v10, 5
buffer_store_dword v10, v11, s[0:3], 0 offen offset:0x48
buffer_store_dword v10, v11, s[0:3], 0 offen offset:0x4c
v_mov_b32_e32 v10, 0x11111111
buffer_store_dword v10, v11, s[0:3], 0 offen offset:0x50
v_mov_b32_e32 v10, 0x100
buffer_store_dword v10, v11, s[0:3], 0 offen offset:0x54
v_mov_b32_e32 v10, 0xf0f0
buffer_store_dword v10, v11, s[0:3], 0 offen offset:0x58
v_mov_b32_e32 v10, 0xaaaa
buffer_store_dword v10, v11, s[0:3], 0 offen offset:0x5c
s_waitcnt vmcnt(0)
s_waitcnt_vscnt null, 0x0
s_mov_b32 s20, 0
s_mov_b32 s21, 0
v_mov_b32 v1, 0
v_mov_b32 v2, 0x0f0f0f0f
global_atomic_xor v1, v2, s[20:21] offset:0x40
v_mov_b32 v2, 0xcafef00d
global_atomic_swap v1, v2, s[20:21] offset:0x44
v_mov_b32 v2, 10
global_atomic_inc v1, v2, s[20:21] offset:0x48
global_atomic_dec v1, v2, s[20:21] offset:0x4c
v_mov_b32 v2, 0x22222222
global_atomic_add v6, v1, v2, s[20:21] offset:0x50 glc
v_mov_b32 v2, 1
global_atomic_sub v7, v1, v2, s[20:21] offset:0x54 glc
v_mov_b32 v2, 0x0ff0
global_atomic_xor v8, v1, v2, s[20:21] offset:0x58 glc
v_mov_b32 v2, 0x5555
global_atomic_swap v9, v1, v2, s[20:21] offset:0x5c glc
s_waitcnt vmcnt(0)
s_waitcnt_vscnt null, 0x0
buffer_load_dword v3, v11, s[0:3], 0 offen offset:0x40
buffer_load_dword v4, v11, s[0:3], 0 offen offset:0x44
buffer_load_dword v5, v11, s[0:3], 0 offen offset:0x48
buffer_load_dword v10, v11, s[0:3], 0 offen offset:0x4c
s_waitcnt vmcnt(0)
buffer_store_dword v3, v11, s[0:3], 0 offen
buffer_store_dword v4, v11, s[0:3], 0 offen offset:4
buffer_store_dword v5, v11, s[0:3], 0 offen offset:8
buffer_store_dword v10, v11, s[0:3], 0 offen offset:12
buffer_store_dword v6, v11, s[0:3], 0 offen offset:16
buffer_store_dword v7, v11, s[0:3], 0 offen offset:20
buffer_store_dword v8, v11, s[0:3], 0 offen offset:24
buffer_store_dword v9, v11, s[0:3], 0 offen offset:28
s_endpgm
