; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; The four 64-bit-element GLOBAL atomics (#226): global_atomic_{and,or,xor,swap}_x2 on 8-byte-aligned elements.
; Lane 0 of a wave32 group seeds four 64-bit elements through the output V# (base 0), runs one atomic on each
; through a global pointer (s[20:21] = 0, vaddr v1 = 0, so the address is the instruction offset), waits, reads the
; elements back and stores them:
;   0x40 and_x2  {0xF0F0F0F0, 0xFFFFFFFF} & {0xFF00FF00, 0x0F0F0F0F} = {0xF000F000, 0x0F0F0F0F}
;   0x48 or_x2   {0x0F0F0F0F, 0x00000000} | {0xFF00FF00, 0xFFFFFFFF} = {0xFF0FFF0F, 0xFFFFFFFF}
;   0x50 xor_x2  {0xAAAA5555, 0x12345678} ^ {0xFFFF0000, 0xFFFFFFFF} = {0x55555555, 0xEDCBA987}
;   0x58 swap_x2 {0x11112222, 0x33334444} <- {0xAAAABBBB, 0xCCCCDDDD} = {0xAAAABBBB, 0xCCCCDDDD}
; Each element is two dwords, low word first, as one 32-bit atomic per dword: a bitwise op or an exchange leaves
; the same memory that way. out[0..1] = 0x40, out[2..3] = 0x48, out[4..5] = 0x50, out[6..7] = 0x58 (low, high).
; group: 32,1,1
; Run: VKC_SUBGROUP_SIZE=32 vk_compute_test cs_global_atomic32_x2.spv 1 dump -> out[0..7] = 4026593280 252645135 4279238415 4294967295 1431655765 3989547399 2863315899 3435978205
; mattr: +wavefrontsize32,-wavefrontsize64
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v11, 0
v_mov_b32_e32 v10, 0xf0f0f0f0
buffer_store_dword v10, v11, s[0:3], 0 offen offset:0x40
v_mov_b32_e32 v10, 0xffffffff
buffer_store_dword v10, v11, s[0:3], 0 offen offset:0x44
v_mov_b32_e32 v10, 0x0f0f0f0f
buffer_store_dword v10, v11, s[0:3], 0 offen offset:0x48
v_mov_b32_e32 v10, 0
buffer_store_dword v10, v11, s[0:3], 0 offen offset:0x4c
v_mov_b32_e32 v10, 0xaaaa5555
buffer_store_dword v10, v11, s[0:3], 0 offen offset:0x50
v_mov_b32_e32 v10, 0x12345678
buffer_store_dword v10, v11, s[0:3], 0 offen offset:0x54
v_mov_b32_e32 v10, 0x11112222
buffer_store_dword v10, v11, s[0:3], 0 offen offset:0x58
v_mov_b32_e32 v10, 0x33334444
buffer_store_dword v10, v11, s[0:3], 0 offen offset:0x5c
s_waitcnt vmcnt(0)
s_waitcnt_vscnt null, 0x0
s_mov_b32 s20, 0
s_mov_b32 s21, 0
v_mov_b32 v1, 0
v_mov_b32 v2, 0xff00ff00
v_mov_b32 v3, 0x0f0f0f0f
global_atomic_and_x2 v1, v[2:3], s[20:21] offset:0x40
v_mov_b32 v2, 0xff00ff00
v_mov_b32 v3, 0xffffffff
global_atomic_or_x2 v1, v[2:3], s[20:21] offset:0x48
v_mov_b32 v2, 0xffff0000
v_mov_b32 v3, 0xffffffff
global_atomic_xor_x2 v1, v[2:3], s[20:21] offset:0x50
v_mov_b32 v2, 0xaaaabbbb
v_mov_b32 v3, 0xccccdddd
global_atomic_swap_x2 v1, v[2:3], s[20:21] offset:0x58
s_waitcnt vmcnt(0)
s_waitcnt_vscnt null, 0x0
buffer_load_dword v4, v11, s[0:3], 0 offen offset:0x40
buffer_load_dword v5, v11, s[0:3], 0 offen offset:0x44
buffer_load_dword v6, v11, s[0:3], 0 offen offset:0x48
buffer_load_dword v7, v11, s[0:3], 0 offen offset:0x4c
buffer_load_dword v8, v11, s[0:3], 0 offen offset:0x50
buffer_load_dword v9, v11, s[0:3], 0 offen offset:0x54
buffer_load_dword v10, v11, s[0:3], 0 offen offset:0x58
buffer_load_dword v12, v11, s[0:3], 0 offen offset:0x5c
s_waitcnt vmcnt(0)
buffer_store_dword v4, v11, s[0:3], 0 offen
buffer_store_dword v5, v11, s[0:3], 0 offen offset:4
buffer_store_dword v6, v11, s[0:3], 0 offen offset:8
buffer_store_dword v7, v11, s[0:3], 0 offen offset:12
buffer_store_dword v8, v11, s[0:3], 0 offen offset:16
buffer_store_dword v9, v11, s[0:3], 0 offen offset:20
buffer_store_dword v10, v11, s[0:3], 0 offen offset:24
buffer_store_dword v12, v11, s[0:3], 0 offen offset:28
s_endpgm
