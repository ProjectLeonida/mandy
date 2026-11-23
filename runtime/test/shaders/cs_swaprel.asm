; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; v_swaprel_b32 (GFX10 VOP1 opcode 104): swap VGPR[D + M0[25:16]] and VGPR[S0 + M0[9:0]], the two
; 10-bit offsets v_movrelsd_2_b32 uses. M0[15:10] and M0[31:26] are ignored, both registers are read before
; either is written, and a swap of one register with itself keeps its value.
; v1..v8 = 11, 22, 33, 44, 55, 66, 77, 88, lane 0 of a wave32 group.
; 1. m0 = (4 << 16) | 2: v_swaprel_b32 v1, v2 -> swap v5 (55) and v4 (44); v1 and v2 keep their values.
; 2. m0 = 0xFC03FC02 (dst 3, src 2, ignored bits set): v_swaprel_b32 v1, v5 -> swap v4 (now 55) and v7 (77).
;    v5 is the S0 base and keeps 44; v6 (v5 + 1) keeps 66.
; 3. m0 = (5 << 16) | 5: v_swaprel_b32 v1, v1 -> v6 with itself, keeps 66.
; Lane 0 stores: out[0..7] = v1 v2 v4 v5 (after 1) v4 v7 (after 2) v6 v8 = 11 22 55 44 77 55 66 88.
; group: 32,1,1
; Run: vk_compute_test cs_swaprel.spv 1 dump -> out[0..7] = 11 22 55 44 77 55 66 88
; mattr: +wavefrontsize32,-wavefrontsize64
v_mov_b32_e32 v1, 11
v_mov_b32_e32 v2, 22
v_mov_b32_e32 v3, 33
v_mov_b32_e32 v4, 44
v_mov_b32_e32 v5, 55
v_mov_b32_e32 v6, 66
v_mov_b32_e32 v7, 77
v_mov_b32_e32 v8, 88
v_cmpx_eq_u32_e32 0, v0
s_mov_b32 m0, 0x00040002
v_swaprel_b32 v1, v2
v_mov_b32_e32 v21, 0
buffer_store_dword v1, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 4
buffer_store_dword v2, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 8
buffer_store_dword v4, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 12
buffer_store_dword v5, v21, s[0:3], 0 offen
s_mov_b32 m0, 0xfc03fc02
v_swaprel_b32 v1, v5
v_mov_b32_e32 v21, 16
buffer_store_dword v4, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 20
buffer_store_dword v7, v21, s[0:3], 0 offen
s_mov_b32 m0, 0x00050005
v_swaprel_b32 v1, v1
v_mov_b32_e32 v21, 24
buffer_store_dword v6, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 28
buffer_store_dword v8, v21, s[0:3], 0 offen
s_endpgm
