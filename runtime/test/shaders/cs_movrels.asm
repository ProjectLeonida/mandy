; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; s_movrels_b32 / s_movrels_b64 (RDNA2 ISA SOP1): D = SGPR[S0 + M0], the b64 form the pair at
; S0 + M0 and S0 + M0 + 1. M0 is used whole, not a field, and SCC is not written. Every read happens
; before the destination is written, so D may overlap the source.
; s10..s14 = 0xAA 0xBB 0xCC 0xDD 0xEE, lane 0 of a wave32 group.
; 1. m0 = 0: s_movrels_b32 s4, s10            -> s4 = s10 = 0xAA
; 2. m0 = 2: s_movrels_b32 s5, s10            -> s5 = s12 = 0xCC
; 3. m0 = 1: s_movrels_b64 s[6:7], s[10:11]   -> s6 = s11 = 0xBB, s7 = s12 = 0xCC
; 4. m0 = 1: s_movrels_b32 s10, s10           -> reads s11 = 0xBB before writing s10 = 0xBB
; 5. s_cmp_eq_u32 s4, 0xAA (SCC = 1), then m0 = 0: s_movrels_b32 s8, s10 -> s8 = 0xBB;
;    s_cselect_b32 s9, 1, 0 reads the SCC the movrels left alone -> s9 = 1
; Lane 0 stores: out[0..7] = s4 s5 s6 s7 s10 s11 s9 s14 = 170 204 187 204 187 187 1 238.
; s11 and s14 were never written (the b64 read does not change its source).
; group: 32,1,1
; Run: vk_compute_test cs_movrels.spv 1 dump -> out[0..7] = 170 204 187 204 187 187 1 238
; mattr: +wavefrontsize32,-wavefrontsize64
s_mov_b32 s10, 0xaa
s_mov_b32 s11, 0xbb
s_mov_b32 s12, 0xcc
s_mov_b32 s13, 0xdd
s_mov_b32 s14, 0xee
s_mov_b32 m0, 0
s_movrels_b32 s4, s10
s_mov_b32 m0, 2
s_movrels_b32 s5, s10
s_mov_b32 m0, 1
s_movrels_b64 s[6:7], s[10:11]
s_mov_b32 m0, 1
s_movrels_b32 s10, s10
s_cmp_eq_u32 s4, 0xaa
s_mov_b32 m0, 0
s_movrels_b32 s8, s10
s_cselect_b32 s9, 1, 0
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v1, s4
v_mov_b32_e32 v21, 0
buffer_store_dword v1, v21, s[0:3], 0 offen
v_mov_b32_e32 v1, s5
v_mov_b32_e32 v21, 4
buffer_store_dword v1, v21, s[0:3], 0 offen
v_mov_b32_e32 v1, s6
v_mov_b32_e32 v21, 8
buffer_store_dword v1, v21, s[0:3], 0 offen
v_mov_b32_e32 v1, s7
v_mov_b32_e32 v21, 12
buffer_store_dword v1, v21, s[0:3], 0 offen
v_mov_b32_e32 v1, s10
v_mov_b32_e32 v21, 16
buffer_store_dword v1, v21, s[0:3], 0 offen
v_mov_b32_e32 v1, s11
v_mov_b32_e32 v21, 20
buffer_store_dword v1, v21, s[0:3], 0 offen
v_mov_b32_e32 v1, s9
v_mov_b32_e32 v21, 24
buffer_store_dword v1, v21, s[0:3], 0 offen
v_mov_b32_e32 v1, s14
v_mov_b32_e32 v21, 28
buffer_store_dword v1, v21, s[0:3], 0 offen
s_endpgm
