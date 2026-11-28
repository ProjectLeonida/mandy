; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; s_movrels_b64 (#1598, merged with CPU-only checks; #1850's cs_w32_pk_rel_ext covers s_movrels_b32 only): D = the
; SGPR pair at S0 + M0 (RDNA2 ISA SOP1). s20..s27 hold W[k] = 0x5EED0000 + 0x111 * k. Lane 0 stores eight dwords:
;   out[0], out[1] = s_movrels_b64 s[10:11], s[20:21] with M0 = 1 (an odd pair): W[1], W[2]  = 0x5EED0111 0x5EED0222
;   out[2], out[3] = the same with M0 = 4 set from v_readfirstlane (not foldable): W[4], W[5]  = 0x5EED0444 0x5EED0555
;   out[4] = s_movrels_b32 s12, s22 with M0 = 3 from v_readfirstlane: s25 = W[5]               = 0x5EED0555
;   out[5], out[6] = s_movrels_b64 s[22:23], s[20:21] with M0 = 1: it reads the pair s21, s22, which overlaps the
;                    destination s22 (an aligned pair, as LLVM 22+ requires), before writing: W[1], W[2]
;                                                                                                = 0x5EED0111 0x5EED0222
;   out[7] = SCC set by s_cmp before an s_movrels and read after it (SCC is not written)        = 1
; group: 32,1,1
; Run: vk_compute_test cs_w32_movrels64.spv 1 dump -> out[0..7] = 1592590609 1592590882 1592591428 1592591701 1592591701 1592590609 1592590882 1
; mattr: +wavefrontsize32,-wavefrontsize64
s_mov_b32 s20, 0x5eed0000
s_mov_b32 s21, 0x5eed0111
s_mov_b32 s22, 0x5eed0222
s_mov_b32 s23, 0x5eed0333
s_mov_b32 s24, 0x5eed0444
s_mov_b32 s25, 0x5eed0555
s_mov_b32 s26, 0x5eed0666
s_mov_b32 s27, 0x5eed0777
s_mov_b32 m0, 1
s_movrels_b64 s[10:11], s[20:21]
v_mov_b32_e32 v1, 4
v_readfirstlane_b32 s5, v1
s_mov_b32 m0, s5
s_movrels_b64 s[14:15], s[20:21]
v_mov_b32_e32 v1, 3
v_readfirstlane_b32 s5, v1
s_mov_b32 m0, s5
s_cmp_eq_u32 s5, 3
s_movrels_b32 s12, s22
s_cselect_b32 s13, 1, 0
s_mov_b32 m0, 1
s_movrels_b64 s[22:23], s[20:21]
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v20, 0
v_mov_b32_e32 v21, s10
buffer_store_dword v21, v20, s[0:3], 0 offen
v_mov_b32_e32 v20, 4
v_mov_b32_e32 v21, s11
buffer_store_dword v21, v20, s[0:3], 0 offen
v_mov_b32_e32 v20, 8
v_mov_b32_e32 v21, s14
buffer_store_dword v21, v20, s[0:3], 0 offen
v_mov_b32_e32 v20, 12
v_mov_b32_e32 v21, s15
buffer_store_dword v21, v20, s[0:3], 0 offen
v_mov_b32_e32 v20, 16
v_mov_b32_e32 v21, s12
buffer_store_dword v21, v20, s[0:3], 0 offen
v_mov_b32_e32 v20, 20
v_mov_b32_e32 v21, s22
buffer_store_dword v21, v20, s[0:3], 0 offen
v_mov_b32_e32 v20, 24
v_mov_b32_e32 v21, s23
buffer_store_dword v21, v20, s[0:3], 0 offen
v_mov_b32_e32 v20, 28
v_mov_b32_e32 v21, s13
buffer_store_dword v21, v20, s[0:3], 0 offen
s_endpgm
