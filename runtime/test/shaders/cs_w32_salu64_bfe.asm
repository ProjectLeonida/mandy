; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; SHADER-930: s_bfe_u64 / s_bfe_i64 edge cases, lowered on the two dwords with 32-bit ops (no Int64 capability).
; X = 0xF0E1D2C3_89ABCDEF in s[6:7]. Control word S1: offset = S1[5:0], width = S1[22:16]; a field past bit 63 stops
; there (width = min(width, 64 - offset)), s_bfe_i64 sign-extends from the top bit kept. SCC = D != 0.
; Lane 0 of a wave32 group stores eight dwords:
;   out[0] = lo of s_bfe_u64(X, o0 w64)          = 0x89ABCDEF  (width exactly 64: both dwords whole)
;   out[1] = hi of s_bfe_u64(X, o0 w64)          = 0xF0E1D2C3
;   out[2] = lo of s_bfe_u64(X, o33 w32)         = 0x7870E961  (offset past 32, width cut to 31: X.hi >> 1)
;   out[3] = hi of s_bfe_i64(X, o1 w127)         = 0xF870E961  (width cut to 63: X >> 1 arithmetic, sign = bit 63)
;   out[4] = lo of s_bfe_i64(X, o1 w127)         = 0xC4D5E6F7  (X.lo >> 1 with bit 32 of X shifted in)
;   out[5] = hi of s_bfe_i64(X, o8 w32)          = 0xFFFFFFFF  (32-bit field 0xC389ABCD: its sign fills the high dword)
;   out[6] = hi of s_bfe_u64(X, o4 w36)          = 0x0000000C  (bits 39..36 of X)
;   out[7] = 0x5CC00000 | SCC bits               = 0x5CC0000A
;            bit 0: s_bfe_u64(X, ctl 0xff80ffc5)   = 0, width 0 (S1[31:23] and S1[15:6] ignored)    -> SCC 0
;            bit 1: s_bfe_u64(X, o0 w64)           = X                                              -> SCC 1
;            bit 2: s_bfe_u64(0x00000000_FFFFFFFF, o32 w32) = 0, a non-zero source                 -> SCC 0
;            bit 3: s_bfe_u64(0x80000000_00000000, o0 w64): only the high dword is non-zero         -> SCC 1
;            Each SCC is preset to the opposite value first, so a kept SCC fails.
; group: 32,1,1
; Run: vk_compute_test cs_w32_salu64_bfe.spv 1 dump -> out[0..7] = 2309737967 4041331395 2020665697 4168149345 3302352631 4294967295 12 1556086794
; mattr: +wavefrontsize32,-wavefrontsize64
s_mov_b32 s6, 0x89abcdef
s_mov_b32 s7, 0xf0e1d2c3
s_mov_b32 s30, 0x5cc00000
s_mov_b32 s5, 0x400000
s_bfe_u64 s[8:9], s[6:7], s5
s_mov_b32 s5, 0x200021
s_bfe_u64 s[10:11], s[6:7], s5
s_mov_b32 s5, 0x7f0001
s_bfe_i64 s[12:13], s[6:7], s5
s_mov_b32 s5, 0x200008
s_bfe_i64 s[14:15], s[6:7], s5
s_mov_b32 s5, 0x240004
s_bfe_u64 s[16:17], s[6:7], s5
s_cmp_eq_u32 s5, s5
s_mov_b32 s5, 0xff80ffc5
s_bfe_u64 s[18:19], s[6:7], s5
s_cselect_b32 s31, 1, 0
s_or_b32 s30, s30, s31
s_cmp_lg_u32 s5, s5
s_mov_b32 s5, 0x400000
s_bfe_u64 s[18:19], s[6:7], s5
s_cselect_b32 s31, 2, 0
s_or_b32 s30, s30, s31
s_mov_b32 s20, -1
s_mov_b32 s21, 0
s_cmp_eq_u32 s5, s5
s_mov_b32 s5, 0x200020
s_bfe_u64 s[18:19], s[20:21], s5
s_cselect_b32 s31, 4, 0
s_or_b32 s30, s30, s31
s_mov_b32 s20, 0
s_mov_b32 s21, 0x80000000
s_cmp_lg_u32 s5, s5
s_mov_b32 s5, 0x400000
s_bfe_u64 s[18:19], s[20:21], s5
s_cselect_b32 s31, 8, 0
s_or_b32 s30, s30, s31
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v20, s8
v_mov_b32_e32 v21, 0
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s9
v_mov_b32_e32 v21, 4
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s10
v_mov_b32_e32 v21, 8
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s13
v_mov_b32_e32 v21, 12
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s12
v_mov_b32_e32 v21, 16
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s15
v_mov_b32_e32 v21, 20
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s17
v_mov_b32_e32 v21, 24
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s30
v_mov_b32_e32 v21, 28
buffer_store_dword v20, v21, s[0:3], 0 offen
s_endpgm
