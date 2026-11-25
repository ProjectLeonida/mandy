; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; SHADER-930: s_brev_b64, s_flbit_i32_b64 / s_flbit_i32_i64 and s_bitset0_b64 edge cases on SGPR data pairs, each
; lowered on the two dwords with 32-bit ops (no Int64 capability). None of these ops writes SCC.
; X = 0xF0E1D2C3_89ABCDEF. Lane 0 of a wave32 group stores eight dwords:
;   out[0] = lo of s_brev_b64(X)                        = 0xC34B870F  (reverse(X.hi))
;   out[1] = hi of s_brev_b64(X)                        = 0xF7B3D591  (reverse(X.lo))
;   out[2] = s_flbit_i32_b64(0)                         = 0xFFFFFFFF  (no 1 bit)
;   out[3] = s_flbit_i32_b64(0x00000000_00000001)       = 63          (only bit 0 set)
;   out[4] = s_flbit_i32_i64(0xFFFFFFFF_FFFFFFFF)       = 0xFFFFFFFF  (every bit equals the sign)
;   out[5] = s_flbit_i32_i64(0xFFFFFFFF_7FFFFFFF)       = 32          (first bit unlike the sign: bit 31, in the low dword)
;   out[6] = hi of P                                     = 0x7FFFFFFE
;   out[7] = SCC ? lo of P : 0xDEADBEEF                  = 0x7FFFFFFF  (SCC set to 1 before the ops above, kept)
;   P = 0xFFFFFFFF_FFFFFFFF after s_bitset0_b64 with indices 32, 127 (127 & 63 = bit 63) and 31.
; group: 32,1,1
; Run: vk_compute_test cs_w32_salu64_bits.spv 1 dump -> out[0..7] = 3276506895 4155757969 4294967295 63 4294967295 32 2147483646 2147483647
; mattr: +wavefrontsize32,-wavefrontsize64
s_mov_b32 s6, 0x89abcdef
s_mov_b32 s7, 0xf0e1d2c3
s_cmp_eq_u32 s6, s6
s_brev_b64 s[8:9], s[6:7]
s_mov_b32 s6, 0
s_mov_b32 s7, 0
s_flbit_i32_b64 s10, s[6:7]
s_mov_b32 s6, 1
s_flbit_i32_b64 s11, s[6:7]
s_mov_b32 s6, -1
s_mov_b32 s7, -1
s_flbit_i32_i64 s12, s[6:7]
s_mov_b32 s6, 0x7fffffff
s_flbit_i32_i64 s13, s[6:7]
s_mov_b32 s14, -1
s_mov_b32 s15, -1
s_mov_b32 s16, 32
s_bitset0_b64 s[14:15], s16
s_mov_b32 s16, 0x7f
s_bitset0_b64 s[14:15], s16
s_bitset0_b64 s[14:15], 31
s_cselect_b32 s17, s14, 0xdeadbeef
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
v_mov_b32_e32 v20, s11
v_mov_b32_e32 v21, 12
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s12
v_mov_b32_e32 v21, 16
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s13
v_mov_b32_e32 v21, 20
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s15
v_mov_b32_e32 v21, 24
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s17
v_mov_b32_e32 v21, 28
buffer_store_dword v20, v21, s[0:3], 0 offen
s_endpgm
