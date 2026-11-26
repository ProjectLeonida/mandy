; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; EXEC restored through an SGPR copy keeps the narrowing's identity (mask-regs EXEC tracking, #1397 review):
; s7 = 0xffffffff comes from v_readfirstlane_b32, so the translator can't fold it; s_mov_b32 exec_lo, s7 is an
; EXEC write the analysis can only name as an unknown narrowing. s8 = the odd-lane mask is written under it,
; s4 = exec_lo / exec_lo = s4 round-trips EXEC through an SGPR unchanged, and the mask-word read must still
; gather — lanes return the wave's mask word, not this lane's 0 / ~0 bit, exactly as s_mov_b32 exec_lo, -1 gives.
; Before the fix the copy lost the narrowing's tag, the restore minted a fresh one, mask_exec_sigs coverage
; failed and every lane stored its own bit (0 or ~0) instead of 0xAAAAAAAA.
; Per lane (stride 16 bytes): out[0] = s8's mask word after the SGPR round-trip (0xAAAAAAAA),
; out[1] = exec_lo's word (0xFFFFFFFF: EXEC really is all-ones), out[2] = the same mask word after the
; literal -1 restore, out[3] = the lane id.
; Lane 0 stores words 0..3, lane 1 words 4..7 (checked by the dump's first eight words).
; group: 32,1,1
; Run: VKC_SUBGROUP_SIZE=32 vk_compute_test cs_exec_copy.spv 1 dump -> out[0..7] = aaaaaaaa ffffffff aaaaaaaa 0 aaaaaaaa ffffffff aaaaaaaa 1
; mattr: +wavefrontsize32,-wavefrontsize64
v_mov_b32_e32 v7, -1
v_readfirstlane_b32 s7, v7
s_mov_b32 exec_lo, s7
v_and_b32 v1, 1, v0
v_cmp_ne_u32_e64 s8, v1, 0
s_mov_b32 s4, exec_lo
s_mov_b32 exec_lo, s4
v_mov_b32_e32 v2, s8
v_mov_b32_e32 v4, exec_lo
s_mov_b32 exec_lo, -1
v_mov_b32_e32 v5, s8
v_lshlrev_b32_e32 v3, 4, v0
v_mov_b32_e32 v20, v2
v_mov_b32_e32 v21, v3
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, v4
v_add_nc_u32_e32 v21, 4, v3
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, v5
v_add_nc_u32_e32 v21, 8, v3
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, v0
v_add_nc_u32_e32 v21, 12, v3
buffer_store_dword v20, v21, s[0:3], 0 offen
s_endpgm
