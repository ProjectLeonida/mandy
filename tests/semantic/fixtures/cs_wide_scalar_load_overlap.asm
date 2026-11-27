; Wide scalar loads whose destination is their own V# (s_buffer_load_dwordx4 s[20:23], s[20:23]), as a pipeline-library
; depth-reduction program has it: the V# comes from memory through the pointer in s[8:9], an x2 and an x4 at +8 go
; through the V# in s[4:7]. All dwords are loaded before any destination SGPR is written, in range or not.
	v_cmpx_eq_u32_e32 0, v0
	s_mov_b32 s12, s8
	s_mov_b32 s13, 0
	s_load_dwordx4 s[20:23], s[12:13], 0x0
	s_waitcnt lgkmcnt(0)
	s_buffer_load_dwordx2 s[8:9], s[4:7], 0x0
	s_buffer_load_dwordx4 s[20:23], s[20:23], 0x0
	s_buffer_load_dwordx4 s[32:35], s[4:7], 0x8
	s_waitcnt lgkmcnt(0)
	v_mov_b32 v2, s8
	v_mov_b32 v3, s9
	v_mov_b32 v4, s20
	v_mov_b32 v5, s21
	buffer_store_dwordx4 v[2:5], off, s[0:3], 0
	v_mov_b32 v2, s22
	v_mov_b32 v3, s23
	v_mov_b32 v4, s32
	v_mov_b32 v5, s35
	buffer_store_dwordx4 v[2:5], off, s[0:3], 0 offset:16
	s_endpgm
