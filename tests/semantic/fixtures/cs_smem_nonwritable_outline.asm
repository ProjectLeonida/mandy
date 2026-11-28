; Hand-written synthetic compute fixture for per-load NonWritable SMEM and outlined %sbuf_slow_xN.
; s[0:3] = output UAV V#, s[4:7] = disjoint CBV V#, s[8] = 32-bit descriptor-table pointer.
	v_cmpx_eq_u32_e32 0, v0
	s_mov_b32 s9, 0
	s_load_dwordx4 s[12:15], s[8:9], 0x0
	s_buffer_load_dwordx4 s[16:19], s[4:7], 0x0
	s_buffer_load_dwordx4 s[20:23], s[4:7], 0x10
	s_buffer_load_dwordx16 s[32:47], s[4:7], 0x0
	s_buffer_load_dwordx16 s[48:63], s[4:7], 0x10
	s_waitcnt lgkmcnt(0)
	v_mov_b32 v2, s12
	v_mov_b32 v3, s16
	v_mov_b32 v4, s20
	v_mov_b32 v5, s36
	buffer_store_dwordx4 v[2:5], off, s[0:3], 0
	v_mov_b32 v2, s48
	v_mov_b32 v3, s51
	v_mov_b32 v4, s62
	v_mov_b32 v5, s63
	buffer_store_dwordx4 v[2:5], off, s[0:3], 0 offset:16
	s_endpgm
