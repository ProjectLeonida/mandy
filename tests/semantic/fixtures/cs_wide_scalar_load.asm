; Wide scalar loads: s_buffer_load_dwordx16/x8/x4/x2 through a V# known only at run time (user SGPRs s[4:7]), and
; s_load_dwordx8 through a pointer (s[8:9]). s[0:3] is the output V#.
; With NUM_RECORDS = 0x48 bytes (18 dwords) the x16 at 0 and the x4 at 0x10 are in range (one wide load each), the x8
; at 0x40 and the x2 at 0x44 are partly out of range: their dwords 18 and up are 0 and are not read.
	v_cmpx_eq_u32_e32 0, v0
	v_mov_b32 v1, 0
	s_buffer_load_dwordx16 s[16:31], s[4:7], 0x0
	s_buffer_load_dwordx8 s[32:39], s[4:7], 0x40
	s_buffer_load_dwordx4 s[40:43], s[4:7], 0x10
	s_buffer_load_dwordx2 s[44:45], s[4:7], 0x44
	s_load_dwordx8 s[48:55], s[8:9], 0x0
	s_waitcnt lgkmcnt(0)
	v_mov_b32 v2, s16
	v_mov_b32 v3, s21
	v_mov_b32 v4, s26
	v_mov_b32 v5, s31
	buffer_store_dwordx4 v[2:5], off, s[0:3], 0
	v_mov_b32 v2, s32
	v_mov_b32 v3, s33
	v_mov_b32 v4, s34
	v_mov_b32 v5, s39
	buffer_store_dwordx4 v[2:5], off, s[0:3], 0 offset:16
	v_mov_b32 v2, s40
	v_mov_b32 v3, s43
	v_mov_b32 v4, s44
	v_mov_b32 v5, s45
	buffer_store_dwordx4 v[2:5], off, s[0:3], 0 offset:32
	v_mov_b32 v2, s48
	v_mov_b32 v3, s51
	v_mov_b32 v4, s52
	v_mov_b32 v5, s55
	buffer_store_dwordx4 v[2:5], off, s[0:3], 0 offset:48
	s_endpgm
