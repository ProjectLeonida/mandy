; Hand-written synthetic fixture for hardware DenormFlushToZero 32 and fmulz lowering.
	v_mov_b32 v0, 0x00000001
	v_mov_b32 v1, 0x7fc00000
	v_mul_legacy_f32 v2, v0, v1
	v_mov_b32 v3, 0x80800000
	v_mov_b32 v4, 0x3f000000
	v_mul_legacy_f32 v5, v3, v4
	v_mov_b32 v6, 0x40000000
	v_mov_b32 v7, 0x40400000
	v_mad_f32 v8, v6, v7, v0
	v_mad_legacy_f32 v9, v0, v1, v6
	exp mrt0 v2, v5, v8, v9 done vm
	s_endpgm
