; v_dot4_u32_u8 and v_dot4c_u32_u8 as SPV_KHR_integer_dot_product: zero-extended bytes, the sum wrapping at 32 bits,
; unsigned saturation under clamp, and the accumulating form reading its destination (twice: 2 * 17027).
; 0x80ff0102 . 0x7f02ff03 = 128*127 + 255*2 + 1*255 + 2*3 = 17027; 4 * 255 * 255 = 260100.
; Expected MRT0 words: 00004288 0003f803 ffffffff 00008506.
	v_mov_b32 v0, 0x80ff0102
	v_mov_b32 v1, 0x7f02ff03
	v_mov_b32 v2, 5
	v_dot4_u32_u8 v3, v0, v1, v2
	v_mov_b32 v4, -1
	v_dot4_u32_u8 v5, v4, v4, v4
	v_dot4_u32_u8 v6, v4, v4, v4 clamp
	v_mov_b32 v7, 0
	v_dot4c_u32_u8 v7, v0, v1
	v_dot4c_u32_u8 v7, v0, v1
	exp mrt0 v3, v5, v6, v7 done vm
	s_endpgm
