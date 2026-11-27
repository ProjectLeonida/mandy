; v_dot4_i32_i8 as SPV_KHR_integer_dot_product: the wrapping sum, both saturations under clamp, and op_sel.
; Operands and words are the gfx1036 measurements of tests/semantic/test_alu_dot4_i8.py.
; Expected MRT0 words: 8000fc03 7fffffff 80000000 0000003e.
	v_mov_b32 v0, 0x7f7f7f7f
	v_mov_b32 v1, 0x7fffffff
	v_dot4_i32_i8 v2, v0, v0, v1
	v_dot4_i32_i8 v3, v0, v0, v1 clamp
	v_mov_b32 v4, 0x80808080
	v_mov_b32 v5, 0x80000000
	v_dot4_i32_i8 v4, v4, v0, v5 clamp
	v_mov_b32 v6, 0x04030201
	v_mov_b32 v7, 0x08070605
	v_mov_b32 v8, 0
	v_dot4_i32_i8 v5, v6, v7, v8 op_sel:[0,1,0] op_sel_hi:[1,0,1]
	exp mrt0 v2, v3, v4, v5 done vm
	s_endpgm
