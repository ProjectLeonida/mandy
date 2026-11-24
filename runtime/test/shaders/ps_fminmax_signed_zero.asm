; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; v_max_f32 / v_min_f32 with a -0 and a +0 operand, in both orders, inputs read from memory at run time (not inline
; constants, which the driver could fold). The V# in s[0:3] points at 0x1000 (num_records 16); the test seeds
; 0x1000..0x1004 with a, b = 0x80000000 0x00000000 (-0, +0).
; RDNA2 orders -0 below +0 (docs/design.md, Float min/max row: "-0 < +0", measured on an AMD Radeon RX 6800, gfx1030):
;   max_f32(-0, +0) = 0x00000000   max_f32(+0, -0) = 0x00000000
;   min_f32(-0, +0) = 0x80000000   min_f32(+0, -0) = 0x80000000
; GLSL.std.450 NMax/NMin alone are "y if x < y, otherwise x": -0 and +0 compare equal, so they give the first operand
; (max(-0, +0) = -0, min(+0, -0) = +0) and this fixture fails without the sign-bit select tools/rdna2spirv.py adds.
; Expected mrt0 bits: 0x00000000, 0x00000000, 0x80000000, 0x80000000
s_buffer_load_dwordx2 s[4:5], s[0:3], 0
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v1, s4
v_mov_b32_e32 v2, s5
v_max_f32_e32 v5, v1, v2
v_max_f32_e32 v6, v2, v1
v_min_f32_e32 v7, v1, v2
v_min_f32_e32 v8, v2, v1
exp mrt0 v5, v6, v7, v8 done vm
s_endpgm
