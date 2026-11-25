; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; The saveexec / wrexec kinds #1657 added (xor, andn1, orn1, orn2, nand, nor, xnor, andn1_wrexec), run on a GPU.
; One wave32 group. S0 = lanes 10..29 and each step's EXEC = lanes 0..23 come from compares (a lane mask held in an
; SGPR, as compiled code has it). Each step runs the op
; and records (s_bcnt1 << 8) | s_ff1 of the new EXEC (a whole-wave count and first lane, so the word does not depend
; on which lane stores it), then restores EXEC. Lane 0 stores the eight words. New EXEC per ISA (tools/cf_ref.py):
;   out[0] xor   S ^ E      lanes 0..9, 24..29   16 lanes, first 0  = 0x1000
;   out[1] andn1 ~S & E     lanes 0..9           10 lanes, first 0  = 0xA00
;   out[2] orn1  ~S | E     lanes 0..23, 30..31  26 lanes, first 0  = 0x1A00
;   out[3] orn2  S | ~E     lanes 10..31         22 lanes, first 10 = 0x160A
;   out[4] nand  ~(S & E)   all but 10..23       18 lanes, first 0  = 0x1200
;   out[5] nor   ~(S | E)   lanes 30..31          2 lanes, first 30 = 0x21E
;   out[6] xnor  ~(S ^ E)   lanes 10..23, 30..31 16 lanes, first 10 = 0x100A
;   out[7] s_andn1_wrexec_b32, also ~S & E                           = 0xA00
; Lanes EXEC had off (24..31) turn on where f gives 1, as on hardware.
; group: 32,1,1
; Run: vk_compute_test cs_w32_saveexec_ext.spv 1 dump -> out[0..7] = 4096 2560 6656 5642 4608 542 4106 2560
; mattr: +wavefrontsize32,-wavefrontsize64
v_cmp_le_u32_e64 s20, 10, v0
v_cmp_gt_u32_e64 s21, 30, v0
s_and_b32 s20, s20, s21
s_mov_b32 exec_lo, -1
v_cmpx_gt_u32_e32 24, v0
s_xor_saveexec_b32 s6, s20
s_bcnt1_i32_b32 s7, exec_lo
s_ff1_i32_b32 s8, exec_lo
s_lshl_b32 s7, s7, 8
s_or_b32 s10, s7, s8
s_mov_b32 exec_lo, -1
v_cmpx_gt_u32_e32 24, v0
s_andn1_saveexec_b32 s6, s20
s_bcnt1_i32_b32 s7, exec_lo
s_ff1_i32_b32 s8, exec_lo
s_lshl_b32 s7, s7, 8
s_or_b32 s11, s7, s8
s_mov_b32 exec_lo, -1
v_cmpx_gt_u32_e32 24, v0
s_orn1_saveexec_b32 s6, s20
s_bcnt1_i32_b32 s7, exec_lo
s_ff1_i32_b32 s8, exec_lo
s_lshl_b32 s7, s7, 8
s_or_b32 s12, s7, s8
s_mov_b32 exec_lo, -1
v_cmpx_gt_u32_e32 24, v0
s_orn2_saveexec_b32 s6, s20
s_bcnt1_i32_b32 s7, exec_lo
s_ff1_i32_b32 s8, exec_lo
s_lshl_b32 s7, s7, 8
s_or_b32 s13, s7, s8
s_mov_b32 exec_lo, -1
v_cmpx_gt_u32_e32 24, v0
s_nand_saveexec_b32 s6, s20
s_bcnt1_i32_b32 s7, exec_lo
s_ff1_i32_b32 s8, exec_lo
s_lshl_b32 s7, s7, 8
s_or_b32 s14, s7, s8
s_mov_b32 exec_lo, -1
v_cmpx_gt_u32_e32 24, v0
s_nor_saveexec_b32 s6, s20
s_bcnt1_i32_b32 s7, exec_lo
s_ff1_i32_b32 s8, exec_lo
s_lshl_b32 s7, s7, 8
s_or_b32 s15, s7, s8
s_mov_b32 exec_lo, -1
v_cmpx_gt_u32_e32 24, v0
s_xnor_saveexec_b32 s6, s20
s_bcnt1_i32_b32 s7, exec_lo
s_ff1_i32_b32 s8, exec_lo
s_lshl_b32 s7, s7, 8
s_or_b32 s16, s7, s8
s_mov_b32 exec_lo, -1
v_cmpx_gt_u32_e32 24, v0
s_andn1_wrexec_b32 s6, s20
s_bcnt1_i32_b32 s7, exec_lo
s_ff1_i32_b32 s8, exec_lo
s_lshl_b32 s7, s7, 8
s_or_b32 s17, s7, s8
s_mov_b32 exec_lo, -1
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v2, 0
v_mov_b32_e32 v1, s10
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32_e32 v2, 4
v_mov_b32_e32 v1, s11
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32_e32 v2, 8
v_mov_b32_e32 v1, s12
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32_e32 v2, 12
v_mov_b32_e32 v1, s13
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32_e32 v2, 16
v_mov_b32_e32 v1, s14
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32_e32 v2, 20
v_mov_b32_e32 v1, s15
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32_e32 v2, 24
v_mov_b32_e32 v1, s16
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32_e32 v2, 28
v_mov_b32_e32 v1, s17
buffer_store_dword v1, v2, s[0:3], 0 offen
s_endpgm
