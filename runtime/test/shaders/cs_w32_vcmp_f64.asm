; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; All 16 binary64 compare predicates in both forms (#381): v_cmp_*_f64 (VCC destination, spelled out as a
; disassembly does) and v_cmpx_*_f64 (EXEC narrowing), 32 instructions packed into eight dwords, four
; predicates per dword.
; Lane 0 of a wave32 group stores out[0..3] from the v_cmp half and out[4..7] from the v_cmpx half. Predicate i of
; each group of four lands in bit i of the group's dword (f, lt, eq, le | gt, lg, ge, o | u, nge, nlg, ngt |
; nle, neq, nlt, tru).
;   v_cmp half:  v[1:2] = -0.0 (0x80000000_00000000), v[3:4] = +0.0. Equal and ordered, so eq/le/ge/o/tru are 1
;                and f/lt/gt/lg/u/nge/neq/nle are 0; nlg, ngt and nlt are 1.
;                dwords 0xC 0xC 0xC 0xC  (-0.0 compares equal to +0.0, not below it)
;   v_cmpx half: v[5:6] = qNaN (0x7FF80000_00000000), v[7:8] = 1.0. Every ordered predicate is 0, so EXEC goes to
;                0 on f/lt/eq/le/gt/lg/ge/o, and u/nge/nlg/ngt/nle/neq/nlt/tru are 1.
;                dwords 0 0 0xF 0xF
; v_cmpx reads EXEC: v9 is preloaded 0, only lanes the compare left on run the trailing v_mov_b32 v9, 1, and EXEC
; is restored before the next predicate.
; group: 32,1,1
; Run: VKC_SUBGROUP_SIZE=32 vk_compute_test cs_w32_vcmp_f64.spv 1 dump -> out[0..7] = 12 12 12 12 0 0 15 15
; mattr: +wavefrontsize32,-wavefrontsize64
v_mov_b32_e32 v1, 0
v_mov_b32_e32 v2, 0x80000000
v_mov_b32_e32 v3, 0
v_mov_b32_e32 v4, 0
v_mov_b32_e32 v5, 0
v_mov_b32_e32 v6, 0x7ff80000
v_mov_b32_e32 v7, 0
v_mov_b32_e32 v8, 0x3ff00000
v_mov_b32_e32 v20, 0
v_mov_b32_e32 v21, 0
v_mov_b32_e32 v22, 0
v_mov_b32_e32 v23, 0
v_mov_b32_e32 v24, 0
v_mov_b32_e32 v25, 0
v_mov_b32_e32 v26, 0
v_mov_b32_e32 v27, 0
v_cmp_f_f64 vcc_lo, v[1:2], v[3:4]
v_cndmask_b32_e64 v9, 0, 1, vcc_lo
v_lshl_or_b32 v20, v9, 0, v20
v_cmp_lt_f64 vcc_lo, v[1:2], v[3:4]
v_cndmask_b32_e64 v9, 0, 1, vcc_lo
v_lshl_or_b32 v20, v9, 1, v20
v_cmp_eq_f64 vcc_lo, v[1:2], v[3:4]
v_cndmask_b32_e64 v9, 0, 1, vcc_lo
v_lshl_or_b32 v20, v9, 2, v20
v_cmp_le_f64 vcc_lo, v[1:2], v[3:4]
v_cndmask_b32_e64 v9, 0, 1, vcc_lo
v_lshl_or_b32 v20, v9, 3, v20
v_cmp_gt_f64 vcc_lo, v[1:2], v[3:4]
v_cndmask_b32_e64 v9, 0, 1, vcc_lo
v_lshl_or_b32 v21, v9, 0, v21
v_cmp_lg_f64 vcc_lo, v[1:2], v[3:4]
v_cndmask_b32_e64 v9, 0, 1, vcc_lo
v_lshl_or_b32 v21, v9, 1, v21
v_cmp_ge_f64 vcc_lo, v[1:2], v[3:4]
v_cndmask_b32_e64 v9, 0, 1, vcc_lo
v_lshl_or_b32 v21, v9, 2, v21
v_cmp_o_f64 vcc_lo, v[1:2], v[3:4]
v_cndmask_b32_e64 v9, 0, 1, vcc_lo
v_lshl_or_b32 v21, v9, 3, v21
v_cmp_u_f64 vcc_lo, v[1:2], v[3:4]
v_cndmask_b32_e64 v9, 0, 1, vcc_lo
v_lshl_or_b32 v22, v9, 0, v22
v_cmp_nge_f64 vcc_lo, v[1:2], v[3:4]
v_cndmask_b32_e64 v9, 0, 1, vcc_lo
v_lshl_or_b32 v22, v9, 1, v22
v_cmp_nlg_f64 vcc_lo, v[1:2], v[3:4]
v_cndmask_b32_e64 v9, 0, 1, vcc_lo
v_lshl_or_b32 v22, v9, 2, v22
v_cmp_ngt_f64 vcc_lo, v[1:2], v[3:4]
v_cndmask_b32_e64 v9, 0, 1, vcc_lo
v_lshl_or_b32 v22, v9, 3, v22
v_cmp_nle_f64 vcc_lo, v[1:2], v[3:4]
v_cndmask_b32_e64 v9, 0, 1, vcc_lo
v_lshl_or_b32 v23, v9, 0, v23
v_cmp_neq_f64 vcc_lo, v[1:2], v[3:4]
v_cndmask_b32_e64 v9, 0, 1, vcc_lo
v_lshl_or_b32 v23, v9, 1, v23
v_cmp_nlt_f64 vcc_lo, v[1:2], v[3:4]
v_cndmask_b32_e64 v9, 0, 1, vcc_lo
v_lshl_or_b32 v23, v9, 2, v23
v_cmp_tru_f64 vcc_lo, v[1:2], v[3:4]
v_cndmask_b32_e64 v9, 0, 1, vcc_lo
v_lshl_or_b32 v23, v9, 3, v23
v_mov_b32_e32 v9, 0
v_cmpx_f_f64 v[5:6], v[7:8]
v_mov_b32_e32 v9, 1
s_mov_b32 exec_lo, -1
v_lshl_or_b32 v24, v9, 0, v24
v_mov_b32_e32 v9, 0
v_cmpx_lt_f64 v[5:6], v[7:8]
v_mov_b32_e32 v9, 1
s_mov_b32 exec_lo, -1
v_lshl_or_b32 v24, v9, 1, v24
v_mov_b32_e32 v9, 0
v_cmpx_eq_f64 v[5:6], v[7:8]
v_mov_b32_e32 v9, 1
s_mov_b32 exec_lo, -1
v_lshl_or_b32 v24, v9, 2, v24
v_mov_b32_e32 v9, 0
v_cmpx_le_f64 v[5:6], v[7:8]
v_mov_b32_e32 v9, 1
s_mov_b32 exec_lo, -1
v_lshl_or_b32 v24, v9, 3, v24
v_mov_b32_e32 v9, 0
v_cmpx_gt_f64 v[5:6], v[7:8]
v_mov_b32_e32 v9, 1
s_mov_b32 exec_lo, -1
v_lshl_or_b32 v25, v9, 0, v25
v_mov_b32_e32 v9, 0
v_cmpx_lg_f64 v[5:6], v[7:8]
v_mov_b32_e32 v9, 1
s_mov_b32 exec_lo, -1
v_lshl_or_b32 v25, v9, 1, v25
v_mov_b32_e32 v9, 0
v_cmpx_ge_f64 v[5:6], v[7:8]
v_mov_b32_e32 v9, 1
s_mov_b32 exec_lo, -1
v_lshl_or_b32 v25, v9, 2, v25
v_mov_b32_e32 v9, 0
v_cmpx_o_f64 v[5:6], v[7:8]
v_mov_b32_e32 v9, 1
s_mov_b32 exec_lo, -1
v_lshl_or_b32 v25, v9, 3, v25
v_mov_b32_e32 v9, 0
v_cmpx_u_f64 v[5:6], v[7:8]
v_mov_b32_e32 v9, 1
s_mov_b32 exec_lo, -1
v_lshl_or_b32 v26, v9, 0, v26
v_mov_b32_e32 v9, 0
v_cmpx_nge_f64 v[5:6], v[7:8]
v_mov_b32_e32 v9, 1
s_mov_b32 exec_lo, -1
v_lshl_or_b32 v26, v9, 1, v26
v_mov_b32_e32 v9, 0
v_cmpx_nlg_f64 v[5:6], v[7:8]
v_mov_b32_e32 v9, 1
s_mov_b32 exec_lo, -1
v_lshl_or_b32 v26, v9, 2, v26
v_mov_b32_e32 v9, 0
v_cmpx_ngt_f64 v[5:6], v[7:8]
v_mov_b32_e32 v9, 1
s_mov_b32 exec_lo, -1
v_lshl_or_b32 v26, v9, 3, v26
v_mov_b32_e32 v9, 0
v_cmpx_nle_f64 v[5:6], v[7:8]
v_mov_b32_e32 v9, 1
s_mov_b32 exec_lo, -1
v_lshl_or_b32 v27, v9, 0, v27
v_mov_b32_e32 v9, 0
v_cmpx_neq_f64 v[5:6], v[7:8]
v_mov_b32_e32 v9, 1
s_mov_b32 exec_lo, -1
v_lshl_or_b32 v27, v9, 1, v27
v_mov_b32_e32 v9, 0
v_cmpx_nlt_f64 v[5:6], v[7:8]
v_mov_b32_e32 v9, 1
s_mov_b32 exec_lo, -1
v_lshl_or_b32 v27, v9, 2, v27
v_mov_b32_e32 v9, 0
v_cmpx_tru_f64 v[5:6], v[7:8]
v_mov_b32_e32 v9, 1
s_mov_b32 exec_lo, -1
v_lshl_or_b32 v27, v9, 3, v27
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v10, 0
buffer_store_dword v20, v10, s[0:3], 0 offen
buffer_store_dword v21, v10, s[0:3], 0 offen offset:4
buffer_store_dword v22, v10, s[0:3], 0 offen offset:8
buffer_store_dword v23, v10, s[0:3], 0 offen offset:12
buffer_store_dword v24, v10, s[0:3], 0 offen offset:16
buffer_store_dword v25, v10, s[0:3], 0 offen offset:20
buffer_store_dword v26, v10, s[0:3], 0 offen offset:24
buffer_store_dword v27, v10, s[0:3], 0 offen offset:28
s_endpgm
