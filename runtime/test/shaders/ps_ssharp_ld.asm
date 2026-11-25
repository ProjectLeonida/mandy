; The S# mip fields on the two explicit forms ps_ssharp doesn't run: image_sample_l (an LOD in the address) and
; image_sample_d (gradients in the address). Same descriptor path, texture and target as ps_ssharp.asm: s0:s1 is the
; console VA of a T# (8 dwords) with the S# (4 dwords) 32 bytes after it.
; Run: vk_sampler_test vs_ids.spv ps_ssharp_ld.spv out.raw <S# words> [<base level> <last level>]  (check.py:
;   SAMPLER_CASES, through sampler_test(..., "ps_ssharp_ld"))
;   image_sample_l takes LOD 0 from v18. image_sample_d takes dU/dx = dV/dy = 1/64 and dU/dy = dV/dx = 0 (v8..v11, then U and V in v12, v13):
;   on the 128x128 level 0 that is two texels per pixel, LOD 1, which is what ps_ssharp's implicit sample sees on the
;   64x64 target. So check.py judges (l.r, l.g, d.r, d.g) exactly like ps_ssharp's (lz.r, lz.g, implicit.r,
;   implicit.g), and the S#'s LOD_BIAS, MIN_LOD, MAX_LOD and MIP_FILTER apply to both forms.
s_load_dwordx8 s[8:15], s[0:1], 0x0
s_load_dwordx4 s[16:19], s[0:1], 0x20
s_waitcnt lgkmcnt(0)
v_interp_p1_f32_e32 v2, v0, attr0.x
v_interp_p2_f32_e32 v2, v1, attr0.x
v_interp_p1_f32_e32 v3, v0, attr0.y
v_interp_p2_f32_e32 v3, v1, attr0.y
v_mov_b32_e32 v16, v2
v_mov_b32_e32 v17, v3
v_mov_b32_e32 v18, 0
v_mov_b32_e32 v8, 0x3c800000
v_mov_b32_e32 v9, 0
v_mov_b32_e32 v10, 0
v_mov_b32_e32 v11, 0x3c800000
v_mov_b32_e32 v12, v2
v_mov_b32_e32 v13, v3
image_sample_l v[4:5], v[16:18], s[8:15], s[16:19] dmask:0x3 dim:SQ_RSRC_IMG_2D
image_sample_d v[6:7], v[8:13], s[8:15], s[16:19] dmask:0x3 dim:SQ_RSRC_IMG_2D
s_waitcnt vmcnt(0)
exp mrt0 v4, v5, v6, v7 done vm
s_endpgm
