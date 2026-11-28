; The S# mip fields on image_sample_b, the implicit form with a bias in the address, which ps_ssharp doesn't run.
; Same descriptor path, texture and target as ps_ssharp.asm: s0:s1 is the console VA of a T# (8 dwords) with the S#
; (4 dwords) 32 bytes after it.
; Run: vk_sampler_test vs_ids.spv ps_ssharp_b.spv out.raw <S# words> [<base level> <last level>]  (check.py:
;   SAMPLER_CASES, through sampler_test(..., "ps_ssharp_b"))
;   image_sample_lz as in ps_ssharp, then image_sample_b with an address bias of 0 (v4): it sees LOD 1 like
;   ps_ssharp's implicit sample, so check.py judges (lz.r, lz.g, b.r, b.g) exactly like ps_ssharp's, and the S#'s
;   LOD_BIAS adds to the address bias.
s_load_dwordx8 s[8:15], s[0:1], 0x0
s_load_dwordx4 s[16:19], s[0:1], 0x20
s_waitcnt lgkmcnt(0)
v_interp_p1_f32_e32 v2, v0, attr0.x
v_interp_p2_f32_e32 v2, v1, attr0.x
v_interp_p1_f32_e32 v3, v0, attr0.y
v_interp_p2_f32_e32 v3, v1, attr0.y
v_mov_b32_e32 v10, 0
v_mov_b32_e32 v11, v2
v_mov_b32_e32 v12, v3
image_sample_b v[6:7], v[10:12], s[8:15], s[16:19] dmask:0x3 dim:SQ_RSRC_IMG_2D
image_sample_lz v[4:5], v[2:3], s[8:15], s[16:19] dmask:0x3 dim:SQ_RSRC_IMG_2D
s_waitcnt vmcnt(0)
exp mrt0 v4, v5, v6, v7 done vm
s_endpgm
