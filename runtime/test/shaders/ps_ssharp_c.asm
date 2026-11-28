; The S# mip fields on depth-compare samples, image_sample_c_lz and image_sample_c (--explicit-lod-bias's _c forms).
; Same descriptor path and target as ps_ssharp.asm: s0:s1 is the console VA of a T# (8 dwords) with the S# (4 dwords)
; 32 bytes after it; the S# carries DEPTH_COMPARE_FUNC = LESS_EQUAL.
; Run: vk_sampler_test --dref vs_ids.spv ps_ssharp_c.spv out.raw <S# words> [<base level> <last level>]
;   (check.py: SAMPLER_CASES through sampler_test(..., "ps_ssharp_c"), with the compare function added). The D32 texture
;   holds 0.25 in level 0 and 0.75 in level 1, and the reference is 0.5: a compare reads 0 on level 0 and 1 on level 1.
;   image_sample_c_lz is at LOD 0 and image_sample_c at LOD 1, as in ps_ssharp. -> mrt0 = (c_lz, c_lz, c, c).
s_load_dwordx8 s[8:15], s[0:1], 0x0
s_load_dwordx4 s[16:19], s[0:1], 0x20
s_waitcnt lgkmcnt(0)
v_interp_p1_f32_e32 v2, v0, attr0.x
v_interp_p2_f32_e32 v2, v1, attr0.x
v_interp_p1_f32_e32 v3, v0, attr0.y
v_interp_p2_f32_e32 v3, v1, attr0.y
v_mov_b32_e32 v10, 0.5
v_mov_b32_e32 v11, v2
v_mov_b32_e32 v12, v3
image_sample_c v6, v[10:12], s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D
image_sample_c_lz v4, v[10:12], s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D
s_waitcnt vmcnt(0)
exp mrt0 v4, v4, v6, v6 done vm
s_endpgm
