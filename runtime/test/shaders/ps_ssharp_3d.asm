; The S# mip fields on a 3D sample, as ps_ssharp.asm does them on 2D (--explicit-lod-bias on images_3d, binding 6).
; Same descriptor path and target: s0:s1 is the console VA of a T# (8 dwords) with the S# (4 dwords) 32 bytes after it.
; Run: vk_sampler_test --3d vs_ids.spv ps_ssharp_3d.spv out.raw <S# words> [<base level> <last level>]
;   (check.py: SAMPLER_CASES through sampler_test(..., "ps_ssharp_3d")). The volume is 128x128x2 then 64x64x1 with every
;   slice the same, sampled at (u, v, 0.25): the third coordinate doesn't vary across the target, so image_sample sees
;   LOD 1 and image_sample_lz LOD 0, as in ps_ssharp. -> mrt0 = (lz.r, lz.g, implicit.r, implicit.g).
s_load_dwordx8 s[8:15], s[0:1], 0x0
s_load_dwordx4 s[16:19], s[0:1], 0x20
s_waitcnt lgkmcnt(0)
v_interp_p1_f32_e32 v2, v0, attr0.x
v_interp_p2_f32_e32 v2, v1, attr0.x
v_interp_p1_f32_e32 v3, v0, attr0.y
v_interp_p2_f32_e32 v3, v1, attr0.y
v_mov_b32_e32 v10, v2
v_mov_b32_e32 v11, v3
v_mov_b32_e32 v12, 0.25
image_sample v[6:7], v[10:12], s[8:15], s[16:19] dmask:0x3 dim:SQ_RSRC_IMG_3D
image_sample_lz v[4:5], v[10:12], s[8:15], s[16:19] dmask:0x3 dim:SQ_RSRC_IMG_3D
s_waitcnt vmcnt(0)
exp mrt0 v4, v5, v6, v7 done vm
s_endpgm
