; The S# mip fields on a cube sample, as ps_ssharp.asm does them on 2D (--explicit-lod-bias on images_arr, binding 3).
; Same descriptor path and target: s0:s1 is the console VA of a T# (8 dwords) with the S# (4 dwords) 32 bytes after it.
; Run: vk_sampler_test --cube vs_ids.spv ps_ssharp_cube.spv out.raw <S# words> [<base level> <last level>]
;   (check.py: SAMPLER_CASES through sampler_test(..., "ps_ssharp_cube")). The cube face coordinates are the way the
;   hardware takes them, s, t = 1 + (u, v) on face 0 (ps_cube_sample.asm), so face 0 is read exactly where ps_ssharp
;   reads the 2D texture: image_sample_lz at LOD 0 and image_sample at LOD 1. Every face holds the same levels.
;   -> mrt0 = (lz.r, lz.g, implicit.r, implicit.g), judged like ps_ssharp.
s_load_dwordx8 s[8:15], s[0:1], 0x0
s_load_dwordx4 s[16:19], s[0:1], 0x20
s_waitcnt lgkmcnt(0)
v_interp_p1_f32_e32 v2, v0, attr0.x
v_interp_p2_f32_e32 v2, v1, attr0.x
v_interp_p1_f32_e32 v3, v0, attr0.y
v_interp_p2_f32_e32 v3, v1, attr0.y
v_add_f32_e32 v10, 1.0, v2
v_add_f32_e32 v11, 1.0, v3
v_mov_b32_e32 v12, 0
image_sample v[6:7], v[10:12], s[8:15], s[16:19] dmask:0x3 dim:SQ_RSRC_IMG_CUBE
image_sample_lz v[4:5], v[10:12], s[8:15], s[16:19] dmask:0x3 dim:SQ_RSRC_IMG_CUBE
s_waitcnt vmcnt(0)
exp mrt0 v4, v5, v6, v7 done vm
s_endpgm
