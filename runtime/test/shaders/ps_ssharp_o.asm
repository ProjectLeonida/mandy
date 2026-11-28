; The S# mip fields on the texel-offset forms under --texel-offset-fallback, where the offset is folded into the
; coordinate and an implicit sample becomes a Grad sample (the derivatives --explicit-lod-bias scales).
; Same descriptor path, texture and target as ps_ssharp.asm: s0:s1 is the console VA of a T# (8 dwords) with the S#
; (4 dwords) 32 bytes after it.
; Run: vk_sampler_test vs_ids.spv ps_ssharp_o.spv out.raw <S# words> [<base level> <last level>] with the shaders
;   translated under LEONIDA_TEXOFF_FALLBACK=1 (check.py: SAMPLER_CASES through sampler_test(..., "ps_ssharp_o"); it
;   is a SKIP otherwise, since vk_sampler_test doesn't enable VK_KHR_maintenance8 for a non-constant Offset).
;   image_sample_lz_o and image_sample_o with an offset dword of 0 (v10), so the texels are ps_ssharp's:
;   -> mrt0 = (lz.r, lz.g, implicit.r, implicit.g), judged like ps_ssharp.
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
image_sample_o v[6:7], v[10:12], s[8:15], s[16:19] dmask:0x3 dim:SQ_RSRC_IMG_2D
image_sample_lz_o v[4:5], v[10:12], s[8:15], s[16:19] dmask:0x3 dim:SQ_RSRC_IMG_2D
s_waitcnt vmcnt(0)
exp mrt0 v4, v5, v6, v7 done vm
s_endpgm
