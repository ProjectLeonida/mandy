; SHADER-820 tier C: a vote whose mask register is rewritten before its branch (#4620; fixture by @AwkGrep256).
; Only the pixel (0, 0) (x = floor(u * 64) = 0, y = floor(v * 64) = 0 on the 64x64 target of vs_ids) is in the voted
; mask s[20:21] = EXEC & VCC; s_cmp_lg_u64 votes on it, then v_cmp_le_u32_e64 rewrites s[20:21] (all lanes: 0 <= v4)
; without touching SCC. s_cbranch_scc0 still decides on the old mask, but the run side narrows EXEC to the new one: on
; a wave64 the wave with pixel (0, 0) runs it in every lane and writes v6 = 1; on two 32-lane subgroups the half
; without pixel (0, 0) votes "no" and keeps v6 = 0. r0 = v6, r1 = pixel (0, 0).
; Run: vk_harness vs_ids.spv ps_w64_c_vote_rewrite.spv out.raw with VKH_SUBGROUP_SIZE=64 and 32: r1 equal, r0 differs
; in the pixels of the other half (the detector must call this shader tier C, tainted-branch; before SHADER-820 it
; called it tier A).
; mattr: +wavefrontsize64,-wavefrontsize32
v_interp_p1_f32_e32 v2, v0, attr0.x
v_interp_p2_f32_e32 v2, v1, attr0.x
v_interp_p1_f32_e32 v3, v0, attr0.y
v_interp_p2_f32_e32 v3, v1, attr0.y
v_mul_f32_e32 v2, 0x42800000, v2
v_mul_f32_e32 v3, 0x42800000, v3
v_cvt_u32_f32_e32 v2, v2
v_cvt_u32_f32_e32 v3, v3
s_mov_b64 s[10:11], exec
v_or_b32_e32 v4, v2, v3
v_mov_b32_e32 v6, 0
v_cmp_eq_u32_e32 vcc, 0, v4
s_and_b64 s[20:21], exec, vcc
s_cmp_lg_u64 s[20:21], 0
v_cmp_le_u32_e64 s[20:21], 0, v4
s_cbranch_scc0 3
s_and_b64 exec, exec, s[20:21]
v_mov_b32_e32 v6, 1
s_mov_b64 exec, s[10:11]
v_cndmask_b32_e64 v7, 0, 1, vcc
v_mov_b32_e32 v8, 0
exp mrt0 v6, v7, v8, v8 done vm
s_endpgm
