; SHADER-820 tier C, the EXEC form (#4620; @shdann's case, written as a GPU fixture by @sennecools): a vote on EXEC
; whose EXEC gets a new value before the branch, from an op that leaves SCC alone. Only the pixel (0, 0) (x = floor(u
; * 64) = 0, y = floor(v * 64) = 0 on the 64x64 target of vs_ids) passes the compare; s_and_saveexec_b64 narrows EXEC to
; it and sets SCC = (EXEC != 0), the vote; s_mov_b64 exec restores the launch mask without touching SCC. s_cbranch_scc0
; still decides on the narrowed EXEC, but the run side is the full mask: on a wave64 the wave with pixel (0, 0) writes
; v6 = 1 in every lane; on two 32-lane subgroups the half without pixel (0, 0) votes "no" and keeps v6 = 0.
; r0 = v6, r1 = pixel (0, 0).
; Run: vk_harness vs_ids.spv ps_w64_c_vote_exec.spv out.raw with VKH_SUBGROUP_SIZE=64 and 32: r1 equal, r0 differs in
; the pixels of the other half (the detector must call this shader tier C, tainted-branch; before #4620's EXEC rule it
; called it tier A). Measured by @sennecools on an RX 9070 XT (RADV): r0 differs between the two sizes.
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
s_and_saveexec_b64 s[12:13], vcc
s_mov_b64 exec, s[10:11]
s_cbranch_scc0 1
v_mov_b32_e32 v6, 1
v_cndmask_b32_e64 v7, 0, 1, vcc
v_mov_b32_e32 v8, 0
exp mrt0 v6, v7, v8, v8 done vm
s_endpgm
