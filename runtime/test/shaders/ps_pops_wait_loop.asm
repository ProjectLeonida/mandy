; POPS (primitive-ordered pixel shading) wait loop (docs/design.md "POPS"): the entry of an ordered section as console
; pixel programs write it, wave64. The hardware puts the collision word in the SGPR after the user SGPRs and the
; primitive mask (s18 here): collision wave id in bits 0-9, this wave's id in bits 16-25, overlap in bit 31. The
; program extracts the ids, computes s5 = wave_id + ~collision_id, skips the loop when the overlap bit is clear (or
; its turn has come) for a lane in EXEC, and otherwise waits until s5 < src_pops_exiting_wave_id + ~collision_id.
; The translator orders the section with fragment-shader interlock, reads src_pops_exiting_wave_id as 0 and seeds the
; collision word with 0x00010000 (no overlap, wave id 1, collision id 0), so a wave that enters the loop leaves it
; after one round: s5 = 1 + 0xffffffff = 0 < 0 + 0xffffffff.
; x: rounds of the loop with EXEC all off at s_cbranch_vccnz (the vote fails, the wave falls into the loop): 1.
;   A collision word with wave id == collision id (0, what the push constants hold for s18) gives s5 = 0xffffffff,
;   never below 0xffffffff: the loop ran until the loop guard (65536 rounds, x = 0x00010000).
; y: rounds with EXEC all on: the overlap test holds for every lane, the branch skips the loop: 0.
; z, w: 1.0, so a pixel that is written shows it.
; Each s_cbranch_vccnz skips the 5 dwords of its loop; each s_cbranch_scc0 -5 goes back to the loop's s_sleep.
; Expected mrt0 bits: 0x00000001, 0x00000000, 0x3f800000, 0x3f800000
s_mov_b64 s[20:21], exec
s_mov_b32 s12, 0
s_mov_b32 s13, 0
s_mov_b64 exec, 0
s_bfe_u32 s2, s18, 0xa0000
s_bfe_u32 s5, s18, 0xa0010
s_not_b32 s4, s2
s_lshr_b32 s0, s18, 31
s_add_i32 s5, s5, s4
s_cmp_eq_u32 s0, 0
s_cselect_b64 s[0:1], exec, 0
s_add_i32 s2, src_pops_exiting_wave_id, s4
s_cmp_gt_u32 s2, s5
s_cselect_b64 s[2:3], exec, 0
s_or_b64 s[0:1], s[0:1], s[2:3]
s_and_b64 vcc, exec, s[0:1]
s_cbranch_vccnz 5
s_sleep 0xffff
s_add_i32 s12, s12, 1
s_add_i32 s0, src_pops_exiting_wave_id, s4
s_cmp_lt_u32 s5, s0
s_cbranch_scc0 -5
s_mov_b64 exec, s[20:21]
s_bfe_u32 s2, s18, 0xa0000
s_bfe_u32 s5, s18, 0xa0010
s_not_b32 s4, s2
s_lshr_b32 s0, s18, 31
s_add_i32 s5, s5, s4
s_cmp_eq_u32 s0, 0
s_cselect_b64 s[0:1], exec, 0
s_add_i32 s2, src_pops_exiting_wave_id, s4
s_cmp_gt_u32 s2, s5
s_cselect_b64 s[2:3], exec, 0
s_or_b64 s[0:1], s[0:1], s[2:3]
s_and_b64 vcc, exec, s[0:1]
s_cbranch_vccnz 5
s_sleep 0xffff
s_add_i32 s13, s13, 1
s_add_i32 s0, src_pops_exiting_wave_id, s4
s_cmp_lt_u32 s5, s0
s_cbranch_scc0 -5
v_mov_b32_e32 v0, s12
v_mov_b32_e32 v1, s13
v_mov_b32_e32 v2, 1.0
exp mrt0 v0, v1, v2, v2 done vm
s_endpgm
