; buffer_atomic_inc / buffer_atomic_dec (#2448) under contention: all 32 lanes of a wave32 group run every op on the
; same dwords, so the translated OpAtomicCompareExchange loops race for real (the CPU interpreter runs lanes in
; lockstep). The words start as the harness's 0xEE fill, F = 0xEEEEEEEE. Every lane applies the same op, so the final
; word does not depend on the order; the returned old values of one op are all different, so each lane adds
; 1 << (its old value's distance from F) into another word, and a lost or doubled update changes that sum.
;   out[0] = 32 x inc, D = ~0: F + 32                                                      = 0xEEEEEF0E
;   out[1] = F + sum of 1 << (old - F) over out[0]'s returned values: F + 0xFFFFFFFF iff each of 0..31 came once
;                                                                                            = 0xEEEEEEED
;   out[2] = 32 x inc, D = 20: the first gives 0 (F >= D), then it wraps every 21: 31 mod 21  = 0x0000000A
;   out[3] = 32 x dec, D = ~0: F - 32                                                       = 0xEEEEEECE
;   out[4] = 32 x dec, D = 5: the first gives 5 (F > D), then 5,4,..,0 repeats: 5 - 31 mod 6 = 0x00000004
;   out[5] = umax of F and out[0]'s returned values: F + 31                                  = 0xEEEEEF0D
;   out[6] = umin of F and out[3]'s returned values: F - 31                                  = 0xEEEEEECF
;   out[7] = F + sum of 1 << (F - old) over out[3]'s returned values                         = 0xEEEEEEED
; group: 32,1,1
; Run: vk_compute_test cs_w32_atomic_contend.spv 1 dump -> out[0..7] = 4008636174 4008636141 10 4008636110 4 4008636173 4008636111 4008636141
; mattr: +wavefrontsize32,-wavefrontsize64
; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
v_mov_b32_e32 v3, 0xeeeeeeee
v_mov_b32_e32 v5, 1
v_mov_b32_e32 v20, 0
v_mov_b32_e32 v1, -1
buffer_atomic_inc v1, v20, s[0:3], 0 offen glc
v_mov_b32_e32 v20, 8
v_mov_b32_e32 v8, 20
buffer_atomic_inc v8, v20, s[0:3], 0 offen
v_mov_b32_e32 v20, 12
v_mov_b32_e32 v6, -1
buffer_atomic_dec v6, v20, s[0:3], 0 offen glc
v_mov_b32_e32 v20, 16
v_mov_b32_e32 v9, 5
buffer_atomic_dec v9, v20, s[0:3], 0 offen
s_waitcnt vmcnt(0)
v_sub_nc_u32_e32 v2, v1, v3
v_lshlrev_b32_e32 v4, v2, v5
v_mov_b32_e32 v20, 4
buffer_atomic_add v4, v20, s[0:3], 0 offen
v_mov_b32_e32 v20, 20
buffer_atomic_umax v1, v20, s[0:3], 0 offen
v_sub_nc_u32_e32 v7, v3, v6
v_lshlrev_b32_e32 v10, v7, v5
v_mov_b32_e32 v20, 28
buffer_atomic_add v10, v20, s[0:3], 0 offen
v_mov_b32_e32 v20, 24
buffer_atomic_umin v6, v20, s[0:3], 0 offen
s_endpgm
