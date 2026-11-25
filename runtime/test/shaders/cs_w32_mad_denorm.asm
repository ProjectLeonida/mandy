; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; Denormal flush in v_mad_f32, v_mac_f32, v_mad_legacy_f32 and v_mac_legacy_f32 (#379 #380): a subnormal operand or
; result reads as signed zero, so the Vulkan words check the translator's model even though the gfx1036 part does
; not decode these four GFX10.1 opcodes (the native measurement is an RDNA1 one).
; Lane 0 of a wave32 group computes one result per form and stores eight dwords. SUBN = 0x00000001 (2^-149),
; P23 = 0x4B000000 (2^23), FLT_MIN = 0x00800000, HALF = 0x3F000000:
;   out[0] = v_mad_f32 SUBN * P23 + 0            = 0  (operand flushed; 2^-149 * 2^23 = FLT_MIN otherwise)
;   out[1] = v_mad_f32 FLT_MIN * HALF + 0        = 0  (product 2^-127 flushed; 0x00400000 otherwise)
;   out[2] = v_mad_f32 SUBN * P23 + 0 clamp      = 0  (the flush runs before clamp)
;   out[3] = v_mac_f32 D=0, SUBN * P23           = 0  (D is read as the addend; 0x00800000 otherwise)
;   out[4] = v_mac_f32 D=1.0, 2.0 * 3.0          = 0x40E00000 (7.0; a control: the op itself runs)
;   out[5] = v_mad_legacy_f32 SUBN * P23 + 0     = 0
;   out[6] = v_mad_legacy_f32 FLT_MIN * HALF + 0 = 0
;   out[7] = v_mac_legacy_f32 D=0, SUBN * P23    = 0
; group: 32,1,1
; Run: VKC_SUBGROUP_SIZE=32 vk_compute_test cs_w32_mad_denorm.spv 1 dump -> out[0..7] = 0 0 0 0 1088421888 0 0 0
; mattr: +wavefrontsize32,-wavefrontsize64
v_mov_b32_e32 v1, 1
v_mov_b32_e32 v2, 0x4b000000
v_mov_b32_e32 v4, 0
v_mad_f32 v3, v1, v2, v4
v_mov_b32_e32 v1, 0x00800000
v_mov_b32_e32 v2, 0.5
v_mad_f32 v5, v1, v2, v4
v_mov_b32_e32 v1, 1
v_mov_b32_e32 v2, 0x4b000000
v_mad_f32 v6, v1, v2, v4 clamp
v_mov_b32_e32 v7, 0
v_mac_f32 v7, v1, v2
v_mov_b32_e32 v8, 1.0
v_mov_b32_e32 v1, 2.0
v_mov_b32_e32 v2, 3.0
v_mac_f32 v8, v1, v2
v_mov_b32_e32 v1, 1
v_mov_b32_e32 v2, 0x4b000000
v_mad_legacy_f32 v9, v1, v2, v4
v_mov_b32_e32 v1, 0x00800000
v_mov_b32_e32 v2, 0.5
v_mad_legacy_f32 v10, v1, v2, v4
v_mov_b32_e32 v1, 1
v_mov_b32_e32 v2, 0x4b000000
v_mov_b32_e32 v11, 0
v_mac_legacy_f32 v11, v1, v2
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v20, 0
buffer_store_dword v3, v20, s[0:3], 0 offen
buffer_store_dword v5, v20, s[0:3], 0 offen offset:4
buffer_store_dword v6, v20, s[0:3], 0 offen offset:8
buffer_store_dword v7, v20, s[0:3], 0 offen offset:12
buffer_store_dword v8, v20, s[0:3], 0 offen offset:16
buffer_store_dword v9, v20, s[0:3], 0 offen offset:20
buffer_store_dword v10, v20, s[0:3], 0 offen offset:24
buffer_store_dword v11, v20, s[0:3], 0 offen offset:28
s_endpgm
