; Resident buffer_load* tfe on the GPU (docs/design.md, "Resident buffer_load*/tbuffer_load* tfe"; CPU-checked by
; tests/semantic/test_buffer_load_tfe.py, which also covers the MTBUF tbuffer_load_format_* form llvm-mc cannot
; assemble). With tfe the VDATA range holds one status dword after the data VGPRs; the ISA writes it only when the
; fetch NACKs, which a Vulkan buffer cannot produce, so it keeps its prior value. Heap bytes 0x40..0x4f are seeded
; with 0x5eed0100..0x5eed0103 and the three status VGPRs with their own seeds; the V# has base 0x40, stride 0,
; NUM_RECORDS 0x10 and word3 0x31016fac (OOB_SELECT 3, RAW), as in cs_buf_tail.asm:
;   buffer_load_dword   v[4:5]   v20=0  offen tfe -> v4 = 0x5eed0100, v5 keeps 0x5eed0055
;   buffer_load_dwordx2 v[6:8]   v21=4  offen tfe -> v6 = 0x5eed0101, v7 = 0x5eed0102, v8 keeps 0x5eed0066
;   buffer_load_ushort  v[9:10]  v22=8  offen tfe -> v9 = 0x0102, v10 keeps 0x5eed0077
; and v[4:10] is written to out[0..6]. Every lane loads the same addresses, so all lanes store the same words.
; Run: vk_compute_test cs_buf_tfe.spv 1 dump -> out[0..6] = 0x5eed0100 0x5eed0055 0x5eed0101 0x5eed0102
; 0x5eed0066 0x00000102 0x5eed0077.
v_cmpx_eq_u32_e32 0, v0
v_mov_b32 v1, 0x5eed0100
v_mov_b32 v2, 0x5eed0101
v_mov_b32 v3, 0x5eed0102
buffer_store_dwordx4 v[1:4], off, s[0:3], 0 offset:64
s_waitcnt vmcnt(0)
s_mov_b32 s4, 0x40
s_mov_b32 s5, 0
s_mov_b32 s6, 0x10
s_mov_b32 s7, 0x31016fac
v_mov_b32 v20, 0
v_mov_b32 v21, 4
v_mov_b32 v22, 8
v_mov_b32 v5, 0x5eed0055
v_mov_b32 v8, 0x5eed0066
v_mov_b32 v10, 0x5eed0077
buffer_load_dword v[4:5], v20, s[4:7], 0 offen tfe
buffer_load_dwordx2 v[6:8], v21, s[4:7], 0 offen tfe
buffer_load_ushort v[9:10], v22, s[4:7], 0 offen tfe
s_waitcnt vmcnt(0)
buffer_store_dwordx4 v[4:7], off, s[0:3], 0
buffer_store_dwordx2 v[8:9], off, s[0:3], 0 offset:16
buffer_store_dword v10, off, s[0:3], 0 offset:24
s_endpgm
