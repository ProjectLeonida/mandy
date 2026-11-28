; Mesh shader (NGG, wave64), PSV0: 128 threads, 64/64 outputs, triangles; the group-shared size is the DXBC's.
; One ds_write_b32 gives the shader an LDS array; the workgroup-memory limit tests size it through PSV0.
s_mov_b64 exec, -1
v_lshlrev_b32_e32 v1, 2, v0
ds_write_b32 v1, v0
s_endpgm
