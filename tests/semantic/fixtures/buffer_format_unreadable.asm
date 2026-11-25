; s[4:7] is the tested V#. The harness gives s[0:3] a separate output range.
; s[4:7] is the tested V#: its FORMAT picks the decode at run time. s[0:3] is a raw V# over a separate output range.
v_cmpx_eq_u32_e32 0, v0
buffer_load_format_xyzw v[4:7], off, s[4:7], 0
s_waitcnt vmcnt(0)
buffer_store_dwordx4 v[4:7], off, s[0:3], 0
s_endpgm
