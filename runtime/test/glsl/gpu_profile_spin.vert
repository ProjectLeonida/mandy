#version 450
// vk_gpu_profile_test: a draw whose GPU time is its vertex shader's loop of s0 turns. Every vertex lands outside the
// viewport, so nothing is rasterised; the position depends on the loop, so the loop is kept.
layout(push_constant) uniform UserData {
    uint s[32];
    uvec2 heap_base;
    uvec2 code_base;
} pc;

void main() {
    uint v = uint(gl_VertexIndex);
    for (uint i = 0; i < pc.s[0]; ++i) v = (v * 1664525u + 1013904223u) ^ (v >> 13);
    gl_Position = vec4(3.0 + float(v & 1u), 3.0, 0.0, 1.0);
}
