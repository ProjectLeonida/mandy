#version 460
// A pixel shader written to the translator's runtime contract, standing in for a translated one in
// d3d12_draw_test. The tests' signatures are two root constants then a root CBV, so in the console's graphics layout
// (from s8, two dwords per root view: docs/d3d12_vulkan.md, D3D-448) s10:s11 is the CBV: the console VA of three
// floats (a tint), read at heap base + VA. The output is (red * tint.x, tint.y, blue * tint.z, 1).
// d3d12_mesh_test uses it as is: its mesh shader reads the same block through the root window (push dword 8 = s16).
// Compiled with -DPC_TAIL it is a --pc-tail-ubo shader: s0..s31 alone fill the 128-byte push-constant block, and the
// heap base comes from the uniform block at set 1, binding 0.
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference_uvec2 : require

#ifndef ROOT_BASE
#define ROOT_BASE 8
#endif

layout(buffer_reference, std430, buffer_reference_align = 4) buffer Floats {
    float f[];
};

layout(location = 0) in vec4 color;
layout(location = 0) out vec4 target;

#ifdef PC_TAIL
layout(push_constant) uniform UserData {
    uint s[32];
} pc;
layout(set = 1, binding = 0) uniform PcTail {
    uvec2 heap_base;
    uvec2 code_base;
} tail;
#define HEAP_BASE tail.heap_base
#else
layout(push_constant) uniform UserData {
    uint s[32];
    uvec2 heap_base;
    uvec2 code_base;
} pc;
#define HEAP_BASE pc.heap_base
#endif

uvec2 add64(uvec2 a, uvec2 b) {
    uint carry;
    const uint low = uaddCarry(a.x, b.x, carry);
    return uvec2(low, a.y + b.y + carry);
}

void main() {
    Floats tint = Floats(add64(HEAP_BASE, uvec2(pc.s[ROOT_BASE + 2], pc.s[ROOT_BASE + 3])));
    target = vec4(color.r * tint.f[0], tint.f[1], color.b * tint.f[2], 1.0);
}
