#version 460

layout(vertices = 3) out;
layout(push_constant) uniform UserData {
    uint s[32];
    uvec2 heap_base;
    uvec2 code_base;
} pc;

void main() {
    if (gl_InvocationID == 0) {
        // Drop this patch if the IA topology did not give the hull shader the requested input count.
        float level = uint(gl_PatchVerticesIn) == pc.s[8] ? 1.0 : 0.0;
        gl_TessLevelOuter[0] = level;
        gl_TessLevelOuter[1] = level;
        gl_TessLevelOuter[2] = level;
        gl_TessLevelInner[0] = level;
    }
    gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;
}
