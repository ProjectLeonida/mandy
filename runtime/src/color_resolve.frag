#version 450
#extension GL_EXT_samplerless_texture_functions : require

#if defined(RESOLVE_UINT)
#define Texture utexture2DMS
#define Value uvec4
#elif defined(RESOLVE_SINT)
#define Texture itexture2DMS
#define Value ivec4
#else
#define Texture texture2DMS
#define Value vec4
#endif

layout(set = 0, binding = 0) uniform Texture source_image;
layout(push_constant) uniform Resolve {
    ivec2 source_offset;
    int samples;
    int maximum;
} resolve;
layout(location = 0) out Value result;

void main() {
    ivec2 pixel = ivec2(gl_FragCoord.xy) + resolve.source_offset;
    Value value = texelFetch(source_image, pixel, 0);
    for (int i = 1; i < resolve.samples; ++i) {
        Value sample_value = texelFetch(source_image, pixel, i);
        value = resolve.maximum != 0 ? max(value, sample_value) : min(value, sample_value);
    }
    result = value;
}
