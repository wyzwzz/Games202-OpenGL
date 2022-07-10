#version 460 core
layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(binding = 0, rgba32f) uniform readonly image2D CurGBuffer;
layout(binding = 1, rgba8) uniform readonly image2D CurIndirect;
layout(binding = 2, rgba8) uniform writeonly image2D Dst;

layout(binding = 0) uniform sampler2D Src;
layout(binding = 1) uniform sampler2D PreGBuffer;

uniform mat4 PreViewProj;
uniform float Ratio;

void main() {
    ivec2 g_index = ivec2(gl_WorkGroupSize.xy * gl_WorkGroupID.xy + gl_LocalInvocationID.xy);
    ivec2 res = imageSize(CurIndirect);
    if (g_index.x >= res.x || g_index.y >= res.y){
        return;
    }

    vec4 cur_indirect = imageLoad(CurIndirect, g_index);
    if (cur_indirect.w == 0){
        imageStore(Dst, g_index, vec4(0));
        return;
    }
    vec3 pos = imageLoad(CurGBuffer, g_index).xyz;
    vec4 pre_pos_clip = PreViewProj * vec4(pos, 1.0);
    vec2 pre_uv = pre_pos_clip.xy / pre_pos_clip.w * 0.5 + 0.5;
    if (any(notEqual(clamp(pre_uv, vec2(0), vec2(1)), pre_uv))){
        imageStore(Dst, g_index, cur_indirect);
        return;
    }
    vec4 accu_src = texture(Src, pre_uv);
    vec3 pre_pos = texture(PreGBuffer, pre_uv).xyz;
    float alpha = Ratio;
    alpha *= exp(min(distance(pre_pos, pos), 1));
    alpha = clamp(alpha, 0, 1);
    imageStore(Dst, g_index, mix(accu_src, cur_indirect, alpha));
}