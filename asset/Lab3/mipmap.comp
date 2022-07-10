#version 460 core
layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;
//float infinity = 1.0 / 0.0;

layout(r32f, binding = 0) uniform image2D PreLevel;
layout(r32f, binding = 1) uniform image2D CurLevel;

uniform vec2 PreSize;
uniform vec2 CurSize;

void main() {
    ivec2 g_index = ivec2(gl_WorkGroupSize.xy * gl_WorkGroupID.xy + gl_LocalInvocationID.xy);
    if (g_index.x >= CurSize.x || g_index.y >= CurSize.y){
        return;
    }

    vec2 uv00 = vec2(g_index) / CurSize;
    vec2 uv11 = vec2(g_index + ivec2(1)) / CurSize;

    ivec2 beg = ivec2(floor(uv00 * PreSize));
    ivec2 end = ivec2(ceil(uv11 * PreSize));
    float depth = 1.0 / 0.0;
    for (int x = beg.x; x < end.x; ++x){
        for (int y = beg.y; y < end.y; ++y){
            float d = imageLoad(PreLevel, ivec2(x, y)).r;
            if (d > 0)
            depth = min(d, depth);
        }
    }
    //depth of 0 will generate infinity in next level
    imageStore(CurLevel, g_index, vec4(depth));
}
