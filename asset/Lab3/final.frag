#version 460 core

layout(location = 0) in vec2 iUV;

layout(location = 0) out vec4 oFragColor;

uniform sampler2D Direct;
uniform sampler2D Indirect;
uniform sampler2D GBuffer1;//color part in xy

uniform int EnableDirect;
uniform int EnableIndirect;
uniform int EnableTonemap;
uniform float Exposure;

#define A 2.51
#define B 0.03
#define C 2.43
#define D 0.59
#define E 0.14

vec3 tonemap(vec3 x)
{
    vec3 v = x * Exposure;
    return (v * (A * v + B)) / (v * (C * v + D) + E);
}

void main() {

    vec3 color = vec3(0);
    vec4 direct = texture(Direct, iUV);
    vec4 indirect = texture(Indirect, iUV);
    vec4 p1 = texture(GBuffer1, iUV);
    vec2 color1 = unpackHalf2x16(floatBitsToUint(p1.x));
    vec3 albedo = vec3(color1.x, p1.g, color1.y);

    if (bool(EnableDirect))
        color += direct.rgb * direct.w;
    if (bool(EnableIndirect))
        color += albedo * indirect.rgb * indirect.w;

    if (bool(EnableTonemap)){
        oFragColor = vec4(pow(tonemap(color), vec3(1.0/2.2)), 1);
    }
    else {
        oFragColor = vec4(pow(color, vec3(1.0/2.2)), 1.0);
    }
}
