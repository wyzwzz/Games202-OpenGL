#version 460 core

layout(location = 0) in vec2 iUV;

layout(location = 0) out vec4 oFragColor;

layout(std140, binding = 0) uniform DirectionalLight{
    vec3 LightDir;
    vec3 LightRadiance;
    mat4 LightProjView;
};

uniform sampler2D ShadowMap;
uniform sampler2D GBuffer0;
uniform sampler2D GBuffer1;

vec3 decodeNormal(vec2 f)
{
    f = f * 2.0 - 1.0;
    vec3 n = vec3(f.x, f.y, 1.0 - abs(f.x) - abs(f.y));
    float t = clamp(-n.z, 0, 1);
    n.xy += vec2((n.x >= 0 ? -t : t), (n.y >= 0 ? -t : t));
    return normalize(n);
}

void main() {
    vec4 p0 = texture(GBuffer0, iUV);
    vec4 p1 = texture(GBuffer1, iUV);
    vec3 pos = p0.xyz;
    vec2 oct_normal = vec2(p0.w, p1.w);
    vec3 normal = decodeNormal(oct_normal);
    vec2 color1 = unpackHalf2x16(floatBitsToUint(p1.x));
    vec3 albedo = vec3(color1.x, p1.g, color1.y);

    float view_z = p1.z;

    vec4 clip_coord = LightProjView * vec4(pos + normal * 0.02, 1.0);
    vec3 ndc_coord = (clip_coord.xyz / clip_coord.w) * 0.5 + 0.5;
    float shadow_z = texture(ShadowMap, ndc_coord.xy).r;
    float shadow_factor = shadow_z >= ndc_coord.z ? 1 : 0;

    vec3 color = LightRadiance * albedo * shadow_factor * max(0, dot(normal, -LightDir));
    oFragColor = vec4(color, 1.0);
}
