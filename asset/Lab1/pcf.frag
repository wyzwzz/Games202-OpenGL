#version 460 core

layout(location = 0) in vec3 iFragPos;
layout(location = 1) in vec3 iFragNormal;
layout(location = 2) in vec2 iUV;
layout(location = 0) out vec4 oFragColor;

layout(std140,binding = 0) uniform PointLight{
    vec3 light_pos;
    float fade_cos_begin;//1

    vec3 light_dir;
    float fade_cos_end;

    vec3 light_radiance;
    float ambient;
};

layout(location = 1) uniform sampler2D ShadowMap;
layout(location = 2) uniform sampler2D AlbedoMap;
layout(location = 3) uniform mat4 LightProjView;
layout(location = 4) uniform float FilterSize;
layout(location = 5) uniform int SampleCount;

vec3 computeLightFactor(){
    vec3 pos = iFragPos;
    vec3 normal = normalize(iFragNormal);

    vec3 light_to_pos = pos - light_pos;
    float dist2 = dot(light_to_pos,light_to_pos);
    vec3 atten_radiance = light_radiance / dist2;
    light_to_pos = normalize(light_to_pos);

    float cos_theta = dot(light_to_pos,light_dir);
    float fade_factor = (cos_theta - fade_cos_end) / (fade_cos_begin - fade_cos_end);
    fade_factor = pow(clamp(fade_factor,0,1),3);
    return fade_factor * atten_radiance * max(dot(normal,-light_to_pos),0);
}
#define PI2 (3.1415926 * 3.1415926)
float rand(vec3 co)
{
    return fract(sin(dot(co.xy * co.z, vec2(12.9898, 78.233))) * 43758.5453);
}

float computeShadowFactor(){
    vec4 ndc_coord = LightProjView * vec4(iFragPos,1.0);
    ndc_coord = (ndc_coord / ndc_coord.w) * 0.5 + 0.5;
    vec2 shadow_uv = ndc_coord.xy;
    float shadow_z = ndc_coord.z;

    vec2 texel = vec2(FilterSize) / textureSize(ShadowMap,0);

    float possion_angle_step = PI2 * 10 / float(SampleCount);
    float possion_angle = rand(iFragPos) * PI2;
    float possion_radius = 1.0 / SampleCount;
    float possion_radius_step = possion_radius;
    float sum = 0;
    for(int i = 0; i < SampleCount; i++){
        vec2 offset = vec2(cos(possion_angle),sin(possion_angle)) * pow(possion_radius,0.75);
        possion_radius += possion_radius_step;
        possion_angle += possion_angle_step;
        float z = texture(ShadowMap,shadow_uv + offset * texel).r;
        sum += float(shadow_z - 0.001 <= z);
    }
    return sum / SampleCount;
}
void main() {
    vec3 albedo = texture(AlbedoMap,iUV).rgb;
    vec3 light_factor = computeLightFactor();
    float shadow_factor = computeShadowFactor();
    vec3 color =  (light_factor * shadow_factor + ambient) * albedo;
    color = pow(color,vec3(1.0/2.2));
    oFragColor = vec4(color,1.0);
}