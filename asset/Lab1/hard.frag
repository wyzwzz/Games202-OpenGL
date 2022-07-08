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
float computeShadowFactor(){
    vec4 ndc_coord = LightProjView * vec4(iFragPos,1.0);
    ndc_coord = (ndc_coord / ndc_coord.w) * 0.5 + 0.5;
    vec2 shadow_uv = ndc_coord.xy;
    float shadow_z = ndc_coord.z;
    if(all(equal(clamp(shadow_uv,vec2(0),vec2(1)),shadow_uv))){
        float z = texture(ShadowMap,shadow_uv).r;
        float shadow_factor = float(shadow_z - 0.001 <= z);
        return shadow_factor;
    }
    return 1;
}
void main() {
    vec3 albedo = texture(AlbedoMap,iUV).rgb;
    vec3 light_factor = computeLightFactor();
    float shadow_factor = computeShadowFactor();
    vec3 color =  (light_factor * shadow_factor + ambient) * albedo;
    color = pow(color,vec3(1.0/2.2));
    oFragColor = vec4(color,1.0);
}