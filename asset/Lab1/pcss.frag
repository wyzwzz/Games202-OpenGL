#version 460 core
#define PI2 6.283185307179586
#define NUM_RINGS 10
#define EPS 1e-3
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
layout(location = 4) uniform float LightNearPlane;
layout(location = 5) uniform float LightRadius;
layout(location = 6) uniform int ShadowSampleCount;
layout(location = 7) uniform int BlockSearchSampleCount;

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

float rand(vec3 co)
{
    return fract(sin(dot(co.xy * co.z, vec2(12.9898, 78.233))) * 43758.5453);
}
float findBlockerDepth(vec2 shadow_uv,float z_receiver,float rnd){
    //search radius depends on the light size and receiver's distance from the light
    //set shadow map in the near plane between ligth and receiver
    float block_search_r = LightRadius * (z_receiver - LightNearPlane) / z_receiver;

    float possion_angle_step = PI2 * NUM_RINGS / BlockSearchSampleCount;
    float possion_angle = rnd * PI2;
    float possion_radius = 1.0 / BlockSearchSampleCount;
    float possion_radius_step = possion_radius;
    float sum_block_depth = 0;
    int block_count = 0;
    for(int i = 0; i < BlockSearchSampleCount; i++){
        vec2 offset = vec2(cos(possion_angle),sin(possion_angle)) * pow(possion_radius,0.75);
        possion_radius += possion_radius_step;
        possion_angle += possion_angle_step;
        float z = texture(ShadowMap,shadow_uv + offset * block_search_r).r;
        if(z + EPS < z_receiver){
            sum_block_depth += z;
            block_count++;
        }
    }
    return block_count > 0 ? sum_block_depth / block_count : -1.0;

}
float computeShadowFactor(){
    vec4 ndc_coord = LightProjView * vec4(iFragPos,1.0);
    ndc_coord = (ndc_coord / ndc_coord.w) * 0.5 + 0.5;
    vec2 shadow_uv = ndc_coord.xy;
    float shadow_z = ndc_coord.z;

    float z_blocker = findBlockerDepth(shadow_uv,shadow_z,rand(iFragPos + 1.5));
    if(z_blocker < EPS)
        return 1;

    float pcf_radius = LightRadius * (shadow_z - z_blocker) / z_blocker;

    float possion_angle_step = PI2 * NUM_RINGS / float(ShadowSampleCount);
    float possion_angle = rand(iFragPos) * PI2;
    float possion_radius = 1.0 / ShadowSampleCount;
    float possion_radius_step = possion_radius;
    float sum = 0;
    for(int i = 0; i < ShadowSampleCount; i++){
        vec2 offset = vec2(cos(possion_angle),sin(possion_angle)) * pow(possion_radius,0.75);
        possion_radius += possion_radius_step;
        possion_angle += possion_angle_step;
        float z = texture(ShadowMap,shadow_uv + offset * pcf_radius).r;
        sum += float(shadow_z - EPS <= z);
    }
    return sum / ShadowSampleCount;
}
void main() {
    vec3 albedo = texture(AlbedoMap,iUV).rgb;
    vec3 light_factor = computeLightFactor();
    float shadow_factor = computeShadowFactor();
    vec3 color =  (light_factor * shadow_factor + ambient) * albedo;
    color = pow(color,vec3(1.0/2.2));
    oFragColor = vec4(color,1.0);
}