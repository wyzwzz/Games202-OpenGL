#version 460 core
layout(location = 0) in vec3 inWorldPos;
layout(location = 0) out vec4 outFragColor;

layout(binding = 0) uniform sampler2D environmentMap;

uniform mat4 Model;

#define PI 3.14159265
vec2 sampleSphericalMap(vec3 v){
    float theta = acos(v.y);
    float phi = (v.x == 0 && v.z == 0) ? 0 : atan(v.z,v.x);
    if(phi < 0) phi += 2 * PI;
    if(phi > 2 * PI) phi -= 2 * PI;
    vec2 uv = vec2(phi / (2 * PI),theta / PI);
    return uv;
}

#define A 2.51
#define B 0.03
#define C 2.43
#define D 0.59
#define E 0.14

vec3 tonemap(vec3 v)
{
    return (v * (A * v + B)) / (v * (C * v + D) + E);
}

void main(){
    vec3 dir = normalize(inWorldPos);
    dir = mat3(Model) * dir;
    vec3 env_color = texture(environmentMap,sampleSphericalMap(dir)).rgb;

    env_color = tonemap(env_color);

    env_color = pow(env_color,vec3(1.0/2.2));

    outFragColor = vec4(env_color,1.f);
}