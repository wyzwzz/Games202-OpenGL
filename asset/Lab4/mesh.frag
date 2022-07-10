#version 460 core
layout(location = 0) in vec3 iFragPos;
layout(location = 1) in vec3 iFragNormal;

layout(location = 0) out vec4 oFragColor;

layout(binding = 0) uniform Params{
    vec3 Albedo;
    float Roughness;

    vec3 EdgeTint;
    float Metallic;

    vec3 ViewPos;
    int EnableKC;

    vec3 LightDir;
    vec3 LightIntensity;
};

layout(binding = 0) uniform sampler2D EMiu;
layout(binding = 1) uniform sampler1D EAvg;

#define PI 3.14159265359

float distributionGGX(vec3 N, vec3 H, float roughness){
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N,H),0.f);
    float NdotH2 = NdotH * NdotH;

    float nom = a2;
    float denom = (NdotH2 * (a2 - 1.f) + 1.f);
    denom = PI * denom * denom;

    return nom / denom;
}

float geometrySchlickGGX(float NdotV, float roughness){
    float a = roughness;
    float k = (a * a) / 2.0;

    float nom   = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return nom / denom;
}

float geometrySmith(vec3 N, vec3 V, vec3 L, float roughness){
    float NdotV = max(dot(N,V),0.f);
    float NdotH = max(dot(N,L),0.f);
    float ggx2 = geometrySchlickGGX(NdotV,roughness);
    float ggx1 = geometrySchlickGGX(NdotH,roughness);

    return ggx1 * ggx2;
}

vec3 fresnelSchlick(float cosTheta, vec3 F0){
    return F0 + (1.f - F0) * pow(max(1.f - cosTheta, 0.f), 5.f);
}

//https://blog.selfshadow.com/publications/s2017-shading-course/imageworks/s2017_pbs_imageworks_slides_v2.pdf
vec3 averageFresnel(vec3 r, vec3 g)
{
    return vec3(0.087237) + 0.0230685*g - 0.0864902*g*g + 0.0774594*g*g*g
    + 0.782654*r - 0.136432*r*r + 0.278708*r*r*r
    + 0.19744*g*r + 0.0360605*g*g*r - 0.2586*g*r*r;
}

vec3 multiScatterBRDF(float NdotL, float NdotV)
{
    vec3 E_o = texture(EMiu,vec2(NdotL, Roughness)).xxx;
    vec3 E_i = texture(EMiu,vec2(NdotV, Roughness)).xxx;

    vec3 E_avg = texture(EAvg,Roughness).xxx;

    vec3 F_avg = averageFresnel(Albedo, EdgeTint);

    vec3 fms = (1.f - E_o) * (1.f - E_i) / (PI * (1.0 - E_avg));
    vec3 fadd = F_avg * E_avg / (1.0 - F_avg * (1.0 - E_avg));

    return fms * fadd;
}

void main() {

    vec3 F0 = mix(vec3(0.04),Albedo,Metallic);

    vec3 N = normalize(iFragNormal);
    vec3 V = normalize(ViewPos - iFragPos);
    float NdotV = max(dot(N,V),0.0);

    vec3 L = normalize(-LightDir);
    float NdotL = max(dot(N,L),0.0);

    vec3 H = 0.5 * (L + V);
    float HdotV = max(dot(H,V),0.0);

    float NDF = distributionGGX(N,H,Roughness);
    float G = geometrySmith(N,V,L,Roughness);
    vec3 F = fresnelSchlick(HdotV,F0);

    vec3 numerator = NDF * G * F;
    float denominator = 4.0 * NdotV * NdotL;
    vec3 Fmicro = numerator / max(denominator,0.001f);

    vec3 BRDF = bool(EnableKC) ? Fmicro + multiScatterBRDF(NdotL,NdotV) : Fmicro;

    vec3 color = LightIntensity * BRDF * NdotL;

    oFragColor = vec4(pow(color,vec3(1.0/2.2)),1.0);
}
