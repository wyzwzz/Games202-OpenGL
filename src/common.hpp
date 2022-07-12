#pragma once
#include <CGUtils/api.hpp>
#include <CGUtils/model.hpp>
using namespace wzz::gl;
using namespace wzz::model;

constexpr float PI = wzz::math::PI_f;
constexpr float invPI = wzz::math::invPI<float>;

vec2f ConcentricSampleDisk(float u, float v){
    vec2f u_offset = vec2f(u * 2, v * 2) - vec2f(1, 1);
    if(u_offset.x == 0 && u_offset.y == 0) return vec2f(0,0);

    float theta,r;
    if(std::abs(u_offset.x) > std::abs(u_offset.y)){
        r = u_offset.x;
        theta = PI / 4 * (u_offset.y / u_offset.x);
    }
    else{
        r = u_offset.y;
        theta = PI / 2 - PI / 4 * (u_offset.x / u_offset.y);
    }
    return r * vec2f(std::cos(theta),std::sin(theta));
}

vec3f CosineSampleHemisphere(float u, float v){
    vec2f d = ConcentricSampleDisk(u,v);
    float z = std::sqrt(std::max<float>(0,1-d.x*d.x-d.y*d.y));
    return vec3f(d.x,d.y,z);
}

inline float CosineHemispherePdf(float cos_theta){ return cos_theta * invPI;}
