#pragma once
#include <CGUtils/api.hpp>
#include <CGUtils/model.hpp>
using namespace wzz::gl;
using namespace wzz::model;

constexpr float PI = wzz::math::PI_f;
constexpr float invPI = wzz::math::invPI<float>;

using color3f = wzz::math::color3f;

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

template<typename T, typename Func>
void parallel_forrange(T beg, T end, Func &&func, int worker_count = 0)
{
    std::mutex it_mutex;
    T it = beg;
    auto next_item = [&]() -> std::optional<T>
    {
        std::lock_guard lk(it_mutex);
        if(it == end)
            return std::nullopt;
        return std::make_optional(it++);
    };

    std::mutex except_mutex;
    std::exception_ptr except_ptr = nullptr;

    auto worker_func = [&](int thread_index)
    {
        for(;;)
        {
            auto item = next_item();
            if(!item)
                break;

            try
            {
                func(thread_index, *item);
            }
            catch(...)
            {
                std::lock_guard lk(except_mutex);
                if(!except_ptr)
                    except_ptr = std::current_exception();
            }

            std::lock_guard lk(except_mutex);
            if(except_ptr)
                break;
        }
    };

    std::vector<std::thread> workers;
    for(int i = 0; i < worker_count; ++i)
        workers.emplace_back(worker_func, i);

    for(auto &w : workers)
        w.join();

    if(except_ptr)
        std::rethrow_exception(except_ptr);
}