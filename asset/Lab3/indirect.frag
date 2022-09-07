#version 460 core
#define PI 3.14159265

layout(location = 0) in vec2 iUV;

layout(location = 0) out vec4 oFragColor;

layout(std140, binding = 0) uniform IndirectParams{
    mat4 View;
    mat4 Proj;

    int IndirectSampleCount;
    int IndirectRayMaxSteps;
    int FrameIndex;
    int TraceMaxLevel;
    int Width;
    int Height;
    float DepthThreshold;
    float RayMarchingStep;
    int UseHierarchicalTrace;
};

layout(std430, binding = 0) buffer RandomSamples{
    vec2 RawSamples[];//sample count with IndirectSampleCount
};

uniform sampler2D Direct;
uniform sampler2D GBuffer0;
uniform sampler2D GBuffer1;
uniform sampler2D ViewDepth;

vec3 decodeNormal(vec2 f)
{
    f = f * 2.0 - 1.0;
    vec3 n = vec3(f.x, f.y, 1.0 - abs(f.x) - abs(f.y));
    float t = clamp(-n.z, 0, 1);
    n.xy += vec2((n.x >= 0 ? -t : t), (n.y >= 0 ? -t : t));
    return normalize(n);
}

// http://advances.realtimerendering.com/s2015/
// stochastic screen-space reflections
bool hierarchicalRayTrace(in float jitter, in vec3 ori, in vec3 dir, out vec2 res_uv){
    //basic idea: trace ray in view space
    //stackless ray walk of min-z pyramid
    //mip = 0
    //while(level > -1 )
    //    step through current cell
    //    if(above z plane) ++level;
    //    if(below z plane) --level;

    float ray_len = (ori.z + dir.z < 0.1) ? (0.1 - ori.z) / dir.z : 1;
    vec3 end = ori + dir * ray_len;

    vec4 ori_clip = Proj * vec4(ori, 1.0);
    vec4 end_clip = Proj * vec4(end, 1.0);

    float inv_ori_w = 1.0 / ori_clip.w;
    float inv_end_w = 1.0 / end_clip.w;

    vec2 ori_ndc = ori_clip.xy * inv_ori_w;//-1 ~ 1
    vec2 end_ndc = end_clip.xy * inv_end_w;
    vec2 extend_ndc = end_ndc - ori_ndc;

    int width = Width;
    int height = Height;
    vec2 delta_pixel = 0.5 * extend_ndc * vec2(width, height);//not multiply 0.5 is also ok

    //step along x or y which has bigger differential
    bool swap_xy = false;
    if (abs(extend_ndc.x * width) < abs(extend_ndc.y * height)){
        swap_xy = true;
        ori_ndc = ori_ndc.yx;
        end_ndc = end_ndc.yx;
        extend_ndc = extend_ndc.yx;
        delta_pixel = delta_pixel.yx;
        width = Height;
        height = Width;
    }

    // x01 y01 z01 is view space coord
    // dx ~ near * (x1 / z1 - x0 / z0)
    // dy ~ near * (y1 / z1 - y0 / z0)
    // dz ~ near * far * (1 / z1 - 1 / z0) ==> z ~ (1 / z1 - 1 / z0)
    // w = view z
    // d(1/w) is linear in ndc space coord

    //ndc x delta per pixel
    float dx = sign(extend_ndc.x) * 2 / width;
    //normalize dx
    dx *= abs(delta_pixel.x) / length(delta_pixel);
    float dy = extend_ndc.y / extend_ndc.x * dx;
    // unit delta ndc per pixel
    vec2 dp = vec2(dx, dy);

    float ori_z_over_w = ori.z * inv_ori_w;//-1 ~ 1
    float end_z_over_w = end.z * inv_end_w;
    // z / w is linear in ndc coord
    float dz_over_w = (end_z_over_w - ori_z_over_w) / extend_ndc.x * dx;
    // 1 / w is linear in ndc coord
    float dinv_w = (inv_end_w - inv_ori_w) / extend_ndc.x * dx;

    #define HIT_STEPS 3
    #ifndef NO_MIPMAP
    //start from level 0
    int level = 0;
    //ray advance step, will change according to current level
    float level_advance_dist[16];
    //measure in pixel size
    float step = RayMarchingStep;
    int total_steps = IndirectRayMaxSteps;
    int steps_count = 0;
    level_advance_dist[0] = jitter * step;
    //if hit a level than keep search this level for some times
    int hit = 0;
    while (true){
        if (++steps_count > total_steps)
            return false;

        //get curret ray advance start pos
        float t = level_advance_dist[level] + step;
        //update ray advance dist after current step
        level_advance_dist[level] = t;

        //compute current ray ndc
        vec2 p = ori_ndc + t * dp;
        float z_over_w = ori_z_over_w + t * dz_over_w;
        float inv_w = inv_ori_w + t * dinv_w;

        vec2 ndc = swap_xy ? p.yx : p;
        vec2 uv = ndc * 0.5 + 0.5;
        if (any(notEqual(clamp(uv, vec2(0), vec2(1)), uv)))
            return false;

        float ray_depth = z_over_w / inv_w - 0.1;
        float cell_z = textureLod(ViewDepth, uv, level).r;

        if (ray_depth < cell_z){
            if (--hit < 0){
                level = min(level + 1, TraceMaxLevel);
                level_advance_dist[level] = t;
                step = float(1 << level) * RayMarchingStep;
            }
            continue;
        }
        //only check if level is zero and check depth threshould handle if ray is in shadow
        if (level == 0){
            bool find = (ray_depth - DepthThreshold) <= cell_z;
            if (find){
                res_uv = uv;
                return true;
            }
        }
        hit = HIT_STEPS;
        level = max(0, level - 1);
        //update new level's advance distance
        level_advance_dist[level] = t - step;
        step = float(1 << level) * RayMarchingStep;
    }
    return false;
    #else
    //this not use mipmap but trace in ndc space
    float step = RayMarchingStep;
    int total_steps = IndirectRayMaxSteps;
    int steps_count = 0;
    float t = jitter * step;
    while (++steps_count < total_steps){
        vec2 p = ori_ndc + t * dp;
        float z_over_w = ori_z_over_w + t * dz_over_w;
        float inv_w = inv_ori_w + t * dinv_w;

        vec2 ndc = swap_xy ? p.yx : p;
        vec2 uv = ndc * 0.5 + 0.5;
        if (any(notEqual(clamp(uv, vec2(0), vec2(1)), uv)))
            return false;

        float ray_depth = z_over_w / inv_w - 0.1;
        float cell_z = textureLod(ViewDepth, uv, 0).r;

        if (ray_depth > cell_z){
            res_uv = uv;
            return ray_depth - DepthThreshold <= cell_z;
        }

        t += step;
    }
    return false;
    #endif
}

//linear ray marching in camera view space
bool linearRayMarching(in float jitter, in vec3 ori, in vec3 dir, out vec2 res_uv){
    float step = RayMarchingStep;
    float t = jitter * step;
    for (int i = 0; i < IndirectRayMaxSteps; i++){
        vec3 p = ori + dir * (t + 0.5 * step);
        t += step;
        vec4 clip_pos = Proj * vec4(p, 1.0);
        vec2 uv = clip_pos.xy/clip_pos.w * 0.5 + 0.5;
        float ray_depth = p.z;
        if (any(notEqual(clamp(uv, vec2(0), vec2(1)), uv)))
        return false;
        float z = textureLod(ViewDepth, uv, 0).r;
        if (ray_depth >= z){
            res_uv = uv;
            return ray_depth - DepthThreshold <= z;
        }
    }
    return false;
}

void main() {
    // get from g-buffer
    vec4 p0 = texture(GBuffer0, iUV);
    vec4 p1 = texture(GBuffer1, iUV);
    vec3 pos = p0.xyz;
    vec2 oct_normal = vec2(p0.w, p1.w);
    vec3 normal = decodeNormal(oct_normal);
    vec2 color1 = unpackHalf2x16(floatBitsToUint(p1.x));
    vec3 albedo = vec3(color1.x, p1.g, color1.y);
    float view_z = p1.z;

    //transform to view space
    vec3 ori = vec3(View * vec4(pos, 1.0));

    //build local coord from normal for pos
    vec3 local_z = normal;
    vec3 local_y;
    if (abs(dot(local_z, vec3(1, 0, 0))) < 0.7)// approximate cos45
        local_y = cross(local_z, vec3(1, 0, 0));
    else
        local_y = cross(local_z, vec3(0, 1, 0));
        local_y = normalize(local_y);
    vec3 local_x = cross(local_y, local_z);

    //indirect
    //each fragment and frame random offset for accumulate
    vec2 sample_offset = vec2(fract(sin(FrameIndex + dot(pos.xy, vec2(12.9898, 78.233) * 2.0)) * 43758.5453));
    vec3 indirect_sum = vec3(0);
    for (int i = 0; i < IndirectSampleCount; ++i){
        vec2 rand = RawSamples[i];
        rand = fract(rand + sample_offset);

        //sample hemisphere
        float sin_theta = sqrt(1 - rand.x);
        float phi = 2 * PI * rand.y;
        float cos_theta = sqrt(rand.x);

        //local dir in z-hemisphere
        vec3 local_dir = vec3(cos_theta * cos(phi), cos_theta * sin(phi), sin_theta);
        vec3 dir = local_dir.x * local_x + local_dir.y * local_y + local_dir.z * local_z;
        dir = vec3(View * vec4(dir, 0));

        float jitter = fract(sin(FrameIndex + rand.x * 12.9898 * 2) * 43758.5453);

        //perform ray marching/trace accoding to ori and dir
        vec2 uv;
        if (bool(UseHierarchicalTrace)){
            if (hierarchicalRayTrace(jitter, ori, dir, uv)){
                //using mc cos-weight hemisphere sample, pdf is cos/PI
                vec3 direct = texture(Direct, uv).rgb;
                indirect_sum += direct;
            }
        }
        else {
            if (linearRayMarching(jitter, ori, dir, uv)){
                vec3 direct = texture(Direct, uv).rgb;
                indirect_sum += direct;
            }
        }
    }

    oFragColor = vec4(PI * indirect_sum / IndirectSampleCount, 1);
}
