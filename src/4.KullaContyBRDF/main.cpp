#include "../common.hpp"

float geometrySchlickGGX(float NdotV, float roughness){
    float a = roughness;
    float k = (a * a) / 2.0;

    float nom   = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return nom / denom;
}

float geometrySmith(float NdotV,float NdotL, float roughness){
    float ggx2 = geometrySchlickGGX(NdotV,roughness);
    float ggx1 = geometrySchlickGGX(NdotL,roughness);

    return ggx1 * ggx2;
}


vec2 hammersley(uint32_t i, uint32_t N)
{
    uint32_t bits = (i << 16u) | (i >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    float rdi = static_cast<float>(bits * 2.3283064365386963e-10);
    return { static_cast<float>(i) / N, rdi };
}

vec3 sampleGGX(float roughness, float u1, float u2)
{
    float alpha = roughness * roughness;
    float theta = std::atan(alpha * std::sqrt(u1) / std::sqrt(1 - u1));
    float phi   = 2 * PI * u2;

    return {std::sin(phi) * std::sin(theta),
            std::cos(phi) * std::sin(theta),
            std::cos(theta)};
}


auto generateEmiu(const vec2i& res,int spp){
    wzz::texture::image2d_t<float> Emiu(res.x,res.y);
    parallel_forrange(0,res.y,[&](int,int y){
        const float roughness = (y + 0.5f) / res.y;
        for(int x = 0; x < res.x; ++x){
            const float NdotV = (x + 0.5f) / res.x;
            const vec3f wo = vec3f(std::sqrt(std::max(0.f,1.f - NdotV * NdotV)),0,NdotV);
            float sum = 0;
            for(int i = 0; i < spp; ++i){
                const vec2 xi = hammersley(i,spp);
                const vec3 wh = sampleGGX(roughness,xi.x,xi.y);
                const vec3 wi = (2.f * wh * dot(wh,wo) - wo).normalized();
                if(wi.z <= 0) continue;
                const float G = geometrySmith(wi.z,wo.z,roughness);
                float weight = dot(wo,wh) * G / (wo.z * wh.z);
                if(std::isfinite(weight))
                    sum += weight;
            }
            Emiu.at(x,y) = std::min(sum / spp, 1.f);
        }
    },20);
    return Emiu;
}

auto generateEavg(const wzz::texture::image2d_t<float>& Emiu){
    wzz::texture::image1d_t<float> Eavg(Emiu.height());
    const float dmu = 1.f / Emiu.width();
    for(int y = 0; y < Emiu.height(); ++y){
        float sum = 0;
        for(int x = 0; x < Emiu.width(); ++x){
            const float miu = (x + 0.5f) / Emiu.width();
            sum += miu * Emiu(x,y) * dmu;
        }
        Eavg(y) = 2 * sum;
    }
    return Eavg;
}

class KullaContyBRDFApp final: public gl_app_t{
public:
    using gl_app_t::gl_app_t;
private:
    void initialize() override {
        GL_EXPR(glEnable(GL_DEPTH_TEST));
        GL_EXPR(glClearColor(0,0,0,0));
        GL_EXPR(glClearDepth(1));

        shader = program_t::build_from(
                shader_t<GL_VERTEX_SHADER>::from_file("asset/Lab4/mesh.vert"),
                shader_t<GL_FRAGMENT_SHADER>::from_file("asset/Lab4/mesh.frag"));

        camera.set_position({0,2,5});
        camera.set_perspective(60,0.1,20);
        camera.set_direction(-PI / 2,0);


        initLUT();


        params_buffer.initialize_handle();
        params_buffer.reinitialize_buffer_data(&params,GL_STATIC_DRAW);
        params_buffer.bind(0);

        auto model = load_model_from_obj_file("asset/ball.obj","asset/ball.mtl");

        std::vector<std::shared_ptr<Mesh>> mesh_rsc;
        for(auto& m:model->meshes){
            mesh_rsc.push_back(std::make_shared<Mesh>());
            auto& mesh = mesh_rsc.back();
            mesh->vao.initialize_handle();
            mesh->vbo.initialize_handle();
            mesh->ebo.initialize_handle();
            mesh->vbo.reinitialize_buffer_data(m.vertices.data(),m.vertices.size(),GL_STATIC_DRAW);
            mesh->ebo.reinitialize_buffer_data(m.indices.data(),m.indices.size(),GL_STATIC_DRAW);
            mesh->vao.bind_vertex_buffer_to_attrib(attrib_var_t<vec3f>(0),mesh->vbo,&vertex_t::pos,0);
            mesh->vao.bind_vertex_buffer_to_attrib(attrib_var_t<vec3f>(1),mesh->vbo,&vertex_t::normal,1);
            mesh->vao.bind_vertex_buffer_to_attrib(attrib_var_t<vec2f>(2),mesh->vbo,&vertex_t::tex_coord,2);
            mesh->vao.enable_attrib(attrib_var_t<vec3f>(0));
            mesh->vao.enable_attrib(attrib_var_t<vec3f>(1));
            mesh->vao.enable_attrib(attrib_var_t<vec2f>(2));
            mesh->vao.bind_index_buffer(mesh->ebo);
        }
        for(int i = 0; i < 10; i++){
            draw_models.emplace_back();
            auto& draw_model = draw_models.back();
            draw_model.meshes = mesh_rsc;
            draw_model.roughness = i * 0.1;
            draw_model.model = transform::translate((i-2.5) * 1.5,0,0) * transform::scale(0.2,0.2,0.2);
        }

        linear_sampler.initialize_handle();
        linear_sampler.set_param(GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE);
        linear_sampler.set_param(GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);
        linear_sampler.set_param(GL_TEXTURE_WRAP_R,GL_CLAMP_TO_EDGE);
        linear_sampler.set_param(GL_TEXTURE_MIN_FILTER,GL_LINEAR);
        linear_sampler.set_param(GL_TEXTURE_MAG_FILTER,GL_LINEAR);
        linear_sampler.bind(0);
        linear_sampler.bind(1);
    }

    void frame() override {
         handle_events();
        if(ImGui::Begin("Settings",nullptr,ImGuiWindowFlags_AlwaysAutoResize)){
            //ImGui::ColorEdit3("Albedo",&albedo.x);
            ImGui::ColorEdit3("Edge Tint",&params.edge_tint.x);
            ImGui::SliderFloat("Roughness",&roughness,0,1);
            ImGui::SliderFloat("Metallic",&params.metallic,0,1);
            ImGui::Checkbox("Enable KC",reinterpret_cast<bool*>(&params.enable_kc));
            if(params.enable_kc){
                ImGui::Text("Update LUT may take some times...");
                if(ImGui::InputInt2("LUT Size",&lut_res.x)){
                    initLUT();
                }
            }
            ImGui::SliderFloat("Light X Degree",&light_x_degree,0,360);
            ImGui::SliderFloat("Light Y Degree",&light_y_degree,0,90);
            ImGui::InputFloat3("Light Intensity",&params.light_intensity.x);
        }
        ImGui::End();
        params.albedo = albedo;
        params.view_pos = camera.get_position();
        params.light_dir = getLight();

        framebuffer_t::clear_buffer(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
        shader.bind();
        shader.set_uniform_var("ProjView",camera.get_view_proj());
        for(auto& m:draw_models){
            params.roughness = m.roughness;
            params_buffer.set_buffer_data(&params);
            shader.set_uniform_var("Model",m.model);
            for(auto& mesh:m.meshes){
                mesh->vao.bind();
                GL_EXPR(glDrawElements(GL_TRIANGLES,mesh->ebo.index_count(),GL_UNSIGNED_INT,nullptr));
                mesh->vao.unbind();
            }
        }

        shader.unbind();

    }

    void destroy() override {

    }

    vec3 getLight(){
        float x = -std::cos(wzz::math::deg2rad(light_y_degree)) * std::cos(wzz::math::deg2rad(light_x_degree));
        float y = -std::sin(wzz::math::deg2rad(light_y_degree));
        float z = -std::cos(wzz::math::deg2rad(light_y_degree)) * std::sin(wzz::math::deg2rad(light_x_degree));
        return {x,y,z};
    }

    void initLUT(){
        e_miu.destroy();
        e_avg.destroy();
        e_miu.initialize_handle();
        e_avg.initialize_handle();
        e_miu.initialize_texture(1,GL_R32F,lut_res.x,lut_res.y);
        e_avg.initialize_texture(1,GL_R32F,lut_res.x);

        e_miu.bind(0);
        e_avg.bind(1);
        auto Emiu = generateEmiu(lut_res,sample_count);
        auto Eavg = generateEavg(Emiu);
        e_miu.set_texture_data(lut_res.x,lut_res.y,Emiu.get_raw_data());
        e_avg.set_texture_data(lut_res.x,Eavg.get_raw_data());
    }
private:
    program_t shader;
    sampler_t linear_sampler;
    struct Mesh{
        vertex_array_t vao;
        vertex_buffer_t<vertex_t> vbo;
        index_buffer_t<uint32_t> ebo;
    };

    struct DrawModel{
        float roughness;
        mat4 model;
        std::vector<std::shared_ptr<Mesh>> meshes;
    };

    std::vector<DrawModel> draw_models;

    texture2d_t e_miu;
    texture1d_t e_avg;
    vec2i lut_res = {128,128};
    int sample_count = 256;
    struct alignas(16) Params{
        vec3f albedo = vec3(0.17,0.57,0.7);
        float roughness = 1;
        vec3 edge_tint = vec3(0.827,0.792,0.678);
        float metallic = 1;
        vec3 view_pos;
        int enable_kc = 1;
        vec3 light_dir;float pad0;
        vec3 light_intensity = vec3(1,1,1);
    }params;
    vec3 albedo = vec3(0.17,0.57,0.7);
    float roughness = 0.7;
    float light_x_degree = 90;
    float light_y_degree = 45;
    std140_uniform_block_buffer_t<Params> params_buffer;
};
int main(){
    KullaContyBRDFApp(window_desc_t{
        .size = {1280,720},
        .title = "Kulla-Conty BRDF",
        .multisamples = 4}
        ).run();
}