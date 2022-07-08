#include "../common.hpp"
#include <CGUtils/image.hpp>
#define set_uniform_var set_uniform_var_unchecked
struct DirectionalLight{
    vec3f direction;
    float pad0;
    vec3f radiance;
    float pad1;
    mat4 proj_view;
};

struct Vertex{
    vec3f pos;
    vec3f normal;
    vec3f tangent;
    vec2f tex_coord;
};

struct DrawMesh{
    mat4 model;
    vertex_array_t vao;
    vertex_buffer_t<Vertex> vbo;
    index_buffer_t<uint32_t> ebo;
    std::shared_ptr<texture2d_t> albedo;
    std::shared_ptr<texture2d_t> normal;
};

class GBufferGenerator{
public:
    void initialize(int w,int h){
        shader = program_t::build_from(
                shader_t<GL_VERTEX_SHADER>::from_file("asset/Lab3/gbuffer.vert"),
                shader_t<GL_FRAGMENT_SHADER>::from_file("asset/Lab3/gbuffer.frag"));
        shader.bind();
        shader.set_uniform_var("AlbedoMap",0);
        shader.set_uniform_var("NormalMap",1);
        shader.unbind();
        gbuffer.g0.initialize_handle();
        gbuffer.g1.initialize_handle();
        gbuffer.g0.initialize_texture(1,GL_RGBA32F,w,h);
        gbuffer.g1.initialize_texture(1,GL_RGBA32F,w,h);
    }
    void setCamera(const mat4& view,const mat4& proj){
        this->view = view;
        this->proj = proj;
    }
    void setFramebuffer(const framebuffer_t& fbo){
        assert(fbo.is_complete());
        fbo.attach(GL_COLOR_ATTACHMENT0,gbuffer.g0);
        fbo.attach(GL_COLOR_ATTACHMENT1,gbuffer.g1);
        GLenum targets[2] = {GL_COLOR_ATTACHMENT0,GL_COLOR_ATTACHMENT1};
        GL_EXPR(glDrawBuffers(2,targets));
        framebuffer_t::clear_buffer(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    }
    void setSample(const sampler_t& sampler){
        sampler.bind(0);
        sampler.bind(1);
    }
    void begin(){
        shader.bind();
    }
    void end(){
        shader.unbind();
    }
    void render(const DrawMesh& m){
        shader.set_uniform_var("Model",m.model);
        shader.set_uniform_var("ViewModel",view * m.model);
        shader.set_uniform_var("ProjViewModel",proj * view * m.model);
        m.albedo->bind(0);
        m.normal->bind(1);
        m.vao.bind();
        GL_EXPR(glDrawElements(GL_TRIANGLES,m.ebo.index_count(),GL_UNSIGNED_INT,nullptr));
        m.vao.unbind();
    }
    const texture2d_t& getGBuffer(int index) const{
        assert(index == 0 || index == 1);
        return index == 0 ? gbuffer.g0 : gbuffer.g1;
    }
private:
    mat4 view,proj;
    struct{
        texture2d_t g0;
        texture2d_t g1;
    }gbuffer;
    program_t shader;
};

class MipMapGenerator{
public:
    void initialize(int w,int h){
        width = w;
        height = h;
        copy_viewz_shader = program_t::build_from(
                shader_t<GL_COMPUTE_SHADER>::from_file("asset/Lab3/viewz.comp"));

        mipmap_shader = program_t::build_from(
                shader_t<GL_COMPUTE_SHADER>::from_file("asset/Lab3/mipmap.comp"));
        mipmap.initialize_handle();
        levels = computeLevels(w,h);
        mipmap.initialize_texture(levels,GL_R32F,width,height);
        LOG_INFO("mipmap levels: {}",levels);
    }
    int computeLevels(int w,int h){
        int levels = 1;
        while(w > 1 || h > 1){
            levels += 1;
            w >>= 1;
            h >>= 1;
        }
        return levels;
    }
    void generate(const texture2d_t& g1){
        constexpr int group_thread_size_x = 16;
        constexpr int group_thread_size_y = 16;
        auto get_group_size = [&](int x,int y)->vec2i{
            return vec2i{
                    (x + group_thread_size_x - 1) / group_thread_size_x,
                    (y + group_thread_size_y - 1) / group_thread_size_y
            };
        };
        auto group_size = get_group_size(width,height);
        copy_viewz_shader.bind();
        g1.bind_image(0,0,GL_READ_ONLY,GL_RGBA32F);
        mipmap.bind_image(1,0,GL_READ_WRITE,GL_R32F);
        GL_EXPR(glDispatchCompute(group_size.x,group_size.y,1));
        copy_viewz_shader.unbind();

        mipmap_shader.bind();
        vec2i pre_res = {width,height};
        for(int i = 1; i < levels; ++i){
            vec2i cur_res = {
                    std::max(pre_res.x / 2,1),
                    std::max(pre_res.y / 2,1)};
            mipmap_shader.set_uniform_var("PreSize",vec2f(pre_res.x,pre_res.y));
            mipmap_shader.set_uniform_var("CurSize",vec2f(cur_res.x,cur_res.y));
            mipmap.bind_image(0,i - 1,GL_READ_ONLY,GL_R32F);
            mipmap.bind_image(1,i,GL_READ_WRITE,GL_R32F);
            auto gs = get_group_size(cur_res.x,cur_res.y);
            GL_EXPR(glDispatchCompute(gs.x,gs.y,1));
            pre_res = cur_res;
        }
        mipmap_shader.unbind();
    }
    const texture2d_t& getMipMap() const{
        return mipmap;
    }
private:
    int width = 0,height = 0;
    program_t copy_viewz_shader;
    texture2d_t mipmap;
    int levels = 1;
    program_t mipmap_shader;
};

class DirectRenderer{
public:
    void initialize(int w,int h){
        shader = program_t::build_from(
                shader_t<GL_VERTEX_SHADER>::from_file("asset/Lab3/quad.vert"),
                shader_t<GL_FRAGMENT_SHADER>::from_file("asset/Lab3/direct.frag"));
        shader.bind();
        shader.set_uniform_var_unchecked("ShadowMap",0);
        shader.set_uniform_var_unchecked("GBuffer0",1);
        shader.set_uniform_var_unchecked("GBuffer1",2);
        shader.unbind();
        color.initialize_handle();
        color.initialize_texture(1,GL_RGBA8,w,h);
        vao.initialize_handle();
    }
    void setFramebuffer(const framebuffer_t& fbo){
        fbo.attach(GL_COLOR_ATTACHMENT0,color);
        GL_EXPR(glDrawBuffer(GL_COLOR_ATTACHMENT0));
        framebuffer_t::clear_buffer(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    }
    void setLight(const std140_uniform_block_buffer_t<DirectionalLight>& light_buffer){
        light_buffer.bind(0);
    }
    void setSampler(const sampler_t& sampler){
        sampler.bind(0);
        sampler.bind(1);
        sampler.bind(2);
    }
    void render(
            const texture2d_t& shadow_map,
            const texture2d_t& gbuffer0,
            const texture2d_t& gbuffer1){

        shader.bind();
        shadow_map.bind(0);
        gbuffer0.bind(1);
        gbuffer1.bind(2);
        vao.bind();
        GL_EXPR(glDepthFunc(GL_LEQUAL));

        GL_EXPR(glDrawArrays(GL_TRIANGLE_STRIP,0,4));

        GL_EXPR(glDepthFunc(GL_LESS));
        vao.unbind();
        shader.unbind();
    }

    const texture2d_t& getColor(){
        return color;
    }
private:
    vertex_array_t vao;
    program_t shader;
    texture2d_t color;
};

class IndirectRenderer{
public:
    void initialize(int w,int h){

    }
private:
    program_t shader;
    texture2d_t color;
};

class FinalRenderer{
public:
    void initialize(int w,int t){

    }
private:
    program_t shader;
    texture2d_t color;
};

/**
 * @param BA vertex B' pos minus vertex A' pos
 */
inline vec3f ComputeTangent(const vec3f& BA,const vec3f CA,
                            const vec2f& uvBA,const vec2f& uvCA,
                            const vec3f& normal)
{
    const float m00 = uvBA.x, m01 = uvBA.y;
    const float m10 = uvCA.x, m11 = uvCA.y;
    const float det = m00 * m11 - m01 * m10;
    if(std::abs(det) < 0.0001f)
        return wzz::math::tcoord3<float>::from_z(normal).x;
    const float inv_det = 1 / det;
    return (m11 * inv_det * BA - m01 * inv_det * CA).normalized();
}

class SSRTApp final:public gl_app_t{
public:
    using gl_app_t::gl_app_t;
private:
    virtual void initialize() {
        SET_LOG_LEVEL_DEBUG
        GL_EXPR(glEnable(GL_DEPTH_TEST));
        GL_EXPR(glClearColor(0, 0, 0, 0));
        GL_EXPR(glClearDepth(1.0));

        auto normal = wzz::image::load_rgb_from_file("asset/normal.jpg");
        auto albedo = wzz::image::load_rgb_from_file("asset/albedo.jpg");
        std::shared_ptr<texture2d_t> normal_tex = std::make_shared<texture2d_t>(true);
        std::shared_ptr<texture2d_t> albedo_tex = std::make_shared<texture2d_t>(true);
        normal_tex->initialize_format_and_data(1,GL_RGB8,normal);
        albedo_tex->initialize_format_and_data(1,GL_RGB8,albedo);
        auto load_mesh = [&](const std::string& obj_filename,const mat4& model)
        {
            auto meshes = load_mesh_from_obj_file(obj_filename);
            std::vector<DrawMesh> draw_meshes;
            for(auto& mesh:meshes){
                draw_meshes.emplace_back();
                auto& draw_mesh = draw_meshes.back();
                draw_mesh.model = model;
                draw_mesh.albedo = albedo_tex;
                draw_mesh.normal = normal_tex;
                draw_mesh.vao.initialize_handle();
                draw_mesh.vbo.initialize_handle();
                draw_mesh.ebo.initialize_handle();
                std::vector<Vertex> vertices;
                int n = mesh.indices.size() / 3;
                auto& indices = mesh.indices;
                for(int i = 0; i < n; i++){
                    const auto& A = mesh.vertices[indices[3 * i]];
                    const auto& B = mesh.vertices[indices[3 * i + 1]];
                    const auto& C = mesh.vertices[indices[3 * i + 2]];
                    const auto BA = B.pos - A.pos;
                    const auto CA = C.pos - A.pos;
                    const auto uvBA = B.tex_coord - A.tex_coord;
                    const auto uvCA = C.tex_coord - A.tex_coord;
                    for(int j = 0; j < 3; j++){
                        const auto& v = mesh.vertices[indices[3 * i + j]];
                        vertices.push_back({
                                                   v.pos,v.normal, ComputeTangent(BA,CA,uvBA,uvCA,v.normal),v.tex_coord
                                           });
                    }
                }
                assert(vertices.size() == mesh.vertices.size());
                draw_mesh.vbo.reinitialize_buffer_data(vertices.data(),vertices.size(),GL_STATIC_DRAW);
                draw_mesh.ebo.reinitialize_buffer_data(indices.data(),indices.size(),GL_STATIC_DRAW);
                draw_mesh.vao.bind_vertex_buffer_to_attrib(attrib_var_t<vec3f>(0), draw_mesh.vbo, &Vertex::pos, 0);
                draw_mesh.vao.bind_vertex_buffer_to_attrib(attrib_var_t<vec3f>(1), draw_mesh.vbo, &Vertex::normal, 1);
                draw_mesh.vao.bind_vertex_buffer_to_attrib(attrib_var_t<vec3f>(2), draw_mesh.vbo, &Vertex::tangent, 2);
                draw_mesh.vao.bind_vertex_buffer_to_attrib(attrib_var_t<vec2f>(3), draw_mesh.vbo, &Vertex::tex_coord, 3);
                draw_mesh.vao.enable_attrib(attrib_var_t<vec3f>(0));
                draw_mesh.vao.enable_attrib(attrib_var_t<vec3f>(1));
                draw_mesh.vao.enable_attrib(attrib_var_t<vec3f>(2));
                draw_mesh.vao.enable_attrib(attrib_var_t<vec2f>(3));
                draw_mesh.vao.bind_index_buffer(draw_mesh.ebo);
            }
            return draw_meshes;
        };
        draw_meshes = load_mesh("asset/cave.obj",mat4::identity());

        linear_clamp_sampler.initialize_handle();
        linear_clamp_sampler.set_param(GL_TEXTURE_MIN_FILTER,GL_LINEAR);
        linear_clamp_sampler.set_param(GL_TEXTURE_MAG_FILTER,GL_LINEAR);
        linear_clamp_sampler.set_param(GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE);
        linear_clamp_sampler.set_param(GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);
        linear_clamp_sampler.set_param(GL_TEXTURE_WRAP_R,GL_CLAMP_TO_EDGE);

        nearest_clamp_sampler.initialize_handle();
        nearest_clamp_sampler.set_param(GL_TEXTURE_MIN_FILTER,GL_NEAREST);
        nearest_clamp_sampler.set_param(GL_TEXTURE_MAG_FILTER,GL_NEAREST);
        nearest_clamp_sampler.set_param(GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE);
        nearest_clamp_sampler.set_param(GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);
        nearest_clamp_sampler.set_param(GL_TEXTURE_WRAP_R,GL_CLAMP_TO_EDGE);

        //shadow
        shadow_shader = program_t::build_from(
                shader_t<GL_VERTEX_SHADER>::from_file("asset/Lab3/mesh.vert"),
                shader_t<GL_FRAGMENT_SHADER>::from_file("asset/Lab3/depth.frag"));
        shadow.fbo.initialize_handle();
        shadow.rbo.initialize_handle();
        shadow.depth_tex.initialize_handle();
        shadow.rbo.set_format(GL_DEPTH32F_STENCIL8,4096,4096);
        shadow.fbo.attach(GL_DEPTH_STENCIL_ATTACHMENT,shadow.rbo);
        shadow.depth_tex.initialize_texture(1,GL_R32F,4096,4096);
        shadow.depth_tex.set_texture_param(GL_TEXTURE_MAG_FILTER,GL_NEAREST);
        shadow.depth_tex.set_texture_param(GL_TEXTURE_MIN_FILTER,GL_NEAREST);
        shadow.fbo.attach(GL_COLOR_ATTACHMENT0,shadow.depth_tex);
        assert(shadow.fbo.is_complete());

        //light
        light_buffer.initialize_handle();
        light_buffer.reinitialize_buffer_data(nullptr,GL_STATIC_DRAW);

        //camera
        camera.set_position({3.54,0.58,1.55});
        camera.set_direction(3.19,0.17);
        camera.set_perspective(60.0,0.1,100.0);

        //gbuffer
        off_frame.fbo.initialize_handle();
        off_frame.rbo.initialize_handle();
        off_frame.rbo.set_format(GL_DEPTH32F_STENCIL8, window->get_window_width(), window->get_window_height());
        off_frame.fbo.attach(GL_DEPTH_STENCIL_ATTACHMENT, off_frame.rbo);
        assert(off_frame.fbo.is_complete());
        gbuffer0.initialize(window->get_window_width(),window->get_window_height());
        gbuffer1.initialize(window->get_window_width(),window->get_window_height());
        gbuffer = &gbuffer0;

        //mipmap
        mipmap.initialize(window->get_window_width(),window->get_window_height());

        //direct
        direct.initialize(window->get_window_width(),window->get_window_height());
    }

    virtual void frame() {

        auto last_camera_proj_view = camera.get_view_proj();

        handle_events();
        //gui
        if (ImGui::Begin("Settings", nullptr, ImGuiWindowFlags_AlwaysAutoResize)){
            ImGui::Text("Press LCtrl to show/hide cursor");
            ImGui::Text("Use W/A/S/D/Space/LShift to move");
            ImGui::Text("FPS: %.0f", ImGui::GetIO().Framerate);


        }

        //render
        //shadow
        DirectionalLight directional_light = getLightProjView();
        auto light_proj_view = directional_light.proj_view;
        light_buffer.set_buffer_data(&directional_light);

        shadow.fbo.bind();
        shadow.fbo.clear_buffer(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
        GL_EXPR(glViewport(0,0,4096,4096));

        shadow_shader.bind();
        for(auto& draw_mesh:draw_meshes){
            draw_mesh.vao.bind();
            shadow_shader.set_uniform_var("Model",draw_mesh.model);
            shadow_shader.set_uniform_var("ProjView",light_proj_view);
            GL_EXPR(glDrawElements(GL_TRIANGLES,draw_mesh.ebo.index_count(),GL_UNSIGNED_INT,nullptr));
            draw_mesh.vao.unbind();
        }
        shadow_shader.unbind();
        shadow.fbo.unbind();
        GL_EXPR(glViewport(0,0,window->get_window_width(),window->get_window_height()))

        //gbuffer
        auto prev_gbuffer = gbuffer;
        gbuffer = gbuffer == &gbuffer0 ? &gbuffer1 : &gbuffer0;

        off_frame.fbo.bind();
        gbuffer->setFramebuffer(off_frame.fbo);
        gbuffer->setCamera(camera.get_view(),camera.get_proj());
        gbuffer->setSample(linear_clamp_sampler);
        gbuffer->begin();
        for(auto& m:draw_meshes){
            gbuffer->render(m);
        }
        gbuffer->end();
        off_frame.fbo.unbind();

        //mipmap
        mipmap.generate(gbuffer->getGBuffer(1));

        //direct
        off_frame.fbo.bind();
        direct.setFramebuffer(off_frame.fbo);
        direct.setLight(light_buffer);
        direct.setSampler(nearest_clamp_sampler);
        direct.render(shadow.depth_tex,
                      gbuffer->getGBuffer(0),
                      gbuffer->getGBuffer(1));
        off_frame.fbo.unbind();



        ImGui::End();
    }

    virtual void destroy() { }
private:
    DirectionalLight getLightProjView();
private:

    bool enable_direct = true;
    bool enable_indirect = true;

    int sample_count = 6;
    int max_ray_march_step_count = 32;


    //light
    struct{
        float light_x_degree = 206;
        float light_y_degree = 74;
        float light_radiance = 15;
    }light;
    std140_uniform_block_buffer_t<DirectionalLight> light_buffer;

    //mesh

    std::vector<DrawMesh> draw_meshes;

    sampler_t linear_clamp_sampler;
    sampler_t nearest_clamp_sampler;

    //shadow map
    struct{
        framebuffer_t fbo;
        renderbuffer_t rbo;
        texture2d_t depth_tex;
    }shadow;
    program_t shadow_shader;

    struct{
        framebuffer_t fbo;
        renderbuffer_t rbo;
    }off_frame;
    GBufferGenerator gbuffer0;
    GBufferGenerator gbuffer1;

    GBufferGenerator* gbuffer = nullptr;

    //mipmap
    MipMapGenerator mipmap;

    //direct
    DirectRenderer direct;

    //indirect
    IndirectRenderer indirect;

    //final
    FinalRenderer final;
};

DirectionalLight SSRTApp::getLightProjView() {
    const vec3f light_target = {0,0,0};
    vec3f direction = {
        -std::cos(wzz::math::deg2rad(light.light_y_degree)) *
         std::cos(wzz::math::deg2rad(light.light_x_degree)),
        -std::sin(wzz::math::deg2rad(light.light_y_degree)),
        -std::cos(wzz::math::deg2rad(light.light_y_degree)) *
         std::sin(wzz::math::deg2rad(light.light_x_degree))
    };

    return {direction,0,light.light_radiance,0,
            transform::orthographic(-5,5,-5,5,3,40) * transform::look_at(light_target - 15.f * direction,light_target,{1,0,0})};
}

int main() {
    SSRTApp(window_desc_t{
            .size = {1600, 900},
            .title = "ShadowMap",
            .multisamples = 4
    }).run();
}