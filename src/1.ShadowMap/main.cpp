#include "../common.hpp"

struct SpotLight {
    vec3f pos;
    float fade_cos_begin;

    vec3f dir;
    float fade_cos_end;

    vec3f radiance;
    float ambient;
};
static_assert(sizeof(SpotLight) == 48, "");

class NoShadow {
public:
    void initialize() {
        mesh_shader = program_t::build_from(
                shader_t<GL_VERTEX_SHADER>::from_file("asset/Lab1/mesh.vert"),
                shader_t<GL_FRAGMENT_SHADER>::from_file("asset/Lab1/mesh.frag"));

    }

    void setCamera(const fps_camera_t &camera) {
        proj_view = camera.get_view_proj();
    }

    void setPointLight(const std140_uniform_block_buffer_t<SpotLight> &light_buffer) {
        light_buffer.bind(0);
    }

    void setAlbedoMap(const texture2d_t &albedo_tex) {
        albedo_tex.bind(0);
    }

    void begin() {
        mesh_shader.bind();
        mesh_shader.set_uniform_var("ProjView", proj_view);
    }

    void render(const vertex_array_t &vao, size_t count, const mat4 &model) {
        mesh_shader.set_uniform_var("Model", model);
        vao.bind();
        GL_EXPR(glDrawElements(GL_TRIANGLES, count, GL_UNSIGNED_INT, nullptr));
        vao.unbind();
    }

    void end() {
        mesh_shader.unbind();
    }

private:
    mat4 proj_view;
    program_t mesh_shader;
};

class HardShadow {
public:
    void initialize() {
        hard_shader = program_t::build_from(
                shader_t<GL_VERTEX_SHADER>::from_file("asset/Lab1/mesh.vert"),
                shader_t<GL_FRAGMENT_SHADER>::from_file("asset/Lab1/hard.frag"));
        hard_shader.bind();
        hard_shader.set_uniform_var("ShadowMap", 0);
        hard_shader.set_uniform_var("AlbedoMap", 1);
        hard_shader.unbind();
    }

    void setCamera(const fps_camera_t &camera) {
        camera_proj_view = camera.get_view_proj();
    }

    void setPointLight(const std140_uniform_block_buffer_t<SpotLight> &light_buffer) {
        light_buffer.bind(0);
    }

    void setShadowMap(const texture2d_t &shadow_depth, const mat4 &vp) {
        shadow_depth.bind(0);
        this->light_proj_view = vp;
    }

    void setAlbedoMap(const texture2d_t &albedo_tex) {
        albedo_tex.bind(1);
    }

    void begin() {
        hard_shader.bind();
        hard_shader.set_uniform_var("ProjView", camera_proj_view);
        hard_shader.set_uniform_var("LightProjView", light_proj_view);
    }

    void render(const vertex_array_t &vao, size_t count, const mat4 &model) {
        hard_shader.set_uniform_var("Model", model);
        vao.bind();
        GL_EXPR(glDrawElements(GL_TRIANGLES, count, GL_UNSIGNED_INT, nullptr));
        vao.unbind();
    }

    void end() {
        hard_shader.unbind();
    }

private:
    program_t hard_shader;
    mat4 camera_proj_view;
    mat4 light_proj_view;

};

class PCF {
public:
    void initialize() {
        pcf_shader = program_t::build_from(
                shader_t<GL_VERTEX_SHADER>::from_file("asset/Lab1/mesh.vert"),
                shader_t<GL_FRAGMENT_SHADER>::from_file("asset/Lab1/pcf.frag"));
        pcf_shader.bind();
        pcf_shader.set_uniform_var("ShadowMap", 0);
        pcf_shader.set_uniform_var("AlbedoMap", 1);
        pcf_shader.unbind();
    }

    void setCamera(const fps_camera_t &camera) {
        camera_proj_view = camera.get_view_proj();
    }

    void setPointLight(const std140_uniform_block_buffer_t<SpotLight> &light_buffer) {
        light_buffer.bind(0);
    }

    void setShadowMap(const texture2d_t &shadow_depth, const mat4 &vp) {
        shadow_depth.bind(0);
        this->light_proj_view = vp;
    }

    void setAlbedoMap(const texture2d_t &albedo_tex) {
        albedo_tex.bind(1);
    }

    void setFilter(float filter_radius, int sample_count) {
        pcf_shader.bind();
        pcf_shader.set_uniform_var("FilterSize", filter_radius);
        pcf_shader.set_uniform_var("SampleCount", sample_count);
        pcf_shader.unbind();
    }

    void begin() {
        pcf_shader.bind();
        pcf_shader.set_uniform_var("ProjView", camera_proj_view);
        pcf_shader.set_uniform_var("LightProjView", light_proj_view);
    }

    void render(const vertex_array_t &vao, size_t count, const mat4 &model) {
        pcf_shader.set_uniform_var("Model", model);
        vao.bind();
        GL_EXPR(glDrawElements(GL_TRIANGLES, count, GL_UNSIGNED_INT, nullptr));
        vao.unbind();
    }

    void end() {
        pcf_shader.unbind();
    }

private:
    program_t pcf_shader;
    mat4 camera_proj_view;
    mat4 light_proj_view;
};

class PCSS {
public:
    void initialize() {
        pcss_shader = program_t::build_from(
                shader_t<GL_VERTEX_SHADER>::from_file("asset/Lab1/mesh.vert"),
                shader_t<GL_FRAGMENT_SHADER>::from_file("asset/Lab1/pcss.frag"));
        pcss_shader.bind();
        pcss_shader.set_uniform_var("ShadowMap", 0);
        pcss_shader.set_uniform_var("AlbedoMap", 1);
        pcss_shader.unbind();
    }

    void setCamera(const fps_camera_t &camera) {
        camera_proj_view = camera.get_view_proj();
    }

    void setPointLight(const std140_uniform_block_buffer_t<SpotLight> &light_buffer) {
        light_buffer.bind(0);
    }

    void setShadowMap(const texture2d_t &shadow_depth, const mat4 &vp) {
        shadow_depth.bind(0);
        this->light_proj_view = vp;
    }

    void setAlbedoMap(const texture2d_t &albedo_tex) {
        albedo_tex.bind(1);
    }

    void setFilter(float light_near, float light_radius, int shadow_sample_count, int block_sample_count) {
        pcss_shader.bind();
        pcss_shader.set_uniform_var("LightNearPlane", light_near);
        pcss_shader.set_uniform_var("LightRadius", light_radius);
        pcss_shader.set_uniform_var("ShadowSampleCount", shadow_sample_count);
        pcss_shader.set_uniform_var("BlockSearchSampleCount", block_sample_count);
        pcss_shader.unbind();
    }

    void begin() {
        pcss_shader.bind();
        pcss_shader.set_uniform_var("ProjView", camera_proj_view);
        pcss_shader.set_uniform_var("LightProjView", light_proj_view);
    }

    void render(const vertex_array_t &vao, size_t count, const mat4 &model) {
        pcss_shader.set_uniform_var("Model", model);
        vao.bind();
        GL_EXPR(glDrawElements(GL_TRIANGLES, count, GL_UNSIGNED_INT, nullptr));
        vao.unbind();
    }

    void end() {
        pcss_shader.unbind();
    }

private:
    program_t pcss_shader;
    mat4 camera_proj_view;
    mat4 light_proj_view;
};

class ShadowMapAPP final : public gl_app_t {
public:
    using gl_app_t::gl_app_t;

private:
    void initialize() override {
        GL_EXPR(glEnable(GL_DEPTH_TEST));
        GL_EXPR(glClearColor(0, 0, 0, 0));
        GL_EXPR(glClearDepth(1.0));
        this->vsync = window->is_vsync();
        //meshes
        auto marry = load_model_from_obj_file("asset/Marry.obj", "asset/Marry.mtl");
        auto floor = load_model_from_obj_file("asset/floor.obj", "asset/floor.mtl");
        auto load_model = [&](const model_t &model) {
            for (auto &mesh: model.meshes) {
                draw_meshes.emplace_back();
                auto &m = draw_meshes.back();
                m.vao.initialize_handle();
                m.vbo.initialize_handle();
                m.ebo.initialize_handle();
                m.vbo.reinitialize_buffer_data(mesh.vertices.data(), mesh.vertices.size(), GL_STATIC_DRAW);
                m.ebo.reinitialize_buffer_data(mesh.indices.data(), mesh.indices.size(), GL_STATIC_DRAW);
                m.vao.bind_vertex_buffer_to_attrib(attrib_var_t<vec3f>(0), m.vbo, &vertex_t::pos, 0);
                m.vao.bind_vertex_buffer_to_attrib(attrib_var_t<vec3f>(1), m.vbo, &vertex_t::normal, 1);
                m.vao.bind_vertex_buffer_to_attrib(attrib_var_t<vec2f>(2), m.vbo, &vertex_t::tex_coord, 2);
                m.vao.enable_attrib(attrib_var_t<vec3f>(0));
                m.vao.enable_attrib(attrib_var_t<vec3f>(1));
                m.vao.enable_attrib(attrib_var_t<vec2f>(2));
                m.vao.bind_index_buffer(m.ebo);
                if (mesh.material != -1) {
                    m.albedo_tex.initialize_handle();
                    m.albedo_tex.initialize_format_and_data(1, GL_RGB8, model.materials[mesh.material].albedo);
                    m.albedo_tex.set_texture_param(GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                    m.albedo_tex.set_texture_param(GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                }
            }
        };
        load_model(*marry);
        load_model(*floor);

        //depth framebuffer
        shadow_depth.shader = program_t::build_from(
                shader_t<GL_VERTEX_SHADER>::from_file("asset/Lab1/mesh.vert"),
                shader_t<GL_FRAGMENT_SHADER>::from_file("asset/Lab1/depth.frag"));
        shadow_depth.shader.bind();
        shadow_depth.shader.set_uniform_var("Model", mat4::identity());
        shadow_depth.shader.unbind();
        shadow_depth.fbo.initialize_handle();
        shadow_depth.rbo.initialize_handle();
        shadow_depth.tex.initialize_handle();
        shadow_depth.rbo.set_format(GL_DEPTH32F_STENCIL8, 4096, 4096);
        shadow_depth.fbo.attach(GL_DEPTH_STENCIL_ATTACHMENT, shadow_depth.rbo);
        shadow_depth.tex.initialize_texture(1, GL_R32F, 4096, 4096);
        shadow_depth.tex.set_texture_param(GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        shadow_depth.tex.set_texture_param(GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        shadow_depth.fbo.attach(GL_COLOR_ATTACHMENT0, shadow_depth.tex);
        assert(shadow_depth.fbo.is_complete());

        //light
        light_buffer.initialize_handle();
        light_buffer.reinitialize_buffer_data(nullptr, GL_STATIC_DRAW);

        //camera
        camera.set_position({0.0, 2.5, 5.0});
        camera.set_direction(wzz::math::deg2rad(-90.0), wzz::math::deg2rad(-10.0));
        camera.set_perspective(45.0, 0.1, 20.0);

        //renderer
        no_shadow.initialize();
        hard.initialize();

        pcf.initialize();
        pcf.setFilter(pcf_sample_radius, pcf_sample_count);

        pcss.initialize();
        pcss.setFilter(pcss_light_near, pcss_light_radius, pcss_sample_count, pcss_block_search_sample_count);
    }

    void frame() override {
        handle_events();
        // gui
        if (ImGui::Begin("Settings", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::Text("Press LCtrl to show/hide cursor");
            ImGui::Text("Use W/A/S/D/Space/LShift to move");
            ImGui::Text("FPS: %.0f", ImGui::GetIO().Framerate);
            if (ImGui::Checkbox("VSync", &vsync))
                window->set_vsync(vsync);

            ImGui::Checkbox("Rotate Light", &light.rotate_light);
            ImGui::SliderFloat("Light Y Degree", &light.light_y_degree, 0, 90);
            ImGui::SliderFloat("Light X Degree", &light.light_x_degree, 0, 360);

            static const char *shadow_types[] = {
                    "None",
                    "Hard",
                    "PCF",
                    "PCSS"
            };
            if (ImGui::BeginCombo("Shadow Type", shadow_types[static_cast<int>(shadow_type)])) {

                for (int i = 0; i < 4; i++) {
                    bool select = i == static_cast<int>(shadow_type);
                    if (ImGui::Selectable(shadow_types[i], select)) {
                        shadow_type = static_cast<ShadowType>(i);
                    }
                }

                ImGui::EndCombo();
            }
        }


        //render

        //shadow depth
        auto light_info = getLight();
        light_buffer.set_buffer_data(&light_info);
        auto light_proj_view = getLightViewProj();
        shadow_depth.fbo.bind();
        GL_EXPR(glViewport(0, 0, 4096, 4096));
        GL_EXPR(glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT));
        shadow_depth.shader.bind();
        shadow_depth.shader.set_uniform_var("ProjView", light_proj_view);
        for (auto &mesh: draw_meshes) {
            mesh.vao.bind();
            GL_EXPR(glDrawElements(GL_TRIANGLES, mesh.ebo.index_count(), GL_UNSIGNED_INT, nullptr));
            mesh.vao.unbind();
        }
        shadow_depth.shader.unbind();
        shadow_depth.fbo.unbind();
        GL_EXPR(glViewport(0, 0, window->get_window_width(), window->get_window_height()));
        GL_EXPR(glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT));

        switch (shadow_type) {
            case ShadowType::NO_SHADOW:
                frameNoShadow();
                break;
            case ShadowType::HARD:
                frameHard();
                break;
            case ShadowType::PCF:
                framePCF();
                break;
            case ShadowType::PCSS:
                framePCSS();
                break;
        }

        ImGui::End();
    }

    void destroy() override {

    }

private:
    void frameNoShadow();

    void frameHard();

    void framePCF();

    void framePCSS();

    SpotLight getLight(bool update = true);

    mat4 getLightViewProj();

private:
    bool vsync;

    struct {
        framebuffer_t fbo;
        renderbuffer_t rbo;
        texture2d_t tex;
        program_t shader;
    } shadow_depth;

    struct DrawMesh {
        vertex_array_t vao;
        vertex_buffer_t<vertex_t> vbo;
        index_buffer_t<uint32_t> ebo;
        texture2d_t albedo_tex;
    };
    std::vector<DrawMesh> draw_meshes;

    struct {
        float light_y_degree = 45;
        float light_x_degree = 0;
        float light_dist = 10;
        bool rotate_light = true;

    } light;

    std140_uniform_block_buffer_t<SpotLight> light_buffer;

    enum class ShadowType : int {
        NO_SHADOW = 0,
        HARD,
        PCF,
        PCSS
    };

    ShadowType shadow_type = ShadowType::HARD;

    NoShadow no_shadow;

    HardShadow hard;

    PCF pcf;
    float pcf_sample_radius = 10;
    int pcf_sample_count = 20;

    PCSS pcss;
    float pcss_light_near = 1.2f;
    float pcss_light_radius = 0.1f;
    int pcss_block_search_sample_count = 20;
    int pcss_sample_count = 30;

};

void ShadowMapAPP::frameNoShadow() {
    no_shadow.setCamera(camera);
    no_shadow.setPointLight(light_buffer);
    no_shadow.begin();
    for (auto &m: draw_meshes) {
        no_shadow.setAlbedoMap(m.albedo_tex);
        no_shadow.render(m.vao, m.ebo.index_count(), mat4::identity());
    }
    no_shadow.end();
}

void ShadowMapAPP::frameHard() {
    hard.setCamera(camera);
    hard.setPointLight(light_buffer);
    hard.setShadowMap(shadow_depth.tex, getLightViewProj());
    hard.begin();
    for (auto &mesh: draw_meshes) {
        hard.setAlbedoMap(mesh.albedo_tex);
        hard.render(mesh.vao, mesh.ebo.index_count(), mat4::identity());
    }
    hard.end();
}

void ShadowMapAPP::framePCF() {
    if (ImGui::SliderFloat("Sample Radius", &pcf_sample_radius, 0, 100))
        pcf.setFilter(pcf_sample_radius, pcf_sample_count);
    if (ImGui::SliderInt("Sample Count", &pcf_sample_count, 1, 1000))
        pcf.setFilter(pcf_sample_radius, pcf_sample_count);

    pcf.setCamera(camera);
    pcf.setPointLight(light_buffer);
    pcf.setShadowMap(shadow_depth.tex, getLightViewProj());
    pcf.begin();
    for (auto &mesh: draw_meshes) {
        pcf.setAlbedoMap(mesh.albedo_tex);
        pcf.render(mesh.vao, mesh.ebo.index_count(), mat4::identity());
    }
    pcf.end();
}

void ShadowMapAPP::framePCSS() {
    if (ImGui::SliderFloat("Light Near Plane Z", &pcss_light_near, 0, 5))
        pcss.setFilter(pcss_light_near, pcss_light_radius, pcss_sample_count, pcss_block_search_sample_count);
    if (ImGui::SliderFloat("Light Radius", &pcss_light_radius, 0, 10))
        pcss.setFilter(pcss_light_near, pcss_light_radius, pcss_sample_count, pcss_block_search_sample_count);
    if (ImGui::SliderInt("Shadow Sample Count", &pcss_sample_count, 0, 100))
        pcss.setFilter(pcss_light_near, pcss_light_radius, pcss_sample_count, pcss_block_search_sample_count);
    if (ImGui::SliderInt("Block Search Sample Count", &pcss_block_search_sample_count, 0, 100))
        pcss.setFilter(pcss_light_near, pcss_light_radius, pcss_sample_count, pcss_block_search_sample_count);

    pcss.setCamera(camera);
    pcss.setPointLight(light_buffer);
    pcss.setShadowMap(shadow_depth.tex, getLightViewProj());
    pcss.begin();
    for (auto &mesh: draw_meshes) {
        pcss.setAlbedoMap(mesh.albedo_tex);
        pcss.render(mesh.vao, mesh.ebo.index_count(), mat4::identity());
    }
    pcss.end();
}

SpotLight ShadowMapAPP::getLight(bool update) {
    if (update && light.rotate_light) {
        light.light_x_degree += 0.2;
        while (light.light_x_degree > 360) light.light_x_degree -= 360;
    }
    float light_x_rad = wzz::math::deg2rad(light.light_x_degree);
    float light_y_rad = wzz::math::deg2rad(light.light_y_degree);
    float dy = light.light_dist * std::sin(light_y_rad);
    float dx = light.light_dist * std::cos(light_y_rad) * std::cos(light_x_rad);
    float dz = light.light_dist * std::cos(light_y_rad) * std::sin(light_x_rad);
    vec3f light_pos = {dx, dy, dz};
    vec3f light_dst = {0, 1, 0};
    return {
            .pos = light_pos,
            .fade_cos_begin = 1,
            .dir = (light_dst - light_pos).normalized(),
            .fade_cos_end = 0.97,
            .radiance = vec3f(80),
            .ambient = 0.01
    };
}

mat4 ShadowMapAPP::getLightViewProj() {
    auto l = getLight(false);
    auto view = transform::look_at(l.pos, l.pos + l.dir, {0, 1, 0});
    auto fov = std::min(PI - 0.01, 0.01 + (2 * std::acos(l.fade_cos_end)));
    auto proj = transform::perspective(fov,1, 5, 20);
    return proj * view;
}


int main() {
    ShadowMapAPP(window_desc_t{
            .size = {1600, 900},
            .title = "ShadowMap",
            .multisamples = 4
    }).run();
}