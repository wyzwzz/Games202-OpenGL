#include <array>
#include "../common.hpp"
#include "bvh.hpp"
#include "triangle.hpp"
#include <CGUtils/image.hpp>

enum class LightMode:int{
    NoShadow = 0,
    Shadow = 1,
    InterReflect = 2
};

class PRTApp final: public gl_app_t{
public:
    using gl_app_t::gl_app_t;
private:
    virtual void initialize() {
        GL_EXPR(glEnable(GL_DEPTH_TEST));
        GL_EXPR(glClearColor(0, 0, 0, 0));
        GL_EXPR(glClearDepth(1.0));

        prt_shader = program_t::build_from(
                shader_t<GL_VERTEX_SHADER>::from_file("asset/Lab2/mesh.vert"),
                shader_t<GL_FRAGMENT_SHADER>::from_file("asset/Lab2/mesh.frag"));

        skybox_shader = program_t::build_from(
                shader_t<GL_VERTEX_SHADER>::from_file("asset/Lab2/skybox.vert"),
                shader_t<GL_FRAGMENT_SHADER>::from_file("asset/Lab2/skybox.frag"));

        light_sh_buffer.initialize_handle();
        light_sh_buffer.reinitialize_buffer_data(nullptr, GL_STATIC_DRAW);
        light_sh_buffer.bind(0);

        loadModel("asset/Marry.obj", "asset/Marry.mtl");

        loadEnvLight("asset/newport_loft.hdr","newport loft");
        loadEnvLight("asset/small_hangar.hdr","small hangar");
        loadEnvLight("asset/environment.hdr","room");


        camera.set_position({0.0, 2.5, 5.0});
        camera.set_direction(wzz::math::deg2rad(-90.0), wzz::math::deg2rad(-10.0));
        camera.set_perspective(45.0, 0.1, 20.0);

        //skybox
        vec3f vertices[] = {
                // back face
                {-1.0f, -1.0f, -1.0f},{1.0f, 1.0f, -1.0f},{1.0f, -1.0f, -1.0f},
                {1.0f, 1.0f, -1.0f},{-1.0f, -1.0f, -1.0f},{-1.0f, 1.0f, -1.0f},
                // front face
                {-1.0f, -1.0f, 1.0f},{1.0f, -1.0f, 1.0f},{1.0f, 1.0f, 1.0f},
                {1.0f, 1.0f, 1.0f},{-1.0f, 1.0f, 1.0f},{-1.0f, -1.0f, 1.0f},
                // left face
                {-1.0f, 1.0f, 1.0f},{-1.0f, 1.0f, -1.0f},{-1.0f, -1.0f, -1.0f},
                {-1.0f, -1.0f, -1.0f},{-1.0f, -1.0f, 1.0f},{-1.0f, 1.0f, 1.0f},
                // right face
                {1.0f, 1.0f, 1.0f},{1.0f, -1.0f, -1.0f},{1.0f, 1.0f, -1.0f},
                {1.0f, -1.0f, -1.0f},{1.0f, 1.0f, 1.0f},{1.0f, -1.0f, 1.0f},
                // bottom face
                {-1.0f, -1.0f, -1.0f},{1.0f, -1.0f, -1.0f},{1.0f, -1.0f, 1.0f},
                {1.0f, -1.0f, 1.0f},{-1.0f, -1.0f, 1.0f},{-1.0f, -1.0f, -1.0f},
                // top face
                {-1.0f, 1.0f, -1.0f},{1.0f, 1.0f, 1.0f},{1.0f, 1.0f, -1.0f},
                {1.0f, 1.0f, 1.0f},{-1.0f, 1.0f, -1.0f},{-1.0f, 1.0f, 1.0f},
        };
        skybox.vao.initialize_handle();
        skybox.vbo.initialize_handle();
        skybox.vbo.reinitialize_buffer_data(vertices,36,GL_STATIC_DRAW);
        skybox.vao.bind_vertex_buffer_to_attrib(attrib_var_t<vec3f>(0),skybox.vbo,0);
        skybox.vao.enable_attrib(attrib_var_t<vec3f>(0));
    }

    virtual void frame() {
        handle_events();

        updateLight();

        //gui
        if (ImGui::Begin("Settings", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::Text("Press LCtrl to show/hide cursor");
            ImGui::Text("Use W/A/S/D/Space/LShift to move");
            ImGui::Text("FPS: %.0f", ImGui::GetIO().Framerate);

            ImGui::Text("First time switch to maybe slow");
            static const std::string env_lights[] = {
                    "room",
                    "newport loft",
                    "small hangar"
            };
            if(ImGui::BeginCombo("EnvLight Map",env_light_name.c_str())){
                for(int i = 0; i < 3; i++){
                    bool select = env_lights[i] == env_light_name;
                    if(ImGui::Selectable(env_lights[i].c_str(),select)){
                        env_light_name = env_lights[i];
                        loadEnvLight("",env_light_name);
                    }
                }
                ImGui::EndCombo();
            }


            if (ImGui::RadioButton("NoShadow", light_mode == LightMode::NoShadow)) {
                light_mode = LightMode::NoShadow;
                generateMeshSHCoefs();
            }
            ImGui::SameLine();
            if (ImGui::RadioButton("Shadow", light_mode == LightMode::Shadow)) {
                light_mode = LightMode::Shadow;
                generateMeshSHCoefs();
            }
            ImGui::SameLine();
            if (ImGui::RadioButton("InterReflect", light_mode == LightMode::InterReflect)) {
                light_mode = LightMode::InterReflect;
                generateMeshSHCoefs();
            }

            ImGui::SliderInt("SH Max Degree", &sh_degree, 0, MaxSHDegree);

            ImGui::Checkbox("Show Env Light",&show_env_light);

            ImGui::Checkbox("Rotate Env Light", &light.rotate_light);

            if (ImGui::SliderFloat("Light X Angle Degree", &light.light_x_angle_degree, 0, 360) || light.rotate_light) {
                auto tmp = light_map[env_light_name].coef;
                rotateEnvSHCoefs(wzz::math::mat3f_c::rotate_y(wzz::math::deg2rad(light.light_x_angle_degree)),
                                 tmp);
                updateEnvLightBuffer(tmp);
            }
        }
        ImGui::End();

        //render
        framebuffer_t::clear_color_depth_buffer();

        prt_shader.bind();
        prt_shader.set_uniform_var("ProjView", camera.get_view_proj());
        prt_shader.set_uniform_var("LightSHCount", sh::get_coef_count(sh_degree));
        prt_shader.set_uniform_var("Model", mat4::identity());
        for (auto &draw_mesh: draw_meshes) {
            draw_mesh.vao.bind();
            draw_mesh.vertex_sh_ssbo.bind(0);
            GL_EXPR(glDrawElements(GL_TRIANGLES, draw_mesh.ebo.index_count(), GL_UNSIGNED_INT, nullptr));

            draw_mesh.vao.unbind();
        }
        prt_shader.unbind();

        if(show_env_light){
            skybox_shader.bind();
            skybox_shader.set_uniform_var("Model",transform::rotate_y(-wzz::math::deg2rad(light.light_x_angle_degree)));
            skybox_shader.set_uniform_var("View",camera.get_view());
            skybox_shader.set_uniform_var("Proj",camera.get_proj());
            skybox.vao.bind();
            GL_EXPR(glDepthFunc(GL_LEQUAL));
            GL_EXPR(glDrawArrays(GL_TRIANGLES,0,36));
            GL_EXPR(glDepthFunc(GL_LESS));
            skybox.vao.unbind();
            skybox_shader.unbind();
        }
    }

    virtual void destroy() {

    }
private:
    void loadModel(const std::string& filename,const std::string& mtl = "");

    void loadEnvLight(const std::string& filename,const std::string& name);

    std::vector<vec3f> computeEnvSHCoefs(const wzz::texture::image2d_t<color3f>& env,
                                         int max_degree,int sample_count);

    void rotateEnvSHCoefs(const mat3& rot,std::vector<vec3f>& env_coefs);

    void generateMeshSHCoefs();

    void generateEnvSHCoefs(const wzz::texture::image2d_t<color3f>& env_light);

    std::vector<vec3f> computeMeshSHCoefs(const BVHAccel& bvh,
                                          const std::vector<vec3f> vertex_pos,
                                          const std::vector<vec3f> vertex_normal,
                                          const std::vector<vec2f> vertex_texcoord,
                                          const std::vector<material_t>& materials,
                                          int material_id,
                                          int max_degree,int sample_count_per_vertex,
                                          LightMode mode);

    void updateEnvLightBuffer(const std::vector<vec3f>& light_coefs_arr);

    void updateLight();

private:
    //some constants
    static constexpr int MaxSHDegree =  4;
    static constexpr int MaxSHCount = wzz::math::sh::get_coef_count(MaxSHDegree);
    static constexpr int SampleCountPerVertex = 1 << 10;
    static constexpr int LightSampleCount = 1 << 20;
    static constexpr float DefaultAlbedo = 0.7f;
    static constexpr int MaxInterReflectDepth = 5;
    //use for debug
    static constexpr bool UseFixedAlbedo = false;

    //current used sh degree
    int sh_degree = 2;

    struct{
        vertex_array_t vao;
        vertex_buffer_t<vec3f> vbo;
    }skybox;

    struct LightSHInfo{
        vec4 light_sh_coefs[25];
    };

    LightSHInfo light_sh_coefs;

    bool show_env_light = false;
    std::string env_light_name;
    struct LightRsc{
        texture2d_t tex;
        wzz::texture::image2d_t<color3f> img;
        std::vector<vec3f> coef;
    };
    std::unordered_map<std::string,LightRsc> light_map;


    std140_uniform_block_buffer_t<LightSHInfo> light_sh_buffer;

    LightMode light_mode = LightMode::NoShadow;

    struct{
        float light_x_angle_degree = 0.f;
        bool rotate_light = true;
    }light;

    struct DrawMesh {
        vertex_array_t vao;
        vertex_buffer_t<vertex_t> vbo;
        index_buffer_t<uint32_t> ebo;
        texture2d_t albedo_tex;
        storage_buffer_t<float> vertex_sh_ssbo;
    };

    std::vector<DrawMesh> draw_meshes;

    //cache pre-computed mesh sh coefs
    std::unordered_map<DrawMesh*,std::array<std::vector<vec3f>,3>> mp;

    std::unique_ptr<model_t> model_;

    std::unique_ptr<BVHAccel> bvh_;

    program_t prt_shader;

    program_t skybox_shader;
};

void PRTApp::loadModel(const std::string& filename,const std::string& mtl){
    auto model = wzz::model::load_model_from_obj_file(filename, mtl);
    std::vector<RC<Primitive>> primitives;
    std::vector<vec3f> vertex_pos;
    std::vector<vec3f> vertex_normal;
    for (auto &mesh: model->meshes) {
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
        m.albedo_tex.initialize_handle();
        if (mesh.material != -1) {
            m.albedo_tex.initialize_format_and_data(1, GL_RGB8, model->materials[mesh.material].albedo);
        } else {
            m.albedo_tex.initialize_format_and_data(1, GL_RGB8, wzz::texture::image2d_t<wzz::math::color3b>(
                    2, 2, wzz::math::color3b(uint8_t(255 * DefaultAlbedo))));
        }
        m.albedo_tex.set_texture_param(GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        m.albedo_tex.set_texture_param(GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        auto mesh_primitives = create_triangle_mesh(mesh);
        primitives.insert(primitives.end(), mesh_primitives.begin(), mesh_primitives.end());
        vertex_pos.insert(vertex_pos.end(), mesh.pos.begin(), mesh.pos.end());
        vertex_normal.insert(vertex_normal.end(), mesh.normal.begin(), mesh.normal.end());
    }
    std::unique_ptr<BVHAccel> bvh = std::make_unique<BVHAccel>(3);
    bvh->build(std::move(primitives));
    int count = draw_meshes.size();
    int total_count = 0;
    for (int i = 0; i < count; ++i) {
        const auto &mesh = model->meshes[i];
        auto coefs = computeMeshSHCoefs(*bvh, mesh.pos, mesh.normal, mesh.tex_coord, model->materials,
                                        UseFixedAlbedo ? -1 : mesh.material,
                                        MaxSHDegree, SampleCountPerVertex, light_mode);

        // setup gl shader storage buffer
        auto &draw_mesh = draw_meshes[i];
        draw_mesh.vertex_sh_ssbo.initialize_handle();
        draw_mesh.vertex_sh_ssbo.initialize_buffer_data(reinterpret_cast<const float *>(coefs.data()),
                                                        coefs.size() * sizeof(vec3f),
                                                        GL_DYNAMIC_STORAGE_BIT);
        //store pre-computed results for all degree
        mp[&draw_mesh][static_cast<int>(light_mode)] = std::move(coefs);
        total_count += coefs.size();
    }

    model_ = std::move(model);
    bvh_ = std::move(bvh);
    LOG_DEBUG("total mesh sh coefs count: {}", total_count);
    LOG_DEBUG("load model ok");
}

void PRTApp::generateMeshSHCoefs() {
    int count = draw_meshes.size();
    int total_count = 0;
    for (int i = 0; i < count; ++i) {
        const auto &mesh = model_->meshes[i];

        // setup gl buffer
        auto &draw_mesh = draw_meshes[i];
        auto &coefs = mp[&draw_mesh][static_cast<int>(light_mode)];
        if (coefs.empty())//compute and cache
            coefs = computeMeshSHCoefs(*bvh_, mesh.pos, mesh.normal, mesh.tex_coord, model_->materials,
                                       UseFixedAlbedo ? -1 : mesh.material,
                                       MaxSHDegree, SampleCountPerVertex, light_mode);
        total_count += coefs.size();
        draw_mesh.vertex_sh_ssbo.set_buffer_data(reinterpret_cast<const float *>(coefs.data()), 0,
                                                 coefs.size() * sizeof(vec3f));
    }
    LOG_DEBUG("total mesh sh coefs count: {}", total_count);
}

static void computeVertexSHCoefsNoShadow(const Ray& ray,
                                         const vec3f& coef,
                                         int sh_count,
                                         vec3f* res){
    if (!isfinite(coef.x) || !isfinite(coef.y) || !isfinite(coef.z)) return;
    auto table = sh::sh_linear_table<float>();
    for (int i = 0; i < sh_count; i++) {
        res[i] += coef * table[i](ray.d);
    }
}

static void computeVertexSHCoefsShadow(const BVHAccel& bvh,
                                       const Ray& ray,
                                       const vec3f& coef,
                                       int sh_count,
                                       vec3f* res){
    if (!isfinite(coef.x) || !isfinite(coef.y) || !isfinite(coef.z)) return;
    if (bvh.intersect(ray)) {
        return;
    }
    auto table = sh::sh_linear_table<float>();
    for (int i = 0; i < sh_count; i++) {
        res[i] += coef * table[i](ray.d);
    }
}

static void computeVertexSHCoefsInterReflect(const BVHAccel& bvh,
                                             const Ray& r,
                                             vec3f& coef,
                                             const std::vector<material_t>& materials,
                                             float default_albedo,
                                             int max_depth,
                                             int sh_count,
                                             vec3f* res,
                                             std::function<std::pair<float,float>()> sampler){
    auto table = sh::sh_linear_table<float>();
    Ray ray = r;
    for(int i = 0; i < max_depth; ++i){
        if(!isfinite(coef.x) || !isfinite(coef.y) || !isfinite(coef.z)) return;

        Intersection isect;
        if (!bvh.intersect_p(ray, &isect)) {
            for (int i = 0; i < sh_count; ++i)
                res[i] += coef * table[i](ray.d);
            return;
        }

        vec3f brdf = vec3f(default_albedo / PI_f);
        if(isect.material_id != -1){
            const auto& albedo = materials[isect.material_id].albedo;
            brdf = wzz::texture::linear_sampler_t::sample2d(isect.uv,[&](int x,int y)->vec3f{
                auto c =  albedo.at(x,y);
                return vec3f(static_cast<float>(c.r) / 255.f,
                             static_cast<float>(c.g) / 255.f,
                             static_cast<float>(c.b) / 255.f).map([](float x){
                                 return x / PI_f;
                             });
                },
            albedo.width(),albedo.height());
        }

        auto [u,v] = sampler();

        auto new_dir = CosineSampleHemisphere(u,v);
        if(new_dir.z <= 0) {
            for(int i = 0; i < sh_count; ++i)
                res[i] += coef * table[i](ray.d);
            return;
        }
        auto new_pdf = CosineHemispherePdf(new_dir.z);
        auto local_coord = coord3f::from_z(isect.normal);
        ray.d = local_coord.local_to_global(new_dir);
        ray.o = isect.pos + 0.0002f * isect.normal;

        coef *= brdf * abs_cos(ray.d,isect.normal) / new_pdf;
    }
}

std::vector<vec3f> PRTApp::computeMeshSHCoefs(const BVHAccel &bvh,
                                              const std::vector<vec3f> vertex_pos,
                                              const std::vector<vec3f> vertex_normal,
                                              const std::vector<vec2f> vertex_texcoord,
                                              const std::vector<material_t>& materials,
                                              int material_id,
                                              int max_degree,int sample_count_per_v, LightMode mode)
{
    int vertex_count  = vertex_pos.size();
    int sh_coef_count = sh::get_coef_count(max_degree);
    std::vector<vec3f> ret(vertex_count * sh_coef_count,0);

    auto albedo = material_id == -1 ?
                  wzz::texture::image2d_t<vec3f>(2,2,vec3f(DefaultAlbedo))
                  : materials[material_id].albedo.map([](const color3b& c){
                      return vec3f(static_cast<float>(c.r) / 255.f,
                                   static_cast<float>(c.g) / 255.f,
                                   static_cast<float>(c.b) / 255.f);
                  });

    parallel_forrange(0,vertex_count,[&](int thread_index,int vertex_id){
        const auto& vertex_p  = vertex_pos[vertex_id];
        const auto& vertex_n  = vertex_normal[vertex_id];
        const auto& vertex_uv = vertex_texcoord[vertex_id];
        const auto brdf = wzz::texture::linear_sampler_t::sample2d(albedo,vertex_uv) * ::invPI;

        auto res = &ret[vertex_id * sh_coef_count];

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> rng(0.0, 1.0);


        const auto ori = vertex_p + vertex_n * 0.0002f;
        const auto local_coord = coord3f::from_z(vertex_n);
        for(int i = 0; i < sample_count_per_v; ++i){
            auto u = rng(gen);
            auto v = rng(gen);

            //sample cos-weight hemisphere
            auto ldir = CosineSampleHemisphere(u,v);
            if(ldir.z <= 0) continue;
            auto pdf = CosineHemispherePdf(ldir.z);

            const vec3f dir = local_coord.local_to_global(ldir);
            const Ray ray(ori,dir);

            vec3f coef = brdf * abs_cos(dir,vertex_n) / pdf;

            if(mode == LightMode::NoShadow){
                computeVertexSHCoefsNoShadow(ray,coef,sh_coef_count,res);
            }
            else if(mode == LightMode::Shadow){
                computeVertexSHCoefsShadow(bvh,ray,coef,sh_coef_count,res);
            }
            else{
                computeVertexSHCoefsInterReflect(bvh,ray,coef,materials,DefaultAlbedo,
                                                 MaxInterReflectDepth,sh_coef_count,res,[&](){
                    return std::make_pair(rng(gen),rng(gen));
                });
            }
        }
        float weight = 1.f / sample_count_per_v;
        for(int i = 0; i < sh_coef_count; ++i)
            res[i] *= weight;
        //ok
    },20);
    return ret;
}

void PRTApp::loadEnvLight(const std::string &filename,const std::string& name) {
    assert(!name.empty());
    env_light_name = name;
    if(light_map.count(name) == 0){
        auto& rsc = light_map[name];
        rsc.img = wzz::texture::image2d_t<color3f>(wzz::image::load_rgb_from_hdr_file(filename));
        rsc.tex.initialize_handle();
        rsc.tex.initialize_format_and_data(1,GL_RGB32F,rsc.img);
        rsc.tex.set_texture_param(GL_TEXTURE_MAG_FILTER,GL_LINEAR);
        rsc.tex.set_texture_param(GL_TEXTURE_MIN_FILTER,GL_LINEAR);
        generateEnvSHCoefs(rsc.img);
        assert(light_map.count(name) == 1);
    }
    light_map[name].tex.bind(0);
    updateEnvLightBuffer(light_map[name].coef);
}

void PRTApp::generateEnvSHCoefs(const wzz::texture::image2d_t<color3f>& env_light) {
    light_map[env_light_name].coef = computeEnvSHCoefs(env_light, MaxSHDegree, LightSampleCount);

    updateEnvLightBuffer(light_map[env_light_name].coef);

    LOG_DEBUG("light sh coefs ok");
}

std::vector<vec3f> PRTApp::computeEnvSHCoefs(const wzz::texture::image2d_t<color3f> &env, int max_degree, int sample_count) {
    std::vector<vec3f> ret;

    auto coefs = wzz::math::sh::project_func_to_sh<float,3>(max_degree,[&](float phi,float theta)->wzz::math::tvec<float,3>{
        //from local coord to view coord
        assert(theta >= 0 && theta <= PI_f);
        auto r = std::sin(theta);
        auto x = r * std::cos(phi);
        auto y = r * std::sin(phi);// -1 ~ 1
        auto z = std::cos(theta);
        auto _theta = std::acos(y);
        auto _phi = (x == 0 && z == 0) ? 0.f : std::atan2(z, x);
        while (_phi < 0) _phi += 2 * PI_f;
        while (_phi > 2 * PI_f) _phi -= 2 * PI_f;
        assert(_phi >= 0 && _phi <= 2 * PI_f);
        auto color = wzz::texture::linear_sampler_t::sample2d(env, vec2f(_phi / (2.f * PI_f), _theta / PI_f));
        return {color.r, color.g, color.b};
    },sample_count);

    for(auto& x:coefs) ret.emplace_back(x.data[0],x.data[1],x.data[2]);
    return ret;
}

void PRTApp::updateEnvLightBuffer(const std::vector<vec3f> &light_coefs_arr) {
    //clear
    for (int i = 0; i < MaxSHCount; ++i)
        light_sh_coefs.light_sh_coefs[i] = vec4(0);

    //copy
    for (int i = 0; i < sh::get_coef_count(sh_degree); i++)
        light_sh_coefs.light_sh_coefs[i] = vec4f(light_coefs_arr[i].x,
                                                 light_coefs_arr[i].y,
                                                 light_coefs_arr[i].z, 0.0);

    //upload
    light_sh_buffer.set_buffer_data(&light_sh_coefs);
}

void PRTApp::updateLight() {
    if (light.rotate_light) {
        light.light_x_angle_degree += 1.f;
        if (light.light_x_angle_degree > 360) light.light_x_angle_degree -= 360;
    }
}

void PRTApp::rotateEnvSHCoefs(const mat3 &rot, std::vector<vec3f> &env_coefs) {
    int n = env_coefs.size();
    int max_degree = static_cast<int>(std::sqrt(n)) - 1;
    assert(n == sh::get_coef_count(max_degree));
    std::vector<float> coefs;

    for (int i = 0; i <= max_degree; i++) {
        int order_coef_count = 2 * i + 1;
        coefs.resize(order_coef_count);

        // rotate rgb channel separately
        for (int c = 0; c < 3; c++) {
            int offset = sh::get_coef_count(i - 1);
            for (int j = 0; j < order_coef_count; j++)
                coefs[j] = env_coefs[offset + j][c];

            sh::rotate_sh_coefs(rot, i, coefs.data());

            for (int j = 0; j < order_coef_count; j++)
                env_coefs[offset + j][c] = coefs[j];
        }
    }
    LOG_DEBUG("rotate env light ok");
}

int main() {
    PRTApp(window_desc_t{
            .size = {1280, 720},
            .title = "PRT",
            .multisamples = 4
    }).run();
}