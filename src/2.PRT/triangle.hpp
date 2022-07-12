#include "bvh.hpp"
#include <vector>

template<typename T>
using Box = std::unique_ptr<T>;

template<typename T, typename...Args>
Box<T> newBox(Args&&...args)
{
    return std::make_unique<T>(std::forward<Args>(args)...);
}

struct TriangleMesh{
    TriangleMesh(const transform3f & t,int tri_count,int v_count)
            :local_to_world(t),triangles_count(tri_count),vertices_count(v_count),
             indices(tri_count * 3),
             p(newBox<vec3f[]>(v_count)),
             n(newBox<vec3f[]>(v_count)),
             uv(newBox<vec2f[]>(v_count))
    {
        assert(triangles_count && vertices_count);
    }
    const int triangles_count;
    const int vertices_count;
    std::vector<uint32_t> indices;
    Box<vec3f[]> p;
    Box<vec3f[]> n;
    Box<vec2f[]> uv;
    transform3f local_to_world;

};

class Triangle:public Primitive{
public:
    Triangle(const RC<TriangleMesh>& m,int triangle_index,int mid)
            :mesh(m),material_id(mid)
    {
        vertex = &m->indices[triangle_index * 3];
    }

    bool intersect(const Ray& ray) const noexcept override;

    bool intersect_p(const Ray& ray,Intersection* isect) const noexcept override;

    aabb3f world_bounds() const noexcept override{
        const vec3f& A = mesh->p[vertex[0]];
        const vec3f& B = mesh->p[vertex[1]];
        const vec3f& C = mesh->p[vertex[2]];
        return aabb3f(A,B) | C;
    }

private:
    RC<TriangleMesh> mesh;
    const uint32_t* vertex = nullptr;
    int material_id = -1;
};

bool Triangle::intersect(const Ray& ray) const noexcept {
    const vec3f A = mesh->p[vertex[0]];
    const vec3f B = mesh->p[vertex[1]];
    const vec3f C = mesh->p[vertex[2]];

    const vec3f AB = B - A;
    const vec3f AC = C - A;

    vec3f s1 = cross(ray.d, AC);
    real div = dot(s1,AB);
    if(!div) return false;

    real inv_div = 1 / div;

    const vec3f AO = ray.o - A;
    real alpha = dot(AO,s1) * inv_div;
    if(alpha < 0) return false;

    vec3f s2 = cross(AO,AB);
    real beta = dot(ray.d,s2) * inv_div;
    if(beta < 0 || alpha + beta > 1)
        return false;
    real t = dot(AC,s2) * inv_div;

    if(t < ray.t_min || t > ray.t_max) return false;
    return true;
}

bool Triangle::intersect_p(const Ray &ray, Intersection *isect) const noexcept {
    const vec3f A = mesh->p[vertex[0]];
    const vec3f B = mesh->p[vertex[1]];
    const vec3f C = mesh->p[vertex[2]];

    const vec3f AB = B - A;
    const vec3f AC = C - A;

    vec3f s1 = cross(ray.d, AC);
    real div = dot(s1,AB);
    if(!div) return false;

    real inv_div = 1 / div;

    const vec3f AO = ray.o - A;
    real alpha = dot(AO,s1) * inv_div;
    if(alpha < 0) return false;

    vec3f s2 = cross(AO,AB);
    real beta = dot(ray.d,s2) * inv_div;
    if(beta < 0 || alpha + beta > 1)
        return false;
    real t = dot(AC,s2) * inv_div;

    if(t < ray.t_min || t > ray.t_max) return false;
    ray.t_max = t;

    const vec2f uvA = mesh->uv[vertex[0]];
    const vec2f uvB = mesh->uv[vertex[1]];
    const vec2f uvC = mesh->uv[vertex[2]];

    const vec3f nA = mesh->n[vertex[0]].normalized();
    const vec3f nB = mesh->n[vertex[1]].normalized();
    const vec3f nC = mesh->n[vertex[2]].normalized();

    isect->normal = nA + alpha * (nB - nA) + beta * (nC - nA);

    isect->uv = uvA + alpha * (uvB - uvA) + beta * (uvC - uvA);

    isect->pos = ray(t);

    isect->material_id = material_id;

    return true;
}

std::vector<RC<Primitive>> create_triangle_mesh(const mesh_t& mesh){
    std::vector<RC<Primitive>> triangles;
    const auto triangles_count = mesh.indices.size() / 3;
    const auto vertices_count = mesh.vertices.size();
    assert(mesh.indices.size() % 3 == 0);
    transform3f t;
    auto triangle_mesh = newRC<TriangleMesh>(t,triangles_count,vertices_count);
    triangle_mesh->indices = mesh.indices;
    for(size_t i = 0; i < vertices_count; ++i){
        const auto& vertex = mesh.vertices[i];
        triangle_mesh->p[i] = t.apply_to_point(vertex.pos);
        triangle_mesh->n[i] = t.apply_to_vector(vertex.normal);
        triangle_mesh->uv[i] = vertex.tex_coord;
    }
    for(size_t i = 0; i < triangles_count; ++i){
        triangles.emplace_back(newRC<Triangle>(triangle_mesh,i,mesh.material));
    }

    return triangles;
}
