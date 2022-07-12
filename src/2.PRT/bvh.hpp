#pragma once
#include "../common.hpp"
#include <CGUtils/geometry.hpp>
#include <CGUtils/memory.hpp>
#include <stack>
using namespace wzz::geometry;
using real = float;
class Ray{
public:
    Ray(const vec3f& o, const vec3f& d,real t_min = 0,real t_max = std::numeric_limits<real>::max())
            :o(o),d(d.normalized()),t(0),t_min(t_min),t_max(t_max)
    {}
    Ray()
            :Ray(vec3f(),vec3f(1,0,0))
    {}

    vec3f operator()(real time) const{
        return o + d * time;
    }

    bool between(real t) const{
        return t_min <= t && t <= t_max;
    }

    vec3f o;
    vec3f d;
    real t;
    mutable real t_min;
    mutable real t_max;
};
struct Intersection{
    vec3f pos;
    vec3f normal;
    vec2f uv;
    int material_id = -1;
    float t = 0;
};
class Primitive{
public:
    virtual aabb3f world_bounds() const noexcept = 0;

    virtual bool intersect(const Ray& ray) const noexcept = 0;

    virtual bool intersect_p(const Ray& ray,Intersection* isect) const noexcept = 0;
};
template<typename T>
using RC = std::shared_ptr<T>;
template<typename T, typename...Args>
RC<T> newRC(Args&&...args)
{
    return std::make_shared<T>(std::forward<Args>(args)...);
}
using MemoryArena = wzz::alloc::memory_arena_t;
namespace {

    struct BVHPrimitiveInfo{
        BVHPrimitiveInfo(){}
        BVHPrimitiveInfo(size_t index,const aabb3f& bounds)
                :primitive_index(index),bounds(bounds),centroid(0.5f * bounds.low + 0.5f * bounds.high)
        {}
        size_t primitive_index = 0;
        aabb3f bounds;
        vec3f centroid;
    };
    struct BVHBuildNode{
        void init_leaf(size_t first,size_t count,const aabb3f& b){
            first_prim_offset = first;
            primitive_count = count;
            bounds = b;
            left = right = nullptr;

        }

        void init_interior(int axis,BVHBuildNode* l,BVHBuildNode* r){
            assert(l && r);
            left = l;
            right = r;
            split_axis = axis;
            bounds = l->bounds | r->bounds;
            primitive_count = 0;
        }
        bool is_leaf_node() const{
            return primitive_count > 0;
        }
        aabb3f bounds;
        BVHBuildNode* left = nullptr;
        BVHBuildNode* right = nullptr;
        int split_axis;
        size_t first_prim_offset;
        size_t primitive_count = 0;//if > 0 then it is a leaf node
    };
    struct LinearBVHNode{
        aabb3f bounds;
        union {
            int primitive_offset;
            int second_child_offset;
        };
        uint16_t primitive_count;
        uint8_t axis;
        uint8_t pad[1];
        bool is_leaf_node() const{
            return primitive_count > 0;
        }
    };
    static_assert(sizeof(LinearBVHNode) == 32,"");

}

//using SAH
class BVHAccel{
public:


    BVHAccel(int max_leaf_prims):max_leaf_prims(max_leaf_prims){}

    ~BVHAccel() {

    }

    static bool rayIntersectWithAABB(const aabb3f& box,
                              const vec3f& ori,const vec3f& inv_dir,
                              float t_min,float t_max,
                              const int dir_is_neg[3]){
        real tMin = (box[dir_is_neg[0]].x - ori.x) * inv_dir.x;
        real tMax = (box[1 - dir_is_neg[0]].x - ori.x) * inv_dir.x;
        real tyMin = (box[dir_is_neg[1]].y - ori.y) * inv_dir.y;
        real tyMax = (box[1 - dir_is_neg[1]].y - ori.y) * inv_dir.y;

        tMax *= 1 + 0.0000001;
        tyMax *= 1 + 0.0000001;
        if (tMin > tyMax || tyMin > tMax) return false;
        if (tyMin > tMin) tMin = tyMin;
        if (tyMax < tMax) tMax = tyMax;

        real tzMin = (box[dir_is_neg[2]].z - ori.z) * inv_dir.z;
        real tzMax = (box[1 - dir_is_neg[2]].z - ori.z) * inv_dir.z;

        tzMax *= 1 + 0.0000001;
        if (tMin > tzMax || tzMin > tMax) return false;
        if (tzMin > tMin) tMin = tzMin;
        if (tzMax < tMax) tMax = tzMax;
        return (tMin < t_max) && (tMax > 0);
    }


    void build(std::vector<RC<Primitive>> prims) {
        if(prims.empty()) return;
        primitives = std::move(prims);
        size_t n = primitives.size();
        std::vector<BVHPrimitiveInfo> primitive_infos;
        primitive_infos.reserve(n);
        for(size_t i = 0; i < n; i++){
            primitive_infos.emplace_back(i,primitives[i]->world_bounds());
        }

        MemoryArena arena(1<<20);
        size_t total_nodes_count = 0;
        std::vector<RC<Primitive>> ordered_prims;
        ordered_prims.reserve(n);

        BVHBuildNode* root = nullptr;

        root = recursive_build(primitive_infos,ordered_prims,0,n,total_nodes_count,arena);
        assert(root);
        primitives = std::move(ordered_prims);

        linear_nodes = wzz::alloc::aligned_alloc<LinearBVHNode>(total_nodes_count);
        size_t offset = 0;
        flatten_bvh_tree(root,offset);
        assert(offset == total_nodes_count);

        LOG_INFO("bvh tree build node count: {}",total_nodes_count);
    }

    virtual aabb3f world_bound() const noexcept{
        if(!linear_nodes) return aabb3f ();
        return linear_nodes[0].bounds;
    }

    bool intersect(const Ray& ray) const noexcept{
        if(!linear_nodes) return false;

        vec3f inv_dir(1.0 / ray.d.x, 1.0 / ray.d.y, 1.0 / ray.d.z);
        int dir_is_neg[3] = {inv_dir.x < 0,inv_dir.y < 0,inv_dir.z < 0};

        std::stack<size_t> s;
        s.push(0);
        while(!s.empty()){
            assert(s.size() < 64);
            auto node_index = s.top();
            s.pop();
            const auto node = linear_nodes + node_index;
            if(!rayIntersectWithAABB(node->bounds,ray.o,inv_dir,ray.t_min,ray.t_max,dir_is_neg)) continue;
            if(node->is_leaf_node()){
                for(int i = 0; i < node->primitive_count; i++){
                    if(primitives[node->primitive_offset + i]->intersect(ray)){
                        //if find one just return true
                        return true;
                    }
                }
            }
            else{
                if(dir_is_neg[node->axis]){
                    //second child first
                    s.push(node_index + 1);
                    s.push(node->second_child_offset);
                }
                else{
                    s.push(node->second_child_offset);
                    s.push(node_index + 1);
                }
            }
        }
        return false;
    }

    bool intersect_p(const Ray& ray,Intersection* isect) const noexcept {
        if(!linear_nodes) return false;

        vec3f inv_dir(1.0 / ray.d.x, 1.0 / ray.d.y, 1.0 / ray.d.z);
        int dir_is_neg[3] = {inv_dir.x < 0,inv_dir.y < 0,inv_dir.z < 0};
        bool hit = false;
        std::stack<size_t> s;
        s.push(0);
        while(!s.empty()){
            assert(s.size() < 64);
            auto node_index = s.top();
            s.pop();
            const auto node = linear_nodes + node_index;
            if(!rayIntersectWithAABB(node->bounds,ray.o,inv_dir,ray.t_min,ray.t_max,dir_is_neg)) continue;
            if(node->is_leaf_node()){
                for(int i = 0; i < node->primitive_count; i++){
                    //note intersect_p will change ray.max_t which is mutable
                    //in order to find closet intersection
                    if(primitives[node->primitive_offset + i]->intersect_p(ray,isect)){
                        hit = true;
                    }
                }
            }
            else{
                if(dir_is_neg[node->axis]){
                    //second child first
                    s.push(node_index + 1);
                    s.push(node->second_child_offset);
                }
                else{
                    s.push(node->second_child_offset);
                    s.push(node_index + 1);
                }
            }
        }
        return hit;
    }
private:
    struct BucketInfo{
        size_t count = 0;
        aabb3f bounds;
    };
    BVHBuildNode* recursive_build(std::vector<BVHPrimitiveInfo>& primitive_infos,
                                  std::vector<RC<Primitive>>& ordered_prims,
                                  size_t start,size_t end,
                                  size_t& total_nodes_count,MemoryArena& arena){
        assert(start < end);
        auto node = arena.alloc<BVHBuildNode>();
        ++total_nodes_count;
        aabb3f bounds;
        for(size_t i = start; i < end; ++i)
            bounds |= primitive_infos[i].bounds;

        size_t primitives_count = end - start;

        auto insert_leaf = [&](){
            size_t first_prim_offset = ordered_prims.size();
            for(size_t i = start; i < end; ++i){
                auto prim_index = primitive_infos[i].primitive_index;
                ordered_prims.emplace_back(primitives[prim_index]);
            }
            node->init_leaf(first_prim_offset,primitives_count,bounds);
        };
        if(primitives_count == 1){
            insert_leaf();
        }
        else
        {
            aabb3f centroid_bounds;
            for(size_t i = start; i < end; ++i)
                centroid_bounds |= primitive_infos[i].centroid;

            int mid = (start + end) >> 1;
            int dim = centroid_bounds.maximum_extent();

            if(centroid_bounds.high[dim] == centroid_bounds.low[dim]){
                insert_leaf();
            }
            else{
                //using SAH
                constexpr size_t SAH_PRIMITIVES_THRESHOLD = 2;
                if(primitives_count <= SAH_PRIMITIVES_THRESHOLD){
                    std::nth_element(&primitive_infos[start],&primitive_infos[mid],
                                     &primitive_infos[end-1]+1,
                                     [dim](const BVHPrimitiveInfo& a,const BVHPrimitiveInfo& b){
                                         return a.centroid[dim] < b.centroid[dim];
                                     });
                }
                else
                {
                    constexpr int N_BUCKETS = 12;
                    BucketInfo buckets[N_BUCKETS];

                    for(size_t i = start; i < end; ++i){
                        int b = N_BUCKETS * centroid_bounds.offset(primitive_infos[i].centroid)[dim];
                        if(b == N_BUCKETS) b = N_BUCKETS - 1;
                        assert(b >= 0 && b < N_BUCKETS);
                        buckets[b].count++;
                        buckets[b].bounds |= primitive_infos[i].bounds;
                    }
                    real cost[N_BUCKETS - 1];
                    //选择划分后包围盒面积更小的
                    for(int i = 0; i < N_BUCKETS - 1; ++i){
                        aabb3f lb,rb;
                        int l_count = 0, r_count = 0;
                        for(int j = 0; j <= i; ++j){
                            lb |= buckets[j].bounds;
                            l_count += buckets[j].count;
                        }
                        for(int j = i + 1; j < N_BUCKETS; ++j){
                            rb |= buckets[j].bounds;
                            r_count += buckets[j].count;
                        }
                        cost[i] = 1 + (l_count * lb.surface_area() + r_count * rb.surface_area()) / bounds.surface_area();
                    }

                    auto min_bucket_pos = std::min_element(cost,cost+(N_BUCKETS)-1) - cost;
                    auto min_cost = cost[min_bucket_pos];

                    if(primitives_count > max_leaf_prims){
                        BVHPrimitiveInfo* p_mid = std::partition(
                                &primitive_infos[start],&primitive_infos[end-1]+1,
                                [=](const BVHPrimitiveInfo& prim_info){
                                    int b = N_BUCKETS * centroid_bounds.offset(prim_info.centroid)[dim];
                                    if(b == N_BUCKETS) b = N_BUCKETS - 1;
                                    return b <= min_bucket_pos;
                                });
                        mid = p_mid - &primitive_infos[0];
                    }
                    else{
                        insert_leaf();
                        return node;
                    }
                }
                node->init_interior(dim, recursive_build(primitive_infos,ordered_prims,start,mid,total_nodes_count,arena),
                                    recursive_build(primitive_infos,ordered_prims,mid,end,total_nodes_count,arena));
            }
        }
        return node;
    }

    size_t flatten_bvh_tree(BVHBuildNode* node,size_t& offset){
        //将原来的BVH节点以深度搜索的顺序存储
        LinearBVHNode* linear_node = linear_nodes + offset;
        size_t node_offset = offset++;
        linear_node->bounds = node->bounds;
        if(node->is_leaf_node()){
            linear_node->primitive_count = node->primitive_count;
            linear_node->primitive_offset = node->first_prim_offset;
        }
        else{
            linear_node->axis = node->split_axis;
            linear_node->primitive_count = 0;
            flatten_bvh_tree(node->left,offset);
            linear_node->second_child_offset = flatten_bvh_tree(node->right,offset);
        }
        return node_offset;
    }
private:

    const int max_leaf_prims;

    std::vector<RC<Primitive>> primitives;
    LinearBVHNode* linear_nodes = nullptr;
};
