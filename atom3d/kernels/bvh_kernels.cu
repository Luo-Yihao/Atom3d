/**
 * BVH CUDA Kernels for Atom3D
 * Optimized with SAH Build and Stack-Based Traversal
 * 
 * Key features:
 * - Binary BVH with SAH (Surface Area Heuristic) binning for high-quality trees
 * - Stack-based traversal with distance sorting (visits closest nodes first)
 * - Optimized UDF query with early pruning
 * - Fixed barycentric computation (O(1) instead of re-traversal)
 * - Fallback: Longest Axis Median Split (Spatial) for robustness
 * 
 * Node layout: [bb_min(3), bb_max(3), left_idx, right_idx, unused] = 9 floats
 * For leaves: left_idx = -(start+1), right_idx = -(end+1)
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// ============================================================
// Helper Math Functions (inline device)
// ============================================================

__device__ __forceinline__ float3 make_float3_v(float x, float y, float z) {
    return make_float3(x, y, z);
}

__device__ __forceinline__ float dot3(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __forceinline__ float3 cross3(float3 a, float3 b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

__device__ __forceinline__ float3 sub3(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __forceinline__ float3 add3(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __forceinline__ float3 mul3(float3 a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__device__ __forceinline__ float len3(float3 a) {
    return sqrtf(a.x * a.x + a.y * a.y + a.z * a.z);
}

__device__ __forceinline__ float clamp_f(float x, float lo, float hi) {
    return fmaxf(lo, fminf(hi, x));
}

__device__ __forceinline__ float sign_f(float x) {
    return x >= 0.0f ? 1.0f : -1.0f;
}

// ============================================================
// Triangle Structure
// ============================================================

struct Triangle {
    float3 a, b, c;
    int original_id;  // Original face index before reordering
    
    __host__ __device__ float3 centroid() const {
        return make_float3(
            (a.x + b.x + c.x) / 3.0f,
            (a.y + b.y + c.y) / 3.0f,
            (a.z + b.z + c.z) / 3.0f
        );
    }
    
    __device__ float3 normal() const {
        float3 e1 = sub3(b, a);
        float3 e2 = sub3(c, a);
        float3 n = cross3(e1, e2);
        float len = len3(n);
        if (len > 1e-10f) {
            return mul3(n, 1.0f / len);
        }
        return make_float3(0, 0, 1);
    }
    
    // Point-triangle squared distance
    __device__ float distance_sq(float3 pos) const {
        float3 v21 = sub3(b, a);
        float3 p1 = sub3(pos, a);
        float3 v32 = sub3(c, b);
        float3 p2 = sub3(pos, b);
        float3 v13 = sub3(a, c);
        float3 p3 = sub3(pos, c);
        float3 nor = cross3(v21, v13);
        float nor_sq = dot3(nor, nor);
        
        bool is_degenerate = (nor_sq < 1e-12f);
        float sign_test = 0.0f;
        
        if (!is_degenerate) {
             sign_test = sign_f(dot3(cross3(v21, nor), p1)) +
                         sign_f(dot3(cross3(v32, nor), p2)) +
                         sign_f(dot3(cross3(v13, nor), p3));
        }

        if (is_degenerate || sign_test < 2.0f) {
            // Outside - distance to edges
            auto edge_dist = [&](float3 v, float3 p) {
                float d = dot3(v, p) / fmaxf(dot3(v, v), 1e-12f);
                d = clamp_f(d, 0.0f, 1.0f);
                float3 c = sub3(mul3(v, d), p);
                return dot3(c, c);
            };
            
            return fminf(edge_dist(v21, p1), fminf(edge_dist(v32, p2), edge_dist(v13, p3)));
        } else {
            // Inside - distance to plane
            float d = dot3(nor, p1);
            return d * d / fmaxf(nor_sq, 1e-12f);
        }
    }
    
    // Closest point calculation (must match distance_sq logic)
    __device__ float3 closest_point(float3 pos) const {
        float3 v21 = sub3(b, a);
        float3 p1 = sub3(pos, a);
        float3 v32 = sub3(c, b);
        float3 p2 = sub3(pos, b);
        float3 v13 = sub3(a, c);
        float3 p3 = sub3(pos, c);
        float3 nor = cross3(v21, v13);
        float nor_sq = dot3(nor, nor);
        
        bool is_degenerate = (nor_sq < 1e-12f);
        float sign_test = 0.0f;
        
        if (!is_degenerate) {
            sign_test = sign_f(dot3(cross3(v21, nor), p1)) +
                        sign_f(dot3(cross3(v32, nor), p2)) +
                        sign_f(dot3(cross3(v13, nor), p3));
        }
        
        if (is_degenerate || sign_test < 2.0f) {
            // Outside - find closest point on edges
            auto get_edge_closest = [&](float3 v, float3 p, float3 origin) {
                float d = dot3(v, p) / fmaxf(dot3(v, v), 1e-12f);
                d = clamp_f(d, 0.0f, 1.0f);
                return add3(origin, mul3(v, d));
            };
            
            float3 c1 = get_edge_closest(v21, p1, a);
            float3 c2 = get_edge_closest(v32, p2, b);
            float3 c3 = get_edge_closest(v13, p3, c);
            
            float d1 = dot3(sub3(c1, pos), sub3(c1, pos));
            float d2 = dot3(sub3(c2, pos), sub3(c2, pos));
            float d3 = dot3(sub3(c3, pos), sub3(c3, pos));
            
            if (d1 < d2 && d1 < d3) return c1;
            if (d2 < d3) return c2;
            return c3;
        } else {
            // Inside - project to plane
            float d = dot3(nor, p1);
            float3 proj = mul3(nor, d / fmaxf(nor_sq, 1e-12f));
            return sub3(pos, proj);
        }
    }
    
    // Ray-triangle intersection
    __device__ float ray_intersect(float3 ro, float3 rd) const {
        const float eps = 1e-8f;
        float3 e1 = sub3(b, a);
        float3 e2 = sub3(c, a);
        float3 h = cross3(rd, e2);
        float det = dot3(e1, h);
        
        if (fabsf(det) < eps) return 1e10f;
        
        float f = 1.0f / det;
        float3 s = sub3(ro, a);
        float u = f * dot3(s, h);
        if (u < 0.0f || u > 1.0f) return 1e10f;
        
        float3 q = cross3(s, e1);
        float v = f * dot3(rd, q);
        if (v < 0.0f || u + v > 1.0f) return 1e10f;
        
        float t = f * dot3(e2, q);
        return (t > eps) ? t : 1e10f;
    }
    
    // Barycentric coordinates
    __device__ float3 barycentric(float3 p) const {
        float3 v0 = sub3(b, a);
        float3 v1 = sub3(c, a);
        float3 v2 = sub3(p, a);
        
        float d00 = dot3(v0, v0);
        float d01 = dot3(v0, v1);
        float d11 = dot3(v1, v1);
        float d20 = dot3(v2, v0);
        float d21 = dot3(v2, v1);
        
        float denom = d00 * d11 - d01 * d01;
        float v = (d11 * d20 - d01 * d21) / fmaxf(denom, 1e-10f);
        float w = (d00 * d21 - d01 * d20) / fmaxf(denom, 1e-10f);
        float u = 1.0f - v - w;
        
        return make_float3(u, v, w);
    }
};

// ============================================================
// BVH Node Structure
// ============================================================

struct BVHNode {
    float bb_min[3];
    float bb_max[3];
    int left_idx;    // <0 for leaf: start = -left_idx - 1
    int right_idx;   // <0 for leaf: end = -right_idx - 1
    int pad;         // Padding/Unused
};

// ============================================================
// AABB Helper Functions
// ============================================================

__device__ bool aabb_overlap(const float* bb_min, const float* bb_max, float3 q_min, float3 q_max) {
    return (bb_min[0] <= q_max.x && bb_max[0] >= q_min.x) &&
           (bb_min[1] <= q_max.y && bb_max[1] >= q_min.y) &&
           (bb_min[2] <= q_max.z && bb_max[2] >= q_min.z);
}

// Point-AABB squared distance
__device__ float point_aabb_dist_sq(float3 p, const float* bb_min, const float* bb_max) {
    float dx = fmaxf(fmaxf(bb_min[0] - p.x, 0.0f), p.x - bb_max[0]);
    float dy = fmaxf(fmaxf(bb_min[1] - p.y, 0.0f), p.y - bb_max[1]);
    float dz = fmaxf(fmaxf(bb_min[2] - p.z, 0.0f), p.z - bb_max[2]);
    return dx * dx + dy * dy + dz * dz;
}

// Ray-AABB intersection
__device__ float ray_aabb_intersect(float3 ro, float3 rd, const float* bb_min, const float* bb_max) {
    float3 inv_d = make_float3(
        1.0f / (fabsf(rd.x) > 1e-8f ? rd.x : (rd.x >= 0 ? 1e-8f : -1e-8f)),
        1.0f / (fabsf(rd.y) > 1e-8f ? rd.y : (rd.y >= 0 ? 1e-8f : -1e-8f)),
        1.0f / (fabsf(rd.z) > 1e-8f ? rd.z : (rd.z >= 0 ? 1e-8f : -1e-8f))
    );
    
    float t1 = (bb_min[0] - ro.x) * inv_d.x;
    float t2 = (bb_max[0] - ro.x) * inv_d.x;
    float tmin = fminf(t1, t2);
    float tmax = fmaxf(t1, t2);
    
    t1 = (bb_min[1] - ro.y) * inv_d.y;
    t2 = (bb_max[1] - ro.y) * inv_d.y;
    tmin = fmaxf(tmin, fminf(t1, t2));
    tmax = fminf(tmax, fmaxf(t1, t2));
    
    t1 = (bb_min[2] - ro.z) * inv_d.z;
    t2 = (bb_max[2] - ro.z) * inv_d.z;
    tmin = fmaxf(tmin, fminf(t1, t2));
    tmax = fminf(tmax, fmaxf(t1, t2));
    
    if (tmax >= tmin && tmax >= 0) {
        return fmaxf(tmin, 0.0f);
    }
    return 1e10f;
}

// Triangle-AABB SAT with epsilon tolerance for edge cases
__device__ bool triangle_aabb_sat(const Triangle& tri, float3 box_min, float3 box_max) {
    // Small epsilon for edge-case detection (borderline intersections)
    const float sat_eps = 1e-6f;
    
    float3 box_center = make_float3(
        (box_min.x + box_max.x) * 0.5f,
        (box_min.y + box_max.y) * 0.5f,
        (box_min.z + box_max.z) * 0.5f
    );
    float3 box_half = make_float3(
        (box_max.x - box_min.x) * 0.5f,
        (box_max.y - box_min.y) * 0.5f,
        (box_max.z - box_min.z) * 0.5f
    );
    
    // Translate triangle to box-centered coordinates
    float3 v0 = sub3(tri.a, box_center);
    float3 v1 = sub3(tri.b, box_center);
    float3 v2 = sub3(tri.c, box_center);
    
    // Test box axes with epsilon
    float min_v, max_v;
    
    min_v = fminf(fminf(v0.x, v1.x), v2.x);
    max_v = fmaxf(fmaxf(v0.x, v1.x), v2.x);
    if (min_v > box_half.x + sat_eps || max_v < -box_half.x - sat_eps) return false;
    
    min_v = fminf(fminf(v0.y, v1.y), v2.y);
    max_v = fmaxf(fmaxf(v0.y, v1.y), v2.y);
    if (min_v > box_half.y + sat_eps || max_v < -box_half.y - sat_eps) return false;
    
    min_v = fminf(fminf(v0.z, v1.z), v2.z);
    max_v = fmaxf(fmaxf(v0.z, v1.z), v2.z);
    if (min_v > box_half.z + sat_eps || max_v < -box_half.z - sat_eps) return false;
    
    // Triangle edges
    float3 e0 = sub3(v1, v0);
    float3 e1 = sub3(v2, v1);
    float3 e2 = sub3(v0, v2);
    
    // Test 9 cross-product axes with epsilon
    #define SAT_AXIS_TEST(edge, axis_idx) do { \
        float3 axis; \
        if (axis_idx == 0) axis = make_float3(0.0f, edge.z, -edge.y); \
        else if (axis_idx == 1) axis = make_float3(-edge.z, 0.0f, edge.x); \
        else axis = make_float3(edge.y, -edge.x, 0.0f); \
        float p0 = dot3(v0, axis); \
        float p1 = dot3(v1, axis); \
        float p2 = dot3(v2, axis); \
        float min_p = fminf(fminf(p0, p1), p2); \
        float max_p = fmaxf(fmaxf(p0, p1), p2); \
        float rad = box_half.x * fabsf(axis.x) + box_half.y * fabsf(axis.y) + box_half.z * fabsf(axis.z) + sat_eps; \
        if (min_p > rad || max_p < -rad) return false; \
    } while(0)
    
    SAT_AXIS_TEST(e0, 0); SAT_AXIS_TEST(e0, 1); SAT_AXIS_TEST(e0, 2);
    SAT_AXIS_TEST(e1, 0); SAT_AXIS_TEST(e1, 1); SAT_AXIS_TEST(e1, 2);
    SAT_AXIS_TEST(e2, 0); SAT_AXIS_TEST(e2, 1); SAT_AXIS_TEST(e2, 2);
    
    #undef SAT_AXIS_TEST
    
    // Test triangle normal with epsilon
    float3 normal = cross3(e0, sub3(v2, v0));
    float d = -dot3(normal, v0);
    float r = box_half.x * fabsf(normal.x) + box_half.y * fabsf(normal.y) + box_half.z * fabsf(normal.z) + sat_eps;
    if (fabsf(d) > r) return false;
    
    return true;
}

// ============================================================
// BVH Traversal Kernels (Stack-Based)
// ============================================================

/**
 * UDF query: find closest point to mesh
 * Optimized with stack traversal and distance sorting
 */
__global__ void bvh_udf_kernel(
    const BVHNode* __restrict__ nodes,
    const Triangle* __restrict__ triangles,
    const float* __restrict__ points,
    int num_points,
    float* __restrict__ distances,
    int* __restrict__ closest_face_ids,
    float* __restrict__ closest_points,
    float* __restrict__ uvw
) {
    int p_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (p_idx >= num_points) return;
    
    float3 point = make_float3(points[p_idx * 3], points[p_idx * 3 + 1], points[p_idx * 3 + 2]);
    
    float best_dist_sq = 1e30f;
    int best_face = -1;
    int best_tri_idx = -1;  // Direct index for O(1) barycentric lookup
    
    // Fixed size stack - 128 for robustness on deep trees (degenerate meshes)
    int stack[128];
    int stack_ptr = 0;
    
    // Push root
    stack[stack_ptr++] = 0;
    
    while (stack_ptr > 0) {
        int idx = stack[--stack_ptr];
        const BVHNode& node = nodes[idx];
        
        // Culling: check if node AABB is further than current best distance
        // Note: For root(0), dist is 0 unless outside, so always visited.
        float node_dist_sq = point_aabb_dist_sq(point, node.bb_min, node.bb_max);
        if (node_dist_sq >= best_dist_sq) continue;
        
        if (node.left_idx < 0) {
            // Leaf: test triangles
            int start = -node.left_idx - 1;
            int end = -node.right_idx - 1;
            
            for (int i = start; i < end; i++) {
                float dist_sq = triangles[i].distance_sq(point);
                if (dist_sq < best_dist_sq) {
                    best_dist_sq = dist_sq;
                    best_face = triangles[i].original_id;
                    best_tri_idx = i;
                }
            }
        } else {
            // Internal: push children
            // Optimization: sort children by distance
            int left_child = node.left_idx;
            int right_child = node.right_idx;
            
            const BVHNode& l_node = nodes[left_child];
            const BVHNode& r_node = nodes[right_child];
            
            float d1 = point_aabb_dist_sq(point, l_node.bb_min, l_node.bb_max);
            float d2 = point_aabb_dist_sq(point, r_node.bb_min, r_node.bb_max);
            
            // Push FURTHEST child first, so CLOSEST is popped first
            if (d1 < d2) {
                // Left is closer
                if (d2 < best_dist_sq) {
                    if (stack_ptr < 128) stack[stack_ptr++] = right_child;
                }
                if (d1 < best_dist_sq) {
                   if (stack_ptr < 128) stack[stack_ptr++] = left_child;
                }
            } else {
                // Right is closer (or equal)
                if (d1 < best_dist_sq) {
                   if (stack_ptr < 128) stack[stack_ptr++] = left_child;
                }
                if (d2 < best_dist_sq) {
                   if (stack_ptr < 128) stack[stack_ptr++] = right_child;
                }
            }
        }
    }
    
    distances[p_idx] = sqrtf(best_dist_sq);
    closest_face_ids[p_idx] = best_face;
    
    float3 best_closest = point;
    if (best_tri_idx >= 0) {
        best_closest = triangles[best_tri_idx].closest_point(point);
    }
    
    closest_points[p_idx * 3 + 0] = best_closest.x;
    closest_points[p_idx * 3 + 1] = best_closest.y;
    closest_points[p_idx * 3 + 2] = best_closest.z;
    
    // Compute barycentric directly using found index (no re-traversal!)
    if (uvw && best_tri_idx >= 0) {
        float3 bary = triangles[best_tri_idx].barycentric(best_closest);
        uvw[p_idx * 3 + 0] = bary.x;
        uvw[p_idx * 3 + 1] = bary.y;
        uvw[p_idx * 3 + 2] = bary.z;
    }
}

/**
 * Ray-BVH intersection
 * Stack-based traversal with sorting
 */
__global__ void bvh_ray_intersect_kernel(
    const BVHNode* __restrict__ nodes,
    const Triangle* __restrict__ triangles,
    const float* __restrict__ rays_o,
    const float* __restrict__ rays_d,
    int num_rays,
    float max_t,
    bool* __restrict__ hit_mask,
    float* __restrict__ hit_t,
    int* __restrict__ hit_face_ids,
    float* __restrict__ hit_points
) {
    int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_idx >= num_rays) return;
    
    float3 ro = make_float3(rays_o[ray_idx * 3], rays_o[ray_idx * 3 + 1], rays_o[ray_idx * 3 + 2]);
    float3 rd = make_float3(rays_d[ray_idx * 3], rays_d[ray_idx * 3 + 1], rays_d[ray_idx * 3 + 2]);
    
    float mint = max_t;
    int best_face = -1;
    
    int stack[96];
    int stack_ptr = 0;
    stack[stack_ptr++] = 0;
    
    while (stack_ptr > 0) {
        int idx = stack[--stack_ptr];
        const BVHNode& node = nodes[idx];
        
        float tbb = ray_aabb_intersect(ro, rd, node.bb_min, node.bb_max);
        if (tbb >= mint) continue;
        
        if (node.left_idx < 0) {
            // Leaf
            int start = -node.left_idx - 1;
            int end = -node.right_idx - 1;
            
            for (int i = start; i < end; i++) {
                float t = triangles[i].ray_intersect(ro, rd);
                if (t < mint) {
                    mint = t;
                    best_face = triangles[i].original_id;
                }
            }
        } else {
            // Push children (simple order for ray tracing, could be optimized with front-to-back)
            // For now, push right then left
            int left = node.left_idx;
            int right = node.right_idx;
            
            // We could sort by ray entry time (tbb) to visit closer nodes first (Front-to-Back)
            // But ray_aabb_intersect handles the check.
            
            float t1 = ray_aabb_intersect(ro, rd, nodes[left].bb_min, nodes[left].bb_max);
            float t2 = ray_aabb_intersect(ro, rd, nodes[right].bb_min, nodes[right].bb_max);
            
            if (t1 < t2) {
                if (t2 < mint && stack_ptr < 96) stack[stack_ptr++] = right;
                if (t1 < mint && stack_ptr < 96) stack[stack_ptr++] = left;
            } else {
                if (t1 < mint && stack_ptr < 96) stack[stack_ptr++] = left;
                if (t2 < mint && stack_ptr < 96) stack[stack_ptr++] = right;
            }
        }
    }
    
    hit_mask[ray_idx] = (best_face >= 0);
    hit_t[ray_idx] = mint;
    hit_face_ids[ray_idx] = best_face;
    
    if (hit_points && best_face >= 0) {
        hit_points[ray_idx * 3 + 0] = ro.x + mint * rd.x;
        hit_points[ray_idx * 3 + 1] = ro.y + mint * rd.y;
        hit_points[ray_idx * 3 + 2] = ro.z + mint * rd.z;
    }
}

/**
 * AABB-BVH intersection with exact SAT test
 */
__global__ void bvh_aabb_intersect_kernel(
    const BVHNode* __restrict__ nodes,
    const Triangle* __restrict__ triangles,
    const float* __restrict__ query_min,
    const float* __restrict__ query_max,
    int num_queries,
    bool* __restrict__ hit_mask,
    int* __restrict__ hit_aabb_ids,
    int* __restrict__ hit_face_ids,
    int* __restrict__ hit_counter,
    int max_hits
) {
    int q_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (q_idx >= num_queries) return;
    
    float3 q_min = make_float3(query_min[q_idx * 3], query_min[q_idx * 3 + 1], query_min[q_idx * 3 + 2]);
    float3 q_max = make_float3(query_max[q_idx * 3], query_max[q_idx * 3 + 1], query_max[q_idx * 3 + 2]);
    
    bool any_hit = false;
    
    int stack[128];
    int stack_ptr = 0;
    stack[stack_ptr++] = 0;
    
    while (stack_ptr > 0) {
        int idx = stack[--stack_ptr];
        const BVHNode& node = nodes[idx];
        
        if (!aabb_overlap(node.bb_min, node.bb_max, q_min, q_max)) continue;
        
        if (node.left_idx < 0) {
            // Leaf: exact SAT test
            int start = -node.left_idx - 1;
            int end = -node.right_idx - 1;
            
            for (int i = start; i < end; i++) {
                if (triangle_aabb_sat(triangles[i], q_min, q_max)) {
                    any_hit = true;
                    
                    int write_idx = atomicAdd(hit_counter, 1);
                    if (write_idx < max_hits) {
                        hit_aabb_ids[write_idx] = q_idx;
                        hit_face_ids[write_idx] = triangles[i].original_id;
                    }
                }
            }
        } else {
            if (stack_ptr < 127) {
                stack[stack_ptr++] = node.right_idx;
                stack[stack_ptr++] = node.left_idx;
            }
        }
    }
    
    hit_mask[q_idx] = any_hit;
}

// ============================================================
// BVH Build (CPU, SAH-Based)
// ============================================================

struct TriangleInfo {
    float3 centroid;
    float3 a, b, c; // Cache vertices for bounds calculation
    int original_id;
    int idx;
};

struct SAHBin {
    float bb_min[3];
    float bb_max[3];
    int count;
    
    SAHBin() {
        bb_min[0] = bb_min[1] = bb_min[2] = 1e30f;
        bb_max[0] = bb_max[1] = bb_max[2] = -1e30f;
        count = 0;
    }
    
    void grow(const float3& a, const float3& b, const float3& c) {
        bb_min[0] = fminf(bb_min[0], fminf(fminf(a.x, b.x), c.x));
        bb_min[1] = fminf(bb_min[1], fminf(fminf(a.y, b.y), c.y));
        bb_min[2] = fminf(bb_min[2], fminf(fminf(a.z, b.z), c.z));
        bb_max[0] = fmaxf(bb_max[0], fmaxf(fmaxf(a.x, b.x), c.x));
        bb_max[1] = fmaxf(bb_max[1], fmaxf(fmaxf(a.y, b.y), c.y));
        bb_max[2] = fmaxf(bb_max[2], fmaxf(fmaxf(a.z, b.z), c.z));
        count++;
    }
    
    void grow(const SAHBin& other) {
        bb_min[0] = fminf(bb_min[0], other.bb_min[0]);
        bb_min[1] = fminf(bb_min[1], other.bb_min[1]);
        bb_min[2] = fminf(bb_min[2], other.bb_min[2]);
        bb_max[0] = fmaxf(bb_max[0], other.bb_max[0]);
        bb_max[1] = fmaxf(bb_max[1], other.bb_max[1]);
        bb_max[2] = fmaxf(bb_max[2], other.bb_max[2]);
        count += other.count;
    }
    
    float area() const {
        if (count == 0) return 0.0f;
        float dx = bb_max[0] - bb_min[0];
        float dy = bb_max[1] - bb_min[1];
        float dz = bb_max[2] - bb_min[2];
        // Ensure non-negative dimensions (handles empty bins effectively)
        dx = fmaxf(0.0f, dx);
        dy = fmaxf(0.0f, dy);
        dz = fmaxf(0.0f, dz);
        return 2.0f * (dx * dy + dy * dz + dz * dx);
    }
};

void build_bvh_recursive(
    std::vector<TriangleInfo>& tris,
    int start, int end,
    std::vector<BVHNode>& nodes,
    int node_idx,
    int n_primitives_per_leaf
) {
    BVHNode& node = nodes[node_idx];
    
    // Compute AABB
    node.bb_min[0] = node.bb_min[1] = node.bb_min[2] = 1e30f;
    node.bb_max[0] = node.bb_max[1] = node.bb_max[2] = -1e30f;
    
    for (int i = start; i < end; i++) {
        const TriangleInfo& t = tris[i];
        node.bb_min[0] = fminf(node.bb_min[0], fminf(fminf(t.a.x, t.b.x), t.c.x));
        node.bb_min[1] = fminf(node.bb_min[1], fminf(fminf(t.a.y, t.b.y), t.c.y));
        node.bb_min[2] = fminf(node.bb_min[2], fminf(fminf(t.a.z, t.b.z), t.c.z));
        node.bb_max[0] = fmaxf(node.bb_max[0], fmaxf(fmaxf(t.a.x, t.b.x), t.c.x));
        node.bb_max[1] = fmaxf(node.bb_max[1], fmaxf(fmaxf(t.a.y, t.b.y), t.c.y));
        node.bb_max[2] = fmaxf(node.bb_max[2], fmaxf(fmaxf(t.a.z, t.b.z), t.c.z));
    }
    
    int count = end - start;
    if (count <= n_primitives_per_leaf) {
        node.left_idx = -(start + 1);
        node.right_idx = -(end + 1);
        return;
    }
    
    // SAH Binning
    int best_axis = -1;
    float best_cost = 1e30f;
    int best_split_idx = -1;
    
    // Current node surface area
    float root_area;
    {
        float dx = node.bb_max[0] - node.bb_min[0];
        float dy = node.bb_max[1] - node.bb_min[1];
        float dz = node.bb_max[2] - node.bb_min[2];
        root_area = 2.0f * (dx * dy + dy * dz + dz * dx);
    }
    
    const int NUM_BINS = 16;
    bool sah_possible = false;
    
    float box_dims[3];
    box_dims[0] = node.bb_max[0] - node.bb_min[0];
    box_dims[1] = node.bb_max[1] - node.bb_min[1];
    box_dims[2] = node.bb_max[2] - node.bb_min[2];
    
    for (int axis = 0; axis < 3; axis++) {
        // Skip axis if extent is too small
        if (box_dims[axis] < 1e-6f) continue;
        
        sah_possible = true;
        
        float bounds_min = 1e30f, bounds_max = -1e30f;
        for(int i=start; i<end; ++i) {
            float c;
            if(axis==0) c = tris[i].centroid.x;
            else if(axis==1) c = tris[i].centroid.y;
            else c = tris[i].centroid.z;
            bounds_min = fminf(bounds_min, c);
            bounds_max = fmaxf(bounds_max, c);
        }
        
        if (bounds_max - bounds_min < 1e-6f) continue;
        
        SAHBin bins[NUM_BINS];
        float scale = NUM_BINS / (bounds_max - bounds_min);
        // Clamp scale to avoid index out of bounds slightly
        scale *= 0.999f;
        
        for (int i = start; i < end; i++) {
            float c;
            if(axis==0) c = tris[i].centroid.x;
            else if(axis==1) c = tris[i].centroid.y;
            else c = tris[i].centroid.z;
            
            int bin_idx = (int)((c - bounds_min) * scale);
            bin_idx = std::max(0, std::min(NUM_BINS - 1, bin_idx));
            bins[bin_idx].grow(tris[i].a, tris[i].b, tris[i].c);
        }
        
        float left_area[NUM_BINS - 1], right_area[NUM_BINS - 1];
        int left_count[NUM_BINS - 1], right_count[NUM_BINS - 1];
        
        SAHBin current_box;
        int current_count = 0;
        
        for (int i = 0; i < NUM_BINS - 1; i++) {
            current_box.grow(bins[i]);
            current_count += bins[i].count;
            left_area[i] = current_box.area();
            left_count[i] = current_count;
        }
        
        current_box = SAHBin();
        current_count = 0;
        
        for (int i = NUM_BINS - 1; i > 0; i--) {
            current_box.grow(bins[i]);
            current_count += bins[i].count;
            right_area[i - 1] = current_box.area();
            right_count[i - 1] = current_count;
        }
        
        for (int i = 0; i < NUM_BINS - 1; i++) {
            float cost = left_count[i] * left_area[i] + right_count[i] * right_area[i];
            if (cost < best_cost) {
                best_cost = cost;
                best_axis = axis;
                best_split_idx = i;
            }
        }
    }
    
    int mid = start + count / 2;
    
    if (best_axis != -1) {
        // ... SAH Implementation (unchanged logic) ...
         float bounds_min = 1e30f, bounds_max = -1e30f;
        for(int i=start; i<end; ++i) {
            float c;
            if(best_axis==0) c = tris[i].centroid.x;
            else if(best_axis==1) c = tris[i].centroid.y;
            else c = tris[i].centroid.z;
            bounds_min = fminf(bounds_min, c);
            bounds_max = fmaxf(bounds_max, c);
        }
        
        float split_val = bounds_min + (best_split_idx + 1) * (bounds_max - bounds_min) / NUM_BINS;
        
        auto it = std::partition(tris.begin() + start, tris.begin() + end,
            [&](const TriangleInfo& t) {
                float c = (best_axis == 0) ? t.centroid.x : ((best_axis == 1) ? t.centroid.y : t.centroid.z);
                return c < split_val;
            });
            
        mid = std::distance(tris.begin(), it);
        
        if (mid == start || mid == end) {
            // SAH split empty, fallback to median on best_axis
            std::nth_element(
                tris.begin() + start,
                tris.begin() + start + count / 2,
                tris.begin() + end,
                [best_axis](const TriangleInfo& a, const TriangleInfo& b) {
                    float ca = (best_axis == 0) ? a.centroid.x : ((best_axis == 1) ? a.centroid.y : a.centroid.z);
                    float cb = (best_axis == 0) ? b.centroid.x : ((best_axis == 1) ? b.centroid.y : b.centroid.z);
                    return ca < cb;
                }
            );
            mid = start + count / 2;
        }
    } else {
        // FALLBACK: Spatial Median Split on Longest Axis
        int fallback_axis = 0;
        if (box_dims[1] > box_dims[0]) fallback_axis = 1;
        if (box_dims[2] > box_dims[fallback_axis]) fallback_axis = 2;
        
        // Ensure we actually split spatially
        std::nth_element(
            tris.begin() + start,
            tris.begin() + start + count / 2,
            tris.begin() + end,
            [fallback_axis](const TriangleInfo& a, const TriangleInfo& b) {
                float ca = (fallback_axis == 0) ? a.centroid.x : ((fallback_axis == 1) ? a.centroid.y : a.centroid.z);
                float cb = (fallback_axis == 0) ? b.centroid.x : ((fallback_axis == 1) ? b.centroid.y : b.centroid.z);
                return ca < cb;
            }
        );
        mid = start + count / 2;
    }

    // Create child nodes
    int left_idx = nodes.size();
    nodes.emplace_back();
    nodes.emplace_back();
    
    nodes[node_idx].left_idx = left_idx;
    nodes[node_idx].right_idx = left_idx + 1; // Indices are now sequential
    
    build_bvh_recursive(tris, start, mid, nodes, left_idx, n_primitives_per_leaf);
    build_bvh_recursive(tris, mid, end, nodes, left_idx + 1, n_primitives_per_leaf);
}

/**
 * Build BVH from mesh
 * Returns: (nodes_tensor, triangles_tensor)
 */
std::vector<at::Tensor> build_bvh_cuda(
    at::Tensor vertices,
    at::Tensor faces,
    int n_primitives_per_leaf
) {
    auto vertices_cpu = vertices.cpu().contiguous();
    auto faces_cpu = faces.cpu().contiguous();
    
    int num_faces = faces_cpu.size(0);
    const float* v_ptr = vertices_cpu.data_ptr<float>();
    const int* f_ptr = faces_cpu.data_ptr<int>();
    
    std::vector<TriangleInfo> tri_info(num_faces);
    for (int i = 0; i < num_faces; i++) {
        int i0 = f_ptr[i * 3 + 0];
        int i1 = f_ptr[i * 3 + 1];
        int i2 = f_ptr[i * 3 + 2];
        
        TriangleInfo& info = tri_info[i];
        
        info.a = make_float3(v_ptr[i0 * 3], v_ptr[i0 * 3 + 1], v_ptr[i0 * 3 + 2]);
        info.b = make_float3(v_ptr[i1 * 3], v_ptr[i1 * 3 + 1], v_ptr[i1 * 3 + 2]);
        info.c = make_float3(v_ptr[i2 * 3], v_ptr[i2 * 3 + 1], v_ptr[i2 * 3 + 2]);
        info.original_id = i;
        
        info.centroid = make_float3(
            (info.a.x + info.b.x + info.c.x) / 3.0f,
            (info.a.y + info.b.y + info.c.y) / 3.0f,
            (info.a.z + info.b.z + info.c.z) / 3.0f
        );
    }
    
    std::vector<BVHNode> nodes;
    nodes.emplace_back();
    // Pre-reserve to avoid reallocations
    nodes.reserve(num_faces * 2);
    
    build_bvh_recursive(tri_info, 0, num_faces, nodes, 0, n_primitives_per_leaf);
    
    // Create tensors
    auto opts_float = torch::TensorOptions().dtype(torch::kFloat32);
    auto nodes_tensor = torch::empty({(int)nodes.size(), 9}, opts_float);
    float* n_ptr = nodes_tensor.data_ptr<float>();
    
    // Direct copy
    std::memcpy(n_ptr, nodes.data(), nodes.size() * sizeof(BVHNode));
    
    auto triangles_tensor = torch::empty({num_faces, 10}, opts_float);
    float* t_ptr = triangles_tensor.data_ptr<float>();
    
    for (int i = 0; i < num_faces; i++) {
        const TriangleInfo& t = tri_info[i];
        t_ptr[i * 10 + 0] = t.a.x;
        t_ptr[i * 10 + 1] = t.a.y;
        t_ptr[i * 10 + 2] = t.a.z;
        t_ptr[i * 10 + 3] = t.b.x;
        t_ptr[i * 10 + 4] = t.b.y;
        t_ptr[i * 10 + 5] = t.b.z;
        t_ptr[i * 10 + 6] = t.c.x;
        t_ptr[i * 10 + 7] = t.c.y;
        t_ptr[i * 10 + 8] = t.c.z;
        t_ptr[i * 10 + 9] = *reinterpret_cast<const float*>(&t.original_id);
    }
    
    return {nodes_tensor.to(vertices.device()), triangles_tensor.to(vertices.device())};
}

// ============================================================
// Python Bindings (wrappers)
// ============================================================

std::vector<at::Tensor> bvh_udf_cuda(
    at::Tensor nodes,
    at::Tensor triangles,
    at::Tensor points
) {
    CHECK_INPUT(nodes);
    CHECK_INPUT(triangles);
    CHECK_INPUT(points);
    
    // Ensure kernel runs on the correct GPU
    c10::cuda::CUDAGuard device_guard(points.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    int num_points = points.size(0);
    
    auto opts_float = points.options();
    auto opts_int = points.options().dtype(torch::kInt32);
    
    auto distances = torch::empty({num_points}, opts_float);
    auto face_ids = torch::empty({num_points}, opts_int);
    auto closest_points = torch::empty({num_points, 3}, opts_float);
    auto uvw = torch::empty({num_points, 3}, opts_float);
    
    int block_size = 256;
    int grid_size = (num_points + block_size - 1) / block_size;
    
    bvh_udf_kernel<<<grid_size, block_size, 0, stream>>>(
        reinterpret_cast<const BVHNode*>(nodes.data_ptr<float>()),
        reinterpret_cast<const Triangle*>(triangles.data_ptr<float>()),
        points.data_ptr<float>(),
        num_points,
        distances.data_ptr<float>(),
        face_ids.data_ptr<int>(),
        closest_points.data_ptr<float>(),
        uvw.data_ptr<float>()
    );
    
    return {distances, face_ids, closest_points, uvw};
}

std::vector<at::Tensor> bvh_ray_intersect_cuda(
    at::Tensor nodes,
    at::Tensor triangles,
    at::Tensor rays_o,
    at::Tensor rays_d,
    float max_t
) {
    CHECK_INPUT(nodes);
    CHECK_INPUT(triangles);
    CHECK_INPUT(rays_o);
    CHECK_INPUT(rays_d);
    
    // Ensure kernel runs on the correct GPU
    c10::cuda::CUDAGuard device_guard(rays_o.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    int num_rays = rays_o.size(0);
    
    auto opts_bool = rays_o.options().dtype(torch::kBool);
    auto opts_float = rays_o.options();
    auto opts_int = rays_o.options().dtype(torch::kInt32);
    
    auto hit_mask = torch::zeros({num_rays}, opts_bool);
    auto hit_t = torch::full({num_rays}, max_t, opts_float);
    auto hit_face_ids = torch::full({num_rays}, -1, opts_int);
    auto hit_points = torch::empty({num_rays, 3}, opts_float);
    
    int block_size = 256;
    int grid_size = (num_rays + block_size - 1) / block_size;
    
    bvh_ray_intersect_kernel<<<grid_size, block_size, 0, stream>>>(
        reinterpret_cast<const BVHNode*>(nodes.data_ptr<float>()),
        reinterpret_cast<const Triangle*>(triangles.data_ptr<float>()),
        rays_o.data_ptr<float>(),
        rays_d.data_ptr<float>(),
        num_rays,
        max_t,
        hit_mask.data_ptr<bool>(),
        hit_t.data_ptr<float>(),
        hit_face_ids.data_ptr<int>(),
        hit_points.data_ptr<float>()
    );
    
    return {hit_mask, hit_t, hit_face_ids, hit_points};
}

std::vector<at::Tensor> bvh_aabb_intersect_cuda(
    at::Tensor nodes,
    at::Tensor triangles,
    at::Tensor query_min,
    at::Tensor query_max
) {
    CHECK_INPUT(nodes);
    CHECK_INPUT(triangles);
    CHECK_INPUT(query_min);
    CHECK_INPUT(query_max);
    
    // Ensure kernel runs on the correct GPU
    c10::cuda::CUDAGuard device_guard(query_min.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    int num_queries = query_min.size(0);
    
    auto opts_bool = query_min.options().dtype(torch::kBool);
    auto opts_int = query_min.options().dtype(torch::kInt32);
    
    auto hit_mask = torch::zeros({num_queries}, opts_bool);
    
    int64_t max_hits = std::min((int64_t)num_queries * 100, (int64_t)50000000);
    auto hit_aabb_ids = torch::empty({max_hits}, opts_int);
    auto hit_face_ids = torch::empty({max_hits}, opts_int);
    auto hit_counter = torch::zeros({1}, opts_int);
    
    int block_size = 256;
    int grid_size = (num_queries + block_size - 1) / block_size;
    
    bvh_aabb_intersect_kernel<<<grid_size, block_size, 0, stream>>>(
        reinterpret_cast<const BVHNode*>(nodes.data_ptr<float>()),
        reinterpret_cast<const Triangle*>(triangles.data_ptr<float>()),
        query_min.data_ptr<float>(),
        query_max.data_ptr<float>(),
        num_queries,
        hit_mask.data_ptr<bool>(),
        hit_aabb_ids.data_ptr<int>(),
        hit_face_ids.data_ptr<int>(),
        hit_counter.data_ptr<int>(),
        max_hits
    );
    
    int final_hits = std::min(hit_counter[0].item<int>(), (int)max_hits);
    
    return {
        hit_mask,
        hit_aabb_ids.slice(0, 0, final_hits),
        hit_face_ids.slice(0, 0, final_hits)
    };
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("build_bvh", &build_bvh_cuda, "Build BVH from mesh");
    m.def("bvh_udf", &bvh_udf_cuda, "BVH UDF query");
    m.def("bvh_ray_intersect", &bvh_ray_intersect_cuda, "BVH ray intersection");
    m.def("bvh_aabb_intersect", &bvh_aabb_intersect_cuda, "BVH AABB intersection");
}
