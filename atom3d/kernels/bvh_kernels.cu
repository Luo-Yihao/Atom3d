/**
 * BVH CUDA Kernels for Atom3D
 * Ported from cubvh with modifications for AABB intersection support.
 * 
 * Key features:
 * - 4-way BVH with escape links for stackless traversal
 * - CPU build, GPU traversal
 * - Supports UDF, ray, and AABB queries
 * 
 * Node layout: [bb_min(3), bb_max(3), left_idx, right_idx, escape_idx] = 9 floats
 * For leaves: left_idx = -(start+1), right_idx = -(end+1)
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include <stack>
#include <algorithm>
#include <cmath>

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
// Triangle Structure (ported from cubvh)
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
    
    // Point-triangle squared distance (from cubvh)
    __device__ float distance_sq(float3 pos) const {
        float3 v21 = sub3(b, a);
        float3 p1 = sub3(pos, a);
        float3 v32 = sub3(c, b);
        float3 p2 = sub3(pos, b);
        float3 v13 = sub3(a, c);
        float3 p3 = sub3(pos, c);
        float3 nor = cross3(v21, v13);
        float nor_sq = dot3(nor, nor);
        
        bool is_degenerate = (nor_sq < 1e-20f);  // Area < 1e-10
        float sign_test = 0.0f;
        
        if (!is_degenerate) {
             sign_test = sign_f(dot3(cross3(v21, nor), p1)) +
                         sign_f(dot3(cross3(v32, nor), p2)) +
                         sign_f(dot3(cross3(v13, nor), p3));
        }

        if (is_degenerate || sign_test < 2.0f) {
            // Outside - distance to edges
            float d1 = dot3(v21, p1) / fmaxf(dot3(v21, v21), 1e-12f);
            d1 = clamp_f(d1, 0.0f, 1.0f);
            float3 c1 = sub3(mul3(v21, d1), p1);
            float dist1 = dot3(c1, c1);
            
            float d2 = dot3(v32, p2) / fmaxf(dot3(v32, v32), 1e-12f);
            d2 = clamp_f(d2, 0.0f, 1.0f);
            float3 c2 = sub3(mul3(v32, d2), p2);
            float dist2 = dot3(c2, c2);
            
            float d3 = dot3(v13, p3) / fmaxf(dot3(v13, v13), 1e-12f);
            d3 = clamp_f(d3, 0.0f, 1.0f);
            float3 c3 = sub3(mul3(v13, d3), p3);
            float dist3 = dot3(c3, c3);
            
            return fminf(dist1, fminf(dist2, dist3));
        } else {
            // Inside - distance to plane
            float d = dot3(nor, p1);
            return d * d / fmaxf(nor_sq, 1e-12f);
        }
    }
    
    // Closest point on triangle - must match distance_sq logic exactly
    __device__ float3 closest_point(float3 pos) const {
        float3 v21 = sub3(b, a);
        float3 p1 = sub3(pos, a);
        float3 v32 = sub3(c, b);
        float3 p2 = sub3(pos, b);
        float3 v13 = sub3(a, c);
        float3 p3 = sub3(pos, c);
        float3 nor = cross3(v21, v13);
        float nor_sq = dot3(nor, nor);
        
        bool is_degenerate = (nor_sq < 1e-20f);
        float sign_test = 0.0f;
        
        if (!is_degenerate) {
            sign_test = sign_f(dot3(cross3(v21, nor), p1)) +
                        sign_f(dot3(cross3(v32, nor), p2)) +
                        sign_f(dot3(cross3(v13, nor), p3));
        }
        
        if (is_degenerate || sign_test < 2.0f) {
            // Outside - find closest point on edges
            float d1 = dot3(v21, p1) / fmaxf(dot3(v21, v21), 1e-12f);
            d1 = clamp_f(d1, 0.0f, 1.0f);
            float3 c1_vec = sub3(mul3(v21, d1), p1);
            float dist1 = dot3(c1_vec, c1_vec);
            
            float d2 = dot3(v32, p2) / fmaxf(dot3(v32, v32), 1e-12f);
            d2 = clamp_f(d2, 0.0f, 1.0f);
            float3 c2_vec = sub3(mul3(v32, d2), p2);
            float dist2 = dot3(c2_vec, c2_vec);
            
            float d3 = dot3(v13, p3) / fmaxf(dot3(v13, v13), 1e-12f);
            d3 = clamp_f(d3, 0.0f, 1.0f);
            float3 c3_vec = sub3(mul3(v13, d3), p3);
            float dist3 = dot3(c3_vec, c3_vec);
            
            if (dist1 <= dist2 && dist1 <= dist3) {
                // Closest to edge a-b
                return add3(a, mul3(v21, d1));
            } else if (dist2 <= dist3) {
                // Closest to edge b-c
                return add3(b, mul3(v32, d2));
            } else {
                // Closest to edge c-a
                return add3(c, mul3(v13, d3));
            }
        } else {
            // Inside - project to plane
            float d = dot3(nor, p1);
            float3 proj = mul3(nor, d / fmaxf(nor_sq, 1e-12f));
            return sub3(pos, proj);
        }
    }
    
    // Ray-triangle intersection (Möller-Trumbore)
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
    int escape_idx;  // Next node after subtree, -1 = terminate
};

// ============================================================
// AABB Helper Functions
// ============================================================

// AABB-AABB overlap test
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

// Ray-AABB intersection (slab method)
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

// Triangle-AABB SAT intersection (from cubvh bounding_box.cuh)
__device__ bool triangle_aabb_sat(const Triangle& tri, float3 box_min, float3 box_max) {
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
    
    // Test box axes
    float min_v, max_v;
    
    min_v = fminf(fminf(v0.x, v1.x), v2.x);
    max_v = fmaxf(fmaxf(v0.x, v1.x), v2.x);
    if (min_v > box_half.x || max_v < -box_half.x) return false;
    
    min_v = fminf(fminf(v0.y, v1.y), v2.y);
    max_v = fmaxf(fmaxf(v0.y, v1.y), v2.y);
    if (min_v > box_half.y || max_v < -box_half.y) return false;
    
    min_v = fminf(fminf(v0.z, v1.z), v2.z);
    max_v = fmaxf(fmaxf(v0.z, v1.z), v2.z);
    if (min_v > box_half.z || max_v < -box_half.z) return false;
    
    // Triangle edges
    float3 e0 = sub3(v1, v0);
    float3 e1 = sub3(v2, v1);
    float3 e2 = sub3(v0, v2);
    
    // Test 9 cross-product axes
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
        float rad = box_half.x * fabsf(axis.x) + box_half.y * fabsf(axis.y) + box_half.z * fabsf(axis.z); \
        if (min_p > rad || max_p < -rad) return false; \
    } while(0)
    
    SAT_AXIS_TEST(e0, 0); SAT_AXIS_TEST(e0, 1); SAT_AXIS_TEST(e0, 2);
    SAT_AXIS_TEST(e1, 0); SAT_AXIS_TEST(e1, 1); SAT_AXIS_TEST(e1, 2);
    SAT_AXIS_TEST(e2, 0); SAT_AXIS_TEST(e2, 1); SAT_AXIS_TEST(e2, 2);
    
    #undef SAT_AXIS_TEST
    
    // Test triangle normal
    float3 normal = cross3(e0, sub3(v2, v0));
    float d = -dot3(normal, v0);
    float r = box_half.x * fabsf(normal.x) + box_half.y * fabsf(normal.y) + box_half.z * fabsf(normal.z);
    if (fabsf(d) > r) return false;
    
    return true;
}

// ============================================================
// BVH Traversal Kernels
// ============================================================

/**
 * UDF query: find closest point to mesh
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
    int best_face = 0;
    float3 best_closest = point;
    
    // Stackless traversal
    int idx = 0;
    while (idx != -1) {
        const BVHNode& node = nodes[idx];
        
        float dbb = point_aabb_dist_sq(point, node.bb_min, node.bb_max);
        if (dbb > best_dist_sq) {
            idx = node.escape_idx;
            continue;
        }
        
        if (node.left_idx < 0) {
            // Leaf: test triangles
            int start = -node.left_idx - 1;
            int end = -node.right_idx - 1;
            
            for (int i = start; i < end; i++) {
                float dist_sq = triangles[i].distance_sq(point);
                if (dist_sq < best_dist_sq) {
                    best_dist_sq = dist_sq;
                    best_face = triangles[i].original_id;
                    best_closest = triangles[i].closest_point(point);
                }
            }
            idx = node.escape_idx;
        } else {
            idx = node.left_idx;
        }
    }
    
    distances[p_idx] = sqrtf(best_dist_sq);
    closest_face_ids[p_idx] = best_face;
    closest_points[p_idx * 3 + 0] = best_closest.x;
    closest_points[p_idx * 3 + 1] = best_closest.y;
    closest_points[p_idx * 3 + 2] = best_closest.z;
    
    // Compute barycentric
    if (uvw) {
        // Load triangle for barycentric
        float3 bary = make_float3(0.33f, 0.33f, 0.34f);
        for (int idx2 = 0; idx2 != -1; ) {
            const BVHNode& node = nodes[idx2];
            if (node.left_idx < 0) {
                int start = -node.left_idx - 1;
                int end = -node.right_idx - 1;
                for (int i = start; i < end; i++) {
                    if (triangles[i].original_id == best_face) {
                        bary = triangles[i].barycentric(best_closest);
                        idx2 = -1;
                        break;
                    }
                }
                if (idx2 != -1) idx2 = node.escape_idx;
            } else {
                idx2 = node.left_idx;
            }
        }
        uvw[p_idx * 3 + 0] = bary.x;
        uvw[p_idx * 3 + 1] = bary.y;
        uvw[p_idx * 3 + 2] = bary.z;
    }
}

/**
 * Ray-BVH intersection
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
    
    // Stackless traversal
    int idx = 0;
    while (idx != -1) {
        const BVHNode& node = nodes[idx];
        
        float tbb = ray_aabb_intersect(ro, rd, node.bb_min, node.bb_max);
        if (tbb >= mint) {
            idx = node.escape_idx;
            continue;
        }
        
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
            idx = node.escape_idx;
        } else {
            idx = node.left_idx;
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
 * Returns which query AABBs intersect any triangle (exact, no false positives)
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
    
    // Stackless traversal
    int idx = 0;
    while (idx != -1) {
        const BVHNode& node = nodes[idx];
        
        // AABB-AABB broadphase
        if (!aabb_overlap(node.bb_min, node.bb_max, q_min, q_max)) {
            idx = node.escape_idx;
            continue;
        }
        
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
                        hit_face_ids[write_idx] = triangles[i].original_id;  // ORIGINAL index
                    }
                }
            }
            idx = node.escape_idx;
        } else {
            idx = node.left_idx;
        }
    }
    
    hit_mask[q_idx] = any_hit;
}

// ============================================================
// BVH Build (CPU, ported from cubvh)
// ============================================================

constexpr int BRANCHING_FACTOR = 4;

struct TriangleInfo {
    float3 centroid;
    int idx;
    Triangle tri;
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
        const Triangle& t = tris[i].tri;
        node.bb_min[0] = fminf(node.bb_min[0], fminf(fminf(t.a.x, t.b.x), t.c.x));
        node.bb_min[1] = fminf(node.bb_min[1], fminf(fminf(t.a.y, t.b.y), t.c.y));
        node.bb_min[2] = fminf(node.bb_min[2], fminf(fminf(t.a.z, t.b.z), t.c.z));
        node.bb_max[0] = fmaxf(node.bb_max[0], fmaxf(fmaxf(t.a.x, t.b.x), t.c.x));
        node.bb_max[1] = fmaxf(node.bb_max[1], fmaxf(fmaxf(t.a.y, t.b.y), t.c.y));
        node.bb_max[2] = fmaxf(node.bb_max[2], fmaxf(fmaxf(t.a.z, t.b.z), t.c.z));
    }
    
    int count = end - start;
    if (count <= n_primitives_per_leaf) {
        // Leaf
        node.left_idx = -(start + 1);
        node.right_idx = -(end + 1);
        return;
    }
    
    // Choose split axis (max variance)
    float mean[3] = {0, 0, 0};
    for (int i = start; i < end; i++) {
        mean[0] += tris[i].centroid.x;
        mean[1] += tris[i].centroid.y;
        mean[2] += tris[i].centroid.z;
    }
    mean[0] /= count;
    mean[1] /= count;
    mean[2] /= count;
    
    float var[3] = {0, 0, 0};
    for (int i = start; i < end; i++) {
        float dx = tris[i].centroid.x - mean[0];
        float dy = tris[i].centroid.y - mean[1];
        float dz = tris[i].centroid.z - mean[2];
        var[0] += dx * dx;
        var[1] += dy * dy;
        var[2] += dz * dz;
    }
    
    int axis = 0;
    if (var[1] > var[0]) axis = 1;
    if (var[2] > var[axis]) axis = 2;
    
    // Partition at median
    int mid = start + count / 2;
    std::nth_element(
        tris.begin() + start,
        tris.begin() + mid,
        tris.begin() + end,
        [axis](const TriangleInfo& a, const TriangleInfo& b) {
            if (axis == 0) return a.centroid.x < b.centroid.x;
            if (axis == 1) return a.centroid.y < b.centroid.y;
            return a.centroid.z < b.centroid.z;
        }
    );
    
    // Create child nodes
    int left_idx = nodes.size();
    nodes.emplace_back();
    nodes.emplace_back();
    
    nodes[node_idx].left_idx = left_idx;
    nodes[node_idx].right_idx = left_idx + 2;  // +2 because we have 2 children (indices left_idx and left_idx+1)
    
    // Recursively build children
    build_bvh_recursive(tris, start, mid, nodes, left_idx, n_primitives_per_leaf);
    build_bvh_recursive(tris, mid, end, nodes, left_idx + 1, n_primitives_per_leaf);
}

void thread_bvh(std::vector<BVHNode>& nodes, int node_idx, int escape_idx) {
    BVHNode& node = nodes[node_idx];
    node.escape_idx = escape_idx;
    
    if (node.left_idx < 0) return;  // Leaf
    
    int first_child = node.left_idx;
    int num_children = node.right_idx - first_child;
    
    for (int c = 0; c < num_children; c++) {
        int next_escape = (c + 1 < num_children) ? (first_child + c + 1) : escape_idx;
        thread_bvh(nodes, first_child + c, next_escape);
    }
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
    // Copy to CPU for construction (BVH is built on CPU)
    auto vertices_cpu = vertices.cpu().contiguous();
    auto faces_cpu = faces.cpu().contiguous();
    
    int num_verts = vertices_cpu.size(0);
    int num_faces = faces_cpu.size(0);
    
    const float* v_ptr = vertices_cpu.data_ptr<float>();
    const int* f_ptr = faces_cpu.data_ptr<int>();
    
    // Create triangle infos
    std::vector<TriangleInfo> tri_info(num_faces);
    for (int i = 0; i < num_faces; i++) {
        int i0 = f_ptr[i * 3 + 0];
        int i1 = f_ptr[i * 3 + 1];
        int i2 = f_ptr[i * 3 + 2];
        
        Triangle& t = tri_info[i].tri;
        t.a = make_float3(v_ptr[i0 * 3], v_ptr[i0 * 3 + 1], v_ptr[i0 * 3 + 2]);
        t.b = make_float3(v_ptr[i1 * 3], v_ptr[i1 * 3 + 1], v_ptr[i1 * 3 + 2]);
        t.c = make_float3(v_ptr[i2 * 3], v_ptr[i2 * 3 + 1], v_ptr[i2 * 3 + 2]);
        t.original_id = i;  // Store original index
        
        tri_info[i].centroid = t.centroid();
        tri_info[i].idx = i;
    }
    
    // Build BVH
    std::vector<BVHNode> nodes;
    nodes.emplace_back();
    build_bvh_recursive(tri_info, 0, num_faces, nodes, 0, n_primitives_per_leaf);
    
    // Thread with escape links
    if (!nodes.empty()) {
        thread_bvh(nodes, 0, -1);
    }
    
    // Create nodes tensor (9 floats per node)
    auto opts_float = torch::TensorOptions().dtype(torch::kFloat32);
    auto nodes_tensor = torch::empty({(int)nodes.size(), 9}, opts_float);
    float* n_ptr = nodes_tensor.data_ptr<float>();
    
    for (size_t i = 0; i < nodes.size(); i++) {
        const auto& node = nodes[i];
        n_ptr[i * 9 + 0] = node.bb_min[0];
        n_ptr[i * 9 + 1] = node.bb_min[1];
        n_ptr[i * 9 + 2] = node.bb_min[2];
        n_ptr[i * 9 + 3] = node.bb_max[0];
        n_ptr[i * 9 + 4] = node.bb_max[1];
        n_ptr[i * 9 + 5] = node.bb_max[2];
        n_ptr[i * 9 + 6] = *reinterpret_cast<const float*>(&node.left_idx);
        n_ptr[i * 9 + 7] = *reinterpret_cast<const float*>(&node.right_idx);
        n_ptr[i * 9 + 8] = *reinterpret_cast<const float*>(&node.escape_idx);
    }
    
    // Create triangles tensor (10 floats per tri: a(3), b(3), c(3), original_id)
    auto triangles_tensor = torch::empty({num_faces, 10}, opts_float);
    float* t_ptr = triangles_tensor.data_ptr<float>();
    
    for (int i = 0; i < num_faces; i++) {
        const Triangle& t = tri_info[i].tri;
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
// Python Bindings for Traversal
// ============================================================

std::vector<at::Tensor> bvh_udf_cuda(
    at::Tensor nodes,
    at::Tensor triangles,
    at::Tensor points
) {
    CHECK_INPUT(nodes);
    CHECK_INPUT(triangles);
    CHECK_INPUT(points);
    
    int num_points = points.size(0);
    
    auto opts_float = points.options();
    auto opts_int = points.options().dtype(torch::kInt32);
    
    auto distances = torch::empty({num_points}, opts_float);
    auto face_ids = torch::empty({num_points}, opts_int);
    auto closest_points = torch::empty({num_points, 3}, opts_float);
    auto uvw = torch::empty({num_points, 3}, opts_float);
    
    int block_size = 256;
    int grid_size = (num_points + block_size - 1) / block_size;
    
    bvh_udf_kernel<<<grid_size, block_size>>>(
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
    
    bvh_ray_intersect_kernel<<<grid_size, block_size>>>(
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
    
    bvh_aabb_intersect_kernel<<<grid_size, block_size>>>(
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

// ============================================================
// Module Definition
// ============================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("build_bvh", &build_bvh_cuda, "Build BVH from mesh");
    m.def("bvh_udf", &bvh_udf_cuda, "BVH UDF query");
    m.def("bvh_ray_intersect", &bvh_ray_intersect_cuda, "BVH ray intersection");
    m.def("bvh_aabb_intersect", &bvh_aabb_intersect_cuda, "BVH AABB intersection");
}
