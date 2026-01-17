/**
 * cuMTV CUDA Kernels
 * 
 * Integrated kernels from @SparC3D, @FC_dev, and @cubvh
 * 
 * Core Operations:
 * - Triangle-AABB SAT intersection (from @SparC3D)
 * - Segment-Triangle intersection (from @FC_dev) 
 * - Point-Triangle distance (for UDF)
 * - Ray-Triangle intersection (from @cubvh style)
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <cmath>

// ============================================================
// Utility Functions
// ============================================================

#define CUDA_CHECK(x) do { \
    cudaError_t err = (x); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
    } \
} while(0)

__device__ inline float3 make_float3_from_ptr(const float* ptr) {
    return make_float3(ptr[0], ptr[1], ptr[2]);
}

__device__ inline float dot3(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ inline float3 cross3(float3 a, float3 b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

__device__ inline float3 sub3(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ inline float length3(float3 v) {
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

// ============================================================
// Triangle-AABB SAT Intersection (from @SparC3D)
// ============================================================

/**
 * Test if a triangle intersects an axis-aligned box using SAT
 * Based on Akenine-Möller's Triangle-Box Overlap Test
 * 
 * @param v0, v1, v2: Triangle vertices in world space
 * @param box_min, box_max: AABB bounds
 */
__device__ bool triangle_aabb_sat(
    float3 v0, float3 v1, float3 v2,
    float3 box_min, float3 box_max
) {
    // Compute box center and half-extents
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
    float3 tv0 = sub3(v0, box_center);
    float3 tv1 = sub3(v1, box_center);
    float3 tv2 = sub3(v2, box_center);
    
    // Small epsilon for edge-case detection (borderline intersections)
    const float sat_eps = 1e-6f;
    
    // Quick AABB rejection test with epsilon
    float min_v, max_v;
    min_v = fminf(fminf(tv0.x, tv1.x), tv2.x);
    max_v = fmaxf(fmaxf(tv0.x, tv1.x), tv2.x);
    if (min_v > box_half.x + sat_eps || max_v < -box_half.x - sat_eps) return false;
    
    min_v = fminf(fminf(tv0.y, tv1.y), tv2.y);
    max_v = fmaxf(fmaxf(tv0.y, tv1.y), tv2.y);
    if (min_v > box_half.y + sat_eps || max_v < -box_half.y - sat_eps) return false;
    
    min_v = fminf(fminf(tv0.z, tv1.z), tv2.z);
    max_v = fmaxf(fmaxf(tv0.z, tv1.z), tv2.z);
    if (min_v > box_half.z + sat_eps || max_v < -box_half.z - sat_eps) return false;
    
    // Triangle edges
    float3 e0 = sub3(tv1, tv0);
    float3 e1 = sub3(tv2, tv1);
    float3 e2 = sub3(tv0, tv2);
    
    // Test 9 axes (3 edges x 3 coordinate axes) with epsilon
    #define AXIS_TEST(edge, axis) do { \
        float3 a; \
        if (axis == 0) a = make_float3(0.0f, edge.z, -edge.y); \
        else if (axis == 1) a = make_float3(-edge.z, 0.0f, edge.x); \
        else a = make_float3(edge.y, -edge.x, 0.0f); \
        float p0 = dot3(tv0, a); \
        float p1 = dot3(tv1, a); \
        float p2 = dot3(tv2, a); \
        float min_p = fminf(fminf(p0, p1), p2); \
        float max_p = fmaxf(fmaxf(p0, p1), p2); \
        float rad = box_half.x * fabsf(a.x) + box_half.y * fabsf(a.y) + box_half.z * fabsf(a.z) + sat_eps; \
        if (min_p > rad || max_p < -rad) return false; \
    } while(0)
    
    AXIS_TEST(e0, 0); AXIS_TEST(e0, 1); AXIS_TEST(e0, 2);
    AXIS_TEST(e1, 0); AXIS_TEST(e1, 1); AXIS_TEST(e1, 2);
    AXIS_TEST(e2, 0); AXIS_TEST(e2, 1); AXIS_TEST(e2, 2);
    
    #undef AXIS_TEST
    
    // Test triangle plane with epsilon
    float3 normal = cross3(e0, e1);
    float d = -dot3(normal, tv0);
    float r = box_half.x * fabsf(normal.x) + box_half.y * fabsf(normal.y) + box_half.z * fabsf(normal.z) + sat_eps;
    if (fabsf(d) > r) return false;
    
    return true;
}

/**
 * CUDA Kernel: Triangle-AABB batch intersection test
 * 
 * Grid: (num_aabbs, num_faces / BLOCK_SIZE)
 * Thread: process one (aabb, face) pair
 */
__global__ void triangle_aabb_intersect_kernel(
    const float* __restrict__ vertices,     // [N, 3]
    const int* __restrict__ faces,          // [M, 3]
    const float* __restrict__ aabb_min,     // [K, 3]
    const float* __restrict__ aabb_max,     // [K, 3]
    int num_faces,
    int num_aabbs,
    int max_hits,                           // P0-3 FIX: max output capacity
    bool* __restrict__ hit_mask,            // [K]
    int* __restrict__ aabb_ids,             // [max_hits] output pairs
    int* __restrict__ face_ids,             // [max_hits]
    int* __restrict__ hit_counter           // atomic counter
) {
    int aabb_idx = blockIdx.x;
    int face_idx = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (aabb_idx >= num_aabbs || face_idx >= num_faces) return;
    
    // Load AABB
    float3 box_min = make_float3_from_ptr(aabb_min + aabb_idx * 3);
    float3 box_max = make_float3_from_ptr(aabb_max + aabb_idx * 3);
    
    // Load face vertices
    int idx0 = faces[face_idx * 3 + 0];
    int idx1 = faces[face_idx * 3 + 1];
    int idx2 = faces[face_idx * 3 + 2];
    
    float3 v0 = make_float3_from_ptr(vertices + idx0 * 3);
    float3 v1 = make_float3_from_ptr(vertices + idx1 * 3);
    float3 v2 = make_float3_from_ptr(vertices + idx2 * 3);
    
    // SAT test
    if (triangle_aabb_sat(v0, v1, v2, box_min, box_max)) {
        hit_mask[aabb_idx] = true;
        
        // P0-3 FIX: Bounds check before write to prevent memory overwrite
        int write_idx = atomicAdd(hit_counter, 1);
        if (write_idx < max_hits) {
            aabb_ids[write_idx] = aabb_idx;
            face_ids[write_idx] = face_idx;
        }
    }
}

// ============================================================
// SAT Clip Polygon (from @FC_dev)
// Sutherland-Hodgman polygon clipping for precise intersection
// ============================================================

#define MAX_CLIP_VERTS 8

/**
 * Clip polygon against a plane using Sutherland-Hodgman algorithm
 * 
 * @param in_poly: input polygon vertices [MAXV][3]
 * @param in_count: number of input vertices
 * @param n: plane normal [3]
 * @param d: plane offset (plane: dot(n, p) = d)
 * @param out_poly: output clipped polygon
 * @param eps: tolerance
 * @return: number of output vertices
 */
template<int MAXV>
__device__ __forceinline__ int clip_with_plane(
    const float in_poly[MAXV][3], int in_count,
    const float n[3], float d,
    float out_poly[MAXV][3], float eps
) {
    if (in_count <= 0) return 0;
    
    float dist[MAXV];
    bool inside[MAXV];
    
    #pragma unroll
    for (int i = 0; i < in_count; ++i) {
        dist[i] = in_poly[i][0] * n[0] + in_poly[i][1] * n[1] + in_poly[i][2] * n[2] - d;
        inside[i] = (dist[i] <= eps);
    }
    
    int out_count = 0;
    
    #pragma unroll
    for (int i = 0; i < in_count; ++i) {
        int j = (i + 1 == in_count) ? 0 : (i + 1);
        const float* P = in_poly[i];
        const float* Q = in_poly[j];
        float dP = dist[i], dQ = dist[j];
        bool inP = inside[i], inQ = inside[j];
        
        if (inP && inQ) {
            out_poly[out_count][0] = Q[0];
            out_poly[out_count][1] = Q[1];
            out_poly[out_count][2] = Q[2];
            ++out_count;
        } else if (inP && !inQ) {
            float denom = dP - dQ;
            if (fabsf(denom) > eps) {
                float t = dP / denom;
                out_poly[out_count][0] = P[0] + t * (Q[0] - P[0]);
                out_poly[out_count][1] = P[1] + t * (Q[1] - P[1]);
                out_poly[out_count][2] = P[2] + t * (Q[2] - P[2]);
                ++out_count;
            }
        } else if (!inP && inQ) {
            float denom = dP - dQ;
            if (fabsf(denom) > eps) {
                float t = dP / denom;
                out_poly[out_count][0] = P[0] + t * (Q[0] - P[0]);
                out_poly[out_count][1] = P[1] + t * (Q[1] - P[1]);
                out_poly[out_count][2] = P[2] + t * (Q[2] - P[2]);
                ++out_count;
            }
            out_poly[out_count][0] = Q[0];
            out_poly[out_count][1] = Q[1];
            out_poly[out_count][2] = Q[2];
            ++out_count;
        }
        
        if (out_count >= MAXV) return MAXV;
    }
    return out_count;
}

/**
 * SAT Clip Kernel - Compute intersection polygon, centroid, and area
 * 
 * For each (AABB, triangle) candidate pair:
 * 1. Clip triangle against AABB's 6 faces
 * 2. Compute clipped polygon centroid
 * 3. Project centroid to triangle plane
 * 
 * @param mode: 0=hit only, 1=centroid, 2=full polygon
 */
__global__ void sat_clip_polygon_kernel(
    const float* __restrict__ aabbs_min,    // [K, 3]
    const float* __restrict__ aabbs_max,    // [K, 3]
    const float* __restrict__ tris_verts,   // [M, 3, 3] = [M, 9]
    const long* __restrict__ cand_a,        // [N] aabb indices
    const long* __restrict__ cand_t,        // [N] triangle indices
    int64_t K,
    float eps,
    int mode,                               // 0: hit, 1: centroid, 2: full polygon
    bool* __restrict__ hit_mask,            // [N]
    int* __restrict__ poly_counts,          // [N] number of polygon vertices
    float* __restrict__ poly_verts,         // [N, MAX_CLIP_VERTS, 3]
    float* __restrict__ centroids,          // [N, 3]
    float* __restrict__ areas,              // [N]
    long* __restrict__ out_a_idx,           // [N]
    long* __restrict__ out_t_idx            // [N]
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;
    
    long ai = cand_a[k];
    long ti = cand_t[k];
    out_a_idx[k] = ai;
    out_t_idx[k] = ti;
    
    const float* bmin = aabbs_min + ai * 3;
    const float* bmax = aabbs_max + ai * 3;
    
    // Load triangle
    float tri[3][3];
    #pragma unroll
    for (int v = 0; v < 3; ++v) {
        const float* tv = tris_verts + ti * 9 + v * 3;
        tri[v][0] = tv[0];
        tri[v][1] = tv[1];
        tri[v][2] = tv[2];
    }
    
    // Initialize polygon = triangle
    float polyA[MAX_CLIP_VERTS][3], polyB[MAX_CLIP_VERTS][3];
    int nA = 3;
    for (int i = 0; i < 3; ++i) {
        polyA[i][0] = tri[i][0];
        polyA[i][1] = tri[i][1];
        polyA[i][2] = tri[i][2];
    }
    
    // Clip against 6 AABB faces
    const float nxp[3] = {1, 0, 0}, nyp[3] = {0, 1, 0}, nzp[3] = {0, 0, 1};
    const float nxn[3] = {-1, 0, 0}, nyn[3] = {0, -1, 0}, nzn[3] = {0, 0, -1};
    
    // P0-1 FIX: Replaced lambda with direct macro to avoid CUDA compatibility issues
    #define CLIP_ONCE(inP, inN, n, d, outP) clip_with_plane<MAX_CLIP_VERTS>(inP, inN, n, d, outP, eps)
    
    int nB;
    nB = CLIP_ONCE(polyA, nA, nxp, bmax[0], polyB); nA = nB;
    if (!nA) { hit_mask[k] = false; poly_counts[k] = 0; areas[k] = 0; return; }
    
    nB = CLIP_ONCE(polyB, nA, nxn, -bmin[0], polyA); nA = nB;
    if (!nA) { hit_mask[k] = false; poly_counts[k] = 0; areas[k] = 0; return; }
    
    nB = CLIP_ONCE(polyA, nA, nyp, bmax[1], polyB); nA = nB;
    if (!nA) { hit_mask[k] = false; poly_counts[k] = 0; areas[k] = 0; return; }
    
    nB = CLIP_ONCE(polyB, nA, nyn, -bmin[1], polyA); nA = nB;
    if (!nA) { hit_mask[k] = false; poly_counts[k] = 0; areas[k] = 0; return; }
    
    nB = CLIP_ONCE(polyA, nA, nzp, bmax[2], polyB); nA = nB;
    if (!nA) { hit_mask[k] = false; poly_counts[k] = 0; areas[k] = 0; return; }
    
    nA = CLIP_ONCE(polyB, nB, nzn, -bmin[2], polyA);
    if (!nA) { hit_mask[k] = false; poly_counts[k] = 0; areas[k] = 0; return; }
    
    // Hit!
    hit_mask[k] = true;
    int cnt = nA;
    if (cnt > MAX_CLIP_VERTS) cnt = MAX_CLIP_VERTS;
    poly_counts[k] = cnt;
    
    // Store polygon vertices if mode == 2
    if (mode == 2 && poly_verts != nullptr) {
        for (int i = 0; i < MAX_CLIP_VERTS; ++i) {
            if (i < cnt) {
                poly_verts[(k * MAX_CLIP_VERTS + i) * 3 + 0] = polyA[i][0];
                poly_verts[(k * MAX_CLIP_VERTS + i) * 3 + 1] = polyA[i][1];
                poly_verts[(k * MAX_CLIP_VERTS + i) * 3 + 2] = polyA[i][2];
            } else {
                poly_verts[(k * MAX_CLIP_VERTS + i) * 3 + 0] = 0;
                poly_verts[(k * MAX_CLIP_VERTS + i) * 3 + 1] = 0;
                poly_verts[(k * MAX_CLIP_VERTS + i) * 3 + 2] = 0;
            }
        }
    }
    
    if (cnt < 1) {
        hit_mask[k] = false;  // CRITICAL FIX: No intersection if polygon degenerates  
        areas[k] = 0;
        centroids[k * 3 + 0] = 0;
        centroids[k * 3 + 1] = 0;
        centroids[k * 3 + 2] = 0;
        return;
    }
    
    // Compute centroid (arithmetic mean)
    float cx = 0, cy = 0, cz = 0;
    for (int i = 0; i < cnt; ++i) {
        cx += polyA[i][0];
        cy += polyA[i][1];
        cz += polyA[i][2];
    }
    cx /= (float)cnt;
    cy /= (float)cnt;
    cz /= (float)cnt;
    
    // Clamp to voxel bounds
    cx = fminf(fmaxf(cx, bmin[0]), bmax[0]);
    cy = fminf(fmaxf(cy, bmin[1]), bmax[1]);
    cz = fminf(fmaxf(cz, bmin[2]), bmax[2]);
    
    // Project centroid onto triangle plane with barycentric clamping
    float ax = tri[0][0], ay = tri[0][1], az = tri[0][2];
    float bx = tri[1][0], by = tri[1][1], bz = tri[1][2];
    float cx_t = tri[2][0], cy_t = tri[2][1], cz_t = tri[2][2];
    
    float e1x = bx - ax, e1y = by - ay, e1z = bz - az;
    float e2x = cx_t - ax, e2y = cy_t - ay, e2z = cz_t - az;
    float vx = cx - ax, vy = cy - ay, vz = cz - az;
    
    float d00 = e1x * e1x + e1y * e1y + e1z * e1z;
    float d01 = e1x * e2x + e1y * e2y + e1z * e2z;
    float d11 = e2x * e2x + e2y * e2y + e2z * e2z;
    float d20 = vx * e1x + vy * e1y + vz * e1z;
    float d21 = vx * e2x + vy * e2y + vz * e2z;
    float denom = d00 * d11 - d01 * d01;
    
    // P0 FIX: Guard against degenerate triangle (denom near zero -> NaN/Inf)
    if (fabsf(denom) < eps * eps) {
        // Fallback: use first vertex as centroid for degenerate triangles
        centroids[k * 3 + 0] = ax;
        centroids[k * 3 + 1] = ay;
        centroids[k * 3 + 2] = az;
        return;
    }
    
    float v_bc = (d11 * d20 - d01 * d21) / denom;
    float w_bc = (d00 * d21 - d01 * d20) / denom;
    float u_bc = 1.0f - v_bc - w_bc;
    
    // Clamp barycentric coords
    if (u_bc < 0) u_bc = 0;
    if (v_bc < 0) v_bc = 0;
    if (w_bc < 0) w_bc = 0;
    float norm = u_bc + v_bc + w_bc;
    if (norm <= eps) { u_bc = 1; v_bc = 0; w_bc = 0; norm = 1; }
    u_bc /= norm; v_bc /= norm; w_bc /= norm;
    
    float projx = u_bc * ax + v_bc * bx + w_bc * cx_t;
    float projy = u_bc * ay + v_bc * by + w_bc * cy_t;
    float projz = u_bc * az + v_bc * bz + w_bc * cz_t;
    
    centroids[k * 3 + 0] = projx;
    centroids[k * 3 + 1] = projy;
    centroids[k * 3 + 2] = projz;
    
    // Compute approximate area (fan triangulation from centroid)
    float area = 0;
    for (int i = 0; i < cnt; ++i) {
        int j = (i + 1) % cnt;
        float e1_x = polyA[i][0] - cx, e1_y = polyA[i][1] - cy, e1_z = polyA[i][2] - cz;
        float e2_x = polyA[j][0] - cx, e2_y = polyA[j][1] - cy, e2_z = polyA[j][2] - cz;
        float cross_x = e1_y * e2_z - e1_z * e2_y;
        float cross_y = e1_z * e2_x - e1_x * e2_z;
        float cross_z = e1_x * e2_y - e1_y * e2_x;
        area += 0.5f * sqrtf(cross_x * cross_x + cross_y * cross_y + cross_z * cross_z);
    }
    areas[k] = area;
    
    // NOTE: For area < eps (point/line contact), we keep hit_mask=true
    // because contact counts as collision. Only set area=0 explicitly.
    if (area < eps) {
        areas[k] = 0.0f;  // Explicit zero for degenerate cases
        // hit_mask stays true - contact counts as collision
    }
}

// ============================================================
// Ray-Triangle Intersection 
// ============================================================

/**
 * Möller-Trumbore ray-triangle intersection
 * Returns t parameter, -1 if no hit
 */
__device__ float ray_triangle_intersect(
    float3 ray_o, float3 ray_d,
    float3 v0, float3 v1, float3 v2,
    float* out_u, float* out_v
) {
    const float eps = 1e-8f;
    
    float3 e1 = sub3(v1, v0);
    float3 e2 = sub3(v2, v0);
    float3 h = cross3(ray_d, e2);
    float a = dot3(e1, h);
    
    if (fabsf(a) < eps) return -1.0f;
    
    float f = 1.0f / a;
    float3 s = sub3(ray_o, v0);
    float u = f * dot3(s, h);
    
    if (u < 0.0f || u > 1.0f) return -1.0f;
    
    float3 q = cross3(s, e1);
    float v = f * dot3(ray_d, q);
    
    if (v < 0.0f || u + v > 1.0f) return -1.0f;
    
    float t = f * dot3(e2, q);
    
    if (t > eps) {
        *out_u = u;
        *out_v = v;
        return t;
    }
    
    return -1.0f;
}

/**
 * CUDA Kernel: Ray-Mesh intersection (brute force)
 * Each thread processes one ray against all triangles
 */
__global__ void ray_mesh_intersect_kernel(
    const float* __restrict__ vertices,     // [N, 3]
    const int* __restrict__ faces,          // [M, 3]
    const float* __restrict__ rays_o,       // [K, 3]
    const float* __restrict__ rays_d,       // [K, 3]
    int num_rays,
    int num_faces,
    float max_t,
    bool* __restrict__ hit_mask,            // [K]
    float* __restrict__ hit_t,              // [K]
    int* __restrict__ hit_face_ids,         // [K]
    float* __restrict__ hit_points,         // [K, 3]
    float* __restrict__ hit_uvs             // [K, 2]
) {
    int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_idx >= num_rays) return;
    
    float3 ray_o = make_float3_from_ptr(rays_o + ray_idx * 3);
    float3 ray_d = make_float3_from_ptr(rays_d + ray_idx * 3);
    
    float best_t = max_t;
    int best_face = -1;
    float best_u = 0, best_v = 0;
    
    for (int f = 0; f < num_faces; f++) {
        int idx0 = faces[f * 3 + 0];
        int idx1 = faces[f * 3 + 1];
        int idx2 = faces[f * 3 + 2];
        
        float3 v0 = make_float3_from_ptr(vertices + idx0 * 3);
        float3 v1 = make_float3_from_ptr(vertices + idx1 * 3);
        float3 v2 = make_float3_from_ptr(vertices + idx2 * 3);
        
        float u, v;
        float t = ray_triangle_intersect(ray_o, ray_d, v0, v1, v2, &u, &v);
        
        if (t > 0 && t < best_t) {
            best_t = t;
            best_face = f;
            best_u = u;
            best_v = v;
        }
    }
    
    if (best_face >= 0) {
        hit_mask[ray_idx] = true;
        hit_t[ray_idx] = best_t;
        hit_face_ids[ray_idx] = best_face;
        
        // Compute hit point
        hit_points[ray_idx * 3 + 0] = ray_o.x + best_t * ray_d.x;
        hit_points[ray_idx * 3 + 1] = ray_o.y + best_t * ray_d.y;
        hit_points[ray_idx * 3 + 2] = ray_o.z + best_t * ray_d.z;
        
        hit_uvs[ray_idx * 2 + 0] = best_u;
        hit_uvs[ray_idx * 2 + 1] = best_v;
    } else {
        hit_mask[ray_idx] = false;
        hit_t[ray_idx] = max_t;
        hit_face_ids[ray_idx] = -1;
    }
}

// ============================================================
// Point-Triangle Distance (for UDF)
// ============================================================

/**
 * Compute closest point on triangle to query point
 * Returns squared distance
 */
__device__ float point_triangle_distance_sq(
    float3 p, float3 v0, float3 v1, float3 v2,
    float3* closest_point, float* out_u, float* out_v, float* out_w
) {
    float3 e0 = sub3(v1, v0);
    float3 e1 = sub3(v2, v0);
    float3 v0_to_p = sub3(p, v0);
    
    float d00 = dot3(e0, e0);
    float d01 = dot3(e0, e1);
    float d11 = dot3(e1, e1);
    float d20 = dot3(v0_to_p, e0);
    float d21 = dot3(v0_to_p, e1);
    
    float denom = d00 * d11 - d01 * d01;
    if (fabsf(denom) < 1e-8f) denom = 1e-8f;
    
    float v = (d11 * d20 - d01 * d21) / denom;
    float w = (d00 * d21 - d01 * d20) / denom;
    float u = 1.0f - v - w;
    
    // Clamp to triangle
    if (u < 0) {
        // Project to edge v1-v2
        float3 e = sub3(v2, v1);
        float len_sq = dot3(e, e);
        float3 diff = sub3(p, v1);
        float t = fmaxf(0.0f, fminf(1.0f, dot3(diff, e) / len_sq));
        *closest_point = make_float3(v1.x + t * e.x, v1.y + t * e.y, v1.z + t * e.z);
        *out_u = 0; *out_v = 1 - t; *out_w = t;
    } else if (v < 0) {
        // Project to edge v0-v2
        float3 e = sub3(v2, v0);
        float len_sq = dot3(e, e);
        float3 diff = sub3(p, v0);
        float t = fmaxf(0.0f, fminf(1.0f, dot3(diff, e) / len_sq));
        *closest_point = make_float3(v0.x + t * e.x, v0.y + t * e.y, v0.z + t * e.z);
        *out_u = 1 - t; *out_v = 0; *out_w = t;
    } else if (w < 0) {
        // Project to edge v0-v1
        float3 e = sub3(v1, v0);
        float len_sq = dot3(e, e);
        float3 diff = sub3(p, v0);
        float t = fmaxf(0.0f, fminf(1.0f, dot3(diff, e) / len_sq));
        *closest_point = make_float3(v0.x + t * e.x, v0.y + t * e.y, v0.z + t * e.z);
        *out_u = 1 - t; *out_v = t; *out_w = 0;
    } else {
        // Inside triangle
        *closest_point = make_float3(
            u * v0.x + v * v1.x + w * v2.x,
            u * v0.y + v * v1.y + w * v2.y,
            u * v0.z + v * v1.z + w * v2.z
        );
        *out_u = u; *out_v = v; *out_w = w;
    }
    
    float3 diff = sub3(p, *closest_point);
    return dot3(diff, diff);
}

/**
 * CUDA Kernel: Point-Mesh UDF (brute force)
 */
__global__ void point_mesh_udf_kernel(
    const float* __restrict__ vertices,     // [N, 3]
    const int* __restrict__ faces,          // [M, 3]
    const float* __restrict__ points,       // [K, 3]
    int num_points,
    int num_faces,
    float* __restrict__ distances,          // [K]
    int* __restrict__ closest_face_ids,     // [K]
    float* __restrict__ closest_points,     // [K, 3]
    float* __restrict__ uvw                 // [K, 3]
) {
    int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (point_idx >= num_points) return;
    
    float3 p = make_float3_from_ptr(points + point_idx * 3);
    
    float best_dist_sq = 1e30f;
    int best_face = -1;
    float3 best_closest;
    float best_u, best_v, best_w;
    
    for (int f = 0; f < num_faces; f++) {
        int idx0 = faces[f * 3 + 0];
        int idx1 = faces[f * 3 + 1];
        int idx2 = faces[f * 3 + 2];
        
        float3 v0 = make_float3_from_ptr(vertices + idx0 * 3);
        float3 v1 = make_float3_from_ptr(vertices + idx1 * 3);
        float3 v2 = make_float3_from_ptr(vertices + idx2 * 3);
        
        float3 closest;
        float u, v, w;
        float dist_sq = point_triangle_distance_sq(p, v0, v1, v2, &closest, &u, &v, &w);
        
        if (dist_sq < best_dist_sq) {
            best_dist_sq = dist_sq;
            best_face = f;
            best_closest = closest;
            best_u = u; best_v = v; best_w = w;
        }
    }
    
    distances[point_idx] = sqrtf(best_dist_sq);
    closest_face_ids[point_idx] = best_face;
    closest_points[point_idx * 3 + 0] = best_closest.x;
    closest_points[point_idx * 3 + 1] = best_closest.y;
    closest_points[point_idx * 3 + 2] = best_closest.z;
    uvw[point_idx * 3 + 0] = best_u;
    uvw[point_idx * 3 + 1] = best_v;
    uvw[point_idx * 3 + 2] = best_w;
}

// ============================================================
// Segment-Triangle Intersection (from @FC_dev)
// ============================================================

#define FUSED_TILE_SIZE 256

/**
 * Segment-Triangle intersection using Möller-Trumbore
 * Used for mesh-mesh collision detection
 */
__global__ void segment_tri_intersection_kernel(
    const float* __restrict__ seg_verts,    // [N_seg, 6] (p0, p1)
    const float* __restrict__ tri_verts,    // [N_tri, 9] (v0, v1, v2)
    const float* __restrict__ tri_aabb_min, // [N_tri, 3]
    const float* __restrict__ tri_aabb_max, // [N_tri, 3]
    int num_segs,
    int num_tris,
    int max_hits,  // P0 FIX: max output capacity
    float eps,
    long* __restrict__ out_seg_ids,
    long* __restrict__ out_tri_ids,
    float* __restrict__ out_t,
    int* __restrict__ counter
) {
    int seg_idx = blockIdx.x;
    if (seg_idx >= num_segs) return;
    
    // Load segment
    float3 p0 = make_float3_from_ptr(seg_verts + seg_idx * 6);
    float3 p1 = make_float3_from_ptr(seg_verts + seg_idx * 6 + 3);
    
    float3 seg_min = make_float3(
        fminf(p0.x, p1.x), fminf(p0.y, p1.y), fminf(p0.z, p1.z)
    );
    float3 seg_max = make_float3(
        fmaxf(p0.x, p1.x), fmaxf(p0.y, p1.y), fmaxf(p0.z, p1.z)
    );
    
    // FIX: Normalize ray direction for correct t interpretation
    float3 dir = sub3(p1, p0);
    float seg_len = length3(dir);
    if (seg_len <= eps) return;  // Skip zero-length segments
    float inv_len = 1.0f / seg_len;
    float3 ray_d = make_float3(dir.x * inv_len, dir.y * inv_len, dir.z * inv_len);
    
    extern __shared__ float smem[];
    float* tri_min_tile = smem;
    float* tri_max_tile = tri_min_tile + FUSED_TILE_SIZE * 3;
    
    for (int tile = 0; tile < num_tris; tile += FUSED_TILE_SIZE) {
        // Load tile
        int local_idx = threadIdx.x;
        int global_tri = tile + local_idx;
        
        if (global_tri < num_tris) {
            tri_min_tile[local_idx * 3 + 0] = tri_aabb_min[global_tri * 3 + 0];
            tri_min_tile[local_idx * 3 + 1] = tri_aabb_min[global_tri * 3 + 1];
            tri_min_tile[local_idx * 3 + 2] = tri_aabb_min[global_tri * 3 + 2];
            tri_max_tile[local_idx * 3 + 0] = tri_aabb_max[global_tri * 3 + 0];
            tri_max_tile[local_idx * 3 + 1] = tri_aabb_max[global_tri * 3 + 1];
            tri_max_tile[local_idx * 3 + 2] = tri_aabb_max[global_tri * 3 + 2];
        }
        __syncthreads();
        
        int current_tri = tile + threadIdx.x;
        if (current_tri < num_tris) {
            // AABB overlap test
            bool overlap = 
                (seg_max.x >= tri_min_tile[threadIdx.x * 3 + 0] && 
                 seg_min.x <= tri_max_tile[threadIdx.x * 3 + 0]) &&
                (seg_max.y >= tri_min_tile[threadIdx.x * 3 + 1] && 
                 seg_min.y <= tri_max_tile[threadIdx.x * 3 + 1]) &&
                (seg_max.z >= tri_min_tile[threadIdx.x * 3 + 2] && 
                 seg_min.z <= tri_max_tile[threadIdx.x * 3 + 2]);
            
            if (overlap) {
                float3 v0 = make_float3_from_ptr(tri_verts + current_tri * 9);
                float3 v1 = make_float3_from_ptr(tri_verts + current_tri * 9 + 3);
                float3 v2 = make_float3_from_ptr(tri_verts + current_tri * 9 + 6);
                
                float u, v;
                float t_dist = ray_triangle_intersect(p0, ray_d, v0, v1, v2, &u, &v);
                
                // FIX: Use distance-based check (t_dist is actual distance since ray_d is normalized)
                float t_eps = 1e-6f;
                if (t_dist >= -t_eps && t_dist <= seg_len + t_eps) {
                    float t_seg = t_dist * inv_len;  // Convert to [0,1] segment parameter
                    // P0 FIX: Bounds check before write
                    int write_idx = atomicAdd(counter, 1);
                    if (write_idx < max_hits) {
                        out_seg_ids[write_idx] = seg_idx;
                        out_tri_ids[write_idx] = current_tri;
                        out_t[write_idx] = t_seg;  // Store normalized segment parameter
                    }
                }
            }
        }
        __syncthreads();
    }
}

// ============================================================
// PyTorch C++ Bindings
// ============================================================

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

/**
 * Triangle-AABB intersection
 */
std::vector<at::Tensor> triangle_aabb_intersect_cuda(
    at::Tensor vertices,
    at::Tensor faces,
    at::Tensor aabb_min,
    at::Tensor aabb_max
) {
    CHECK_INPUT(vertices);
    CHECK_INPUT(faces);
    CHECK_INPUT(aabb_min);
    CHECK_INPUT(aabb_max);
    
    // P0-2 FIX: Enforce strict dtype checks to prevent memory corruption
    TORCH_CHECK(vertices.scalar_type() == at::kFloat, "vertices must be float32");
    TORCH_CHECK(faces.scalar_type() == at::kInt, "faces must be int32, got ", faces.scalar_type());
    TORCH_CHECK(aabb_min.scalar_type() == at::kFloat, "aabb_min must be float32");
    TORCH_CHECK(aabb_max.scalar_type() == at::kFloat, "aabb_max must be float32");
    
    int num_faces = faces.size(0);
    int num_aabbs = aabb_min.size(0);
    
    auto opts_bool = aabb_min.options().dtype(torch::kBool);
    auto opts_int = aabb_min.options().dtype(torch::kInt32);
    
    auto hit_mask = torch::zeros({num_aabbs}, opts_bool);
    
    // Pre-allocate output buffers
    int64_t max_hits = (int64_t)num_aabbs * num_faces;  // Worst case
    max_hits = std::min(max_hits, (int64_t)num_aabbs * 100);  // Reasonable limit
    
    auto aabb_ids = torch::empty({max_hits}, opts_int);
    auto face_ids = torch::empty({max_hits}, opts_int);
    auto hit_counter = torch::zeros({1}, opts_int);
    
    // Launch kernel
    int block_size = 256;
    dim3 grid(num_aabbs, (num_faces + block_size - 1) / block_size);
    
    triangle_aabb_intersect_kernel<<<grid, block_size>>>(
        vertices.data_ptr<float>(),
        faces.data_ptr<int>(),
        aabb_min.data_ptr<float>(),
        aabb_max.data_ptr<float>(),
        num_faces,
        num_aabbs,
        (int)max_hits,  // P0-3 FIX: Pass max_hits for bounds check
        hit_mask.data_ptr<bool>(),
        aabb_ids.data_ptr<int>(),
        face_ids.data_ptr<int>(),
        hit_counter.data_ptr<int>()
    );
    
    int final_hits = hit_counter[0].item<int>();
    
    return {
        hit_mask,
        aabb_ids.slice(0, 0, final_hits),
        face_ids.slice(0, 0, final_hits)
    };
}

/**
 * Ray-Mesh intersection
 */
std::vector<at::Tensor> ray_mesh_intersect_cuda(
    at::Tensor vertices,
    at::Tensor faces,
    at::Tensor rays_o,
    at::Tensor rays_d,
    float max_t
) {
    CHECK_INPUT(vertices);
    CHECK_INPUT(faces);
    CHECK_INPUT(rays_o);
    CHECK_INPUT(rays_d);
    
    // P0-2 FIX: dtype validation
    TORCH_CHECK(vertices.scalar_type() == at::kFloat, "vertices must be float32");
    TORCH_CHECK(faces.scalar_type() == at::kInt, "faces must be int32, got ", faces.scalar_type());
    TORCH_CHECK(rays_o.scalar_type() == at::kFloat, "rays_o must be float32");
    TORCH_CHECK(rays_d.scalar_type() == at::kFloat, "rays_d must be float32");
    
    int num_faces = faces.size(0);
    int num_rays = rays_o.size(0);
    
    auto opts_bool = rays_o.options().dtype(torch::kBool);
    auto opts_int = rays_o.options().dtype(torch::kInt32);
    auto opts_float = rays_o.options();
    
    auto hit_mask = torch::zeros({num_rays}, opts_bool);
    auto hit_t = torch::full({num_rays}, max_t, opts_float);
    auto hit_face_ids = torch::full({num_rays}, -1, opts_int);
    auto hit_points = torch::zeros({num_rays, 3}, opts_float);
    auto hit_uvs = torch::zeros({num_rays, 2}, opts_float);
    
    int block_size = 256;
    int grid_size = (num_rays + block_size - 1) / block_size;
    
    ray_mesh_intersect_kernel<<<grid_size, block_size>>>(
        vertices.data_ptr<float>(),
        faces.data_ptr<int>(),
        rays_o.data_ptr<float>(),
        rays_d.data_ptr<float>(),
        num_rays,
        num_faces,
        max_t,
        hit_mask.data_ptr<bool>(),
        hit_t.data_ptr<float>(),
        hit_face_ids.data_ptr<int>(),
        hit_points.data_ptr<float>(),
        hit_uvs.data_ptr<float>()
    );
    
    return {hit_mask, hit_t, hit_face_ids, hit_points, hit_uvs};
}

/**
 * Point-Mesh UDF
 */
std::vector<at::Tensor> point_mesh_udf_cuda(
    at::Tensor vertices,
    at::Tensor faces,
    at::Tensor points
) {
    CHECK_INPUT(vertices);
    CHECK_INPUT(faces);
    CHECK_INPUT(points);
    
    // P0-2 FIX: dtype validation
    TORCH_CHECK(vertices.scalar_type() == at::kFloat, "vertices must be float32");
    TORCH_CHECK(faces.scalar_type() == at::kInt, "faces must be int32, got ", faces.scalar_type());
    TORCH_CHECK(points.scalar_type() == at::kFloat, "points must be float32");
    
    int num_faces = faces.size(0);
    int num_points = points.size(0);
    
    auto opts_int = points.options().dtype(torch::kInt32);
    auto opts_float = points.options();
    
    auto distances = torch::zeros({num_points}, opts_float);
    auto closest_face_ids = torch::zeros({num_points}, opts_int);
    auto closest_points = torch::zeros({num_points, 3}, opts_float);
    auto uvw = torch::zeros({num_points, 3}, opts_float);
    
    int block_size = 256;
    int grid_size = (num_points + block_size - 1) / block_size;
    
    point_mesh_udf_kernel<<<grid_size, block_size>>>(
        vertices.data_ptr<float>(),
        faces.data_ptr<int>(),
        points.data_ptr<float>(),
        num_points,
        num_faces,
        distances.data_ptr<float>(),
        closest_face_ids.data_ptr<int>(),
        closest_points.data_ptr<float>(),
        uvw.data_ptr<float>()
    );
    
    return {distances, closest_face_ids, closest_points, uvw};
}

/**
 * Segment-Triangle intersection
 */
std::vector<at::Tensor> segment_tri_intersect_cuda(
    at::Tensor seg_verts,
    at::Tensor tri_verts,
    at::Tensor tri_aabb_min,
    at::Tensor tri_aabb_max,
    float eps
) {
    CHECK_INPUT(seg_verts);
    CHECK_INPUT(tri_verts);
    CHECK_INPUT(tri_aabb_min);
    CHECK_INPUT(tri_aabb_max);
    
    // P0-2 FIX: dtype validation
    TORCH_CHECK(seg_verts.scalar_type() == at::kFloat, "seg_verts must be float32");
    TORCH_CHECK(tri_verts.scalar_type() == at::kFloat, "tri_verts must be float32");
    TORCH_CHECK(tri_aabb_min.scalar_type() == at::kFloat, "tri_aabb_min must be float32");
    TORCH_CHECK(tri_aabb_max.scalar_type() == at::kFloat, "tri_aabb_max must be float32");
    
    int num_segs = seg_verts.size(0);
    int num_tris = tri_verts.size(0);
    
    int64_t max_hits = (int64_t)num_segs * 12 + 8192;
    
    auto opts_long = seg_verts.options().dtype(torch::kInt64);
    auto opts_float = seg_verts.options();
    auto opts_int = seg_verts.options().dtype(torch::kInt32);
    
    auto out_seg_ids = torch::empty({max_hits}, opts_long);
    auto out_tri_ids = torch::empty({max_hits}, opts_long);
    auto out_t = torch::empty({max_hits}, opts_float);
    auto counter = torch::zeros({1}, opts_int);
    
    dim3 blocks(num_segs);
    int threads = FUSED_TILE_SIZE;
    size_t smem = threads * 3 * 2 * sizeof(float);
    
    segment_tri_intersection_kernel<<<blocks, threads, smem>>>(
        seg_verts.data_ptr<float>(),
        tri_verts.data_ptr<float>(),
        tri_aabb_min.data_ptr<float>(),
        tri_aabb_max.data_ptr<float>(),
        num_segs,
        num_tris,
        (int)max_hits,  // P0 FIX: pass max_hits for bounds check
        eps,
        out_seg_ids.data_ptr<long>(),
        out_tri_ids.data_ptr<long>(),
        out_t.data_ptr<float>(),
        counter.data_ptr<int>()
    );
    
    int final_hits = counter[0].item<int>();
    
    return {
        out_seg_ids.slice(0, 0, final_hits),
        out_tri_ids.slice(0, 0, final_hits),
        out_t.slice(0, 0, final_hits)
    };
}

// ============================================================
// Python Module Registration
// ============================================================

/**
 * SAT Clip Polygon - compute intersection polygon, centroid, area
 * 
 * mode: 0 = hit only, 1 = centroid only, 2 = full polygon
 */
std::vector<at::Tensor> sat_clip_polygon_cuda(
    at::Tensor aabbs_min,      // [K, 3]
    at::Tensor aabbs_max,      // [K, 3]
    at::Tensor tris_verts,     // [M, 9]
    at::Tensor cand_a,         // [N] aabb indices
    at::Tensor cand_t,         // [N] triangle indices
    int mode,                  // 0=hit, 1=centroid, 2=full
    float eps
) {
    CHECK_INPUT(aabbs_min);
    CHECK_INPUT(aabbs_max);
    CHECK_INPUT(tris_verts);
    CHECK_INPUT(cand_a);
    CHECK_INPUT(cand_t);
    
    // P0-2 FIX: dtype validation
    TORCH_CHECK(aabbs_min.scalar_type() == at::kFloat, "aabbs_min must be float32");
    TORCH_CHECK(aabbs_max.scalar_type() == at::kFloat, "aabbs_max must be float32");
    TORCH_CHECK(tris_verts.scalar_type() == at::kFloat, "tris_verts must be float32");
    TORCH_CHECK(cand_a.scalar_type() == at::kLong, "cand_a must be int64");
    TORCH_CHECK(cand_t.scalar_type() == at::kLong, "cand_t must be int64");
    
    int64_t K = cand_a.size(0);
    
    auto opts_bool = aabbs_min.options().dtype(torch::kBool);
    auto opts_int = aabbs_min.options().dtype(torch::kInt32);
    auto opts_long = aabbs_min.options().dtype(torch::kInt64);
    auto opts_float = aabbs_min.options();
    
    auto hit_mask = torch::zeros({K}, opts_bool);
    auto poly_counts = torch::zeros({K}, opts_int);
    auto centroids = torch::zeros({K, 3}, opts_float);
    auto areas = torch::zeros({K}, opts_float);
    auto out_a_idx = torch::empty({K}, opts_long);
    auto out_t_idx = torch::empty({K}, opts_long);
    
    at::Tensor poly_verts;
    if (mode == 2) {
        poly_verts = torch::zeros({K, MAX_CLIP_VERTS, 3}, opts_float);
    } else {
        poly_verts = torch::empty({0}, opts_float);
    }
    
    int block_size = 256;
    int grid_size = (K + block_size - 1) / block_size;
    
    sat_clip_polygon_kernel<<<grid_size, block_size>>>(
        aabbs_min.data_ptr<float>(),
        aabbs_max.data_ptr<float>(),
        tris_verts.data_ptr<float>(),
        cand_a.data_ptr<long>(),
        cand_t.data_ptr<long>(),
        K,
        eps,
        mode,
        hit_mask.data_ptr<bool>(),
        poly_counts.data_ptr<int>(),
        (mode == 2) ? poly_verts.data_ptr<float>() : nullptr,
        centroids.data_ptr<float>(),
        areas.data_ptr<float>(),
        out_a_idx.data_ptr<long>(),
        out_t_idx.data_ptr<long>()
    );
    
    return {hit_mask, poly_counts, poly_verts, centroids, areas, out_a_idx, out_t_idx};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("triangle_aabb_intersect", &triangle_aabb_intersect_cuda,
          "Triangle-AABB SAT intersection (CUDA)");
    m.def("ray_mesh_intersect", &ray_mesh_intersect_cuda,
          "Ray-Mesh intersection (CUDA)");
    m.def("point_mesh_udf", &point_mesh_udf_cuda,
          "Point-Mesh UDF query (CUDA)");
    m.def("segment_tri_intersect", &segment_tri_intersect_cuda,
          "Segment-Triangle intersection (CUDA)");
    m.def("sat_clip_polygon", &sat_clip_polygon_cuda,
          "SAT clip polygon - returns intersection polygon, centroid, area (CUDA)");
}
