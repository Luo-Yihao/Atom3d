"""
cuMTV CUDA Kernels Python Interface

Provides JIT compilation of CUDA kernels using torch.utils.cpp_extension
"""

import os
import torch
from torch.utils.cpp_extension import load

_kernel_loaded = False
_cumtv_cuda = None


def get_cuda_kernels():
    """
    Load CUDA kernels - use cached .so if available, else JIT compile.
    
    Returns:
        Compiled CUDA module with functions:
        - triangle_aabb_intersect(vertices, faces, aabb_min, aabb_max)
        - ray_mesh_intersect(vertices, faces, rays_o, rays_d, max_t)
        - point_mesh_udf(vertices, faces, points)
        - segment_tri_intersect(seg_verts, tri_verts, tri_aabb_min, tri_aabb_max, eps)
    """
    global _kernel_loaded, _cumtv_cuda
    
    if _kernel_loaded and _cumtv_cuda is not None:
        return _cumtv_cuda
    
    # Get kernel source directory
    kernel_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(kernel_dir, 'build')
    so_file = os.path.join(build_dir, 'cumtv_cuda.so')
    
    # Fast path: if .so exists with expected torch version, load directly
    if os.path.exists(so_file):
        import importlib.util
        spec = importlib.util.spec_from_file_location('cumtv_cuda', so_file)
        _cumtv_cuda = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_cumtv_cuda)
        _kernel_loaded = True
        return _cumtv_cuda
    
    # Slow path: JIT compile
    kernel_file = os.path.join(kernel_dir, 'cumtv_kernels.cu')
    if not os.path.exists(kernel_file):
        raise RuntimeError(f"CUDA kernel file not found: {kernel_file}")
    
    os.makedirs(build_dir, exist_ok=True)
    
    _cumtv_cuda = load(
        name='cumtv_cuda',
        sources=[kernel_file],
        build_directory=build_dir,
        extra_cuda_cflags=['-O3', '--use_fast_math', '-gencode=arch=compute_90,code=sm_90'],
        verbose=False
    )
    
    _kernel_loaded = True
    return _cumtv_cuda


# Convenience functions

def triangle_aabb_intersect(vertices, faces, aabb_min, aabb_max):
    """
    CUDA Triangle-AABB intersection using SAT
    
    Args:
        vertices: [N, 3] float32
        faces: [M, 3] int32
        aabb_min: [K, 3] float32
        aabb_max: [K, 3] float32
    
    Returns:
        hit_mask: [K] bool
        aabb_ids: [num_hits] int32
        face_ids: [num_hits] int32
    """
    cuda = get_cuda_kernels()
    return cuda.triangle_aabb_intersect(
        vertices.contiguous().float(),
        faces.contiguous().int(),
        aabb_min.contiguous().float(),
        aabb_max.contiguous().float()
    )


def ray_mesh_intersect(vertices, faces, rays_o, rays_d, max_t=1e10):
    """
    CUDA Ray-Mesh intersection
    
    Args:
        vertices: [N, 3] float32
        faces: [M, 3] int32
        rays_o: [K, 3] float32
        rays_d: [K, 3] float32
        max_t: float
    
    Returns:
        hit_mask: [K] bool
        hit_t: [K] float32
        hit_face_ids: [K] int32
        hit_points: [K, 3] float32
        hit_uvs: [K, 2] float32
    """
    cuda = get_cuda_kernels()
    return cuda.ray_mesh_intersect(
        vertices.contiguous().float(),
        faces.contiguous().int(),
        rays_o.contiguous().float(),
        rays_d.contiguous().float(),
        float(max_t)
    )


def point_mesh_udf(vertices, faces, points):
    """
    CUDA Point-Mesh UDF query
    
    Args:
        vertices: [N, 3] float32
        faces: [M, 3] int32
        points: [K, 3] float32
    
    Returns:
        distances: [K] float32
        closest_face_ids: [K] int32
        closest_points: [K, 3] float32
        uvw: [K, 3] float32
    """
    cuda = get_cuda_kernels()
    return cuda.point_mesh_udf(
        vertices.contiguous().float(),
        faces.contiguous().int(),
        points.contiguous().float()
    )


def segment_tri_intersect(seg_verts, tri_verts, tri_aabb_min, tri_aabb_max, eps=1e-8):
    """
    CUDA Segment-Triangle intersection
    
    Args:
        seg_verts: [N_seg, 6] float32 (p0, p1)
        tri_verts: [N_tri, 9] float32 (v0, v1, v2)
        tri_aabb_min: [N_tri, 3] float32
        tri_aabb_max: [N_tri, 3] float32
        eps: float
    
    Returns:
        seg_ids: [num_hits] int64
        tri_ids: [num_hits] int64
        t: [num_hits] float32
    """
    cuda = get_cuda_kernels()
    return cuda.segment_tri_intersect(
        seg_verts.contiguous().float(),
        tri_verts.contiguous().float(),
        tri_aabb_min.contiguous().float(),
        tri_aabb_max.contiguous().float(),
        float(eps)
    )


def sat_clip_polygon(aabbs_min, aabbs_max, tris_verts, cand_a, cand_t, mode=1, eps=1e-8):
    """
    CUDA SAT Clip Polygon - compute intersection polygon, centroid, area
    
    For each (AABB, triangle) candidate pair, clips the triangle against the
    AABB using Sutherland-Hodgman algorithm and outputs the resulting polygon.
    
    Args:
        aabbs_min: [K, 3] float32 - AABB min bounds
        aabbs_max: [K, 3] float32 - AABB max bounds
        tris_verts: [M, 9] float32 - Triangle vertices (v0, v1, v2 flattened)
        cand_a: [N] int64 - Candidate AABB indices
        cand_t: [N] int64 - Candidate triangle indices
        mode: int - Output mode:
            0 = hit mask only
            1 = hit mask + centroid + area (default)
            2 = hit mask + centroid + area + full polygon vertices
        eps: float - Tolerance
    
    Returns:
        hit_mask: [N] bool - True if intersection exists
        poly_counts: [N] int32 - Number of vertices in clipped polygon
        poly_verts: [N, 8, 3] float32 - Polygon vertices (only if mode=2)
        centroids: [N, 3] float32 - Centroid of clipped polygon (projected to triangle)
        areas: [N] float32 - Area of clipped polygon
        out_a_idx: [N] int64 - AABB indices
        out_t_idx: [N] int64 - Triangle indices
    """
    cuda = get_cuda_kernels()
    return cuda.sat_clip_polygon(
        aabbs_min.contiguous().float(),
        aabbs_max.contiguous().float(),
        tris_verts.contiguous().float(),
        cand_a.contiguous().long(),
        cand_t.contiguous().long(),
        int(mode),
        float(eps)
    )


# Check if CUDA is available
def cuda_available():
    """Check if CUDA kernels can be compiled and used"""
    if not torch.cuda.is_available():
        return False
    try:
        get_cuda_kernels()
        return True
    except Exception as e:
        print(f"Warning: CUDA kernels not available: {e}")
        return False
