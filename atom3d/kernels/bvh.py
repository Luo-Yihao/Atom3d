"""
BVH Accelerator for Atom3D

Python interface to BVH CUDA kernels. Provides accelerated:
- UDF queries (closest point to mesh)
- Ray-mesh intersection
- AABB-mesh intersection (with exact SAT)
"""

import torch
from torch.utils.cpp_extension import load
import os

# Get the CUDA kernel source
_KERNEL_DIR = os.path.dirname(os.path.abspath(__file__))
_BUILD_DIR = os.path.join(_KERNEL_DIR, 'build', 'bvh')

# Global cache for JIT compiled module
_bvh_cuda = None

def get_bvh_kernels():
    """Get BVH CUDA kernels - use cached .so if available, else JIT compile."""
    global _bvh_cuda
    
    if _bvh_cuda is not None:
        return _bvh_cuda
    
    so_file = os.path.join(_BUILD_DIR, 'bvh_cuda.so')
    
    # Fast path: if .so exists, load directly
    if os.path.exists(so_file):
        import importlib.util
        spec = importlib.util.spec_from_file_location('bvh_cuda', so_file)
        _bvh_cuda = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_bvh_cuda)
        return _bvh_cuda
    
    # Slow path: JIT compile
    os.makedirs(_BUILD_DIR, exist_ok=True)
    
    kernel_path = os.path.join(_KERNEL_DIR, 'bvh_kernels.cu')
    
    _bvh_cuda = load(
        name='bvh_cuda',
        sources=[kernel_path],
        build_directory=_BUILD_DIR,
        extra_cuda_cflags=['-O3', '-gencode=arch=compute_90,code=sm_90'],
        verbose=False
    )
    
    return _bvh_cuda


class BVHAccelerator:
    """
    BVH-accelerated mesh queries.
    
    Provides O(log M) queries instead of O(M) brute-force:
    - udf: closest point to mesh
    - ray_intersect: ray-mesh intersection
    - aabb_intersect: AABB-mesh intersection with exact SAT
    
    All returned face_ids are in ORIGINAL mesh order (not BVH reordered order).
    """
    
    def __init__(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        n_primitives_per_leaf: int = 4  # Reduced from 8 for better accuracy on dense meshes
    ):
        """
        Build BVH from mesh.
        
        Args:
            vertices: [N, 3] float32 vertices
            faces: [M, 3] int32 face indices
            n_primitives_per_leaf: Max triangles per leaf node
        """
        self.device = vertices.device
        self.vertices = vertices.contiguous().float()
        self.faces = faces.contiguous().int()
        self.num_faces = faces.shape[0]
        
        # Build BVH - returns (nodes, triangles with original_id)
        cuda = get_bvh_kernels()
        result = cuda.build_bvh(
            self.vertices,
            self.faces,
            n_primitives_per_leaf
        )
        self.nodes = result[0]        # [num_nodes, 9]
        self.triangles = result[1]    # [num_faces, 10] - includes original_id
    
    def udf(
        self,
        points: torch.Tensor
    ):
        """
        Unsigned distance field query.
        
        Args:
            points: [K, 3] query points
            
        Returns:
            distances: [K] unsigned distances
            face_ids: [K] closest face indices (ORIGINAL order)
            closest_points: [K, 3] closest points on mesh
            uvw: [K, 3] barycentric coordinates
        """
        cuda = get_bvh_kernels()
        distances, face_ids, closest_points, uvw = cuda.bvh_udf(
            self.nodes,
            self.triangles,
            points.contiguous().float()
        )
        return distances, face_ids, closest_points, uvw
    
    def ray_intersect(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        max_t: float = 1e10
    ):
        """
        Ray-mesh intersection.
        
        Args:
            rays_o: [K, 3] ray origins
            rays_d: [K, 3] ray directions
            max_t: Maximum ray distance
            
        Returns:
            hit_mask: [K] bool - whether ray hit mesh
            hit_t: [K] hit distance (max_t if no hit)
            face_ids: [K] hit face indices (ORIGINAL order, -1 if no hit)
            hit_points: [K, 3] hit positions
        """
        cuda = get_bvh_kernels()
        hit_mask, hit_t, face_ids, hit_points = cuda.bvh_ray_intersect(
            self.nodes,
            self.triangles,
            rays_o.contiguous().float(),
            rays_d.contiguous().float(),
            max_t
        )
        return hit_mask, hit_t, face_ids, hit_points
    
    def aabb_intersect(
        self,
        query_min: torch.Tensor,
        query_max: torch.Tensor
    ):
        """
        AABB-mesh intersection with exact SAT test.
        
        Args:
            query_min: [K, 3] query AABB mins
            query_max: [K, 3] query AABB maxs
            
        Returns:
            hit_mask: [K] bool - whether AABB intersects mesh
            aabb_ids: [N] query indices for each intersection pair
            face_ids: [N] face indices (ORIGINAL order) for each pair
        """
        cuda = get_bvh_kernels()
        hit_mask, aabb_ids, face_ids = cuda.bvh_aabb_intersect(
            self.nodes,
            self.triangles,
            query_min.contiguous().float(),
            query_max.contiguous().float()
        )
        return hit_mask, aabb_ids, face_ids


# Check if BVH kernels are available
def bvh_available():
    """Check if BVH CUDA kernels can be compiled."""
    try:
        get_bvh_kernels()
        return True
    except Exception:
        return False
