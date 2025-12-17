# Atom3D

**Atomize Your 3D Meshes** — High-performance CUDA mesh geometry with internal BVH acceleration for voxelization, distance fields, and intersection queries.

<p align="center">
  <img src="https://img.shields.io/badge/CUDA-Accelerated-76B900?logo=nvidia" alt="CUDA">
  <img src="https://img.shields.io/badge/BVH-Internal-orange" alt="BVH">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/License-MIT-blue" alt="License">
</p>

## Highlights

- **🚀 85x speedup** — BVH-accelerated octree traversal vs brute-force
- **📦 Zero external BVH dependencies** — Fully self-contained, ported from [cubvh](https://github.com/ashawkey/cubvh)
- **⚡ All operations BVH-accelerated** — UDF, ray, segment, AABB intersection

## Features

### Core Geometry (`MeshBVH`)

| Operation | Complexity | BVH Accelerated |
|-----------|------------|-----------------|
| **Triangle-AABB intersection** | O(N log M) | ✅ Exact SAT |
| **Polygon clipping** | O(N log M) | ✅ Broadphase |
| **UDF/SDF queries** | O(N log M) | ✅ Closest point |
| **Ray intersection** | O(N log M) | ✅ Möller-Trumbore |
| **Segment intersection** | O(N log M) | ✅ Via ray |

### Spatial Indexing (`OctreeIndexer`)
- **Octree-accelerated voxelization** — Hierarchical coarse-to-fine surface detection
- **Multi-resolution traversal** — Efficient broadphase filtering from coarse to fine levels
- **Cube topology** — Vertex, edge, and face indexing for primal/dual grids

### CUDA Kernels
- **`bvh_kernels.cu`** — Internal BVH: build, UDF, ray, AABB intersection
- **`cumtv_kernels.cu`** — SAT clip polygon, segment-triangle intersection

## Installation

```bash
git clone https://github.com/your-org/Atom3D.git
cd Atom3D
pip install -e . --no-build-isolation
```

**Requirements:** Python ≥ 3.8, PyTorch ≥ 2.0, CUDA ≥ 11.0

**Optional:** `pip install trimesh pyvista`

## Quick Start

### Surface Voxelization

```python
import torch
from atom3d import MeshBVH
from atom3d.grid import OctreeIndexer

# Load mesh
bvh = MeshBVH(vertices.cuda(), faces.cuda(), device='cuda')

# Create octree (256³ max resolution)
octree = OctreeIndexer(max_level=8, device='cuda')

# BVH-accelerated octree traversal (85x faster than brute-force!)
candidates = octree.octree_traverse(bvh, min_level=4)

# Narrowphase: precise SAT intersection
voxel_min, voxel_max = octree.cube_aabb_level(candidates)
result = bvh.intersect_aabb(voxel_min, voxel_max, mode=1)
surface_voxels = candidates[result.hit]
```

### Polygon Clipping

```python
# Mode 2: Get clipped polygon centroids and areas
result = bvh.intersect_aabb(voxel_min, voxel_max, mode=2)

# result.aabb_ids: which voxel each hit belongs to
# result.face_ids: which triangle was intersected  
# result.centroids: [N, 3] clipped polygon centroids
# result.areas: [N] clipped polygon areas
```

### UDF/SDF Query

```python
points = torch.randn(1000, 3, device='cuda', requires_grad=True)

# Unsigned distance with closest point (BVH-accelerated)
result = bvh.udf(points, return_closest=True, return_uvw=True)
# result.distances, result.closest_points, result.uvw

# Signed distance (requires watertight mesh)
distances = bvh.sdf(points)

# Gradient support
result = bvh.udf(points, return_grad=True)
result.distances.mean().backward()
```

### Ray Intersection

```python
rays_o = torch.randn(1000, 3, device='cuda')
rays_d = torch.randn(1000, 3, device='cuda')
rays_d = rays_d / rays_d.norm(dim=1, keepdim=True)

# BVH-accelerated ray-mesh intersection
result = bvh.intersect_ray(rays_o, rays_d)
# result.hit, result.t, result.face_ids, result.hit_points
```

## Performance

Benchmarks on robot.glb (687K faces) with FaithC_v2 encoding:

| Resolution | Before BVH | After BVH | Speedup |
|------------|-----------|-----------|---------|
| 256 | 15.5s | 0.18s | **85x** |
| 512 | 10.6s | 2.2s | **4.8x** |
| 2048 | 210s | 32.5s | **6.5x** |

## API Reference

### MeshBVH

| Method | Description |
|--------|-------------|
| `intersect_aabb(min, max, mode)` | Triangle-AABB SAT intersection. mode: 0=hit, 1=pairs, 2=clip |
| `udf(points, ...)` | BVH-accelerated unsigned distance field query |
| `sdf(points, ...)` | Signed distance field (watertight mesh required) |
| `intersect_ray(o, d, max_t)` | BVH-accelerated ray-mesh intersection |
| `intersect_segment(start, end)` | Segment-mesh intersection |
| `get_bounds()` | Mesh AABB bounds |
| `get_face_aabb()` | Per-triangle AABBs |

### OctreeIndexer

| Method | Description |
|--------|-------------|
| `octree_traverse(bvh, min_level)` | BVH-accelerated hierarchical broadphase |
| `cube_aabb_level(cubes, level)` | Get voxel AABB at level |
| `ijk_to_cube(ijk)` | Grid coords to linear index |
| `cube_to_ijk(idx)` | Linear index to grid coords |
| `get_cell_size(level)` | Voxel size at level |

### Data Structures

```python
# AABBIntersectResult
result.hit          # [N] bool
result.aabb_ids     # [H] int - which AABB
result.face_ids     # [H] int - which triangle
result.centroids    # [H, 3] float (mode >= 2)
result.areas        # [H] float (mode >= 2)

# ClosestPointResult  
result.distances      # [N] float
result.closest_points # [N, 3] float
result.face_ids       # [N] int
result.uvw            # [N, 3] barycentric coords
```

## Examples

```bash
python examples/basic_voxelization.py
python examples/udf_gradient.py
```

## Acknowledgements

This project builds upon excellent open-source work:

- **[cubvh](https://github.com/ashawkey/cubvh)** — BVH implementation reference (ported and extended internally)
- **[diso](https://github.com/SarahWeiii/diso)** — Differentiable isosurface extraction
- **[FlexiCubes](https://github.com/nv-tlabs/FlexiCubes)** — NVIDIA's flexible isosurface extraction
- **[instant-ngp](https://github.com/NVlabs/instant-ngp)** — NVIDIA's instant neural graphics primitives

## License

[MIT](LICENSE)
