# Atom3D

**Atomize Your 3D Meshes** - CUDA-accelerated mesh voxelization and distance field queries for 3D deep learning.

<p align="center">
  <img src="https://img.shields.io/badge/CUDA-Accelerated-76B900?logo=nvidia" alt="CUDA">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/License-MIT-blue" alt="License">
</p>

## Features

- **Octree-accelerated voxelization** — Hierarchical coarse-to-fine surface voxel detection
- **SAT intersection** — Precise triangle-AABB collision using Separating Axis Theorem
- **UDF/SDF queries** — Unsigned and signed distance fields with gradient support
- **Ray casting** — Möller-Trumbore ray-triangle intersection

## Installation

```bash
git clone https://github.com/your-org/Atom3D.git
cd Atom3D
pip install -e . --no-build-isolation
```

**Requirements:** Python ≥ 3.8, PyTorch ≥ 2.0, CUDA ≥ 11.0

**Optional:** `pip install trimesh pyvista`

## Quick Start

### Mesh Voxelization

```python
import torch
from atom3d import MeshBVH
from atom3d.grid import OctreeIndexer

# Load mesh
bvh = MeshBVH(vertices.cuda(), faces.cuda(), device='cuda')

# Octree (256³ max resolution)
octree = OctreeIndexer(max_level=8, device='cuda')

# Broadphase: octree traversal
face_min, face_max = bvh.get_face_aabb()
candidates = octree.octree_traverse(face_min, face_max, min_level=4)

# Narrowphase: SAT intersection
voxel_min, voxel_max = octree.cube_aabb_level(candidates, level=8)
result = bvh.intersect_aabb(voxel_min, voxel_max)
surface_voxels = candidates[result.hit]
```

### UDF/SDF Query

```python
points = torch.randn(1000, 3, device='cuda', requires_grad=True)

# Unsigned distance
result = bvh.udf(points, return_grad=True)
# result.distances, result.closest_points

# Signed distance (watertight mesh)
result = bvh.sdf(points, return_grad=True)

# Backprop
result.distances.mean().backward()
```

## API

### MeshBVH

| Method | Description |
|--------|-------------|
| `intersect_aabb(min, max, mode)` | Triangle-AABB SAT intersection |
| `udf(points, return_grad)` | Unsigned distance field |
| `sdf(points, return_grad)` | Signed distance field |
| `intersect_ray(o, d, max_t)` | Ray-mesh intersection |
| `get_face_aabb()` | Per-triangle AABBs |

### Grid

| Class | Description |
|-------|-------------|
| `OctreeIndexer` | Multi-resolution octree with traversal |
| `GridIndexer` | Uniform grid indexer |
| `CubeGrid` | Cube topology (vertex/edge/face indices) |

## Examples

```bash
python examples/basic_voxelization.py
python examples/udf_gradient.py
```

## License

MIT
