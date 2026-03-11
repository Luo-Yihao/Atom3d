# Atom3D

**High-performance CUDA mesh processing library** with BVH acceleration for voxelization, distance fields, mesh extraction, and intersection queries.

<p align="center">
  <img src="https://img.shields.io/badge/CUDA-Accelerated-76B900?logo=nvidia" alt="CUDA">
  <img src="https://img.shields.io/badge/Sparse-Optimized-orange" alt="Sparse">
  <img src="https://img.shields.io/badge/License-MIT-blue" alt="License">
</p>

## 🎉 News
- **[2026-01]** `atom3d.mesh_extractor.SparseDiffDMC` — Differentiable mesh extraction from sparse voxel input
- **[2026-01]** `atom3d.apps.VisibilityQuery` — Ray-based visibility testing
- **[2026-01]** `atom3d.apps.sparse_flood_fill` — High-performance CUDA flood fill from multi-resolution voxelization


## ✨ Features

Atom3D provides high-performance primitives for modern 3D deep learning:

- **Core Geometry**: `MeshBVH` with SAT intersection, clipping, and UDF/SDF queries.
- **Mesh Extraction**: `SparseDiffDMC` for differentiable, sparse-grid isosurface extraction.
- **Spatial Indexing**: `OctreeIndexer` and `CubeGrid` for efficient sparse data management.
- **Applications**: `VisibilityQuery` and CUDA-optimized `sparse_flood_fill`.

👉 **[See Detailed Features Documentation](docs/features.md)**

## Installation

```bash
pip install -e . --no-build-isolation
```
or

```bash
pip install git+https://github.com/Luo-Yihao/atom3d.git --no-build-isolation
```

**Requirements:** Python ≥ 3.8, PyTorch ≥ 2.0, CUDA ≥ 11.0

## Quick Start

```python
from atom3d import MeshBVH
from atom3d.grid import OctreeIndexer
from atom3d.mesh_extractor import SparseDiffDMC

# BVH-accelerated mesh queries
bvh = MeshBVH(vertices, faces, device='cuda')
result = bvh.udf(points, return_closest=True)

# Octree-based voxelization
octree = OctreeIndexer(max_level=10, device='cuda')
candidates = octree.octree_traverse(bvh, min_level=4)

# Differentiable mesh extraction
dmc = SparseDiffDMC(device='cuda')
mesh_verts, mesh_faces = dmc(voxel_coords, sdf, cube_idx, resolution)
```

## Acknowledgements

- **[cubvh](https://github.com/ashawkey/cubvh)** — BVH implementation reference
- **[FlexiCubes](https://github.com/nv-tlabs/FlexiCubes)** — Differentiable mesh extraction
- **[diso](https://github.com/SarahWeiii/diso)** — Differentiable isosurface extraction

## License

This project is licensed under the [MIT License](LICENSE).
