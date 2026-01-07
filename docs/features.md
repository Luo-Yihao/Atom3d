# Features Overview

Atom3D provides a comprehensive suite of GPU-accelerated geometry processing tools designed for high-performance 3D learning and reconstruction tasks.

## 🔷 Core Geometry (`atom3d.core`)

The backbone of Atom3D is its highly optimized BVH (Bounding Volume Hierarchy) implementation, enabling blazing fast geometric queries on large meshes.

| Feature | Description | Accelerated By |
|---------|-------------|----------------|
| **MeshBVH** | A CUDA-based BVH structure for triangle meshes. Supports millions of triangles with O(log N) query time. | Internal CUDA BVH |
| **UDF / SDF** | Exact Unsigned and Signed Distance Field queries. Supports finding closest points, face IDs, and barycentric coordinates. | BVH Traversal |
| **Ray Intersection** | Fast ray-mesh intersection testing using Möller–Trumbore algorithm. | BVH Traversal |
| **AABB Intersection** | Precise Separating Axis Theorem (SAT) tests for Triangle-Box intersection. | SAT + BVH |
| **Polygon Clipping** | Efficiently clips mesh triangles against voxel AABBs, returning centroids and areas. | SAT + Sutherland-Hodgman |

## 🔶 Mesh Extraction (`atom3d.mesh_extractor`)

Differentiable isosurface extraction specifically optimized for sparse voxel grids, enabling end-to-end learning of mesh topology and geometry.

- **`SparseDiffDMC`**: A PyTorch-native implementation of Dual Marching Cubes (FlexiCubes variant) that operates on sparse voxel data.
    - **Sparse Input**: No need for dense 3D grids; processes only active voxels.
    - **Differentiable**: Gradients propagate through vertex positions (`deform`) and topological weights (`alpha`, `beta`, `gamma`).
    - **Topology Control**: Implicitly handles topological changes and sharp features via dual contouring principles.

## 🌲 Spatial Indexing (`atom3d.grid`)

Tools for managing sparse 3D data structures and coordinate mapping.

- **`OctreeIndexer`**: A GPU-accelerated octree builder for hierarchical scene analysis.
    - **Multi-resolution**: Supports traversing from coarse to fine levels to quickly identify regions of interest.
    - **Cell Generation**: internal logic to generate candidate cubic cells for voxelization.
- **`CubeGrid`**: A unified topology provider properly implementing standard graphics conventions (x-first vertex ordering).
    - Provides lookups for cube corners, edges, faces, and their connectivity.

## 🛠️ Applications (`atom3d.apps`)

High-level applications built on top of the core primitives.

- **`VisibilityQuery`**: Determines visibility of points from various viewpoints using ray casting.
    - **Omnidirectional**: Fibonacci spiral sampling for robust "ambient occlusion" style visibility.
    - **Camera-based**: Standard frustum/ray visibility from specific camera poses.
- **`sparse_flood_fill`**: A high-performance CUDA flood fill algorithm.
    - **Sign Assignment**: Rapidly propagates inside/outside signs across a voxel grid starting from a seed.
    - **Differentiable-ready**: Provides the necessary sign field for subsequent mesh extraction steps.
