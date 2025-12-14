"""
Atom3D: Atomize Your 3D Meshes

A high-performance CUDA library for mesh processing and representation
to support 3D deep learning.
"""

__version__ = "0.1.0"
__author__ = "Atom3D Contributors"

from .core.mesh_bvh import MeshBVH
from .core.data_structures import (
    AABBIntersectResult,
    RayIntersectResult,
    SegmentIntersectResult,
    ClosestPointResult,
    TriangleIntersectResult,
    VoxelFaceMapping,
    VoxelPolygonMapping,
    VisibilityResult,
)
from .grid.grid_indexer import GridIndexer
from .grid.octree_indexer import OctreeIndexer
from .grid.cube_grid import CubeGrid

__all__ = [
    # Core
    "MeshBVH",
    # Data structures
    "AABBIntersectResult",
    "RayIntersectResult",
    "SegmentIntersectResult",
    "ClosestPointResult",
    "TriangleIntersectResult",
    "VoxelFaceMapping",
    "VoxelPolygonMapping",
    "VisibilityResult",
    # Grid
    "GridIndexer",
    "OctreeIndexer",
    "CubeGrid",
]
