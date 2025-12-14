"""Core module exports"""

from .mesh_bvh import MeshBVH
from .data_structures import (
    AABBIntersectResult,
    RayIntersectResult,
    SegmentIntersectResult,
    ClosestPointResult,
    TriangleIntersectResult,
    VoxelFaceMapping,
    VoxelPolygonMapping,
    VisibilityResult,
)

__all__ = [
    "MeshBVH",
    "AABBIntersectResult",
    "RayIntersectResult",
    "SegmentIntersectResult",
    "ClosestPointResult",
    "TriangleIntersectResult",
    "VoxelFaceMapping",
    "VoxelPolygonMapping",
    "VisibilityResult",
]
