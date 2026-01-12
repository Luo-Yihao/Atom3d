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
from .device_utils import resolve_device, ensure_same_device, get_default_cuda_device

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
    "resolve_device",
    "ensure_same_device",
    "get_default_cuda_device",
]
