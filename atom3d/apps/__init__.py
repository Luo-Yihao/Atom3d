"""
Atom3D Apps Module

High-level application components built on core primitives.
"""

from .voxelizer import Voxelizer
from .mesh_intersector import MeshIntersector
from .visibility_query import (
    VisibilityQuery,
    uniform_sphere_cameras,
    fibonacci_sphere_cameras,
    hierarchical_cameras,
)
from .udf_query import UDFQuery
from .sdf_query import SDFQuery
from .sparse_flood_fill import sparse_flood_fill, get_dam_face_labels

__all__ = [
    # Query Applications
    "VisibilityQuery",
    "UDFQuery",
    "SDFQuery",
    
    # Mesh Operations
    "Voxelizer",
    "MeshIntersector",
    
    # Camera Sampling
    "uniform_sphere_cameras",
    "fibonacci_sphere_cameras",
    "hierarchical_cameras",
    
    # Flood Fill
    "sparse_flood_fill",
    "get_dam_face_labels",
]
