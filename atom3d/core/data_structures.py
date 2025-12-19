"""
Data structures for cuMTV
"""

from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class AABBIntersectResult:
    """AABB intersection result"""
    hit: torch.Tensor           # [N] bool whether each AABB has collision
    aabb_ids: Optional[torch.Tensor] = None   # [total_hits] int32 colliding AABB indices
    face_ids: Optional[torch.Tensor] = None   # [total_hits] int32 colliding face indices
    centroids: Optional[torch.Tensor] = None  # [total_hits, 3] float32 clipped polygon centroids (mode>=2)
    areas: Optional[torch.Tensor] = None      # [total_hits] float32 clipped polygon areas (mode>=2)
    poly_verts: Optional[torch.Tensor] = None # [total_hits, 8, 3] float32 clipped polygon vertices (mode==3)
    poly_counts: Optional[torch.Tensor] = None # [total_hits] int32 polygon vertex counts (mode>=2)


@dataclass
class RayIntersectResult:
    """Ray intersection result"""
    hit: torch.Tensor           # [N] bool
    t: torch.Tensor             # [N] float32 (miss=inf)
    face_ids: torch.Tensor      # [N] int32 (miss=-1)
    hit_points: torch.Tensor    # [N, 3]
    normals: torch.Tensor       # [N, 3]
    bary_coords: torch.Tensor   # [N, 3]


@dataclass
class SegmentIntersectResult:
    """Segment intersection result"""
    hit: torch.Tensor           # [N] bool
    hit_points: torch.Tensor    # [N, 3] or [total, 3]
    face_ids: torch.Tensor      # [N] or [total] int32
    bary_coords: torch.Tensor   # [N, 3] or [total, 3]
    segment_ids: Optional[torch.Tensor] = None  # [total] (if return_all=True)


@dataclass
class ClosestPointResult:
    """Closest point query result (UDF)"""
    distances: torch.Tensor     # [N] float32 unsigned distance
    face_ids: torch.Tensor      # [N] int32 closest face
    closest_points: torch.Tensor  # [N, 3]
    uvw: Optional[torch.Tensor] = None  # [N, 3] barycentric coordinates


@dataclass
class TriangleIntersectResult:
    """Triangle-triangle intersection result"""
    edge_hit: torch.Tensor      # [num_edges] bool whether each edge intersects
    hit_points: torch.Tensor    # [num_hits, 3] intersection point coordinates
    hit_face_ids: torch.Tensor  # [num_hits] int32 hit faces in this mesh
    hit_edge_ids: torch.Tensor  # [num_hits] int32 hit edges in other mesh


@dataclass
class VoxelFaceMapping:
    """Voxel-face mapping (CSR sparse format)"""
    voxel_coords: torch.Tensor  # [K, 3] int32
    face_indices: torch.Tensor  # [total] int32
    face_start: torch.Tensor    # [K] int32
    face_count: torch.Tensor    # [K] int32
    
    def get_faces_for_voxel(self, voxel_idx: int) -> torch.Tensor:
        """Get all faces intersecting the specified voxel"""
        start = self.face_start[voxel_idx].item()
        count = self.face_count[voxel_idx].item()
        return self.face_indices[start:start+count]


@dataclass
class VoxelPolygonMapping:
    """Voxel-polygon mapping (exact intersection region)"""
    voxel_coords: torch.Tensor      # [K, 3] int32
    polygons: torch.Tensor          # [total, max_verts, 3] float32
    polygon_counts: torch.Tensor    # [total] int32 vertex count per polygon
    face_indices: torch.Tensor      # [total] int32
    voxel_ids: torch.Tensor         # [total] int32
    
    def get_polygon(self, idx: int) -> torch.Tensor:
        """Get intersection polygon at specified index"""
        count = self.polygon_counts[idx].item()
        return self.polygons[idx, :count]


@dataclass
class VisibilityResult:
    """Visibility query result"""
    visibility: torch.Tensor        # [N] float32 visibility probability [0, 1]
    visible_mask: Optional[torch.Tensor] = None  # [N, M] bool visibility of each point from each viewpoint
    hit_distances: Optional[torch.Tensor] = None  # [N, M] float32 occlusion distance
