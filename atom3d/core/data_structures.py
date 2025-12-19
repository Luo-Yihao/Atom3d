"""
Data structures for cuMTV
"""

from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class AABBIntersectResult:
    """AABB碰撞检测结果"""
    hit: torch.Tensor           # [N] bool 每个AABB是否有碰撞
    aabb_ids: Optional[torch.Tensor] = None   # [total_hits] int32 碰撞的AABB索引
    face_ids: Optional[torch.Tensor] = None   # [total_hits] int32 碰撞的面索引
    centroids: Optional[torch.Tensor] = None
    areas: Optional[torch.Tensor] = None
    poly_verts: Optional[torch.Tensor] = None
    poly_counts: Optional[torch.Tensor] = None


@dataclass
class RayIntersectResult:
    """射线相交检测结果"""
    hit: torch.Tensor           # [N] bool
    t: torch.Tensor             # [N] float32 (未击中=inf)
    face_ids: torch.Tensor      # [N] int32 (未击中=-1)
    hit_points: torch.Tensor    # [N, 3]
    normals: torch.Tensor       # [N, 3]
    bary_coords: torch.Tensor   # [N, 3]


@dataclass
class SegmentIntersectResult:
    """线段相交检测结果"""
    hit: torch.Tensor           # [N] bool
    hit_points: torch.Tensor    # [N, 3] or [total, 3]
    face_ids: torch.Tensor      # [N] or [total] int32
    bary_coords: torch.Tensor   # [N, 3] or [total, 3]
    segment_ids: Optional[torch.Tensor] = None  # [total] (if return_all=True)


@dataclass
class ClosestPointResult:
    """最近点查询结果（UDF）"""
    distances: torch.Tensor     # [N] float32 无符号距离
    face_ids: torch.Tensor      # [N] int32 最近面
    closest_points: torch.Tensor  # [N, 3]
    uvw: Optional[torch.Tensor] = None  # [N, 3] 重心坐标


@dataclass
class TriangleIntersectResult:
    """三角形-三角形相交结果"""
    edge_hit: torch.Tensor      # [num_edges] bool 每条边是否相交
    hit_points: torch.Tensor    # [num_hits, 3] 相交点坐标
    hit_face_ids: torch.Tensor  # [num_hits] int32 本网格被击中的面
    hit_edge_ids: torch.Tensor  # [num_hits] int32 另一网格击中的边


@dataclass
class VoxelFaceMapping:
    """体素-面映射（CSR稀疏格式）"""
    voxel_coords: torch.Tensor  # [K, 3] int32
    face_indices: torch.Tensor  # [total] int32
    face_start: torch.Tensor    # [K] int32
    face_count: torch.Tensor    # [K] int32
    
    def get_faces_for_voxel(self, voxel_idx: int) -> torch.Tensor:
        """获取指定体素相交的所有面"""
        start = self.face_start[voxel_idx].item()
        count = self.face_count[voxel_idx].item()
        return self.face_indices[start:start+count]


@dataclass
class VoxelPolygonMapping:
    """体素-多边形映射（精确相交区域）"""
    voxel_coords: torch.Tensor      # [K, 3] int32
    polygons: torch.Tensor          # [total, max_verts, 3] float32
    polygon_counts: torch.Tensor    # [total] int32 每个多边形的顶点数
    face_indices: torch.Tensor      # [total] int32
    voxel_ids: torch.Tensor         # [total] int32
    
    def get_polygon(self, idx: int) -> torch.Tensor:
        """获取指定索引的相交多边形"""
        count = self.polygon_counts[idx].item()
        return self.polygons[idx, :count]


@dataclass
class VisibilityResult:
    """可见性查询结果"""
    visibility: torch.Tensor        # [N] float32 可见性概率 [0, 1]
    visible_mask: Optional[torch.Tensor] = None  # [N, M] bool 每个点在每个视角的可见性
    hit_distances: Optional[torch.Tensor] = None  # [N, M] float32 遮挡距离
