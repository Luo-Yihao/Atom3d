"""
SDFQuery: Signed Distance Field query with multiple methods
"""

from typing import Optional
import torch

from ..core.mesh_bvh import MeshBVH
from ..grid.grid_indexer import GridIndexer
from .flood_fill import FloodFill


class SDFQuery:
    """
    SDF查询应用
    
    = MeshBVH.query_closest_point + 全局符号算法
    
    支持多种SDF计算方法:
    - winding: Winding number（精确，需要水密网格）
    - flood: Flood fill（开放网格）
    - raystab: Ray stabbing（鲁棒但慢）
    
    Args:
        bvh: MeshBVH实例
    """
    
    def __init__(self, bvh: MeshBVH):
        self.bvh = bvh
    
    def query_winding(self, points: torch.Tensor) -> torch.Tensor:
        """
        基于Winding Number的SDF
        
        Args:
            points: [N, 3]
        
        Returns:
            sdf: [N] float32 (内部为负，外部为正)
        
        适用: 水密或近水密网格
        """
        N = points.shape[0]
        device = points.device
        
        # Get UDF
        result = self.bvh.query_closest_point(points, return_uvw=False)
        distances = result.distances
        
        # Compute winding number
        winding = self._compute_winding_number(points)
        
        # Inside if winding > 0.5
        inside = winding > 0.5
        
        # SDF = distance * sign
        sdf = distances.clone()
        sdf[inside] = -sdf[inside]
        
        return sdf
    
    def _compute_winding_number(self, points: torch.Tensor) -> torch.Tensor:
        """
        Compute generalized winding number
        
        For each point, sum solid angles subtended by all triangles
        """
        N = points.shape[0]
        device = points.device
        
        # Get triangle vertices
        tri_verts = self.bvh.vertices[self.bvh.faces]  # [M, 3, 3]
        v0, v1, v2 = tri_verts[:, 0], tri_verts[:, 1], tri_verts[:, 2]
        
        winding = torch.zeros(N, device=device)
        
        for i in range(N):
            p = points[i]
            
            # Vectors from point to vertices
            a = v0 - p
            b = v1 - p
            c = v2 - p
            
            # Normalize
            la = a.norm(dim=1, keepdim=True) + 1e-8
            lb = b.norm(dim=1, keepdim=True) + 1e-8
            lc = c.norm(dim=1, keepdim=True) + 1e-8
            
            a = a / la
            b = b / lb
            c = c / lc
            
            # Solid angle formula
            det = (a * torch.cross(b, c)).sum(dim=1)
            denom = 1 + (a * b).sum(dim=1) + (b * c).sum(dim=1) + (c * a).sum(dim=1)
            
            solid_angle = 2 * torch.atan2(det, denom)
            winding[i] = solid_angle.sum() / (4 * torch.pi)
        
        return winding
    
    def query_flood(
        self,
        points: torch.Tensor,
        voxel_coords: torch.Tensor,
        grid: GridIndexer
    ) -> torch.Tensor:
        """
        基于Flood Fill的SDF
        
        Args:
            points: [N, 3]
            voxel_coords: [K, 3] 表面体素坐标
            grid: GridIndexer
        
        Returns:
            sdf: [N] float32
        
        适用: 开放网格
        """
        # Get UDF
        result = self.bvh.query_closest_point(points, return_uvw=False)
        distances = result.distances
        
        # Perform flood fill to get inside/outside labels
        labels = FloodFill.fill(voxel_coords, grid)
        
        # Convert points to grid coords and check labels
        grid_coords = grid.world_to_grid(points).floor().int()
        grid_coords = grid_coords.clamp(0, grid.resolution - 1)
        
        # Create lookup
        exterior_label = 0  # Connected to seed (assumed exterior)
        
        # For each point, check if it's inside or outside
        # Simplified: interpolate from voxel labels
        sdf = distances.clone()
        
        # TODO: Proper interpolation from flood fill labels
        # For now, use visibility-based estimation
        
        return sdf
    
    def query_raystab(
        self,
        points: torch.Tensor,
        num_rays: int = 8,
        seed: Optional[int] = None
    ) -> torch.Tensor:
        """
        基于Ray Stabbing的SDF
        
        从每个点发射多条射线，统计相交次数的奇偶性
        
        Args:
            points: [N, 3]
            num_rays: 每个点发射的射线数量
            seed: 随机种子
        
        Returns:
            sdf: [N] float32
        
        适用: 开放、非流形网格
        """
        N = points.shape[0]
        device = points.device
        
        if seed is not None:
            torch.manual_seed(seed)
        
        # Get UDF first
        result = self.bvh.query_closest_point(points, return_uvw=False)
        distances = result.distances
        
        # Generate random directions
        directions = torch.randn(num_rays, 3, device=device)
        directions = directions / directions.norm(dim=1, keepdim=True)
        
        # Count intersections for each point
        inside_votes = torch.zeros(N, device=device)
        
        for direction in directions:
            rays_o = points
            rays_d = direction.expand_as(points)
            
            # Count intersections (simplified: just check if hits)
            result_ray = self.bvh.intersect_ray(rays_o, rays_d)
            
            # If hit, it's either entering or leaving
            # Odd number of hits = inside
            inside_votes += result_ray.hit.float()
        
        # Majority vote: if more than half rays hit, likely inside
        # (This is a simplification; proper impl would count all intersections)
        inside = (inside_votes / num_rays) > 0.5
        
        sdf = distances.clone()
        sdf[inside] = -sdf[inside]
        
        return sdf
