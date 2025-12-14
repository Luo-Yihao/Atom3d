"""
VisibilityQuery: Visibility query application with statistical probability
"""

from typing import Union, Optional
import torch

from ..core.mesh_bvh import MeshBVH
from ..core.data_structures import VisibilityResult


class VisibilityQuery:
    """
    可见性查询应用
    
    = MeshBVH.intersect_ray + 统计概率
    
    对任意query point查询其可见性，用多视角射线的统计概率表达
    
    Args:
        bvh: MeshBVH实例
    """
    
    def __init__(self, bvh: MeshBVH):
        self.bvh = bvh
    
    def query(
        self,
        points: torch.Tensor,
        view_directions: torch.Tensor,
        return_details: bool = False
    ) -> Union[torch.Tensor, VisibilityResult]:
        """
        查询点的可见性（统计概率）
        
        对任意query point，从多个视角方向发射射线检测遮挡
        
        Args:
            points: [N, 3] 任意查询点
            view_directions: [M, 3] 多个视角方向（归一化）
            return_details: 是否返回详细信息
        
        Returns:
            如果return_details=False:
                visibility: [N] float32 可见性概率 [0, 1]
            如果return_details=True:
                result: VisibilityResult
        """
        N = points.shape[0]
        M = view_directions.shape[0]
        device = points.device
        
        visible_mask = torch.zeros(N, M, dtype=torch.bool, device=device)
        hit_distances = torch.zeros(N, M, device=device)
        
        for j in range(M):
            direction = view_directions[j]
            
            # All points use the same direction
            rays_o = points
            rays_d = direction.expand_as(points)
            
            result = self.bvh.intersect_ray(rays_o, rays_d)
            
            # Not hit = visible
            visible_mask[:, j] = ~result.hit
            hit_distances[:, j] = result.t
        
        # Compute visibility probability
        visibility = visible_mask.float().mean(dim=1)
        
        if return_details:
            return VisibilityResult(
                visibility=visibility,
                visible_mask=visible_mask,
                hit_distances=hit_distances
            )
        else:
            return visibility
    
    def query_from_cameras(
        self,
        points: torch.Tensor,
        camera_positions: torch.Tensor
    ) -> torch.Tensor:
        """
        从相机位置查询可见性
        
        Args:
            points: [N, 3] 查询点
            camera_positions: [M, 3] 相机位置
        
        Returns:
            visibility: [N] float32 可见性概率
                = 能看到该点的相机比例
        """
        N = points.shape[0]
        M = camera_positions.shape[0]
        device = points.device
        
        visible_count = torch.zeros(N, device=device)
        
        for cam_pos in camera_positions:
            # 从点向相机发射射线
            rays_o = points
            rays_d = cam_pos - points
            dists = rays_d.norm(dim=1, keepdim=True)
            rays_d = rays_d / (dists + 1e-8)
            
            # 检测遮挡
            result = self.bvh.intersect_ray(rays_o, rays_d, max_t=dists.squeeze().max().item())
            
            # 如果未击中或击中距离 >= 到相机距离，则可见
            visible = ~result.hit | (result.t >= dists.squeeze() - 1e-4)
            visible_count += visible.float()
        
        return visible_count / M
    
    def query_uniform_sphere(
        self,
        points: torch.Tensor,
        num_samples: int = 32,
        seed: Optional[int] = None
    ) -> torch.Tensor:
        """
        均匀球面采样查询可见性
        
        Args:
            points: [N, 3] 查询点
            num_samples: int 球面采样数量
            seed: 随机种子（用于可重复性）
        
        Returns:
            visibility: [N] float32 可见性概率
                = 均匀球面上未被遮挡的方向比例
        
        用途:
            - 无特定视角时的通用可见性度量
            - 可用于SDF符号判断（内部点可见性低）
        """
        device = points.device
        
        if seed is not None:
            torch.manual_seed(seed)
        
        # Generate uniform sphere directions using Fibonacci spiral
        directions = self._fibonacci_sphere(num_samples, device)
        
        return self.query(points, directions, return_details=False)
    
    def _fibonacci_sphere(self, n: int, device: str) -> torch.Tensor:
        """Generate n uniformly distributed points on sphere"""
        indices = torch.arange(n, dtype=torch.float32, device=device)
        
        phi = torch.acos(1 - 2 * (indices + 0.5) / n)
        theta = torch.pi * (1 + 5**0.5) * indices
        
        x = torch.sin(phi) * torch.cos(theta)
        y = torch.sin(phi) * torch.sin(theta)
        z = torch.cos(phi)
        
        return torch.stack([x, y, z], dim=1)
