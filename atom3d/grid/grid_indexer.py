"""
GridIndexer: Uniform grid indexer for cuMTV
"""

from typing import Tuple, Optional
import torch


class GridIndexer:
    """
    网格索引器
    
    职责:
        - 管理空间网格（resolution、bounds）
        - 坐标转换（world ↔ grid）
        - 生成grid AABB（供碰撞检测使用）
    
    不包含:
        - 任何碰撞检测逻辑
    
    Args:
        resolution: int 网格分辨率
        bounds: [2, 3] 网格边界 [[min_xyz], [max_xyz]]
        device: 运行设备
    """
    
    def __init__(
        self,
        resolution: int,
        bounds: torch.Tensor,
        device: str = 'cuda'
    ):
        self.resolution = resolution
        self.bounds = bounds.to(device)
        self.device = device
        self.cell_size = (self.bounds[1] - self.bounds[0]) / resolution
    
    def world_to_grid(self, world_coords: torch.Tensor) -> torch.Tensor:
        """
        世界坐标 → 网格坐标（连续）
        
        Args:
            world_coords: [N, 3]
        
        Returns:
            grid_coords: [N, 3] float32
        """
        return (world_coords - self.bounds[0]) / self.cell_size
    
    def grid_to_world(self, grid_coords: torch.Tensor) -> torch.Tensor:
        """
        网格坐标 → 世界坐标（cell中心）
        
        Args:
            grid_coords: [N, 3] (可以是int或float)
        
        Returns:
            world_coords: [N, 3] float32
        """
        return (grid_coords.float() + 0.5) * self.cell_size + self.bounds[0]
    
    def get_cell_aabb(
        self, 
        grid_coords: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取网格单元的AABB
        
        这是给碰撞原语使用的输入
        
        Args:
            grid_coords: [N, 3] int32 网格坐标
        
        Returns:
            aabb_min: [N, 3] float32
            aabb_max: [N, 3] float32
        """
        aabb_min = grid_coords.float() * self.cell_size + self.bounds[0]
        aabb_max = (grid_coords.float() + 1) * self.cell_size + self.bounds[0]
        return aabb_min, aabb_max
    
    def generate_all_cells(self) -> torch.Tensor:
        """
        生成所有网格坐标
        
        Returns:
            coords: [resolution^3, 3] int32
        """
        r = self.resolution
        x = torch.arange(r, device=self.device)
        y = torch.arange(r, device=self.device)
        z = torch.arange(r, device=self.device)
        
        # Create meshgrid
        xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
        coords = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=1)
        return coords.int()
    
    def generate_candidate_cells(
        self,
        face_aabb_min: torch.Tensor,
        face_aabb_max: torch.Tensor
    ) -> torch.Tensor:
        """
        根据三角形AABB生成候选网格单元
        
        Args:
            face_aabb_min: [M, 3] 三角形AABB最小坐标
            face_aabb_max: [M, 3] 三角形AABB最大坐标
        
        Returns:
            candidate_cells: [K, 3] int32 可能相交的网格坐标（去重）
        """
        # Convert to grid coordinates
        grid_min = self.world_to_grid(face_aabb_min).floor().int()
        grid_max = self.world_to_grid(face_aabb_max).ceil().int()
        
        # Clamp to valid range
        grid_min = grid_min.clamp(0, self.resolution - 1)
        grid_max = grid_max.clamp(0, self.resolution - 1)
        
        # Generate all cells for each face
        all_cells = []
        for i in range(face_aabb_min.shape[0]):
            xmin, ymin, zmin = grid_min[i].tolist()
            xmax, ymax, zmax = grid_max[i].tolist()
            
            for x in range(xmin, xmax + 1):
                for y in range(ymin, ymax + 1):
                    for z in range(zmin, zmax + 1):
                        all_cells.append([x, y, z])
        
        if len(all_cells) == 0:
            return torch.empty(0, 3, dtype=torch.int32, device=self.device)
        
        cells = torch.tensor(all_cells, dtype=torch.int32, device=self.device)
        
        # Remove duplicates
        cells_unique = torch.unique(cells, dim=0)
        return cells_unique
    
    def generate_candidate_cells_fast(
        self,
        face_aabb_min: torch.Tensor,
        face_aabb_max: torch.Tensor
    ) -> torch.Tensor:
        """
        快速生成候选网格单元（使用全局范围）
        
        Args:
            face_aabb_min: [M, 3]
            face_aabb_max: [M, 3]
        
        Returns:
            candidate_cells: [K, 3] int32
        """
        # Get global bounds
        global_min = face_aabb_min.min(dim=0)[0]
        global_max = face_aabb_max.max(dim=0)[0]
        
        # Convert to grid coordinates
        grid_min = self.world_to_grid(global_min).floor().int().clamp(0, self.resolution - 1)
        grid_max = self.world_to_grid(global_max).ceil().int().clamp(0, self.resolution - 1)
        
        # Generate cells in range
        x = torch.arange(grid_min[0], grid_max[0] + 1, device=self.device)
        y = torch.arange(grid_min[1], grid_max[1] + 1, device=self.device)
        z = torch.arange(grid_min[2], grid_max[2] + 1, device=self.device)
        
        xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
        coords = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=1)
        return coords.int()
