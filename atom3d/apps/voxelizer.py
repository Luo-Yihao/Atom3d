"""
Voxelizer: Voxelization application
"""

import torch
from typing import Union

from ..core.mesh_bvh import MeshBVH
from ..core.data_structures import VoxelFaceMapping, VoxelPolygonMapping
from ..grid.grid_indexer import GridIndexer


class Voxelizer:
    """
    体素化应用
    
    = GridIndexer生成的AABB + MeshBVH.intersect_aabb
    
    这是一个特化实例，组合了：
    - GridIndexer: 生成grid cell AABBs
    - MeshBVH: 检测AABB-mesh碰撞
    
    Args:
        bvh: MeshBVH实例
        grid: GridIndexer实例
    """
    
    def __init__(
        self,
        bvh: MeshBVH,
        grid: GridIndexer
    ):
        self.bvh = bvh
        self.grid = grid
    
    def voxelize_surface(
        self,
        strategy: str = 'candidate'
    ) -> torch.Tensor:
        """
        表面体素化（Level 1）
        
        流程:
            1. GridIndexer生成候选cells（或所有cells）
            2. GridIndexer.get_cell_aabb() 获取AABB
            3. MeshBVH.intersect_aabb() 检测碰撞
        
        Args:
            strategy: 'all' 测试所有cells，'candidate' 仅测试候选cells
        
        Returns:
            voxel_coords: [K, 3] int32 相交的体素坐标
        """
        if strategy == 'all':
            # 测试所有cells（慢但完整）
            all_coords = self.grid.generate_all_cells()
            aabb_min, aabb_max = self.grid.get_cell_aabb(all_coords)
            result = self.bvh.intersect_aabb(aabb_min, aabb_max, return_pairs=False)
            return all_coords[result.hit]
        
        else:  # 'candidate'
            # 使用候选cells（快速）
            face_aabb_min, face_aabb_max = self.bvh.get_face_aabb()
            candidates = self.grid.generate_candidate_cells_fast(face_aabb_min, face_aabb_max)
            
            if candidates.shape[0] == 0:
                return torch.empty(0, 3, dtype=torch.int32, device=self.bvh.device)
            
            aabb_min, aabb_max = self.grid.get_cell_aabb(candidates)
            result = self.bvh.intersect_aabb(aabb_min, aabb_max, return_pairs=False)
            return candidates[result.hit]
    
    def voxelize_with_faces(self) -> VoxelFaceMapping:
        """
        体素化 + 体素-面映射（Level 2）
        
        流程:
            1. GridIndexer生成候选cells
            2. MeshBVH.intersect_aabb(return_pairs=True)
            3. 构建CSR格式映射
        
        Returns:
            mapping: VoxelFaceMapping
        """
        # Get candidates
        face_aabb_min, face_aabb_max = self.bvh.get_face_aabb()
        candidates = self.grid.generate_candidate_cells_fast(face_aabb_min, face_aabb_max)
        
        if candidates.shape[0] == 0:
            device = self.bvh.device
            return VoxelFaceMapping(
                voxel_coords=torch.empty(0, 3, dtype=torch.int32, device=device),
                face_indices=torch.empty(0, dtype=torch.int32, device=device),
                face_start=torch.empty(0, dtype=torch.int32, device=device),
                face_count=torch.empty(0, dtype=torch.int32, device=device)
            )
        
        # Get AABB
        aabb_min, aabb_max = self.grid.get_cell_aabb(candidates)
        
        # Intersect with pairs
        result = self.bvh.intersect_aabb(aabb_min, aabb_max, return_pairs=True)
        
        # Build CSR format
        # result.aabb_ids: which candidate cell
        # result.face_ids: which face
        
        # Get unique voxels that have hits
        unique_aabb_ids, inverse = torch.unique(result.aabb_ids, return_inverse=True)
        voxel_coords = candidates[unique_aabb_ids]
        
        # Count faces per voxel
        num_voxels = unique_aabb_ids.shape[0]
        face_count = torch.bincount(inverse, minlength=num_voxels)
        
        # Compute start indices
        face_start = torch.cumsum(face_count, dim=0) - face_count
        
        # Sort face_ids by voxel
        sorted_indices = torch.argsort(inverse)
        face_indices = result.face_ids[sorted_indices]
        
        return VoxelFaceMapping(
            voxel_coords=voxel_coords,
            face_indices=face_indices.int(),
            face_start=face_start.int(),
            face_count=face_count.int()
        )
    
    def voxelize_with_polygons(
        self,
        max_polygon_verts: int = 8
    ) -> VoxelPolygonMapping:
        """
        体素化 + 精确相交多边形（Level 3）
        
        流程:
            1. 先执行voxelize_with_faces
            2. 对每个体素-面对，计算裁剪多边形
        
        Args:
            max_polygon_verts: 最大多边形顶点数
        
        Returns:
            mapping: VoxelPolygonMapping
        
        Note:
            当前使用简化实现，完整实现需要Sutherland-Hodgman算法
        """
        # Get voxel-face mapping first
        vf_mapping = self.voxelize_with_faces()
        
        device = self.bvh.device
        total_pairs = vf_mapping.face_indices.shape[0]
        
        if total_pairs == 0:
            return VoxelPolygonMapping(
                voxel_coords=vf_mapping.voxel_coords,
                polygons=torch.empty(0, max_polygon_verts, 3, device=device),
                polygon_counts=torch.empty(0, dtype=torch.int32, device=device),
                face_indices=torch.empty(0, dtype=torch.int32, device=device),
                voxel_ids=torch.empty(0, dtype=torch.int32, device=device)
            )
        
        # For each voxel-face pair, compute intersection polygon
        # Simplified: just return triangle vertices clipped to AABB
        
        polygons = torch.zeros(total_pairs, max_polygon_verts, 3, device=device)
        polygon_counts = torch.zeros(total_pairs, dtype=torch.int32, device=device)
        voxel_ids = torch.zeros(total_pairs, dtype=torch.int32, device=device)
        
        idx = 0
        for v_idx in range(vf_mapping.voxel_coords.shape[0]):
            start = vf_mapping.face_start[v_idx].item()
            count = vf_mapping.face_count[v_idx].item()
            
            voxel_coord = vf_mapping.voxel_coords[v_idx]
            aabb_min, aabb_max = self.grid.get_cell_aabb(voxel_coord.unsqueeze(0))
            
            for f_offset in range(count):
                face_id = vf_mapping.face_indices[start + f_offset]
                
                # Get triangle vertices
                tri_verts = self.bvh.vertices[self.bvh.faces[face_id]]  # [3, 3]
                
                # Simplified: store triangle vertices (proper impl needs clipping)
                polygons[idx, :3] = tri_verts
                polygon_counts[idx] = 3
                voxel_ids[idx] = v_idx
                
                idx += 1
        
        return VoxelPolygonMapping(
            voxel_coords=vf_mapping.voxel_coords,
            polygons=polygons,
            polygon_counts=polygon_counts,
            face_indices=vf_mapping.face_indices,
            voxel_ids=voxel_ids
        )
