"""
MeshIntersector: Mesh-mesh collision detection application
"""

import torch

from ..core.mesh_bvh import MeshBVH
from ..core.data_structures import TriangleIntersectResult


class MeshIntersector:
    """
    网格碰撞检测应用
    
    = MeshBVH.intersect_triangles 的封装
    
    用于网格自相交检测、多网格碰撞检测
    
    Args:
        bvh: MeshBVH实例
    """
    
    def __init__(self, bvh: MeshBVH):
        self.bvh = bvh
    
    def check_self_intersection(
        self,
        skip_adjacent: bool = True
    ) -> TriangleIntersectResult:
        """
        检测网格自相交
        
        Args:
            skip_adjacent: 是否跳过相邻面（共享顶点）
        
        Returns:
            result: TriangleIntersectResult
        """
        # Use the mesh against itself
        result = self.bvh.intersect_triangles(
            self.bvh.vertices,
            self.bvh.faces
        )
        
        if skip_adjacent:
            # Filter out adjacent face collisions
            # (faces that share vertices)
            result = self._filter_adjacent(result)
        
        return result
    
    def _filter_adjacent(
        self,
        result: TriangleIntersectResult
    ) -> TriangleIntersectResult:
        """Filter out collisions from adjacent faces"""
        if result.hit_points.shape[0] == 0:
            return result
        
        # Get edge face indices
        edge_face_ids = result.hit_edge_ids // 3  # Each face has 3 edges
        hit_face_ids = result.hit_face_ids
        
        # Check if faces share vertices
        valid_mask = torch.ones(result.hit_points.shape[0], dtype=torch.bool, device=self.bvh.device)
        
        for i in range(result.hit_points.shape[0]):
            face1 = self.bvh.faces[hit_face_ids[i]]
            face2 = self.bvh.faces[edge_face_ids[i]]
            
            # Check for shared vertices
            shared = (face1.unsqueeze(1) == face2.unsqueeze(0)).any()
            if shared:
                valid_mask[i] = False
        
        return TriangleIntersectResult(
            edge_hit=result.edge_hit,  # Keep original
            hit_points=result.hit_points[valid_mask],
            hit_face_ids=result.hit_face_ids[valid_mask],
            hit_edge_ids=result.hit_edge_ids[valid_mask]
        )
    
    def intersect_with_mesh(
        self,
        other_vertices: torch.Tensor,
        other_faces: torch.Tensor
    ) -> TriangleIntersectResult:
        """
        与另一个网格碰撞检测
        
        Args:
            other_vertices: [M, 3]
            other_faces: [K, 3]
        
        Returns:
            result: TriangleIntersectResult
        """
        return self.bvh.intersect_triangles(other_vertices, other_faces)
