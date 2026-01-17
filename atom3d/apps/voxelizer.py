"""
Voxelizer: Mesh voxelization application

Combines CubeGrid for spatial indexing with MeshBVH for collision detection.
"""

import torch
from typing import Union

from ..core.mesh_bvh import MeshBVH
from ..core.data_structures import VoxelFaceMapping, VoxelPolygonMapping
from ..grid.cube_grid import CubeGrid


class Voxelizer:
    """
    Mesh voxelization application.
    
    Combines:
        - CubeGrid: Generate grid cell AABBs
        - MeshBVH: Detect AABB-mesh collisions
    
    Args:
        bvh: MeshBVH instance
        grid: CubeGrid instance
    """
    
    def __init__(
        self,
        bvh: MeshBVH,
        grid: CubeGrid
    ):
        self.bvh = bvh
        self.grid = grid
    
    def voxelize_surface(
        self,
        strategy: str = 'candidate'
    ) -> torch.Tensor:
        """
        Surface voxelization.
        
        Workflow:
            1. Generate candidate cells (or all cells)
            2. Get cell AABBs via CubeGrid.cube_aabb()
            3. Detect collisions via MeshBVH.intersect_aabb()
        
        Args:
            strategy: 'all' test all cells, 'candidate' only test candidate cells
        
        Returns:
            voxel_coords: [K, 3] int32 intersecting voxel coordinates
        """
        if strategy == 'all':
            # Test all cells (slow but complete)
            all_coords = self.grid.generate_all_cells()
            cube_indices = self.grid.ijk_to_cube(all_coords)
            aabb_min, aabb_max = self.grid.cube_aabb(cube_indices)
            result = self.bvh.intersect_aabb(aabb_min, aabb_max, mode=0)
            return all_coords[result.hit]
        
        else:  # 'candidate'
            # Use candidate cells (fast)
            face_aabb_min, face_aabb_max = self.bvh.get_face_aabb()
            candidates = self.grid.generate_candidate_cells_from_aabb(face_aabb_min, face_aabb_max)
            
            if candidates.shape[0] == 0:
                return torch.empty(0, 3, dtype=torch.int32, device=self.bvh.device)
            
            cube_indices = self.grid.ijk_to_cube(candidates)
            aabb_min, aabb_max = self.grid.cube_aabb(cube_indices)
            result = self.bvh.intersect_aabb(aabb_min, aabb_max, mode=0)
            return candidates[result.hit]
    
    def voxelize_with_faces(self) -> VoxelFaceMapping:
        """
        Voxelization with voxel-face mapping.
        
        Workflow:
            1. Generate candidate cells
            2. MeshBVH.intersect_aabb(return_pairs=True)
            3. Build CSR format mapping
        
        Returns:
            mapping: VoxelFaceMapping
        """
        # Get candidates
        face_aabb_min, face_aabb_max = self.bvh.get_face_aabb()
        candidates = self.grid.generate_candidate_cells_from_aabb(face_aabb_min, face_aabb_max)
        
        if candidates.shape[0] == 0:
            device = self.bvh.device
            return VoxelFaceMapping(
                voxel_coords=torch.empty(0, 3, dtype=torch.int32, device=device),
                face_indices=torch.empty(0, dtype=torch.int32, device=device),
                face_start=torch.empty(0, dtype=torch.int32, device=device),
                face_count=torch.empty(0, dtype=torch.int32, device=device)
            )
        
        # Get AABB
        cube_indices = self.grid.ijk_to_cube(candidates)
        aabb_min, aabb_max = self.grid.cube_aabb(cube_indices)
        
        # Intersect with pairs (mode=1 returns aabb_ids and face_ids)
        result = self.bvh.intersect_aabb(aabb_min, aabb_max, mode=1)
        
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
        Voxelization with exact intersection polygons.
        
        Workflow:
            1. Run voxelize_with_faces first
            2. For each voxel-face pair, compute clipped polygon
        
        Args:
            max_polygon_verts: Maximum polygon vertex count
        
        Returns:
            mapping: VoxelPolygonMapping
        
        Note:
            Current implementation is simplified. Full implementation
            requires Sutherland-Hodgman clipping algorithm.
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
            cube_idx = self.grid.ijk_to_cube(voxel_coord.unsqueeze(0))
            aabb_min, aabb_max = self.grid.cube_aabb(cube_idx)
            
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
