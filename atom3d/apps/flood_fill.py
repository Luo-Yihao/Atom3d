"""
FloodFill: Flood fill for voxel connectivity analysis

Used to distinguish interior/exterior voxels for solid voxelization.
"""

from typing import Optional
import torch

from ..grid.cube_grid import CubeGrid


class FloodFill:
    """
    Flood fill application for connectivity analysis.
    
    Uses CubeGrid for boundary checking and coordinate conversion.
    """
    
    @staticmethod
    def fill(
        voxel_coords: torch.Tensor,
        grid: CubeGrid,
        seed: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Connected component labeling via flood fill.
        
        Args:
            voxel_coords: [N, 3] int32 occupied voxel coordinates
            grid: CubeGrid (for boundary checking)
            seed: [3] seed point (default [0,0,0] as exterior)
        
        Returns:
            labels: [N] int32 connected component labels
                label=0 means connected to seed (exterior region)
                label>0 means interior connected components (may have multiple)
        """
        N = voxel_coords.shape[0]
        device = voxel_coords.device
        resolution = grid.res
        
        if seed is None:
            seed = torch.tensor([0, 0, 0], device=device)
        
        # Create 3D occupancy grid
        occupied = torch.zeros(resolution, resolution, resolution, dtype=torch.bool, device=device)
        
        # Mark occupied voxels
        valid_mask = (
            (voxel_coords[:, 0] >= 0) & (voxel_coords[:, 0] < resolution) &
            (voxel_coords[:, 1] >= 0) & (voxel_coords[:, 1] < resolution) &
            (voxel_coords[:, 2] >= 0) & (voxel_coords[:, 2] < resolution)
        )
        valid_coords = voxel_coords[valid_mask]
        occupied[valid_coords[:, 0], valid_coords[:, 1], valid_coords[:, 2]] = True
        
        # Initialize labels
        labels_3d = torch.full((resolution, resolution, resolution), -1, dtype=torch.int32, device=device)
        labels_3d[occupied] = 0  # Occupied voxels start with label 0
        
        # BFS from seed
        visited = torch.zeros_like(occupied)
        current_label = 0
        
        # 6-connectivity offsets
        offsets = torch.tensor([
            [1, 0, 0], [-1, 0, 0],
            [0, 1, 0], [0, -1, 0],
            [0, 0, 1], [0, 0, -1]
        ], device=device)
        
        # Start flood fill from seed (marking exterior)
        if not occupied[seed[0], seed[1], seed[2]]:
            queue = [seed.clone()]
            visited[seed[0], seed[1], seed[2]] = True
            labels_3d[seed[0], seed[1], seed[2]] = 0  # Exterior
            
            while queue:
                current = queue.pop(0)
                
                for offset in offsets:
                    neighbor = current + offset
                    
                    # Check bounds
                    if (neighbor >= 0).all() and (neighbor < resolution).all():
                        nx, ny, nz = neighbor[0].item(), neighbor[1].item(), neighbor[2].item()
                        
                        if not visited[nx, ny, nz] and not occupied[nx, ny, nz]:
                            visited[nx, ny, nz] = True
                            labels_3d[nx, ny, nz] = 0  # Exterior
                            queue.append(neighbor)
        
        # Extract labels for input coordinates
        labels = torch.zeros(N, dtype=torch.int32, device=device)
        labels[valid_mask] = labels_3d[valid_coords[:, 0], valid_coords[:, 1], valid_coords[:, 2]]
        
        return labels
    
    @staticmethod
    def get_interior_voxels(
        voxel_coords: torch.Tensor,
        grid: CubeGrid,
        seed: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get interior voxel coordinates.
        
        Args:
            voxel_coords: [N, 3] surface voxel coordinates
            grid: CubeGrid
            seed: [3] exterior seed point
        
        Returns:
            interior_coords: [K, 3] interior voxel coordinates 
                             (empty voxels not connected to exterior)
        """
        device = voxel_coords.device
        resolution = grid.res
        
        if seed is None:
            seed = torch.tensor([0, 0, 0], device=device)
        
        # Create occupancy grid from surface voxels
        occupied = torch.zeros(resolution, resolution, resolution, dtype=torch.bool, device=device)
        
        valid_mask = (
            (voxel_coords[:, 0] >= 0) & (voxel_coords[:, 0] < resolution) &
            (voxel_coords[:, 1] >= 0) & (voxel_coords[:, 1] < resolution) &
            (voxel_coords[:, 2] >= 0) & (voxel_coords[:, 2] < resolution)
        )
        valid_coords = voxel_coords[valid_mask]
        if valid_coords.shape[0] > 0:
            occupied[valid_coords[:, 0], valid_coords[:, 1], valid_coords[:, 2]] = True
        
        # Mark exterior via flood fill
        exterior = torch.zeros_like(occupied)
        
        offsets = torch.tensor([
            [1, 0, 0], [-1, 0, 0],
            [0, 1, 0], [0, -1, 0],
            [0, 0, 1], [0, 0, -1]
        ], device=device)
        
        if not occupied[seed[0], seed[1], seed[2]]:
            queue = [seed.clone()]
            exterior[seed[0], seed[1], seed[2]] = True
            
            while queue:
                current = queue.pop(0)
                
                for offset in offsets:
                    neighbor = current + offset
                    
                    if (neighbor >= 0).all() and (neighbor < resolution).all():
                        nx, ny, nz = neighbor[0].item(), neighbor[1].item(), neighbor[2].item()
                        
                        if not exterior[nx, ny, nz] and not occupied[nx, ny, nz]:
                            exterior[nx, ny, nz] = True
                            queue.append(neighbor)
        
        # Interior = not occupied and not exterior
        interior = ~occupied & ~exterior
        
        # Convert to coordinates
        interior_indices = torch.where(interior)
        interior_coords = torch.stack([
            interior_indices[0],
            interior_indices[1],
            interior_indices[2]
        ], dim=1).int()
        
        return interior_coords
