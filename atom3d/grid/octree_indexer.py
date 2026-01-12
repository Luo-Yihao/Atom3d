"""
OctreeIndexer: Multi-resolution octree grid with full cube topology

Inherits from CubeGrid to provide cube vertex/edge/face indexing
at multiple resolution levels.
"""

from typing import Optional, Tuple, Union, TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from atom3d.core import MeshBVH
from .cube_grid import CubeGrid
from ..core.device_utils import resolve_device


class OctreeIndexer(CubeGrid):
    """
    Multi-resolution octree indexer.
    
    Inherits all CubeGrid functionality and adds:
    - Morton encoding/decoding
    - Multi-level support (level 0 to max_level)
    - Level subdivision and merging
    - Level-aware coordinate conversion
    
    Args:
        max_level: Maximum octree level (resolution = 2^max_level)
        bounds: Optional [2, 3] grid bounds (default [-1, 1])
        device: Compute device. If None, auto-detects from bounds tensor.
                Priority: bounds.device > explicit device > 'cuda:0'
    """
    
    def __init__(
        self,
        max_level: int,
        bounds: Optional[torch.Tensor] = None,
        device: Optional[Union[str, torch.device]] = None
    ):
        self.max_level = max_level
        self._max_resolution = 2 ** max_level
        
        # Initialize parent CubeGrid at maximum resolution
        # Device resolution is handled by CubeGrid.__init__
        super().__init__(
            resolution=self._max_resolution,
            bounds=bounds,
            device=device
        )
        
        # Precompute per-level resolution and cell size
        self._level_resolutions = [2 ** level for level in range(max_level + 1)]
        self._level_cell_sizes = [
            (self.bounds[1] - self.bounds[0]) / res 
            for res in self._level_resolutions
        ]
    
    # ============================================================
    # Level-Aware Properties
    # ============================================================
    
    def get_resolution(self, level: int) -> int:
        """Get resolution at specified level."""
        return self._level_resolutions[level]
    
    def get_cell_size(self, level: int) -> torch.Tensor:
        """Get cell size at specified level."""
        return self._level_cell_sizes[level]
    
    def get_num_cubes(self, level: int) -> int:
        """Get cube count at specified level."""
        res = self.get_resolution(level)
        return res ** 3
    
    # ============================================================
    # Level-Aware Coordinate Conversion
    # ============================================================
    
    def world_to_grid_level(
        self, 
        world_coords: torch.Tensor,
        level: int
    ) -> torch.Tensor:
        """
        World coordinates -> grid coordinates at specified level.
        
        Args:
            world_coords: [N, 3]
            level: Octree level
        
        Returns:
            grid_coords: [N, 3] int64
        """
        cell_size = self.get_cell_size(level)
        return ((world_coords - self.bounds[0]) / cell_size).floor().to(self.index_dtype)
    
    def grid_to_world_level(
        self, 
        grid_coords: torch.Tensor,
        level: int
    ) -> torch.Tensor:
        """Grid coordinates -> world coordinates (cell center) at specified level."""
        cell_size = self.get_cell_size(level)
        return (grid_coords.float() + 0.5) * cell_size + self.bounds[0]
    
    def get_cell_aabb_level(
        self,
        grid_coords: torch.Tensor,
        level: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get cell AABB at specified level.
        
        Args:
            grid_coords: [N, 3] int
            level: Octree level
        
        Returns:
            aabb_min: [N, 3]
            aabb_max: [N, 3]
        """
        cell_size = self.get_cell_size(level)
        aabb_min = grid_coords.float() * cell_size + self.bounds[0]
        aabb_max = (grid_coords.float() + 1) * cell_size + self.bounds[0]
        return aabb_min, aabb_max
    
    # ============================================================
    # Morton Encoding (Z-order curve)
    # ============================================================
    
    def linear_to_morton(self, linear_idx: torch.Tensor, level: int) -> torch.Tensor:
        """
        Linear index -> Morton code.
        
        Args:
            linear_idx: [N] linear indices
            level: Octree level
        
        Returns:
            morton: [N] Morton codes
        """
        res = self.get_resolution(level)
        dims = (res, res, res)
        ijk = self.unravel_idx(linear_idx, dims)
        return self._ijk_to_morton(ijk)
    
    def morton_to_linear(self, morton: torch.Tensor, level: int) -> torch.Tensor:
        """
        Morton code -> linear index.
        
        Args:
            morton: [N] Morton codes
            level: Octree level
        
        Returns:
            linear_idx: [N] linear indices
        """
        ijk = self._morton_to_ijk(morton, level)
        res = self.get_resolution(level)
        return self.ravel_ijk(ijk, (res, res, res))
    
    def _ijk_to_morton(self, ijk: torch.Tensor) -> torch.Tensor:
        """ijk coordinates -> Morton code (bit interleaving)."""
        x, y, z = ijk[:, 0], ijk[:, 1], ijk[:, 2]
        
        morton = torch.zeros_like(x)
        for i in range(21):  # Support up to 2^21 resolution
            morton |= ((x >> i) & 1) << (3 * i)
            morton |= ((y >> i) & 1) << (3 * i + 1)
            morton |= ((z >> i) & 1) << (3 * i + 2)
        
        return morton
    
    def _morton_to_ijk(self, morton: torch.Tensor, level: int) -> torch.Tensor:
        """Morton code -> ijk coordinates (bit de-interleaving)."""
        x = torch.zeros_like(morton)
        y = torch.zeros_like(morton)
        z = torch.zeros_like(morton)
        
        for i in range(level + 1):
            x |= ((morton >> (3 * i)) & 1) << i
            y |= ((morton >> (3 * i + 1)) & 1) << i
            z |= ((morton >> (3 * i + 2)) & 1) << i
        
        return torch.stack([x, y, z], dim=-1)
    
    # ============================================================
    # Level Transitions
    # ============================================================
    
    def subdivide(
        self,
        parent_coords: torch.Tensor,
        parent_level: int
    ) -> torch.Tensor:
        """
        Subdivide parent cells into 8 children.
        
        Args:
            parent_coords: [N, 3] parent cell ijk coordinates
            parent_level: Parent level
        
        Returns:
            child_coords: [N*8, 3] child cell ijk coordinates (level = parent_level + 1)
        """
        # Child offsets (same as CUBE_CORNERS)
        offsets = self.CUBE_CORNERS  # [8, 3]
        
        # Scale parent coords to child level (×2)
        parent_scaled = parent_coords * 2  # [N, 3]
        
        # Generate all children
        children = parent_scaled[:, None, :] + offsets[None, :, :]  # [N, 8, 3]
        children = children.reshape(-1, 3)  # [N*8, 3]
        
        return children.to(self.index_dtype)
    
    def merge(
        self,
        child_coords: torch.Tensor,
        child_level: int
    ) -> torch.Tensor:
        """
        Merge child cells to parent (floor divide by 2).
        
        Args:
            child_coords: [N, 3] child cell ijk coordinates
            child_level: Child level
        
        Returns:
            parent_coords: [N, 3] parent cell ijk coordinates (level = child_level - 1)
        """
        return torch.div(child_coords, 2, rounding_mode='floor')
    
    def get_root_coords(self) -> torch.Tensor:
        """Get root cell coordinates (level 0)."""
        return torch.tensor([[0, 0, 0]], dtype=self.index_dtype, device=self.device)
    
    # ============================================================
    # Level-Aware Cube Topology
    # ============================================================
    
    def cube_corner_coords_level(
        self,
        cube_ijk: torch.Tensor,
        level: int
    ) -> torch.Tensor:
        """
        Get 8 corner world coordinates at specified level.
        
        Args:
            cube_ijk: [B, 3] cube ijk coordinates
            level: Octree level
        
        Returns:
            coords: [B, 8, 3]
        """
        cell_size = self.get_cell_size(level)
        corners_ijk = cube_ijk[:, None, :] + self.CUBE_CORNERS[None, :, :]  # [B, 8, 3]
        return corners_ijk.float() * cell_size + self.bounds[0]
    
    def cube_to_ijk_level(
        self,
        cube_idx: torch.Tensor,
        level: Optional[int] = None
    ) -> torch.Tensor:
        """
        Convert linear cube index to ijk coordinates at specified level.
        
        Args:
            cube_idx: [B] linear indices
            level: Octree level (default: max_level)
        
        Returns:
            ijk: [B, 3] ijk coordinates
        """
        if level is None:
            level = self.max_level
        res = self.get_resolution(level)
        return self.unravel_idx(cube_idx, (res, res, res))
    
    def ijk_to_cube_level(
        self,
        ijk: torch.Tensor,
        level: Optional[int] = None
    ) -> torch.Tensor:
        """
        Convert ijk coordinates to linear cube index at specified level.
        
        Args:
            ijk: [B, 3] ijk coordinates
            level: Octree level (default: max_level)
        
        Returns:
            cube_idx: [B] linear indices
        """
        if level is None:
            level = self.max_level
        res = self.get_resolution(level)
        return self.ravel_ijk(ijk, (res, res, res))
    
    def cube_aabb_level(
        self,
        cubes: torch.Tensor,
        level: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get cube AABB at specified level.
        
        Automatically detects input format (ijk or linear index) and handles
        level-appropriate conversion. When level=None (default), behaves
        identically to cube_aabb().
        
        Args:
            cubes: [B, 3] ijk coordinates OR [B] linear indices
            level: Octree level (default: max_level)
        
        Returns:
            aabb_min, aabb_max: [B, 3]
        """
        if level is None:
            level = self.max_level
        
        # Auto-detect format and convert appropriately for the level
        if cubes.dim() == 2 and cubes.shape[1] == 3:
            # Already ijk coordinates
            cube_ijk = cubes
        elif cubes.dim() == 1:
            # Linear index - use level-aware conversion
            cube_ijk = self.cube_to_ijk_level(cubes, level)
        else:
            raise ValueError(f"cubes must be [B, 3] ijk or [B] linear idx, got shape {cubes.shape}")
        
        cell_size = self.get_cell_size(level)
        aabb_min = cube_ijk.float() * cell_size + self.bounds[0]
        aabb_max = aabb_min + cell_size
        return aabb_min, aabb_max
    
    def cube_edge_endpoints_level(
        self,
        cube_ijk: torch.Tensor,
        level: int
    ) -> torch.Tensor:
        """
        Get 12 edge endpoint coordinates at specified level.
        
        Args:
            cube_ijk: [B, 3]
            level: int
        
        Returns:
            endpoints: [B, 12, 2, 3] two endpoints per edge
        """
        cell_size = self.get_cell_size(level)
        
        # Get 8 corner coordinates
        corners = cube_ijk[:, None, :] + self.CUBE_CORNERS[None, :, :]  # [B, 8, 3]
        corners_world = corners.float() * cell_size + self.bounds[0]  # [B, 8, 3]
        
        # Index edge endpoints using CUBE_EDGES
        v0 = corners_world[:, self.CUBE_EDGES[:, 0]]  # [B, 12, 3]
        v1 = corners_world[:, self.CUBE_EDGES[:, 1]]  # [B, 12, 3]
        
        return torch.stack([v0, v1], dim=2)  # [B, 12, 2, 3]
    
    
    # ============================================================
    # Node Expansion
    # ============================================================
    
    def expand_nodes(
        self,
        nodes: torch.Tensor
    ) -> torch.Tensor:
        """
        Expand octree nodes to voxel coordinates at max_level.
        
        Args:
            nodes: [N, 2] int32 - (level, node_idx) pairs where
                   node_idx is linearized index at that level
        
        Returns:
            coords: [M, 3] int32 - Voxel coords at max_level
                    M can be > N if nodes are at coarse levels
        
        Example:
            Node at level=6 represents 2^(9-6) = 8³ = 512 voxels at level=9
        """
        if nodes.numel() == 0:
            return torch.empty(0, 3, dtype=self.index_dtype, device=self.device)
        
        all_coords = []
        
        for i in range(nodes.shape[0]):
            level = nodes[i, 0].item()
            node_idx = nodes[i, 1].item()
            
            # Convert node index to ijk at node level
            res_at_level = self.get_resolution(level)
            node_ijk = self.unravel_idx(
                nodes[i:i+1, 1], 
                (res_at_level, res_at_level, res_at_level)
            )  # [1, 3]
            
            # How many levels to expand
            expand_levels = self.max_level - level
            
            if expand_levels == 0:
                # Already at max level
                all_coords.append(node_ijk)
            else:
                # Recursively subdivide to max level
                current_coords = node_ijk
                for _ in range(expand_levels):
                    current_coords = self.subdivide(current_coords, level)
                    level += 1
                
                all_coords.append(current_coords)
        
        return torch.cat(all_coords, dim=0) if all_coords else torch.empty(0, 3, dtype=self.index_dtype, device=self.device)
    
    # ============================================================
    # Octree Traversal
    # ============================================================
    
    def all_cubes_at_level(self, level: int) -> torch.Tensor:
        """
        Get all cube ijk coordinates at specified level.
        
        Args:
            level: int
        
        Returns:
            ijk: [res^3, 3]
        """
        res = self.get_resolution(level)
        x = torch.arange(res, device=self.device, dtype=self.index_dtype)
        xx, yy, zz = torch.meshgrid(x, x, x, indexing='ij')
        return torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=1)
    
    def filter_active_cubes(
        self,
        cube_ijk: torch.Tensor,
        level: int,
        mesh_aabb_min: torch.Tensor,
        mesh_aabb_max: torch.Tensor
    ) -> torch.Tensor:
        """
        Filter cubes that intersect mesh AABBs.
        
        Args:
            cube_ijk: [N, 3] candidate cubes
            level: Octree level
            mesh_aabb_min, mesh_aabb_max: [M, 3] mesh triangle AABBs
        
        Returns:
            active_cube_ijk: [K, 3] active cubes
        """
        # Get cube AABBs
        cube_min, cube_max = self.cube_aabb_level(cube_ijk, level)  # [N, 3]
        
        # Broadphase: check cube overlap with any mesh AABB
        # [N, 1, 3] vs [1, M, 3]
        overlap = (cube_min[:, None, :] <= mesh_aabb_max[None, :, :]) & \
                  (cube_max[:, None, :] >= mesh_aabb_min[None, :, :])
        overlap_all = overlap.all(dim=2).any(dim=1)  # [N]
        
        return cube_ijk[overlap_all]
    
    def octree_traverse(
        self,
        bvh: "MeshBVH",
        min_level: int = 2
    ) -> torch.Tensor:
        """
        Octree traversal: coarse-to-fine active node discovery using BVH acceleration.
        
        Uses BVH's intersect_aabb for O(N log M) broadphase instead of 
        brute-force O(N × M) comparison.
        
        Args:
            bvh: MeshBVH instance for triangle-AABB intersection
            min_level: Starting level for traversal
        
        Returns:
            active_cubes: [K, 3] active cube ijk at max level
        """
        # Start from min_level
        current_cubes = self.all_cubes_at_level(min_level)
        
        # Initial filter at min_level
        cube_min, cube_max = self.cube_aabb_level(current_cubes, min_level)
        result = bvh.intersect_aabb(cube_min, cube_max, mode=0)
        current_cubes = current_cubes[result.hit]
        
        # Refine level by level
        for level in range(min_level + 1, self.max_level + 1):
            if current_cubes.numel() == 0:
                break
            
            # Subdivide
            current_cubes = self.subdivide(current_cubes, level - 1)
            
            # Filter using BVH broadphase (mode=0 = hit mask only)
            cube_min, cube_max = self.cube_aabb_level(current_cubes, level)
            result = bvh.intersect_aabb(cube_min, cube_max, mode=0)
            current_cubes = current_cubes[result.hit]
        
        return current_cubes
    
    # ============================================================
    # Flood Fill Helpers
    # ============================================================
    
    def mark_active_nodes(
        self,
        coords: torch.Tensor
    ) -> dict:
        """
        Build active nodes dictionary from max_level surface coordinates.
        
        Args:
            coords: [N, 3] int32 tensor, coordinates at max_level
        
        Returns:
            dict: {(level, node_idx): is_occupied}
        """
        device = coords.device
        max_level = self.max_level
        active_nodes = {}
        
        # ensure coords is on CPU for fast dict operations
        coords_cpu = coords.cpu()
        current_coords = coords_cpu.numpy() if hasattr(coords_cpu, 'numpy') else coords_cpu
        resolution = 2 ** max_level
        
        # 1. Dictionary at max_level
        current_indices = set()
        for coord in current_coords:
            idx = int(coord[0] * resolution * resolution + coord[1] * resolution + coord[2])
            active_nodes[(max_level, idx)] = True
            current_indices.add(idx)
            
        # 2. Aggregate upward
        for level in range(max_level, 0, -1):
            parent_res = 2 ** (level - 1)
            res = 2 ** level
            next_indices = set()
            
            for idx in current_indices:
                x = idx // (res * res)
                rem = idx % (res * res)
                y = rem // res
                z = rem % res
                
                px = x // 2
                py = y // 2
                pz = z // 2
                
                p_idx = px * parent_res * parent_res + py * parent_res + pz
                
                if (level - 1, p_idx) not in active_nodes:
                    active_nodes[(level - 1, p_idx)] = False
                    next_indices.add(p_idx)
            
            current_indices = next_indices
            
        return active_nodes

    def get_node_neighbors(
        self,
        level: int,
        node_idx: int,
        connectivity: int = 6
    ) -> list:
        """
        Get neighbors of a node for flood fill.
        
        Args:
            level: Node level
            node_idx: Node linear index
            connectivity: 6 (face) or 26 (full)
        
        Returns:
            List of (neighbor_level, neighbor_idx) tuples - all at SAME level
        """
        res = 2 ** level
        
        # Decode node index to coords
        x = node_idx // (res * res)
        rem = node_idx % (res * res)
        y = rem // res
        z = rem % res
        
        neighbors = []
        
        # Define offsets
        if connectivity == 6:
            offsets = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
        else:
            offsets = []
            for dz in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dx == 0 and dy == 0 and dz == 0:
                            continue
                        offsets.append((dx, dy, dz))
        
        # Get same-level neighbors
        for dx, dy, dz in offsets:
            nx, ny, nz = x + dx, y + dy, z + dz
            
            # Bounds check
            if 0 <= nx < res and 0 <= ny < res and 0 <= nz < res:
                neighbor_idx = nx * res * res + ny * res + nz
                neighbors.append((level, neighbor_idx))
        
        return neighbors
