"""
CubeGrid: Comprehensive cube topology indexing

Provides complete cube vertex/edge/face relationships:
- Cube corner (vertex) indices
- Cube edge indices with direction
- Cube face indices
- Edge-cube incidence
- Coordinate conversions
"""

from typing import Optional, Tuple
import torch


# ============================================================
# Cube Topology Constants
# ============================================================

# 8 corners of a unit cube, ordered by (x, y, z) bit pattern
# Corner i = (i&4, i&2, i&1) / 4 for each axis
CUBE_CORNERS = torch.tensor([
    [0, 0, 0],  # 0: origin
    [0, 0, 1],  # 1
    [0, 1, 0],  # 2
    [0, 1, 1],  # 3
    [1, 0, 0],  # 4
    [1, 0, 1],  # 5
    [1, 1, 0],  # 6
    [1, 1, 1],  # 7
], dtype=torch.int64)

# 12 edges defined by corner pairs
# First 4: z=0 face loop, next 4: z=1 face loop, last 4: vertical edges
CUBE_EDGES = torch.tensor([
    # z=0 face loop (0-2-6-4)
    [0, 2], [2, 6], [6, 4], [4, 0],
    # z=1 face loop (1-3-7-5)
    [1, 3], [3, 7], [7, 5], [5, 1],
    # vertical edges along z
    [0, 1], [2, 3], [6, 7], [4, 5],
], dtype=torch.int64)

# 6 faces defined by 4 corners each (CCW when viewed from outside)
CUBE_FACES = torch.tensor([
    [0, 2, 6, 4],  # -Z face (z=0)
    [1, 5, 7, 3],  # +Z face (z=1)
    [0, 1, 3, 2],  # -X face (x=0)
    [4, 6, 7, 5],  # +X face (x=1)
    [0, 4, 5, 1],  # -Y face (y=0)
    [2, 3, 7, 6],  # +Y face (y=1)
], dtype=torch.int64)

# Edge to face mapping: which 2 faces each edge belongs to
EDGE_TO_FACES = torch.tensor([
    # z=0 loop edges
    [0, 2], [0, 5], [0, 3], [0, 4],  # edges 0-3 belong to -Z face
    # z=1 loop edges
    [1, 2], [1, 5], [1, 3], [1, 4],  # edges 4-7 belong to +Z face
    # vertical edges
    [2, 4], [2, 5], [3, 5], [3, 4],  # edges 8-11
], dtype=torch.int64)

# Face normal directions
FACE_NORMALS = torch.tensor([
    [0, 0, -1],  # face 0: -Z
    [0, 0, 1],   # face 1: +Z
    [-1, 0, 0],  # face 2: -X
    [1, 0, 0],   # face 3: +X
    [0, -1, 0],  # face 4: -Y
    [0, 1, 0],   # face 5: +Y
], dtype=torch.int64)


class CubeGrid:
    """
    Cube grid topology indexer.
    
    Provides complete cube-vertex-edge-face indexing relationships.
    
    Conventions:
        - Coordinate range: [-1, 1] by default
        - Grid resolution: res (res cells per axis)
        - Vertex count: (res+1)^3
        - Cube count: res^3
        - Edge count: 3 * res * (res+1)^2
        - Face count: 3 * res^2 * (res+1)
    
    Args:
        resolution: Grid resolution per axis
        bounds: Optional [2, 3] custom bounds [[min_xyz], [max_xyz]]
        device: Compute device
    """
    
    def __init__(
        self,
        resolution: int,
        bounds: Optional[torch.Tensor] = None,
        device: str = 'cuda'
    ):
        self.res = resolution
        self.device = device
        self.dtype = torch.float32
        self.index_dtype = torch.int64
        
        # Vertex/cube counts
        self.num_vertices_per_axis = resolution + 1
        self.num_cubes = resolution ** 3
        self.num_vertices = self.num_vertices_per_axis ** 3
        
        # Bounds and cell size
        if bounds is not None:
            self.bounds = bounds.to(device)
        else:
            self.bounds = torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]], device=device)
        
        self.cell_size = (self.bounds[1] - self.bounds[0]) / resolution
        
        # 1D coordinates
        self.coords_1d = torch.linspace(
            self.bounds[0, 0].item(), 
            self.bounds[1, 0].item(),
            self.num_vertices_per_axis,
            device=device
        )
        
        # Move topology constants to device
        self.CUBE_CORNERS = CUBE_CORNERS.to(device)
        self.CUBE_EDGES = CUBE_EDGES.to(device)
        self.CUBE_FACES = CUBE_FACES.to(device)
        self.EDGE_TO_FACES = EDGE_TO_FACES.to(device)
        self.FACE_NORMALS = FACE_NORMALS.to(device)
        
        # Precompute edge indexing
        self._setup_edge_indexing()
        
        # Precompute face indexing
        self._setup_face_indexing()
    
    def _setup_edge_indexing(self):
        """Precompute edge index offsets."""
        # Edge direction (0=x, 1=y, 2=z)
        edge_dirs = self.CUBE_CORNERS[self.CUBE_EDGES[:, 1]] - self.CUBE_CORNERS[self.CUBE_EDGES[:, 0]]
        self._edge_axis = torch.argmax(edge_dirs.abs(), dim=1)  # [12]
        
        # Edge anchor (smaller endpoint)
        self._edge_anchor_local = torch.minimum(
            self.CUBE_CORNERS[self.CUBE_EDGES[:, 0]], 
            self.CUBE_CORNERS[self.CUBE_EDGES[:, 1]]
        )  # [12, 3]
        
        # Global edge counts (per axis)
        r = self.res
        p = self.num_vertices_per_axis
        
        self._num_edges_x = r * p * p       # x-direction: res x (res+1) x (res+1)
        self._num_edges_y = p * r * p       # y-direction
        self._num_edges_z = p * p * r       # z-direction
        
        self._edge_offset_x = 0
        self._edge_offset_y = self._num_edges_x
        self._edge_offset_z = self._num_edges_x + self._num_edges_y
        
        self.num_edges = self._num_edges_x + self._num_edges_y + self._num_edges_z
    
    def _setup_face_indexing(self):
        """Precompute face index offsets."""
        r = self.res
        p = self.num_vertices_per_axis
        
        # Global face counts (per normal axis)
        self._num_faces_x = p * r * r       # x-normal: (res+1) x res x res
        self._num_faces_y = r * p * r       # y-normal
        self._num_faces_z = r * r * p       # z-normal
        
        self._face_offset_x = 0
        self._face_offset_y = self._num_faces_x
        self._face_offset_z = self._num_faces_x + self._num_faces_y
        
        self.num_faces = self._num_faces_x + self._num_faces_y + self._num_faces_z
    
    # ============================================================
    # Coordinate Conversions
    # ============================================================
    
    def world_to_grid(self, world_coords: torch.Tensor) -> torch.Tensor:
        """World coordinates -> continuous grid coordinates."""
        return (world_coords - self.bounds[0]) / self.cell_size
    
    def grid_to_world(self, grid_coords: torch.Tensor) -> torch.Tensor:
        """Grid coordinates -> world coordinates (cell center)."""
        return (grid_coords.float() + 0.5) * self.cell_size + self.bounds[0]
    
    def vertex_to_world(self, vertex_ijk: torch.Tensor) -> torch.Tensor:
        """Vertex ijk coordinates -> world coordinates."""
        return vertex_ijk.float() * self.cell_size + self.bounds[0]
    
    # ============================================================
    # Index Utilities
    # ============================================================
    
    def ravel_ijk(self, ijk: torch.Tensor, dims: Tuple[int, int, int]) -> torch.Tensor:
        """ijk coordinates -> linear index."""
        nx, ny, nz = dims
        ijk = ijk.to(device=self.device, dtype=self.index_dtype)
        mul = torch.tensor([ny * nz, nz, 1], device=self.device, dtype=self.index_dtype)
        return (ijk * mul).sum(dim=-1)
    
    def unravel_idx(self, idx: torch.Tensor, dims: Tuple[int, int, int]) -> torch.Tensor:
        """Linear index -> ijk coordinates."""
        nx, ny, nz = dims
        idx = idx.to(device=self.device, dtype=self.index_dtype)
        i = torch.div(idx, ny * nz, rounding_mode='floor')
        rem = idx - i * (ny * nz)
        j = torch.div(rem, nz, rounding_mode='floor')
        k = rem - j * nz
        return torch.stack([i, j, k], dim=-1)
    
    # ============================================================
    # Cube Indexing
    # ============================================================
    
    def all_cube_indices(self) -> torch.Tensor:
        """Get all cube linear indices."""
        return torch.arange(self.num_cubes, device=self.device, dtype=self.index_dtype)
    
    def cube_to_ijk(self, cube_idx: torch.Tensor) -> torch.Tensor:
        """Cube linear index -> ijk coordinates."""
        return self.unravel_idx(cube_idx, (self.res, self.res, self.res))
    
    def ijk_to_cube(self, ijk: torch.Tensor) -> torch.Tensor:
        """ijk coordinates -> cube linear index."""
        return self.ravel_ijk(ijk, (self.res, self.res, self.res))
    
    def cube_aabb(self, cube_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get cube AABB bounds."""
        ijk = self.cube_to_ijk(cube_idx)
        mins = self.vertex_to_world(ijk)
        maxs = mins + self.cell_size
        return mins, maxs
    
    def cube_center(self, cube_idx: torch.Tensor) -> torch.Tensor:
        """Get cube center coordinates."""
        ijk = self.cube_to_ijk(cube_idx)
        return self.grid_to_world(ijk)
    
    # ============================================================
    # Cube -> Vertex (8 corners)
    # ============================================================
    
    def cube_corner_vertex_indices(self, cube_idx: torch.Tensor) -> torch.Tensor:
        """
        Get 8 corner vertex indices for each cube.
        
        Args:
            cube_idx: [B] cube indices
        
        Returns:
            vertex_indices: [B, 8] vertex indices
        """
        ijk = self.cube_to_ijk(cube_idx)  # [B, 3]
        corners_ijk = ijk[:, None, :] + self.CUBE_CORNERS[None, :, :]  # [B, 8, 3]
        
        p = self.num_vertices_per_axis
        vertex_indices = self.ravel_ijk(corners_ijk.view(-1, 3), (p, p, p))
        return vertex_indices.view(-1, 8)
    
    def cube_corner_coords(self, cube_idx: torch.Tensor) -> torch.Tensor:
        """
        Get 8 corner world coordinates for each cube.
        
        Args:
            cube_idx: [B]
        
        Returns:
            coords: [B, 8, 3]
        """
        ijk = self.cube_to_ijk(cube_idx)  # [B, 3]
        corners_ijk = ijk[:, None, :] + self.CUBE_CORNERS[None, :, :]  # [B, 8, 3]
        return self.vertex_to_world(corners_ijk)
    
    # ============================================================
    # Cube -> Edge (12 edges)
    # ============================================================
    
    def cube_edge_indices(self, cube_idx: torch.Tensor) -> torch.Tensor:
        """
        Get 12 global edge indices for each cube.
        
        Args:
            cube_idx: [B] cube indices
        
        Returns:
            edge_indices: [B, 12] edge indices
        """
        if cube_idx is None:
            cube_idx = self.all_cube_indices()
        
        cube_ijk = self.cube_to_ijk(cube_idx)  # [B, 3]
        B = cube_ijk.shape[0]
        
        # Edge anchor vertex (offset within cube)
        anchor_local = self._edge_anchor_local.to(self.index_dtype)  # [12, 3]
        anchor_ijk = cube_ijk[:, None, :] + anchor_local[None, :, :]  # [B, 12, 3]
        
        # Edge direction axis
        edge_axis = self._edge_axis  # [12]
        edge_axis_b = edge_axis.view(1, 12).expand(B, 12)  # [B, 12]
        
        E = torch.empty((B, 12), device=self.device, dtype=self.index_dtype)
        
        p = self.num_vertices_per_axis
        r = self.res
        
        # X-direction edges: dims = (res, pnum, pnum)
        mask_x = (edge_axis_b == 0)
        if mask_x.any():
            ijk_x = anchor_ijk[mask_x]  # [Nx, 3]
            local = ijk_x[:, 0] * (p * p) + ijk_x[:, 1] * p + ijk_x[:, 2]
            E[mask_x] = local + self._edge_offset_x
        
        # Y-direction edges: dims = (pnum, res, pnum)
        mask_y = (edge_axis_b == 1)
        if mask_y.any():
            ijk_y = anchor_ijk[mask_y]
            local = ijk_y[:, 0] * (r * p) + ijk_y[:, 1] * p + ijk_y[:, 2]
            E[mask_y] = local + self._edge_offset_y
        
        # Z-direction edges: dims = (pnum, pnum, res)
        mask_z = (edge_axis_b == 2)
        if mask_z.any():
            ijk_z = anchor_ijk[mask_z]
            local = ijk_z[:, 0] * (p * r) + ijk_z[:, 1] * r + ijk_z[:, 2]
            E[mask_z] = local + self._edge_offset_z
        
        return E
    
    def edge_endpoints(self, edge_idx: torch.Tensor) -> torch.Tensor:
        """
        Get world coordinates of edge endpoints.
        
        Args:
            edge_idx: [E] edge indices
        
        Returns:
            endpoints: [E, 2, 3] two endpoint coordinates
        """
        E = edge_idx.numel()
        coords = torch.empty(E, 2, 3, dtype=self.dtype, device=self.device)
        
        p = self.num_vertices_per_axis
        r = self.res
        
        # X-direction edges
        mask_x = (edge_idx >= self._edge_offset_x) & (edge_idx < self._edge_offset_y)
        if mask_x.any():
            local = edge_idx[mask_x] - self._edge_offset_x
            ijk = self.unravel_idx(local, (r, p, p))
            v0_ijk = ijk.clone()
            v1_ijk = ijk.clone()
            v1_ijk[:, 0] = v1_ijk[:, 0] + 1
            coords[mask_x, 0] = self.vertex_to_world(v0_ijk)
            coords[mask_x, 1] = self.vertex_to_world(v1_ijk)
        
        # Y-direction edges
        mask_y = (edge_idx >= self._edge_offset_y) & (edge_idx < self._edge_offset_z)
        if mask_y.any():
            local = edge_idx[mask_y] - self._edge_offset_y
            ijk = self.unravel_idx(local, (p, r, p))
            v0_ijk = ijk.clone()
            v1_ijk = ijk.clone()
            v1_ijk[:, 1] = v1_ijk[:, 1] + 1
            coords[mask_y, 0] = self.vertex_to_world(v0_ijk)
            coords[mask_y, 1] = self.vertex_to_world(v1_ijk)
        
        # Z-direction edges
        mask_z = edge_idx >= self._edge_offset_z
        if mask_z.any():
            local = edge_idx[mask_z] - self._edge_offset_z
            ijk = self.unravel_idx(local, (p, p, r))
            v0_ijk = ijk.clone()
            v1_ijk = ijk.clone()
            v1_ijk[:, 2] = v1_ijk[:, 2] + 1
            coords[mask_z, 0] = self.vertex_to_world(v0_ijk)
            coords[mask_z, 1] = self.vertex_to_world(v1_ijk)
        
        return coords
    
    # ============================================================
    # Cube -> Face (6 faces)
    # ============================================================
    
    def cube_face_indices(self, cube_idx: torch.Tensor) -> torch.Tensor:
        """
        Get 6 global face indices for each cube.
        
        Args:
            cube_idx: [B]
        
        Returns:
            face_indices: [B, 6] (order: -Z, +Z, -X, +X, -Y, +Y)
        """
        ijk = self.cube_to_ijk(cube_idx)  # [B, 3]
        cx, cy, cz = ijk[:, 0], ijk[:, 1], ijk[:, 2]
        
        p = self.num_vertices_per_axis
        r = self.res
        
        # Z faces: dims = (res, res, pnum)
        f_minus_z = self.ravel_ijk(torch.stack([cx, cy, cz], dim=-1), (r, r, p)) + self._face_offset_z
        f_plus_z = self.ravel_ijk(torch.stack([cx, cy, cz + 1], dim=-1), (r, r, p)) + self._face_offset_z
        
        # X faces: dims = (pnum, res, res)
        f_minus_x = self.ravel_ijk(torch.stack([cx, cy, cz], dim=-1), (p, r, r)) + self._face_offset_x
        f_plus_x = self.ravel_ijk(torch.stack([cx + 1, cy, cz], dim=-1), (p, r, r)) + self._face_offset_x
        
        # Y faces: dims = (res, pnum, res)
        f_minus_y = self.ravel_ijk(torch.stack([cx, cy, cz], dim=-1), (r, p, r)) + self._face_offset_y
        f_plus_y = self.ravel_ijk(torch.stack([cx, cy + 1, cz], dim=-1), (r, p, r)) + self._face_offset_y
        
        return torch.stack([f_minus_z, f_plus_z, f_minus_x, f_plus_x, f_minus_y, f_plus_y], dim=1)
    
    # ============================================================
    # Edge <-> Cube Incidence
    # ============================================================
    
    def edge_incident_cubes(self, edge_idx: torch.Tensor) -> torch.Tensor:
        """
        Get incident cubes for each edge.
        
        Args:
            edge_idx: [E]
        
        Returns:
            cube_indices: [E, 4] up to 4 incident cubes, -1 for boundary
        """
        E = edge_idx.numel()
        out = torch.full((E, 4), -1, dtype=self.index_dtype, device=self.device)
        
        r = self.res
        p = self.num_vertices_per_axis
        
        # X-direction edges
        mask_x = (edge_idx >= self._edge_offset_x) & (edge_idx < self._edge_offset_y)
        if mask_x.any():
            idx_x = torch.where(mask_x)[0]
            local = edge_idx[idx_x] - self._edge_offset_x
            ijk = self.unravel_idx(local, (r, p, p))  # (ix, jv, kv)
            
            # 4 incident cubes: (ix, jv-1|jv, kv-1|kv)
            for slot, (dy, dz) in enumerate([(0, 0), (0, -1), (-1, -1), (-1, 0)]):
                cy = ijk[:, 1] + dy
                cz = ijk[:, 2] + dz
                cx = ijk[:, 0]
                
                valid = (cx >= 0) & (cx < r) & (cy >= 0) & (cy < r) & (cz >= 0) & (cz < r)
                cube_ijk = torch.stack([cx, cy, cz], dim=-1)
                cube_idx_valid = self.ijk_to_cube(cube_ijk)
                
                out[idx_x[valid], slot] = cube_idx_valid[valid]
        
        # Y-direction edges
        mask_y = (edge_idx >= self._edge_offset_y) & (edge_idx < self._edge_offset_z)
        if mask_y.any():
            idx_y = torch.where(mask_y)[0]
            local = edge_idx[idx_y] - self._edge_offset_y
            ijk = self.unravel_idx(local, (p, r, p))  # (iv, jy, kv)
            
            for slot, (dx, dz) in enumerate([(0, 0), (0, -1), (-1, -1), (-1, 0)]):
                cx = ijk[:, 0] + dx
                cz = ijk[:, 2] + dz
                cy = ijk[:, 1]
                
                valid = (cx >= 0) & (cx < r) & (cy >= 0) & (cy < r) & (cz >= 0) & (cz < r)
                cube_ijk = torch.stack([cx, cy, cz], dim=-1)
                cube_idx_valid = self.ijk_to_cube(cube_ijk)
                
                out[idx_y[valid], slot] = cube_idx_valid[valid]
        
        # Z-direction edges
        mask_z = edge_idx >= self._edge_offset_z
        if mask_z.any():
            idx_z = torch.where(mask_z)[0]
            local = edge_idx[idx_z] - self._edge_offset_z
            ijk = self.unravel_idx(local, (p, p, r))  # (iv, jv, kz)
            
            for slot, (dx, dy) in enumerate([(0, 0), (0, -1), (-1, -1), (-1, 0)]):
                cx = ijk[:, 0] + dx
                cy = ijk[:, 1] + dy
                cz = ijk[:, 2]
                
                valid = (cx >= 0) & (cx < r) & (cy >= 0) & (cy < r) & (cz >= 0) & (cz < r)
                cube_ijk = torch.stack([cx, cy, cz], dim=-1)
                cube_idx_valid = self.ijk_to_cube(cube_ijk)
                
                out[idx_z[valid], slot] = cube_idx_valid[valid]
        
        return out
    
    # ============================================================
    # Cell Generation
    # ============================================================
    
    def generate_all_cells(self) -> torch.Tensor:
        """Generate all grid cell coordinates."""
        x = torch.arange(self.res, device=self.device)
        xx, yy, zz = torch.meshgrid(x, x, x, indexing='ij')
        return torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=1).to(self.index_dtype)
    
    def generate_candidate_cells_from_aabb(
        self,
        aabb_min: torch.Tensor,
        aabb_max: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate candidate cells from AABB bounds.
        
        Args:
            aabb_min: [M, 3]
            aabb_max: [M, 3]
        
        Returns:
            cells: [K, 3] deduplicated candidate cells
        """
        # Convert to grid coordinates
        grid_min = self.world_to_grid(aabb_min).floor().int().clamp(0, self.res - 1)
        grid_max = self.world_to_grid(aabb_max).ceil().int().clamp(0, self.res - 1)
        
        # Use global range (fast version)
        global_min = grid_min.min(dim=0)[0]
        global_max = grid_max.max(dim=0)[0]
        
        x = torch.arange(global_min[0], global_max[0] + 1, device=self.device)
        y = torch.arange(global_min[1], global_max[1] + 1, device=self.device)
        z = torch.arange(global_min[2], global_max[2] + 1, device=self.device)
        
        xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
        return torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=1).to(self.index_dtype)
