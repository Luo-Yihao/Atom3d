"""
CubeGrid: Comprehensive cube topology indexing

Provides complete cube vertex/edge/face relationships:
- Cube corner (vertex) indices
- Cube edge indices with direction
- Cube face indices
- Edge-cube incidence (CCW loop ordered by right-hand rule w.r.t. edge_endpoints direction)
- Coordinate conversions
"""

from typing import Optional, Tuple
import torch


# ============================================================
# Cube Topology Constants
# ============================================================

# 8 corners of a unit cube, ordered by (x, y, z) bit pattern
CUBE_CORNERS = torch.tensor(
    [
        [0, 0, 0],  # 0
        [0, 0, 1],  # 1
        [0, 1, 0],  # 2
        [0, 1, 1],  # 3
        [1, 0, 0],  # 4
        [1, 0, 1],  # 5
        [1, 1, 0],  # 6
        [1, 1, 1],  # 7
    ],
    dtype=torch.int64,
)

# 12 edges defined by corner pairs
# First 4: z=0 face loop, next 4: z=1 face loop, last 4: vertical edges
CUBE_EDGES = torch.tensor(
    [
        # z=0 face loop (0-2-6-4)
        [0, 2],
        [2, 6],
        [6, 4],
        [4, 0],
        # z=1 face loop (1-3-7-5)
        [1, 3],
        [3, 7],
        [7, 5],
        [5, 1],
        # vertical edges along z
        [0, 1],
        [2, 3],
        [6, 7],
        [4, 5],
    ],
    dtype=torch.int64,
)

# 6 faces defined by 4 corners each (CCW when viewed from outside)
CUBE_FACES = torch.tensor(
    [
        [0, 2, 6, 4],  # -Z
        [1, 5, 7, 3],  # +Z
        [0, 1, 3, 2],  # -X
        [4, 6, 7, 5],  # +X
        [0, 4, 5, 1],  # -Y
        [2, 3, 7, 6],  # +Y
    ],
    dtype=torch.int64,
)

# Edge to face mapping: which 2 faces each edge belongs to
EDGE_TO_FACES = torch.tensor(
    [
        # z=0 loop edges
        [0, 2],
        [0, 5],
        [0, 3],
        [0, 4],
        # z=1 loop edges
        [1, 2],
        [1, 5],
        [1, 3],
        [1, 4],
        # vertical edges
        [2, 4],
        [2, 5],
        [3, 5],
        [3, 4],
    ],
    dtype=torch.int64,
)

# Face normal directions (integer axis normals)
FACE_NORMALS = torch.tensor(
    [
        [0, 0, -1],  # 0: -Z
        [0, 0, 1],  # 1: +Z
        [-1, 0, 0],  # 2: -X
        [1, 0, 0],  # 3: +X
        [0, -1, 0],  # 4: -Y
        [0, 1, 0],  # 5: +Y
    ],
    dtype=torch.int64,
)


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
    """

    def __init__(
        self,
        resolution: int,
        bounds: Optional[torch.Tensor] = None,
        device: str = "cuda",
    ):
        if not isinstance(resolution, int) or resolution < 1:
            raise ValueError(f"resolution must be positive int, got {resolution}")

        self.res = int(resolution)
        self.device = device
        self.dtype = torch.float32
        self.index_dtype = torch.int64

        # Vertex/cube counts
        self.num_vertices_per_axis = self.res + 1
        self.num_cubes = self.res**3
        self.num_vertices = self.num_vertices_per_axis**3

        # Bounds and cell size
        if bounds is not None:
            if bounds.shape != (2, 3):
                raise ValueError(f"bounds must be [2,3], got {tuple(bounds.shape)}")
            self.bounds = bounds.to(device=self.device, dtype=self.dtype)
        else:
            self.bounds = torch.tensor(
                [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
                device=self.device,
                dtype=self.dtype,
            )

        self.cell_size = (self.bounds[1] - self.bounds[0]) / float(self.res)

        # 1D coordinates (x-axis only; y/z can be derived similarly if needed)
        self.coords_1d = torch.linspace(
            self.bounds[0, 0].item(),
            self.bounds[1, 0].item(),
            self.num_vertices_per_axis,
            device=self.device,
            dtype=self.dtype,
        )

        # Move topology constants to device
        self.CUBE_CORNERS = CUBE_CORNERS.to(self.device)
        self.CUBE_EDGES = CUBE_EDGES.to(self.device)
        self.CUBE_FACES = CUBE_FACES.to(self.device)
        self.EDGE_TO_FACES = EDGE_TO_FACES.to(self.device)
        self.FACE_NORMALS = FACE_NORMALS.to(self.device)

        # Precompute edge indexing
        self._setup_edge_indexing()

        # Precompute face indexing
        self._setup_face_indexing()

    def _setup_edge_indexing(self):
        """Precompute edge index offsets and local edge metadata."""
        edge_dirs = self.CUBE_CORNERS[self.CUBE_EDGES[:, 1]] - self.CUBE_CORNERS[self.CUBE_EDGES[:, 0]]
        self._edge_axis = torch.argmax(edge_dirs.abs(), dim=1)  # [12], values in {0,1,2}

        self._edge_anchor_local = torch.minimum(
            self.CUBE_CORNERS[self.CUBE_EDGES[:, 0]],
            self.CUBE_CORNERS[self.CUBE_EDGES[:, 1]],
        )  # [12,3]

        r = self.res
        p = self.num_vertices_per_axis

        self._num_edges_x = r * p * p
        self._num_edges_y = p * r * p
        self._num_edges_z = p * p * r

        self._edge_offset_x = 0
        self._edge_offset_y = self._num_edges_x
        self._edge_offset_z = self._num_edges_x + self._num_edges_y

        self.num_edges = self._num_edges_x + self._num_edges_y + self._num_edges_z

    def _setup_face_indexing(self):
        """Precompute face index offsets."""
        r = self.res
        p = self.num_vertices_per_axis

        self._num_faces_x = p * r * r
        self._num_faces_y = r * p * r
        self._num_faces_z = r * r * p

        self._face_offset_x = 0
        self._face_offset_y = self._num_faces_x
        self._face_offset_z = self._num_faces_x + self._num_faces_y

        self.num_faces = self._num_faces_x + self._num_faces_y + self._num_faces_z

    # ============================================================
    # Coordinate Conversions
    # ============================================================

    def world_to_grid(self, world_coords: torch.Tensor) -> torch.Tensor:
        return (world_coords.to(self.device, dtype=self.dtype) - self.bounds[0]) / self.cell_size

    def grid_to_world(self, grid_coords: torch.Tensor) -> torch.Tensor:
        return (grid_coords.to(self.device).float() + 0.5) * self.cell_size + self.bounds[0]

    def vertex_to_world(self, vertex_ijk: torch.Tensor) -> torch.Tensor:
        return vertex_ijk.to(self.device).float() * self.cell_size + self.bounds[0]

    # ============================================================
    # Index Utilities
    # ============================================================

    def ravel_ijk(self, ijk: torch.Tensor, dims: Tuple[int, int, int]) -> torch.Tensor:
        nx, ny, nz = dims
        ijk = ijk.to(device=self.device, dtype=self.index_dtype)
        mul = torch.tensor([ny * nz, nz, 1], device=self.device, dtype=self.index_dtype)
        return (ijk * mul).sum(dim=-1)

    def unravel_idx(self, idx: torch.Tensor, dims: Tuple[int, int, int]) -> torch.Tensor:
        nx, ny, nz = dims
        idx = idx.to(device=self.device, dtype=self.index_dtype)
        i = torch.div(idx, ny * nz, rounding_mode="floor")
        rem = idx - i * (ny * nz)
        j = torch.div(rem, nz, rounding_mode="floor")
        k = rem - j * nz
        return torch.stack([i, j, k], dim=-1)

    # ============================================================
    # Cube Indexing
    # ============================================================

    def all_cube_indices(self) -> torch.Tensor:
        return torch.arange(self.num_cubes, device=self.device, dtype=self.index_dtype)

    def cube_to_ijk(self, cube_idx: torch.Tensor) -> torch.Tensor:
        return self.unravel_idx(cube_idx, (self.res, self.res, self.res))

    def ijk_to_cube(self, ijk: torch.Tensor) -> torch.Tensor:
        return self.ravel_ijk(ijk, (self.res, self.res, self.res))

    def cube_aabb(self, cube_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ijk = self.cube_to_ijk(cube_idx)
        mins = self.vertex_to_world(ijk)
        maxs = mins + self.cell_size
        return mins, maxs

    def cube_center(self, cube_idx: torch.Tensor) -> torch.Tensor:
        ijk = self.cube_to_ijk(cube_idx)
        return self.grid_to_world(ijk)

    # ============================================================
    # Cube -> Vertex (8 corners)
    # ============================================================

    def cube_corner_vertex_indices(self, cube_idx: torch.Tensor) -> torch.Tensor:
        ijk = self.cube_to_ijk(cube_idx)
        corners_ijk = ijk[:, None, :] + self.CUBE_CORNERS[None, :, :]
        p = self.num_vertices_per_axis
        vertex_indices = self.ravel_ijk(corners_ijk.reshape(-1, 3), (p, p, p))
        return vertex_indices.view(-1, 8)

    def cube_corner_coords(self, cube_idx: torch.Tensor) -> torch.Tensor:
        ijk = self.cube_to_ijk(cube_idx)
        corners_ijk = ijk[:, None, :] + self.CUBE_CORNERS[None, :, :]
        return self.vertex_to_world(corners_ijk)

    def vertex_ijk_from_indices(self, vertex_idx: torch.Tensor) -> torch.Tensor:
        p = self.num_vertices_per_axis
        return self.unravel_idx(vertex_idx.flatten(), (p, p, p)).view(*vertex_idx.shape, 3)

    def vertex_coords_from_indices(self, vertex_idx: torch.Tensor) -> torch.Tensor:
        orig_shape = vertex_idx.shape
        vertex_idx_flat = vertex_idx.flatten()
        p = self.num_vertices_per_axis
        ijk = self.unravel_idx(vertex_idx_flat, (p, p, p))
        coords = self.vertex_to_world(ijk)
        return coords.view(*orig_shape, 3)

    def voxel_unique_vertices(self, voxels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if voxels.dim() == 2 and voxels.shape[1] == 3:
            cube_idx = self.ijk_to_cube(voxels)
        elif voxels.dim() == 1:
            cube_idx = voxels
        else:
            raise ValueError(f"voxels must be [N,3] or [N], got {tuple(voxels.shape)}")

        N = cube_idx.shape[0]
        corner_indices = self.cube_corner_vertex_indices(cube_idx)
        all_vertex_indices = corner_indices.flatten()
        unique_vertices, inverse = torch.unique(all_vertex_indices, return_inverse=True)
        unique_coords = self.vertex_coords_from_indices(unique_vertices)
        voxel_to_vertex = inverse.view(N, 8)
        return unique_vertices, unique_coords, voxel_to_vertex

    # ============================================================
    # Cube -> Edge (12)
    # ============================================================

    def voxel_unique_edges(self, voxels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if voxels.dim() == 2 and voxels.shape[1] == 3:
            cube_idx = self.ijk_to_cube(voxels)
        elif voxels.dim() == 1:
            cube_idx = voxels
        else:
            raise ValueError(f"voxels must be [N,3] or [N], got {tuple(voxels.shape)}")

        edge_indices = self.cube_edge_indices(cube_idx)
        all_edge_indices = edge_indices.flatten()
        unique_edges, inverse = torch.unique(all_edge_indices, return_inverse=True)
        voxel_to_edge = inverse.view(edge_indices.shape[0], 12)
        return unique_edges, voxel_to_edge

    def cube_edge_indices(self, cube_idx: torch.Tensor) -> torch.Tensor:
        if cube_idx is None:
            cube_idx = self.all_cube_indices()

        cube_ijk = self.cube_to_ijk(cube_idx)
        B = cube_ijk.shape[0]

        anchor_local = self._edge_anchor_local.to(self.index_dtype)
        anchor_ijk = cube_ijk[:, None, :] + anchor_local[None, :, :]

        edge_axis = self._edge_axis
        edge_axis_b = edge_axis.view(1, 12).expand(B, 12)

        Eout = torch.empty((B, 12), device=self.device, dtype=self.index_dtype)

        p = self.num_vertices_per_axis
        r = self.res

        mask_x = edge_axis_b == 0
        if mask_x.any():
            ijk_x = anchor_ijk[mask_x]
            local = ijk_x[:, 0] * (p * p) + ijk_x[:, 1] * p + ijk_x[:, 2]
            Eout[mask_x] = local + self._edge_offset_x

        mask_y = edge_axis_b == 1
        if mask_y.any():
            ijk_y = anchor_ijk[mask_y]
            local = ijk_y[:, 0] * (r * p) + ijk_y[:, 1] * p + ijk_y[:, 2]
            Eout[mask_y] = local + self._edge_offset_y

        mask_z = edge_axis_b == 2
        if mask_z.any():
            ijk_z = anchor_ijk[mask_z]
            local = ijk_z[:, 0] * (p * r) + ijk_z[:, 1] * r + ijk_z[:, 2]
            Eout[mask_z] = local + self._edge_offset_z

        return Eout

    def edge_endpoints(self, edge_idx: torch.Tensor) -> torch.Tensor:
        edge_idx = edge_idx.to(self.device, dtype=self.index_dtype).flatten()
        E = edge_idx.numel()
        coords = torch.empty((E, 2, 3), dtype=self.dtype, device=self.device)

        p = self.num_vertices_per_axis
        r = self.res

        mask_x = (edge_idx >= self._edge_offset_x) & (edge_idx < self._edge_offset_y)
        if mask_x.any():
            local = edge_idx[mask_x] - self._edge_offset_x
            ijk = self.unravel_idx(local, (r, p, p))
            v0 = ijk.clone()
            v1 = ijk.clone()
            v1[:, 0] = v1[:, 0] + 1
            coords[mask_x, 0] = self.vertex_to_world(v0)
            coords[mask_x, 1] = self.vertex_to_world(v1)

        mask_y = (edge_idx >= self._edge_offset_y) & (edge_idx < self._edge_offset_z)
        if mask_y.any():
            local = edge_idx[mask_y] - self._edge_offset_y
            ijk = self.unravel_idx(local, (p, r, p))
            v0 = ijk.clone()
            v1 = ijk.clone()
            v1[:, 1] = v1[:, 1] + 1
            coords[mask_y, 0] = self.vertex_to_world(v0)
            coords[mask_y, 1] = self.vertex_to_world(v1)

        mask_z = edge_idx >= self._edge_offset_z
        if mask_z.any():
            local = edge_idx[mask_z] - self._edge_offset_z
            ijk = self.unravel_idx(local, (p, p, r))
            v0 = ijk.clone()
            v1 = ijk.clone()
            v1[:, 2] = v1[:, 2] + 1
            coords[mask_z, 0] = self.vertex_to_world(v0)
            coords[mask_z, 1] = self.vertex_to_world(v1)

        return coords

    # ============================================================
    # Cube -> Face (6)
    # ============================================================

    def cube_face_indices(self, cube_idx: torch.Tensor) -> torch.Tensor:
        ijk = self.cube_to_ijk(cube_idx)
        cx, cy, cz = ijk[:, 0], ijk[:, 1], ijk[:, 2]

        p = self.num_vertices_per_axis
        r = self.res

        f_minus_z = self.ravel_ijk(torch.stack([cx, cy, cz], dim=-1), (r, r, p)) + self._face_offset_z
        f_plus_z = self.ravel_ijk(torch.stack([cx, cy, cz + 1], dim=-1), (r, r, p)) + self._face_offset_z

        f_minus_x = self.ravel_ijk(torch.stack([cx, cy, cz], dim=-1), (p, r, r)) + self._face_offset_x
        f_plus_x = self.ravel_ijk(torch.stack([cx + 1, cy, cz], dim=-1), (p, r, r)) + self._face_offset_x

        f_minus_y = self.ravel_ijk(torch.stack([cx, cy, cz], dim=-1), (r, p, r)) + self._face_offset_y
        f_plus_y = self.ravel_ijk(torch.stack([cx, cy + 1, cz], dim=-1), (r, p, r)) + self._face_offset_y

        return torch.stack([f_minus_z, f_plus_z, f_minus_x, f_plus_x, f_minus_y, f_plus_y], dim=1)

    # ============================================================
    # Edge <-> Cube Incidence (FIXED: strict right-hand CCW around edge_endpoints dir)
    # ============================================================

    def edge_incident_cubes(
        self,
        edge_idx: torch.Tensor,
        edge_directions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        For each global edge index, return up to 4 incident cubes [E,4].

        Ordering rule (matches your old primal_edge_incident_cubes_righthand):
        - Take edge direction as (p1 - p0) from edge_endpoints().
          In this implementation, that is always +X/+Y/+Z along grid axes.
        - Treat the edge direction as +Z of a local frame; the 4 incident cubes in the
          orthogonal plane are returned as a right-handed CCW loop.

        Invalid / boundary cubes are filled with -1.

        edge_directions:
          +1 (default): keep CCW
          -1: reverse to CW, per edge
        """
        edge_idx = edge_idx.to(device=self.device, dtype=self.index_dtype).flatten()
        E = edge_idx.numel()
        out = torch.full((E, 4), -1, device=self.device, dtype=self.index_dtype)

        r = self.res
        p = self.num_vertices_per_axis

        num_ex = self._num_edges_x
        num_ey = self._num_edges_y
        # num_ez = self._num_edges_z

        # ---------- X-direction edges ----------
        # local dims: (r, p, p) => (ix, jv, kv)
        mask_x = edge_idx < num_ex
        if mask_x.any():
            idx_x = torch.nonzero(mask_x, as_tuple=False).squeeze(-1)
            local = edge_idx[idx_x] - self._edge_offset_x
            ijk = self.unravel_idx(local, (r, p, p)).to(self.index_dtype)  # (ix, jv, kv)

            ix = ijk[:, 0:1]
            jv = ijk[:, 1:2]
            kv = ijk[:, 2:3]

            # View along +X, plane is (Y,Z). CCW loop:
            # (0) (y,   z)
            # (1) (y,   z-1)
            # (2) (y-1, z-1)
            # (3) (y-1, z)
            dy = torch.tensor([0, 0, 1, 1], device=self.device, dtype=self.index_dtype).view(1, 4)
            dz = torch.tensor([0, 1, 1, 0], device=self.device, dtype=self.index_dtype).view(1, 4)

            cy = jv - dy
            cz = kv - dz
            cx = ix.expand_as(cy)

            valid = (cx >= 0) & (cx < r) & (cy >= 0) & (cy < r) & (cz >= 0) & (cz < r)
            cubes_ijk = torch.stack([cx, cy, cz], dim=-1)  # [Nx,4,3]

            cubes_lin = self.ijk_to_cube(cubes_ijk.view(-1, 3)).view(-1, 4)
            cubes_lin = torch.where(valid, cubes_lin, torch.full_like(cubes_lin, -1))
            out[idx_x] = cubes_lin

        # ---------- Y-direction edges ----------
        # local dims: (p, r, p) => (iv, jy, kv)
        mask_y = (edge_idx >= num_ex) & (edge_idx < num_ex + num_ey)
        if mask_y.any():
            idx_y = torch.nonzero(mask_y, as_tuple=False).squeeze(-1)
            local = edge_idx[idx_y] - self._edge_offset_y
            ijk = self.unravel_idx(local, (p, r, p)).to(self.index_dtype)  # (iv, jy, kv)

            iv = ijk[:, 0:1]
            jy = ijk[:, 1:2]
            kv = ijk[:, 2:3]

            # View along +Y, plane is (X,Z). CCW loop:
            # (0) (x-1, z)
            # (1) (x-1, z-1)
            # (2) (x,   z-1)
            # (3) (x,   z)
            dx = torch.tensor([-1, -1, 0, 0], device=self.device, dtype=self.index_dtype).view(1, 4)
            dz = torch.tensor([0, -1, -1, 0], device=self.device, dtype=self.index_dtype).view(1, 4)

            cx = iv + dx
            cy = jy.expand_as(cx)
            cz = kv + dz

            valid = (cx >= 0) & (cx < r) & (cy >= 0) & (cy < r) & (cz >= 0) & (cz < r)
            cubes_ijk = torch.stack([cx, cy, cz], dim=-1)

            cubes_lin = self.ijk_to_cube(cubes_ijk.view(-1, 3)).view(-1, 4)
            cubes_lin = torch.where(valid, cubes_lin, torch.full_like(cubes_lin, -1))
            out[idx_y] = cubes_lin

        # ---------- Z-direction edges ----------
        # local dims: (p, p, r) => (iv, jv, kz)
        mask_z = ~(mask_x | mask_y)
        if mask_z.any():
            idx_z = torch.nonzero(mask_z, as_tuple=False).squeeze(-1)
            local = edge_idx[idx_z] - self._edge_offset_z
            ijk = self.unravel_idx(local, (p, p, r)).to(self.index_dtype)  # (iv, jv, kz)

            iv = ijk[:, 0:1]
            jv = ijk[:, 1:2]
            kz = ijk[:, 2:3]

            # View along +Z, plane is (X,Y). CCW loop:
            # (0) (x-1, y)
            # (1) (x,   y)
            # (2) (x,   y-1)
            # (3) (x-1, y-1)
            dx = torch.tensor([-1, 0, 0, -1], device=self.device, dtype=self.index_dtype).view(1, 4)
            dy = torch.tensor([0, 0, -1, -1], device=self.device, dtype=self.index_dtype).view(1, 4)

            cx = iv + dx
            cy = jv + dy
            cz = kz.expand_as(cx)

            valid = (cx >= 0) & (cx < r) & (cy >= 0) & (cy < r) & (cz >= 0) & (cz < r)
            cubes_ijk = torch.stack([cx, cy, cz], dim=-1)

            cubes_lin = self.ijk_to_cube(cubes_ijk.view(-1, 3)).view(-1, 4)
            cubes_lin = torch.where(valid, cubes_lin, torch.full_like(cubes_lin, -1))
            out[idx_z] = cubes_lin

        # Optional per-edge reversal
        if edge_directions is not None:
            edge_directions = edge_directions.to(self.device).flatten()
            if edge_directions.numel() != E:
                raise ValueError(
                    f"edge_directions must have same length as edge_idx ({E}), got {edge_directions.numel()}"
                )
            rev = edge_directions < 0
            if rev.any():
                out[rev] = out[rev].flip(dims=[1])

        return out

    # ============================================================
    # Cell Generation
    # ============================================================

    def generate_all_cells(self) -> torch.Tensor:
        x = torch.arange(self.res, device=self.device, dtype=self.index_dtype)
        xx, yy, zz = torch.meshgrid(x, x, x, indexing="ij")
        return torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=1)

    def all_vertex_indices(self) -> torch.Tensor:
        return torch.arange(self.num_vertices, device=self.device, dtype=self.index_dtype)

    def all_vertex_ijk(self) -> torch.Tensor:
        p = self.num_vertices_per_axis
        x = torch.arange(p, device=self.device, dtype=self.index_dtype)
        xx, yy, zz = torch.meshgrid(x, x, x, indexing="ij")
        return torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=1)

    def all_vertex_coords(self) -> torch.Tensor:
        return self.vertex_to_world(self.all_vertex_ijk())

    def generate_candidate_cells_from_aabb(self, aabb_min: torch.Tensor, aabb_max: torch.Tensor) -> torch.Tensor:
        aabb_min = aabb_min.to(self.device, dtype=self.dtype)
        aabb_max = aabb_max.to(self.device, dtype=self.dtype)

        grid_min = self.world_to_grid(aabb_min).floor().to(self.index_dtype).clamp(0, self.res - 1)
        grid_max = self.world_to_grid(aabb_max).ceil().to(self.index_dtype).clamp(0, self.res - 1)

        global_min = grid_min.min(dim=0)[0]
        global_max = grid_max.max(dim=0)[0]

        x = torch.arange(global_min[0], global_max[0] + 1, device=self.device, dtype=self.index_dtype)
        y = torch.arange(global_min[1], global_max[1] + 1, device=self.device, dtype=self.index_dtype)
        z = torch.arange(global_min[2], global_max[2] + 1, device=self.device, dtype=self.index_dtype)

        xx, yy, zz = torch.meshgrid(x, y, z, indexing="ij")
        return torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=1)
