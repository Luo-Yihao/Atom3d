
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Union, NamedTuple
from ..constants import CUBE_CORNERS
from .tables import *


class SparseDiffDMCOutput(NamedTuple):
    """Output of SparseDiffDMC forward pass."""
    vertices: torch.Tensor          # [V, 3] mesh vertices
    faces: torch.Tensor             # [F, 3] mesh faces (int64)
    L_dev: torch.Tensor             # [E] per-edge regularization loss
    p_alpha: torch.Tensor           # [E, 3] alpha-weighted zero-crossing points
    n_alpha: torch.Tensor           # [E, 3] normals at crossing points
    v_features: Optional[torch.Tensor] = None  # [V, C] interpolated vertex features


class SparseDMC(nn.Module):
    """
    Sparse Dual Marching Cubes (DMC).
    
    A sparse, differentiable mesh extraction layer based on FlexiCubes.
    
    This is an alias for SparseDiffDMC.
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.module = SparseDiffDMC(*args, **kwargs)
        
    def forward(self, *args, **kwargs):
        return self.module.forward(*args, **kwargs)


class SparseDiffDMC(nn.Module):
    """
    Sparse Differentiable Dual Marching Cubes.
    
    A sparse, differentiable mesh extraction layer based on FlexiCubes.
    
    Modes:
    - train(): Uses weighted average (better differentiability)
    - eval(): Uses QEF solver by default (better precision)
    
    The p_alpha and n_alpha intermediate outputs are always returned,
    allowing external QEF solvers or post-processing.
    """

    def __init__(self, device: str = "cuda", dtype: torch.dtype = torch.float32):
        super().__init__()
        self.device = device
        self.dtype = dtype
        
        # Load tables
        self.register_buffer('dmc_table', torch.tensor(dmc_table, dtype=torch.long, device=device))
        self.register_buffer('num_vd_table', torch.tensor(num_vd_table, dtype=torch.long, device=device))
        self.register_buffer('check_table', torch.tensor(check_table, dtype=torch.long, device=device))
        self.register_buffer('tet_table', torch.tensor(tet_table, dtype=torch.long, device=device))

        self.register_buffer('quad_split_1', torch.tensor([0, 1, 2, 0, 2, 3], dtype=torch.long, device=device))
        self.register_buffer('quad_split_2', torch.tensor([0, 1, 3, 3, 1, 2], dtype=torch.long, device=device))
        self.register_buffer('quad_split_train', torch.tensor([0, 1, 1, 2, 2, 3, 3, 0], dtype=torch.long, device=device))

        self.register_buffer('cube_corners', CUBE_CORNERS.to(device).float())
        self.register_buffer('cube_corners_idx', torch.pow(2, torch.arange(8, dtype=torch.long, device=device)))
        
        self.register_buffer('cube_edges', torch.tensor([
            0, 1, 1, 5, 4, 5, 0, 4, 
            2, 3, 3, 7, 6, 7, 2, 6,
            2, 0, 3, 1, 7, 5, 6, 4
        ], dtype=torch.long, device=device))

        self.adj_pairs = torch.tensor([0, 1, 1, 3, 3, 2, 2, 0], dtype=torch.long, device=device)

    def forward(
        self,
        voxel_coords: torch.Tensor,
        sdf: torch.Tensor,
        cube_idx: torch.Tensor,
        resolution: int,
        deform: Optional[torch.Tensor] = None,
        beta: Optional[torch.Tensor] = None,
        alpha: Optional[torch.Tensor] = None,
        gamma: Optional[torch.Tensor] = None,
        v_features: Optional[torch.Tensor] = None,
        isovalue: float = 0.0,
        qef_reg_scale: float = 1e-2,
        weight_scale: float = 0.99,
        inference_mode: str = "auto",
        dtype: Optional[torch.dtype] = None,
        corner_gradients: Optional[torch.Tensor] = None,
    ) -> SparseDiffDMCOutput:
        """
        Extract mesh from sparse voxel SDF.

        Args:
            voxel_coords: [N, 3] integer coordinates of active cubes
            sdf: [M] SDF values at unique grid corners
            cube_idx: [N, 8] indices into sdf for each cube's 8 corners
            resolution: Grid resolution
            deform: [M, 3] optional vertex deformation in world space
            beta: [N, 12] edge weights for dual vertex computation
            alpha: [N, 8] corner weights for interpolation
            gamma: [N] quad split decision weights
            v_features: [M, C] optional vertex features (e.g., colors)
            isovalue: Isosurface value (default 0)
            qef_reg_scale: Regularization scale for QEF
            weight_scale: Scale for weight normalization
            inference_mode:
                - "auto": train→weighted_avg, eval→qef
                - "weighted_avg": force FlexiCubes style
                - "qef": force QEF solver
            dtype: Optional dtype for precision control
            corner_gradients: [M, 3] pre-computed SDF gradients at grid corners.
                If provided, skips internal finite-difference estimation.
                For BVH-based SDF: normalize(p - closest_point(p)) is exact.
            
        Returns:
            SparseDiffDMCOutput NamedTuple with fields:
                vertices, faces, L_dev, p_alpha, n_alpha, v_features (optional)
        """
        # Determine actual mode
        if inference_mode == "auto":
            mode = "weighted_avg" if self.training else "qef"
        else:
            mode = inference_mode
        
        # Unify dtype
        if dtype is None:
            if sdf.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
                dtype = sdf.dtype
            else:
                dtype = torch.float32
        
        device = self.device
        
        # Cast inputs
        sdf = sdf.to(dtype)
        if deform is not None:
            deform = deform.to(dtype)
        if beta is not None:
            beta = beta.to(dtype)
        if alpha is not None:
            alpha = alpha.to(dtype)
        if gamma is not None:
            gamma = gamma.to(dtype)
        if v_features is not None:
            v_features = v_features.to(dtype)
        
        # 1. Recover Unique Corner Positions
        M = sdf.shape[0]
        unique_corners_pos = torch.zeros((M, 3), dtype=dtype, device=device)
        
        N = voxel_coords.shape[0]
        cube_corner_pos = (voxel_coords.unsqueeze(1) + self.cube_corners.unsqueeze(0)).contiguous()
        
        flat_indices = cube_idx.contiguous().view(-1).long()
        flat_pos = cube_corner_pos.view(-1, 3).to(dtype)
        unique_corners_pos.index_put_((flat_indices,), flat_pos)

        # 2. Apply Deformation
        world_scale = 2.0 / resolution
        grid_pos = unique_corners_pos
        world_pos = grid_pos * world_scale - 1.0
        
        if deform is not None:
            world_pos = world_pos + deform

        voxelgrid_vertices = world_pos

        # 3. Adjust SDF
        scalar_field = sdf - isovalue

        # 4. Identify Surface Cubes
        surf_cubes, occ_fx8 = self._identify_surf_cubes(scalar_field, cube_idx)
        
        if surf_cubes.sum() == 0:
            empty_3 = torch.zeros((0, 3), device=device, dtype=dtype)
            empty_1 = torch.zeros((0,), device=device, dtype=dtype)
            return SparseDiffDMCOutput(
                vertices=empty_3,
                faces=torch.zeros((0, 3), dtype=torch.int64, device=device),
                L_dev=empty_1,
                p_alpha=empty_3,
                n_alpha=empty_3,
                v_features=None,
            )

        # 5. Normalize Weights
        beta_s, alpha_s, gamma_s = self._normalize_weights(beta, alpha, gamma, surf_cubes, weight_scale, dtype)
        
        # 6. Case IDs
        case_ids = self._get_case_id_sparse(occ_fx8, surf_cubes, resolution, voxel_coords)

        # 7. Identify Surface Edges
        surf_edges, idx_map, edge_counts, surf_edges_mask = self._identify_surf_edges(
            scalar_field, cube_idx, surf_cubes
        )

        # 8. Compute Dual Vertices
        # In QEF mode we need per-corner SDF gradients for accurate surface normals.
        # Prefer caller-supplied exact gradients (e.g. normalize(p - bvh.closest(p)))
        # over the internal finite-difference estimation, which is less accurate.
        if mode == "qef":
            if corner_gradients is None:
                corner_gradients = self._estimate_corner_gradients(
                    voxelgrid_vertices, scalar_field, cube_idx, dtype
                )
            else:
                corner_gradients = corner_gradients.to(dtype)
        vd, L_dev, vd_gamma, vd_idx_map, vd_features, p_alpha, n_alpha = self._compute_vd(
            voxelgrid_vertices, cube_idx[surf_cubes], surf_edges, scalar_field,
            case_ids, beta_s, alpha_s, gamma_s, idx_map, qef_reg_scale, v_features, dtype, mode,
            corner_gradients=corner_gradients,
        )

        # 9. Triangulate
        vertices, faces, vd_features = self._triangulate(
            scalar_field, surf_edges, vd, vd_gamma, edge_counts, idx_map,
            vd_idx_map, surf_edges_mask, self.training, vd_features, dtype,
            corner_gradients=corner_gradients if mode == "qef" else None,
        )

        return SparseDiffDMCOutput(
            vertices=vertices,
            faces=faces.long(),
            L_dev=L_dev,
            p_alpha=p_alpha,
            n_alpha=n_alpha,
            v_features=vd_features,
        )

    def _normalize_weights(self, beta, alpha, gamma_f, surf_cubes, weight_scale, dtype):
        n_cubes = surf_cubes.shape[0]
        device = self.device
        
        if beta is not None:
            beta_out = (torch.tanh(beta) * weight_scale + 1)
        else:
            beta_out = torch.ones((n_cubes, 12), dtype=dtype, device=device)
             
        if alpha is not None:
            alpha_out = (torch.tanh(alpha) * weight_scale + 1)
        else:
            alpha_out = torch.ones((n_cubes, 8), dtype=dtype, device=device)
             
        if gamma_f is not None:
            gamma_out = torch.sigmoid(gamma_f) * weight_scale + (1 - weight_scale) / 2
        else:
            gamma_out = torch.ones((n_cubes,), dtype=dtype, device=device)
             
        return beta_out[surf_cubes], alpha_out[surf_cubes], gamma_out[surf_cubes]

    def _compute_reg_loss(self, vd, ue, edge_group_to_vd, vd_num_edges):
        """L_dev regularization loss."""
        dist = torch.norm(ue - torch.index_select(input=vd, index=edge_group_to_vd, dim=0), dim=-1)
        mean_l2 = torch.zeros_like(vd[:, 0])
        mean_l2 = mean_l2.index_add_(0, edge_group_to_vd, dist) / vd_num_edges.squeeze(1).float()
        mad = (dist - torch.index_select(input=mean_l2, index=edge_group_to_vd, dim=0)).abs()
        return mad

    def _compute_normals_from_gradient(self, scalar_field, surf_edges, voxelgrid_vertices):
        """
        Estimate normals at edge crossing points from SDF gradient.
        Uses the edge direction and SDF difference.
        """
        # Get edge endpoints
        p0 = voxelgrid_vertices[surf_edges[:, 0]]
        p1 = voxelgrid_vertices[surf_edges[:, 1]]
        s0 = scalar_field[surf_edges[:, 0]]
        s1 = scalar_field[surf_edges[:, 1]]
        
        # Edge direction
        edge_vec = p1 - p0
        edge_len = edge_vec.norm(dim=-1, keepdim=True) + 1e-8
        edge_dir = edge_vec / edge_len
        
        # Gradient along edge (magnitude and sign)
        grad_mag = (s1 - s0).unsqueeze(-1) / edge_len
        
        # Normal approximation: perpendicular to edge, scaled by gradient
        # For a more accurate normal, we'd need cross-edge gradients
        # Here we use edge direction as a proxy (works for axis-aligned edges)
        normals = edge_dir * grad_mag.sign()
        
        return F.normalize(normals, dim=-1)

    def _solve_qef_scatter(self, crossing_points, normals, edge_group_to_vd, num_vd, qef_reg_scale,
                           vd_cell_center=None, vd_cell_min=None, vd_cell_max=None):
        """
        QEF solver: Tikhonov regularization toward cell center + mass-point fallback.

        Standard "Dual Contouring: The Secret Sauce" approach (Schaefer & Warren):
          1. Solve min_x Σ (n_j · (x - p_j))² + λ ||x - cell_center||²
          2. If solution falls outside cell bounds, fall back to mass point (centroid).

        Tikhonov regularization handles rank-deficient cases (coplanar normals)
        by pulling toward cell center. Mass-point fallback catches remaining outliers.

        Uses lstsq on [A; √λ I] δ = [b; √λ (cell_center - centroid)]
        in centroid-relative coordinates for numerical stability.
        """
        device = crossing_points.device
        dtype = crossing_points.dtype
        K = crossing_points.shape[0]

        if num_vd == 0 or K == 0:
            return torch.zeros((max(num_vd, 1), 3), device=device, dtype=dtype)[:num_vd]

        edge_group_to_vd = edge_group_to_vd.long()

        # Mass point (centroid of crossing points per VD)
        centroid = torch.zeros(num_vd, 3, device=device, dtype=dtype)
        centroid.scatter_add_(0, edge_group_to_vd.view(-1, 1).expand(-1, 3), crossing_points)
        counts = torch.zeros(num_vd, device=device, dtype=dtype)
        counts.scatter_add_(0, edge_group_to_vd, torch.ones(K, device=device, dtype=dtype))
        counts = counts.clamp_min(1)
        centroid = centroid / counts.view(-1, 1)

        # Regularization target: cell center if available, else centroid
        reg_target = vd_cell_center if vd_cell_center is not None else centroid

        # Shift to local coordinates (centroid-relative)
        local_pts = crossing_points - centroid[edge_group_to_vd]  # [K, 3]

        # Constraint: n_j · (x - q_j) = 0  →  n_j · δ = n_j · q_j
        rhs = (normals * local_pts).sum(dim=-1)  # [K]

        MAX_EDGES = 7
        ROWS = MAX_EDGES + 3  # 7 constraints + 3 regularization

        A = torch.zeros(num_vd, ROWS, 3, device=device, dtype=dtype)
        b = torch.zeros(num_vd, ROWS, 1, device=device, dtype=dtype)

        # Scatter constraints into per-VD fixed-size slots
        _, sort_idx = torch.sort(edge_group_to_vd, stable=True)
        sorted_vd = edge_group_to_vd[sort_idx]
        group_start = torch.zeros(K, dtype=torch.long, device=device)
        group_start[1:] = (sorted_vd[1:] != sorted_vd[:-1]).long()
        group_start = group_start.cumsum(0)
        group_offsets = torch.zeros(num_vd, dtype=torch.long, device=device)
        group_offsets.scatter_(0, sorted_vd, torch.arange(K, device=device))
        local_idx = torch.arange(K, device=device) - group_offsets[sorted_vd]
        local_idx = local_idx.clamp(max=MAX_EDGES - 1)
        inv_sort = torch.empty_like(sort_idx)
        inv_sort[sort_idx] = torch.arange(K, device=device)
        slot = local_idx[inv_sort]

        # Fill A and b
        A[edge_group_to_vd, slot] = normals
        b[edge_group_to_vd, slot, 0] = rhs

        # Tikhonov regularization: √λ I δ = √λ (reg_target - centroid)
        # In world coords this pulls toward reg_target (cell center).
        sqrt_lambda = qef_reg_scale ** 0.5
        reg_rhs = (reg_target - centroid) * sqrt_lambda  # [G, 3]
        A[:, MAX_EDGES + 0, 0] = sqrt_lambda
        A[:, MAX_EDGES + 1, 1] = sqrt_lambda
        A[:, MAX_EDGES + 2, 2] = sqrt_lambda
        b[:, MAX_EDGES + 0, 0] = reg_rhs[:, 0]
        b[:, MAX_EDGES + 1, 0] = reg_rhs[:, 1]
        b[:, MAX_EDGES + 2, 0] = reg_rhs[:, 2]

        # Batch lstsq
        result = torch.linalg.lstsq(A, b)
        delta = result.solution.squeeze(-1)  # [G, 3]

        # World coordinates
        vd = delta + centroid

        # NaN fallback → mass point
        nan_mask = torch.isnan(vd).any(dim=-1)
        if nan_mask.any():
            vd[nan_mask] = centroid[nan_mask]

        # Mass-point fallback: if QEF solution is far outside cell bounds, use centroid.
        # Allow 1 cell of slack for sharp features, but catch extreme outliers.
        if vd_cell_min is not None and vd_cell_max is not None:
            cell_size = (vd_cell_max - vd_cell_min)  # [G, 3]
            margin = cell_size  # allow 1 full cell outside
            outside = (
                (vd < vd_cell_min - margin).any(dim=-1) |
                (vd > vd_cell_max + margin).any(dim=-1)
            )
            if outside.any():
                vd[outside] = centroid[outside]

        return vd

    def _compute_normals_scatter(self, surf_edges_x, surf_edges_s, edge_group_to_vd, num_vd):
        """
        Compute normals by solving least-squares from edge gradient projections.
        Uses scatter-gather pattern for parallel execution.
        
        For each dual vertex, solves:
            min_g  Σ (d_i · g - ∂f/∂d_i)²
        where d_i is edge direction and ∂f/∂d_i is gradient projection along edge.
        """
        device = surf_edges_x.device
        dtype = surf_edges_x.dtype
        K = surf_edges_x.shape[0]
        
        if num_vd == 0 or K == 0:
            return torch.zeros((max(K, 1), 3), device=device, dtype=dtype)[:K]
        
        edge_group_to_vd = edge_group_to_vd.long()
        
        # Compute edge directions and gradient projections
        p0, p1 = surf_edges_x[:, 0], surf_edges_x[:, 1]
        s0, s1 = surf_edges_s[:, 0, 0], surf_edges_s[:, 1, 0]
        
        edge_vec = p1 - p0
        edge_len = edge_vec.norm(dim=-1, keepdim=True) + 1e-8
        edge_dir = edge_vec / edge_len  # [K, 3]
        grad_proj = (s1 - s0) / edge_len.squeeze(-1)  # [K]
        
        # Build normal equations: D^T D g = D^T b
        # where D[i] = edge_dir[i], b[i] = grad_proj[i]
        outer = edge_dir.unsqueeze(2) * edge_dir.unsqueeze(1)  # [K, 3, 3]
        
        DtD = torch.zeros(num_vd, 3, 3, device=device, dtype=dtype)
        DtD.scatter_add_(0, edge_group_to_vd.view(-1, 1, 1).expand(-1, 3, 3), outer)
        
        Dtb_terms = grad_proj.view(-1, 1) * edge_dir  # [K, 3]
        Dtb = torch.zeros(num_vd, 3, device=device, dtype=dtype)
        Dtb.scatter_add_(0, edge_group_to_vd.view(-1, 1).expand(-1, 3), Dtb_terms)
        
        # Add regularization
        I3 = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)
        DtD = DtD + 1e-3 * I3 + 1e-8 * I3  # regularization + epsilon
        
        # Solve for gradient vectors
        try:
            gradients = torch.linalg.solve(DtD, Dtb.unsqueeze(-1)).squeeze(-1)  # [G, 3]
        except RuntimeError:
            gradients = torch.bmm(torch.linalg.pinv(DtD), Dtb.unsqueeze(-1)).squeeze(-1)
        
        # Normalize to get normals (negate gradient for outward normal)
        normals_per_vd = F.normalize(-gradients, dim=-1)
        
        # Handle NaN (replace with z-axis)
        nan_mask = torch.isnan(normals_per_vd).any(dim=-1)
        if nan_mask.any():
            normals_per_vd[nan_mask] = torch.tensor([0., 0., 1.], device=device, dtype=dtype)
        
        # Expand to per-edge normals
        return normals_per_vd[edge_group_to_vd]

    @torch.no_grad()
    def _estimate_corner_gradients(self, voxelgrid_vertices, scalar_field, cube_idx, dtype):
        """
        Estimate the full 3D gradient at every grid corner using all incident edges.

        On a deformed grid, the edge-direction proxy (used in
        _compute_normals_from_gradient) is inaccurate: each edge only captures the
        1D projection of the gradient along its direction. Here we collect ALL edges
        incident to each corner and solve the over-determined system

            edge_dir_j · g_i  ≈  (s_j - s_i) / |p_j - p_i|   for each neighbor j

        by least squares (D^T D) g = D^T b via scatter_add, giving a proper 3D
        gradient even after arbitrary deformation.
        """
        device = self.device
        M = voxelgrid_vertices.shape[0]

        # All unique edges from all active cubes
        all_edges = cube_idx[:, self.cube_edges.long()].reshape(-1, 2).long()
        unique_edges = torch.unique(all_edges, dim=0)          # [E, 2]
        i0, i1 = unique_edges[:, 0], unique_edges[:, 1]

        p0 = voxelgrid_vertices[i0]
        p1 = voxelgrid_vertices[i1]
        s0 = scalar_field[i0].to(dtype)
        s1 = scalar_field[i1].to(dtype)

        edge_vec = p1 - p0
        edge_len = edge_vec.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        edge_dir = edge_vec / edge_len                          # [E, 3]
        grad_proj = (s1 - s0) / edge_len.squeeze(-1)           # [E]  (Δs / |Δp|)

        # Build D^T D and D^T b for each corner via scatter_add.
        # Each undirected edge (i0, i1) contributes to both endpoints:
        #   corner i0:  direction = +edge_dir,  projection = +grad_proj
        #   corner i1:  direction = -edge_dir,  projection = -grad_proj
        # The outer product d⊗d is the same in both directions, so we accumulate
        # it symmetrically; only the rhs vector flips sign.
        outer   = edge_dir.unsqueeze(2) * edge_dir.unsqueeze(1)  # [E, 3, 3]
        Dtb_fwd = grad_proj.unsqueeze(-1) * edge_dir             # [E, 3]

        DtD = torch.zeros(M, 3, 3, device=device, dtype=dtype)
        Dtb = torch.zeros(M, 3, device=device, dtype=dtype)

        DtD.scatter_add_(0, i0.view(-1, 1, 1).expand(-1, 3, 3), outer)
        DtD.scatter_add_(0, i1.view(-1, 1, 1).expand(-1, 3, 3), outer)
        Dtb.scatter_add_(0, i0.view(-1, 1).expand(-1, 3),  Dtb_fwd)
        Dtb.scatter_add_(0, i1.view(-1, 1).expand(-1, 3), -Dtb_fwd)

        # Regularise (isolated corners get identity → zero gradient fallback)
        I3 = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)
        DtD = DtD + 1e-4 * I3

        try:
            gradients = torch.linalg.solve(DtD, Dtb.unsqueeze(-1)).squeeze(-1)  # [M, 3]
        except RuntimeError:
            gradients = torch.bmm(torch.linalg.pinv(DtD), Dtb.unsqueeze(-1)).squeeze(-1)

        nan_mask = torch.isnan(gradients).any(dim=-1)
        if nan_mask.any():
            gradients[nan_mask] = 0.0

        return gradients  # [M, 3], points in direction of increasing SDF (outward)

    @torch.no_grad()
    def _identify_surf_cubes(self, scalar_field, cube_idx):
        occ_n = scalar_field < 0
        occ_fx8 = occ_n[cube_idx.reshape(-1)].reshape(-1, 8)
        _occ_sum = torch.sum(occ_fx8, -1)
        surf_cubes = (_occ_sum > 0) & (_occ_sum < 8)
        return surf_cubes, occ_fx8

    @torch.no_grad()
    def _get_case_id_sparse(self, occ_fx8, surf_cubes, res, voxel_coords):
        """
        Resolve DMC topology ambiguity (C16/C19 configurations).

        For each problematic surface cube, checks whether its neighbor (given
        by the check_table direction offset) is also problematic.  If so, the
        cube's case_id is replaced with the inverted variant to ensure
        consistent topology across the shared face.

        Neighbor lookup uses a sorted hash table built only over the
        problematic subset, since non-problematic neighbors never trigger
        inversion.  This keeps the sort O(P log P) with P << N_surf.
        """
        device = self.device
        res_long = int(res)

        occ_int = occ_fx8[surf_cubes].long()
        case_ids = (occ_int * self.cube_corners_idx.unsqueeze(0)).sum(-1)

        prob_cfg = self.check_table[case_ids]
        to_check = prob_cfg[..., 0] == 1

        if not to_check.any():
            return case_ids

        prob_coords = voxel_coords[surf_cubes][to_check].long()  # [P, 3]
        prob_cfg_sub = prob_cfg[to_check]                         # [P, 5]

        # Neighbor coordinates from the direction offset in the check table.
        adj_coords = prob_coords + prob_cfg_sub[:, 1:4]           # [P, 3]
        within_range = (adj_coords >= 0).all(dim=-1) & (adj_coords < res_long).all(dim=-1)

        if not within_range.any():
            return case_ids

        # Hash table over all problematic cubes for neighbor lookup.
        # Only problematic cubes can trigger inversion, so the search
        # domain need not include the full surface-cube set.
        prob_hash = prob_coords[:, 0] * (res_long * res_long) + prob_coords[:, 1] * res_long + prob_coords[:, 2]
        sorted_hash, sort_order = prob_hash.sort()

        adj_hash = (adj_coords[within_range, 0] * (res_long * res_long)
                   + adj_coords[within_range, 1] * res_long
                   + adj_coords[within_range, 2])

        pos = torch.searchsorted(sorted_hash, adj_hash).clamp(max=sorted_hash.shape[0] - 1)
        found = sorted_hash[pos] == adj_hash

        if not found.any():
            return case_ids

        # Invert case_ids for cubes whose neighbor is confirmed problematic.
        idx = torch.arange(case_ids.shape[0], device=device)[to_check][within_range][found]
        case_ids.index_put_((idx,), prob_cfg_sub[within_range][found][:, -1])

        return case_ids

    @torch.no_grad()
    def _identify_surf_edges(self, scalar_field, cube_idx, surf_cubes):
        occ_n = scalar_field < 0
        all_edges = cube_idx[surf_cubes][:, self.cube_edges.long()].reshape(-1, 2)
        unique_edges, _idx_map, counts = torch.unique(all_edges, dim=0, return_inverse=True, return_counts=True)
        unique_edges = unique_edges.long()

        mask_edges = occ_n[unique_edges.reshape(-1)].reshape(-1, 2).sum(-1) == 1

        surf_edges_mask = mask_edges[_idx_map]
        counts = counts[_idx_map]

        mapping = torch.ones((unique_edges.shape[0]), dtype=torch.long, device=self.device) * -1
        mapping[mask_edges] = torch.arange(mask_edges.sum(), device=self.device)

        idx_map = mapping[_idx_map]
        surf_edges = unique_edges[mask_edges]

        return surf_edges, idx_map, counts, surf_edges_mask

    def _compute_vd(self, voxelgrid_vertices, surf_cubes_fx8, surf_edges, scalar_field,
                    case_ids, beta, alpha, gamma_f, idx_map, qef_reg_scale, v_features, dtype, mode,
                    corner_gradients=None):
        """Compute dual vertices with optional QEF or weighted average."""

        # 1. Zero crossings
        surf_edges_x = voxelgrid_vertices[surf_edges.reshape(-1)].reshape(-1, 2, 3)
        surf_edges_s = scalar_field[surf_edges.reshape(-1)].reshape(-1, 2, 1)

        zero_crossing = self._linear_interp(surf_edges_s, surf_edges_x)

        # Compute per-crossing normals.
        # QEF mode: interpolate 3D corner gradients to each crossing point.
        #   This correctly handles deformed grids — the full gradient is estimated
        #   from ALL edges at each corner (not just the one surface edge direction).
        # Weighted-avg mode: fall back to the fast edge-direction approximation.
        if corner_gradients is not None:
            s0_e = scalar_field[surf_edges[:, 0]].to(dtype)
            s1_e = scalar_field[surf_edges[:, 1]].to(dtype)
            t = (s0_e / (s0_e - s1_e + 1e-8)).clamp(0, 1).unsqueeze(-1)  # [E, 1]
            g0 = corner_gradients[surf_edges[:, 0].long()]  # [E, 3]
            g1 = corner_gradients[surf_edges[:, 1].long()]  # [E, 3]

            # Confidence-weighted interpolation.
            # Corners with few incident edges have near-zero gradient magnitude
            # (regularisation collapses DtD → 0 solution).  Multiplying by the
            # gradient norm gives confidence: a weak corner contributes little,
            # so the crossing normal comes mostly from the well-constrained endpoint.
            #
            #   w0 = |g0| * (1-t)   — position × confidence for corner 0
            #   w1 = |g1| *   t     — position × confidence for corner 1
            #   g_cross = (w0*g0 + w1*g1) / (w0 + w1)
            #
            # Reduces to plain linear interp when both norms are equal; gives full
            # weight to the valid endpoint when the other has near-zero gradient.
            norm0 = g0.norm(dim=-1, keepdim=True)          # [E, 1]
            norm1 = g1.norm(dim=-1, keepdim=True)          # [E, 1]
            w0 = norm0 * (1 - t)                           # [E, 1]
            w1 = norm1 * t                                 # [E, 1]
            denom = w0 + w1                                # [E, 1]

            # Where denom is too small both corners lack gradient info → fall back
            # to the simple edge-direction approximation for those crossings only.
            has_grad = (denom > 1e-6).squeeze(-1)          # [E]
            g_cross = (w0 * g0 + w1 * g1) / denom.clamp(min=1e-8)

            if not has_grad.all():
                edge_fallback = self._compute_normals_from_gradient(
                    scalar_field, surf_edges, voxelgrid_vertices
                )
                g_cross = torch.where(has_grad.unsqueeze(-1), g_cross, edge_fallback)

            all_normals = F.normalize(g_cross, dim=-1)
        else:
            all_normals = self._compute_normals_from_gradient(scalar_field, surf_edges, voxelgrid_vertices)
        
        # Features
        if v_features is not None:
            C = v_features.shape[-1]
            surf_edges_f = v_features[surf_edges.reshape(-1)].reshape(-1, 2, C)
        else:
            C = None
            surf_edges_f = None
        
        # 2. Build edge groups
        idx_map = idx_map.reshape(-1, 12)
        num_vd = self.num_vd_table[case_ids]
        
        edge_group, edge_group_to_vd, edge_group_to_cube, vd_num_edges, vd_gamma = [], [], [], [], []
        
        total_num_vd = 0
        vd_idx_map = torch.zeros((case_ids.shape[0], 12), dtype=torch.long, device=self.device)

        unique_nums = torch.unique(num_vd)
        for num in unique_nums:
            cur_cubes = (num_vd == num)
            count = cur_cubes.sum()
            if count == 0: 
                continue
            
            curr_num_vd = count * num
            curr_edge_group = self.dmc_table[case_ids[cur_cubes], :num].reshape(-1, num * 7)
            
            curr_edge_group_to_vd = torch.arange(curr_num_vd, device=self.device).unsqueeze(-1).repeat(1, 7) + total_num_vd
            total_num_vd += curr_num_vd
            
            curr_edge_group_to_cube = torch.arange(idx_map.shape[0], device=self.device)[cur_cubes].unsqueeze(-1).repeat(1, num * 7).reshape_as(curr_edge_group)
            
            curr_mask = (curr_edge_group != -1)
            edge_group.append(torch.masked_select(curr_edge_group, curr_mask))
            edge_group_to_vd.append(torch.masked_select(curr_edge_group_to_vd.reshape_as(curr_edge_group), curr_mask))
            edge_group_to_cube.append(torch.masked_select(curr_edge_group_to_cube, curr_mask))
            vd_num_edges.append(curr_mask.reshape(-1, 7).sum(-1, keepdims=True))
            vd_gamma.append(torch.masked_select(gamma_f, cur_cubes).unsqueeze(-1).repeat(1, num).reshape(-1))

        if len(edge_group) == 0:
            empty_3 = torch.zeros((0, 3), device=self.device, dtype=dtype)
            empty_1 = torch.zeros((0,), device=self.device, dtype=dtype)
            return empty_3, empty_1, empty_1, vd_idx_map, None, zero_crossing, all_normals

        edge_group = torch.cat(edge_group)
        edge_group_to_vd = torch.cat(edge_group_to_vd)
        edge_group_to_cube = torch.cat(edge_group_to_cube)
        vd_num_edges = torch.cat(vd_num_edges)
        vd_gamma = torch.cat(vd_gamma)

        # Gather indices
        idx_group = torch.gather(idx_map.view(-1), 0, edge_group_to_cube * 12 + edge_group.long())
        
        zero_crossing_group = zero_crossing[idx_group.long()]
        s_group = surf_edges_s[idx_group.long()]
        x_group = surf_edges_x[idx_group.long()]
        n_group = all_normals[idx_group.long()]
        
        # Alpha-weighted crossing points (p_alpha)
        alpha_nx12x2 = alpha[:, self.cube_edges.long()].reshape(-1, 12, 2)
        alpha_group = alpha_nx12x2.view(-1, 2)[edge_group_to_cube * 12 + edge_group.long()].unsqueeze(-1)
        ue_group = self._linear_interp(s_group * alpha_group, x_group)
        
        # Beta
        beta_group = beta.view(-1)[edge_group_to_cube * 12 + edge_group.long()].unsqueeze(-1)
        
        # Compute dual vertices based on mode
        if mode == "qef":
            # Use per-edge normals directly (NOT averaged per-VD normals).
            # Sharp feature preservation requires distinct normals per crossing point:
            #   - smooth face: all normals in a VD are parallel → QEF is underdetermined
            #     → regularization pulls vertex to centroid (correct)
            #   - sharp edge/corner: crossings have distinct X/Y/Z normals → constraint
            #     planes intersect at the feature point → QEF solves to the sharp vertex
            # Averaging normals per VD (old _compute_normals_scatter approach) collapses
            # all constraints to the same direction, destroying sharp feature information.
            #
            # Also use zero_crossing_group (true linear interpolation) instead of
            # alpha-weighted ue_group — alpha is a training-time flexibility parameter
            # and shifts crossing positions away from the true zero crossing.
            # Compute cell geometry for each dual vertex.
            cell_verts = voxelgrid_vertices[surf_cubes_fx8.long()]  # [N_surf, 8, 3]
            cell_centers = cell_verts.mean(dim=1)       # [N_surf, 3]
            cell_mins = cell_verts.min(dim=1).values     # [N_surf, 3]
            cell_maxs = cell_verts.max(dim=1).values     # [N_surf, 3]
            # Map each VD to its owning cube
            vd_to_cube = torch.zeros(total_num_vd, dtype=torch.long, device=self.device)
            vd_to_cube.scatter_(0, edge_group_to_vd, edge_group_to_cube.long())
            vd_cell_center = cell_centers[vd_to_cube]
            vd_cell_min = cell_mins[vd_to_cube]
            vd_cell_max = cell_maxs[vd_to_cube]

            vd = self._solve_qef_scatter(
                zero_crossing_group, n_group, edge_group_to_vd, total_num_vd,
                qef_reg_scale, vd_cell_center=vd_cell_center,
                vd_cell_min=vd_cell_min, vd_cell_max=vd_cell_max,
            )
        else:
            # Weighted average (original FlexiCubes approach)
            vd = torch.zeros((total_num_vd, 3), device=self.device, dtype=dtype)
            beta_sum = torch.zeros((total_num_vd, 1), device=self.device, dtype=dtype)
            beta_sum = beta_sum.index_add_(0, edge_group_to_vd, beta_group)
            vd = vd.index_add_(0, edge_group_to_vd, ue_group * beta_group) / (beta_sum + 1e-8)
        
        # Feature interpolation
        if v_features is not None and surf_edges_f is not None:
            vd_features = torch.zeros((total_num_vd, C), device=self.device, dtype=dtype)
            f_group = surf_edges_f[idx_group.long()]
            uf_group = self._linear_interp(s_group * alpha_group, f_group)
            beta_sum = torch.zeros((total_num_vd, 1), device=self.device, dtype=dtype)
            beta_sum = beta_sum.index_add_(0, edge_group_to_vd, beta_group)
            vd_features = vd_features.index_add_(0, edge_group_to_vd, uf_group * beta_group) / (beta_sum + 1e-8)
        else:
            vd_features = None
        
        # L_dev loss
        L_dev = self._compute_reg_loss(vd, zero_crossing_group, edge_group_to_vd, vd_num_edges)
        
        # Update vd_idx_map
        v_idx = torch.arange(total_num_vd, dtype=torch.int64, device=self.device)
        index = edge_group_to_cube * 12 + edge_group.long()
        vd_idx_map.view(-1).scatter_(0, index, v_idx[edge_group_to_vd])
        
        return vd, L_dev, vd_gamma, vd_idx_map, vd_features, ue_group, n_group

    def _triangulate(self, scalar_field, surf_edges, vd, vd_gamma, edge_counts, idx_map,
                     vd_idx_map, surf_edges_mask, training, vd_features, dtype,
                     corner_gradients=None):
        """Aligned with official FlexiCubes _triangulate (no .int() casts, in-place flip)."""
        with torch.no_grad():
            group_mask = (edge_counts == 4) & surf_edges_mask
            group = idx_map.reshape(-1)[group_mask]
            vd_idx = vd_idx_map.reshape(-1)[group_mask]
            edge_indices, indices = torch.sort(group, stable=True)
            quad_vd_idx = vd_idx[indices].reshape(-1, 4)

            # Consistent winding (same as official)
            s_edges = scalar_field[surf_edges[edge_indices.reshape(-1, 4)[:, 0]].reshape(-1)].reshape(-1, 2)
            flip_mask = s_edges[:, 0] > 0
            quad_vd_idx[flip_mask] = quad_vd_idx[flip_mask][:, [0, 1, 3, 2]]
            quad_vd_idx[~flip_mask] = quad_vd_idx[~flip_mask][:, [2, 3, 1, 0]]

        # Gamma for quad split decision (aligned with official FlexiCubes)
        quad_gamma = torch.index_select(input=vd_gamma, index=quad_vd_idx.reshape(-1), dim=0).reshape(-1, 4)
        gamma_02 = quad_gamma[:, 0:1] * quad_gamma[:, 2:3]  # [N, 1]
        gamma_13 = quad_gamma[:, 1:2] * quad_gamma[:, 3:4]  # [N, 1]

        if not training:
            mask = (gamma_02 > gamma_13).squeeze(1)
            faces = torch.zeros((quad_gamma.shape[0], 6), dtype=torch.long, device=self.device)
            faces[mask] = quad_vd_idx[mask][:, self.quad_split_1]
            faces[~mask] = quad_vd_idx[~mask][:, self.quad_split_2]
            faces = faces.reshape(-1, 3)
        else:
            vd_quad = torch.index_select(input=vd, index=quad_vd_idx.reshape(-1), dim=0).reshape(-1, 4, 3)
            vd_02 = (vd_quad[:, 0:1] + vd_quad[:, 2:3]) / 2
            vd_13 = (vd_quad[:, 1:2] + vd_quad[:, 3:4]) / 2
            weight_sum = (gamma_02 + gamma_13) + 1e-8
            vd_center = ((vd_02 * gamma_02.unsqueeze(-1) + vd_13 * gamma_13.unsqueeze(-1)) /
                         weight_sum.unsqueeze(-1)).squeeze(1)

            if vd_features is not None:
                feat_quad = vd_features[quad_vd_idx.reshape(-1)].reshape(-1, 4, vd_features.shape[-1])
                feat_02 = (feat_quad[:, 0:1] + feat_quad[:, 2:3]) / 2
                feat_13 = (feat_quad[:, 1:2] + feat_quad[:, 3:4]) / 2
                feat_center = ((feat_02 * gamma_02.unsqueeze(-1) + feat_13 * gamma_13.unsqueeze(-1)) /
                               weight_sum.unsqueeze(-1)).squeeze(1)
                vd_features = torch.cat([vd_features, feat_center])

            vd_center_idx = torch.arange(vd_center.shape[0], device=self.device) + vd.shape[0]
            vd = torch.cat([vd, vd_center])

            faces = quad_vd_idx[:, self.quad_split_train].reshape(-1, 4, 2)
            faces = torch.cat([faces, vd_center_idx.reshape(-1, 1, 1).repeat(1, 4, 1)], -1).reshape(-1, 3)

        return vd, faces, vd_features

    def _estimate_vd_normals(self, vd, corner_gradients, surf_edges, edge_counts,
                              idx_map, vd_idx_map, surf_edges_mask, scalar_field):
        """Estimate per-dual-vertex normals by interpolating corner gradients at crossing points."""
        num_vd = vd.shape[0]
        device = vd.device
        dtype = vd.dtype

        # Interpolate corner gradients to each surface edge crossing
        s0 = scalar_field[surf_edges[:, 0]].to(dtype)
        s1 = scalar_field[surf_edges[:, 1]].to(dtype)
        t = (s0 / (s0 - s1 + 1e-8)).clamp(0, 1).unsqueeze(-1)  # [E, 1]
        g0 = corner_gradients[surf_edges[:, 0].long()]
        g1 = corner_gradients[surf_edges[:, 1].long()]
        edge_normals = F.normalize((1 - t) * g0 + t * g1, dim=-1)  # [E, 3]

        # Map edges to dual vertices via the same group structure used in _compute_vd
        # Reuse vd_idx_map: for each (cube, edge) → dual vertex index
        group_mask = (edge_counts == 4) & surf_edges_mask
        group = idx_map.reshape(-1)[group_mask]
        vd_idx = vd_idx_map.reshape(-1)[group_mask]

        # Average normals per dual vertex
        vd_normals = torch.zeros(num_vd, 3, device=device, dtype=dtype)
        vd_counts = torch.zeros(num_vd, 1, device=device, dtype=dtype)
        edge_n = edge_normals[group.long()]
        vd_normals.scatter_add_(0, vd_idx.long().unsqueeze(1).expand(-1, 3), edge_n)
        vd_counts.scatter_add_(0, vd_idx.long().unsqueeze(1), torch.ones(vd_idx.shape[0], 1, device=device, dtype=dtype))
        vd_normals = vd_normals / vd_counts.clamp(min=1)

        return vd_normals

    def _linear_interp(self, edges_weight, edges_x):
        """Linear interpolation for zero-crossing computation."""
        edge_dim = edges_weight.dim() - 2
        assert edges_weight.shape[edge_dim] == 2
        
        edges_weight = torch.cat([
            torch.index_select(input=edges_weight, index=torch.tensor(1, device=self.device), dim=edge_dim), 
            -torch.index_select(input=edges_weight, index=torch.tensor(0, device=self.device), dim=edge_dim)
        ], edge_dim)
        
        denominator = edges_weight.sum(edge_dim)
        ue = (edges_x * edges_weight).sum(edge_dim) / denominator
        return ue
