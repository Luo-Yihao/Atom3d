"""
Sparse Flood Fill: CUDA-accelerated flood fill for mesh voxelization.
"""

from typing import Tuple, Dict, List, Optional
import logging
import torch
import numpy as np

from ..core.mesh_bvh import MeshBVH
from ..grid.octree_indexer import OctreeIndexer
from ..constants import (
    CUBE_CORNERS, 
    NEIGHBOR_OFFSETS_6, 
    NEIGHBOR_OFFSETS_26, 
    EDGE_NEIGHBOR_OFFSETS, 
    CORNER_NEIGHBOR_OFFSETS,
    SIGN_WATER,
    SIGN_DRY,
    SIGN_DAM_DEFAULT
)

# Module logger
logger = logging.getLogger(__name__)

# Import CUDA flood fill kernels
try:
    from ..kernels.flood_fill import flood_fill_3d_sparse, floodfill_available
    HAS_CUDA_FLOODFILL = floodfill_available()
except ImportError:
    HAS_CUDA_FLOODFILL = False


def get_dam_neighborhood_signs(
    dam_ijk: torch.Tensor,
    water_ijk: torch.Tensor,
    dry_ijk: torch.Tensor,
    resolution: int,
    k_ring: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get k-ring neighbors of the Dam and determine their signs using centralized constants.
    """
    if len(dam_ijk) == 0:
        return (torch.empty(0, 3, dtype=torch.int32, device=dam_ijk.device),
                torch.empty(0, dtype=torch.int8, device=dam_ijk.device))
        
    device = dam_ijk.device
    R = resolution
    
    # Use centralized 26-neighbor offsets
    offsets_26 = NEIGHBOR_OFFSETS_26.to(device)
    
    # Start with Dam cells
    current_frontier = dam_ijk.long()
    all_neighbors = dam_ijk.long()
    
    # Iteratively expand k times
    for ring in range(k_ring):
        # Expand frontier by 26-neighbors
        neighbors = current_frontier.unsqueeze(1) + offsets_26.unsqueeze(0)  # [N, 26, 3]
        flat_nb = neighbors.reshape(-1, 3)
        
        # Bounds check
        valid = (flat_nb >= 0) & (flat_nb < R)
        flat_nb = flat_nb[valid.all(dim=1)]
        
        # Add to all neighbors
        all_neighbors = torch.cat([all_neighbors, flat_nb], dim=0)
        all_neighbors = torch.unique(all_neighbors, dim=0)
        
        if ring < k_ring - 1:
            current_frontier = all_neighbors
    
    neighbor_ijk = all_neighbors
    neighbor_lin = neighbor_ijk[:,0]*R*R + neighbor_ijk[:,1]*R + neighbor_ijk[:,2]
    
    K = len(neighbor_ijk)
    signs = torch.full((K,), SIGN_WATER, dtype=torch.int8, device=device)
    
    def check_membership(subset_ijk):
        if len(subset_ijk) == 0:
            return torch.zeros(K, dtype=torch.bool, device=device)
        subset_lin = subset_ijk[:,0].long()*R*R + subset_ijk[:,1].long()*R + subset_ijk[:,2].long()
        subset_sorted, _ = torch.sort(subset_lin)
        idx = torch.searchsorted(subset_sorted, neighbor_lin)
        idx = torch.clamp(idx, 0, len(subset_sorted)-1)
        return (subset_sorted[idx] == neighbor_lin)
        
    is_dam = check_membership(dam_ijk)
    is_water = check_membership(water_ijk)
    is_dry = check_membership(dry_ijk)
        
    # Assign Signs (Priority: Dam > Dry > Water)
    # Note: Dam cells get 0 here, will be converted to SIGN_DAM_DEFAULT in dual graph.
    signs[is_dry] = SIGN_DRY
    signs[is_water] = SIGN_WATER
    signs[is_dam] = 0 
    
    return neighbor_ijk.int(), signs


def sparse_flood_fill(
    bvh: MeshBVH,
    resolution: int,
    source: Tuple[int, int, int] = (0, 0, 0),
    connectivity: int = 26,
    min_level: int = 0,
    k_ring: int = 1,
    verbose: bool = True
) -> Dict[str, torch.Tensor]:
    """
    CUDA-accelerated sparse flood fill for mesh inside/outside classification.

    WARNING: connectivity=26 can leak through thin walls (1-cell thick).
    The octree voxelization may miss diagonal cells at triangle edges due to
    floating-point precision, allowing water to flow diagonally past dam cells.
    Use connectivity=6 for reliable inside/outside classification on thin-wall meshes.
    """
    if not HAS_CUDA_FLOODFILL:
        raise RuntimeError("CUDA flood fill not available.")
    
    from ..kernels.flood_fill import flood_fill_3d_sparse
    import time
    
    device = bvh.device
    max_level = int(np.log2(resolution))
    octree = OctreeIndexer(max_level=max_level, device=device)
    
    # 1. Intersected voxels
    all_intersected_ijk = octree.octree_traverse(bvh, min_level=min_level)
    
    # 2. Flood fill
    t0 = time.time()
    water_coords, dry_coords, collision_coords, true_dam_coords, _, _ = flood_fill_3d_sparse(
        all_intersected_ijk, resolution, source, padding=10, connectivity=connectivity
    )
    cuda_time = time.time() - t0
    
    # 3. k-ring Neighborhood
    t1 = time.time()
    neighbor_ijk, neighborhood_signs = get_dam_neighborhood_signs(
        true_dam_coords, water_coords, dry_coords, resolution, k_ring=k_ring
    )
    
    extended_tide_ijk = neighbor_ijk[neighborhood_signs == SIGN_WATER]
    dry_adjacent_ijk = neighbor_ijk[neighborhood_signs == SIGN_DRY]
    neighborhood_time = time.time() - t1

    if verbose:
        logger.info(f"Sparse Flood Fill completed in {cuda_time+neighborhood_time:.4f}s")
    
    return {
        'dam_ijk': true_dam_coords,
        'water_ijk': water_coords,
        'dry_ijk': dry_coords,
        'tide_ijk': collision_coords,
        'extended_tide_ijk': extended_tide_ijk,
        'dry_adjacent_ijk': dry_adjacent_ijk
    }


def get_dam_wet_faces(dam_ijk: torch.Tensor, collision_ijk: torch.Tensor, resolution: int) -> torch.Tensor:
    if len(dam_ijk) == 0: return torch.zeros((0, 6), dtype=torch.bool, device=dam_ijk.device)
    device = dam_ijk.device
    N = dam_ijk.shape[0]
    collision_mask = torch.zeros(resolution**3, dtype=torch.bool, device=device)
    if len(collision_ijk) > 0:
        c_idx = (collision_ijk[:,0].long()*resolution*resolution + collision_ijk[:,1].long()*resolution + collision_ijk[:,2].long())
        collision_mask[c_idx] = True
    offsets = NEIGHBOR_OFFSETS_6.to(device)
    neighbors = dam_ijk.unsqueeze(1) + offsets.unsqueeze(0)
    flat_nb = neighbors.reshape(-1, 3).long()
    in_bounds = (flat_nb >= 0) & (flat_nb < resolution)
    valid_mask = in_bounds.all(dim=1)
    is_wet_flat = torch.zeros(N*6, dtype=torch.bool, device=device)
    if valid_mask.any():
        v_lin = (flat_nb[valid_mask,0]*resolution*resolution + flat_nb[valid_mask,1]*resolution + flat_nb[valid_mask,2])
        is_wet_flat[valid_mask] = collision_mask[v_lin]
    return is_wet_flat.view(N, 6)


def get_dam_wet_edges(dam_ijk: torch.Tensor, collision_ijk: torch.Tensor, resolution: int) -> torch.Tensor:
    if len(dam_ijk) == 0: return torch.zeros((0, 12), dtype=torch.bool, device=dam_ijk.device)
    device = dam_ijk.device
    N = dam_ijk.shape[0]
    collision_mask = torch.zeros(resolution**3, dtype=torch.bool, device=device)
    if len(collision_ijk) > 0:
        c_idx = (collision_ijk[:,0].long()*resolution*resolution + collision_ijk[:,1].long()*resolution + collision_ijk[:,2].long())
        collision_mask[c_idx] = True
    offsets = EDGE_NEIGHBOR_OFFSETS.to(device)
    neighbors = dam_ijk.unsqueeze(1).unsqueeze(1) + offsets.unsqueeze(0)
    flat_nb = neighbors.reshape(-1, 3).long()
    in_bounds = (flat_nb >= 0) & (flat_nb < resolution)
    valid_mask = in_bounds.all(dim=1)
    is_water = torch.zeros(N*12*3, dtype=torch.bool, device=device)
    if valid_mask.any():
        v_lin = (flat_nb[valid_mask,0]*resolution*resolution + flat_nb[valid_mask,1]*resolution + flat_nb[valid_mask,2])
        is_water[valid_mask] = collision_mask[v_lin]
    return is_water.view(N, 12, 3).any(dim=2)


def get_dam_wet_corners(dam_ijk: torch.Tensor, collision_ijk: torch.Tensor, resolution: int) -> torch.Tensor:
    if len(dam_ijk) == 0: return torch.zeros((0, 8), dtype=torch.bool, device=dam_ijk.device)
    device = dam_ijk.device
    N = dam_ijk.shape[0]
    collision_mask = torch.zeros(resolution**3, dtype=torch.bool, device=device)
    if len(collision_ijk) > 0:
        c_idx = (collision_ijk[:,0].long()*resolution*resolution + collision_ijk[:,1].long()*resolution + collision_ijk[:,2].long())
        collision_mask[c_idx] = True
    offsets = CORNER_NEIGHBOR_OFFSETS.to(device)
    neighbors = dam_ijk.unsqueeze(1).unsqueeze(1) + offsets.unsqueeze(0)
    flat_nb = neighbors.reshape(-1, 3).long()
    in_bounds = (flat_nb >= 0) & (flat_nb < resolution)
    valid_mask = in_bounds.all(dim=1)
    is_water = torch.zeros(N*8*7, dtype=torch.bool, device=device)
    if valid_mask.any():
        v_lin = (flat_nb[valid_mask,0]*resolution*resolution + flat_nb[valid_mask,1]*resolution + flat_nb[valid_mask,2])
        is_water[valid_mask] = collision_mask[v_lin]
    return is_water.view(N, 8, 7).any(dim=2)


def construct_surface_dual_graph(
    dam_ijk: torch.Tensor,
    collision_ijk: torch.Tensor,
    dry_adjacent_ijk: torch.Tensor,
    octree: OctreeIndexer,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Construct surface dual graph — pure topology, no sign logic.
    
    Returns:
        voxelgrid_vertices: [V, 3] world-space vertex positions
        vertices_ijk: [V, 3] integer grid coordinates of vertices
        cube_idx: [N, 8] indices into vertex array for each cube's 8 corners
        adj_idx: [N, 6] indices into cell array for each cube's 6 face-neighbors
        active_cells_ijk: [N, 3] integer coordinates of active cells
    """
    if dry_adjacent_ijk is None:
        dry_adjacent_ijk = torch.empty((0, 3), dtype=torch.int32, device=dam_ijk.device if len(dam_ijk)>0 else 'cuda')
    device = dam_ijk.device if len(dam_ijk) > 0 else (collision_ijk.device if len(collision_ijk) > 0 else torch.device('cuda'))
    
    R = 2 ** octree.max_level
    all_cells = [dam_ijk, collision_ijk, dry_adjacent_ijk]
    active_cells_ijk = torch.cat([c for c in all_cells if len(c)>0], dim=0)
    active_cells_ijk = torch.unique(active_cells_ijk, dim=0)
    num_cells = len(active_cells_ijk)

    cells_flat = (active_cells_ijk[:,0].long() * R * R + active_cells_ijk[:,1].long() * R + active_cells_ijk[:,2].long())
    cells_sorted, cells_perm = torch.sort(cells_flat)

    off_v = CUBE_CORNERS.to(device)
    vertex_cand = active_cells_ijk.unsqueeze(1) + off_v.unsqueeze(0)
    vertices_ijk = torch.unique(vertex_cand.view(-1, 3), dim=0)
    num_verts = len(vertices_ijk)
    
    R_v = R + 1
    verts_flat = (vertices_ijk[:,0].long() * R_v * R_v + vertices_ijk[:,1].long() * R_v + vertices_ijk[:,2].long())
    verts_sorted, verts_perm = torch.sort(verts_flat)

    cube_idx = torch.full((num_cells, 8), -1, dtype=torch.long, device=device)
    if num_cells > 0:
        c_flat = (vertex_cand[...,0].long() * R_v * R_v + vertex_cand[...,1].long() * R_v + vertex_cand[...,2].long())
        idx_in_sorted = torch.searchsorted(verts_sorted, c_flat.view(-1))
        idx_in_sorted = torch.clamp(idx_in_sorted, 0, num_verts - 1)
        found = (verts_sorted[idx_in_sorted] == c_flat.view(-1))
        cube_idx.view(-1)[found] = verts_perm[idx_in_sorted[found]]
    
    adj_idx = torch.full((num_cells, 6), -1, dtype=torch.long, device=device)
    if num_cells > 0:
        off_adj = NEIGHBOR_OFFSETS_6.to(device)
        adj_flat = ((active_cells_ijk.unsqueeze(1) + off_adj.unsqueeze(0))[...,0].long() * R * R + 
                    (active_cells_ijk.unsqueeze(1) + off_adj.unsqueeze(0))[...,1].long() * R + 
                    (active_cells_ijk.unsqueeze(1) + off_adj.unsqueeze(0))[...,2].long())
        idx_in_sorted = torch.searchsorted(cells_sorted, adj_flat.view(-1))
        idx_in_sorted = torch.clamp(idx_in_sorted, 0, num_cells - 1)
        found = (cells_sorted[idx_in_sorted] == adj_flat.view(-1))
        adj_idx.view(-1)[found] = cells_perm[idx_in_sorted[found]]

    # --- World coordinates ---
    cmin = octree.bounds[0]
    cell_size = (octree.bounds[1] - octree.bounds[0]) / float(R)
    voxelgrid_vertices = cmin + vertices_ijk.float() * cell_size

    return voxelgrid_vertices, vertices_ijk, cube_idx, adj_idx, active_cells_ijk


def reconstruct_dense_from_sparse(dam_ijk: torch.Tensor, resolution: int, connectivity: int = 6, device: str = 'cuda') -> Dict[str, torch.Tensor]:
    from scipy import ndimage
    R = resolution
    can_flood = np.ones((R, R, R), dtype=bool)
    if dam_ijk.numel() > 0:
        dam_np = dam_ijk.cpu().numpy().astype(np.int64)
        valid = (dam_np >= 0).all(axis=1) & (dam_np < R).all(axis=1)
        dam_np = dam_np[valid]
        can_flood[dam_np[:, 0], dam_np[:, 1], dam_np[:, 2]] = False
    struct = ndimage.generate_binary_structure(3, 1 if connectivity == 6 else 3)
    labels, num_labels = ndimage.label(can_flood, structure=struct)
    boundary_labels = set(labels[[0,-1],:,:].flatten()) | set(labels[:,[0,-1],:].flatten()) | set(labels[:,:,[0,-1]].flatten())
    boundary_labels.discard(0)
    is_exterior = np.isin(labels, list(boundary_labels))
    water_mask = is_exterior & can_flood
    dry_mask = ~is_exterior & can_flood
    dam_mask = ~can_flood
    dam_dilated = ndimage.binary_dilation(dam_mask, structure=struct)
    dry_collision_mask = dry_mask & dam_dilated
    return {
        'dry_ijk': torch.tensor(np.argwhere(dry_mask), dtype=torch.int32, device=device),
        'water_ijk': torch.tensor(np.argwhere(water_mask), dtype=torch.int32, device=device),
        'dry_adjacent_ijk': torch.tensor(np.argwhere(dry_collision_mask), dtype=torch.int32, device=device),
        'dry_mask': dry_mask, 'water_mask': water_mask
    }
