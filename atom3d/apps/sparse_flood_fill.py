"""
Sparse Flood Fill: CUDA-accelerated flood fill for mesh voxelization.
"""

from typing import Tuple, Dict
import logging
import torch
import numpy as np

from ..core.mesh_bvh import MeshBVH
from ..grid.octree_indexer import OctreeIndexer

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
    Get k-ring neighbors of the Dam and determine their signs.
    Sparse implementation using iterative expansion.
    
    signs logic:
        0: Dam
        1: Water (Present in the provided water_ijk set)
       -1: Dry (Present in the provided dry_ijk set)
       
    Args:
        dam_ijk: [N, 3] Dam voxels
        water_ijk: [M, 3] Global Water voxels (defined by connectivity)
        dry_ijk: [D, 3] Global Dry voxels (interior from flood fill)
        resolution: Grid resolution
        k_ring: Number of expansion rings (default=1 for 26-connected)
        
    Returns:
        (neighbor_ijk, signs)
    """
    if len(dam_ijk) == 0:
        return (torch.empty(0, 3, dtype=torch.int32, device=dam_ijk.device),
                torch.empty(0, dtype=torch.int8, device=dam_ijk.device))
        
    device = dam_ijk.device
    R = resolution
    
    # Generate 26-neighbor offsets
    offsets_26 = []
    for dz in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                offsets_26.append((dx, dy, dz))
    offsets_26 = torch.tensor(offsets_26, dtype=torch.int32, device=device)
    
    # Start with Dam cells
    current_frontier = dam_ijk.long()
    all_neighbors = dam_ijk.long()
    
    # Iteratively expand k times
    for ring in range(k_ring):
        # Expand frontier by 26-neighbors
        neighbors = current_frontier.unsqueeze(1) + offsets_26.unsqueeze(0)  # [N, 27, 3]
        flat_nb = neighbors.reshape(-1, 3)
        
        # Bounds check
        valid = (flat_nb >= 0) & (flat_nb < R)
        flat_nb = flat_nb[valid.all(dim=1)]
        
        # Get unique new neighbors
        new_neighbors = torch.unique(flat_nb, dim=0)
        
        # Add to all neighbors
        all_neighbors = torch.cat([all_neighbors, new_neighbors], dim=0)
        all_neighbors = torch.unique(all_neighbors, dim=0)
        
        # New frontier for next iteration (exclude Dam which is already processed)
        if ring < k_ring - 1:
            # Frontier = new cells not in previous all_neighbors
            current_frontier = new_neighbors
    
    # Get unique neighbor coordinates
    neighbor_ijk = all_neighbors
    neighbor_lin = neighbor_ijk[:,0]*R*R + neighbor_ijk[:,1]*R + neighbor_ijk[:,2]
    
    K = len(neighbor_ijk)
    # Default to Water/Exterior (+1) for unmarked cells
    signs = torch.full((K,), 1, dtype=torch.int8, device=device)
    
    # Check membership using Search Sorted (Sparse)
    def check_membership(subset_ijk):
        if len(subset_ijk) == 0:
            return torch.zeros(K, dtype=torch.bool, device=device)
        
        subset_lin = subset_ijk[:,0].long()*R*R + subset_ijk[:,1].long()*R + subset_ijk[:,2].long()
        subset_sorted, _ = torch.sort(subset_lin)
        
        idx = torch.searchsorted(subset_sorted, neighbor_lin)
        idx = torch.clamp(idx, 0, len(subset_sorted)-1)
        found = (subset_sorted[idx] == neighbor_lin)
        return found
        
    is_dam = check_membership(dam_ijk)
    is_water = check_membership(water_ijk)
    is_dry = check_membership(dry_ijk)
        
    # Assign Signs
    # Priority: Dam > Dry > Water
    signs[is_dry] = -1  # Dry = interior
    signs[is_water] = 1  # Water = exterior (overwrites)
    signs[is_dam] = 0    # Dam = surface (highest priority)
    
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
    
    Args:
        bvh: MeshBVH object
        resolution: Grid resolution (power of 2)
        source: Flood source point
        connectivity: Flood connectivity (6 or 26)
        min_level: Minimum octree level
        k_ring: Number of neighborhood expansion rings (1=26-connected, 2=2-ring, etc.)
        verbose: Print logging info
        
    Returns:
        {
            'dam_ijk': [N, 3] Surface voxels
            'dry_ijk': [D, 3] Full interior voxels
            'tide_ijk': [C, 3] Water voxels adjacent to Dam (Connectivity specific)
            'extended_tide_ijk': [E, 3] Water voxels in k-ring Neighborhood
            'dry_adjacent_ijk': [A, 3] Interior voxels in k-ring Neighborhood
        }
    """
    if not HAS_CUDA_FLOODFILL:
        raise RuntimeError(
            "CUDA flood fill not available. Install pytorch-floodfill-3d:\n"
            "  pip install pytorch-floodfill-3d"
        )
    
    from ..kernels.flood_fill import flood_fill_3d_sparse
    import time
    
    device = bvh.device
    max_level = int(np.log2(resolution))

    octree = OctreeIndexer(max_level=max_level, device=device)
    
    # Step 1: Find all mesh-intersecting voxels (potential dams)
    all_intersected_ijk = octree.octree_traverse(bvh, min_level=min_level)
    
    # Step 2: Run CUDA sparse flood fill kernel
    t0 = time.time()
    water_coords, dry_coords, collision_coords, true_dam_coords, _, _ = flood_fill_3d_sparse(
        all_intersected_ijk, resolution, source, padding=10, connectivity=connectivity
    )
    cuda_time = time.time() - t0
    
    # Step 3: Compute Full k-ring Neighborhood Partition
    # Goal: Dam + ExtendedTide + DryAdjacent = Full k-ring Neighborhood(Dam)
    # Signs determined by Global Water/Dry Sets which respect flood fill results
    t1 = time.time()
    
    neighborhood_ijk, neighborhood_signs = get_dam_neighborhood_signs(
        true_dam_coords, water_coords, dry_coords, resolution, k_ring=k_ring
    )
    
    extended_tide_ijk = neighborhood_ijk[neighborhood_signs == 1]
    dry_adjacent_ijk = neighborhood_ijk[neighborhood_signs == -1]
    
    neighborhood_time = time.time() - t1

    if verbose:
        logger.info(
            "Sparse Flood Fill completed in %.4fs (Flood) + %.4fs (Neighborhood):\n"
            "  Resolution:      %d³\n"
            "  k_ring:          %d\n"
            "  Intersected:     %d voxels\n"
            "  True dam:        %d voxels\n"
            "  Tide (Raw):      %d voxels\n"
            "  Ext. Tide (Wet): %d voxels\n"
            "  Dry Adj (Dry):   %d voxels",
            cuda_time, neighborhood_time, resolution, k_ring, len(all_intersected_ijk),
            len(true_dam_coords), len(collision_coords), 
            len(extended_tide_ijk), len(dry_adjacent_ijk)
        )
    
    return {
        'dam_ijk': true_dam_coords,       # Surface
        'tide_ijk': collision_coords,     # Water (Connectivity-Specific, Raw)
        # 'dry_ijk': dry_coords,            # Full Interior (All voxels inside Dam)
        'extended_tide_ijk': extended_tide_ijk, # Water (Subset in 26-Neighborhood)
        'dry_adjacent_ijk': dry_adjacent_ijk    # Dry (26-Neighborhood only)
    }




def get_dam_wet_faces(
    dam_ijk: torch.Tensor,
    collision_ijk: torch.Tensor,
    resolution: int
) -> torch.Tensor:
    """
    For each dam voxel, determine which of its 6 faces are wet (adjacent to water).
    Vectorized implementation.
    """
    if len(dam_ijk) == 0:
        return torch.zeros((0, 6), dtype=torch.bool, device=dam_ijk.device)
        
    device = dam_ijk.device
    N = dam_ijk.shape[0]
    
    # Enable dense lookup for speed (O(1) batched query)
    # 512^3 requires ~134MB for bool mask, fitting in GPU memory.
    # Use uint8 for better compatibility with some ops if needed, but bool is fine.
    collision_mask = torch.zeros(resolution**3, dtype=torch.bool, device=device)
    if len(collision_ijk) > 0:
        c_idx = (collision_ijk[:, 0].long() * resolution * resolution + 
                 collision_ijk[:, 1].long() * resolution + 
                 collision_ijk[:, 2].long())
        collision_mask[c_idx] = True
    
    # 6 face offsets: [+x, -x, +y, -y, +z, -z]
    offsets = torch.tensor([
        [1, 0, 0], [-1, 0, 0], 
        [0, 1, 0], [0, -1, 0], 
        [0, 0, 1], [0, 0, -1],
    ], dtype=torch.int32, device=device) # [6, 3]
    
    # Broadcast add: [N, 1, 3] + [1, 6, 3] -> [N, 6, 3]
    neighbors = dam_ijk.unsqueeze(1) + offsets.unsqueeze(0)
    
    # Flatten neighbor coords: [N*6, 3]
    flat_nb = neighbors.reshape(-1, 3).long()
    
    # Bounds check
    in_bounds = (flat_nb >= 0) & (flat_nb < resolution)
    valid_mask = in_bounds.all(dim=1) # [N*6]
    
    # Result buffer
    is_wet_flat = torch.zeros(N*6, dtype=torch.bool, device=device)
    
    # Only query valid neighbors
    if valid_mask.any():
        valid_nb = flat_nb[valid_mask]
        valid_linear = (valid_nb[:, 0] * resolution * resolution + 
                        valid_nb[:, 1] * resolution + 
                        valid_nb[:, 2])
        is_wet_flat[valid_mask] = collision_mask[valid_linear]
        
    return is_wet_flat.view(N, 6)



def get_dam_wet_edges(
    dam_ijk: torch.Tensor,
    collision_ijk: torch.Tensor,
    resolution: int
) -> torch.Tensor:
    """
    For each dam voxel, determine which of its 12 edges are wet.
    Geometric definition: An edge is wet if ANY of the 3 adjacent neighbor voxels (sharing that edge) are water.
    Vectorized implementation.
    """
    if len(dam_ijk) == 0:
        return torch.zeros((0, 12), dtype=torch.bool, device=dam_ijk.device)

    device = dam_ijk.device
    N = dam_ijk.shape[0]
    
    collision_mask = torch.zeros(resolution**3, dtype=torch.bool, device=device)
    if len(collision_ijk) > 0:
        c_idx = (collision_ijk[:, 0].long() * resolution * resolution + 
                 collision_ijk[:, 1].long() * resolution + 
                 collision_ijk[:, 2].long())
        collision_mask[c_idx] = True
    
    # 12 edges. Each has 3 neighbor voxels sharing it.
    # Order: 0-3 (Z-aligned), 4-7 (Y-aligned), 8-11 (X-aligned)
    # Z-aligned: neighbors along X and Y axes
    # Edge 0: (+x, +y). Neighbors: (+1,0,0), (0,+1,0), (+1,+1,0)
    offsets_list = [
        # Z-aligned (vary x, y)
        [[1,0,0], [0,1,0], [1,1,0]],     # 0: (+x,+y)
        [[1,0,0], [0,-1,0], [1,-1,0]],   # 1: (+x,-y)
        [[-1,0,0], [0,1,0], [-1,1,0]],   # 2: (-x,+y)
        [[-1,0,0], [0,-1,0], [-1,-1,0]], # 3: (-x,-y)
        
        # Y-aligned (vary x, z)
        [[1,0,0], [0,0,1], [1,0,1]],     # 4: (+x,+z)
        [[1,0,0], [0,0,-1], [1,0,-1]],   # 5: (+x,-z)
        [[-1,0,0], [0,0,1], [-1,0,1]],   # 6: (-x,+z)
        [[-1,0,0], [0,0,-1], [-1,0,-1]], # 7: (-x,-z)
        
        # X-aligned (vary y, z)
        [[0,1,0], [0,0,1], [0,1,1]],     # 8: (+y,+z)
        [[0,1,0], [0,0,-1], [0,1,-1]],   # 9: (+y,-z)
        [[0,-1,0], [0,0,1], [0,-1,1]],   # 10: (-y,+z)
        [[0,-1,0], [0,0,-1], [0,-1,-1]], # 11: (-y,-z)
    ]
    
    # Shape: [12, 3, 3]
    offsets = torch.tensor(offsets_list, dtype=torch.int32, device=device)
    
    # Neighbor Expansion: [N, 1, 1, 3] + [1, 12, 3, 3] -> [N, 12, 3, 3]
    neighbors = dam_ijk.unsqueeze(1).unsqueeze(1) + offsets.unsqueeze(0)
    
    # Flatten to [N * 12 * 3, 3] for linear query
    flat_nb = neighbors.reshape(-1, 3).long()
    
    in_bounds = (flat_nb >= 0) & (flat_nb < resolution)
    valid_mask = in_bounds.all(dim=1)
    
    is_water = torch.zeros(N*12*3, dtype=torch.bool, device=device)
    if valid_mask.any():
        valid_nb = flat_nb[valid_mask]
        valid_linear = (valid_nb[:, 0] * resolution * resolution + 
                        valid_nb[:, 1] * resolution + 
                        valid_nb[:, 2])
        is_water[valid_mask] = collision_mask[valid_linear]
    
    # Reshape to [N, 12, 3] and aggregate check (ANY neighbor is water)
    is_water_grouped = is_water.view(N, 12, 3)
    wet_edges = is_water_grouped.any(dim=2)
    
    return wet_edges


def get_dam_wet_corners(
    dam_ijk: torch.Tensor,
    collision_ijk: torch.Tensor,
    resolution: int
) -> torch.Tensor:
    """
    For each dam voxel, determine which of its 8 corners are wet.
    Geometric definition: A corner is wet if ANY of the 7 adjacent neighbor voxels (sharing that corner) are water.
    Vectorized implementation.
    """
    if len(dam_ijk) == 0:
        return torch.zeros((0, 8), dtype=torch.bool, device=dam_ijk.device)

    device = dam_ijk.device
    N = dam_ijk.shape[0]
    
    collision_mask = torch.zeros(resolution**3, dtype=torch.bool, device=device)
    if len(collision_ijk) > 0:
        c_idx = (collision_ijk[:, 0].long() * resolution * resolution + 
                 collision_ijk[:, 1].long() * resolution + 
                 collision_ijk[:, 2].long())
        collision_mask[c_idx] = True
    
    # 8 corners. Each has 7 neighbor voxels sharing it.
    # Corner order:
    # 0: (-,-,-), 1: (+,-,-), 2: (-,+,-), 3: (+,+,-)
    # 4: (-,-,+), 5: (+,-,+), 6: (-,+,+), 7: (+,+,+)
    
    corners_signs = [
        (-1,-1,-1), (1,-1,-1), (-1,1,-1), (1,1,-1),
        (-1,-1, 1), (1,-1, 1), (-1,1, 1), (1,1, 1)
    ]
    
    offsets_list = []
    for (sx, sy, sz) in corners_signs:
        # 7 neighbors composed of combinations of sx, sy, sz
        # (sx,0,0), (0,sy,0), (0,0,sz) -> Faces
        # (sx,sy,0), (sx,0,sz), (0,sy,sz) -> Edges
        # (sx,sy,sz) -> Corner
        corner_group = [
            [sx, 0, 0], [0, sy, 0], [0, 0, sz],
            [sx, sy, 0], [sx, 0, sz], [0, sy, sz],
            [sx, sy, sz]
        ]
        offsets_list.append(corner_group)
        
    # Shape: [8, 7, 3]
    offsets = torch.tensor(offsets_list, dtype=torch.int32, device=device)
    
    # Neighbor Expansion: [N, 1, 1, 3] + [1, 8, 7, 3] -> [N, 8, 7, 3]
    neighbors = dam_ijk.unsqueeze(1).unsqueeze(1) + offsets.unsqueeze(0)
    
    flat_nb = neighbors.reshape(-1, 3).long()
    
    in_bounds = (flat_nb >= 0) & (flat_nb < resolution)
    valid_mask = in_bounds.all(dim=1)
    
    is_water = torch.zeros(N*8*7, dtype=torch.bool, device=device)
    if valid_mask.any():
        valid_nb = flat_nb[valid_mask]
        valid_linear = (valid_nb[:, 0] * resolution * resolution + 
                        valid_nb[:, 1] * resolution + 
                        valid_nb[:, 2])
        is_water[valid_mask] = collision_mask[valid_linear]
        
    # Reshape to [N, 8, 7] and aggregate
    is_water_grouped = is_water.view(N, 8, 7)
    wet_corners = is_water_grouped.any(dim=2)
    
    return wet_corners


def construct_surface_dual_graph(
    dam_ijk: torch.Tensor,
    collision_ijk: torch.Tensor,
    dry_adjacent_ijk: torch.Tensor,
    octree: OctreeIndexer,
    full_dry_ijk: torch.Tensor = None,
    full_water_ijk: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Construct Surface Dual Graph (Primal Cube Connectivity) for Sparse FlexiCubes.
    
    Active cells are limited to Dam + neighborhood (sparse).
    Vertex types are determined using FULL classification sets for correctness.
    
    Args:
        dam_ijk: [N, 3] int32 - Dam voxels (surface)
        collision_ijk: [M, 3] int32 - Water voxels in 26-neighborhood of Dam (for active cells)
        octree: OctreeIndexer - Context data.
        dry_adjacent_ijk: [D, 3] int32 - Dry voxels in 26-neighborhood of Dam (for active cells)
        full_dry_ijk: [I, 3] int32 - FULL interior voxels for vertex classification (from sparse_flood_fill)
        full_water_ijk: [E, 3] int32 - FULL exterior voxels for vertex classification (from sparse_flood_fill)
        
    Returns:
        (voxelgrid_vertices, cube_idx, adj_idx, vertex_types, active_cells_ijk)
        
    Vertex Types (determined from full sets):
        0: Dam (surface) - sign=0
        1: Water/Tide (exterior) - sign=+1
        2: Dry (interior) - sign=-1
    """
    if dry_adjacent_ijk is None:
        dry_adjacent_ijk = torch.empty((0, 3), dtype=torch.int32, device=dam_ijk.device if len(dam_ijk)>0 else 'cuda')

    device = dam_ijk.device if len(dam_ijk) > 0 else (collision_ijk.device if len(collision_ijk) > 0 else torch.device('cuda'))
    
    # Use int64 for hashing logic to avoid overflow at 2048^3
    R = 2 ** octree.max_level
    resolution = R

    # 1. Active Cells
    all_cells = []
    if len(dam_ijk)>0: all_cells.append(dam_ijk)
    if len(collision_ijk)>0: all_cells.append(collision_ijk)
    if len(dry_adjacent_ijk)>0: all_cells.append(dry_adjacent_ijk)
    
    active_cells_ijk = torch.cat(all_cells, dim=0) if all_cells else torch.empty((0,3), device=device)
    active_cells_ijk = torch.unique(active_cells_ijk, dim=0) # [C, 3]
    num_cells = len(active_cells_ijk)

    # 2. Cell Lookup (Sparse)
    # Linearize: z + y*R + x*R*R
    # Note: Ensure coords are long
    cells_flat = (active_cells_ijk[:,0].long() * R * R + 
                  active_cells_ijk[:,1].long() * R + 
                  active_cells_ijk[:,2].long())
    
    # Sort for searchsorted
    cells_sorted, cells_perm = torch.sort(cells_flat)
    # We need map: FlatKey -> Index in active_cells_ijk
    # But searchsorted gives index in cells_sorted.
    # Map back: real_index = cells_perm[search_result].

    # 3. Vertices (Corners of these cells)
    off_v = torch.tensor([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]], device=device)
    vertex_cand = active_cells_ijk.unsqueeze(1) + off_v.unsqueeze(0) # [C, 8, 3]
    vertex_cand_flat = vertex_cand.view(-1, 3)
    
    vertices_ijk = torch.unique(vertex_cand_flat, dim=0) # [V, 3]
    num_verts = len(vertices_ijk)
    
    # 4. Vertex Lookup (Sparse)
    R_v = R + 1 # Vertices use R+1 grid
    verts_flat = (vertices_ijk[:,0].long() * R_v * R_v + 
                  vertices_ijk[:,1].long() * R_v + 
                  vertices_ijk[:,2].long())
    verts_sorted, verts_perm = torch.sort(verts_flat)

    # 5. Cube Idx (Connectivity: Cell -> 8 Vertices)
    cube_idx = torch.full((num_cells, 8), -1, dtype=torch.long, device=device)
    if num_cells > 0:
        c_corners = vertex_cand # [C, 8, 3]
        c_flat = (c_corners[...,0].long() * R_v * R_v + 
                  c_corners[...,1].long() * R_v + 
                  c_corners[...,2].long()) # [C, 8]
        
        # Search
        idx_in_sorted = torch.searchsorted(verts_sorted, c_flat.view(-1))
        idx_in_sorted = torch.clamp(idx_in_sorted, 0, num_verts - 1)
        found = (verts_sorted[idx_in_sorted] == c_flat.view(-1))
        
        # If valid, map to original index
        final_idx = torch.full_like(idx_in_sorted, -1)
        final_idx[found] = verts_perm[idx_in_sorted[found]]
        
        cube_idx = final_idx.view(num_cells, 8)
    
    # 6. Adjacency (Cell -> Neighbor Cell)
    adj_idx = torch.full((num_cells, 6), -1, dtype=torch.long, device=device)
    if num_cells > 0:
        off_adj = torch.tensor([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]], device=device)
        adj_cand = active_cells_ijk.unsqueeze(1) + off_adj.unsqueeze(0) # [C, 6, 3]
        adj_flat = (adj_cand[...,0].long() * R * R + 
                    adj_cand[...,1].long() * R + 
                    adj_cand[...,2].long()) # [C, 6]
        
        # Filter bounds before search (strictly keys must be valid)
        # Assuming keys outside 0..R^3 are impossible or handled by value mismatch
        # But negative coords wrap around in modulo arithmetic if not careful? 
        # Here we use standard multiplication, negative gives negative key.
        # searchsorted handles value mismatch fine.
        
        idx_in_sorted = torch.searchsorted(cells_sorted, adj_flat.view(-1))
        idx_in_sorted = torch.clamp(idx_in_sorted, 0, num_cells - 1)
        found = (cells_sorted[idx_in_sorted] == adj_flat.view(-1))
        
        final_idx = torch.full_like(idx_in_sorted, -1)
        final_idx[found] = cells_perm[idx_in_sorted[found]]
        
        adj_idx = final_idx.view(num_cells, 6)

    # Fast reconstruction:
    # 1. Create active_keys = encoded active_cells
    # 2. Create dam_keys, water_keys, dry_keys
    # 3. Use searchsorted to find positions of dam_keys in active_keys and set type=0.
    
    cell_vals = torch.full((num_cells,), 255, dtype=torch.uint8, device=device)
    
    # Because active_cells is sorted by `torch.unique`? No, unique sorts by value (coords).
    # But `cells_sorted` is sorted by KEY. It's permuted.
    # Let's treat distinct sets.
    
    def fill_types(subset_ijk, type_id):
        if len(subset_ijk) == 0: return
        keys = (subset_ijk[:,0].long()*R*R + subset_ijk[:,1].long()*R + subset_ijk[:,2].long())
        idx = torch.searchsorted(cells_sorted, keys)
        idx = torch.clamp(idx, 0, num_cells-1)
        found = (cells_sorted[idx] == keys)
        # Original indices
        orig_indices = cells_perm[idx[found]]
        cell_vals[orig_indices] = type_id

    # Order matters: Dry first, then Water, then Dam (overwrites to ensure priority)
    # Actually active_cells are unique. A cell is EITHER Dam OR Water OR Dry.
    # But we might have overlaps in input sets? "extended_tide" vs "dry".
    # Assume Dam > Water > Dry priority if overlap.
    if len(dry_adjacent_ijk) > 0: fill_types(dry_adjacent_ijk, 2)
    if len(collision_ijk) > 0: fill_types(collision_ijk, 1)
    if len(dam_ijk) > 0: fill_types(dam_ijk, 0)
    
    # Lookup vertex type from classification sets
    # Use FULL sets if provided, otherwise fall back to neighborhood sets (true sparse)
    
    # Build sorted hash tables for classification
    # Priority: use full sets if available, otherwise use neighborhood sets
    if full_dry_ijk is not None and len(full_dry_ijk) > 0:
        dry_for_lookup = full_dry_ijk
    elif len(dry_adjacent_ijk) > 0:
        dry_for_lookup = dry_adjacent_ijk
    else:
        dry_for_lookup = torch.empty((0,3), device=device, dtype=torch.int32)
    
    if len(dry_for_lookup) > 0:
        dry_keys = (dry_for_lookup[:,0].long()*R*R + dry_for_lookup[:,1].long()*R + dry_for_lookup[:,2].long())
        dry_sorted, _ = torch.sort(dry_keys)
    else:
        dry_sorted = torch.empty(0, device=device, dtype=torch.long)
    
    if full_water_ijk is not None and len(full_water_ijk) > 0:
        water_for_lookup = full_water_ijk
    elif len(collision_ijk) > 0:
        water_for_lookup = collision_ijk
    else:
        water_for_lookup = torch.empty((0,3), device=device, dtype=torch.int32)
    
    if len(water_for_lookup) > 0:
        water_keys = (water_for_lookup[:,0].long()*R*R + water_for_lookup[:,1].long()*R + water_for_lookup[:,2].long())
        water_sorted, _ = torch.sort(water_keys)
    else:
        water_sorted = torch.empty(0, device=device, dtype=torch.long)
    # Direct position-based vertex classification (matching dense semantics)
    # In dense: scalar_field[i,j,k] = sign at vertex position (i,j,k)
    # In sparse: look up vertex position in classification sets
    #
    # Classification: Dry=+1 (interior), Water=-1 (exterior), Dam=0 (surface)
    
    vertex_keys = (vertices_ijk[:,0].long()*R*R + vertices_ijk[:,1].long()*R + vertices_ijk[:,2].long())
    
    # Build lookup tables for each classification
    dam_keys = (dam_ijk[:,0].long()*R*R + dam_ijk[:,1].long()*R + dam_ijk[:,2].long()) if len(dam_ijk) > 0 else torch.empty(0, device=device, dtype=torch.long)
    dam_sorted, _ = torch.sort(dam_keys) if len(dam_keys) > 0 else (dam_keys, None)
    
    # Check membership in each set
    def check_membership(sorted_keys, query_keys):
        if len(sorted_keys) == 0:
            return torch.zeros(len(query_keys), dtype=torch.bool, device=device)
        idx = torch.searchsorted(sorted_keys, query_keys)
        idx = torch.clamp(idx, 0, len(sorted_keys)-1)
        return sorted_keys[idx] == query_keys
    
    is_dam = check_membership(dam_sorted, vertex_keys)
    is_dry = check_membership(dry_sorted, vertex_keys) if len(dry_sorted) > 0 else torch.zeros(num_verts, dtype=torch.bool, device=device)
    is_tide = check_membership(water_sorted, vertex_keys) if len(water_sorted) > 0 else torch.zeros(num_verts, dtype=torch.bool, device=device)
    
    # Assign vertex types based on direct lookup
    # Priority: Dam(0) > Dry(2) > Water(1) > default(exterior=1)
    vertex_mask = torch.full((num_verts,), 1, dtype=torch.uint8, device=device)  # Default: exterior
    vertex_mask[is_tide] = 1  # Water = exterior
    vertex_mask[is_dry] = -1   # Dry = interior
    vertex_mask[is_dam] = 0   # Dam = surface

    # 8. Coords
    # Use octree bounds logic or simple sizing
    if num_verts > 0:
        zeros = torch.zeros((1,3), dtype=torch.int32, device=device)
        cmin, cmax = octree.get_cell_aabb_level(zeros, octree.max_level)
        cell_size = cmax - cmin
        voxelgrid_vertices = cmin + vertices_ijk.float() * cell_size
    else:
        voxelgrid_vertices = torch.empty((0, 3), dtype=torch.float32, device=device)

    return voxelgrid_vertices, cube_idx, adj_idx, vertex_mask, active_cells_ijk


def filter_mesh_by_dam(
    bvh: MeshBVH,
    dam_ijk: torch.Tensor,
    octree: OctreeIndexer
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Filter mesh to keep only faces that intersect with dam voxels.
    
    This creates a "shell" mesh corresponding to the dam region,
    filtering out geometry that is far from the dam.
    
    Args:
        bvh: MeshBVH - Input mesh with BVH acceleration
        dam_ijk: [N, 3] int32 - Active voxels (e.g. extended_dam)
        octree: OctreeIndexer - Grid context for coordinate conversion
        
    Returns:
        new_vertices: [V', 3] float32 - Subset of vertices
        new_faces: [F', 3] int32 - Subset of faces
    """
    if len(dam_ijk) == 0:
        return (torch.empty(0, 3, dtype=torch.float32, device=bvh.device), 
                torch.empty(0, 3, dtype=torch.int32, device=bvh.device))
    
    level = octree.max_level
    
    # 1. Get AABBs for all voxels in world space
    # [N, 3], [N, 3]
    voxel_mins, voxel_maxs = octree.get_cell_aabb_level(dam_ijk, level)
    
    # 2. Query BVH for intersections
    # intersect_aabb returns active pairs (box_idx, face_idx)
    # We only care about unique face indices
    result = bvh.intersect_aabb(voxel_mins, voxel_maxs, mode=1)
    
    if result.face_ids is None or len(result.face_ids) == 0:
        return (torch.empty(0, 3, dtype=torch.float32, device=bvh.device), 
                torch.empty(0, 3, dtype=torch.int32, device=bvh.device))

    # 3. Extract unique faces
    kept_face_indices = torch.unique(result.face_ids)
    # Ensure sorted for determinism
    kept_face_indices, _ = torch.sort(kept_face_indices)
    
    # 4. Filter faces
    # [F', 3]
    old_faces = bvh.faces[kept_face_indices]
    
    # 5. Compact vertices (optional but recommended to save memory)
    # Find unique vertex indices used by kept faces
    active_verts_idx = torch.unique(old_faces)
    active_verts_idx, _ = torch.sort(active_verts_idx)
    
    # Create mapping: old_idx -> new_idx
    # We can use searchsorted since active_verts_idx is sorted
    # Or strict mapping. Since it's a subset, searchsorted works if we verify.
    
    # New vertices list
    new_vertices = bvh.vertices[active_verts_idx]
    
    # Remap faces indices
    # Method: create a dense map if range is small, or use searchsorted
    # searchsorted is generally efficient for sparse remapping
    new_faces = torch.searchsorted(active_verts_idx, old_faces)
    
    return new_vertices, new_faces


def reconstruct_dense_from_sparse(
    dam_ijk: torch.Tensor,
    resolution: int,
    connectivity: int = 6,
    device: str = 'cuda'
) -> Dict[str, torch.Tensor]:
    """
    Reconstruct dense interior/exterior masks from sparse dam_ijk.
    
    This allows storing only `dam_ijk` and reconstructing the full
    interior/exterior classification on demand.
    
    Algorithm:
        1. Use dam_ijk as barrier (blocks flood)
        2. Flood fill from grid boundary
        3. Voxels reachable from boundary = exterior (water)
        4. Voxels NOT reachable = interior (dry)
    
    Args:
        dam_ijk: [N, 3] int32 - Surface/dam voxel coordinates
        resolution: Grid resolution R (R³ grid)
        connectivity: 6 (face) or 26 (full), default 6
        device: Output device for tensors
    
    Returns:
        {
            'dry_ijk': [D, 3] int32 - Interior voxels
            'water_ijk': [W, 3] int32 - Exterior voxels
            'dry_mask': [R, R, R] bool - Dense interior mask (on CPU)
            'water_mask': [R, R, R] bool - Dense exterior mask (on CPU)
        }
    
    Example:
        >>> flood_res = sparse_flood_fill(bvh, resolution=256)
        >>> dam_ijk = flood_res['dam_ijk']
        >>> dense = reconstruct_dense_from_sparse(dam_ijk, resolution=256)
        >>> dry_ijk = dense['dry_ijk']  # Interior voxels
    """
    from scipy import ndimage
    import numpy as np
    
    R = resolution
    
    # 1. Build barrier mask (dam blocks flood)
    can_flood = np.ones((R, R, R), dtype=bool)
    
    if dam_ijk.numel() > 0:
        dam_np = dam_ijk.cpu().numpy().astype(np.int64)
        # Bounds check
        valid = (dam_np >= 0).all(axis=1) & (dam_np < R).all(axis=1)
        dam_np = dam_np[valid]
        can_flood[dam_np[:, 0], dam_np[:, 1], dam_np[:, 2]] = False
    
    # 2. Connected component labeling
    if connectivity == 6:
        struct = ndimage.generate_binary_structure(3, 1)  # 6-connectivity
    else:
        struct = ndimage.generate_binary_structure(3, 3)  # 26-connectivity
    
    labels, num_labels = ndimage.label(can_flood, structure=struct)
    
    # 3. Find labels connected to boundary (exterior)
    boundary_labels = set()
    # All 6 faces of the cube
    boundary_labels.update(labels[0, :, :].flatten())
    boundary_labels.update(labels[-1, :, :].flatten())
    boundary_labels.update(labels[:, 0, :].flatten())
    boundary_labels.update(labels[:, -1, :].flatten())
    boundary_labels.update(labels[:, :, 0].flatten())
    boundary_labels.update(labels[:, :, -1].flatten())
    boundary_labels.discard(0)  # 0 is background (dam voxels)
    
    # 4. Create masks
    is_exterior = np.isin(labels, list(boundary_labels))
    water_mask = is_exterior & can_flood  # Exterior, excluding dam
    dry_mask = ~is_exterior & can_flood   # Interior, excluding dam
    
    # 5. Find dry voxels adjacent to dam (interior tide line)
    # Dilate dam by 1 voxel using the same connectivity
    dam_mask = ~can_flood  # dam is where we can't flood
    if connectivity == 6:
        dilate_struct = ndimage.generate_binary_structure(3, 1)
    else:
        dilate_struct = ndimage.generate_binary_structure(3, 3)
    
    dam_dilated = ndimage.binary_dilation(dam_mask, structure=dilate_struct)
    # dry_collision = dry voxels that are adjacent to dam
    dry_collision_mask = dry_mask & dam_dilated
    
    # 6. Convert to sparse coordinates
    water_coords = np.argwhere(water_mask)
    dry_coords = np.argwhere(dry_mask)
    dry_adjacent_coords = np.argwhere(dry_collision_mask)
    
    # 7. Convert to torch tensors
    water_ijk = torch.tensor(water_coords, dtype=torch.int32, device=device)
    dry_ijk = torch.tensor(dry_coords, dtype=torch.int32, device=device)
    dry_adjacent_ijk = torch.tensor(dry_adjacent_coords, dtype=torch.int32, device=device)
    
    return {
        'dry_ijk': dry_ijk,                  # All interior voxels
        'water_ijk': water_ijk,              # All exterior voxels
        'dry_adjacent_ijk': dry_adjacent_ijk,  # Interior voxels adjacent to dam
        'dry_mask': dry_mask,                # Dense interior mask (CPU)
        'water_mask': water_mask,            # Dense exterior mask (CPU)
    }


