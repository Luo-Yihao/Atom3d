"""
Sparse Flood Fill: Hierarchical Octree-based flood fill for mesh voxelization.
"""

from typing import Tuple, Dict, Set, List
import torch

from ..core.mesh_bvh import MeshBVH
from ..grid.octree_indexer import OctreeIndexer

# Try to import CUDA flood fill kernels
try:
    from ..kernels.flood_fill import flood_fill_3d_sparse, floodfill_available
    HAS_CUDA_FLOODFILL = floodfill_available()
except ImportError:
    HAS_CUDA_FLOODFILL = False


def sparse_flood_fill(
    bvh: MeshBVH,
    octree: OctreeIndexer,
    source: Tuple[int, int, int],
    connectivity: int = 6,
    backend: str = 'auto',
    min_level: int = 0
) -> Dict[str, torch.Tensor]:
    """
    Sparse flood fill with optional CUDA acceleration.
    
    Args:
        bvh: MeshBVH
        octree: OctreeIndexer
        source: Seed point (exterior voxel coord)
        connectivity: 6 (face) or 26 (full)
        backend: 'auto', 'cuda', or 'python'
            - auto: Use CUDA if available, else Python
            - cuda: Force CUDA (raises if unavailable)
            - python: Force Python (hierarchical mixed-level)
    
    Returns:
        {
            'dam_nodes': [N, 2] int32 - (level, node_idx) Surface nodes
            'water_nodes': [K, 2] int32 - (level, node_idx) Exterior nodes
            'dry_nodes': [M, 2] int32 - (level, node_idx) Interior nodes
        }
    """
    # Backend selection
    use_cuda = False
    if backend == 'cuda':
        if not HAS_CUDA_FLOODFILL:
            raise RuntimeError("CUDA flood fill not available. Install pytorch-floodfill-3d.")
        use_cuda = True
    elif backend == 'auto':
        use_cuda = HAS_CUDA_FLOODFILL
    # else: python
    
    if use_cuda:
        return _sparse_flood_fill_cuda(bvh, octree, source, min_level)
    else:
        return _sparse_flood_fill_python(bvh, octree, source, connectivity)


def _sparse_flood_fill_cuda(
    bvh: MeshBVH,
    octree: OctreeIndexer,
    source: Tuple[int, int, int],
    min_level: int = 0
) -> Dict[str, torch.Tensor]:
    """CUDA-accelerated flood fill using cropped dense grid."""
    from ..kernels.flood_fill import flood_fill_3d_sparse
    
    device = octree.device
    max_level = octree.max_level
    resolution = 2 ** max_level
    
    # Step 1: Surface voxels
    print(f"[CUDA Flood Fill] Finding surface voxels...")
    dam_coords = octree.octree_traverse(bvh, min_level=min_level)
    print(f"  Dam voxels: {len(dam_coords)}")
    
    # Dam nodes
    dam_indices = (dam_coords[:, 0] * resolution * resolution + 
                   dam_coords[:, 1] * resolution + 
                   dam_coords[:, 2])
    dam_levels = torch.full((len(dam_coords),), max_level, dtype=torch.int32, device=device)
    dam_nodes = torch.stack([dam_levels, dam_indices], dim=1)
    
    # Step 2: CUDA sparse flood fill
    print(f"[CUDA Flood Fill] Running CUDA kernel on cropped region...")
    import time
    t0 = time.time()
    # Use larger padding to ensure flood fill can find exterior space
    water_coords, dry_coords, collision_coords, dam_boundary_mask = flood_fill_3d_sparse(
        dam_coords, resolution, source, padding=10
    )
    cuda_time = time.time() - t0
    print(f"  CUDA flood fill took {cuda_time:.4f}s")
    
    # Convert to nodes format
    if len(water_coords) > 0:
        water_indices = (water_coords[:, 0] * resolution * resolution + 
                        water_coords[:, 1] * resolution + 
                        water_coords[:, 2])
        water_levels = torch.full((len(water_coords),), max_level, dtype=torch.int32, device=device)
        water_nodes = torch.stack([water_levels, water_indices], dim=1)
    else:
        water_nodes = torch.empty(0, 2, dtype=torch.int32, device=device)
    
    if len(dry_coords) > 0:
        dry_indices = (dry_coords[:, 0] * resolution * resolution + 
                      dry_coords[:, 1] * resolution + 
                      dry_coords[:, 2])
        dry_levels = torch.full((len(dry_coords),), max_level, dtype=torch.int32, device=device)
        dry_nodes = torch.stack([dry_levels, dry_indices], dim=1)
    else:
        dry_nodes = torch.empty(0, 2, dtype=torch.int32, device=device)
    
    print(f"  Water nodes: {len(water_nodes)}")
    print(f"  Dry nodes: {len(dry_nodes)}")
    print(f"  Collision voxels (flood_mask==0): {len(collision_coords)}")
    
    return {
        'dam_nodes': dam_nodes,
        'dam_coords': dam_coords,                  # Raw dam coordinates (mesh-intersecting)
        'collision_coords': collision_coords,      # Water voxels adjacent to dam (tide line)
        'dam_boundary_mask': dam_boundary_mask,    # Which dams have water neighbors
        'water_nodes': water_nodes,
        'dry_nodes': dry_nodes
    }


def _sparse_flood_fill_python(
    bvh: MeshBVH,
    octree: OctreeIndexer,
    source: Tuple[int, int, int],
    connectivity: int = 6
) -> Dict[str, torch.Tensor]:
    """
    Python hierarchical flood fill with mixed-level nodes.
    
    Optimized with on-the-fly coarsening and top-down dry detection.
    """
    device = octree.device
    max_level = octree.max_level
    resolution = 2 ** max_level
    
    # Step 1: Surface voxels
    print(f"[Sparse Flood Fill] Finding surface voxels...")
    dam_coords = octree.octree_traverse(bvh, min_level=max(4, max_level - 3))
    print(f"  Dam voxels: {len(dam_coords)}")
    
    dam_indices = (dam_coords[:, 0] * resolution * resolution + 
                   dam_coords[:, 1] * resolution + 
                   dam_coords[:, 2])
    dam_levels = torch.full((len(dam_coords),), max_level, dtype=torch.int32, device=device)
    dam_nodes_tensor = torch.stack([dam_levels, dam_indices], dim=1)
    
    # Step 2: Build active nodes
    print(f"[Sparse Flood Fill] Building active nodes dictionary...")
    active_nodes = octree.mark_active_nodes(dam_coords)
    print(f"  Total active nodes: {len(active_nodes)}")
    
    # Step 3: BFS with On-the-Fly Coarsening
    print(f"[Sparse Flood Fill] Running hierarchical BFS...")
    
    # Start at max_level
    source_idx = source[0] * resolution * resolution + source[1] * resolution + source[2]
    source_node = (max_level, source_idx)
    
    if active_nodes.get(source_node, False):
        raise ValueError(f"Source is on surface!")
    
    # BFS
    current = {source_node}
    done = set()
    water_nodes_dict = {}  # {node: True}
    
    # Initialize with source
    water_nodes_dict[source_node] = True
    
    iteration = 0
    coarsened_count = 0
    
    while current:
        done.update(current)
        next_current = set()
        
        # Track newly added water nodes for coarsening check
        parents_to_check = set()
        
        for node in current:
            level, idx = node
            
            # Get neighbors at same level
            neighbors = octree.get_node_neighbors(level, idx, connectivity=connectivity)
            
            for neighbor in neighbors:
                if neighbor not in done and neighbor not in water_nodes_dict:
                    if not active_nodes.get(neighbor, False):
                        # Found new water node
                        next_current.add(neighbor)
                        water_nodes_dict[neighbor] = True
                        
                        # Mark parent for checking
                        if level > 0:
                            res = 2 ** level
                            nx = neighbor[1] // (res * res)
                            rem = neighbor[1] % (res * res)
                            ny = rem // res
                            nz = rem % res
                            
                            px, py, pz = nx // 2, ny // 2, nz // 2
                            p_res = 2 ** (level - 1)
                            p_idx = px * p_res * p_res + py * p_res + pz
                            parents_to_check.add((level - 1, p_idx))

        # On-the-fly Coarsening Logic
        while parents_to_check:
            next_parents_to_check = set()
            
            for parent_node in parents_to_check:
                p_level, p_idx = parent_node
                
                # 1. Check if parent is blocked by active_nodes (surface)
                if active_nodes.get(parent_node, False):
                    continue
                
                # 2. Check if all 8 children are in water_nodes_dict
                p_res = 2 ** p_level
                px = p_idx // (p_res * p_res)
                p_rem = p_idx % (p_res * p_res)
                py = p_rem // p_res
                pz = p_rem % p_res
                
                child_level = p_level + 1
                child_res = 2 ** child_level
                base_x, base_y, base_z = px * 2, py * 2, pz * 2
                
                siblings = []
                all_water = True
                
                for dz in [0, 1]:
                    for dy in [0, 1]:
                        for dx in [0, 1]:
                            sx, sy, sz = base_x + dx, base_y + dy, base_z + dz
                            s_idx = sx * child_res * child_res + sy * child_res + sz
                            s_node = (child_level, s_idx)
                            siblings.append(s_node)
                            
                            if s_node not in water_nodes_dict:
                                all_water = False
                                break
                        if not all_water: break
                    if not all_water: break
                
                if all_water:
                    # COARSEN!
                    # Remove children from dict and BFS queues
                    for s_node in siblings:
                        del water_nodes_dict[s_node]
                        if s_node in next_current:
                            next_current.remove(s_node)
                    
                    # Add parent to water dict
                    water_nodes_dict[parent_node] = True
                    next_current.add(parent_node)
                    
                    coarsened_count += 1
                    
                    # Queue Grandparent for check
                    if p_level > 0:
                        gp_res = 2 ** (p_level - 1)
                        gpx, gpy, gpz = px // 2, py // 2, pz // 2
                        gp_idx = gpx * gp_res * gp_res + gpy * gp_res + gpz
                        next_parents_to_check.add((p_level - 1, gp_idx))
            
            parents_to_check = next_parents_to_check
        
        current = next_current
        iteration += 1
        
        if iteration % 50 == 0:
            print(f"  Iteration {iteration}: {len(done)} explored, {len(water_nodes_dict)} water nodes (coarsened {coarsened_count})")

    print(f"  Converged at iteration {iteration}")
    print(f"  Total coarsening events: {coarsened_count}")
    
    # Final level distribution
    level_counts = {}
    for (level, idx) in water_nodes_dict.keys():
        level_counts[level] = level_counts.get(level, 0) + 1
    
    print(f"  Final water nodes: {len(water_nodes_dict)} (mixed-level)")
    for level in sorted(level_counts.keys()):
        print(f"    Level {level}: {level_counts[level]:,} nodes")

    # Step 4: Dry Detection (Top-Down)
    print(f"[Sparse Flood Fill] Finding dry nodes (top-down)...")
    dry_nodes_list = []
    
    # Stack for non-recursive traversal
    stack = [(0, 0, 0, 0)] # level, x, y, z
    
    while stack:
        l, x, y, z = stack.pop()
        res = 2 ** l
        idx = x * res * res + y * res + z
        node = (l, idx)
        
        # 1. Is it Water?
        if node in water_nodes_dict:
            continue
            
        # 2. Is it blocked by Surface (Active)?
        is_active = active_nodes.get(node, False)
        
        if not is_active:
            # Not active AND not water implies DRY
            dry_nodes_list.append(node)
            continue
        
        # 3. It IS active (contains surface). Recurse.
        if l < max_level:
            nl = l + 1
            # Add 8 children to stack
            base_x, base_y, base_z = x*2, y*2, z*2
            for dz in [1, 0]:
                for dy in [1, 0]:
                    for dx in [1, 0]:
                        stack.append((nl, base_x+dx, base_y+dy, base_z+dz))
    
    print(f"  Dry nodes found: {len(dry_nodes_list)} (mixed-level)")

    # Convert to Tensors
    if len(water_nodes_dict) > 0:
        water_nodes_list = list(water_nodes_dict.keys())
        water_nodes_tensor = torch.tensor(water_nodes_list, dtype=torch.int32, device=device)
    else:
        water_nodes_tensor = torch.empty(0, 2, dtype=torch.int32, device=device)
        
    if len(dry_nodes_list) > 0:
        dry_nodes_tensor = torch.tensor(dry_nodes_list, dtype=torch.int32, device=device)
    else:
        dry_nodes_tensor = torch.empty(0, 2, dtype=torch.int32, device=device)

    return {
        'dam_nodes': dam_nodes_tensor,
        'water_nodes': water_nodes_tensor,
        'dry_nodes': dry_nodes_tensor
    }


def get_dam_face_labels(
    dam_coords: torch.Tensor,
    dry_coords: torch.Tensor,
    level: int
) -> torch.Tensor:
    """
    For each dam voxel, label its 6 face neighbors.
    Useful for training data generation.
    """
    resolution = 2 ** level
    device = dam_coords.device
    
    # Build hash sets for O(1) lookup
    dam_set = set((dam_coords[:, 0] * resolution * resolution + 
                   dam_coords[:, 1] * resolution + 
                   dam_coords[:, 2]).cpu().tolist())
    
    dry_set = set((dry_coords[:, 0] * resolution * resolution + 
                   dry_coords[:, 1] * resolution + 
                   dry_coords[:, 2]).cpu().tolist()) if len(dry_coords) > 0 else set()
    
    # 6 face offsets: [-x, +x, -y, +y, -z, +z]
    face_offsets = torch.tensor([
        [-1, 0, 0], [1, 0, 0], 
        [0, -1, 0], [0, 1, 0], 
        [0, 0, -1], [0, 0, 1],
    ], dtype=torch.int32, device=device)
    
    N = dam_coords.shape[0]
    labels = torch.full((N, 6), -2, dtype=torch.int8, device=device)
    
    for i in range(N):
        for j in range(6):
            nb = dam_coords[i] + face_offsets[j]
            if (nb < 0).any() or (nb >= resolution).any():
                continue
            
            nb_idx = int(nb[0] * resolution * resolution + nb[1] * resolution + nb[2])
            
            if nb_idx in dam_set:
                labels[i, j] = 0    # Dam
            elif nb_idx in dry_set:
                labels[i, j] = -1   # Dry
            else:
                labels[i, j] = 1    # Water
    return labels


__all__ = ['sparse_flood_fill', 'get_dam_face_labels']
