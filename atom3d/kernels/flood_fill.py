"""
CUDA Flood Fill Kernel Wrapper

Uses pytorch-floodfill-3d CUDA kernels for GPU-accelerated 3D flood fill.
"""

import os
import torch
from typing import Tuple, Optional

# Module-level cache
_floodfill_cuda = None
_floodfill_loaded = False


def get_floodfill_kernels():
    """
    Load CUDA flood fill kernels via JIT compilation.
    
    No external dependencies - uses local kernel source.
    
    Returns:
        Compiled extension module with flood_fill function
    """
    global _floodfill_cuda, _floodfill_loaded
    
    if _floodfill_loaded and _floodfill_cuda is not None:
        return _floodfill_cuda
    
    kernel_dir = os.path.dirname(os.path.abspath(__file__))
    floodfill_src = os.path.join(kernel_dir, 'flood_fill_kernels.cu')
    
    if not os.path.exists(floodfill_src):
        raise ImportError(
            f"Flood fill kernel source not found at {floodfill_src}. "
            "Please ensure Atom3D is installed correctly."
        )
    
    from torch.utils.cpp_extension import load
    build_dir = os.path.join(kernel_dir, 'build')
    os.makedirs(build_dir, exist_ok=True)
    
    _floodfill_cuda = load(
        name='floodfill_cuda',
        sources=[floodfill_src],
        build_directory=build_dir,
        extra_cuda_cflags=['-O3', '--use_fast_math'],
        verbose=False
    )
    _floodfill_loaded = True
    return _floodfill_cuda


def floodfill_available() -> bool:
    """Check if CUDA flood fill is available"""
    if not torch.cuda.is_available():
        return False
    try:
        get_floodfill_kernels()
        return True
    except ImportError:
        return False


def flood_fill_3d(
    occupancy: torch.Tensor,
    start_point: Tuple[int, int, int] = (0, 0, 0),
    connectivity: int = 26
) -> torch.Tensor:
    """
    CUDA-accelerated 3D flood fill with connectivity.
    
    Args:
        occupancy: [D, H, W] bool tensor on CUDA
            True = occupied (surface)
            False = free space
        start_point: (z, y, x) starting coordinate (z is D-dim, y is H-dim, x is W-dim)
            Note: Standard sequence is usually (x,y,z) in user code, but tensor is (z,y,x).
            Please ensure caller provides correct axis order. 
            Typical usage matches tensor indexing: occupancy[z, y, x].
        connectivity: 6, 18, or 26 (default 26)
    
    Returns:
        mask: [D, H, W] int32 tensor
            -2 = Dry (Unreachable)
            -1 = Dam (Occupied voxel reached/touched by water)
             1 = Collision (Water voxel adjacent to Dam)
             2 = Water (Pure Water voxel)
    """
    if not occupancy.is_cuda:
        raise ValueError("occupancy must be on CUDA device")
    
    if occupancy.dim() != 3:
        raise ValueError(f"Expected 3D tensor, got {occupancy.dim()}D")
    
    cuda = get_floodfill_kernels()
    
    # Pass start point coordinates explicitly
    # Kernel expects (x, y, z) corresponding to W, H, D dimensions if treated as 3D coords
    # But tensor indexing is [D, H, W] -> [z, y, x]
    # The kernel calculates idx = z * (H*W) + y * W + x
    # So we pass z, y, x corresponding to D, H, W
    z, y, x = start_point
    return cuda.flood_fill(occupancy.contiguous(), x, y, z, connectivity)


def flood_fill_3d_sparse(
    dam_coords: torch.Tensor,
    resolution: int,
    source: Tuple[int, int, int] = (0, 0, 0),
    padding: int = 2,
    connectivity: int = 26
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sparse flood fill using CUDA on a cropped region.
    
    Args:
        dam_coords: [N, 3] int32 surface voxel coordinates
        resolution: Full grid resolution
        source: Seed point in full resolution coords
        padding: Bounding box padding
        connectivity: 6, 18, or 26 (default 26)
    
    Returns:
        water_coords: [K, 3] int32 (mask == 2)
        dry_coords: [M, 3] int32 (mask == -2)
        collision_coords: [C, 3] int32 (mask == 1)
        true_dam_coords: [D, 3] int32 (mask == -1)
        dam_mask: [N] bool indicating which input dam_coords are in true_dam_coords
        extended_dam_coords: [E, 3] int32 (diagonal dam voxels)
        
    If start point is occupied, returns empty tensors and prints warning.
    """
    device = dam_coords.device
    
    if len(dam_coords) == 0:
        return _empty_result(device)
    
    # Compute tight bounding box
    bbox_min = dam_coords.min(dim=0)[0] - padding
    bbox_max = dam_coords.max(dim=0)[0] + padding
    
    # Clamp to valid range
    bbox_min = torch.clamp(bbox_min, min=0)
    bbox_max = torch.clamp(bbox_max, max=resolution - 1)
    
    cropped_size = (bbox_max - bbox_min + 1).tolist()
    
    # Build cropped occupancy grid
    occupancy = torch.zeros(cropped_size, dtype=torch.bool, device=device)
    local_coords = dam_coords - bbox_min
    occupancy[local_coords[:, 0], local_coords[:, 1], local_coords[:, 2]] = True
    
    # Convert source to local coordinates
    source_tensor = torch.tensor(source, dtype=torch.int32, device=device)
    local_source = source_tensor - bbox_min
    
    # Check if source is within cropped region
    ls = local_source.long()
    in_bounds = (local_source >= 0).all() and (local_source < torch.tensor(cropped_size, device=device)).all()
    
    if in_bounds:
        # Check if source is occupied
        if occupancy[ls[0], ls[1], ls[2]]:
            print(f"[Flood Fill Warning] Start point {source} is OCCUPIED. Returning empty result.")
            return _empty_result(device)
        start_point_tuple = (ls[0].item(), ls[1].item(), ls[2].item())
    else:
        # Default starting point if source is outside bounds
        start_point_tuple = (0, 0, 0)
        if occupancy[0, 0, 0]:
             print(f"[Flood Fill Warning] Default start (0,0,0) is OCCUPIED. Returning empty result.")
             return _empty_result(device)

    # Run CUDA flood fill
    # mask values: -2 (Dry), -1 (Dam), 1 (Collision), 2 (Water)
    mask = flood_fill_3d(occupancy, start_point=start_point_tuple, connectivity=connectivity)
    
    # --- Extend Dam Logic ---
    # User request: Dry (-2) voxels that are 26-adjacent to flooded voxels (>0) should be considered Dam (-1)
    # This captures diagonal leaks or disconnected components that are touching water diagonally
    
    # 1. Identify flooded region (Water or Collision)
    flooded = (mask > 0).float() # [D, H, W]
    
    # 2. Dilate flooded region by 1 voxel (26-connectivity) using max_pool3d
    # input needs to be [N, C, D, H, W]
    flooded_expanded = flooded.unsqueeze(0).unsqueeze(0)
    dilated_flooded = torch.nn.functional.max_pool3d(
        flooded_expanded, kernel_size=3, stride=1, padding=1
    ).squeeze(0).squeeze(0)
    
    # 3. Identify candidates: Dry (-2) AND Occupied AND Adjacent to Flooded
    extended_mask = (mask == -2) & occupancy & (dilated_flooded > 0.5)
    
    # 4. Update Mask and Extract Coordinates
    mask[extended_mask] = -1
    extended_dam_local = torch.nonzero(extended_mask, as_tuple=False)
    
    # Extract results (true_dam includes the extended ones now)
    collision_mask_local = (mask == 1)
    pure_water_mask_local = (mask == 2)
    true_dam_mask_local = (mask == -1)
    dry_mask_local = (mask == -2)
    
    collision_local = torch.nonzero(collision_mask_local, as_tuple=False)
    water_local = torch.nonzero(pure_water_mask_local, as_tuple=False)
    true_dam_local = torch.nonzero(true_dam_mask_local, as_tuple=False)
    dry_local = torch.nonzero(dry_mask_local, as_tuple=False)
    
    water_global = water_local.int() + bbox_min if len(water_local) > 0 else torch.empty(0, 3, dtype=torch.int32, device=device)
    dry_global = dry_local.int() + bbox_min if len(dry_local) > 0 else torch.empty(0, 3, dtype=torch.int32, device=device)
    collision_global = collision_local.int() + bbox_min if len(collision_local) > 0 else torch.empty(0, 3, dtype=torch.int32, device=device)
    true_dam_global = true_dam_local.int() + bbox_min if len(true_dam_local) > 0 else torch.empty(0, 3, dtype=torch.int32, device=device)
    extended_dam_global = extended_dam_local.int() + bbox_min if len(extended_dam_local) > 0 else torch.empty(0, 3, dtype=torch.int32, device=device)

    # Dam boundary mask: which input dam_coords are in true_dam_global
    # Vectorized check: simply check the mask value at the dam coordinates
    dam_values_at_coords = mask[local_coords[:, 0], local_coords[:, 1], local_coords[:, 2]]
    dam_mask = (dam_values_at_coords == -1)
    
    return water_global, dry_global, collision_global, true_dam_global, dam_mask, extended_dam_global


def _empty_result(device):
    return (
        torch.empty(0, 3, dtype=torch.int32, device=device),
        torch.empty(0, 3, dtype=torch.int32, device=device),
        torch.empty(0, 3, dtype=torch.int32, device=device),
        torch.empty(0, 3, dtype=torch.int32, device=device),
        torch.empty(0, dtype=torch.bool, device=device),
        torch.empty(0, 3, dtype=torch.int32, device=device)
    )





__all__ = [
    'get_floodfill_kernels',
    'floodfill_available', 
    'flood_fill_3d',
    'flood_fill_3d_sparse'
]
