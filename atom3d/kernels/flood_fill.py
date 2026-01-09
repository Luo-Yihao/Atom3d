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
    start_point: Tuple[int, int, int] = (0, 0, 0)
) -> torch.Tensor:
    """
    CUDA-accelerated 3D flood fill with 26-connectivity.
    
    Args:
        occupancy: [D, H, W] bool tensor on CUDA
            True = occupied (dam/surface)
            False = free space
        start_point: (z, y, x) starting coordinate
    
    Returns:
        mask: [D, H, W] int32 tensor (raw kernel output)
            -1 = unreachable (dry/interior)
             0 = dam surface (occupancy=True)
             1 = filled (water/exterior)
    """
    if not occupancy.is_cuda:
        raise ValueError("occupancy must be on CUDA device")
    
    if occupancy.dim() != 3:
        raise ValueError(f"Expected 3D tensor, got {occupancy.dim()}D")
    
    cuda = get_floodfill_kernels()
    
    # The pytorch-floodfill-3d kernel starts from (0,0,0) by default
    # We need to handle custom start_point by modifying the mask initialization
    return cuda.flood_fill(occupancy.contiguous())


def flood_fill_3d_sparse(
    dam_coords: torch.Tensor,
    resolution: int,
    source: Tuple[int, int, int] = (0, 0, 0),
    padding: int = 2
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sparse flood fill using CUDA on a cropped region.
    
    Only processes the bounding box around dam_coords, then maps back to full resolution.
    
    Args:
        dam_coords: [N, 3] int32 surface voxel coordinates (mesh-intersecting)
        resolution: Full grid resolution
        source: Seed point in full resolution coords
        padding: Bounding box padding
    
    Returns:
        water_coords: [K, 3] int32 pure water voxels (exterior, NOT adjacent to dam)
        dry_coords: [M, 3] int32 dry voxels (interior)
        collision_coords: [C, 3] int32 collision voxels (water voxels ADJACENT to dam)
        dam_mask: [N] bool indicating which dam coords have water neighbors
        
    Semantics:
        - Dam ∩ Mesh = Dam (all dam voxels intersect mesh)
        - Water ∩ Mesh = ∅ (water never intersects mesh)
        - Collision ∩ Mesh = ∅ (collision never intersects mesh)
        - Collision ⊂ "original water" (collision is the tide line)
        - (Dam + Water + Collision + Dry) = all, mutually exclusive
    """
    device = dam_coords.device
    
    if len(dam_coords) == 0:
        return (
            torch.empty(0, 3, dtype=torch.int32, device=device),
            torch.empty(0, 3, dtype=torch.int32, device=device),
            torch.empty(0, dtype=torch.bool, device=device)
        )
    
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
    if (local_source < 0).any() or (local_source >= torch.tensor(cropped_size, device=device)).any():
        local_source = torch.tensor([0, 0, 0], dtype=torch.int32, device=device)
    
    # Run CUDA flood fill
    mask = flood_fill_3d(occupancy)
    
    # Extract results
    # CUDA kernel semantics:
    #   mask == -1: dry (interior, unreachable from source)
    #   mask == 0: boundary (water voxels touching dam, OR dam voxels themselves)
    #   mask == 1: water (exterior, reachable and not touching dam)
    
    # Collision = mask==0 AND occupancy==False (water voxels at dam boundary)
    collision_mask_local = (mask == 0) & (~occupancy)
    # Pure Water = mask==1 (water not touching dam)
    pure_water_mask_local = (mask == 1)
    # Dry = mask==-1
    dry_local = torch.nonzero(mask == -1, as_tuple=False)
    
    water_local = torch.nonzero(pure_water_mask_local, as_tuple=False)
    collision_local = torch.nonzero(collision_mask_local, as_tuple=False)
    
    water_global = water_local.int() + bbox_min if len(water_local) > 0 else torch.empty(0, 3, dtype=torch.int32, device=device)
    dry_global = dry_local.int() + bbox_min if len(dry_local) > 0 else torch.empty(0, 3, dtype=torch.int32, device=device)
    collision_global = collision_local.int() + bbox_min if len(collision_local) > 0 else torch.empty(0, 3, dtype=torch.int32, device=device)
    
    # Dam boundary mask: dam voxels that have mask==0 neighbors (i.e., touched by water)
    dam_mask = (mask[local_coords[:, 0], local_coords[:, 1], local_coords[:, 2]] == 0)
    
    return water_global, dry_global, collision_global, dam_mask


__all__ = [
    'get_floodfill_kernels',
    'floodfill_available', 
    'flood_fill_3d',
    'flood_fill_3d_sparse'
]
