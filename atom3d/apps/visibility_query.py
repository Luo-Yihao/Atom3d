"""
Visibility Query Module

Ray-based visibility testing using arbitrary camera configurations.
Supports multiple camera sampling strategies for robust coverage.
"""

from typing import Optional, Tuple
import torch

from ..core.mesh_bvh import MeshBVH


__all__ = [
    "VisibilityQuery",
    "uniform_sphere_cameras",
    "fibonacci_sphere_cameras", 
    "hierarchical_cameras",
]


# =============================================================================
# Camera Samplers
# =============================================================================

def uniform_sphere_cameras(
    n: int,
    radius: float = 2.0,
    device: str = "cuda",
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Sample cameras uniformly on a sphere using randn + normalize.
    
    Args:
        n: number of cameras
        radius: sphere radius
        device: torch device
        seed: random seed
    
    Returns:
        cameras: [n, 3] camera positions
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    dirs = torch.randn(n, 3, device=device)
    dirs = dirs / (dirs.norm(dim=1, keepdim=True) + 1e-8)
    return dirs * radius


def fibonacci_sphere_cameras(
    n: int,
    radius: float = 2.0,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Sample cameras using Fibonacci spiral (deterministic, uniform).
    
    Args:
        n: number of cameras
        radius: sphere radius
        device: torch device
    
    Returns:
        cameras: [n, 3] camera positions
    """
    indices = torch.arange(n, dtype=torch.float32, device=device)
    phi = torch.acos(1 - 2 * (indices + 0.5) / n)
    theta = torch.pi * (1 + 5**0.5) * indices
    
    x = torch.sin(phi) * torch.cos(theta)
    y = torch.sin(phi) * torch.sin(theta)
    z = torch.cos(phi)
    
    return torch.stack([x, y, z], dim=1) * radius


def hierarchical_cameras(
    bvh: MeshBVH,
    *,
    n_outer: int = 64,
    n_inner_per_stage: int = 64,
    n_stages: int = 1,
    outer_radius: float = 2.0,
    inner_radius: float = 1.8,
    min_visible_from_ref: int = 3,
    min_camera_distance: Optional[float] = None,
    surface_eps: float = 1e-3,
    max_iterations_per_stage: int = 20,
    batch_size: int = 64,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Sample cameras using hierarchical cascading expansion.
    
    Cameras are sampled in stages. Each stage uses all previously accumulated
    cameras as reference, enabling coverage of deep concave regions.
    
    Stage 0: outer sphere cameras (reference)
    Stage 1: interior cameras visible from outer
    Stage 2: interior cameras visible from (outer + stage1)
    ...
    
    Args:
        bvh: MeshBVH for visibility testing
        n_outer: number of outer sphere cameras
        n_inner_per_stage: target interior cameras per stage
        n_stages: number of cascading stages
        outer_radius: bounding sphere radius
        inner_radius: interior sampling radius
        min_visible_from_ref: min reference cameras that must see each camera
        min_camera_distance: min spacing (auto if None)
        surface_eps: surface tolerance
        max_iterations_per_stage: max iterations per stage
        batch_size: samples per iteration
        seed: random seed
    
    Returns:
        cameras: [n_outer + n_inner_per_stage * n_stages, 3] all cameras
    """
    device = bvh.vertices.device
    
    if seed is not None:
        torch.manual_seed(seed)
    
    # Outer cameras
    outer = uniform_sphere_cameras(n_outer, outer_radius, device)
    
    if n_stages == 0:
        return outer
    
    # Auto-compute spacing
    total_inner = n_inner_per_stage * n_stages
    if min_camera_distance is None:
        min_camera_distance = 1.6 * inner_radius / (total_inner ** (1/3))
    
    reference = outer
    all_inner = []
    
    for stage in range(n_stages):
        n_ref = reference.shape[0]
        vis_threshold = min_visible_from_ref / n_ref
        
        stage_cams = []
        n_collected = 0
        
        for _ in range(max_iterations_per_stage):
            if n_collected >= n_inner_per_stage:
                break
            
            # Sample candidates
            dirs = torch.randn(batch_size, 3, device=device)
            dirs = dirs / (dirs.norm(dim=1, keepdim=True) + 1e-8)
            radii = torch.rand(batch_size, 1, device=device) * inner_radius
            candidates = dirs * radii
            
            # Visibility filter
            vis = _query_visibility(bvh, candidates, reference, surface_eps)
            candidates = candidates[vis >= vis_threshold]
            
            if candidates.shape[0] == 0:
                continue
            
            # Distance filter vs existing
            existing = all_inner + stage_cams
            if len(existing) > 0:
                existing_t = torch.cat(existing, dim=0)
                dists = torch.cdist(candidates, existing_t)
                candidates = candidates[dists.min(dim=1).values >= min_camera_distance]
            
            # Greedy distance filter within batch
            if candidates.shape[0] > 1:
                keep = [0]
                for i in range(1, candidates.shape[0]):
                    if (candidates[i] - candidates[keep]).norm(dim=1).min() >= min_camera_distance:
                        keep.append(i)
                candidates = candidates[keep]
            
            if candidates.shape[0] > 0:
                stage_cams.append(candidates)
                n_collected += candidates.shape[0]
        
        if len(stage_cams) > 0:
            stage_inner = torch.cat(stage_cams, dim=0)[:n_inner_per_stage]
            all_inner.append(stage_inner)
            reference = torch.cat([outer] + all_inner, dim=0)
    
    if len(all_inner) > 0:
        return torch.cat([outer] + all_inner, dim=0)
    return outer


def _query_visibility(
    bvh: MeshBVH,
    points: torch.Tensor,
    cameras: torch.Tensor,
    surface_eps: float = 1e-3,
) -> torch.Tensor:
    """Query visibility of points from cameras."""
    N, M = points.shape[0], cameras.shape[0]
    device = points.device
    
    visible_count = torch.zeros(N, device=device)
    
    for cam in cameras:
        rays_o = cam.unsqueeze(0).expand(N, 3)
        rays_d = points - cam
        dists = rays_d.norm(dim=1, keepdim=True)
        rays_d = rays_d / (dists + 1e-8)
        
        result = bvh.intersect_ray(rays_o, rays_d, max_t=dists.max().item() + surface_eps)
        
        target_dist = dists.squeeze()
        visible = ~result.hit | (result.t >= target_dist - surface_eps)
        visible_count += visible.float()
    
    return visible_count / M


# =============================================================================
# VisibilityQuery Class
# =============================================================================

class VisibilityQuery:
    """
    Query point visibility from arbitrary camera configurations.
    
    The class provides a simple interface that accepts any camera positions.
    Use camera sampler functions to generate appropriate camera configurations.
    
    Args:
        bvh: MeshBVH instance for ray intersection
    
    Example:
        >>> bvh = MeshBVH(vertices, faces)
        >>> vis = VisibilityQuery(bvh)
        >>> 
        >>> # Simple: uniform sphere cameras
        >>> cameras = uniform_sphere_cameras(64, radius=2.0)
        >>> prob = vis.query(points, cameras)
        >>> 
        >>> # Advanced: hierarchical cameras for concave regions
        >>> cameras = hierarchical_cameras(bvh, n_outer=64, n_stages=3)
        >>> prob = vis.query(points, cameras)
    """
    
    def __init__(self, bvh: MeshBVH):
        self.bvh = bvh
    
    def query(
        self,
        points: torch.Tensor,
        cameras: torch.Tensor,
        surface_eps: float = 1e-3,
    ) -> torch.Tensor:
        """
        Query visibility of points from cameras.
        
        Casts rays from each camera to each point. A point is visible from
        a camera if the ray is unoccluded or hits the point's surface.
        
        Args:
            points: [N, 3] query positions
            cameras: [M, 3] camera positions (should be outside mesh)
            surface_eps: tolerance for surface points
        
        Returns:
            visibility: [N] probability in [0, 1] (fraction of cameras seeing point)
        """
        return _query_visibility(self.bvh, points, cameras, surface_eps)
    
    def query_directions(
        self,
        points: torch.Tensor,
        directions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Query visibility along specified directions (not from cameras).
        
        For each point, casts rays in all directions and returns the fraction
        of unoccluded directions.
        
        Args:
            points: [N, 3] query positions
            directions: [M, 3] normalized direction vectors
        
        Returns:
            visibility: [N] fraction of unoccluded directions
        """
        N, M = points.shape[0], directions.shape[0]
        device = points.device
        
        visible = torch.zeros(N, M, dtype=torch.bool, device=device)
        
        for j, d in enumerate(directions):
            result = self.bvh.intersect_ray(points, d.expand(N, 3))
            visible[:, j] = ~result.hit
        
        return visible.float().mean(dim=1)
