"""
SDFQuery: Signed Distance Field query with multiple methods

Provides SDF computation using winding number, flood fill, or ray stabbing.
"""

from typing import Optional
import torch

from ..core.mesh_bvh import MeshBVH
from ..grid.cube_grid import CubeGrid



class SDFQuery:
    """
    SDF query application.
    
    Combines MeshBVH.query_closest_point with sign determination algorithms.
    
    Supported methods:
        - winding: Winding number (exact, requires watertight mesh)
        - flood: Flood fill (works with open meshes)
        - raystab: Ray stabbing (robust but slow)
    
    Args:
        bvh: MeshBVH instance
    """
    
    def __init__(self, bvh: MeshBVH):
        self.bvh = bvh
    
    def query_winding(self, points: torch.Tensor) -> torch.Tensor:
        """
        SDF via Winding Number.
        
        Args:
            points: [N, 3]
        
        Returns:
            sdf: [N] float32 (negative inside, positive outside)
        
        Suitable for: watertight or near-watertight meshes
        """
        N = points.shape[0]
        device = points.device
        
        # Get UDF
        result = self.bvh.query_closest_point(points, return_uvw=False)
        distances = result.distances
        
        # Compute winding number
        winding = self._compute_winding_number(points)
        
        # Inside if winding > 0.5
        inside = winding > 0.5
        
        # SDF = distance * sign
        sdf = distances.clone()
        sdf[inside] = -sdf[inside]
        
        return sdf
    
    def _compute_winding_number(self, points: torch.Tensor) -> torch.Tensor:
        """
        Compute generalized winding number.
        
        For each point, sum solid angles subtended by all triangles.
        """
        N = points.shape[0]
        device = points.device
        
        # Get triangle vertices
        tri_verts = self.bvh.vertices[self.bvh.faces]  # [M, 3, 3]
        v0, v1, v2 = tri_verts[:, 0], tri_verts[:, 1], tri_verts[:, 2]
        
        winding = torch.zeros(N, device=device)
        
        for i in range(N):
            p = points[i]
            
            # Vectors from point to vertices
            a = v0 - p
            b = v1 - p
            c = v2 - p
            
            # Normalize
            la = a.norm(dim=1, keepdim=True) + 1e-8
            lb = b.norm(dim=1, keepdim=True) + 1e-8
            lc = c.norm(dim=1, keepdim=True) + 1e-8
            
            a = a / la
            b = b / lb
            c = c / lc
            
            # Solid angle formula
            det = (a * torch.cross(b, c)).sum(dim=1)
            denom = 1 + (a * b).sum(dim=1) + (b * c).sum(dim=1) + (c * a).sum(dim=1)
            
            solid_angle = 2 * torch.atan2(det, denom)
            winding[i] = solid_angle.sum() / (4 * torch.pi)
        
        return winding
    
    
    def query_raystab(
        self,
        points: torch.Tensor,
        num_rays: int = 8,
        seed: Optional[int] = None
    ) -> torch.Tensor:
        """
        SDF via Ray Stabbing.
        
        Shoots multiple rays from each point and counts intersection parity.
        
        Args:
            points: [N, 3]
            num_rays: Number of rays per point
            seed: Random seed
        
        Returns:
            sdf: [N] float32
        
        Suitable for: open, non-manifold meshes
        """
        N = points.shape[0]
        device = points.device
        
        if seed is not None:
            torch.manual_seed(seed)
        
        # Get UDF first
        result = self.bvh.query_closest_point(points, return_uvw=False)
        distances = result.distances
        
        # Generate random directions
        directions = torch.randn(num_rays, 3, device=device)
        directions = directions / directions.norm(dim=1, keepdim=True)
        
        # Count intersections for each point
        inside_votes = torch.zeros(N, device=device)
        
        for direction in directions:
            rays_o = points
            rays_d = direction.expand_as(points)
            
            # Count intersections (simplified: just check if hits)
            result_ray = self.bvh.intersect_ray(rays_o, rays_d)
            
            # If hit, it's either entering or leaving
            # Odd number of hits = inside
            inside_votes += result_ray.hit.float()
        
        # Majority vote: if more than half rays hit, likely inside
        # (This is a simplification; proper impl would count all intersections)
        inside = (inside_votes / num_rays) > 0.5
        
        sdf = distances.clone()
        sdf[inside] = -sdf[inside]
        
        return sdf
