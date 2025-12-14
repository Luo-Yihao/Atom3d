"""
UDFQuery: Unsigned Distance Field query with gradient support.
"""

from typing import Optional
import torch

from ..core.mesh_bvh import MeshBVH
from ..core.data_structures import ClosestPointResult


class UDFQuery:
    """
    Unsigned Distance Field query application.
    
    Wraps MeshBVH.query_closest_point with:
    - Gradient support via autograd
    - Batch processing to avoid OOM
    
    Args:
        bvh: MeshBVH instance
    """
    
    def __init__(self, bvh: MeshBVH):
        self.bvh = bvh
    
    def query(
        self,
        points: torch.Tensor,
        compute_grad: bool = False,
        batch_size: Optional[int] = None
    ) -> ClosestPointResult:
        """
        Query unsigned distance field.
        
        Args:
            points: [N, 3] query points
                If compute_grad=True, should have requires_grad=True
            compute_grad: Whether to enable gradient computation
            batch_size: Batch size (None = process all at once)
        
        Returns:
            result: ClosestPointResult
                - distances: [N] float32
                - face_ids: [N] int32
                - closest_points: [N, 3]
                - uvw: [N, 3]
        """
        if batch_size is not None and points.shape[0] > batch_size:
            return self._query_batched(points, compute_grad, batch_size)
        
        if compute_grad:
            return self._query_with_grad(points)
        else:
            return self.bvh.query_closest_point(points, return_uvw=True)
    
    def _query_batched(
        self,
        points: torch.Tensor,
        compute_grad: bool,
        batch_size: int
    ) -> ClosestPointResult:
        """Batched query to avoid OOM."""
        N = points.shape[0]
        
        all_distances = []
        all_face_ids = []
        all_closest_points = []
        all_uvw = []
        
        for i in range(0, N, batch_size):
            batch_points = points[i:i+batch_size]
            
            if compute_grad:
                result = self._query_with_grad(batch_points)
            else:
                result = self.bvh.query_closest_point(batch_points, return_uvw=True)
            
            all_distances.append(result.distances)
            all_face_ids.append(result.face_ids)
            all_closest_points.append(result.closest_points)
            if result.uvw is not None:
                all_uvw.append(result.uvw)
        
        return ClosestPointResult(
            distances=torch.cat(all_distances),
            face_ids=torch.cat(all_face_ids),
            closest_points=torch.cat(all_closest_points),
            uvw=torch.cat(all_uvw) if all_uvw else None
        )
    
    def _query_with_grad(self, points: torch.Tensor) -> ClosestPointResult:
        """
        Query with gradient computation.
        
        Gradient: d(distance)/d(point) = (point - closest_point) / distance
        """
        # Get closest points (no grad)
        with torch.no_grad():
            result = self.bvh.query_closest_point(points, return_uvw=True)
        
        # Compute distances with gradient
        closest_points = result.closest_points.detach()
        diff = points - closest_points
        distances = diff.norm(dim=1)
        
        # If input requires grad, the output distances will have grad_fn
        return ClosestPointResult(
            distances=distances,
            face_ids=result.face_ids,
            closest_points=closest_points,
            uvw=result.uvw
        )
