"""
MeshBVH: Core mesh class for Atom3D

Provides BVH acceleration and all collision/query primitives.
"""

from typing import Optional, Tuple, Union
import torch

from .data_structures import (
    AABBIntersectResult,
    RayIntersectResult,
    SegmentIntersectResult,
    ClosestPointResult,
    TriangleIntersectResult,
)

# Try to import CUDA kernels
try:
    from ..kernels import (
        cuda_available, 
        triangle_aabb_intersect, 
        ray_mesh_intersect, 
        point_mesh_udf,
        sat_clip_polygon
    )
    HAS_CUDA = cuda_available()
except ImportError:
    HAS_CUDA = False

# Try to import internal BVH accelerator
try:
    from ..kernels.bvh import BVHAccelerator, bvh_available
    HAS_BVH = bvh_available()
except ImportError:
    HAS_BVH = False
    BVHAccelerator = None


class MeshBVH:
    """
    Mesh with BVH acceleration structure.
    
    Provides all collision and query primitives:
    - intersect_aabb: Triangle-AABB intersection with optional polygon clipping
    - intersect_ray: Ray-mesh intersection
    - intersect_segment: Segment-mesh intersection
    - udf: Unsigned distance field query
    - sdf: Signed distance field query (requires watertight mesh)
    
    Args:
        vertices: [N, 3] float32 vertex coordinates
        faces: [M, 3] int32 triangle face indices
        device: Compute device ('cuda' or 'cpu')
    """
    
    def __init__(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        device: str = 'cuda',
        face_normals: Optional[torch.Tensor] = None,
        vertex_normals: Optional[torch.Tensor] = None
    ):
        """Initialize MeshBVH with optional pre-computed normals.
        
        Args:
            vertices: [V, 3] mesh vertices
            faces: [F, 3] mesh face indices
            device: 'cuda' or 'cpu'
            face_normals: [F, 3] pre-computed face normals (optional, will compute if None)
            vertex_normals: [V, 3] pre-computed vertex normals (optional, will compute if None)
        """
        self.vertices = vertices.to(device).float()
        self.faces = faces.to(device).int()
        self.device = device
        self.num_vertices = self.vertices.shape[0]
        self.num_faces = self.faces.shape[0]
        
        # Compute mesh bounds
        self._bounds = torch.stack([
            self.vertices.min(dim=0)[0],
            self.vertices.max(dim=0)[0]
        ])
        
        # Precompute face vertices for SAT clip
        self._face_verts_flat = None
        
        # Precompute or store face normals (FV)
        if face_normals is not None:
            self._face_normals = face_normals.to(device).float()
        else:
            self._face_normals = self._precompute_face_normals()
        
        # Precompute or store vertex normals (NV)
        if vertex_normals is not None:
            self._vertex_normals = vertex_normals.to(device).float()
        else:
            self._vertex_normals = None  # Lazy compute if needed
        
        # Build BVH
        self._build_bvh()
    
    def _build_bvh(self):
        """Build BVH acceleration structure."""
        self._bvh = None
        if HAS_BVH and self.device == 'cuda':
            try:
                self._bvh = BVHAccelerator(self.vertices, self.faces)
            except Exception as e:
                print(f"BVH build failed: {e}")
                self._bvh = None
    
    def _precompute_face_normals(self) -> torch.Tensor:
        """Precompute face normals for all faces.
        
        Returns:
            face_normals: [F, 3] normalized face normals
        """
        tri_verts = self.vertices[self.faces]  # [F, 3, 3]
        v0, v1, v2 = tri_verts[:, 0], tri_verts[:, 1], tri_verts[:, 2]
        normals = torch.cross(v1 - v0, v2 - v0, dim=1)
        normals = normals / (normals.norm(dim=1, keepdim=True) + 1e-8)
        return normals
    
    def _get_face_verts_flat(self) -> torch.Tensor:
        """Get flattened triangle vertices [M, 9] for SAT clip kernel."""
        if self._face_verts_flat is None:
            tri_verts = self.vertices[self.faces]  # [M, 3, 3]
            self._face_verts_flat = tri_verts.reshape(-1, 9)
        return self._face_verts_flat
    
    # ==================== Properties ====================
    
    def get_bounds(self) -> torch.Tensor:
        """
        Get mesh bounding box.
        
        Returns:
            bounds: [2, 3] [[min_xyz], [max_xyz]]
        """
        return self._bounds
    
    def get_face_normals(self) -> torch.Tensor:
        """
        Get pre-computed face normals.
        
        Returns:
            face_normals: [F, 3] normalized face normals
        """
        return self._face_normals
    
    def get_vertex_normals(self) -> Optional[torch.Tensor]:
        """
        Get pre-computed vertex normals (if available).
        
        Returns:
            vertex_normals: [V, 3] normalized vertex normals or None
        """
        return self._vertex_normals
    
    def get_face_aabb(
        self, 
        face_indices: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get per-triangle AABBs.
        
        Args:
            face_indices: [K] face indices (None = all faces)
        
        Returns:
            aabb_min: [K, 3]
            aabb_max: [K, 3]
        """
        if face_indices is None:
            faces = self.faces
        else:
            faces = self.faces[face_indices]
        
        tri_verts = self.vertices[faces]  # [K, 3, 3]
        aabb_min = tri_verts.min(dim=1)[0]
        aabb_max = tri_verts.max(dim=1)[0]
        
        return aabb_min, aabb_max
    
    # ==================== Triangle-AABB Intersection ====================
    
    def intersect_aabb(
        self,
        aabb_min: torch.Tensor,
        aabb_max: torch.Tensor,
        mode: int = 1
    ) -> AABBIntersectResult:
        """
        Triangle-AABB batch intersection using exact clipping.
        
        All modes now use the clip-based test to eliminate false positives
        from degenerate intersections (e.g., corner-only touches with area=0).
        
        Args:
            aabb_min: [N, 3] AABB min coordinates
            aabb_max: [N, 3] AABB max coordinates
            mode: Output mode
                0 = hit mask only (exact, no false positives)
                1 = hit mask + (aabb_id, face_id) pairs (default, exact)
                2 = hit mask + pairs + centroid + area
                3 = hit mask + pairs + centroid + area + polygon vertices
        
        Returns:
            result: AABBIntersectResult
                - hit: [N] bool
                - aabb_ids: [num_hits] int (if mode >= 1)
                - face_ids: [num_hits] int (if mode >= 1)
                - centroids: [num_hits, 3] float (if mode >= 2)
                - areas: [num_hits] float (if mode >= 2)
                - poly_verts: [num_hits, 8, 3] float (if mode == 3)
                - poly_counts: [num_hits] int (if mode == 3)
        """
        N = aabb_min.shape[0]
        
        # All modes now use clip-based test for accuracy
        # Mode 0/1 will use clip internally but only return hit/pairs
        return self._intersect_aabb_clip(aabb_min, aabb_max, mode)
    
    def _intersect_aabb_sat(
        self,
        aabb_min: torch.Tensor,
        aabb_max: torch.Tensor,
        return_pairs: bool
    ) -> AABBIntersectResult:
        """Basic SAT intersection (mode 0/1).
        
        Uses BVH-accelerated exact SAT test when available (O(N log M)).
        Falls back to brute-force CUDA kernel (O(N × M)).
        """
        N = aabb_min.shape[0]
        device = aabb_min.device
        
        # Use internal BVH (O(N log M))
        if self._bvh is not None:
            try:
                hit_mask, aabb_ids, face_ids = self._bvh.aabb_intersect(
                    aabb_min.contiguous(), aabb_max.contiguous()
                )
                if return_pairs:
                    return AABBIntersectResult(hit=hit_mask, aabb_ids=aabb_ids, face_ids=face_ids)
                else:
                    return AABBIntersectResult(hit=hit_mask)
            except Exception as e:
                print(f"BVH AABB intersect failed: {e}")
        
        # Fallback: brute-force CUDA kernel (O(N × M))
        if HAS_CUDA and self.device == 'cuda':
            try:
                hit_mask, aabb_ids, face_ids = triangle_aabb_intersect(
                    self.vertices, self.faces,
                    aabb_min.contiguous(), aabb_max.contiguous()
                )
                if return_pairs:
                    return AABBIntersectResult(hit=hit_mask, aabb_ids=aabb_ids, face_ids=face_ids)
                else:
                    return AABBIntersectResult(hit=hit_mask)
            except Exception as e:
                print(f"CUDA kernel failed: {e}")
        
        # Fallback: PyTorch broadphase
        return self._intersect_aabb_pytorch(aabb_min, aabb_max, return_pairs)
    
    def _intersect_aabb_clip(
        self,
        aabb_min: torch.Tensor,
        aabb_max: torch.Tensor,
        mode: int
    ) -> AABBIntersectResult:
        """SAT with polygon clipping for all modes (0-3).
        
        Uses BVH for fast broadphase, then sat_clip_polygon kernel for exact clipping.
        This eliminates false positives from degenerate intersections.
        """
        N = aabb_min.shape[0]
        device = aabb_min.device
        
        # Use BVH for fast broadphase (O(N log M) instead of O(N×M))
        if self._bvh is not None:
            try:
                hit_mask, cand_a, cand_t = self._bvh.aabb_intersect(
                    aabb_min.contiguous(), aabb_max.contiguous()
                )
                
                if cand_a.numel() == 0:
                    return AABBIntersectResult(
                        hit=torch.zeros(N, dtype=torch.bool, device=device)
                    )
            except Exception as e:
                print(f"BVH broadphase failed: {e}, falling back to brute-force")
                cand_a, cand_t = self._get_candidates_bruteforce(aabb_min, aabb_max)
        elif HAS_CUDA and self.device == 'cuda':
            # Fallback: brute-force CUDA kernel
            try:
                hit_mask, cand_a, cand_t = triangle_aabb_intersect(
                    self.vertices, self.faces,
                    aabb_min.contiguous(), aabb_max.contiguous()
                )
                
                if cand_a.numel() == 0:
                    return AABBIntersectResult(
                        hit=torch.zeros(N, dtype=torch.bool, device=device)
                    )
            except Exception as e:
                print(f"CUDA SAT kernel failed: {e}")
                cand_a, cand_t = self._chunked_broadphase(aabb_min, aabb_max)
        else:
            cand_a, cand_t = self._chunked_broadphase(aabb_min, aabb_max)
        
        if cand_a.numel() == 0:
            return AABBIntersectResult(
                hit=torch.zeros(N, dtype=torch.bool, device=device)
            )
        
        # Use sat_clip_polygon kernel
        if HAS_CUDA and self.device == 'cuda':
            try:
                tris_verts = self._get_face_verts_flat()
                # Mode mapping: 0/1 use mode 0 (hit only), 2 uses mode 1 (centroid), 3 uses mode 2 (polygon)
                clip_mode = 0 if mode <= 1 else (1 if mode == 2 else 2)
                
                hit_mask, poly_counts, poly_verts, centroids, areas, out_a, out_t = sat_clip_polygon(
                    aabb_min, aabb_max, tris_verts,
                    cand_a.long(), cand_t.long(),
                    mode=clip_mode
                )
                
                # Filter to actual hits (fixed bug: degenerate polygons now correctly marked as no-hit)
                valid = hit_mask
                
                # Build per-aabb hit mask
                aabb_hit = torch.zeros(N, dtype=torch.bool, device=device)
                aabb_hit[out_a[valid]] = True
                
                # Mode 0: hit mask only
                if mode == 0:
                    return AABBIntersectResult(hit=aabb_hit)
                
                # Mode 1+: include pairs
                result = AABBIntersectResult(
                    hit=aabb_hit,
                    aabb_ids=out_a[valid],
                    face_ids=out_t[valid]
                )
                
                # Mode 2+: add clip data
                if mode >= 2:
                    result.centroids = centroids[valid]
                    result.areas = areas[valid]
                    result.poly_counts = poly_counts[valid]
                
                # Mode 3: add polygon vertices
                if mode == 3:
                    result.poly_verts = poly_verts[valid]
                
                return result
                
            except Exception as e:
                print(f"SAT clip kernel failed: {e}")
        
        # Fallback: just return SAT result without clip data
        return self._intersect_aabb_sat(aabb_min, aabb_max, return_pairs=(mode >= 1))
    
    def _chunked_broadphase(
        self,
        aabb_min: torch.Tensor,
        aabb_max: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Chunked AABB-AABB overlap for CPU fallback."""
        N = aabb_min.shape[0]
        face_aabb_min, face_aabb_max = self.get_face_aabb()
        M = self.num_faces
        
        aabb_chunk = 50000
        tri_chunk = 20000
        
        all_cand_a = []
        all_cand_t = []
        
        for a0 in range(0, N, aabb_chunk):
            a1 = min(a0 + aabb_chunk, N)
            aabb_min_chunk = aabb_min[a0:a1]
            aabb_max_chunk = aabb_max[a0:a1]
            
            for t0 in range(0, M, tri_chunk):
                t1 = min(t0 + tri_chunk, M)
                face_min_chunk = face_aabb_min[t0:t1]
                face_max_chunk = face_aabb_max[t0:t1]
                
                overlap = (aabb_min_chunk[:, None, :] <= face_max_chunk[None, :, :]) & \
                          (aabb_max_chunk[:, None, :] >= face_min_chunk[None, :, :])
                overlap_all = overlap.all(dim=2)
                
                local_a, local_t = torch.where(overlap_all)
                if local_a.numel() > 0:
                    all_cand_a.append(local_a + a0)
                    all_cand_t.append(local_t + t0)
        
        if len(all_cand_a) == 0:
            device = aabb_min.device
            return torch.empty(0, dtype=torch.long, device=device), \
                   torch.empty(0, dtype=torch.long, device=device)
        
        return torch.cat(all_cand_a), torch.cat(all_cand_t)
    
    def _get_candidates_bruteforce(
        self,
        aabb_min: torch.Tensor,
        aabb_max: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get candidate pairs using brute-force SAT kernel."""
        if HAS_CUDA and self.device == 'cuda':
            try:
                hit_mask, cand_a, cand_t = triangle_aabb_intersect(
                    self.vertices, self.faces,
                    aabb_min.contiguous(), aabb_max.contiguous()
                )
                return cand_a, cand_t
            except Exception as e:
                print(f"Brute-force SAT failed: {e}")
        return self._chunked_broadphase(aabb_min, aabb_max)
    
    def _intersect_aabb_pytorch(
        self,
        aabb_min: torch.Tensor,
        aabb_max: torch.Tensor,
        return_pairs: bool
    ) -> AABBIntersectResult:
        """PyTorch AABB-AABB overlap (broadphase only)."""
        N = aabb_min.shape[0]
        device = aabb_min.device
        
        face_aabb_min, face_aabb_max = self.get_face_aabb()
        
        aabb_min_exp = aabb_min.unsqueeze(1)
        aabb_max_exp = aabb_max.unsqueeze(1)
        face_min_exp = face_aabb_min.unsqueeze(0)
        face_max_exp = face_aabb_max.unsqueeze(0)
        
        overlap = (aabb_min_exp <= face_max_exp) & (aabb_max_exp >= face_min_exp)
        overlap_all = overlap.all(dim=2)
        
        hit = overlap_all.any(dim=1)
        
        if return_pairs:
            aabb_ids, face_ids = torch.where(overlap_all)
            return AABBIntersectResult(hit=hit, aabb_ids=aabb_ids, face_ids=face_ids)
        else:
            return AABBIntersectResult(hit=hit)
    
    # ==================== UDF/SDF ====================
    
    def udf(
        self,
        points: torch.Tensor,
        return_grad: bool = False,
        return_closest: bool = False,
        return_uvw: bool = False,
        return_face_ids: bool = False
    ) -> Union[torch.Tensor, ClosestPointResult]:
        """
        Unsigned Distance Field query.
        
        Args:
            points: [N, 3] query points
            return_grad: If True, distances will have grad_fn (requires points.requires_grad=True)
            return_closest: If True, include closest_points in result
            return_uvw: If True, include uvw barycentric coordinates in result
            return_face_ids: If True, include face_ids in result
        
        Returns:
            If all return_* are False: distances [N] tensor
            Otherwise: ClosestPointResult with requested fields
        
        Example:
            # Simple: just distances
            distances = bvh.udf(points)
            
            # With extras
            result = bvh.udf(points, return_closest=True)
            print(result.distances, result.closest_points)
        """
        if return_grad:
            result = self._udf_with_grad(points)
        else:
            result = self._udf_query(points)
        
        # If no extras requested, return tensor directly
        if not return_closest and not return_uvw and not return_face_ids:
            return result.distances
        
        # Filter result based on options
        if not return_closest:
            result.closest_points = None
        if not return_uvw:
            result.uvw = None
        if not return_face_ids:
            result.face_ids = None
        
        return result
    
    def _udf_query(self, points: torch.Tensor) -> ClosestPointResult:
        """UDF query without gradient."""
        N = points.shape[0]
        device = points.device
        
        # Use internal BVH (O(N log M))
        if self._bvh is not None:
            try:
                distances, face_ids, closest_points, uvw = self._bvh.udf(points.contiguous())
                return ClosestPointResult(
                    distances=distances,
                    face_ids=face_ids,
                    closest_points=closest_points,
                    uvw=uvw
                )
            except Exception as e:
                print(f"BVH UDF failed: {e}")
        
        # Fallback: brute-force CUDA kernel (O(N × M))
        if HAS_CUDA and self.device == 'cuda':
            try:
                distances, face_ids, closest_points, uvw = point_mesh_udf(
                    self.vertices, self.faces, points.contiguous()
                )
                return ClosestPointResult(
                    distances=distances,
                    face_ids=face_ids,
                    closest_points=closest_points,
                    uvw=uvw
                )
            except Exception as e:
                print(f"CUDA UDF kernel failed: {e}")
        
        return self._udf_bruteforce(points)
    
    def _udf_with_grad(self, points: torch.Tensor) -> ClosestPointResult:
        """UDF query with gradient support."""
        with torch.no_grad():
            result = self._udf_query(points)
        
        # Recompute distance with gradient
        closest_points = result.closest_points.detach()
        diff = points - closest_points
        distances = diff.norm(dim=1)
        
        return ClosestPointResult(
            distances=distances,
            face_ids=result.face_ids,
            closest_points=closest_points,
            uvw=result.uvw
        )
    
    def sdf(
        self,
        points: torch.Tensor,
        return_grad: bool = False,
        return_closest: bool = False,
        return_uvw: bool = False,
        return_face_ids: bool = False
    ) -> Union[torch.Tensor, ClosestPointResult]:
        """
        Signed Distance Field query.
        
        Requires watertight mesh with consistent winding.
        Sign is determined by face normal direction.
        
        Args:
            points: [N, 3] query points
            return_grad: If True, distances will have grad_fn
            return_closest: If True, include closest_points in result
            return_uvw: If True, include uvw barycentric coordinates in result
            return_face_ids: If True, include face_ids in result
        
        Returns:
            If all return_* are False: distances [N] tensor (signed)
            Otherwise: ClosestPointResult with requested fields
        """
        # Need full result for sign computation
        result = self._udf_query(points) if not return_grad else self._udf_with_grad(points)
        
        # Compute sign using closest face normal
        signs = self._compute_sign(points, result.closest_points, result.face_ids)
        result.distances = result.distances * signs
        
        # If no extras requested, return tensor directly
        if not return_closest and not return_uvw and not return_face_ids:
            return result.distances
        
        # Filter result based on options
        if not return_closest:
            result.closest_points = None
        if not return_uvw:
            result.uvw = None
        if not return_face_ids:
            result.face_ids = None
        
        return result
    
    def _compute_sign(
        self,
        points: torch.Tensor,
        closest_points: torch.Tensor,
        face_ids: torch.Tensor
    ) -> torch.Tensor:
        """Compute SDF sign based on cached face normals."""
        device = points.device
        N = points.shape[0]
        
        # Get cached face normals
        valid = face_ids >= 0
        signs = torch.ones(N, device=device)
        
        if valid.any():
            # Use cached face normals instead of recomputing
            normals = self._face_normals[face_ids[valid]]
            
            # Sign = dot(point - closest, normal)
            to_point = points[valid] - closest_points[valid]
            dot = (to_point * normals).sum(dim=1)
            signs[valid] = torch.sign(dot)
            signs[valid][signs[valid] == 0] = 1
        
        return signs
    
    def _compute_barycentric(
        self,
        points: torch.Tensor,
        closest_points: torch.Tensor,
        face_ids: torch.Tensor
    ) -> torch.Tensor:
        """Compute barycentric coordinates."""
        N = points.shape[0]
        device = points.device
        uvw = torch.zeros(N, 3, device=device)
        
        valid = face_ids >= 0
        if valid.any():
            valid_faces = face_ids[valid]
            valid_closest = closest_points[valid]
            
            tri_verts = self.vertices[self.faces[valid_faces]]
            v0, v1, v2 = tri_verts[:, 0], tri_verts[:, 1], tri_verts[:, 2]
            
            v0v1 = v1 - v0
            v0v2 = v2 - v0
            v0p = valid_closest - v0
            
            d00 = (v0v1 * v0v1).sum(dim=1)
            d01 = (v0v1 * v0v2).sum(dim=1)
            d11 = (v0v2 * v0v2).sum(dim=1)
            d20 = (v0p * v0v1).sum(dim=1)
            d21 = (v0p * v0v2).sum(dim=1)
            
            denom = d00 * d11 - d01 * d01
            v = (d11 * d20 - d01 * d21) / (denom + 1e-8)
            w = (d00 * d21 - d01 * d20) / (denom + 1e-8)
            u = 1.0 - v - w
            
            uvw[valid, 0] = u
            uvw[valid, 1] = v
            uvw[valid, 2] = w
        
        return uvw
    
    def _udf_bruteforce(self, points: torch.Tensor) -> ClosestPointResult:
        """Brute force UDF (fallback)."""
        N = points.shape[0]
        device = points.device
        
        distances = torch.full((N,), float('inf'), device=device)
        face_ids = torch.full((N,), -1, dtype=torch.int32, device=device)
        closest_points = torch.zeros(N, 3, device=device)
        
        tri_verts = self.vertices[self.faces]
        
        for i in range(N):
            dists, closest = self._point_to_triangles(points[i], tri_verts)
            best_idx = dists.argmin()
            distances[i] = dists[best_idx]
            face_ids[i] = best_idx
            closest_points[i] = closest[best_idx]
        
        uvw = self._compute_barycentric(points, closest_points, face_ids)
        
        return ClosestPointResult(
            distances=distances,
            face_ids=face_ids,
            closest_points=closest_points,
            uvw=uvw
        )
    
    def _point_to_triangles(
        self,
        point: torch.Tensor,
        tri_verts: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute distance from point to all triangles."""
        v0, v1, v2 = tri_verts[:, 0], tri_verts[:, 1], tri_verts[:, 2]
        
        edge0 = v1 - v0
        edge1 = v2 - v0
        v0_to_p = point - v0
        
        a = (edge0 * edge0).sum(dim=1)
        b = (edge0 * edge1).sum(dim=1)
        c = (edge1 * edge1).sum(dim=1)
        d = (edge0 * v0_to_p).sum(dim=1)
        e = (edge1 * v0_to_p).sum(dim=1)
        
        det = a * c - b * b
        s = b * e - c * d
        t = b * d - a * e
        
        s = torch.clamp(s / (det + 1e-8), 0, 1)
        t = torch.clamp(t / (det + 1e-8), 0, 1)
        
        sum_st = s + t
        mask = sum_st > 1
        s[mask] = s[mask] / sum_st[mask]
        t[mask] = t[mask] / sum_st[mask]
        
        closest = v0 + s.unsqueeze(1) * edge0 + t.unsqueeze(1) * edge1
        distances = (point - closest).norm(dim=1)
        
        return distances, closest
    
    # ==================== Ray Intersection ====================
    
    def intersect_ray(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        max_t: float = 1e10
    ) -> RayIntersectResult:
        """
        Ray-mesh intersection.
        
        Args:
            rays_o: [N, 3] ray origins
            rays_d: [N, 3] ray directions (normalized)
            max_t: Maximum distance
        
        Returns:
            result: RayIntersectResult
        """
        N = rays_o.shape[0]
        device = rays_o.device
        
        # Use internal BVH (O(N log M))
        if self._bvh is not None:
            try:
                hit_mask, hit_t, face_ids, hit_points = self._bvh.ray_intersect(
                    rays_o.contiguous(), rays_d.contiguous(), max_t
                )
                normals = self._compute_normals_from_faces(face_ids, hit_mask)
                bary_coords = torch.zeros(N, 3, device=device)
                
                return RayIntersectResult(
                    hit=hit_mask,
                    t=hit_t,
                    face_ids=face_ids,
                    hit_points=hit_points,
                    normals=normals,
                    bary_coords=bary_coords
                )
            except Exception as e:
                print(f"BVH ray intersect failed: {e}")
        
        # Fallback: brute-force CUDA kernel (O(N × M))
        if HAS_CUDA and self.device == 'cuda':
            try:
                from ..kernels import ray_mesh_intersect
                hit_mask, hit_t, hit_face_ids, hit_points, hit_uvs = ray_mesh_intersect(
                    self.vertices, self.faces,
                    rays_o.contiguous(), rays_d.contiguous(), max_t
                )
                
                normals = self._compute_normals_from_faces(hit_face_ids, hit_mask)
                bary_coords = torch.zeros(N, 3, device=device)
                bary_coords[:, 1] = hit_uvs[:, 0]
                bary_coords[:, 2] = hit_uvs[:, 1]
                bary_coords[:, 0] = 1.0 - hit_uvs[:, 0] - hit_uvs[:, 1]
                
                return RayIntersectResult(
                    hit=hit_mask,
                    t=hit_t,
                    face_ids=hit_face_ids,
                    hit_points=hit_points,
                    normals=normals,
                    bary_coords=bary_coords
                )
            except Exception as e:
                print(f"CUDA ray kernel failed: {e}")
        
        # Fallback
        return self._intersect_ray_bruteforce(rays_o, rays_d, max_t)
    
    def _compute_normals_from_faces(
        self, 
        face_ids: torch.Tensor, 
        valid_mask: torch.Tensor
    ) -> torch.Tensor:
        """Get normals from cached face normals using face indices.
        
        Note: Now uses pre-computed face normals instead of recomputing.
        """
        N = face_ids.shape[0]
        device = face_ids.device
        normals = torch.zeros(N, 3, device=device)
        
        if valid_mask.any():
            valid_faces = face_ids[valid_mask]
            # Use cached face normals instead of recomputing
            normals[valid_mask] = self._face_normals[valid_faces]
        
        return normals
    
    def _intersect_ray_bruteforce(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        max_t: float
    ) -> RayIntersectResult:
        """Brute force ray intersection (fallback)."""
        N = rays_o.shape[0]
        device = rays_o.device
        
        hit = torch.zeros(N, dtype=torch.bool, device=device)
        t = torch.full((N,), float('inf'), device=device)
        face_ids = torch.full((N,), -1, dtype=torch.int32, device=device)
        hit_points = torch.zeros(N, 3, device=device)
        normals = torch.zeros(N, 3, device=device)
        bary_coords = torch.zeros(N, 3, device=device)
        
        tri_verts = self.vertices[self.faces]
        v0, v1, v2 = tri_verts[:, 0], tri_verts[:, 1], tri_verts[:, 2]
        
        for i in range(N):
            ray_o = rays_o[i]
            ray_d = rays_d[i]
            
            edge1 = v1 - v0
            edge2 = v2 - v0
            h = torch.cross(ray_d.expand_as(edge1), edge2)
            a = (edge1 * h).sum(dim=1)
            
            valid = torch.abs(a) > 1e-8
            f = torch.zeros_like(a)
            f[valid] = 1.0 / a[valid]
            
            s = ray_o - v0
            u = f * (s * h).sum(dim=1)
            valid &= (u >= 0) & (u <= 1)
            
            q = torch.cross(s, edge1)
            v = f * (ray_d.expand_as(q) * q).sum(dim=1)
            valid &= (v >= 0) & (u + v <= 1)
            
            t_hit = f * (edge2 * q).sum(dim=1)
            valid &= (t_hit > 1e-6) & (t_hit < max_t)
            
            if valid.any():
                t_hit[~valid] = float('inf')
                best_idx = t_hit.argmin()
                if t_hit[best_idx] < t[i]:
                    t[i] = t_hit[best_idx]
                    face_ids[i] = best_idx
                    hit[i] = True
                    hit_points[i] = ray_o + t[i] * ray_d
                    face_normal = torch.cross(edge1[best_idx], edge2[best_idx])
                    normals[i] = face_normal / (face_normal.norm() + 1e-8)
        
        return RayIntersectResult(
            hit=hit, t=t, face_ids=face_ids,
            hit_points=hit_points, normals=normals, bary_coords=bary_coords
        )
    
    # ==================== Segment Intersection ====================
    
    def intersect_segment(
        self,
        seg_start: torch.Tensor,
        seg_end: torch.Tensor
    ) -> SegmentIntersectResult:
        """
        Segment-mesh intersection.
        
        Args:
            seg_start: [N, 3]
            seg_end: [N, 3]
        
        Returns:
            result: SegmentIntersectResult
        """
        rays_o = seg_start
        rays_d = seg_end - seg_start
        seg_lengths = rays_d.norm(dim=1, keepdim=True)
        rays_d = rays_d / (seg_lengths + 1e-8)
        
        ray_result = self.intersect_ray(rays_o, rays_d, max_t=seg_lengths.squeeze().max().item())
        hit = ray_result.hit & (ray_result.t <= seg_lengths.squeeze())
        
        return SegmentIntersectResult(
            hit=hit,
            hit_points=ray_result.hit_points,
            face_ids=ray_result.face_ids,
            bary_coords=ray_result.bary_coords
        )
    
    # ==================== Triangle-Triangle Intersection ====================
    
    def intersect_triangles(
        self,
        other_vertices: torch.Tensor,
        other_faces: torch.Tensor
    ) -> TriangleIntersectResult:
        """
        Mesh-mesh collision detection.
        
        Args:
            other_vertices: [M, 3]
            other_faces: [K, 3]
        
        Returns:
            result: TriangleIntersectResult
        """
        device = other_vertices.device
        other_vertices = other_vertices.to(self.device)
        other_faces = other_faces.to(self.device)
        
        K = other_faces.shape[0]
        edge_v0_idx = other_faces[:, [0, 1, 2]].reshape(-1)
        edge_v1_idx = other_faces[:, [1, 2, 0]].reshape(-1)
        
        edge_starts = other_vertices[edge_v0_idx]
        edge_ends = other_vertices[edge_v1_idx]
        
        seg_result = self.intersect_segment(edge_starts, edge_ends)
        
        edge_hit = seg_result.hit
        hit_indices = torch.where(edge_hit)[0]
        
        return TriangleIntersectResult(
            edge_hit=edge_hit,
            hit_points=seg_result.hit_points[hit_indices],
            hit_face_ids=seg_result.face_ids[hit_indices],
            hit_edge_ids=hit_indices
        )
