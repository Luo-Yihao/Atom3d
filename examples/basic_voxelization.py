#!/usr/bin/env python3
"""
Basic Mesh Voxelization Example

Demonstrates octree-accelerated mesh voxelization with SAT intersection.
"""

import torch
import time

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from atom3d import MeshBVH
from atom3d.grid import OctreeIndexer


def create_test_mesh():
    """Create a simple icosphere mesh."""
    try:
        import trimesh
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.8)
        vertices = torch.tensor(mesh.vertices, dtype=torch.float32)
        faces = torch.tensor(mesh.faces, dtype=torch.int64)
        return vertices, faces
    except ImportError:
        raise RuntimeError("trimesh required: pip install trimesh")


def main():
    print("=" * 50)
    print("Atom3D: Basic Voxelization Example")
    print("=" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Load mesh
    vertices, faces = create_test_mesh()
    vertices = vertices.to(device)
    faces = faces.to(device)
    print(f"Mesh: {len(vertices)} vertices, {len(faces)} faces")
    
    # Create BVH
    print("\n[1] Building BVH...")
    t0 = time.time()
    bvh = MeshBVH(vertices, faces, device=device)
    print(f"    Time: {time.time()-t0:.3f}s")
    print(f"    Bounds: {bvh.get_bounds()[0].tolist()}")
    
    # Create octree
    max_level = 7  # 128^3 resolution
    print(f"\n[2] Creating octree (level {max_level}, res {2**max_level})...")
    octree = OctreeIndexer(max_level=max_level, device=device)
    
    # Octree traversal using BVH-accelerated broadphase
    print("\n[3] Octree traversal (BVH-accelerated broadphase)...")
    t0 = time.time()
    
    min_level = 4
    candidates = octree.octree_traverse(bvh, min_level=min_level)
    
    print(f"    Candidates at level {max_level}: {len(candidates)}")
    print(f"    Broadphase time: {time.time()-t0:.3f}s")
    
    # SAT intersection (narrowphase)
    print("\n[4] SAT intersection (narrowphase)...")
    t0 = time.time()
    
    voxel_min, voxel_max = octree.cube_aabb_level(candidates, max_level)
    result = bvh.intersect_aabb(voxel_min, voxel_max, mode=1)
    
    surface_voxels = candidates[result.hit]
    print(f"    Surface voxels: {len(surface_voxels)}")
    print(f"    SAT time: {time.time()-t0:.3f}s")
    print(f"    Reduction: {len(candidates)} -> {len(surface_voxels)} ({100*(1-len(surface_voxels)/len(candidates)):.1f}%)")
    
    # UDF at voxel corners
    print("\n[5] UDF at voxel corners...")
    t0 = time.time()
    
    corners = octree.cube_corner_coords_level(surface_voxels[:100], max_level)  # First 100
    corner_points = corners.reshape(-1, 3)
    
    udf_distances = bvh.udf(corner_points)  # Returns tensor directly when no extras requested
    print(f"    UDF range: [{udf_distances.min():.4f}, {udf_distances.max():.4f}]")
    print(f"    UDF time: {time.time()-t0:.3f}s")
    
    # Polygon clipping (mode 2)
    print("\n[6] Polygon clipping (mode=2)...")
    t0 = time.time()
    
    clip_result = bvh.intersect_aabb(voxel_min[:100], voxel_max[:100], mode=2)
    if hasattr(clip_result, 'centroids'):
        print(f"    Centroids shape: {clip_result.centroids.shape}")
        print(f"    Areas range: [{clip_result.areas.min():.6f}, {clip_result.areas.max():.6f}]")
    print(f"    Clip time: {time.time()-t0:.3f}s")
    
    print("\n" + "=" * 50)
    print("Done!")
    print("=" * 50)


if __name__ == "__main__":
    main()
