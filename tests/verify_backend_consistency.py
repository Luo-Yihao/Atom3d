#!/usr/bin/env python3
"""
Verify that CUDA and Python backends produce equivalent results.

Usage:
    python verify_backend_consistency.py <mesh_path> [--resolution 512]
"""

import argparse
import torch
import numpy as np

def verify_backends(mesh_path: str, resolution_level: int = 9):
    """
    Compare CUDA and Python flood fill results.
    
    Args:
        mesh_path: Path to mesh file (.obj, .glb, etc.)
        resolution_level: Octree max level (resolution = 2^level)
    """
    from atom3d.core.mesh_bvh import MeshBVH
    from atom3d.grid.octree_indexer import OctreeIndexer
    from atom3d.apps.sparse_flood_fill import sparse_flood_fill
    
    print(f"Loading mesh: {mesh_path}")
    print(f"Resolution level: {resolution_level} (grid: {2**resolution_level}³)")
    
    # Load mesh and build BVH
    import trimesh
    mesh = trimesh.load(mesh_path, force='mesh')
    vertices = torch.from_numpy(mesh.vertices.astype(np.float32)).cuda()
    faces = torch.from_numpy(mesh.faces.astype(np.int32)).cuda()
    
    # Normalize to [-0.9, 0.9] to ensure source (0,0,0) is outside
    center = (vertices.max(0)[0] + vertices.min(0)[0]) / 2
    scale = (vertices.max(0)[0] - vertices.min(0)[0]).max() / 1.8
    vertices = (vertices - center) / scale
    
    bvh = MeshBVH(vertices, faces)
    octree = OctreeIndexer(max_level=resolution_level, device='cuda')
    
    source = (0, 0, 0)
    resolution = 2 ** resolution_level
    
    print("\n" + "="*60)
    print("Running CUDA backend...")
    print("="*60)
    result_cuda = sparse_flood_fill(bvh, octree, source=source, backend='cuda', connectivity=26, verbose=True)
    
    print("\n" + "="*60)
    print("Running Python backend...")
    print("="*60)
    result_python = sparse_flood_fill(bvh, octree, source=source, backend='python', connectivity=26, verbose=True)
    
    # ============================================================
    # Compare dam_ijk
    # ============================================================
    print("\n" + "="*60)
    print("Comparing dam_ijk...")
    print("="*60)
    
    dam_cuda = result_cuda['dam_ijk']
    dam_python = result_python['dam_ijk']
    
    dam_cuda_set = set(tuple(x.tolist()) for x in dam_cuda.cpu())
    dam_python_set = set(tuple(x.tolist()) for x in dam_python.cpu())
    
    dam_intersection = dam_cuda_set & dam_python_set
    dam_only_cuda = dam_cuda_set - dam_python_set
    dam_only_python = dam_python_set - dam_cuda_set
    
    print(f"CUDA dam:   {len(dam_cuda_set)}")
    print(f"Python dam: {len(dam_python_set)}")
    print(f"Intersection: {len(dam_intersection)}")
    print(f"Only in CUDA: {len(dam_only_cuda)}")
    print(f"Only in Python: {len(dam_only_python)}")
    
    if dam_cuda_set == dam_python_set:
        print("✅ dam_ijk: MATCH")
    else:
        print("❌ dam_ijk: MISMATCH")
    
    # ============================================================
    # Compare collision_ijk
    # ============================================================
    print("\n" + "="*60)
    print("Comparing collision_ijk...")
    print("="*60)
    
    collision_cuda = result_cuda['collision_ijk']
    collision_python = result_python['collision_ijk']
    
    collision_cuda_set = set(tuple(x.tolist()) for x in collision_cuda.cpu())
    collision_python_set = set(tuple(x.tolist()) for x in collision_python.cpu())
    
    print(f"CUDA collision:   {len(collision_cuda_set)}")
    print(f"Python collision: {len(collision_python_set)}")
    
    if collision_cuda_set == collision_python_set:
        print("✅ collision_ijk: MATCH")
    else:
        collision_intersection = collision_cuda_set & collision_python_set
        print(f"Intersection: {len(collision_intersection)}")
        print("❌ collision_ijk: MISMATCH (may be acceptable due to implementation differences)")
    
    # ============================================================
    # Compare water_octree_node (expand Python to max_level)
    # ============================================================
    print("\n" + "="*60)
    print("Comparing water_octree_node (after expanding Python to max_level)...")
    print("="*60)
    
    water_cuda = result_cuda['water_octree_node']
    water_python = result_python['water_octree_node']
    
    # Expand Python hierarchical nodes to max_level
    water_python_expanded = octree.expand_nodes(water_python)
    
    # Convert CUDA nodes to ijk
    water_cuda_ijk = octree.cube_to_ijk_level(water_cuda[:, 1], level=resolution_level)
    
    water_cuda_set = set(tuple(x.tolist()) for x in water_cuda_ijk.cpu())
    water_python_set = set(tuple(x.tolist()) for x in water_python_expanded.cpu())
    
    print(f"CUDA water (raw nodes):     {len(water_cuda)}")
    print(f"Python water (raw nodes):   {len(water_python)}")
    print(f"CUDA water (as ijk):        {len(water_cuda_set)}")
    print(f"Python water (expanded):    {len(water_python_set)}")
    
    water_intersection = water_cuda_set & water_python_set
    water_only_cuda = water_cuda_set - water_python_set
    water_only_python = water_python_set - water_cuda_set
    
    print(f"Intersection: {len(water_intersection)}")
    print(f"Only in CUDA: {len(water_only_cuda)}")
    print(f"Only in Python: {len(water_only_python)}")
    
    if water_cuda_set == water_python_set:
        print("✅ water_octree_node: MATCH")
    else:
        print("❌ water_octree_node: MISMATCH")
    
    # ============================================================
    # Compare dry_octree_node (expand Python to max_level)
    # ============================================================
    print("\n" + "="*60)
    print("Comparing dry_octree_node (after expanding Python to max_level)...")
    print("="*60)
    
    dry_cuda = result_cuda['dry_octree_node']
    dry_python = result_python['dry_octree_node']
    
    # Expand Python hierarchical nodes to max_level
    dry_python_expanded = octree.expand_nodes(dry_python)
    
    # Convert CUDA nodes to ijk
    if len(dry_cuda) > 0:
        dry_cuda_ijk = octree.cube_to_ijk_level(dry_cuda[:, 1], level=resolution_level)
        dry_cuda_set = set(tuple(x.tolist()) for x in dry_cuda_ijk.cpu())
    else:
        dry_cuda_set = set()
    
    dry_python_set = set(tuple(x.tolist()) for x in dry_python_expanded.cpu())
    
    print(f"CUDA dry (raw nodes):     {len(dry_cuda)}")
    print(f"Python dry (raw nodes):   {len(dry_python)}")
    print(f"CUDA dry (as ijk):        {len(dry_cuda_set)}")
    print(f"Python dry (expanded):    {len(dry_python_set)}")
    
    dry_intersection = dry_cuda_set & dry_python_set
    dry_only_cuda = dry_cuda_set - dry_python_set
    dry_only_python = dry_python_set - dry_cuda_set
    
    print(f"Intersection: {len(dry_intersection)}")
    print(f"Only in CUDA: {len(dry_only_cuda)}")
    print(f"Only in Python: {len(dry_only_python)}")
    
    if dry_cuda_set == dry_python_set:
        print("✅ dry_octree_node: MATCH")
    else:
        print("❌ dry_octree_node: MISMATCH")
    
    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    all_match = True
    results = {
        'dam_ijk': dam_cuda_set == dam_python_set,
        'collision_ijk': collision_cuda_set == collision_python_set,
        'water_octree_node': water_cuda_set == water_python_set,
        'dry_octree_node': dry_cuda_set == dry_python_set,
    }
    
    for key, match in results.items():
        status = "✅ MATCH" if match else "❌ MISMATCH"
        print(f"  {key}: {status}")
        if not match:
            all_match = False
    
    if all_match:
        print("\n🎉 All outputs are consistent between CUDA and Python backends!")
    else:
        print("\n⚠️  Some outputs differ. See details above.")
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Verify CUDA vs Python flood fill consistency')
    parser.add_argument('mesh_path', type=str, help='Path to mesh file')
    parser.add_argument('--resolution', type=int, default=9, help='Octree max level (default: 9 = 512³)')
    args = parser.parse_args()
    
    verify_backends(args.mesh_path, args.resolution)
