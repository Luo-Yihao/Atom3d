#!/usr/bin/env python3
"""
UDF/SDF Query Example

Demonstrates distance field queries with gradient support.
"""

import torch
import time

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from atom3d import MeshBVH


def create_sphere_mesh():
    """Create icosphere mesh."""
    try:
        import trimesh
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.5)
        return (
            torch.tensor(mesh.vertices, dtype=torch.float32),
            torch.tensor(mesh.faces, dtype=torch.int64)
        )
    except ImportError:
        raise RuntimeError("trimesh required: pip install trimesh")


def main():
    print("=" * 50)
    print("Atom3D: UDF/SDF Query Example")
    print("=" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create sphere mesh
    vertices, faces = create_sphere_mesh()
    vertices = vertices.to(device)
    faces = faces.to(device)
    print(f"Mesh: sphere with {len(faces)} faces, radius=0.5")
    
    # Create BVH
    bvh = MeshBVH(vertices, faces, device=device)
    
    # Generate random query points
    num_points = 10000
    points = torch.randn(num_points, 3, device=device)
    points = points / points.norm(dim=1, keepdim=True) * torch.rand(num_points, 1, device=device) * 2
    
    print(f"\n--- UDF Query (no gradient) ---")
    t0 = time.time()
    result = bvh.udf(points, return_grad=False)
    print(f"Time: {time.time()-t0:.4f}s")
    print(f"Distance range: [{result.distances.min():.4f}, {result.distances.max():.4f}]")
    
    print(f"\n--- UDF Query (with gradient) ---")
    points_grad = points.clone().requires_grad_(True)
    
    t0 = time.time()
    result = bvh.udf(points_grad, return_grad=True)
    print(f"Time: {time.time()-t0:.4f}s")
    
    # Backprop
    loss = result.distances.mean()
    loss.backward()
    
    gradients = points_grad.grad
    print(f"Gradient shape: {gradients.shape}")
    print(f"Gradient norm mean: {gradients.norm(dim=1).mean():.4f}")
    
    # Verify gradient
    print(f"\n--- Gradient Verification ---")
    closest = result.closest_points
    expected_grad = (points_grad.detach() - closest)
    expected_grad = expected_grad / (expected_grad.norm(dim=1, keepdim=True) + 1e-8)
    
    dot = (gradients * expected_grad).sum(dim=1)
    print(f"Gradient · Expected: {dot.mean():.6f} (should be ~1.0)")
    
    print(f"\n--- SDF Query ---")
    sdf_result = bvh.sdf(points, return_grad=False)
    
    inside = (sdf_result.distances < 0).sum()
    outside = (sdf_result.distances >= 0).sum()
    print(f"Inside mesh: {inside}, Outside: {outside}")
    print(f"SDF range: [{sdf_result.distances.min():.4f}, {sdf_result.distances.max():.4f}]")
    
    # Application: move points towards surface
    print(f"\n--- Application: Move towards surface ---")
    
    with torch.no_grad():
        step_size = 0.1
        direction = (points - result.closest_points)
        direction = direction / (direction.norm(dim=1, keepdim=True) + 1e-8)
        
        new_points = points - step_size * direction * result.distances[:, None].sign()
        new_result = bvh.udf(new_points)
        
        print(f"Before: mean distance = {result.distances.abs().mean():.4f}")
        print(f"After:  mean distance = {new_result.distances.abs().mean():.4f}")
    
    print("\n" + "=" * 50)
    print("Done!")
    print("=" * 50)


if __name__ == "__main__":
    main()
