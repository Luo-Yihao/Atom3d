"""
Mesh Extractor Module for Atom3D

Provides differentiable mesh extraction algorithms:
- SparseDiffDMC: Sparse Differentiable Dual Marching Cubes
"""

from .sparse_diffdmc import SparseDiffDMC

__all__ = ["SparseDiffDMC"]
