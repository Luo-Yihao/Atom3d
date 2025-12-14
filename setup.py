"""
Atom3D: Atomize Your 3D Meshes

Installation:
    pip install -e .
"""

from setuptools import setup, find_packages

setup(
    name="atom3d",
    version="0.1.0",
    author="Atom3D Contributors",
    description="Atomize your 3D meshes - High-performance CUDA mesh voxelization and distance field queries",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/Atom3D",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.20.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "trimesh>=3.0",
        ],
        "full": [
            "trimesh>=3.0",
            "cubvh>=0.1.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    include_package_data=True,
)
