
import torch

# ============================================================
# Cube Topology Templates (X-major ordering)
# Matches FlexiCubes, DISO, and modern graphics standards
# ============================================================

# 8 corners of a unit cube
# Index i has coordinate [x, y, z] = [(i>>0)&1, (i>>1)&1, (i>>2)&1]
CUBE_CORNERS = torch.tensor([
    [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
    [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]
], dtype=torch.int64)

# 12 edges defined by corner indices
CUBE_EDGES = torch.tensor([
    [0, 1], [2, 3], [4, 5], [6, 7], # X-aligned
    [0, 2], [1, 3], [4, 6], [5, 7], # Y-aligned
    [0, 4], [1, 5], [2, 6], [3, 7]  # Z-aligned
], dtype=torch.int64)

# 6 faces defined by 4 corner indices (CCW from outside)
CUBE_FACES = torch.tensor([
    [0, 4, 6, 2], # -X (x=0)
    [1, 3, 7, 5], # +X (x=1)
    [0, 1, 5, 4], # -Y (y=0)
    [2, 6, 7, 3], # +Y (y=1)
    [0, 2, 3, 1], # -Z (z=0)
    [4, 5, 7, 6]  # +Z (z=1)
], dtype=torch.int64)

# Edge to face mapping: which 2 faces each edge belongs to
EDGE_TO_FACES = torch.tensor([
    [0, 2], [0, 5], [0, 3], [0, 4], # E0-3
    [1, 2], [1, 5], [1, 3], [1, 4], # E4-7
    [2, 4], [2, 5], [3, 5], [3, 4]  # E8-11
], dtype=torch.int64)

# Sign mapping for corners (maps 0/1 to -1/1)
# Used for coordinate generation and sign logic
CUBE_CORNER_SIGNS = (CUBE_CORNERS * 2 - 1).float()

# ============================================================
# Neighbor Offsets
# ============================================================

# 6-neighbor offsets: [+x, -x, +y, -y, +z, -z]
# Standard order for flood fill and face queries
NEIGHBOR_OFFSETS_6 = torch.tensor([
    [1, 0, 0], [-1, 0, 0],
    [0, 1, 0], [0, -1, 0],
    [0, 0, 1], [0, 0, -1]
], dtype=torch.int32)

# 26-neighbor offsets (Full 3x3x3 neighborhood excluding center)
_offsets_26 = []
for dz in [-1, 0, 1]:
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dx == 0 and dy == 0 and dz == 0:
                continue
            _offsets_26.append([dx, dy, dz])
NEIGHBOR_OFFSETS_26 = torch.tensor(_offsets_26, dtype=torch.int32)

# ============================================================
# Edge/Corner Geometric Adjacency
# ============================================================

# For each of the 12 edges, which neighbor voxels share that edge?
# Used by get_dam_wet_edges to detect if an edge is on a water boundary.
# A neighbor voxel shares an edge if it's offset by the combination of
# directions perpendicular to the edge axis.
EDGE_NEIGHBOR_OFFSETS = torch.tensor([
    # X-aligned edges (vary y, z)
    [[0, 1, 0], [0, 0, 1], [0, 1, 1]],     # Edge 0: y=0, z=0 -> neighbors at +y, +z, +yz
    [[0, -1, 0], [0, 0, 1], [0, -1, 1]],   # Edge 1: y=1, z=0 -> neighbors at -y, +z, -yz
    [[0, 1, 0], [0, 0, -1], [0, 1, -1]],   # Edge 2: y=0, z=1 -> neighbors at +y, -z, +y-z
    [[0, -1, 0], [0, 0, -1], [0, -1, -1]], # Edge 3: y=1, z=1 -> neighbors at -y, -z, -yz
    
    # Y-aligned edges (vary x, z)
    [[1, 0, 0], [0, 0, 1], [1, 0, 1]],     # Edge 4
    [[-1, 0, 0], [0, 0, 1], [-1, 0, 1]],   # Edge 5
    [[1, 0, 0], [0, 0, -1], [1, 0, -1]],   # Edge 6
    [[-1, 0, 0], [0, 0, -1], [-1, 0, -1]], # Edge 7
    
    # Z-aligned edges (vary x, y)
    [[1, 0, 0], [0, 1, 0], [1, 1, 0]],     # Edge 8
    [[-1, 0, 0], [0, 1, 0], [-1, 1, 0]],   # Edge 9
    [[1, 0, 0], [0, -1, 0], [1, -1, 0]],   # Edge 10
    [[-1, 0, 0], [0, -1, 0], [-1, -1, 0]], # Edge 11
], dtype=torch.int32)

# For each of the 8 corners, which 7 neighbor voxels share that corner?
# Used by get_dam_wet_corners.
CORNER_NEIGHBOR_OFFSETS = torch.empty((8, 7, 3), dtype=torch.int32)
for i in range(8):
    sx, sy, sz = CUBE_CORNER_SIGNS[i].int().tolist()
    # 7 neighbors: 3 faces, 3 edges, 1 corner
    CORNER_NEIGHBOR_OFFSETS[i] = torch.tensor([
        [sx, 0, 0], [0, sy, 0], [0, 0, sz],
        [sx, sy, 0], [sx, 0, sz], [0, sy, sz],
        [sx, sy, sz]
    ], dtype=torch.int32)

# ============================================================
# Sign Logic Presets
# ============================================================
SIGN_WATER = 1
SIGN_DRY = -1
SIGN_DAM_DEFAULT = -1 # Matches successful single-layered extraction
