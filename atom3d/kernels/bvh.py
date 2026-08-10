"""
BVH Accelerator for Atom3D

Python interface to BVH CUDA kernels. Provides accelerated:
- UDF queries (closest point to mesh)
- Ray-mesh intersection
- AABB-mesh intersection (with exact SAT)
"""

import torch
from torch.utils.cpp_extension import load
import os

# Get the CUDA kernel source
_KERNEL_DIR = os.path.dirname(os.path.abspath(__file__))
_BUILD_DIR = os.path.join(_KERNEL_DIR, 'build', 'bvh')

# Global cache for JIT compiled module
_bvh_cuda = None

def get_bvh_kernels():
    """Get BVH CUDA kernels - use cached .so if available, else JIT compile."""
    global _bvh_cuda
    
    if _bvh_cuda is not None:
        return _bvh_cuda
    
    so_file = os.path.join(_BUILD_DIR, 'bvh_cuda.so')
    kernel_path = os.path.join(_KERNEL_DIR, 'bvh_kernels.cu')

    # Fast path: load the cached .so only if it was built from the current
    # kernel source (and torch/CUDA env) — otherwise a stale binary would
    # silently shadow source changes. Tag formula matches kernels/__init__.py.
    import hashlib
    if not os.path.exists(kernel_path):
        raise RuntimeError(f"CUDA kernel file not found: {kernel_path}")
    with open(kernel_path, 'rb') as f:
        src_hash = hashlib.sha256(f.read()).hexdigest()[:16]
    tag = f"{torch.__version__}_{torch.version.cuda}_{torch.cuda.get_arch_list()}_{src_hash}"
    tag_file = os.path.join(_BUILD_DIR, '.build_tag')
    tag_ok = False
    if os.path.exists(tag_file):
        with open(tag_file) as f:
            tag_ok = f.read().strip() == tag

    if os.path.exists(so_file) and tag_ok:
        import importlib.util
        spec = importlib.util.spec_from_file_location('bvh_cuda', so_file)
        _bvh_cuda = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_bvh_cuda)
        return _bvh_cuda

    # Slow path: JIT compile
    os.makedirs(_BUILD_DIR, exist_ok=True)

    _bvh_cuda = load(
        name='bvh_cuda',
        sources=[kernel_path],
        build_directory=_BUILD_DIR,
        extra_cuda_cflags=['-O3', '--use_fast_math'],
        verbose=False
    )

    with open(tag_file, 'w') as f:
        f.write(tag)

    return _bvh_cuda


class BVHAccelerator:
    """
    BVH-accelerated mesh queries.
    
    Provides O(log M) queries instead of O(M) brute-force:
    - udf: closest point to mesh
    - ray_intersect: ray-mesh intersection
    - aabb_intersect: AABB-mesh intersection with exact SAT
    
    All returned face_ids are in ORIGINAL mesh order (not BVH reordered order).
    """
    
    def __init__(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        n_primitives_per_leaf: int = 4  # Reduced from 8 for better accuracy on dense meshes
    ):
        """
        Build BVH from mesh.
        
        Args:
            vertices: [N, 3] float32 vertices
            faces: [M, 3] int32 face indices
            n_primitives_per_leaf: Max triangles per leaf node
        """
        self.device = vertices.device
        self.vertices = vertices.contiguous().float()
        self.faces = faces.contiguous().int()
        self.num_faces = faces.shape[0]
        
        # Build BVH - returns (nodes, triangles with original_id)
        built = None
        if self.vertices.is_cuda and self.num_faces > 0:
            built = _build_lbvh_torch(
                self.vertices, self.faces, n_primitives_per_leaf)
        if built is not None:
            self.nodes, self.triangles = built
        else:
            cuda = get_bvh_kernels()
            result = cuda.build_bvh(
                self.vertices,
                self.faces,
                n_primitives_per_leaf
            )
            self.nodes = result[0]        # [num_nodes, 9]
            self.triangles = result[1]    # [num_faces, 10] - includes original_id
    
    def udf(
        self,
        points: torch.Tensor
    ):
        """
        Unsigned distance field query.
        
        Args:
            points: [K, 3] query points
            
        Returns:
            distances: [K] unsigned distances
            face_ids: [K] closest face indices (ORIGINAL order)
            closest_points: [K, 3] closest points on mesh
            uvw: [K, 3] barycentric coordinates
        """
        cuda = get_bvh_kernels()
        distances, face_ids, closest_points, uvw = cuda.bvh_udf(
            self.nodes,
            self.triangles,
            points.contiguous().float()
        )
        return distances, face_ids, closest_points, uvw
    
    def ray_intersect(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        max_t: float = 1e10
    ):
        """
        Ray-mesh intersection.
        
        Args:
            rays_o: [K, 3] ray origins
            rays_d: [K, 3] ray directions
            max_t: Maximum ray distance
            
        Returns:
            hit_mask: [K] bool - whether ray hit mesh
            hit_t: [K] hit distance (max_t if no hit)
            face_ids: [K] hit face indices (ORIGINAL order, -1 if no hit)
            hit_points: [K, 3] hit positions
        """
        cuda = get_bvh_kernels()
        hit_mask, hit_t, face_ids, hit_points, _uvw = cuda.bvh_ray_intersect(
            self.nodes,
            self.triangles,
            rays_o.contiguous().float(),
            rays_d.contiguous().float(),
            max_t
        )
        return hit_mask, hit_t, face_ids, hit_points
    
    def ray_all_hit(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        max_t: float = 1e10
    ):
        """
        Ray all-hit query: collect ALL intersections via two-pass BVH traversal.

        Pass 1: count hits per ray.  Pass 2: write hits with prefix-sum offsets.
        No Python ray-marching loop.

        Args:
            rays_o: [K, 3] ray origins
            rays_d: [K, 3] ray directions
            max_t: Maximum ray distance

        Returns:
            ray_idx:    [H] int32 — which ray each hit belongs to
            t:          [H] float — hit distance along ray
            face_id:    [H] int32 — hit face index (original order)
            sign:       [H] int32 — sign(dot(face_normal, ray_dir)), +1/-1/0
            hit_points: [H, 3] float — hit positions
            uvw:        [H, 3] float — barycentric coordinates at hit
        """
        cuda = get_bvh_kernels()
        return cuda.bvh_ray_all_hit(
            self.nodes,
            self.triangles,
            rays_o.contiguous().float(),
            rays_d.contiguous().float(),
            max_t
        )

    def aabb_intersect(
        self,
        query_min: torch.Tensor,
        query_max: torch.Tensor,
        eps: float = 1e-6,
        pairs: bool = True
    ):
        """
        AABB-mesh intersection with exact SAT test.
        
        Args:
            query_min: [K, 3] query AABB mins
            query_max: [K, 3] query AABB maxs
            
        Returns:
            hit_mask: [K] bool - whether AABB intersects mesh
            aabb_ids: [N] query indices for each intersection pair
            face_ids: [N] face indices (ORIGINAL order) for each pair
        """
        cuda = get_bvh_kernels()
        hit_mask, aabb_ids, face_ids = cuda.bvh_aabb_intersect(
            self.nodes,
            self.triangles,
            query_min.contiguous().float(),
            query_max.contiguous().float(),
            float(eps),
            not pairs
        )
        return hit_mask, aabb_ids, face_ids


def _msb(x):
    """floor(log2(x)) for int64 x >= 1, exact.

    Doubles hold every integer below 2**53 exactly, but log2 of (2**b - 1) can
    round UP to b at the precision edge, so the float answer is clamp-corrected
    with two integer comparisons instead of being trusted."""
    m = torch.floor(torch.log2(x.double())).long()
    m = torch.where((1 << m) > x, m - 1, m)
    m = torch.where((1 << (m + 1)) <= x, m + 1, m)
    return m


def _build_lbvh_torch(vertices: torch.Tensor, faces: torch.Tensor,
                      n_per_leaf: int = 4):
    """
    GPU LBVH build (Karras 2012), vectorized torch, no custom kernels.

    Exists because build_bvh_cuda is CUDA in name only: it copies the mesh to
    the host and builds the tree in single-threaded recursive C++ - 42 s for an
    8.15M-triangle mesh whose every other pipeline stage runs in milliseconds.
    This produces the same byte layout the traversal kernels reinterpret_cast
    (BVHNode = bb_min[3], bb_max[3], left, right, pad; Triangle = a, b, c,
    original_id; leaf ranges encoded as left = -start-1, right = -end-1), so
    traversal, SAT, UDF and ray kernels run on it unchanged. The tree SHAPE
    differs from the C++ builder's median split - query RESULTS do not, which
    the equivalence test asserts on masks, pair sets and closest points.

    All primitive coordinates enter leaf boxes exactly; only the split ORDER
    comes from the quantized morton codes, so quantization affects quality,
    never correctness.
    """
    dev = vertices.device
    tri = vertices[faces.long()].float()                        # [M, 3, 3]
    M = tri.shape[0]
    cen = tri.mean(1)
    lo, hi = tri.amin((0, 1)), tri.amax((0, 1))
    q = ((cen - lo) / (hi - lo).clamp_min(1e-30) * 1023.0).clamp(0, 1023).long()

    def spread(x):
        x = (x | (x << 16)) & 0x030000FF
        x = (x | (x << 8)) & 0x0300F00F
        x = (x | (x << 4)) & 0x030C30C3
        x = (x | (x << 2)) & 0x09249249
        return x

    morton = (spread(q[:, 0]) << 2) | (spread(q[:, 1]) << 1) | spread(q[:, 2])
    # stable: duplicate morton codes are common (co-located primitives) and
    # an unstable sort would make the tree shape run-to-run nondeterministic
    order = torch.argsort(morton, stable=True)
    tri_s = tri[order].reshape(M, 9)
    oid = order.to(torch.int32)

    L = (M + n_per_leaf - 1) // n_per_leaf
    starts = torch.arange(L, device=dev) * n_per_leaf
    ends = torch.clamp(starts + n_per_leaf, max=M)
    ibits = max(1, int(L - 1).bit_length())
    if 30 + ibits > 52:
        # the exact-log2 delta runs out of double-precision bits around 16M
        # faces; the caller falls back to the host builder rather than failing
        return None
    # first primitive's code, index-augmented: strictly increasing, unique
    key = (morton[order][starts] << ibits) | torch.arange(L, device=dev)

    # leaf AABBs (exact, from the primitives themselves)
    leaf_of = torch.arange(M, device=dev) // n_per_leaf
    lbb_min = torch.full((L, 3), float("inf"), device=dev)
    lbb_max = torch.full((L, 3), float("-inf"), device=dev)
    prim = tri_s.reshape(M, 3, 3)
    ix = leaf_of[:, None].expand(M, 3)
    lbb_min.scatter_reduce_(0, ix, prim.amin(1), "amin")
    lbb_max.scatter_reduce_(0, ix, prim.amax(1), "amax")

    if L == 1:
        nodes = torch.zeros(1, 9, dtype=torch.float32, device=dev)
        nodes[0, 0:3], nodes[0, 3:6] = lbb_min[0], lbb_max[0]
        iview = nodes.view(torch.int32)
        iview[0, 6], iview[0, 7] = -1, -int(M) - 1
        tris = torch.empty(M, 10, dtype=torch.float32, device=dev)
        tris[:, :9] = tri_s
        tris[:, 9] = oid.view(torch.float32)
        return nodes, tris

    def delta(i, j):
        ok = (j >= 0) & (j < L)
        x = key[i] ^ key[j.clamp(0, L - 1)]
        d = 63 - _msb(x.clamp_min(1))
        return torch.where(ok, d, torch.full_like(d, -1))

    i = torch.arange(L - 1, device=dev)
    d = torch.sign(delta(i, i + 1) - delta(i, i - 1)).long()
    d[d == 0] = 1
    dmin = delta(i, i - d)
    # upper bound for the range length, then binary-search it down
    lmax = torch.full_like(i, 2)
    for _ in range(int(L).bit_length() + 1):
        grow = delta(i, i + lmax * d) > dmin
        if not bool(grow.any()):
            break
        lmax = torch.where(grow, lmax * 2, lmax)
    ln = torch.zeros_like(i)
    t = lmax // 2
    while bool((t > 0).any()):
        better = (t > 0) & (delta(i, i + (ln + t) * d) > dmin)
        ln = torch.where(better, ln + t, ln)
        t = t // 2
    j = i + ln * d
    dnode = delta(i, j)
    # split position: largest s with delta(i, i + s*d) > dnode
    s = torch.zeros_like(i)
    t = (ln + 1) // 2
    while True:
        better = delta(i, i + (s + t) * d) > dnode
        s = torch.where(better, s + t, s)
        if bool((t <= 1).all()):
            break
        t = (t + 1) // 2
    gamma = i + s * d + torch.minimum(d, torch.zeros_like(d))
    left_is_leaf = torch.minimum(i, j) == gamma
    right_is_leaf = torch.maximum(i, j) == gamma + 1
    left = torch.where(left_is_leaf, L - 1 + gamma, gamma)
    right = torch.where(right_is_leaf, L + gamma, gamma + 1)

    N = 2 * L - 1
    nodes = torch.zeros(N, 9, dtype=torch.float32, device=dev)
    iview = nodes.view(torch.int32)
    iview[: L - 1, 6] = left.to(torch.int32)
    iview[: L - 1, 7] = right.to(torch.int32)
    iview[L - 1:, 6] = (-starts - 1).to(torch.int32)
    iview[L - 1:, 7] = (-ends - 1).to(torch.int32)
    nodes[L - 1:, 0:3] = lbb_min
    nodes[L - 1:, 3:6] = lbb_max

    # internal boxes bottom-up: a fixed-point sweep; each pass finalizes every
    # node whose children are done, so the pass count is the tree depth
    done = torch.zeros(N, dtype=torch.bool, device=dev)
    done[L - 1:] = True
    for _ in range(64):
        ready = ~done[: L - 1] & done[left] & done[right]
        if not bool(ready.any()):
            break
        r = torch.nonzero(ready).squeeze(-1)
        nodes[r, 0:3] = torch.minimum(nodes[left[r], 0:3], nodes[right[r], 0:3])
        nodes[r, 3:6] = torch.maximum(nodes[left[r], 3:6], nodes[right[r], 3:6])
        done[r] = True
    if not bool(done.all()):
        raise RuntimeError("LBVH box sweep did not converge (malformed tree)")

    tris = torch.empty(M, 10, dtype=torch.float32, device=dev)
    tris[:, :9] = tri_s
    tris[:, 9] = oid.view(torch.float32)
    return nodes, tris


# Check if BVH kernels are available
def bvh_available():
    """Check if BVH CUDA kernels can be compiled."""
    try:
        get_bvh_kernels()
        return True
    except Exception:
        return False
