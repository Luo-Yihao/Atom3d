"""
Tests for the SAT clip polygon kernel and the pair-emitting intersection
kernels, checked against an exact float64 numpy reference.

Covers the regressions fixed in 2026-07:
  * clip_with_plane stack overflow / silent truncation on 9-vertex clips
    (a triangle clipped by a box yields up to 3 + 6 = 9 vertices; the old
    8-vertex work buffers wrote out of bounds and dropped edges)
  * silent pair-list truncation in triangle_aabb_intersect,
    segment_tri_intersect, and bvh_aabb_intersect when hits exceeded the
    heuristic buffer capacity

Run:  pytest tests/test_sat_clip_polygon.py -v   (requires CUDA)
"""

import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from atom3d.kernels import sat_clip_polygon, triangle_aabb_intersect, segment_tri_intersect  # noqa: E402
from atom3d.core.mesh_bvh import MeshBVH  # noqa: E402


# ------------------------------------------------------------------
# exact float64 reference
# ------------------------------------------------------------------

def clip_exact(tri, bmin, bmax):
    """Sutherland-Hodgman triangle-vs-box clip, float64, no vertex cap.

    Same plane order as the kernel (+x, -x, +y, -y, +z, -z) so vertex
    ordering is comparable.
    """
    poly = [np.asarray(v, dtype=np.float64) for v in tri]
    for axis in range(3):
        for sign, bound in ((1.0, bmax[axis]), (-1.0, bmin[axis])):
            out = []
            n = len(poly)
            for i in range(n):
                P, Q = poly[i], poly[(i + 1) % n]
                dP = sign * P[axis] - sign * bound
                dQ = sign * Q[axis] - sign * bound
                inP, inQ = dP <= 0, dQ <= 0
                if inP and inQ:
                    out.append(Q)
                elif inP and not inQ:
                    out.append(P + dP / (dP - dQ) * (Q - P))
                elif not inP and inQ:
                    out.append(P + dP / (dP - dQ) * (Q - P))
                    out.append(Q)
            poly = out
            if not poly:
                return []
    return poly


def poly_area(poly):
    if len(poly) < 3:
        return 0.0
    p = np.asarray(poly)
    c = p.mean(0)
    return sum(
        0.5 * np.linalg.norm(np.cross(p[i] - c, p[(i + 1) % len(p)] - c))
        for i in range(len(p))
    )


def run_kernel(tris, boxes_min, boxes_max, pairs_a, pairs_t, mode=2):
    dev = "cuda"
    tv = torch.tensor(np.asarray(tris, np.float32).reshape(-1, 9), device=dev)
    bmin = torch.tensor(np.asarray(boxes_min, np.float32), device=dev)
    bmax = torch.tensor(np.asarray(boxes_max, np.float32), device=dev)
    ca = torch.tensor(np.asarray(pairs_a, np.int64), device=dev)
    ct = torch.tensor(np.asarray(pairs_t, np.int64), device=dev)
    return sat_clip_polygon(bmin, bmax, tv, ca, ct, mode=mode)


# A triangle whose clip against the unit box is a 9-gon (found by exact
# search; verified by clip_exact in the test itself).
NINE_GON_TRI = np.array([
    [-0.233561, 1.187059, 0.732607],
    [0.968992, 0.44565, -0.023344],
    [1.849645, -1.80253, 2.105212],
])


# ------------------------------------------------------------------
# clip kernel vs reference
# ------------------------------------------------------------------

def test_random_agreement_with_exact_reference():
    rng = np.random.default_rng(42)
    n = 4000
    tris, bmins, bmaxs = [], [], []
    for _ in range(n):
        scale = 10.0 ** rng.uniform(-2, 1)
        c = rng.uniform(-1, 1, 3) * scale
        bmins.append(c)
        bmaxs.append(c + rng.uniform(0.5, 1.5, 3) * scale)
        tris.append(c + rng.uniform(-2, 2, (3, 3)) * scale)

    hit, pc, pv, cents, areas, _, _ = run_kernel(
        tris, bmins, bmaxs, np.arange(n), np.arange(n))
    hit = hit.cpu().numpy()
    areas = areas.cpu().numpy()
    cents = cents.cpu().numpy()

    checked_hits = checked_miss = 0
    for i in range(n):
        ref = clip_exact(tris[i], bmins[i], bmaxs[i])
        a_ref = poly_area(ref)
        scale2 = float(np.prod(bmaxs[i] - bmins[i]) ** (2 / 3))
        if a_ref > 1e-5 * scale2:
            # clearly intersecting: kernel must agree, area must match
            assert hit[i], f"pair {i}: reference area {a_ref}, kernel says miss"
            assert areas[i] == pytest.approx(a_ref, rel=2e-3, abs=1e-6 * scale2), \
                f"pair {i}: area {areas[i]} vs exact {a_ref}"
            c_ref = np.asarray(ref).mean(0)
            assert np.linalg.norm(cents[i] - c_ref) < 1e-3 * np.sqrt(scale2), \
                f"pair {i}: centroid off by {np.linalg.norm(cents[i] - c_ref)}"
            checked_hits += 1
        elif not ref:
            # reference says empty with fp64: allow the kernel's eps padding
            # to flip only borderline cases — re-check with a shrunken box
            grown = clip_exact(tris[i],
                               np.asarray(bmins[i]) - 1e-4 * np.sqrt(scale2),
                               np.asarray(bmaxs[i]) + 1e-4 * np.sqrt(scale2))
            if not grown:
                assert not hit[i], f"pair {i}: clearly separated but kernel hit"
                assert areas[i] == 0
                checked_miss += 1
    # make sure the test actually exercised both classes
    assert checked_hits > 500 and checked_miss > 500


def test_nine_gon_area_and_centroid_exact():
    bmin, bmax = np.zeros(3), np.ones(3)
    # axis permutations keep the box invariant and the clip a 9-gon
    perms = [(0, 1, 2), (1, 2, 0), (2, 0, 1), (0, 2, 1), (1, 0, 2), (2, 1, 0)]
    tris = [NINE_GON_TRI[:, p] for p in perms]
    for tri in tris:
        assert len(clip_exact(tri, bmin, bmax)) == 9  # guard: still a 9-gon

    n = len(tris)
    hit, pc, pv, cents, areas, _, _ = run_kernel(
        tris, [bmin] * n, [bmax] * n, np.arange(n), np.arange(n))

    for i, tri in enumerate(tris):
        ref = clip_exact(tri, bmin, bmax)
        a_ref = poly_area(ref)
        c_ref = np.asarray(ref).mean(0)
        assert bool(hit[i])
        # the old 8-vertex buffer dropped clip edges here (≈5% area error
        # in this configuration) and overflowed the work buffer
        assert areas[i].item() == pytest.approx(a_ref, rel=1e-3), \
            f"perm {i}: area {areas[i].item()} vs exact {a_ref}"
        # kernel centroid is the vertex mean projected to the triangle;
        # for an interior polygon that equals the vertex mean
        assert np.linalg.norm(cents[i].cpu().numpy() - c_ref) < 1e-4
        # export layout: at most 8 vertices stored, count matches storage
        assert pc[i].item() == 8
        stored = pv[i, :8].cpu().numpy()
        ref8 = np.asarray(ref[:8], dtype=np.float32)
        assert np.abs(stored - ref8).max() < 1e-4


def test_degenerate_candidates():
    bmin, bmax = np.zeros(3), np.ones(3)
    tris = [
        np.array([[3.0, 3.0, 3.0], [4.0, 3.0, 3.0], [3.0, 4.0, 3.0]]),  # far away
        np.array([[0.0, 0.0, 0.0], [-1.0, -1.0, 0.0], [-1.0, 0.0, -1.0]]),  # corner touch
    ]
    n = len(tris)
    hit, pc, pv, cents, areas, _, _ = run_kernel(
        tris, [bmin] * n, [bmax] * n, np.arange(n), np.arange(n))

    # non-intersecting broadphase candidate: no hit, zeroed outputs
    assert not bool(hit[0])
    assert areas[0].item() == 0.0
    # single-point contact: hit by convention, explicit zero area
    assert bool(hit[1])
    assert areas[1].item() == 0.0


# ------------------------------------------------------------------
# pair-emitting kernels: no silent truncation
# ------------------------------------------------------------------

def _grid_mesh(nx=60, nz=60, device="cuda"):
    """Gently curved triangulated height field over [0,1]^2."""
    xs = torch.linspace(0, 1, nx + 1)
    zs = torch.linspace(0, 1, nz + 1)
    gx, gz = torch.meshgrid(xs, zs, indexing="ij")
    gy = 0.5 + 0.15 * torch.sin(4 * gx) * torch.cos(4 * gz)
    verts = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)
    idx = torch.arange((nx + 1) * (nz + 1)).reshape(nx + 1, nz + 1)
    a, b, c, d = idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]
    faces = torch.cat([
        torch.stack([a, b, c], -1).reshape(-1, 3),
        torch.stack([a, c, d], -1).reshape(-1, 3),
    ])
    return verts.float().to(device), faces.int().to(device)


def _sat_fp64(tris, bmin, bmax):
    """Vectorized exact 13-axis SAT, float64. tris [K,3,3], boxes [K,3]."""
    c = (bmin + bmax) / 2
    e = (bmax - bmin) / 2
    v = tris.double() - c.double().unsqueeze(1)
    sep = torch.zeros(len(tris), dtype=torch.bool, device=tris.device)

    def axis_test(ax):
        p = (v * ax.unsqueeze(1)).sum(-1)
        r = (e.double() * ax.abs()).sum(-1)
        return (p.amin(-1) > r) | (p.amax(-1) < -r)

    for a in range(3):
        ax = torch.zeros(len(tris), 3, dtype=torch.float64, device=tris.device)
        ax[:, a] = 1
        sep |= axis_test(ax)
    f = [v[:, 1] - v[:, 0], v[:, 2] - v[:, 1], v[:, 0] - v[:, 2]]
    sep |= axis_test(torch.cross(f[0], f[1], dim=-1))
    for fe in f:
        for a in range(3):
            u = torch.zeros(len(tris), 3, dtype=torch.float64, device=tris.device)
            u[:, a] = 1
            sep |= axis_test(torch.cross(u, fe, dim=-1))
    return ~sep


def test_meshbvh_intersect_aabb_no_leaks_and_complete():
    verts, faces = _grid_mesh()
    bvh = MeshBVH(verts, faces)

    g = 12
    lin = torch.linspace(0, 1, g + 1, device="cuda")
    lo = torch.stack(torch.meshgrid(lin[:-1], lin[:-1], lin[:-1], indexing="ij"),
                     -1).reshape(-1, 3)
    hi = lo + 1.0 / g
    res = bvh.intersect_aabb(lo.contiguous(), hi.contiguous(), mode=2)
    assert res.aabb_ids is not None and len(res.aabb_ids) > 0

    tri_all = verts[faces.long()]
    # (a) no junk rows: every returned pair truly intersects (allow eps slack)
    pad = 1e-5
    ok = _sat_fp64(tri_all[res.face_ids.long()],
                   lo[res.aabb_ids.long()] - pad, hi[res.aabb_ids.long()] + pad)
    assert bool(ok.all()), f"{(~ok).sum().item()} returned pairs do not intersect"
    # centroids of returned pairs must never be the (0,0,0) placeholder
    assert not (res.centroids == 0).all(-1).any()

    # (b) completeness vs exact SAT with a strict margin (pairs whose
    # intersection survives shrinking the box cannot be dropped)
    got = set(zip(res.aabb_ids.tolist(), res.face_ids.tolist()))
    K, M = lo.shape[0], faces.shape[0]
    bi = torch.repeat_interleave(torch.arange(K, device="cuda"), M)
    ti = torch.arange(M, device="cuda").repeat(K)
    strict = _sat_fp64(tri_all[ti], lo[bi] + 1e-4, hi[bi] - 1e-4)
    must = set(zip(bi[strict].tolist(), ti[strict].tolist()))
    missing = must - got
    assert not missing, f"{len(missing)} clearly-intersecting pairs missing"


def test_triangle_aabb_intersect_overflow_regrowth():
    # 4 boxes covering the whole mesh: true pairs = 4 * 7200, far beyond the
    # heuristic capacity num_aabbs * 100 — exercises the re-launch path
    verts, faces = _grid_mesh()
    n_box = 4
    lo = torch.full((n_box, 3), -1.0, device="cuda")
    hi = torch.full((n_box, 3), 2.0, device="cuda")
    hit_mask, aabb_ids, face_ids = triangle_aabb_intersect(
        verts, faces, lo.contiguous(), hi.contiguous())
    assert bool(hit_mask.all())
    assert len(aabb_ids) == n_box * faces.shape[0]
    # every (box, face) pair exactly once
    key = aabb_ids.long() * faces.shape[0] + face_ids.long()
    assert torch.unique(key).numel() == n_box * faces.shape[0]


def test_bvh_aabb_intersect_overflow_regrowth():
    verts, faces = _grid_mesh()
    bvh = MeshBVH(verts, faces)
    assert bvh._bvh is not None
    n_box = 4
    lo = torch.full((n_box, 3), -1.0, device="cuda")
    hi = torch.full((n_box, 3), 2.0, device="cuda")
    hit_mask, aabb_ids, face_ids = bvh._bvh.aabb_intersect(
        lo.contiguous(), hi.contiguous())
    assert bool(hit_mask.all())
    assert len(aabb_ids) == n_box * faces.shape[0]
    key = aabb_ids.long() * faces.shape[0] + face_ids.long()
    assert torch.unique(key).numel() == n_box * faces.shape[0]


def test_non_default_stream_consistency():
    # kernels must run on the caller's current stream: with a non-default
    # (non-blocking) stream, launching on legacy stream 0 would break the
    # ordering between the kernel and the counter readback that the
    # overflow-regrowth logic relies on
    verts, faces = _grid_mesh(30, 30)
    n_box = 2
    lo = torch.full((n_box, 3), -1.0, device="cuda")
    hi = torch.full((n_box, 3), 2.0, device="cuda")
    expected = n_box * faces.shape[0]  # > n_box * 100: regrowth path active

    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        hit_mask, aabb_ids, face_ids = triangle_aabb_intersect(
            verts, faces, lo.contiguous(), hi.contiguous())
        tris = verts[faces.long()]
        hit, pc, pv, cents, areas, _, _ = sat_clip_polygon(
            lo[:1].contiguous(), hi[:1].contiguous(),
            tris.reshape(-1, 9).contiguous(),
            torch.zeros(faces.shape[0], dtype=torch.int64, device="cuda"),
            torch.arange(faces.shape[0], dtype=torch.int64, device="cuda"),
            mode=1)
    torch.cuda.synchronize()

    assert len(aabb_ids) == expected
    key = aabb_ids.long() * faces.shape[0] + face_ids.long()
    assert torch.unique(key).numel() == expected
    # every face is fully inside the huge box: all hits, area = face area
    assert bool(hit.all())
    e1 = tris[:, 1] - tris[:, 0]
    e2 = tris[:, 2] - tris[:, 0]
    ref_area = 0.5 * torch.cross(e1, e2, dim=-1).norm(dim=-1)
    assert torch.allclose(areas, ref_area, rtol=1e-4, atol=1e-7)


def test_segment_tri_intersect_overflow_regrowth():
    # one segment piercing 20k stacked triangles: 20000 hits > 12 + 8192
    n_tri = 20000
    z = torch.linspace(0.1, 0.9, n_tri, device="cuda")
    v0 = torch.stack([torch.full_like(z, -1.0), torch.full_like(z, -1.0), z], -1)
    v1 = torch.stack([torch.full_like(z, 3.0), torch.full_like(z, -1.0), z], -1)
    v2 = torch.stack([torch.full_like(z, -1.0), torch.full_like(z, 3.0), z], -1)
    tri_verts = torch.cat([v0, v1, v2], dim=-1).contiguous()          # [M, 9]
    tris = tri_verts.reshape(-1, 3, 3)
    tmin = tris.amin(1).contiguous()
    tmax = tris.amax(1).contiguous()
    segs = torch.tensor([[0.2, 0.2, -0.5, 0.2, 0.2, 1.5]], device="cuda")
    seg_ids, tri_ids, t = segment_tri_intersect(segs, tri_verts, tmin, tmax)
    assert len(tri_ids) == n_tri
    assert torch.unique(tri_ids).numel() == n_tri
    assert bool((t > 0).all()) and bool((t < 1).all())
