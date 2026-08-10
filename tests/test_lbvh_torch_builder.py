"""
The torch LBVH builder must be indistinguishable from the host builder
through every query path - different tree SHAPE, identical ANSWERS.
"""
import numpy as np
import pytest
import torch

torch.manual_seed(0)
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _mesh(n=4000):
    """A bumpy sphere-ish soup with duplicated positions mixed in."""
    g = torch.Generator().manual_seed(1)
    v = torch.nn.functional.normalize(torch.randn(n, 3, 3, generator=g), dim=-1)
    v = (v * (1 + 0.1 * torch.rand(n, 1, 1, generator=g))).reshape(-1, 3)
    v[: n // 10 * 3] = v[0]                      # co-located block: equal mortons
    f = torch.arange(3 * n).reshape(n, 3)
    return v.cuda().float(), f.cuda()


def _both_trees(v, f):
    from atom3d import MeshBVH
    import atom3d.kernels.bvh as B

    b_torch = MeshBVH(v, f)
    res = B.get_bvh_kernels().build_bvh(v.contiguous().float(),
                                        f.contiguous().int(), 4)
    b_cpp = MeshBVH(v, f)
    b_cpp._bvh.nodes, b_cpp._bvh.triangles = res[0], res[1]
    return b_torch, b_cpp


@cuda
def test_masks_and_pair_sets_match_the_host_builder():
    v, f = _mesh()
    bt, bc = _both_trees(v, f)
    g = torch.Generator(device="cuda").manual_seed(2)
    lo = torch.rand(100_000, 3, generator=g, device="cuda") * 2.4 - 1.2
    hi = lo + torch.rand(100_000, 3, generator=g, device="cuda") * 0.2
    mt = bt._bvh.aabb_intersect(lo.contiguous(), hi.contiguous(), eps=1e-5)
    mc = bc._bvh.aabb_intersect(lo.contiguous(), hi.contiguous(), eps=1e-5)
    assert torch.equal(mt[0], mc[0])
    NF = f.shape[0] + 1
    pt = torch.sort(mt[1].long() * NF + mt[2].long()).values
    pc = torch.sort(mc[1].long() * NF + mc[2].long()).values
    assert pt.shape == pc.shape and torch.equal(pt, pc)


@cuda
def test_mask_only_equals_pairs_mask():
    v, f = _mesh()
    bt, _ = _both_trees(v, f)
    g = torch.Generator(device="cuda").manual_seed(3)
    lo = torch.rand(100_000, 3, generator=g, device="cuda") * 2.4 - 1.2
    hi = lo + 0.05
    a = bt._bvh.aabb_intersect(lo.contiguous(), hi.contiguous(), eps=1e-5, pairs=True)[0]
    b = bt._bvh.aabb_intersect(lo.contiguous(), hi.contiguous(), eps=1e-5, pairs=False)[0]
    assert torch.equal(a, b)


@cuda
def test_udf_matches_the_host_builder():
    v, f = _mesh()
    bt, bc = _both_trees(v, f)
    g = torch.Generator(device="cuda").manual_seed(4)
    p = torch.rand(50_000, 3, generator=g, device="cuda") * 3 - 1.5
    assert float((bt.udf(p) - bc.udf(p)).abs().max()) < 1e-5


@cuda
def test_single_and_colocated_primitives():
    from atom3d import MeshBVH
    v = torch.tensor([[0., 0, 0], [1, 0, 0], [0, 1, 0]]).cuda()
    f = torch.tensor([[0, 1, 2]]).cuda()
    assert float(MeshBVH(v, f).udf(torch.zeros(1, 3).cuda())) == 0.0
    v5 = v.repeat(5, 1)
    f5 = torch.arange(15).reshape(5, 3).cuda()
    assert abs(float(MeshBVH(v5, f5).udf(torch.tensor([[2., 0, 0]]).cuda())) - 1.0) < 1e-6


@cuda
def test_build_is_deterministic():
    v, f = _mesh()
    from atom3d import MeshBVH
    a, b = MeshBVH(v, f), MeshBVH(v, f)
    # compare as int32 bit patterns: the child-index columns hold negative ints
    # bit-cast into float slots, and many of those patterns are NaN, for which
    # float equality is false even when the bits agree
    assert torch.equal(a._bvh.nodes.view(torch.int32), b._bvh.nodes.view(torch.int32))
    assert torch.equal(a._bvh.triangles.view(torch.int32),
                       b._bvh.triangles.view(torch.int32))
