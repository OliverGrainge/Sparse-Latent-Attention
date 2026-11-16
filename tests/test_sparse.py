"""Checks for the lightning indexer and top-k sparse attention (pure tensor tests,
no model weights needed)."""

import math

import pytest
import torch

from sla.model import causal_attention
from sla.sparse import Indexer, indices_to_mask, sparse_attention, topk_indices

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
DEV = "cuda"


def _inputs(B=2, H=4, Hkv=2, T=96, D=32, C=64, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    r = lambda *s: torch.randn(*s, device=DEV, generator=g)
    idx = Indexer(C, n_heads=4, head_dim=16, rope_dim=8).to(DEV)
    with torch.no_grad():
        idx.ww.normal_(generator=g)
    x = r(B, T, C)
    pos = torch.arange(T, device=DEV)
    iq, iw = idx.queries(x, pos)
    ik = idx.keys(x, pos)
    return r(B, H, T, D), r(B, Hkv, T, D), r(B, Hkv, T, D), iq, iw, ik


def test_full_topk_is_dense():
    q, k, v, iq, iw, ik = _inputs()
    T = q.shape[2]
    dense = causal_attention(q, k, v, 0, 0.2, impl="eager")
    # topk >= S takes the dense shortcut; topk = S - 1 exercises the masked path with
    # every query except the last seeing all its predecessors.
    assert torch.allclose(sparse_attention(q, k, v, iq, iw, ik, 0, T, 0.2), dense, atol=1e-5)
    out = sparse_attention(q, k, v, iq, iw, ik, 0, T - 1, 0.2, chunk=32)
    assert torch.allclose(out[:, :, : T - 1], dense[:, :, : T - 1], atol=1e-5)


def test_sparse_matches_explicit_gather():
    """Masked-dense implementation == literally gathering the k selected keys."""
    q, k, v, iq, iw, ik = _inputs()
    B, H, T, D = q.shape
    Hkv, topk, scale = k.shape[1], 16, 1 / math.sqrt(D)
    out = sparse_attention(q, k, v, iq, iw, ik, 0, topk, scale, chunk=40)
    idx = topk_indices(iq, iw, ik, 0, topk)                                # (B, T, k)
    for b in range(B):
        for t in range(T):
            sel = idx[b, t][idx[b, t] >= 0]
            assert sel.numel() == min(topk, t + 1) and (sel <= t).all()
            for h in range(H):
                kh, vh = k[b, h // (H // Hkv), sel], v[b, h // (H // Hkv), sel]
                p = ((kh @ q[b, h, t]) * scale).softmax(0)
                assert torch.allclose(out[b, h, t], p @ vh, atol=1e-5)


def test_chunked_decode_matches_prefill():
    """Scoring queries one at a time (as in cached decoding) selects the same keys."""
    q, k, v, iq, iw, ik = _inputs(T=64)
    full = topk_indices(iq, iw, ik, 0, 8)
    for t in range(40, 64):
        step = topk_indices(iq[:, t : t + 1], iw[:, t : t + 1], ik[:, : t + 1], t, 8)
        assert torch.equal(step[:, 0].sort(-1).values, full[:, t].sort(-1).values)


def test_indices_to_mask_padding():
    idx = torch.tensor([[[0, 2, -1]]], device=DEV)
    assert indices_to_mask(idx, 4).tolist() == [[[True, False, True, False]]]
