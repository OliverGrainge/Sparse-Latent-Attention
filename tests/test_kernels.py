"""Triton decode kernels vs. plain torch references (synthetic tensors)."""

import math

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
triton = pytest.importorskip("triton")

from sla.kernels import index_scores_decode, sparse_latent_decode  # noqa: E402
from sla.model import RMSNorm, apply_rope, rope_cos_sin  # noqa: E402

DEV = "cuda"


def reference_sparse_latent(q, c, w_up, knorm, idx, theta, Hkv):
    """Gather -> expand -> k_norm -> RoPE -> softmax attention, per batch element (fp32)."""
    B, H, D = q.shape
    out = torch.empty_like(q, dtype=torch.float32)
    for b in range(B):
        sel = idx[b][idx[b] >= 0].long()
        kv = c[b, sel].float() @ w_up.float().T                              # (n, 2*Hkv*D)
        k, v = kv.split(Hkv * D, dim=-1)
        k = knorm(k.view(-1, Hkv, D)).transpose(0, 1)                          # (Hkv, n, D)
        cos, sin = rope_cos_sin(sel, D, theta, torch.float32)
        k = apply_rope(k, cos, sin)
        v = v.view(-1, Hkv, D).transpose(0, 1)
        g = H // Hkv
        qb = q[b].float().view(Hkv, g, D)
        p = (qb @ k.transpose(1, 2) / math.sqrt(D)).softmax(-1)               # (Hkv, g, n)
        out[b] = (p @ v).reshape(H, D)
    return out


@pytest.mark.parametrize("K,S", [(64, 300), (1000, 5000), (2048, 2048)])
def test_sparse_latent_decode(K, S):
    torch.manual_seed(0)
    B, H, Hkv, D, R, theta = 2, 16, 8, 128, 512, 1e6
    dt = torch.bfloat16
    q = torch.randn(B, H, D, device=DEV, dtype=dt)
    c = torch.randn(B, S, R, device=DEV, dtype=dt)
    w_up = (torch.randn(2 * Hkv * D, R, device=DEV) / math.sqrt(R)).to(dt)
    knorm = RMSNorm(D, 1e-6).to(DEV)
    with torch.no_grad():
        knorm.weight.uniform_(0.5, 2.0)
    idx = torch.stack([torch.randperm(S, device=DEV)[:K] for _ in range(B)]).int()
    idx[1, K // 2 :] = -1                                                       # padding is ignored
    inv_freq = 1.0 / (theta ** (torch.arange(0, D, 2, device=DEV).float() / D))
    out = sparse_latent_decode(q, c, w_up.T.contiguous(), knorm.weight.to(dt), inv_freq, idx,
                               1 / math.sqrt(D), 1e-6, Hkv)
    ref = reference_sparse_latent(q, c, w_up, knorm, idx, theta, Hkv)
    err = (out.float() - ref).abs().max().item()
    assert err < 3e-2, err


def test_index_scores_decode():
    torch.manual_seed(0)
    B, Hi, d, S = 2, 8, 64, 3000
    q = torch.randn(B, Hi, d, device=DEV, dtype=torch.bfloat16)
    w = torch.rand(B, Hi, device=DEV)
    k = torch.randn(B, S + 100, d, device=DEV, dtype=torch.bfloat16)
    out = index_scores_decode(q, w, k, S)
    ref = torch.einsum("bh,bhs->bs", w.bfloat16().float(),
                       torch.einsum("bhd,bsd->bhs", q.float(), k[:, :S].float()).relu())
    assert out.shape == (B, S)
    assert torch.allclose(out, ref, rtol=2e-2, atol=2e-1)
