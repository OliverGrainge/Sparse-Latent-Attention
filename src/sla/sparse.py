"""Sparse attention driven by a lightning indexer (DeepSeek Sparse Attention style).

For every query t the indexer scores every earlier token s with a tiny,
cheap attention-like function

    I(t, s) = sum_j  w_{t,j} * relu(q_{t,j} . k_s)          (j = indexer heads)

and the real attention then only looks at the top-k tokens by I(t, .). The
indexer key k_s is a single small vector per token (shared by all indexer
heads), cached next to the latent.

The selection is shared by all query heads of a layer, so attention reads k
tokens of the latent cache per query instead of all S of them. With a latent
KV cache that matters twice: only the selected latents have to be expanded
back into keys/values.

This file holds the torch reference path (used for training and evaluation);
the fused decode kernel lives in `kernels.py`.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Always-kept tokens (they count towards the top-k budget): the first few tokens
# ("attention sinks", which hold most of the attention mass in Qwen3's late
# layers) and a short window of the most recent ones. The indexer only has to
# rank everything else, which is where it earns its keep.
SINK = 4
LOCAL = 128


def _rope_half(x: torch.Tensor, positions: torch.Tensor, rope_dim: int, theta: float) -> torch.Tensor:
    """Half-split RoPE on the first `rope_dim` channels of x (..., T, [heads,] d).
    positions: (T,). x layout: (B, T, d) or (B, T, H, d)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, rope_dim, 2, device=x.device).float() / rope_dim))
    ang = positions.float()[:, None] * inv_freq[None, :]                  # (T, rope_dim/2)
    cos, sin = ang.cos(), ang.sin()
    if x.dim() == 4:
        cos, sin = cos[:, None], sin[:, None]
    r, rest = x[..., :rope_dim].float(), x[..., rope_dim:]
    a, b = r.chunk(2, dim=-1)
    r = torch.cat([a * cos - b * sin, b * cos + a * sin], dim=-1).to(x.dtype)
    return torch.cat([r, rest], dim=-1)


class Indexer(nn.Module):
    """Lightning indexer for one layer.

    Weights are fp32 (they are what we train) and cast to the activation dtype
    on use. Keys go through a weightless RMSNorm so their scale is fixed and
    the query side carries all the magnitude.
    """

    def __init__(self, hidden_size: int, n_heads: int = 8, head_dim: int = 64, rope_dim: int = 32,
                 rope_theta: float = 10_000.0):
        super().__init__()
        self.n_heads, self.head_dim, self.rope_dim, self.rope_theta = n_heads, head_dim, rope_dim, rope_theta
        self.wq = nn.Parameter(torch.randn(n_heads * head_dim, hidden_size) / math.sqrt(hidden_size))
        self.wk = nn.Parameter(torch.randn(head_dim, hidden_size) / math.sqrt(hidden_size))
        self.ww = nn.Parameter(torch.zeros(n_heads, hidden_size))
        self.w_bias = nn.Parameter(torch.zeros(n_heads))

    def keys(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C) -> post-RoPE indexer keys (B, T, d). This is what gets cached."""
        k = F.linear(x, self.wk.to(x.dtype))
        k = k * torch.rsqrt(k.float().pow(2).mean(-1, keepdim=True) + 1e-6).to(k.dtype)
        return _rope_half(k, positions, self.rope_dim, self.rope_theta)

    def queries(self, x: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, T, C) -> queries (B, T, H, d) and per-head weights (B, T, H).
        The 1/sqrt(d) and 1/sqrt(H) scales are folded in here."""
        B, T, _ = x.shape
        q = F.linear(x, self.wq.to(x.dtype)).view(B, T, self.n_heads, self.head_dim)
        q = _rope_half(q, positions, self.rope_dim, self.rope_theta) / math.sqrt(self.head_dim)
        w = (F.linear(x, self.ww.to(x.dtype)) + self.w_bias.to(x.dtype)).float()
        w = F.softplus(w + 1.0) / math.sqrt(self.n_heads)
        return q, w


def index_scores(q: torch.Tensor, w: torch.Tensor, k: torch.Tensor, q_start: int) -> torch.Tensor:
    """I(t, s) for queries q (B, T, H, d), weights w (B, T, H) and keys k (B, S, d)
    whose first entry is position 0. Future keys get -inf. Returns (B, T, S) fp32."""
    T, S = q.shape[1], k.shape[1]
    dots = torch.einsum("bthd,bsd->bths", q.float(), k.float()).relu_()    # (B, T, H, S), fp32 to avoid ties
    s = torch.einsum("bths,bth->bts", dots, w)
    q_pos = torch.arange(q_start, q_start + T, device=q.device)
    future = torch.arange(S, device=q.device)[None, :] > q_pos[:, None]
    return force_keep(s.masked_fill(future, float("-inf")), q_pos)


def always_kept(q_pos: torch.Tensor, S: int, sink: int = SINK, local: int = LOCAL) -> torch.Tensor:
    """(Q, S) bool: the sink tokens and the `local` most recent tokens (incl. the query's own)."""
    pos = torch.arange(S, device=q_pos.device)[None, :]
    dist = q_pos[:, None] - pos
    return (dist >= 0) & ((pos < sink) | (dist < local))


def force_keep(scores: torch.Tensor, q_pos: torch.Tensor, sink: int = SINK, local: int = LOCAL) -> torch.Tensor:
    """Give the always-kept keys an infinite score so top-k selects them.
    scores: (..., Q, S) for queries at positions q_pos (Q,). In place."""
    return scores.masked_fill_(always_kept(q_pos, scores.shape[-1], sink, local), float("inf"))


@torch.no_grad()
def topk_indices(q, w, k, q_start: int, topk: int, chunk: int = 1024) -> torch.Tensor:
    """Top-k key indices per query, (B, T, min(topk, S)) int64. Entries that point
    at future tokens (only when t < topk) are -1."""
    B, T = q.shape[:2]
    S = k.shape[1]
    kk = min(topk, S)
    out = torch.empty((B, T, kk), device=q.device, dtype=torch.long)
    for t0 in range(0, T, chunk):
        t1 = min(T, t0 + chunk)
        s = index_scores(q[:, t0:t1], w[:, t0:t1], k, q_start + t0)
        vals, idx = s.topk(kk, dim=-1)
        out[:, t0:t1] = idx.masked_fill(vals == float("-inf"), -1)
    return out


def indices_to_mask(idx: torch.Tensor, S: int) -> torch.Tensor:
    """(B, T, k) indices (with -1 padding) -> boolean (B, T, S) visibility mask."""
    B, T, _ = idx.shape
    mask = torch.zeros((B, T, S + 1), device=idx.device, dtype=torch.bool)
    mask.scatter_(-1, idx.clamp_min(-1) % (S + 1), True)  # -1 lands in the spare last column
    return mask[..., :S]


def sink_window_mask(q_start: int, T: int, S: int, topk: int, sink: int = 4, device=None) -> torch.Tensor:
    """(1, 1, T, S) mask keeping the first `sink` tokens and the topk - sink most
    recent ones (StreamingLLM-style baseline with the same budget as top-k)."""
    q_pos = torch.arange(q_start, q_start + T, device=device)[:, None]
    pos = torch.arange(S, device=device)[None, :]
    dist = q_pos - pos
    return ((dist >= 0) & ((dist < topk - sink) | (pos < sink)))[None, None]


def sparse_attention(q, k, v, iq, iw, ik, q_start: int, topk: int, scale: float,
                     chunk: int = 2048, selector: str = "indexer") -> torch.Tensor:
    """Reference top-k sparse attention.

    q: (B, H, T, D); k, v: (B, Hkv, S, D); indexer queries/weights iq (B, T, Hi, di),
    iw (B, T, Hi) and keys ik (B, S, di). Each query attends to its top-k keys by
    indexer score (all of them if it has fewer than k predecessors).
    Implemented as masked dense attention, chunked over queries.
    """
    B, H, T, D = q.shape
    S = k.shape[2]
    if S <= topk:  # every query sees all its predecessors: plain causal attention
        from .model import causal_attention
        return causal_attention(q, k, v, q_start, scale, impl="sdpa")
    out = torch.empty_like(q)
    for t0 in range(0, T, chunk):
        t1 = min(T, t0 + chunk)
        if selector == "sink_window":
            mask = sink_window_mask(q_start + t0, t1 - t0, S, topk, device=q.device)
        else:
            idx = topk_indices(iq[:, t0:t1], iw[:, t0:t1], ik, q_start + t0, topk)
            mask = indices_to_mask(idx, S)[:, None]                          # (B, 1, c, S)
        out[:, :, t0:t1] = F.scaled_dot_product_attention(
            q[:, :, t0:t1], k, v, attn_mask=mask, scale=scale, enable_gqa=True)
    return out


def attention_target(q: torch.Tensor, k: torch.Tensor, q_pos: torch.Tensor, scale: float) -> torch.Tensor:
    """Head-averaged attention distribution of the real attention for a subset of
    queries. q: (B, H, Q, D) at absolute positions q_pos (Q,), k: (B, Hkv, S, D).
    Returns (B, Q, S) fp32 rows summing to 1 (zeros on future keys)."""
    B, H, Q, D = q.shape
    Hkv, S = k.shape[1], k.shape[2]
    g = H // Hkv
    scores = torch.einsum("bjgqd,bjsd->bjgqs", q.view(B, Hkv, g, Q, D), k).float() * scale
    future = torch.arange(S, device=q.device)[None, :] > q_pos[:, None]
    p = scores.masked_fill(future, float("-inf")).softmax(-1)
    return p.view(B, H, Q, S).mean(1)


def indexer_kl(indexer: Indexer, x: torch.Tensor, target: torch.Tensor, q_pos: torch.Tensor,
               restrict: torch.Tensor | None = None, sink: int = SINK, local: int = LOCAL) -> torch.Tensor:
    """Mean over queries of KL(target || softmax(I)). x: (B, T, C) attention input,
    target: (B, Q, S) from attention_target for queries at q_pos (Q,).
    `restrict` (B, Q, S) bool limits both distributions to a key subset (the
    DSA sparse-stage loss); the target is renormalised over it. The always-kept
    tokens (sinks + local window) are excluded from the loss: the indexer only
    has to rank the rest."""
    S = target.shape[-1]
    pos = torch.arange(S, device=x.device)
    ik = indexer.keys(x[:, :S], pos)
    iq, iw = indexer.queries(x[:, q_pos], q_pos)
    # Scores for the sampled queries: each query masks keys beyond its own position.
    dots = torch.einsum("bqhd,bsd->bqhs", iq, ik).float().relu()
    logits = torch.einsum("bqhs,bqh->bqs", dots, iw)
    kept = always_kept(q_pos, S, sink, local)
    invalid = (pos[None, :] > q_pos[:, None]) | kept
    if sink or local:
        target = target.masked_fill(kept, 0.0)
        target = target / target.sum(-1, keepdim=True).clamp_min(1e-12)
    if restrict is not None:
        invalid = invalid | ~restrict
        target = target.masked_fill(~restrict, 0.0)
        target = target / target.sum(-1, keepdim=True).clamp_min(1e-12)
    # Zero the log-probs of excluded keys after the softmax so no inf/nan reaches the gradient.
    logp = logits.masked_fill(invalid, float("-inf")).log_softmax(-1).masked_fill(invalid, 0.0)
    kl = target * (target.clamp_min(1e-30).log() - logp)
    return kl.sum(-1).mean()


@torch.no_grad()
def recall_at_k(indexer: Indexer, x: torch.Tensor, target: torch.Tensor, q_pos: torch.Tensor,
                topks: list[int], sink: int = 4) -> dict[str, dict[int, float]]:
    """Fraction of the real (head-averaged) attention mass that falls on the keys a
    selection rule keeps, averaged over the sampled queries, for each k:

      indexer      sinks + local window, then the best of the rest by indexer score (what sparse attention uses)
      indexer_raw  top-k by indexer score alone
      oracle       top-k by the true attention mass (upper bound for any rule)
      window       the k most recent tokens
      sink_window  the first `sink` tokens plus the k - sink most recent (StreamingLLM)
    """
    S = target.shape[-1]
    pos = torch.arange(S, device=x.device)
    ik = indexer.keys(x[:, :S], pos)
    iq, iw = indexer.queries(x[:, q_pos], q_pos)
    scores = torch.einsum("bqh,bqhs->bqs", iw, torch.einsum("bqhd,bsd->bqhs", iq, ik).float().relu_())
    scores = scores.masked_fill(pos[None, :] > q_pos[:, None], float("-inf"))
    dist = q_pos[:, None] - pos[None, :]                                  # (Q, S), >= 0 for past keys
    forced = force_keep(scores.clone(), q_pos, sink)
    out: dict[str, dict[int, float]] = {"indexer": {}, "indexer_raw": {}, "oracle": {}, "window": {},
                                        "sink_window": {}}
    for k in topks:
        kk = min(k, S)
        out["indexer"][k] = target.gather(-1, forced.topk(kk, dim=-1).indices).sum(-1).mean().item()
        out["indexer_raw"][k] = target.gather(-1, scores.topk(kk, dim=-1).indices).sum(-1).mean().item()
        out["oracle"][k] = target.topk(kk, dim=-1).values.sum(-1).mean().item()
        win = (dist >= 0) & (dist < k)
        out["window"][k] = (target * win).sum(-1).mean().item()
        sw = (dist >= 0) & ((dist < k - sink) | (pos[None, :] < sink))
        out["sink_window"][k] = (target * sw).sum(-1).mean().item()
    return out
