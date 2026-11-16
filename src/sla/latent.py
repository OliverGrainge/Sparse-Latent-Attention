"""PCA initialisation of the latent KV projections from real activations.

For layer l let y = [W_k x ; W_v x] in R^{2*Hkv*D} be the pre-norm keys and
values. We want the best rank-r linear code c = A x with y ~= B c.

Since y is linear in x, we take the top-r principal subspace U of the
*weighted* activations W y (uncentred second moment on calibration text):

    W_down = U^T W W_kv          W_up = W^-1 U

The weighting W matters a lot. Plain PCA minimises error in pre-norm K/V
space, but that is not what the model sees:

  * keys go through k_norm (per-dim weights up to ~70x, divided by the
    head's RMS) and are then dotted with queries, so a key dim's importance
    is |k_norm_w[d]| / rms(head) * rms(q[d]);
  * values only matter through o_proj, so a value dim's importance is the
    norm of the o_proj columns it feeds.

modes:  "plain"      W = I
        "balance"    K and V blocks rescaled to equal RMS
        "importance" per-dim weights above, then K/V blocks balanced
"""

from __future__ import annotations

import torch
from tqdm import tqdm

from .model import Qwen3


@torch.no_grad()
def collect_kv_second_moments(model: Qwen3, batches, progress: bool = True) -> dict:
    """Per layer: second moment of y = [k_pre; v] (kv_dim x kv_dim) and the mean
    squared post-norm query per (head, dim)."""
    cfg = model.cfg
    H, D = cfg.num_heads, cfg.head_dim
    kv_dim = 2 * cfg.num_kv_heads * D
    dev = model.embed.weight.device
    moments = [torch.zeros(kv_dim, kv_dim, device=dev, dtype=torch.float64) for _ in model.layers]
    q_sq = [torch.zeros(H, D, device=dev, dtype=torch.float64) for _ in model.layers]
    count = 0

    def make_hook(i, attn):
        def hook(_mod, args):
            x = args[0].flatten(0, 1)                             # (N, C) attention input
            y = torch.cat([attn.k_proj(x), attn.v_proj(x)], dim=-1).float()
            moments[i] += (y.T @ y).double()
            q = attn.q_norm(attn.q_proj(x).view(-1, H, D)).float()
            q_sq[i] += q.pow(2).sum(0).double()
        return hook

    was_latent = model.layers[0].attn.use_latent
    if was_latent:
        model.set_latent(False)
    handles = [l.attn.register_forward_pre_hook(make_hook(i, l.attn)) for i, l in enumerate(model.layers)]
    try:
        for ids in tqdm(batches, disable=not progress, desc="calibrate"):
            ids = ids.to(dev)
            model.hidden_states(ids)
            count += ids.numel()
    finally:
        for h in handles:
            h.remove()
        if was_latent:
            model.set_latent(True)
    return {"moments": [m / count for m in moments], "q_sq": [q / count for q in q_sq], "tokens": count}


@torch.no_grad()
def _importance(attn, M: torch.Tensor, q_sq: torch.Tensor, cfg) -> torch.Tensor:
    """Per-dim importance of y = [k_pre; v] for one layer, shape (kv_dim,)."""
    Hkv, D, H = cfg.num_kv_heads, cfg.head_dim, cfg.num_heads
    g = H // Hkv
    half = Hkv * D
    # RoPE rotates dim d together with d + D/2: average query energy over each pair.
    q_pair = q_sq.view(H, 2, D // 2).mean(1).repeat(1, 2)                   # (H, D)
    q_grp = q_pair.view(Hkv, g, D).sum(1).sqrt()                            # (Hkv, D)
    head_rms = M.diagonal()[:half].view(Hkv, D).mean(1).sqrt()              # (Hkv,)
    w_k = attn.k_norm.weight.double().abs()                                 # (D,)
    imp_k = w_k[None, :] * q_grp / head_rms[:, None]
    # o_proj columns h*D + d for every query head h in the group of KV head j.
    Wo = attn.o_proj.weight.double()                                        # (C, H*D)
    col_sq = Wo.pow(2).sum(0).view(Hkv, g, D).sum(1)                        # (Hkv, D)
    imp_v = col_sq.sqrt()
    return torch.cat([imp_k.flatten(), imp_v.flatten()]).clamp_min(1e-6)


@torch.no_grad()
def init_latent_from_pca(model: Qwen3, calib: dict, rank: int, mode: str = "importance") -> list[dict]:
    """Attach LatentKV of `rank` to every layer, initialised by weighted PCA.
    Returns per-layer statistics (errors are in original, unweighted units)."""
    cfg = model.cfg
    half = cfg.num_kv_heads * cfg.head_dim
    model.add_latent_kv(rank)
    stats = []
    for i, layer in enumerate(model.layers):
        attn, M = layer.attn, calib["moments"][i]
        n = M.shape[0]
        if mode == "plain":
            w = torch.ones(n, device=M.device, dtype=M.dtype)
        elif mode in ("balance", "importance"):
            w = _importance(attn, M, calib["q_sq"][i], cfg) if mode == "importance" else \
                torch.ones(n, device=M.device, dtype=M.dtype)
            # Balance: equal weighted energy in the K and V blocks.
            e = w.pow(2) * M.diagonal()
            w = torch.cat([w[:half] / e[:half].mean().sqrt(), w[half:] / e[half:].mean().sqrt()])
        else:
            raise ValueError(mode)

        Mw = w[:, None] * M * w[None, :]                           # moment of W y
        evals, evecs = torch.linalg.eigh(Mw)                       # ascending
        U = evecs[:, -rank:].flip(-1)                              # (kv_dim, r), top first
        W_kv = torch.cat([attn.k_proj.weight, attn.v_proj.weight], 0).double()  # (kv_dim, C)
        attn.latent.down.copy_((U.T @ (w[:, None] * W_kv)).float())
        attn.latent.up.copy_((U / w[:, None]).float())

        Q = (U @ U.T) * w[None, :] / w[:, None]                    # y -> reconstruction
        R = torch.eye(n, device=M.device, dtype=M.dtype) - Q
        resid = R @ M @ R.T                                        # second moment of the error
        stats.append({
            "layer": i,
            "energy_kept": float(evals[-rank:].sum() / evals.sum()),
            "k_rel_err": float(resid.diagonal()[:half].sum() / M.diagonal()[:half].sum()),
            "v_rel_err": float(resid.diagonal()[half:].sum() / M.diagonal()[half:].sum()),
        })
    return stats
