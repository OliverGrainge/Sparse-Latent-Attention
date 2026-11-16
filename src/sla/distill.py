"""Logit distillation: the latent-KV model (student) learns to match the
original dense-attention model (teacher). Both are the *same* weights; only
the attention K/V path differs, so no second model copy is needed."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .model import Qwen3


def _chunk_kl(lm_head_w: torch.Tensor, h_s: torch.Tensor, h_t: torch.Tensor) -> torch.Tensor:
    """sum over positions of KL(teacher || student) for one chunk of positions."""
    log_s = F.linear(h_s, lm_head_w).float().log_softmax(-1)
    log_t = F.linear(h_t, lm_head_w).float().log_softmax(-1)
    return F.kl_div(log_s, log_t, log_target=True, reduction="sum")


def distill_loss(model: Qwen3, ids: torch.Tensor, chunk: int = 1024) -> tuple[torch.Tensor, dict]:
    """Mean per-token KL(teacher || student). Returns (loss, metrics)."""
    with torch.no_grad():
        model.set_latent(False)
        h_t = model.hidden_states(ids)
    model.set_latent(True)
    h_s = model.hidden_states(ids)

    W = model.lm_head.weight
    total = h_s.new_zeros((), dtype=torch.float32)
    for t0 in range(0, ids.shape[1], chunk):
        sl = slice(t0, t0 + chunk)
        # Recompute logits in backward instead of storing (chunk x vocab) fp32 twice.
        total = total + checkpoint(_chunk_kl, W, h_s[:, sl], h_t[:, sl], use_reentrant=False)
    n = ids.numel()
    return total / n, {"kl": total.item() / n}


@torch.no_grad()
def evaluate(model: Qwen3, batches, chunk: int = 1024) -> dict:
    """Mean KL to the teacher and next-token NLL for teacher and student."""
    kl = nll_s = nll_t = 0.0
    n = 0
    for ids in batches:
        ids = ids.to(model.embed.weight.device)
        model.set_latent(False)
        h_t = model.hidden_states(ids)
        model.set_latent(True)
        h_s = model.hidden_states(ids)
        for t0 in range(0, ids.shape[1] - 1, chunk):
            sl = slice(t0, min(t0 + chunk, ids.shape[1] - 1))
            tgt = ids[:, sl.start + 1 : sl.stop + 1]
            log_s = model.lm_head(h_s[:, sl]).float().log_softmax(-1)
            log_t = model.lm_head(h_t[:, sl]).float().log_softmax(-1)
            kl += F.kl_div(log_s, log_t, log_target=True, reduction="sum").item()
            nll_s += -log_s.gather(-1, tgt[..., None]).sum().item()
            nll_t += -log_t.gather(-1, tgt[..., None]).sum().item()
            n += tgt.numel()
    return {"kl": kl / n, "nll_student": nll_s / n, "nll_teacher": nll_t / n}
