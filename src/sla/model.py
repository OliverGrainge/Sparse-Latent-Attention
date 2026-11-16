"""Qwen3 dense decoder, written from scratch in native PyTorch.

Every operation (RMSNorm, RoPE, grouped-query attention, SwiGLU MLP) is
spelled out explicitly so later we can swap the attention for a sparse,
indexer-driven version without fighting a framework.

Shapes:  B = batch, T = query length, S = key length, H = query heads,
         Hkv = key/value heads, D = head dim, C = hidden size.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class ModelConfig:
    vocab_size: int = 151936
    hidden_size: int = 2048
    intermediate_size: int = 6144
    num_layers: int = 28
    num_heads: int = 16
    num_kv_heads: int = 8
    head_dim: int = 128
    rope_theta: float = 1_000_000.0
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 40960
    tie_word_embeddings: bool = True

    @classmethod
    def from_hf_config(cls, path: str | Path) -> "ModelConfig":
        cfg = json.loads(Path(path).read_text())
        if cfg.get("rope_scaling"):
            raise NotImplementedError(f"rope_scaling not supported: {cfg['rope_scaling']}")
        return cls(
            vocab_size=cfg["vocab_size"],
            hidden_size=cfg["hidden_size"],
            intermediate_size=cfg["intermediate_size"],
            num_layers=cfg["num_hidden_layers"],
            num_heads=cfg["num_attention_heads"],
            num_kv_heads=cfg["num_key_value_heads"],
            head_dim=cfg.get("head_dim", cfg["hidden_size"] // cfg["num_attention_heads"]),
            rope_theta=cfg["rope_theta"],
            rms_norm_eps=cfg["rms_norm_eps"],
            max_position_embeddings=cfg["max_position_embeddings"],
            tie_word_embeddings=cfg.get("tie_word_embeddings", False),
        )


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalise in fp32 for stability, then cast back before the scale.
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


def rope_cos_sin(positions: torch.Tensor, head_dim: int, theta: float, dtype: torch.dtype):
    """cos/sin tables for the given positions. positions: (T,) -> (T, D)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=positions.device).float() / head_dim))
    freqs = positions.float()[:, None] * inv_freq[None, :]          # (T, D/2)
    emb = torch.cat([freqs, freqs], dim=-1)                          # (T, D)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, H, T, D); cos/sin: (T, D). Uses the 'half-split' RoPE layout."""
    return x * cos + rotate_half(x) * sin


# --------------------------------------------------------------------------- #
# Attention
# --------------------------------------------------------------------------- #
# Upper bound on the fp32 score block materialised at once by eager attention.
ATTN_SCORE_BUDGET_BYTES = 1 << 30


def causal_attention(
    q: torch.Tensor,            # (B, H, T, D)
    k: torch.Tensor,            # (B, Hkv, S, D)
    v: torch.Tensor,            # (B, Hkv, S, D)
    q_start: int,               # absolute position of q[:, :, 0]; keys start at position 0
    scale: float,
    impl: str = "eager",
) -> torch.Tensor:
    """Causal grouped-query attention. Returns (B, H, T, D).

    'eager' is the explicit softmax(QK^T)V reference, computed in chunks of
    queries so the (T x S) score matrix never has to exist all at once.
    'sdpa' calls torch's fused kernel and is used only as a speed/sanity path.
    """
    B, H, T, D = q.shape
    Hkv, S = k.shape[1], k.shape[2]
    group = H // Hkv

    if impl == "sdpa":
        if T == S:
            mask, is_causal = None, True
        else:
            q_pos = torch.arange(q_start, q_start + T, device=q.device)
            mask, is_causal = torch.arange(S, device=q.device)[None, :] <= q_pos[:, None], False
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=is_causal, scale=scale, enable_gqa=True
        )

    if impl != "eager":
        raise ValueError(f"unknown attention impl {impl!r}")

    # Group query heads under their shared KV head: (B, Hkv, group, T, D). The group
    # and query dims are folded into the matmul rows, so K/V are never broadcast
    # (broadcasting would make torch materialise an expanded copy of the cache).
    qg = q.view(B, Hkv, group, T, D)
    kt = k.transpose(-1, -2)                                         # (B, Hkv, D, S)
    key_pos = torch.arange(S, device=q.device)

    out = torch.empty_like(qg)
    # fp32 scores are (B, H, chunk, S) -> pick the chunk to stay under the byte budget.
    chunk = max(1, min(T, ATTN_SCORE_BUDGET_BYTES // (B * H * S * 4)))
    for t0 in range(0, T, chunk):
        t1 = min(T, t0 + chunk)
        c = t1 - t0
        rows = qg[:, :, :, t0:t1].reshape(B, Hkv, group * c, D)
        scores = (rows @ kt).view(B, Hkv, group, c, S).float() * scale  # (B, Hkv, g, c, S)
        q_pos = torch.arange(q_start + t0, q_start + t1, device=q.device)
        visible = key_pos[None, :] <= q_pos[:, None]                # (c, S)
        scores = scores.masked_fill(~visible, float("-inf"))
        probs = scores.softmax(dim=-1).to(v.dtype)
        out[:, :, :, t0:t1] = (probs.view(B, Hkv, group * c, S) @ v).view(B, Hkv, group, c, D)
    return out.view(B, H, T, D)


class KVCache:
    """Pre-allocated per-layer buffers for incremental decoding.

    Dense attention caches full keys/values (B, Hkv, S, D) per layer. With
    `latent_rank` set, only the compressed latent (B, S, r) is cached and K/V
    are re-expanded from it on every step - that is the whole point of MLA.
    """

    def __init__(self, cfg: ModelConfig, batch: int, max_len: int, device, dtype, latent_rank: int | None = None,
                 index_dim: int | None = None):
        self.max_len = max_len
        self.length = 0  # tokens already stored (shared by all layers)
        self.latent_rank = latent_rank
        L = cfg.num_layers
        if latent_rank is None:
            shape = (batch, cfg.num_kv_heads, max_len, cfg.head_dim)
            self.k = [torch.empty(shape, device=device, dtype=dtype) for _ in range(L)]
            self.v = [torch.empty(shape, device=device, dtype=dtype) for _ in range(L)]
        else:
            self.c = [torch.empty((batch, max_len, latent_rank), device=device, dtype=dtype) for _ in range(L)]
        # Lightning-indexer keys (one small vector per token), when sparse attention is on.
        self.ik = None if index_dim is None else             [torch.empty((batch, max_len, index_dim), device=device, dtype=dtype) for _ in range(L)]

    def _check(self, end: int):
        if end > self.max_len:
            raise ValueError(f"KV cache overflow: {end} > {self.max_len}")

    def update(self, layer: int, k: torch.Tensor, v: torch.Tensor, start: int):
        end = start + k.shape[2]
        self._check(end)
        self.k[layer][:, :, start:end] = k
        self.v[layer][:, :, start:end] = v
        return self.k[layer][:, :, :end], self.v[layer][:, :, :end]

    def update_latent(self, layer: int, c: torch.Tensor, start: int):
        end = start + c.shape[1]
        self._check(end)
        self.c[layer][:, start:end] = c
        return self.c[layer][:, :end]

    def update_index(self, layer: int, ik: torch.Tensor, start: int):
        end = start + ik.shape[1]
        self._check(end)
        self.ik[layer][:, start:end] = ik
        return self.ik[layer][:, :end]

    def nbytes(self) -> int:
        bufs = self.c if self.latent_rank is not None else self.k + self.v
        bufs = bufs + (self.ik or [])
        return sum(b.numel() * b.element_size() for b in bufs)


class LatentKV(nn.Module):
    """MLA-style joint low-rank KV compression for one layer.

        c   = W_down x          (rank r, the only thing cached)
        k,v = split(W_up c)     (pre-norm keys and values for all KV heads)

    Weights are kept in fp32 (they are what we train) and cast to the
    activation dtype on use.
    """

    def __init__(self, hidden_size: int, rank: int, kv_dim: int):
        super().__init__()
        self.rank = rank
        self.down = nn.Parameter(torch.zeros(rank, hidden_size))
        self.up = nn.Parameter(torch.zeros(kv_dim, rank))

    def compress(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.down.to(x.dtype))

    def expand(self, c: torch.Tensor) -> torch.Tensor:
        return F.linear(c, self.up.to(c.dtype))


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        H, Hkv, D, C = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim, cfg.hidden_size
        self.q_proj = nn.Linear(C, H * D, bias=False)
        self.k_proj = nn.Linear(C, Hkv * D, bias=False)
        self.v_proj = nn.Linear(C, Hkv * D, bias=False)
        self.o_proj = nn.Linear(H * D, C, bias=False)
        # Qwen3 applies RMSNorm to each query/key head before RoPE.
        self.q_norm = RMSNorm(D, cfg.rms_norm_eps)
        self.k_norm = RMSNorm(D, cfg.rms_norm_eps)
        self.scale = 1.0 / math.sqrt(D)
        # Optional MLA-style latent KV path (see LatentKV); off unless use_latent.
        self.latent: LatentKV | None = None
        self.use_latent = False
        # Optional lightning indexer + top-k sparse attention (see sparse.py).
        self.indexer = None
        self.sparse_topk = 0          # 0 = dense attention
        self.selector = "indexer"     # or "sink_window" (baseline with the same token budget)
        self.observer = None          # callback(attn, x, q, k, start_pos), used to train the indexer

    def forward(self, x, cos, sin, start_pos: int, cache: KVCache | None, impl: str):
        B, T, _ = x.shape
        H, Hkv, D = self.cfg.num_heads, self.cfg.num_kv_heads, self.cfg.head_dim

        q = self.q_norm(self.q_proj(x).view(B, T, H, D)).transpose(1, 2)    # (B, H, T, D)
        q = apply_rope(q, cos, sin)

        if impl == "triton" and self.use_latent and self.sparse_topk and cache is not None and T == 1:
            o = self._sparse_latent_decode(x, q[:, :, 0], start_pos, cache)            # (B, H, D)
            return self.o_proj(o.reshape(B, 1, H * D))
        if impl == "triton":  # the kernels only cover sparse latent decoding; SDPA for the rest
            impl = "sdpa"

        if self.use_latent:
            k, v = self._latent_kv(x, cos, sin, start_pos, cache)
        else:
            k = self.k_norm(self.k_proj(x).view(B, T, Hkv, D)).transpose(1, 2)  # (B, Hkv, T, D)
            v = self.v_proj(x).view(B, T, Hkv, D).transpose(1, 2)
            k = apply_rope(k, cos, sin)
            if cache is not None:
                k, v = cache.update(self.layer_idx, k, v, start_pos)

        if self.observer is not None:
            self.observer(self, x, q, k, start_pos)
        if self.sparse_topk:
            o = self._sparse(x, q, k, v, start_pos, cache)
        else:
            o = causal_attention(q, k, v, q_start=start_pos, scale=self.scale, impl=impl)
        return self.o_proj(o.transpose(1, 2).reshape(B, T, H * D))

    def _sparse(self, x, q, k, v, start_pos, cache):
        from .sparse import sparse_attention
        positions = torch.arange(start_pos, start_pos + x.shape[1], device=x.device)
        ik = self.indexer.keys(x, positions)
        if cache is not None:
            ik = cache.update_index(self.layer_idx, ik, start_pos)
        iq, iw = self.indexer.queries(x, positions)
        return sparse_attention(q, k, v, iq, iw, ik, start_pos, self.sparse_topk, self.scale,
                                selector=self.selector)

    def _sparse_latent_decode(self, x, q, start_pos, cache):
        """One decode step with the fused Triton kernels: score every cached token with
        the indexer, keep the top-k, and attend over them straight from the latent cache."""
        from .kernels import index_scores_decode, sparse_latent_decode
        from .sparse import force_keep
        L, S = self.layer_idx, start_pos + 1
        pos = torch.arange(start_pos, S, device=x.device)
        cache.update_latent(L, self.latent.compress(x), start_pos)
        cache.update_index(L, self.indexer.keys(x, pos), start_pos)
        iq, iw = self.indexer.queries(x, pos)
        if S <= self.sparse_topk:
            idx = torch.arange(S, device=x.device, dtype=torch.int32).expand(x.shape[0], S).contiguous()
        else:
            scores = force_keep(index_scores_decode(iq[:, 0], iw[:, 0], cache.ik[L], S), pos)
            idx = scores.topk(self.sparse_topk, dim=-1, sorted=False).indices.int()
        w_up_t, inv_freq = self._decode_consts()
        return sparse_latent_decode(q, cache.c[L], w_up_t, self.k_norm.weight, inv_freq, idx,
                                    self.scale, self.k_norm.eps, self.cfg.num_kv_heads)

    def _decode_consts(self):
        """W_up^T in the activation dtype and RoPE inverse frequencies, rebuilt if W_up changes."""
        up = self.latent.up
        try:
            version = up._version
        except RuntimeError:  # inference tensors have no version counter (and cannot change)
            version = None
        key = (up.data_ptr(), version, self.k_norm.weight.dtype)
        if getattr(self, "_consts_key", None) != key:
            D = self.cfg.head_dim
            self._w_up_t = up.detach().to(self.k_norm.weight.dtype).T.contiguous()
            self._inv_freq = 1.0 / (self.cfg.rope_theta ** (torch.arange(0, D, 2, device=up.device).float() / D))
            self._consts_key = key
        return self._w_up_t, self._inv_freq

    def _latent_kv(self, x, cos, sin, start_pos, cache):
        """Keys/values reconstructed from the cached latent. k_norm and RoPE are
        applied after expansion, exactly as on the original path."""
        Hkv, D = self.cfg.num_kv_heads, self.cfg.head_dim
        c = self.latent.compress(x)                                          # (B, T, r)
        if cache is not None:
            c = cache.update_latent(self.layer_idx, c, start_pos)            # (B, S, r)
        B, S, _ = c.shape
        k, v = self.latent.expand(c).split(Hkv * D, dim=-1)
        k = self.k_norm(k.view(B, S, Hkv, D)).transpose(1, 2)
        v = v.view(B, S, Hkv, D).transpose(1, 2)
        if S != cos.shape[0]:  # cached decode: keys span positions [0, S)
            cos, sin = rope_cos_sin(torch.arange(S, device=x.device), D, self.cfg.rope_theta, x.dtype)
        return apply_rope(k, cos, sin), v


# --------------------------------------------------------------------------- #
# MLP / block / model
# --------------------------------------------------------------------------- #
class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.attn = Attention(cfg, layer_idx)
        self.mlp_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin, start_pos, cache, impl):
        x = x + self.attn(self.attn_norm(x), cos, sin, start_pos, cache, impl)
        return x + self.mlp(self.mlp_norm(x))


class Qwen3(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(Block(cfg, i) for i in range(cfg.num_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.embed.weight
        self.attn_impl = "eager"
        self.grad_checkpointing = False  # recompute each block in backward (long-sequence training)

    def hidden_states(
        self, input_ids: torch.Tensor, start_pos: int = 0, cache: KVCache | None = None
    ) -> torch.Tensor:
        """Final normalised hidden states (B, T, C), before the LM head."""
        T = input_ids.shape[1]
        x = self.embed(input_ids)
        positions = torch.arange(start_pos, start_pos + T, device=input_ids.device)
        cos, sin = rope_cos_sin(positions, self.cfg.head_dim, self.cfg.rope_theta, x.dtype)
        for layer in self.layers:
            if self.grad_checkpointing and torch.is_grad_enabled():
                x = checkpoint(layer, x, cos, sin, start_pos, cache, self.attn_impl, use_reentrant=False)
            else:
                x = layer(x, cos, sin, start_pos, cache, self.attn_impl)
        if cache is not None:
            cache.length = start_pos + T
        return self.norm(x)

    def forward(self, input_ids, start_pos: int = 0, cache: KVCache | None = None) -> torch.Tensor:
        return self.lm_head(self.hidden_states(input_ids, start_pos, cache))

    # ---- lightning indexer / sparse attention ---------------------------- #
    def add_indexer(self, **kw) -> None:
        from .sparse import Indexer
        dev = self.embed.weight.device
        for layer in self.layers:
            layer.attn.indexer = Indexer(self.cfg.hidden_size, **kw).to(dev)

    def set_sparse(self, topk: int, selector: str = "indexer") -> None:
        """Top-k sparse attention in every layer (0 turns it off)."""
        for layer in self.layers:
            if topk and layer.attn.indexer is None:
                raise RuntimeError("call add_indexer() first")
            layer.attn.sparse_topk = topk
            layer.attn.selector = selector

    @property
    def index_dim(self) -> int | None:
        idx = self.layers[0].attn.indexer
        return idx.head_dim if (idx is not None and self.layers[0].attn.sparse_topk) else None

    def indexer_state_dict(self) -> dict:
        idx = self.layers[0].attn.indexer
        return {"config": dict(n_heads=idx.n_heads, head_dim=idx.head_dim, rope_dim=idx.rope_dim,
                               rope_theta=idx.rope_theta),
                "state": {k: v for k, v in self.state_dict().items() if ".indexer." in k}}

    def load_indexer(self, path: str | Path, topk: int = 0) -> None:
        ckpt = torch.load(path, map_location=self.embed.weight.device)
        self.add_indexer(**ckpt["config"])
        _, unexpected = self.load_state_dict(ckpt["state"], strict=False)
        if unexpected:
            raise RuntimeError(f"bad indexer checkpoint: unexpected={unexpected}")
        self.set_sparse(topk)

    def make_cache(self, batch: int, max_len: int) -> KVCache:
        w = self.embed.weight
        return KVCache(self.cfg, batch, max_len, w.device, w.dtype, self.latent_rank, self.index_dim)

    # ---- latent KV (MLA conversion) -------------------------------------- #
    def add_latent_kv(self, rank: int) -> None:
        """Attach an (uninitialised) LatentKV of the given rank to every layer."""
        kv_dim = 2 * self.cfg.num_kv_heads * self.cfg.head_dim
        dev = self.embed.weight.device
        for layer in self.layers:
            layer.attn.latent = LatentKV(self.cfg.hidden_size, rank, kv_dim).to(dev)

    def set_latent(self, enabled: bool) -> None:
        for layer in self.layers:
            if enabled and layer.attn.latent is None:
                raise RuntimeError("call add_latent_kv() first")
            layer.attn.use_latent = enabled

    @property
    def latent_rank(self) -> int | None:
        lat = self.layers[0].attn.latent
        return lat.rank if (lat is not None and self.layers[0].attn.use_latent) else None

    def latent_state_dict(self) -> dict[str, torch.Tensor]:
        return {k: v for k, v in self.state_dict().items() if ".latent." in k}

    def load_latent(self, path: str | Path, enable: bool = True) -> int:
        """Load trained latent projections saved by latent_state_dict(). Returns the rank."""
        state = torch.load(path, map_location=self.embed.weight.device)
        rank = state["layers.0.attn.latent.down"].shape[0]
        self.add_latent_kv(rank)
        missing, unexpected = self.load_state_dict(state, strict=False)
        if unexpected or any(".latent." in m for m in missing):
            raise RuntimeError(f"bad latent checkpoint: unexpected={unexpected}")
        self.set_latent(enable)
        return rank


# --------------------------------------------------------------------------- #
# Loading pretrained weights
# --------------------------------------------------------------------------- #
def _hf_to_ours(name: str) -> str:
    """Map a Hugging Face Qwen3 parameter name onto our module names."""
    name = name.removeprefix("model.")
    return (
        name.replace("embed_tokens.", "embed.")
        .replace("self_attn.", "attn.")
        .replace("input_layernorm.", "attn_norm.")
        .replace("post_attention_layernorm.", "mlp_norm.")
    )


def load_qwen3(
    repo_or_path: str = "Qwen/Qwen3-1.7B",
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> Qwen3:
    """Build our Qwen3 and fill it with pretrained weights from the Hub or a local dir."""
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    path = Path(repo_or_path)
    if not path.is_dir():
        path = Path(snapshot_download(repo_or_path, allow_patterns=["*.json", "*.safetensors", "*.txt"]))

    cfg = ModelConfig.from_hf_config(path / "config.json")
    with torch.device("meta"):
        model = Qwen3(cfg)

    state = {}
    for f in sorted(path.glob("*.safetensors")):
        for k, t in load_file(f, device=str(device)).items():
            state[_hf_to_ours(k)] = t.to(dtype)
    if cfg.tie_word_embeddings:
        state.pop("lm_head.weight", None)  # tied to embed.weight

    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    missing = [m for m in missing if not (cfg.tie_word_embeddings and m == "lm_head.weight")]
    if missing or unexpected:
        raise RuntimeError(f"weight mismatch: missing={missing} unexpected={unexpected}")
    if cfg.tie_word_embeddings:
        model.lm_head.weight = model.embed.weight  # re-tie after assign replaced the tensor

    model.path = path
    return model.eval()
