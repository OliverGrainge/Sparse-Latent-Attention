"""Triton kernels for sparse latent attention at decode time (one new query per sequence).

Why a custom kernel: our latent cache cannot use MLA's weight-absorption trick,
because Qwen3 applies RMSNorm (k_norm) and RoPE to keys *after* they are
expanded from the latent. So keys/values must be rebuilt from the cached
latents before attention. Done naively that means expanding the whole cache
every step: S x r x (2 Hkv D) multiply-adds per layer, and writing the full K/V
back to memory, which throws away the point of caching a latent.

With top-k sparse attention only the k selected latents are needed, and a
fused kernel can do the whole thing on-chip, block by block:

    gather c[idx]  ->  [k; v] = c W_up  ->  RMSNorm(k), RoPE(k, idx)  ->  online-softmax attention

so the expanded keys/values never touch global memory. Two kernels:

  index_scores_decode   I(s) = sum_j w_j relu(q_j . k_s) for every cached token
  sparse_latent_decode  the fused gather/expand/norm/rope/attention above,
                        split over blocks of selected tokens (flash-decoding),
                        followed by a small cross-split reduction.
"""

from __future__ import annotations

import os

os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")  # silence Triton 3.9's default-change warning

import torch
import triton
import triton.language as tl


# --------------------------------------------------------------------------- #
# Indexer scores
# --------------------------------------------------------------------------- #
@triton.jit
def _index_scores_kernel(
    q_ptr, w_ptr, k_ptr, out_ptr,
    S,
    stride_qb, stride_qh,          # q: (B, Hi, d)
    stride_wb,                     # w: (B, Hi)
    stride_kb, stride_ks,          # k: (B, S_max, d)
    stride_ob,                     # out: (B, S_pad)
    HI: tl.constexpr, HI_PAD: tl.constexpr, DI: tl.constexpr, BLOCK_S: tl.constexpr,
):
    b = tl.program_id(0)
    s0 = tl.program_id(1) * BLOCK_S
    offs_s = s0 + tl.arange(0, BLOCK_S)
    offs_h = tl.arange(0, HI_PAD)
    offs_d = tl.arange(0, DI)
    q = tl.load(q_ptr + b * stride_qb + offs_h[:, None] * stride_qh + offs_d[None, :],
                mask=offs_h[:, None] < HI, other=0.0)                               # (HI_PAD, d)
    w = tl.load(w_ptr + b * stride_wb + offs_h, mask=offs_h < HI, other=0.0)     # (HI_PAD,)
    k = tl.load(k_ptr + b * stride_kb + offs_s[:, None] * stride_ks + offs_d[None, :],
                mask=offs_s[:, None] < S, other=0.0)                                # (BLOCK_S, d)
    dots = tl.dot(q, tl.trans(k))                                                  # (HI_PAD, BLOCK_S) fp32
    s = tl.sum(tl.maximum(dots, 0.0) * w[:, None], axis=0)
    s = tl.where(offs_s < S, s, float("-inf"))
    tl.store(out_ptr + b * stride_ob + offs_s, s)


def index_scores_decode(q: torch.Tensor, w: torch.Tensor, k: torch.Tensor, S: int) -> torch.Tensor:
    """q (B, Hi, d), w (B, Hi) fp32, k (B, >=S, d) -> scores (B, S) fp32."""
    B, HI, DI = q.shape
    BLOCK_S = 256
    n_blk = triton.cdiv(S, BLOCK_S)
    out = torch.empty((B, n_blk * BLOCK_S), device=q.device, dtype=torch.float32)
    w = w.to(q.dtype)
    _index_scores_kernel[(B, n_blk)](
        q, w, k, out, S,
        q.stride(0), q.stride(1), w.stride(0), k.stride(0), k.stride(1), out.stride(0),
        HI=HI, HI_PAD=max(16, triton.next_power_of_2(HI)), DI=DI, BLOCK_S=BLOCK_S,
    )
    return out[:, :S]


# --------------------------------------------------------------------------- #
# Fused sparse latent attention
# --------------------------------------------------------------------------- #
@triton.jit
def _sparse_latent_kernel(
    q_ptr,            # (B, H, D)  post-norm, post-RoPE queries
    c_ptr,            # (B, S_max, R) cached latents
    wt_ptr,           # (R, 2*HKV*D) W_up transposed: columns [k heads | v heads]
    knw_ptr,          # (D,) k_norm weight
    invf_ptr,         # (D/2,) RoPE inverse frequencies (fp32)
    idx_ptr,          # (B, K) selected positions, -1 = unused
    m_ptr, l_ptr, acc_ptr,   # per-split partials: (B, HKV, NS, G_PAD), same, (B, HKV, NS, G_PAD, D)
    K, scale, eps,
    stride_qb, stride_qh,
    stride_cb, stride_cs,
    stride_ib,
    G: tl.constexpr, G_PAD: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr, R: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_R: tl.constexpr, BLOCKS_PER_SPLIT: tl.constexpr, NS: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    split = tl.program_id(1)
    b = pid_bh // HKV
    j = pid_bh % HKV
    HALF: tl.constexpr = D // 2
    W_COLS: tl.constexpr = 2 * HKV * D

    offs_g = tl.arange(0, G_PAD)
    offs_half = tl.arange(0, HALF)
    offs_d = tl.arange(0, D)
    offs_r = tl.arange(0, BLOCK_R)

    # The G query heads that share KV head j, split into RoPE halves.
    q_rows = q_ptr + b * stride_qb + (j * G + offs_g)[:, None] * stride_qh
    gmask = offs_g[:, None] < G
    q_lo = tl.load(q_rows + offs_half[None, :], mask=gmask, other=0.0)            # (G_PAD, D/2)
    q_hi = tl.load(q_rows + HALF + offs_half[None, :], mask=gmask, other=0.0)

    knw_lo = tl.load(knw_ptr + offs_half).to(tl.float32)
    knw_hi = tl.load(knw_ptr + HALF + offs_half).to(tl.float32)
    inv_freq = tl.load(invf_ptr + offs_half)

    col_k = j * D                    # this head's key columns in W_up^T
    col_v = HKV * D + j * D          # and value columns

    m_i = tl.full((G_PAD,), float("-inf"), tl.float32)
    l_i = tl.zeros((G_PAD,), tl.float32)
    acc = tl.zeros((G_PAD, D), tl.float32)

    for blk in range(BLOCKS_PER_SPLIT):
        n0 = (split * BLOCKS_PER_SPLIT + blk) * BLOCK_N
        offs_n = n0 + tl.arange(0, BLOCK_N)
        pos = tl.load(idx_ptr + b * stride_ib + offs_n, mask=offs_n < K, other=-1)
        valid = pos >= 0
        safe = tl.where(valid, pos, 0)

        # [k_lo | k_hi | v] = c[pos] @ W_up^T[:, head j], accumulated over the latent dim.
        k_lo = tl.zeros((BLOCK_N, HALF), tl.float32)
        k_hi = tl.zeros((BLOCK_N, HALF), tl.float32)
        v = tl.zeros((BLOCK_N, D), tl.float32)
        for r0 in range(0, R, BLOCK_R):
            c = tl.load(c_ptr + b * stride_cb + safe[:, None] * stride_cs + (r0 + offs_r)[None, :],
                        mask=valid[:, None], other=0.0)                                # (BLOCK_N, BLOCK_R)
            w_rows = wt_ptr + (r0 + offs_r)[:, None] * W_COLS
            w_klo = tl.load(w_rows + col_k + offs_half[None, :])                     # (BLOCK_R, D/2)
            w_khi = tl.load(w_rows + col_k + HALF + offs_half[None, :])
            w_v = tl.load(w_rows + col_v + offs_d[None, :])                          # (BLOCK_R, D)
            k_lo = tl.dot(c, w_klo, k_lo)
            k_hi = tl.dot(c, w_khi, k_hi)
            v = tl.dot(c, w_v, v)

        # k_norm (RMSNorm over the full head dim), then RoPE at the selected positions.
        ms = (tl.sum(k_lo * k_lo, axis=1) + tl.sum(k_hi * k_hi, axis=1)) / D
        rs = tl.rsqrt(ms + eps)[:, None]
        k_lo = k_lo * rs * knw_lo[None, :]
        k_hi = k_hi * rs * knw_hi[None, :]
        ang = safe.to(tl.float32)[:, None] * inv_freq[None, :]                        # (BLOCK_N, D/2)
        cos = tl.cos(ang)
        sin = tl.sin(ang)
        kr_lo = (k_lo * cos - k_hi * sin).to(q_lo.dtype)
        kr_hi = (k_hi * cos + k_lo * sin).to(q_lo.dtype)

        s = tl.dot(q_lo, tl.trans(kr_lo)) + tl.dot(q_hi, tl.trans(kr_hi))             # (G_PAD, BLOCK_N)
        s = tl.where(valid[None, :], s * scale, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp(m_i - m_safe)
        p = tl.exp(s - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(q_lo.dtype), v.to(q_lo.dtype))
        m_i = m_new

    base = (pid_bh * NS + split) * G_PAD
    tl.store(m_ptr + base + offs_g, m_i)
    tl.store(l_ptr + base + offs_g, l_i)
    tl.store(acc_ptr + (base + offs_g)[:, None] * D + offs_d[None, :], acc)


@triton.jit
def _combine_kernel(m_ptr, l_ptr, acc_ptr, out_ptr, stride_ob, stride_oh,
                    G: tl.constexpr, G_PAD: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr,
                    NS: tl.constexpr, NS_PAD: tl.constexpr):
    pid_bh = tl.program_id(0)
    g = tl.program_id(1)
    b = pid_bh // HKV
    j = pid_bh % HKV
    offs_s = tl.arange(0, NS_PAD)
    offs_d = tl.arange(0, D)
    smask = offs_s < NS
    rows = (pid_bh * NS + offs_s) * G_PAD + g
    m = tl.load(m_ptr + rows, mask=smask, other=float("-inf"))
    l = tl.load(l_ptr + rows, mask=smask, other=0.0)
    acc = tl.load(acc_ptr + rows[:, None] * D + offs_d[None, :], mask=smask[:, None], other=0.0)
    m_max = tl.max(m, axis=0)
    wgt = tl.where(m == float("-inf"), 0.0, tl.exp(m - m_max))
    o = tl.sum(acc * wgt[:, None], axis=0) / tl.sum(l * wgt, axis=0)
    tl.store(out_ptr + b * stride_ob + (j * G + g) * stride_oh + offs_d, o.to(out_ptr.dtype.element_ty))


def sparse_latent_decode(q: torch.Tensor, c: torch.Tensor, w_up_t: torch.Tensor, k_norm_w: torch.Tensor,
                         inv_freq: torch.Tensor, idx: torch.Tensor, scale: float, eps: float,
                         num_kv_heads: int, block_n: int = 32, target_programs: int = 256) -> torch.Tensor:
    """Fused top-k sparse attention over a latent KV cache, for one query per sequence.

    q (B, H, D) post-norm/RoPE queries; c (B, >=S, R) latent cache; w_up_t (R, 2*Hkv*D);
    k_norm_w (D,); inv_freq (D/2,) fp32; idx (B, K) int32 selected positions (-1 = none).
    Returns (B, H, D).
    """
    B, H, D = q.shape
    R = c.shape[-1]
    K = idx.shape[1]
    G = H // num_kv_heads
    G_PAD = max(16, triton.next_power_of_2(G))
    n_blocks = triton.cdiv(K, block_n)
    # Enough splits to fill the GPU, but at least one block each.
    ns = max(1, min(n_blocks, triton.cdiv(target_programs, B * num_kv_heads)))
    bps = triton.cdiv(n_blocks, ns)
    ns = triton.cdiv(n_blocks, bps)
    m = torch.empty((B * num_kv_heads, ns, G_PAD), device=q.device, dtype=torch.float32)
    l = torch.empty_like(m)
    acc = torch.empty((B * num_kv_heads, ns, G_PAD, D), device=q.device, dtype=torch.float32)
    _sparse_latent_kernel[(B * num_kv_heads, ns)](
        q, c, w_up_t, k_norm_w, inv_freq, idx, m, l, acc,
        K, scale, eps,
        q.stride(0), q.stride(1), c.stride(0), c.stride(1), idx.stride(0),
        G=G, G_PAD=G_PAD, HKV=num_kv_heads, D=D, R=R,
        BLOCK_N=block_n, BLOCK_R=64, BLOCKS_PER_SPLIT=bps, NS=ns,
        num_warps=4, num_stages=2,
    )
    out = torch.empty((B, H, D), device=q.device, dtype=q.dtype)
    _combine_kernel[(B * num_kv_heads, G)](
        m, l, acc, out, out.stride(0), out.stride(1),
        G=G, G_PAD=G_PAD, HKV=num_kv_heads, D=D, NS=ns, NS_PAD=max(2, triton.next_power_of_2(ns)),
    )
    return out
