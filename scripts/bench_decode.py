"""Decode-step attention latency for one layer of Qwen3-1.7B shapes, vs. context length.

Compares, for a single new query (batch 1):
  dense          attention over a full bf16 K/V cache (torch; see note below)
  dense_roofline the time to just read that K/V cache once at measured DRAM bandwidth,
                 i.e. a lower bound for *any* dense decode kernel
  latent         expand the whole latent cache -> k_norm -> RoPE -> SDPA  (current torch path)
  sparse_torch   indexer (torch) -> top-k -> gather + expand only those -> SDPA
  sparse_triton  indexer kernel -> top-k -> fused gather/expand/norm/RoPE/attention kernel

Note: torch on Windows ships no flash-attention kernel, and SDPA's
"efficient" backend has no split-KV path for a single query, so it is ~7x
slower than a plain matmul-softmax-matmul at long context. All torch baselines
therefore use that plain path (`attend` below), and the roofline shows what an
ideal dense kernel could reach.

    uv run python scripts/bench_decode.py
"""

import argparse
import json
import math
from pathlib import Path

import torch
import triton.testing

from sla.kernels import index_scores_decode, sparse_latent_decode
from sla.model import RMSNorm, apply_rope, rope_cos_sin


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--contexts", type=int, nargs="+",
                   default=[1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072])
    p.add_argument("--topk", type=int, default=2048)
    p.add_argument("--out", default="results/bench_decode.json")
    args = p.parse_args()

    dev, dt = "cuda", torch.bfloat16
    H, Hkv, D, R, C, Hi, di, theta = 16, 8, 128, 512, 2048, 8, 64, 1e6
    scale = 1 / math.sqrt(D)
    torch.manual_seed(0)
    w_up = (torch.randn(2 * Hkv * D, R, device=dev) / math.sqrt(R)).to(dt)
    w_up_t = w_up.T.contiguous()
    knorm = RMSNorm(D, 1e-6).to(dev).to(dt)
    inv_freq = 1.0 / (theta ** (torch.arange(0, D, 2, device=dev).float() / D))
    q = torch.randn(1, H, D, device=dev, dtype=dt)
    iq = torch.randn(1, Hi, di, device=dev, dtype=dt)
    iw = torch.rand(1, Hi, device=dev)

    # Achievable DRAM bandwidth (device-to-device copy of 1 GiB: read + write).
    buf = torch.empty(2**29, device=dev, dtype=dt)
    out_buf = torch.empty_like(buf)
    bw = 2 * buf.numel() * 2 / (triton.testing.do_bench(lambda: out_buf.copy_(buf)) * 1e-3)  # bytes/s
    del buf, out_buf
    print(f"measured copy bandwidth: {bw / 1e12:.2f} TB/s")

    def attend(q_, k, v):
        """Plain decode attention. q_: (1, H, D), k/v: (1, Hkv, S, D) -> (1, H, D)."""
        qg = q_.view(1, Hkv, H // Hkv, D)
        p_ = (qg @ k.transpose(-1, -2)).float().mul_(scale).softmax(-1).to(dt)
        return (p_ @ v).view(1, H, D)

    results = []
    for S in args.contexts:
        k_cache = torch.randn(1, Hkv, S, D, device=dev, dtype=dt)
        v_cache = torch.randn(1, Hkv, S, D, device=dev, dtype=dt)
        c = torch.randn(1, S, R, device=dev, dtype=dt)
        ik = torch.randn(1, S, di, device=dev, dtype=dt)
        kk = min(args.topk, S)
        cos_all, sin_all = rope_cos_sin(torch.arange(S, device=dev), D, theta, dt)

        def dense():
            return attend(q, k_cache, v_cache)

        def expand_attend(cs, cos, sin):
            k, v = (cs @ w_up_t).split(Hkv * D, dim=-1)
            k = apply_rope(knorm(k.view(1, -1, Hkv, D)).transpose(1, 2), cos, sin)
            v = v.view(1, -1, Hkv, D).transpose(1, 2)
            return attend(q, k, v)

        def latent():
            return expand_attend(c, cos_all, sin_all)

        def select_torch():
            s = torch.einsum("bh,bhs->bs", iw, torch.einsum("bhd,bsd->bhs", iq, ik).float().relu())
            return s.topk(kk, dim=-1, sorted=False).indices

        def sparse_torch():
            idx = select_torch()[0]
            return expand_attend(c[:, idx], cos_all[idx], sin_all[idx])

        def select_triton():
            return index_scores_decode(iq, iw, ik, S).topk(kk, dim=-1, sorted=False).indices.int()

        idx_fixed = select_triton()

        def sparse_triton():
            return sparse_latent_decode(q, c, w_up_t, knorm.weight, inv_freq, select_triton(), scale, 1e-6, Hkv)

        def fused_only():
            return sparse_latent_decode(q, c, w_up_t, knorm.weight, inv_freq, idx_fixed, scale, 1e-6, Hkv)

        def scores_only():
            return index_scores_decode(iq, iw, ik, S)

        row = {"context": S, "dense_roofline": 2 * Hkv * S * D * 2 / bw * 1e6}
        for name, fn in [("dense", dense), ("latent", latent), ("sparse_torch", sparse_torch),
                         ("sparse_triton", sparse_triton), ("triton_fused_attn", fused_only),
                         ("triton_index_scores", scores_only)]:
            try:
                row[name] = triton.testing.do_bench(fn, warmup=50, rep=200) * 1e3  # microseconds
            except torch.OutOfMemoryError:
                row[name] = None
                torch.cuda.empty_cache()
        row["topk_only"] = row["sparse_triton"] - row["triton_fused_attn"] - row["triton_index_scores"]
        # Cache bytes per token per layer.
        row["bytes_per_token"] = {"dense": 2 * Hkv * D * 2, "latent": R * 2, "sparse_latent": (R + di) * 2}
        results.append(row)
        print("  ".join(f"{k}={v:.1f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()
                        if k != "bytes_per_token"), flush=True)
        del k_cache, v_cache, c, ik
        torch.cuda.empty_cache()

    Path(args.out).write_text(json.dumps({"topk": args.topk, "unit": "us", "bandwidth_TBps": bw / 1e12, "results": results}, indent=1))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
