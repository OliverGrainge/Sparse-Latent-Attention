"""Profile of the fused sparse-latent decode kernel on its own (one layer, batch 1).

Sweeps the things the kernel's cost could depend on:
  context   cache length S, with k fixed                (should be flat)
  topk      tokens attended k, with S fixed              (should be linear)
  block_n   selected tokens rebuilt per inner iteration
  programs  how many Triton programs the work is split over

    uv run python scripts/bench_kernel.py
"""

import json
import math
from pathlib import Path

import torch
import triton.testing

from sla.kernels import sparse_latent_decode
from sla.model import RMSNorm


def main():
    dev, dt = "cuda", torch.bfloat16
    H, Hkv, D, R, theta = 16, 8, 128, 512, 1e6
    scale = 1 / math.sqrt(D)
    torch.manual_seed(0)
    w_up_t = (torch.randn(2 * Hkv * D, R, device=dev) / math.sqrt(R)).to(dt).T.contiguous()
    knorm = RMSNorm(D, 1e-6).to(dev).to(dt)
    inv_freq = 1.0 / (theta ** (torch.arange(0, D, 2, device=dev).float() / D))
    q = torch.randn(1, H, D, device=dev, dtype=dt)

    def bench(S, k, **kw):
        c = torch.randn(1, S, R, device=dev, dtype=dt)
        idx = torch.randperm(S, device=dev)[:k].int()[None].contiguous()
        fn = lambda: sparse_latent_decode(q, c, w_up_t, knorm.weight, inv_freq, idx, scale, 1e-6, Hkv, **kw)
        return triton.testing.do_bench(fn, warmup=50, rep=200) * 1e3  # microseconds

    out = {"unit": "us"}
    out["context"] = [{"context": S, "us": bench(S, 2048)} for S in [4096, 16384, 65536, 131072]]
    out["topk"] = [{"topk": k, "us": bench(32768, k)} for k in [256, 512, 1024, 2048, 4096, 8192]]
    out["block_n"] = [{"block_n": b, "us": bench(32768, 2048, block_n=b)} for b in [16, 32, 64]]
    out["programs"] = [{"programs": p, "us": bench(32768, 2048, target_programs=p)}
                       for p in [8, 32, 64, 128, 256, 512]]
    for name in ("context", "topk", "block_n", "programs"):
        print(name, "  ".join(f"{list(r.values())[0]}: {r['us']:.1f}" for r in out[name]), flush=True)
    Path("results/bench_kernel.json").write_text(json.dumps(out, indent=1))
    print("saved results/bench_kernel.json")


if __name__ == "__main__":
    main()
