"""End-to-end decode speed and KV-cache memory of the whole model (batch 1).

The cache is filled with random values up to the target context (timing does
not depend on the contents), then one-token decode steps are timed with the
new token landing at position S - 1, so attention covers exactly S tokens.

S is a power of two on purpose: torch's matmul picks a ~4x slower kernel when
the number of cached tokens is not a multiple of 8, which would penalise the
torch baselines on 7 of every 8 real decode steps. Aligned lengths are their
best case, so the comparison is conservative.

  dense          original Qwen3-1.7B, full K/V cache, plain matmul attention
  latent         latent cache, whole cache re-expanded every step (torch)
  sparse_latent  latent cache + indexer, top-2048, fused Triton kernels

    uv run python scripts/bench_e2e.py
"""

import argparse
import json
import time
from pathlib import Path

import torch

from sla.model import load_qwen3


def fill_(cache):
    for bufs in (getattr(cache, "k", None), getattr(cache, "v", None), getattr(cache, "c", None), cache.ik):
        for b in bufs or []:
            b.normal_(0, 0.5)


@torch.inference_mode()
def time_decode(model, S, steps):
    cache = model.make_cache(1, S)
    fill_(cache)
    tok = torch.tensor([[1000]], device="cuda")
    for _ in range(3):
        model(tok, S - 1, cache)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        model(tok, S - 1, cache)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / steps * 1e3
    # Same step captured in a CUDA graph: no Python / launch overhead, i.e. the
    # GPU time per token that a serving stack (which uses graphs) would see.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        model(tok, S - 1, cache)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        model(tok, S - 1, cache)
    graph.replay()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        graph.replay()
    torch.cuda.synchronize()
    ms_graph = (time.perf_counter() - t0) / steps * 1e3
    del graph
    nbytes = cache.nbytes()
    del cache
    torch.cuda.empty_cache()
    return ms, ms_graph, nbytes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--contexts", type=int, nargs="+", default=[4096, 16384, 32768, 65536, 131072])
    p.add_argument("--latent", default="checkpoints/latent_r512_16k/latent.pt")
    p.add_argument("--indexer", default="checkpoints/indexer/indexer.pt")
    p.add_argument("--topk", type=int, default=2048)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--out", default="results/bench_e2e.json")
    args = p.parse_args()

    model = load_qwen3("Qwen/Qwen3-1.7B")
    configs = {}

    def run(name):
        rows = []
        for S in args.contexts:
            ms, ms_g, nb = time_decode(model, S, args.steps)
            rows.append({"context": S, "ms_per_token": ms, "ms_per_token_graph": ms_g, "cache_GiB": nb / 2**30})
            print(f"{name:>14}  ctx {S:>7}  eager {ms:7.2f} ms/token  cuda-graph {ms_g:7.2f} ms/token  "
                  f"cache {nb / 2**30:6.2f} GiB", flush=True)
        configs[name] = rows

    model.attn_impl = "eager"
    run("dense")
    model.load_latent(args.latent)
    run("latent")
    model.load_indexer(args.indexer, topk=args.topk)
    model.attn_impl = "triton"
    run("sparse_latent")
    Path(args.out).write_text(json.dumps({"topk": args.topk, "batch": 1, "configs": configs}, indent=1))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
