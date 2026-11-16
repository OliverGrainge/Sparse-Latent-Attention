"""Dense-baseline perplexity on WikiText-2 at several context lengths.

    uv run python scripts/eval_ppl.py --context-lens 512 2048 4096 8192 16384 32768
"""

import argparse
import json
import time
from pathlib import Path

import torch

from sla.generate import load_tokenizer
from sla.model import load_qwen3
from sla.perplexity import evaluate_perplexity, load_wikitext2_tokens


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--context-lens", type=int, nargs="+", default=[512, 2048, 4096, 8192])
    p.add_argument("--stride", type=int, default=None, help="default: non-overlapping windows")
    p.add_argument("--attn", default="eager", choices=["eager", "sdpa"])
    p.add_argument("--latent", default=None, help="path to a trained latent.pt (MLA conversion)")
    p.add_argument("--out", default="results/ppl_dense_wikitext2.json")
    args = p.parse_args()

    model = load_qwen3(args.model)
    model.attn_impl = args.attn
    if args.latent:
        print(f"latent KV rank {model.load_latent(args.latent)} from {args.latent}")
    tokens = load_wikitext2_tokens(load_tokenizer(model))
    print(f"WikiText-2 test: {tokens.numel():,} tokens")

    results = []
    for ctx in args.context_lens:
        torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        r = evaluate_perplexity(model, tokens, context_len=ctx, stride=args.stride)
        dt = time.perf_counter() - t
        mem = torch.cuda.max_memory_allocated() / 2**30
        print(f"ctx={ctx:>6}  ppl={r.ppl:7.3f}  tokens={r.tokens_scored:,}  {dt:6.1f}s  peak={mem:.1f} GiB")
        results.append({**r.__dict__, "ppl": r.ppl, "seconds": dt, "peak_gib": mem})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": args.model, "attn": args.attn, "results": results}, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
