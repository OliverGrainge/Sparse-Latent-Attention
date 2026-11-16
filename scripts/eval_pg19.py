"""Long-context dense baseline on PG19 test books.

Main metric: perplexity on a fixed 1,024-token span ending at token 32K of each
book, scored with increasing amounts of preceding context.

    uv run python scripts/eval_pg19.py --attn sdpa
    uv run python scripts/eval_pg19.py --by-position   # raw loss-vs-position curve
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch

from sla.generate import load_tokenizer
from sla.model import load_qwen3
from sla.perplexity import bucket_ppl, load_pg19_test_books, nll_by_position, ppl_vs_context

DEFAULT_CONTEXTS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 31744]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--end", type=int, default=32768, help="targets end at this token of each book")
    p.add_argument("--score-len", type=int, default=1024)
    p.add_argument("--context-lens", type=int, nargs="+", default=DEFAULT_CONTEXTS)
    p.add_argument("--num-books", type=int, default=None, help="default: every book long enough")
    p.add_argument("--attn", default="eager", choices=["eager", "sdpa"])
    p.add_argument("--by-position", action="store_true", help="also compute the per-position curve")
    p.add_argument("--latent", default=None, help="path to a trained latent.pt (MLA conversion)")
    p.add_argument("--out", default="results/pg19_dense.json")
    args = p.parse_args()

    model = load_qwen3(args.model)
    model.attn_impl = args.attn
    if args.latent:
        print(f"latent KV rank {model.load_latent(args.latent)} from {args.latent}")
    books = load_pg19_test_books(load_tokenizer(model), num_books=args.num_books, min_tokens=args.end)
    lens = torch.tensor([b.numel() for b in books])
    print(f"PG19 test: {len(books)} books with >= {args.end:,} tokens (median {int(lens.median()):,})")

    result = {"model": args.model, "attn": args.attn, "end": args.end, "score_len": args.score_len,
              "num_books": len(books), "tokens_scored": len(books) * args.score_len}

    t = time.perf_counter()
    by_ctx = ppl_vs_context(model, books, args.context_lens, end=args.end, score_len=args.score_len)
    print(f"perplexity of the fixed span vs context ({time.perf_counter() - t:.0f}s, attn={args.attn}):")
    for c, v in by_ctx.items():
        print(f"  context {c:>6,}: ppl {v:7.3f}")
    result["ppl_vs_context"] = {str(c): v for c, v in by_ctx.items()}

    if args.by_position:
        curve = nll_by_position(model, books, args.end)
        edges = [0] + [e for e in (1024, 2048, 4096, 8192, 16384) if e < args.end] + [args.end - 1]
        result["bucket_ppl"] = bucket_ppl(curve, edges)
        result["overall_ppl"] = math.exp(curve.mean().item())
        result["nll_by_position"] = curve.tolist()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
