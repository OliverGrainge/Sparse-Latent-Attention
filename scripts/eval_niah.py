"""Needle in a haystack: can the model retrieve one fact buried in a long book?

A sentence "The secret passcode is NNNNNNN." is inserted at a given depth of a
PG19 test book, and the text ends with a question. Retrieval counts as a
success if every digit of the passcode is the model's top-1 prediction
(teacher-forced, i.e. what greedy decoding would produce).

    uv run python scripts/eval_niah.py --out results/niah_dense.json
    uv run python scripts/eval_niah.py --latent ... --indexer ... --out results/niah_sparse.json
"""

import argparse
import json
import random
from pathlib import Path

import torch

from sla.generate import load_tokenizer
from sla.model import load_qwen3
from sla.perplexity import load_pg19_test_books

QUESTION = "\n\nQuestion: What is the secret passcode mentioned in the text above?\nAnswer: The secret passcode is "


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--latent", default=None)
    p.add_argument("--indexer", default=None)
    p.add_argument("--topk", type=int, default=2048)
    p.add_argument("--selector", default="indexer", choices=["indexer", "sink_window"])
    p.add_argument("--lengths", type=int, nargs="+", default=[4096, 8192, 16384, 32768])
    p.add_argument("--depths", type=int, default=11, help="evenly spaced depths in [0, 1]")
    p.add_argument("--samples", type=int, default=3, help="haystacks (books) per cell")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    model = load_qwen3(args.model)
    model.attn_impl = "sdpa"
    if args.latent:
        model.load_latent(args.latent)
    if args.indexer or args.selector == "sink_window":
        model.load_indexer(args.indexer) if args.indexer else model.add_indexer()
        model.set_sparse(args.topk, args.selector)
    tok = load_tokenizer(model)
    enc = lambda s: tok.encode(s, add_special_tokens=False).ids
    books = load_pg19_test_books(tok, min_tokens=max(args.lengths))
    q_ids = enc(QUESTION)
    rng = random.Random(0)
    depths = [i / (args.depths - 1) for i in range(args.depths)]

    grid = {}
    for L in args.lengths:
        for d in depths:
            hits = []
            for s in range(args.samples):
                code = str(rng.randrange(1_000_000, 10_000_000))
                needle = enc(f" The secret passcode is {code}. ")
                answer = enc(code)
                book = books[(s * 7 + int(d * 10)) % len(books)]
                n_hay = L - len(needle) - len(q_ids) - len(answer)
                hay = book[:n_hay].tolist()
                at = int(d * n_hay)
                ids = hay[:at] + needle + hay[at:] + q_ids + answer
                x = torch.tensor([ids], device="cuda")
                with torch.inference_mode():
                    h = model.hidden_states(x[:, :-1])[:, -len(answer):]
                    pred = model.lm_head(h).argmax(-1)[0].tolist()
                hits.append(pred == answer)
            grid[f"{L}/{d:.1f}"] = sum(hits) / len(hits)
            print(f"len {L:>6}  depth {d:.1f}  acc {grid[f'{L}/{d:.1f}']:.2f}", flush=True)
    per_len = {L: sum(grid[f"{L}/{d:.1f}"] for d in depths) / len(depths) for L in args.lengths}
    print("mean accuracy per length:", {k: round(v, 3) for k, v in per_len.items()})
    Path(args.out).write_text(json.dumps({"args": vars(args), "grid": grid, "per_length": per_len}, indent=1))


if __name__ == "__main__":
    main()
