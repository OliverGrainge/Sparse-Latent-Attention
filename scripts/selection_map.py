"""Which tokens does the sparse model actually read? (data for the write-up's hero figure)

A passcode sentence is hidden in a PG19 book and the text ends with a question
about it (same prompt as eval_niah.py). For two query positions, the last word
of the book text and the last word of the question, this records, in every
layer, which cached tokens the indexer selects and how much of the real
attention mass lands on the needle.

    uv run python scripts/selection_map.py
"""

import argparse
import json
from pathlib import Path

import torch

from sla.generate import load_tokenizer
from sla.model import load_qwen3
from sla.perplexity import load_pg19_test_books
from sla.sparse import attention_target, index_scores

QUESTION = "\n\nQuestion: What is the secret passcode mentioned in the text above?\nAnswer: The secret passcode is "


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--latent", default="checkpoints/latent_r512_16k/latent.pt")
    p.add_argument("--indexer", default="checkpoints/indexer/indexer.pt")
    p.add_argument("--topk", type=int, default=2048)
    p.add_argument("--length", type=int, default=16384)
    p.add_argument("--depth", type=float, default=0.3)
    p.add_argument("--book", type=int, default=3)
    p.add_argument("--code", default="4817263")
    p.add_argument("--out", default="results/selection_map.json")
    args = p.parse_args()

    model = load_qwen3("Qwen/Qwen3-1.7B")
    model.attn_impl = "sdpa"
    model.load_latent(args.latent)
    model.load_indexer(args.indexer, topk=args.topk)
    tok = load_tokenizer(model)
    enc = lambda s: tok.encode(s, add_special_tokens=False).ids
    book = load_pg19_test_books(tok, min_tokens=args.length)[args.book]
    needle, q_ids, answer = enc(f" The secret passcode is {args.code}. "), enc(QUESTION), enc(args.code)
    n_hay = args.length - len(needle) - len(q_ids)
    hay = book[:n_hay].tolist()
    at = int(args.depth * n_hay)
    ids = hay[:at] + needle + hay[at:] + q_ids
    S = len(ids)
    needle_span = (at, at + len(needle))
    queries = {"book": S - len(q_ids) - 1, "question": S - 1}
    layers = []

    def observer(attn, x, q, k, start_pos):
        rec = {}
        for name, t in queries.items():
            pos = torch.tensor([t], device=x.device)
            ik = attn.indexer.keys(x[:, : t + 1], torch.arange(t + 1, device=x.device))
            iq, iw = attn.indexer.queries(x[:, t : t + 1], pos)
            sel = index_scores(iq, iw, ik, t).topk(min(args.topk, t + 1), dim=-1).indices[0, 0]
            target = attention_target(q[:, :, t : t + 1], k[:, :, : t + 1], pos, attn.scale)[0, 0]
            in_needle = (sel >= needle_span[0]) & (sel < needle_span[1])
            rec[name] = {
                "selected": sorted(sel.tolist()),
                "needle_tokens_selected": int(in_needle.sum()),
                "needle_attention": target[needle_span[0] : needle_span[1]].sum().item(),
                "selected_attention": target[sel].sum().item(),
            }
        layers.append(rec)

    for layer in model.layers:
        layer.attn.observer = observer
    with torch.inference_mode():
        h = model.hidden_states(torch.tensor([ids], device="cuda"))
        first = model.lm_head(h[:, -1]).argmax(-1).item()
    print(f"S={S} needle at {needle_span}  first answer token correct: {first == answer[0]}")
    for i, rec in enumerate(layers):
        b, qn = rec["book"], rec["question"]
        print(f"layer {i:>2}  needle tokens selected  book {b['needle_tokens_selected']:>2}  "
              f"question {qn['needle_tokens_selected']:>2} / {len(needle)}   "
              f"attention on needle  book {b['needle_attention']:.4f}  question {qn['needle_attention']:.4f}")
    Path(args.out).write_text(json.dumps({
        "length": S, "topk": args.topk, "needle_span": needle_span, "needle_len": len(needle),
        "queries": queries, "first_token_correct": first == answer[0], "layers": layers}))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
