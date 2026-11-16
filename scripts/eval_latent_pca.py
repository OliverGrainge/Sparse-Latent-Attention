"""PCA-only latent KV (no training): perplexity vs. latent rank and PCA weighting.

    uv run python scripts/eval_latent_pca.py --ranks 256 512 768 1024 --modes plain importance
"""

import argparse
import json
from pathlib import Path

import torch

from sla.data import PackedSequences
from sla.generate import load_tokenizer
from sla.latent import collect_kv_second_moments, init_latent_from_pca
from sla.model import load_qwen3
from sla.perplexity import evaluate_perplexity, load_wikitext2_tokens


def get_calibration(model, data_dir: str, n_seqs: int, path: str) -> dict:
    p = Path(path)
    if p.exists():
        return torch.load(p, map_location=model.embed.weight.device)
    train = PackedSequences(data_dir, "train")
    calib = collect_kv_second_moments(model, [train[i][None] for i in range(n_seqs)])
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(calib, p)
    return calib


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--data", default="data/mix_4k")
    p.add_argument("--calib-seqs", type=int, default=64)
    p.add_argument("--calib-path", default="checkpoints/kv_calib.pt")
    p.add_argument("--ranks", type=int, nargs="+", default=[128, 256, 512, 768, 1024])
    p.add_argument("--modes", nargs="+", default=["plain", "balance", "importance"])
    p.add_argument("--context-len", type=int, default=4096)
    p.add_argument("--out", default="results/latent_pca_sweep.json")
    args = p.parse_args()

    model = load_qwen3(args.model)
    model.attn_impl = "sdpa"
    tokens = load_wikitext2_tokens(load_tokenizer(model))
    calib = get_calibration(model, args.data, args.calib_seqs, args.calib_path)

    dense = evaluate_perplexity(model, tokens, args.context_len, progress=False).ppl
    full = 2 * model.cfg.num_kv_heads * model.cfg.head_dim
    print(f"dense: ppl {dense:.3f}  (cache {full} values/token/layer)")
    rows = [{"mode": "dense", "rank": full, "ppl": dense, "compression": 1.0}]
    for mode in args.modes:
        for r in args.ranks:
            stats = init_latent_from_pca(model, calib, r, mode=mode)
            model.set_latent(True)
            ppl = evaluate_perplexity(model, tokens, args.context_len, progress=False).ppl
            model.set_latent(False)
            k_err = sum(s["k_rel_err"] for s in stats) / len(stats)
            v_err = sum(s["v_rel_err"] for s in stats) / len(stats)
            print(f"{mode:>10} rank {r:>5}: ppl {ppl:10.3f}  ({full / r:4.1f}x smaller)  "
                  f"rel. err K {k_err:.3f}  V {v_err:.3f}")
            rows.append({"mode": mode, "rank": r, "ppl": ppl, "compression": full / r,
                         "k_rel_err": k_err, "v_rel_err": v_err, "per_layer": stats})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"context_len": args.context_len, "calib_tokens": calib["tokens"],
                               "rows": rows}, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
