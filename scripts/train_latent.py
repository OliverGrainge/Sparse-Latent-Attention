"""Distil Qwen3's attention into a low-rank latent KV cache (MLA-style).

Only the per-layer latent projections (W_down, W_up) are trained; everything
else is frozen. Initialised from importance-weighted PCA.

    uv run python scripts/train_latent.py --rank 512 --steps 1000
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch

from sla.data import PackedSequences
from sla.distill import distill_loss, evaluate
from sla.generate import load_tokenizer
from sla.latent import collect_kv_second_moments, init_latent_from_pca
from sla.model import load_qwen3
from sla.perplexity import evaluate_perplexity, load_wikitext2_tokens


def lr_at(step: int, total: int, peak: float, warmup: int) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))  # cosine to 10%


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--data", default="data/mix_4k")
    p.add_argument("--rank", type=int, default=512)
    p.add_argument("--pca-mode", default="importance")
    p.add_argument("--calib-path", default="checkpoints/kv_calib.pt")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=1, help="sequences per micro-batch")
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--val-seqs", type=int, default=16)
    p.add_argument("--out", default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    out = Path(args.out or f"checkpoints/latent_r{args.rank}")
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    model = load_qwen3(args.model)
    model.attn_impl = "sdpa"
    model.requires_grad_(False)

    calib = torch.load(args.calib_path, map_location="cuda")
    init_latent_from_pca(model, calib, args.rank, mode=args.pca_mode)
    params = [p_ for n, p_ in model.named_parameters() if ".latent." in n]
    for p_ in params:
        p_.requires_grad_(True)
    n_params = sum(p_.numel() for p_ in params)
    print(f"trainable latent params: {n_params / 1e6:.1f}M  (rank {args.rank}, "
          f"cache {2 * model.cfg.num_kv_heads * model.cfg.head_dim / args.rank:.1f}x smaller)")

    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    train = PackedSequences(args.data, "train")
    val = PackedSequences(args.data, "val")
    val_batches = [val[i][None, : args.seq_len] for i in range(args.val_seqs)]
    wikitext = load_wikitext2_tokens(load_tokenizer(model))

    def run_eval(step):
        model.eval()
        m = evaluate(model, val_batches)
        model.set_latent(True)
        m["wikitext_ppl_4k"] = evaluate_perplexity(model, wikitext, 4096, progress=False).ppl
        m["step"] = step
        print(f"[eval {step:>5}] val KL {m['kl']:.4f}  val ppl student {math.exp(m['nll_student']):.3f} "
              f"/ teacher {math.exp(m['nll_teacher']):.3f}  wikitext ppl@4k {m['wikitext_ppl_4k']:.3f}", flush=True)
        return m

    log = {"args": vars(args), "evals": [run_eval(0)], "train": []}
    order = torch.randperm(len(train))
    cursor = 0
    tokens_per_step = args.batch_size * args.grad_accum * args.seq_len
    t0 = time.perf_counter()
    for step in range(args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args.steps, args.lr, args.warmup)
        kl_acc = 0.0
        for _ in range(args.grad_accum):
            idx = order[cursor : cursor + args.batch_size]
            cursor += args.batch_size
            if cursor >= len(order):
                order, cursor = torch.randperm(len(train)), 0
            ids = torch.stack([train[int(i)][: args.seq_len] for i in idx]).cuda()
            loss, m = distill_loss(model, ids)
            (loss / args.grad_accum).backward()
            kl_acc += m["kl"] / args.grad_accum
        gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
        opt.step()
        opt.zero_grad(set_to_none=True)

        if step % 10 == 0 or step == args.steps - 1:
            dt = time.perf_counter() - t0
            tps = (step + 1) * tokens_per_step / dt
            print(f"step {step:>5}  KL {kl_acc:.4f}  gnorm {gnorm:.3f}  lr {opt.param_groups[0]['lr']:.2e}  "
                  f"{tps / 1e3:.1f}K tok/s  mem {torch.cuda.max_memory_allocated() / 2**30:.1f}GiB", flush=True)
            log["train"].append({"step": step, "kl": kl_acc, "gnorm": gnorm, "tok_per_s": tps})
        if (step + 1) % args.eval_every == 0 or step == args.steps - 1:
            log["evals"].append(run_eval(step + 1))
            torch.save(model.latent_state_dict(), out / "latent.pt")
            (out / "log.json").write_text(json.dumps(log, indent=2))
    print(f"done; saved {out / 'latent.pt'}")


if __name__ == "__main__":
    main()
