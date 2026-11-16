"""Train the lightning indexers (DSA "dense warm-up" stage).

The model (with its latent KV cache) is frozen and runs dense attention. In
every layer a callback takes the attention input x and the real queries/keys,
builds the head-averaged attention distribution for a random subset of
queries, and trains that layer's indexer to match it with
KL(attention || softmax(indexer scores)). Each layer's loss is local, so we
never backpropagate through the model itself.

    uv run python scripts/train_indexer.py --latent checkpoints/latent_r512_16k/latent.pt
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch

from sla.data import PackedSequences
from sla.model import load_qwen3
from sla.sparse import attention_target, indexer_kl, recall_at_k


def lr_at(step, total, peak, warmup):
    if step < warmup:
        return peak * (step + 1) / warmup
    return peak * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup))))


def sample_queries(T: int, n: int, min_pos: int, device) -> torch.Tensor:
    return torch.randperm(T - min_pos, device=device)[:n].sort().values + min_pos


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--latent", default="checkpoints/latent_r512_16k/latent.pt")
    p.add_argument("--data", default="data/mix_16k")
    p.add_argument("--seq-len", type=int, default=16384)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--rope-dim", type=int, default=32)
    p.add_argument("--queries", type=int, default=512, help="sampled query positions per layer per sequence")
    p.add_argument("--min-pos", type=int, default=256)
    p.add_argument("--eval-min-pos", type=int, default=4096, help="recall is measured on queries past this")
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--val-seqs", type=int, default=4)
    p.add_argument("--topks", type=int, nargs="+", default=[256, 512, 1024, 2048])
    p.add_argument("--init", default=None, help="start from this indexer.pt (with --steps 0: eval only)")
    p.add_argument("--recall-out", default="results/indexer_recall.json")
    p.add_argument("--out", default="checkpoints/indexer")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    model = load_qwen3(args.model)
    model.attn_impl = "sdpa"
    model.requires_grad_(False)
    if args.latent:
        print(f"latent KV rank {model.load_latent(args.latent)}")
    if args.init:
        model.load_indexer(args.init)
    else:
        model.add_indexer(n_heads=args.heads, head_dim=args.head_dim, rope_dim=args.rope_dim)
    params = [p_ for n, p_ in model.named_parameters() if ".indexer." in n]
    for p_ in params:
        p_.requires_grad_(True)
    print(f"indexer params: {sum(p_.numel() for p_ in params) / 1e6:.1f}M")
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)

    L = model.cfg.num_layers
    state = {"mode": "train", "kl": [0.0] * L, "recall": None}

    def observer(attn, x, q, k, start_pos):
        T = x.shape[1]
        min_pos = args.min_pos if state["mode"] == "train" else min(args.eval_min_pos, T // 2)
        q_pos = sample_queries(T, args.queries, min_pos, x.device)
        target = attention_target(q[:, :, q_pos], k, q_pos, attn.scale)
        if state["mode"] == "train":
            with torch.enable_grad():
                loss = indexer_kl(attn.indexer, x, target, q_pos)
                (loss / args.grad_accum).backward()
            state["kl"][attn.layer_idx] += loss.item() / args.grad_accum
        else:
            kl = indexer_kl(attn.indexer, x, target, q_pos).item()
            r = recall_at_k(attn.indexer, x, target, q_pos, args.topks)
            acc = state["recall"][attn.layer_idx]
            acc["kl"] += kl
            for name, vals in r.items():
                for kk, v in vals.items():
                    acc.setdefault(name, {}).setdefault(kk, 0.0)
                    acc[name][kk] += v

    for layer in model.layers:
        layer.attn.observer = observer

    train = PackedSequences(args.data, "train")
    val = PackedSequences(args.data, "val")

    @torch.no_grad()
    def run_eval(step):
        state["mode"] = "eval"
        state["recall"] = [{"kl": 0.0} for _ in range(L)]
        rng_state = torch.cuda.get_rng_state()
        torch.cuda.manual_seed(1234)
        for i in range(args.val_seqs):
            model.hidden_states(val[i][None, : args.seq_len].cuda())
        torch.cuda.set_rng_state(rng_state)
        n = args.val_seqs
        per_layer = []
        for acc in state["recall"]:
            per_layer.append({k: (v / n if not isinstance(v, dict) else {kk: vv / n for kk, vv in v.items()})
                              for k, v in acc.items()})
        mean = {"kl": sum(l["kl"] for l in per_layer) / L}
        for name in ("indexer", "indexer_raw", "oracle", "window", "sink_window"):
            mean[name] = {kk: sum(l[name][kk] for l in per_layer) / L for kk in per_layer[0][name]}
        state["mode"] = "train"
        rec = " ".join(f"k={kk}: {mean['indexer'][kk]:.3f}/{mean['oracle'][kk]:.3f}/{mean['window'][kk]:.3f}"
                       for kk in mean["indexer"])
        print(f"[eval {step:>4}] KL {mean['kl']:.4f}  recall indexer/oracle/window  {rec}", flush=True)
        return {"step": step, "mean": mean, "per_layer": per_layer}

    log = {"args": vars(args), "evals": [run_eval(0)], "train": []}
    order = torch.randperm(len(train))
    cursor = 0
    t0 = time.perf_counter()
    for step in range(args.steps):
        for g_ in opt.param_groups:
            g_["lr"] = lr_at(step, args.steps, args.lr, args.warmup)
        state["kl"] = [0.0] * L
        for _ in range(args.grad_accum):
            ids = train[int(order[cursor])][None, : args.seq_len].cuda()
            cursor = (cursor + 1) % len(order)
            with torch.no_grad():
                model.hidden_states(ids)
        gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
        opt.step()
        opt.zero_grad(set_to_none=True)
        kl = sum(state["kl"]) / L
        if step % 10 == 0 or step == args.steps - 1:
            dt = time.perf_counter() - t0
            print(f"step {step:>4}  KL {kl:.4f}  (layer 0 {state['kl'][0]:.3f}, last {state['kl'][-1]:.3f})  "
                  f"gnorm {gnorm:.3f}  lr {opt.param_groups[0]['lr']:.2e}  {dt / (step + 1):.1f}s/step  "
                  f"mem {torch.cuda.max_memory_allocated() / 2**30:.1f}GiB", flush=True)
            log["train"].append({"step": step, "kl": kl, "kl_per_layer": list(state["kl"]), "gnorm": gnorm})
        if (step + 1) % args.eval_every == 0 or step == args.steps - 1:
            log["evals"].append(run_eval(step + 1))
            torch.save(model.indexer_state_dict(), out / "indexer.pt")
            (out / "log.json").write_text(json.dumps(log, indent=1))
    final = log["evals"][-1]
    Path("results").mkdir(exist_ok=True)
    Path(args.recall_out).write_text(json.dumps(
        {"context": args.eval_min_pos, "seq_len": args.seq_len, "mean": final["mean"],
         "per_layer": final["per_layer"]}, indent=1))
    if args.steps:
        print(f"done; saved {out / 'indexer.pt'}")


if __name__ == "__main__":
    main()
