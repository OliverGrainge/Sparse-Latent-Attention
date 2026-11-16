"""Quick inference check: chat with our native-torch Qwen3.

    uv run python scripts/generate.py "Explain attention in one paragraph."
"""

import argparse
import time

import torch

from sla.generate import chat_prompt, generate, load_tokenizer
from sla.model import load_qwen3


def main():
    p = argparse.ArgumentParser()
    p.add_argument("prompt", nargs="?", default="Explain what the attention mechanism in a transformer does, in three sentences.")
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--raw", action="store_true", help="plain completion, no chat template")
    p.add_argument("--attn", default="eager", choices=["eager", "sdpa"])
    p.add_argument("--latent", default=None, help="path to a trained latent.pt (MLA conversion)")
    args = p.parse_args()

    model = load_qwen3(args.model)
    model.attn_impl = args.attn
    if args.latent:
        print(f"[latent KV rank {model.load_latent(args.latent)}]")
    tok = load_tokenizer(model)
    prompt = args.prompt if args.raw else chat_prompt(args.prompt)

    torch.cuda.synchronize()
    t = time.perf_counter()
    text = generate(model, tok, prompt, args.max_new_tokens, args.temperature, args.top_p, seed=0)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t
    n = len(tok.encode(text, add_special_tokens=False).ids)
    print(text)
    print(f"\n[{n} tokens in {dt:.2f}s, {n / dt:.1f} tok/s, attn={args.attn}]")


if __name__ == "__main__":
    main()
