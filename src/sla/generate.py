"""Tokenizer loading and autoregressive generation with a KV cache."""

from __future__ import annotations

from pathlib import Path

import torch
from tokenizers import Tokenizer

from .model import KVCache, Qwen3


def load_tokenizer(model: Qwen3) -> Tokenizer:
    return Tokenizer.from_file(str(Path(model.path) / "tokenizer.json"))


def chat_prompt(user_message: str, thinking: bool = False) -> str:
    """Minimal Qwen3 chat template for a single user turn."""
    prompt = f"<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"
    if not thinking:
        prompt += "<think>\n\n</think>\n\n"  # Qwen3's switch for non-thinking mode
    return prompt


@torch.inference_mode()
def generate(
    model: Qwen3,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    top_p: float = 1.0,
    stop_tokens: tuple[str, ...] = ("<|im_end|>", "<|endoftext|>"),
    seed: int | None = None,
) -> str:
    device = model.embed.weight.device
    ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    stop_ids = {tokenizer.token_to_id(t) for t in stop_tokens} - {None}
    gen = torch.Generator(device=device).manual_seed(seed) if seed is not None else None

    cache = KVCache(model.cfg, 1, len(ids) + max_new_tokens, device, model.embed.weight.dtype,
                    latent_rank=model.latent_rank)
    x = torch.tensor([ids], device=device)
    out: list[int] = []
    pos = 0
    for _ in range(max_new_tokens):
        logits = model(x, start_pos=pos, cache=cache)[:, -1].float()  # (1, V)
        pos += x.shape[1]
        next_id = sample(logits, temperature, top_p, gen)
        if next_id in stop_ids:
            break
        out.append(next_id)
        x = torch.tensor([[next_id]], device=device)
    return tokenizer.decode(out, skip_special_tokens=False)


def sample(logits: torch.Tensor, temperature: float, top_p: float, gen=None) -> int:
    if temperature <= 0:
        return int(logits.argmax(-1))
    probs = (logits / temperature).softmax(-1)
    if top_p < 1.0:
        sorted_p, idx = probs.sort(-1, descending=True)
        keep = sorted_p.cumsum(-1) - sorted_p < top_p  # always keeps the top token
        sorted_p = sorted_p * keep
        choice = torch.multinomial(sorted_p / sorted_p.sum(-1, keepdim=True), 1, generator=gen)
        return int(idx.gather(-1, choice))
    return int(torch.multinomial(probs, 1, generator=gen))
