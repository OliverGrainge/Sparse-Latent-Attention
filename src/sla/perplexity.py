"""Perplexity evaluation with a sliding window over a long token stream."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .model import Qwen3


def load_wikitext2_tokens(tokenizer, split: str = "test") -> torch.Tensor:
    """WikiText-2 (raw), joined the standard way, as one long 1-D token tensor."""
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(ds["text"])
    return torch.tensor(tokenizer.encode(text, add_special_tokens=False).ids, dtype=torch.long)


def load_pg19_test_books(tokenizer, num_books: int | None = None, min_tokens: int = 0) -> list[torch.Tensor]:
    """PG19 test split (100 full-length books), one token tensor per book."""
    from datasets import load_dataset

    ds = load_dataset("emozilla/pg19-test", split="test")
    books = []
    for text in ds["text"]:
        ids = torch.tensor(tokenizer.encode(text, add_special_tokens=False).ids, dtype=torch.long)
        if ids.numel() >= min_tokens:
            books.append(ids)
        if num_books is not None and len(books) == num_books:
            break
    return books


def per_token_nll(model: Qwen3, input_ids: torch.Tensor, targets: torch.Tensor, vocab_chunk: int = 1024):
    """Next-token NLL at every position, shape (B, T), fp32; ignored targets (-100) give 0.
    The LM head is applied in chunks of positions so the full (T x vocab) fp32
    logit matrix never has to be held in memory."""
    h = model.hidden_states(input_ids)                           # (B, T, C)
    out = torch.zeros(targets.shape, device=h.device, dtype=torch.float32)
    for t0 in range(0, h.shape[1], vocab_chunk):
        t1 = t0 + vocab_chunk
        tgt = targets[:, t0:t1]
        if not (tgt != -100).any():
            continue
        logits = model.lm_head(h[:, t0:t1]).float()
        out[:, t0:t1] = F.cross_entropy(
            logits.transpose(1, 2), tgt, ignore_index=-100, reduction="none"
        )
    return out


def token_nll(model: Qwen3, input_ids: torch.Tensor, targets: torch.Tensor, vocab_chunk: int = 1024):
    """Summed next-token NLL and number of scored tokens."""
    nll = per_token_nll(model, input_ids, targets, vocab_chunk)
    return nll.double().sum(), int((targets != -100).sum())


@torch.inference_mode()
def nll_by_position(model: Qwen3, books: list[torch.Tensor], seq_len: int, progress: bool = True) -> torch.Tensor:
    """Mean NLL at each position 1..seq_len-1 over the first `seq_len` tokens of
    each book (books shorter than seq_len are skipped). Returns (seq_len - 1,).

    Position t is predicted with t tokens of context, so this curve shows how
    much the model gains from distant context — exactly what sparse attention
    has to preserve."""
    device = model.embed.weight.device
    usable = [b for b in books if b.numel() >= seq_len]
    if not usable:
        raise ValueError(f"no book has >= {seq_len} tokens")
    acc = torch.zeros(seq_len - 1, device=device, dtype=torch.float64)
    for book in tqdm(usable, disable=not progress, desc=f"nll@{seq_len}"):
        x = book[:seq_len].to(device)[None]
        acc += per_token_nll(model, x[:, :-1], x[:, 1:])[0].double()
    return (acc / len(usable)).float().cpu()


@torch.inference_mode()
def ppl_vs_context(
    model: Qwen3,
    books: list[torch.Tensor],
    context_lens: list[int],
    end: int = 32768,
    score_len: int = 1024,
    progress: bool = True,
) -> dict[int, float]:
    """Perplexity on a FIXED set of target tokens, varying only the context.

    For every book, the targets are tokens [end - score_len, end). For each
    context length c the model sees tokens [end - score_len - c, end), so the
    first target has c tokens of history and the last has c + score_len - 1.
    Because the targets never change, differences between context lengths are
    caused by context alone (unlike position-wise curves, where position is
    confounded with content such as front matter)."""
    device = model.embed.weight.device
    assert max(context_lens) + score_len <= end
    usable = [b for b in books if b.numel() >= end]
    sums = {c: 0.0 for c in context_lens}
    for book in tqdm(usable, disable=not progress, desc="ppl-vs-context"):
        for c in context_lens:
            x = book[end - score_len - c : end].to(device)[None]
            targets = x.clone()
            targets[:, :c] = -100  # only the fixed span is scored
            nll, _ = token_nll(model, x[:, :-1], targets[:, 1:])
            sums[c] += float(nll)
    n = len(usable) * score_len
    return {c: math.exp(s / n) for c, s in sums.items()}


def bucket_ppl(nll_curve: torch.Tensor, edges: list[int]) -> dict[str, float]:
    """Perplexity within position buckets, e.g. edges=[0, 1024, 4096, 32768].
    nll_curve[i] is the loss for predicting token i+1 (context length i+1)."""
    out = {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        seg = nll_curve[lo:hi]
        if seg.numel():
            out[f"{lo}-{hi}"] = math.exp(seg.mean().item())
    return out


@dataclass
class PPLResult:
    context_len: int
    stride: int
    tokens_scored: int
    nll: float

    @property
    def ppl(self) -> float:
        return math.exp(self.nll)


@torch.inference_mode()
def evaluate_perplexity(
    model: Qwen3,
    tokens: torch.Tensor,
    context_len: int = 4096,
    stride: int | None = None,
    max_tokens: int | None = None,
    progress: bool = True,
) -> PPLResult:
    """Sliding-window perplexity.

    Each window is `context_len` tokens; only the last `stride` targets of each
    window are scored (all of them in the first window), so every token is
    scored exactly once with at least `context_len - stride` tokens of history.
    stride == context_len gives plain non-overlapping chunks.
    """
    stride = stride or context_len
    assert 0 < stride <= context_len
    device = model.embed.weight.device
    if max_tokens is not None:
        tokens = tokens[:max_tokens]
    n = tokens.numel()

    total, count = 0.0, 0
    prev_end = 0
    starts = range(0, n - 1, stride)
    for begin in tqdm(starts, disable=not progress, desc=f"ppl@{context_len}"):
        end = min(begin + context_len, n)
        window = tokens[begin:end].to(device)[None]
        targets = window.clone()
        # Score only tokens not already scored by a previous window.
        n_new = end - prev_end
        targets[:, : window.shape[1] - n_new] = -100
        # Next-token prediction: inputs[t] predicts targets[t+1].
        nll, c = token_nll(model, window[:, :-1], targets[:, 1:])
        total += float(nll)
        count += c
        prev_end = end
        if end == n:
            break
    return PPLResult(context_len, stride, count, total / count)
