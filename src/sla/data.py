"""Training data: tokenise documents and pack them into fixed-length sequences.

Sequences are stored as a flat uint32 array (num_seqs * seq_len) on disk so
training just memory-maps it. Documents are separated by <|endoftext|>.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import torch
from tqdm import tqdm

SOURCES = {
    # Long books (train split; disjoint from the PG19 test books we evaluate on).
    "pg19": dict(path="emozilla/pg19", data_files="data/train-00000-of-00023-*.parquet", text="text"),
    # Diverse, high-quality web text.
    "fineweb_edu": dict(path="HuggingFaceFW/fineweb-edu", data_files="sample/10BT/000_00000.parquet", text="text"),
}


def iter_texts(source: str) -> Iterator[str]:
    from datasets import load_dataset

    s = SOURCES[source]
    ds = load_dataset(s["path"], data_files=s["data_files"], split="train", streaming=True)
    for row in ds:
        yield row[s["text"]]


def pack_tokens(tokenizer, texts: Iterable[str], seq_len: int, max_tokens: int, eos_id: int,
                batch_docs: int = 64, desc: str = "") -> np.ndarray:
    """Tokenise texts (in batches, multi-threaded) and pack into (N, seq_len) uint32."""
    buf: list[np.ndarray] = []
    total = 0
    pbar = tqdm(total=max_tokens, unit="tok", unit_scale=True, desc=desc)
    batch: list[str] = []

    def flush():
        nonlocal total
        for enc in tokenizer.encode_batch(batch, add_special_tokens=False):
            ids = np.asarray(enc.ids + [eos_id], dtype=np.uint32)
            buf.append(ids)
            total += ids.size
            pbar.update(ids.size)
        batch.clear()

    for text in texts:
        batch.append(text)
        if len(batch) == batch_docs:
            flush()
            if total >= max_tokens:
                break
    if batch and total < max_tokens:
        flush()
    pbar.close()

    flat = np.concatenate(buf)[:max_tokens]
    n = flat.size // seq_len
    return flat[: n * seq_len].reshape(n, seq_len)


def build_mixture(tokenizer, out_dir: str | Path, seq_len: int, tokens_per_source: dict[str, int],
                  val_fraction: float = 0.01, seed: int = 0) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    parts, meta = [], {"seq_len": seq_len, "sources": {}}
    for src, n_tok in tokens_per_source.items():
        seqs = pack_tokens(tokenizer, iter_texts(src), seq_len, n_tok, eos_id, desc=src)
        parts.append(seqs)
        meta["sources"][src] = int(seqs.shape[0])
    allseq = np.concatenate(parts)
    rng = np.random.default_rng(seed)
    rng.shuffle(allseq)  # shuffles rows (whole sequences)
    n_val = max(1, int(len(allseq) * val_fraction))
    allseq[:n_val].tofile(out / "val.bin")
    allseq[n_val:].tofile(out / "train.bin")
    meta.update(train_seqs=int(len(allseq) - n_val), val_seqs=n_val)
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


class PackedSequences(torch.utils.data.Dataset):
    """Memory-mapped (N, seq_len) uint32 token sequences."""

    def __init__(self, data_dir: str | Path, split: str = "train"):
        d = Path(data_dir)
        self.seq_len = json.loads((d / "meta.json").read_text())["seq_len"]
        self.data = np.memmap(d / f"{split}.bin", dtype=np.uint32, mode="r").reshape(-1, self.seq_len)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, i):
        return torch.from_numpy(self.data[i].astype(np.int64))
