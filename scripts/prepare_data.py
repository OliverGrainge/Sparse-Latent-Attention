"""Build the packed training mixture (PG19 train books + FineWeb-Edu).

    uv run python scripts/prepare_data.py --tokens-per-source 50e6 --seq-len 4096
"""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download
from tokenizers import Tokenizer

from sla.data import build_mixture


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--out", default="data/mix_4k")
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--tokens-per-source", type=float, default=50e6)
    p.add_argument("--sources", nargs="+", default=["pg19", "fineweb_edu"])
    args = p.parse_args()

    path = Path(snapshot_download(args.model, allow_patterns=["tokenizer.json"]))
    tok = Tokenizer.from_file(str(path / "tokenizer.json"))
    build_mixture(tok, args.out, args.seq_len, {s: int(args.tokens_per_source) for s in args.sources})


if __name__ == "__main__":
    main()
