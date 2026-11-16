"""Plot PG19 perplexity vs. context length from one or more eval_pg19.py results.

    uv run python scripts/plot_pg19.py results/pg19_dense.json [more.json ...] --labels dense ...

Each file is one series. Series keep a fixed colour order so 'dense' stays the
same colour in every figure.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def fmt(n: int) -> str:
    return f"{n / 1024:g}K" if n >= 1024 else str(n)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+")
    p.add_argument("--labels", nargs="*")
    p.add_argument("--out", default="results/pg19_ppl_vs_context.png")
    args = p.parse_args()
    labels = args.labels or [Path(f).stem for f in args.files]
    multi = len(args.files) > 1

    fig, ax = plt.subplots(figsize=(7.5, 4.4), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for i, (f, label) in enumerate(zip(args.files, labels)):
        r = json.loads(Path(f).read_text())
        xs = [int(c) for c in r["ppl_vs_context"]]
        ys = list(r["ppl_vs_context"].values())
        ax.plot(xs, ys, color=SERIES[i], lw=2, marker="o", ms=5, label=label,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
        if multi:
            ax.annotate(label, (xs[-1], ys[-1]), xytext=(8, 0), textcoords="offset points",
                        va="center", fontsize=8, color=INK2)
        else:  # label the two ends only
            for x, y in ((xs[0], ys[0]), (xs[-1], ys[-1])):
                ax.annotate(f"{y:.2f}", (x, y), xytext=(0, 8), textcoords="offset points",
                            ha="center", fontsize=8, color=INK2)

    ax.set_xscale("log", base=2)
    ax.set_xticks(xs, [fmt(x) for x in xs])
    ax.set_xlabel("tokens of preceding context", color=INK2, fontsize=9)
    ax.set_ylabel("perplexity on fixed span", color=INK2, fontsize=9)
    title = "PG19: perplexity vs. context length" + ("" if multi else f" — {labels[0]}")
    fig.suptitle(title, x=0.012, y=0.975, ha="left", color=INK, fontsize=11)
    fig.text(0.012, 0.915,
             f"{r['num_books']} books · same {r['score_len']:,} target tokens per book "
             f"(ending at token {r['end']:,}) · only the context changes",
             ha="left", fontsize=8, color=INK2)
    ax.grid(True, color=GRID, lw=0.6)
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    if multi:
        ax.legend(frameon=False, fontsize=8, labelcolor=INK2)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out, facecolor=SURFACE)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
