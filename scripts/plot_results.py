"""Figures for the write-up, from the JSON files in results/.

    uv run python scripts/plot_results.py

One house style: a finding as the headline, labels on the marks instead of
legends, and a fixed colour per method across every figure.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

R = Path("results")
FIG = R / "figures"

# One colour per method everywhere (validated categorical palette); text stays in ink.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
GRAY = "#8a8984"
INK, INK2, GRID, SURFACE, WASH = "#161614", "#5f5d57", "#dedad0", "#f7f5f0", "#ebe7dd"  # matches the site
SERIF, SANS = "Georgia", "Segoe UI"
SEQ = LinearSegmentedColormap.from_list("seq", [SURFACE, "#9cc3f0", BLUE, "#0d2f5c"])

plt.rcParams.update({
    "font.family": [SANS, "DejaVu Sans"], "font.size": 10.5,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "xtick.major.size": 0, "ytick.major.size": 0,
    "legend.frameon": False, "lines.linewidth": 2.2, "lines.markersize": 5.5,
    "lines.solid_capstyle": "round",
})


def load(name):
    p = R / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def ctx_label(v, _=None):
    return f"{int(v) // 1024}K" if v >= 1024 else str(int(v))


def headline(fig, title, sub, y=1.0):
    """Finding as a serif headline with a muted one-line explanation under it."""
    fig.text(0.012, y, title, fontfamily=SERIF, fontsize=15, fontweight="bold", ha="left", va="bottom")
    fig.text(0.012, y - 0.012, sub, fontsize=10, color=INK2, ha="left", va="top")


def ctx_axis(ax):
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(ctx_label))
    ax.xaxis.set_minor_locator(mticker.NullLocator())


def end_label(ax, x, y, text, dy=0):
    ax.annotate(text, (x, y), xytext=(9, dy), textcoords="offset points", va="center",
                fontsize=9.5, color=INK2, linespacing=1.25)


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight", pad_inches=0.25)
    fig.savefig(FIG / f"{name}.svg", bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("saved", FIG / f"{name}.png")


# --------------------------------------------------------------------------- #
# 1. Hero: which tokens each layer reads, for two different queries
# --------------------------------------------------------------------------- #
def fig_selection():
    d = load("selection_map.json")
    if d is None:
        return
    S, n_layers = d["length"], len(d["layers"])
    n0, n1 = d["needle_span"]
    bins = 256
    w = S / bins
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4),
                             gridspec_kw={"width_ratios": [7, 1], "wspace": 0.04, "hspace": 0.34})
    rows = [("book", "While reading the book", "predicting the next word of the story"),
            ("question", "When asked for the passcode", "predicting the first digit of the answer")]
    for (ax, bx), (key, name, what) in zip(axes, rows):
        dens = np.zeros((n_layers, bins))
        hits = []
        for i, layer in enumerate(d["layers"]):
            sel = np.asarray(layer[key]["selected"])
            dens[i] = np.bincount((sel / w).astype(int), minlength=bins)[:bins] / w
            hits.append(layer[key]["needle_tokens_selected"])
        ax.imshow(dens, cmap=SEQ, norm=PowerNorm(0.6, vmin=0, vmax=1), aspect="auto",
                  extent=(0, S, n_layers - 0.5, -0.5), interpolation="nearest")
        ax.grid(False)
        ax.set_xticks([0, 4096, 8192, 12288, 16384], ["start", "4K", "8K", "12K", "16K"])
        ax.set_yticks([0, 9, 18, 27], ["layer 1", "10", "19", "28"])
        mid = (n0 + n1) / 2
        ax.plot([mid], [-1.25], marker="v", color=ORANGE, markersize=8, clip_on=False)
        ax.plot([mid], [n_layers + 0.25], marker="^", color=ORANGE, markersize=8, clip_on=False)
        ax.text(mid + 220, -1.3, "the passcode sentence (15 tokens)", fontsize=9.5, color=INK2, va="center")
        ax.set_title(name, fontfamily=SERIF, fontsize=12.5, fontweight="bold", loc="left", pad=22)
        ax.text(0.0, 1.025, what, transform=ax.transAxes, fontsize=9.5, color=INK2, va="bottom")
        for s in ax.spines.values():
            s.set_visible(False)
        bx.barh(range(n_layers), hits, color=ORANGE, height=0.72)
        bx.set_ylim(n_layers - 0.5, -0.5)
        bx.set_xlim(0, d["needle_len"])
        bx.set_xticks([d["needle_len"]])
        bx.set_yticks([])
        bx.grid(False)
        bx.axvline(d["needle_len"], color=GRID, lw=1)
        bx.spines["bottom"].set_visible(False)
        bx.set_title("passcode tokens\npicked (of 15)", fontsize=9.5, color=INK2, loc="left", pad=6)
    axes[1][0].set_xlabel("position in the 16K-token context   (each layer may read 2,048 of these tokens; "
                          "darker = more picked there)")
    headline(fig, "The indexer reads what the question needs",
             "Tokens selected by each layer of the sparse model for one query. "
             "Same book, same hidden passcode, two different moments.", y=0.985)
    save(fig, "selection")


# --------------------------------------------------------------------------- #
# 2. Cost: what a token costs to store, and how long the next one takes
# --------------------------------------------------------------------------- #
def fig_cost():
    d = load("bench_e2e.json")
    if d is None:
        return
    cfg = d["configs"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.1), gridspec_kw={"width_ratios": [1, 1.25], "wspace": 0.22})

    # Left: bytes per token per layer, drawn to scale.
    gap = 28
    bars = [(1, [("keys", 2048, BLUE), ("values", 2048, BLUE)], "Dense cache", "4,096 bytes"),
            (0, [("latent", 1024, AQUA), ("indexer key", 128, AQUA)], "Sparse latent cache", "1,152 bytes")]
    for y, parts, name, total in bars:
        x = 0
        for label, width, color in parts:
            ax.add_patch(Rectangle((x, y - 0.2), width - gap, 0.4, fc=color, ec="none"))
            if width > 500:
                ax.text(x + (width - gap) / 2, y, label, color="white", ha="center", va="center", fontsize=10,
                        fontweight="bold")
            else:
                ax.annotate(label, (x + width / 2, y - 0.2), xytext=(18, -16), textcoords="offset points",
                            fontsize=9.5, color=INK2, arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.8))
            x += width
        ax.text(0, y + 0.3, name, fontsize=10.5, fontweight="bold", va="bottom")
        ax.text(4096, y + 0.3, total, fontsize=10, color=INK2, va="bottom", ha="right")
    ax.set_xlim(0, 4096)
    ax.set_ylim(-0.95, 1.75)
    ax.axis("off")
    ax.text(0, -0.95, "per token, per layer, drawn to scale (bf16)\n128K tokens x 28 layers: 14.0 GiB vs 3.9 GiB",
            fontsize=9.5, color=INK2, va="bottom", linespacing=1.4)

    # Right: whole-model time per generated token.
    for key, label, color in [("latent", "Latent cache,\nfull attention", ORANGE), ("dense", "Dense (original)", BLUE),
                              ("sparse_latent", "Sparse latent", AQUA)]:
        rows = cfg[key]
        xs = [r["context"] for r in rows]
        ys = [r["ms_per_token_graph"] for r in rows]
        ax2.plot(xs, ys, color=color, marker="o", zorder=3)
        end_label(ax2, xs[-1], ys[-1], f"{label}\n{ys[-1]:.0f} ms")
    ctx_axis(ax2)
    ax2.set_yscale("log")
    ax2.set_yticks([5, 10, 20, 50, 100, 200], ["5", "10", "20", "50", "100", "200"])
    ax2.yaxis.set_minor_locator(mticker.NullLocator())
    ax2.set_xlim(right=xs[-1] * 2.6)
    ax2.set_ylim(4.5, 420)
    ax2.set_xlabel("context length (tokens)")
    ax2.set_ylabel("ms per generated token")
    headline(fig, "A quarter of the memory, and decoding stops slowing down",
             "Qwen3-1.7B on an RTX 5090, batch 1, whole model, decode step captured in a CUDA graph.", y=1.02)
    save(fig, "cost")


# --------------------------------------------------------------------------- #
# 3. The indexer bug: recall per layer, before and after
# --------------------------------------------------------------------------- #
def fig_recall():
    d, v1 = load("indexer_recall.json"), load("indexer_recall_v1.json")
    if d is None:
        return
    k = "2048"
    layers = np.arange(1, len(d["per_layer"]) + 1)
    fig, ax = plt.subplots(figsize=(11, 4.1))
    ax.axvspan(20.5, 28.5, color=WASH, zorder=0, lw=0)
    ax.text(28.3, 0.615, "deep layers: attention\nsits on the first tokens\nand the most recent ones",
            fontsize=9.5, color=INK2, ha="right", va="bottom", linespacing=1.35)
    series = [("oracle", d, GRAY, "--", "Best possible 2,048 tokens", 14),
              ("sink_window", d, YELLOW, "-", "First 4 + most recent", -15)]
    if v1:
        series.append(("indexer", v1, MAGENTA, "-", "Indexer, first attempt", 0))
    series.append(("indexer", d, AQUA, "-", "Indexer + always-kept tokens", 0))
    for key, src, color, ls, label, dy in series:
        ys = [l[key][k] for l in src["per_layer"]]
        final = src is d and key == "indexer"
        ax.plot(layers, ys, color=color, ls=ls, lw=2.6 if final else 1.9, marker="o" if final else None,
                markersize=4, zorder=4 if final else 3)
        end_label(ax, layers[-1], ys[-1], label, dy)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlim(0.5, 34.5)
    ax.set_xticks([1, 5, 10, 15, 20, 25, 28])
    ax.set_ylim(0.6, 1.005)
    ax.set_xlabel("layer")
    ax.set_ylabel("attention mass on the kept tokens")
    ax.xaxis.grid(False)
    headline(fig, "The first indexer went blind in the deep layers",
             "Share of the real attention that lands on the 2,048 tokens each rule keeps "
             "(queries 4K+ into 16K-token sequences).", y=1.02)
    save(fig, "indexer_recall")


# --------------------------------------------------------------------------- #
# 4. Quality: perplexity vs context, and needle retrieval
# --------------------------------------------------------------------------- #
def fig_quality():
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(11, 6.6), gridspec_kw={"height_ratios": [2.3, 1], "hspace": 0.42})
    series = [("pg19_sink_window.json", "Latent, first 4 +\nrecent window", YELLOW, 0),
              ("pg19_sparse_latent.json", "Sparse latent", AQUA, 3),
              ("pg19_latent_r512_16k.json", "Latent, full attention", ORANGE, -2),
              ("pg19_dense.json", "Dense (original)", BLUE, 0)]
    for f, label, color, dy in series:
        d = load(f)
        if d is None:
            continue
        xs = [int(k) for k in d["ppl_vs_context"]]
        ys = list(d["ppl_vs_context"].values())
        ax.plot(xs, ys, color=color, marker="o", zorder=4 if color == AQUA else 3)
        end_label(ax, xs[-1], ys[-1], label, dy)
    ctx_axis(ax)
    ax.axvspan(100, 2048, color=WASH, zorder=0, lw=0)
    ax.text(140, 21.62, "context fits in the 2,048-token budget:\nsparse and full attention are the same thing",
            fontsize=9.5, color=INK2, va="top", linespacing=1.35)
    ax.set_xlim(110, 31744 * 2.4)
    ax.text(31744, 19.66, "32K is twice the\ntraining length", fontsize=9.5, color=INK2, ha="center", va="bottom",
            linespacing=1.3)
    ax.set_xlabel("context before the scored passage (tokens)")
    ax.set_ylabel("perplexity (lower is better)")
    ax.set_title("Predicting the same 1,024 tokens of a book, given more and more of what came before",
                 fontsize=10.5, color=INK2, loc="left", pad=8)

    runs = [("niah_dense.json", "Dense (original)", BLUE), ("niah_latent_16k.json", "Latent, full attention", ORANGE),
            ("niah_sparse_latent.json", "Sparse latent", AQUA),
            ("niah_sink_window.json", "First 4 + recent window", YELLOW)]
    runs = [(load(f), t, c) for f, t, c in runs if load(f)]
    lengths, nd = runs[0][0]["args"]["lengths"], runs[0][0]["args"]["depths"]
    for row, (d, name, color) in enumerate(runs):
        accs = []
        for gi, L in enumerate(lengths):
            for di in range(nd):
                acc = d["grid"][f"{L}/{di / (nd - 1):.1f}"]
                accs.append(acc)
                x = gi * (nd + 2) + di
                if acc > 0:
                    bx.add_patch(Rectangle((x + 0.08, row - 0.36), 0.84, 0.72, fc=color, alpha=acc, ec="none"))
                else:
                    bx.add_patch(Rectangle((x + 0.08, row - 0.36), 0.84, 0.72, fc=SURFACE, ec="#cdc8bb", lw=0.9))
        bx.text(-1.2, row, name, ha="right", va="center", fontsize=10)
        bx.text(len(lengths) * (nd + 2) - 1, row, f"{np.mean(accs):.0%}", ha="left", va="center", fontsize=10,
                color=INK2)
    for gi, L in enumerate(lengths):
        bx.text(gi * (nd + 2) + nd / 2, len(runs) - 0.25, f"{ctx_label(L)} context", ha="center", va="top",
                fontsize=9.5, color=INK2)
    bx.text(0, -0.85, "hidden at the start → at the end", ha="left", va="bottom", fontsize=8.5, color=INK2)
    bx.set_xlim(-12, len(lengths) * (nd + 2) + 2)
    bx.set_ylim(len(runs) + 0.1, -1.0)
    bx.axis("off")
    bx.set_title("Finding a 7-digit passcode hidden in a book: each square is one hiding place, "
                 "filled = recalled exactly", fontsize=10.5, color=INK2, loc="left", pad=10)
    headline(fig, "Well-chosen tokens match full attention up to 16K; a fixed window does not",
             "PG19 test books. Every sparse rule gets the same budget of 2,048 tokens per layer.", y=0.975)
    save(fig, "quality")


# --------------------------------------------------------------------------- #
# 5. The kernel: a schematic, and where the time goes
# --------------------------------------------------------------------------- #
def fig_kernel():
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.set_xlim(0, 110)
    ax.set_ylim(-2, 46)
    ax.axis("off")

    def box(x, y, w, h, fc, text=None, sub=None, ec="none", grid=None):
        ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=1.1))
        if grid:
            for i in range(1, grid[0]):
                ax.plot([x, x + w], [y + h * i / grid[0]] * 2, color=SURFACE, lw=0.9)
            for i in range(1, grid[1]):
                ax.plot([x + w * i / grid[1]] * 2, [y, y + h], color=SURFACE, lw=0.9)
        if text:
            ax.text(x + w / 2, y + h + 1.1, text, ha="center", va="bottom", fontsize=9.8, fontweight="bold")
        if sub:
            ax.text(x + w / 2, y - 1.0, sub, ha="center", va="top", fontsize=8.8, color=INK2, linespacing=1.3)

    def arrow(x0, y0, x1, y1, rad=0.0):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=11, color=INK2, lw=1.2,
                                     connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))

    # The latent cache as a strip of tokens, with the picked ones highlighted.
    n, x0, cw = 64, 4, 1.56
    picked = set(range(4)) | {11, 12, 19, 27, 28, 29, 36, 43, 49} | set(range(58, 64))
    for i in range(n):
        ax.add_patch(Rectangle((x0 + i * cw + 0.12, 38), cw - 0.24, 3.2, fc=AQUA if i in picked else GRID, ec="none"))
    ax.text(x0, 42.6, "latent cache: one row of 512 numbers per token", fontsize=9.8, fontweight="bold")
    ax.text(x0 + 2 * cw, 36.9, "first 4", fontsize=8.8, color=INK2, ha="center", va="top")
    ax.text(x0 + 30 * cw, 36.9, "picked by the indexer", fontsize=8.8, color=INK2, ha="center", va="top")
    ax.text(x0 + 61 * cw, 36.9, "most recent", fontsize=8.8, color=INK2, ha="center", va="top")

    # One Triton program.
    ax.add_patch(FancyBboxPatch((2.5, -1.5), 105, 32.5, boxstyle="round,pad=0,rounding_size=1.2", fc=WASH,
                                ec="#cdc8bb", lw=1.1, ls=(0, (5, 4))))
    ax.text(4, -0.6, "inside one Triton program: the rebuilt keys and values exist only here "
            "and are never written back to GPU memory", fontsize=9, color=INK2, va="bottom")
    arrow(9.5, 35.2, 9.5, 28.2)
    ax.text(10.8, 32.9, "gather 32 rows at a time", fontsize=9, color=INK2, va="center")

    box(5, 12, 9, 12, AQUA, "latents", "32 x 512", grid=(6, 4))
    arrow(15.5, 18, 26.5, 21.5, rad=-0.12)
    arrow(15.5, 18, 26.5, 12.5, rad=0.12)
    ax.text(20.3, 23.4, r"$\times\,W_{up}$", fontsize=10.5, color=INK2, ha="center")
    box(28, 18.5, 8, 6, BLUE, "keys", None, grid=(4, 3))
    box(28, 9.5, 8, 6, BLUE, None, "values\n32 x 128 each", grid=(4, 3))
    arrow(37.5, 21.5, 44.5, 21.5)
    box(46, 18.5, 15, 6, SURFACE, "normalise + rotate", None, ec="#b5b0a3")
    ax.text(53.5, 21.5, "RMSNorm, RoPE\nfrom position", ha="center", va="center", fontsize=8.8, color=INK2,
            linespacing=1.3)
    arrow(62.5, 21.5, 69.5, 21.5)
    box(71, 18.5, 11, 6, SURFACE, "scores", None, ec="#b5b0a3")
    ax.text(76.5, 21.5, "query · key", ha="center", va="center", fontsize=8.8, color=INK2)
    arrow(83.5, 21.5, 89.5, 19.0, rad=-0.1)
    arrow(37.5, 12.5, 89.5, 14.0, rad=0.04)
    box(91, 10.5, 14, 11, SURFACE, "running softmax", None, ec="#b5b0a3")
    ax.text(98, 16, "keep max, sum,\nweighted values;\nthen next 32 rows", ha="center", va="center", fontsize=8.8,
            color=INK2, linespacing=1.3)
    headline(fig, "One kernel: gather, rebuild, attend",
             "The decode step for one layer. Only the picked rows of the cache are touched, "
             "however long the context is.", y=0.99)
    save(fig, "kernel")


def fig_decode():
    d = load("bench_decode.json")
    if d is None:
        return
    rows = d["results"]
    xs = np.array([r["context"] for r in rows])
    fused = np.array([r["triton_fused_attn"] for r in rows])
    scores = np.array([r["triton_index_scores"] for r in rows])
    topk = np.array([r["topk_only"] for r in rows])
    dense = np.array([r["dense"] for r in rows])
    fig, ax = plt.subplots(figsize=(11, 4.1))
    bands = [(fused, AQUA, "fused attention kernel"), (scores, MAGENTA, "indexer scores"),
             (topk, YELLOW, "top-k (torch)")]
    base = np.zeros_like(fused)
    for vals, color, label in bands:
        ax.fill_between(xs, base, base + vals, color=color, lw=0, zorder=2)
        ax.plot(xs, base + vals, color=SURFACE, lw=2, zorder=3)
        ax.text(xs[-1] * 1.06, base[-1] + vals[-1] / 2, f"{label}  {vals[-1]:.0f} µs", fontsize=9.5,
                color=INK2, va="center")
        base = base + vals
    ax.plot(xs, dense, color=BLUE, marker="o", zorder=4)
    i = int(np.searchsorted(dense, 400)) - 1
    ax.annotate(f"Dense attention (torch)\nkeeps climbing: {dense[-1] / 1000:.2f} ms at 128K", (xs[i], dense[i]),
                xytext=(-12, 16), textcoords="offset points", fontsize=9.5, color=INK2, ha="right", va="bottom",
                linespacing=1.3)
    if "dense_roofline" in rows[0]:
        roof = np.array([r["dense_roofline"] for r in rows])
        ax.plot(xs, roof, color=GRAY, ls="--", lw=1.6, zorder=4)
        ax.text(xs[-1] * 1.06, roof[-1], "time just to read a dense cache\nonce (no kernel can beat it)",
                fontsize=9.5, color=INK2, va="center", linespacing=1.3)
    ctx_axis(ax)
    ax.set_xlim(xs[0] * 0.92, xs[-1] * 3.4)
    ax.set_ylim(0, 420)
    ax.set_xticks(xs)
    ax.set_xlabel("context length (tokens)")
    ax.set_ylabel("µs per decode step, one layer")
    ax.xaxis.grid(False)
    headline(fig, "The kernel is flat; top-k is now the slow part",
             "Time for one layer's attention during decoding, stacked by stage "
             "(RTX 5090, batch 1, 2,048 tokens kept).", y=1.02)
    save(fig, "decode_latency")


if __name__ == "__main__":
    fig_selection()
    fig_cost()
    fig_recall()
    fig_quality()
    fig_kernel()
    fig_decode()
