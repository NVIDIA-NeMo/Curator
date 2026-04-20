"""Generate the memory-comparison bar plot for PARAM_TUNE.md.

Compares peak RAM of brute-force AHC vs BIRCH+AHC across N in
{280k, 1M, 5M, 20M, 30M}.  Uses log scale because the spread is many
orders of magnitude.

Usage::

    python make_memory_plot.py --out param_tune_assets
"""
from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


def upper_tri(n: int) -> int:
    return n * (n - 1) // 2


def brute_force_peak_bytes(N: int, D: int = 192) -> int:
    # Best case: float32 condensed pdist + embeddings.
    return N * D * 4 + upper_tri(N) * 4


def birch_ahc_peak_bytes(
    N: int,
    D: int = 192,
    n_leaves_cap: int = 200_000,
    leaf_ratio: float = 0.25,
    assign_tile: int = 16_384,
) -> tuple[int, int]:
    n_leaves = min(n_leaves_cap, max(1_000, int(N * leaf_ratio)))
    embs = N * D * 4
    leaves = n_leaves * D * 4
    pdist_leaves = upper_tri(n_leaves) * 4
    assign_ws = assign_tile * n_leaves * 4
    return embs + leaves + pdist_leaves + assign_ws, n_leaves


def fmt_bytes(b: float) -> str:
    for unit, sz in [("PB", 2 ** 50), ("TB", 2 ** 40), ("GB", 2 ** 30), ("MB", 2 ** 20)]:
        if b >= sz:
            return f"{b / sz:.1f} {unit}"
    return f"{b / 2 ** 20:.1f} MB"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    scales = [
        ("LibriSpeech\n(280k utts)",         280_971),
        ("1 M utts",                          1_000_000),
        ("5 M utts",                          5_000_000),
        ("YODAS-scale\n(20 M utts)",          20_000_000),
        ("YODAS-scale\n(30 M utts)",          30_000_000),
    ]
    labels = [s[0] for s in scales]
    Ns = [s[1] for s in scales]

    bf = [brute_force_peak_bytes(N) for N in Ns]
    ba = [birch_ahc_peak_bytes(N)[0] for N in Ns]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(scales))
    width = 0.36

    b1 = ax.bar(x - width / 2, bf, width, label="Brute-force AHC (best case fp32)",
                color="#c0392b")
    b2 = ax.bar(x + width / 2, ba, width, label="BIRCH + AHC (this work)",
                color="#1f3b73")

    # Reference horizontal lines: 128 GB and 1 TB.
    ax.axhline(128 * 2 ** 30, color="gray", lw=0.8, ls="--")
    ax.text(len(scales) - 0.5, 128 * 2 ** 30 * 1.15, "128 GB (single-node RAM)",
            color="gray", ha="right", fontsize=8)
    ax.axhline(1 * 2 ** 40, color="gray", lw=0.8, ls=":")
    ax.text(len(scales) - 0.5, 1 * 2 ** 40 * 1.15, "1 TB", color="gray",
            ha="right", fontsize=8)

    for bars, vals in [(b1, bf), (b2, ba)]:
        for rect, v in zip(bars, vals):
            ax.text(rect.get_x() + rect.get_width() / 2,
                    rect.get_height() * 1.20,
                    fmt_bytes(v),
                    ha="center", va="bottom", fontsize=8.5)

    ax.set_yscale("log")
    ax.set_ylim(1e7, 5e15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Peak RAM (log scale, bytes)")
    ax.set_title(
        "Peak memory: brute-force AHC vs. BIRCH + AHC\n"
        "Embeddings dim D = 192 (TitaNet-large).  "
        "Brute-force = condensed fp32 pdist + embeddings.  "
        "BIRCH+AHC capped at n_leaves = 200,000."
    )
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(axis="y", which="both", lw=0.4, alpha=0.5)

    out_path = os.path.join(args.out, "memory_comparison.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")

    # Second plot: ratio (how many times more RAM brute-force needs).
    ratios = [b / a for b, a in zip(bf, ba)]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.bar(x, ratios, color="#1f3b73")
    for r, rect in zip(ratios, bars):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() * 1.05,
                f"{r:,.0f}×",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_yscale("log")
    ax.set_ylim(1, 1e5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Brute-force / BIRCH+AHC peak RAM (log scale)")
    ax.set_title("Memory savings of BIRCH+AHC over brute-force AHC\n"
                 "(higher is more savings; ratio is cubic in N for large N)")
    ax.grid(axis="y", which="both", lw=0.4, alpha=0.5)
    out_path2 = os.path.join(args.out, "memory_ratio.png")
    fig.tight_layout()
    fig.savefig(out_path2, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
