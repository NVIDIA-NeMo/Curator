"""Generate the silhouette-confidence histogram bar plot for PARAM_TUNE.md.

Reads the per-utterance ``confidence_score`` from a SCOTCH run's
``clusters_summary.jsonl`` (or any JSONL file with one record per
utterance carrying ``confidence_score`` and ``speaker_label``) and
draws a 50-bin histogram in ``[0, 1]``, restricted to *kept*
utterances (``speaker_label != -1``).

Output PNG (default ``confidence_histogram.png``) is referenced from
PARAM_TUNE.md  9.1a.

Usage
-----

::

    python make_confidence_histogram.py \
        --summary /disk_f_nvd/datasets/librispeech/tarred_train/scotch_speaker_clustering_results/clusters_summary.jsonl \
        --out param_tune_assets

The ``--out`` directory must exist; the file
``<out>/confidence_histogram.png`` is overwritten.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def load_kept_confidences(summary_path: str) -> Tuple[np.ndarray, int]:
    """Return ``(kept_confidences, n_dropped)``."""
    kept: List[float] = []
    n_dropped = 0
    with open(summary_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            label = rec.get("speaker_label", -1)
            conf = rec.get("confidence_score")
            if conf is None:
                conf = 0.0
            if label == -1:
                n_dropped += 1
                continue
            kept.append(float(conf))
    return np.asarray(kept, dtype=np.float64), n_dropped


def plot_hist(
    kept: np.ndarray,
    n_dropped: int,
    out_path: str,
    n_bins: int = 50,
    title_suffix: str = "",
) -> None:
    n_kept = kept.size
    n_total = n_kept + n_dropped

    counts, edges = np.histogram(kept, bins=n_bins, range=(0.0, 1.0))
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = edges[1] - edges[0]

    mode_idx = int(np.argmax(counts))
    pct = 100.0 * counts / max(n_kept, 1)

    fig, ax = plt.subplots(figsize=(11, 5.0))

    bar_color = "#1f3b73"
    mode_color = "#c0392b"
    colors = [mode_color if i == mode_idx else bar_color for i in range(n_bins)]
    ax.bar(centers, counts, width=width * 0.95, color=colors, edgecolor="white",
           linewidth=0.5, zorder=3)

    # Annotate the mode bin.
    ax.annotate(
        f"mode bin\n[{edges[mode_idx]:.3f}, {edges[mode_idx + 1]:.3f})\n"
        f"{counts[mode_idx]:,} utts ({pct[mode_idx]:.2f}%)",
        xy=(centers[mode_idx], counts[mode_idx]),
        xytext=(centers[mode_idx] + 0.18, counts[mode_idx] * 0.92),
        fontsize=9, color=mode_color,
        arrowprops=dict(arrowstyle="->", color=mode_color, lw=1.0),
    )

    # Mark the three operating points recommended in PARAM_TUNE.md  9.5.
    ops = [
        (0.41, "Permissive (0.41)\nret 91.0%, pur 98.78%"),
        (0.55, "Balanced (0.55)\nret 17.0%, pur 99.50%"),
        (0.62, "Gold-only (0.62)\nret 0.6%, pur 100%"),
    ]
    op_color = "#27ae60"
    label_y = max(counts) * 0.40
    for thr, label in ops:
        ax.axvline(thr, color=op_color, lw=1.0, ls="--", alpha=0.85, zorder=4)
        ax.text(thr + 0.005, label_y, label, rotation=90, va="bottom",
                ha="left", fontsize=8, color=op_color)

    # Cosmetics.
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel(
        "Per-utterance silhouette confidence_score\n"
        "((a-b)/max(a,b),  a = cos(emb, own centroid),  b = max cos(emb, other centroid))"
    )
    ax.set_ylabel(f"Utterance count  (50 bins of width {width:.3f})")
    ax.grid(axis="y", alpha=0.35, zorder=0)

    mean = kept.mean()
    median = float(np.median(kept))
    std = kept.std()
    title = (
        "LibriSpeech silhouette confidence_score distribution"
        f"{title_suffix}\n"
        f"N kept = {n_kept:,}   N dropped = {n_dropped:,}   "
        f"mean = {mean:.3f}   median = {median:.3f}   std = {std:.3f}   "
        f"max = {kept.max():.3f}"
    )
    ax.set_title(title, fontsize=11)

    # Custom legend explaining colours.
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend = [
        Patch(facecolor=bar_color, edgecolor="white", label="kept utts"),
        Patch(facecolor=mode_color, edgecolor="white", label="mode bin"),
        Line2D([0], [0], color=op_color, ls="--",
               label="recommended cutoffs (\u00a79.5)"),
    ]
    ax.legend(handles=legend, loc="upper left", fontsize=9, framealpha=0.95)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--summary", required=True,
        help=(
            "Path to clusters_summary.jsonl (or equivalent JSONL with "
            "speaker_label + confidence_score fields per record)."
        ),
    )
    p.add_argument(
        "--out", required=True,
        help="Output directory; writes <out>/confidence_histogram.png.",
    )
    p.add_argument("--bins", type=int, default=50,
                   help="Number of histogram bins in [0, 1] (default: 50).")
    p.add_argument(
        "--title_suffix", default="",
        help=(
            "Optional extra string appended to the plot title (e.g. "
            "' (SCOTCH-v1.large_scale.librispeech-2026-04)')."
        ),
    )
    args = p.parse_args()

    if not os.path.isdir(args.out):
        raise SystemExit(f"--out directory does not exist: {args.out}")

    kept, n_dropped = load_kept_confidences(args.summary)
    if kept.size == 0:
        raise SystemExit(
            f"No kept utterances found in {args.summary}; nothing to plot."
        )

    out_path = os.path.join(args.out, "confidence_histogram.png")
    plot_hist(kept, n_dropped, out_path, n_bins=args.bins,
              title_suffix=args.title_suffix)
    print(f"  wrote {out_path}")
    print(f"  N kept = {kept.size:,}   N dropped = {n_dropped:,}")
    print(f"  min/max = {kept.min():.4f} / {kept.max():.4f}   "
          f"mean = {kept.mean():.4f}   std = {kept.std():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
