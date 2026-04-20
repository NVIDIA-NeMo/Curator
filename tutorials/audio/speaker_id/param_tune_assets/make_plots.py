"""Generate the bar plots referenced from PARAM_TUNE.md.

Reads ``tuning_results.csv`` + ``best_threshold.json`` and writes:

  * ``f1_per_birch_floor.png``   - best F1 attainable at each BIRCH cosine floor
  * ``best_cell_metrics.png``    - P / R / F1 / NMI / ARI at the chosen cell
  * ``ahc_pr_curve_at_best_birch.png`` -
        precision, recall, F1 vs. AHC threshold (BIRCH floor fixed at the
        chosen value).  The reason "why F1 picks 0.50" lives here.
  * ``cluster_count_vs_ahc.png`` -
        predicted number of clusters vs. AHC threshold, with the true
        speaker count drawn as a horizontal line.  Shows the over- /
        under-merging tradeoff that the threshold trades off.

Usage::

    python make_plots.py \
        --csv  /home/taejinp/work/librispeech_threshold_tune/tuning_results.csv \
        --best /home/taejinp/work/librispeech_threshold_tune/best_threshold.json \
        --out  /home/taejinp/projects/Curator/Curator/tutorials/audio/speaker_id/param_tune_assets
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(fig: plt.Figure, path: str) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def plot_f1_per_birch_floor(df: pd.DataFrame, out_dir: str) -> None:
    """Bar chart: best B-cubed F1 attainable at each BIRCH cosine floor."""
    rows = []
    for bf, g in df.groupby("birch_cosine_floor"):
        r = g.loc[g.bcubed_f1.idxmax()]
        rows.append((bf, r.bcubed_f1, r.ahc_threshold, int(r.num_pred_clusters)))
    rows.sort()
    floors, f1s, ahcs, kpreds = zip(*rows)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar([f"{f:.2f}" for f in floors], f1s,
                  color=["#9bb7d4"] * (len(floors) - 1) + ["#1f3b73"])
    for b, ahc, kp, f1 in zip(bars, ahcs, kpreds, f1s):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                f"F1={f1:.4f}\nAHC={ahc:.2f}\nK={kp}",
                ha="center", va="bottom", fontsize=8.5)
    ax.set_ylim(0.85, 1.005)
    ax.set_xlabel("BIRCH per-leaf cosine floor")
    ax.set_ylabel("Best B-cubed F1 attainable")
    ax.set_title("Tighter BIRCH leaves → better F1 (best AHC chosen per cell)\n"
                 "LibriSpeech 960h tarred train, 280,971 utts, 2,337 spk")
    ax.axhline(1.0, color="gray", lw=0.6, ls=":")
    ax.grid(axis="y", lw=0.4, alpha=0.6)
    _save(fig, os.path.join(out_dir, "f1_per_birch_floor.png"))


def plot_best_cell_metrics(best: dict, out_dir: str) -> None:
    """Bar chart: P / R / F1 / NMI / ARI / Purity / InvPurity at the best cell."""
    metrics = [
        ("B-cubed P",   best["bcubed_precision"]),
        ("B-cubed R",   best["bcubed_recall"]),
        ("B-cubed F1",  best["bcubed_f1"]),
        ("Purity",      best["purity"]),
        ("Inv. purity", best["inverse_purity"]),
        ("NMI",         best["nmi"]),
        ("V-measure",   best["v_measure"]),
        ("ARI",         best["ari"]),
    ]
    names, vals = zip(*metrics)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(names, vals, color="#1f3b73")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0.95, 1.005)
    ax.axhline(1.0, color="gray", lw=0.6, ls=":")
    ax.set_ylabel("Score")
    ax.set_title(f"Quality at the chosen cell  "
                 f"(AHC={best['ahc_threshold']:.2f}, "
                 f"BIRCH cosine floor={best['birch_cosine_floor']:.2f})\n"
                 f"K_pred={int(best['num_pred_clusters'])}  "
                 f"vs.  K_true={int(best['num_true_speakers'])}")
    ax.grid(axis="y", lw=0.4, alpha=0.6)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    _save(fig, os.path.join(out_dir, "best_cell_metrics.png"))


def plot_ahc_pr_curve(df: pd.DataFrame, best: dict, out_dir: str) -> None:
    """P / R / F1 / NMI vs AHC threshold at the chosen BIRCH floor."""
    bf = best["birch_cosine_floor"]
    g = df[df.birch_cosine_floor == bf].sort_values("ahc_threshold").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 5.0))
    x = np.arange(len(g))
    width = 0.21
    ax.bar(x - 1.5 * width, g.bcubed_precision, width, label="B-cubed P",  color="#d97a4a")
    ax.bar(x - 0.5 * width, g.bcubed_recall,    width, label="B-cubed R",  color="#6aa84f")
    ax.bar(x + 0.5 * width, g.bcubed_f1,        width, label="B-cubed F1", color="#1f3b73")
    ax.bar(x + 1.5 * width, g.nmi,              width, label="NMI",        color="#a44ed1")

    # Highlight best column
    best_idx = int(g.ahc_threshold.sub(best["ahc_threshold"]).abs().idxmin())
    ax.axvspan(best_idx - 0.5, best_idx + 0.5, color="gold", alpha=0.18,
               label=f"Chosen AHC={best['ahc_threshold']:.2f}")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{a:.2f}" for a in g.ahc_threshold], rotation=70, fontsize=8)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("AHC cosine threshold (lower = merge more aggressively)")
    ax.set_ylabel("Score")
    ax.set_title(f"Why AHC={best['ahc_threshold']:.2f} wins (BIRCH floor fixed at {bf:.2f})\n"
                 "Recall stays ~1.0 across the band; precision climbs with tighter merges; "
                 "F1 follows precision and saturates at the upper end.")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", lw=0.4, alpha=0.5)
    _save(fig, os.path.join(out_dir, "ahc_pr_curve_at_best_birch.png"))


def plot_cluster_count_vs_ahc(df: pd.DataFrame, best: dict, out_dir: str) -> None:
    """Predicted K vs AHC, comparing against true speaker count."""
    bf = best["birch_cosine_floor"]
    g = df[df.birch_cosine_floor == bf].sort_values("ahc_threshold").reset_index(drop=True)
    k_true = int(best["num_true_speakers"])

    fig, ax = plt.subplots(figsize=(10, 4.6))
    x = np.arange(len(g))
    bars = ax.bar(x, g.num_pred_clusters, color="#7494c4")
    best_idx = int(g.ahc_threshold.sub(best["ahc_threshold"]).abs().idxmin())
    bars[best_idx].set_color("#1f3b73")
    ax.axhline(k_true, color="#c0392b", lw=1.5, ls="--",
               label=f"True K = {k_true}")
    ax.text(len(g) - 0.5, k_true + 25, f"True K = {k_true}",
            color="#c0392b", ha="right", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{a:.2f}" for a in g.ahc_threshold], rotation=70, fontsize=8)
    ax.set_xlabel("AHC cosine threshold")
    ax.set_ylabel("# predicted clusters K_pred")
    ax.set_title(f"Predicted vs. true cluster count  (BIRCH floor = {bf:.2f})\n"
                 "Low AHC → severe over-merging (~few hundred clusters); "
                 f"chosen cell sits slightly above K_true (K_pred="
                 f"{int(best['num_pred_clusters'])} vs {k_true}).")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", lw=0.4, alpha=0.5)
    _save(fig, os.path.join(out_dir, "cluster_count_vs_ahc.png"))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv",  required=True)
    p.add_argument("--best", required=True)
    p.add_argument("--out",  required=True)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = pd.read_csv(args.csv)
    with open(args.best) as f:
        best = json.load(f)["best"]

    plot_f1_per_birch_floor(df, args.out)
    plot_best_cell_metrics(best, args.out)
    plot_ahc_pr_curve(df, best, args.out)
    plot_cluster_count_vs_ahc(df, best, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
