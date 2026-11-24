# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render reported paper results; does not synthesize audio or train models."""

import argparse
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def draw(ax: plt.Axes, rows: list[dict], value: str, spread: str, title: str) -> None:
    """Plot the manuscript's aggregate and its reported ± value."""
    y = np.arange(len(rows))
    colors = ["#64748b" if i == 0 else "#087f8c" for i in y]
    ax.barh(
        y,
        [r[value] for r in rows],
        xerr=[r[spread] for r in rows],
        color=colors,
        error_kw={"capsize": 3, "linewidth": 1},
        height=0.65,
    )
    ax.set_yticks(y, [r["label"] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 28)
    ax.set_xlabel("DER (%) — lower is better")
    ax.set_title(title, loc="left", fontweight="bold", pad=16)
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    for i, row in enumerate(rows):
        ax.text(row[value] + row[spread] + 0.35, i, f"{row[value]:.1f}", va="center", fontsize=10)
    ax.spines[["top", "right", "left"]].set_visible(False)


def render(data_path: Path, output_dir: Path) -> list[Path]:
    """Create exportable SVG/PNG charts from the transcribed paper tables."""
    data = json.loads(data_path.read_text())
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "svg.fonttype": "none",
            "svg.hashsalt": "tts-diarization-paper",
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = []
    for table, name, title in (
        ("I", "direct_cloning", "Direct voice cloning · Table I"),
        ("II", "cross_lingual", "Cross-lingual references · Table II"),
    ):
        fig, ax = plt.subplots(figsize=(11, 5.0), layout="constrained")
        draw(ax, data[table], "macro", "spread", title)
        fig.suptitle(
            "Reported results · Exploring Text-to-Speech as a Data Source for Speaker Diarization", fontsize=10
        )
        figures.append((name, fig))
    fig, axes = plt.subplots(1, 2, figsize=(17, 6), layout="constrained")
    draw(axes[0], data["III"], "out_of_domain", "out_spread", "AMI references · Table III\nAverage excluding AMI")
    draw(axes[1], data["IV"], "in_domain", "in_spread", "AliMeeting references · Table IV\nAverage on AliMeeting")
    fig.suptitle("Reported target-domain experiments · aggregates cover different benchmark subsets", fontsize=12)
    figures.append(("target_domain", fig))
    paths = []
    for name, fig in figures:
        for extension in ("svg", "png"):
            path = output_dir / f"{name}.{extension}"
            metadata = {"Title": name, "Creator": None, "Date": None} if extension == "svg" else {"Title": name}
            fig.savefig(path, dpi=150, facecolor="white", metadata=metadata)
            if extension == "svg":
                path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
            paths.append(path)
        plt.close(fig)
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "figures")
    args = parser.parse_args()
    render(Path(__file__).with_name("results.json"), args.output_dir)
