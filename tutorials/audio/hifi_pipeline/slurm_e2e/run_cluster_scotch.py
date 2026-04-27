#!/usr/bin/env python3
"""Corpus-wide speaker clustering via SCOTCH (BIRCH + AHC).

Gathers all per-shard embeddings + manifests for one corpus subset,
runs :func:`cluster_embeddings_large_scale` once over the concatenated
embedding matrix using the stock ``librispeech-2026-04`` preset, and
scatters speaker labels / confidence scores back to per-shard JSONLs.
Writes a single ``cluster_config.json`` sidecar next to the outputs.

This is the alternative to ``run_stage.py --stage cluster`` (which
clusters per-video as a memory-avoidance hack). BIRCH makes corpus-wide
feasible, so we can drop the per-video grouping.

Usage::

    python run_cluster_scotch.py \\
        --manifest_dir /path/to/e2e_output/<sub>/transcribe \\
        --embedding_dir /path/to/e2e_output/<sub>/embeddings \\
        --output_dir   /path/to/e2e_output/<sub>/clustered_scotch \\
        --num_shards 64
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np

sys.stdout.reconfigure(line_buffering=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Corpus-wide SCOTCH clustering")
    p.add_argument("--manifest_dir", required=True,
                   help="Dir with shard_<N>.jsonl (transcribe stage outputs).")
    p.add_argument("--embedding_dir", required=True,
                   help="Dir with embeddings_<N>.npz from the embed stage.")
    p.add_argument("--output_dir", required=True,
                   help="Per-shard clustered JSONLs + cluster_config.json land here.")
    p.add_argument("--num_shards", type=int, required=True)
    p.add_argument("--preset", default="librispeech-2026-04",
                   help="SCOTCH preset name from cluster_config.PRESETS.")
    p.add_argument("--audio_filepath_key", default="audio_filepath")
    p.add_argument("--max_leaf_subclusters", type=int, default=150_000,
                   help="Upper bound on BIRCH leaves before AHC. Lower "
                        "this on memory-tight nodes; the clustering code "
                        "auto-relaxes the BIRCH cosine floor to honour it.")
    p.add_argument("--birch_initial_floor", type=float, default=None,
                   help="Override BIRCH starting cosine floor (default: "
                        "preset value). Use lower values for very large "
                        "corpora to skip the known-too-tight first fit.")
    return p.parse_args()


def _load_shard_manifest(path: str) -> List[Dict]:
    rows: List[Dict] = []
    if not os.path.isfile(path):
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_shard_embeddings(path: str) -> Dict[str, np.ndarray]:
    if not os.path.isfile(path):
        return {}
    data = np.load(path, allow_pickle=True)
    cut_ids = data["cut_ids"]
    embs = data["embeddings"].astype(np.float32)
    return {str(cid): embs[i] for i, cid in enumerate(cut_ids)}


def gather_corpus(
    manifest_dir: str,
    embedding_dir: str,
    num_shards: int,
    audio_filepath_key: str,
) -> Tuple[np.ndarray, List[Tuple[int, int]], List[List[Dict]]]:
    """Return concatenated embeddings + per-row (shard_id, row_idx) + per-shard row lists.

    * ``embeddings``: ``(N, D)`` float32 — only rows with a matching embedding.
    * ``origin``: list of ``(shard_id, row_idx_in_shard)`` for each embedding
       row (same length as ``embeddings``).
    * ``shard_rows``: list indexed by shard_id. Each entry is the list of
       *all* rows for that shard (including ones without embeddings, which
       will be labelled as dropped).
    """
    all_embs: List[np.ndarray] = []
    origin: List[Tuple[int, int]] = []
    shard_rows: List[List[Dict]] = [[] for _ in range(num_shards)]
    missing_emb_total = 0

    t0 = time.time()
    for sid in range(num_shards):
        manifest_path = os.path.join(manifest_dir, f"shard_{sid}.jsonl")
        emb_path = os.path.join(embedding_dir, f"embeddings_{sid}.npz")
        rows = _load_shard_manifest(manifest_path)
        shard_rows[sid] = rows
        id_to_emb = _load_shard_embeddings(emb_path)

        for row_idx, row in enumerate(rows):
            afp = row.get(audio_filepath_key, "")
            emb = id_to_emb.get(afp)
            if emb is None:
                emb = id_to_emb.get(os.path.basename(afp))
            if emb is None:
                missing_emb_total += 1
                continue
            all_embs.append(emb)
            origin.append((sid, row_idx))

    if not all_embs:
        raise RuntimeError(
            f"No embeddings matched any manifest row across {num_shards} shards. "
            f"manifest_dir={manifest_dir}  embedding_dir={embedding_dir}"
        )

    embeddings = np.stack(all_embs, axis=0)
    print(
        f"Gathered {embeddings.shape[0]:,} embeddings "
        f"(dim={embeddings.shape[1]}) across {num_shards} shards "
        f"in {time.time() - t0:.1f}s. "
        f"Rows without embeddings: {missing_emb_total:,}"
    )
    return embeddings, origin, shard_rows


def scatter_labels(
    shard_rows: List[List[Dict]],
    origin: List[Tuple[int, int]],
    labels: np.ndarray,
    confidence: np.ndarray,
    dropped_label: int,
) -> None:
    """Write speaker_label / confidence_score into each row in place.

    Rows whose embedding wasn't present get speaker_label=-1, confidence=0.0.
    """
    for row_list in shard_rows:
        for row in row_list:
            row["speaker_label"] = dropped_label
            row["confidence_score"] = 0.0

    for i, (sid, row_idx) in enumerate(origin):
        lab = int(labels[i])
        conf = float(confidence[i]) if confidence is not None else 0.0
        shard_rows[sid][row_idx]["speaker_label"] = lab
        shard_rows[sid][row_idx]["confidence_score"] = round(conf, 6)


def write_shards(shard_rows: List[List[Dict]], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for sid, rows in enumerate(shard_rows):
        out_path = os.path.join(output_dir, f"shard_{sid}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()

    from nemo_curator.stages.audio.speaker_id.clustering.cluster_config import (
        PRESETS, build_cluster_config, cosine_floor_to_birch_radius,
        write_cluster_config,
    )
    from nemo_curator.stages.audio.speaker_id.clustering.large_scale_clustering_and_scoring import (
        DROPPED_LABEL, cluster_embeddings_large_scale, print_large_scale_summary,
    )

    if args.preset not in PRESETS:
        raise ValueError(
            f"Unknown preset {args.preset!r}. Known: {sorted(PRESETS)}"
        )
    preset_values = PRESETS[args.preset]
    if not preset_values:
        raise ValueError(
            f"Preset {args.preset!r} is empty (custom); supply explicit params "
            "or pick a named preset such as 'librispeech-2026-04'."
        )

    cluster_threshold = float(preset_values["cluster_threshold"])
    linkage_method = str(preset_values["cluster_linkage"])
    min_cluster_size = int(preset_values["min_cluster_size"])
    birch_cosine_floor = float(preset_values["birch_cosine_floor"])
    if args.birch_initial_floor is not None:
        birch_cosine_floor = float(args.birch_initial_floor)
    branching_factor = int(preset_values["birch_branching_factor"])
    partial_fit_batch = int(preset_values["birch_partial_fit_batch"])
    assign_tile = int(preset_values["assign_tile"])
    embedding_normalization = str(preset_values["embedding_normalization"])
    birch_radius = cosine_floor_to_birch_radius(birch_cosine_floor)

    print(f"SCOTCH preset: {args.preset}")
    print(f"  cluster_threshold    = {cluster_threshold}")
    print(f"  min_cluster_size     = {min_cluster_size}")
    print(f"  birch_cosine_floor   = {birch_cosine_floor}  "
          f"(radius {birch_radius:.4f})")
    print(f"  branching_factor     = {branching_factor}")
    print(f"  partial_fit_batch    = {partial_fit_batch}")
    print(f"  assign_tile          = {assign_tile}")

    embeddings, origin, shard_rows = gather_corpus(
        args.manifest_dir, args.embedding_dir, args.num_shards,
        args.audio_filepath_key,
    )

    if embedding_normalization == "center_global":
        print("Applying embedding_normalization=center_global")
        embeddings -= embeddings.mean(axis=0, keepdims=True)

    t_cluster = time.time()
    labels, confidence, stats = cluster_embeddings_large_scale(
        embeddings,
        threshold=cluster_threshold,
        linkage_method=linkage_method,
        min_cluster_size=min_cluster_size,
        birch_threshold=birch_radius,
        branching_factor=branching_factor,
        partial_fit_batch=partial_fit_batch,
        assign_tile=assign_tile,
        compute_confidence=True,
        dropped_label=DROPPED_LABEL,
        max_leaf_subclusters=args.max_leaf_subclusters,
    )
    cluster_runtime = time.time() - t_cluster
    print(f"Clustering runtime: {cluster_runtime:.1f}s")
    if stats.get("birch_retries", 0) > 0:
        eff_rad = stats.get("effective_birch_threshold", birch_radius)
        eff_cos = 1.0 - (eff_rad ** 2) / 2.0
        print(f"BIRCH leaf-cap backoff: {stats['birch_retries']} retries, "
              f"effective radius={eff_rad:.4f} (cosine floor ~{eff_cos:.3f})")
    print_large_scale_summary(labels, stats)

    scatter_labels(shard_rows, origin, labels, confidence, DROPPED_LABEL)
    write_shards(shard_rows, args.output_dir)
    print(f"Wrote {args.num_shards} shard_*.jsonl files to {args.output_dir}")

    n_kept = int((labels != DROPPED_LABEL).sum())
    n_dropped = int((labels == DROPPED_LABEL).sum())
    effective_radius = float(stats.get("effective_birch_threshold", birch_radius))
    effective_cos_floor = max(-1.0, min(1.0, 1.0 - (effective_radius ** 2) / 2.0))
    cfg = build_cluster_config(
        backend="large_scale",
        preset=args.preset,
        cluster_threshold=cluster_threshold,
        cluster_linkage=linkage_method,
        min_cluster_size=min_cluster_size,
        n_input=int(embeddings.shape[0]),
        embedding_dim=int(embeddings.shape[1]),
        embedding_normalization=embedding_normalization,
        confidence_enabled=True,
        birch_cosine_floor=effective_cos_floor,
        birch_radius=effective_radius,
        birch_branching_factor=branching_factor,
        birch_partial_fit_batch=partial_fit_batch,
        assign_tile=assign_tile,
        n_leaf_subclusters=stats.get("n_leaf_subclusters"),
        n_clusters_raw=stats.get("n_clusters_raw"),
        n_clusters_kept=stats.get("filter", {}).get("n_clusters_after"),
        n_utts_kept=n_kept,
        n_utts_dropped=n_dropped,
        runtime_seconds=cluster_runtime,
        extra={
            "birch_retries": int(stats.get("birch_retries", 0)),
            "max_leaf_subclusters": int(stats.get("max_leaf_subclusters", args.max_leaf_subclusters)),
            "birch_cosine_floor_requested": birch_cosine_floor,
            "birch_radius_requested": birch_radius,
        },
    )
    write_cluster_config(args.output_dir, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
