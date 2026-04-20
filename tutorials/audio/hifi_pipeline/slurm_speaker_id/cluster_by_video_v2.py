#!/usr/bin/env python3
"""Per-video speaker clustering v2.

Uses GroupByVideoStage.extract_video_id for video ID resolution (supports
manifest 'id', 'youtube_id', 'audio_item_id' fields + filepath regex fallback).

Flow:
  1. Load manifests + embeddings for all shards
  2. Resolve video_id per row via extract_video_id()
  3. AHC clustering per video group (center_global normalization)
  4. Write annotated manifests with speaker_label, confidence_score, video_id
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-video speaker clustering v2")
    p.add_argument("--manifest_dir", required=True)
    p.add_argument("--embedding_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--threshold", type=float, default=0.292)
    p.add_argument("--audio_filepath_key", default="audio_filepath")
    return p.parse_args()


def _get_extract_video_id():
    """Import extract_video_id from GroupByVideoStage module."""
    try:
        from nemo_curator.stages.audio.preprocessing.group_by_video import extract_video_id
        return extract_video_id
    except ImportError:
        # Fallback: inline extraction for environments without nemo_curator
        _converted_re = re.compile(r"_converted_\w+_([A-Za-z0-9_-]{11})_")
        _yt_re = re.compile(r"[A-Za-z0-9_-]{11}")

        def extract_video_id(row, id_key="video_id", parse_filepath=True, filepath_key="audio_filepath"):
            for key in ("video_id", "id", "youtube_id", "audio_item_id"):
                val = row.get(key)
                if val and str(val) not in ("", "None", "nan"):
                    s = str(val)
                    if "__" in s:
                        s = s.rsplit("__", 1)[-1]
                    if _yt_re.fullmatch(s):
                        return s
                    return str(val)
            if parse_filepath:
                afp = row.get(filepath_key, "")
                m = _converted_re.search(afp)
                if m:
                    return m.group(1)
            return "unknown"

        return extract_video_id


def l2_normalize(embs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    return embs / np.maximum(norms, 1e-8)


def cluster_embeddings(embs: np.ndarray, threshold: float) -> np.ndarray:
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    n = embs.shape[0]
    if n <= 1:
        return np.ones(n, dtype=int)

    normed = l2_normalize(embs)
    sim = np.clip(normed @ normed.T, -1.0, 1.0)
    dist = 1.0 - sim
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")
    return fcluster(Z, t=1.0 - threshold, criterion="distance")


def speaker_confidence(embs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    n = len(labels)
    if n <= 1:
        return np.zeros(n, dtype=np.float32)
    normed = l2_normalize(embs)
    sim = np.clip(normed @ normed.T, -1.0, 1.0)

    cluster_idx = defaultdict(list)
    for i, lab in enumerate(labels):
        cluster_idx[lab].append(i)

    unique = sorted(cluster_idx.keys())
    label_to_k = {lab: k for k, lab in enumerate(unique)}
    K = len(unique)

    membership = np.zeros((n, K), dtype=np.float32)
    sizes = np.zeros(K, dtype=np.float32)
    for lab, idxs in cluster_idx.items():
        k = label_to_k[lab]
        membership[idxs, k] = 1.0
        sizes[k] = len(idxs)

    mean_sim = (sim @ membership) / np.maximum(sizes, 1.0)
    scores = np.zeros(n, dtype=np.float32)

    for i in range(n):
        my_k = label_to_k[labels[i]]
        if sizes[my_k] < 2:
            continue
        cohesion = (mean_sim[i, my_k] * sizes[my_k] - 1.0) / (sizes[my_k] - 1.0)
        rival = mean_sim[i].copy()
        rival[my_k] = -2.0
        best_rival = rival.max()
        if best_rival <= -2.0:
            scores[i] = 1.0
            continue
        denom = max(cohesion, best_rival)
        scores[i] = max(0.0, min(1.0, (cohesion - best_rival) / denom)) if denom > 0 else 0.0

    return scores


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    extract_video_id = _get_extract_video_id()

    # Phase 1: Load manifests + embeddings, resolve video_id
    print("Phase 1: Loading manifests and embeddings...")

    video_groups: dict[str, list[tuple[str, int, dict, np.ndarray | None]]] = defaultdict(list)

    manifest_files = sorted(
        [f for f in os.listdir(args.manifest_dir) if f.endswith(".jsonl")],
        key=lambda f: int(m.group(1)) if (m := re.search(r"_(\d+)\.", f)) else 0,
    )

    total_rows = 0
    total_with_emb = 0

    for mf in manifest_files:
        shard_match = re.search(r"_(\d+)\.", mf)
        if not shard_match:
            continue
        shard_id = shard_match.group(1)

        manifest_path = os.path.join(args.manifest_dir, mf)
        with open(manifest_path) as f:
            rows = [json.loads(l) for l in f if l.strip()]

        emb_path = os.path.join(args.embedding_dir, f"embeddings_{shard_id}.npz")
        id_to_emb = {}
        if os.path.isfile(emb_path):
            data = np.load(emb_path, allow_pickle=True)
            cut_ids = data["cut_ids"]
            embs = data["embeddings"].astype(np.float32)
            for i, cid in enumerate(cut_ids):
                id_to_emb[str(cid)] = embs[i]

        for row_idx, row in enumerate(rows):
            vid = extract_video_id(row)
            row["video_id"] = vid
            afp = row.get(args.audio_filepath_key, "")
            emb = id_to_emb.get(afp)
            video_groups[vid].append((mf, row_idx, row, emb))
            total_rows += 1
            if emb is not None:
                total_with_emb += 1

    n_videos = len(video_groups)
    n_unknown = sum(1 for v in video_groups if v == "unknown")
    print(f"  {total_rows} rows, {total_with_emb} with embeddings, {n_videos} videos ({n_unknown} unknown)")

    # Phase 2: Cluster per video
    print("Phase 2: Clustering per video...")

    total_speakers = 0
    videos_processed = 0

    for vid, entries in video_groups.items():
        valid_indices = []
        valid_embs = []
        for i, (mf, row_idx, row, emb) in enumerate(entries):
            if emb is not None:
                valid_indices.append(i)
                valid_embs.append(emb)

        if not valid_embs:
            for mf, row_idx, row, emb in entries:
                row["speaker_label"] = -1
                row["confidence_score"] = 0.0
            continue

        emb_matrix = np.stack(valid_embs)
        emb_matrix = emb_matrix - emb_matrix.mean(axis=0, keepdims=True)

        labels = cluster_embeddings(emb_matrix, args.threshold)
        scores = speaker_confidence(emb_matrix, labels)

        n_spk = len(set(labels))
        total_speakers += n_spk

        for idx_in_valid, entry_idx in enumerate(valid_indices):
            mf, row_idx, row, emb = entries[entry_idx]
            row["speaker_label"] = int(labels[idx_in_valid])
            row["confidence_score"] = round(float(scores[idx_in_valid]), 6)

        for i, (mf, row_idx, row, emb) in enumerate(entries):
            if emb is None:
                row["speaker_label"] = -1
                row["confidence_score"] = 0.0

        # Globally unique speaker labels: video hash prefix
        vid_prefix = abs(hash(vid)) % 1_000_000
        for mf, row_idx, row, emb in entries:
            if row["speaker_label"] != -1:
                row["speaker_label"] = vid_prefix * 10000 + row["speaker_label"]

        videos_processed += 1
        if videos_processed % 5000 == 0:
            print(f"  [{videos_processed}/{n_videos}] videos, {total_speakers} speakers")

    print(f"  Done: {videos_processed} videos, {total_speakers} speakers")

    # Phase 3: Write output manifests (preserve shard structure)
    print("Phase 3: Writing output manifests...")

    shard_rows: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for vid, entries in video_groups.items():
        for mf, row_idx, row, emb in entries:
            shard_rows[mf].append((row_idx, row))

    for mf, rows in shard_rows.items():
        rows.sort(key=lambda x: x[0])
        out_path = os.path.join(args.output_dir, mf)
        with open(out_path, "w") as f:
            for row_idx, row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"  Wrote {len(shard_rows)} manifest files to {args.output_dir}")

    # Stats
    spk_counts = Counter()
    for vid, entries in video_groups.items():
        for mf, row_idx, row, emb in entries:
            spk_counts[row.get("speaker_label", -1)] += 1
    n_labeled = sum(c for l, c in spk_counts.items() if l != -1)
    n_unlabeled = spk_counts.get(-1, 0)
    print(f"  {n_labeled} labeled, {n_unlabeled} unlabeled, {len(spk_counts) - (1 if -1 in spk_counts else 0)} unique speakers")


if __name__ == "__main__":
    main()
