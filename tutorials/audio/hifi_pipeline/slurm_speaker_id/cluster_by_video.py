#!/usr/bin/env python3
"""Cluster speaker embeddings per video ID.

Groups utterances by video ID (from 'id' field or parsed from audio_filepath),
runs AHC clustering within each video, writes annotated manifests.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest_dir", required=True)
    p.add_argument("--embedding_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--threshold", type=float, default=0.292)
    p.add_argument("--audio_filepath_key", default="audio_filepath")
    return p.parse_args()


def extract_video_id(row: dict) -> str:
    """Extract video ID from manifest row."""
    # Try explicit 'id' field first
    vid = row.get("id")
    if vid and vid != "None" and str(vid) != "None":
        return str(vid)
    # Parse from audio_filepath: ..._VIDEOID_offset_dur_... pattern
    # e.g. _data0_ameister_converted_ru103_eZ0a2mvLOQE_181_36_0_16.wav
    afp = row.get("audio_filepath", "")
    # YouTube IDs are 11 chars: [A-Za-z0-9_-]
    # Pattern: after "converted_ruNNN_" comes the video ID
    m = re.search(r"_converted_\w+_([A-Za-z0-9_-]{11})_", afp)
    if m:
        return m.group(1)
    # Fallback: try youtube_id field
    yt = row.get("youtube_id")
    if yt:
        return str(yt)
    return "unknown"


def l2_normalize(embs):
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    return embs / np.maximum(norms, 1e-8)


def cluster_embeddings(embs, threshold):
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


def speaker_confidence(embs, labels):
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


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Phase 1: Load all manifests + embeddings, build per-video index
    print("Phase 1: Loading manifests and embeddings...")

    # Structure: video_id -> list of (shard_file, row_idx, row_dict, embedding_or_None)
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

        # Load embeddings for this shard
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
            afp = row.get(args.audio_filepath_key, "")
            emb = id_to_emb.get(afp)
            video_groups[vid].append((mf, row_idx, row, emb))
            total_rows += 1
            if emb is not None:
                total_with_emb += 1

    n_videos = len(video_groups)
    print(f"  {total_rows} rows, {total_with_emb} with embeddings, {n_videos} videos")

    # Phase 2: Cluster per video
    print("Phase 2: Clustering per video...")

    total_speakers = 0
    videos_processed = 0

    for vid, entries in video_groups.items():
        # Collect embeddings for this video
        valid_indices = []
        valid_embs = []
        for i, (mf, row_idx, row, emb) in enumerate(entries):
            if emb is not None:
                valid_indices.append(i)
                valid_embs.append(emb)

        if not valid_embs:
            # No embeddings — mark all as unknown
            for mf, row_idx, row, emb in entries:
                row["speaker_label"] = -1
                row["confidence_score"] = 0.0
            continue

        emb_matrix = np.stack(valid_embs)
        # Center (batch mean subtraction)
        emb_matrix = emb_matrix - emb_matrix.mean(axis=0, keepdims=True)

        labels = cluster_embeddings(emb_matrix, args.threshold)
        scores = speaker_confidence(emb_matrix, labels)

        n_spk = len(set(labels))
        total_speakers += n_spk

        # Assign labels back
        for idx_in_valid, entry_idx in enumerate(valid_indices):
            mf, row_idx, row, emb = entries[entry_idx]
            row["speaker_label"] = int(labels[idx_in_valid])
            row["confidence_score"] = round(float(scores[idx_in_valid]), 6)

        # Mark entries without embeddings
        for i, (mf, row_idx, row, emb) in enumerate(entries):
            if emb is None:
                row["speaker_label"] = -1
                row["confidence_score"] = 0.0

        # Prefix speaker labels with video hash to make globally unique
        vid_prefix = abs(hash(vid)) % 1_000_000
        for mf, row_idx, row, emb in entries:
            if row["speaker_label"] != -1:
                row["speaker_label"] = vid_prefix * 10000 + row["speaker_label"]

        videos_processed += 1
        if videos_processed % 5000 == 0:
            print(f"  [{videos_processed}/{n_videos}] videos clustered, {total_speakers} speakers so far")

    print(f"  Done: {videos_processed} videos, {total_speakers} speakers")

    # Phase 3: Write output manifests (same shard structure as input)
    print("Phase 3: Writing output manifests...")

    # Regroup by shard file
    shard_rows: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for vid, entries in video_groups.items():
        for mf, row_idx, row, emb in entries:
            shard_rows[mf].append((row_idx, row))

    for mf, rows in shard_rows.items():
        rows.sort(key=lambda x: x[0])  # preserve original order
        out_path = os.path.join(args.output_dir, mf)
        with open(out_path, "w") as f:
            for row_idx, row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"  Wrote {len(shard_rows)} manifest files to {args.output_dir}")


if __name__ == "__main__":
    main()
