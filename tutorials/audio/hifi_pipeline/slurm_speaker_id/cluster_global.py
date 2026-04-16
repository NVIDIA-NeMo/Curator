#!/usr/bin/env python3
"""Global speaker clustering across all shards. Loads all embeddings into memory."""
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


def l2_normalize(embs):
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    return embs / np.maximum(norms, 1e-8)


def cluster_embeddings(embs, threshold):
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    n = embs.shape[0]
    if n <= 1:
        return np.ones(n, dtype=int)

    print(f"  Computing cosine distance matrix ({n}x{n})...")
    normed = l2_normalize(embs)
    sim = np.clip(normed @ normed.T, -1.0, 1.0)
    dist = 1.0 - sim
    del sim

    print(f"  Converting to condensed form...")
    condensed = squareform(dist, checks=False)
    del dist

    print(f"  Running AHC linkage...")
    Z = linkage(condensed, method="average")
    del condensed

    print(f"  Cutting tree at threshold {threshold}...")
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

    manifest_files = sorted(
        [f for f in os.listdir(args.manifest_dir) if f.endswith(".jsonl")],
        key=lambda f: int(m.group(1)) if (m := re.search(r"_(\d+)\.", f)) else 0,
    )

    # Phase 1: Load everything
    print("Phase 1: Loading all manifests and embeddings...")
    all_rows = []  # (shard_file, row_idx, row_dict)
    all_embs = []  # parallel list, None if no embedding
    all_emb_indices = []  # indices into all_rows that have embeddings

    for mf in manifest_files:
        shard_match = re.search(r"_(\d+)\.", mf)
        if not shard_match:
            continue
        shard_id = shard_match.group(1)

        with open(os.path.join(args.manifest_dir, mf)) as f:
            rows = [json.loads(l) for l in f if l.strip()]

        emb_path = os.path.join(args.embedding_dir, f"embeddings_{shard_id}.npz")
        id_to_emb = {}
        if os.path.isfile(emb_path):
            data = np.load(emb_path, allow_pickle=True)
            for i, cid in enumerate(data["cut_ids"]):
                id_to_emb[str(cid)] = data["embeddings"][i]

        for row_idx, row in enumerate(rows):
            global_idx = len(all_rows)
            all_rows.append((mf, row_idx, row))
            afp = row.get(args.audio_filepath_key, "")
            emb = id_to_emb.get(afp)
            if emb is not None:
                all_emb_indices.append(global_idx)
                all_embs.append(emb)

    n_total = len(all_rows)
    n_emb = len(all_embs)
    print(f"  {n_total} total rows, {n_emb} with embeddings")

    # Phase 2: Global clustering
    print("Phase 2: Global clustering...")
    emb_matrix = np.stack(all_embs).astype(np.float32)
    print(f"  Embedding matrix: {emb_matrix.shape}, ~{emb_matrix.nbytes / 1e9:.1f} GB")

    # Center
    emb_matrix = emb_matrix - emb_matrix.mean(axis=0, keepdims=True)

    labels = cluster_embeddings(emb_matrix, args.threshold)
    n_speakers = len(set(labels))
    print(f"  {n_emb} utterances -> {n_speakers} speakers")

    sizes = sorted(Counter(labels).values(), reverse=True)
    print(f"  Top 10 speaker sizes: {sizes[:10]}")
    print(f"  Singletons: {sum(1 for s in sizes if s == 1)}")
    print(f"  Avg utt/speaker: {n_emb / n_speakers:.1f}")

    print("  Computing confidence scores...")
    scores = speaker_confidence(emb_matrix, labels)
    print(f"  Mean confidence: {scores.mean():.3f}")

    # Phase 3: Assign labels back
    print("Phase 3: Assigning labels...")
    # Map: global_idx -> (label, score) for rows with embeddings
    label_map = {}
    for i, global_idx in enumerate(all_emb_indices):
        label_map[global_idx] = (int(labels[i]), round(float(scores[i]), 6))

    for global_idx, (mf, row_idx, row) in enumerate(all_rows):
        if global_idx in label_map:
            row["speaker_label"], row["confidence_score"] = label_map[global_idx]
        else:
            row["speaker_label"] = -1
            row["confidence_score"] = 0.0

    # Phase 4: Write output (same shard structure)
    print("Phase 4: Writing output manifests...")
    shard_rows = defaultdict(list)
    for mf, row_idx, row in all_rows:
        shard_rows[mf].append((row_idx, row))

    for mf, rows in shard_rows.items():
        rows.sort(key=lambda x: x[0])
        out_path = os.path.join(args.output_dir, mf)
        with open(out_path, "w") as f:
            for _, row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"  Wrote {len(shard_rows)} files to {args.output_dir}")
    print(f"\nSummary: {n_emb} utterances -> {n_speakers} speakers (avg {n_emb/n_speakers:.1f} utt/spk)")


if __name__ == "__main__":
    main()
