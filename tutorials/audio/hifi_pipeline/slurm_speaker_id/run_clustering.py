#!/usr/bin/env python3
"""Standalone per-shard speaker clustering. No NeMo dependency.

Reads manifest + embeddings_N.npz pairs, runs AHC, writes annotated manifests.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest_dir", required=True, help="Dir with shard_N.jsonl files")
    p.add_argument("--embedding_dir", required=True, help="Dir with embeddings_N.npz files")
    p.add_argument("--output_dir", required=True, help="Output dir for annotated manifests")
    p.add_argument("--threshold", type=float, default=0.292)
    p.add_argument("--audio_filepath_key", default="audio_filepath")
    return p.parse_args()


def l2_normalize(embs):
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    return embs / np.maximum(norms, 1e-8)


def cluster_embeddings(embs, threshold, linkage_method="average"):
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    n = embs.shape[0]
    if n <= 1:
        return np.ones(n, dtype=int)

    normed = l2_normalize(embs)
    sim = np.clip(normed @ normed.T, -1.0, 1.0)
    dist = 1.0 - sim
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method=linkage_method)
    return fcluster(Z, t=1.0 - threshold, criterion="distance")


def speaker_confidence(embs, labels):
    n = len(labels)
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

    # Find all shard manifest files
    manifests = sorted(
        [f for f in os.listdir(args.manifest_dir) if f.endswith(".jsonl")],
        key=lambda f: int(re.search(r"_(\d+)\.", f).group(1)) if re.search(r"_(\d+)\.", f) else 0,
    )

    total_utt = 0
    total_clusters = 0

    for mf in manifests:
        shard_match = re.search(r"_(\d+)\.", mf)
        if not shard_match:
            continue
        shard_id = shard_match.group(1)

        emb_path = os.path.join(args.embedding_dir, f"embeddings_{shard_id}.npz")
        if not os.path.isfile(emb_path):
            print(f"  Shard {shard_id}: no embedding file, skipping")
            continue

        # Read manifest
        manifest_path = os.path.join(args.manifest_dir, mf)
        with open(manifest_path) as f:
            rows = [json.loads(l) for l in f if l.strip()]

        # Load embeddings
        data = np.load(emb_path, allow_pickle=True)
        cut_ids = data["cut_ids"]
        embs = data["embeddings"].astype(np.float32)

        # Build mapping: audio_filepath -> embedding index
        id_to_idx = {str(cid): i for i, cid in enumerate(cut_ids)}

        # Center embeddings (batch mean subtraction)
        embs = embs - embs.mean(axis=0, keepdims=True)

        # Cluster
        labels = cluster_embeddings(embs, args.threshold)
        scores = speaker_confidence(embs, labels)

        n_clusters = len(set(labels))
        total_utt += len(rows)
        total_clusters += n_clusters

        # Annotate manifest
        for row in rows:
            afp = row.get(args.audio_filepath_key, "")
            idx = id_to_idx.get(afp)
            if idx is not None:
                row["speaker_label"] = int(labels[idx])
                row["confidence_score"] = round(float(scores[idx]), 6)
            else:
                row["speaker_label"] = -1
                row["confidence_score"] = 0.0

        out_path = os.path.join(args.output_dir, mf)
        with open(out_path, "w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        print(f"  Shard {shard_id}: {len(rows)} utt -> {n_clusters} speakers -> {out_path}")

    print(f"\nDone. {total_utt} utterances, {total_clusters} total clusters across {len(manifests)} shards.")


if __name__ == "__main__":
    main()
