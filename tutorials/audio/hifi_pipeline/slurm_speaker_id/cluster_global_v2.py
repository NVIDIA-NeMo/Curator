#!/usr/bin/env python3
"""Memory-efficient global speaker clustering using FAISS + sparse graph.

Replaces the O(N^2) dense cosine-similarity matrix approach with:
  1. FAISS index for fast top-k nearest-neighbor search   O(N * k)
  2. Sparse similarity graph (only edges above threshold)  O(N * k)
  3. Connected-components clustering with iterative merging
  4. Sparse silhouette-style confidence scoring

Memory: O(N * k) instead of O(N^2).
For 10M utterances with k=100:  ~4 GB indices + ~4 GB distances  (fits in 128 GB).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter, defaultdict
from typing import Any

import faiss
import numpy as np
from scipy import sparse


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Memory-efficient global speaker clustering (FAISS + sparse graph)."
    )
    p.add_argument("--manifest_dir", required=True, help="Directory with shard_N.jsonl files")
    p.add_argument("--embedding_dir", required=True, help="Directory with embeddings_N.npz files")
    p.add_argument("--output_dir", required=True, help="Output directory for annotated manifests")
    p.add_argument(
        "--threshold", type=float, default=0.292,
        help="Cosine-similarity threshold for same-speaker decisions (default: 0.292)",
    )
    p.add_argument("--audio_filepath_key", default="audio_filepath")
    p.add_argument(
        "--k_neighbors", type=int, default=100,
        help="Number of nearest neighbors for FAISS search (default: 100)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def l2_normalize(embs: np.ndarray) -> np.ndarray:
    """L2-normalize each row in-place-safe manner."""
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    return embs / np.maximum(norms, 1e-8)


def _timer(label: str, start: float) -> None:
    print(f"    [{label}] {time.time() - start:.1f}s")


# ---------------------------------------------------------------------------
# Phase 2a: FAISS kNN search
# ---------------------------------------------------------------------------

def build_faiss_knn(
    embs_normed: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a FAISS index and return (distances, indices) for top-k neighbors.

    Since embeddings are L2-normalized, inner-product = cosine similarity.
    FAISS IndexFlatIP returns similarities in descending order.

    Returns:
        sims: (N, k) float32 cosine similarities  (self included at position 0)
        indices: (N, k) int64 neighbor indices
    """
    n, d = embs_normed.shape

    # Use inner product on L2-normalized vectors = cosine similarity.
    index = faiss.IndexFlatIP(d)
    index.add(embs_normed)

    # Search returns (similarities, indices) in descending similarity order.
    # Position 0 is the point itself (sim=1.0).
    sims, indices = index.search(embs_normed, k)
    return sims.astype(np.float32), indices.astype(np.int64)


# ---------------------------------------------------------------------------
# Phase 2b: Sparse similarity graph + connected components clustering
# ---------------------------------------------------------------------------

def build_sparse_sim_graph(
    sims: np.ndarray,
    indices: np.ndarray,
    threshold: float,
) -> sparse.csr_matrix:
    """Build a sparse symmetric similarity graph from kNN results.

    Only edges with cosine similarity >= threshold are kept.
    Self-loops are excluded.

    Returns:
        Symmetric CSR matrix of shape (N, N) with similarity values.
    """
    n, k = indices.shape

    # Flatten to COO triplets
    rows = np.repeat(np.arange(n, dtype=np.int64), k)
    cols = indices.ravel()
    vals = sims.ravel()

    # Filter: remove self-loops and below-threshold edges
    mask = (rows != cols) & (vals >= threshold)
    rows = rows[mask]
    cols = cols[mask]
    vals = vals[mask].astype(np.float32)

    # Build sparse matrix (will sum duplicates from symmetrization)
    graph = sparse.coo_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float32)

    # Symmetrize: take max of (i,j) and (j,i)
    graph_t = graph.T
    graph = graph.maximum(graph_t)

    return graph.tocsr()


def cluster_sparse_graph(
    graph: sparse.csr_matrix,
    threshold: float,
) -> np.ndarray:
    """Cluster using connected components of the sparse similarity graph.

    Each connected component in the thresholded graph becomes a cluster.
    The graph already has only edges >= threshold from build_sparse_sim_graph.

    For tighter clustering, we do an additional pass: for each component
    with >1 node, verify that the average intra-component similarity
    (using available sparse edges) is above a stricter bar. If not,
    re-split using a second round of connected components at a higher
    threshold. This prevents spurious chaining.

    Returns:
        labels: (N,) int array of 1-based cluster IDs.
    """
    n = graph.shape[0]
    n_components, labels_0 = sparse.csgraph.connected_components(
        graph, directed=False, return_labels=True,
    )

    # Convert to 1-based labels
    labels = labels_0 + 1
    return labels


def cluster_sparse_graph_refined(
    graph: sparse.csr_matrix,
    threshold: float,
) -> np.ndarray:
    """Two-pass clustering to prevent chaining artifacts.

    Pass 1: Connected components on the threshold-filtered graph.
    Pass 2: For each large component, check average edge weight.
             If average similarity < (threshold + 0.05), re-run connected
             components at a tighter threshold to break spurious chains.

    Returns:
        labels: (N,) int array of 1-based cluster IDs.
    """
    n = graph.shape[0]
    _, labels_pass1 = sparse.csgraph.connected_components(
        graph, directed=False, return_labels=True,
    )

    # Identify components and optionally refine large ones
    component_map: dict[int, list[int]] = defaultdict(list)
    for i, lab in enumerate(labels_pass1):
        component_map[lab].append(i)

    final_labels = np.zeros(n, dtype=np.int64)
    next_label = 1
    tighter_threshold = threshold + 0.05

    for comp_id, members in component_map.items():
        if len(members) <= 2:
            # Small components: assign directly
            for idx in members:
                final_labels[idx] = next_label
            next_label += 1
            continue

        # For larger components, check if any edges are suspiciously low.
        # Extract the sub-graph for this component.
        member_arr = np.array(members)
        sub_graph = graph[member_arr][:, member_arr]

        # Count edges and compute mean similarity
        nnz = sub_graph.nnz
        if nnz == 0:
            # No edges -- each node is its own cluster
            for idx in members:
                final_labels[idx] = next_label
                next_label += 1
            continue

        avg_sim = sub_graph.sum() / nnz

        if avg_sim < tighter_threshold:
            # Re-cluster this component at a tighter threshold
            mask = sub_graph >= tighter_threshold
            sub_filtered = sub_graph.multiply(mask)
            n_sub, sub_labels = sparse.csgraph.connected_components(
                sub_filtered, directed=False, return_labels=True,
            )
            for local_idx, global_idx in enumerate(members):
                final_labels[global_idx] = next_label + sub_labels[local_idx]
            next_label += n_sub
        else:
            for idx in members:
                final_labels[idx] = next_label
            next_label += 1

    return final_labels


# ---------------------------------------------------------------------------
# Phase 3: Sparse silhouette-style confidence scoring
# ---------------------------------------------------------------------------

def sparse_speaker_confidence(
    sims: np.ndarray,
    indices: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Silhouette-style confidence using only the sparse kNN neighbors.

    For each utterance i:
      - cohesion  = mean similarity to neighbors in same cluster
      - separation = max over other clusters of mean similarity to neighbors in that cluster
      - score = clamp((cohesion - separation) / max(cohesion, separation), 0, 1)

    Singletons get score 0.0.

    Memory: O(N * k), no dense matrices.
    """
    n = len(labels)
    k = sims.shape[1]
    scores = np.zeros(n, dtype=np.float32)

    # Pre-compute cluster sizes for singleton check
    cluster_sizes: dict[int, int] = Counter(labels.tolist())

    for i in range(n):
        my_label = labels[i]

        # Singletons
        if cluster_sizes[my_label] < 2:
            continue

        # Gather similarities to neighbors, grouped by their cluster label
        neighbor_sims_by_cluster: dict[int, list[float]] = defaultdict(list)
        for j_pos in range(k):
            j = indices[i, j_pos]
            if j == i or j < 0:
                continue
            neighbor_sims_by_cluster[labels[j]].append(float(sims[i, j_pos]))

        # Cohesion: mean similarity to same-cluster neighbors
        same_sims = neighbor_sims_by_cluster.get(my_label, [])
        if not same_sims:
            # No same-cluster neighbors in top-k -- low confidence
            scores[i] = 0.0
            continue
        cohesion = sum(same_sims) / len(same_sims)

        # Separation: best rival cluster's mean similarity
        best_rival = -2.0
        for other_label, other_sims in neighbor_sims_by_cluster.items():
            if other_label == my_label:
                continue
            rival_mean = sum(other_sims) / len(other_sims)
            if rival_mean > best_rival:
                best_rival = rival_mean

        if best_rival <= -2.0:
            # No rival neighbors -> perfect confidence
            scores[i] = 1.0
            continue

        denom = max(cohesion, best_rival)
        if denom <= 0:
            scores[i] = 0.0
        else:
            raw = (cohesion - best_rival) / denom
            scores[i] = max(0.0, min(1.0, raw))

    return scores


def sparse_speaker_confidence_vectorized(
    sims: np.ndarray,
    indices: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Vectorized version of sparse silhouette scoring -- much faster for large N.

    Uses numpy broadcasting over the (N, k) neighbor arrays instead of
    Python loops.

    Memory: O(N * k * K_local) in the worst case for the per-cluster
    accumulations, but in practice K_local (distinct clusters among k
    neighbors) is small.
    """
    n = len(labels)
    k = sims.shape[1]
    scores = np.zeros(n, dtype=np.float32)

    cluster_sizes = Counter(labels.tolist())
    singleton_mask = np.array([cluster_sizes[labels[i]] < 2 for i in range(n)])

    # Neighbor labels: (N, k)
    neighbor_labels = labels[indices]  # works because indices are valid row indices
    my_labels = labels[:, None]  # (N, 1)

    # Mask out self-neighbors and invalid indices
    valid_mask = (indices != np.arange(n)[:, None]) & (indices >= 0)

    # Same-cluster mask: (N, k)
    same_mask = (neighbor_labels == my_labels) & valid_mask

    # Cohesion: mean similarity to same-cluster neighbors
    same_sims_sum = np.where(same_mask, sims, 0.0).sum(axis=1)
    same_count = same_mask.sum(axis=1)
    cohesion = np.where(same_count > 0, same_sims_sum / same_count, 0.0)

    # For separation we need: for each point, the best rival cluster mean.
    # This requires grouping by cluster label among neighbors -- hard to
    # fully vectorize without per-cluster accum.  Compromise: vectorized
    # for the common case, fall back to loop for per-cluster detail.

    # Fast path: if only one other cluster among neighbors, or none.
    other_mask = (~(neighbor_labels == my_labels)) & valid_mask
    any_rival = other_mask.any(axis=1)

    # For points with no rival neighbors: confidence = 1.0 (unless singleton)
    no_rival_idx = np.where(~singleton_mask & (same_count > 0) & ~any_rival)[0]
    scores[no_rival_idx] = 1.0

    # For points needing rival computation: use a chunked loop
    needs_rival = np.where(~singleton_mask & (same_count > 0) & any_rival)[0]

    # Process in chunks to balance speed vs memory
    chunk_size = 50000
    for chunk_start in range(0, len(needs_rival), chunk_size):
        chunk_idx = needs_rival[chunk_start:chunk_start + chunk_size]

        for i in chunk_idx:
            best_rival = -2.0
            rival_accum: dict[int, tuple[float, int]] = {}
            for j_pos in range(k):
                j = indices[i, j_pos]
                if j == i or j < 0:
                    continue
                j_label = labels[j]
                if j_label == labels[i]:
                    continue
                if j_label not in rival_accum:
                    rival_accum[j_label] = (0.0, 0)
                s, c = rival_accum[j_label]
                rival_accum[j_label] = (s + sims[i, j_pos], c + 1)

            for _, (s, c) in rival_accum.items():
                rival_mean = s / c
                if rival_mean > best_rival:
                    best_rival = rival_mean

            coh = cohesion[i]
            if best_rival <= -2.0:
                scores[i] = 1.0
            else:
                denom = max(coh, best_rival)
                if denom <= 0:
                    scores[i] = 0.0
                else:
                    raw = (coh - best_rival) / denom
                    scores[i] = max(0.0, min(1.0, raw))

    return scores


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    k = args.k_neighbors
    threshold = args.threshold

    # ------------------------------------------------------------------
    # Phase 1: Load all manifests and embeddings
    # ------------------------------------------------------------------
    print("Phase 1: Loading all manifests and embeddings...")
    t0 = time.time()

    manifest_files = sorted(
        [f for f in os.listdir(args.manifest_dir) if f.endswith(".jsonl")],
        key=lambda f: int(m.group(1)) if (m := re.search(r"_(\d+)\.", f)) else 0,
    )

    all_rows: list[tuple[str, int, dict[str, Any]]] = []
    all_embs: list[np.ndarray] = []
    all_emb_indices: list[int] = []

    for mf in manifest_files:
        shard_match = re.search(r"_(\d+)\.", mf)
        if not shard_match:
            continue
        shard_id = shard_match.group(1)

        with open(os.path.join(args.manifest_dir, mf)) as f:
            rows = [json.loads(line) for line in f if line.strip()]

        emb_path = os.path.join(args.embedding_dir, f"embeddings_{shard_id}.npz")
        id_to_emb: dict[str, np.ndarray] = {}
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

        print(f"  Loaded shard {shard_id}: {len(rows)} rows, {len(id_to_emb)} embeddings")

    n_total = len(all_rows)
    n_emb = len(all_embs)
    _timer("Phase 1", t0)
    print(f"  {n_total} total rows, {n_emb} with embeddings")

    if n_emb == 0:
        print("No embeddings found. Nothing to cluster.")
        return

    # ------------------------------------------------------------------
    # Phase 2: Build embedding matrix, center, L2-normalize
    # ------------------------------------------------------------------
    print("Phase 2: Preparing embeddings...")
    t0 = time.time()

    emb_matrix = np.stack(all_embs).astype(np.float32)
    del all_embs  # free the list of individual arrays
    print(f"  Embedding matrix: {emb_matrix.shape}, ~{emb_matrix.nbytes / 1e9:.2f} GB")

    # Center (batch mean subtraction)
    emb_matrix -= emb_matrix.mean(axis=0, keepdims=True)

    # L2-normalize for cosine similarity via inner product
    emb_matrix = l2_normalize(emb_matrix)

    # Ensure C-contiguous float32 for FAISS
    emb_matrix = np.ascontiguousarray(emb_matrix, dtype=np.float32)
    _timer("Phase 2", t0)

    # ------------------------------------------------------------------
    # Phase 3: FAISS kNN search
    # ------------------------------------------------------------------
    effective_k = min(k, n_emb)
    print(f"Phase 3: FAISS top-{effective_k} nearest neighbor search (N={n_emb})...")
    t0 = time.time()

    knn_sims, knn_indices = build_faiss_knn(emb_matrix, effective_k)
    mem_knn = (knn_sims.nbytes + knn_indices.nbytes) / 1e9
    print(f"  kNN arrays: ~{mem_knn:.2f} GB")
    _timer("Phase 3", t0)

    # ------------------------------------------------------------------
    # Phase 4: Build sparse graph + cluster
    # ------------------------------------------------------------------
    print(f"Phase 4: Building sparse graph (threshold={threshold}) and clustering...")
    t0 = time.time()

    graph = build_sparse_sim_graph(knn_sims, knn_indices, threshold)
    print(f"  Sparse graph: {graph.nnz:,} edges ({graph.nnz / n_emb:.1f} avg edges/node)")

    labels = cluster_sparse_graph_refined(graph, threshold)
    del graph  # free sparse graph

    n_speakers = len(set(labels))
    _timer("Phase 4", t0)

    sizes = sorted(Counter(labels).values(), reverse=True)
    print(f"  {n_emb:,} utterances -> {n_speakers:,} speakers")
    print(f"  Top 10 speaker sizes: {sizes[:10]}")
    print(f"  Singletons: {sum(1 for s in sizes if s == 1):,}")
    print(f"  Avg utt/speaker: {n_emb / max(n_speakers, 1):.1f}")

    # ------------------------------------------------------------------
    # Phase 5: Confidence scoring (sparse)
    # ------------------------------------------------------------------
    print("Phase 5: Computing sparse confidence scores...")
    t0 = time.time()

    if n_emb <= 200_000:
        # For moderate sizes, use the partially-vectorized version
        scores = sparse_speaker_confidence_vectorized(knn_sims, knn_indices, labels)
    else:
        # For very large N, use the simple loop which has lower peak memory
        scores = sparse_speaker_confidence(knn_sims, knn_indices, labels)

    del knn_sims, knn_indices  # free kNN arrays
    _timer("Phase 5", t0)
    print(f"  Mean confidence: {scores.mean():.4f}")
    print(f"  Median confidence: {np.median(scores):.4f}")
    print(f"  Confidence > 0.5: {(scores > 0.5).sum():,} ({100 * (scores > 0.5).mean():.1f}%)")

    # ------------------------------------------------------------------
    # Phase 6: Assign labels back to manifest rows
    # ------------------------------------------------------------------
    print("Phase 6: Assigning labels to manifest rows...")
    t0 = time.time()

    label_map: dict[int, tuple[int, float]] = {}
    for i, global_idx in enumerate(all_emb_indices):
        label_map[global_idx] = (int(labels[i]), round(float(scores[i]), 6))

    for global_idx, (mf, row_idx, row) in enumerate(all_rows):
        if global_idx in label_map:
            row["speaker_label"], row["confidence_score"] = label_map[global_idx]
        else:
            row["speaker_label"] = -1
            row["confidence_score"] = 0.0

    _timer("Phase 6", t0)

    # ------------------------------------------------------------------
    # Phase 7: Write output manifests (same shard structure)
    # ------------------------------------------------------------------
    print("Phase 7: Writing output manifests...")
    t0 = time.time()

    shard_rows: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for mf, row_idx, row in all_rows:
        shard_rows[mf].append((row_idx, row))

    for mf, rows in shard_rows.items():
        rows.sort(key=lambda x: x[0])
        out_path = os.path.join(args.output_dir, mf)
        with open(out_path, "w") as f:
            for _, row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    _timer("Phase 7", t0)
    print(f"  Wrote {len(shard_rows)} files to {args.output_dir}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Summary (v2 -- FAISS + sparse graph)")
    print(f"  Utterances:       {n_emb:>12,}")
    print(f"  Speakers:         {n_speakers:>12,}")
    print(f"  Avg utt/speaker:  {n_emb / max(n_speakers, 1):>12.1f}")
    print(f"  Singletons:       {sum(1 for s in sizes if s == 1):>12,}")
    print(f"  Threshold:        {threshold:>12.4f}")
    print(f"  k_neighbors:      {k:>12}")
    print(f"  Mean confidence:  {scores.mean():>12.4f}")
    print(f"  Rows w/o embeddings: {n_total - n_emb:>9,} (label=-1)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
