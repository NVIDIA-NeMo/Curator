"""Agglomerative Hierarchical Clustering (AHC) for speaker embeddings.

Threshold constants are derived from SimAM_ResNet100 (voxblink2_samresnet100_ft)
evaluated on VoxCeleb1-O cleaned trials (37,611 pairs):

  EER = 0.2287%  =>  cosine-similarity threshold = 0.3483
  FPR = 0.05%    =>  cosine-similarity threshold ~ 0.40

For clustering, false merges (different speakers collapsed) are far more
damaging than missed merges (same speaker split into two clusters), so the
default threshold is set stricter than EER.
"""

import logging
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

logger = logging.getLogger(__name__)

# Cosine-similarity thresholds for SimAM_ResNet100 on VoxCeleb1-O cleaned.
# Pairs scoring above the threshold are considered same-speaker.
EER_THRESHOLD = 0.3483
DEFAULT_THRESHOLD = 0.40


def _l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalize rows, avoiding division by zero."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return embeddings / norms


def _cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Return a symmetric (N, N) cosine-similarity matrix."""
    normed = _l2_normalize(embeddings)
    sim = normed @ normed.T
    np.clip(sim, -1.0, 1.0, out=sim)
    return sim


def _cosine_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Return a symmetric (N, N) cosine-distance matrix: 1 - cos_sim."""
    return 1.0 - _cosine_similarity_matrix(embeddings)


def cluster_embeddings(
    embeddings: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
    linkage_method: str = "average",
) -> np.ndarray:
    """Cluster speaker embeddings via AHC with a cosine-similarity threshold.

    Args:
        embeddings: (N, D) float array of L2-normalised (or raw) speaker
            embeddings.
        threshold: Cosine-similarity cutoff.  Pairs with similarity >= threshold
            are considered same-speaker.  Converted internally to a distance
            cutoff of ``1 - threshold`` for scipy.
        linkage_method: Linkage criterion passed to
            ``scipy.cluster.hierarchy.linkage``.  One of
            ``"average"`` (default), ``"complete"``, ``"single"``.

    Returns:
        labels: (N,) int array of 1-based cluster IDs.
    """
    n = embeddings.shape[0]
    if n <= 1:
        return np.ones(n, dtype=int)

    dist_mat = _cosine_distance_matrix(embeddings)
    condensed = squareform(dist_mat, checks=False)

    Z = linkage(condensed, method=linkage_method)
    distance_cutoff = 1.0 - threshold
    labels = fcluster(Z, t=distance_cutoff, criterion="distance")
    return labels


def cluster_stats(labels: np.ndarray) -> Dict:
    """Return a summary dict for a set of cluster labels."""
    counts = Counter(labels.tolist())
    sizes = sorted(counts.values(), reverse=True)
    return {
        "num_clusters": len(counts),
        "largest_cluster": sizes[0],
        "smallest_cluster": sizes[-1],
        "median_cluster": int(np.median(sizes)),
        "singletons": sum(1 for s in sizes if s == 1),
        "size_distribution": sizes,
    }


def print_cluster_summary(
    labels: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
) -> None:
    """Log a human-readable clustering summary."""
    stats = cluster_stats(labels)
    n = len(labels)
    print(f"\n{'='*50}")
    print(f"  AHC Clustering Results  (threshold={threshold:.4f})")
    print(f"{'='*50}")
    print(f"  Utterances        : {n:,}")
    print(f"  Speakers found    : {stats['num_clusters']:,}")
    print(f"  Largest cluster   : {stats['largest_cluster']:,} utts")
    print(f"  Smallest cluster  : {stats['smallest_cluster']:,} utts")
    print(f"  Median cluster    : {stats['median_cluster']:,} utts")
    print(f"  Singletons        : {stats['singletons']:,}")

    dist = stats["size_distribution"]
    top = min(20, len(dist))
    print(f"\n  Top-{top} cluster sizes: {dist[:top]}")
    print(f"{'='*50}\n")


# ---------------------------------------------------------------------------
# Cluster quality & per-utterance speaker-ID confidence
# ---------------------------------------------------------------------------

def cluster_quality(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> Dict:
    """Compute within-cluster and inter-cluster cosine-similarity statistics.

    Singletons (clusters with 1 sample) are excluded from both metrics.

    Returns a dict with keys:
        within_cluster_sims: per-cluster mean within-cluster cosine sim (array)
        within_stats: {mean, std, min, median, max, p25, p75}
        inter_cluster_sims: all pairwise centroid-to-centroid cosine sims (array)
        inter_stats: {mean, std, min, median, max, p25, p75}
        n_multi_clusters: number of clusters with >= 2 samples
        n_singletons: number of singleton clusters
    """
    sim_mat = _cosine_similarity_matrix(embeddings)

    cluster_indices = defaultdict(list)
    for i, lab in enumerate(labels):
        cluster_indices[lab].append(i)

    multi = {k: v for k, v in cluster_indices.items() if len(v) >= 2}
    n_singletons = len(cluster_indices) - len(multi)

    # Within-cluster avg cosine sim per cluster
    within_sims = []
    for idxs in multi.values():
        sub = sim_mat[np.ix_(idxs, idxs)]
        n = len(idxs)
        triu = np.triu_indices(n, k=1)
        within_sims.append(sub[triu].mean())
    within_sims = np.array(within_sims)

    # Centroids for multi-sample clusters
    normed = _l2_normalize(embeddings)
    cids = sorted(multi.keys())
    centroids = np.empty((len(cids), embeddings.shape[1]), dtype=np.float32)
    for j, cid in enumerate(cids):
        c = normed[multi[cid]].mean(axis=0)
        centroids[j] = c / max(np.linalg.norm(c), 1e-8)

    inter_sim = centroids @ centroids.T
    np.clip(inter_sim, -1.0, 1.0, out=inter_sim)
    triu = np.triu_indices(len(cids), k=1)
    inter_pairwise = inter_sim[triu]

    def _stats(arr):
        if len(arr) == 0:
            return {}
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "p25": float(np.percentile(arr, 25)),
            "median": float(np.median(arr)),
            "p75": float(np.percentile(arr, 75)),
            "max": float(arr.max()),
        }

    return {
        "within_cluster_sims": within_sims,
        "within_stats": _stats(within_sims),
        "inter_cluster_sims": inter_pairwise,
        "inter_stats": _stats(inter_pairwise),
        "n_multi_clusters": len(multi),
        "n_singletons": n_singletons,
    }


def speaker_confidence(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Compute a per-utterance speaker-ID confidence score in [0, 1].

    Uses a silhouette-style metric in cosine-similarity space:

      cohesion  (a) = mean cosine sim to other members of own cluster
      separation(b) = max over other clusters of (mean cosine sim to that cluster)
      confidence    = clamp((a - b) / max(a, b), 0, 1)

    Singletons get confidence = 0.0 (no within-cluster evidence).

    Returns:
        scores: (N,) float32 array of confidence scores.
    """
    n = len(labels)
    sim_mat = _cosine_similarity_matrix(embeddings)

    cluster_indices = defaultdict(list)
    for i, lab in enumerate(labels):
        cluster_indices[lab].append(i)

    # Build a (N, K) matrix: mean similarity of each utterance to each cluster.
    # cluster_id_list[k] -> original cluster label;  member_mask[k] -> indices.
    unique_labels = sorted(cluster_indices.keys())
    label_to_k = {lab: k for k, lab in enumerate(unique_labels)}
    K = len(unique_labels)

    # Mean sim to each cluster via matrix multiply with membership indicator
    membership = np.zeros((n, K), dtype=np.float32)
    cluster_sizes = np.zeros(K, dtype=np.float32)
    for lab, idxs in cluster_indices.items():
        k = label_to_k[lab]
        membership[idxs, k] = 1.0
        cluster_sizes[k] = len(idxs)

    # mean_sim[i, k] = mean cosine sim of utterance i to cluster k
    mean_sim = (sim_mat @ membership) / np.maximum(cluster_sizes, 1.0)

    scores = np.zeros(n, dtype=np.float32)

    for i in range(n):
        my_k = label_to_k[labels[i]]
        my_size = cluster_sizes[my_k]

        if my_size < 2:
            continue

        # Cohesion: correct for self-similarity (remove sim[i,i]=1 from the sum)
        cohesion = (mean_sim[i, my_k] * my_size - 1.0) / (my_size - 1.0)

        # Separation: best rival = max mean sim to any other cluster
        rival_sims = mean_sim[i].copy()
        rival_sims[my_k] = -2.0
        best_rival = rival_sims.max()

        if best_rival <= -2.0:
            scores[i] = 1.0
            continue

        denom = max(cohesion, best_rival)
        if denom <= 0:
            scores[i] = 0.0
        else:
            raw = (cohesion - best_rival) / denom
            scores[i] = max(0.0, min(1.0, raw))

    return scores


def print_quality_summary(
    embeddings: np.ndarray,
    labels: np.ndarray,
    confidence_scores: np.ndarray,
) -> None:
    """Print a comprehensive cluster quality + confidence report."""
    qual = cluster_quality(embeddings, labels)
    ws = qual["within_stats"]
    ics = qual["inter_stats"]

    print(f"\n{'='*60}")
    print(f"  Cluster Quality Report")
    print(f"{'='*60}")
    print(f"  Multi-sample clusters : {qual['n_multi_clusters']:,}")
    print(f"  Singletons (excluded) : {qual['n_singletons']:,}")

    if ws:
        print(f"\n  Within-Cluster Avg Cosine Similarity (per cluster):")
        print(f"    mean={ws['mean']:.4f}  std={ws['std']:.4f}  "
              f"min={ws['min']:.4f}  median={ws['median']:.4f}  max={ws['max']:.4f}")
        print(f"    [p25={ws['p25']:.4f}  p75={ws['p75']:.4f}]")

    if ics:
        print(f"\n  Inter-Cluster Centroid Cosine Similarity:")
        print(f"    mean={ics['mean']:.4f}  std={ics['std']:.4f}  "
              f"min={ics['min']:.4f}  median={ics['median']:.4f}  max={ics['max']:.4f}")
        print(f"    [p25={ics['p25']:.4f}  p75={ics['p75']:.4f}]")

    # Confidence score stats (only non-singleton utterances)
    non_singleton = confidence_scores[confidence_scores > 0]
    print(f"\n  Speaker-ID Confidence (per utterance, singletons excluded):")
    print(f"    count={len(non_singleton):,}")
    if len(non_singleton) > 0:
        print(f"    mean={non_singleton.mean():.4f}  std={non_singleton.std():.4f}  "
              f"min={non_singleton.min():.4f}  median={np.median(non_singleton):.4f}  "
              f"max={non_singleton.max():.4f}")
        print(f"    [p25={np.percentile(non_singleton, 25):.4f}  "
              f"p75={np.percentile(non_singleton, 75):.4f}]")

        # Distribution buckets
        buckets = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
        print(f"\n    Confidence distribution:")
        for lo, hi in buckets:
            ct = ((non_singleton >= lo) & (non_singleton < hi)).sum()
            pct = ct / len(non_singleton) * 100
            label = f"[{lo:.1f}, {hi:.1f})" if hi < 1.01 else f"[{lo:.1f}, 1.0]"
            print(f"      {label:12s}: {ct:6,} ({pct:5.1f}%)")

    total_singletons = (confidence_scores == 0).sum()
    print(f"\n    Singletons (confidence=0): {total_singletons:,}")
    print(f"{'='*60}\n")
