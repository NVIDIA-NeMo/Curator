"""Large-scale speaker clustering for datasets that do not fit in RAM.

This module is the **memory-bounded** counterpart to ``ahc.py``.  Use it when
the embedding count is too large for a full ``N x N`` cosine-similarity matrix
to fit in RAM (roughly ``N > 150,000`` on a 256 GB box, or any dataset with
**more than 500 hours** of audio).

Pipeline
--------

1. **BIRCH (stage 1, streaming)** — ``sklearn.cluster.Birch`` is fed the
   L2-normalised embeddings in mini-batches via ``partial_fit``.  Each leaf
   subcluster ends up tightly grouping near-duplicate utterances of the same
   speaker.  Memory is independent of ``N`` — bounded by the CF-tree size.

2. **Leaf assignment** — each utterance is assigned to its nearest leaf
   centroid in tiled batches (avoids the ``(N, n_subclusters)`` blow-up that
   ``Birch.predict`` would cause).

3. **AHC on centroids (stage 2)** — the leaf centroids are re-L2-normalised
   and clustered with the same SciPy AHC + cosine-similarity threshold used
   in :mod:`ahc`.  Because there are typically only ``10k - 150k`` leaves, the
   full distance matrix at this stage easily fits in RAM.

4. **Back-propagation** — utterance labels are assigned by looking up
   ``centroid_label[leaf_idx]``.

5. **``min_cluster_size`` filter** — clusters smaller than the configured
   threshold are dropped (utterances get ``cluster_id = -1``).  This is the
   purity-first filter for downstream consumers that prefer dropping
   long-tail / noise utterances over keeping them.

6. **Silhouette-based confidence** — per-utterance ``confidence_score`` in
   ``[0, 1]`` defined as ``(a - b) / max(a, b)`` where ``a`` is the cosine
   similarity of the (L2-normalised) embedding to its **own** speaker
   centroid and ``b`` is the maximum cosine similarity to **any other**
   speaker centroid.  Computed against the ``K`` surviving speaker
   centroids (not the full ``N x N`` matrix), so it scales to tens of
   millions of utterances.  Singletons and dropped utterances get ``0.0``.

Threshold semantics
-------------------

* ``threshold`` (cosine) keeps the **same meaning** as in :mod:`ahc`.  It is
  applied at the centroid AHC step, so a value tuned on small data
  (e.g. 0.40 for ``ResNet293_LM``) transfers as-is.
* ``birch_threshold`` (Euclidean on L2-normalised vectors) controls how
  tight each BIRCH leaf is.  Tighter leaves yield more (purer) centroids and
  more accurate downstream AHC, at the cost of memory at stage 2.  The
  default value corresponds to a per-leaf cosine similarity of ``~0.8``
  (``birch_thr = sqrt(2 * (1 - 0.8))``), which is well above the speaker
  decision threshold and so does not pre-merge distinct speakers.

Recommended for:
  * Datasets larger than ``500 hours`` of audio.
  * Single-language clustering runs over hundreds of thousands to tens of
    millions of utterances.
"""

import logging
from collections import Counter, defaultdict
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Recommendation thresholds
# ---------------------------------------------------------------------------

# Use the large-scale path when the dataset is larger than this many hours.
LARGE_SCALE_HOURS_THRESHOLD = 500.0

# Default minimum cluster size for the purity-first filter.  Clusters with
# fewer than this many utterances are dropped (label -> -1).  Tuned for
# downstream training-data curation where the long tail is not useful.
DEFAULT_MIN_CLUSTER_SIZE = 30

# Default speaker-decision threshold (cosine), inherited from ``ahc.py``.
# Override when calling for a different embedding model.
DEFAULT_THRESHOLD = 0.40

# Default BIRCH leaf radius (Euclidean on L2-normalised vectors).
# Corresponds to a per-leaf cosine similarity floor of ~0.8.
#   euclidean_thr = sqrt(2 * (1 - cos_thr))
DEFAULT_BIRCH_THRESHOLD = float(np.sqrt(2.0 * (1.0 - 0.8)))  # ~0.6325

# BIRCH partial_fit batch size.  Trade-off: larger -> fewer Python overhead
# round-trips, smaller -> lower peak RAM during a single update.
DEFAULT_BIRCH_PARTIAL_FIT_BATCH = 50_000

# Tile size for the utterance-to-leaf assignment step.  Picks the nearest
# centroid for ``ASSIGN_TILE`` utterances at a time.  Peak memory for the
# assignment step is ``~ ASSIGN_TILE * n_subclusters * 4`` bytes.
DEFAULT_ASSIGN_TILE = 16_384

# Drop label sentinel used for utterances that fall in a too-small cluster.
DROPPED_LABEL = -1


# ---------------------------------------------------------------------------
# Linear-algebra helpers (kept local so this module has no cross-import
# dependency on ``ahc``)
# ---------------------------------------------------------------------------

def _l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalise rows; rows with zero norm are left as zero vectors."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return embeddings / norms


def _cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Symmetric ``(M, M)`` cosine-similarity matrix on row-vectors."""
    normed = _l2_normalize(embeddings)
    sim = normed @ normed.T
    np.clip(sim, -1.0, 1.0, out=sim)
    return sim


def cosine_threshold_to_birch_radius(cos_threshold: float) -> float:
    """Convert a per-leaf cosine-similarity floor to a BIRCH Euclidean radius.

    On the unit sphere, ``||x - y||^2 = 2 * (1 - cos(x, y))``.  A leaf whose
    members are all within cosine ``>= cos_threshold`` of its centroid has a
    radius of at most ``sqrt(2 * (1 - cos_threshold))``.

    Args:
        cos_threshold: Per-leaf cosine-similarity floor in ``[-1, 1]``.

    Returns:
        Euclidean radius for ``sklearn.cluster.Birch.threshold``.
    """
    if cos_threshold > 1.0 or cos_threshold < -1.0:
        raise ValueError(f"cos_threshold must be in [-1, 1], got {cos_threshold}")
    return float(np.sqrt(2.0 * (1.0 - cos_threshold)))


# ---------------------------------------------------------------------------
# Recommendation helper
# ---------------------------------------------------------------------------

def recommend_clustering_method(
    num_hours: Optional[float] = None,
    num_utterances: Optional[int] = None,
    hours_threshold: float = LARGE_SCALE_HOURS_THRESHOLD,
    utterance_threshold: int = 150_000,
) -> str:
    """Recommend ``"standard"`` vs ``"large_scale"`` clustering for a dataset.

    Either ``num_hours`` or ``num_utterances`` may be provided (or both).
    The large-scale path is recommended whenever **either** axis is past its
    cutoff:

    * Audio duration above ``hours_threshold`` (default 500h).  This is the
      headline rule documented in the README — datasets larger than half a
      thousand hours almost always exceed the RAM budget of the standard
      ``N x N`` AHC.
    * Utterance count above ``utterance_threshold`` (default 150,000).
      Tracks the actual driver of OOM, which is ``N``, not duration.

    Args:
        num_hours: Total dataset duration in hours.  Optional.
        num_utterances: Total utterance count.  Optional.
        hours_threshold: Hours cutoff for switching to large-scale.
        utterance_threshold: Utterance-count cutoff for switching.

    Returns:
        Either ``"standard"`` (use :mod:`ahc`) or ``"large_scale"``
        (use this module).
    """
    if num_hours is None and num_utterances is None:
        raise ValueError("Provide num_hours, num_utterances, or both")

    if num_hours is not None and num_hours > hours_threshold:
        return "large_scale"
    if num_utterances is not None and num_utterances > utterance_threshold:
        return "large_scale"
    return "standard"


# ---------------------------------------------------------------------------
# Stage 1: BIRCH (streaming)
# ---------------------------------------------------------------------------

def _build_birch_tree(
    embeddings: np.ndarray,
    birch_threshold: float,
    branching_factor: int,
    partial_fit_batch: int,
):
    """Stream the embeddings into a fitted ``sklearn.cluster.Birch``.

    Embeddings are expected to be L2-normalised already (so that the BIRCH
    Euclidean threshold reflects a per-leaf cosine-similarity floor).
    """
    try:
        from sklearn.cluster import Birch
    except ImportError as exc:
        raise ImportError(
            "large_scale_clustering_and_scoring requires scikit-learn.  "
            "Install with `pip install scikit-learn>=1.3`."
        ) from exc

    birch = Birch(
        threshold=birch_threshold,
        branching_factor=branching_factor,
        n_clusters=None,  # we run our own AHC on the leaf centroids
        compute_labels=False,  # avoid the (N, n_subclusters) blow-up in fit()
        copy=False,
    )

    n = embeddings.shape[0]
    n_batches = (n + partial_fit_batch - 1) // partial_fit_batch
    for b in range(n_batches):
        start = b * partial_fit_batch
        end = min(start + partial_fit_batch, n)
        birch.partial_fit(embeddings[start:end])
        if (b + 1) % 10 == 0 or b == n_batches - 1:
            n_sub = len(birch.subcluster_centers_)
            logger.info(
                "  BIRCH: %d / %d batches, %d leaf subclusters so far",
                b + 1, n_batches, n_sub,
            )

    return birch


# ---------------------------------------------------------------------------
# Stage 2: utterance -> leaf assignment (tiled, no Birch.predict)
# ---------------------------------------------------------------------------

def _assign_to_nearest_leaf(
    normed_embeddings: np.ndarray,
    normed_centroids: np.ndarray,
    tile: int = DEFAULT_ASSIGN_TILE,
) -> np.ndarray:
    """For each utterance, return the index of the nearest leaf centroid.

    Both inputs must be L2-normalised.  Equivalent to ``argmax`` over cosine
    similarity, computed in tiles of ``tile`` utterances at a time so peak
    memory is ``O(tile * n_subclusters)`` instead of ``O(N * n_subclusters)``.
    """
    n = normed_embeddings.shape[0]
    leaf_idx = np.empty(n, dtype=np.int32)

    for start in range(0, n, tile):
        end = min(start + tile, n)
        sim_tile = normed_embeddings[start:end] @ normed_centroids.T
        leaf_idx[start:end] = sim_tile.argmax(axis=1)

    return leaf_idx


# ---------------------------------------------------------------------------
# Stage 3: AHC on leaf centroids
# ---------------------------------------------------------------------------

def _ahc_on_centroids(
    centroids: np.ndarray,
    threshold: float,
    linkage_method: str,
) -> np.ndarray:
    """Run cosine-distance AHC on leaf centroids.  Returns 1-based labels."""
    n = centroids.shape[0]
    if n <= 1:
        return np.ones(n, dtype=int)

    dist_mat = 1.0 - _cosine_similarity_matrix(centroids)
    condensed = squareform(dist_mat, checks=False)
    Z = linkage(condensed, method=linkage_method)
    distance_cutoff = 1.0 - threshold
    return fcluster(Z, t=distance_cutoff, criterion="distance")


# ---------------------------------------------------------------------------
# Stage 5: ``min_cluster_size`` filter
# ---------------------------------------------------------------------------

def filter_small_clusters(
    labels: np.ndarray,
    min_cluster_size: int,
    dropped_label: int = DROPPED_LABEL,
) -> Tuple[np.ndarray, Dict]:
    """Drop clusters with fewer than ``min_cluster_size`` utterances.

    Utterances in dropped clusters get label ``dropped_label`` (default -1).
    Surviving cluster labels are **kept as-is** (not renumbered) so callers
    can join back to the pre-filter cluster IDs if needed.

    Args:
        labels: ``(N,)`` int array of cluster IDs.
        min_cluster_size: Drop clusters strictly smaller than this.
        dropped_label: Sentinel value written into dropped utterance slots.

    Returns:
        ``(filtered_labels, stats)`` where stats summarises how much was
        dropped at the cluster and utterance level.
    """
    if min_cluster_size <= 1:
        return labels.copy(), {
            "min_cluster_size": min_cluster_size,
            "n_clusters_before": int(len(set(labels.tolist()))),
            "n_clusters_after": int(len(set(labels.tolist()))),
            "n_clusters_dropped": 0,
            "n_utts_before": int(len(labels)),
            "n_utts_kept": int(len(labels)),
            "n_utts_dropped": 0,
            "fraction_dropped": 0.0,
        }

    counts = Counter(labels.tolist())
    keep = {lab for lab, ct in counts.items() if ct >= min_cluster_size}

    filtered = np.where(np.isin(labels, list(keep)), labels, dropped_label)
    n_total = len(labels)
    n_kept = int((filtered != dropped_label).sum())
    n_dropped = n_total - n_kept

    stats = {
        "min_cluster_size": min_cluster_size,
        "n_clusters_before": len(counts),
        "n_clusters_after": len(keep),
        "n_clusters_dropped": len(counts) - len(keep),
        "n_utts_before": n_total,
        "n_utts_kept": n_kept,
        "n_utts_dropped": n_dropped,
        "fraction_dropped": (n_dropped / n_total) if n_total > 0 else 0.0,
    }
    return filtered, stats


# ---------------------------------------------------------------------------
# Stage 6: centroid-based per-utterance confidence
# ---------------------------------------------------------------------------

def _speaker_confidence_from_centroids(
    normed_embeddings: np.ndarray,
    labels: np.ndarray,
    dropped_label: int = DROPPED_LABEL,
    tile: int = DEFAULT_ASSIGN_TILE,
) -> np.ndarray:
    """Per-utterance confidence based on speaker centroids (memory-bounded).

    Replacement for :func:`ahc.speaker_confidence` that does **not** build the
    full ``N x N`` similarity matrix.  Instead:

    1. Compute one centroid per surviving speaker (mean of L2-normalised
       embeddings, then re-L2-normalise).
    2. For each utterance, compute cosine sim to all ``K`` speaker centroids
       in a single tile.  Take ``a = sim`` to own centroid and
       ``b = max sim`` to any other centroid.  Confidence
       ``= clamp((a - b) / max(a, b), 0, 1)``.

    Dropped utterances (``labels == dropped_label``) get confidence ``0.0``.
    Singleton speakers also get confidence ``0.0`` (no separation signal).
    """
    n = normed_embeddings.shape[0]
    scores = np.zeros(n, dtype=np.float32)

    # Group surviving utterances by speaker label.
    cluster_indices = defaultdict(list)
    for i, lab in enumerate(labels):
        if lab == dropped_label:
            continue
        cluster_indices[int(lab)].append(i)

    if not cluster_indices:
        return scores

    # Build (K, D) matrix of L2-normalised speaker centroids in a stable order.
    speaker_ids = sorted(cluster_indices.keys())
    label_to_k = {lab: k for k, lab in enumerate(speaker_ids)}
    K = len(speaker_ids)
    D = normed_embeddings.shape[1]

    centroids = np.empty((K, D), dtype=np.float32)
    cluster_sizes = np.empty(K, dtype=np.int64)
    for k, lab in enumerate(speaker_ids):
        idxs = cluster_indices[lab]
        c = normed_embeddings[idxs].mean(axis=0)
        nrm = max(np.linalg.norm(c), 1e-8)
        centroids[k] = (c / nrm).astype(np.float32, copy=False)
        cluster_sizes[k] = len(idxs)

    # Compute confidence in tiles of utterances.
    for start in range(0, n, tile):
        end = min(start + tile, n)
        sim_tile = normed_embeddings[start:end] @ centroids.T  # (B, K)
        np.clip(sim_tile, -1.0, 1.0, out=sim_tile)

        for local_i, global_i in enumerate(range(start, end)):
            lab = int(labels[global_i])
            if lab == dropped_label:
                continue
            k = label_to_k.get(lab)
            if k is None or cluster_sizes[k] < 2:
                continue

            a = float(sim_tile[local_i, k])
            row = sim_tile[local_i].copy()
            row[k] = -2.0
            b = float(row.max())

            denom = max(a, b)
            if denom <= 0.0:
                continue
            raw = (a - b) / denom
            scores[global_i] = float(max(0.0, min(1.0, raw)))

    return scores


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def cluster_embeddings_large_scale(
    embeddings: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
    linkage_method: str = "average",
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
    birch_threshold: float = DEFAULT_BIRCH_THRESHOLD,
    branching_factor: int = 50,
    partial_fit_batch: int = DEFAULT_BIRCH_PARTIAL_FIT_BATCH,
    assign_tile: int = DEFAULT_ASSIGN_TILE,
    compute_confidence: bool = True,
    dropped_label: int = DROPPED_LABEL,
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict]:
    """Cluster a large embedding set with BIRCH (stage 1) + AHC (stage 2).

    Memory peak:
        ``O(N * D + n_subclusters^2)`` instead of ``O(N^2)``.
        With ``N = 30,000,000`` and typical settings this stays well under
        100 GB on commodity hardware.

    Args:
        embeddings: ``(N, D)`` float array of speaker embeddings.  Will be
            L2-normalised internally; pre-normalisation is harmless.
        threshold: Cosine-similarity cutoff used at the AHC step.  Same
            meaning as in :mod:`ahc`.
        linkage_method: SciPy AHC linkage; one of
            ``"average"`` (default), ``"complete"``, ``"single"``.
        min_cluster_size: Drop clusters strictly smaller than this.  Dropped
            utterances get label ``dropped_label``.  Default: 30.  Set to
            ``1`` to disable filtering.
        birch_threshold: Euclidean radius for BIRCH leaves on the unit
            sphere.  Use :func:`cosine_threshold_to_birch_radius` to derive
            from a desired per-leaf cosine floor.
        branching_factor: BIRCH ``branching_factor``.  Default 50.
        partial_fit_batch: How many utterances to feed BIRCH per
            ``partial_fit`` call.  Tunes Python overhead vs peak RAM.
        assign_tile: Tile size for the utterance -> leaf assignment and the
            confidence computation.
        compute_confidence: If True, return per-utterance silhouette-style
            confidence scores in ``[0, 1]``.  Otherwise the second return
            value is ``None``.
        dropped_label: Sentinel label written for utterances dropped by the
            ``min_cluster_size`` filter.

    Returns:
        ``(labels, confidence, stats)`` where:

        * ``labels`` is ``(N,)`` int with dropped utterances marked as
          ``dropped_label``.
        * ``confidence`` is ``(N,)`` float32 in ``[0, 1]`` (or ``None`` when
          ``compute_confidence=False``).  Dropped utterances get ``0.0``.
        * ``stats`` is a dict summarising every stage of the pipeline.
    """
    n, d = embeddings.shape
    logger.info(
        "Large-scale clustering: N=%d, D=%d, threshold=%.4f, "
        "min_cluster_size=%d, birch_threshold=%.4f",
        n, d, threshold, min_cluster_size, birch_threshold,
    )

    if n == 0:
        return (
            np.empty(0, dtype=int),
            (np.empty(0, dtype=np.float32) if compute_confidence else None),
            {"n_input": 0},
        )
    if n == 1:
        labels = np.array([1], dtype=int)
        if min_cluster_size > 1:
            labels[:] = dropped_label
        conf = np.zeros(1, dtype=np.float32) if compute_confidence else None
        return labels, conf, {"n_input": 1}

    # ---- L2-normalise once.  All downstream stages assume unit-norm input.
    logger.info("Stage 0: L2-normalising %d embeddings", n)
    normed = _l2_normalize(embeddings.astype(np.float32, copy=False))

    # ---- Stage 1: BIRCH.
    logger.info(
        "Stage 1: BIRCH partial_fit (batch=%d, branching=%d, threshold=%.4f)",
        partial_fit_batch, branching_factor, birch_threshold,
    )
    birch = _build_birch_tree(
        normed,
        birch_threshold=birch_threshold,
        branching_factor=branching_factor,
        partial_fit_batch=partial_fit_batch,
    )
    leaf_centroids = np.asarray(birch.subcluster_centers_, dtype=np.float32)
    n_sub = leaf_centroids.shape[0]
    logger.info("Stage 1 done: %d leaf subclusters", n_sub)

    # Re-L2-normalise centroids (BIRCH means are *inside* the unit sphere).
    normed_centroids = _l2_normalize(leaf_centroids)

    # ---- Stage 2: utterance -> leaf assignment.
    logger.info("Stage 2: assigning %d utterances to %d leaves (tile=%d)",
                n, n_sub, assign_tile)
    leaf_idx = _assign_to_nearest_leaf(
        normed, normed_centroids, tile=assign_tile,
    )

    # ---- Stage 3: AHC on centroids.
    logger.info("Stage 3: AHC on %d leaf centroids (linkage=%s, threshold=%.4f)",
                n_sub, linkage_method, threshold)
    centroid_labels = _ahc_on_centroids(
        normed_centroids,
        threshold=threshold,
        linkage_method=linkage_method,
    )
    n_speakers_raw = int(len(set(centroid_labels.tolist())))
    logger.info("Stage 3 done: %d centroid clusters (= raw speakers)",
                n_speakers_raw)

    # ---- Stage 4: back-propagate to utterances.
    logger.info("Stage 4: back-propagating speaker labels to utterances")
    labels = centroid_labels[leaf_idx].astype(np.int64, copy=False)

    # ---- Stage 5: filter small clusters.
    logger.info("Stage 5: filtering clusters smaller than %d utterances",
                min_cluster_size)
    labels, filter_stats = filter_small_clusters(
        labels, min_cluster_size=min_cluster_size, dropped_label=dropped_label,
    )
    logger.info(
        "Stage 5 done: kept %d / %d clusters, %d / %d utterances "
        "(dropped %.1f%%)",
        filter_stats["n_clusters_after"], filter_stats["n_clusters_before"],
        filter_stats["n_utts_kept"], filter_stats["n_utts_before"],
        100.0 * filter_stats["fraction_dropped"],
    )

    # ---- Stage 6: centroid-based confidence.
    confidence: Optional[np.ndarray] = None
    if compute_confidence:
        logger.info("Stage 6: per-utterance confidence vs speaker centroids")
        confidence = _speaker_confidence_from_centroids(
            normed, labels, dropped_label=dropped_label, tile=assign_tile,
        )

    stats = {
        "n_input": n,
        "embedding_dim": d,
        "threshold": threshold,
        "linkage_method": linkage_method,
        "birch_threshold": birch_threshold,
        "branching_factor": branching_factor,
        "n_leaf_subclusters": n_sub,
        "n_clusters_raw": n_speakers_raw,
        "filter": filter_stats,
    }
    return labels, confidence, stats


# ---------------------------------------------------------------------------
# Logging / pretty-print helpers
# ---------------------------------------------------------------------------

def print_large_scale_summary(
    labels: np.ndarray,
    stats: Dict,
    confidence: Optional[np.ndarray] = None,
    dropped_label: int = DROPPED_LABEL,
) -> None:
    """Print a human-readable summary of a large-scale clustering run."""
    kept_mask = labels != dropped_label
    kept_labels = labels[kept_mask]
    kept_counts = Counter(kept_labels.tolist())
    kept_sizes = sorted(kept_counts.values(), reverse=True)

    print(f"\n{'='*60}")
    print("  Large-Scale Clustering Results")
    print(f"  (BIRCH leaves={stats.get('n_leaf_subclusters', 'N/A')}, "
          f"threshold={stats.get('threshold', float('nan')):.4f}, "
          f"linkage={stats.get('linkage_method', 'N/A')})")
    print(f"{'='*60}")
    print(f"  Utterances input    : {stats.get('n_input', len(labels)):,}")
    print(f"  BIRCH leaves        : {stats.get('n_leaf_subclusters', 'N/A')}")
    print(f"  Speakers (raw)      : {stats.get('n_clusters_raw', 'N/A'):,}")

    fstats = stats.get("filter", {})
    print(f"\n  min_cluster_size    : {fstats.get('min_cluster_size', 'N/A')}")
    print(f"  Speakers after filter: {fstats.get('n_clusters_after', 'N/A'):,}")
    print(f"  Utterances kept     : {fstats.get('n_utts_kept', 'N/A'):,}")
    print(f"  Utterances dropped  : {fstats.get('n_utts_dropped', 'N/A'):,} "
          f"({100.0 * fstats.get('fraction_dropped', 0.0):.1f}%)")

    if kept_sizes:
        print("\n  Kept cluster sizes:")
        print(f"    largest         : {kept_sizes[0]:,}")
        print(f"    median          : {int(np.median(kept_sizes)):,}")
        print(f"    smallest        : {kept_sizes[-1]:,}")
        top = min(20, len(kept_sizes))
        print(f"    top-{top}          : {kept_sizes[:top]}")

    if confidence is not None:
        kept_conf = confidence[kept_mask]
        if len(kept_conf) > 0:
            print("\n  Confidence (kept utterances):")
            print(f"    mean={kept_conf.mean():.4f}  "
                  f"median={float(np.median(kept_conf)):.4f}  "
                  f"min={kept_conf.min():.4f}  max={kept_conf.max():.4f}")

    print(f"{'='*60}\n")
