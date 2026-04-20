#!/usr/bin/env python3
"""Cluster speaker embeddings via Agglomerative Hierarchical Clustering (AHC).

Two backends are available:

* ``--backend standard`` (default for small/medium datasets) — full ``N x N``
  cosine-similarity AHC.  Best quality, but RAM grows as ``O(N^2)``.  Use
  for up to ~150,000 utterances (~500 hours of audio).

* ``--backend large_scale`` — BIRCH (streaming) followed by AHC on the leaf
  centroids, with a ``--min-cluster-size`` filter that drops the long tail.
  Memory peak is bounded by ``n_subclusters^2`` instead of ``N^2``.
  **Recommended for datasets larger than 500 hours of audio** or more than
  150,000 utterances.

* ``--backend auto`` (default) — picks ``large_scale`` when the embedding
  count exceeds ``--auto-utterance-threshold`` (default 150,000), else uses
  ``standard``.  See ``recommend_clustering_method`` in
  ``speaker_id/clustering/large_scale_clustering_and_scoring.py``.

Usage:
    python scripts/cluster_speakers.py \\
        --embeddings-dir /disk_f_nvd/datasets/Yodas/da/wespeaker_embeddings/ \\
        --threshold 0.40

    # Force the large-scale backend, drop clusters with < 30 utterances:
    python scripts/cluster_speakers.py \\
        --embeddings-dir /disk_f_nvd/datasets/Yodas/de/wespeaker_embeddings/ \\
        --backend large_scale \\
        --min-cluster-size 30 \\
        --threshold 0.40

Reads ``embeddings.npy`` and ``utt_names.txt`` produced by extract_embeddings.py
(or run_multigpu.py).  Writes a JSONL file mapping each utterance to its cluster.
"""

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    from nemo_curator.stages.audio.speaker_id.clustering.ahc import (
        DEFAULT_THRESHOLD,
        EER_THRESHOLD,
        cluster_embeddings,
        print_cluster_summary,
        speaker_confidence,
        print_quality_summary,
    )
    from nemo_curator.stages.audio.speaker_id.clustering.large_scale_clustering_and_scoring import (
        DEFAULT_MIN_CLUSTER_SIZE,
        DROPPED_LABEL,
        LARGE_SCALE_HOURS_THRESHOLD,
        cluster_embeddings_large_scale,
        print_large_scale_summary,
        recommend_clustering_method,
    )
    from nemo_curator.stages.audio.speaker_id.utils.io import load_embeddings

    p = argparse.ArgumentParser(
        description="Cluster speaker embeddings with AHC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            f"Threshold reference (SimAM_ResNet100, VoxCeleb1-O cleaned):\n"
            f"  EER threshold  = {EER_THRESHOLD:.4f}  (balanced, more merging)\n"
            f"  Default        = {DEFAULT_THRESHOLD:.4f}  (stricter, fewer false merges)\n"
            f"\n"
            f"Backend recommendation:\n"
            f"  Use --backend large_scale for datasets > {LARGE_SCALE_HOURS_THRESHOLD:.0f}h\n"
            f"  of audio or > 150,000 utterances.  --backend auto picks for you.\n"
        ),
    )
    p.add_argument(
        "--embeddings-dir",
        required=True,
        help="Directory containing embeddings.npy and utt_names.txt",
    )
    p.add_argument(
        "--backend",
        choices=["auto", "standard", "large_scale"],
        default="auto",
        help=(
            "Clustering backend.  'auto' picks 'large_scale' when the "
            "embedding count exceeds --auto-utterance-threshold "
            "(default behaviour for datasets > 500h of audio)."
        ),
    )
    p.add_argument(
        "--auto-utterance-threshold",
        type=int,
        default=150_000,
        help=(
            "When --backend=auto, switch to 'large_scale' if the number of "
            "embeddings exceeds this value.  Default 150,000 corresponds "
            "roughly to 500 hours of audio."
        ),
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Cosine-similarity threshold for same-speaker decision (default: {DEFAULT_THRESHOLD})",
    )
    p.add_argument(
        "--linkage",
        default="average",
        choices=["average", "complete", "single"],
        help="Linkage method for AHC (default: average)",
    )
    p.add_argument(
        "--min-cluster-size",
        type=int,
        default=DEFAULT_MIN_CLUSTER_SIZE,
        help=(
            f"large_scale backend only: drop clusters with fewer than this "
            f"many utterances (label set to {DROPPED_LABEL}).  "
            f"Default: {DEFAULT_MIN_CLUSTER_SIZE}.  Set to 1 to disable."
        ),
    )
    p.add_argument(
        "--birch-threshold",
        type=float,
        default=None,
        help=(
            "large_scale backend only: BIRCH leaf radius (Euclidean on "
            "L2-normalised embeddings).  Defaults to ~0.6325, equivalent to "
            "a per-leaf cosine floor of ~0.8."
        ),
    )
    p.add_argument(
        "--no-confidence",
        action="store_true",
        help="Skip confidence scoring (faster, no quality report)",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Output JSONL path (default: <embeddings-dir>/clusters.jsonl)",
    )
    args = p.parse_args()

    embeddings, utt_ids = load_embeddings(args.embeddings_dir)
    logger.info(
        "Loaded %d embeddings of dim %d from %s",
        embeddings.shape[0],
        embeddings.shape[1],
        args.embeddings_dir,
    )

    # ---- Backend selection
    if args.backend == "auto":
        chosen = recommend_clustering_method(
            num_utterances=embeddings.shape[0],
            utterance_threshold=args.auto_utterance_threshold,
        )
        logger.info(
            "auto backend selection: N=%d, threshold=%d -> %s",
            embeddings.shape[0], args.auto_utterance_threshold, chosen,
        )
    else:
        chosen = args.backend

    # ---- Cluster
    if chosen == "large_scale":
        birch_kwargs = {}
        if args.birch_threshold is not None:
            birch_kwargs["birch_threshold"] = args.birch_threshold

        labels, conf_scores, stats = cluster_embeddings_large_scale(
            embeddings,
            threshold=args.threshold,
            linkage_method=args.linkage,
            min_cluster_size=args.min_cluster_size,
            compute_confidence=not args.no_confidence,
            **birch_kwargs,
        )
        print_large_scale_summary(labels, stats, confidence=conf_scores)

    else:
        labels = cluster_embeddings(
            embeddings,
            threshold=args.threshold,
            linkage_method=args.linkage,
        )
        print_cluster_summary(labels, threshold=args.threshold)

        conf_scores = None
        if not args.no_confidence:
            logger.info("Computing speaker-ID confidence scores...")
            conf_scores = speaker_confidence(embeddings, labels)
            print_quality_summary(embeddings, labels, conf_scores)

    # ---- Write JSONL
    output_path = args.output or os.path.join(args.embeddings_dir, "clusters.jsonl")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    n_dropped = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for i, (utt_id, label) in enumerate(zip(utt_ids, labels)):
            label_int = int(label)
            if label_int == DROPPED_LABEL:
                n_dropped += 1
            record = {"utt_id": utt_id, "cluster_id": label_int}
            if conf_scores is not None:
                record["speaker_confidence"] = round(float(conf_scores[i]), 4)
            f.write(json.dumps(record) + "\n")

    logger.info("Cluster assignments written to %s", output_path)
    if chosen == "large_scale" and n_dropped > 0:
        logger.info(
            "  %d / %d utterances marked dropped (cluster_id = %d) by "
            "--min-cluster-size=%d filter",
            n_dropped, len(utt_ids), DROPPED_LABEL, args.min_cluster_size,
        )


if __name__ == "__main__":
    main()
