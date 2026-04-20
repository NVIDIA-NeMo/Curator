#!/usr/bin/env python3
"""SCOTCH-v1.large_scale.<preset>: cluster + annotate manifests (cluster edition).

Cluster-side driver that ships **inside** the Curator tutorial.  Take an
``embeddings_merged.npz`` produced by ``run_pipeline.py --direct`` (then
``--merge``), apply the SCOTCH-v1.large_scale clustering preset, and write
out:

    ``<output_dir>/manifest_<sid>.json``  --  each input shard manifest
        copied verbatim with two extra fields per row:
        ``speaker_label`` (int, ``-1`` if dropped by ``min_cluster_size``)
        ``confidence_score`` (float in ``[0, 1]``, silhouette-style;
        see ``PARAM_TUNE.md`` 9.1 for the exact formula).
    ``<output_dir>/clusters_summary.jsonl``  --  flat one-line-per-utterance
        ``{audio_filepath, speaker_label, confidence_score}`` index.
    ``<output_dir>/cluster_config.json``  --  SCOTCH sidecar built via
        :func:`build_cluster_config`; documents *exactly* which preset and
        which knobs produced these labels.  See ``PARAM_TUNE.md`` 3.

Why a separate driver (instead of ``run_pipeline.py --cluster``)?
-----------------------------------------------------------------
``run_pipeline.py --cluster`` runs the *standard* AHC backend
(:class:`SpeakerClusteringStage`).  The SCOTCH-v1.large_scale preset uses
the BIRCH + AHC backend (``cluster_embeddings_large_scale``) which is
required for 20--30M-utterance corpora and which produces silhouette
``confidence_score`` instead of the standard self-cosine.  This driver
calls that backend directly with the preset values pulled from
``nemo_curator.stages.audio.speaker_id.clustering.cluster_config.PRESETS``
so the result is reproducible and self-documenting via the sidecar.

Cluster paths
-------------
This script and its launcher (``run_scotch_cluster_only.sh``) target two
NVIDIA clusters:

* **CS-OCI-ORD**:
  ``DATA_ROOT = /lustre/fs11/portfolios/llmservice/projects/llmservice_nemo_speechlm/data/ASR/librispeech/tarred_train``
* **DRACO-OCI-IAD**:
  ``DATA_ROOT = /lustre/fs12/portfolios/llmservice/projects/llmservice_nemo_speechlm/data/ASR/librispeech/tarred_train``

The launcher auto-detects the cluster from the hostname; this Python
driver is path-agnostic and only requires ``--merged_npz`` /
``--manifest_dir`` / ``--output_dir``.

Usage
-----

Cluster-typical invocation (paths are the CS-OCI-ORD layout)::

    DATA_ROOT=/lustre/fs11/portfolios/llmservice/projects/llmservice_nemo_speechlm/data/ASR/librispeech/tarred_train
    WORK_DIR=/lustre/fsw/portfolios/llmservice/users/${USER}/scotch_librispeech

    python scotch_cluster_and_annotate.py \\
        --merged_npz   "${WORK_DIR}/embeddings/embeddings_merged.npz" \\
        --manifest_dir "${DATA_ROOT}/raw_sharded_manifests" \\
        --output_dir   "${WORK_DIR}/scotch_speaker_clustering_results"

All preset values are loaded from
``nemo_curator.stages.audio.speaker_id.clustering.cluster_config.PRESETS``;
override any single knob with ``--threshold`` / ``--min_cluster_size`` /
``--birch_cosine_floor`` / etc., and the override will be recorded in
``cluster_config.json`` so the sidecar always reflects what actually ran.

Use ``run_scotch_cluster_only.sh`` if you just want sensible cluster-aware
defaults wired up for you.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("scotch_cluster")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _load_merged(merged_npz: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(cut_ids[N], embeddings[N, D])``."""
    logger.info("Loading merged embeddings: %s", merged_npz)
    data = np.load(merged_npz, allow_pickle=True)
    cut_ids = np.asarray(data["cut_ids"], dtype=object)
    embeddings = np.asarray(data["embeddings"], dtype=np.float32)
    logger.info("  N=%d, D=%d", embeddings.shape[0], embeddings.shape[1])
    return cut_ids, embeddings


def _enumerate_shards(manifest_dir: str) -> List[Tuple[int, str]]:
    """Return ``[(shard_id, manifest_path), ...]`` sorted by ``shard_id``."""
    out: List[Tuple[int, str]] = []
    for fn in os.listdir(manifest_dir):
        if not (fn.startswith("manifest_") and fn.endswith(".json")):
            continue
        try:
            sid = int(fn[len("manifest_"):-len(".json")])
        except ValueError:
            continue
        out.append((sid, os.path.join(manifest_dir, fn)))
    out.sort(key=lambda t: t[0])
    return out


def _normalize_for_clustering(
    embeddings: np.ndarray, mode: str
) -> np.ndarray:
    """Mirror Curator's ``center_global`` / ``none`` semantics.

    The SCOTCH-v1.large_scale preset uses ``center_global``, which is also
    the only mode this driver is wired up for.  ``external`` (cohort mean
    / std) is intentionally not exposed here; if you need it, run the
    standard ``run_pipeline.py --cluster`` flow.
    """
    if mode == "none":
        return embeddings.astype(np.float32, copy=False)
    if mode == "center_global":
        x = embeddings.astype(np.float32, copy=True)
        x -= x.mean(axis=0, keepdims=True)
        return x
    raise ValueError(
        f"Unsupported embedding_normalization={mode!r} for this driver "
        "(only 'center_global' / 'none' are wired up here)."
    )


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--merged_npz", required=True,
        help="Path to embeddings_merged.npz (produced by run_pipeline.py --merge).",
    )
    p.add_argument(
        "--manifest_dir", required=True,
        help="Directory containing per-shard input manifests "
             "(manifest_<shard_id>.json).",
    )
    p.add_argument(
        "--output_dir", required=True,
        help="Where to write annotated per-shard manifests + cluster_config.json "
             "+ clusters_summary.jsonl.",
    )
    p.add_argument(
        "--preset", default="librispeech-2026-04",
        help="SCOTCH preset name; PRESETS[preset] supplies the default values. "
             "See nemo_curator.stages.audio.speaker_id.clustering.cluster_config.PRESETS.",
    )

    # SCOTCH-v1.large_scale knobs.  Default = None so we can tell whether
    # the user overrode the preset and record it in the sidecar.
    p.add_argument("--threshold", type=float, default=None,
                   help="AHC cosine cutoff (preset default: 0.50).")
    p.add_argument("--linkage", type=str, default=None,
                   choices=[None, "average", "complete", "single"],
                   help="AHC linkage method (preset default: average).")
    p.add_argument("--min_cluster_size", type=int, default=None,
                   help="Drop clusters smaller than this (preset default: 30; "
                        "set to 1 to disable).")
    p.add_argument("--birch_cosine_floor", type=float, default=None,
                   help="Per-leaf cosine floor; converted to Euclidean radius "
                        "via sqrt(2*(1-c)) (preset default: 0.95).")
    p.add_argument("--branching_factor", type=int, default=None,
                   help="BIRCH branching_factor (preset default: 50).")
    p.add_argument("--partial_fit_batch", type=int, default=None,
                   help="Utterances per BIRCH partial_fit (preset default: 50000).")
    p.add_argument("--assign_tile", type=int, default=None,
                   help="Tile size for utt -> leaf assignment "
                        "(preset default: 16384).")
    p.add_argument("--embedding_normalization", type=str, default=None,
                   choices=[None, "none", "center_global"],
                   help="Pre-cosine normalisation (preset default: center_global).")
    p.add_argument("--no_confidence", action="store_true",
                   help="Skip per-utterance confidence scoring "
                        "(faster, but no confidence_score field in output).")

    p.add_argument("--audio_filepath_key", type=str, default="audio_filepath",
                   help="Manifest key holding the cut_id "
                        "(default: audio_filepath).")
    p.add_argument(
        "--curator_repo", type=str, default="",
        help="Optional path to the Curator checkout (a directory containing "
             "nemo_curator/).  Only needed if Curator is not already on "
             "PYTHONPATH / pip-installed; the launcher exports PYTHONPATH "
             "for you.",
    )
    args = p.parse_args()

    # Late import so --help works even if PYTHONPATH isn't set yet.
    if args.curator_repo:
        cr = os.path.abspath(os.path.expanduser(args.curator_repo))
        if os.path.isdir(os.path.join(cr, "nemo_curator")) and cr not in sys.path:
            sys.path.insert(0, cr)

    try:
        from nemo_curator.stages.audio.speaker_id.clustering.cluster_config import (
            PRESETS, build_cluster_config, cosine_floor_to_birch_radius,
            write_cluster_config,
        )
        from nemo_curator.stages.audio.speaker_id.clustering.large_scale_clustering_and_scoring import (
            DROPPED_LABEL, cluster_embeddings_large_scale,
        )
    except ImportError as e:
        logger.error(
            "Could not import nemo_curator.  Either pip-install Curator, or "
            "ensure the Curator repo is on PYTHONPATH, or pass "
            "--curator_repo /path/to/Curator.  Error: %s", e,
        )
        return 1

    if args.preset not in PRESETS:
        logger.error("Unknown preset %r.  Known: %s",
                     args.preset, sorted(PRESETS))
        return 1
    preset = PRESETS[args.preset]
    if not preset:
        logger.error(
            "Preset %r has no expected values; pick a concrete preset like "
            "'librispeech-2026-04' or pass every knob explicitly.", args.preset,
        )
        return 1

    # Resolve effective parameters.  CLI flag > preset.
    def _pick(name: str, default):
        v = getattr(args, name, None)
        return default if v is None else v

    cluster_threshold = _pick("threshold", preset["cluster_threshold"])
    cluster_linkage = _pick("linkage", preset["cluster_linkage"])
    min_cluster_size = _pick("min_cluster_size", preset["min_cluster_size"])
    birch_cosine_floor = _pick("birch_cosine_floor", preset["birch_cosine_floor"])
    birch_branching_factor = _pick("branching_factor", preset["birch_branching_factor"])
    birch_partial_fit_batch = _pick("partial_fit_batch", preset["birch_partial_fit_batch"])
    assign_tile = _pick("assign_tile", preset["assign_tile"])
    embedding_normalization = _pick("embedding_normalization",
                                    preset["embedding_normalization"])
    confidence_enabled = not args.no_confidence

    birch_radius = cosine_floor_to_birch_radius(birch_cosine_floor)

    config_id = f"SCOTCH-v1.large_scale.{args.preset}"
    logger.info("=" * 72)
    logger.info("Config:           %s", config_id)
    logger.info("  threshold:        %.4f", cluster_threshold)
    logger.info("  linkage:          %s", cluster_linkage)
    logger.info("  min_cluster_size: %d", min_cluster_size)
    logger.info("  birch cos floor:  %.4f  -> radius %.4f",
                birch_cosine_floor, birch_radius)
    logger.info("  branching_factor: %d", birch_branching_factor)
    logger.info("  partial_fit_bat:  %d", birch_partial_fit_batch)
    logger.info("  assign_tile:      %d", assign_tile)
    logger.info("  emb normalisation:%s", embedding_normalization)
    logger.info("  confidence:       %s", confidence_enabled)
    logger.info("=" * 72)

    # Load embeddings, normalise, cluster.
    cut_ids, embeddings = _load_merged(args.merged_npz)
    n_input, embedding_dim = embeddings.shape

    logger.info("Normalising (mode=%s)...", embedding_normalization)
    embeddings = _normalize_for_clustering(embeddings, embedding_normalization)

    logger.info("Clustering with cluster_embeddings_large_scale ...")
    t0 = time.time()
    labels, conf_scores, stats = cluster_embeddings_large_scale(
        embeddings,
        threshold=cluster_threshold,
        linkage_method=cluster_linkage,
        min_cluster_size=min_cluster_size,
        birch_threshold=birch_radius,
        branching_factor=birch_branching_factor,
        partial_fit_batch=birch_partial_fit_batch,
        assign_tile=assign_tile,
        compute_confidence=confidence_enabled,
    )
    runtime = time.time() - t0
    logger.info("Clustering done in %.1fs", runtime)

    # Stats from cluster_embeddings_large_scale (best-effort, depending on impl).
    n_leaf = stats.get("n_leaf_subclusters") if isinstance(stats, dict) else None
    n_clusters_raw = stats.get("n_clusters_raw") if isinstance(stats, dict) else None
    n_clusters_kept = stats.get("n_clusters_kept") if isinstance(stats, dict) else None
    n_utts_dropped = int(np.sum(labels == DROPPED_LABEL))
    n_utts_kept = int(n_input - n_utts_dropped)

    if n_clusters_kept is None:
        kept_mask = labels != DROPPED_LABEL
        n_clusters_kept = int(np.unique(labels[kept_mask]).size) if kept_mask.any() else 0

    logger.info("  N=%d  kept=%d  dropped=%d  clusters_kept=%s  leaves=%s",
                n_input, n_utts_kept, n_utts_dropped, n_clusters_kept, n_leaf)

    # Index cut_id -> row.
    id_to_idx: Dict[str, int] = {str(c): i for i, c in enumerate(cut_ids)}

    # Write the SCOTCH cluster_config.json sidecar BEFORE we touch any
    # manifest output, so a partial run still leaves a record of what was
    # asked for.
    os.makedirs(args.output_dir, exist_ok=True)
    cfg = build_cluster_config(
        backend="large_scale",
        preset=args.preset,
        cluster_threshold=cluster_threshold,
        cluster_linkage=cluster_linkage,
        min_cluster_size=min_cluster_size,
        n_input=n_input,
        embedding_dim=embedding_dim,
        embedding_normalization=embedding_normalization,
        confidence_enabled=confidence_enabled,
        birch_cosine_floor=birch_cosine_floor,
        birch_radius=birch_radius,
        birch_branching_factor=birch_branching_factor,
        birch_partial_fit_batch=birch_partial_fit_batch,
        assign_tile=assign_tile,
        n_leaf_subclusters=n_leaf,
        n_clusters_raw=n_clusters_raw,
        n_clusters_kept=n_clusters_kept,
        n_utts_kept=n_utts_kept,
        n_utts_dropped=n_utts_dropped,
        runtime_seconds=round(runtime, 3),
        extra={
            "merged_npz": os.path.abspath(args.merged_npz),
            "manifest_dir": os.path.abspath(args.manifest_dir),
            "output_dir": os.path.abspath(args.output_dir),
            "audio_filepath_key": args.audio_filepath_key,
        },
    )
    cfg_path = write_cluster_config(args.output_dir, cfg)
    logger.info("Sidecar: %s", cfg_path)

    # Walk every input manifest shard, annotate, write to output_dir.
    shards = _enumerate_shards(args.manifest_dir)
    if not shards:
        logger.error("No manifest_*.json files found in %s", args.manifest_dir)
        return 1
    logger.info("Annotating %d manifest shards from %s",
                len(shards), args.manifest_dir)

    summary_path = os.path.join(args.output_dir, "clusters_summary.jsonl")
    n_lines_total = 0
    n_no_emb = 0
    with open(summary_path, "w", encoding="utf-8") as summary_f:
        for sid, mp in shards:
            out_path = os.path.join(args.output_dir, os.path.basename(mp))
            n_in_shard = 0
            with open(mp, "r", encoding="utf-8") as fin, \
                 open(out_path, "w", encoding="utf-8") as fout:
                for raw in fin:
                    raw = raw.strip()
                    if not raw:
                        continue
                    rec = json.loads(raw)
                    afp = rec.get(args.audio_filepath_key)
                    idx = id_to_idx.get(str(afp))
                    if idx is None:
                        rec["speaker_label"] = -1
                        rec["confidence_score"] = 0.0
                        n_no_emb += 1
                    else:
                        rec["speaker_label"] = int(labels[idx])
                        if conf_scores is not None:
                            rec["confidence_score"] = round(float(conf_scores[idx]), 6)
                        else:
                            rec["confidence_score"] = None
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

                    summary_f.write(json.dumps({
                        args.audio_filepath_key: afp,
                        "speaker_label": rec["speaker_label"],
                        "confidence_score": rec["confidence_score"],
                    }) + "\n")
                    n_in_shard += 1
                    n_lines_total += 1
            if sid % 64 == 0:
                logger.info("  shard %3d -> %s (%d lines)", sid, out_path, n_in_shard)

    logger.info("=" * 72)
    logger.info("Wrote %d annotated manifest shards to %s",
                len(shards), args.output_dir)
    logger.info("  total utterances annotated: %d", n_lines_total)
    if n_no_emb:
        logger.warning(
            "  %d / %d manifest lines had no matching embedding "
            "(speaker_label=-1, confidence_score=0.0)",
            n_no_emb, n_lines_total,
        )
    logger.info("  clusters_summary.jsonl:    %s", summary_path)
    logger.info("  cluster_config.json:       %s", cfg_path)
    logger.info("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
