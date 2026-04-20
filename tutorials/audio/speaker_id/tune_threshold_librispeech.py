# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tune the speaker-clustering cosine threshold on LibriSpeech train.

This is a *measurement* tool: it runs the production large-scale clustering
pipeline (BIRCH + AHC, from
``nemo_curator.stages.audio.speaker_id.clustering.large_scale_clustering_and_scoring``)
across a 2-D grid of cosine-similarity thresholds and reports the value that
maximises B-cubed F1 against LibriSpeech's gold speaker labels.

LibriSpeech filenames follow ``<spkid>-<chap>-<utt>.flac``.  In NeMo
Canary-style tarred manifests the original path is encoded in
``audio_filepath`` as e.g.

    "_disk7_..._1638_84448_1638-84448-0036.flac"

so the LibriSpeech basename is the part after the last underscore and the
speaker id is the first ``-``-separated field of that basename.

Pipeline (three independent subcommands)
----------------------------------------

1. ``extract`` -- delegate to Curator's existing
   ``run_pipeline.py --direct`` and ``--merge`` to produce a single
   ``embeddings_merged.npz``.  GPU stage.  Skipped if the merged file
   already exists.

2. ``labels``  -- read every manifest line, parse the LibriSpeech speaker id
   from the encoded path, align to the merged embeddings, and write
   ``labels.npz`` (``cut_ids``, ``embeddings``, ``true_speakers``).  Pure
   I/O; ~seconds.

3. ``tune``    -- sweep the AHC cosine threshold (and optionally the BIRCH
   leaf threshold) on the labelled subset and report the best.  Uses the
   same ``cluster_embeddings_large_scale`` function that ``run_pipeline.py``
   calls in production, with ``min_cluster_size=1`` so that nothing is
   dropped (LibriSpeech is gold-labelled -- all utterances are scored).

You can run all three at once with the ``all`` subcommand.

Metrics
-------

Reported per (ahc_threshold, birch_threshold) cell:

* B-cubed precision / recall / F1     -- primary.  Robust to imbalanced
  cluster sizes, which speakers always are.
* NMI, ARI, V-measure                 -- standard sanity-check metrics.
* Cluster purity, inverse purity      -- diagnostic for over/under-merge.
* Predicted speaker count             -- vs ground truth ``num_speakers``.

The chosen threshold is the one with maximum B-cubed F1, with NMI as
tiebreaker.  Results are written as CSV + JSON next to the merged npz.

Cluster paths
-------------

The defaults below assume the LibriSpeech tarred train set lives at one of
the two NVIDIA cluster locations; the module auto-detects which one based
on the local hostname:

* CS-OCI-ORD::

      /lustre/fs11/portfolios/llmservice/projects/llmservice_nemo_speechlm/data/ASR/librispeech/tarred_train

* DRACO-OCI-IAD::

      /lustre/fs12/portfolios/llmservice/projects/llmservice_nemo_speechlm/data/ASR/librispeech/tarred_train

Both clusters expose ``raw_sharded_manifests/manifest_<sid>.json`` (512
shards) alongside the matching ``audio_<sid>.tar`` files.  Override with
``--manifest_glob`` / ``--tar_glob`` to point at a different location.

Example
-------

::

    # Phase 1 (GPU): extract embeddings into <work>/embeddings/.
    CUDA_VISIBLE_DEVICES=0 python tune_threshold_librispeech.py extract \\
        --work_dir      /scratch/${USER}/librispeech_threshold_tune

    # Phase 2 (CPU, fast): build labels.npz from the manifests.
    python tune_threshold_librispeech.py labels \\
        --work_dir      /scratch/${USER}/librispeech_threshold_tune

    # Phase 3 (CPU): sweep thresholds and pick best.
    python tune_threshold_librispeech.py tune \\
        --work_dir      /scratch/${USER}/librispeech_threshold_tune \\
        --ahc_thresholds 0.20,0.50,0.01 \\
        --birch_thresholds 0.7,0.9,0.05
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

# --------------------------------------------------------------------------
# Defaults
# --------------------------------------------------------------------------

# LibriSpeech tarred-train roots on the two NVIDIA clusters this script
# targets.  We pick one based on a substring match against the local
# hostname; if neither matches, the data-path defaults are left empty and
# the user must supply --manifest_glob / --tar_glob explicitly.
_CLUSTER_DATA_ROOTS = {
    "cs-oci": (
        "/lustre/fs11/portfolios/llmservice/projects/llmservice_nemo_speechlm"
        "/data/ASR/librispeech/tarred_train"
    ),
    "draco-oci": (
        "/lustre/fs12/portfolios/llmservice/projects/llmservice_nemo_speechlm"
        "/data/ASR/librispeech/tarred_train"
    ),
}


def _resolve_default_data_root() -> str:
    """Return the LibriSpeech tarred-train root for the current cluster.

    Resolution order:

    1. ``LIBRISPEECH_DATA_ROOT`` env var, if set.
    2. ``CLUSTER`` env var (``cs-oci-ord`` / ``draco-oci-iad``).
    3. Hostname substring match against the keys of ``_CLUSTER_DATA_ROOTS``.
    4. Empty string -- caller must supply --manifest_glob / --tar_glob.
    """
    env = os.environ.get("LIBRISPEECH_DATA_ROOT", "").strip()
    if env:
        return env
    cluster = os.environ.get("CLUSTER", "").strip().lower()
    if cluster:
        for key, root in _CLUSTER_DATA_ROOTS.items():
            if key in cluster:
                return root
    try:
        import socket
        host = socket.getfqdn() or socket.gethostname()
    except OSError:
        host = ""
    host = host.lower()
    for key, root in _CLUSTER_DATA_ROOTS.items():
        if key in host:
            return root
    return ""


_DEFAULT_DATA_ROOT = _resolve_default_data_root()

DEFAULT_MANIFEST_GLOB = (
    f"{_DEFAULT_DATA_ROOT}/raw_sharded_manifests/manifest__OP_0..511_CL_.json"
    if _DEFAULT_DATA_ROOT
    else ""
)
DEFAULT_TAR_GLOB = (
    f"{_DEFAULT_DATA_ROOT}/audio__OP_0..511_CL_.tar"
    if _DEFAULT_DATA_ROOT
    else ""
)
DEFAULT_WORK_DIR = "tune_librispeech_work"

# Legacy: path to the (deprecated) speaker_id_for_asr_data repo.  Kept as an
# argument for backwards compatibility with old launcher scripts; the actual
# clustering module now ships with Curator at
# ``nemo_curator.stages.audio.speaker_id.clustering.large_scale_clustering_and_scoring``.
DEFAULT_SPEAKER_ID_REPO = ""

# 2-D sweep defaults.  AHC range covers the typical 0.2--0.5 cosine band for
# TitaNet + center_global; step 0.01 is granular enough.  BIRCH range is
# expressed as a per-leaf cosine floor (we convert to Euclidean radius
# internally).  0.7 is loose, 0.95 is tight; 0.85 is the production default.
DEFAULT_AHC_RANGE = (0.20, 0.50, 0.01)
DEFAULT_BIRCH_COS_RANGE = (0.70, 0.95, 0.05)

# Default extraction settings.
DEFAULT_BATCH_SIZE = 64
DEFAULT_MODEL = "nvidia/speakerverification_en_titanet_large"

# B-cubed evaluation: tile size for the per-utterance kernel (uses
# O(B * n_clusters) intermediate memory, never the full N x N matrix).
BCUBED_TILE = 4096


logger = logging.getLogger("tune_threshold_librispeech")


# --------------------------------------------------------------------------
# Path helpers (mirror Curator's _expand_nemo_path)
# --------------------------------------------------------------------------

def expand_nemo_path(path: str) -> List[str]:
    """Expand NeMo brace patterns (e.g. ``manifest__OP_0..511_CL_.json``)."""
    for opener in ("(", "[", "<", "_OP_"):
        path = path.replace(opener, "{")
    for closer in (")", "]", ">", "_CL_"):
        path = path.replace(closer, "}")
    m = re.search(r"\{(\d+)\.\.(\d+)\}", path)
    if not m:
        return [path]
    start, end = int(m.group(1)), int(m.group(2))
    prefix, suffix = path[: m.start()], path[m.end() :]
    return [f"{prefix}{i}{suffix}" for i in range(start, end + 1)]


# --------------------------------------------------------------------------
# LibriSpeech speaker-id parsing
# --------------------------------------------------------------------------

# audio_filepath looks like:
#   "_disk7_..._1638_84448_1638-84448-0036.flac"             (single LibriSpeech utt)
#   "_disk7_..._708_129393_708-129393-0098_0099.flac"        (LibriSpeech-PNC pair)
#   "_disk7_..._5266_41151_5266-41151-0000_0001_0002.flac"   (LibriSpeech-PNC triple)
# In every case the speaker id is the first integer of the LibriSpeech tag
# "<spk>-<chap>-<utt>" that appears (possibly followed by "_<utt>..." suffixes)
# at the very end of the path before ".flac".
_LIBRISPEECH_TAIL_RE = re.compile(
    r"_(\d+)-(\d+)-(\d+)(?:_\d+)*\.(?:flac|wav|opus|mp3)$"
)


def parse_speaker_id(audio_filepath: str) -> Optional[str]:
    """Extract the LibriSpeech speaker id from a Canary-style audio_filepath.

    Handles both single-utterance LibriSpeech files (``<spk>-<chap>-<utt>.flac``)
    and LibriSpeech-PNC concatenated files
    (``<spk>-<chap>-<utt0>_<utt1>[_<utt2>...].flac``).  Returns ``None`` if
    the path doesn't match either convention.
    """
    if not audio_filepath:
        return None
    m = _LIBRISPEECH_TAIL_RE.search(audio_filepath)
    return m.group(1) if m else None


# --------------------------------------------------------------------------
# Phase 1: extract  (delegate to run_pipeline.py)
# --------------------------------------------------------------------------

def _require_data_globs(args: argparse.Namespace, *, need_tar: bool) -> None:
    """Fail fast with a friendly message if the data globs are unresolved."""
    missing = []
    if not getattr(args, "manifest_glob", ""):
        missing.append("--manifest_glob")
    if need_tar and not getattr(args, "tar_glob", ""):
        missing.append("--tar_glob")
    if missing:
        cluster_paths = "\n".join(
            f"  * {key}-* hostnames: {root}" for key, root in _CLUSTER_DATA_ROOTS.items()
        )
        raise SystemExit(
            "ERROR: " + ", ".join(missing) + " is required.\n"
            "       Auto-detection found no matching cluster path for this hostname.\n"
            "       Either run on one of:\n"
            f"{cluster_paths}\n"
            "       or pass --manifest_glob / --tar_glob explicitly, or set\n"
            "       LIBRISPEECH_DATA_ROOT=/path/to/tarred_train (or CLUSTER=cs-oci-ord|draco-oci-iad)."
        )


def cmd_extract(args: argparse.Namespace) -> None:
    """Invoke ``run_pipeline.py --direct`` then ``--merge``."""
    _require_data_globs(args, need_tar=True)
    work_dir = os.path.abspath(args.work_dir)
    emb_dir = os.path.join(work_dir, "embeddings")
    os.makedirs(emb_dir, exist_ok=True)
    merged = os.path.join(emb_dir, "embeddings_merged.npz")

    run_pipeline = os.path.join(os.path.dirname(__file__), "run_pipeline.py")
    if not os.path.isfile(run_pipeline):
        raise FileNotFoundError(f"run_pipeline.py not found next to this script: {run_pipeline}")

    if os.path.isfile(merged) and not args.force:
        logger.info("Merged embeddings already exist (%s); skipping extract.", merged)
        return

    # 1a. Per-shard extraction (GPU).
    logger.info("Phase 1a: per-shard embedding extraction -> %s", emb_dir)
    extract_cmd = [
        sys.executable, run_pipeline,
        "--direct",
        "--input_manifest", args.manifest_glob,
        "--input_tar", args.tar_glob,
        "--lhotse_mode", "nemo_tarred",
        "--output_dir", emb_dir,
        "--batch_size", str(args.batch_size),
        "--model_name", args.model_name,
    ]
    if args.max_cuts is not None:
        extract_cmd += ["--max_cuts", str(args.max_cuts)]
    logger.info("$ %s", " ".join(extract_cmd))
    subprocess.run(extract_cmd, check=True)

    # 1b. Merge per-shard files into a single npz.
    logger.info("Phase 1b: merging per-shard embeddings -> %s", merged)
    merge_cmd = [
        sys.executable, run_pipeline,
        "--merge",
        "--output_dir", emb_dir,
    ]
    logger.info("$ %s", " ".join(merge_cmd))
    subprocess.run(merge_cmd, check=True)

    if not os.path.isfile(merged):
        raise FileNotFoundError(f"Merge step did not produce {merged}")
    logger.info("Phase 1 done.  Merged: %s", merged)


# --------------------------------------------------------------------------
# Phase 2: labels  (build labels.npz)
# --------------------------------------------------------------------------

def cmd_labels(args: argparse.Namespace) -> None:
    """Join manifest speaker labels with merged embeddings."""
    _require_data_globs(args, need_tar=False)
    work_dir = os.path.abspath(args.work_dir)
    emb_dir = os.path.join(work_dir, "embeddings")
    merged = os.path.join(emb_dir, "embeddings_merged.npz")
    out_path = os.path.join(work_dir, "labels.npz")

    if not os.path.isfile(merged):
        raise FileNotFoundError(
            f"Merged embeddings not found at {merged}.  Run the 'extract' "
            "subcommand first."
        )

    if os.path.isfile(out_path) and not args.force:
        logger.info("labels.npz already exists (%s); skipping.", out_path)
        return

    logger.info("Loading merged embeddings: %s", merged)
    data = np.load(merged, allow_pickle=True)
    cut_ids = np.asarray(data["cut_ids"], dtype=object)
    embeddings = np.asarray(data["embeddings"], dtype=np.float32)
    logger.info("  -> N=%d, D=%d", embeddings.shape[0], embeddings.shape[1])

    # Build cut_id -> idx map.
    id_to_idx: Dict[str, int] = {str(c): i for i, c in enumerate(cut_ids)}

    # Walk every manifest line and look up the embedding row + speaker id.
    manifest_paths = expand_nemo_path(args.manifest_glob)
    logger.info("Reading %d manifest shards", len(manifest_paths))

    matched = np.full(embeddings.shape[0], -1, dtype=np.int64)  # row -> spk_int
    spk_to_int: Dict[str, int] = {}
    n_lines = 0
    n_no_spk = 0
    n_no_emb = 0
    for mp in manifest_paths:
        if not os.path.isfile(mp):
            logger.warning("Manifest missing, skipping: %s", mp)
            continue
        with open(mp, "r", encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                n_lines += 1
                rec = json.loads(raw)
                afp = rec.get("audio_filepath", "")
                spk = parse_speaker_id(afp)
                if spk is None:
                    n_no_spk += 1
                    continue
                idx = id_to_idx.get(afp)
                if idx is None:
                    n_no_emb += 1
                    continue
                if spk not in spk_to_int:
                    spk_to_int[spk] = len(spk_to_int)
                matched[idx] = spk_to_int[spk]

    n_matched = int((matched >= 0).sum())
    logger.info(
        "Matched %d / %d manifest lines to embeddings (%d unique speakers; "
        "no_spkid=%d, no_embedding=%d)",
        n_matched, n_lines, len(spk_to_int), n_no_spk, n_no_emb,
    )

    if n_matched == 0:
        raise RuntimeError("No manifest lines matched embeddings; check paths / cut id format.")

    # Restrict to matched rows only.
    keep_mask = matched >= 0
    kept_ids = cut_ids[keep_mask]
    kept_embs = embeddings[keep_mask]
    kept_spks = matched[keep_mask].astype(np.int64)

    int_to_spk = np.empty(len(spk_to_int), dtype=object)
    for s, k in spk_to_int.items():
        int_to_spk[k] = s

    np.savez(
        out_path,
        cut_ids=np.asarray(kept_ids, dtype=object),
        embeddings=kept_embs,
        true_speakers=kept_spks,
        speaker_id_str=int_to_spk,
    )
    logger.info(
        "Wrote %s (N=%d, D=%d, num_speakers=%d)",
        out_path, kept_embs.shape[0], kept_embs.shape[1], len(spk_to_int),
    )


# --------------------------------------------------------------------------
# Phase 3: tune  (the threshold sweep)
# --------------------------------------------------------------------------

def _import_large_scale(repo_path: str = ""):
    """Import the ``large_scale_clustering_and_scoring`` module.

    The module now lives inside Curator at
    ``nemo_curator.stages.audio.speaker_id.clustering``.  ``repo_path`` is
    accepted only for backwards compatibility with old launcher scripts that
    still pass ``--speaker_id_repo``; if given, it is added to ``sys.path``
    as a fallback for environments where Curator is not yet importable.
    """
    if repo_path:
        repo_path = os.path.expanduser(repo_path)
        if os.path.isdir(repo_path) and repo_path not in sys.path:
            sys.path.insert(0, repo_path)
    from nemo_curator.stages.audio.speaker_id.clustering import large_scale_clustering_and_scoring as ls  # noqa: E402
    return ls


def _parse_range(spec: str) -> Tuple[float, float, float]:
    """Parse a ``"start,stop,step"`` triple."""
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"Expected 'start,stop,step', got {spec!r}"
        )
    return float(parts[0]), float(parts[1]), float(parts[2])


def _frange(start: float, stop: float, step: float) -> List[float]:
    """Inclusive float range."""
    out: List[float] = []
    v = start
    eps = step / 1e6
    while v <= stop + eps:
        out.append(round(v, 6))
        v += step
    return out


# ----- Embedding normalisation (matches Curator's ``center_global``) -----

def center_global(embeddings: np.ndarray) -> np.ndarray:
    """Subtract the batch mean; matches Curator's default normalisation."""
    x = embeddings.astype(np.float64, copy=False)
    x = x - x.mean(axis=0, keepdims=True)
    return x.astype(np.float32, copy=False)


# ----- Memory-bounded clustering metrics -----------------------------------

def bcubed_precision_recall(
    pred: np.ndarray,
    true: np.ndarray,
) -> Tuple[float, float, float]:
    """B-cubed precision, recall, F1 (Bagga & Baldwin 1998).

    For each utterance ``i``:
      * precision_i = (#j in same pred-cluster as i AND same true-speaker as i) /
                      (#j in same pred-cluster as i)
      * recall_i    = (#j in same true-speaker as i AND same pred-cluster as i) /
                      (#j in same true-speaker as i)
    Returns the mean over all utterances.

    Memory: O(N + n_pred + n_true).  Uses the contingency-table closed form
    sum_{c, k} |c \\cap k|^2.
    """
    n = len(pred)
    if n == 0:
        return 0.0, 0.0, 0.0

    pred_sizes = np.bincount(pred)
    true_sizes = np.bincount(true)

    # Build (pred, true) -> count via combined-key sort + run-length.
    # We need both the cell sizes and which (c, k) each cell corresponds to,
    # so we sort the combined key and decode (c, k) from its sorted values.
    n_true = int(true.max()) + 1
    keys = pred.astype(np.int64) * n_true + true.astype(np.int64)
    keys.sort()
    diffs = np.flatnonzero(np.diff(keys))
    starts = np.concatenate([[0], diffs + 1])
    ends = np.concatenate([diffs + 1, [n]])
    sizes = ends - starts            # |c ∩ k| per cell
    cell_keys = keys[starts]         # combined (c, k) key per cell -- READ FROM SORTED keys
    cell_pred = (cell_keys // n_true).astype(np.int64)
    cell_true = (cell_keys %  n_true).astype(np.int64)

    # Per-utterance precision = |c_i ∩ k_i| / |c_i|.  Sum over utterances:
    #   sum_i |c_i ∩ k_i| / |c_i|
    # = sum_{c,k} (|c ∩ k| / |c|) * |c ∩ k|     [each cell contributes |c∩k| utts]
    # = sum_{c,k} |c ∩ k|^2 / |c|
    # Same logic for recall using |k|.
    sq = sizes.astype(np.float64) ** 2
    p_sum = float(np.sum(sq / pred_sizes[cell_pred]))
    r_sum = float(np.sum(sq / true_sizes[cell_true]))

    precision = p_sum / n
    recall = r_sum / n
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return precision, recall, f1


def cluster_purity_inverse(pred: np.ndarray, true: np.ndarray) -> Tuple[float, float]:
    """Cluster purity and inverse purity.

    purity         = sum_c max_k |c ∩ k| / N
    inverse_purity = sum_k max_c |c ∩ k| / N  (= "completeness" in spirit)
    """
    n = len(pred)
    if n == 0:
        return 0.0, 0.0

    n_true = int(true.max()) + 1
    keys = pred.astype(np.int64) * n_true + true.astype(np.int64)
    keys.sort()
    diffs = np.flatnonzero(np.diff(keys))
    starts = np.concatenate([[0], diffs + 1])
    ends = np.concatenate([diffs + 1, [n]])
    sizes = ends - starts
    cell_keys = keys[starts]
    cell_pred = (cell_keys // n_true).astype(np.int64)
    cell_true = (cell_keys %  n_true).astype(np.int64)

    # max_k |c ∩ k| per pred-cluster c
    max_per_pred: Dict[int, int] = {}
    for c, sz in zip(cell_pred.tolist(), sizes.tolist()):
        if sz > max_per_pred.get(c, 0):
            max_per_pred[c] = sz
    purity = sum(max_per_pred.values()) / n

    # max_c |c ∩ k| per true-speaker k
    max_per_true: Dict[int, int] = {}
    for k, sz in zip(cell_true.tolist(), sizes.tolist()):
        if sz > max_per_true.get(k, 0):
            max_per_true[k] = sz
    inverse_purity = sum(max_per_true.values()) / n

    return purity, inverse_purity


def compute_all_metrics(
    pred: np.ndarray,
    true: np.ndarray,
) -> Dict[str, float]:
    """Compute all clustering metrics for one (pred, true) pair."""
    # Coerce labels to dense 0..K-1 ints (b-cubed and bincount need this).
    pred_dense = _densify(pred)
    true_dense = _densify(true)

    p, r, f1 = bcubed_precision_recall(pred_dense, true_dense)
    purity, inv_purity = cluster_purity_inverse(pred_dense, true_dense)

    metrics: Dict[str, float] = {
        "bcubed_precision": p,
        "bcubed_recall": r,
        "bcubed_f1": f1,
        "purity": purity,
        "inverse_purity": inv_purity,
        "num_pred_clusters": float(int(len(set(pred_dense.tolist())))),
        "num_true_speakers": float(int(len(set(true_dense.tolist())))),
    }

    # Optional sklearn metrics.  Not strictly required; skip on import error
    # so the script still runs in minimal envs.
    try:
        from sklearn.metrics import (
            adjusted_rand_score,
            normalized_mutual_info_score,
            v_measure_score,
        )
        metrics["nmi"] = float(normalized_mutual_info_score(true_dense, pred_dense))
        metrics["ari"] = float(adjusted_rand_score(true_dense, pred_dense))
        metrics["v_measure"] = float(v_measure_score(true_dense, pred_dense))
    except Exception as exc:  # pragma: no cover
        logger.warning("sklearn metrics unavailable: %s", exc)

    return metrics


def _densify(labels: np.ndarray) -> np.ndarray:
    """Map labels to dense 0..K-1 ints.  Preserves grouping."""
    _, inv = np.unique(labels, return_inverse=True)
    return inv.astype(np.int64, copy=False)


# ----- The actual sweep ----------------------------------------------------

def cmd_tune(args: argparse.Namespace) -> None:
    """Sweep AHC threshold (x optional BIRCH threshold) and pick the best."""
    work_dir = os.path.abspath(args.work_dir)
    labels_path = os.path.join(work_dir, "labels.npz")
    if not os.path.isfile(labels_path):
        raise FileNotFoundError(
            f"labels.npz not found at {labels_path}.  Run the 'labels' "
            "subcommand first."
        )

    logger.info("Loading %s", labels_path)
    data = np.load(labels_path, allow_pickle=True)
    embeddings = np.asarray(data["embeddings"], dtype=np.float32)
    true_speakers = np.asarray(data["true_speakers"], dtype=np.int64)
    n, d = embeddings.shape
    n_speakers_true = int(len(set(true_speakers.tolist())))
    logger.info(
        "  N=%d, D=%d, ground-truth speakers=%d",
        n, d, n_speakers_true,
    )

    # Optional centring.  Matches Curator's production default.
    if args.embedding_normalization == "center_global":
        logger.info("Applying center_global normalisation (subtracting batch mean)")
        embeddings = center_global(embeddings)
    elif args.embedding_normalization == "none":
        logger.info("Skipping embedding normalisation (raw embeddings)")
    else:
        raise ValueError(
            f"Unknown embedding_normalization: {args.embedding_normalization!r}"
        )

    ls = _import_large_scale(args.speaker_id_repo)
    L2 = ls._l2_normalize  # private but stable; we only call once per BIRCH refit

    # Pre-compute L2-normed embeddings once.  Used both for BIRCH partial_fit
    # and the leaf-assignment step.
    logger.info("L2-normalising embeddings (once)")
    normed = L2(embeddings.astype(np.float32, copy=False))

    ahc_grid = _frange(*args.ahc_thresholds)
    birch_grid = _frange(*args.birch_thresholds)
    logger.info(
        "Sweeping %d AHC thresholds x %d BIRCH thresholds = %d cells",
        len(ahc_grid), len(birch_grid), len(ahc_grid) * len(birch_grid),
    )

    csv_path = os.path.join(work_dir, "tuning_results.csv")
    json_path = os.path.join(work_dir, "best_threshold.json")
    fieldnames = [
        "ahc_threshold", "birch_cosine_floor", "birch_radius",
        "n_leaf_subclusters", "num_pred_clusters", "num_true_speakers",
        "bcubed_precision", "bcubed_recall", "bcubed_f1",
        "purity", "inverse_purity",
        "nmi", "ari", "v_measure",
        "ahc_seconds", "birch_seconds",
    ]
    csv_f = open(csv_path, "w", encoding="utf-8", newline="")
    writer = csv.DictWriter(csv_f, fieldnames=fieldnames)
    writer.writeheader()

    rows: List[Dict[str, float]] = []
    best: Optional[Dict[str, float]] = None

    # Outer loop: BIRCH (expensive, refit per value).
    # Inner loop: AHC (cheap, just runs on the leaf centroids).
    for birch_cos in birch_grid:
        birch_radius = float(np.sqrt(2.0 * (1.0 - birch_cos)))
        logger.info(
            "[BIRCH] cosine_floor=%.4f -> radius=%.4f", birch_cos, birch_radius,
        )
        t0 = time.time()
        birch = ls._build_birch_tree(
            normed,
            birch_threshold=birch_radius,
            branching_factor=args.branching_factor,
            partial_fit_batch=args.partial_fit_batch,
        )
        leaf_centroids = np.asarray(birch.subcluster_centers_, dtype=np.float32)
        normed_centroids = L2(leaf_centroids)
        n_sub = normed_centroids.shape[0]
        logger.info("  BIRCH built %d leaf subclusters", n_sub)

        # Assign every utterance to its nearest leaf centroid (once per BIRCH).
        leaf_idx = ls._assign_to_nearest_leaf(
            normed, normed_centroids, tile=args.assign_tile,
        )
        birch_seconds = time.time() - t0

        # Pre-compute the leaf cosine distance matrix once (per BIRCH).
        # Then linkage is the only cost per AHC threshold.
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform
        sim = normed_centroids @ normed_centroids.T
        np.clip(sim, -1.0, 1.0, out=sim)
        dist_mat = 1.0 - sim
        condensed = squareform(dist_mat, checks=False)
        # Average linkage (Curator default).  We do NOT sweep linkage because
        # the user's question was specifically about thresholds.
        Z = linkage(condensed, method=args.linkage)

        for ahc_thr in ahc_grid:
            t1 = time.time()
            distance_cutoff = 1.0 - ahc_thr
            centroid_labels = fcluster(Z, t=distance_cutoff, criterion="distance")
            pred = centroid_labels[leaf_idx].astype(np.int64, copy=False)
            ahc_seconds = time.time() - t1

            metrics = compute_all_metrics(pred, true_speakers)
            row = {
                "ahc_threshold": ahc_thr,
                "birch_cosine_floor": birch_cos,
                "birch_radius": birch_radius,
                "n_leaf_subclusters": n_sub,
                "ahc_seconds": ahc_seconds,
                "birch_seconds": birch_seconds,
                **metrics,
            }
            writer.writerow({k: row.get(k, "") for k in fieldnames})
            csv_f.flush()
            rows.append(row)

            logger.info(
                "  AHC thr=%.3f  bF1=%.4f (P=%.4f R=%.4f)  "
                "NMI=%.4f ARI=%.4f  pred_K=%d / true_K=%d  "
                "(BIRCH=%.1fs, AHC=%.2fs)",
                ahc_thr,
                metrics["bcubed_f1"], metrics["bcubed_precision"], metrics["bcubed_recall"],
                metrics.get("nmi", float("nan")), metrics.get("ari", float("nan")),
                int(metrics["num_pred_clusters"]), int(metrics["num_true_speakers"]),
                birch_seconds, ahc_seconds,
            )

            if best is None or _is_better(row, best):
                best = row

    csv_f.close()

    if best is None:
        raise RuntimeError("Sweep produced no results")

    logger.info("=" * 60)
    logger.info("Best threshold:")
    logger.info("  AHC cosine threshold   : %.4f", best["ahc_threshold"])
    logger.info("  BIRCH cosine floor     : %.4f (radius=%.4f)",
                best["birch_cosine_floor"], best["birch_radius"])
    logger.info("  B-cubed F1             : %.4f", best["bcubed_f1"])
    logger.info("  B-cubed P / R          : %.4f / %.4f",
                best["bcubed_precision"], best["bcubed_recall"])
    logger.info("  NMI / ARI / V-measure  : %.4f / %.4f / %.4f",
                best.get("nmi", float("nan")),
                best.get("ari", float("nan")),
                best.get("v_measure", float("nan")))
    logger.info("  Predicted speakers     : %d (truth: %d)",
                int(best["num_pred_clusters"]), int(best["num_true_speakers"]))
    logger.info("=" * 60)

    summary = {
        "best": best,
        "config": {
            "n_utterances": int(n),
            "embedding_dim": int(d),
            "num_true_speakers": n_speakers_true,
            "embedding_normalization": args.embedding_normalization,
            "linkage": args.linkage,
            "min_cluster_size": 1,  # always disabled for tuning
            "ahc_thresholds": list(ahc_grid),
            "birch_cosine_floors": list(birch_grid),
            "branching_factor": args.branching_factor,
            "partial_fit_batch": args.partial_fit_batch,
            "assign_tile": args.assign_tile,
        },
        "csv_path": csv_path,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Wrote %s and %s", csv_path, json_path)


def _is_better(row: Dict[str, float], best: Dict[str, float]) -> bool:
    """Return True if ``row`` is strictly better than ``best``.

    Primary key: B-cubed F1.  Tiebreaker: NMI.  Final tiebreaker: |pred_K - true_K|
    (prefer cluster counts close to ground truth).
    """
    if row["bcubed_f1"] != best["bcubed_f1"]:
        return row["bcubed_f1"] > best["bcubed_f1"]
    rn, bn = row.get("nmi", -1.0), best.get("nmi", -1.0)
    if rn != bn:
        return rn > bn
    rk = abs(row["num_pred_clusters"] - row["num_true_speakers"])
    bk = abs(best["num_pred_clusters"] - best["num_true_speakers"])
    return rk < bk


# --------------------------------------------------------------------------
# 'all' subcommand  -- run extract + labels + tune in sequence
# --------------------------------------------------------------------------

def cmd_all(args: argparse.Namespace) -> None:
    cmd_extract(args)
    cmd_labels(args)
    cmd_tune(args)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    def _add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument(
            "--work_dir", type=str, default=DEFAULT_WORK_DIR,
            help="All artefacts live under this directory.",
        )

    manifest_help = (
        "NeMo brace-glob over per-shard manifests, e.g. "
        "'.../raw_sharded_manifests/manifest__OP_0..511_CL_.json'.  "
        "Defaults to the LibriSpeech tarred train manifest on the current "
        "cluster (cs-oci-* or draco-oci-*); set explicitly when running off-cluster."
    )
    tar_help = (
        "NeMo brace-glob over the matching tar shards, e.g. "
        "'.../audio__OP_0..511_CL_.tar'.  Defaults to the LibriSpeech "
        "tarred train tars on the current cluster."
    )

    # extract
    sp = sub.add_parser("extract", help="GPU: extract embeddings (delegates to run_pipeline.py)")
    _add_common(sp)
    sp.add_argument("--manifest_glob", type=str, default=DEFAULT_MANIFEST_GLOB,
                    help=manifest_help)
    sp.add_argument("--tar_glob", type=str, default=DEFAULT_TAR_GLOB,
                    help=tar_help)
    sp.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    sp.add_argument("--model_name", type=str, default=DEFAULT_MODEL)
    sp.add_argument("--max_cuts", type=int, default=None,
                    help="Cap utterances for a quick smoke test.")
    sp.add_argument("--force", action="store_true",
                    help="Re-run even if merged file already exists.")
    sp.set_defaults(func=cmd_extract)

    # labels
    sp = sub.add_parser("labels", help="CPU: build labels.npz (manifest -> speaker id)")
    _add_common(sp)
    sp.add_argument("--manifest_glob", type=str, default=DEFAULT_MANIFEST_GLOB,
                    help=manifest_help)
    sp.add_argument("--force", action="store_true",
                    help="Overwrite existing labels.npz.")
    sp.set_defaults(func=cmd_labels)

    # tune
    sp = sub.add_parser("tune", help="CPU: sweep thresholds on labelled embeddings")
    _add_common(sp)
    sp.add_argument(
        "--ahc_thresholds", type=_parse_range,
        default=DEFAULT_AHC_RANGE,
        help="AHC cosine threshold range as 'start,stop,step'.",
    )
    sp.add_argument(
        "--birch_thresholds", type=_parse_range,
        default=DEFAULT_BIRCH_COS_RANGE,
        help=(
            "BIRCH per-leaf cosine-similarity floor range, expressed as "
            "'start,stop,step' on cosine.  Internally converted to "
            "Euclidean radius via sqrt(2*(1-cos))."
        ),
    )
    sp.add_argument(
        "--linkage", type=str, default="average",
        choices=["average", "complete", "single"],
        help="SciPy AHC linkage method on the BIRCH leaf centroids.",
    )
    sp.add_argument(
        "--embedding_normalization", type=str, default="center_global",
        choices=["center_global", "none"],
        help="Pre-normalisation; matches Curator production default.",
    )
    sp.add_argument(
        "--branching_factor", type=int, default=50,
        help="BIRCH branching_factor.",
    )
    sp.add_argument(
        "--partial_fit_batch", type=int, default=50_000,
        help="Utterances per BIRCH partial_fit call.",
    )
    sp.add_argument(
        "--assign_tile", type=int, default=16_384,
        help="Tile size for utterance -> leaf assignment.",
    )
    sp.add_argument(
        "--speaker_id_repo", type=str, default=DEFAULT_SPEAKER_ID_REPO,
        help=(
            "DEPRECATED: legacy path to the speaker_id_for_asr_data repo. "
            "The BIRCH+AHC module now ships with Curator at "
            "nemo_curator.stages.audio.speaker_id.clustering."
        ),
    )
    sp.set_defaults(func=cmd_tune)

    # all
    sp = sub.add_parser("all", help="Run extract + labels + tune in sequence")
    _add_common(sp)
    sp.add_argument("--manifest_glob", type=str, default=DEFAULT_MANIFEST_GLOB,
                    help=manifest_help)
    sp.add_argument("--tar_glob", type=str, default=DEFAULT_TAR_GLOB,
                    help=tar_help)
    sp.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    sp.add_argument("--model_name", type=str, default=DEFAULT_MODEL)
    sp.add_argument("--max_cuts", type=int, default=None)
    sp.add_argument("--ahc_thresholds", type=_parse_range, default=DEFAULT_AHC_RANGE)
    sp.add_argument("--birch_thresholds", type=_parse_range, default=DEFAULT_BIRCH_COS_RANGE)
    sp.add_argument("--linkage", type=str, default="average",
                    choices=["average", "complete", "single"])
    sp.add_argument("--embedding_normalization", type=str, default="center_global",
                    choices=["center_global", "none"])
    sp.add_argument("--branching_factor", type=int, default=50)
    sp.add_argument("--partial_fit_batch", type=int, default=50_000)
    sp.add_argument("--assign_tile", type=int, default=16_384)
    sp.add_argument("--speaker_id_repo", type=str, default=DEFAULT_SPEAKER_ID_REPO)
    sp.add_argument("--force", action="store_true")
    sp.set_defaults(func=cmd_all)

    return p


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
