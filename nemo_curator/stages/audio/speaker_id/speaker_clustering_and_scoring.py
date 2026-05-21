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

"""Agglomerative Hierarchical Clustering (AHC) and per-utterance confidence
scoring for speaker embeddings.

This is a **CPU-only** stage.  It reads:

1. NeMo JSONL manifest files (the same ones fed to the embedding stage).
2. Per-shard embedding ``.npz`` files produced by
   :class:`~nemo_curator.stages.audio.speaker_id.speaker_embedding_lhotse.SpeakerEmbeddingLhotseStage`.

It verifies that every manifest line has a matching embedding, clusters the
utterances into speaker groups, and writes **new manifest files** (one per
input shard) with ``speaker_label`` and ``confidence_score`` fields added to
each JSON line.

Two clustering granularities are supported via ``shard_level_clustering``:

* ``False`` (default) -- all shards are loaded together and clustered
  globally.  Produces consistent speaker IDs across shards.
* ``True``  -- each shard is clustered independently.  Faster and lower
  memory, but speaker IDs are local to each shard.

**Embedding normalization** (``embedding_normalization``): default
``center_global`` subtracts the mean of the utterances being clustered (global
batch or per shard).  Use ``external`` with ``cohort_mean.npy`` / optional
``cohort_std.npy`` for fixed cohort stats (e.g. VoxCeleb).  Use ``none`` for
legacy raw-embedding geometry.

Threshold constants are derived from VoxCeleb1-O cleaned trials.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from loguru import logger
from tqdm.auto import tqdm

from nemo_curator.stages.audio.speaker_id.speaker_embedding_lhotse import (
    _tqdm_enabled,
    _worker_tag,
)

EmbeddingNormalization = Literal["none", "center_global", "external"]

from nemo_curator.backends.base import WorkerMetadata
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import EmptyTask, _EmptyTask

# SimAM_ResNet100 / legacy VoxCeleb1-O reference (different model family).
EER_THRESHOLD = 0.3483
# Default AHC cosine threshold for TitaNet + batch mean removal (``center_global``):
# local eval: TitaNet with cohort mean subtraction, cosine @ EER ≈ 0.291670
# (see tutorials/audio/speaker_id/TITANET_VS_WESPKResNet_benchmark.md).  Raise toward
# 0.35–0.40 if false merges (different speakers merged) hurt more than splits.
DEFAULT_THRESHOLD = 0.292


# ---------------------------------------------------------------------------
# Path helpers (shared with speaker_embedding_lhotse.py)
# ---------------------------------------------------------------------------

def _expand_nemo_path(path: str) -> list[str]:
    """Expand NeMo-style brace patterns (``_OP_0..49_CL_``) to file list."""
    for opener in ("(", "[", "<", "_OP_"):
        path = path.replace(opener, "{")
    for closer in (")", "]", ">", "_CL_"):
        path = path.replace(closer, "}")
    match = re.search(r"\{(\d+)\.\.(\d+)\}", path)
    if not match:
        return [path]
    start_idx, end_idx = int(match.group(1)), int(match.group(2))
    prefix, suffix = path[: match.start()], path[match.end() :]
    return [f"{prefix}{i}{suffix}" for i in range(start_idx, end_idx + 1)]


def _extract_shard_id(path: str) -> str:
    m = re.search(r"_(\d+)\.\w+$", os.path.basename(path))
    return m.group(1) if m else os.path.splitext(os.path.basename(path))[0]


# ---------------------------------------------------------------------------
# Linear-algebra helpers
# ---------------------------------------------------------------------------

def _l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return embeddings / norms


def normalize_embeddings_for_clustering(
    embeddings: np.ndarray,
    mode: EmbeddingNormalization = "center_global",
    *,
    external_mean_npy: str = "",
    external_std_npy: str = "",
    eps: float = 1e-8,
) -> np.ndarray:
    """Return embeddings prepared before AHC / confidence (cosine geometry).

    Cosine similarity internally L2-normalizes rows; this step applies optional
    **affine** transforms in raw space first.

    * ``none`` -- no change (legacy behaviour).
    * ``center_global`` -- subtract the **batch** mean vector (same utterances
      you cluster).  Good default for unlabeled web/audio without external stats.
    * ``external`` -- subtract ``cohort_mean.npy``; optionally divide by
      ``cohort_std.npy`` (e.g. VoxCeleb cohort from
      ``tutorials/audio/speaker_id/embedding_norm_stats``).

    Args:
        embeddings: ``(N, D)`` float array.
        mode: Normalization strategy.
        external_mean_npy: Path to ``(D,)`` mean (required if mode is ``external``).
        external_std_npy: Path to ``(D,)`` std; if empty, std scaling is skipped.
        eps: Added to std before division.

    Returns:
        ``(N, D)`` float32 array.
    """
    if embeddings.ndim != 2:
        raise ValueError(f"Expected (N, D) embeddings, got shape {embeddings.shape}")

    x = np.asarray(embeddings, dtype=np.float64)

    if mode == "none":
        pass
    elif mode == "center_global":
        x = x - x.mean(axis=0, keepdims=True)
    elif mode == "external":
        if not external_mean_npy or not os.path.isfile(external_mean_npy):
            raise ValueError(
                "embedding_normalization='external' requires a valid "
                f"external_mean_npy file; got {external_mean_npy!r}"
            )
        mean = np.load(external_mean_npy).astype(np.float64).reshape(-1)
        if mean.shape[0] != x.shape[1]:
            raise ValueError(
                f"Mean dim {mean.shape[0]} != embedding dim {x.shape[1]}"
            )
        x = x - mean
        if external_std_npy:
            if not os.path.isfile(external_std_npy):
                raise ValueError(f"external_std_npy not found: {external_std_npy}")
            std = np.load(external_std_npy).astype(np.float64).reshape(-1)
            if std.shape[0] != x.shape[1]:
                raise ValueError(
                    f"Std dim {std.shape[0]} != embedding dim {x.shape[1]}"
                )
            x = x / np.maximum(std, eps)
    else:
        raise ValueError(f"Unknown embedding_normalization: {mode!r}")

    return x.astype(np.float32)


def _cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    normed = _l2_normalize(embeddings)
    sim = normed @ normed.T
    np.clip(sim, -1.0, 1.0, out=sim)
    return sim


# ---------------------------------------------------------------------------
# Core clustering
# ---------------------------------------------------------------------------

def cluster_embeddings(
    embeddings: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
    linkage_method: str = "average",
) -> np.ndarray:
    """Cluster speaker embeddings via AHC with a cosine-similarity threshold.

    Returns:
        labels: ``(N,)`` int array of 1-based cluster IDs.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    n = embeddings.shape[0]
    if n <= 1:
        return np.ones(n, dtype=int)

    dist_mat = 1.0 - _cosine_similarity_matrix(embeddings)
    condensed = squareform(dist_mat, checks=False)

    Z = linkage(condensed, method=linkage_method)
    distance_cutoff = 1.0 - threshold
    labels = fcluster(Z, t=distance_cutoff, criterion="distance")
    return labels


def cluster_stats(labels: np.ndarray) -> dict:
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


# ---------------------------------------------------------------------------
# Per-utterance confidence
# ---------------------------------------------------------------------------

def speaker_confidence(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Compute a per-utterance speaker-ID confidence score in [0, 1].

    Silhouette-style metric in cosine-similarity space.
    Singletons get confidence = 0.0.
    """
    n = len(labels)
    sim_mat = _cosine_similarity_matrix(embeddings)

    cluster_indices: dict[int, list[int]] = defaultdict(list)
    for i, lab in enumerate(labels):
        cluster_indices[lab].append(i)

    unique_labels = sorted(cluster_indices.keys())
    label_to_k = {lab: k for k, lab in enumerate(unique_labels)}
    K = len(unique_labels)

    membership = np.zeros((n, K), dtype=np.float32)
    cluster_sizes = np.zeros(K, dtype=np.float32)
    for lab, idxs in cluster_indices.items():
        k = label_to_k[lab]
        membership[idxs, k] = 1.0
        cluster_sizes[k] = len(idxs)

    mean_sim = (sim_mat @ membership) / np.maximum(cluster_sizes, 1.0)

    scores = np.zeros(n, dtype=np.float32)
    for i in range(n):
        my_k = label_to_k[labels[i]]
        my_size = cluster_sizes[my_k]

        if my_size < 2:
            continue

        cohesion = (mean_sim[i, my_k] * my_size - 1.0) / (my_size - 1.0)

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


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _log_cluster_summary(labels: np.ndarray, threshold: float) -> None:
    stats = cluster_stats(labels)
    logger.info(
        f"AHC (threshold={threshold:.4f}): "
        f"{len(labels):,} utterances -> {stats['num_clusters']:,} speakers "
        f"(largest={stats['largest_cluster']:,}, singletons={stats['singletons']:,})"
    )


# ---------------------------------------------------------------------------
# I/O: load embeddings, read/write manifests
# ---------------------------------------------------------------------------

def _load_embedding_shard(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load ``(cut_ids, embeddings)`` from one ``.npz`` file."""
    data = np.load(path, allow_pickle=True)
    return data["cut_ids"], data["embeddings"]


def _read_manifest(path: str) -> list[dict]:
    """Read a NeMo JSONL manifest, return list of dicts."""
    lines = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(json.loads(line))
    return lines


def _write_manifest(path: str, entries: list[dict]) -> None:
    """Write a NeMo JSONL manifest."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _verify_manifest_vs_embeddings(
    manifest_entries: list[dict],
    cut_ids: np.ndarray,
    manifest_path: str,
    audio_filepath_key: str = "audio_filepath",
) -> dict[str, int]:
    """Build a cut_id -> embedding index map and verify coverage.

    Raises if any manifest line is missing from the embeddings.
    Returns mapping from audio_filepath to embedding array index.
    """
    id_to_idx = {str(cid): i for i, cid in enumerate(cut_ids)}

    missing = []
    mapping: dict[str, int] = {}
    for line_num, entry in enumerate(manifest_entries):
        afp = entry.get(audio_filepath_key, "")
        if afp in id_to_idx:
            mapping[afp] = id_to_idx[afp]
        else:
            missing.append((line_num, afp))

    if missing:
        sample = missing[:5]
        logger.warning(
            f"{len(missing)} manifest lines in {manifest_path} have no matching "
            f"embedding (will be assigned speaker_label=-1). First few: {sample}"
        )

    return mapping


# ---------------------------------------------------------------------------
# NeMo Curator ProcessingStage
# ---------------------------------------------------------------------------

@dataclass
class SpeakerClusteringStage(ProcessingStage[_EmptyTask, _EmptyTask]):
    """Cluster speaker embeddings and annotate manifests with speaker IDs.

    Reads NeMo JSONL manifests and per-shard ``.npz`` embedding files,
    verifies every manifest line has a matching embedding, clusters via AHC,
    and writes new manifest files with ``speaker_label`` and
    ``confidence_score`` fields added.

    **CPU-only** -- no GPU required.

    Args:
        input_manifest: NeMo manifest path pattern (brace-expand).
        embedding_dir: Directory with ``embeddings_*.npz`` files.
        output_manifest_dir: Where to write annotated manifests.
        threshold: Cosine-similarity cutoff for same-speaker decisions.
        linkage_method: ``"average"``, ``"complete"``, or ``"single"``.
        shard_level_clustering: If ``False`` (default), cluster all shards
            globally.  If ``True``, cluster each shard independently.
            Ignored when ``batch_size`` is set.
        batch_size: Number of shards to cluster together per group.
            ``None`` (default) falls back to ``shard_level_clustering``
            behaviour (all-global or per-shard).  ``1`` is equivalent to
            ``shard_level_clustering=True``.  Values > 1 cluster shards in
            groups of this size -- speaker labels are offset per group so
            they remain globally unique across the full dataset.
        audio_filepath_key: Manifest key used as the cut ID.
        embedding_normalization: ``none`` | ``center_global`` | ``external``.
            ``center_global`` (default) subtracts the mean of the utterances being
            clustered (per shard if ``shard_level_clustering``, else over the
            merged batch).  ``external`` loads fixed stats (e.g. VoxCeleb cohort).
        external_norm_mean_npy: Path to ``(D,)`` ``cohort_mean.npy`` when using
            ``external``.
        external_norm_std_npy: Optional ``(D,)`` ``cohort_std.npy`` for
            per-dim scaling after centering.
        norm_eps: Small constant added to std before division.
    """

    name: str = "SpeakerClusteringStage"
    input_manifest: str = ""
    embedding_dir: str = "embeddings"
    output_manifest_dir: str = "output_manifests"
    threshold: float = DEFAULT_THRESHOLD
    linkage_method: str = "average"
    shard_level_clustering: bool = False
    batch_size: int | None = None
    audio_filepath_key: str = "audio_filepath"
    embedding_normalization: EmbeddingNormalization = "center_global"
    external_norm_mean_npy: str = ""
    external_norm_std_npy: str = ""
    norm_eps: float = 1e-8
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))
    show_progress: bool = True

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        pass

    def _load_shard_pair(
        self, manifest_path: str,
    ) -> tuple[list[dict], np.ndarray, np.ndarray, dict[str, int]]:
        """Load one manifest + its matching embedding file, verify alignment."""
        shard_id = _extract_shard_id(manifest_path)
        emb_path = os.path.join(self.embedding_dir, f"embeddings_{shard_id}.npz")

        if not os.path.isfile(emb_path):
            raise FileNotFoundError(
                f"Embedding file {emb_path} not found for manifest {manifest_path}"
            )

        manifest_entries = _read_manifest(manifest_path)
        cut_ids, embeddings = _load_embedding_shard(emb_path)

        mapping = _verify_manifest_vs_embeddings(
            manifest_entries, cut_ids, manifest_path, self.audio_filepath_key
        )

        return manifest_entries, cut_ids, embeddings, mapping

    def _normalize(self, embeddings: np.ndarray) -> np.ndarray:
        return normalize_embeddings_for_clustering(
            embeddings,
            self.embedding_normalization,
            external_mean_npy=self.external_norm_mean_npy,
            external_std_npy=self.external_norm_std_npy,
            eps=self.norm_eps,
        )

    # ------------------------------------------------------------------
    # Shard-level clustering
    # ------------------------------------------------------------------

    def _process_shard_level(self, manifest_paths: list[str]) -> int:
        os.makedirs(self.output_manifest_dir, exist_ok=True)
        total = 0
        bar_disable = not _tqdm_enabled(self.show_progress)

        with tqdm(
            total=len(manifest_paths),
            desc=f"clustering shards [{_worker_tag()}]",
            unit="shard",
            dynamic_ncols=True,
            disable=bar_disable,
        ) as bar:
            for idx, mp in enumerate(manifest_paths, start=1):
                manifest_entries, cut_ids, embeddings, mapping = self._load_shard_pair(mp)
                n = embeddings.shape[0]
                if n == 0:
                    logger.warning(f"Shard {mp} has 0 embeddings, skipping")
                    bar.update(1)
                    continue

                embs = self._normalize(embeddings)
                labels = cluster_embeddings(embs, self.threshold, self.linkage_method)
                scores = speaker_confidence(embs, labels)
                _log_cluster_summary(labels, self.threshold)

                for entry in manifest_entries:
                    afp = entry[self.audio_filepath_key]
                    emb_idx = mapping.get(afp)
                    if emb_idx is not None:
                        entry["speaker_label"] = int(labels[emb_idx])
                        entry["confidence_score"] = round(float(scores[emb_idx]), 6)
                    else:
                        entry["speaker_label"] = -1
                        entry["confidence_score"] = 0.0

                out_path = os.path.join(
                    self.output_manifest_dir, os.path.basename(mp)
                )
                _write_manifest(out_path, manifest_entries)
                total += n
                logger.info(
                    f"  Shard {idx}/{len(manifest_paths)}: "
                    f"{os.path.basename(mp)} -> {out_path}"
                )
                bar.set_postfix(utts=total, last_n_clusters=int(labels.max() + 1))
                bar.update(1)

        return total

    # ------------------------------------------------------------------
    # Global clustering
    # ------------------------------------------------------------------

    def _process_global(self, manifest_paths: list[str]) -> int:
        os.makedirs(self.output_manifest_dir, exist_ok=True)
        bar_disable = not _tqdm_enabled(self.show_progress)

        shard_data: list[tuple[str, list[dict], np.ndarray, dict[str, int]]] = []
        all_embs: list[np.ndarray] = []

        for mp in tqdm(
            manifest_paths,
            desc=f"loading shards [{_worker_tag()}]",
            unit="shard",
            dynamic_ncols=True,
            disable=bar_disable,
        ):
            manifest_entries, cut_ids, embeddings, mapping = self._load_shard_pair(mp)
            if embeddings.shape[0] == 0:
                logger.warning(f"Shard {mp} has 0 embeddings, skipping")
                continue
            shard_data.append((mp, manifest_entries, embeddings, mapping))
            all_embs.append(embeddings)

        if not all_embs:
            logger.warning("No embeddings found, nothing to cluster")
            return 0

        merged_embs = np.concatenate(all_embs)
        logger.info(
            f"Global clustering: {merged_embs.shape[0]:,} utterances "
            f"from {len(shard_data)} shards"
        )

        # AHC + confidence are single SciPy/NumPy calls on the merged matrix;
        # they cannot be tqdm-tracked without rewriting the algorithm. They are
        # the bulk of CPU time for global mode.
        logger.info("  AHC clustering (no progress bar -- single SciPy call) ...")
        merged_embs = self._normalize(merged_embs)
        labels = cluster_embeddings(merged_embs, self.threshold, self.linkage_method)
        logger.info("  Computing per-utterance confidence scores ...")
        scores = speaker_confidence(merged_embs, labels)
        _log_cluster_summary(labels, self.threshold)

        offset = 0
        total = 0
        for mp, manifest_entries, embeddings, mapping in tqdm(
            shard_data,
            desc=f"writing manifests [{_worker_tag()}]",
            unit="shard",
            dynamic_ncols=True,
            disable=bar_disable,
        ):
            n = embeddings.shape[0]
            shard_labels = labels[offset:offset + n]
            shard_scores = scores[offset:offset + n]
            offset += n

            for entry in manifest_entries:
                afp = entry[self.audio_filepath_key]
                emb_idx = mapping.get(afp)
                if emb_idx is not None:
                    entry["speaker_label"] = int(shard_labels[emb_idx])
                    entry["confidence_score"] = round(float(shard_scores[emb_idx]), 6)
                else:
                    entry["speaker_label"] = -1
                    entry["confidence_score"] = 0.0

            out_path = os.path.join(
                self.output_manifest_dir, os.path.basename(mp)
            )
            _write_manifest(out_path, manifest_entries)
            total += n

        return total

    # ------------------------------------------------------------------
    # Grouped (batched) clustering
    # ------------------------------------------------------------------

    def _process_grouped(
        self, manifest_paths: list[str], batch_size: int,
    ) -> int:
        """Cluster shards in groups of ``batch_size``.

        Speaker labels are offset per group so they remain globally unique
        across the full dataset.  Each group is clustered independently
        using the same AHC logic as ``_process_global``.
        """
        os.makedirs(self.output_manifest_dir, exist_ok=True)
        total = 0
        label_offset = 0
        num_groups = (len(manifest_paths) + batch_size - 1) // batch_size
        bar_disable = not _tqdm_enabled(self.show_progress)

        with tqdm(
            total=num_groups,
            desc=f"clustering groups [{_worker_tag()}]",
            unit="group",
            dynamic_ncols=True,
            disable=bar_disable,
        ) as group_bar:
            for group_idx in range(num_groups):
                start = group_idx * batch_size
                end = min(start + batch_size, len(manifest_paths))
                group_paths = manifest_paths[start:end]

                logger.info(
                    f"Group {group_idx + 1}/{num_groups}: "
                    f"shards {start}-{end - 1} ({len(group_paths)} shards)"
                )

                shard_data: list[tuple[str, list[dict], np.ndarray, dict[str, int]]] = []
                all_embs: list[np.ndarray] = []

                for mp in tqdm(
                    group_paths,
                    desc=f"  group {group_idx + 1}/{num_groups}: loading",
                    unit="shard",
                    leave=False,
                    dynamic_ncols=True,
                    disable=bar_disable,
                ):
                    try:
                        manifest_entries, cut_ids, embeddings, mapping = (
                            self._load_shard_pair(mp)
                        )
                    except FileNotFoundError:
                        logger.warning(f"Missing embeddings for {mp}, skipping")
                        continue
                    if embeddings.shape[0] == 0:
                        logger.warning(f"Shard {mp} has 0 embeddings, skipping")
                        continue
                    shard_data.append((mp, manifest_entries, embeddings, mapping))
                    all_embs.append(embeddings)

                if not all_embs:
                    logger.warning(f"Group {group_idx + 1}: no embeddings, skipping")
                    group_bar.update(1)
                    continue

                merged_embs = np.concatenate(all_embs)
                logger.info(
                    f"  Clustering {merged_embs.shape[0]:,} utterances "
                    f"from {len(shard_data)} shards"
                )

                merged_embs = self._normalize(merged_embs)
                labels = cluster_embeddings(
                    merged_embs, self.threshold, self.linkage_method,
                )
                scores = speaker_confidence(merged_embs, labels)
                _log_cluster_summary(labels, self.threshold)

                # Offset labels so they don't collide across groups.
                labels = labels + label_offset
                label_offset = int(labels.max())

                offset = 0
                for mp, manifest_entries, embeddings, mapping in tqdm(
                    shard_data,
                    desc=f"  group {group_idx + 1}/{num_groups}: writing",
                    unit="shard",
                    leave=False,
                    dynamic_ncols=True,
                    disable=bar_disable,
                ):
                    n = embeddings.shape[0]
                    shard_labels = labels[offset:offset + n]
                    shard_scores = scores[offset:offset + n]
                    offset += n

                    for entry in manifest_entries:
                        afp = entry[self.audio_filepath_key]
                        emb_idx = mapping.get(afp)
                        if emb_idx is not None:
                            entry["speaker_label"] = int(shard_labels[emb_idx])
                            entry["confidence_score"] = round(
                                float(shard_scores[emb_idx]), 6,
                            )
                        else:
                            entry["speaker_label"] = -1
                            entry["confidence_score"] = 0.0

                    out_path = os.path.join(
                        self.output_manifest_dir, os.path.basename(mp),
                    )
                    _write_manifest(out_path, manifest_entries)
                    total += n

                group_bar.set_postfix(utts=total, max_label=label_offset)
                group_bar.update(1)

        return total

    # ------------------------------------------------------------------
    # Stage entry point
    # ------------------------------------------------------------------

    def _resolve_mode(self) -> str:
        """Return ``"shard"`` | ``"grouped"`` | ``"global"``."""
        if self.batch_size is not None:
            if self.batch_size < 1:
                raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
            if self.batch_size == 1:
                return "shard"
            return "grouped"
        return "shard" if self.shard_level_clustering else "global"

    def process(self, _: _EmptyTask) -> _EmptyTask:
        manifest_paths = _expand_nemo_path(self.input_manifest.strip())
        mode = self._resolve_mode()

        mode_desc = mode
        if mode == "grouped":
            mode_desc = f"grouped (batch_size={self.batch_size})"
        logger.info(
            f"SpeakerClusteringStage: {len(manifest_paths)} manifest shards, "
            f"mode={mode_desc}, threshold={self.threshold:.4f}, "
            f"linkage={self.linkage_method}, "
            f"embedding_normalization={self.embedding_normalization!r}"
        )
        if self.embedding_normalization == "external":
            logger.info(
                f"  external mean: {self.external_norm_mean_npy or '(missing)'} "
                f"std: {self.external_norm_std_npy or '(none)'}"
            )

        if mode == "shard":
            total = self._process_shard_level(manifest_paths)
        elif mode == "grouped":
            total = self._process_grouped(manifest_paths, self.batch_size)
        else:
            total = self._process_global(manifest_paths)

        logger.info(
            f"Done. {total:,} utterances clustered -> "
            f"{self.output_manifest_dir}"
        )
        return EmptyTask
