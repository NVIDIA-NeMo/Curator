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

"""Per-recording TitaNet speaker clustering.

Within one sample row (one source recording),
single-speaker segments are embedded with TitaNet and grouped by average-linkage agglomerative clustering on cosine
distance. Each segment gets ``<sample_id>_speaker_<k>``, its cosine similarity to its own cluster centroid and the
margin to the nearest other centroid. Context selection later draws prompts from the same cluster.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from loguru import logger

from nemo_curator.stages.audio.tts.audio_io import make_audio_config, sample_id
from nemo_curator.stages.audio.tts.fields import get_dotted, set_dotted
from nemo_curator.stages.audio.tts.speaker_embedding import DEFAULT_TITANET_MODEL, TitaNetEmbedder, l2_normalize
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

if TYPE_CHECKING:
    from nemo_curator.backends.base import NodeInfo, WorkerMetadata

MISSING_NUM_SPEAKERS = "missing_num_speakers"
INVALID_NUM_SPEAKERS = "invalid_num_speakers"
NOT_SINGLE_SPEAKER = "not_single_speaker"
NOT_PROCESSED = "not_processed"
CLUSTERING_FAILED = "clustering_failed"


def empty_annotation(error: str | None) -> dict[str, Any]:
    return {"speaker_id": None, "cluster_similarity": None, "cluster_margin": None, "error": error}


def _round(value: float | None, digits: int = 6) -> float | None:
    return None if value is None else round(float(value), digits)


def make_clusterer(distance_threshold: float) -> Any:  # noqa: ANN401
    """Agglomerative cosine clusterer with ``distance_threshold``."""
    from sklearn.cluster import AgglomerativeClustering

    try:
        return AgglomerativeClustering(
            n_clusters=None, metric="cosine", linkage="average", distance_threshold=distance_threshold
        )
    except TypeError:  # scikit-learn < 1.2
        return AgglomerativeClustering(
            n_clusters=None, affinity="cosine", linkage="average", distance_threshold=distance_threshold
        )


def cluster_embeddings(
    embeddings: np.ndarray, threshold: float = 0.4
) -> tuple[np.ndarray, list[float], list[float | None]]:
    """Cluster ``[N, D]`` embeddings.

    Returns ``(labels, similarities, margins)``: labels are renumbered ``0..K-1`` in ascending order of the raw
    cluster labels; similarity is the cosine to the segment's own (re-normalised mean) centroid; margin is that
    similarity minus the best similarity to any other centroid (``None`` when there is a single cluster).
    """
    normed = l2_normalize(np.asarray(embeddings, dtype=np.float32))
    if normed.shape[0] == 0:
        return np.zeros(0, dtype=np.int64), [], []
    raw = np.zeros(1, dtype=np.int64) if normed.shape[0] == 1 else make_clusterer(threshold).fit_predict(normed)
    raw = np.asarray(raw, dtype=np.int64)
    ordered = sorted(int(v) for v in np.unique(raw))
    remap = {label: index for index, label in enumerate(ordered)}
    labels = np.array([remap[int(v)] for v in raw], dtype=np.int64)

    centroids = l2_normalize(np.stack([normed[labels == k].mean(axis=0) for k in range(len(ordered))]))
    sims = normed @ centroids.T
    similarities: list[float] = []
    margins: list[float | None] = []
    for row, label in enumerate(labels):
        own = float(sims[row, label])
        others = [float(sims[row, k]) for k in range(len(ordered)) if k != label]
        similarities.append(own)
        margins.append(own - max(others) if others else None)
    return labels, similarities, margins


@dataclass
class SpeakerClusteringStage(ProcessingStage[AudioTask, AudioTask]):
    """Cluster the single-speaker segments of each recording by TitaNet speaker embeddings.

    Only segments whose diarization annotation reports exactly one speaker are embedded; others get an annotation
    with ``speaker_id=None`` and an error (``missing_num_speakers`` / ``invalid_num_speakers`` /
    ``not_single_speaker``). Requires sample rows with a ``segments`` list (one row = one recording).

    Args:
        model_name / model_path: TitaNet checkpoint (``from_pretrained`` name or local ``.nemo``).
        embedding_batch_size: segments per TitaNet forward pass.
        distance_threshold: cosine distance threshold of average-linkage clustering.
        num_speakers_key: dotted path of the diarization speaker count on each segment.
        speaker_id_source: row key used as speaker id prefix (falls back to sample_id/id/audio_id/audio_filepath).
        output_key: dotted path written on each segment.
    """

    model_name: str = DEFAULT_TITANET_MODEL
    model_path: str | None = None
    embedder: TitaNetEmbedder | None = None
    embedding_batch_size: int = 32
    distance_threshold: float = 0.4
    num_speakers_key: str = "diarization.num_speakers"
    speaker_id_source: str = "orig_sample_id"
    output_key: str = "speaker_cluster"
    tar_root: str | None = None
    strip_tar_key_prefixes: tuple[str, ...] = ()
    audio_source: str = "original"
    enhanced_key: str = "enhanced_audio"
    enhanced_fallback: bool = False
    name: str = "SpeakerClustering"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0, gpus=1.0))
    batch_size: int = 1

    def __post_init__(self) -> None:
        if not 0.0 < self.distance_threshold <= 2.0:  # noqa: PLR2004
            msg = "distance_threshold must be in (0, 2] for cosine distance"
            raise ValueError(msg)
        self._audio_config = make_audio_config(self)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def _get_embedder(self) -> TitaNetEmbedder:
        if self.embedder is None:
            self.embedder = TitaNetEmbedder(
                model_name=self.model_name, model_path=self.model_path, batch_size=self.embedding_batch_size
            )
        return self.embedder

    def setup_on_node(
        self, _node_info: NodeInfo | None = None, _worker_metadata: WorkerMetadata | None = None
    ) -> None:
        self._get_embedder().download()

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        self._get_embedder().load()

    def _eligible(self, segments: list[Any]) -> list[dict[str, Any]]:
        eligible: list[dict[str, Any]] = []
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            num_speakers = get_dotted(segment, self.num_speakers_key)
            if num_speakers is None:
                set_dotted(segment, self.output_key, empty_annotation(MISSING_NUM_SPEAKERS))
                continue
            try:
                count = int(num_speakers)
            except (TypeError, ValueError):
                set_dotted(segment, self.output_key, empty_annotation(INVALID_NUM_SPEAKERS))
                continue
            if count != 1:
                set_dotted(segment, self.output_key, empty_annotation(NOT_SINGLE_SPEAKER))
                continue
            set_dotted(segment, self.output_key, empty_annotation(NOT_PROCESSED))
            eligible.append(segment)
        return eligible

    def annotate_row(self, row: dict[str, Any]) -> None:
        segments = row.get("segments")
        if not isinstance(segments, list):
            return
        eligible = self._eligible(segments)
        if not eligible:
            return
        embeddings, errors = self._get_embedder().embed_segments(eligible, self._audio_config)
        valid: list[dict[str, Any]] = []
        vectors: list[np.ndarray] = []
        for segment, vector, error in zip(eligible, embeddings, errors, strict=True):
            if error is not None or vector is None:
                set_dotted(segment, self.output_key, empty_annotation(error or "embedding_failed"))
                continue
            valid.append(segment)
            vectors.append(vector)
        if not valid:
            return
        prefix = str(row.get(self.speaker_id_source) or sample_id(row))
        try:
            labels, similarities, margins = cluster_embeddings(np.stack(vectors), self.distance_threshold)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Speaker clustering failed for {}: {}", prefix, exc)
            for segment in valid:
                set_dotted(segment, self.output_key, empty_annotation(CLUSTERING_FAILED))
            return
        for segment, label, similarity, margin in zip(valid, labels, similarities, margins, strict=True):
            set_dotted(
                segment,
                self.output_key,
                {
                    "speaker_id": f"{prefix}_speaker_{int(label)}",
                    "cluster_similarity": _round(similarity),
                    "cluster_margin": _round(margin),
                    "error": None,
                },
            )

    def process(self, task: AudioTask) -> AudioTask:
        self.annotate_row(task.data)
        return task
