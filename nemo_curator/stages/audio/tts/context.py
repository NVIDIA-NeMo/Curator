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

"""Same-speaker context (voice prompt) selection.

For every target segment, candidates come from the same speaker (``speaker_cluster.speaker_id``, or a sample-level
field for corpora with known speakers such as LibriLight) within the same sample row. Candidates are walked in
descending TitaNet cosine similarity and the first one with similarity >= ``min_ssim``, duration >=
``min_duration``, text overlap <= ``max_text_overlap`` and complete tar metadata is selected.
"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from nemo_curator.stages.audio.tts.audio_io import (
    PNC_TEXT_FIELD,
    TAR_BYTES_FIELD,
    TAR_MEMBER_FIELD,
    TAR_OFFSET_FIELD,
    TN_RAW_FIELD,
    make_audio_config,
)
from nemo_curator.stages.audio.tts.fields import MISSING, get_dotted, set_dotted
from nemo_curator.stages.audio.tts.speaker_embedding import DEFAULT_TITANET_MODEL, TitaNetEmbedder, l2_normalize
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

MASKED_SIMILARITY_VALUE = -2.0
_CJK_RE = re.compile(r"[一-鿿㐀-䶿぀-ヿ가-힯]")
_MIN_SIZE_FOR_CONTAINMENT = 3


def empty_context(error: str) -> dict[str, Any]:
    return {
        "context_recording_id": None,
        "context_audio_duration": None,
        "context_audio_text": None,
        "context_audio_speaker_similarity": None,
        "context_audio_text_overlap": None,
        "context_audio_offset": None,
        "context_audio_bytes": None,
        "error": error,
    }


def normalize_text_for_overlap(text: str) -> str:
    """Casefold, strip punctuation and collapse whitespace before overlap scoring."""
    text = text.casefold()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize_for_overlap(text: str) -> list[str]:
    """Whitespace tokens; CJK / kana / hangul characters become single-character tokens."""
    tokens: list[str] = []
    for word in text.split():
        if not _CJK_RE.search(word):
            tokens.append(word)
            continue
        buffer: list[str] = []
        for char in word:
            if _CJK_RE.match(char):
                if buffer:
                    tokens.append("".join(buffer))
                    buffer = []
                tokens.append(char)
            else:
                buffer.append(char)
        if buffer:
            tokens.append("".join(buffer))
    return tokens


def word_ngrams(words: list[str], n: int) -> set[tuple[str, ...]]:
    if len(words) < n:
        return set()
    return {tuple(words[index : index + n]) for index in range(len(words) - n + 1)}


def compute_text_overlap(text_a: str, text_b: str) -> float:
    """Language-agnostic text overlap (max of unigram Jaccard, bigram Jaccard and containment).

    Both empty -> 1.0; exactly one empty -> 0.0. Containment ``|A∩B| / min(|A|,|B|)`` only counts when the smaller
    word set has at least 3 words.
    """
    words_a = tokenize_for_overlap(normalize_text_for_overlap(text_a))
    words_b = tokenize_for_overlap(normalize_text_for_overlap(text_b))
    if not words_a and not words_b:
        return 1.0
    if not words_a or not words_b:
        return 0.0
    set_a = set(words_a)
    set_b = set(words_b)
    intersection_size = len(set_a & set_b)
    union_size = len(set_a | set_b)
    scores: list[float] = []
    if union_size:
        scores.append(intersection_size / union_size)
    bigrams_a = word_ngrams(words_a, 2)
    bigrams_b = word_ngrams(words_b, 2)
    bigram_union = len(bigrams_a | bigrams_b)
    if bigram_union:
        scores.append(len(bigrams_a & bigrams_b) / bigram_union)
    min_size = min(len(set_a), len(set_b))
    if min_size >= _MIN_SIZE_FOR_CONTAINMENT:
        scores.append(intersection_size / min_size)
    return max(scores) if scores else 0.0


def rejection_reason(stats: dict[str, int]) -> str:
    """First reason that explains *all* candidates, in fixed priority order."""
    if stats["total"] == 0:
        return "no_context_segment_for_speaker"
    if stats["below_min_ssim"] == stats["total"]:
        return "no_context_audio_below_min_ssim"
    if stats["below_min_duration"] == stats["total"]:
        return "no_context_audio_below_min_duration"
    if stats["text_overlap"] == stats["total"]:
        return "no_context_audio_text_overlap"
    if stats["missing_metadata"] == stats["total"]:
        return "no_context_audio_missing_metadata"
    return "no_context_audio_after_filters"


def context_metadata(
    segment: dict[str, Any],
    text_field: str = PNC_TEXT_FIELD,
    extra_text_field: str | None = TN_RAW_FIELD,
) -> dict[str, Any] | None:
    """Context payload of a candidate, or ``None`` if tar member / offset / bytes / duration / text is missing."""
    recording_id = get_dotted(segment, TAR_MEMBER_FIELD, MISSING)
    offset = get_dotted(segment, TAR_OFFSET_FIELD, MISSING)
    size = get_dotted(segment, TAR_BYTES_FIELD, MISSING)
    duration = segment.get("duration", MISSING)
    text = get_dotted(segment, text_field, MISSING)
    if any(value is MISSING or value is None for value in (recording_id, offset, size, duration, text)):
        return None
    try:
        metadata = {
            "context_recording_id": str(recording_id),
            "context_audio_duration": float(duration),
            "context_audio_text": str(text),
            "context_audio_offset": int(offset),
            "context_audio_bytes": int(size),
        }
    except (TypeError, ValueError):
        return None
    if extra_text_field:
        extra = get_dotted(segment, extra_text_field)
        metadata["context_audio_tn_raw"] = extra if isinstance(extra, str) else None
    return metadata


def round_float(value: float | None, digits: int = 6) -> float | None:
    """Round to ``digits`` decimals, passing None through."""
    return None if value is None else round(float(value), digits)


def select_context(  # noqa: PLR0913
    target_position: int,
    target_indices: list[int],
    segments: list[dict[str, Any]],
    similarities: np.ndarray,
    pool_positions: list[int],
    *,
    min_ssim: float = 0.6,
    min_duration: float = 3.0,
    max_text_overlap: float | None = 0.5,
    text_field: str = PNC_TEXT_FIELD,
    extra_text_field: str | None = TN_RAW_FIELD,
) -> dict[str, Any]:
    """Pick the context for one target (row ``target_position`` of the speaker's similarity matrix).

    ``similarities`` is ``[n_targets, n_pool]`` with the target's own pool entry already masked to
    ``MASKED_SIMILARITY_VALUE``; ``pool_positions[j]`` is the target position of pool column ``j``.
    """
    target_text_value = get_dotted(segments[target_indices[target_position]], text_field, "")
    target_text = "" if target_text_value is None else str(target_text_value)
    # The target's own pool entry (if sampled into the pool) is the only masked value: cosine similarity >= -1.
    in_pool = bool(np.any(similarities[target_position] <= MASKED_SIMILARITY_VALUE))
    stats = {
        "total": len(pool_positions) - (1 if in_pool else 0),
        "below_min_ssim": 0,
        "below_min_duration": 0,
        "text_overlap": 0,
        "missing_metadata": 0,
    }
    for pool_position in np.argsort(-similarities[target_position], kind="stable"):
        similarity = float(similarities[target_position, int(pool_position)])
        if similarity <= MASKED_SIMILARITY_VALUE:
            continue
        if similarity < min_ssim:
            stats["below_min_ssim"] += 1
            continue
        candidate = segments[target_indices[pool_positions[int(pool_position)]]]
        try:
            candidate_duration = float(candidate.get("duration") or 0.0)
        except (TypeError, ValueError):
            candidate_duration = 0.0
        if candidate_duration < min_duration:
            stats["below_min_duration"] += 1
            continue
        candidate_text_value = get_dotted(candidate, text_field, "")
        candidate_text = "" if candidate_text_value is None else str(candidate_text_value)
        overlap = compute_text_overlap(target_text, candidate_text)
        if max_text_overlap is not None and overlap > max_text_overlap:
            stats["text_overlap"] += 1
            continue
        metadata = context_metadata(candidate, text_field, extra_text_field)
        if metadata is None:
            stats["missing_metadata"] += 1
            continue
        return {
            **metadata,
            "context_audio_speaker_similarity": round_float(similarity),
            "context_audio_text_overlap": round_float(overlap, digits=4),
            "error": None,
        }
    return empty_context(rejection_reason(stats))


def assign_contexts(  # noqa: PLR0913
    row: dict[str, Any],
    segments: list[dict[str, Any]],
    embeddings: list[np.ndarray | None],
    embedding_errors: list[str | None],
    *,
    output_key: str = "context",
    speaker_id_key: str = "speaker_cluster.speaker_id",
    speaker_id_scope: str = "segment",
    text_field: str = PNC_TEXT_FIELD,
    extra_text_field: str | None = TN_RAW_FIELD,
    min_ssim: float = 0.6,
    min_duration: float = 3.0,
    max_text_overlap: float | None = 0.5,
    max_speaker_items: int | None = 10000,
    random_seed: int = 12345,
) -> None:
    """Write a context annotation on every segment (most similar valid same-speaker candidate)."""
    if speaker_id_scope not in {"segment", "sample"}:
        msg = f"Unsupported speaker_id_scope: {speaker_id_scope}"
        raise ValueError(msg)
    groups: dict[str, list[int]] = defaultdict(list)
    for index, segment in enumerate(segments):
        if embedding_errors[index] is not None:
            set_dotted(segment, output_key, empty_context(str(embedding_errors[index])))
            continue
        if embeddings[index] is None:
            set_dotted(segment, output_key, empty_context("embedding_failed"))
            continue
        speaker_payload = row if speaker_id_scope == "sample" else segment
        speaker_id = get_dotted(speaker_payload, speaker_id_key, MISSING)
        if speaker_id is MISSING or speaker_id is None or str(speaker_id) == "":
            set_dotted(segment, output_key, empty_context("missing_speaker_id"))
            continue
        groups[str(speaker_id)].append(index)

    for target_indices in groups.values():
        normalized = l2_normalize(np.stack([embeddings[index] for index in target_indices]))
        pool_positions = list(range(len(target_indices)))
        if max_speaker_items is not None and len(pool_positions) > max_speaker_items:
            pool_positions = random.Random(random_seed).sample(pool_positions, max_speaker_items)  # noqa: S311
        similarities = normalized @ normalized[pool_positions].T
        for pool_position, target_position in enumerate(pool_positions):
            similarities[target_position, pool_position] = MASKED_SIMILARITY_VALUE
        for target_position, target_index in enumerate(target_indices):
            selected = select_context(
                target_position,
                target_indices,
                segments,
                similarities,
                pool_positions,
                min_ssim=min_ssim,
                min_duration=min_duration,
                max_text_overlap=max_text_overlap,
                text_field=text_field,
                extra_text_field=extra_text_field,
            )
            set_dotted(segments[target_index], output_key, selected)


def has_usable_context(segment: dict[str, Any], output_key: str = "context") -> bool:
    """Whether a context annotation is usable: no error and recording id / offset / bytes present."""
    context = get_dotted(segment, output_key)
    if not isinstance(context, dict):
        return False
    return context.get("error") is None and all(
        context.get(key) is not None for key in ("context_recording_id", "context_audio_offset", "context_audio_bytes")
    )


@dataclass
class ContextSelectionStage(ProcessingStage[AudioTask, AudioTask]):
    """Select a same-speaker context segment for every target segment of a Granary sample row.

    Embeddings are recomputed with TitaNet on the (post-filter) segments. Precomputed
    embeddings can be supplied per segment under ``embeddings_key`` (any length-D list); when every segment has
    one, TitaNet is not run for that row. The transient field is removed after use.

    Args:
        output_key: Where the context annotation is written (dotted path).
        speaker_id_key / speaker_id_scope: Speaker grouping field and whether it lives on the ``segment`` or the
            sample ``row`` (LibriLight: ``speaker_id`` / ``sample``).
        text_field: Text used for overlap and stored as ``context_audio_text`` (default ``GranaryV2.pnc_text``).
        extra_text_field: Additional context text copied as ``context_audio_tn_raw`` (used by training flags /
            SHAR); ``None`` disables it.
        drop_without_context: Remove segments without a usable context.
    """

    output_key: str = "context"
    speaker_id_key: str = "speaker_cluster.speaker_id"
    speaker_id_scope: str = "segment"
    text_field: str = PNC_TEXT_FIELD
    extra_text_field: str | None = TN_RAW_FIELD
    min_ssim: float = 0.6
    min_duration: float = 3.0
    max_text_overlap: float | None = 0.5
    max_speaker_items: int | None = 10000
    random_seed: int = 12345
    drop_without_context: bool = False
    embeddings_key: str | None = None
    model_name: str = DEFAULT_TITANET_MODEL
    model_path: str | None = None
    embedding_batch_size: int = 16
    tar_root: str | None = None
    strip_tar_key_prefixes: tuple[str, ...] = ()
    audio_source: str = "original"
    enhanced_key: str = "enhanced_audio"
    enhanced_fallback: bool = False
    embedder: Any | None = None
    name: str = "ContextSelection"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0, gpus=0.25))
    batch_size: int = 1

    def __post_init__(self) -> None:
        super().__init__()
        self.speaker_id_scope = self.speaker_id_scope.lower()
        if self.speaker_id_scope not in {"segment", "sample"}:
            msg = f"Unsupported speaker_id_scope: {self.speaker_id_scope}"
            raise ValueError(msg)
        self._audio_config = make_audio_config(self)

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], ["segments"]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], ["segments"]

    def _make_embedder(self) -> TitaNetEmbedder:
        if self.embedder is None:
            self.embedder = TitaNetEmbedder(
                model_name=self.model_name, model_path=self.model_path, batch_size=self.embedding_batch_size
            )
        return self.embedder

    def setup_on_node(self, _node_info: Any = None, _worker_metadata: Any = None) -> None:  # noqa: ANN401
        self._make_embedder().download()

    def setup(self, _worker_metadata: Any = None) -> None:  # noqa: ANN401
        self._make_embedder().load()

    def _precomputed(self, segments: list[dict[str, Any]]) -> list[np.ndarray] | None:
        if not self.embeddings_key:
            return None
        values = [get_dotted(segment, self.embeddings_key) for segment in segments]
        if any(value is None for value in values):
            return None
        return [np.asarray(value, dtype=np.float32) for value in values]

    def _embed(self, segments: list[dict[str, Any]]) -> tuple[list[np.ndarray | None], list[str | None]]:
        precomputed = self._precomputed(segments)
        if precomputed is not None:
            return list(precomputed), [None] * len(segments)
        embedder = self._make_embedder()
        if getattr(embedder, "model", None) is None:
            embedder.load()
        return embedder.embed_segments(segments, self._audio_config)

    def annotate_row(self, row: dict[str, Any]) -> None:
        raw_segments = row.get("segments")
        if not isinstance(raw_segments, list):
            return
        segments = [segment for segment in raw_segments if isinstance(segment, dict)]
        if segments:
            embeddings, errors = self._embed(segments)
            assign_contexts(
                row,
                segments,
                embeddings,
                errors,
                output_key=self.output_key,
                speaker_id_key=self.speaker_id_key,
                speaker_id_scope=self.speaker_id_scope,
                text_field=self.text_field,
                extra_text_field=self.extra_text_field,
                min_ssim=self.min_ssim,
                min_duration=self.min_duration,
                max_text_overlap=self.max_text_overlap,
                max_speaker_items=self.max_speaker_items,
                random_seed=self.random_seed,
            )
        if self.embeddings_key:
            for segment in segments:
                _pop_dotted(segment, self.embeddings_key)
        with_context = sum(has_usable_context(segment, self.output_key) for segment in segments)
        self._log_metrics(
            {"context_segments": float(with_context), "missing_context_segments": float(len(segments) - with_context)}
        )
        if self.drop_without_context:
            row["segments"] = [
                segment
                for segment in raw_segments
                if isinstance(segment, dict) and has_usable_context(segment, self.output_key)
            ]

    def process(self, task: AudioTask) -> AudioTask:
        self.annotate_row(task.data)
        return task

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        if len(tasks) == 0:
            return []
        for task in tasks:
            self.annotate_row(task.data)
        return list(tasks)


def _pop_dotted(payload: dict[str, Any], path: str) -> None:
    parts = path.split(".")
    current: Any = payload
    for part in parts[:-1]:
        if not isinstance(current, dict):
            return
        current = current.get(part)
    if isinstance(current, dict):
        current.pop(parts[-1], None)
