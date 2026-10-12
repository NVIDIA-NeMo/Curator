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

"""Per-segment streaming Sortformer diarization.

Every Granary segment is diarized on its own and
annotated with the diarized speech rate and speaker count used by the common segment filter and by speaker
clustering. Rows and segments are never dropped; failures are recorded in ``error``.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from loguru import logger

from nemo_curator.stages.audio.tts.audio_io import (
    SegmentAudioError,
    iter_segments,
    load_segment_audio,
    make_audio_config,
)
from nemo_curator.stages.audio.tts.fields import set_dotted
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

if TYPE_CHECKING:
    from collections.abc import Iterable

    import numpy as np

    from nemo_curator.backends.base import NodeInfo, WorkerMetadata

DIARIZATION_FAILED = "diarization_failed"
DEFAULT_SORTFORMER_MODEL = "nvidia/diar_streaming_sortformer_4spk-v2.1"
_SEGMENT_FIELDS = 3


def parse_diarization_output(raw_segments: Iterable[Any]) -> list[dict[str, Any]]:
    """Convert Sortformer output (``"begin end speaker"`` strings or objects) to ``{start_time, end_time, speaker}`` dicts."""
    segments: list[dict[str, Any]] = []
    for raw in raw_segments:
        if isinstance(raw, str):
            parts = raw.strip().split(maxsplit=2)
            if len(parts) != _SEGMENT_FIELDS:
                msg = f"Unexpected diarization segment format: {raw!r}"
                raise ValueError(msg)
            start, end, speaker = parts
        elif isinstance(raw, (list, tuple)) and len(raw) >= _SEGMENT_FIELDS:
            start, end, speaker = raw[0], raw[1], raw[2]
        elif hasattr(raw, "start") and hasattr(raw, "end"):
            start, end, speaker = raw.start, raw.end, getattr(raw, "speaker", getattr(raw, "label", "unknown"))
        else:
            msg = f"Unexpected diarization segment format: {raw!r}"
            raise ValueError(msg)
        segments.append({"start_time": float(start), "end_time": float(end), "speaker": str(speaker)})
    return segments


def union_duration(intervals: Iterable[tuple[float, float]]) -> float:
    """Total length of the union of ``(start, end)`` intervals (overlaps merged), rounded to 6 decimals."""
    spans = sorted((float(s), float(e)) for s, e in intervals if float(e) > float(s))
    if not spans:
        return 0.0
    total = 0.0
    current_start, current_end = spans[0]
    for start, end in spans[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
            continue
        total += current_end - current_start
        current_start, current_end = start, end
    total += current_end - current_start
    return round(total, 6)


def summarize_diarization(diarized_segments: list[dict[str, Any]], duration: float | None) -> dict[str, Any]:
    """Build the segment annotation from parsed diarization segments and the Granary segment duration."""
    speech_duration = union_duration((seg["start_time"], seg["end_time"]) for seg in diarized_segments)
    rate = None if duration is None or float(duration) <= 0 else round(speech_duration / float(duration), 6)
    return {
        "diarized_segments": diarized_segments,
        "num_speakers": len({str(seg["speaker"]) for seg in diarized_segments}),
        "diarized_speech_duration": speech_duration,
        "diarized_speech_rate": rate,
        "error": None,
    }


def empty_annotation(error: str) -> dict[str, Any]:
    return {
        "diarized_segments": None,
        "num_speakers": None,
        "diarized_speech_duration": None,
        "diarized_speech_rate": None,
        "error": error,
    }


@dataclass
class SegmentDiarizationStage(ProcessingStage[AudioTask, AudioTask]):
    """Diarize every Granary segment with streaming Sortformer and annotate speech rate / speaker count.

    Args:
        model_name: Hugging Face repo with the Sortformer ``.nemo`` (used when ``model_path`` is unset).
        model_path: Local ``.nemo`` checkpoint (overrides ``model_name``).
        input_mode: ``"wav"`` (default) writes a temporary PCM_16 WAV per segment;
            ``"numpy"`` passes float 16 kHz waveforms in memory (batched by ``inference_batch_size``, no temp
            files); results can differ slightly from the WAV path.
        chunk_len, chunk_right_context, fifo_len, spkcache_update_period, spkcache_len: streaming
            parameters set on ``model.sortformer_modules`` (defaults 340/40/40/300/188).
        chunk_left_context: optional; left untouched when ``None``.
        output_key: dotted path written on each segment.
        tar_root, strip_tar_key_prefixes: resolution of relative ``tar_key`` values.
    """

    model_name: str = DEFAULT_SORTFORMER_MODEL
    model_path: str | None = None
    cache_dir: str | None = None
    diar_model: Any | None = None
    input_mode: str = "wav"
    sample_rate: int = 16000
    inference_batch_size: int = 1
    chunk_len: int = 340
    chunk_right_context: int = 40
    fifo_len: int = 40
    spkcache_update_period: int = 300
    spkcache_len: int = 188
    chunk_left_context: int | None = None
    output_key: str = "diarization"
    tar_root: str | None = None
    strip_tar_key_prefixes: tuple[str, ...] = ()
    audio_source: str = "original"
    enhanced_key: str = "enhanced_audio"
    enhanced_fallback: bool = False
    name: str = "SegmentDiarization"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0, gpus=1.0))
    batch_size: int = 1

    def __post_init__(self) -> None:
        if self.input_mode not in ("numpy", "wav"):
            msg = f"input_mode must be 'numpy' or 'wav', got {self.input_mode!r}"
            raise ValueError(msg)
        if self.inference_batch_size < 1:
            msg = "inference_batch_size must be >= 1"
            raise ValueError(msg)
        self._audio_config = make_audio_config(self)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def setup_on_node(
        self, _node_info: NodeInfo | None = None, _worker_metadata: WorkerMetadata | None = None
    ) -> None:
        if self.model_path is None and self.diar_model is None:
            from huggingface_hub import snapshot_download

            snapshot_download(repo_id=self.model_name, cache_dir=self.cache_dir)

    def _resolve_model_path(self) -> str:
        if self.model_path is not None:
            return self.model_path
        from huggingface_hub import snapshot_download

        repo_dir = snapshot_download(repo_id=self.model_name, cache_dir=self.cache_dir)
        nemo_files = sorted(f for f in os.listdir(repo_dir) if f.endswith(".nemo"))
        if not nemo_files:
            msg = f"No .nemo file found in {repo_dir} for model {self.model_name}"
            raise FileNotFoundError(msg)
        return os.path.join(repo_dir, nemo_files[0])

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        if self.diar_model is None:
            import torch
            from nemo.collections.asr.models import SortformerEncLabelModel

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.diar_model = SortformerEncLabelModel.restore_from(
                restore_path=self._resolve_model_path(), map_location=device, strict=False
            )
        self.diar_model.eval()
        self._configure_streaming()

    def _configure_streaming(self) -> None:
        modules = self.diar_model.sortformer_modules
        modules.chunk_len = self.chunk_len
        modules.chunk_right_context = self.chunk_right_context
        modules.fifo_len = self.fifo_len
        modules.spkcache_update_period = self.spkcache_update_period
        modules.spkcache_len = self.spkcache_len
        if self.chunk_left_context is not None:
            modules.chunk_left_context = self.chunk_left_context
        check = getattr(modules, "_check_streaming_parameters", None)
        if callable(check):
            check()

    def _diarize(self, waveforms: list[np.ndarray]) -> list[list[Any]]:
        if self.input_mode == "numpy":
            return list(
                self.diar_model.diarize(
                    audio=waveforms,
                    sample_rate=self.sample_rate,
                    batch_size=self.inference_batch_size,
                    verbose=False,
                )
            )
        import soundfile as sf

        outputs: list[list[Any]] = []
        for waveform in waveforms:
            with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
                sf.write(tmp.name, waveform, self.sample_rate)
                predicted = self.diar_model.diarize(audio=[tmp.name], batch_size=1, verbose=False)
            outputs.append(predicted[0] if predicted else [])
        return outputs

    def annotate_segments(self, segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return one annotation per segment (same order)."""
        annotations: list[dict[str, Any] | None] = [None] * len(segments)
        pending: list[tuple[int, np.ndarray]] = []

        def flush() -> None:
            if not pending:
                return
            try:
                results = self._diarize([wave for _, wave in pending])
            except Exception as exc:  # noqa: BLE001
                logger.warning("Sortformer diarization batch failed: {}", exc)
                results = None
            for position, (index, _) in enumerate(pending):
                if results is None or position >= len(results):
                    annotations[index] = empty_annotation(DIARIZATION_FAILED)
                    continue
                try:
                    parsed = parse_diarization_output(results[position] or [])
                    annotations[index] = summarize_diarization(parsed, _duration(segments[index]))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Could not parse diarization output: {}", exc)
                    annotations[index] = empty_annotation(DIARIZATION_FAILED)
            pending.clear()

        for index, segment in enumerate(segments):
            try:
                wave, _ = load_segment_audio(segment, self.sample_rate, self._audio_config)
            except SegmentAudioError as exc:
                annotations[index] = empty_annotation(exc.code)
                continue
            pending.append((index, wave))
            if len(pending) >= self.inference_batch_size:
                flush()
        flush()
        return [a if a is not None else empty_annotation(DIARIZATION_FAILED) for a in annotations]

    def process(self, task: AudioTask) -> AudioTask:
        pairs = list(iter_segments(task.data))
        annotations = self.annotate_segments([segment for _, segment in pairs])
        for (_, segment), annotation in zip(pairs, annotations, strict=True):
            set_dotted(segment, self.output_key, annotation)
        return task


def _duration(segment: dict[str, Any]) -> float | None:
    value = segment.get("duration")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
