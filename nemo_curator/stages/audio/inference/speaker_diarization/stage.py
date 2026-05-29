# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

"""Generic speaker-diarization Curator stage (SDP-V2 design doc §3).

Implements the stage half of the SDP-V2 stage-adapter split for the
diarization family. The stage owns Curator-side glue:

* validates ``task.data`` against ``inputs()`` / ``outputs()``;
* unpacks per-task knobs (audio filepath, optional in-memory waveform,
  ``audio_item_id``, ``duration``) into a single item dict;
* dispatches the adapter's ``diarize_batch`` once per task;
* converts the adapter's typed :class:`DiarSegment` results into the
  on-disk dict shape downstream consumers expect
  (``{"speaker": ..., "start": ..., "end": ...}``);
* fills the inter-turn gaps with ``no-speaker`` segments via
  :func:`add_non_speaker_segments`;
* writes ``task.data[segments_key]`` and optionally
  ``task.data[overlap_segments_key]``;
* emits performance metrics in the shape ``perf_summary_merged.json``
  consumers already expect.

The stage knows nothing about which diarizer is running. The concrete
adapter class is resolved at runtime from the YAML's ``adapter_target``
string via ``hydra.utils.get_class``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import hydra.utils
from loguru import logger

from nemo_curator.adapters.diarization.base import DiarizationAdapter, DiarizationResult
from nemo_curator.stages.audio.common import get_audio_duration
from nemo_curator.stages.audio.tagging.utils import add_non_speaker_segments
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

if TYPE_CHECKING:
    from nemo_curator.backends.base import NodeInfo, WorkerMetadata


@dataclass
class DiarizationStage(ProcessingStage[AudioTask, AudioTask]):
    """Speaker-diarization Curator stage with pluggable adapter.

    Args:
        adapter_target: Tier-1 swap surface. Fully-qualified class path
            of the concrete
            :class:`~nemo_curator.adapters.diarization.DiarizationAdapter`
            implementation (e.g.
            ``"nemo_curator.adapters.diarization.PyAnnoteDiarizationAdapter"``).
            Resolved at ``setup()`` time via ``hydra.utils.get_class``.
        model_id: Tier-1. Model checkpoint identifier, forwarded both to
            :meth:`DiarizationAdapter.prefetch_weights` (in
            ``setup_on_node``) and to the adapter constructor.
        revision: Tier-1. Optional model revision to pin.
        audio_filepath_key: Key into ``task.data`` carrying the decoded
            audio path. Defaults to ``"resampled_audio_filepath"`` for
            symmetry with the §1.2 ResampleAudioStage output.
        waveform_key: Optional key into ``task.data`` carrying an
            in-memory waveform. When present alongside
            ``sample_rate_key`` and ``filepath_fallback_key`` is
            enabled, adapters MAY use the in-memory buffer to avoid a
            re-decode.
        sample_rate_key: Key into ``task.data`` carrying the
            ``waveform_key`` array's sample rate.
        segments_key: Key under which the canonical per-speaker turn
            list is written. Each entry is a dict
            ``{"speaker": str, "start": float, "end": float}``;
            includes ``no-speaker`` gap-fill segments emitted by
            :func:`add_non_speaker_segments`.
        overlap_segments_key: When set, the stage also writes a list
            of overlap turns under this key. Set ``None`` for adapters
            that don't surface overlap detection.
        non_speaker_max_length: Optional ceiling for the
            ``no-speaker`` gap-fill segments. When set, long gaps are
            split into ``<= non_speaker_max_length`` second chunks
            (preserves the pre-split behaviour where this was tied to
            the PyAnnote adapter's ``max_length`` knob).
        prefetch_fail_on_error: When False, ``setup_on_node`` warns and
            defers weight prefetch to ``setup()`` instead of raising.
        adapter_kwargs: Tier-2. Opaque dict forwarded to the adapter
            constructor as ``**adapter_kwargs``. The stage NEVER reads
            inside this dict - it is the adapter's private knob bag.
        resources / batch_size: Standard Curator stage knobs.
        xenna_num_workers: Optional cluster-wide cap forwarded to the
            Xenna scheduler. ``None`` (default) lets Xenna autoscale.
    """

    name: str = "Diarization"

    # ---- Tier 1: swap surface ----
    adapter_target: str = ""
    model_id: str = ""
    revision: str | None = None

    # ---- Tier 1: universal stage knobs ----
    audio_filepath_key: str = "resampled_audio_filepath"
    waveform_key: str | None = None
    sample_rate_key: str | None = None
    segments_key: str = "segments"
    overlap_segments_key: str | None = "overlap_segments"
    non_speaker_max_length: float | None = 40.0

    prefetch_fail_on_error: bool = True

    # ---- Tier 2: opaque adapter knob bag ----
    adapter_kwargs: dict[str, Any] = field(default_factory=dict)

    # ---- Standard Curator stage knobs ----
    resources: Resources = field(default_factory=lambda: Resources(gpus=1))
    xenna_num_workers: int | None = None

    # ---- Internal state ----
    _adapter: DiarizationAdapter | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.adapter_target:
            msg = (
                "DiarizationStage.adapter_target is required - set it in YAML to a fully-qualified "
                "adapter class path (e.g. 'nemo_curator.adapters.diarization.PyAnnoteDiarizationAdapter')."
            )
            raise ValueError(msg)

    # ------------------------------------------------------------------
    # I/O contract
    # ------------------------------------------------------------------

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.audio_filepath_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        keys = [self.audio_filepath_key, self.segments_key]
        if self.overlap_segments_key:
            keys.append(self.overlap_segments_key)
        return [], keys

    def xenna_stage_spec(self) -> dict[str, Any]:
        spec: dict[str, Any] = {}
        if self.xenna_num_workers is not None:
            spec["num_workers"] = self.xenna_num_workers
        return spec

    @property
    def _device(self) -> str:
        return "cuda" if self.resources.requires_gpu else "cpu"

    # ------------------------------------------------------------------
    # Adapter lifecycle
    # ------------------------------------------------------------------

    def _adapter_class(self) -> type:
        return hydra.utils.get_class(self.adapter_target)

    def setup_on_node(
        self,
        _node_info: NodeInfo | None = None,
        _worker_metadata: WorkerMetadata | None = None,
    ) -> None:
        """Cache diarizer weights once per node (no GPU allocation)."""
        try:
            prefetch_t0 = time.perf_counter()
            self._adapter_class().prefetch_weights(self.model_id, self.revision)
            logger.info(
                "Diarization weights cached on node for {} ({}) in {:.3f}s",
                self.model_id,
                self.adapter_target,
                time.perf_counter() - prefetch_t0,
            )
        except Exception as exc:  # noqa: BLE001
            msg = f"DiarizationStage: prefetch_weights failed for {self.model_id}"
            if self.prefetch_fail_on_error:
                raise RuntimeError(msg) from exc
            logger.warning("{}; setup() will retry: {}", msg, exc)

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        if self._adapter is None:
            cls = self._adapter_class()
            kwargs = dict(self.adapter_kwargs)
            # The stage owns model_id/revision (Tier-1); pass through.
            kwargs.setdefault("model_id", self.model_id) if self.model_id else None
            kwargs.setdefault("revision", self.revision)
            # Inject device hint when the adapter accepts one (PyAnnote does).
            kwargs.setdefault("device", self._device)
            self._adapter = cls(**kwargs)
            self._adapter.setup()
            logger.info(
                "[{}] Diarization adapter ready ({})",
                self.name,
                self.adapter_target,
            )

    def teardown(self) -> None:
        if self._adapter is not None:
            self._adapter.teardown()
            self._adapter = None

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def _build_item(self, task: AudioTask) -> dict[str, Any]:
        data = task.data
        item: dict[str, Any] = {
            "audio_filepath": data.get(self.audio_filepath_key),
            "audio_item_id": data.get("audio_item_id"),
            "speaker_id": data.get("speaker_id"),
            "duration": data.get("duration"),
            "task_id": getattr(task, "task_id", None),
        }
        if self.waveform_key:
            item["waveform"] = data.get(self.waveform_key)
        if self.sample_rate_key:
            item["sample_rate"] = data.get(self.sample_rate_key)
        return item

    @staticmethod
    def _segment_to_dict(seg: Any) -> dict[str, Any]:
        out: dict[str, Any] = {
            "speaker": seg.speaker,
            "start": float(seg.start),
            "end": float(seg.end),
        }
        if getattr(seg, "confidence", None) is not None:
            out["confidence"] = float(seg.confidence)
        return out

    def process(self, task: AudioTask) -> AudioTask:
        t0 = time.perf_counter()
        data_entry = task.data

        if self._adapter is None:
            msg = "Adapter not initialized - setup() was not called"
            raise RuntimeError(msg)

        file_path = data_entry.get(self.audio_filepath_key)
        if not file_path:
            msg = (
                f"[{self.name}] Missing key '{self.audio_filepath_key}' in entry: "
                f"{data_entry.get('audio_item_id', 'unknown')}"
            )
            raise ValueError(msg)

        item = self._build_item(task)
        results: list[DiarizationResult] = self._adapter.diarize_batch([item])
        result = results[0] if results else DiarizationResult(diar_segments=[], model_id=self.model_id)

        # Convert typed segments -> on-disk dict shape.
        segments: list[dict[str, Any]] = [self._segment_to_dict(seg) for seg in result.diar_segments]
        overlap_segments: list[dict[str, Any]] = [
            self._segment_to_dict(seg) for seg in result.overlap_segments
        ]

        # Non-speaker gap fill (works on dicts; mirrors pre-split behaviour).
        audio_duration = data_entry.get("duration", get_audio_duration(file_path))
        add_non_speaker_segments(segments, audio_duration, self.non_speaker_max_length)

        data_entry[self.segments_key] = segments
        if self.overlap_segments_key:
            data_entry[self.overlap_segments_key] = overlap_segments

        speakers = {seg["speaker"] for seg in segments if seg.get("speaker") != "no-speaker"}
        metrics: dict[str, float] = {
            "process_time": time.perf_counter() - t0,
            "segments_detected": float(len(segments)),
            "overlap_segments_detected": float(len(overlap_segments)),
            "speakers_detected": float(len(speakers)),
            "audio_duration": float(audio_duration),
        }
        for key, value in (self._adapter.last_metrics or {}).items():
            metrics[f"model_{key}"] = float(value)
        self._log_metrics(metrics)
        return task
