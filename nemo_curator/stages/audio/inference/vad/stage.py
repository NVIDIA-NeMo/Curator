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

"""Generic VAD Curator stage (SDP-V2 design doc §4).

Implements the stage half of the SDP-V2 stage-adapter split for the
VAD family. The stage owns Curator-side glue:

* validates ``task.data`` against ``inputs()`` / ``outputs()``;
* unpacks per-task knobs (audio filepath, optional in-memory waveform,
  ``duration``) into a single item dict;
* dispatches the adapter's ``detect_batch`` once per task;
* writes the adapter's :class:`VADInterval` list onto
  ``task.data[segments_key]`` as canonical ``{"start": float,
  "end": float}`` dicts;
* emits performance metrics in the shape ``perf_summary_merged.json``
  consumers already expect.

The stage knows nothing about which VAD model is running. The concrete
adapter class is resolved at runtime from the YAML's ``adapter_target``
string via ``hydra.utils.get_class``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import hydra.utils
from loguru import logger

from nemo_curator.adapters.vad.base import VADAdapter, VADResult
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

if TYPE_CHECKING:
    from nemo_curator.backends.base import NodeInfo, WorkerMetadata


@dataclass
class VADStage(ProcessingStage[AudioTask, AudioTask]):
    """Voice-activity-detection Curator stage with pluggable adapter.

    Args:
        adapter_target: Tier-1 swap surface. Fully-qualified class path
            of the concrete :class:`~nemo_curator.adapters.vad.VADAdapter`
            implementation (e.g.
            ``"nemo_curator.adapters.vad.WhisperXVADAdapter"``).
            Resolved at ``setup()`` time via ``hydra.utils.get_class``.
        model_id: Tier-1. Model checkpoint identifier, forwarded to
            :meth:`VADAdapter.prefetch_weights` and to the adapter
            constructor.
        revision: Tier-1. Optional model revision to pin.
        audio_filepath_key: Key into ``task.data`` carrying the audio
            path. Defaults to ``"resampled_audio_filepath"``.
        waveform_key / sample_rate_key: Optional keys for in-memory
            waveform reuse.
        segments_key: Key under which the canonical VAD interval list
            is written.
        prefetch_fail_on_error: When False, ``setup_on_node`` warns and
            defers weight prefetch to ``setup()`` instead of raising.
        adapter_kwargs: Tier-2. Opaque dict forwarded to the adapter
            constructor as ``**adapter_kwargs``. The stage NEVER reads
            inside this dict.
        resources / xenna_num_workers: Standard Curator stage knobs.
    """

    name: str = "VAD"

    # ---- Tier 1: swap surface ----
    adapter_target: str = ""
    model_id: str = ""
    revision: str | None = None

    # ---- Tier 1: universal stage knobs ----
    audio_filepath_key: str = "resampled_audio_filepath"
    waveform_key: str | None = None
    sample_rate_key: str | None = None
    segments_key: str = "vad_segments"

    prefetch_fail_on_error: bool = True

    # ---- Tier 2: opaque adapter knob bag ----
    adapter_kwargs: dict[str, Any] = field(default_factory=dict)

    # ---- Standard Curator stage knobs ----
    resources: Resources = field(default_factory=lambda: Resources(gpus=1))
    xenna_num_workers: int | None = None

    # ---- Internal state ----
    _adapter: VADAdapter | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.adapter_target:
            msg = (
                "VADStage.adapter_target is required - set it in YAML to a fully-qualified "
                "adapter class path (e.g. 'nemo_curator.adapters.vad.WhisperXVADAdapter')."
            )
            raise ValueError(msg)

    # ------------------------------------------------------------------
    # I/O contract
    # ------------------------------------------------------------------

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.audio_filepath_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.audio_filepath_key, self.segments_key]

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
        try:
            prefetch_t0 = time.perf_counter()
            self._adapter_class().prefetch_weights(self.model_id, self.revision)
            logger.info(
                "VAD weights cached on node for {} ({}) in {:.3f}s",
                self.model_id,
                self.adapter_target,
                time.perf_counter() - prefetch_t0,
            )
        except Exception as exc:  # noqa: BLE001
            msg = f"VADStage: prefetch_weights failed for {self.model_id}"
            if self.prefetch_fail_on_error:
                raise RuntimeError(msg) from exc
            logger.warning("{}; setup() will retry: {}", msg, exc)

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        if self._adapter is None:
            cls = self._adapter_class()
            kwargs = dict(self.adapter_kwargs)
            if self.model_id:
                kwargs.setdefault("model_id", self.model_id)
            kwargs.setdefault("revision", self.revision)
            kwargs.setdefault("device", self._device)
            self._adapter = cls(**kwargs)
            self._adapter.setup()
            logger.info("[{}] VAD adapter ready ({})", self.name, self.adapter_target)

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
            "duration": data.get("duration"),
            "task_id": getattr(task, "task_id", None),
        }
        if self.waveform_key:
            item["waveform"] = data.get(self.waveform_key)
        if self.sample_rate_key:
            item["sample_rate"] = data.get(self.sample_rate_key)
        return item

    @staticmethod
    def _interval_to_dict(interval: Any) -> dict[str, float]:
        return {"start": float(interval.start), "end": float(interval.end)}

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
        results: list[VADResult] = self._adapter.detect_batch([item])
        result = results[0] if results else VADResult(intervals=[], model_id=self.model_id)

        intervals = [self._interval_to_dict(iv) for iv in result.intervals]
        data_entry[self.segments_key] = intervals

        duration = float(item.get("duration") or 0.0)
        if duration == 0.0:
            duration = float(result.extras.get("duration_s", 0.0) or 0.0)

        metrics: dict[str, float] = {
            "process_time": time.perf_counter() - t0,
            "audio_duration": duration,
            "vad_segments_detected": float(len(intervals)),
            "skipped_short": float(result.extras.get("skipped_short", 0.0))
            if "skipped_short" in result.extras
            else (1.0 if not intervals and result.extras.get("duration_s", 0.0) < 0 else 0.0),
        }
        for key, value in (self._adapter.last_metrics or {}).items():
            metrics[f"model_{key}"] = float(value)
        self._log_metrics(metrics)
        return task
