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

"""WhisperX VAD adapter.

Implements :class:`~nemo_curator.adapters.vad.VADAdapter` on top of
WhisperX's ``Pyannote.merge_chunks``-based VAD helper. The underlying
:class:`~nemo_curator.stages.audio.inference.vad.whisperx_vad.WhisperXVADModel`
class is kept where it is - PyAnnote diarization also depends on it
for its long-turn micro-split path - and this adapter just wraps it
behind the canonical Protocol.

Logic moved verbatim from the pre-split
``nemo_curator.stages.audio.inference.vad.whisperx_vad.WhisperXVADStage``
process() body.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import soundfile as sf
from loguru import logger

from nemo_curator.adapters.vad.base import VADInterval, VADResult
from nemo_curator.stages.audio.common import get_audio_duration
from nemo_curator.stages.audio.inference.vad.whisperx_vad import WhisperXVADModel


@dataclass
class WhisperXVADAdapter:
    """WhisperX-backed implementation of :class:`VADAdapter`.

    Attributes:
        model_id: Identifier for diagnostics (WhisperX VAD doesn't use
            a HF id; default ``"whisperx/vad"`` is a label only).
        revision: Accepted for protocol uniformity; not used by
            WhisperX VAD.
        device: ``"cuda"`` or ``"cpu"``. The stage passes the worker's
            actual device.
        vad_onset: Onset probability threshold forwarded to WhisperX.
        vad_offset: Offset probability threshold forwarded to WhisperX.
        max_length: Maximum chunk length passed to
            :meth:`WhisperXVADModel.get_vad_segments`.
        min_length: Minimum clip duration; clips shorter than this are
            skipped entirely (empty :class:`VADResult`). Matches the
            pre-split ``WhisperXVADStage`` behaviour.
    """

    # ---- Required protocol fields ----
    model_id: str = "whisperx/vad"
    revision: str | None = None

    # ---- Adapter-specific knobs ----
    device: str = "cuda"
    vad_onset: float = 0.5
    vad_offset: float = 0.363
    max_length: float = 40.0
    min_length: float = 0.5

    # ---- Internal state ----
    _vad_model: Any = field(default=None, repr=False)
    last_metrics: dict[str, float] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Adapter contract
    # ------------------------------------------------------------------

    @classmethod
    def prefetch_weights(cls, model_id: str, revision: str | None = None) -> None:
        """Force a CPU-side download of the WhisperX VAD weights.

        WhisperX VAD downloads on first instantiation; we trigger that
        once on the node by constructing the model on CPU and then
        dropping it. The actual GPU placement happens in
        :meth:`setup`.
        """
        del model_id, revision  # WhisperX VAD has no public model_id knob.
        _ = WhisperXVADModel(device="cpu")

    def setup(self) -> None:
        if self._vad_model is None:
            self._vad_model = WhisperXVADModel(
                device=self.device,
                vad_onset=self.vad_onset,
                vad_offset=self.vad_offset,
            )
        self._vad_model.to(self.device)
        logger.info("WhisperXVADAdapter ready on {} (model_id={})", self.device, self.model_id)

    def teardown(self) -> None:
        self._vad_model = None

    def detect_batch(self, items: list[dict[str, Any]]) -> list[VADResult]:
        if not items:
            return []
        if self._vad_model is None:
            msg = "WhisperXVADAdapter.setup() must be called before detect_batch()"
            raise RuntimeError(msg)

        results: list[VADResult] = []
        per_item_times: list[float] = []
        per_item_skip: list[float] = []
        per_item_count: list[int] = []

        for item in items:
            t0 = time.perf_counter()
            result, skipped = self._detect_one(item)
            results.append(result)
            per_item_times.append(time.perf_counter() - t0)
            per_item_skip.append(1.0 if skipped else 0.0)
            per_item_count.append(len(result.intervals))

        self.last_metrics = {
            "batch_size": float(len(items)),
            "vad_time_s_total": float(sum(per_item_times)),
            "vad_time_s_max": float(max(per_item_times)) if per_item_times else 0.0,
            "skipped_short_total": float(sum(per_item_skip)),
            "vad_intervals_detected_total": float(sum(per_item_count)),
        }
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect_one(self, item: dict[str, Any]) -> tuple[VADResult, bool]:
        audio_filepath = item.get("audio_filepath")
        if not audio_filepath:
            return VADResult(intervals=[], model_id=self.model_id), True

        duration = item.get("duration")
        if duration is None:
            duration = get_audio_duration(audio_filepath)
        if duration < self.min_length:
            logger.warning(
                "Skipping {} because it is less than {} seconds", audio_filepath, self.min_length
            )
            return (
                VADResult(intervals=[], model_id=self.model_id, extras={"duration_s": float(duration)}),
                True,
            )

        data, sr = sf.read(audio_filepath, dtype="float32")
        audio = np.expand_dims(data, axis=0) if data.ndim == 1 else data.T
        raw_segments = self._vad_model.get_vad_segments(audio, self.max_length, sample_rate=sr)
        intervals = [
            VADInterval(start=float(seg["start"]), end=float(seg["end"])) for seg in raw_segments
        ]
        return (
            VADResult(
                intervals=intervals,
                model_id=self.model_id,
                extras={"duration_s": float(duration)},
            ),
            False,
        )
