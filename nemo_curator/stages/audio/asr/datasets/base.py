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

"""Base interface for ASR dataset handler stages.

A *dataset handler* is a fan-out source stage that takes a raw, already-downloaded
dataset directory, extracts/decodes the audio into the ASR-training format
(WAV, 16 kHz, mono, PCM16), and emits one :class:`AudioTask` per utterance.

Concrete handlers (e.g. ``IndicVoicesHandler``) implement :meth:`process`,
reusing the shared helpers provided here (audio conversion and task construction).
Heavy extraction is parallelized *inside* a single Xenna worker via
``extraction_workers`` (joblib), so handlers run with ``xenna_workers=1`` by
default.
"""

from __future__ import annotations

import os
from abc import ABC
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from nemo_curator.backends.utils import RayStageSpecKeys
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask, _EmptyTask

if TYPE_CHECKING:
    from nemo_curator.backends.base import WorkerMetadata
    from nemo_curator.stages.audio.asr.metadata import ASRMetadata


@dataclass
class BaseASRDatasetHandlerStage(ProcessingStage[_EmptyTask, AudioTask], ABC):
    """Base interface/protocol for ASR dataset handlers.

    Subclasses MUST implement :meth:`process`, which should:
      1. discover the raw units (e.g. HF arrow dirs / tar archives) under
         ``raw_data_dir`` for each requested language and native split;
      2. extract/decode audio in parallel (use ``extraction_workers``) and
         convert each clip to WAV/16 kHz/mono/PCM16 via :meth:`convert_audio`;
      3. assign dataset-specific ``split_type`` values in the concrete handler;
      4. return one ``AudioTask`` per utterance via :meth:`build_audio_task`.

    Args:
        raw_data_dir: Directory containing the already-downloaded raw dataset.
        output_dir: Destination root for converted audio.
        langs: Languages to process.
        xenna_workers: Number of Xenna workers for this stage (kept at 1; the
            stage parallelizes extraction internally).
        extraction_workers: Internal joblib worker count for parallel extraction.
            (Named separately from the framework ``num_workers()`` method.)
        target_sample_rate: Output sample rate (Hz).
        target_channels: Output channel count (1 = mono).
        skip_untar: If True, reuse already-extracted WAV files when present
            instead of re-decoding/writing them.
    """

    raw_data_dir: str = ""
    output_dir: str = ""
    langs: list[str] = field(default_factory=list)
    name: str = "asr_dataset_handler"
    source_name: str = "asr_dataset"
    xenna_workers: int = 1
    extraction_workers: int = 10
    target_sample_rate: int = 16000
    target_channels: int = 1
    skip_untar: bool = False
    audio_filepath_key: str = "audio_filepath"
    text_key: str = "text"
    batch_size: int = 1
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    def __post_init__(self) -> None:
        super().__init__()
        for attr in ("raw_data_dir", "output_dir"):
            if not getattr(self, attr):
                msg = f"{attr} is required for {type(self).__name__}"
                raise ValueError(msg)
        if not self.langs:
            msg = f"langs is required for {type(self).__name__}"
            raise ValueError(msg)
        # Give the single Xenna worker enough CPUs for internal parallel extraction.
        self.resources = Resources(cpus=float(max(self.extraction_workers, 1)))

    # ------------------------------------------------------------------
    # Framework wiring
    # ------------------------------------------------------------------
    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.audio_filepath_key, self.text_key, "duration", "lang", "split_type"]

    def num_workers(self) -> int | None:
        return self.xenna_workers

    def ray_stage_spec(self) -> dict[str, Any]:
        return {RayStageSpecKeys.IS_FANOUT_STAGE: True}

    def xenna_stage_spec(self) -> dict[str, Any]:
        return {"num_workers": self.xenna_workers}

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        """Lazy-import heavy deps on the worker (not the driver)."""
        import librosa
        import numpy as np
        import soundfile

        self._np = np
        self._sf = soundfile
        self._librosa = librosa

    # ------------------------------------------------------------------
    # Shared helpers for subclasses
    # ------------------------------------------------------------------
    def convert_audio(self, array: Any, sample_rate: int, orig_channels: int, dst_path: str) -> dict[str, Any]:  # noqa: ANN401
        """Convert one clip to WAV/target-SR/mono/PCM16 and write it to ``dst_path``.

        ``array`` must already be decoded by the concrete dataset handler. Returns
        a dict with ``duration``, ``orig_sample_rate`` and ``orig_num_channels``.
        When ``skip_untar`` is set and ``dst_path`` already exists, the file is
        probed instead of rewritten.
        """
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)

        if self.skip_untar and os.path.exists(dst_path):
            info = self._sf.info(dst_path)
            return {
                "duration": float(info.frames / info.samplerate) if info.samplerate else 0.0,
                "orig_sample_rate": int(info.samplerate),
                "orig_num_channels": int(info.channels),
            }

        arr = self._np.asarray(array, dtype=self._np.float32)
        orig_sample_rate = int(sample_rate)
        if orig_sample_rate != self.target_sample_rate:
            arr = self._librosa.resample(arr, orig_sr=orig_sample_rate, target_sr=self.target_sample_rate)

        self._sf.write(dst_path, arr, self.target_sample_rate, subtype="PCM_16")
        duration = float(len(arr) / self.target_sample_rate) if self.target_sample_rate else 0.0
        return {
            "duration": duration,
            "orig_sample_rate": orig_sample_rate,
            "orig_num_channels": orig_channels,
        }

    def build_audio_task(self, meta: ASRMetadata) -> AudioTask:
        """Wrap an :class:`ASRMetadata` into an ``AudioTask``."""
        return AudioTask(
            data=meta.to_dict(),
            dataset_name=f"{self.source_name}_{meta.lang}_{meta.split_type}",
            filepath_key=self.audio_filepath_key,
        )

    def audio_output_dir(self, lang: str, split_type: str) -> str:
        """Standard per-language/per-split audio output directory."""
        return os.path.join(self.output_dir, lang, split_type, "audio")
