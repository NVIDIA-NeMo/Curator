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

Concrete handlers (e.g. ``HuggingFaceASRDatasetHandler``) implement :meth:`process`,
reusing the shared helpers provided here (audio conversion, task construction,
and optional per-language/per-split manifest writing). Heavy extraction is
parallelized *inside* a single Xenna worker via ``extraction_workers`` (joblib),
so handlers run with ``xenna_workers=1`` by default.
"""

from __future__ import annotations

import json
import os
from abc import ABC
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from loguru import logger

from nemo_curator.backends.utils import RayStageSpecKeys
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask, _EmptyTask

if TYPE_CHECKING:
    from nemo_curator.backends.base import NodeInfo, WorkerMetadata
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
      4. optionally write per-split JSONL manifests via
         :meth:`write_manifest_entry` when ``write_manifest`` is enabled;
      5. return one ``AudioTask`` per utterance via :meth:`build_audio_task`.

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
        write_manifest: If True, write each emitted metadata record to
            ``{output_dir}/{lang}/{split_type}.jsonl`` from this source stage.
            Downstream writer stages can be used instead by leaving this False.
        manifest_splits: Optional split names to pre-create empty manifest files
            for in :meth:`setup_on_node`. Dataset handlers with custom split
            logic can override :meth:`_output_splits`.
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
    write_manifest: bool = False
    manifest_splits: list[str] | None = None
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

    def setup_on_node(
        self,
        _node_info: NodeInfo | None = None,
        _worker_metadata: WorkerMetadata | None = None,
    ) -> None:
        self._manifest_handles = {}
        if not self.write_manifest:
            return
        for lang in self.langs:
            for split_type in self._output_splits():
                os.makedirs(self.audio_output_dir(lang, split_type), exist_ok=True)
                self._manifest_handles[(lang, split_type)] = self._open_manifest(lang, split_type)

    def teardown(self) -> None:
        for handle in getattr(self, "_manifest_handles", {}).values():
            handle.close()
        self._manifest_handles = {}

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

    def _output_splits(self) -> list[str]:
        """Return split names whose manifest files should be pre-created."""
        return list(dict.fromkeys(self.manifest_splits or []))

    def manifest_path(self, lang: str, split_type: str) -> str:
        """Return the JSONL manifest path for one language/split pair."""
        return os.path.join(self.output_dir, lang, f"{split_type}.jsonl")

    def _open_manifest(self, lang: str, split_type: str) -> Any:  # noqa: ANN401
        """Open a manifest handle for one language/split pair."""
        manifest_path = self.manifest_path(lang, split_type)
        os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
        logger.info(f"[{self.name}] writing manifest -> {manifest_path}")
        return open(manifest_path, "w", encoding="utf-8")

    def write_manifest_entry(self, meta: ASRMetadata) -> None:
        """Write one ``ASRMetadata`` row to its split manifest when enabled."""
        if not self.write_manifest:
            return
        key = (meta.lang, meta.split_type)
        if not hasattr(self, "_manifest_handles"):
            self._manifest_handles = {}
        if key not in self._manifest_handles:
            os.makedirs(self.audio_output_dir(meta.lang, meta.split_type), exist_ok=True)
            self._manifest_handles[key] = self._open_manifest(*key)
        handle = self._manifest_handles[key]
        handle.write(json.dumps(meta.to_dict(), ensure_ascii=False) + "\n")
        handle.flush()
