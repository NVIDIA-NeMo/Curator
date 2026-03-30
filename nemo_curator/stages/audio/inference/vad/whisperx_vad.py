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

"""
WhisperX VAD for NeMo Curator.

Provides WhisperXVADModel (shared VAD logic for pyannote and standalone VAD)
and WhisperXVADStage (LegacySpeechStage for VAD-only pipeline use).
"""

import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import soundfile as sf
import torch
from loguru import logger
from whisperx.audio import SAMPLE_RATE
from whisperx.vads.pyannote import Pyannote, load_vad_model

from nemo_curator.backends.base import NodeInfo, WorkerMetadata
from nemo_curator.stages.audio.common import LegacySpeechStage, get_audio_duration
from nemo_curator.tasks import AudioBatch


class WhisperXVADModel:
    """Shared VAD model and get_vad_segments logic for PyAnnote and standalone VAD.

    Used by PyAnnoteDiarizationStage for sub-segment VAD and by WhisperXVADStage
    for VAD-only processing.
    """

    def __init__(
        self,
        device: str = "cuda",
        vad_onset: float = 0.5,
        vad_offset: float = 0.363,
        use_auth_token: Any = None,  # noqa: ANN401
    ):
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA is not available, falling back to CPU for VAD model")
            device = "cpu"
        self._device = device
        self._vad_onset = vad_onset
        self._vad_offset = vad_offset
        default_vad_options = {
            "vad_onset": vad_onset,
            "vad_offset": vad_offset,
        }

        if "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD" not in os.environ:
            os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "true"

        self._model = load_vad_model(torch.device(device), token=use_auth_token, **default_vad_options)

    def to(self, device: str) -> None:
        """Move the model to the given device."""
        self._model = self._model.to(torch.device(device))

    def get_vad_segments(
        self,
        audio: "np.ndarray",
        merge_max_length: float,
        sample_rate: int = SAMPLE_RATE,
    ) -> list[dict]:
        """Get voice activity detection segments for the given audio.

        Args:
            audio: NumPy array of shape (C, N).
            merge_max_length: Maximum length for merging chunks in seconds.
            sample_rate: Sample rate of the audio.

        Returns:
            List of VAD segment dicts with "start" and "end" keys.
        """
        vad_segments = self._model(
            {
                "waveform": torch.from_numpy(audio),
                "sample_rate": sample_rate,
            }
        )
        return Pyannote.merge_chunks(vad_segments, merge_max_length, onset=self._vad_onset)


@dataclass
class WhisperXVADStage(LegacySpeechStage):
    """
    Stage that performs Voice Activity Detection (VAD) using WhisperX's VAD model.

    Adds VAD segments to each entry under segments_key (e.g. "vad_segments").
    Entries shorter than min_length are skipped (not emitted).
    """

    min_length: float = 0.5
    max_length: float = 40.0
    device: str = "cuda"
    vad_onset: float = 0.5
    vad_offset: float = 0.363
    segments_key: str = "vad_segments"
    audio_filepath_key: str = "resampled_audio_filepath"

    name: str = "WhisperXVAD"
    output_dir: str = None

    _vad_model: Any = field(default=None, repr=False)
    _model_initialized: bool = field(default=False, repr=False)

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.audio_filepath_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.audio_filepath_key, self.segments_key]

    def setup_on_node(self, node_info: NodeInfo, worker_metadata: WorkerMetadata) -> None:  # noqa: ARG002
        """Setup stage on node."""
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, using CPU for VAD")
            self.device = "cpu"
        if self._vad_model is None:
            self._vad_model = WhisperXVADModel(
                device=self.device,
                vad_onset=self.vad_onset,
                vad_offset=self.vad_offset,
            )

    def setup(self, worker_metadata: Any = None) -> None:  # noqa: ARG002, ANN401
        if self._model_initialized:
            return
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, using CPU for VAD")
            self.device = "cpu"

        if self._vad_model is None:
            self._vad_model = WhisperXVADModel(
                device=self.device,
                vad_onset=self.vad_onset,
                vad_offset=self.vad_offset,
            )
        self._vad_model.to(self.device)
        self._model_initialized = True
        logger.info(f"[{self.name}] Initialized WhisperX VAD on {self.device}")

    def process_dataset_entry(self, data_entry: dict[str, Any]) -> list[AudioBatch]:
        if not self._model_initialized:
            self.setup()

        file_path = data_entry[self.audio_filepath_key]
        duration = data_entry.get("duration", get_audio_duration(file_path))
        if duration < self.min_length:
            logger.warning(f"Skipping {file_path} because it is less than {self.min_length} seconds")
            return [AudioBatch(data=[data_entry])]

        data, sr = sf.read(file_path, dtype="float32")
        audio = np.expand_dims(data, axis=0) if data.ndim == 1 else data.T
        vad_segments = self._vad_model.get_vad_segments(audio, self.max_length, sample_rate=sr)
        data_entry[self.segments_key] = vad_segments
        return [AudioBatch(data=[data_entry])]
