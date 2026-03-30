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

"""TorchSQUIM audio quality metrics stage (PESQ, STOI, SI-SDR)."""

import math
from dataclasses import dataclass, field
from typing import Any

import librosa
import soundfile as sf
import torch
import torchaudio.functional as torchaudio_F  # noqa: N812
from loguru import logger
from torchaudio.pipelines import SQUIM_OBJECTIVE

from nemo_curator.backends.base import NodeInfo, WorkerMetadata
from nemo_curator.stages.audio.common import LegacySpeechStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioBatch


@dataclass
class TorchSquimQualityMetricsStage(LegacySpeechStage):
    """
    Stage that calculates Squim quality metrics for audio files.

    Uses a pre-trained Squim model to calculate audio quality metrics like
    PESQ, STOI, and SI-SDR for each audio segment.

    Args:
        device: Device to run the model on. Defaults to "cuda".

    Returns:
        The same data as in the input data, but with Squim quality metrics added to each segment.
    """

    device: str = "cuda"
    audio_filepath_key: str = "resampled_audio_filepath"

    # Stage metadata
    name: str = "TorchSquimQualityMetrics"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0, gpus=1.0))

    # Internal state (lazy loaded to allow serialization)
    model: Any = field(default=None, repr=False)
    _model_initialized: bool = field(default=False, repr=False)

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.audio_filepath_key, "segments"]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.audio_filepath_key, "segments", "metrics"]

    def setup_on_node(self, node_info: NodeInfo, worker_metadata: WorkerMetadata) -> None:  # noqa: ARG002
        """Setup stage on node."""
        if self.model is None:
            self.model = SQUIM_OBJECTIVE.get_model()

    def setup(self, worker_metadata: Any = None) -> None:  # noqa: ARG002, ANN401
        """Load model. Called once per worker before processing."""
        if self._model_initialized:
            return

        if not torch.cuda.is_available():
            self.device = "cpu"
            logger.warning("CUDA not available, using CPU")

        if self.model is None:
            self.model = SQUIM_OBJECTIVE.get_model()

        if self.device == "cuda":
            self.model = self.model.cuda()

        self._model_initialized = True
        logger.info(f"[{self.name}] Initialized SQUIM model on {self.device}")

    def process_dataset_entry(self, data_entry: dict[str, Any]) -> list[AudioBatch]:
        """Calculate Squim quality metrics for audio entry."""
        if not self._model_initialized:
            self.setup()

        audio_path = data_entry.get(self.audio_filepath_key)
        if not audio_path:
            logger.error(
                f"[{self.name}] Missing '{self.audio_filepath_key}' for entry: "
                f"{data_entry.get('audio_item_id', 'unknown')}"
            )
            return [AudioBatch(data=[data_entry])]
        info = sf.info(audio_path)
        sr = info.samplerate

        try:
            audio, _ = librosa.load(path=audio_path, sr=sr)
        except Exception as ex:  # noqa: BLE001
            logger.error(f"Failed to load audio path: {audio_path}, exception={ex}")
            return [AudioBatch(data=[data_entry])]

        segments = data_entry.get("segments", [])

        for segment in segments:
            if segment.get("speaker") == "no-speaker" or segment.get("text", "").strip() == "":
                continue

            start = segment["start"]
            end = segment["end"]

            start_frame = math.floor(start * sr)
            end_frame = math.floor(end * sr)
            num_frames = end_frame - start_frame

            if num_frames <= 0:
                logger.warning(f"[{self.name}] Zero-length segment at {start}-{end}s in {audio_path}, skipping")
                continue

            y = audio[start_frame:end_frame]
            y = torch.from_numpy(y).unsqueeze(0)

            target_sr = 16000
            if sr != target_sr:
                y = torchaudio_F.resample(y, sr, target_sr)

            try:
                with torch.no_grad():
                    y_cuda = y.to(self.device)
                    stoi_hyp, pesq_hyp, si_sdr_hyp = self.model(y_cuda)
                if "metrics" not in segment:
                    segment["metrics"] = {}

                segment["metrics"]["pesq_squim"] = round(pesq_hyp.item(), 3)
                segment["metrics"]["stoi_squim"] = round(stoi_hyp.item(), 3)
                segment["metrics"]["sisdr_squim"] = round(si_sdr_hyp.item(), 3)

            except Exception as e:  # noqa: BLE001
                torch.cuda.empty_cache()
                logger.error(
                    f"Failed to compute Squim metrics: {e}, frame_offset={start}, num_frames={num_frames}, file={audio_path}"
                )
        return [
            AudioBatch(
                data=[data_entry],
            )
        ]
