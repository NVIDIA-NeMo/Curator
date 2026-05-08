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
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask


@dataclass
class TorchSquimQualityMetricsStage(ProcessingStage[AudioTask, AudioTask]):
    """
    Stage that calculates Squim quality metrics for audio files.

    Uses a pre-trained Squim model to calculate audio quality metrics like
    PESQ, STOI, and SI-SDR for each audio segment.

    Args:
        audio_filepath_key: Key for the audio file path in the manifest. Defaults to "resampled_audio_filepath".
        target_sr: Target sample rate for SQUIM model input. Defaults to 16000.
        batch_size: Number of audio tasks to be processed at once. Defaults to 32.
        compute_batch_size: Number of waveforms to process per GPU inference call. Defaults to 64.

    Returns:
        The same data as in the input data, but with Squim quality metrics added to each segment.
    """

    audio_filepath_key: str = "resampled_audio_filepath"
    target_sr: int = 16000
    batch_size: int = 32
    compute_batch_size: int = 64

    # Stage metadata
    name: str = "TorchSquimQualityMetrics"
    resources: Resources = field(default_factory=lambda: Resources(gpus=1.0))

    model: Any = field(default=None, repr=False)

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.audio_filepath_key, "segments"]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.audio_filepath_key, "segments", "metrics"]

    @property
    def _device(self) -> str:
        """Derive device from resources configuration."""
        if self.resources.requires_gpu:
            if not torch.cuda.is_available():
                msg = f"[{self.name}] GPU requested via resources but CUDA is not available."
                raise RuntimeError(msg)
            return "cuda"
        return "cpu"

    def setup_on_node(
        self, _node_info: NodeInfo | None = None, _worker_metadata: WorkerMetadata | None = None
    ) -> None:
        """Pre-download SQUIM model weights (cache warming, no GPU allocation)."""
        SQUIM_OBJECTIVE.get_model()

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        """Load model onto the target device. Called once per worker."""
        self.model = SQUIM_OBJECTIVE.get_model()
        if self._device == "cuda":
            self.model = self.model.cuda()
        logger.info(f"[{self.name}] Initialized SQUIM model on {self._device}")

    def _compute_metrics_batched(self, waveforms: list[torch.Tensor]) -> list[tuple[float, float, float]]:
        """Run SQUIM on a list of 1-D waveform tensors in batches."""
        results: list[tuple[float, float, float]] = []
        for i in range(0, len(waveforms), self.compute_batch_size):
            batch = waveforms[i : i + self.compute_batch_size]
            max_len = max(w.shape[0] for w in batch)
            padded = torch.zeros(len(batch), max_len)
            for j, w in enumerate(batch):
                padded[j, : w.shape[0]] = w
            padded = padded.to(self._device)
            with torch.no_grad():
                stoi, pesq, si_sdr = self.model(padded)
            for j in range(len(batch)):
                results.append(
                    (
                        round(pesq[j].item(), 3),
                        round(stoi[j].item(), 3),
                        round(si_sdr[j].item(), 3),
                    )
                )
        return results

    def _collect_waveforms_for_entry(self, task_idx: int, data_entry: dict) -> list[tuple[int, int, torch.Tensor]]:
        """Extract valid segment waveforms from a single audio entry.

        Returns a list of (task_idx, segment_idx, waveform) tuples.
        """
        audio_path = data_entry.get(self.audio_filepath_key)
        if not audio_path:
            logger.error(
                f"[{self.name}] Missing '{self.audio_filepath_key}' for entry: "
                f"{data_entry.get('audio_item_id', 'unknown')}"
            )
            return []

        try:
            info = sf.info(audio_path)
            sr = info.samplerate
        except Exception as ex:  # noqa: BLE001
            logger.error(f"[{self.name}] Failed to read audio info: {audio_path}, exception={ex}")
            return []

        try:
            audio, _ = librosa.load(path=audio_path, sr=sr)
        except Exception as ex:  # noqa: BLE001
            logger.error(f"[{self.name}] Failed to load audio: {audio_path}, exception={ex}")
            return []

        collected: list[tuple[int, int, torch.Tensor]] = []
        segments = data_entry.get("segments", [])

        for seg_idx, segment in enumerate(segments):
            if segment.get("speaker") == "no-speaker" or segment.get("text", "").strip() == "":
                continue

            start = segment.get("start", 0)
            end = segment.get("end", 0)
            start_frame = math.floor(start * sr)
            end_frame = math.floor(end * sr)

            if end_frame - start_frame <= 0:
                logger.warning(f"[{self.name}] Zero-length segment at {start}-{end}s in {audio_path}, skipping")
                continue

            y = torch.from_numpy(audio[start_frame:end_frame])
            if sr != self.target_sr:
                y = torchaudio_F.resample(y.unsqueeze(0), sr, self.target_sr).squeeze(0)

            collected.append((task_idx, seg_idx, y))

        return collected

    def process(self, task: AudioTask) -> AudioTask:
        """Delegate single-task processing to process_batch."""
        msg = f"[{self.name}] is a GPU/batched inference stage. Use process_batch() instead."
        raise NotImplementedError(msg)

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        """Calculate Squim quality metrics across all tasks using batched GPU inference.

        Collects waveforms from every valid segment across all tasks, sorts them
        by duration so similarly-sized segments are padded together, runs SQUIM
        in batches on GPU, then scatters results back to the originating
        task's segment.
        """
        if not tasks:
            return tasks

        # Collect all valid waveforms with their origin (task_idx, segment_idx)
        all_waveform_metadata: list[tuple[int, int, torch.Tensor]] = []
        for task_idx, task in enumerate(tasks):
            all_waveform_metadata.extend(self._collect_waveforms_for_entry(task_idx, task.data))

        if not all_waveform_metadata:
            return tasks

        # Sort by waveform length so similarly-sized segments share a batch
        sorted_indices = sorted(range(len(all_waveform_metadata)), key=lambda i: all_waveform_metadata[i][2].shape[0])
        sorted_waveforms = [all_waveform_metadata[i][2] for i in sorted_indices]

        try:
            sorted_results = self._compute_metrics_batched(sorted_waveforms)
            for rank, (pesq_val, stoi_val, sisdr_val) in enumerate(sorted_results):
                orig_idx = sorted_indices[rank]
                task_idx, seg_idx, _ = all_waveform_metadata[orig_idx]
                segment = tasks[task_idx].data["segments"][seg_idx]
                if "metrics" not in segment:
                    segment["metrics"] = {}
                segment["metrics"]["pesq_squim"] = pesq_val
                segment["metrics"]["stoi_squim"] = stoi_val
                segment["metrics"]["sisdr_squim"] = sisdr_val
        except Exception as e:  # noqa: BLE001
            torch.cuda.empty_cache()
            logger.error(f"[{self.name}] Failed to compute Squim metrics: {e}")

        return tasks
