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

"""
VAD (Voice Activity Detection) segmentation stage.

Segments audio into speech chunks using Silero VAD model,
filtering out silence and creating manageable segments for further processing.

Supports both CPU and GPU execution. GPU is used when available and requested
via _resources configuration.

Example:
    from nemo_curator.pipeline import Pipeline
    from nemo_curator.stages.audio.segmentation import VADSegmentationStage
    from nemo_curator.stages.resources import Resources

    # Default execution (CPU-only)
    pipeline.add_stage(VADSegmentationStage(min_duration_sec=2.0, threshold=0.5))

    # Opt into GPU if desired
    pipeline.add_stage(
        VADSegmentationStage(min_duration_sec=2.0)
        .with_(resources=Resources(gpus=0.3))
    )
"""

import os
import warnings
from dataclasses import dataclass, field
from typing import Any

import torch
import torchaudio
from loguru import logger
from silero_vad import get_speech_timestamps, load_silero_vad

from nemo_curator.backends.base import WorkerMetadata
from nemo_curator.backends.experimental.utils import RayStageSpecKeys
from nemo_curator.stages.audio.common import ensure_waveform_2d, load_audio_file
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

SILERO_SUPPORTED_RATES = {8000, 16000, 32000, 48000, 64000, 96000}
SILERO_TARGET_RATE = 16000


@dataclass
class VADSegmentationStage(ProcessingStage[AudioTask, AudioTask]):
    """
    Stage to segment audio using Voice Activity Detection (VAD).

    This stage takes a single AudioTask and segments it into speech chunks based on VAD,
    filtering out silence and creating manageable segments for further processing.
    Uses Silero VAD model loaded via torch.hub.

    Returns a list[AudioTask] with one AudioTask per detected speech segment (fan-out).

    Args:
        min_interval_ms: Minimum silence interval between speech segments in milliseconds.
        min_duration_sec: Minimum segment duration in seconds.
        max_duration_sec: Maximum segment duration in seconds.
        threshold: Voice activity detection threshold (0.0-1.0).
        speech_pad_ms: Padding in ms to add before/after speech segments.
        waveform_key: Key to get waveform data.
        sample_rate_key: Key to get sample rate.

    Note:
        Default resources: cpus=1.0, gpus=0.0 (CPU). Silero VAD is lightweight.
        Use .with_(resources=Resources(gpus=X)) to opt into GPU execution.
    """

    min_interval_ms: int = 500
    min_duration_sec: float = 2.0
    max_duration_sec: float = 60.0
    threshold: float = 0.5
    speech_pad_ms: int = 300
    waveform_key: str = "waveform"
    sample_rate_key: str = "sample_rate"
    nested: bool = False

    name: str = "VADSegmentation"
    batch_size: int = 1
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0, gpus=0.0))

    def __post_init__(self):
        super().__init__()
        self._vad_model = None
        self._device = None

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], ["waveform", "sample_rate", "start_ms", "end_ms", "segment_num", "duration_sec"]

    def ray_stage_spec(self) -> dict[str, Any]:
        if self.nested:
            return {}
        return {RayStageSpecKeys.IS_FANOUT_STAGE: True}

    def setup(self, _: WorkerMetadata | None = None) -> None:
        self._initialize_model()

    def teardown(self) -> None:
        if self._vad_model is not None:
            del self._vad_model
            self._vad_model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @staticmethod
    def _check_gpu_availability(gpus: float) -> None:
        if gpus > 0 and not torch.cuda.is_available():
            msg = (
                "Resources request GPU (gpus > 0) but CUDA is not available. "
                "Either set resources=Resources(gpus=0) for CPU-only or install CUDA."
            )
            raise RuntimeError(msg)

    def _initialize_model(self) -> None:
        if self._vad_model is not None:
            return
        self._check_gpu_availability(self._resources.gpus)
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Sampling rate is a multiple of 16000")
                model = load_silero_vad()

            use_gpu = self._resources.gpus > 0 and torch.cuda.is_available()

            if use_gpu:
                self._device = torch.device("cuda")
                model = model.to(self._device)
                logger.info(f"Silero VAD model loaded on GPU: {self._device}")
            else:
                self._device = torch.device("cpu")
                logger.info("Silero VAD model loaded on CPU")

            self._vad_model = model
        except Exception as e:
            logger.error(f"Failed to load VAD model: {e}")
            raise

    def _build_segment_item(
        self,
        item: dict[str, Any],
        waveform: torch.Tensor,
        sample_rate: int,
        segment: dict[str, float],
        segment_num: int,
    ) -> dict[str, Any]:
        """Build a single segment item dict from a VAD result."""
        start_ms = int(segment["start"] * 1000)
        end_ms = int(segment["end"] * 1000)
        start_sample = int(segment["start"] * sample_rate)
        end_sample = int(segment["end"] * sample_rate)

        if waveform.dim() == 1:
            segment_waveform = waveform[start_sample:end_sample].unsqueeze(0).clone()
        else:
            segment_waveform = waveform[:, start_sample:end_sample].clone()

        segment_data: dict[str, Any] = {
            k: v
            for k, v in item.items()
            if k
            not in (
                self.waveform_key,
                self.sample_rate_key,
                "start_ms",
                "end_ms",
                "segment_num",
                "duration_sec",
                "duration",
                "num_samples",
            )
        }
        segment_data.update(
            {
                "waveform": segment_waveform,
                "sample_rate": sample_rate,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "segment_num": segment_num,
                "duration_sec": (end_ms - start_ms) / 1000.0,
                "original_file": item.get("original_file", item.get("audio_filepath", "unknown")),
            }
        )
        return segment_data

    def _resolve_audio(self, item: dict[str, Any]) -> tuple[torch.Tensor, int] | None:
        """Resolve waveform and sample_rate from task data. Returns None on failure."""
        waveform = item.get(self.waveform_key)
        sample_rate = item.get(self.sample_rate_key)

        if waveform is None:
            audio_filepath = item.get("audio_filepath")
            if audio_filepath and os.path.exists(audio_filepath):
                try:
                    waveform, sample_rate = load_audio_file(audio_filepath)
                    item[self.waveform_key] = waveform
                    item[self.sample_rate_key] = sample_rate
                except Exception as e:  # noqa: BLE001
                    logger.error(f"Failed to load audio file {audio_filepath}: {e}")
                    return None
            else:
                logger.error("Missing waveform and no valid audio_filepath provided")
                return None
        elif sample_rate is None:
            logger.warning("Waveform present but sample_rate missing - task skipped")
            return None

        return ensure_waveform_2d(waveform), sample_rate

    def process(self, task: AudioTask) -> AudioTask | list[AudioTask]:
        """
        Process a single AudioTask.

        When ``nested=False`` (default), returns ``list[AudioTask]`` with one
        task per speech segment (fan-out).

        When ``nested=True``, returns a single ``AudioTask`` with all segment
        dicts stored in ``task.data["segments"]`` (no fan-out).
        """
        if self._vad_model is None:
            msg = "VAD model failed to initialize. Cannot process audio."
            raise RuntimeError(msg)

        audio_result = self._resolve_audio(task.data)
        if audio_result is None:
            return []
        waveform, sample_rate = audio_result

        try:
            segments = self._get_vad_segments(waveform, sample_rate)
            if not segments:
                logger.warning("No speech segments detected by VAD")
                if self.nested:
                    task.data["segments"] = []
                    return task
                return []

            original_file = task.data.get("audio_filepath", "unknown")
            file_name = os.path.basename(original_file) if original_file != "unknown" else task.task_id
            total_duration = sum((s["end"] - s["start"]) for s in segments)
            logger.info(
                f"[VADSegmentation] {file_name}: {len(segments)} segments extracted ({total_duration:.1f}s total speech)"
            )

            if self.nested:
                task.data["segments"] = [
                    self._build_segment_item(task.data, waveform, sample_rate, seg, i)
                    for i, seg in enumerate(segments)
                ]
                del task.data[self.waveform_key]
                return task

            output_tasks: list[AudioTask] = []
            for i, segment in enumerate(segments):
                seg_data = self._build_segment_item(task.data, waveform, sample_rate, segment, i)
                seg_task = AudioTask(
                    data=seg_data,
                    task_id=f"{task.task_id}_seg_{i}",
                    dataset_name=task.dataset_name,
                )
                if task._metadata:
                    seg_task._metadata = dict(task._metadata)
                output_tasks.append(seg_task)

        except Exception as e:  # noqa: BLE001
            logger.exception(f"Error during VAD segmentation: {e}")
            return []
        else:
            return output_tasks

    def _get_vad_segments(self, waveform: torch.Tensor, sample_rate: int) -> list[dict[str, float]]:
        """Get speech segments using VAD."""
        if waveform.dim() > 1:
            waveform = waveform.mean(dim=0) if waveform.shape[0] > 1 else waveform.squeeze(0)

        if self._device is not None and waveform.device != self._device:
            waveform = waveform.to(self._device)

        vad_sample_rate = sample_rate
        vad_waveform = waveform
        if sample_rate not in SILERO_SUPPORTED_RATES:
            logger.debug(f"Resampling audio from {sample_rate}Hz to {SILERO_TARGET_RATE}Hz for VAD")
            device = waveform.device
            waveform_cpu = waveform.cpu() if waveform.device.type != "cpu" else waveform
            if waveform_cpu.dim() == 1:
                waveform_cpu = waveform_cpu.unsqueeze(0)
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=SILERO_TARGET_RATE)
            vad_waveform = resampler(waveform_cpu).squeeze(0)
            if device.type != "cpu":
                vad_waveform = vad_waveform.to(device)
            vad_sample_rate = SILERO_TARGET_RATE

        speech_timestamps = get_speech_timestamps(
            vad_waveform,
            self._vad_model,
            sampling_rate=vad_sample_rate,
            threshold=self.threshold,
            min_speech_duration_ms=self.min_duration_sec * 1000,
            max_speech_duration_s=self.max_duration_sec,
            min_silence_duration_ms=self.min_interval_ms,
            speech_pad_ms=self.speech_pad_ms,
        )

        segments = []
        for ts in speech_timestamps:
            start_sec = ts["start"] / vad_sample_rate
            end_sec = ts["end"] / vad_sample_rate
            segments.append(
                {
                    "start": start_sec,
                    "end": end_sec,
                }
            )

        return segments
