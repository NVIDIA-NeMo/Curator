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
from nemo_curator.backends.utils import RayStageSpecKeys
from nemo_curator.stages.audio._agent_ready import AgentReady, Gates, IOSpec, StageContract
from nemo_curator.stages.audio._residency import InputResidency, residency_read_specs, resolve_audio
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

SILERO_SUPPORTED_RATES = {8000, 16000, 32000, 48000, 64000, 96000}
SILERO_TARGET_RATE = 16000


@dataclass
class VADSegmentationStage(AgentReady, ProcessingStage[AudioTask, AudioTask]):
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
        audio_filepath_key: Key in data dict for the input audio file path.
        segments_key: Key where the nested segments list is written (nested=True).
        start_ms_key: Key where each segment's start time in milliseconds is written.
        end_ms_key: Key where each segment's end time in milliseconds is written.
        segment_num_key: Key where each segment's index is written.
        duration_key: Key where each segment's duration in seconds is written.
        original_file_key: Key carrying the source file path for provenance.
        nested: If True, return one task with all segment dicts under segments_key
            instead of fanning out one task per segment (default False).
        input_residency: Which input to use — "waveform" (in-memory only), "file"
            (audio_filepath only), or "auto" (waveform first, file fallback; default).
        keep_segment_waveform_in_task: If True (default), store each segment's waveform
            in the segment item. If False, nested segments are metadata-only — waveform
            consumers such as SegmentConcatenation will skip them.

    Note:
        Default resources: cpus=1.0, gpus=0.0 (CPU). Silero VAD is lightweight.
        Use .with_(resources=Resources(gpus=X)) to opt into GPU execution.
    """

    min_interval_ms: int = 500
    min_duration_sec: float = 2.0
    max_duration_sec: float = 60.0
    threshold: float = 0.5
    speech_pad_ms: int = 300
    audio_filepath_key: str = "audio_filepath"
    waveform_key: str = "waveform"
    sample_rate_key: str = "sample_rate"
    segments_key: str = "segments"
    start_ms_key: str = "start_ms"
    end_ms_key: str = "end_ms"
    segment_num_key: str = "segment_num"
    duration_key: str = "duration"
    original_file_key: str = "original_file"
    nested: bool = False
    input_residency: InputResidency = "auto"
    keep_segment_waveform_in_task: bool = True

    name: str = "VADSegmentation"
    batch_size: int = 1
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0, gpus=0.0))

    def __post_init__(self):
        super().__init__()
        self._vad_model = None
        self._device = None
        if self.nested and not self.keep_segment_waveform_in_task:
            logger.warning(
                "[VADSegmentation] nested=True with keep_segment_waveform_in_task=False: "
                "segments will carry no audio — SegmentConcatenation (and any waveform "
                "consumer) will silently drop every segment. Metadata-only use intended?"
            )

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        if self.nested:
            return [], [self.segments_key]
        outputs = [self.sample_rate_key, self.start_ms_key, self.end_ms_key, self.segment_num_key, self.duration_key]
        if self.keep_segment_waveform_in_task:
            outputs.append(self.waveform_key)
        outputs.append(self.original_file_key)
        return [], outputs

    def describe(self) -> StageContract:
        writes = [
            self.sample_rate_key,
            self.start_ms_key,
            self.end_ms_key,
            self.segment_num_key,
            self.duration_key,
            self.original_file_key,  # _build_segment_item always writes it
        ]
        produces = []
        if self.keep_segment_waveform_in_task:
            writes.append(self.waveform_key)
            produces.append("tensor")
        if self.nested:
            writes = [self.segments_key]
        return StageContract(
            reads_one_of=residency_read_specs(
                self.input_residency,
                audio_filepath_key=self.audio_filepath_key,
                waveform_key=self.waveform_key,
                sample_rate_key=self.sample_rate_key,
            ),
            writes=IOSpec(data_keys=writes, produces=produces),
            cardinality="1:1 nested-list" if self.nested else "1:N fan-out",
            cardinality_options=["fan_out", "nested"],
            iteration_key=self.segments_key,
            gates=Gates(requires_gpu=self.resources.gpus > 0),
        )

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
                self.start_ms_key,
                self.end_ms_key,
                self.segment_num_key,
                self.duration_key,
                "num_samples",
            )
        }
        if not self.keep_segment_waveform_in_task:
            segment_waveform = None
        segment_data.update(
            {
                self.sample_rate_key: sample_rate,
                self.start_ms_key: start_ms,
                self.end_ms_key: end_ms,
                self.segment_num_key: segment_num,
                self.duration_key: (end_ms - start_ms) / 1000.0,
                self.original_file_key: item.get(self.original_file_key, item.get(self.audio_filepath_key, "unknown")),
            }
        )
        if segment_waveform is not None:
            segment_data[self.waveform_key] = segment_waveform
        return segment_data

    def _resolve_audio(self, item: dict[str, Any]) -> tuple[torch.Tensor, int] | None:
        """Resolve waveform and sample_rate from task data. Returns None on failure."""
        resolved = resolve_audio(
            item,
            residency=self.input_residency,  # type: ignore[arg-type]
            audio_filepath_key=self.audio_filepath_key,
            waveform_key=self.waveform_key,
            sample_rate_key=self.sample_rate_key,
        )
        if resolved is None:
            logger.error("Missing waveform/sample_rate and no valid audio path provided")
        return resolved

    def process(self, task: AudioTask) -> AudioTask | list[AudioTask]:  # noqa: PLR0911 (complexity accepted: one early return per input/error condition)
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

        try:
            audio_result = self._resolve_audio(task.data)
        except (OSError, RuntimeError) as e:  # corrupt/unreadable audio -> skip the row, don't crash the batch
            logger.error(f"Failed to load audio for {task.data.get(self.audio_filepath_key)!r}: {e}")
            return []
        if audio_result is None:
            return []
        waveform, sample_rate = audio_result

        try:
            segments = self._get_vad_segments(waveform, sample_rate)
            if not segments:
                logger.warning("No speech segments detected by VAD")
                if self.nested:
                    task.data[self.segments_key] = []
                    return task
                return []

            original_file = task.data.get(self.audio_filepath_key, "unknown")
            file_name = os.path.basename(original_file) if original_file != "unknown" else task.task_id
            total_duration = sum((s["end"] - s["start"]) for s in segments)
            logger.info(
                f"[VADSegmentation] {file_name}: {len(segments)} segments extracted ({total_duration:.1f}s total speech)"
            )

            if self.nested:
                task.data[self.segments_key] = [
                    self._build_segment_item(task.data, waveform, sample_rate, seg, i)
                    for i, seg in enumerate(segments)
                ]
                task.data.pop(self.waveform_key, None)
                return task

            output_tasks: list[AudioTask] = []
            for i, segment in enumerate(segments):
                seg_data = self._build_segment_item(task.data, waveform, sample_rate, segment, i)
                seg_task = AudioTask(
                    data=seg_data,
                    dataset_name=task.dataset_name,
                    _metadata=dict(task._metadata or {}),
                    _stage_perf=list(task._stage_perf),
                )
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
