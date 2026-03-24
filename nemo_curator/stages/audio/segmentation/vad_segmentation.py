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
    
    # Default execution (GPU with cpus=1.0, gpus=0.3)
    pipeline.add_stage(VADSegmentationStage(min_duration_sec=2.0, threshold=0.5))
    
    # Custom GPU allocation
    pipeline.add_stage(
        VADSegmentationStage(min_duration_sec=2.0)
        .with_(resources=Resources(gpus=0.1))
    )
"""

import os
import warnings
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
import torchaudio
from loguru import logger

try:
    from silero_vad import load_silero_vad, get_speech_timestamps
    _SILERO_AVAILABLE = True
except ImportError:
    _SILERO_AVAILABLE = False

from nemo_curator.backends.experimental.utils import RayStageSpecKeys
from nemo_curator.stages.audio.common import ensure_waveform_2d, load_audio_file
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioBatch

SILERO_SUPPORTED_RATES = {8000, 16000, 32000, 48000, 64000, 96000}
SILERO_TARGET_RATE = 16000


@dataclass
class VADSegmentationStage(ProcessingStage[AudioBatch, AudioBatch]):
    """
    Stage to segment audio using Voice Activity Detection (VAD).
    
    This stage takes audio and segments it into speech chunks based on VAD,
    filtering out silence and creating manageable segments for further processing.
    Uses Silero VAD model loaded via torch.hub.
    
    Args:
        mode (str): Output mode. "fanout" (default) returns list[AudioBatch] with one
            task per segment (fan-out, IS_FANOUT_STAGE=True). "batch" returns a single
            AudioBatch with all segments as items (1:1, no fan-out).
        min_interval_ms (int): Minimum silence interval between speech segments in milliseconds. Default: 500
        min_duration_sec (float): Minimum segment duration in seconds. Default: 2.0
        max_duration_sec (float): Maximum segment duration in seconds. Default: 60.0
        threshold (float): Voice activity detection threshold (0.0-1.0). Default: 0.5
        speech_pad_ms (int): Padding in ms to add before/after speech segments. Default: 300
        waveform_key (str): Key to get waveform data. Default: "waveform"
        sample_rate_key (str): Key to get sample rate. Default: "sample_rate"
        
    Returns:
        mode="fanout": List of AudioBatch objects, one per detected speech segment.
        mode="batch": Single AudioBatch with all segments as items (or None if empty).
        
        Each segment item contains (canonical format):
        - waveform: torch.Tensor of the segment
        - sample_rate: Sample rate
        - start_ms, end_ms: Segment time in milliseconds
        - segment_num: Segment index
        - duration_sec: Segment duration in seconds
        - original_file: Source file path (from audio_filepath)
        
    Example:
        # Fan-out mode (default) - one task per segment
        stage = VADSegmentationStage(min_duration_sec=3.0, max_duration_sec=30.0)
        
        # Batch mode - all segments in one task (for use in decomposed pipelines)
        stage = VADSegmentationStage(mode="batch", min_duration_sec=3.0)
        
        # Custom GPU allocation
        stage = VADSegmentationStage().with_(resources=Resources(gpus=0.1))
    
    Note:
        Default resources: cpus=1.0, gpus=0.3. Silero VAD is lightweight.
        Use .with_(resources=Resources(gpus=X)) to override GPU allocation.
        GPU is used automatically when resources specify gpus > 0.
    """
    
    mode: Literal["fanout", "batch"] = "fanout"
    min_interval_ms: int = 500
    min_duration_sec: float = 2.0
    max_duration_sec: float = 60.0
    threshold: float = 0.5
    speech_pad_ms: int = 300
    waveform_key: str = "waveform"
    sample_rate_key: str = "sample_rate"
    
    name: str = "VADSegmentation"
    batch_size: int = 1
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0, gpus=0.3))
    
    def __post_init__(self):
        """Initialize after dataclass fields are set."""
        super().__init__()
        
        if self.mode not in ("fanout", "batch"):
            raise ValueError(f"mode must be 'fanout' or 'batch', got '{self.mode}'")
        if not _SILERO_AVAILABLE:
            raise ImportError("silero_vad is required for VADSegmentationStage. Install it with: pip install silero-vad")
        
        self._vad_model = None
        self._device = None
    
    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        """Define outputs."""
        return [], ['waveform', 'sample_rate', 'start_ms', 'end_ms', 'segment_num', 'duration_sec']
    
    def ray_stage_spec(self) -> dict[str, Any]:
        if self.mode == "fanout":
            return {RayStageSpecKeys.IS_FANOUT_STAGE: True}
        return {}

    def setup(self, worker_metadata=None) -> None:
        """Load VAD model on worker initialization."""
        self._initialize_model()
    
    def teardown(self) -> None:
        """Clean up resources."""
        if self._vad_model is not None:
            del self._vad_model
            self._vad_model = None
            if self._device is not None and self._device.type == "cuda":
                torch.cuda.empty_cache()
    
    def _initialize_model(self):
        """Initialize the VAD model."""
        if self._vad_model is not None:
            return
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Sampling rate is a multiply of 16000")
                model = load_silero_vad()
            
            if self._resources.gpus > 0 and not torch.cuda.is_available():
                raise RuntimeError(
                    "Resources request GPU (gpus > 0) but CUDA is not available. "
                    "Either set resources=Resources(gpus=0) for CPU-only or install CUDA."
                )
            use_gpu = self._resources.gpus > 0 and torch.cuda.is_available()
            
            if use_gpu:
                self._device = torch.device("cuda")
                model = model.to(self._device)
                logger.info(f"Silero VAD model loaded on GPU: {self._device}")
            else:
                self._device = torch.device('cpu')
                logger.info("Silero VAD model loaded on CPU")
            
            self._vad_model = model
        except Exception as e:
            logger.error(f"Failed to load VAD model: {e}")
            raise
    
    @property
    def vad_model(self):
        """Get VAD model."""
        self._initialize_model()
        return self._vad_model
    
    def _build_segment_items(
        self, item: dict[str, Any], waveform: torch.Tensor, sample_rate: int,
        segments: list[dict[str, float]], segment_offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Build segment item dicts from VAD results. Shared by both modes."""
        items = []
        for i, segment in enumerate(segments):
            start_ms = int(segment['start'] * 1000)
            end_ms = int(segment['end'] * 1000)
            start_sample = int(segment['start'] * sample_rate)
            end_sample = int(segment['end'] * sample_rate)

            if waveform.dim() == 1:
                segment_waveform = waveform[start_sample:end_sample].unsqueeze(0).clone()
            else:
                segment_waveform = waveform[:, start_sample:end_sample].clone()

            segment_data: dict[str, Any] = {
                k: v for k, v in item.items()
                if k not in (self.waveform_key, self.sample_rate_key, 'start_ms', 'end_ms',
                             'segment_num', 'duration_sec', 'duration', 'num_samples')
            }
            segment_data.update({
                'waveform': segment_waveform,
                'sample_rate': sample_rate,
                'start_ms': start_ms,
                'end_ms': end_ms,
                'segment_num': segment_offset + i,
                'duration_sec': (end_ms - start_ms) / 1000.0,
                'original_file': item.get('original_file', item.get('audio_filepath', 'unknown')),
            })
            items.append(segment_data)
        return items

    def process(self, task: AudioBatch):
        """
        Process audio and return segments.

        In ``mode="fanout"`` (default): returns ``List[AudioBatch]`` with one
        task per segment (fan-out, compatible with IS_FANOUT_STAGE).
        In ``mode="batch"``: returns a single ``AudioBatch`` with all segments
        as items (1:1, no fan-out).
        """
        self._initialize_model()

        if self._vad_model is None:
            raise RuntimeError("VAD model failed to initialize. Cannot process audio.")

        all_segment_items: list[dict[str, Any]] = []

        if not task.data:
            if self.mode == "batch":
                return AudioBatch(data=[], task_id=task.task_id, dataset_name=task.dataset_name,
                                  _metadata=dict(task._metadata) if task._metadata else {},
                                  _stage_perf=list(task._stage_perf))
            return []

        for item in task.data:
            waveform = item.get(self.waveform_key)
            sample_rate = item.get(self.sample_rate_key)

            if waveform is None:
                audio_filepath = item.get("audio_filepath")
                if audio_filepath and os.path.exists(audio_filepath):
                    try:
                        waveform, sample_rate = load_audio_file(audio_filepath)
                        item[self.waveform_key] = waveform
                        item[self.sample_rate_key] = sample_rate
                    except Exception as e:
                        logger.error(f"Failed to load audio file {audio_filepath}: {e}")
                        continue
                else:
                    logger.error("Missing waveform and no valid audio_filepath provided")
                    continue
            elif sample_rate is None:
                logger.warning("Waveform present but sample_rate missing – item skipped")
                continue

            waveform = ensure_waveform_2d(waveform)

            try:
                segments = self._get_vad_segments(waveform, sample_rate)
                if not segments:
                    logger.warning("No speech segments detected by VAD")
                    continue

                seg_items = self._build_segment_items(
                    item, waveform, sample_rate, segments,
                    segment_offset=len(all_segment_items),
                )
                all_segment_items.extend(seg_items)

                total_duration = sum((s['end'] - s['start']) for s in segments)
                original_file = item.get('audio_filepath', 'unknown')
                file_name = os.path.basename(original_file) if original_file != 'unknown' else task.task_id
                logger.info(f"[VADSegmentation] {file_name}: {len(segments)} segments extracted ({total_duration:.1f}s total speech)")

            except Exception as e:
                logger.exception(f"Error during VAD segmentation: {e}")
                continue

        if self.mode == "batch":
            return AudioBatch(
                data=all_segment_items,
                task_id=task.task_id,
                dataset_name=task.dataset_name,
                _metadata=dict(task._metadata) if task._metadata else {},
                _stage_perf=list(task._stage_perf),
            )

        # Fan-out mode: one AudioBatch per segment item
        output_tasks: list[AudioBatch] = []
        for global_idx, seg_item in enumerate(all_segment_items):
            output_tasks.append(AudioBatch(
                data=seg_item,
                task_id=f"{task.task_id}_seg_{global_idx}",
                dataset_name=task.dataset_name,
                _metadata=dict(task._metadata) if task._metadata else {},
                _stage_perf=list(task._stage_perf),
            ))
        return output_tasks
    
    def _get_vad_segments(self, waveform: torch.Tensor, sample_rate: int) -> list[dict[str, float]]:
        """Get speech segments using VAD."""
        # Ensure waveform is 1D for VAD
        if waveform.dim() > 1:
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0)
            else:
                waveform = waveform.squeeze(0)
        
        # Move waveform to the same device as the model
        if self._device is not None and waveform.device != self._device:
            waveform = waveform.to(self._device)
        
        # Resample if sample rate is not supported by Silero VAD
        # Silero only supports 8kHz, 16kHz, and multiples of 16kHz (32k, 48k, etc.)
        # Sample rates like 22050 Hz need to be resampled to 16kHz
        vad_sample_rate = sample_rate
        vad_waveform = waveform
        if sample_rate not in SILERO_SUPPORTED_RATES:
            logger.debug(f"Resampling audio from {sample_rate}Hz to {SILERO_TARGET_RATE}Hz for VAD")
            # Move to CPU for resampling if needed
            device = waveform.device
            waveform_cpu = waveform.cpu() if waveform.device.type != 'cpu' else waveform
            # Ensure 2D for torchaudio resampling (channels, samples)
            if waveform_cpu.dim() == 1:
                waveform_cpu = waveform_cpu.unsqueeze(0)
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=SILERO_TARGET_RATE)
            vad_waveform = resampler(waveform_cpu).squeeze(0)
            # Move back to original device
            if device.type != 'cpu':
                vad_waveform = vad_waveform.to(device)
            vad_sample_rate = SILERO_TARGET_RATE
        
        # Get speech timestamps using silero_vad package
        speech_timestamps = get_speech_timestamps(
            vad_waveform,
            self.vad_model,
            sampling_rate=vad_sample_rate,
            threshold=self.threshold,
            min_speech_duration_ms=self.min_duration_sec * 1000,
            max_speech_duration_s=self.max_duration_sec,
            min_silence_duration_ms=self.min_interval_ms,
            speech_pad_ms=self.speech_pad_ms
        )
        
        # Convert to seconds using the ORIGINAL sample rate
        # VAD timestamps are in samples of the resampled audio, need to convert back
        segments = []
        for ts in speech_timestamps:
            # Convert from resampled samples to original time
            start_sec = ts['start'] / vad_sample_rate
            end_sec = ts['end'] / vad_sample_rate
            segments.append({
                'start': start_sec,
                'end': end_sec
            })
        
        return segments
