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
Speaker separation stage using NeMo SortFormer diarization model.

Performs speaker diarization and separates audio by speaker,
creating separate AudioBatch outputs for each speaker.

Example:
    from nemo_curator.pipeline import Pipeline
    from nemo_curator.stages.audio.segmentation import SpeakerSeparationStage
    from nemo_curator.stages.resources import Resources
    
    pipeline = Pipeline(name="speaker_pipeline")
    pipeline.add_stage(
        SpeakerSeparationStage(exclude_overlaps=True, min_duration=0.8)
        .with_(resources=Resources(cpus=1.0, gpus=1.0))
    )
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from loguru import logger
from pydub import AudioSegment

from nemo_curator.backends.experimental.utils import RayStageSpecKeys
from nemo_curator.stages.audio.segmentation.speaker_separation_module.speaker_sep import SpeakerSeparator
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioBatch

from ..common import resolve_waveform_from_item


def _pydub_to_waveform_sr(seg: AudioSegment) -> tuple[torch.Tensor, int]:
    """Convert PyDub AudioSegment to (waveform, sample_rate). Output is canonical format only."""
    max_val = float(1 << (8 * seg.sample_width - 1))
    samples = np.array(seg.get_array_of_samples(), dtype=np.float32) / max_val
    if seg.channels > 1:
        samples = samples.reshape((-1, seg.channels)).mean(axis=1)
    return torch.from_numpy(samples).unsqueeze(0), seg.frame_rate


@dataclass
class SpeakerSeparationStage(ProcessingStage[AudioBatch, AudioBatch]):
    """
    Speaker separation stage using NeMo SortFormer diarization model.
    
    Separates audio by speaker and creates separate AudioBatch outputs
    for each speaker's segments. Downloads the NeMo model from
    HuggingFace Hub (nvidia/diar_sortformer_4spk-v1).
    
    Args:
        model_path: HuggingFace model ID or path to NeMo diarization model
        exclude_overlaps: Whether to exclude overlapping speaker regions
        min_duration: Minimum segment duration in seconds
        gap_threshold: Gap threshold for merging speaker segments
        buffer_time: Buffer time around speaker segments
    Note:
        GPU assignment is handled by the executor via _resources.
        Use .with_(resources=Resources(gpus=X)) to configure GPU allocation.
    
    Example:
        stage = SpeakerSeparationStage(exclude_overlaps=True, min_duration=0.8)
    """
    
    model_path: str = "nvidia/diar_sortformer_4spk-v1"
    exclude_overlaps: bool = True
    min_duration: float = 0.8
    gap_threshold: float = 0.1
    buffer_time: float = 0.5
    
    name: str = "SpeakerSeparation"
    batch_size: int = 1
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0, gpus=1.0))
    
    def __post_init__(self):
        """Initialize after dataclass fields are set."""
        super().__init__()
        self._separator = None
    
    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        """Define outputs produced by this stage."""
        return [], ["waveform", "sample_rate", "speaker_id", "num_speakers", "duration_sec"]
    
    def ray_stage_spec(self) -> dict[str, Any]:
        return {RayStageSpecKeys.IS_FANOUT_STAGE: True}

    def setup_on_node(self, _node_info=None, _worker_metadata=None) -> None:  # noqa: ARG002
        """Download the NeMo diarization model from HuggingFace (once per node)."""
        try:
            from nemo.collections.asr.models import SortformerEncLabelModel
            SortformerEncLabelModel.from_pretrained(self.model_path)
        except Exception:
            logger.warning("Model pre-download in setup_on_node failed; will retry in setup().")

    def setup(self, _worker_metadata=None) -> None:  # noqa: ARG002
        """Load NeMo diarization model on worker initialization."""
        self._initialize_separator()
    
    def teardown(self) -> None:
        """Clean up resources."""
        if self._separator is not None:
            del self._separator
            self._separator = None
            torch.cuda.empty_cache()
    
    def _initialize_separator(self):
        """Initialize the NeMo speaker separator with HuggingFace model."""
        if self._separator is None:
            try:
                if self._resources.gpus > 0 and not torch.cuda.is_available():
                    raise RuntimeError(
                        "Resources request GPU (gpus > 0) but CUDA is not available. "
                        "Either set resources=Resources(gpus=0) for CPU-only or install CUDA."
                    )
                use_gpu = self._resources.gpus > 0 and torch.cuda.is_available()
                
                separator_config = {
                    'speaker_model_path': self.model_path,
                    'speaker_gap_threshold': self.gap_threshold,
                    'speaker_exclude_overlaps': self.exclude_overlaps,
                    'speaker_min_duration': self.min_duration,
                    'speaker_buffer_time': self.buffer_time,
                    'use_gpu': use_gpu,
                }
                
                self._separator = SpeakerSeparator(
                    model_name=self.model_path,
                    config=separator_config,
                )
                
                logger.info(f"NeMo speaker separator loaded from HuggingFace: {self.model_path}")
            except ImportError as e:
                logger.error(f"Failed to import speaker separation module: {e}")
                raise
            except Exception as e:
                logger.error(f"Failed to load speaker separator: {e}")
                raise
    
    def process(self, task: AudioBatch) -> list[AudioBatch]:
        """
        Separate audio by speaker.
        
        Args:
            task: AudioBatch with audio data
            
        Returns:
            List of AudioBatch objects, one per speaker
        """
        self._initialize_separator()
        
        if self._separator is None:
            raise RuntimeError("Speaker separator failed to initialize. Cannot process audio.")
        
        results = []
        
        for item_idx, item in enumerate(task.data):
            item = dict(item)
            waveform = None
            try:
                audio_result = resolve_waveform_from_item(item, task.task_id)
                if audio_result is None:
                    continue
                waveform, sample_rate = audio_result
            
                speaker_audio_data = self._separator.get_speaker_audio_data(
                    waveform,
                    sample_rate=sample_rate,
                    gap_threshold=self.gap_threshold,
                    exclude_overlaps=self.exclude_overlaps,
                    min_duration=self.min_duration,
                    buffer_time=self.buffer_time
                )
                
                num_speakers = len(speaker_audio_data)
                
                if num_speakers == 0:
                    logger.warning("No speakers detected")
                    continue
                
                logger.info(f"Detected {num_speakers} speakers")
                
                for speaker_id, (speaker_audio_pydub, duration) in speaker_audio_data.items():
                    if duration < self.min_duration:
                        logger.debug(f"Skipping {speaker_id}: duration {duration:.2f}s < {self.min_duration}s")
                        continue
                    spk_waveform, spk_sr = _pydub_to_waveform_sr(speaker_audio_pydub)
                    speaker_data = {
                        **{k: v for k, v in item.items() if k not in ('audio', 'waveform')},
                        'waveform': spk_waveform,
                        'sample_rate': spk_sr,
                        'speaker_id': speaker_id,
                        'num_speakers': num_speakers,
                        'duration_sec': duration,
                    }
                    results.append(AudioBatch(
                        data=[speaker_data],
                        task_id=f"{task.task_id}_{speaker_id}",
                        dataset_name=task.dataset_name,
                        _metadata=dict(task._metadata) if task._metadata else {},
                        _stage_perf=list(task._stage_perf),
                    ))
                        
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                raise RuntimeError(
                    f"CUDA out of memory processing item {item_idx}. "
                    "Consider splitting long audio files into shorter segments, "
                    "using a GPU with more memory, or setting resources=Resources(gpus=0) for CPU mode."
                )
            except Exception as e:
                logger.warning(f"Skipping item {item_idx}: {e}")
            finally:
                if waveform is not None:
                    del waveform
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        
        return results
