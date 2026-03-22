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
        .with_(resources=Resources(gpus=1.0))
    )
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
from loguru import logger
from pydub import AudioSegment

from nemo_curator.backends.experimental.utils import RayStageSpecKeys
from nemo_curator.stages.audio.segmentation.speaker_separation_module.speaker_sep import SpeakerSeparator
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioBatch

from ..common import resolve_model_path, resolve_waveform_from_item
from ..configs import SpeakerSeparationConfig


def _pydub_to_waveform_sr(seg: AudioSegment) -> Tuple[torch.Tensor, int]:
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
    for each speaker's segments. Uses local NeMo model from 
    speaker_separation_module/model/diar_sortformer_4spk-v1.nemo.
    
    Args:
        config: SpeakerSeparationConfig object (overrides other params if provided)
        model_path: Path to NeMo diarization model (.nemo file)
        exclude_overlaps: Whether to exclude overlapping speaker regions
        min_duration: Minimum segment duration in seconds
        gap_threshold: Gap threshold for merging speaker segments
        buffer_time: Buffer time around speaker segments
    
    Note:
        GPU assignment is handled by the executor via _resources.
        Use .with_(resources=Resources(gpus=X)) to configure GPU allocation.
    
    Example:
        # Using config
        config = SpeakerSeparationConfig(exclude_overlaps=True)
        stage = SpeakerSeparationStage(config=config)
        
        # Using parameters
        stage = SpeakerSeparationStage(exclude_overlaps=True, min_duration=0.8)
    """
    
    config: Optional[SpeakerSeparationConfig] = None
    model_path: str = "model/diar_sortformer_4spk-v1.nemo"
    exclude_overlaps: bool = True
    min_duration: float = 0.8
    gap_threshold: float = 0.1
    buffer_time: float = 0.5
    
    name: str = "SpeakerSeparation"
    batch_size: int = 1
    resources: Resources = field(default_factory=lambda: Resources(gpus=1.0))
    
    def __post_init__(self):
        """Initialize after dataclass fields are set."""
        super().__init__()
        self._separator = None
        
        # Apply user-facing config fields only; gap_threshold, min_duration,
        # and buffer_time are internal stage params, not exposed via config.
        if self.config is not None:
            if self.config.model_path:
                self.model_path = self.config.model_path
            self.exclude_overlaps = self.config.exclude_overlaps
    
    def inputs(self) -> Tuple[List[str], List[str]]:
        return ["data"], []

    def outputs(self) -> Tuple[List[str], List[str]]:
        """Define outputs produced by this stage."""
        return [], ["waveform", "sample_rate", "speaker_id", "num_speakers", "duration_sec"]
    
    def ray_stage_spec(self) -> dict[str, Any]:
        return {RayStageSpecKeys.IS_FANOUT_STAGE: True}

    def setup(self, worker_metadata=None) -> None:
        """Load NeMo diarization model on worker initialization."""
        self._initialize_separator()
    
    def teardown(self) -> None:
        """Clean up resources."""
        if self._separator is not None:
            del self._separator
            self._separator = None
            torch.cuda.empty_cache()
    
    def _initialize_separator(self):
        """Initialize the NeMo speaker separator."""
        if self._separator is None:
            try:
                model_path = resolve_model_path(self.model_path, __file__, 'speaker_separation_module')
                use_gpu = self._resources.gpus > 0 and torch.cuda.is_available()
                
                separator_config = {
                    'speaker_model_path': model_path,
                    'speaker_gap_threshold': self.gap_threshold,
                    'speaker_exclude_overlaps': self.exclude_overlaps,
                    'speaker_min_duration': self.min_duration,
                    'speaker_buffer_time': self.buffer_time,
                    'use_gpu': use_gpu,
                }
                
                self._separator = SpeakerSeparator(
                    model_name=model_path,
                    config=separator_config
                )
                
                logger.info(f"NeMo speaker separator loaded from {model_path}")
            except ImportError as e:
                logger.error(f"Failed to import speaker separation module: {e}")
                raise
            except Exception as e:
                logger.error(f"Failed to load speaker separator: {e}")
                raise
    
    def process(self, task: AudioBatch) -> List[AudioBatch]:
        """
        Separate audio by speaker.
        
        Args:
            task: AudioBatch with audio data
            
        Returns:
            List of AudioBatch objects, one per speaker
        """
        self._initialize_separator()
        
        if self._separator is None:
            logger.error("Speaker separator not available")
            return []
        
        results = []
        
        for item in task.data:
            item = dict(item)
            audio_result = resolve_waveform_from_item(item, task.task_id)
            if audio_result is None:
                continue
            waveform, sample_rate = audio_result
            
            try:
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
                
                # Output canonical format only: waveform + sample_rate (no item['audio'])
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
                        
            except Exception as e:
                logger.exception(f"Error in speaker separation: {e}")
        
        return results
