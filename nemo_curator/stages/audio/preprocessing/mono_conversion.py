# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
Audio mono conversion stage.

Converts multi-channel audio to mono and verifies sample rate.
Typically the first stage in an audio processing pipeline.

Example:
    from nemo_curator.pipeline import Pipeline
    from nemo_curator.stages.audio.preprocessing import MonoConversionStage
    
    pipeline = Pipeline(name="audio_pipeline")
    pipeline.add_stage(MonoConversionStage(output_sample_rate=48000))
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torchaudio
import soundfile as sf
from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioBatch

from ..configs import MonoConversionConfig


@dataclass
class MonoConversionStage(ProcessingStage[AudioBatch, AudioBatch]):
    """
    Audio mono conversion and sample rate verification stage.
    
    Converts multi-channel audio to mono by averaging channels.
    Optionally verifies that audio matches expected sample rate.
    
    Args:
        config: MonoConversionConfig object (overrides other params if provided)
        output_sample_rate: Expected sample rate in Hz (default: 48000)
        audio_filepath_key: Key in data dict for audio file path
        strict_sample_rate: If True, reject audio with wrong sample rate
    
    Example:
        # Basic usage
        stage = MonoConversionStage(output_sample_rate=48000)
        
        # Using config object
        config = MonoConversionConfig(output_sample_rate=16000)
        stage = MonoConversionStage(config=config)
        
        # Allow any sample rate
        stage = MonoConversionStage(strict_sample_rate=False)
    """
    
    config: Optional[MonoConversionConfig] = None
    output_sample_rate: int = 48000
    audio_filepath_key: str = "audio_filepath"
    strict_sample_rate: bool = True
    
    name: str = "MonoConversion"
    batch_size: int = 1
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))
    
    def __post_init__(self):
        """Initialize after dataclass fields are set."""
        super().__init__()
        
        # Apply config if provided
        if self.config is not None:
            self.output_sample_rate = self.config.output_sample_rate
            self.audio_filepath_key = self.config.audio_filepath_key
            self.strict_sample_rate = self.config.strict_sample_rate
    
    def inputs(self) -> Tuple[List[str], List[str]]:
        """Define required inputs."""
        return [], [self.audio_filepath_key]
    
    def outputs(self) -> Tuple[List[str], List[str]]:
        """Define outputs produced by this stage."""
        return [], ["waveform", "sample_rate", "is_mono", "duration", "num_samples"]
    
    def process(self, task: AudioBatch) -> Optional[AudioBatch]:
        """
        Convert audio to mono and verify sample rate.
        
        Args:
            task: AudioBatch with audio_filepath in data
            
        Returns:
            AudioBatch with waveform data, or None if doesn't meet requirements
        """
        results = []
        
        for item in task.data:
            audio_filepath = item.get(self.audio_filepath_key)
            
            if not audio_filepath or not os.path.exists(audio_filepath):
                logger.error(f"Audio file not found: {audio_filepath}")
                continue
            
            try:
                # Load audio using soundfile (more reliable than torchaudio backends)
                audio_data, sample_rate = sf.read(audio_filepath, dtype='float32')
                
                # Convert to torch tensor and reshape to (channels, samples)
                waveform = torch.from_numpy(audio_data)
                if waveform.dim() == 1:
                    # Mono audio: reshape to (1, samples)
                    waveform = waveform.unsqueeze(0)
                else:
                    # Multi-channel: transpose from (samples, channels) to (channels, samples)
                    waveform = waveform.T
                
                num_channels = waveform.shape[0]
                
                if self.strict_sample_rate and sample_rate != self.output_sample_rate:
                    logger.warning(
                        f"Sample rate {sample_rate}Hz != expected {self.output_sample_rate}Hz: {audio_filepath}"
                    )
                    continue
                
                sr = sample_rate
                
                if num_channels > 1:
                    mono_waveform = torch.mean(waveform, dim=0, keepdim=True)
                    logger.debug(f"Converted {num_channels} channels to mono")
                else:
                    mono_waveform = waveform
                
                item['waveform'] = mono_waveform
                item['sample_rate'] = sr
                item['is_mono'] = True
                item['duration'] = mono_waveform.shape[1] / sr
                item['num_samples'] = mono_waveform.shape[1]
                
                results.append(item)
                
            except Exception as e:
                logger.error(f"Error processing {audio_filepath}: {e}")
                continue
        
        if not results:
            return None
        
        return AudioBatch(data=results, task_id=task.task_id, dataset_name=task.dataset_name)
