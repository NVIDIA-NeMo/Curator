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
Audio segment concatenation stage.

Concatenates multiple audio segments with silence in between.

Example:
    from nemo_curator.stages.audio.preprocessing import SegmentConcatenationStage
    
    stage = SegmentConcatenationStage(silence_duration_sec=1.0)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from loguru import logger
from pydub import AudioSegment

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioBatch

from ..configs import SegmentConcatenationConfig


@dataclass
class SegmentConcatenationStage(ProcessingStage[AudioBatch, AudioBatch]):
    """
    Concatenate audio segments with silence in between.
    
    This stage takes multiple AudioBatch inputs and concatenates their
    audio into a single AudioBatch output.
    
    Args:
        config: SegmentConcatenationConfig object (overrides other params if provided)
        silence_duration_sec: Duration of silence between segments (seconds)
        audio_key: Key in data dict containing audio
    
    Example:
        # Basic usage
        stage = SegmentConcatenationStage(silence_duration_sec=1.0)
        
        # Using config object
        config = SegmentConcatenationConfig(silence_duration_sec=0.5)
        stage = SegmentConcatenationStage(config=config)
    """
    
    config: Optional[SegmentConcatenationConfig] = None
    silence_duration_sec: float = 1.0
    audio_key: str = "audio"
    
    name: str = "SegmentConcatenation"
    batch_size: int = 1
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))
    
    def __post_init__(self):
        """Initialize after dataclass fields are set."""
        super().__init__()
        
        # Apply config if provided
        if self.config is not None:
            self.silence_duration_sec = self.config.silence_duration_sec
            self.audio_key = self.config.audio_key
    
    def inputs(self) -> Tuple[List[str], List[str]]:
        """Define required inputs."""
        return [], [self.audio_key]
    
    def outputs(self) -> Tuple[List[str], List[str]]:
        """Define outputs produced by this stage."""
        return [], [self.audio_key, "num_segments", "total_duration_sec", "concatenated"]
    
    def process_batch(self, tasks: List[AudioBatch]) -> List[AudioBatch]:
        """
        Concatenate multiple audio segments.
        
        Args:
            tasks: List of AudioBatch objects to concatenate
            
        Returns:
            List containing single AudioBatch with concatenated audio
        """
        if not tasks:
            logger.error("No segments to concatenate")
            return []
        
        try:
            silence = AudioSegment.silent(duration=int(self.silence_duration_sec * 1000))
            combined = AudioSegment.empty()
            segment_count = 0
            
            for i, task in enumerate(tasks):
                for item in task.data:
                    audio = item.get(self.audio_key)
                    
                    if isinstance(audio, AudioSegment):
                        combined += audio
                        segment_count += 1
                        
                        logger.debug(f"Added segment {i}: {len(audio)/1000:.2f}s")
                        
                        # Add silence between segments (except after last)
                        if i < len(tasks) - 1:
                            combined += silence
                    else:
                        logger.warning(f"Segment {i} has no AudioSegment")
            
            if segment_count == 0:
                logger.error("No valid audio segments found")
                return []
            
            total_duration = len(combined) / 1000.0
            
            logger.debug(f"Concatenated {segment_count} segments: {total_duration:.2f}s")
            
            output_data = {
                self.audio_key: combined,
                'num_segments': segment_count,
                'total_duration_sec': total_duration,
                'concatenated': True,
                'sample_rate': 48000,
            }
            
            return [AudioBatch(
                data=output_data,
                task_id="concatenated",
                dataset_name=tasks[0].dataset_name if tasks else "",
            )]
            
        except Exception as e:
            logger.error(f"Error concatenating segments: {e}")
            return []
    
    def process(self, task: AudioBatch) -> Optional[AudioBatch]:
        """Process single task (delegates to process_batch)."""
        results = self.process_batch([task])
        return results[0] if results else None
