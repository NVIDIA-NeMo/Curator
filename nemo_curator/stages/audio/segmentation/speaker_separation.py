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

import os
import tempfile
import threading
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import soundfile as sf
from loguru import logger
from pydub import AudioSegment

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioBatch

from ..configs import SpeakerSeparationConfig


def _load_audio_as_pydub(audio_path: str) -> AudioSegment:
    """
    Load audio file as PyDub AudioSegment.
    
    Supports standalone usage of stages without requiring previous stages.
    Supports multiple audio formats: wav, mp3, flac, ogg, m4a, aac, wma, opus, webm.
    
    Note: Non-wav formats require ffmpeg to be installed on the system.
    """
    return AudioSegment.from_file(audio_path)


def _load_audio_as_tensor(audio_path: str):
    """
    Load audio file and return waveform tensor and sample rate.
    
    Supports standalone usage of stages without requiring previous stages.
    """
    data, sample_rate = sf.read(audio_path, dtype='float32')
    waveform = torch.from_numpy(data)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    else:
        waveform = waveform.T
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform, sample_rate


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
        self._init_lock = None  # Lazy initialization to avoid pickle issues
        
        # Apply config if provided
        if self.config is not None:
            if hasattr(self.config, 'model_path') and self.config.model_path:
                self.model_path = self.config.model_path
            if hasattr(self.config, 'exclude_overlaps'):
                self.exclude_overlaps = self.config.exclude_overlaps
            if hasattr(self.config, 'min_duration'):
                self.min_duration = self.config.min_duration
            if hasattr(self.config, 'gap_threshold'):
                self.gap_threshold = self.config.gap_threshold
            if hasattr(self.config, 'buffer_time'):
                self.buffer_time = self.config.buffer_time
            # Apply resources from config
            if hasattr(self.config, 'cpus') and hasattr(self.config, 'gpus'):
                self.resources = Resources(cpus=self.config.cpus, gpus=self.config.gpus)
    
    def __getstate__(self):
        """Return state for pickling, excluding unpicklable objects."""
        state = self.__dict__.copy()
        # Remove the lock and separator - they'll be recreated
        state['_init_lock'] = None
        state['_separator'] = None
        return state
    
    def __setstate__(self, state):
        """Restore state after unpickling."""
        self.__dict__.update(state)
        self._init_lock = None
        self._separator = None
    
    def _get_lock(self):
        """Get or create the initialization lock (lazy initialization)."""
        if self._init_lock is None:
            self._init_lock = threading.Lock()
        return self._init_lock
    
    def inputs(self) -> Tuple[List[str], List[str]]:
        """Define required inputs."""
        return [], ["audio"]
    
    def outputs(self) -> Tuple[List[str], List[str]]:
        """Define outputs produced by this stage."""
        return [], ["speaker_id", "num_speakers", "duration_sec"]
    
    def setup(self, worker_metadata=None) -> None:
        """Load NeMo diarization model on worker initialization."""
        self._initialize_separator()
    
    def teardown(self) -> None:
        """Clean up resources."""
        if self._separator is not None:
            del self._separator
            self._separator = None
            torch.cuda.empty_cache()
    
    def _resolve_model_path(self) -> str:
        """Resolve model path to absolute path."""
        if os.path.isabs(self.model_path):
            return self.model_path
        
        # Try relative to speaker_separation_module first (default location)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        module_dir = os.path.join(current_dir, 'speaker_separation_module')
        resolved = os.path.join(module_dir, self.model_path)
        if os.path.exists(resolved):
            return resolved
        
        # Try relative to segmentation directory
        resolved = os.path.join(current_dir, self.model_path)
        if os.path.exists(resolved):
            return resolved
        
        # Return the module path as default
        return os.path.join(module_dir, self.model_path)
    
    def _initialize_separator(self):
        """Initialize the NeMo speaker separator (thread-safe)."""
        if self._separator is None:
            with self._get_lock():
                if self._separator is None:
                    try:
                        # Import from local speaker_separation_module
                        from nemo_curator.stages.audio.segmentation.speaker_separation_module.speaker_sep import SpeakerSeparator
                        
                        model_path = self._resolve_model_path()
                        
                        # Determine GPU usage based on resources
                        use_gpu = self._resources.gpus > 0 and torch.cuda.is_available()
                        
                        # Create config for the separator
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
            audio = item.get('audio')
            waveform = item.get('waveform')
            sample_rate = item.get('sample_rate', 48000)
            
            # Auto-load from file if audio/waveform not provided (standalone usage)
            if audio is None and waveform is None:
                audio_filepath = item.get('audio_filepath')
                if audio_filepath and os.path.exists(audio_filepath):
                    try:
                        audio = _load_audio_as_pydub(audio_filepath)
                        waveform, sample_rate = _load_audio_as_tensor(audio_filepath)
                        item['audio'] = audio
                        item['waveform'] = waveform
                        item['sample_rate'] = sample_rate
                    except Exception as e:
                        logger.error(f"Failed to load audio file {audio_filepath}: {e}")
                        continue
                else:
                    logger.warning("No audio/waveform or valid audio_filepath found")
                    continue
            
            try:
                # Use the get_speaker_audio_data method from SpeakerSeparator
                if audio is not None:
                    # Export audio to temp file for processing
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                        temp_path = temp_file.name
                    
                    try:
                        audio.export(temp_path, format="wav")
                        
                        speaker_audio_data = self._separator.get_speaker_audio_data(
                            temp_path,
                            sample_rate=None,
                            gap_threshold=self.gap_threshold,
                            exclude_overlaps=self.exclude_overlaps,
                            min_duration=self.min_duration,
                            buffer_time=self.buffer_time
                        )
                    finally:
                        if os.path.exists(temp_path):
                            os.unlink(temp_path)
                else:
                    # Use waveform directly
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
                
                # Create output for each speaker
                for speaker_id, (speaker_audio, duration) in speaker_audio_data.items():
                    if duration < self.min_duration:
                        logger.debug(f"Skipping {speaker_id}: duration {duration:.2f}s < {self.min_duration}s")
                        continue
                    
                    speaker_data = {
                        **{k: v for k, v in item.items() if k not in ['audio', 'waveform']},
                        'audio': speaker_audio,
                        'speaker_id': speaker_id,
                        'num_speakers': num_speakers,
                        'duration_sec': duration,
                    }
                    
                    results.append(AudioBatch(
                        data=[speaker_data],
                        task_id=f"{task.task_id}_{speaker_id}",
                        dataset_name=task.dataset_name
                    ))
                        
            except Exception as e:
                logger.exception(f"Error in speaker separation: {e}")
        
        return results
