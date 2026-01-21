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
VAD (Voice Activity Detection) segmentation stage.

Segments audio into speech chunks using Silero VAD model,
filtering out silence and creating manageable segments for further processing.

Supports both CPU and GPU execution. GPU is used when available and requested
via _resources configuration.

Example:
    from nemo_curator.pipeline import Pipeline
    from nemo_curator.stages.audio.segmentation import VADSegmentationStage
    from nemo_curator.stages.resources import Resources
    
    # CPU execution (default)
    pipeline.add_stage(VADSegmentationStage(min_duration_sec=2.0, threshold=0.5))
    
    # GPU execution
    pipeline.add_stage(
        VADSegmentationStage(min_duration_sec=2.0)
        .with_(resources=Resources(gpus=0.1))
    )
"""

import os
import threading
import warnings
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import torch
import torchaudio
import soundfile as sf
from loguru import logger
from pydub import AudioSegment
from silero_vad import load_silero_vad, get_speech_timestamps

# Suppress Silero VAD sample rate warning (48kHz -> 16kHz is expected)
warnings.filterwarnings('ignore', message='Sampling rate is a multiply of 16000')

# Silero VAD only supports 8kHz, 16kHz, and multiples of 16kHz (32k, 48k, etc.)
# Sample rates like 22050 Hz need to be resampled to 16kHz
SILERO_SUPPORTED_RATES = {8000, 16000, 32000, 48000, 64000, 96000}
SILERO_TARGET_RATE = 16000

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioBatch
from nemo_curator.stages.audio.configs.vad import VADConfig


def _load_audio_file(audio_path: str) -> Tuple[torch.Tensor, int]:
    """
    Load audio file and return waveform tensor and sample rate.
    
    Supports standalone usage of stages without requiring MonoConversionStage.
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
class VADSegmentationStage(ProcessingStage[AudioBatch, AudioBatch]):
    """
    Stage to segment audio using Voice Activity Detection (VAD).
    
    This stage takes audio and segments it into speech chunks based on VAD,
    filtering out silence and creating manageable segments for further processing.
    Uses Silero VAD model loaded via torch.hub.
    
    Args:
        config (VADConfig): Optional configuration object. If provided, overrides individual params.
        min_interval_ms (int): Minimum silence interval between speech segments in milliseconds. Default: 500
        min_duration_sec (float): Minimum segment duration in seconds. Default: 2.0
        max_duration_sec (float): Maximum segment duration in seconds. Default: 60.0
        threshold (float): Voice activity detection threshold (0.0-1.0). Default: 0.5
        speech_pad_ms (int): Padding in ms to add before/after speech segments. Default: 300
        waveform_key (str): Key to get waveform data. Default: "waveform"
        sample_rate_key (str): Key to get sample rate. Default: "sample_rate"
        
    Returns:
        List of AudioBatch objects, one per detected speech segment, each containing:
        - audio: PyDub AudioSegment
        - waveform: torch.Tensor of the segment
        - sample_rate: Sample rate
        - start_ms: Start time in milliseconds
        - end_ms: End time in milliseconds
        - segment_num: Segment index
        - duration_sec: Segment duration in seconds
        
    Example:
        # Using direct parameters
        stage = VADSegmentationStage(min_duration_sec=3.0, max_duration_sec=30.0)
        
        # Using config object
        config = VADConfig(min_duration_sec=3.0, max_duration_sec=30.0)
        stage = VADSegmentationStage(config=config)
        
        # With GPU support (via resources)
        stage = VADSegmentationStage().with_(resources=Resources(gpus=0.1))
    
    Note:
        GPU assignment is handled by the executor via _resources.
        Use .with_(resources=Resources(gpus=X)) to configure GPU allocation.
        Silero VAD is lightweight, so gpus=0.1 is usually sufficient.
        GPU is used automatically when resources specify gpus > 0.
    """
    
    config: Optional[VADConfig] = None
    min_interval_ms: int = 500
    min_duration_sec: float = 2.0
    max_duration_sec: float = 60.0
    threshold: float = 0.5
    speech_pad_ms: int = 300
    waveform_key: str = "waveform"
    sample_rate_key: str = "sample_rate"
    
    name: str = "VADSegmentation"
    batch_size: int = 1
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))
    
    def __post_init__(self):
        """Initialize after dataclass fields are set."""
        super().__init__()
        
        # Apply config if provided
        if self.config is not None:
            self.min_interval_ms = self.config.min_interval_ms
            self.min_duration_sec = self.config.min_duration_sec
            self.max_duration_sec = self.config.max_duration_sec
            self.threshold = self.config.threshold
            self.speech_pad_ms = self.config.speech_pad_ms
            # Apply resources from config
            self.resources = Resources(cpus=self.config.cpus, gpus=self.config.gpus)
        
        self._vad_model = None
        self._vad_utils = None
        self._init_lock = None  # Lazy initialization to avoid pickle issues
        self._device = None  # Will be set during model initialization
    
    def __getstate__(self):
        """Return state for pickling, excluding unpicklable objects."""
        state = self.__dict__.copy()
        # Remove the lock and model - they'll be recreated
        state['_init_lock'] = None
        state['_vad_model'] = None
        state['_vad_utils'] = None
        state['_device'] = None
        return state
    
    def __setstate__(self, state):
        """Restore state after unpickling."""
        self.__dict__.update(state)
        self._init_lock = None
        self._vad_model = None
        self._vad_utils = None
        self._device = None
    
    def _get_lock(self):
        """Get or create the initialization lock (lazy initialization)."""
        if self._init_lock is None:
            self._init_lock = threading.Lock()
        return self._init_lock
    
    def inputs(self) -> Tuple[List[str], List[str]]:
        """Define required inputs."""
        return [], [self.waveform_key, self.sample_rate_key]
    
    def outputs(self) -> Tuple[List[str], List[str]]:
        """Define outputs."""
        return [], ['audio', 'waveform', 'sample_rate', 'start_ms', 'end_ms', 'segment_num', 'duration_sec']
    
    def setup(self, worker_metadata=None) -> None:
        """Load VAD model on worker initialization."""
        self._initialize_model()
    
    def teardown(self) -> None:
        """Clean up resources."""
        if self._vad_model is not None:
            del self._vad_model
            del self._vad_utils
            self._vad_model = None
            self._vad_utils = None
            torch.cuda.empty_cache()
    
    def _initialize_model(self):
        """Initialize the VAD model (thread-safe with double-checked locking)."""
        if self._vad_model is None:
            with self._get_lock():
                if self._vad_model is None:
                    try:
                        # Use silero-vad pip package instead of torch.hub.load
                        # This avoids GitHub API rate limiting issues
                        model = load_silero_vad()
                        
                        # Determine device based on resources and CUDA availability
                        # Use GPU if _resources.gpus > 0 and CUDA is available
                        use_gpu = self._resources.gpus > 0 and torch.cuda.is_available()
                        
                        if use_gpu:
                            self._device = torch.device(f'cuda:{torch.cuda.current_device()}')
                            model = model.to(self._device)
                            logger.info(f"Silero VAD model loaded on GPU: {self._device}")
                        else:
                            self._device = torch.device('cpu')
                            logger.info("Silero VAD model loaded on CPU")
                        
                        self._vad_model = model
                        self._vad_utils = None  # No longer needed, using get_speech_timestamps directly
                    except Exception as e:
                        logger.error(f"Failed to load VAD model: {e}")
                        raise
    
    @property
    def vad_model(self):
        """Get VAD model (lazy load with thread safety)."""
        self._initialize_model()
        return self._vad_model
    
    @property 
    def vad_utils(self):
        """Get VAD utilities."""
        self._initialize_model()
        return self._vad_utils
    
    def process(self, task: AudioBatch) -> List[AudioBatch]:
        """
        Process audio and return segments.
        
        Note: AudioBatch.data is always a list of dictionaries.
        This stage creates multiple output AudioBatch objects (one per segment).
        
        Args:
            task: AudioBatch with waveform data in task.data[i]
            
        Returns:
            List of AudioBatch objects, one per detected speech segment
        """
        # Ensure model is initialized
        self._initialize_model()
        
        if self._vad_model is None:
            logger.error("VAD model not available")
            return []
        
        output_tasks = []
        
        # Iterate over all items in the batch (AudioBatch.data is always a list)
        for item in task.data:
            waveform = item.get(self.waveform_key)
            sample_rate = item.get(self.sample_rate_key)
            
            # Auto-load from file if waveform not provided (standalone usage)
            if waveform is None or sample_rate is None:
                audio_filepath = item.get('audio_filepath')
                if audio_filepath and os.path.exists(audio_filepath):
                    try:
                        waveform, sample_rate = _load_audio_file(audio_filepath)
                        item['waveform'] = waveform
                        item['sample_rate'] = sample_rate
                    except Exception as e:
                        logger.error(f"Failed to load audio file {audio_filepath}: {e}")
                        continue
                else:
                    logger.error("Missing waveform/sample_rate and no valid audio_filepath provided")
                    continue
            
            try:
                # Get VAD segments
                segments = self._get_vad_segments(waveform, sample_rate)
                
                if not segments:
                    logger.warning("No speech segments detected by VAD")
                    continue
                
                # Convert waveform to PyDub AudioSegment for easier manipulation
                audio_segment = self._tensor_to_pydub(waveform, sample_rate)
                
                # Create output tasks for each segment
                for i, segment in enumerate(segments):
                    start_ms = int(segment['start'] * 1000)
                    end_ms = int(segment['end'] * 1000)
                    
                    # Extract segment audio (PyDub)
                    segment_audio = audio_segment[start_ms:end_ms]
                    
                    # Extract segment waveform (torch tensor) for downstream stages
                    start_sample = int(segment['start'] * sample_rate)
                    end_sample = int(segment['end'] * sample_rate)
                    
                    # Handle waveform dimensions
                    if waveform.dim() == 1:
                        segment_waveform = waveform[start_sample:end_sample].unsqueeze(0)
                    else:
                        segment_waveform = waveform[:, start_sample:end_sample]
                    
                    # Create new data dict for this segment
                    segment_data = {
                        'audio': segment_audio,
                        'waveform': segment_waveform,  # Segment waveform for downstream
                        'sample_rate': sample_rate,
                        'start_ms': start_ms,
                        'end_ms': end_ms,
                        'segment_num': i,
                        'duration_sec': (end_ms - start_ms) / 1000.0,
                        'original_file': item.get('audio_filepath', 'unknown'),
                    }
                    
                    # Copy any metadata from original item
                    for key in ['audio_filepath', 'text', 'dataset_name']:
                        if key in item:
                            segment_data[key] = item[key]
                    
                    # Create new AudioBatch for this segment
                    output_tasks.append(AudioBatch(
                        data=segment_data,
                        task_id=f"{task.task_id}_seg_{i}",
                        dataset_name=task.dataset_name,
                    ))
                
                # Log with more context
                total_duration = sum((s['end'] - s['start']) for s in segments)
                original_file = item.get('audio_filepath', 'unknown')
                file_name = os.path.basename(original_file) if original_file != 'unknown' else task.task_id
                logger.info(f"[VADSegmentation] {file_name}: {len(segments)} segments extracted ({total_duration:.1f}s total speech)")
                
            except Exception as e:
                logger.error(f"Error during VAD segmentation: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        return output_tasks
    
    def _get_vad_segments(self, waveform: torch.Tensor, sample_rate: int) -> List[Dict[str, float]]:
        """Get speech segments using VAD."""
        # Ensure waveform is 1D for VAD
        if waveform.dim() > 1:
            waveform = waveform.squeeze()
        
        # Move waveform to the same device as the model
        if self._device is not None and waveform.device != self._device:
            waveform = waveform.to(self._device)
        
        # Resample if sample rate is not supported by Silero VAD
        # Silero only supports 8kHz, 16kHz, and multiples of 16kHz (32k, 48k, etc.)
        # Sample rates like 22050 Hz need to be resampled to 16kHz
        vad_sample_rate = sample_rate
        vad_waveform = waveform
        if sample_rate not in SILERO_SUPPORTED_RATES and sample_rate % 16000 != 0:
            logger.debug(f"Resampling audio from {sample_rate}Hz to {SILERO_TARGET_RATE}Hz for VAD")
            # Move to CPU for resampling if needed
            device = waveform.device
            waveform_cpu = waveform.cpu() if waveform.device.type != 'cpu' else waveform
            # Ensure 2D for torchaudio resampling (channels, samples)
            if waveform_cpu.dim() == 1:
                waveform_cpu = waveform_cpu.unsqueeze(0)
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=SILERO_TARGET_RATE)
            vad_waveform = resampler(waveform_cpu).squeeze()
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
    
    def _tensor_to_pydub(self, waveform: torch.Tensor, sample_rate: int) -> AudioSegment:
        """Convert PyTorch tensor to PyDub AudioSegment."""
        # Ensure waveform is on CPU and convert to numpy
        waveform = waveform.cpu()
        if waveform.dim() > 1:
            waveform = waveform.squeeze(0)
        
        # Convert to int16 for PyDub
        waveform_int16 = (waveform * 32767).numpy().astype(np.int16)
        
        # Create AudioSegment
        audio = AudioSegment(
            waveform_int16.tobytes(),
            frame_rate=sample_rate,
            sample_width=2,  # 16-bit
            channels=1  # Mono
        )
        
        return audio
