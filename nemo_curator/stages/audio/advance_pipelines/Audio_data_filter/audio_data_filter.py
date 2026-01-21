# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0

"""
Audio Data Filter Stage - Complete audio curation pipeline in a single stage.

This stage combines:
1. Mono Conversion
2. VAD Segmentation
3. Quality Filters (NISQA, SIGMOS, Band)
4. Concatenation with timestamp tracking
5. Speaker Separation
6. Per-speaker VAD and filtering
7. Timestamp mapping back to original file

Usage:
    from advance_pipelines import (
        AudioDataFilterStage,
        AudioDataFilterConfig,
    )
    
    config = AudioDataFilterConfig(
        enable_nisqa=True,
        enable_speaker_separation=True,
    )
    stage = AudioDataFilterStage(config=config)
    results = stage.process(audio_batch)
"""

import os
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from loguru import logger
from pydub import AudioSegment

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioBatch

from nemo_curator.stages.audio import (
    MonoConversionStage,
    VADSegmentationStage,
    NISQAFilterStage,
    SIGMOSFilterStage,
    BandFilterStage,
    SpeakerSeparationStage,
)
from nemo_curator.stages.audio.configs import (
    VADConfig,
    NISQAConfig,
    SIGMOSConfig,
    BandFilterConfig,
    SpeakerSeparationConfig,
)

from .config import AudioDataFilterConfig


# =============================================================================
# Timestamp Tracking
# =============================================================================

@dataclass
class SegmentMapping:
    """Mapping from concatenated position to original file position."""
    original_file: str
    original_start_ms: int
    original_end_ms: int
    concat_start_ms: int
    concat_end_ms: int
    segment_index: int


class TimestampTracker:
    """Tracks segment timestamps through pipeline transformations."""
    
    def __init__(self):
        self.mappings: List[SegmentMapping] = []
        self.total_duration_ms: int = 0
    
    def build_from_segments(
        self, 
        segments: List[AudioBatch],
        silence_duration_ms: int = 500
    ) -> Tuple[AudioSegment, torch.Tensor, int]:
        """
        Build concatenated audio from segments and track mappings.
        
        Returns: (combined_audio, combined_waveform, sample_rate)
        """
        self.mappings = []
        combined = AudioSegment.empty()
        silence = AudioSegment.silent(duration=silence_duration_ms)
        current_pos_ms = 0
        sample_rate = 48000
        
        for i, task in enumerate(segments):
            for item in task.data:
                audio = item.get('audio')
                if not isinstance(audio, AudioSegment):
                    continue
                
                original_start = item.get('start_ms', 0)
                original_end = item.get('end_ms', 0)
                original_file = item.get('original_file', item.get('audio_filepath', 'unknown'))
                sample_rate = item.get('sample_rate', audio.frame_rate)
                
                segment_duration = len(audio)
                
                mapping = SegmentMapping(
                    original_file=original_file,
                    original_start_ms=original_start,
                    original_end_ms=original_end,
                    concat_start_ms=current_pos_ms,
                    concat_end_ms=current_pos_ms + segment_duration,
                    segment_index=i,
                )
                self.mappings.append(mapping)
                
                combined += audio
                current_pos_ms += segment_duration
                
                combined += silence
                current_pos_ms += silence_duration_ms
        
        # Remove trailing silence
        if self.mappings and len(combined) > 0:
            combined = combined[:-silence_duration_ms]
            current_pos_ms -= silence_duration_ms
        
        self.total_duration_ms = current_pos_ms
        waveform = self._to_tensor(combined)
        
        return combined, waveform, sample_rate
    
    def translate_to_original(
        self, 
        concat_start_ms: int, 
        concat_end_ms: int
    ) -> List[Dict[str, Any]]:
        """Translate concatenated range to original file positions."""
        results = []
        
        for m in self.mappings:
            if m.concat_end_ms <= concat_start_ms or m.concat_start_ms >= concat_end_ms:
                continue
            
            overlap_start = max(concat_start_ms, m.concat_start_ms)
            overlap_end = min(concat_end_ms, m.concat_end_ms)
            
            start_offset = overlap_start - m.concat_start_ms
            end_offset = overlap_end - m.concat_start_ms
            
            results.append({
                'original_file': m.original_file,
                'original_start_ms': m.original_start_ms + start_offset,
                'original_end_ms': m.original_start_ms + end_offset,
                'duration_ms': end_offset - start_offset,
            })
        
        return results
    
    def _to_tensor(self, audio: AudioSegment) -> torch.Tensor:
        """Convert AudioSegment to torch tensor."""
        samples = np.array(audio.get_array_of_samples())
        if audio.channels == 2:
            samples = samples.reshape((-1, 2)).mean(axis=1)
        samples = samples.astype(np.float32) / 32768.0
        return torch.from_numpy(samples).unsqueeze(0)


# =============================================================================
# Audio Data Filter Stage
# =============================================================================

@dataclass
class AudioDataFilterStage(ProcessingStage[AudioBatch, AudioBatch]):
    """
    Complete audio data filtering and curation pipeline.
    
    This stage performs the following operations:
    1. Mono Conversion - Convert multi-channel to mono
    2. VAD Segmentation - Extract speech segments
    3. Quality Filters - Apply NISQA, SIGMOS, Band filters
    4. Concatenation - Combine quality segments with timestamp tracking
    5. Speaker Separation - Separate by speaker (optional)
    6. Per-Speaker Processing - VAD + quality filters per speaker
    7. Timestamp Mapping - Map final segments back to original file
    
    Args:
        config: AudioDataFilterConfig with all pipeline settings
        
    Returns:
        AudioBatch with filtered segments, each containing:
        - original_file: Path to source audio
        - original_start_ms: Start position in original file
        - original_end_ms: End position in original file
        - duration_ms: Segment duration
        - speaker_id: Speaker identifier (if speaker separation enabled)
        - Quality scores (nisqa_*, sigmos_*, band_*)
    
    Example:
        config = AudioDataFilterConfig(
            enable_nisqa=True,
            enable_speaker_separation=True,
            nisqa_mos_threshold=4.5,
        )
        stage = AudioDataFilterStage(config=config)
        
        # Process audio file
        input_batch = AudioBatch(data={"audio_filepath": "/path/to/audio.wav"})
        results = stage.process(input_batch)
    """
    
    config: AudioDataFilterConfig = field(default_factory=AudioDataFilterConfig)
    
    name: str = "AudioDataFilter"
    batch_size: int = 1
    resources: Resources = field(default_factory=lambda: Resources(gpus=1.0))
    
    def __post_init__(self):
        super().__init__()
        self._mono_stage = None
        self._vad_stage = None
        self._band_stage = None
        self._nisqa_stage = None
        self._sigmos_stage = None
        self._speaker_stage = None
        self._init_lock = None
        self._initialized = False
        
        # Apply resources from config
        self.resources = Resources(cpus=self.config.cpus, gpus=self.config.gpus)
    
    def __getstate__(self):
        state = self.__dict__.copy()
        state['_init_lock'] = None
        state['_mono_stage'] = None
        state['_vad_stage'] = None
        state['_band_stage'] = None
        state['_nisqa_stage'] = None
        state['_sigmos_stage'] = None
        state['_speaker_stage'] = None
        state['_initialized'] = False
        return state
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        self._init_lock = None
        self._initialized = False
    
    def _get_lock(self):
        if self._init_lock is None:
            self._init_lock = threading.Lock()
        return self._init_lock
    
    def inputs(self) -> Tuple[List[str], List[str]]:
        # Don't validate data attributes since AudioBatch wraps dict in list
        # Validation is handled in process() method
        return [], []
    
    def outputs(self) -> Tuple[List[str], List[str]]:
        return [], ["original_file", "original_start_ms", "original_end_ms", 
                    "duration_ms", "duration_sec", "speaker_id"]
    
    def setup(self, worker_metadata=None) -> None:
        self._initialize_stages()
    
    def teardown(self) -> None:
        if self._nisqa_stage:
            self._nisqa_stage.teardown()
        if self._sigmos_stage:
            self._sigmos_stage.teardown()
        if self._band_stage:
            self._band_stage.teardown()
        if self._speaker_stage:
            self._speaker_stage.teardown()
        if self._vad_stage:
            self._vad_stage.teardown()
        torch.cuda.empty_cache()
    
    def _initialize_stages(self):
        """Initialize all sub-stages based on config.
        
        Resources from the parent AudioDataFilterStage are propagated to all
        GPU-enabled sub-stages. This ensures that when you call:
            stage.with_(resources=Resources(gpus=1.0))
        all sub-stages will use the same GPU allocation.
        """
        if self._initialized:
            return
        
        with self._get_lock():
            if self._initialized:
                return
            
            cfg = self.config
            
            # Log resource allocation
            logger.info(f"Initializing sub-stages with resources: "
                       f"gpus={self._resources.gpus}, cpus={self._resources.cpus}")
            
            # Mono conversion (CPU only, no GPU needed)
            self._mono_stage = MonoConversionStage(
                output_sample_rate=cfg.sample_rate,
                strict_sample_rate=cfg.strict_sample_rate,
            )
            
            # VAD - propagate GPU resources (only if enabled)
            if cfg.enable_vad:
                vad_config = VADConfig(
                    min_duration_sec=cfg.vad_min_duration_sec,
                    max_duration_sec=cfg.vad_max_duration_sec,
                    min_interval_ms=cfg.vad_min_interval_ms,
                    threshold=cfg.vad_threshold,
                    speech_pad_ms=cfg.vad_speech_pad_ms,
                )
                self._vad_stage = VADSegmentationStage(config=vad_config).with_(
                    resources=self._resources
                )
            
            # Band filter - propagate GPU resources
            if cfg.enable_band_filter:
                band_config = BandFilterConfig(band_value=cfg.band_value)
                self._band_stage = BandFilterStage(config=band_config).with_(
                    resources=self._resources
                )
            
            # NISQA filter - propagate GPU resources
            if cfg.enable_nisqa:
                nisqa_config = NISQAConfig(
                    mos_threshold=cfg.nisqa_mos_threshold,
                    noi_threshold=cfg.nisqa_noi_threshold,
                    col_threshold=cfg.nisqa_col_threshold,
                    dis_threshold=cfg.nisqa_dis_threshold,
                    loud_threshold=cfg.nisqa_loud_threshold,
                )
                self._nisqa_stage = NISQAFilterStage(config=nisqa_config).with_(
                    resources=self._resources
                )
            
            # SIGMOS filter - propagate GPU resources
            if cfg.enable_sigmos:
                sigmos_config = SIGMOSConfig(
                    noise_threshold=cfg.sigmos_noise_threshold,
                    ovrl_threshold=cfg.sigmos_ovrl_threshold,
                    sig_threshold=cfg.sigmos_sig_threshold,
                    col_threshold=cfg.sigmos_col_threshold,
                    disc_threshold=cfg.sigmos_disc_threshold,
                    loud_threshold=cfg.sigmos_loud_threshold,
                    reverb_threshold=cfg.sigmos_reverb_threshold,
                )
                self._sigmos_stage = SIGMOSFilterStage(config=sigmos_config).with_(
                    resources=self._resources
                )
            
            # Speaker separation - propagate GPU resources
            if cfg.enable_speaker_separation:
                speaker_config = SpeakerSeparationConfig(
                    exclude_overlaps=cfg.speaker_exclude_overlaps,
                    min_duration=cfg.speaker_min_duration,
                    gap_threshold=cfg.speaker_gap_threshold,
                    buffer_time=cfg.speaker_buffer_time,
                )
                self._speaker_stage = SpeakerSeparationStage(config=speaker_config).with_(
                    resources=self._resources
                )
            
            self._initialized = True
            logger.info(f"AudioDataFilterStage initialized with filters: {cfg.get_enabled_filters()}")
            if cfg.enable_speaker_separation:
                logger.info("Speaker separation: enabled")
    
    def _run_stage(self, stage, tasks: List[AudioBatch], name: str) -> List[AudioBatch]:
        """Run a stage on tasks, handling None results.
        
        Note: Parallel processing is handled internally by each stage based on
        the resources allocated via with_(resources=Resources(...)). Each stage
        automatically uses CPU parallelism or GPU batching when processing
        multiple items in task.data based on its resource allocation.
        """
        if stage is None:
            return tasks
        
        results = []
        for task in tasks:
            if task is None:
                continue
            try:
                result = stage.process(task)
                if result is None:
                    continue
                if isinstance(result, list):
                    results.extend([r for r in result if r is not None])
                else:
                    results.append(result)
            except Exception as e:
                logger.error(f"Error in {name}: {e}")
                import traceback
                traceback.print_exc()
        
        logger.debug(f"[{name}] {len(tasks)} -> {len(results)}")
        return results
    
    def _convert_mono_to_audio_segments(
        self, 
        mono_results: List[AudioBatch], 
        original_file: str
    ) -> List[AudioBatch]:
        """
        Convert mono conversion output to format expected by quality filters.
        
        When VAD is disabled, we need to convert the waveform tensor to 
        PyDub AudioSegment format that NISQA/SIGMOS/Band filters expect.
        """
        converted = []
        
        for task in mono_results:
            for item in task.data:
                waveform = item.get('waveform')
                sample_rate = item.get('sample_rate', 48000)
                
                if waveform is None:
                    logger.warning("No waveform in mono output")
                    continue
                
                try:
                    # Convert torch tensor to numpy
                    if hasattr(waveform, 'numpy'):
                        # Handle different tensor shapes
                        if waveform.dim() == 1:
                            audio_np = waveform.numpy()
                        else:
                            audio_np = waveform.squeeze().numpy()
                    else:
                        audio_np = np.array(waveform).squeeze()
                    
                    # Convert to int16 for PyDub
                    audio_int16 = (audio_np * 32767).astype(np.int16)
                    
                    # Create PyDub AudioSegment
                    audio_segment = AudioSegment(
                        audio_int16.tobytes(),
                        frame_rate=sample_rate,
                        sample_width=2,  # 16-bit
                        channels=1  # mono
                    )
                    
                    # Calculate duration
                    duration_ms = len(audio_segment)
                    duration_sec = duration_ms / 1000.0
                    
                    # Create output data with all required fields
                    output_data = {
                        'audio': audio_segment,
                        'waveform': waveform,
                        'sample_rate': sample_rate,
                        'start_ms': 0,
                        'end_ms': duration_ms,
                        'duration_sec': duration_sec,
                        'original_file': original_file,
                        'audio_filepath': item.get('audio_filepath', original_file),
                    }
                    
                    converted.append(AudioBatch(
                        data=[output_data],
                        task_id=task.task_id,
                        dataset_name=task.dataset_name
                    ))
                    
                except Exception as e:
                    logger.error(f"Error converting mono to audio segment: {e}")
                    import traceback
                    traceback.print_exc()
        
        return converted
    
    def process(self, task: AudioBatch) -> Optional[AudioBatch]:
        """
        Process audio through the complete filtering pipeline.
        
        Args:
            task: AudioBatch with audio_filepath
            
        Returns:
            AudioBatch with filtered segments and original timestamps
        """
        self._initialize_stages()
        
        # Handle both dict and list data formats
        if isinstance(task.data, list) and len(task.data) > 0:
            original_file = task.data[0].get('audio_filepath', 'unknown')
        elif isinstance(task.data, dict):
            original_file = task.data.get('audio_filepath', 'unknown')
        else:
            logger.error("Invalid task data format")
            return None
        
        file_name = os.path.basename(original_file)
        
        logger.info(f"Processing: {file_name}")
        
        # Stage 1: Mono conversion
        mono_results = self._run_stage(self._mono_stage, [task], "MonoConversion")
        if not mono_results:
            logger.warning(f"Mono conversion failed for {file_name}")
            return None
        
        # Stage 2: VAD segmentation (only if enabled)
        if self._vad_stage:
            vad_results = self._run_stage(self._vad_stage, mono_results, "VAD")
            if not vad_results:
                logger.warning(f"No VAD segments for {file_name}")
                return None
            logger.info(f"VAD segments: {len(vad_results)}")
            quality_segments = vad_results
        else:
            # Skip VAD - use mono results directly
            # Need to convert waveform to AudioSegment for downstream filters
            logger.info(f"VAD disabled - using full audio")
            quality_segments = self._convert_mono_to_audio_segments(mono_results, original_file)
        
        # Stage 3: Quality filters
        
        if self._band_stage:
            quality_segments = self._run_stage(self._band_stage, quality_segments, "BandFilter")
            logger.info(f"After BandFilter: {len(quality_segments)} segments")
        
        if self._nisqa_stage:
            quality_segments = self._run_stage(self._nisqa_stage, quality_segments, "NISQA")
            logger.info(f"After NISQA: {len(quality_segments)} segments")
        
        if self._sigmos_stage:
            quality_segments = self._run_stage(self._sigmos_stage, quality_segments, "SIGMOS")
            logger.info(f"After SIGMOS: {len(quality_segments)} segments")
        
        if not quality_segments:
            logger.warning(f"No segments passed quality filters for {file_name}")
            return None
        
        # If no speaker separation, return quality segments directly
        if not self._speaker_stage:
            return self._build_output(quality_segments, original_file, task)
        
        # Stage 4: Concatenate with timestamp tracking
        logger.info("Concatenating quality segments...")
        tracker = TimestampTracker()
        combined_audio, combined_waveform, sample_rate = tracker.build_from_segments(
            quality_segments, 
            self.config.silence_duration_ms
        )
        
        logger.info(f"Concatenated: {len(combined_audio)/1000:.2f}s, {len(tracker.mappings)} segments")
        
        concat_task = AudioBatch(
            data=[{
                'audio': combined_audio,
                'waveform': combined_waveform,
                'sample_rate': sample_rate,
                'original_file': original_file,
            }],
            task_id=f"{task.task_id}_concat",
            dataset_name=task.dataset_name
        )
        
        # Stage 5: Speaker separation
        speaker_results = self._run_stage(self._speaker_stage, [concat_task], "SpeakerSep")
        if not speaker_results:
            logger.warning(f"Speaker separation failed for {file_name}")
            return None
        
        num_speakers = len(speaker_results)
        logger.info(f"Speakers detected: {num_speakers}")
        
        # Stage 6 & 7: VAD + Quality filters per speaker
        final_results = []
        
        for speaker_task in speaker_results:
            if isinstance(speaker_task.data, list) and len(speaker_task.data) > 0:
                speaker_id = speaker_task.data[0].get('speaker_id', 'unknown')
            else:
                speaker_id = 'unknown'
            
            logger.info(f"  Processing speaker: {speaker_id}")
            
            # Ensure waveform exists for VAD
            for item in speaker_task.data:
                if 'waveform' not in item and 'audio' in item:
                    audio = item['audio']
                    if isinstance(audio, AudioSegment):
                        samples = np.array(audio.get_array_of_samples())
                        if audio.channels == 2:
                            samples = samples.reshape((-1, 2)).mean(axis=1)
                        samples = samples.astype(np.float32) / 32768.0
                        item['waveform'] = torch.from_numpy(samples).unsqueeze(0)
                        item['sample_rate'] = audio.frame_rate
            
            # VAD on speaker audio
            speaker_vad = self._run_stage(self._vad_stage, [speaker_task], f"VAD_{speaker_id}")
            if not speaker_vad:
                logger.info(f"    No VAD segments for {speaker_id}")
                continue
            
            # Quality filters
            speaker_quality = speaker_vad
            
            if self._band_stage:
                speaker_quality = self._run_stage(self._band_stage, speaker_quality, f"Band_{speaker_id}")
            if self._nisqa_stage:
                speaker_quality = self._run_stage(self._nisqa_stage, speaker_quality, f"NISQA_{speaker_id}")
            if self._sigmos_stage:
                speaker_quality = self._run_stage(self._sigmos_stage, speaker_quality, f"SIGMOS_{speaker_id}")
            
            if not speaker_quality:
                logger.info(f"    No segments passed filters for {speaker_id}")
                continue
            
            logger.info(f"    {speaker_id}: {len(speaker_quality)} segments passed")
            
            # Map back to original timestamps
            for sq_task in speaker_quality:
                for item in sq_task.data:
                    concat_start = item.get('start_ms', 0)
                    concat_end = item.get('end_ms', 0)
                    
                    original_ranges = tracker.translate_to_original(concat_start, concat_end)
                    
                    for orig in original_ranges:
                        result = {
                            'original_file': orig['original_file'],
                            'original_start_ms': orig['original_start_ms'],
                            'original_end_ms': orig['original_end_ms'],
                            'duration_ms': orig['duration_ms'],
                            'duration_sec': orig['duration_ms'] / 1000.0,
                            'speaker_id': speaker_id,
                            'num_speakers': num_speakers,
                        }
                        
                        # Add quality scores only for enabled filters (non-null values only)
                        if self.config.enable_nisqa:
                            for key in ['nisqa_mos', 'nisqa_noi', 'nisqa_col', 'nisqa_dis', 'nisqa_loud']:
                                if key in item and item[key] is not None:
                                    result[key] = item[key]
                        
                        if self.config.enable_sigmos:
                            for key in ['sigmos_noise', 'sigmos_ovrl', 'sigmos_sig', 'sigmos_col',
                                       'sigmos_disc', 'sigmos_loud', 'sigmos_reverb']:
                                if key in item and item[key] is not None:
                                    result[key] = item[key]
                        
                        if self.config.enable_band_filter:
                            if 'band_prediction' in item and item['band_prediction'] is not None:
                                result['band_prediction'] = item['band_prediction']
                        
                        final_results.append(result)
        
        if not final_results:
            logger.warning(f"No final results for {file_name}")
            return None
        
        logger.info(f"Total output segments: {len(final_results)}")
        
        return AudioBatch(
            data=final_results,
            task_id=task.task_id,
            dataset_name=task.dataset_name
        )
    
    def _build_output(
        self, 
        segments: List[AudioBatch], 
        original_file: str,
        original_task: AudioBatch
    ) -> AudioBatch:
        """Build output AudioBatch from filtered segments (no speaker separation)."""
        results = []
        
        for task in segments:
            for item in task.data:
                # Get timing info - calculate from audio if not provided (e.g., when VAD is disabled)
                start_ms = item.get('start_ms', 0)
                end_ms = item.get('end_ms', 0)
                duration_sec = item.get('duration_sec', 0)
                
                # If end_ms is 0, try to calculate from audio data
                if end_ms == 0:
                    audio = item.get('audio')
                    waveform = item.get('waveform')
                    sample_rate = item.get('sample_rate', 48000)
                    
                    if audio is not None and hasattr(audio, '__len__'):
                        # PyDub AudioSegment - length in ms
                        end_ms = len(audio)
                        duration_sec = end_ms / 1000.0
                    elif waveform is not None and hasattr(waveform, 'shape'):
                        # Torch tensor - calculate from shape and sample rate
                        if len(waveform.shape) == 1:
                            num_samples = waveform.shape[0]
                        else:
                            num_samples = waveform.shape[-1]
                        duration_sec = num_samples / sample_rate
                        end_ms = int(duration_sec * 1000)
                
                duration_ms = end_ms - start_ms
                if duration_sec == 0 and duration_ms > 0:
                    duration_sec = duration_ms / 1000.0
                
                result = {
                    'original_file': item.get('original_file', original_file),
                    'original_start_ms': start_ms,
                    'original_end_ms': end_ms,
                    'duration_ms': duration_ms,
                    'duration_sec': duration_sec,
                }
                
                # Add quality scores only for enabled filters (non-null values only)
                if self.config.enable_nisqa:
                    for key in ['nisqa_mos', 'nisqa_noi', 'nisqa_col', 'nisqa_dis', 'nisqa_loud']:
                        if key in item and item[key] is not None:
                            result[key] = item[key]
                
                if self.config.enable_sigmos:
                    for key in ['sigmos_noise', 'sigmos_ovrl', 'sigmos_sig', 'sigmos_col',
                               'sigmos_disc', 'sigmos_loud', 'sigmos_reverb']:
                        if key in item and item[key] is not None:
                            result[key] = item[key]
                
                if self.config.enable_band_filter:
                    if 'band_prediction' in item and item['band_prediction'] is not None:
                        result['band_prediction'] = item['band_prediction']
                
                results.append(result)
        
        logger.info(f"Output segments (no speaker sep): {len(results)}")
        
        return AudioBatch(
            data=results,
            task_id=original_task.task_id,
            dataset_name=original_task.dataset_name
        )

