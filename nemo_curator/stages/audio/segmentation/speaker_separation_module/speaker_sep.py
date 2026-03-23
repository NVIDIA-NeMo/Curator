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

import os
import tempfile
import time
from typing import List, Tuple, Dict, Optional
import torch
import soundfile as sf
from loguru import logger
from pydub import AudioSegment


def load_audio(audio_path: str) -> Tuple[torch.Tensor, int]:
    """
    Load audio file using soundfile.
    
    Uses soundfile directly to avoid torchaudio/torchcodec/FFmpeg dependency issues.
    
    Args:
        audio_path: Path to the audio file
        
    Returns:
        Tuple of (waveform tensor, sample_rate)
    """
    data, sample_rate = sf.read(audio_path, dtype='float32')
    # Convert to torch tensor with shape (channels, samples)
    waveform = torch.from_numpy(data)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)  # Add channel dimension
    else:
        waveform = waveform.T  # soundfile returns (samples, channels), we need (channels, samples)
    return waveform, sample_rate

from nemo.collections.asr.models import SortformerEncLabelModel

class SpeakerSeparator:
    """
    Class for separating speakers in an audio file using diarization.
    """
    
    def __init__(self, model_name=None, config=None):
        """
        Initialize the speaker separator.
        
        Args:
            model_name: The name of the pretrained model to use
            config: Configuration object (dict or class with .get method)
        """
        self.config = config or {}
        
        # Get model name
        if model_name:
            self.model_name = model_name
        else:
            # Try to find in config
            val = None
            if hasattr(self.config, 'speaker_model_path'):
                val = getattr(self.config, 'speaker_model_path')
            elif isinstance(self.config, dict):
                val = self.config.get('speaker_model_path')
                 
            self.model_name = val or "nvidia/diar_sortformer_4spk-v1"

        # Check for GPU usage
        self.use_gpu = False
        if hasattr(self.config, 'use_gpu'):
             self.use_gpu = getattr(self.config, 'use_gpu')
        elif isinstance(self.config, dict):
             self.use_gpu = self.config.get('use_gpu', False)
             
        self.device = "cpu"
        if self.use_gpu:
            self.device = "cuda"
            
        self.diar_model = None
        self._load_model()
        
    def _load_model(self, max_retries: int = 3, retry_delay: float = 5.0):
        """Load the diarization model from HuggingFace Hub with retry logic."""
        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"Loading speaker separation model from HuggingFace: {self.model_name} (attempt {attempt}/{max_retries})")
                self.diar_model = SortformerEncLabelModel.from_pretrained(
                    self.model_name,
                    map_location=self.device,
                )
                self.diar_model.eval()
                return
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    logger.warning(f"HuggingFace model load attempt {attempt} failed: {e}. Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
        logger.error(f"Failed to load model from HuggingFace after {max_retries} attempts: {last_error}")
        raise last_error
    
    def _get_param(self, param_name, default_value):
        """Helper to get a parameter from config, handling different config structures."""
        # Try direct attribute on config object
        if hasattr(self.config, param_name):
            val = getattr(self.config, param_name)
            if val is not None:
                return val
            
        # Try dictionary access
        if isinstance(self.config, dict):
             if param_name in self.config:
                 return self.config[param_name]
             # Try nested 'speaker_separation' key if it exists (legacy support)
             if 'speaker_separation' in self.config and isinstance(self.config['speaker_separation'], dict):
                 if param_name in self.config['speaker_separation']:
                     return self.config['speaker_separation'][param_name]
        
        # Try method access .get(key, default)
        if hasattr(self.config, 'get'):
            try:
                val = self.config.get(param_name)
                if val is not None:
                    return val
            except Exception:
                pass

        return default_value

    def clean_cut_overlapping_segments(self, speaker_segments):
        """
        Handle overlaps by cutting segments at overlap points.
        """
        # Flatten all segments into a timeline with speaker information
        timeline = []
        for speaker, segments in speaker_segments.items():
            for start, end in segments:
                timeline.append((start, 1, speaker))  # 1 indicates segment start
                timeline.append((end, -1, speaker))   # -1 indicates segment end

        # Sort the timeline by time
        timeline.sort(key=lambda x: (x[0], x[1]))

        # Process the timeline to find non-overlapping segments
        active_speakers = set()
        result_segments = {spk: [] for spk in speaker_segments.keys()}
        current_segments = {spk: None for spk in speaker_segments.keys()}

        for time, event_type, speaker in timeline:
            # Process any segment endings first
            if event_type == -1:
                if speaker in active_speakers:
                    if current_segments[speaker] is not None:
                        start_time = current_segments[speaker]
                        if start_time < time:
                            result_segments[speaker].append((start_time, time))
                        current_segments[speaker] = None
                    active_speakers.remove(speaker)
                    # Restart tracking for speakers still active after this overlap ends
                    for active_spk in active_speakers:
                        if current_segments[active_spk] is None:
                            current_segments[active_spk] = time

            # Then handle any new overlaps with existing active speakers
            elif event_type == 1:
                # If there are already active speakers, end their current segments
                for active_spk in active_speakers:
                    if current_segments[active_spk] is not None:
                        start_time = current_segments[active_spk]
                        if start_time < time:  # Only add if segment has length
                            result_segments[active_spk].append((start_time, time))
                        current_segments[active_spk] = None

                # Mark the new speaker as active
                active_speakers.add(speaker)
                current_segments[speaker] = time

        return result_segments

    def exclude_overlapping_segments(self, speaker_segments, buffer_time=None):
        """
        Completely exclude any segments where multiple speakers are talking simultaneously.
        """
        if not speaker_segments:
            return {}
        
        if buffer_time is None:
            buffer_time = self._get_param('speaker_buffer_time', 0.5)
            
        # Flatten all segments into a timeline with speaker information
        timeline = []
        for speaker, segments in speaker_segments.items():
            for start, end in segments:
                timeline.append((start, 1, speaker))   # 1 indicates segment start
                timeline.append((end, -1, speaker))    # -1 indicates segment end
        
        # Sort the timeline by time
        timeline.sort(key=lambda x: (x[0], x[1]))
        
        # Process the timeline to find periods of single-speaker speech
        active_speakers = set()
        result_segments = {spk: [] for spk in speaker_segments.keys()}
        single_speaker_start = None
        current_single_speaker = None
        
        # Process each event in the timeline
        for i, (time, event_type, speaker) in enumerate(timeline):
            # Speaker starts talking
            if event_type == 1:
                active_speakers.add(speaker)
                
                # If this is the only active speaker, mark the start of a single-speaker segment
                if len(active_speakers) == 1:
                    single_speaker_start = time
                    current_single_speaker = speaker
                # If we now have multiple speakers, end any existing single-speaker segment
                elif len(active_speakers) == 2 and single_speaker_start is not None:
                    # Add completed single-speaker segment to results with buffer
                    if current_single_speaker is not None and single_speaker_start < time:
                        # Apply buffer (end segment earlier for cleaner transition)
                        end_with_buffer = max(single_speaker_start, time - buffer_time)
                        if single_speaker_start < end_with_buffer:  # Only add if segment has positive length
                            result_segments[current_single_speaker].append((single_speaker_start, end_with_buffer))
                    single_speaker_start = None
                    current_single_speaker = None
            
            # Speaker stops talking
            elif event_type == -1:
                # If this was the only active speaker, end the single-speaker segment
                if len(active_speakers) == 1 and speaker in active_speakers:
                    if single_speaker_start is not None and single_speaker_start < time:
                        result_segments[speaker].append((single_speaker_start, time))
                    single_speaker_start = None
                    current_single_speaker = None
                
                # Remove the speaker from active set
                active_speakers.discard(speaker)
                
                # If we now have exactly one speaker, start a new single-speaker segment with buffer
                if len(active_speakers) == 1:
                    # Apply buffer (start segment later for cleaner transition)
                    start_with_buffer = time + buffer_time
                    single_speaker_start = start_with_buffer
                    current_single_speaker = next(iter(active_speakers))
        
        if all(len(segments) == 0 for segments in result_segments.values()):
            total_original = sum(len(segments) for segments in speaker_segments.values())
            logger.warning(f"All segments were excluded during overlap filtering (original count: {total_original})")
            
        return result_segments

    def filter_short_segments(self, speaker_segments, min_duration=None):
        """
        Filter out segments that are shorter than the minimum duration.
        """
        if min_duration is None:
            min_duration = self._get_param('speaker_min_duration', 2.0)
            
        result_segments = {spk: [] for spk in speaker_segments.keys()}
        
        for speaker, segments in speaker_segments.items():
            for start, end in segments:
                duration = end - start
                if duration >= min_duration:
                    result_segments[speaker].append((start, end))
                    
        return result_segments

    def merge_adjacent_segments(self, segments, gap_threshold=None):
        """
        Merge adjacent segments for the same speaker if they are close enough.
        """
        if gap_threshold is None:
            gap_threshold = self._get_param('speaker_gap_threshold', 0.1)
            
        if not segments:
            return []

        # Sort segments by start time
        sorted_segments = sorted(segments)

        merged = [sorted_segments[0]]
        for current in sorted_segments[1:]:
            previous = merged[-1]
            # If the gap between current and previous is small enough, merge them
            if current[0] - previous[1] <= gap_threshold:
                merged[-1] = (previous[0], max(previous[1], current[1]))
            else:
                merged.append(current)

        return merged
    
    def diarize_audio(self, audio_path_or_waveform, sample_rate=None):
        """
        Run speaker diarization on an audio file or waveform.
        """
        if not self.diar_model:
            self._load_model()
        
        # Check if input is a path or waveform
        if isinstance(audio_path_or_waveform, str):
            waveform, sample_rate = load_audio(audio_path_or_waveform)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
        else:
            if sample_rate is None:
                raise ValueError("Sample rate must be provided when passing a waveform")
            waveform = audio_path_or_waveform
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
        
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                temp_path = tmp.name
            wav = waveform.squeeze(0) if waveform.dim() > 1 else waveform
            sf.write(temp_path, wav.cpu().numpy(), sample_rate)
            with torch.no_grad():
                return self.diar_model.diarize(audio=temp_path, batch_size=1)
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def get_speaker_segments(self, predicted_segments: List[str]) -> Dict[str, List[Tuple[float, float]]]:
        """
        Parse predicted segments and organize by speaker.
        """
        speaker_segments = {}
        
        # Handle the nested list structure from the model output
        segments = (
            predicted_segments[0]
            if isinstance(predicted_segments, list) and predicted_segments and isinstance(predicted_segments[0], list)
            else predicted_segments
        )
        
        for segment in segments:
            parts = segment.split()
            if len(parts) < 3:
                logger.warning(f"Skipping malformed diarization segment: {segment!r}")
                continue
            start_time = float(parts[0])
            end_time = float(parts[1])
            speaker = parts[2]
            
            if speaker not in speaker_segments:
                speaker_segments[speaker] = []
                
            speaker_segments[speaker].append((start_time, end_time))
        
        return speaker_segments
    
    def process_audio(self, audio_path_or_waveform, sample_rate=None, gap_threshold=None, exclude_overlaps=None, min_duration=None, buffer_time=None):
        """
        Process an audio file or waveform to get speaker segments.
        """
        # Get parameters from config if not provided
        if gap_threshold is None:
            gap_threshold = self._get_param('speaker_gap_threshold', 0.1)
        if exclude_overlaps is None:
            exclude_overlaps = self._get_param('speaker_exclude_overlaps', False)
        if min_duration is None:
            min_duration = self._get_param('speaker_min_duration', 2.0)
        if buffer_time is None:
            buffer_time = self._get_param('speaker_buffer_time', 0.5)
        
        try:
            # Run diarization
            predicted_segments = self.diarize_audio(audio_path_or_waveform, sample_rate)
            
            # Parse segments by speaker
            speaker_segments = self.get_speaker_segments(predicted_segments)
            
            if not speaker_segments:
                logger.warning("No speakers detected, skipping item")
                return {}
            
            # Process segments based on overlap handling preference
            if exclude_overlaps:
                # Completely exclude overlapping segments with buffer
                processed_segments = self.exclude_overlapping_segments(speaker_segments, buffer_time)
                logger.debug(f"After excluding overlaps with {buffer_time}s buffer: {sum(len(segs) for segs in processed_segments.values())} segments remaining")
            else:
                # Clean cut overlapping segments (divide between speakers)
                processed_segments = self.clean_cut_overlapping_segments(speaker_segments)
            
            if all(len(segments) == 0 for segments in processed_segments.values()):
                logger.warning("All segments removed during overlap processing, skipping item")
                return {}
            
            # Merge adjacent segments with small gaps
            for speaker in processed_segments:
                processed_segments[speaker] = self.merge_adjacent_segments(
                    processed_segments[speaker], gap_threshold
                )
            
            # Filter out segments shorter than minimum duration if specified
            if min_duration > 0:
                processed_segments = self.filter_short_segments(processed_segments, min_duration)
                
                if all(len(segments) == 0 for segments in processed_segments.values()):
                    logger.warning("All segments removed after duration filtering, skipping item")
                    return {}
            
            return processed_segments
            
        except torch.cuda.OutOfMemoryError:
            logger.warning("CUDA out of memory during diarization, skipping item")
            torch.cuda.empty_cache()
            return {}
        except Exception as e:
            logger.error(f"Error processing audio for speaker segments: {e}")
            return {}
    
    def get_speaker_audio_data(self, audio_path_or_waveform, sample_rate=None, gap_threshold=None, exclude_overlaps=None, min_duration=None, buffer_time=None):
        """
        Process an audio file or waveform and return AudioSegment objects for each speaker.
        """
        if gap_threshold is None:
            gap_threshold = self._get_param('speaker_gap_threshold', 0.1)
        if exclude_overlaps is None:
            exclude_overlaps = self._get_param('speaker_exclude_overlaps', False)
        if min_duration is None:
            min_duration = self._get_param('speaker_min_duration', 2.0)
        if buffer_time is None:
            buffer_time = self._get_param('speaker_buffer_time', 0.5)
        
        if isinstance(audio_path_or_waveform, str):
            original_audio = AudioSegment.from_file(audio_path_or_waveform)
            speaker_segments = self.process_audio(
                audio_path_or_waveform, 
                None, 
                gap_threshold, 
                exclude_overlaps, 
                min_duration,
                buffer_time
            )
        else:
            if sample_rate is None:
                raise ValueError("Sample rate must be provided when passing a waveform")
            
            temp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    temp_path = tmp.name
                wav = audio_path_or_waveform.squeeze(0) if audio_path_or_waveform.dim() > 1 else audio_path_or_waveform
                sf.write(temp_path, wav.cpu().numpy(), sample_rate)
                original_audio = AudioSegment.from_file(temp_path)
            finally:
                if temp_path and os.path.exists(temp_path):
                    os.unlink(temp_path)
            
            speaker_segments = self.process_audio(
                audio_path_or_waveform, 
                sample_rate, 
                gap_threshold, 
                exclude_overlaps, 
                min_duration,
                buffer_time
            )
            
        duration_ms = len(original_audio)
        
        speaker_audio = {}
        
        for speaker, segments in speaker_segments.items():
            if not segments:
                continue
                
            total_duration = sum(end - start for start, end in segments)
            
            if total_duration < 0.1:
                continue
            
            silent_audio = AudioSegment.silent(duration=duration_ms)
            
            for start_time, end_time in segments:
                start_ms = max(0, min(int(start_time * 1000), duration_ms))
                end_ms = max(0, min(int(end_time * 1000), duration_ms))
                if start_ms >= end_ms:
                    continue
                
                segment_audio = original_audio[start_ms:end_ms]
                silent_audio = silent_audio.overlay(segment_audio, position=start_ms)
            
            if silent_audio.rms < 1:
                continue
                
            speaker_audio[speaker] = (silent_audio, total_duration)
        
        # Free the original audio to release memory before returning
        del original_audio
        
        return speaker_audio
