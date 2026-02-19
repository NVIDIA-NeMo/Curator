import os
import tempfile
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

# NeMo imports
# We assume the environment is set up correctly for NeMo
try:
    from nemo.collections.asr.models import SortformerEncLabelModel
except ImportError:
    logger.warning("NeMo not found. Speaker separation will not work.")
    SortformerEncLabelModel = None

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
                 
            self.model_name = val or "model/diar_sortformer_4spk-v1.nemo"

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
        
        if SortformerEncLabelModel is not None:
            self._load_model()
        
    def _load_model(self):
        """Load the diarization model."""
        try:
            logger.info(f"Loading speaker separation model from: {self.model_name}")
            self.diar_model = SortformerEncLabelModel.restore_from(
                restore_path=self.model_name,
                map_location=self.device,
                strict=False
            )
            self.diar_model.eval()
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
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
                    # Only end segments if this speaker was active
                    if current_segments[speaker] is not None:
                        start_time = current_segments[speaker]
                        if start_time < time:  # Only add if segment has length
                            result_segments[speaker].append((start_time, time))
                        current_segments[speaker] = None
                    active_speakers.remove(speaker)

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
            # Input is a file path
            waveform, sample_rate = load_audio(audio_path_or_waveform)
            if waveform.shape[0] == 2:
                waveform = waveform.mean(dim=0, keepdim=True)
        else:
            # Input is a waveform tensor
            if sample_rate is None:
                raise ValueError("Sample rate must be provided when passing a waveform")
            waveform = audio_path_or_waveform
            if waveform.shape[0] == 2:
                waveform = waveform.mean(dim=0, keepdim=True)
        
        try:
            # Create temporary file for diarization
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
                # Use soundfile instead of torchaudio
                wav = waveform.squeeze(0) if waveform.dim() > 1 else waveform
                sf.write(temp_audio_file.name, wav.cpu().numpy(), sample_rate)
                # Run diarization on temp mono file
                result = self.diar_model.diarize(audio=temp_audio_file.name, batch_size=1)
                os.unlink(temp_audio_file.name)
                return result
        except Exception as e:
            # If there's an error in diarization, log it and return a fallback result with one speaker
            logger.warning(f"Error during diarization: {e}. Falling back to single speaker mode")
            
            # Calculate duration in seconds
            duration_sec = waveform.shape[1] / sample_rate
            
            # Create a single segment covering the entire audio
            return [f"0.0 {duration_sec:.3f} speaker_0"]
    
    def get_speaker_segments(self, predicted_segments: List[str]) -> Dict[str, List[Tuple[float, float]]]:
        """
        Parse predicted segments and organize by speaker.
        """
        speaker_segments = {}
        
        # Handle the nested list structure from the model output
        segments = predicted_segments[0] if isinstance(predicted_segments, list) and isinstance(predicted_segments[0], list) else predicted_segments
        
        for segment in segments:
            parts = segment.split()
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
            
            # Make sure we have at least one speaker
            if not speaker_segments:
                logger.warning("No speakers detected. Creating default single speaker.")
                # Create a default speaker segment if none found
                if isinstance(audio_path_or_waveform, str):
                    # Input is a file path - use soundfile
                    waveform, sample_rate = load_audio(audio_path_or_waveform)
                    duration_sec = waveform.shape[1] / sample_rate
                else:
                    # Input is a waveform tensor
                    duration_sec = audio_path_or_waveform.shape[1] / sample_rate
                
                speaker_segments = {"speaker_0": [(0.0, duration_sec)]}
            
            # Process segments based on overlap handling preference
            if exclude_overlaps:
                # Completely exclude overlapping segments with buffer
                processed_segments = self.exclude_overlapping_segments(speaker_segments, buffer_time)
                logger.debug(f"After excluding overlaps with {buffer_time}s buffer: {sum(len(segs) for segs in processed_segments.values())} segments remaining")
            else:
                # Clean cut overlapping segments (divide between speakers)
                processed_segments = self.clean_cut_overlapping_segments(speaker_segments)
            
            # Check if we still have segments after processing
            if all(len(segments) == 0 for segments in processed_segments.values()):
                logger.warning("All segments were removed during processing. Creating a new single segment.")
                # Create a default speaker segment if all were removed
                if isinstance(audio_path_or_waveform, str):
                    # Input is a file path - use soundfile
                    waveform, sample_rate = load_audio(audio_path_or_waveform)
                    duration_sec = waveform.shape[1] / sample_rate
                else:
                    # Input is a waveform tensor
                    duration_sec = audio_path_or_waveform.shape[1] / sample_rate
                
                processed_segments = {"speaker_0": [(0.0, duration_sec)]}
            
            # Merge adjacent segments with small gaps
            for speaker in processed_segments:
                processed_segments[speaker] = self.merge_adjacent_segments(
                    processed_segments[speaker], gap_threshold
                )
            
            # Filter out segments shorter than minimum duration if specified
            if min_duration > 0:
                processed_segments = self.filter_short_segments(processed_segments, min_duration)
                
                # Check again if we have any segments left
                if all(len(segments) == 0 for segments in processed_segments.values()):
                    logger.warning("All segments were removed after duration filtering. Creating a new single segment.")
                    # Create a default speaker segment with a slightly shorter duration to pass the filter
                    if isinstance(audio_path_or_waveform, str):
                        # Input is a file path - use soundfile
                        waveform, sample_rate = load_audio(audio_path_or_waveform)
                        duration_sec = waveform.shape[1] / sample_rate
                    else:
                        # Input is a waveform tensor
                        duration_sec = audio_path_or_waveform.shape[1] / sample_rate
                    
                    processed_segments = {"speaker_0": [(0.0, duration_sec)]}
            
            return processed_segments
            
        except Exception as e:
            logger.warning(f"Error during audio processing: {e}. Falling back to single speaker mode")
            
            # Create a fallback result with one speaker covering the entire audio
            if isinstance(audio_path_or_waveform, str):
                # Input is a file path
                waveform, sample_rate = load_audio(audio_path_or_waveform)
                duration_sec = waveform.shape[1] / sample_rate
            else:
                # Input is a waveform tensor
                duration_sec = audio_path_or_waveform.shape[1] / sample_rate
            
            return {"speaker_0": [(0.0, duration_sec)]}
    
    def get_speaker_audio_data(self, audio_path_or_waveform, sample_rate=None, gap_threshold=None, exclude_overlaps=None, min_duration=None, buffer_time=None):
        """
        Process an audio file or waveform and return AudioSegment objects for each speaker.
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
        
        # Check if input is a path or waveform
        if isinstance(audio_path_or_waveform, str):
            # Load the original audio file from path
            # Using from_file() to support multiple formats (wav, mp3, flac, ogg, etc.)
            original_audio = AudioSegment.from_file(audio_path_or_waveform)
            # Process the audio to get speaker segments
            speaker_segments = self.process_audio(
                audio_path_or_waveform, 
                None, 
                gap_threshold, 
                exclude_overlaps, 
                min_duration,
                buffer_time
            )
        else:
            # Input is a waveform tensor
            if sample_rate is None:
                raise ValueError("Sample rate must be provided when passing a waveform")
            
            # Create a temporary file to convert waveform to AudioSegment
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
                # Use soundfile instead of torchaudio to avoid FFmpeg dependency
                wav = audio_path_or_waveform.squeeze(0) if audio_path_or_waveform.dim() > 1 else audio_path_or_waveform
                sf.write(temp_audio_file.name, wav.cpu().numpy(), sample_rate)
                # Using from_file() for consistency, though temp file is always wav
                original_audio = AudioSegment.from_file(temp_audio_file.name)
                os.unlink(temp_audio_file.name)  # Clean up temp file
            
            # Process the audio to get speaker segments
            speaker_segments = self.process_audio(
                audio_path_or_waveform, 
                sample_rate, 
                gap_threshold, 
                exclude_overlaps, 
                min_duration,
                buffer_time
            )
            
        duration_ms = len(original_audio)
        
        # Create audio segments for each speaker
        speaker_audio = {}
        
        for speaker, segments in speaker_segments.items():
            # Skip if no segments for this speaker
            if not segments:
                continue
                
            # Calculate total duration of non-overlapping speech for this speaker
            total_duration = sum(end - start for start, end in segments)
            
            # Skip if total duration is too low (speaker is essentially silent)
            if total_duration < 0.1:  # less than 100ms of speech
                continue
            
            # Create a silent audio with the same duration as the original
            silent_audio = AudioSegment.silent(duration=duration_ms)
            
            # Add segments for this speaker
            for start_time, end_time in segments:
                # Convert times from seconds to milliseconds
                start_ms = int(start_time * 1000)
                end_ms = int(end_time * 1000)
                
                # Extract segment from original audio
                segment_audio = original_audio[start_ms:end_ms]
                
                # Overlay segment onto silent audio
                silent_audio = silent_audio.overlay(segment_audio, position=start_ms)
            
            # Check if audio is silent (RMS amplitude close to 0)
            if silent_audio.rms < 1:  # Very low RMS indicates silence
                continue
                
            # Store the speaker's audio and duration
            speaker_audio[speaker] = (silent_audio, total_duration)
        
        return speaker_audio
