# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
Audio Splitting and Joining Stages.

"""

import math
import os
from dataclasses import dataclass

import torchaudio
from loguru import logger

from nemo_curator.stages.audio.tagging.inference.nemo_asr_align import NeMoASRAlignerStage
from nemo_curator.stages.base import CompositeStage, ProcessingStage
from nemo_curator.tasks import AudioTask


@dataclass
class SplitLongAudioStage(ProcessingStage[AudioTask, AudioTask]):
    """
    Stage that splits long audio files into smaller segments.

    Processes audio files that exceed a specified maximum length by splitting
    them at natural pauses to maintain speech coherence.

    Args:
        suggested_max_len: Target maximum length for audio segments in seconds
        min_len: Minimum length for any split segment
    """

    # Split parameters
    suggested_max_len: float = 3600.0
    min_len: float = 1.0

    # Stage metadata
    name: str = "SplitLongAudio"

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], ["duration", "segments", "resampled_audio_filepath"]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [
            "duration",
            "segments",
            "resampled_audio_filepath",
            "split_filepaths",
            "split_metadata",
            "split_offsets",
            "split_timestamps",
        ]

    def get_split_points(self, metadata: dict) -> list[float]:
        """Get the split points for the audio file based on segments."""
        splits = []
        split_start = 0
        prev_end = 0

        segments = metadata.get("segments", [])
        for segment in segments:
            end = segment.get("end", 0)

            if end - split_start > self.suggested_max_len:
                splits.append(prev_end)
                split_start = prev_end

            prev_end = end

        return splits

    def process(self, task: AudioTask) -> AudioTask:
        """Process entry to split long audio files."""
        data_entry = task.data
        duration = data_entry["duration"]

        # If audio is short enough, no splitting needed
        if duration < self.suggested_max_len:
            data_entry["split_filepaths"] = [data_entry["resampled_audio_filepath"]]
            data_entry["split_metadata"] = [
                {
                    "audio_item_id": data_entry.get("audio_item_id", "unknown"),
                    "resampled_audio_filepath": data_entry["resampled_audio_filepath"],
                    "duration": duration,
                }
            ]
            data_entry["split_offsets"] = [0.0]
            data_entry["split_timestamps"] = [0.0]
            return task

        # Get split points
        splits = self.get_split_points(data_entry)

        # Load audio
        audio_path = data_entry["resampled_audio_filepath"]
        audio, sr = torchaudio.load(audio_path)

        path, filename = os.path.split(audio_path)
        split_start = 0
        split_filepaths = []
        actual_splits = []
        split_durations = []

        # Process each split
        for k, split in enumerate(splits):
            split_filepath = os.path.join(path, os.path.splitext(filename)[0] + f".{k + 1}_of_{1 + len(splits)}.wav")
            split_end = math.ceil(split * sr)

            if split_end - split_start > self.min_len * sr:
                torchaudio.save(split_filepath, audio[:, split_start:split_end], sr)
                split_filepaths.append(split_filepath)
                actual_splits.append(split_start / sr)
                split_durations.append((split_end - split_start) / sr)
                split_start = split_end

        # Handle the last split
        split_filepath = os.path.join(
            path, os.path.splitext(filename)[0] + f".{1 + len(splits)}_of_{1 + len(splits)}.wav"
        )
        last_frame = len(audio[0])
        remaining_frames = last_frame - split_start

        if remaining_frames > self.min_len * sr and remaining_frames < (self.suggested_max_len + 1) * sr:
            torchaudio.save(split_filepath, audio[:, split_start:], sr)
            split_filepaths.append(split_filepath)
            split_durations.append(remaining_frames / sr)
            actual_splits.append(split_start / sr)

        audio_item_id = data_entry.get("audio_item_id", "unknown")

        if not split_filepaths:
            duration = len(audio[0]) / sr
            logger.warning(
                f"[{self.name}] No split files produced for entry "
                f"'{audio_item_id}' (duration={duration:.1f}s, splits={splits}). "
                f"Falling back to full audio file."
            )
            split_filepaths = [audio_path]
            split_durations = [duration]
            actual_splits = [0.0]
            all_split_metadata = [
                {
                    "audio_item_id": audio_item_id,
                    "resampled_audio_filepath": audio_path,
                    "duration": duration,
                }
            ]
        else:
            # Create entries for each split
            all_split_metadata = []
            for idx, split_path in enumerate(split_filepaths):
                split_metadata = {
                    "audio_item_id": f"{audio_item_id}_{idx}",
                    "resampled_audio_filepath": split_path,
                    "duration": split_durations[idx],
                }
                all_split_metadata.append(split_metadata)

        # Create meta-entry with split information
        data_entry["split_metadata"] = all_split_metadata
        data_entry["split_filepaths"] = split_filepaths
        data_entry["split_offsets"] = actual_splits
        data_entry["split_timestamps"] = splits
        return task


@dataclass
class JoinSplitAudioMetadataStage(ProcessingStage[AudioTask, AudioTask]):
    """
    Stage for joining metadata of previously split audio files.

    Combines the metadata (transcripts and alignments) of audio files that were
    previously split by SplitLongAudioStage. Adjusts timestamps and concatenates
    transcripts to recreate the original audio's metadata.
    """

    # Stage metadata
    name: str = "JoinSplitAudioMetadata"

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], ["split_filepaths", "split_metadata", "split_offsets", "split_timestamps"]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], ["text", "alignment"]

    def process(self, task: AudioTask) -> AudioTask:
        """
        Process entries and join split audio metadata.

        This stage collects all entries and processes meta-entries to join
        split audio files back together.
        """
        data_entry = task.data
        # Check if this is a meta-entry with split information
        if "split_filepaths" in data_entry:
            if data_entry["split_filepaths"] is None:
                # No splitting occurred, pass through
                del data_entry["split_filepaths"]
                return task
            else:
                # This is a meta-entry, process joining
                self._join_split_metadata(data_entry)
                return task

        # Regular entry without split info
        return task

    def _join_split_metadata(self, meta_entry: dict) -> None:
        """Join metadata from split audio files."""
        split_metadata = meta_entry.get("split_metadata", [])
        split_offsets = meta_entry.get("split_offsets", [])

        if not split_metadata:
            del meta_entry["split_filepaths"]
            return

        transcripts = []
        alignments = []

        # Find and join metadata from each split
        for idx, split_entry in enumerate(split_metadata):
            text = split_entry.get("text", "")
            if text:
                transcripts.append(text)

            alignment = split_entry.get("alignment", [])
            offset = split_offsets[idx] if idx < len(split_offsets) else 0

            for word in alignment:
                adjusted_word = dict(word)
                adjusted_word["start"] = round(word.get("start", 0) + offset, 3)
                adjusted_word["end"] = round(word.get("end", 0) + offset, 3)
                alignments.append(adjusted_word)

        # Create joined entry
        meta_entry["text"] = " ".join(transcripts)
        meta_entry["alignment"] = alignments

        # Remove split-related fields
        for key in ["split_filepaths", "split_metadata"]:
            meta_entry.pop(key, None)


@dataclass
class SplitASRAlignJoinStage(CompositeStage[AudioTask, AudioTask]):
    """Composite stage: Split long audio -> ASR align -> Join results.

    Decomposes into three sequential stages that always run together:
    1. SplitLongAudioStage — splits audio exceeding ``suggested_max_len``
    2. NeMoASRAlignerStage — transcribes and aligns each chunk
    3. JoinSplitAudioMetadataStage — merges transcripts back into original entries

    Args:
        suggested_max_len: Target max length for audio segments (seconds).
        min_len: Minimum length for any split segment (also used by ASR).
        max_len: Maximum length of audio segments for ASR processing (seconds).
        model_name: Pretrained NeMo ASR model name.
        model_path: Local model file path (overrides ``model_name`` if set).
        is_fastconformer: Whether the model encoder is FastConformer.
        decoder_type: Decoder type — ``"ctc"`` or ``"rnnt"``.
        batch_size: Entries per processing chunk in ASR.
        transcribe_batch_size: Batch size passed to the ASR model's transcribe call.
        split_batch_size: Max entries/paths per batch when chunking segments.
        num_workers: Data-loading workers for ASR inference.
        infer_segment_only: If True, run ASR only on individual segments
            rather than full audio / meta-entries.
        compute_timestamps: Whether to compute word-level timestamps.
        timestamp_type: Timestamp granularity (``"word"`` or ``"char"``).
        text_key: Output key for predicted text.
        words_key: Output key for word-level alignments.
        disable_word_confidence: Whether to disable word confidence scores.
        segments_key: Key for the segments list in each manifest entry.
    """

    # Split parameters
    suggested_max_len: float = 3600.0
    min_len: float = 1.0

    # ASR model configuration
    model_name: str = "nvidia/parakeet-tdt_ctc-1.1b"
    model_path: str | None = None
    is_fastconformer: bool = True
    decoder_type: str = "rnnt"

    # ASR length constraints
    max_len: float = 40.0

    # ASR processing parameters
    batch_size: int = 100
    transcribe_batch_size: int = 32
    split_batch_size: int = 5000
    num_workers: int = 10
    infer_segment_only: bool = False

    # ASR timestamp settings
    compute_timestamps: bool = True
    timestamp_type: str = "word"

    # ASR output keys
    text_key: str = "text"
    words_key: str = "words"
    disable_word_confidence: bool = False
    segments_key: str = "segments"

    name: str = "SplitASRAlignJoin"

    def __post_init__(self) -> None:
        super().__init__()

    def decompose(self) -> list[ProcessingStage]:
        return [
            SplitLongAudioStage(
                suggested_max_len=self.suggested_max_len,
                min_len=self.min_len,
            ),
            NeMoASRAlignerStage(
                model_name=self.model_name,
                model_path=self.model_path,
                is_fastconformer=self.is_fastconformer,
                decoder_type=self.decoder_type,
                min_len=self.min_len,
                max_len=self.max_len,
                batch_size=self.batch_size,
                transcribe_batch_size=self.transcribe_batch_size,
                split_batch_size=self.split_batch_size,
                num_workers=self.num_workers,
                infer_segment_only=self.infer_segment_only,
                compute_timestamps=self.compute_timestamps,
                timestamp_type=self.timestamp_type,
                text_key=self.text_key,
                words_key=self.words_key,
                disable_word_confidence=self.disable_word_confidence,
                segments_key=self.segments_key,
            ),
            JoinSplitAudioMetadataStage(),
        ]
