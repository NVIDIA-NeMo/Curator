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

import hashlib
import math
import posixpath
import time
from dataclasses import dataclass

import torchaudio
from fsspec.core import url_to_fs
from loguru import logger

from nemo_curator.stages.audio._agent._agent_ready import AgentReady, Gates, IOSpec, StageContract
from nemo_curator.stages.audio.tagging.inference.nemo_asr_align import NeMoASRAlignerStage
from nemo_curator.stages.base import CompositeStage, ProcessingStage
from nemo_curator.tasks import AudioTask


@dataclass
class SplitLongAudioStage(AgentReady, ProcessingStage[AudioTask, AudioTask]):
    """
    Stage that splits long audio files into smaller segments.

    Processes audio files that exceed a specified maximum length by splitting
    them at natural pauses to maintain speech coherence.

    Args:
        suggested_max_len: Target maximum length for audio segments in seconds
        min_len: Minimum length for any split segment
        output_dir: Optional directory for written split audio. When unset,
            split files remain beside the source audio for backward compatibility.
    """

    # Split parameters
    suggested_max_len: float = 3600.0
    min_len: float = 1.0
    duration_key: str = "duration"
    segments_key: str = "segments"
    audio_filepath_key: str = "resampled_audio_filepath"
    audio_item_id_key: str = "audio_item_id"
    split_filepaths_key: str = "split_filepaths"
    split_metadata_key: str = "split_metadata"
    split_offsets_key: str = "split_offsets"
    split_timestamps_key: str = "split_timestamps"

    # Stage metadata
    name: str = "SplitLongAudio"
    # Additive agent-only routing knob. Keep it after every legacy field so
    # positional construction retains its historical argument order.
    output_dir: str | None = None

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.duration_key, self.segments_key, self.audio_filepath_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [
            self.duration_key,
            self.segments_key,
            self.audio_filepath_key,
            self.split_filepaths_key,
            self.split_metadata_key,
            self.split_offsets_key,
            self.split_timestamps_key,
        ]

    def describe(self) -> StageContract:
        return StageContract(
            reads=IOSpec(data_keys=[self.duration_key, self.segments_key, self.audio_filepath_key]),
            writes=IOSpec(
                data_keys=[
                    self.split_filepaths_key,
                    self.split_metadata_key,
                    self.split_offsets_key,
                    self.split_timestamps_key,
                ],
                produces=["disk"],
            ),
            cardinality="1:1 nested-list",
            iteration_key=self.split_metadata_key,
            # A row splits by its own duration and segments. With an ``output_dir`` every file
            # shares one flat namespace, so the stem carries a hash of the source path -- which
            # separates ``spk1/utt1.wav`` from ``spk2/utt1.wav`` but not two rows naming the same
            # path with different segments. Without one, splits land beside their source.
            gates=Gates(
                writes_to_disk=True,
                output_path_params=["output_dir"],
                per_row_independent=self.output_dir is None,
            ),
        )

    def get_split_points(self, metadata: dict) -> list[float]:
        """Get the split points for the audio file based on segments."""
        splits = []
        split_start = 0
        prev_end = 0

        segments = sorted(metadata.get(self.segments_key, []), key=lambda s: s.get("start", 0))
        for segment in segments:
            end = segment.get("end", 0)

            if end - split_start > self.suggested_max_len:
                splits.append(prev_end)
                split_start = prev_end

            prev_end = end

        return splits

    def _prepare_output_dir(self) -> str:
        """Create and resolve an explicit output directory."""
        if self.output_dir is None:
            return ""
        output_fs, resolved_output_dir = url_to_fs(self.output_dir)
        output_fs.makedirs(resolved_output_dir, exist_ok=True)
        return resolved_output_dir

    def _split_paths(
        self,
        split_name: str,
        parent_url: str,
        resolved_parent: str,
        resolved_output_dir: str,
    ) -> tuple[str, str]:
        """Build the stored and resolved paths for one split."""
        if self.output_dir is None:
            split_filepath = f"{parent_url}/{split_name}" if parent_url else split_name
            split_resolved = f"{resolved_parent}/{split_name}" if resolved_parent else split_name
            return split_filepath, split_resolved
        return posixpath.join(self.output_dir, split_name), posixpath.join(resolved_output_dir, split_name)

    def process(self, task: AudioTask) -> AudioTask:
        """Process entry to split long audio files."""
        with self._time_metric("process_time"):
            return self._do_split(task)

    def _do_split(self, task: AudioTask) -> AudioTask:
        """Core splitting logic, separated to keep statement count within limits."""
        data_entry = task.data
        duration = data_entry[self.duration_key]

        if duration < self.suggested_max_len:
            data_entry[self.split_filepaths_key] = [data_entry[self.audio_filepath_key]]
            data_entry[self.split_metadata_key] = [
                {
                    self.audio_item_id_key: data_entry.get(self.audio_item_id_key, "unknown"),
                    self.audio_filepath_key: data_entry[self.audio_filepath_key],
                    self.duration_key: duration,
                }
            ]
            data_entry[self.split_offsets_key] = [0.0]
            data_entry[self.split_timestamps_key] = [0.0]
            self._log_metrics({"input_duration": duration, "splits_produced": 1})
            return task

        splits = self.get_split_points(data_entry)

        audio_path = data_entry[self.audio_filepath_key]
        _fs, resolved_path = url_to_fs(audio_path)

        # parent_url preserves protocol prefix (e.g. "s3://bucket/dir") for stored paths;
        # resolved_parent is the fsspec-resolved counterpart for torchaudio I/O.
        parent_url, filename = audio_path.rsplit("/", 1) if "/" in audio_path else ("", audio_path)
        resolved_parent = resolved_path.rsplit("/", 1)[0] if "/" in resolved_path else ""
        stem = filename.rsplit(".", 1)[0] if "." in filename else filename
        if self.output_dir is not None:
            # output_dir flattens the corpus into one namespace, and the parent folder was the
            # only thing keeping two ``utt1.wav`` apart. Unset, splits land beside their source,
            # so that route keeps byte-identical names.
            stem = f"{stem}_{hashlib.sha256(audio_path.encode()).hexdigest()[:8]}"

        resolved_output_dir = self._prepare_output_dir()

        audio, sr = torchaudio.load(resolved_path)

        split_start = 0
        split_filepaths, actual_splits, split_durations = [], [], []

        for k, split in enumerate(splits):
            split_name = f"{stem}.{k + 1}_of_{1 + len(splits)}.wav"
            split_filepath, split_resolved = self._split_paths(
                split_name,
                parent_url,
                resolved_parent,
                resolved_output_dir,
            )
            split_end = math.ceil(split * sr)

            if split_end - split_start > self.min_len * sr:
                torchaudio.save(split_resolved, audio[:, split_start:split_end], sr)
                split_filepaths.append(split_filepath)
                actual_splits.append(split_start / sr)
                split_durations.append((split_end - split_start) / sr)
                split_start = split_end

        split_name = f"{stem}.{1 + len(splits)}_of_{1 + len(splits)}.wav"
        split_filepath, split_resolved = self._split_paths(
            split_name,
            parent_url,
            resolved_parent,
            resolved_output_dir,
        )
        last_frame = len(audio[0])
        remaining_frames = last_frame - split_start

        if remaining_frames > self.min_len * sr:
            torchaudio.save(split_resolved, audio[:, split_start:], sr)
            split_filepaths.append(split_filepath)
            split_durations.append(remaining_frames / sr)
            actual_splits.append(split_start / sr)

        audio_item_id, split_filepaths_before = (
            data_entry.get(self.audio_item_id_key, "unknown"),
            bool(split_filepaths),
        )

        if not split_filepaths:
            logger.warning(
                f"[{self.name}] No split files produced for entry "
                f"'{audio_item_id}' (duration={duration:.1f}s, splits={splits}). "
                f"Falling back to full audio file."
            )
            split_filepaths = [audio_path]
            split_durations = [duration]
            actual_splits = [0.0]

        data_entry[self.split_metadata_key] = self._build_split_metadata(
            audio_item_id,
            split_filepaths,
            split_durations,
            fallback=not split_filepaths_before,
        )
        data_entry[self.split_filepaths_key] = split_filepaths
        data_entry[self.split_offsets_key] = actual_splits
        data_entry[self.split_timestamps_key] = splits
        self._log_metrics({"input_duration": duration, "splits_produced": len(split_filepaths)})
        return task

    def _build_split_metadata(
        self,
        audio_item_id: str,
        split_filepaths: list[str],
        split_durations: list[float],
        *,
        fallback: bool = False,
    ) -> list[dict]:
        """Build per-split metadata dicts from filepaths and durations."""
        if fallback:
            return [
                {
                    self.audio_item_id_key: audio_item_id,
                    self.audio_filepath_key: split_filepaths[0],
                    self.duration_key: split_durations[0],
                }
            ]
        return [
            {
                self.audio_item_id_key: f"{audio_item_id}_{idx}",
                self.audio_filepath_key: path,
                self.duration_key: split_durations[idx],
            }
            for idx, path in enumerate(split_filepaths)
        ]


@dataclass
class JoinSplitAudioMetadataStage(AgentReady, ProcessingStage[AudioTask, AudioTask]):
    """
    Stage for joining metadata of previously split audio files.

    Combines the metadata (transcripts and alignments) of audio files that were
    previously split by SplitLongAudioStage. Adjusts timestamps and concatenates
    transcripts to recreate the original audio's metadata.

    Args:
        text_key: Key used for transcript text in split entries.
                  Defaults to ``"text"`` for backward compatibility.
    """

    text_key: str = "text"
    split_filepaths_key: str = "split_filepaths"
    split_metadata_key: str = "split_metadata"
    split_offsets_key: str = "split_offsets"
    split_timestamps_key: str = "split_timestamps"
    alignment_key: str = "alignment"
    name: str = "JoinSplitAudioMetadata"

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [
            self.split_filepaths_key,
            self.split_metadata_key,
            self.split_offsets_key,
            self.split_timestamps_key,
        ]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.text_key, self.alignment_key]

    def describe(self) -> StageContract:
        return StageContract(
            reads=IOSpec(
                data_keys=[
                    self.split_filepaths_key,
                    self.split_metadata_key,
                    self.split_offsets_key,
                    self.split_timestamps_key,
                ]
            ),
            writes=IOSpec(data_keys=[self.text_key, self.alignment_key]),
            # Rejoins the chunks THIS row was split into, all of which came from its own file.
            gates=Gates(per_row_independent=True),
        )

    def process(self, task: AudioTask) -> AudioTask:
        """
        Process entries and join split audio metadata.

        This stage collects all entries and processes meta-entries to join
        split audio files back together.
        """
        t0 = time.perf_counter()
        data_entry = task.data
        splits_joined = 0
        words_aligned = 0

        # Check if this is a meta-entry with split information
        if self.split_filepaths_key in data_entry:
            if data_entry[self.split_filepaths_key] is None:
                del data_entry[self.split_filepaths_key]
            else:
                splits_joined = len(data_entry.get(self.split_metadata_key, []))
                self._join_split_metadata(data_entry)
                words_aligned = len(data_entry.get(self.alignment_key, []))

        self._log_metrics(
            {
                "process_time": time.perf_counter() - t0,
                "splits_joined": splits_joined,
                "words_aligned": words_aligned,
            }
        )
        return task

    def _join_split_metadata(self, meta_entry: dict) -> None:
        """Join metadata from split audio files."""
        split_metadata = meta_entry.get(self.split_metadata_key, [])
        split_offsets = meta_entry.get(self.split_offsets_key, [])

        if not split_metadata:
            del meta_entry[self.split_filepaths_key]
            return

        transcripts = []
        alignments = []

        # Find and join metadata from each split
        for idx, split_entry in enumerate(split_metadata):
            text = split_entry.get(self.text_key, "")
            if text:
                transcripts.append(text)

            alignment = split_entry.get(self.alignment_key, [])
            offset = split_offsets[idx] if idx < len(split_offsets) else 0

            for word in alignment:
                adjusted_word = dict(word)
                adjusted_word["start"] = round(word.get("start", 0) + offset, 3)
                adjusted_word["end"] = round(word.get("end", 0) + offset, 3)
                alignments.append(adjusted_word)

        # Create joined entry
        meta_entry[self.text_key] = " ".join(transcripts)
        meta_entry[self.alignment_key] = alignments

        # Remove split-related fields
        for key in [self.split_filepaths_key, self.split_metadata_key]:
            meta_entry.pop(key, None)


@dataclass
class SplitASRAlignJoinStage(AgentReady, CompositeStage[AudioTask, AudioTask]):
    """Composite stage: Split long audio -> ASR align -> Join results.

    Decomposes into three sequential stages that always run together:
    1. SplitLongAudioStage — splits audio exceeding ``suggested_max_len``
    2. NeMoASRAlignerStage — transcribes and aligns each chunk
    3. JoinSplitAudioMetadataStage — merges transcripts back into original entries

    Args:
        suggested_max_len: Target max length for audio segments (seconds).
        min_len: Minimum length for any split segment (also used by ASR).
        output_dir: Optional directory for split audio chunks. When unset,
            chunks are written beside their source audio.
        max_len: Maximum length of audio segments for ASR processing (seconds).
        model_name: Pretrained NeMo ASR model name.
        model_path: Local model file path (overrides ``model_name`` if set).
        is_fastconformer: Whether the model encoder is FastConformer.
        decoder_type: Decoder type — ``"ctc"`` or ``"rnnt"``.
        batch_size: Entries per processing chunk in ASR.
        transcribe_batch_size: Batch size passed to the ASR model's transcribe call.
        split_batch_size: Max entries/paths per batch when chunking segments.
        dataloader_num_workers: Data-loading workers for ASR inference.
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
    dataloader_num_workers: int = 10
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
    # Additive agent-only routing knob. Keep it after every legacy field so
    # positional construction retains its historical argument order.
    output_dir: str | None = None

    def __post_init__(self) -> None:
        super().__init__()

    def describe(self) -> StageContract:
        return StageContract(
            wrappable=False,
            # Mirrors the delegate that decides it: the aligner and the join are per-row, so the
            # composite is independent exactly when its ``SplitLongAudioStage`` is -- which is
            # when no ``output_dir`` flattens every source's splits into one namespace.
            gates=Gates(per_row_independent=self.output_dir is None),
        )

    def decompose(self) -> list[ProcessingStage]:
        return [
            SplitLongAudioStage(
                suggested_max_len=self.suggested_max_len,
                min_len=self.min_len,
                output_dir=self.output_dir,
                # Forwarded, or configuring the composite would silently not reach the splitter:
                # it would keep reading "segments" while the aligner read the configured key,
                # which is what blocks feeding diarization segments into ASR.
                segments_key=self.segments_key,
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
                dataloader_num_workers=self.dataloader_num_workers,
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
