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
NeMo ASR Aligner Stage.

Contains BaseASRProcessorStage (shared config and segment preparation)
and NeMoASRAlignerStage (forced alignment via NeMo FastConformer).

These stages are tagging-pipeline-specific because they operate on
tagging manifest keys like ``split_filepaths``, ``split_metadata``,
and ``segments``.
"""

from dataclasses import dataclass, field
from typing import Any

import nemo.collections.asr as nemo_asr
import torch
import torchaudio
from loguru import logger
from nemo.collections.asr.parts.submodules.ctc_decoding import CTCDecodingConfig
from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTDecodingConfig

from nemo_curator.backends.base import NodeInfo, WorkerMetadata
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioTask


@dataclass
class BaseASRProcessorStage(ProcessingStage[AudioTask, AudioTask]):
    """Base class for ASR stages with shared config and segment preparation.

    Provides common fields and _prepare_segment_batch_with_metadata for
    segment-only inference. Subclasses must implement setup() and process().

    Args:
        min_len: Minimum length of audio segments to process (seconds).
        max_len: Maximum length of audio segments to process (seconds).
        num_workers: Number of workers for data loading.
        split_batch_size: Max entries/paths per batch when chunking.
        infer_segment_only: If True, process segments only; else full audio / meta-entries.
        text_key: Key for predicted text in manifest.
        words_key: Key for word alignments in manifest (same as SDP alignment_key).
        device: Device to run the model on ("cuda" or "cpu").
        compute_timestamps: Whether to compute word-level timestamps.
        segments_key: Key for segments list in manifest.
    """

    # Length constraints
    min_len: float = 1.0
    max_len: float = 40.0

    # Processing parameters
    batch_size: int = 32
    num_workers: int = 10
    split_batch_size: int = 5000
    infer_segment_only: bool = False

    # Output keys
    text_key: str = "text"
    words_key: str = "words"

    # Device and timestamps
    device: str = "cuda"
    compute_timestamps: bool = True
    segments_key: str = "segments"

    # Stage metadata (subclasses can override)
    name: str = "BaseASRProcessor"
    output_dir: str = None

    def _prepare_segment_batch_with_metadata(
        self,
        metadata_batch: list[dict],
        cut_audio_segments: bool = False,
        segments_key: str = "segments",
    ) -> list[dict]:
        """Prepare segment metadata for a batch.

        Collects segment metadata with indices for later processing. Mirrors
        generic_sdp BaseASRProcessor._prepare_segment_batch_with_metadata.

        Args:
            metadata_batch: List of metadata dicts, each with a segments list.
            cut_audio_segments: If True, load audio and cut segments (numpy);
                if False, only collect resampled_audio_filepath from segments.
            segments_key: Key for the segments list in each metadata dict.

        Returns:
            List of segment metadata dicts with metadata_idx, segment_idx, and
            either "audio_segment" (numpy) or "resampled_audio_filepath".
        """
        segment_metadata_list: list[dict] = []

        if cut_audio_segments:
            for metadata_idx, metadata in enumerate(metadata_batch):
                audio_path = metadata.get("resampled_audio_filepath", metadata.get("audio_filepath"))
                if not audio_path:
                    continue
                audio, sr = torchaudio.load(audio_path)
                for segment_idx, segment in enumerate(metadata.get(segments_key, [])):
                    duration = segment.get("end", 0) - segment.get("start", 0)
                    if duration >= self.min_len:
                        start = int(segment["start"] * sr)
                        end = int(segment["end"] * sr)
                        audio_segment = audio[:, start:end].squeeze(0)
                        if len(audio_segment) > 0:
                            segment_metadata_list.append(
                                {
                                    "audio_segment": audio_segment.numpy(),
                                    "metadata_idx": metadata_idx,
                                    "segment_idx": segment_idx,
                                }
                            )
        else:
            for metadata_idx, metadata in enumerate(metadata_batch):
                for segment_idx, segment in enumerate(metadata.get(segments_key, [])):
                    if "resampled_audio_filepath" in segment:
                        segment_metadata_list.append(
                            {
                                "resampled_audio_filepath": segment["resampled_audio_filepath"],
                                "metadata_idx": metadata_idx,
                                "segment_idx": segment_idx,
                            }
                        )

        return segment_metadata_list


@dataclass
class NeMoASRAlignerStage(BaseASRProcessorStage):
    """
    Stage that aligns text and audio using NeMo ASR models.

    Uses a pre-trained ASR model to transcribe audio files and generate
    word-level alignments with timestamps. Supports both CTC and RNNT decoders and
    can process either full audio files or just specific segments.

    Args:
        model_name (str): Name of pretrained model to use. Defaults to "nvidia/parakeet-tdt_ctc-1.1b"
        model_path (str, Optional): Path to local model file. If provided, overrides model_name
        is_fastconformer (bool): Whether model's encoder is FastConformer
        decoder_type (str): Type of decoder ('ctc' or 'rnnt'). Defaults to "rnnt"
        transcribe_batch_size (int): Batch size for transcribing. Defaults to 32
        timestamp_type (str): Type of timestamp ('word' or 'char')
        disable_word_confidence (bool): Whether to disable word confidence score computation
        device (str): Device to run model on. Defaults to "cuda"
    """

    # Model configuration
    model_name: str = "nvidia/parakeet-tdt_ctc-1.1b"
    model_path: str | None = None

    # Length constraints
    min_len: float = 1.0
    max_len: float = 40.0

    # Model settings
    is_fastconformer: bool = True
    decoder_type: str = "rnnt"

    # Processing parameters
    transcribe_batch_size: int = 32
    num_workers: int = 10
    batch_size: int = 100

    # Timestamp settings
    compute_timestamps: bool = True
    timestamp_type: str = "word"

    # Processing mode
    infer_segment_only: bool = False

    # Output keys
    text_key: str = "text"
    words_key: str = "words"
    disable_word_confidence: bool = False
    segments_key: str = "segments"
    # Device
    device: str = "cuda"

    # Stage metadata
    name: str = "NeMoASRAligner"
    _asr_model: Any = None
    _override_cfg: Any = None
    _model_initialized: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        """Validate config."""
        if self.decoder_type not in ["ctc", "rnnt"]:
            msg = f"decoder_type must be 'ctc' or 'rnnt', got {self.decoder_type}"
            raise ValueError(msg)

    def load_model(self) -> None:
        if self.model_path:
            self._asr_model = nemo_asr.models.ASRModel.restore_from(restore_path=self.model_path)
        else:
            self._asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=self.model_name)

    def setup_on_node(self, node_info: NodeInfo, worker_metadata: WorkerMetadata) -> None:  # noqa: ARG002
        """Download model weights (called once per node)."""
        if self._asr_model is None:
            self.load_model()

    def setup(self, worker_metadata: Any = None) -> None:  # noqa: ARG002, ANN401
        """Load model to device and configure decoding (called per replica)."""
        if self._model_initialized:
            return

        if not torch.cuda.is_available() and self.device == "cuda":
            logger.warning("CUDA not available, using CPU")
            self.device = "cpu"

        if self._asr_model is None:
            self.load_model()

        self._asr_model.to(self.device)
        self._asr_model.eval()

        if self.is_fastconformer:
            self._asr_model.change_attention_model(
                self_attention_model="rel_pos_local_attn", att_context_size=[128, 128]
            )
            self._asr_model.change_subsampling_conv_chunking_factor(1)

        decoding_cfg = CTCDecodingConfig() if self.decoder_type == "ctc" else RNNTDecodingConfig()

        if self.decoder_type == "ctc":
            decoding_cfg.strategy = "greedy_batch"
        else:
            decoding_cfg.rnnt_timestamp_type = self.timestamp_type

        decoding_cfg.preserve_alignments = self.compute_timestamps
        decoding_cfg.confidence_cfg.preserve_word_confidence = not self.disable_word_confidence
        decoding_cfg.compute_timestamps = self.compute_timestamps
        decoding_cfg.greedy.compute_timestamps = self.compute_timestamps

        self._asr_model.change_decoding_strategy(decoding_cfg=decoding_cfg)

        self._override_cfg = self._asr_model.get_transcribe_config()
        self._override_cfg.batch_size = self.transcribe_batch_size
        self._override_cfg.num_workers = self.num_workers
        self._override_cfg.return_hypotheses = True
        self._override_cfg.timestamps = self.compute_timestamps

        self._model_initialized = True
        logger.info(f"[{self.name}] Initialized ASR model on {self.device}")

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["duration", "segments", "split_filepaths", "split_metadata"]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["duration", "segments", "split_filepaths", "split_metadata"]

    def get_alignments_text(self, hypotheses: Any) -> tuple[list, str]:  # noqa: ANN401
        """Extract word alignments and text from model hypotheses."""
        if not self.compute_timestamps:
            return [], hypotheses.text

        timestamp_dict = hypotheses.timestamp

        if self.is_fastconformer:
            time_stride = 8 * self._asr_model.cfg.preprocessor.window_stride
        else:
            time_stride = 4 * self._asr_model.cfg.preprocessor.window_stride

        word_timestamps = timestamp_dict[self.timestamp_type]

        alignments = []
        for i, stamp in enumerate(word_timestamps):
            conf = None
            if hypotheses.word_confidence is not None and i < len(hypotheses.word_confidence):
                conf = hypotheses.word_confidence[i]
                if isinstance(conf, torch.Tensor):
                    conf = conf.item()
                conf = round(conf, 4)

            if self.decoder_type == "ctc":
                start = stamp["start_offset"] * time_stride
                end = stamp["end_offset"] * time_stride
            else:
                start = max(0, stamp["start_offset"] * time_stride - 0.08)
                end = max(0, stamp["end_offset"] * time_stride - 0.08)

            word = stamp.get("word", stamp.get("char", ""))
            alignments.append(
                {
                    "word": word,
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "confidence": conf,
                }
            )

        text = " ".join(w["word"] for w in alignments)
        text = text.replace("⁇", "")

        return alignments, text

    def process(self, task: AudioTask) -> AudioTask:
        msg = "NeMoASRAlignerStage only supports process_batch"
        raise NotImplementedError(msg)

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        """Process a batch of AudioTasks for ASR alignment."""
        if len(tasks) == 0:
            return []
        if self.infer_segment_only:
            return self.process_segments(tasks)
        else:
            return self.process_full_audio(tasks)

    def process_full_audio(self, tasks: list[AudioTask]) -> list[AudioTask]:  # noqa: C901, PLR0912, PLR0915
        """Process entries as full audio (or meta-entries with split_filepaths)."""
        entries = [task.data for task in tasks]
        skip_indices = []
        meta_indices = []
        for i, data in enumerate(entries):
            split_filepaths = data.get("split_filepaths")
            has_splits = isinstance(split_filepaths, list) and len(split_filepaths) > 0
            if has_splits or split_filepaths is None:
                meta_indices.append(i)
            else:
                skip_indices.append(i)

        for i in skip_indices:
            entries[i][self.text_key] = ""
            entries[i]["alignment"] = []

        for chunk_start in range(0, len(meta_indices), self.batch_size):
            chunk_meta_idx = meta_indices[chunk_start : chunk_start + self.batch_size]
            all_paths = []
            path_to_entry_and_split = []
            for entry_idx in chunk_meta_idx:
                meta_entry = entries[entry_idx]
                split_filepaths = meta_entry.get("split_filepaths")
                if not split_filepaths:
                    logger.warning(f"[{self.name}] Entry at index {entry_idx} has no split_filepaths, skipping.")
                    continue
                for split_idx, path in enumerate(split_filepaths):
                    all_paths.append(path)
                    path_to_entry_and_split.append((entry_idx, split_idx))

            if not all_paths:
                continue

            try:
                with torch.no_grad():
                    hypotheses_list = self._asr_model.transcribe(all_paths, override_config=self._override_cfg)
                if isinstance(hypotheses_list, tuple) and len(hypotheses_list) == 2:  # noqa: PLR2004
                    hypotheses_list = hypotheses_list[0]
            except Exception as e:  # noqa: BLE001
                logger.error(
                    f"[{self.name}] Exception for meta-entries batch: {e!s} for paths: {all_paths}, transcribing one by one"
                )
                hypotheses_list = []
                for path in all_paths:
                    try:
                        with torch.no_grad():
                            hyp = self._asr_model.transcribe([path], override_config=self._override_cfg)
                        if isinstance(hyp, tuple) and len(hyp) == 2:  # noqa: PLR2004
                            hyp = hyp[0]
                        hypotheses_list.append(hyp[0] if hyp else None)
                    except Exception as e2:  # noqa: BLE001, PERF203
                        logger.error(f"[{self.name}] Exception for {path}: {e2}")
                        hypotheses_list.append(None)

            for path_idx, hyp in enumerate(hypotheses_list):
                if path_idx >= len(path_to_entry_and_split):
                    break
                entry_idx, split_idx = path_to_entry_and_split[path_idx]
                meta_entry = entries[entry_idx]
                if hyp is not None:
                    alignments, text = self.get_alignments_text(hyp)
                else:
                    alignments, text = [], ""

                split_metadata = meta_entry.get("split_metadata")
                if split_metadata and split_idx < len(split_metadata):
                    split_metadata[split_idx][self.text_key] = text
                    split_metadata[split_idx]["alignment"] = alignments
                else:
                    meta_entry[self.text_key] = text
                    meta_entry["alignment"] = alignments

        return tasks

    def process_segments(self, tasks: list[AudioTask]) -> list[AudioTask]:
        """Process entries in segment-only mode (infer per segment)."""
        entries = [task.data for task in tasks]
        if not entries:
            return []

        for chunk_start in range(0, len(entries), self.batch_size):
            metadata_batch = entries[chunk_start : chunk_start + self.batch_size]
            segment_metadata_list = self._prepare_segment_batch_with_metadata(
                metadata_batch,
                cut_audio_segments=True,
                segments_key=self.segments_key,
            )
            all_segments = [seg["audio_segment"] for seg in segment_metadata_list]

            if len(all_segments) == 0:
                continue

            try:
                with torch.no_grad():
                    hypotheses_list = self._asr_model.transcribe(all_segments, override_config=self._override_cfg)
            except Exception as e:
                files_list = [x.get("resampled_audio_filepath", x.get("audio_filepath")) for x in metadata_batch]
                msg = f"[{self.name}] Exception for audio list: {files_list}, error: {e}"
                raise ValueError(msg) from e

            if isinstance(hypotheses_list, tuple) and len(hypotheses_list) == 2:  # noqa: PLR2004
                hypotheses_list = hypotheses_list[0]

            for segment_metadata, hypotheses in zip(segment_metadata_list, hypotheses_list, strict=True):
                alignments, text = self.get_alignments_text(hypotheses)
                metadata_idx = segment_metadata["metadata_idx"]
                segment_idx = segment_metadata["segment_idx"]
                segment = metadata_batch[metadata_idx][self.segments_key][segment_idx]
                segment[self.text_key] = text
                if self.compute_timestamps:
                    seg_start = segment.get("start", 0)
                    for word in alignments:
                        word["start"] = round(word["start"] + seg_start, 3)
                        word["end"] = round(word["end"] + seg_start, 3)
                    segment[self.words_key] = alignments

        return tasks
