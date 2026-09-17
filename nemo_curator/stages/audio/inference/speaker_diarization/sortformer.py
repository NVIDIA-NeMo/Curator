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

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

import soundfile as sf
from huggingface_hub import snapshot_download
from loguru import logger
from nemo.collections.asr.models import SortformerEncLabelModel

from nemo_curator.backends.utils import RayStageSpecKeys
from nemo_curator.stages.audio._agent._agent_ready import AgentReady, Gates, IOSpec, StageContract, StaticHints
from nemo_curator.stages.audio._agent._residency import (
    InputResidency,
    cleanup_temp_files,
    resolve_audio_path,
    validate_input_residency,
)
from nemo_curator.stages.audio.inference.base import (
    _channel_first_waveform,
    _fanout_audio_segment,
    _fanout_original_file,
    _fanout_path_keys,
    _inference_audio_input_spec,
    _inference_audio_read_specs,
    _stable_audio_identity,
    _stable_source_path,
    _validate_fanout_key_contract,
    _validate_inference_audio_input,
)
from nemo_curator.stages.base import ProcessingStage

if TYPE_CHECKING:
    from nemo_curator.backends.base import NodeInfo, WorkerMetadata
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask


def _parse_sortformer_segments(raw_segments: list) -> list[dict[str, Any]]:
    """Convert Sortformer output segments to list of {start, end, speaker} dicts.

    Handles both string format ("start end speaker") and objects with
    start/end/speaker attributes.
    """
    segments: list[dict[str, Any]] = []
    for seg in raw_segments:
        if isinstance(seg, str):
            parts = seg.strip().split()
            segments.append(
                {
                    "start": float(parts[0]),
                    "end": float(parts[1]),
                    "speaker": parts[2] if len(parts) > 2 else "unknown",  # noqa: PLR2004
                }
            )
        elif hasattr(seg, "start") and hasattr(seg, "end"):
            segments.append(
                {
                    "start": float(seg.start),
                    "end": float(seg.end),
                    "speaker": str(getattr(seg, "speaker", getattr(seg, "label", "unknown"))),
                }
            )
        elif isinstance(seg, (tuple, list)) and len(seg) >= 3:  # noqa: PLR2004
            segments.append(
                {
                    "start": float(seg[0]),
                    "end": float(seg[1]),
                    "speaker": str(seg[2]),
                }
            )
        else:
            logger.warning(f"Unrecognised segment format: {seg!r}")
    return segments


def _count_distinct_speakers(segments: list[dict[str, Any]]) -> int:
    """Number of distinct speaker labels in diarization output.

    Sortformer emits no explicit speaker count; it is derived as the number of
    distinct cluster labels across turns. Parse-failure placeholders ("unknown")
    are not counted as a real speaker.
    """
    return len({seg.get("speaker") for seg in segments} - {None, "unknown"})


def _write_rttm(segments: list[dict[str, Any]], sess_name: str, rttm_out_dir: str) -> None:
    """Write diarization segments to an RTTM file."""
    os.makedirs(rttm_out_dir, exist_ok=True)
    rttm_path = os.path.join(rttm_out_dir, f"{sess_name}.rttm")
    with open(rttm_path, "w") as f:
        for seg in segments:
            duration = seg["end"] - seg["start"]
            if duration <= 0:
                logger.warning(f"Skipping degenerate segment with non-positive duration: {seg!r}")
                continue
            f.write(f"SPEAKER {sess_name} 1 {seg['start']:.3f} {duration:.3f} <NA> <NA> {seg['speaker']} <NA> <NA>\n")


@dataclass
class InferenceSortformerStage(AgentReady, ProcessingStage[AudioTask, AudioTask]):
    """Speaker diarization inference using Streaming Sortformer (NeMo).

    Uses the NeMo SortformerEncLabelModel for end-to-end neural speaker
    diarization with streaming support. See:
    https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1

    Args:
        model_name: Hugging Face model id. Defaults to "nvidia/diar_streaming_sortformer_4spk-v2.1".
        model_path: Local path to a .nemo checkpoint file; if set, takes precedence over model_name.
        cache_dir: Directory for caching downloaded model weights. Defaults to HF hub default.
        diar_model: Pre-loaded SortformerEncLabelModel; if provided, setup() is a no-op.
        filepath_key: Key in data for path to audio file. Defaults to "audio_filepath".
        diar_segments_key: Key in output data for diarization segments list. Defaults to "diar_segments".
        num_speakers_key: Optional output key for the distinct-speaker count derived
            from diar_segments. Disabled by default for legacy compatibility.
        rttm_out_dir: Optional directory to write RTTM files. Defaults to None.
        chunk_len: Streaming chunk size in 80 ms frames. Defaults to 340 (~30.4 s latency).
        chunk_left_context: Left context frames. Defaults to 1.
        chunk_right_context: Right context frames. Defaults to 40.
        fifo_len: FIFO queue size in frames. Defaults to 40.
        spkcache_update_period: Speaker cache update period in frames. Defaults to 300.
        spkcache_len: Speaker cache size in frames. Defaults to 188.
        inference_batch_size: Batch size passed to diarize(). Defaults to 1.
        name: Stage name. Defaults to "Sortformer_inference".
    """

    model_name: str = "nvidia/diar_streaming_sortformer_4spk-v2.1"
    model_path: str | None = None
    cache_dir: str | None = None
    diar_model: Any | None = None
    filepath_key: str = "audio_filepath"
    waveform_key: str = field(default="waveform", kw_only=True)
    sample_rate_key: str = field(default="sample_rate", kw_only=True)
    diar_segments_key: str = "diar_segments"
    num_speakers_key: str | None = field(default=None, kw_only=True)
    input_residency: InputResidency = field(default="file", kw_only=True)
    fanout: bool = field(default=False, kw_only=True)
    start_key: str = field(default="start", kw_only=True)
    end_key: str = field(default="end", kw_only=True)
    start_ms_key: str = field(default="start_ms", kw_only=True)
    end_ms_key: str = field(default="end_ms", kw_only=True)
    duration_key: str = field(default="duration", kw_only=True)
    segment_num_key: str = field(default="segment_num", kw_only=True)
    speaker_key: str = field(default="speaker", kw_only=True)
    original_file_key: str = field(default="original_file", kw_only=True)
    rttm_out_dir: str | None = None
    chunk_len: int = 340
    chunk_left_context: int = 1
    chunk_right_context: int = 40
    fifo_len: int = 40
    spkcache_update_period: int = 300
    spkcache_len: int = 188
    inference_batch_size: int = 1
    name: str = "Sortformer_inference"
    batch_size: int = 1
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0, gpu_memory_gb=8.0))

    AGENT_STATIC: ClassVar[StaticHints] = StaticHints(
        cardinality_options=["1:1", "1:N fan-out"],
        gates=Gates(
            writes_to_disk=True,
            requires_gpu=True,
            requires_internet_first_run=True,
            output_path_params=["rttm_out_dir"],
            per_row_independent=False,
        ),
    )

    def __post_init__(self) -> None:
        validate_input_residency(self.input_residency, stage_name=self.name)
        self.is_resumable = not self.fanout
        if self.num_speakers_key is not None and self.num_speakers_key in {
            self.filepath_key,
            self.diar_segments_key,
        }:
            msg = "num_speakers_key must be distinct from path and segment output keys when enabled"
            raise ValueError(msg)
        if self.fanout:
            _validate_fanout_key_contract(
                stage_name=self.name,
                audio_filepath_key=self.filepath_key,
                output_keys=[
                    self.waveform_key,
                    self.sample_rate_key,
                    self.start_key,
                    self.end_key,
                    self.start_ms_key,
                    self.end_ms_key,
                    self.duration_key,
                    self.segment_num_key,
                    self.speaker_key,
                    self.original_file_key,
                ],
                removed_container_keys=(self.diar_segments_key,),
            )

    def setup_on_node(
        self, _node_info: NodeInfo | None = None, _worker_metadata: WorkerMetadata | None = None
    ) -> None:
        """Pre-download model weights on the node so workers load from cache."""
        if self.model_path is not None or self.diar_model is not None:
            return
        snapshot_download(repo_id=self.model_name, cache_dir=self.cache_dir)

    def _resolve_model_path(self) -> str:
        """Resolve the path to the .nemo checkpoint from the HF cache."""
        if self.model_path is not None:
            return self.model_path
        repo_dir = snapshot_download(repo_id=self.model_name, cache_dir=self.cache_dir)
        nemo_files = sorted(f for f in os.listdir(repo_dir) if f.endswith(".nemo"))
        if not nemo_files:
            msg = f"No .nemo file found in {repo_dir} for model {self.model_name}"
            raise FileNotFoundError(msg)
        return os.path.join(repo_dir, nemo_files[0])

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        """Load Sortformer model from Hugging Face or a local .nemo file."""
        if self.diar_model is not None:
            self.diar_model.eval()
            self._configure_streaming()
            self._extend_pos_enc_for_long_audio()
            return

        resolved_path = self._resolve_model_path()
        self.diar_model = SortformerEncLabelModel.restore_from(
            restore_path=resolved_path,
            map_location="cuda",
            strict=False,
        )

        self.diar_model.eval()
        self._configure_streaming()
        self._extend_pos_enc_for_long_audio()

    def _extend_pos_enc_for_long_audio(self, max_len: int = 30000) -> None:
        """Extend RelPositionalEncoding buffer to handle long audio files.

        NeMo's streaming Sortformer initialises pos_enc sized for one chunk (~35
        conformer frames). Files longer than a few seconds overflow it at inference
        time. extend_pe() is a NeMo method that resizes the buffer safely — it just
        isn't called automatically. max_len=30000 covers ~1000 s at any subsampling.
        """
        pos_enc = getattr(getattr(self.diar_model, "encoder", None), "pos_enc", None)
        if pos_enc is None or not hasattr(pos_enc, "extend_pe"):
            logger.warning("pos_enc not found or no extend_pe method — skipping extension")
            return
        params = next(self.diar_model.parameters())
        try:
            pos_enc.extend_pe(max_len, params.device, params.dtype)
            logger.info(f"Extended encoder pos_enc to max_len={max_len} for long-form audio")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Could not extend pos_enc: {e}")

    def _configure_streaming(self) -> None:
        """Apply streaming configuration to the loaded model."""
        sm = self.diar_model.sortformer_modules
        sm.chunk_len = self.chunk_len
        sm.chunk_right_context = self.chunk_right_context
        sm.fifo_len = self.fifo_len
        sm.chunk_left_context = self.chunk_left_context
        if hasattr(sm, "spkcache_update_period"):
            sm.spkcache_update_period = self.spkcache_update_period
        sm.spkcache_len = self.spkcache_len

    def inputs(self) -> tuple[list[str], list[str]]:
        return _inference_audio_input_spec(
            self.input_residency,
            audio_filepath_key=self.filepath_key,
            waveform_key=self.waveform_key,
            sample_rate_key=self.sample_rate_key,
        )

    def validate_input(self, task: AudioTask) -> bool:
        return _validate_inference_audio_input(
            task,
            stage_name=self.name,
            residency=self.input_residency,
            audio_filepath_keys=(self.filepath_key,),
            waveform_key=self.waveform_key,
            sample_rate_key=self.sample_rate_key,
        )

    def outputs(self) -> tuple[list[str], list[str]]:
        if self.fanout:
            return ["data"], [
                self.waveform_key,
                self.sample_rate_key,
                self.start_key,
                self.end_key,
                self.start_ms_key,
                self.end_ms_key,
                self.duration_key,
                self.segment_num_key,
                self.speaker_key,
                self.original_file_key,
            ]
        output_keys = [self.filepath_key, self.diar_segments_key]
        if self.num_speakers_key is not None:
            output_keys.append(self.num_speakers_key)
        return ["data"], output_keys

    def describe(self) -> StageContract:
        if self.fanout:
            writes = [
                self.waveform_key,
                self.sample_rate_key,
                self.start_key,
                self.end_key,
                self.start_ms_key,
                self.end_ms_key,
                self.duration_key,
                self.segment_num_key,
                self.speaker_key,
                self.original_file_key,
            ]
            cardinality = "1:N fan-out"
        else:
            writes = [self.filepath_key, self.diar_segments_key]
            if self.num_speakers_key is not None:
                writes.append(self.num_speakers_key)
            cardinality = "1:1"
        return StageContract(
            reads_one_of=_inference_audio_read_specs(
                self.input_residency,
                audio_filepath_key=self.filepath_key,
                waveform_key=self.waveform_key,
                sample_rate_key=self.sample_rate_key,
            ),
            writes=IOSpec(data_keys=writes, produces=["tensor"] if self.fanout else []),
            cardinality=cardinality,
            cardinality_options=["1:1", "1:N fan-out"],
            iteration_key=self.segment_num_key if self.fanout else None,
            removes_keys=(
                list(dict.fromkeys([*_fanout_path_keys(self.filepath_key), self.diar_segments_key]))
                if self.fanout
                else []
            ),
            gates=Gates(
                requires_gpu=self.resources.requires_gpu or self.diar_model is None,
                writes_to_disk=self.rttm_out_dir is not None or self.input_residency != "file",
                # The ROW is per-file: ``process`` handles one task and calls ``diarize`` with a
                # single-element list, so the model never sees another file whatever
                # ``inference_batch_size`` says. The RTTM is not: its name falls back to the audio
                # BASENAME, so with a shared ``rttm_out_dir`` two files called ``utt1.wav`` in
                # different folders write the same ``utt1.rttm``. Drop the directory and the whole
                # stage is safe to run over a subset.
                per_row_independent=self.rttm_out_dir is None,
                requires_internet_first_run=self.model_path is None and self.diar_model is None,
                output_path_params=(
                    ["rttm_out_dir"]
                    if self.rttm_out_dir is not None
                    else ([] if self.input_residency != "file" else None)
                ),
            ),
        )

    def ray_stage_spec(self) -> dict[str, Any]:
        if self.fanout:
            return {RayStageSpecKeys.IS_FANOUT_STAGE: True}
        return {}

    def _segment_child_data(  # noqa: PLR0913
        self,
        item: dict[str, Any],
        segment: dict[str, Any],
        segment_num: int,
        waveform: Any,  # noqa: ANN401
        sample_rate: int,
        original_file: Any,  # noqa: ANN401
    ) -> dict[str, Any]:
        excluded = {
            self.diar_segments_key,
            self.waveform_key,
            self.sample_rate_key,
            *_fanout_path_keys(self.filepath_key),
        }
        child = {k: v for k, v in item.items() if k not in excluded}
        child.update({k: v for k, v in segment.items() if k not in {"start", "end", "speaker", *excluded}})
        raw_start = float(segment.get("start", 0.0))
        raw_end = float(segment.get("end", raw_start))
        child[self.waveform_key], start, end = _fanout_audio_segment(
            waveform, sample_rate, start=raw_start, end=raw_end
        )
        child[self.sample_rate_key] = int(sample_rate)
        child[self.start_key] = start
        child[self.end_key] = end
        child[self.start_ms_key] = round(start * 1000)
        child[self.end_ms_key] = round(end * 1000)
        child[self.duration_key] = max(0.0, end - start)
        child[self.segment_num_key] = segment_num
        if "speaker" in segment:
            child[self.speaker_key] = segment["speaker"]
        child[self.original_file_key] = original_file
        return child

    def _fanout_segments(
        self,
        task: AudioTask,
        segments: list[dict[str, Any]],
        waveform: Any,  # noqa: ANN401
        sample_rate: int,
        original_file: Any,  # noqa: ANN401
    ) -> list[AudioTask]:
        return [
            AudioTask(
                dataset_name=task.dataset_name,
                filepath_key=task.filepath_key or self.filepath_key,
                data=self._segment_child_data(
                    task.data,
                    segment,
                    index,
                    waveform,
                    sample_rate,
                    original_file,
                ),
                _metadata=dict(task._metadata or {}),
                _stage_perf=list(task._stage_perf),
            )
            for index, segment in enumerate(segments)
        ]

    def diarize(self, audio_paths: list[str]) -> list[list[dict[str, Any]]]:
        """Run Sortformer on a list of audio files.

        Returns a list (one entry per file) of segment lists [{start, end, speaker}].
        """
        predicted_segments = self.diar_model.diarize(
            audio=audio_paths,
            batch_size=self.inference_batch_size,
        )
        return [_parse_sortformer_segments(segs) for segs in predicted_segments]

    def process(self, task: AudioTask) -> AudioTask | list[AudioTask]:
        """Run speaker diarization on the audio file in the task."""
        if not self.validate_input(task):
            msg = f"[{self.name}] task {task.task_id!r} has no valid {self.input_residency} audio input"
            raise ValueError(msg)

        source_path = _stable_source_path(
            task.data,
            self.filepath_key,
            "audio_filepath",
            "resampled_audio_filepath",
        )
        resident_waveform = (
            _channel_first_waveform(task.data[self.waveform_key])
            if self.input_residency != "file"
            and task.data.get(self.waveform_key) is not None
            and task.data.get(self.sample_rate_key) is not None
            else None
        )
        resident_sample_rate = int(task.data[self.sample_rate_key]) if resident_waveform is not None else None
        audio_input = task.data if resident_waveform is None else {**task.data, self.waveform_key: resident_waveform}
        temp_paths: list[str] = []
        file_path = resolve_audio_path(
            audio_input,
            residency=self.input_residency,  # type: ignore[arg-type]
            audio_filepath_key=self.filepath_key,
            waveform_key=self.waveform_key,
            sample_rate_key=self.sample_rate_key,
            register_temp=temp_paths,
        )
        if file_path is None:
            msg = f"Task {task!s} missing audio input for {self.filepath_key}"
            raise ValueError(msg)
        try:
            identity_source_path = source_path or _stable_source_path(task.data, self.original_file_key)
            stable_identity = _stable_audio_identity(
                task.data,
                resident_waveform,
                resident_sample_rate or 0,
                source_path=identity_source_path,
                explicit_keys=("session_name",),
                fallback_keys=("audio_item_id",),
            )
            original_file = _fanout_original_file(
                task.data,
                original_file_key=self.original_file_key,
                source_path=source_path,
                stable_identity=stable_identity,
            )

            all_segments = self.diarize([file_path])
            segments = all_segments[0]

            if self.rttm_out_dir is not None:
                _write_rttm(segments, stable_identity, self.rttm_out_dir)

            if self.fanout:
                if resident_waveform is None:
                    decoded, decoded_rate = sf.read(file_path, dtype="float32")
                    resident_waveform = _channel_first_waveform(decoded if decoded.ndim == 1 else decoded.T)
                    resident_sample_rate = int(decoded_rate)
                return self._fanout_segments(
                    task,
                    segments,
                    resident_waveform,
                    resident_sample_rate,
                    original_file,
                )

            output_data = dict(task.data)
            output_data[self.diar_segments_key] = segments
            if self.num_speakers_key is not None:
                output_data[self.num_speakers_key] = _count_distinct_speakers(segments)

            return AudioTask(
                dataset_name=task.dataset_name,
                filepath_key=task.filepath_key or self.filepath_key,
                data=output_data,
                _metadata=dict(task._metadata or {}),
                _stage_perf=list(task._stage_perf),
            )
        finally:
            cleanup_temp_files(temp_paths)
