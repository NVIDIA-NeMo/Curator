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
from typing import TYPE_CHECKING, Any

from huggingface_hub import snapshot_download
from loguru import logger
from nemo.collections.asr.models import SortformerEncLabelModel

from nemo_curator.backends.utils import RayStageSpecKeys
from nemo_curator.stages.audio._agent_ready import AgentReady, Gates, IOSpec, StageContract
from nemo_curator.stages.audio._residency import InputResidency, cleanup_temp_files, residency_read_specs, resolve_audio_path
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
    waveform_key: str = "waveform"
    sample_rate_key: str = "sample_rate"
    diar_segments_key: str = "diar_segments"
    input_residency: InputResidency = "file"
    fanout: bool = False
    start_key: str = "start"
    end_key: str = "end"
    start_ms_key: str = "start_ms"
    end_ms_key: str = "end_ms"
    duration_key: str = "duration"
    segment_num_key: str = "segment_num"
    speaker_key: str = "speaker"
    original_file_key: str = "original_file"
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

    def setup_on_node(
        self, _node_info: NodeInfo | None = None, _worker_metadata: WorkerMetadata | None = None
    ) -> None:
        """Pre-download model weights on the node so workers load from cache."""
        if self.model_path is not None:
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
        return [], [self.filepath_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        if self.fanout:
            return ["data"], [
                self.filepath_key,
                self.start_key,
                self.end_key,
                self.start_ms_key,
                self.end_ms_key,
                self.duration_key,
                self.segment_num_key,
                self.speaker_key,
                self.original_file_key,
            ]
        return ["data"], [self.filepath_key, self.diar_segments_key]

    def describe(self) -> StageContract:
        if self.fanout:
            writes = [
                self.filepath_key,
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
            cardinality = "1:1"
        return StageContract(
            reads_one_of=residency_read_specs(
                self.input_residency,
                audio_filepath_key=self.filepath_key,
                waveform_key=self.waveform_key,
                sample_rate_key=self.sample_rate_key,
            ),
            writes=IOSpec(data_keys=writes),
            cardinality=cardinality,
            cardinality_options=["passthrough", "fan_out"],
            iteration_key=self.diar_segments_key if self.fanout else None,
            gates=Gates(
                requires_gpu=True,
                writes_to_disk=self.rttm_out_dir is not None,
                requires_internet_first_run=self.model_path is None,
            ),
        )

    def ray_stage_spec(self) -> dict[str, Any]:
        if self.fanout:
            return {RayStageSpecKeys.IS_FANOUT_STAGE: True}
        return {}

    def _segment_child_data(
        self,
        item: dict[str, Any],
        segment: dict[str, Any],
        segment_num: int,
        file_path: str,
        file_path_is_temp: bool = False,
    ) -> dict[str, Any]:
        child = {k: v for k, v in item.items() if k != self.diar_segments_key}
        child.update({k: v for k, v in segment.items() if k not in {"start", "end", "speaker"}})
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", start))
        child[self.start_key] = start
        child[self.end_key] = end
        child[self.start_ms_key] = int(round(start * 1000))
        child[self.end_ms_key] = int(round(end * 1000))
        child[self.duration_key] = max(0.0, end - start)
        child[self.segment_num_key] = segment_num
        if "speaker" in segment:
            child[self.speaker_key] = segment["speaker"]
        # Never pin a fan-out child to a materialized temp WAV: process()'s
        # `finally` deletes it, which would leave every child referencing a
        # missing file. Only carry a real (non-temp) source path forward; when
        # only an in-memory waveform exists, children rely on the copied waveform.
        if not file_path_is_temp:
            child.setdefault(self.filepath_key, file_path)
        original_file = (
            item.get(self.original_file_key)
            or item.get(self.filepath_key)
            or item.get("audio_filepath")
            or (None if file_path_is_temp else file_path)
        )
        if original_file is not None:
            child.setdefault(self.original_file_key, original_file)
        return child

    def _fanout_segments(
        self,
        task: AudioTask,
        segments: list[dict[str, Any]],
        file_path: str,
        file_path_is_temp: bool = False,
    ) -> list[AudioTask]:
        return [
            AudioTask(
                dataset_name=task.dataset_name,
                filepath_key=task.filepath_key or self.filepath_key,
                data=self._segment_child_data(task.data, segment, index, file_path, file_path_is_temp),
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
        has_file = self.filepath_key in task.data
        has_waveform = self.waveform_key in task.data and self.sample_rate_key in task.data
        if not (has_file or has_waveform):
            msg = f"Task {task!s} failed validation for stage {self}"
            raise ValueError(msg)

        temp_paths: list[str] = []
        file_path = resolve_audio_path(
            task.data,
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
            sess_name = task.data.get("session_name")
            resolved_sess_name = (
                sess_name if sess_name is not None else os.path.splitext(os.path.basename(file_path))[0]
            )

            all_segments = self.diarize([file_path])
            segments = all_segments[0]

            if self.rttm_out_dir is not None:
                _write_rttm(segments, resolved_sess_name, self.rttm_out_dir)

            if self.fanout:
                return self._fanout_segments(task, segments, file_path, file_path in temp_paths)

            output_data = dict(task.data)
            output_data[self.diar_segments_key] = segments

            return AudioTask(
                dataset_name=task.dataset_name,
                filepath_key=task.filepath_key or self.filepath_key,
                data=output_data,
                _metadata=dict(task._metadata or {}),
                _stage_perf=list(task._stage_perf),
            )
        finally:
            cleanup_temp_files(temp_paths)
