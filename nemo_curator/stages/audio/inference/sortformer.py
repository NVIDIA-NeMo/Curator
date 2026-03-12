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

from nemo_curator.stages.base import ProcessingStage

if TYPE_CHECKING:
    from nemo_curator.backends.base import NodeInfo, WorkerMetadata
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioBatch


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
            f.write(f"SPEAKER {sess_name} 1 {seg['start']:.3f} {duration:.3f} <NA> <NA> {seg['speaker']} <NA> <NA>\n")


@dataclass
class InferenceSortformerStage(ProcessingStage[AudioBatch, AudioBatch]):
    """Speaker diarization inference using Streaming Sortformer (NeMo).

    Uses the NeMo SortformerEncLabelModel for end-to-end neural speaker
    diarization with streaming support. See:
    https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2

    Args:
        model_name: Hugging Face model id. Defaults to "nvidia/diar_streaming_sortformer_4spk-v2".
        model_path: Local path to a .nemo checkpoint file; if set, takes precedence over model_name.
        diar_model: Pre-loaded SortformerEncLabelModel; if provided, setup() is a no-op.
        filepath_key: Key in data for path to audio file. Defaults to "audio_filepath".
        diar_segments_key: Key in output data for diarization segments list. Defaults to "diar_segments".
        rttm_out_dir: Optional directory to write RTTM files. Defaults to None.
        chunk_len: Streaming chunk size in 80 ms frames. Defaults to 340 (~30.4 s latency).
        chunk_right_context: Right context frames. Defaults to 40.
        fifo_len: FIFO queue size in frames. Defaults to 40.
        spkcache_update_period: Speaker cache update period in frames. Defaults to 300.
        spkcache_len: Speaker cache size in frames. Defaults to 188.
        inference_batch_size: Batch size passed to diarize(). Defaults to 1.
        name: Stage name. Defaults to "Sortformer_inference".
    """

    model_name: str = "nvidia/diar_streaming_sortformer_4spk-v2"
    model_path: str | None = None
    diar_model: Any | None = None
    filepath_key: str = "audio_filepath"
    diar_segments_key: str = "diar_segments"
    rttm_out_dir: str | None = None
    chunk_len: int = 340
    chunk_right_context: int = 40
    fifo_len: int = 40
    spkcache_update_period: int = 300
    spkcache_len: int = 188
    inference_batch_size: int = 1
    name: str = "Sortformer_inference"
    batch_size: int = 1
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0, gpu_memory_gb=8.0))

    def setup_on_node(self, _node_info: NodeInfo | None = None, _worker_metadata: WorkerMetadata = None) -> None:
        """Pre-download model weights on the node so actors load from cache."""
        if self.model_path is not None:
            return
        try:
            snapshot_download(repo_id=self.model_name)
        except Exception:  # noqa: BLE001
            logger.info(f"Could not pre-cache {self.model_name}; actors will download on first use")

    def setup(self, _worker_metadata: WorkerMetadata = None) -> None:
        """Load Sortformer model from Hugging Face or a local .nemo file."""
        if self.diar_model is not None:
            self._configure_streaming()
            return
        try:
            from nemo.collections.asr.models import SortformerEncLabelModel
        except ImportError as e:
            msg = (
                "NeMo ASR is required for InferenceSortformerStage. "
                "Install via: pip install 'git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]'"
            )
            raise ImportError(msg) from e

        if self.model_path is not None:
            self.diar_model = SortformerEncLabelModel.restore_from(
                restore_path=self.model_path,
                map_location="cuda",
                strict=False,
            )
        else:
            self.diar_model = SortformerEncLabelModel.from_pretrained(self.model_name)

        self.diar_model.eval()
        self._configure_streaming()

    def _configure_streaming(self) -> None:
        """Apply streaming configuration to the loaded model."""
        sm = self.diar_model.sortformer_modules
        sm.chunk_len = self.chunk_len
        sm.chunk_right_context = self.chunk_right_context
        sm.fifo_len = self.fifo_len
        sm.spkcache_update_period = self.spkcache_update_period
        sm.spkcache_len = self.spkcache_len

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.filepath_key, self.diar_segments_key]

    def diarize(self, audio_paths: list[str]) -> list[list[dict[str, Any]]]:
        """Run Sortformer on a list of audio files.

        Returns a list (one entry per file) of segment lists [{start, end, speaker}].
        """
        predicted_segments = self.diar_model.diarize(
            audio=audio_paths,
            batch_size=self.inference_batch_size,
        )
        return [_parse_sortformer_segments(segs) for segs in predicted_segments]

    def process(self, task: AudioBatch) -> AudioBatch:
        """Run speaker diarization on each audio file in the task."""
        if not self.validate_input(task):
            msg = f"Task {task!s} failed validation for stage {self}"
            raise ValueError(msg)

        files = [item[self.filepath_key] for item in task.data]
        session_names = [item.get("session_name") if isinstance(item, dict) else None for item in task.data]

        all_segments = self.diarize(files)

        audio_items: list[dict[str, Any]] = []
        for i, (file_path, sess_name) in enumerate(zip(files, session_names, strict=True)):
            resolved_sess_name = (
                sess_name if sess_name is not None else os.path.splitext(os.path.basename(file_path))[0]
            )

            segments = all_segments[i]

            if self.rttm_out_dir is not None:
                _write_rttm(segments, resolved_sess_name, self.rttm_out_dir)

            item = dict(task.data[i]) if isinstance(task.data[i], dict) else {}
            item[self.filepath_key] = file_path
            item[self.diar_segments_key] = segments
            audio_items.append(item)

        return AudioBatch(
            task_id=f"{task.task_id}_sortformer",
            dataset_name=task.dataset_name,
            filepath_key=getattr(task, "filepath_key", self.filepath_key),
            data=audio_items,
            _metadata=task._metadata,
            _stage_perf=task._stage_perf,
        )
