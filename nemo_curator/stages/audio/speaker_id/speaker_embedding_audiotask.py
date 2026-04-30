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
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio
from loguru import logger

from nemo_curator.backends.base import NodeInfo, WorkerMetadata
from nemo_curator.stages.audio.common import resolve_waveform_from_item
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask
from nemo_curator.tasks.audio_task import carry_sample_key, ensure_checkpoint_shard_id


@dataclass
class SpeakerEmbeddingAudioTaskStage(ProcessingStage[AudioTask, AudioTask]):
    """Run a NeMo speaker model over task audio and persist diarized-segment embeddings to NPZ."""

    name: str = "SpeakerEmbeddingAudioTaskStage"
    model_name: str = "nvidia/speakerverification_en_titanet_large"
    cache_dir: str | None = None
    speaker_model: Any | None = field(default=None, repr=False)
    target_sample_rate: int = 16000
    output_dir: str = ""
    diarized_segments_key: str = "diarized_segments"
    audio_filepath_key: str = "audio_filepath"
    output_filepath_key: str = "output_filepath"
    output_embedding_count_key: str = "embedding_count"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0, gpu_memory_gb=4.0))
    batch_size: int = 64

    def __post_init__(self) -> None:
        if not self.model_name and self.speaker_model is None:
            msg = "Either model_name or speaker_model is required."
            raise ValueError(msg)
        if not self.output_dir:
            msg = "output_dir is required for SpeakerEmbeddingAudioTaskStage."
            raise ValueError(msg)

    def setup_on_node(
        self,
        _node_info: NodeInfo | None = None,
        _worker_metadata: WorkerMetadata | None = None,
    ) -> None:
        if self.speaker_model is not None:
            return
        import nemo.collections.asr as nemo_asr

        try:
            kwargs: dict[str, Any] = {"model_name": self.model_name, "return_model_file": True}
            if self.cache_dir is not None:
                kwargs["cache_dir"] = self.cache_dir
            nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(**kwargs)
        except Exception:  # noqa: BLE001
            logger.info(f"Could not pre-cache {self.model_name}; will download on first use")

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        if self.speaker_model is not None:
            if hasattr(self.speaker_model, "eval"):
                self.speaker_model.eval()
            return
        import nemo.collections.asr as nemo_asr

        map_location = "cuda" if torch.cuda.is_available() else "cpu"
        kwargs: dict[str, Any] = {"model_name": self.model_name, "map_location": map_location}
        if self.cache_dir is not None:
            kwargs["cache_dir"] = self.cache_dir
        self.speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(**kwargs)
        self.speaker_model.eval()

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.audio_filepath_key, self.diarized_segments_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["sample_key"], [self.output_filepath_key, self.output_embedding_count_key]

    def process(self, task: AudioTask) -> AudioTask:
        output_data = dict(task.data)
        segments = output_data.get(self.diarized_segments_key)
        if not isinstance(segments, list):
            msg = f"{self.diarized_segments_key} must be a list of diarized segments"
            raise RuntimeError(msg)

        audio_result = resolve_waveform_from_item(output_data, task.task_id)
        if audio_result is None:
            msg = f"Unable to resolve waveform for task {task.task_id}"
            raise RuntimeError(msg)
        waveform, sample_rate = audio_result
        audio = self._prepare_audio_array(waveform, sample_rate)
        segment_payload = self._extract_segment_embeddings(audio, segments)

        output_path = self._build_output_path(task)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(output_path.as_posix(), **segment_payload)

        output_data.pop("waveform", None)
        output_data[self.output_filepath_key] = output_path.as_posix()
        output_data[self.output_embedding_count_key] = int(segment_payload["embeddings"].shape[0])
        return AudioTask(
            task_id=f"{task.task_id}_speaker_embedding",
            dataset_name=task.dataset_name,
            filepath_key=task.filepath_key or self.audio_filepath_key,
            data=output_data,
            sample_key=carry_sample_key(task, data=output_data),
            _metadata=dict(task._metadata),
            _stage_perf=list(task._stage_perf),
        )

    def _prepare_audio_array(self, waveform: torch.Tensor, sample_rate: int) -> np.ndarray:
        mono_waveform = waveform.mean(dim=0) if waveform.shape[0] > 1 else waveform.squeeze(0)
        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)
            mono_waveform = resampler(mono_waveform.unsqueeze(0)).squeeze(0)
        return mono_waveform.detach().cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def _extract_embedding(self, audio: np.ndarray) -> np.ndarray:
        device = getattr(self.speaker_model, "device", torch.device("cpu"))
        signal = torch.tensor(audio[np.newaxis, :], device=device, dtype=torch.float32)
        signal_len = torch.tensor([audio.shape[0]], device=device)
        _, emb = self.speaker_model.forward(input_signal=signal, input_signal_length=signal_len)
        return emb.squeeze().detach().cpu().numpy().astype(np.float32)

    def _extract_segment_embeddings(
        self,
        audio: np.ndarray,
        segments: list[dict[str, Any]],
    ) -> dict[str, np.ndarray]:
        cut_ids: list[str] = []
        speaker_ids: list[str] = []
        start_times: list[np.float32] = []
        end_times: list[np.float32] = []
        embeddings: list[np.ndarray] = []

        for segment_index, segment in enumerate(segments):
            start_time = float(segment["start_time"])
            end_time = float(segment["end_time"])
            speaker_id = str(segment.get("speaker", f"speaker_{segment_index}"))
            start_sample = max(round(start_time * self.target_sample_rate), 0)
            end_sample = min(round(end_time * self.target_sample_rate), audio.shape[0])
            if end_sample <= start_sample:
                logger.warning(
                    "Skipping empty diarized segment {} with start={} end={}",
                    segment_index,
                    start_time,
                    end_time,
                )
                continue
            segment_audio = audio[start_sample:end_sample]
            cut_ids.append(f"{segment_index}:{speaker_id}:{start_time:.3f}:{end_time:.3f}")
            speaker_ids.append(speaker_id)
            start_times.append(np.float32(start_time))
            end_times.append(np.float32(end_time))
            embeddings.append(self._extract_embedding(segment_audio))

        if embeddings:
            embedding_matrix = np.stack(embeddings, axis=0).astype(np.float32)
        else:
            embedding_matrix = np.empty((0, 0), dtype=np.float32)

        return {
            "cut_ids": np.array(cut_ids, dtype=object),
            "speaker_ids": np.array(speaker_ids, dtype=object),
            "start_times": np.array(start_times, dtype=np.float32),
            "end_times": np.array(end_times, dtype=np.float32),
            "embeddings": embedding_matrix,
        }

    def _build_output_path(self, task: AudioTask) -> Path:
        shard_id = ensure_checkpoint_shard_id(task)
        sample_key = carry_sample_key(task, data=task.data)
        return Path(self.output_dir) / shard_id / f"{sample_key}.npz"
