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

"""TorchSQUIM audio quality metrics stage (PESQ, STOI, SI-SDR)."""

import math
from dataclasses import dataclass, field
from typing import Any

import librosa
import soundfile as sf
import torch
import torchaudio.functional as torchaudio_F  # noqa: N812
from loguru import logger
from torchaudio.pipelines import SQUIM_OBJECTIVE

from nemo_curator.backends.base import NodeInfo, WorkerMetadata
from nemo_curator.stages.audio._agent._agent_ready import AgentReady, ConditionalWrite, Gates, IOSpec, StageContract
from nemo_curator.stages.audio._agent._residency import InputResidency, residency_read_specs
from nemo_curator.stages.audio.common import ensure_mono, ensure_waveform_2d
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask


@dataclass
class TorchSquimQualityMetricsStage(AgentReady, ProcessingStage[AudioTask, AudioTask]):
    """
    Stage that calculates Squim quality metrics for audio files.

    Uses a pre-trained Squim model to calculate audio quality metrics like
    PESQ, STOI, and SI-SDR for each audio segment.

    Args:
        audio_filepath_key: Key for the audio file path in the manifest. Defaults to "resampled_audio_filepath".
        target_sr: Target sample rate for SQUIM model input. Defaults to 16000.
        batch_size: Number of audio tasks to be processed at once. Defaults to 32.
        compute_batch_size: Number of waveforms to process per GPU inference call. Defaults to 32.
        segments_key: Key for the segments in the manifest. Defaults to "segments".
        waveform_key: Key for an in-memory waveform tensor. Defaults to "waveform".
        sample_rate_key: Key for the in-memory waveform sample rate. Defaults to "sample_rate".
        input_residency: Which input to use — "file" (audio_filepath only; default, unchanged),
            "waveform" (in-memory only), or "auto" (waveform first, file fallback).

    Returns:
        The same data as in the input data, but with Squim quality metrics added to each segment.
    """

    audio_filepath_key: str = "resampled_audio_filepath"
    target_sr: int = 16000
    batch_size: int = 32
    compute_batch_size: int = 32
    segments_key: str = "segments"
    metrics_key: str = "metrics"
    waveform_key: str = "waveform"
    sample_rate_key: str = "sample_rate"
    input_residency: InputResidency = "file"

    # Stage metadata
    name: str = "TorchSquimQualityMetrics"
    BATCH_ONLY = True  # process() raises; only process_batch is implemented (agent-discovery hint)
    resources: Resources = field(default_factory=lambda: Resources(gpus=1.0))

    model: Any = field(default=None, repr=False)

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.metrics_key]

    def describe(self) -> StageContract:
        # An audio source (file or in-memory waveform, per input_residency) is required;
        # segments only refine WHERE metrics are attached (optional, read at runtime).
        return StageContract(
            reads_one_of=residency_read_specs(
                self.input_residency,
                audio_filepath_key=self.audio_filepath_key,
                waveform_key=self.waveform_key,
                sample_rate_key=self.sample_rate_key,
            ),
            writes=IOSpec(data_keys=[self.metrics_key], segment_data_keys=[self.metrics_key]),
            conditional_writes=[
                ConditionalWrite(
                    writes=IOSpec(data_keys=[self.metrics_key]),
                    condition=(
                        f"'{self.segments_key}' is absent, top-level audio resolves to a waveform, "
                        "and whole-batch model inference completes"
                    ),
                    value_origin="augments_upstream_same_key",
                ),
                ConditionalWrite(
                    writes=IOSpec(segment_data_keys=[self.metrics_key]),
                    condition=(
                        f"'{self.segments_key}' is present; an individual segment is not marked no-speaker, "
                        "has non-blank text and a positive requested frame span, its waveform is collected, "
                        "and whole-batch model inference completes"
                    ),
                    value_origin="augments_upstream_same_key",
                ),
            ],
            gates=Gates(
                requires_gpu=self.resources.requires_gpu,
                requires_internet_first_run=True,
                # ``_compute_metrics_batched`` zero-pads each batch to its longest member and
                # calls the model with no lengths, so padding reads as silence and a clip's
                # scores depend on which clips sorted beside it. Pass lengths and this becomes True.
                per_row_independent=False,
            ),
        )

    def validate_input(self, task: AudioTask) -> bool:
        """An audio source is required; segments are optional refinement.

        When ``input_residency`` allows it, an in-memory ``waveform_key``+``sample_rate_key``
        satisfies the requirement; otherwise ``audio_filepath_key`` must be present (the
        default, unchanged behavior).
        """
        data = task.data
        has_waveform = data.get(self.waveform_key) is not None and data.get(self.sample_rate_key) is not None
        has_file = self.audio_filepath_key in data
        if self.input_residency == "waveform":
            ok = has_waveform
        elif self.input_residency == "file":
            ok = has_file
        else:  # auto
            ok = has_waveform or has_file
        if ok:
            return True
        logger.error(
            f"Task {task.task_id} missing required audio input for input_residency={self.input_residency!r}: "
            f"need '{self.audio_filepath_key}' or '{self.waveform_key}'+'{self.sample_rate_key}' "
            f"(segments alone are not sufficient — SQUIM loads audio)"
        )
        return False

    @property
    def _device(self) -> str:
        """Derive device from resources configuration."""
        if self.resources.requires_gpu:
            if not torch.cuda.is_available():
                msg = f"[{self.name}] GPU requested via resources but CUDA is not available."
                raise RuntimeError(msg)
            return "cuda"
        return "cpu"

    def setup_on_node(
        self, _node_info: NodeInfo | None = None, _worker_metadata: WorkerMetadata | None = None
    ) -> None:
        """Pre-download SQUIM model weights (cache warming, no GPU allocation)."""
        SQUIM_OBJECTIVE.get_model()

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        """Load model onto the target device. Called once per worker."""
        self.model = SQUIM_OBJECTIVE.get_model()
        if self._device == "cuda":
            self.model = self.model.cuda()
        logger.info(f"[{self.name}] Initialized SQUIM model on {self._device}")

    def _compute_metrics_batched(self, waveforms: list[torch.Tensor]) -> list[tuple[float, float, float]]:
        """Run SQUIM on a list of 1-D waveform tensors in batches."""
        results: list[tuple[float, float, float]] = []
        for i in range(0, len(waveforms), self.compute_batch_size):
            batch = waveforms[i : i + self.compute_batch_size]
            max_len = max(w.shape[0] for w in batch)
            padded = torch.zeros(len(batch), max_len)
            for j, w in enumerate(batch):
                padded[j, : w.shape[0]] = w
            padded = padded.to(self._device)
            with torch.no_grad():
                stoi, pesq, si_sdr = self.model(padded)
            for j in range(len(batch)):
                results.append(
                    (
                        round(pesq[j].item(), 3),
                        round(stoi[j].item(), 3),
                        round(si_sdr[j].item(), 3),
                    )
                )
        return results

    def _resolve_entry_audio(self, data_entry: dict) -> tuple[Any, int]:
        """Return ``(mono_1d_audio_ndarray, sample_rate)`` from a waveform or the file.

        When ``input_residency`` allows it and an in-memory ``waveform_key``+
        ``sample_rate_key`` is present, that waveform is used directly; otherwise the
        audio file is read at its native sample rate (the default, unchanged behavior).
        Per-segment slicing downstream is identical for either source.
        """
        if self.input_residency != "file":
            waveform = data_entry.get(self.waveform_key)
            sr = data_entry.get(self.sample_rate_key)
            if waveform is not None and sr is not None:
                audio = ensure_mono(ensure_waveform_2d(waveform)).squeeze(0)
                return audio.detach().cpu().numpy(), int(sr)
            if self.input_residency == "waveform":
                msg = (
                    f"[{self.name}] Missing '{self.waveform_key}'+'{self.sample_rate_key}' for entry: "
                    f"{data_entry.get('audio_item_id', 'unknown')} (input_residency='waveform')"
                )
                raise ValueError(msg)

        audio_path = data_entry.get(self.audio_filepath_key)
        if not audio_path:
            msg = (
                f"[{self.name}] Missing '{self.audio_filepath_key}' for entry: "
                f"{data_entry.get('audio_item_id', 'unknown')}"
            )
            raise ValueError(msg)
        try:
            info = sf.info(audio_path)
            sr = info.samplerate
        except Exception as ex:
            msg = f"[{self.name}] Failed to read audio info: {audio_path}"
            raise RuntimeError(msg) from ex
        try:
            audio, _ = librosa.load(path=audio_path, sr=sr)
        except Exception as ex:
            msg = f"[{self.name}] Failed to load audio: {audio_path}"
            raise RuntimeError(msg) from ex
        return audio, sr

    def _collect_waveforms_for_entry(self, task_idx: int, data_entry: dict) -> list[tuple[int, int, torch.Tensor]]:
        """Extract valid segment waveforms from a single audio entry.

        Returns a list of (task_idx, segment_idx, waveform) tuples.
        """
        audio, sr = self._resolve_entry_audio(data_entry)
        # Names the entry in the zero-length-segment warning below. Not the path directly:
        # ``_resolve_entry_audio`` also serves resident waveforms, which have no file, and the
        # binding it once shared with this method moved inside it when it was extracted --
        # leaving the warning referring to an ``audio_path`` that no longer existed here.
        source = data_entry.get(self.audio_filepath_key) or data_entry.get("audio_item_id", "unknown")

        collected: list[tuple[int, int, torch.Tensor]] = []
        if self.segments_key in data_entry:
            segments = data_entry[self.segments_key]
            for seg_idx, segment in enumerate(segments):
                if segment.get("speaker") == "no-speaker" or segment.get("text", "").strip() == "":
                    continue

                start = segment.get("start", 0)
                end = segment.get("end", 0)
                start_frame = math.floor(start * sr)
                end_frame = math.floor(end * sr)

                if end_frame - start_frame <= 0:
                    logger.warning(f"[{self.name}] Zero-length segment at {start}-{end}s in {source}, skipping")
                    continue

                y = torch.from_numpy(audio[start_frame:end_frame])
                if sr != self.target_sr:
                    y = torchaudio_F.resample(y.unsqueeze(0), sr, self.target_sr).squeeze(0)

                collected.append((task_idx, seg_idx, y))
        else:
            y = torch.from_numpy(audio)
            if sr != self.target_sr:
                y = torchaudio_F.resample(y.unsqueeze(0), sr, self.target_sr).squeeze(0)
            collected.append((task_idx, -1, y))
        return collected

    def update_metrics(
        self, audio_segment: dict[str, Any], pesq_val: float, stoi_val: float, sisdr_val: float
    ) -> None:
        """Update the metrics for an audio segment."""
        if self.metrics_key not in audio_segment:
            audio_segment[self.metrics_key] = {}
        audio_segment[self.metrics_key]["pesq_squim"] = pesq_val
        audio_segment[self.metrics_key]["stoi_squim"] = stoi_val
        audio_segment[self.metrics_key]["sisdr_squim"] = sisdr_val

    def process(self, task: AudioTask) -> AudioTask:
        """Delegate single-task processing to process_batch."""
        msg = f"[{self.name}] is a GPU/batched inference stage. Use process_batch() instead."
        raise NotImplementedError(msg)

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        """Calculate Squim quality metrics across all tasks using batched GPU inference.

        Collects waveforms from every valid segment across all tasks, sorts them
        by duration so similarly-sized segments are padded together, runs SQUIM
        in batches on GPU, then scatters results back to the originating
        task's segment.
        """
        if len(tasks) == 0:
            return tasks

        # Collect all valid waveforms with their origin (task_idx, segment_idx)
        all_waveform_metadata: list[tuple[int, int, torch.Tensor]] = []
        for task_idx, task in enumerate(tasks):
            all_waveform_metadata.extend(self._collect_waveforms_for_entry(task_idx, task.data))

        if not all_waveform_metadata:
            logger.warning(
                f"[{self.name}] No valid waveforms collected from {len(tasks)} task(s). "
                "All tasks returned without SQUIM metrics."
            )
            return tasks

        # Sort by waveform length so similarly-sized segments share a batch
        sorted_indices = sorted(range(len(all_waveform_metadata)), key=lambda i: all_waveform_metadata[i][2].shape[0])
        sorted_waveforms = [all_waveform_metadata[i][2] for i in sorted_indices]

        try:
            sorted_results = self._compute_metrics_batched(sorted_waveforms)
            for rank, (pesq_val, stoi_val, sisdr_val) in enumerate(sorted_results):
                orig_idx = sorted_indices[rank]
                task_idx, seg_idx, _ = all_waveform_metadata[orig_idx]
                if self.segments_key in tasks[task_idx].data:
                    segment = tasks[task_idx].data[self.segments_key][seg_idx]
                    self.update_metrics(segment, pesq_val, stoi_val, sisdr_val)
                else:
                    self.update_metrics(tasks[task_idx].data, pesq_val, stoi_val, sisdr_val)
        except Exception as e:
            torch.cuda.empty_cache()
            msg = f"[{self.name}] Failed to compute Squim metrics: {e}"
            raise RuntimeError(msg) from e

        return tasks
