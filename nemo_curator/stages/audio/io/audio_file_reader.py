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

"""Local audio-file reader stage.

``AudioFileReaderStage`` is the raw audio-byte I/O stage for pipelines that
want to hand an in-memory waveform to downstream processors. Remote object
staging is intentionally outside Curator; launchers such as NvLLMOps should
download Swift/S3 objects and rewrite manifests to node-local paths before
Curator starts.
"""

import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import Any

import torch
from loguru import logger

from nemo_curator.backends.base import NodeInfo, WorkerMetadata
from nemo_curator.backends.utils import RayStageSpecKeys
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioTask

from .waveform_utils import audio_item_id_from_path


@dataclass
class AudioFileReaderStage(ProcessingStage[AudioTask, AudioTask]):
    """Read a local audio file or local audio segment and emit a waveform.

    Segment mode is driven by ``segment_start_s`` and ``segment_duration_s`` in
    task data. The stage uses ffmpeg input seeking so a globally planned segment
    can be decoded without loading the full parent audio into memory.
    """

    target_sample_rate: int = 16000
    target_nchannels: int = 1
    audio_filepath_key: str = "audio_filepath"
    duration_key: str = "duration"
    segment_start_key: str = "segment_start_s"
    segment_duration_key: str = "segment_duration_s"
    audio_item_id_key: str = "audio_item_id"
    waveform_key: str = "waveform"
    sample_rate_key: str = "sample_rate"
    num_samples_key: str = "num_samples"
    skip_me_key: str = "_skip_me"
    read_error_key: str = "audio_read_error"
    skip_on_read_error: bool = True
    ray_num_workers: int | None = None
    xenna_num_workers: int | None = None
    xenna_num_workers_per_node: int | None = None
    verbose: bool = False
    name: str = "AudioFileReader"

    def __post_init__(self) -> None:
        if self.target_sample_rate <= 0:
            msg = f"target_sample_rate must be > 0, got {self.target_sample_rate}"
            raise ValueError(msg)
        if self.target_nchannels <= 0:
            msg = f"target_nchannels must be > 0, got {self.target_nchannels}"
            raise ValueError(msg)
        self._validate_optional_positive_int("ray_num_workers", self.ray_num_workers)
        self._validate_optional_positive_int("xenna_num_workers", self.xenna_num_workers)
        self._validate_optional_positive_int("xenna_num_workers_per_node", self.xenna_num_workers_per_node)
        if self.xenna_num_workers is not None and self.xenna_num_workers_per_node is not None:
            msg = (
                "AudioFileReaderStage: set at most one of xenna_num_workers "
                "(cluster-wide) or xenna_num_workers_per_node (per-node)."
            )
            raise ValueError(msg)
        if not isinstance(self.skip_on_read_error, bool):
            msg = f"skip_on_read_error must be bool, got {type(self.skip_on_read_error).__name__}"
            raise TypeError(msg)

    @staticmethod
    def _validate_optional_positive_int(name: str, value: int | None) -> None:
        if value is not None and value <= 0:
            msg = f"{name} must be > 0 when set, got {value}"
            raise ValueError(msg)

    def setup_on_node(
        self, _node_info: NodeInfo | None = None, _worker_metadata: WorkerMetadata | None = None
    ) -> None:
        if not shutil.which("ffmpeg"):
            msg = "AudioFileReaderStage requires 'ffmpeg'. Install with: sudo apt-get install -y ffmpeg"
            raise RuntimeError(msg)

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.audio_filepath_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [
            self.audio_filepath_key,
            self.audio_item_id_key,
            self.duration_key,
            self.segment_start_key,
            self.segment_duration_key,
            self.waveform_key,
            self.sample_rate_key,
            "is_mono",
            self.num_samples_key,
            self.skip_me_key,
            self.read_error_key,
        ]

    def num_workers(self) -> int | None:
        if self.xenna_num_workers_per_node is not None:
            return self.xenna_num_workers
        if self.xenna_num_workers is not None:
            return self.xenna_num_workers
        return self.ray_num_workers

    def ray_stage_spec(self) -> dict[str, Any]:
        if self.num_workers() is None:
            return {}
        return {RayStageSpecKeys.IS_ACTOR_STAGE: True}

    def xenna_stage_spec(self) -> dict[str, Any]:
        spec: dict[str, Any] = {}
        if self.xenna_num_workers_per_node is not None:
            spec["num_workers_per_node"] = self.xenna_num_workers_per_node
        return spec

    def process(self, task: AudioTask) -> AudioTask:
        t0 = time.perf_counter()
        data_entry = task.data
        if self.audio_filepath_key not in data_entry:
            msg = "Absolute audio filepath is required"
            raise ValueError(msg)

        audio_path = str(data_entry[self.audio_filepath_key])
        segment_start_s = self._optional_seconds(data_entry.get(self.segment_start_key), self.segment_start_key)
        segment_duration_s = self._optional_seconds(
            data_entry.get(self.segment_duration_key),
            self.segment_duration_key,
            strictly_positive=True,
        )
        data_entry.setdefault(self.audio_item_id_key, audio_item_id_from_path(audio_path))
        if self._is_remote_path(audio_path):
            msg = (
                "AudioFileReaderStage only accepts local audio paths. "
                "Stage remote Swift/S3 audio with the launcher before Curator starts."
            )
            raise ValueError(msg)

        try:
            waveform, sample_rate = self._load_waveform(
                audio_path,
                segment_start_s=segment_start_s,
                segment_duration_s=segment_duration_s,
            )
        except Exception as exc:
            if not self.skip_on_read_error:
                raise
            logger.warning("Skipping audio row after read failure for {}: {}", audio_path, exc)
            return self._mark_read_error(task, audio_path, exc, time.perf_counter() - t0)

        num_samples = int(waveform.shape[-1])
        duration = num_samples / float(sample_rate)

        if segment_start_s is not None:
            data_entry[self.segment_start_key] = segment_start_s
        if segment_duration_s is not None:
            data_entry[self.segment_duration_key] = duration
        data_entry[self.waveform_key] = waveform
        data_entry[self.sample_rate_key] = sample_rate
        data_entry["is_mono"] = waveform.shape[0] == 1
        data_entry[self.num_samples_key] = num_samples
        data_entry[self.duration_key] = duration

        metrics = {
            "process_time": time.perf_counter() - t0,
            "duration": duration,
            "waveform_bytes": float(waveform.element_size() * waveform.nelement()),
            "audio_file_read": 1.0,
        }
        if segment_start_s is not None:
            metrics["segment_start_s"] = float(segment_start_s)
        if segment_duration_s is not None:
            metrics["segment_duration_s"] = float(duration)
        self._log_metrics(metrics)
        return task

    @staticmethod
    def _optional_seconds(value: object, key: str, *, strictly_positive: bool = False) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool):
            msg = f"{key} must be numeric, got bool"
            raise TypeError(msg)
        try:
            seconds = float(value)
        except (TypeError, ValueError) as exc:
            msg = f"{key} must be numeric, got {value!r}"
            raise TypeError(msg) from exc
        if strictly_positive and seconds <= 0:
            msg = f"{key} must be > 0 when present, got {seconds}"
            raise ValueError(msg)
        if not strictly_positive and seconds < 0:
            msg = f"{key} must be >= 0 when present, got {seconds}"
            raise ValueError(msg)
        return seconds

    def _mark_read_error(
        self,
        task: AudioTask,
        audio_path: str,
        exc: BaseException,
        elapsed_s: float,
    ) -> AudioTask:
        data_entry = task.data
        waveform = torch.empty((self.target_nchannels, 0), dtype=torch.float32)
        data_entry[self.waveform_key] = waveform
        data_entry[self.sample_rate_key] = self.target_sample_rate
        data_entry["is_mono"] = self.target_nchannels == 1
        data_entry[self.num_samples_key] = 0
        data_entry[self.duration_key] = 0.0
        data_entry[self.skip_me_key] = "audio_read_error"
        data_entry[self.read_error_key] = f"{type(exc).__name__}: {exc}"
        self._log_metrics(
            {
                "process_time": elapsed_s,
                "duration": 0.0,
                "waveform_bytes": 0.0,
                "audio_file_read": 0.0,
                "audio_file_read_errors": 1.0,
                "audio_file_skipped": 1.0,
            }
        )
        logger.debug("Marked {} as skipped due to audio read error", audio_path)
        return task

    @staticmethod
    def _is_remote_path(path: str) -> bool:
        return "://" in str(path)

    def _run_ffmpeg(self, cmd: list[str]) -> subprocess.CompletedProcess[bytes]:
        completed = subprocess.run(  # noqa: S603
            cmd,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=False,
        )
        if completed.returncode:
            raise subprocess.CalledProcessError(
                completed.returncode,
                cmd,
                output=completed.stdout,
                stderr=completed.stderr,
            )
        return completed

    def _load_waveform(
        self,
        input_audio_path: str,
        *,
        segment_start_s: float | None = None,
        segment_duration_s: float | None = None,
    ) -> tuple[torch.Tensor, int]:
        if self._is_remote_path(input_audio_path):
            msg = (
                "AudioFileReaderStage only accepts local audio paths. "
                "Stage remote Swift/S3 audio with the launcher before Curator starts."
            )
            raise ValueError(msg)
        if not os.path.exists(input_audio_path):
            raise FileNotFoundError(input_audio_path)

        cmd = ["ffmpeg", "-v", "error"]
        if segment_start_s is not None and segment_start_s > 0:
            cmd.extend(["-ss", self._format_seconds(segment_start_s)])
        cmd.extend(["-i", input_audio_path])
        if segment_duration_s is not None:
            cmd.extend(["-t", self._format_seconds(segment_duration_s)])
        cmd.extend(
            [
                "-ar",
                str(self.target_sample_rate),
                "-ac",
                str(self.target_nchannels),
                "-f",
                "f32le",
                "-acodec",
                "pcm_f32le",
                "pipe:1",
            ]
        )
        try:
            completed = self._run_ffmpeg(cmd)
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode("utf-8", errors="replace") if isinstance(e.stderr, bytes) else e.stderr
            msg = f"Error loading waveform from {input_audio_path}: {stderr or e}"
            raise RuntimeError(msg) from e

        if not completed.stdout:
            msg = f"ffmpeg produced an empty waveform for {input_audio_path}"
            raise RuntimeError(msg)

        samples = torch.frombuffer(completed.stdout, dtype=torch.float32)
        channels = int(self.target_nchannels)
        usable = (samples.numel() // channels) * channels
        if usable != samples.numel():
            samples = samples[:usable]
        waveform = samples.reshape(-1, channels).transpose(0, 1).contiguous()
        return waveform, self.target_sample_rate

    @staticmethod
    def _format_seconds(seconds: float) -> str:
        return f"{seconds:.6f}".rstrip("0").rstrip(".")
