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

"""Generic sound-event-detection stage with a pluggable model adapter.

The stage owns Curator task I/O, audio loading, mono conversion, resampling,
resume behavior, and output storage. The adapter selected by ``adapter_target``
owns model construction, checkpoint loading, batch padding, and inference.

The included YAML reads a JSONL manifest whose rows contain
``{"audio_filepath": "/absolute/path/to/audio.wav"}`` and writes compressed
framewise NPZ files. Install the optional dependency and run it from the
Curator repository root with the full command below::

    uv run --extra audio_sed python nemo_curator/config/run.py \\
      --config-path ../../tutorials/audio/sed \\
      --config-name pipeline \\
      manifest_path=/absolute/path/to/input.jsonl \\
      'checkpoint_path=/absolute/path/to/Cnn14_DecisionLevelMax_mAP\\=0.385.pth' \\
      output_dir=/absolute/path/to/sed_output

The example configuration is ``tutorials/audio/sed/pipeline.yaml``. Its PANNs
adapter produces a ``(frames, 527)`` AudioSet probability matrix per task at
approximately 100 fps. ``sed_valid_frames`` excludes any model padding.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import hydra.utils
import numpy as np
import soundfile
from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask
from nemo_curator.utils.hash_utils import get_deterministic_hash

if TYPE_CHECKING:
    from nemo_curator.backends.base import WorkerMetadata
    from nemo_curator.models.sed.base import SEDAdapter, SEDResult


_CHANNEL_FIRST_DIMENSIONS = 2


@dataclass
class SEDInferenceStage(ProcessingStage[AudioTask, AudioTask]):
    """Run sound-event detection through a YAML-selectable adapter.

    Set ``waveform_key`` to consume in-memory audio plus ``sample_rate_key``.
    Leave it as ``None`` to load ``audio_filepath_key`` inside the worker. Both
    paths are normalized to contiguous mono float32 audio at ``sample_rate``
    before the adapter is called.

    The stage writes a frame-by-class matrix, its frames-per-second value, and
    the number of valid leading frames. It can additionally write an NPZ
    sidecar while retaining the in-memory matrix for a downstream consumer.

    Args:
        adapter_target: Import path of a class implementing ``SEDAdapter``.
        checkpoint_path: Model checkpoint passed unchanged to the adapter.
        sample_rate: Sample rate supplied to the adapter after resampling.
        waveform_key: In-memory waveform field, or ``None`` to load files.
        sample_rate_key: Source sample-rate field for in-memory waveforms.
        audio_filepath_key: File field used for loading and NPZ naming.
        adapter_kwargs: Model-specific constructor options. For PANNs these
            include ``model_type``, frontend settings, class count, and padding.
    """

    adapter_target: str
    checkpoint_path: str
    name: str = "SEDInference"

    sample_rate: int = 16000
    waveform_key: str | None = None
    sample_rate_key: str = "sample_rate"
    audio_filepath_key: str = "audio_filepath"
    skip_me_key: str = "_skipme"

    framewise_output_key: str = "_sed_framewise"
    valid_frames_key: str = "sed_valid_frames"
    fps_key: str = "sed_fps"
    npz_filepath_key: str = "npz_filepath"

    save_npz: bool = False
    output_dir: str = "sed_output"
    framewise_dtype: str = "float16"
    skip_if_output_exists: bool = False

    adapter_kwargs: dict[str, Any] = field(default_factory=dict)
    batch_size: int = 32
    num_workers_override: int | None = None
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0, gpu_memory_gb=4.0))

    def __post_init__(self) -> None:
        if not self.adapter_target:
            msg = "SEDInferenceStage.adapter_target must be non-empty"
            raise ValueError(msg)
        if not self.checkpoint_path:
            msg = "SEDInferenceStage.checkpoint_path must be non-empty"
            raise ValueError(msg)
        if not isinstance(self.sample_rate, int) or isinstance(self.sample_rate, bool) or self.sample_rate <= 0:
            msg = f"SEDInferenceStage.sample_rate must be a positive integer, got {self.sample_rate!r}"
            raise ValueError(msg)
        if not isinstance(self.batch_size, int) or isinstance(self.batch_size, bool) or self.batch_size <= 0:
            msg = f"SEDInferenceStage.batch_size must be a positive integer, got {self.batch_size!r}"
            raise ValueError(msg)
        if self.framewise_dtype not in {"float16", "float32"}:
            msg = f"SEDInferenceStage.framewise_dtype must be 'float16' or 'float32', got {self.framewise_dtype!r}"
            raise ValueError(msg)
        self.adapter_kwargs = dict(self.adapter_kwargs)
        self._adapter: SEDAdapter | None = None

    def num_workers(self) -> int | None:
        return self.num_workers_override

    def _adapter_class(self) -> type:
        """Resolve the configured adapter without importing its model eagerly."""
        return hydra.utils.get_class(self.adapter_target)

    def _adapter_gpu_count(self) -> int:
        """Return the physical GPU count represented by the resource request."""
        requested_gpus = float(self.resources.gpus)
        if requested_gpus < 0 or not math.isfinite(requested_gpus):
            msg = f"SEDInferenceStage.resources.gpus must be finite and non-negative, got {requested_gpus}"
            raise ValueError(msg)
        return math.ceil(requested_gpus)

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        """Construct and load the worker-local adapter."""
        if self._adapter is not None:
            return
        adapter_cls = self._adapter_class()
        adapter = adapter_cls(
            checkpoint_path=self.checkpoint_path,
            sample_rate=self.sample_rate,
            **self.adapter_kwargs,
        )
        try:
            adapter.load_model(num_gpus=self._adapter_gpu_count())
        except Exception:
            try:
                adapter.unload_model()
            except Exception as teardown_exc:  # noqa: BLE001
                logger.warning("SED adapter cleanup after setup failure also failed: {}", teardown_exc)
            raise
        self._adapter = adapter
        logger.info("SED adapter ready on worker ({})", self.adapter_target)

    def teardown(self) -> None:
        if self._adapter is not None:
            self._adapter.unload_model()
            self._adapter = None

    def inputs(self) -> tuple[list[str], list[str]]:
        if self.waveform_key:
            return [], [self.waveform_key, self.sample_rate_key]
        return [], [self.audio_filepath_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        output_keys = [self.framewise_output_key, self.valid_frames_key, self.fps_key]
        if self.save_npz:
            output_keys.append(self.npz_filepath_key)
        return [], output_keys

    @staticmethod
    def _load_audio(audio_filepath: str) -> tuple[np.ndarray, int]:
        """Load one file and convert sample-major multichannel data to channel-first."""
        waveform, sample_rate = soundfile.read(audio_filepath, dtype="float32")
        if waveform.ndim == _CHANNEL_FIRST_DIMENSIONS:
            waveform = waveform.T
        return np.ascontiguousarray(waveform, dtype=np.float32), int(sample_rate)

    def _prepare_waveform(self, waveform: object, sample_rate: object) -> np.ndarray:
        """Return contiguous mono float32 samples at the adapter sample rate."""
        from nemo_curator.stages.audio.common import ensure_mono, ensure_waveform_2d

        source_sample_rate = int(sample_rate)
        if source_sample_rate <= 0:
            msg = f"sample rate must be positive, got {source_sample_rate}"
            raise ValueError(msg)

        tensor = ensure_waveform_2d(waveform)
        if tensor.ndim != _CHANNEL_FIRST_DIMENSIONS:
            msg = f"waveform must be 1-D mono or 2-D channel-first audio, got shape {tuple(tensor.shape)}"
            raise ValueError(msg)
        tensor = ensure_mono(tensor)
        prepared = np.ascontiguousarray(tensor.squeeze(0).cpu().numpy(), dtype=np.float32)
        if source_sample_rate != self.sample_rate:
            import librosa

            prepared = librosa.resample(prepared, orig_sr=source_sample_rate, target_sr=self.sample_rate)
            prepared = np.ascontiguousarray(prepared, dtype=np.float32)
        return prepared

    def process(self, task: AudioTask) -> AudioTask:
        """Run one task through the same batch path used by executors."""
        return self.process_batch([task])[0]

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        """Prepare task audio, delegate one batch, and assemble results."""
        # Ray Data presents its object column as a NumPy array. Avoid testing
        # the truth value of the whole batch because multi-element arrays have
        # no unambiguous boolean value.
        if len(tasks) == 0:
            return []

        skip_indices = self._resume_indices(tasks)
        if len(skip_indices) == len(tasks):
            return tasks

        valid_indices, items, audio_paths = self._prepare_items(tasks, skip_indices)
        if not items:
            logger.info("SED batch: all {} tasks skipped", len(tasks))
            return tasks
        if self._adapter is None:
            msg = "SED adapter not initialized - setup() was not called"
            raise RuntimeError(msg)

        results = self._adapter.infer_batch(items)
        if len(results) != len(items):
            msg = f"SED adapter returned {len(results)} results for {len(items)} items (must match 1:1)"
            raise RuntimeError(msg)
        self._write_results(tasks, valid_indices, results, audio_paths)

        skipped_count = len(tasks) - len(valid_indices)
        if skipped_count:
            logger.info("SED batch: skipped {}/{} tasks", skipped_count, len(tasks))
        logger.info("SED batch: generated {} predictions", len(results))
        return tasks

    def _resume_indices(self, tasks: list[AudioTask]) -> set[int]:
        if not self.skip_if_output_exists:
            return set()
        indices = {index for index, task in enumerate(tasks) if self._already_has_output(task)}
        if indices:
            logger.info("SED: reusing existing output for {}/{} tasks", len(indices), len(tasks))
        return indices

    def _prepare_items(
        self,
        tasks: list[AudioTask],
        skip_indices: set[int],
    ) -> tuple[list[int], list[dict[str, Any]], list[str]]:
        """Normalize valid task audio into the canonical adapter item schema."""
        valid_indices: list[int] = []
        items: list[dict[str, Any]] = []
        audio_paths: list[str] = []

        for index, task in enumerate(tasks):
            if index in skip_indices or task.data.get(self.skip_me_key):
                continue
            audio_path = str(task.data.get(self.audio_filepath_key, "") or "")
            if not self.waveform_key and not audio_path:
                logger.warning("SED: task {} is missing {!r}", task.task_id, self.audio_filepath_key)
                continue
            try:
                if self.waveform_key:
                    waveform = task.data[self.waveform_key]
                    source_sample_rate = task.data[self.sample_rate_key]
                else:
                    waveform, source_sample_rate = self._load_audio(audio_path)
                waveform = self._prepare_waveform(waveform, source_sample_rate)
            except Exception as exc:  # noqa: BLE001
                logger.warning("SED: failed to prepare task {} from {}: {}", task.task_id, audio_path, exc)
                continue

            valid_indices.append(index)
            items.append(
                {
                    "waveform": waveform,
                    "sample_rate": self.sample_rate,
                    "task_id": task.task_id,
                }
            )
            audio_paths.append(audio_path)
        return valid_indices, items, audio_paths

    def _write_results(
        self,
        tasks: list[AudioTask],
        valid_indices: list[int],
        results: list[SEDResult],
        audio_paths: list[str],
    ) -> None:
        """Write canonical adapter results back to their original tasks."""
        dtype = np.float16 if self.framewise_dtype == "float16" else np.float32
        for task_index, result, audio_path in zip(valid_indices, results, audio_paths, strict=True):
            task = tasks[task_index]
            framewise = np.asarray(result.framewise_output, dtype=dtype)
            task.data[self.framewise_output_key] = framewise
            task.data[self.valid_frames_key] = int(result.valid_frames)
            task.data[self.fps_key] = float(result.fps)
            if self.save_npz and audio_path:
                task.data[self.npz_filepath_key] = self._save_npz(result, framewise, audio_path)

    def _already_has_output(self, task: AudioTask) -> bool:
        keys = (self.framewise_output_key, self.valid_frames_key, self.fps_key)
        if any(task.data.get(key) is None for key in keys):
            return False
        return not self.save_npz or task.data.get(self.npz_filepath_key) is not None

    def _save_npz(self, result: SEDResult, framewise: np.ndarray, audio_path: str) -> str:
        """Write one deterministic compressed sidecar and return its path."""
        framewise_dir = os.path.join(self.output_dir, "framewise")
        os.makedirs(framewise_dir, exist_ok=True)

        stem = os.path.splitext(os.path.basename(audio_path))[0]
        path_hash = get_deterministic_hash([audio_path])[:8]
        npz_path = os.path.join(framewise_dir, f"{stem}__{path_hash}.npz")
        np.savez_compressed(
            npz_path,
            framewise=framewise,
            fps=np.float32(result.fps),
            audio_filepath=str(audio_path),
            original_num_samples=np.int32(result.original_num_samples),
            valid_frames=np.int32(result.valid_frames),
        )
        return npz_path
