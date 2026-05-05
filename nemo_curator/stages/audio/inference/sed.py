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

"""Sound Event Detection inference stage for Granary v2 audio pipelines."""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

if TYPE_CHECKING:
    import numpy as np

    from nemo_curator.backends.base import WorkerMetadata


@dataclass
class SEDInferenceStage(ProcessingStage[AudioTask, AudioTask]):
    """Run AudioSet-style Sound Event Detection on each audio task.

    The stage consumes in-memory waveforms emitted by ``NemoTarredAudioReader``
    and writes framewise class probabilities to task data. It is optional:
    importing this module does not require ``torchlibrosa``; that dependency is
    only needed when ``setup()`` loads a SED model.
    """

    name: str = "SEDInference"
    checkpoint_path: str = ""
    model_type: str = "Cnn14_DecisionLevelMax"
    sample_rate: int = 16000
    window_size: int = 1024
    hop_size: int = 320
    mel_bins: int = 64
    fmin: int = 50
    fmax: int = 14000
    classes_num: int = 527
    save_npz: bool = False
    output_dir: str = "sed_output"
    framewise_dtype: str = "float16"
    pad_short_segments: bool = True
    waveform_key: str = "waveform"
    sample_rate_key: str = "sample_rate"
    filepath_key: str = "audio_filepath"
    skip_me_key: str = "_skip_me"
    framewise_key: str = "_sed_framewise"
    valid_frames_key: str = "sed_valid_frames"
    fps_key: str = "sed_fps"
    npz_filepath_key: str = "npz_filepath"
    batch_size: int = 32
    num_workers_override: int | None = None
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0, gpu_memory_gb=4.0))

    def __post_init__(self) -> None:
        if self.framewise_dtype not in {"float16", "float32"}:
            msg = "framewise_dtype must be 'float16' or 'float32'"
            raise ValueError(msg)
        self._model: Any = None
        self._device: Any = None

    def num_workers(self) -> int | None:
        return self.num_workers_override

    def xenna_stage_spec(self) -> dict[str, Any]:
        if self.num_workers_override is None:
            return {}
        return {"num_workers": self.num_workers_override}

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        """Load the SED model on the assigned worker."""
        import torch

        from nemo_curator.stages.audio.inference.sed_models.cnn14 import MODEL_REGISTRY

        if not self.checkpoint_path:
            msg = "checkpoint_path is required for SEDInferenceStage"
            raise ValueError(msg)
        if self.model_type not in MODEL_REGISTRY:
            available = ", ".join(sorted(MODEL_REGISTRY))
            msg = f"Unknown model_type={self.model_type!r}. Available: {available}"
            raise ValueError(msg)

        setup_t0 = time.perf_counter()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_cls = MODEL_REGISTRY[self.model_type]
        self._model = model_cls(
            sample_rate=self.sample_rate,
            window_size=self.window_size,
            hop_size=self.hop_size,
            mel_bins=self.mel_bins,
            fmin=self.fmin,
            fmax=self.fmax,
            classes_num=self.classes_num,
        )
        try:
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        except TypeError:
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        self._model.load_state_dict(state_dict)
        self._model.to(self._device)
        self._model.eval()
        logger.info(
            "Loaded {} from {} on {} in {:.3f}s",
            self.model_type,
            self.checkpoint_path,
            self._device,
            time.perf_counter() - setup_t0,
        )

    def teardown(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            self._device = None

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.waveform_key, self.sample_rate_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        out_keys = [self.framewise_key, self.valid_frames_key, self.fps_key]
        if self.save_npz:
            out_keys.append(self.npz_filepath_key)
        return [], out_keys

    def process(self, task: AudioTask) -> AudioTask:
        msg = "SEDInferenceStage only supports process_batch"
        raise NotImplementedError(msg)

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        if len(tasks) == 0:
            return []
        for task in tasks:
            if not self.validate_input(task):
                msg = f"Task {task.task_id} missing required columns for {type(self).__name__}: {self.inputs()}"
                raise ValueError(msg)
        if self._model is None:
            msg = "Model not initialized - setup() was not called"
            raise RuntimeError(msg)

        import numpy as np
        import torch

        preprocess_t0 = time.perf_counter()
        valid_indices, waveforms, original_samples, audio_paths = self._preprocess_waveforms(tasks)
        preprocess_elapsed = time.perf_counter() - preprocess_t0

        if not valid_indices:
            self._log_metrics({
                "utterances_input": float(len(tasks)),
                "utterances_inferred": 0.0,
                "utterances_skipped": float(len(tasks)),
                "audio_duration_s": 0.0,
                "waveform_bytes": 0.0,
                "preprocess_time_s": preprocess_elapsed,
                "inference_time_s": 0.0,
                "postprocess_time_s": 0.0,
            })
            logger.info("SED: skipped all {} tasks (flagged or missing waveform)", len(tasks))
            return tasks

        min_input = max(self.window_size, self.hop_size * 32)
        for i, waveform in enumerate(waveforms):
            if self.pad_short_segments and waveform.shape[0] < min_input:
                waveforms[i] = np.pad(waveform, (0, min_input - waveform.shape[0]), mode="constant")

        max_len = max(waveform.shape[0] for waveform in waveforms)
        padded = np.zeros((len(waveforms), max_len), dtype=np.float32)
        for i, waveform in enumerate(waveforms):
            padded[i, : waveform.shape[0]] = waveform

        inference_t0 = time.perf_counter()
        with torch.no_grad():
            out = self._model(torch.from_numpy(padded).to(self._device), None)
        inference_elapsed = time.perf_counter() - inference_t0

        postprocess_t0 = time.perf_counter()
        all_framewise = out["framewise_output"].detach().cpu().numpy()
        fps = float(self.sample_rate) / float(self.hop_size)
        dtype = np.float16 if self.framewise_dtype == "float16" else np.float32
        for batch_idx, task_idx in enumerate(valid_indices):
            task = tasks[task_idx]
            framewise_i = all_framewise[batch_idx]
            valid_frames = min(int(np.ceil(original_samples[batch_idx] / self.hop_size)), framewise_i.shape[0])
            framewise = framewise_i.astype(dtype, copy=False)
            task.data[self.framewise_key] = framewise
            task.data[self.valid_frames_key] = int(valid_frames)
            task.data[self.fps_key] = fps
            if self.save_npz and audio_paths[batch_idx]:
                task.data[self.npz_filepath_key] = self._save_npz(
                    framewise,
                    fps,
                    audio_paths[batch_idx],
                    original_samples[batch_idx],
                    valid_frames,
                )
        postprocess_elapsed = time.perf_counter() - postprocess_t0

        audio_seconds = sum(float(samples) / float(self.sample_rate) for samples in original_samples)
        waveform_bytes = sum(float(getattr(waveform, "nbytes", 0)) for waveform in waveforms)
        self._log_metrics({
            "utterances_input": float(len(tasks)),
            "utterances_inferred": float(len(valid_indices)),
            "utterances_skipped": float(len(tasks) - len(valid_indices)),
            "audio_duration_s": audio_seconds,
            "waveform_bytes": waveform_bytes,
            "max_padded_samples": float(max_len),
            "preprocess_time_s": preprocess_elapsed,
            "inference_time_s": inference_elapsed,
            "postprocess_time_s": postprocess_elapsed,
        })
        logger.info(
            "SED: processed {}/{} samples (max_samples={}, fps={:.1f})",
            len(valid_indices),
            len(tasks),
            max_len,
            fps,
        )
        return tasks

    def _preprocess_waveforms(self, tasks: list[AudioTask]) -> tuple[list[int], list[np.ndarray], list[int], list[str]]:
        import numpy as np

        valid_indices: list[int] = []
        waveforms: list[np.ndarray] = []
        original_samples: list[int] = []
        audio_paths: list[str] = []

        for i, task in enumerate(tasks):
            if task.data.get(self.skip_me_key):
                continue

            waveform = task.data.get(self.waveform_key)
            if waveform is None:
                logger.warning("Missing {!r} in task {}; skipping SED", self.waveform_key, task.task_id)
                continue

            src_sr = int(task.data.get(self.sample_rate_key, self.sample_rate))
            waveform = np.asarray(waveform, dtype=np.float32)
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=1)
            if src_sr != self.sample_rate:
                try:
                    import librosa
                except ImportError as exc:
                    msg = "librosa is required for SED when input sample_rate differs from the model sample_rate"
                    raise RuntimeError(msg) from exc
                waveform = librosa.resample(waveform, orig_sr=src_sr, target_sr=self.sample_rate)

            valid_indices.append(i)
            waveforms.append(waveform)
            original_samples.append(int(waveform.shape[0]))
            audio_paths.append(str(task.data.get(self.filepath_key, "")))

        return valid_indices, waveforms, original_samples, audio_paths

    def _save_npz(
        self,
        framewise: np.ndarray,
        fps: float,
        audio_path: str,
        original_samples: int,
        valid_frames: int,
    ) -> str:
        import numpy as np

        framewise_dir = os.path.join(self.output_dir, "framewise")
        os.makedirs(framewise_dir, exist_ok=True)

        stem = os.path.splitext(os.path.basename(audio_path))[0]
        digest = hashlib.md5(audio_path.encode("utf-8"), usedforsecurity=False).hexdigest()[:8]
        npz_path = os.path.join(framewise_dir, f"{stem}__{digest}.npz")
        np.savez_compressed(
            npz_path,
            framewise=framewise,
            fps=np.float32(fps),
            audio_filepath=str(audio_path),
            original_num_samples=np.int32(original_samples),
            valid_frames=np.int32(valid_frames),
        )
        return npz_path
