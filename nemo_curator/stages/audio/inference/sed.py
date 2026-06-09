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

"""Sound Event Detection (SED) inference stage using AudioSet-pretrained CNN models.

Wraps PANNs (Pretrained Audio Neural Networks) CNN14 variants as a Curator
ProcessingStage.  Each audio file is processed through the CNN model and the
per-frame class probabilities (T x 527 AudioSet classes) are passed to the
next stage via task data.  Optionally, results can also be saved as compressed
NPZ sidecar files (``save_npz=True``).

Requires: ``pip install torchlibrosa`` (``librosa`` only needed if input sample rate differs from model target)
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

if TYPE_CHECKING:
    import numpy as np
    import torch


@dataclass
class SEDInferenceStage(ProcessingStage[AudioTask, AudioTask]):
    """Run Sound Event Detection on each audio task.

    The model produces a ``(T, classes_num)`` probability matrix for each audio
    (default 527 AudioSet classes at ~50 fps for 16 kHz / hop_size 320).

    Expects each ``AudioTask.data`` to carry:

    - ``waveform``: 1-D mono numpy float32 array (any sample rate)
    - ``sample_rate``: int

    These are produced by ``NemoTarShardReaderStage`` which decodes audio
    in memory.  The stage resamples to the target sample rate internally.

    By default, results are passed to the next stage in-memory via task data
    keys (``_sed_framewise``, ``sed_fps``, ``sed_valid_frames``).  Set
    ``save_npz=True`` to also write compressed NPZ sidecar files.

    Args:
        checkpoint_path: Path to the PANNs ``.pth`` checkpoint file.
        model_type: CNN14 variant name (see ``sed_models.MODEL_REGISTRY``).
        sample_rate: Model target sample rate. Defaults to 16000.
        window_size: STFT window size. Defaults to 1024.
        hop_size: STFT hop size. Defaults to 320.
        mel_bins: Number of mel filter banks. Defaults to 64.
        fmin: Minimum frequency for mel filters. Defaults to 50.
        fmax: Maximum frequency for mel filters. Defaults to 14000.
        classes_num: Number of AudioSet classes. Defaults to 527.
        save_npz: Write NPZ sidecar files to ``output_dir``. Defaults to False.
        output_dir: Directory to write NPZ sidecar files (only used when ``save_npz=True``).
        framewise_dtype: Float dtype for framewise output. Defaults to ``"float16"``.
        pad_short_segments: Zero-pad audio shorter than minimum model input. Defaults to True.
        waveform_key: Key in task data for the waveform array. Defaults to ``"waveform"``.
        sample_rate_key: Key in task data for the sample rate. Defaults to ``"sample_rate"``.
        filepath_key: Key in task data for the audio path (used for NPZ naming). Defaults to ``"audio_filepath"``.
    """

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
    skip_me_key: str = "_skipme"

    skip_if_output_exists: bool = False
    name: str = "SEDInference"
    batch_size: int = 32
    num_workers_override: int | None = None
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0, gpu_memory_gb=4.0))

    def num_workers(self) -> int | None:
        return self.num_workers_override

    def xenna_stage_spec(self) -> dict[str, Any]:
        spec: dict[str, Any] = {}
        if self.num_workers_override is not None:
            spec["num_workers"] = self.num_workers_override
        return spec

    def setup(self, _worker_metadata: Any = None) -> None:
        """Load CNN14 model from checkpoint."""
        import torch

        from nemo_curator.stages.audio.inference.sed_models.cnn14 import MODEL_REGISTRY

        if not self.checkpoint_path:
            msg = "checkpoint_path is required for SEDInferenceStage."
            raise ValueError(msg)

        if self.model_type not in MODEL_REGISTRY:
            available = ", ".join(sorted(MODEL_REGISTRY))
            msg = f"Unknown model_type={self.model_type!r}. Available: {available}"
            raise ValueError(msg)

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
        # Always load to CPU first, then move — avoids CUDA conflicts with vLLM
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        self._model.load_state_dict(checkpoint["model"])
        self._model.to(self._device)
        self._model.eval()
        logger.info(f"Loaded {self.model_type} from {self.checkpoint_path} on {self._device}")

    def teardown(self) -> None:
        if hasattr(self, "_model") and self._model is not None:
            del self._model
            self._model = None

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        out_keys = [self.filepath_key, "_sed_framewise", "sed_valid_frames", "sed_fps"]
        if self.save_npz:
            out_keys.append("npz_filepath")
        return ["data"], out_keys

    def process(self, task: AudioTask) -> AudioTask:
        """Run SED on a single task (delegates to process_batch)."""
        return self.process_batch([task])[0]

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        """Run batched SED inference on the GPU for all tasks at once."""
        if len(tasks) == 0:
            return []

        _skip_indices: set[int] | None = None
        if self.skip_if_output_exists:
            _skip_indices = {i for i, t in enumerate(tasks) if t.data.get("_sed_framewise") is not None}
            if len(_skip_indices) == len(tasks):
                return tasks
            if _skip_indices:
                logger.info(
                    f"SED: {len(_skip_indices)}/{len(tasks)} tasks already have output, processing remaining"
                )

        import numpy as np
        import torch

        valid_indices, waveforms, original_samples_list, audio_paths = self._preprocess_waveforms(
            tasks, skip_indices=_skip_indices
        )

        if not valid_indices:
            logger.info(f"SED batch: all {len(tasks)} tasks skipped (no valid waveforms)")
            return tasks

        min_input = max(self.window_size, self.hop_size * 32)
        for i, wav in enumerate(waveforms):
            if self.pad_short_segments and wav.shape[0] < min_input:
                waveforms[i] = np.pad(wav, (0, min_input - wav.shape[0]), mode="constant")

        max_len = max(w.shape[0] for w in waveforms)
        padded = np.zeros((len(waveforms), max_len), dtype=np.float32)
        for i, w in enumerate(waveforms):
            padded[i, : w.shape[0]] = w

        x = torch.from_numpy(padded).to(self._device)
        with torch.no_grad():
            out = self._model(x, None)

        all_framewise: np.ndarray = out["framewise_output"].cpu().numpy()
        fps = float(self.sample_rate) / self.hop_size
        use_fp16 = self.framewise_dtype == "float16"

        for vi, task_idx in enumerate(valid_indices):
            task = tasks[task_idx]
            orig_samples = original_samples_list[vi]
            framewise_i = all_framewise[vi]
            valid_frames = min(int(np.ceil(orig_samples / self.hop_size)), framewise_i.shape[0])
            fw = framewise_i.astype(np.float16 if use_fp16 else np.float32)

            task.data["_sed_framewise"] = fw
            task.data["sed_valid_frames"] = int(valid_frames)
            task.data["sed_fps"] = fps

            if self.save_npz and audio_paths[vi]:
                task.data["npz_filepath"] = self._save_npz(fw, fps, audio_paths[vi], orig_samples, valid_frames)

        n_skipped = len(tasks) - len(valid_indices)
        if n_skipped:
            logger.info(f"SED batch: skipped {n_skipped}/{len(tasks)} tasks (already flagged or missing waveform)")
        logger.info(f"SED batch: processed {len(valid_indices)} samples (max_len={max_len}, fps={fps:.1f})")
        return tasks

    def _preprocess_waveforms(
        self, tasks: list[AudioTask], skip_indices: set[int] | None = None
    ) -> tuple[list[int], list[np.ndarray], list[int], list[str]]:
        """Extract, mono-mix, and resample waveforms from valid (non-skipped) tasks.

        Returns:
            ``(valid_indices, waveforms, original_samples, audio_paths)`` where
            *valid_indices* maps each waveform back to its position in *tasks*.
        """
        import numpy as np

        valid_indices: list[int] = []
        waveforms: list[np.ndarray] = []
        original_samples: list[int] = []
        audio_paths: list[str] = []

        for i, task in enumerate(tasks):
            if skip_indices and i in skip_indices:
                continue
            if task.data.get(self.skip_me_key):
                continue

            wav = task.data.get(self.waveform_key)
            if wav is None:
                logger.warning(f"Missing {self.waveform_key!r} in task {task.task_id}, skipping SED")
                continue

            src_sr = int(task.data.get(self.sample_rate_key, self.sample_rate))
            wav = np.asarray(wav, dtype=np.float32)
            if wav.ndim > 1:
                wav = wav.mean(axis=1)

            if src_sr != self.sample_rate:
                import librosa

                wav = librosa.resample(wav, orig_sr=src_sr, target_sr=self.sample_rate)

            valid_indices.append(i)
            waveforms.append(wav)
            original_samples.append(wav.shape[0])
            audio_paths.append(task.data.get(self.filepath_key, ""))

        return valid_indices, waveforms, original_samples, audio_paths

    def _save_npz(
        self,
        fw: np.ndarray,
        fps: float,
        audio_path: str,
        original_samples: int,
        valid_frames: int,
    ) -> str:
        import numpy as np

        framewise_dir = os.path.join(self.output_dir, "framewise")
        os.makedirs(framewise_dir, exist_ok=True)

        stem = os.path.splitext(os.path.basename(audio_path))[0]
        h = hashlib.md5(audio_path.encode("utf-8")).hexdigest()[:8]
        npz_path = os.path.join(framewise_dir, f"{stem}__{h}.npz")

        np.savez_compressed(
            npz_path,
            framewise=fw,
            fps=np.float32(fps),
            audio_filepath=str(audio_path),
            original_num_samples=np.int32(original_samples),
            valid_frames=np.int32(valid_frames),
        )
        return npz_path