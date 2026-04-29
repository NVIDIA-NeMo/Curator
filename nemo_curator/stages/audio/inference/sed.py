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
per-frame class probabilities (T x 527 AudioSet classes) are saved as a
compressed NPZ sidecar file.  Downstream stages (e.g. SEDPostprocessingStage)
read these NPZ files to extract clean-speech timestamps.

Requires: ``pip install torchlibrosa librosa``
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
    """Run Sound Event Detection on each audio file and save framewise NPZ output.

    The model produces a ``(T, classes_num)`` probability matrix for each audio
    file (default 527 AudioSet classes at ~50 fps for 16 kHz / hop_size 320).
    Results are saved as compressed NPZ files under ``output_dir/framewise/``
    and the ``npz_filepath`` key is added to the output task data.

    Args:
        checkpoint_path: Path to the PANNs ``.pth`` checkpoint file.
        model_type: CNN14 variant name (see ``sed_models.MODEL_REGISTRY``).
        sample_rate: Audio resampling rate. Defaults to 16000.
        window_size: STFT window size. Defaults to 1024.
        hop_size: STFT hop size. Defaults to 320.
        mel_bins: Number of mel filter banks. Defaults to 64.
        fmin: Minimum frequency for mel filters. Defaults to 50.
        fmax: Maximum frequency for mel filters. Defaults to 14000.
        classes_num: Number of AudioSet classes. Defaults to 527.
        output_dir: Directory to write NPZ sidecar files. Defaults to ``"sed_output"``.
        framewise_dtype: NPZ float dtype. Defaults to ``"float16"``.
        pad_short_segments: Zero-pad audio shorter than minimum model input. Defaults to True.
        filepath_key: Key in task data for the audio path. Defaults to ``"audio_filepath"``.
        save_npz: Write framewise probabilities to per-utterance NPZ sidecars under
            ``output_dir/framewise/``.  Defaults to ``True`` to preserve the existing
            dev-branch behaviour.  Set to ``False`` to skip disk I/O and pass the
            framewise tensor in-memory to ``SEDPostprocessingStage`` (under the
            ``sed_framewise`` key) — useful for streaming AIS pipelines where the
            framewise array isn't needed after postprocessing.
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
    output_dir: str = "sed_output"
    framewise_dtype: str = "float16"
    pad_short_segments: bool = True
    filepath_key: str = "audio_filepath"
    save_npz: bool = True

    name: str = "SEDInference"
    batch_size: int = 1
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0, gpu_memory_gb=4.0))

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
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
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
        keys = [self.filepath_key, "sed_valid_frames", "sed_fps"]
        if self.save_npz:
            keys.append("npz_filepath")
        else:
            keys.append("sed_framewise")
        return ["data"], keys

    def process(self, task):
        """Run SED on one audio file, save NPZ, return enriched task.

        Handles both AudioTask (dict) and DocumentBatch (DataFrame) inputs.
        """
        import numpy as np
        import pandas as pd
        import torch

        from nemo_curator.tasks import DocumentBatch

        # --- Handle DocumentBatch (from JsonlReader) ---
        if isinstance(task, DocumentBatch):
            df = task.to_pandas() if hasattr(task, "to_pandas") else task.data
            results = []
            for _, row in df.iterrows():
                audio_path = str(row.get(self.filepath_key, ""))
                if not audio_path:
                    results.append(row.to_dict())
                    continue
                framewise, valid_frames, fps, npz_path = self._run_sed_on_file(audio_path)
                r = row.to_dict()
                if self.save_npz:
                    r["npz_filepath"] = npz_path
                else:
                    r["sed_framewise"] = framewise
                r["sed_valid_frames"] = valid_frames
                r["sed_fps"] = fps
                results.append(r)
            out_df = pd.DataFrame(results)
            return DocumentBatch(data=out_df, dataset_name=task.dataset_name, task_id=task.task_id)

        # --- Handle AudioTask ---
        # Prefer in-memory waveform (AIS-streamed pipeline) so we never
        # need a shared filesystem.  Falls back to audio_filepath for the
        # legacy file-based path.
        from nemo_curator.stages.audio.utils.audio_io import ensure_waveform

        if "waveform" in task.data:
            waveform = ensure_waveform(task, target_sr=self.sample_rate)
            audio_path = str(task.data.get(self.filepath_key, "") or task.task_id)
        else:
            audio_path = str(task.data.get(self.filepath_key, ""))
            if not audio_path:
                msg = f"Missing {self.filepath_key} in task data"
                raise ValueError(msg)
            waveform = None  # _run_sed_on_audio will load from path

        framewise, valid_frames, fps, npz_path = self._run_sed_on_audio(
            audio_path=audio_path, preloaded_waveform=waveform
        )
        output_data = dict(task.data)
        if self.save_npz:
            output_data["npz_filepath"] = npz_path
        else:
            output_data["sed_framewise"] = framewise
        output_data["sed_valid_frames"] = int(valid_frames)
        output_data["sed_fps"] = float(fps)

        return AudioTask(
            task_id=f"{task.task_id}_sed",
            dataset_name=task.dataset_name,
            filepath_key=task.filepath_key or self.filepath_key,
            data=output_data,
            _metadata=task._metadata,
            _stage_perf=task._stage_perf,
        )

    def _run_sed_on_file(self, audio_path: str) -> "tuple[np.ndarray, int, float, str]":
        """Backwards-compatible wrapper that loads from ``audio_path``."""
        return self._run_sed_on_audio(audio_path=audio_path, preloaded_waveform=None)

    def _run_sed_on_audio(
        self,
        audio_path: str,
        preloaded_waveform: "np.ndarray | None",
    ) -> "tuple[np.ndarray, int, float, str]":
        """Core SED logic: get audio (preloaded or via librosa), run model, save NPZ.

        Returns ``(framewise, valid_frames, fps, npz_path)``.  ``npz_path``
        is ``""`` when ``save_npz=False``.
        """
        import numpy as np
        import torch

        if preloaded_waveform is not None:
            waveform = preloaded_waveform
        else:
            import librosa

            waveform, _ = librosa.core.load(audio_path, sr=self.sample_rate, mono=True)
        original_samples = waveform.shape[0]

        min_input = max(self.window_size, self.hop_size * 32)
        was_padded = False
        if self.pad_short_segments and original_samples < min_input:
            waveform = np.pad(waveform, (0, min_input - original_samples), mode="constant")
            was_padded = True

        x = torch.from_numpy(waveform[None, :]).float().to(self._device)
        with torch.no_grad():
            out = self._model(x, None)

        framewise: np.ndarray = out["framewise_output"].cpu().numpy()[0]
        fps = float(self.sample_rate) / self.hop_size
        valid_frames = min(int(np.ceil(original_samples / self.hop_size)), framewise.shape[0])

        fw = framewise.astype(np.float16 if self.framewise_dtype == "float16" else np.float32)
        npz_path = ""
        if self.save_npz:
            npz_path = self._save_npz(fw, fps, audio_path, original_samples, valid_frames, was_padded)
        return fw, int(valid_frames), fps, npz_path

    def _save_npz(
        self,
        fw: np.ndarray,
        fps: float,
        audio_path: str,
        original_samples: int,
        valid_frames: int,
        was_padded: bool,
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
            was_padded=np.bool_(was_padded),
        )
        return npz_path
