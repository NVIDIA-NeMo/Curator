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

"""
SIGMOS (Signal-based Mean Opinion Score) filter stage.

Filters audio segments based on SIGMOS quality metrics including
noise, overall quality, signal quality, coloration, discontinuity,
loudness, and reverberation.

Accepts a single input format: either in-memory (waveform + sample_rate)
or audio_filepath to a WAV file. Uses the SigMOS ONNX model directly;
no temp files.

Example:
    from nemo_curator.pipeline import Pipeline
    from nemo_curator.stages.audio.filtering import SIGMOSFilterStage
    from nemo_curator.stages.resources import Resources

    pipeline = Pipeline(name="quality_pipeline")
    pipeline.add_stage(
        SIGMOSFilterStage(noise_threshold=4.0, ovrl_threshold=3.5)
        .with_(resources=Resources(cpus=1.0, gpus=0.5))
    )
"""

import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from loguru import logger

from nemo_curator.backends.base import WorkerMetadata
from nemo_curator.stages.audio.common import resolve_model_path
from nemo_curator.stages.audio.filtering.sigmos_filter_module.third_party.sigmos.sigmos import build_sigmos_model
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask


def _get_audio_numpy_sr(item: dict[str, Any], task_id: str) -> tuple[np.ndarray, int] | None:
    """
    Get (audio mono float32 numpy, sample_rate) from item.

    Supports:
      - waveform (torch.Tensor or np.ndarray) + sample_rate (int)
      - audio_filepath (str) to WAV: loaded with librosa, mono.

    Returns None if unavailable or load fails.
    """
    waveform = item.get("waveform")
    sample_rate = item.get("sample_rate")

    if waveform is not None and sample_rate is not None:
        audio = waveform.cpu().numpy() if torch.is_tensor(waveform) else np.asarray(waveform, dtype=np.float32)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0)
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        return audio, int(sample_rate)

    path = item.get("audio_filepath")
    if path and os.path.isfile(path):
        try:
            import librosa

            audio, sr = librosa.load(path, sr=None, mono=True)
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            return audio, int(sr)
        except Exception as e:  # noqa: BLE001
            logger.error(f"[{task_id}] Failed to load audio file: {e}")
            return None

    logger.warning(f"[{task_id}] No waveform+sample_rate or valid audio_filepath found")
    return None


@dataclass
class SIGMOSFilterStage(ProcessingStage[AudioTask, AudioTask]):
    """
    SIGMOS quality assessment filter stage.

    Filters audio segments based on SIGMOS quality metrics.
    Input: items with waveform + sample_rate (tensor/array) or audio_filepath (WAV).
    The ONNX model is loaded once in setup() and reused for all predictions.

    Args:
        model_path: Path to SIGMOS ONNX model
        noise_threshold: Minimum noise score (None to disable)
        ovrl_threshold: Minimum overall score (None to disable)
        sig_threshold: Minimum signal score (None to disable)
        col_threshold: Minimum coloration score (None to disable)
        disc_threshold: Minimum discontinuity score (None to disable)
        loud_threshold: Minimum loudness score (None to disable)
        reverb_threshold: Minimum reverb score (None to disable)

    Note:
        GPU assignment is handled by the executor via _resources.
        Use .with_(resources=Resources(gpus=X)) to configure GPU allocation.
    """

    model_path: str = "model/model-sigmos_1697718653_41d092e8-epo-200.onnx"
    noise_threshold: float | None = 4.0
    ovrl_threshold: float | None = 3.5
    sig_threshold: float | None = None
    col_threshold: float | None = None
    disc_threshold: float | None = None
    loud_threshold: float | None = None
    reverb_threshold: float | None = None

    name: str = "SIGMOSFilter"
    batch_size: int = 1
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0, gpus=0.5))

    def __post_init__(self):
        super().__init__()
        self._model = None

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [
            "sigmos_noise",
            "sigmos_ovrl",
            "sigmos_sig",
            "sigmos_col",
            "sigmos_disc",
            "sigmos_loud",
            "sigmos_reverb",
        ]

    def setup(self, _: WorkerMetadata | None = None) -> None:
        from nemo_curator.utils.gpu_utils import ensure_cudnn_loaded

        ensure_cudnn_loaded()
        self._initialize_model()

    def teardown(self) -> None:
        self._model = None
        torch.cuda.empty_cache()

    def _resolve_model_path(self) -> str:
        return resolve_model_path(self.model_path, __file__, "sigmos_filter_module")

    def _initialize_model(self) -> None:
        if self._model is not None:
            return

        resolved_path = self._resolve_model_path()
        if torch.cuda.is_available():
            device_id = int(torch.cuda.current_device())
            self._model = build_sigmos_model(
                force_cpu=False,
                device_id=device_id,
                model_path=resolved_path,
            )
        else:
            self._model = build_sigmos_model(
                force_cpu=True,
                model_path=resolved_path,
            )
        logger.info("SIGMOS model loaded successfully")

    def _scores_from_prediction(self, score_data: Any) -> dict[str, float]:  # noqa: ANN401
        if isinstance(score_data, dict):
            return {
                "noise": float(score_data.get("MOS_NOISE", 0)),
                "ovrl": float(score_data.get("MOS_OVRL", 0)),
                "sig": float(score_data.get("MOS_SIG", 0)),
                "col": float(score_data.get("MOS_COL", 0)),
                "disc": float(score_data.get("MOS_DISC", 0)),
                "loud": float(score_data.get("MOS_LOUD", 0)),
                "reverb": float(score_data.get("MOS_REVERB", 0)),
            }
        return {
            "noise": 0.0,
            "sig": 0.0,
            "col": 0.0,
            "disc": 0.0,
            "loud": 0.0,
            "reverb": 0.0,
            "ovrl": float(score_data),
        }

    def _check_thresholds(self, scores: dict[str, float]) -> tuple[bool, list[str]]:
        checks = [
            ("noise", self.noise_threshold, "NOISE"),
            ("ovrl", self.ovrl_threshold, "OVRL"),
            ("sig", self.sig_threshold, "SIG"),
            ("col", self.col_threshold, "COL"),
            ("disc", self.disc_threshold, "DISC"),
            ("loud", self.loud_threshold, "LOUD"),
            ("reverb", self.reverb_threshold, "REVERB"),
        ]
        passed = True
        fail_reasons = []
        for key, threshold, label in checks:
            if threshold is not None and scores[key] < threshold:
                passed = False
                fail_reasons.append(f"{label} {scores[key]:.3f} < {threshold}")
        return passed, fail_reasons

    def process(self, task: AudioTask) -> AudioTask | list[AudioTask]:
        """Process a single AudioTask and filter by SIGMOS quality metrics.

        When ``task.data`` contains a ``"segments"`` key (nested mode from VAD),
        each segment is evaluated individually and only survivors are kept.
        """
        if "segments" in task.data:
            survivors = []
            for seg in task.data["segments"]:
                temp = AudioTask(data=seg, task_id=task.task_id)
                result = self._process_single(temp)
                if result is not None:
                    survivors.append(temp.data)
            task.data["segments"] = survivors
            return task if survivors else []
        return self._process_single(task) or []

    def _process_single(self, task: AudioTask) -> AudioTask | None:
        """Run SIGMOS scoring on a single (non-nested) task."""
        audio_result = _get_audio_numpy_sr(task.data, task.task_id)
        if audio_result is None:
            return None
        audio_np, sample_rate = audio_result

        if self._model is None:
            return None

        try:
            score_data = self._model.run(audio=audio_np, sr=sample_rate)
        except Exception as e:  # noqa: BLE001
            logger.exception(f"[{task.task_id}] SIGMOS prediction error: {e}")
            return None

        s = self._scores_from_prediction(score_data)
        passed, fail_reasons = self._check_thresholds(s)

        logger.debug(
            f"[{task.task_id}] SIGMOS NOISE={s['noise']:.3f}, OVRL={s['ovrl']:.3f}, SIG={s['sig']:.3f}, "
            f"COL={s['col']:.3f}, DISC={s['disc']:.3f}, LOUD={s['loud']:.3f}, REVERB={s['reverb']:.3f}"
        )
        if not passed:
            logger.info(f"[{task.task_id}] SIGMOS FAILED: {', '.join(fail_reasons)}")
            return None

        task.data["sigmos_noise"] = s["noise"]
        task.data["sigmos_ovrl"] = s["ovrl"]
        task.data["sigmos_sig"] = s["sig"]
        task.data["sigmos_col"] = s["col"]
        task.data["sigmos_disc"] = s["disc"]
        task.data["sigmos_loud"] = s["loud"]
        task.data["sigmos_reverb"] = s["reverb"]
        return task
