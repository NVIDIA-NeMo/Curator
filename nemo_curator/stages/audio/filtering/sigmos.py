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
or audio_filepath to a WAV file. Uses predict_audio_mos in-memory only;
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

from nemo_curator.stages.audio.filtering.sigmos_filter_module.sigmos_pipeline import predict_audio_mos
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioBatch

from nemo_curator.stages.audio.common import resolve_model_path


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
class SIGMOSFilterStage(ProcessingStage[AudioBatch, AudioBatch]):
    """
    SIGMOS quality assessment filter stage.

    Filters audio segments based on SIGMOS quality metrics.
    Input: items with waveform + sample_rate (tensor/array) or audio_filepath (WAV).
    Uses in-memory predict_audio_mos only; no temp files.

    SIGMOS predicts (1-5 scale):
    - NOISE: Background noise level (higher = less noisy)
    - OVRL: Overall quality
    - SIG: Signal quality
    - COL: Coloration
    - DISC: Discontinuity
    - LOUD: Loudness
    - REVERB: Reverberation (higher = less reverb)

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
        self._predict_audio_mos = None

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [
            "sigmos_noise", "sigmos_ovrl", "sigmos_sig", "sigmos_col",
            "sigmos_disc", "sigmos_loud", "sigmos_reverb",
        ]

    def setup(self, worker_metadata: Any = None) -> None:  # noqa: ARG002, ANN401
        from nemo_curator.utils.gpu_utils import ensure_cudnn_loaded
        ensure_cudnn_loaded()
        self._ensure_predict()

    def teardown(self) -> None:
        self._predict_audio_mos = None
        torch.cuda.empty_cache()

    def _ensure_predict(self) -> None:
        if self._predict_audio_mos is None:
            self._predict_audio_mos = predict_audio_mos
            logger.info("SIGMOS predict_audio_mos loaded successfully")

    def _resolve_model_path(self) -> str:
        return resolve_model_path(self.model_path, __file__, "sigmos_filter_module")

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
            "noise": 0.0, "sig": 0.0, "col": 0.0, "disc": 0.0, "loud": 0.0, "reverb": 0.0,
            "ovrl": float(score_data),
        }

    def _check_thresholds(self, noise: float, ovrl: float, sig: float, col: float,
                          disc: float, loud: float, reverb: float) -> tuple[bool, list[str]]:
        passed = True
        fail_reasons = []
        if self.noise_threshold is not None and noise < self.noise_threshold:
            passed = False
            fail_reasons.append(f"NOISE {noise:.3f} < {self.noise_threshold}")
        if self.ovrl_threshold is not None and ovrl < self.ovrl_threshold:
            passed = False
            fail_reasons.append(f"OVRL {ovrl:.3f} < {self.ovrl_threshold}")
        if self.sig_threshold is not None and sig < self.sig_threshold:
            passed = False
            fail_reasons.append(f"SIG {sig:.3f} < {self.sig_threshold}")
        if self.col_threshold is not None and col < self.col_threshold:
            passed = False
            fail_reasons.append(f"COL {col:.3f} < {self.col_threshold}")
        if self.disc_threshold is not None and disc < self.disc_threshold:
            passed = False
            fail_reasons.append(f"DISC {disc:.3f} < {self.disc_threshold}")
        if self.loud_threshold is not None and loud < self.loud_threshold:
            passed = False
            fail_reasons.append(f"LOUD {loud:.3f} < {self.loud_threshold}")
        if self.reverb_threshold is not None and reverb < self.reverb_threshold:
            passed = False
            fail_reasons.append(f"REVERB {reverb:.3f} < {self.reverb_threshold}")
        return passed, fail_reasons

    def _process_single_item(self, item: dict[str, Any], task_id: str) -> dict[str, Any] | None:
        audio_result = _get_audio_numpy_sr(item, task_id)
        if audio_result is None:
            return None
        audio_np, sample_rate = audio_result

        self._ensure_predict()
        if self._predict_audio_mos is None:
            return None

        try:
            score_data = self._predict_audio_mos(audio_np, sample_rate, model_path=self._resolve_model_path())
        except Exception as e:  # noqa: BLE001
            logger.exception(f"[{task_id}] SIGMOS prediction error: {e}")
            return None

        s = self._scores_from_prediction(score_data)
        passed, fail_reasons = self._check_thresholds(
            s["noise"], s["ovrl"], s["sig"], s["col"], s["disc"], s["loud"], s["reverb"]
        )

        logger.debug(
            f"[{task_id}] SIGMOS NOISE={s['noise']:.3f}, OVRL={s['ovrl']:.3f}, SIG={s['sig']:.3f}, "
            f"COL={s['col']:.3f}, DISC={s['disc']:.3f}, LOUD={s['loud']:.3f}, REVERB={s['reverb']:.3f}"
        )
        if not passed:
            logger.info(f"[{task_id}] SIGMOS FAILED: {', '.join(fail_reasons)}")
            return None

        item["sigmos_noise"] = s["noise"]
        item["sigmos_ovrl"] = s["ovrl"]
        item["sigmos_sig"] = s["sig"]
        item["sigmos_col"] = s["col"]
        item["sigmos_disc"] = s["disc"]
        item["sigmos_loud"] = s["loud"]
        item["sigmos_reverb"] = s["reverb"]
        return item

    def process(self, task: AudioBatch) -> AudioBatch | None:
        self._ensure_predict()
        if self._predict_audio_mos is None:
            logger.error("SIGMOS prediction not available")
            return AudioBatch(
                data=[],
                task_id=task.task_id,
                dataset_name=task.dataset_name,
                _metadata=task._metadata,
                _stage_perf=list(task._stage_perf),
            )

        results = []
        for item in task.data:
            out = self._process_single_item(item, task.task_id)
            if out is not None:
                results.append(out)

        total = len(task.data)
        threshold_parts = []
        if self.noise_threshold is not None:
            threshold_parts.append(f"NOISE>={self.noise_threshold}")
        if self.ovrl_threshold is not None:
            threshold_parts.append(f"OVRL>={self.ovrl_threshold}")
        if self.sig_threshold is not None:
            threshold_parts.append(f"SIG>={self.sig_threshold}")
        if self.col_threshold is not None:
            threshold_parts.append(f"COL>={self.col_threshold}")
        if self.disc_threshold is not None:
            threshold_parts.append(f"DISC>={self.disc_threshold}")
        if self.loud_threshold is not None:
            threshold_parts.append(f"LOUD>={self.loud_threshold}")
        if self.reverb_threshold is not None:
            threshold_parts.append(f"REVERB>={self.reverb_threshold}")
        threshold_str = ", ".join(threshold_parts) if threshold_parts else "none"
        logger.info(f"[SIGMOSFilter] {task.task_id}: {len(results)}/{total} passed (thresholds: {threshold_str})")

        return AudioBatch(
            data=results,
            task_id=task.task_id,
            dataset_name=task.dataset_name,
            _metadata=task._metadata,
            _stage_perf=list(task._stage_perf),
        )
