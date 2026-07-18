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

The ONNX model is downloaded automatically from Microsoft's SIG-Challenge
repository on first use and cached at ~/.cache/nemo_curator/sigmos_model/.
Users can also provide a pre-downloaded model via the ``model_path`` parameter.

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
from pathlib import Path
from typing import Any, Literal

import numpy as np
import requests
import torch
from loguru import logger

from nemo_curator.backends.base import NodeInfo, WorkerMetadata
from nemo_curator.stages.audio._agent_ready import AgentReady, Gates, IOSpec, StageContract
from nemo_curator.stages.audio._residency import residency_read_specs, resolve_audio
from nemo_curator.stages.audio.common import ensure_mono, ensure_waveform_2d
from nemo_curator.stages.audio.filtering.sigmos_filter_module.third_party.sigmos.sigmos import build_sigmos_model
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

_SIGMOS_MODEL_URL = (
    "https://github.com/microsoft/SIG-Challenge/raw/main/"
    "ICASSP2024/sigmos/model-sigmos_1697718653_41d092e8-epo-200.onnx"
)
_SIGMOS_MODEL_FILENAME = "model-sigmos_1697718653_41d092e8-epo-200.onnx"
_DEFAULT_MODEL_DIR = str(Path.home() / ".cache" / "nemo_curator" / "sigmos_model")
_VALID_MODES = {"task", "segments", "auto"}
_VALID_ACTIONS = {"filter", "annotate"}


def _get_audio_numpy_sr(  # noqa: PLR0913 (complexity accepted: keyword-only residency/key knobs mirror the stage fields)
    item: dict[str, Any],
    task_id: str,
    *,
    input_residency: Literal["file", "waveform", "auto"] = "auto",
    audio_filepath_key: str = "audio_filepath",
    waveform_key: str = "waveform",
    sample_rate_key: str = "sample_rate",
) -> tuple[np.ndarray, int] | None:
    """
    Get (audio mono float32 numpy, sample_rate) from item.

    Supports:
      - waveform (torch.Tensor or np.ndarray) + sample_rate (int)
      - audio_filepath (str): loaded with soundfile, mono.

    Returns None if unavailable or load fails.
    """
    # Thin wrapper over the shared resolver (_residency.resolve_audio): delegate
    # file/waveform resolution, then return a mono float32 1-D numpy array.
    # Audio-file load errors are swallowed and reported as None (matches the
    # original); a waveform without sample_rate falls back to the file path,
    # exactly like resolve_audio.
    try:
        resolved = resolve_audio(
            item,
            residency=input_residency,
            audio_filepath_key=audio_filepath_key,
            waveform_key=waveform_key,
            sample_rate_key=sample_rate_key,
            mono=True,
        )
    except Exception as e:  # noqa: BLE001
        logger.error(f"[{task_id}] Failed to load audio file: {e}")
        return None

    if resolved is None:
        if input_residency == "waveform":
            logger.warning(f"[{task_id}] No {waveform_key}+{sample_rate_key} found")
        else:
            logger.warning(f"[{task_id}] No {waveform_key}+{sample_rate_key} or valid {audio_filepath_key} found")
        return None

    waveform, sample_rate = resolved
    mono = ensure_mono(ensure_waveform_2d(waveform))
    audio = mono.squeeze(0).detach().cpu().numpy().astype(np.float32)
    return audio, int(sample_rate)


@dataclass
class SIGMOSFilterStage(AgentReady, ProcessingStage[AudioTask, AudioTask]):
    """
    SIGMOS quality assessment filter stage.

    Filters audio segments based on SIGMOS quality metrics.
    Input: items with waveform + sample_rate (tensor/array) or audio_filepath (WAV).
    The ONNX model is loaded once in setup() and reused for all predictions.

    The model is automatically downloaded from Microsoft's SIG-Challenge
    GitHub repository on first use and cached at
    ``~/.cache/nemo_curator/sigmos_model/``. To skip downloading, place
    the ONNX file there manually or pass ``model_path`` pointing
    directly to the file.

    Args:
        model_dir: Directory to store the downloaded model weights
            (default: ``~/.cache/nemo_curator/sigmos_model/``).
        model_path: Direct path to a local SIGMOS ONNX model file.
            Overrides model_dir when provided.
        noise_threshold: Minimum noise score (None to disable)
        ovrl_threshold: Minimum overall score (None to disable)
        sig_threshold: Minimum signal score (None to disable)
        col_threshold: Minimum coloration score (None to disable)
        disc_threshold: Minimum discontinuity score (None to disable)
        loud_threshold: Minimum loudness score (None to disable)
        reverb_threshold: Minimum reverb score (None to disable)
        input_residency: Which input to use — "waveform" (in-memory only), "file"
            (audio_filepath only), or "auto" (waveform first, file fallback; default).
        mode: Where to score — "task" (top-level audio), "segments" (nested segments list),
            or "auto" (segments when segments_key is present, else task; default). Setting
            "task" or "segments" overrides the auto-detection.
        action: "filter" drops items below the thresholds; "annotate" keeps every item —
            including items that fail or cannot be scored — and only writes the score keys.
        audio_filepath_key: Key in data dict for the input audio file path.
        waveform_key: Key in data dict for the in-memory waveform tensor.
        sample_rate_key: Key in data dict for the waveform sample rate.
        segments_key: Key in data dict holding the nested segments list (segments/auto mode).
        noise_key: Key where the SIGMOS noise score is written.
        ovrl_key: Key where the SIGMOS overall score is written.
        sig_key: Key where the SIGMOS signal score is written.
        col_key: Key where the SIGMOS coloration score is written.
        disc_key: Key where the SIGMOS discontinuity score is written.
        loud_key: Key where the SIGMOS loudness score is written.
        reverb_key: Key where the SIGMOS reverberation score is written.

    Note:
        GPU assignment is handled by the executor via _resources.
        Use .with_(resources=Resources(gpus=X)) to configure GPU allocation.
    """

    model_dir: str = _DEFAULT_MODEL_DIR
    model_path: str | None = None
    noise_threshold: float | None = 4.0
    ovrl_threshold: float | None = 3.5
    sig_threshold: float | None = None
    col_threshold: float | None = None
    disc_threshold: float | None = None
    loud_threshold: float | None = None
    reverb_threshold: float | None = None
    input_residency: Literal["file", "waveform", "auto"] = "auto"
    mode: Literal["task", "segments", "auto"] = "auto"
    action: Literal["filter", "annotate"] = "filter"
    audio_filepath_key: str = "audio_filepath"
    waveform_key: str = "waveform"
    sample_rate_key: str = "sample_rate"
    segments_key: str = "segments"
    noise_key: str = "sigmos_noise"
    ovrl_key: str = "sigmos_ovrl"
    sig_key: str = "sigmos_sig"
    col_key: str = "sigmos_col"
    disc_key: str = "sigmos_disc"
    loud_key: str = "sigmos_loud"
    reverb_key: str = "sigmos_reverb"

    name: str = "SIGMOSFilter"
    batch_size: int = 1
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0, gpus=0.5))

    def __post_init__(self):
        super().__init__()
        if self.mode not in _VALID_MODES:
            msg = f"mode must be one of {_VALID_MODES!r}, got {self.mode!r}"
            raise ValueError(msg)
        if self.action not in _VALID_ACTIONS:
            msg = f"action must be one of {_VALID_ACTIONS!r}, got {self.action!r}"
            raise ValueError(msg)
        self._model = None

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [
            self.noise_key,
            self.ovrl_key,
            self.sig_key,
            self.col_key,
            self.disc_key,
            self.loud_key,
            self.reverb_key,
        ]

    def describe(self) -> StageContract:
        score_keys = [
            self.noise_key,
            self.ovrl_key,
            self.sig_key,
            self.col_key,
            self.disc_key,
            self.loud_key,
            self.reverb_key,
        ]
        return StageContract(
            reads_one_of=[
                *residency_read_specs(
                    self.input_residency,
                    audio_filepath_key=self.audio_filepath_key,
                    waveform_key=self.waveform_key,
                    sample_rate_key=self.sample_rate_key,
                ),
                IOSpec(data_keys=[self.segments_key]),
            ],
            writes=IOSpec(data_keys=score_keys, segment_data_keys=score_keys),
            cardinality="filter" if self.action == "filter" else "1:1",
            cardinality_options=["filter", "annotate"],
            gates=Gates(requires_gpu=self.resources.gpus > 0, requires_internet_first_run=self.model_path is None),
        )

    @staticmethod
    def _download_model(model_dir: str) -> str:
        """Download SIGMOS ONNX model from Microsoft's SIG-Challenge repository.

        Returns the path to the validated model file.
        """
        weights_path = str(Path(model_dir) / _SIGMOS_MODEL_FILENAME)

        if not os.path.exists(weights_path):
            Path(model_dir).mkdir(parents=True, exist_ok=True)

            logger.info(f"Downloading SIGMOS model from {_SIGMOS_MODEL_URL}")
            response = requests.get(_SIGMOS_MODEL_URL, timeout=120)
            response.raise_for_status()

            with open(weights_path, "wb") as f:
                f.write(response.content)

            logger.info(f"SIGMOS model saved to {weights_path}")

        if not os.path.isfile(weights_path) or os.path.getsize(weights_path) == 0:
            msg = (
                f"SIGMOS model file is missing or empty at {weights_path}. "
                f"Download manually from {_SIGMOS_MODEL_URL} "
                f"and place it at {weights_path}"
            )
            raise RuntimeError(msg)

        return weights_path

    def setup_on_node(
        self, _node_info: NodeInfo | None = None, _worker_metadata: WorkerMetadata | None = None
    ) -> None:
        try:
            if self.model_path is None:
                self._download_model(self.model_dir)
                logger.info("SIGMOS model pre-downloaded on node")
        except Exception:  # noqa: BLE001
            logger.warning("SIGMOS model pre-download in setup_on_node failed; will retry in setup().")

    def setup(self, _: WorkerMetadata | None = None) -> None:
        from nemo_curator.utils.gpu_utils import ensure_cudnn_loaded

        ensure_cudnn_loaded()
        self._initialize_model()

    def teardown(self) -> None:
        self._model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _resolve_model_path(self) -> str:
        """Resolve the ONNX model path: model_path override → model_dir download."""
        if self.model_path is not None and os.path.isfile(self.model_path):
            return self.model_path
        return self._download_model(self.model_dir)

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
        """Process a single AudioTask and filter or annotate by SIGMOS quality metrics.

        Segment mode applies when ``mode="segments"``, or when ``mode="auto"``
        (default) and ``task.data`` contains the ``segments_key``; each segment is
        then evaluated individually. With ``action="filter"`` only survivors are
        kept; with ``action="annotate"`` every item is kept — including items that
        fail the thresholds or cannot be scored — and only the scores are written.
        """
        use_segments = self.mode == "segments" or (self.mode == "auto" and self.segments_key in task.data)
        if use_segments:
            survivors = []
            for seg in task.data.get(self.segments_key, []):
                temp = AudioTask(data=seg)
                result = self._process_single(temp)
                if result is not None or self.action == "annotate":
                    survivors.append(temp.data)
            task.data[self.segments_key] = survivors
            return task if survivors or self.action == "annotate" else []
        return self._process_single(task) or (task if self.action == "annotate" else [])

    def _process_single(self, task: AudioTask) -> AudioTask | None:
        """Run SIGMOS scoring on a single (non-nested) task."""
        audio_result = _get_audio_numpy_sr(
            task.data,
            task.task_id,
            input_residency=self.input_residency,
            audio_filepath_key=self.audio_filepath_key,
            waveform_key=self.waveform_key,
            sample_rate_key=self.sample_rate_key,
        )
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
        task.data[self.noise_key] = s["noise"]
        task.data[self.ovrl_key] = s["ovrl"]
        task.data[self.sig_key] = s["sig"]
        task.data[self.col_key] = s["col"]
        task.data[self.disc_key] = s["disc"]
        task.data[self.loud_key] = s["loud"]
        task.data[self.reverb_key] = s["reverb"]

        logger.debug(
            f"[{task.task_id}] SIGMOS NOISE={s['noise']:.3f}, OVRL={s['ovrl']:.3f}, SIG={s['sig']:.3f}, "
            f"COL={s['col']:.3f}, DISC={s['disc']:.3f}, LOUD={s['loud']:.3f}, REVERB={s['reverb']:.3f}"
        )
        if not passed:
            logger.info(f"[{task.task_id}] SIGMOS FAILED: {', '.join(fail_reasons)}")
            return task if self.action == "annotate" else None

        return task
