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

import math
from dataclasses import KW_ONLY, dataclass, field
from typing import Any, ClassVar, Literal

import numpy as np
from loguru import logger

from nemo_curator.backends.base import NodeInfo, WorkerMetadata
from nemo_curator.models.audio.sigmos import _DEFAULT_MODEL_DIR, SIGMOSAdapter
from nemo_curator.stages.audio._agent._agent_ready import AgentReady, Gates, IOSpec, StageContract, StaticHints
from nemo_curator.stages.audio._agent._residency import (
    scoped_audio_conditional_writes,
    scoped_audio_io_specs,
    scoped_file_audio_hydration_writes,
)
from nemo_curator.stages.audio.filtering.base import AudioQualityFilterStage
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask


@dataclass
class SIGMOSFilterStage(AudioQualityFilterStage, AgentReady, ProcessingStage[AudioTask, AudioTask]):
    """
    SIGMOS quality assessment filter stage.

    Filters audio segments based on SIGMOS quality metrics.
    Input: items with waveform + sample_rate (tensor/array) or audio_filepath (WAV).
    Model download and ONNX inference live in ``nemo_curator.models.audio.sigmos.SIGMOSAdapter``,
    loaded once in setup() and reused for all predictions.

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

    SEPARABLE_DECISION_CONSTRAINTS: ClassVar[dict[str, Any]] = {
        "action": "annotate",
        "mode": "task",
    }
    SEPARABLE_DECISION_CONSTRAINTS_BY_SCOPE: ClassVar[dict[str, dict[str, Any]]] = {
        "task": {"action": "annotate", "mode": "task"},
        "segments": {"action": "annotate", "mode": "segments"},
    }
    SEPARABLE_DECISION_DIMENSIONS: ClassVar[tuple[tuple[str, str], ...]] = (
        ("noise_threshold", "noise_key"),
        ("ovrl_threshold", "ovrl_key"),
        ("sig_threshold", "sig_key"),
        ("col_threshold", "col_key"),
        ("disc_threshold", "disc_key"),
        ("loud_threshold", "loud_key"),
        ("reverb_threshold", "reverb_key"),
    )

    model_dir: str = _DEFAULT_MODEL_DIR
    model_path: str | None = None
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

    _: KW_ONLY
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

    AGENT_STATIC: ClassVar[StaticHints] = StaticHints(
        gates=Gates(requires_internet_first_run=True, per_row_independent=True)
    )

    def __post_init__(self):
        super().__init__()
        self._validate_quality_filter_config(
            output_keys={
                "noise_key": self.noise_key,
                "ovrl_key": self.ovrl_key,
                "sig_key": self.sig_key,
                "col_key": self.col_key,
                "disc_key": self.disc_key,
                "loud_key": self.loud_key,
                "reverb_key": self.reverb_key,
            }
        )
        self._adapter: SIGMOSAdapter | None = None

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
        guaranteed_output_keys = [] if self.action == "annotate" or self.mode == "auto" else score_keys
        reads, reads_one_of, writes, conditional_reads = scoped_audio_io_specs(
            self.input_residency,
            mode=self.mode,
            audio_filepath_key=self.audio_filepath_key,
            waveform_key=self.waveform_key,
            sample_rate_key=self.sample_rate_key,
            segments_key=self.segments_key,
            output_keys=guaranteed_output_keys,
        )
        return StageContract(
            reads=reads,
            reads_one_of=reads_one_of,
            conditional_reads=conditional_reads,
            writes=writes,
            invalidates_prior_writes=IOSpec(
                data_keys=score_keys if self.mode in {"task", "auto"} else [],
                segment_data_keys=score_keys if self.mode in {"segments", "auto"} else [],
            ),
            conditional_writes=[
                *scoped_file_audio_hydration_writes(
                    self.input_residency,
                    hydration_policy="auto_partial",
                    mode=self.mode,
                    waveform_key=self.waveform_key,
                    sample_rate_key=self.sample_rate_key,
                    segments_key=self.segments_key,
                ),
                *scoped_audio_conditional_writes(
                    self.mode,
                    segments_key=self.segments_key,
                    output_keys=score_keys,
                    assignment_condition=(
                        "audio and model inference succeed and all configured SIGMOS "
                        "dimensions are finite before any score key is assigned"
                        + (
                            " on an item that meets every enabled threshold and is retained"
                            if self.action == "filter"
                            else ""
                        )
                    ),
                ),
            ],
            cardinality="filter" if self.action == "filter" else "1:1",
            cardinality_options=["filter", "annotate"],
            gates=Gates(
                requires_gpu=self.resources.requires_gpu,
                requires_internet_first_run=self.model_path is None,
                # ``_process_single`` runs the ONNX model on one clip's samples and compares the
                # scores against the configured thresholds, never against the corpus.
                per_row_independent=True,
            ),
        )

    def setup_on_node(
        self, _node_info: NodeInfo | None = None, _worker_metadata: WorkerMetadata | None = None
    ) -> None:
        try:
            SIGMOSAdapter(model_dir=self.model_dir, model_path=self.model_path).download_weights_on_node()
            if self.model_path is None:
                logger.info("SIGMOS model pre-downloaded on node")
        except Exception:  # noqa: BLE001
            logger.warning("SIGMOS model pre-download in setup_on_node failed; will retry in setup().")

    def setup(self, _: WorkerMetadata | None = None) -> None:
        if self._adapter is not None:
            return
        adapter = SIGMOSAdapter(model_dir=self.model_dir, model_path=self.model_path)
        adapter.load_model(num_gpus=math.ceil(self.resources.gpus))
        self._adapter = adapter

    def teardown(self) -> None:
        if self._adapter is not None:
            self._adapter.unload_model()
            self._adapter = None

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

    def _process_single(self, task: AudioTask) -> AudioTask | None:
        """Run SIGMOS scoring on a single (non-nested) task."""
        # The compound annotation is atomic. Remove every prior output before
        # inference so a failed/non-finite rerun cannot expose a stale partial
        # score set to the downstream missing=drop selector.
        for score_key in (
            self.noise_key,
            self.ovrl_key,
            self.sig_key,
            self.col_key,
            self.disc_key,
            self.loud_key,
            self.reverb_key,
        ):
            task.data.pop(score_key, None)
        audio_result = self._resolve_mono_audio(task)
        if audio_result is None:
            return None
        waveform, sample_rate = audio_result
        audio_np = waveform.squeeze(0).detach().cpu().numpy().astype(np.float32)

        if self._adapter is None:
            return None

        try:
            score_data = self._adapter.run(audio=audio_np, sr=sample_rate)
        except Exception as e:  # noqa: BLE001
            logger.exception(f"[{task.task_id}] SIGMOS prediction error: {e}")
            return None

        s = self._scores_from_prediction(score_data)
        if not all(math.isfinite(value) for value in s.values()):
            logger.warning(
                f"[{task.task_id}] SIGMOS returned at least one non-finite dimension; treating item as unscorable"
            )
            return None
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
