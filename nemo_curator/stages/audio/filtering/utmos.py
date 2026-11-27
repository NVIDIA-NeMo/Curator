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
UTMOS (UTokyo-SaruLab MOS Prediction) filter stage.

Filters audio segments based on UTMOS predicted Mean Opinion Score.
Uses the utmos22_strong model from tarepan/SpeechMOS via torch.hub.

Accepts in-memory (waveform + sample_rate) or audio_filepath input.
Audio is resampled to 16 kHz internally for UTMOS inference.

Example:
    from nemo_curator.pipeline import Pipeline
    from nemo_curator.stages.audio.filtering import UTMOSFilterStage
    from nemo_curator.stages.resources import Resources

    pipeline = Pipeline(name="quality_pipeline")
    pipeline.add_stage(
        UTMOSFilterStage(mos_threshold=3.5)
        .with_(resources=Resources(cpus=1.0, gpus=0.5))
    )
"""

import math
from dataclasses import KW_ONLY, dataclass, field
from typing import Any, ClassVar, Literal

from loguru import logger

from nemo_curator.backends.base import NodeInfo, WorkerMetadata
from nemo_curator.models.audio.utmos import _UTMOS_TARGET_SR, UTMOSAdapter
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
class UTMOSFilterStage(AudioQualityFilterStage, AgentReady, ProcessingStage[AudioTask, AudioTask]):
    """
    UTMOS quality assessment filter stage.

    Filters audio segments based on the UTMOS predicted MOS score.
    Model loading and inference live in ``nemo_curator.models.audio.utmos.UTMOSAdapter``
    (utmos22_strong via torch.hub from tarepan/SpeechMOS, resampled to 16 kHz).

    Args:
        mos_threshold: Minimum MOS score to pass (None to disable)
        sample_rate: Target sample rate for UTMOS inference (default 16000)
        input_residency: Which input to use — "waveform" (in-memory only), "file"
            (audio_filepath only), or "auto" (waveform first, file fallback; default).
        mode: Where to score — "task" (top-level audio), "segments" (nested segments list),
            or "auto" (segments when segments_key is present, else task; default). Setting
            "task" or "segments" overrides the auto-detection.
        action: "filter" drops items below mos_threshold; "annotate" keeps every item —
            including items that fail or cannot be scored — and only writes score_key.
        audio_filepath_key: Key in data dict for the input audio file path.
        waveform_key: Key in data dict for the in-memory waveform tensor.
        sample_rate_key: Key in data dict for the waveform sample rate.
        segments_key: Key in data dict holding the nested segments list (segments/auto mode).
        score_key: Key where the UTMOS MOS score is written.

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

    mos_threshold: float | None = 3.5
    sample_rate: int = _UTMOS_TARGET_SR

    name: str = "UTMOSFilter"
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
    score_key: str = "utmos_mos"

    AGENT_STATIC: ClassVar[StaticHints] = StaticHints(
        gates=Gates(requires_internet_first_run=True, per_row_independent=True)
    )

    def __post_init__(self):
        super().__init__()
        self._validate_quality_filter_config(output_keys={"score_key": self.score_key})
        self._adapter: UTMOSAdapter | None = None

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.score_key]

    def describe(self) -> StageContract:
        guaranteed_output_keys = [] if self.action == "annotate" or self.mode == "auto" else [self.score_key]
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
                data_keys=[self.score_key] if self.mode in {"task", "auto"} else [],
                segment_data_keys=[self.score_key] if self.mode in {"segments", "auto"} else [],
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
                    output_keys=[self.score_key],
                    assignment_condition=(
                        "audio and model inference succeed, a finite numeric MOS is produced, "
                        f"and '{self.score_key}' is assigned"
                        + (
                            " on an item that meets the configured threshold and is retained"
                            if self.action == "filter"
                            else ""
                        )
                    ),
                ),
            ],
            cardinality="filter" if self.action == "filter" else "1:1",
            cardinality_options=["filter", "annotate"],
            gates=Gates(
                requires_gpu=self.resources.requires_gpu, requires_internet_first_run=True, per_row_independent=True
            ),
        )

    def setup_on_node(
        self, _node_info: NodeInfo | None = None, _worker_metadata: WorkerMetadata | None = None
    ) -> None:
        try:
            UTMOSAdapter(sample_rate=self.sample_rate).download_weights_on_node()
        except Exception:  # noqa: BLE001
            logger.opt(exception=True).warning("UTMOS repo pre-download in setup_on_node failed.")

    def setup(self, _: WorkerMetadata | None = None) -> None:
        if self._adapter is not None:
            return
        adapter = UTMOSAdapter(sample_rate=self.sample_rate)
        adapter.load_model(num_gpus=math.ceil(self.resources.gpus))
        self._adapter = adapter

    def teardown(self) -> None:
        if self._adapter is not None:
            self._adapter.unload_model()
            self._adapter = None

    def _process_single(self, task: AudioTask) -> AudioTask | None:
        """Run UTMOS scoring on a single (non-nested) task."""
        # This stage owns ``score_key``. Clear a prior annotation before
        # inference so an unscorable rerun cannot leave a stale finite value
        # that a downstream missing=drop selector would incorrectly retain.
        task.data.pop(self.score_key, None)
        audio_result = self._resolve_mono_audio(task)
        if audio_result is None:
            return None
        waveform, sr = audio_result

        if self._adapter is None:
            return None

        try:
            mos = self._adapter.score(waveform, sr)
        except Exception as e:  # noqa: BLE001
            logger.exception(f"[{task.task_id}] UTMOS prediction error: {e}")
            return None
        if not math.isfinite(mos):
            logger.warning(f"[{task.task_id}] UTMOS returned non-finite MOS; treating item as unscorable")
            return None

        logger.debug(f"[{task.task_id}] UTMOS MOS={mos:.3f}")

        task.data[self.score_key] = mos

        if self.mos_threshold is not None and mos < self.mos_threshold:
            logger.info(f"[{task.task_id}] UTMOS FAILED: MOS {mos:.3f} < {self.mos_threshold}")
            return task if self.action == "annotate" else None

        return task
