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
Band filter stage for audio bandwidth classification.

Classifies audio as "full_band" or "narrow_band" based on spectral
characteristics. Useful for filtering low-quality telephone or compressed audio.

Example:
    from nemo_curator.pipeline import Pipeline
    from nemo_curator.stages.audio.filtering import BandFilterStage

    # Pass only full-band audio
    pipeline = Pipeline(name="band_pipeline")
    pipeline.add_stage(BandFilterStage(band_value="full_band"))

    # Pass only narrow-band audio
    pipeline.add_stage(BandFilterStage(band_value="narrow_band"))
"""

import math
from dataclasses import KW_ONLY, dataclass, field
from typing import ClassVar, Literal

from loguru import logger

from nemo_curator.backends.base import NodeInfo, WorkerMetadata
from nemo_curator.models.audio.band.predict import BandAdapter
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
class BandFilterStage(AudioQualityFilterStage, AgentReady, ProcessingStage[AudioTask, AudioTask]):
    """
    Band filter stage for bandwidth classification.

    Classifies audio as "full_band" or "narrow_band". With action="filter"
    (default) only items matching band_value pass; with action="annotate" every
    item is kept and only the prediction is recorded.

    Args:
        model_path: Local path to band classifier model (.joblib). If not provided,
            the model is downloaded from HuggingFace (nvidia/nemocurator-speech-bandwidth-filter).
        cache_dir: Directory to cache downloaded models.
        band_value: Which band type to pass ("full_band" or "narrow_band")
        mode: Where to classify — "task" (top-level audio), "segments" (nested segments list),
            or "auto" (segments when segments_key is present, else task; default). Setting
            "task" or "segments" overrides the auto-detection.
        action: "filter" drops items that fail the band check; "annotate" keeps every item —
            including items that fail or cannot be classified — and only writes prediction_key.
        audio_filepath_key: Key in data dict for the input audio file path.
        waveform_key: Key in data dict for the in-memory waveform tensor.
        sample_rate_key: Key in data dict for the waveform sample rate.
        segments_key: Key in data dict holding the nested segments list (segments/auto mode).
        prediction_key: Key where the band prediction ("full_band"/"narrow_band") is written.
        input_residency: Which input to use — "waveform" (in-memory only), "file"
            (audio_filepath only), or "auto" (waveform first, file fallback; default).

    Note:
        GPU is used automatically when resources specify gpus > 0.
        Use .with_(resources=Resources(gpus=X)) to configure GPU allocation.

    Example:
        # Pass only full-band audio
        stage = BandFilterStage(band_value="full_band")

        # Pass only narrow-band audio
        stage = BandFilterStage(band_value="narrow_band")
    """

    model_path: str | None = None
    cache_dir: str | None = None
    band_value: Literal["full_band", "narrow_band"] = "full_band"

    name: str = "BandFilter"
    batch_size: int = 1
    resources: Resources = field(default_factory=lambda: Resources(cpus=4.0))

    _: KW_ONLY
    mode: Literal["task", "segments", "auto"] = "auto"
    action: Literal["filter", "annotate"] = "filter"
    audio_filepath_key: str = "audio_filepath"
    waveform_key: str = "waveform"
    sample_rate_key: str = "sample_rate"
    segments_key: str = "segments"
    prediction_key: str = "band_prediction"
    input_residency: Literal["file", "waveform", "auto"] = "auto"

    _VALID_BAND_VALUES: ClassVar[set[str]] = {"full_band", "narrow_band"}
    FILE_AUDIO_HYDRATION: ClassVar[str] = "always"
    INFER_SAMPLE_RATE_FROM_FILE: ClassVar[bool] = True
    AGENT_STATIC: ClassVar[StaticHints] = StaticHints(
        gates=Gates(requires_internet_first_run=True, per_row_independent=True)
    )

    def __post_init__(self):
        super().__init__()
        self._adapter: BandAdapter | None = None

        if self.band_value not in self._VALID_BAND_VALUES:
            msg = f"band_value must be one of {self._VALID_BAND_VALUES!r}, got {self.band_value!r}"
            raise ValueError(msg)
        self._validate_quality_filter_config(output_keys={"prediction_key": self.prediction_key})

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.prediction_key]

    def describe(self) -> StageContract:
        guaranteed_output_keys = [] if self.action == "annotate" or self.mode == "auto" else [self.prediction_key]
        reads, reads_one_of, writes, conditional_reads = scoped_audio_io_specs(
            self.input_residency,
            mode=self.mode,
            audio_filepath_key=self.audio_filepath_key,
            waveform_key=self.waveform_key,
            sample_rate_key=self.sample_rate_key,
            segments_key=self.segments_key,
            output_keys=guaranteed_output_keys,
            infer_sample_rate_from_file=True,
        )
        return StageContract(
            reads=reads,
            reads_one_of=reads_one_of,
            conditional_reads=conditional_reads,
            writes=writes,
            invalidates_prior_writes=IOSpec(
                data_keys=[self.prediction_key] if self.mode in {"task", "auto"} else [],
                segment_data_keys=[self.prediction_key] if self.mode in {"segments", "auto"} else [],
            ),
            conditional_writes=[
                *scoped_file_audio_hydration_writes(
                    self.input_residency,
                    hydration_policy="always",
                    mode=self.mode,
                    waveform_key=self.waveform_key,
                    sample_rate_key=self.sample_rate_key,
                    segments_key=self.segments_key,
                    infer_sample_rate_from_file=True,
                ),
                *scoped_audio_conditional_writes(
                    self.mode,
                    segments_key=self.segments_key,
                    output_keys=[self.prediction_key],
                    assignment_condition=(
                        "audio resolves, the predictor returns 'full_band' or 'narrow_band', "
                        f"'{self.prediction_key}' is assigned"
                        + (
                            ", and the item matches the configured band and is retained"
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
                # Each clip's features are extracted from that clip alone.
                per_row_independent=True,
            ),
        )

    def setup_on_node(
        self, _node_info: NodeInfo | None = None, _worker_metadata: WorkerMetadata | None = None
    ) -> None:
        try:
            BandAdapter(model_path=self.model_path, cache_dir=self.cache_dir).download_weights_on_node()
        except Exception:  # noqa: BLE001
            logger.warning("Model pre-download in setup_on_node failed; will retry in setup().")

    def setup(self, _: WorkerMetadata | None = None) -> None:
        if self._adapter is not None:
            return
        adapter = BandAdapter(model_path=self.model_path, cache_dir=self.cache_dir)
        try:
            adapter.load_model(num_gpus=math.ceil(self.resources.gpus))
        except Exception as e:
            logger.error(f"Failed to initialize Band predictor: {e}")
            raise
        self._adapter = adapter

    def teardown(self) -> None:
        if self._adapter is not None:
            self._adapter.unload_model()
            self._adapter = None

    def _process_single(self, task: AudioTask) -> AudioTask | None:
        """Run band classification on a single (non-nested) task."""
        # This stage owns ``prediction_key``. Clear a prior annotation before
        # inference so an unscorable rerun cannot retain stale classification.
        task.data.pop(self.prediction_key, None)
        if self._adapter is None:
            logger.error("Band predictor not available")
            return None

        audio = self._resolve_mono_audio(task)
        if audio is None:
            return None
        waveform, sample_rate = audio

        try:
            pred = self._adapter.predict_audio(waveform, sample_rate)
            if isinstance(pred, str) and not pred.startswith("Error") and pred in ("full_band", "narrow_band"):
                task.data[self.prediction_key] = pred
            else:
                logger.warning(f"[{task.task_id}] BandFilter: unexpected prediction value: {pred!r}")
        except Exception as e:  # noqa: BLE001
            logger.exception(f"[BandFilter] Prediction error: {e}")
            return None

        actual = task.data.get(self.prediction_key, "unknown")
        if actual != self.band_value:
            logger.info(f"[{task.task_id}] BAND FILTER FAILED: prediction '{actual}' != target '{self.band_value}'")
            return task if self.action == "annotate" else None

        return task
