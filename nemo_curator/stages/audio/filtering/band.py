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

from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

import torch
from loguru import logger

from nemo_curator.stages.audio.filtering.band_filter_module.predict import BandPredictor
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioBatch
from nemo_curator.utils.performance_utils import StagePerfStats

from nemo_curator.stages.audio.common import resolve_model_path, resolve_waveform_from_item


@dataclass
class BandFilterStage(ProcessingStage[AudioBatch, AudioBatch]):
    """
    Band filter stage for bandwidth classification.

    Classifies audio as "full_band" or "narrow_band" and filters
    based on the specified band_value to pass.

    Args:
        model_path: Path to band classifier model (.joblib)
        band_value: Which band type to pass ("full_band" or "narrow_band")

    Note:
        GPU is used automatically when resources specify gpus > 0.
        Use .with_(resources=Resources(gpus=X)) to configure GPU allocation.

    Example:
        # Pass only full-band audio
        stage = BandFilterStage(band_value="full_band")

        # Pass only narrow-band audio
        stage = BandFilterStage(band_value="narrow_band")
    """

    model_path: str = "model/band_classifier_model_band_7000_samples.joblib"
    band_value: Literal["full_band", "narrow_band"] = "full_band"

    name: str = "BandFilter"
    batch_size: int = 1
    resources: Resources = field(default_factory=lambda: Resources(cpus=4.0))

    _VALID_BAND_VALUES: ClassVar[set[str]] = {"full_band", "narrow_band"}

    def __post_init__(self):
        """Initialize after dataclass fields are set."""
        super().__init__()
        self._predictor = None

        if self.band_value not in self._VALID_BAND_VALUES:
            msg = f"band_value must be one of {self._VALID_BAND_VALUES!r}, got {self.band_value!r}"
            raise ValueError(msg)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        """Define outputs produced by this stage."""
        return [], ["band_prediction"]

    def setup(self, worker_metadata: Any = None) -> None:  # noqa: ARG002, ANN401
        """Load band predictor on worker initialization."""
        self._initialize_predictor()

    def teardown(self) -> None:
        """Clean up resources."""
        if self._predictor is not None:
            del self._predictor
            self._predictor = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _initialize_predictor(self) -> None:
        """Initialize the band predictor."""
        if self._predictor is None:
            try:
                model_path = resolve_model_path(self.model_path, __file__, "band_filter_module")
                self._predictor = BandPredictor(
                    model_path=model_path,
                    feature_cache_size=100,
                )
                logger.info("Band predictor loaded successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Band predictor: {e}")
                raise

    def process(self, task: AudioBatch) -> AudioBatch | None:
        """
        Filter audio based on bandwidth classification.

        Processes each item sequentially, classifies as full_band/narrow_band,
        then filters by band_value.

        Args:
            task: AudioBatch with waveform data (or audio_filepath for load).

        Returns:
            AudioBatch with only items that pass the band filter.
        """
        self._initialize_predictor()

        if self._predictor is None:
            logger.error("Band predictor not available")
            return AudioBatch(
                data=[],
                task_id=task.task_id,
                dataset_name=task.dataset_name,
                _metadata=task._metadata,
                _stage_perf=list(task._stage_perf),
            )

        for item in task.data:
            audio = resolve_waveform_from_item(item, task.task_id)
            if audio is None:
                continue
            waveform, sample_rate = audio
            try:
                pred = self._predictor.predict_audio(waveform, sample_rate)
                if (
                    isinstance(pred, str)
                    and not pred.startswith("Error")
                    and pred in ("full_band", "narrow_band")
                ):
                    item["band_prediction"] = pred
            except Exception as e:
                logger.exception(f"[BandFilter] Prediction error: {e}")

        filtered_items = [
            item
            for item in task.data
            if item.get("band_prediction") == self.band_value
        ]

        total_items = len(task.data)
        passed_count = len(filtered_items)

        result = AudioBatch(
            data=filtered_items,
            task_id=task.task_id,
            dataset_name=task.dataset_name,
            _metadata=task._metadata,
            _stage_perf=list(task._stage_perf),
        )
        result.add_stage_perf(StagePerfStats(
            stage_name=self.name,
            num_items_processed=total_items,
            custom_metrics={
                "items_passed": float(passed_count),
                "items_failed": float(total_items - passed_count),
            },
        ))
        return result
