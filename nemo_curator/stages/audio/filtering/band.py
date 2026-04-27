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

import os
from dataclasses import dataclass, field
from typing import ClassVar, Literal

import torch
from huggingface_hub import hf_hub_download
from loguru import logger

from nemo_curator.backends.base import NodeInfo, WorkerMetadata
from nemo_curator.stages.audio.common import resolve_waveform_from_item
from nemo_curator.stages.audio.filtering.band_filter_module.predict import BandPredictor
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

_HF_REPO_ID = "nvidia/nemocurator-speech-bandwidth-filter"
_HF_MODEL_FILENAME = "band_classifier_model_band_7000_samples.joblib"


@dataclass
class BandFilterStage(ProcessingStage[AudioTask, AudioTask]):
    """
    Band filter stage for bandwidth classification.

    Classifies audio as "full_band" or "narrow_band" and filters
    based on the specified band_value to pass.

    Args:
        model_path: Local path to band classifier model (.joblib). If not provided,
            the model is downloaded from HuggingFace (nvidia/nemocurator-speech-bandwidth-filter).
        cache_dir: Directory to cache downloaded models.
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

    model_path: str | None = None
    cache_dir: str | None = None
    band_value: Literal["full_band", "narrow_band"] = "full_band"

    name: str = "BandFilter"
    batch_size: int = 1
    resources: Resources = field(default_factory=lambda: Resources(cpus=4.0))

    _VALID_BAND_VALUES: ClassVar[set[str]] = {"full_band", "narrow_band"}

    def __post_init__(self):
        super().__init__()
        self._predictor = None

        if self.band_value not in self._VALID_BAND_VALUES:
            msg = f"band_value must be one of {self._VALID_BAND_VALUES!r}, got {self.band_value!r}"
            raise ValueError(msg)

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], ["band_prediction"]

    def setup_on_node(
        self, _node_info: NodeInfo | None = None, _worker_metadata: WorkerMetadata | None = None
    ) -> None:
        try:
            if self.model_path is None:
                self.model_path = hf_hub_download(
                    repo_id=_HF_REPO_ID,
                    filename=_HF_MODEL_FILENAME,
                    cache_dir=self.cache_dir,
                )
                logger.info(f"Band filter model downloaded to {self.model_path}")
        except Exception:  # noqa: BLE001
            logger.warning("Model pre-download in setup_on_node failed; will retry in setup().")

    def setup(self, _: WorkerMetadata | None = None) -> None:
        self._initialize_predictor()

    def teardown(self) -> None:
        if self._predictor is not None:
            del self._predictor
            self._predictor = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _resolve_model_path(self) -> str:
        if self.model_path is not None and os.path.isfile(self.model_path):
            return self.model_path
        return hf_hub_download(
            repo_id=_HF_REPO_ID,
            filename=_HF_MODEL_FILENAME,
            cache_dir=self.cache_dir,
        )

    def _initialize_predictor(self) -> None:
        if self._predictor is None:
            try:
                model_path = self._resolve_model_path()
                self._predictor = BandPredictor(
                    model_path=model_path,
                    feature_cache_size=100,
                )
                logger.info("Band predictor loaded successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Band predictor: {e}")
                raise

    def process(self, task: AudioTask) -> AudioTask | list[AudioTask]:
        """
        Filter audio based on bandwidth classification.

        When ``task.data`` contains a ``"segments"`` key (nested mode from VAD),
        each segment is evaluated individually and only survivors are kept.

        Returns:
            AudioTask if passes the band filter, [] if filtered out.
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
        """Run band classification on a single (non-nested) task."""
        if self._predictor is None:
            logger.error("Band predictor not available")
            return None

        audio = resolve_waveform_from_item(task.data, task.task_id)
        if audio is None:
            return None
        waveform, sample_rate = audio

        try:
            pred = self._predictor.predict_audio(waveform, sample_rate)
            if isinstance(pred, str) and not pred.startswith("Error") and pred in ("full_band", "narrow_band"):
                task.data["band_prediction"] = pred
            else:
                logger.warning(f"[{task.task_id}] BandFilter: unexpected prediction value: {pred!r}")
        except Exception as e:  # noqa: BLE001
            logger.exception(f"[BandFilter] Prediction error: {e}")
            return None

        actual = task.data.get("band_prediction", "unknown")
        if actual != self.band_value:
            logger.info(
                f"[{task.task_id}] BAND FILTER FAILED: prediction '{actual}' != target '{self.band_value}'"
            )
            return None

        return task
