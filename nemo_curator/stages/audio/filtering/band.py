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
from nemo_curator.stages.audio._agent_ready import AgentReady, Gates, IOSpec, StageContract
from nemo_curator.stages.audio._residency import resolve_audio
from nemo_curator.stages.audio.filtering.band_filter_module.predict import BandPredictor
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

_HF_REPO_ID = "nvidia/nemocurator-speech-bandwidth-filter"
_HF_MODEL_FILENAME = "band_classifier_model_band_7000_samples.joblib"
_VALID_MODES = {"task", "segments", "auto"}
_VALID_ACTIONS = {"filter", "annotate"}


@dataclass
class BandFilterStage(AgentReady, ProcessingStage[AudioTask, AudioTask]):
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
    mode: Literal["task", "segments", "auto"] = "auto"
    action: Literal["filter", "annotate"] = "filter"
    audio_filepath_key: str = "audio_filepath"
    waveform_key: str = "waveform"
    sample_rate_key: str = "sample_rate"
    segments_key: str = "segments"
    prediction_key: str = "band_prediction"
    input_residency: Literal["file", "waveform", "auto"] = "auto"

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
        if self.mode not in _VALID_MODES:
            msg = f"mode must be one of {_VALID_MODES!r}, got {self.mode!r}"
            raise ValueError(msg)
        if self.action not in _VALID_ACTIONS:
            msg = f"action must be one of {_VALID_ACTIONS!r}, got {self.action!r}"
            raise ValueError(msg)

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.prediction_key]

    def describe(self) -> StageContract:
        return StageContract(
            reads_one_of=[
                IOSpec(data_keys=[self.waveform_key, self.sample_rate_key], accepts=["waveform"]),
                IOSpec(data_keys=[self.audio_filepath_key], accepts=["file"]),
                IOSpec(data_keys=[self.segments_key]),
            ],
            writes=IOSpec(data_keys=[self.prediction_key], segment_data_keys=[self.prediction_key]),
            cardinality="filter" if self.action == "filter" else "1:1",
            cardinality_options=["filter", "annotate"],
            gates=Gates(requires_gpu=self.resources.gpus > 0, requires_internet_first_run=self.model_path is None),
        )

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
        Filter or annotate audio based on bandwidth classification.

        Segment handling follows ``mode``: with ``mode="auto"`` (default), nested
        mode is used when ``task.data`` contains the ``segments_key``; ``mode="task"``
        or ``mode="segments"`` overrides that auto-detection. In nested mode each
        segment is evaluated individually. With ``action="filter"`` only survivors
        are kept; with ``action="annotate"`` every item is kept — including items
        that fail the band check or cannot be classified — and only the prediction
        annotation is written.

        Returns:
            AudioTask if it passes (or ``action="annotate"``), [] if filtered out.
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
        """Run band classification on a single (non-nested) task."""
        if self._predictor is None:
            logger.error("Band predictor not available")
            return None

        try:
            audio = resolve_audio(
                task.data,
                residency=self.input_residency,  # type: ignore[arg-type]
                audio_filepath_key=self.audio_filepath_key,
                waveform_key=self.waveform_key,
                sample_rate_key=self.sample_rate_key,
            )
        except (OSError, RuntimeError) as e:  # corrupt/unreadable audio -> skip the row, don't crash the batch
            logger.error(f"Failed to load audio for {task.data.get(self.audio_filepath_key)!r}: {e}")
            return None
        if audio is None:
            return None
        waveform, sample_rate = audio

        try:
            pred = self._predictor.predict_audio(waveform, sample_rate)
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
