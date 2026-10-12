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

"""Annotate Granary segments with UTMOSv2 MOS scores (TTS Granary ``OrigAudioPipeline.UTMOSv2``)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from nemo_curator.stages.audio.tts.audio_io import (
    SegmentAudioError,
    iter_segments,
    load_segment_audio,
    make_audio_config,
)
from nemo_curator.stages.audio.tts.fields import get_dotted, set_dotted
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

UTMOSV2_FAILED = "utmosv2_failed"
UTMOSV2_UNAVAILABLE = "utmosv2_unavailable"
EMPTY_AUDIO = "empty_audio"


def empty_annotation(error: str) -> dict[str, Any]:
    return {"score": None, "error": error}


def coerce_prediction_to_float(prediction: Any) -> float:  # noqa: ANN401
    """Convert a ``utmosv2`` prediction (dict / tensor / array / scalar) to a float score."""
    if isinstance(prediction, dict):
        if "predicted_mos" in prediction:
            return float(prediction["predicted_mos"])
        msg = f"Unsupported dict prediction payload: {prediction}"
        raise TypeError(msg)
    if hasattr(prediction, "detach"):
        prediction = prediction.detach().cpu().numpy()
    array = np.asarray(prediction)
    if array.size == 0:
        msg = "UTMOSv2 returned an empty prediction"
        raise ValueError(msg)
    return float(array.reshape(-1)[0])


def configure_offline_cache(cache_dir: str) -> None:
    """Point the HF / transformers caches at ``cache_dir`` and force offline mode (reference behaviour)."""
    cache_path = Path(cache_dir).expanduser()
    cache_path.mkdir(parents=True, exist_ok=True)
    for key in ("HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE"):
        os.environ[key] = str(cache_path)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


@dataclass
class UTMOSv2AnnotationStage(ProcessingStage[AudioTask, AudioTask]):
    """Score every segment with UTMOSv2 and write ``{"score", "error"}`` at ``output_key``.

    Port of TTS Granary ``pipeline.stages.mos.utmosv2.UTMOSv2Processor``. Audio is scored at its native sample
    rate (UTMOSv2 resamples internally). Rows and segments are never dropped; failures are recorded as error codes
    (audio loading codes from ``audio_io``, ``utmosv2_failed``, ``utmosv2_unavailable``).

    This is **not** Curator's ``UTMOSFilterStage``: that stage runs UTMOS22-strong (``tarepan/SpeechMOS``) and drops
    rows on failure or below threshold. GranaryHiFi MOS thresholds (YTC/VoxPopuli 1.5, YODAS 1.0, LibriLight 2.5)
    are calibrated on UTMOSv2 and applied later by the segment filter.

    Requires the optional ``utmosv2`` package; when it is missing every segment gets ``utmosv2_unavailable``.

    Args:
        checkpoint_path: Local UTMOSv2 checkpoint (reference: ``fold0_s42_best_model.pth``). ``None`` lets
            ``utmosv2`` download its default pretrained weights.
        cache_dir: When set, HF / transformers caches point here and offline mode is forced.
        device: Device passed to ``model.predict``.
        create_model_kwargs: Extra keyword arguments for ``utmosv2.create_model``.
        num_repetitions: Random-crop repetitions averaged by ``predict``. UTMOSv2 is stochastic: with the default 1
            (the published dataset's setting) repeated scoring of the same audio varies by roughly +-0.2 MOS; raise
            it for more stable filtering at proportional cost.
        allow_missing_package: Annotate ``utmosv2_unavailable`` instead of raising when ``utmosv2`` is missing.
            Off by default: a missing score fails the MOS filter condition and would silently drop every segment.
        model: Pre-built model (tests / sharing); skips ``utmosv2.create_model``.
    """

    name: str = "UTMOSv2Annotation"
    output_key: str = "utmosv2"
    checkpoint_path: str | None = None
    cache_dir: str | None = None
    device: str = "cuda"
    pretrained: bool = True
    create_model_kwargs: dict[str, Any] = field(default_factory=dict)
    num_repetitions: int = 1
    allow_missing_package: bool = False
    tar_root: str | None = None
    strip_tar_key_prefixes: tuple[str, ...] = ()
    audio_source: str = "original"
    enhanced_key: str = "enhanced_audio"
    enhanced_fallback: bool = False
    overwrite: bool = False
    model: Any | None = None
    resources: Resources = field(default_factory=lambda: Resources(cpus=2.0, gpus=1.0))
    batch_size: int = 1

    def __post_init__(self) -> None:
        super().__init__()
        self._setup_error: str | None = None
        self._audio_config = make_audio_config(self)

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.output_key.split(".")[0]]

    def setup(self, _worker_metadata: Any = None) -> None:  # noqa: ANN401
        if self.model is not None:
            return
        if self.cache_dir:
            configure_offline_cache(self.cache_dir)
        try:
            import utmosv2
        except ImportError as exc:
            if not self.allow_missing_package:
                msg = "UTMOSv2AnnotationStage requires the 'utmosv2' package (pip install utmosv2)"
                raise ImportError(msg) from exc
            logger.error(
                "UTMOSv2AnnotationStage: the 'utmosv2' package is not installed; segments get {}", UTMOSV2_UNAVAILABLE
            )
            self._setup_error = UTMOSV2_UNAVAILABLE
            return
        kwargs = dict(self.create_model_kwargs)
        if self.checkpoint_path:
            kwargs["checkpoint_path"] = str(Path(self.checkpoint_path).expanduser())
        self.model = utmosv2.create_model(pretrained=self.pretrained, **kwargs)

    def score_segment(self, segment: dict[str, Any]) -> dict[str, Any]:
        if self.model is None:
            return empty_annotation(self._setup_error or UTMOSV2_UNAVAILABLE)
        try:
            waveform, sample_rate = load_segment_audio(segment, None, self._audio_config)
        except SegmentAudioError as exc:
            return empty_annotation(exc.code)
        if waveform.size == 0:
            return empty_annotation(EMPTY_AUDIO)
        try:
            prediction = self.model.predict(
                data=waveform, sr=sample_rate, device=self.device, num_repetitions=self.num_repetitions, verbose=False
            )
            return {"score": coerce_prediction_to_float(prediction), "error": None}
        except Exception as exc:  # noqa: BLE001
            logger.warning("UTMOSv2 scoring failed: {}", exc)
            return empty_annotation(UTMOSV2_FAILED)

    def _annotate(self, data: dict[str, Any]) -> None:
        for _, segment in iter_segments(data):
            if not self.overwrite:
                existing = get_dotted(segment, self.output_key)
                if isinstance(existing, dict) and existing.get("score") is not None:
                    continue
            set_dotted(segment, self.output_key, self.score_segment(segment))

    def process(self, task: AudioTask) -> AudioTask:
        self._annotate(task.data)
        return task
