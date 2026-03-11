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
from collections.abc import Generator
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import soundfile as sf
import torch
from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioBatch

from ..configs import BandFilterConfig


def _load_audio_file(audio_path: str) -> Tuple[torch.Tensor, int]:
    """
    Load audio file and return waveform tensor and sample rate.

    Supports standalone usage of stages without requiring MonoConversionStage.
    """
    data, sample_rate = sf.read(audio_path, dtype='float32')
    waveform = torch.from_numpy(data)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    else:
        waveform = waveform.T
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform, sample_rate


@dataclass
class BandFilterStage(ProcessingStage[AudioBatch, AudioBatch]):
    """
    Band filter stage for bandwidth classification.

    Classifies audio as "full_band" or "narrow_band" and filters
    based on the specified band_value to pass.

    Args:
        config: BandFilterConfig object (overrides other params if provided)
        model_path: Path to band classifier model (.joblib)
        band_value: Which band type to pass ("full_band" or "narrow_band")

    Note:
        GPU is used automatically when resources specify gpus > 0.
        Use .with_(resources=Resources(gpus=X)) to configure GPU allocation.

    Example:
        # Using config - pass only full-band audio
        config = BandFilterConfig(band_value="full_band")
        stage = BandFilterStage(config=config)

        # Using parameters - pass only narrow-band audio
        stage = BandFilterStage(band_value="narrow_band")
    """

    config: Optional[BandFilterConfig] = None
    model_path: str = "model/band_classifier_model_band_7000_samples.joblib"
    model_inference_batch_size: int = 32
    band_value: str = "full_band"

    name: str = "BandFilter"
    batch_size: int = 1
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    def __post_init__(self):
        """Initialize after dataclass fields are set."""
        super().__init__()
        self._predictor = None

        # Apply config if provided
        if self.config is not None:
            self.model_path = self.config.model_path
            self.model_inference_batch_size = getattr(
                self.config, "model_inference_batch_size", 32
            )
            self.band_value = self.config.band_value

    def inputs(self) -> Tuple[List[str], List[str]]:
        return ["data"], []

    def outputs(self) -> Tuple[List[str], List[str]]:
        """Define outputs produced by this stage."""
        return [], ["band_prediction"]

    def setup(self, worker_metadata=None) -> None:
        """Load band predictor on worker initialization."""
        from nemo_curator.utils.gpu_utils import ensure_cudnn_loaded

        ensure_cudnn_loaded()
        self._initialize_predictor()

    def teardown(self) -> None:
        """Clean up resources."""
        if self._predictor is not None:
            del self._predictor
            self._predictor = None
            torch.cuda.empty_cache()

    def _initialize_predictor(self):
        """Initialize the band predictor."""
        if self._predictor is None:
            try:
                from nemo_curator.stages.audio.filtering.band_filter_module.predict import BandPredictor

                model_path = self._resolve_model_path()
                use_gpu = self._resources.gpus > 0 and torch.cuda.is_available()
                # Derive worker count from allocated CPUs (image-stage pattern: no n_workers in config)
                effective_n_workers = (
                    max(1, int(self._resources.cpus))
                    if self._resources and getattr(self._resources, "cpus", None) is not None
                    else 1
                )

                self._predictor = BandPredictor(
                    model_path=model_path,
                    n_workers=effective_n_workers,
                    feature_cache_size=100,  # internal default, not exposed in config
                    use_gpu=use_gpu,
                )
                logger.info(f"Band predictor loaded successfully (GPU: {use_gpu})")
            except ImportError as e:
                logger.error(f"Failed to import Band module: {e}")
                raise

    def _resolve_model_path(self) -> str:
        """Resolve model path to absolute path."""
        if os.path.isabs(self.model_path):
            return self.model_path

        # Try relative to band_filter_module first (default location)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        module_dir = os.path.join(current_dir, 'band_filter_module')
        resolved = os.path.join(module_dir, self.model_path)
        if os.path.exists(resolved):
            return resolved

        # Try relative to filtering directory
        resolved = os.path.join(current_dir, self.model_path)
        if os.path.exists(resolved):
            return resolved

        # Return the module path as default
        return os.path.join(module_dir, self.model_path)

    def _yield_next_batch(self, task: AudioBatch) -> Generator[List[Dict[str, Any]], None, None]:
        """
        Yield batches of items from task.data for batched model inference.

        Mirrors BaseFilterStage.yield_next_batch from image filters: each batch
        has at most model_inference_batch_size items.

        Args:
            task: AudioBatch to process.

        Yields:
            Slices of task.data of length model_inference_batch_size (last may be smaller).
        """
        batch_size = max(1, self.model_inference_batch_size)
        for i in range(0, len(task.data), batch_size):
            yield task.data[i : i + batch_size]

    def _get_audio_for_item(
        self, item: Dict[str, Any], task_id: str
    ) -> Optional[Tuple[torch.Tensor, int]]:
        """
        Get (waveform, sample_rate) for an item, loading from file if needed.

        Updates item with waveform/sample_rate when loaded from file.
        Returns None if waveform is missing and file load fails.

        Args:
            item: Audio item dict (waveform, sample_rate, or audio_filepath).
            task_id: Task id for logging.

        Returns:
            (waveform, sample_rate) or None if unavailable.
        """
        waveform = item.get("waveform")
        sample_rate = item.get("sample_rate", 48000)

        if waveform is None:
            audio_filepath = item.get("audio_filepath")
            if audio_filepath and os.path.exists(audio_filepath):
                try:
                    waveform, sample_rate = _load_audio_file(audio_filepath)
                    item["waveform"] = waveform
                    item["sample_rate"] = sample_rate
                except Exception as e:
                    logger.error(f"[{task_id}] Failed to load audio file: {e}")
                    return None
            else:
                logger.warning(f"[{task_id}] No waveform or valid audio_filepath found")
                return None

        return (waveform, sample_rate)

    def process(self, task: AudioBatch) -> Optional[AudioBatch]:
        """
        Filter audio based on bandwidth classification.

        Uses the same structure as image aesthetic filter: process items in
        batches with one model call per batch (parallel inference within batch),
        then filter by band_value.

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

        # Phase 1: Process items in batches to generate band predictions
        for batch in self._yield_next_batch(task):
            items_with_audio: List[Tuple[Dict[str, Any], torch.Tensor, int]] = []
            for item in batch:
                audio = self._get_audio_for_item(item, task.task_id)
                if audio is not None:
                    waveform, sample_rate = audio
                    items_with_audio.append((item, waveform, sample_rate))

            if not items_with_audio:
                continue

            audio_data = [(w, sr) for (_, w, sr) in items_with_audio]
            try:
                predictions = self._predictor.predict_audio_batch(
                    audio_data, use_parallel=True
                )
            except Exception as e:
                logger.exception(f"[BandFilter] Batch prediction error: {e}")
                continue

            for (item, _, _), pred in zip(items_with_audio, predictions):
                if (
                    isinstance(pred, str)
                    and not pred.startswith("Error")
                    and pred in ("full_band", "narrow_band")
                ):
                    item["band_prediction"] = pred

        # Phase 2: Filter by band_value (same as image: score then filter)
        filtered_items = [
            item
            for item in task.data
            if item.get("band_prediction") == self.band_value
        ]

        total_items = len(task.data)
        passed_count = len(filtered_items)
        logger.info(
            f"[BandFilter] {task.task_id}: {passed_count}/{total_items} passed ({self.band_value})"
        )

        return AudioBatch(
            data=filtered_items,
            task_id=task.task_id,
            dataset_name=task.dataset_name,
            _metadata=task._metadata,
            _stage_perf=list(task._stage_perf),
        )
