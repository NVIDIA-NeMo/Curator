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

import os
import time
import warnings
from dataclasses import dataclass, field

import joblib
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from loguru import logger

from .features import AudioFeatureExtractor

_HF_REPO_ID = "nvidia/nemocurator-speech-bandwidth-filter"
_HF_MODEL_FILENAME = "band_classifier_model_band_7000_samples.joblib"


class BandPredictor:
    """Class to predict band label (full_band/narrow_band) for audio waveforms."""

    def __init__(self, model_path: str, feature_cache_size: int = 100):
        """
        Initialize the band predictor.

        Args:
            model_path: Path to the trained model file
            feature_cache_size: Retained for backward compatibility; feature caching is disabled
        """
        self.model_path = model_path
        self.feature_cache_size = feature_cache_size
        self.model = None
        self.feature_cache: dict = {}

        self._load_model()

    def _load_model(self) -> None:
        """Load the model from disk."""
        try:
            start_time = time.time()
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                self.model = joblib.load(self.model_path)

                if hasattr(self.model, "estimators_"):
                    for estimator in self.model.estimators_:
                        if hasattr(estimator, "tree_") and not hasattr(estimator, "monotonic_cst"):
                            estimator.monotonic_cst = None
                elif hasattr(self.model, "tree_") and not hasattr(self.model, "monotonic_cst"):
                    self.model.monotonic_cst = None

            logger.info(f"Band prediction model loaded in {time.time() - start_time:.2f} seconds")
        except (OSError, RuntimeError, ValueError) as e:
            logger.error(f"Error loading model from {self.model_path}: {e}")
            raise

    def extract_features_from_audio(self, waveform: torch.Tensor, sample_rate: int) -> np.ndarray:
        """
        Extract band energy features directly from waveform tensor.

        Args:
            waveform: Audio waveform tensor [channels, samples]
            sample_rate: Sample rate of the audio

        Returns:
            Array of extracted features
        """
        try:
            feature_dict = AudioFeatureExtractor.extract_band_features_from_waveform(waveform, sample_rate)
            feature_vector, _ = AudioFeatureExtractor.features_dict_to_vector(feature_dict)

            if np.isnan(feature_vector).any():
                feature_vector = np.nan_to_num(feature_vector, nan=0.0)

        except Exception as e:  # noqa: BLE001
            logger.error(f"Error processing audio for features: {e}")
            return None
        else:
            return feature_vector

    def predict_audio(self, waveform: torch.Tensor, sample_rate: int) -> str:
        """
        Predict whether an audio waveform is full band or narrow band.

        Args:
            waveform: Audio waveform tensor [channels, samples]
            sample_rate: Sample rate of the audio

        Returns:
            Prediction result as a string ('full_band' or 'narrow_band')
        """
        if self.model is None:
            try:
                self._load_model()
            except Exception as e:  # noqa: BLE001
                return f"Error: Could not load model: {e}"

        try:
            features = self.extract_features_from_audio(waveform, sample_rate)
            if features is None:
                return None
            prediction = self.model.predict(features.reshape(1, -1))[0]
        except Exception as e:  # noqa: BLE001
            return f"Error during prediction: {e}"
        else:
            return "full_band" if prediction == 1 else "narrow_band"


@dataclass
class BandAdapter:
    """Lifecycle wrapper that loads a ``BandPredictor`` for ``BandFilterStage``.

    ``model_path`` points at a local ``.joblib`` file and skips the download;
    otherwise the model is fetched from the Hugging Face Hub into ``cache_dir``.
    """

    model_path: str | None = None
    cache_dir: str | None = None
    _predictor: BandPredictor | None = field(default=None, init=False, repr=False)

    def download_weights_on_node(self) -> None:
        """Download the default model; an explicit ``model_path`` needs no download."""
        if self.model_path is None:
            path = hf_hub_download(repo_id=_HF_REPO_ID, filename=_HF_MODEL_FILENAME, cache_dir=self.cache_dir)
            logger.info(f"Band filter model downloaded to {path}")

    def _resolve_model_path(self) -> str:
        if self.model_path is not None:
            if os.path.isfile(self.model_path) and os.path.getsize(self.model_path) > 0:
                return self.model_path
            msg = f"Band classifier model_path does not exist, is empty, or is not a file: {self.model_path!r}"
            raise FileNotFoundError(msg)
        return hf_hub_download(repo_id=_HF_REPO_ID, filename=_HF_MODEL_FILENAME, cache_dir=self.cache_dir)

    def load_model(self, *, num_gpus: int) -> None:  # noqa: ARG002 (CPU-only sklearn model)
        """Load the classifier; it always runs on CPU."""
        if self._predictor is None:
            self._predictor = BandPredictor(model_path=self._resolve_model_path())
            logger.info("Band predictor loaded successfully")

    def unload_model(self) -> None:
        """Release the classifier and CUDA cache."""
        if self._predictor is not None:
            self._predictor = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def predict_audio(self, waveform: torch.Tensor, sample_rate: int) -> str | None:
        """Return ``"full_band"``/``"narrow_band"`` (or the predictor's error string)."""
        if self._predictor is None:
            msg = "BandAdapter is not initialized; call load_model() first"
            raise RuntimeError(msg)
        return self._predictor.predict_audio(waveform, sample_rate)
