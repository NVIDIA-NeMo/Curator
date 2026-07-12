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

import time
import warnings

import joblib
import numpy as np
import torch
from loguru import logger

from .features import AudioFeatureExtractor


class BandPredictor:
    """Class to predict band label (full_band/narrow_band) for audio waveforms."""

    def __init__(self, model_path: str, feature_cache_size: int = 100):
        """
        Initialize the band predictor.

        Args:
            model_path: Path to the trained model file
            feature_cache_size: Number of feature vectors to cache
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
            if hasattr(waveform, "cpu"):
                w_np = waveform.cpu().numpy()
            else:
                w_np = waveform.numpy() if hasattr(waveform, "numpy") else waveform

            flat_data = w_np.ravel()
            step = max(1, len(flat_data) // 1000)
            cache_key = hash(flat_data[::step].tobytes())

            if cache_key in self.feature_cache:
                return self.feature_cache[cache_key]

            feature_dict = AudioFeatureExtractor.extract_band_features_from_waveform(waveform, sample_rate)
            feature_vector, _ = AudioFeatureExtractor.features_dict_to_vector(feature_dict)

            if np.isnan(feature_vector).any():
                feature_vector = np.nan_to_num(feature_vector, nan=0.0)

            if len(self.feature_cache) >= self.feature_cache_size:
                oldest_key = next(iter(self.feature_cache))
                del self.feature_cache[oldest_key]

            self.feature_cache[cache_key] = feature_vector

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
