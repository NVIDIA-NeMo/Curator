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

import librosa
import numpy as np
import pyloudnorm as pyln
from loguru import logger


class AudioFeatureExtractor:
    """Audio feature extractor for band energy classification."""

    BAND_DEFINITIONS = {
        "low1": (0, 1000),
        "low2": (1000, 2000),
        "low3": (2000, 3000),
        "mid1": (3000, 4000),
        "mid2": (4000, 5000),
        "mid3": (5000, 6000),
        "mid4": (6000, 7000),
        "mid5": (7000, 8000),
        "mid6": (8000, 9000),
        "mid7": (9000, 10000),
        "mid8": (10000, 11000),
        "mid9": (11000, 12000),
        "mid10": (12000, 13000),
        "high": (13000, 14000),
        "high1": (14000, 15000),
        "high2": (15000, 16000),
        "high3": (16000, 17000),
        "high4": (17000, 18000),
        "high5": (18000, 19000),
        "high6": (19000, 20000),
        "high7": (20000, 21000),
        "high8": (21000, 22000),
        "high9": (22000, 23000),
        "high10": (23000, 24000),
    }

    @staticmethod
    def get_empty_feature_dict() -> dict[str, float]:
        """
        Create an empty feature dictionary with all band energy keys set to 0.0.

        Returns:
            Dictionary with all band energy feature keys initialized to 0.0
        """
        return {f"band_energy_{band}": 0.0 for band in AudioFeatureExtractor.BAND_DEFINITIONS}

    @staticmethod
    def calculate_band_energy(y: np.ndarray, sr: int) -> dict[str, float]:
        """
        Calculate energy in different frequency bands with LUFS normalization.

        Args:
            y: Audio time series
            sr: Sampling rate

        Returns:
            Dictionary with energy levels for each frequency band
        """
        band_energy = {}

        try:
            if y.ndim > 1 and y.shape[0] > 1:
                y = np.mean(y, axis=0)

            if y.ndim > 1:
                y = y.squeeze()

            meter = pyln.Meter(sr)
            original_loudness = meter.integrated_loudness(y)

            if original_loudness > -100.0:
                normalized_audio = pyln.normalize.loudness(y, original_loudness, -25.0)
            else:
                normalized_audio = y

            n_fft = 4096
            D = np.abs(librosa.stft(normalized_audio, n_fft=n_fft))
            power = D ** 2
            freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

            max_power = np.max(power)
            global_max_power = max_power if max_power > 0 else 1e-10

            for band, (f_min, f_max) in AudioFeatureExtractor.BAND_DEFINITIONS.items():
                mask = (freqs >= f_min) & (freqs < f_max)
                if np.any(mask):
                    mean_power = np.mean(power[mask, :])
                    band_energy[f"band_energy_{band}"] = float(librosa.power_to_db(mean_power, ref=global_max_power))

                    if f_min >= 10000:
                        attenuation_factor = (f_min - 10000) / 14000 * 12
                        band_energy[f"band_energy_{band}"] -= attenuation_factor
                else:
                    band_energy[f"band_energy_{band}"] = -120.0
        except Exception as e:  # noqa: BLE001
            logger.error(f"Error calculating band energy: {e}")
            for band in AudioFeatureExtractor.BAND_DEFINITIONS:
                band_energy[f"band_energy_{band}"] = -120.0

        return band_energy

    @staticmethod
    def features_dict_to_vector(features_dict: dict[str, float]) -> tuple[np.ndarray, list[str]]:
        """
        Convert a dictionary of features to a feature vector.

        Args:
            features_dict: Dictionary of feature name-value pairs

        Returns:
            Tuple of (feature_vector, feature_names)
        """
        if not features_dict:
            features_dict = AudioFeatureExtractor.get_empty_feature_dict()

        feature_names = sorted(features_dict.keys())

        feature_vector = []
        for name in feature_names:
            value = features_dict[name]
            if np.isnan(value):
                logger.warning(f"NaN value found for feature {name}, replacing with 0")
                value = 0.0
            feature_vector.append(value)

        return np.array(feature_vector), feature_names

    @staticmethod
    def extract_band_features_from_waveform(waveform, sr: int) -> dict[str, float]:
        """
        Extract band energy features from a waveform tensor/array.

        Args:
            waveform: Audio waveform tensor/array
            sr: Sample rate of the waveform

        Returns:
            Dictionary of band energy feature names and values
        """
        try:
            if hasattr(waveform, "cpu"):
                y = waveform.cpu().numpy()
            elif hasattr(waveform, "numpy"):
                y = waveform.numpy()
            else:
                y = waveform

            if y.ndim > 1 and y.shape[0] > 1:
                y = np.mean(y, axis=0)

            if y.ndim > 1:
                y = y.squeeze()

            all_features = AudioFeatureExtractor.calculate_band_energy(y, sr)

            for key in all_features:
                if np.isnan(all_features[key]):
                    logger.warning(f"NaN value found for feature {key}, replacing with 0")
                    all_features[key] = 0.0

            return all_features

        except Exception as e:  # noqa: BLE001
            logger.error(f"Error processing waveform: {e}")
            return AudioFeatureExtractor.get_empty_feature_dict()
