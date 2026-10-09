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

from unittest.mock import patch

import numpy as np
import torch

from nemo_curator.stages.audio.filtering.band_filter_module.features import AudioFeatureExtractor
from nemo_curator.stages.audio.filtering.band_filter_module.predict import BandPredictor


def test_legacy_cache_constructor_is_accepted_without_caching() -> None:
    with patch.object(BandPredictor, "_load_model"):
        predictor = BandPredictor("model.joblib", feature_cache_size=1)
    waveform = torch.arange(8, dtype=torch.float32).reshape(1, 8)

    with (
        patch.object(
            AudioFeatureExtractor,
            "extract_band_features_from_waveform",
            return_value={"feature": 1.0},
        ) as extract,
        patch.object(
            AudioFeatureExtractor,
            "features_dict_to_vector",
            return_value=(np.array([1.0], dtype=np.float32), ["feature"]),
        ),
    ):
        predictor.extract_features_from_audio(waveform, 16000)
        predictor.extract_features_from_audio(waveform, 16000)

    assert predictor.feature_cache_size == 1
    assert predictor.feature_cache == {}
    assert extract.call_count == 2
