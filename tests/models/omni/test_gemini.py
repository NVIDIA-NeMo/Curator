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

"""Unit tests for nemo_curator.models.omni.gemini."""

from nemo_curator.models.omni.base import NVInferenceModel, NVInferenceModelConfig
from nemo_curator.models.omni.gemini import DEFAULT_MODEL_ID, Gemini3Pro


class TestGemini3Pro:
    def test_default_model_id(self) -> None:
        model = Gemini3Pro(model_config=NVInferenceModelConfig())
        assert model.model_id == DEFAULT_MODEL_ID
        assert model.model_id == "gcp/google/gemini-3-pro"

    def test_custom_model_id(self) -> None:
        model = Gemini3Pro(model_config=NVInferenceModelConfig(), model_id="gcp/google/gemini-3-flash")
        assert model.model_id == "gcp/google/gemini-3-flash"

    def test_is_nvinference_model(self) -> None:
        model = Gemini3Pro(model_config=NVInferenceModelConfig())
        assert isinstance(model, NVInferenceModel)

    def test_not_loaded_initially(self) -> None:
        model = Gemini3Pro(model_config=NVInferenceModelConfig())
        assert model.is_loaded is False
