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

from nemo_curator.models.omni.base import NVInferenceModel, NVInferenceModelConfig

DEFAULT_MODEL_ID = "gcp/google/gemini-3-pro"


class Gemini3Pro(NVInferenceModel):
    """Gemini model using NVIDIA Inference API backend."""

    def __init__(
        self,
        model_config: NVInferenceModelConfig,
        model_id: str = DEFAULT_MODEL_ID,
    ) -> None:
        """Initialize Gemini with NVIDIA Inference API.

        Args:
            model_config: NVIDIA Inference API configuration.
            model_id: Model identifier for NVIDIA Inference API.
        """
        super().__init__(model_id, model_config)
