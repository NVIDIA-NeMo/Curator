# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
from typing import Optional

import requests
import torch
import torch.nn as nn

from nemo_curator.image.classifiers.base import ImageClassifier
from nemo_curator.utils.file_utils import NEMO_CURATOR_HOME


# MLP code taken from LAION's CLIP-based-NSFW-Detector
# https://github.com/LAION-AI/CLIP-based-NSFW-Detector/blob/main/h14_nsfw_model.py
class H14_NSFW_Detector(nn.Module):
    def __init__(self, input_size=1024):
        super().__init__()
        self.input_size = input_size
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)


class NsfwClassifier(ImageClassifier):
    def __init__(
        self,
        embedding_column: str = "image_embedding",
        pred_column: str = "nsfw_score",
        batch_size: int = -1,
        model_path: Optional[str] = None,
    ) -> None:
        super().__init__(
            model_name="nsfw_classifier",
            embedding_column=embedding_column,
            pred_column=pred_column,
            pred_type=float,
            batch_size=batch_size,
            embedding_size=1024,
        )

        if model_path is None:
            model_path = self._get_default_model()

        self.model_path = model_path

    @staticmethod
    def _get_default_model():
        weights_name = "h14_nsfw.pth"
        model_path = os.path.join(NEMO_CURATOR_HOME, weights_name)
        os.makedirs(NEMO_CURATOR_HOME, exist_ok=True)

        if not os.path.exists(model_path):
            url = f"https://github.com/LAION-AI/CLIP-based-NSFW-Detector/blob/main/{weights_name}?raw=true"
            r = requests.get(url)

            with open(model_path, "wb") as f:
                f.write(r.content)

        return model_path

    def load_model(self, device):
        model = H14_NSFW_Detector(input_size=self.embedding_size).to(device)
        weights = torch.load(self.model_path, map_location=torch.device("cpu"))
        model.load_state_dict(weights)
        model.eval()
        model = self.configure_forward(model)

        return model

    def configure_forward(self, model):
        original_forward = model.forward

        def custom_forward(*args, **kwargs):
            return original_forward(*args, **kwargs).squeeze()

        model.forward = custom_forward
        return model

    def postprocess(self, series):
        new_series = series.list.leaves
        new_series.index = series.index
        return new_series
