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
import zipfile
from typing import Optional

import requests
import torch
import torch.nn as nn

from nemo_curator.image.classifiers.base import ImageClassifier
from nemo_curator.utils.file_utils import NEMO_CURATOR_HOME


# MLP code taken from LAION's CLIP-based-NSFW-Detector
# https://github.com/LAION-AI/CLIP-based-NSFW-Detector/issues/7
class Normalization(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.register_buffer("mean", torch.zeros(shape))
        self.register_buffer("variance", torch.ones(shape))

    def forward(self, x):
        return (x - self.mean) / self.variance.sqrt()


class NSFWModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = Normalization([768])
        self.linear_1 = nn.Linear(768, 64)
        self.linear_2 = nn.Linear(64, 512)
        self.linear_3 = nn.Linear(512, 256)
        self.linear_4 = nn.Linear(256, 1)
        self.act = nn.ReLU()
        self.act_out = nn.Sigmoid()

    def forward(self, x):
        x = self.norm(x)
        x = self.act(self.linear_1(x))
        x = self.act(self.linear_2(x))
        x = self.act(self.linear_3(x))
        x = self.act_out(self.linear_4(x))
        return x


class NsfwClassifier(ImageClassifier):
    """
    NSFW Classifier is a small MLP trained on top of
    OpenAI's ViT-L CLIP image embeddings. It is used to assess the likelihood
    of images containing sexually explicit material.
    More information on the model can be found here:
    https://github.com/LAION-AI/CLIP-based-NSFW-Detector.
    """

    def __init__(
        self,
        embedding_column: str = "image_embedding",
        pred_column: str = "nsfw_score",
        batch_size: int = -1,
        model_path: Optional[str] = None,
    ) -> None:
        """
        Constructs the classifier.

        Args:
            embedding_column (str): The column name that stores the image
                embeddings.
            pred_column (str): The column name to be added where the nsfw
                scores will be stored.
            pred_type (Union[str, type]): The datatype of the pred_column.
            batch_size (int): If greater than 0, the image embeddings
                will be processed in batches of at most this size. If less than 0,
                all embeddings will be processed at once.
            model_path (Optional[str]): If specified, will load the model from the
                given path. If not specified, will default to being stored in
                NEMO_CURATOR_HOME.
        """
        super().__init__(
            model_name="nsfw_classifier",
            embedding_column=embedding_column,
            pred_column=pred_column,
            pred_type=float,
            batch_size=batch_size,
            embedding_size=768,
        )

        if model_path is None:
            model_path = self._get_default_model()

        self.model_path = model_path

    @staticmethod
    def _get_default_model():
        weights_name = "clip_autokeras_binary_nsfw.pth"
        model_path = os.path.join(NEMO_CURATOR_HOME, weights_name)
        os.makedirs(NEMO_CURATOR_HOME, exist_ok=True)

        if not os.path.exists(model_path):
            url = "https://github.com/LAION-AI/CLIP-based-NSFW-Detector/files/10250461/clip_autokeras_binary_nsfw.zip"
            r = requests.get(url)

            raw_zip_path = os.path.join(NEMO_CURATOR_HOME, "nsfw.zip")
            with open(raw_zip_path, "wb") as f:
                f.write(r.content)
            with zipfile.ZipFile(raw_zip_path, "r") as f:
                f.extractall(NEMO_CURATOR_HOME)

        return model_path

    def load_model(self, device):
        model = NSFWModel().to(device)
        weights = torch.load(self.model_path, map_location=torch.device("cpu"))
        model.load_state_dict(weights)
        model.eval()
        model = self._configure_forward(model)

        return model

    def _configure_forward(self, model):
        original_forward = model.forward

        def custom_forward(*args, **kwargs):
            return original_forward(*args, **kwargs).squeeze()

        model.forward = custom_forward
        return model

    def postprocess(self, series):
        new_series = series.list.leaves
        new_series.index = series.index
        return new_series
