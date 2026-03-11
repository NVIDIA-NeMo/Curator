# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from loguru import logger
from safetensors.torch import load_file
from torch import nn

from nemo_curator.utils.hf_download_utils import download_model_from_hf

from .base import ModelInterface

_AESTHETICS_MODEL_ID = "ttj/sac-logos-ava1-l14-linearMSE"
_AESTHETICS_MODEL_REVISION = "1e77fa0"


class MLP(nn.Module):
    """Multi-layer perceptron.

    A neural network that processes embeddings to predict aesthetic scores.
    """

    def __init__(self) -> None:
        """Initialize the MLP.

        Args:
            None

        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    @torch.no_grad()
    def forward(self, embed: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP.

        Args:
            embed: Input embeddings tensor.

        Returns:
            Predicted aesthetic scores.

        """
        return self.layers(embed)  # type: ignore[no-any-return]


class AestheticScorer(ModelInterface):
    """Public interface for aesthetic scoring of video embeddings.

    This class provides a standardized interface for scoring the aesthetic quality
    of video embeddings using a pre-trained model.
    """

    def __init__(self, model_dir: str) -> None:
        """Initialize the aesthetic scorer interface."""
        super().__init__()
        # Use explicit CUDA device index for consistency with tests
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float32
        self.model_dir = model_dir
        # These will be initialized in setup()
        self.mlp = None

    @property
    def model_id_names(self) -> list[str]:
        """Get the model ID names associated with this aesthetic scorer.

        Returns:
            A list containing the model ID for aesthetics scoring.

        """
        return [_AESTHETICS_MODEL_ID]

    def setup(self) -> None:
        """Set up the aesthetic scoring model by loading weights."""
        self.mlp = MLP()
        state_dict = load_file(self.get_weights_path())
        self.mlp.load_state_dict(state_dict)
        self.mlp.to(self.device)
        self.mlp.eval()

    def get_weights_path(self) -> str:
        """Get the path to the weights for the aesthetic scorer."""
        return str(Path(self.model_dir) / _AESTHETICS_MODEL_ID / "model.safetensors")

    @torch.no_grad()
    def __call__(self, embeddings: torch.Tensor | npt.NDArray[np.float32]) -> torch.Tensor:
        """Score the aesthetics of input embeddings.

        Args:
            embeddings: Input embeddings as either a torch tensor or numpy array.

        Returns:
            Aesthetic scores for each input embedding.

        """
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.from_numpy(embeddings.copy())
        return self.mlp(embeddings.to(self.device)).squeeze(1)  # type: ignore[no-any-return]

    @classmethod
    def download_weights_on_node(cls, model_dir: str) -> None:
        """Download the weights for the aesthetic scorer on the node."""
        model_dir_path = Path(model_dir) / _AESTHETICS_MODEL_ID
        model_dir_path.mkdir(parents=True, exist_ok=True)
        model_file = model_dir_path / "model.safetensors"
        if model_file.exists():
            return
        download_model_from_hf(
            model_id=_AESTHETICS_MODEL_ID,
            local_dir=model_dir_path,
            filename="model.safetensors",
            revision=_AESTHETICS_MODEL_REVISION,
        )
        logger.info(f"Aesthetic scorer weights downloaded to: {model_file}")
