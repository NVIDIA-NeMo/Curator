"""Model Aesthetics."""

from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from safetensors.torch import load_file
from torch import nn

from .base import ModelInterface

_AESTHETICS_MODEL_ID = "ttj/sac-logos-ava1-l14-linearMSE"


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
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float32
        self.model_dir = model_dir
        # These will be initialized in setup()
        self.mlp = None
        self.weights_path = None

    @property
    def conda_env_name(self) -> str:
        """Get the name of the conda environment required for this model.

        Returns:
            Name of the conda environment.

        """
        return "video_splitting"

    @property
    def model_id_names(self) -> list[str]:
        """Get the model ID names associated with this aesthetic scorer.

        Returns:
            A list containing the model ID for aesthetics scoring.

        """
        return [_AESTHETICS_MODEL_ID]

    def setup(self) -> None:
        """Set up the aesthetic scoring model by loading weights."""
        self.weights_path = str(Path(self.model_dir) / self.model_id_names[0] / "model.safetensors")

        self.mlp = MLP()
        state_dict = load_file(self.weights_path)
        self.mlp.load_state_dict(state_dict)
        self.mlp.to(self.device)
        self.mlp.eval()

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
