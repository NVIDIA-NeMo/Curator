"""Model NSFW Classifier."""

import os
import zipfile
from pathlib import Path

import numpy as np
import numpy.typing as npt
import requests
import torch
from torch import nn

from .base import ModelInterface

_NSFW_MODEL_ID = "laion/clip-autokeras-binary-nsfw"


class Normalization(nn.Module):
    """Normalization layer for NSFW model.
    
    Applies normalization to input tensors using pre-computed mean and variance.
    """

    def __init__(self, shape: list[int]) -> None:
        """Initialize the normalization layer.

        Args:
            shape: Shape of the normalization parameters.
        """
        super().__init__()
        self.register_buffer("mean", torch.zeros(shape))
        self.register_buffer("variance", torch.ones(shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply normalization to input tensor.

        Args:
            x: Input tensor to normalize.

        Returns:
            Normalized tensor.
        """
        return (x - self.mean) / self.variance.sqrt()


class NSFWModel(nn.Module):
    """NSFW detection model.

    A neural network that processes CLIP embeddings to predict NSFW scores.
    Based on LAION's CLIP-based-NSFW-Detector.
    """

    def __init__(self) -> None:
        """Initialize the NSFW model.

        Args:
            None
        """
        super().__init__()
        self.norm = Normalization([768])
        self.linear_1 = nn.Linear(768, 64)
        self.linear_2 = nn.Linear(64, 512)
        self.linear_3 = nn.Linear(512, 256)
        self.linear_4 = nn.Linear(256, 1)
        self.act = nn.ReLU()
        self.act_out = nn.Sigmoid()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the NSFW model.

        Args:
            x: Input embeddings tensor.

        Returns:
            NSFW probability scores.
        """
        x = self.norm(x)
        x = self.act(self.linear_1(x))
        x = self.act(self.linear_2(x))
        x = self.act(self.linear_3(x))
        return self.act_out(self.linear_4(x))  # type: ignore[no-any-return]


class NSFWScorer(ModelInterface):
    """Public interface for NSFW scoring of image embeddings.

    This class provides a standardized interface for scoring the likelihood
    of images containing sexually explicit material using a pre-trained model.
    """

    def __init__(self, model_dir: str) -> None:
        """Initialize the NSFW scorer interface."""
        super().__init__()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float32
        self.model_dir = model_dir
        # These will be initialized in setup()
        self.nsfw_model = None
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
        """Get the model ID names associated with this NSFW scorer.

        Returns:
            A list containing the model ID for NSFW scoring.
        """
        return [_NSFW_MODEL_ID]

    def setup(self) -> None:
        """Set up the NSFW scoring model by loading weights."""
        weights_filename = "clip_autokeras_binary_nsfw.pth"
        self.weights_path = str(Path(self.model_dir) / self.model_id_names[0] / weights_filename)

        # Download weights if they don't exist
        if not os.path.exists(self.weights_path):
            self._download_weights()

        self.nsfw_model = NSFWModel()
        state_dict = torch.load(self.weights_path, map_location=torch.device("cpu"))
        self.nsfw_model.load_state_dict(state_dict)
        self.nsfw_model.to(self.device)
        self.nsfw_model.eval()

    def _download_weights(self) -> None:
        """Download NSFW model weights from LAION repository."""
        model_dir_path = Path(self.model_dir) / self.model_id_names[0]
        model_dir_path.mkdir(parents=True, exist_ok=True)

        url = "https://github.com/LAION-AI/CLIP-based-NSFW-Detector/files/10250461/clip_autokeras_binary_nsfw.zip"
        response = requests.get(url)  # noqa: S113

        raw_zip_path = model_dir_path / "nsfw.zip"
        with open(raw_zip_path, "wb") as f:
            f.write(response.content)

        with zipfile.ZipFile(raw_zip_path, "r") as f:
            f.extractall(model_dir_path)

        # Remove the zip file after extraction
        raw_zip_path.unlink()

    @torch.no_grad()
    def __call__(self, embeddings: torch.Tensor | npt.NDArray[np.float32]) -> torch.Tensor:
        """Score the NSFW likelihood of input embeddings.

        Args:
            embeddings: Input embeddings as either a torch tensor or numpy array.

        Returns:
            NSFW probability scores for each input embedding.
        """
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.from_numpy(embeddings.copy())
        return self.nsfw_model(embeddings.to(self.device)).squeeze(1)  # type: ignore[no-any-return]
