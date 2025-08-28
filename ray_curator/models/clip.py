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
from typing import Final

import numpy as np
import numpy.typing as npt
import torch
from loguru import logger
from torchvision import transforms  # type: ignore[import-untyped]
from transformers import CLIPModel, CLIPProcessor

from nemo_curator.utils.hf_download_utils import download_model_from_hf

from .aesthetics import AestheticScorer
from .base import ModelInterface

_CLIP_MODEL_ID: Final = "openai/clip-vit-large-patch14"
_CLIP_MODEL_REVISION: Final = "32bd642"


class CLIPImageEmbeddings(ModelInterface):
    """Interface for generating CLIP image embeddings from input images."""

    def __init__(self, model_dir: str) -> None:
        """Initialize the CLIPImageEmbeddings model."""
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float32
        self.model_dir = model_dir
        # These will be initialized in setup()
        self.clip = None
        self.processor = None
        self.transforms = None

    @property
    def model_id_names(self) -> list[str]:
        """Get the model ID names.

        Returns:
            A list of model IDs used by this model.

        """
        return [_CLIP_MODEL_ID]

    def setup(self) -> None:
        """Set up the CLIPImageEmbeddings model."""
        weight_file = str(Path(self.model_dir) / self.model_id_names[0])
        self.clip = CLIPModel.from_pretrained(weight_file).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(weight_file)

        # torchvision transforms that match CLIP preprocessor_config.json
        # (https://huggingface.co/openai/clip-vit-base-patch32/blob/main/preprocessor_config.json)
        self.transforms = transforms.Compose(
            [
                transforms.Resize(
                    224,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                transforms.CenterCrop(224),
                transforms.ConvertImageDtype(torch.float32),  # scales [0, 255] to [0, 1]
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ],
        )

    @torch.no_grad()
    def __call__(self, images: torch.Tensor | npt.NDArray[np.uint8] | list[np.ndarray]) -> torch.Tensor:
        """Call the CLIPImageEmbeddings model.

        Args:
            images: The images to embed.

        Returns:
            The embeddings.

        """
        # If images is a numpy array (all images are the same size), run self.transforms
        # on the entire array, which is more efficient. Otherwise, e.g. images is a list
        # (images have different sizes), run self.processor.
        if isinstance(images, np.ndarray):
            # (N, H, W, C) -> (N, C, H, W)
            images = torch.from_numpy(images).permute(0, 3, 1, 2).to(self.device)
            inputs = self.transforms(images)
        else:
            inputs = self.processor(images=images, return_tensors="pt")["pixel_values"]
            inputs = inputs.to(self.device)

        embed = self.clip.get_image_features(pixel_values=inputs)

        # Normalize embeddings
        return embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)  # type: ignore[no-any-return]

    @classmethod
    def download_weights_on_node(cls, model_dir: str) -> None:
        """Download the weights for the CLIPImageEmbeddings model on the node."""
        model_dir_path = Path(model_dir) / _CLIP_MODEL_ID
        if model_dir_path.exists() and any(model_dir_path.glob("*.safetensors")):
            return
        model_dir_path.mkdir(parents=True, exist_ok=True)
        download_model_from_hf(
            model_id=_CLIP_MODEL_ID,
            local_dir=model_dir_path,
            revision=_CLIP_MODEL_REVISION,
            ignore_patterns=["*.msgpack", "*.bin", "*.ot", "*.h5", "*.gz"],
        )
        logger.info(f"CLIPImageEmbeddings weights downloaded to: {model_dir_path}")


class CLIPAestheticScorer(ModelInterface):
    """A model that chains CLIPImageEmbeddings and AestheticScorer models."""

    def __init__(self, model_dir: str) -> None:
        """Initialize the CLIPAestheticScorer model."""
        super().__init__()
        self.model_dir = model_dir
        self._clip_model: CLIPImageEmbeddings | None = None
        self._aesthetic_model: AestheticScorer | None = None

    @property
    def model_id_names(self) -> list[str]:
        """Get the model ID names.

        Returns:
            A list of model IDs used by this model.

        """
        return [_CLIP_MODEL_ID]

    def setup(self) -> None:
        """Set up the CLIPAestheticScorer model."""
        self._clip_model = CLIPImageEmbeddings(model_dir=self.model_dir)
        self._aesthetic_model = AestheticScorer(model_dir=self.model_dir)
        self._clip_model.setup()
        self._aesthetic_model.setup()

    def __call__(self, images: torch.Tensor | npt.NDArray[np.uint8]) -> torch.Tensor:
        """Call the CLIPAestheticScorer model.

        Args:
            images: The images to score.

        Returns:
            The scores.

        """
        if self._clip_model is None or self._aesthetic_model is None:
            msg = "CLIPAestheticScorer model not initialized"
            raise RuntimeError(msg)
        embeddings = self._clip_model(images)
        return self._aesthetic_model(embeddings)

    @classmethod
    def download_weights_on_node(cls, model_dir: str) -> None:
        """Download the weights for the CLIPAestheticScorer model on the node."""
        CLIPImageEmbeddings.download_weights_on_node(model_dir)
        AestheticScorer.download_weights_on_node(model_dir)
