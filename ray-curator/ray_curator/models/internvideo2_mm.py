# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Model Internvideo2."""

import json
import pathlib
from typing import Final

import cv2
import numpy as np
import numpy.typing as npt
import torch
from easydict import EasyDict  # type: ignore[import-untyped]
from loguru import logger
from torch import nn
from transformers import BatchEncoding, PreTrainedTokenizer  # type: ignore[attr-defined]

from ray_curator.models.model_interface import ModelInterface

from .internvideo2_multi_modality.bert.builder import build_bert
from .internvideo2_multi_modality.bert.tokenization_bert import BertTokenizer
from .internvideo2_multi_modality.internvideo2.internvideo2 import (
    pretrain_internvideo2_1b_patch14_224,
)
from .internvideo2_multi_modality.internvideo2.pos_embed import (
    interpolate_pos_embed_internvideo2_new,
)

CONTAINER_PATHS_CODE_DIR = pathlib.Path("/home/aot/codebase/Curator")
_MODEL_CONFIG_PATH = CONTAINER_PATHS_CODE_DIR / pathlib.Path(
    "ray-curator/ray_curator/models/configs/internvideo2_mm_config_model.json",
)
_BERT_CONFIG_PATH = CONTAINER_PATHS_CODE_DIR / pathlib.Path(
    "ray-curator/ray_curator/models/configs/internvideo2_mm_config_bert.json",
)
INTERNVIDEO2_MODEL_ID: Final = "OpenGVLab/InternVideo2-Stage2_1B-224p-f4"
INTERNVIDEO2_MODEL_FILE: Final = "InternVideo2-stage2_1b-224p-f4.pt"
BERT_MODEL_ID: Final = "google-bert/bert-large-uncased"
BERT_MODEL_FILES: Final = [
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
]


def get_local_dir_for_weights_name(weights_name: str) -> pathlib.Path:
    """Get the local directory for the weights name."""
    return pathlib.Path("/mnt/mint/models/cosmos_curator_local_workspace/models") / weights_name

class _InternVideo2Stage2(nn.Module):
    """Wrapper class for InternVideo2 model."""

    def __init__(self, config: EasyDict, tokenizer: PreTrainedTokenizer, *, is_pretrain: bool = True) -> None:
        super().__init__()

        self.config = config
        self.tokenizer = tokenizer

        self.is_pretrain = is_pretrain
        self.vision_width = config.model.vision_encoder.clip_embed_dim
        self.text_width = config.model.text_encoder.d_model
        self.embed_dim = config.model.embed_dim

        # create modules.
        self.vision_encoder = self.build_vision_encoder()
        self.freeze_vision()

        self.text_encoder = self.build_text_encoder()
        self.freeze_text()

        self.vision_proj = nn.Linear(self.vision_width, self.embed_dim)
        self.text_proj = nn.Linear(self.text_width, self.embed_dim)

    def freeze_vision(self) -> None:
        """Freeze vision encoder."""
        for p in self.vision_encoder.parameters():
            p.requires_grad = False

    def freeze_text(self) -> None:
        """Freeze text encoder."""
        for p in self.text_encoder.parameters():
            p.requires_grad = False

    @property
    def dtype(self) -> torch.dtype:
        """Get the dtype of the model."""
        return self.vision_encoder.patch_embed.proj.weight.dtype  # type: ignore[union-attr, return-value]

    def encode_vision(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode image / videos as features.

        Args:
            image (torch.Tensor): The input images.

        Returns: tuple.
            torch.Tensor: The output features. Shape: [B,N,C].
            torch.Tensor: The pooled output features. Shape: [B,1,C].

        """
        t = image.shape[1]
        use_image = t == 1
        image = image.permute(0, 2, 1, 3, 4).to(self.dtype)  # [B,T,C,H,W] -> [B,C,T,H,W]
        vision_embeds, pooled_vision_embeds, _, _ = self.vision_encoder(image, None, use_image)
        return vision_embeds, pooled_vision_embeds

    def encode_text(self, text: BatchEncoding) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode text.

        Args:
            text (BatchEncoding): The output of huggingface's `PreTrainedTokenizer`. contains keys:
                - input_ids (torch.Tensor): Token ids to be fed to a model. Shape: [B,L].
                - attention_mask (torch.Tensor): The mask indicate padded tokens. Shape: [B,L]. 0 is padded token.
                - other keys refer to "https://huggingface.co/docs/transformers/v4.21.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer.__call__".
        Returns: tuple.
            torch.Tensor: The features of all tokens. Shape: [B,L,C].
            torch.Tensor: The pooled features. Shape: [B,C].

        """
        text_output = self.get_text_encoder()(
            text.input_ids,
            attention_mask=text.attention_mask,
            return_dict=True,
            mode="text",
        )
        text_embeds = text_output.last_hidden_state
        pooled_text_embeds = text_embeds[:, 0]
        return text_embeds, pooled_text_embeds

    def build_vision_encoder(self) -> nn.Module:
        """Build vision encoder.

        Returns:
            nn.Module: the vision encoder.

        """
        encoder_name = self.config.model.vision_encoder.name

        if encoder_name == "pretrain_internvideo2_1b_patch14_224":
            vision_encoder = pretrain_internvideo2_1b_patch14_224(self.config.model)
        else:
            error_msg = f"Not implemented: {encoder_name}"
            raise ValueError(error_msg)

        # parameters for mask
        img_size = self.config.model.vision_encoder.img_size
        num_frames = self.config.model.vision_encoder.num_frames
        tublet_size = self.config.model.vision_encoder.tubelet_size
        patch_size = self.config.model.vision_encoder.patch_size
        self.clip_img_size = self.config.model.vision_encoder.clip_input_resolution
        self.video_mask_type = self.config.model.vision_encoder.video_mask_type
        self.video_window_size = (
            num_frames // tublet_size,
            img_size // patch_size,
            img_size // patch_size,
        )
        self.video_mask_ratio = self.config.model.vision_encoder.video_mask_ratio
        self.image_mask_type = self.config.model.vision_encoder.image_mask_type
        self.image_window_size = (1, img_size // patch_size, img_size // patch_size)
        self.image_mask_ratio = self.config.model.vision_encoder.image_mask_ratio

        return vision_encoder  # type: ignore[no-any-return]

    def build_text_encoder(self) -> nn.Module:
        """Build text_encoder and possiblly video-to-text multimodal fusion encoder.

        Returns:
            nn.Module: the text encoder.

        """
        encoder_name = self.config.model.text_encoder.name

        if "bert" in encoder_name:
            text_encoder = build_bert(
                self.config.model,
                self.is_pretrain,
                self.config.gradient_checkpointing,
            )
        else:
            error_msg = f"Not implemented: {encoder_name}"
            raise ValueError(error_msg)

        return text_encoder  # type: ignore[no-any-return]

    def get_text_encoder(self) -> nn.Module:
        """Get text encoder, used for text and cross-modal encoding.

        Returns:
            nn.Module: the text encoder.

        """
        encoder = self.text_encoder
        return encoder.bert if hasattr(encoder, "bert") else encoder  # type: ignore[return-value]

    def get_vid_feat(self, frames: torch.Tensor) -> torch.Tensor:
        """Get the video features for the given frames.

        Args:
            frames (torch.Tensor): The input frames. Shape: [B,T,C,H,W].

        Returns:
            torch.Tensor: the output featuers. Shape: [B,N,C].

        """
        with torch.no_grad():
            _, vfeat = self.encode_vision(frames)
            vfeat = self.vision_proj(vfeat)
            vfeat /= vfeat.norm(dim=-1, keepdim=True)
        return vfeat

    def get_txt_feat(self, text: str) -> torch.Tensor:
        """Get the text features for the given text.

        Args:
            text (str): The input text.

        Returns:
            torch.Tensor: the output featuers. Shape: [B,N,C].

        """
        assert self.tokenizer, "tokenizer is not initialized"
        with torch.no_grad():
            text_for_encoder = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.config.max_txt_l,
                return_tensors="pt",
            ).to(torch.device(self.config.device))
            _, tfeat = self.encode_text(text_for_encoder)
            tfeat = self.text_proj(tfeat)
            tfeat /= tfeat.norm(dim=-1, keepdim=True)
        return tfeat

    def predict_label(
        self,
        vid_feat: torch.Tensor,
        txt_feat: torch.Tensor,
        top: int = 5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        label_probs = (100.0 * vid_feat @ txt_feat.T).softmax(dim=-1)
        top_probs, top_labels = label_probs.float().cpu().topk(top, dim=-1)
        return top_probs, top_labels


def _create_config(model_pt: str, bert_path: str) -> EasyDict:
    """Create model config.

    Args:
        model_pt (str): The path to the model checkpoint file.
        bert_path (str): Path to Bert

    Returns:
        EasyDict: The model config.

    """
    with pathlib.Path(_MODEL_CONFIG_PATH).open() as fin:
        config = json.load(fin)
    config["pretrained_path"] = model_pt
    config["model"]["vision_encoder"]["pretrained"] = model_pt
    config["model"]["text_encoder"]["config"] = _BERT_CONFIG_PATH
    config["model"]["text_encoder"]["pretrained"] = bert_path
    return EasyDict(config)


def _setup_internvideo2(config: EasyDict) -> _InternVideo2Stage2:
    """Set up internvideo2 model.

    Args:
        config (EasyDict): The model config.

    Returns:
        InternVideo2Stage2: The InternVideo2 Stage2 model.

    """
    if "bert" in config.model.text_encoder.name:
        tokenizer = BertTokenizer.from_pretrained(config.model.text_encoder.pretrained, local_files_only=True)
        model = _InternVideo2Stage2(config=config, tokenizer=tokenizer, is_pretrain=True)
    else:
        error_msg = f"Not implemented: {config.model.text_encoder.name}"
        raise ValueError(error_msg)

    if config.get("compile_model", False):
        torch.set_float32_matmul_precision("high")
        model = torch.compile(model)  # type: ignore[assignment]

    model.to_empty(device=torch.device(config.device)) # TODO: confirm the to_empty is needed
    # Load checkpoint before moving to device
    model_without_ddp = model
    if (
        config.pretrained_path.strip() and (pathlib.Path(config.pretrained_path).is_file())
    ) or "s3://" in config.pretrained_path:
        checkpoint = torch.load(config.pretrained_path, map_location="cpu", weights_only=True)
        print("checkpoint loaded")
        try:
            # checkpoint["module"] : This is a deepspeed stage 1 model
            state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint["module"]
        except Exception as e:  # noqa: BLE001
            logger.error(f"Error: error loading checkpoint: {e}")
            state_dict = checkpoint

        if config.get("origin_num_frames", None) is not None:
            a = len(state_dict)
            interpolate_pos_embed_internvideo2_new(
                state_dict,
                model_without_ddp.vision_encoder,
                orig_t_size=config.origin_num_frames,
            )
            assert a == len(state_dict), state_dict.keys()

        _ = model_without_ddp.load_state_dict(state_dict, strict=False)
        print("state_dict loaded")

    # Move to device after loading checkpoint
    model_without_ddp = model

    if config.get("use_bf16", False):
        model_without_ddp = model_without_ddp.to(torch.bfloat16)
    elif config.get("use_half_precision", False):
        model_without_ddp = model_without_ddp.to(torch.float16)
    else:
        model_without_ddp = model_without_ddp.to(torch.float32)

    model_without_ddp.eval()
    return model_without_ddp


class InternVideo2MultiModality(ModelInterface):
    """Actual outside-facing model."""

    def __init__(self, *, utils_only: bool = False) -> None:
        """Initialize the InternVideo2MultiModality model.

        Args:
            utils_only: Whether to only initialize utility functions.

        """
        super().__init__()
        self.utils_only = utils_only
        self._model: _InternVideo2Stage2 | None = None

    def id_file_mapping(self) -> dict[str, list[str] | None]:
        """Get the mapping of model IDs to their weight files.

        Returns:
            Dictionary mapping model IDs to their weight file paths.

        """
        return {
            INTERNVIDEO2_MODEL_ID: [INTERNVIDEO2_MODEL_FILE],
            BERT_MODEL_ID: BERT_MODEL_FILES,
        }

    @property
    def conda_env_name(self) -> str:
        """Get the conda environment name.

        Returns:
            The conda environment name.

        """
        return "unified"

    def setup(self) -> None:
        """Set up the InternVideo2MultiModality model.

        This method initializes the model and its configuration for video and text processing.
        It also sets up the normalization parameters for video frames.

        """
        model_dir = get_local_dir_for_weights_name(INTERNVIDEO2_MODEL_ID)
        self.weights_path = str(model_dir / INTERNVIDEO2_MODEL_FILE)
        self.bert_path = str(get_local_dir_for_weights_name(BERT_MODEL_ID))
        self._v_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self._v_std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
        self._config = _create_config(self.weights_path, self.bert_path)
        if not self.utils_only:
            self._model = _setup_internvideo2(self._config)
        else:
            self._model = None

    def _normalize(self, data: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
        return ((data / np.float32(255.0) - self._v_mean) / self._v_std).astype(np.float32)  # type: ignore[no-any-return]

    def _construct_frames(
        self,
        vid_list: list[npt.NDArray[np.uint8]],
        fnum: int = 8,
        target_size: tuple[int, int] = (224, 224),
    ) -> npt.NDArray[np.float32]:
        if len(vid_list) < fnum:
            logger.error(f"Frame count {len(vid_list)} is smaller than minimal requirement {fnum}")
            return np.empty(0, dtype=np.float32)
        step = len(vid_list) // fnum
        vid_list = vid_list[::step][:fnum]
        vid_list = [cv2.resize(x, target_size) for x in vid_list]  # type: ignore[misc]
        vid_tube1 = [np.expand_dims(self._normalize(x), axis=(0, 1)) for x in vid_list]
        vid_tube2 = np.concatenate(vid_tube1, axis=1)
        return np.transpose(vid_tube2, (0, 1, 4, 2, 3))

    def get_target_num_frames(self) -> int:
        """Get the target number of frames for the model.

        Returns:
            The target number of frames.

        """
        return self._config.get("num_frames", 8)  # type: ignore[no-any-return]

    def formulate_input_frames(self, frames: list[npt.NDArray[np.uint8]]) -> npt.NDArray[np.float32]:
        """Formulate input frames for the model.

        Args:
            frames: List of video frames.

        Returns:
            The formulated input frames.

        """
        fn = self.get_target_num_frames()
        size_t = self._config.get("size_t", 224)
        return self._construct_frames(frames, fnum=fn, target_size=(size_t, size_t))

    def encode_video_frames(self, iv2_frames: npt.NDArray[np.float32]) -> torch.Tensor:
        """Encode video frames for the model.

        Args:
            iv2_frames: The input video frames.

        Returns:
            The encoded video frames.

        """
        if iv2_frames.size == 0:
            return torch.empty(0)
        target_device = torch.device(self._config.device)
        frames_tensor = torch.from_numpy(iv2_frames).to(target_device).float()
        assert self._model is not None
        return self._model.get_vid_feat(frames_tensor)

    def get_text_embedding(self, text: str) -> torch.Tensor:
        """Get the text embedding for the given text.

        Args:
            text: The input text.

        Returns:
            The text embedding.

        """
        assert self._model is not None
        return self._model.get_txt_feat(text)

    def evaluate(self, video_embd: torch.Tensor, text_embds: list[torch.Tensor]) -> tuple[list[float], list[int]]:
        """Evaluate the model.

        Args:
            video_embd: The video embedding.
            text_embds: The text embeddings.

        Returns:
            The predicted probabilities and indices.

        """
        count = len(text_embds)
        text_embds_tensor = torch.cat(text_embds, 0)
        assert self._model is not None
        probs, idxs = self._model.predict_label(video_embd, text_embds_tensor, top=count)
        return probs.cpu().numpy()[0].tolist(), idxs.cpu().long().numpy()[0].tolist()
