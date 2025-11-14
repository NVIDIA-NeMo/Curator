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

import json
import pathlib
from pathlib import Path
from typing import Final

import cv2

# Load config from the internvideo2_multi_modality package
import internvideo2_multi_modality
import numpy as np
import numpy.typing as npt
import torch
from easydict import EasyDict
from internvideo2_multi_modality import InternVideo2_Stage2_visual, interpolate_pos_embed_internvideo2_new
from loguru import logger
from transformers import AutoTokenizer, PreTrainedTokenizer

from nemo_curator.models.base import ModelInterface
from nemo_curator.utils.hf_download_utils import download_model_from_hf

_MODEL_CONFIG_PATH = (
    pathlib.Path(internvideo2_multi_modality.__file__).parent / "configs" / "internvideo2_mm_config_model.json"
)
_BERT_CONFIG_PATH = pathlib.Path(internvideo2_multi_modality.__file__).parent / "configs" / "config_bert_large.json"
INTERNVIDEO2_MODEL_ID: Final = "OpenGVLab/InternVideo2-Stage2_1B-224p-f4"
INTERNVIDEO2_MODEL_FILE: Final = "InternVideo2-stage2_1b-224p-f4.pt"
INTERNVIDEO2_MODEL_REVISION: Final = "4362e1f"
BERT_MODEL_ID: Final = "google-bert/bert-large-uncased"
BERT_MODEL_REVISION: Final = "6da4b6a"


class _InternVideo2Stage2Wrapper(InternVideo2_Stage2_visual):
    """Wrapper class for InternVideo2 model that inherits from the original implementation.

    This wrapper extends the original InternVideo2_Stage2_visual class and overrides
    only the methods needed for inference, while keeping all the original functionality
    intact.
    """

    def __init__(self, config: EasyDict, tokenizer: PreTrainedTokenizer, *, is_pretrain: bool = True) -> None:
        # Call the parent class constructor
        super().__init__(config, tokenizer, is_pretrain)

        # Override the freeze behavior to always freeze encoders for inference
        self.freeze_vision()
        self.freeze_text()

    def encode_vision(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode image / videos as features for inference.

        Overrides the parent method to return only the basic features needed for inference,
        without the complex teacher-student outputs.

        Args:
            image (torch.Tensor): The input images.

        Returns: tuple.
            torch.Tensor: The output features. Shape: [B,N,C].
            torch.Tensor: The pooled output features. Shape: [B,1,C].

        """
        t = image.shape[1]
        use_image = t == 1
        image = image.permute(0, 2, 1, 3, 4).to(self.dtype)  # [B,T,C,H,W] -> [B,C,T,H,W]

        # Call parent method with test=True to get only basic features
        vision_embeds, pooled_vision_embeds, _, _ = self.vision_encoder(image, None, use_image)
        return vision_embeds, pooled_vision_embeds

    def get_vid_feat(self, frames: torch.Tensor) -> torch.Tensor:
        """Get the video features for the given frames.

        Args:
            frames (torch.Tensor): The input frames. Shape: [B,T,C,H,W].

        Returns:
            torch.Tensor: the output features. Shape: [B,N,C].

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
            torch.Tensor: the output features. Shape: [B,N,C].

        """
        assert self.tokenizer, "tokenizer is not initialized"  # noqa: S101
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
        """Predict labels based on video and text features.

        Args:
            vid_feat: Video features
            txt_feat: Text features
            top: Number of top predictions to return

        Returns:
            Tuple of (probabilities, indices)
        """
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


def _setup_internvideo2(config: EasyDict) -> _InternVideo2Stage2Wrapper:
    """Set up internvideo2 model.

    Args:
        config (EasyDict): The model config.

    Returns:
        _InternVideo2Stage2Wrapper: The InternVideo2 Stage2 model wrapper.

    """
    if "bert" in config.model.text_encoder.name:
        tokenizer = AutoTokenizer.from_pretrained(config.model.text_encoder.pretrained, local_files_only=True)
        model = _InternVideo2Stage2Wrapper(config=config, tokenizer=tokenizer, is_pretrain=True)
    else:
        error_msg = f"Not implemented: {config.model.text_encoder.name}"
        raise ValueError(error_msg)

    if config.get("compile_model", False):
        torch.set_float32_matmul_precision("high")
        model = torch.compile(model)  # type: ignore[assignment]

    model.to_empty(device=torch.device(config.device))  # TODO: confirm the to_empty is needed
    # Load checkpoint before moving to device
    model_without_ddp = model
    if (
        config.pretrained_path.strip() and (pathlib.Path(config.pretrained_path).is_file())
    ) or "s3://" in config.pretrained_path:
        checkpoint = torch.load(config.pretrained_path, map_location="cpu", weights_only=True)
        try:
            # checkpoint["module"] : This is a deepspeed stage 1 model
            state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint["module"]
        except Exception as e:  # noqa: BLE001
            logger.error(f"Error: error loading internvideo2 checkpoint: {e}")
            state_dict = checkpoint

        if config.get("origin_num_frames", None) is not None:
            a = len(state_dict)
            interpolate_pos_embed_internvideo2_new(
                state_dict,
                model_without_ddp.vision_encoder,
                orig_t_size=config.origin_num_frames,
            )
            assert a == len(state_dict), state_dict.keys()  # noqa: S101

        _ = model_without_ddp.load_state_dict(state_dict, strict=True)
        logger.debug("InternVideo2 model loaded successfully")

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
    """Actual outside-facing model using the wrapper class."""

    def __init__(self, model_dir: str, utils_only: bool = False) -> None:
        """Initialize the InternVideo2MultiModality model.

        Args:
            utils_only: Whether to only initialize utility functions.

        """
        super().__init__()
        self.model_dir = Path(model_dir)
        self.utils_only = utils_only
        self._model: _InternVideo2Stage2Wrapper | None = None

    def model_id_names(self) -> list[str]:
        return [INTERNVIDEO2_MODEL_ID, BERT_MODEL_ID]

    def setup(self) -> None:
        """Set up the InternVideo2MultiModality model.

        This method initializes the model and its configuration for video and text processing.
        It also sets up the normalization parameters for video frames.

        """

        self.weights_path = str(self.model_dir / INTERNVIDEO2_MODEL_ID / INTERNVIDEO2_MODEL_FILE)
        self.bert_path = str(self.model_dir / BERT_MODEL_ID)
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
        assert self._model is not None  # noqa: S101
        return self._model.get_vid_feat(frames_tensor)

    def get_text_embedding(self, text: str) -> torch.Tensor:
        """Get the text embedding for the given text.

        Args:
            text: The input text.

        Returns:
            The text embedding.

        """
        assert self._model is not None  # noqa: S101
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
        assert self._model is not None  # noqa: S101
        probs, idxs = self._model.predict_label(video_embd, text_embds_tensor, top=count)
        return probs.cpu().numpy()[0].tolist(), idxs.cpu().long().numpy()[0].tolist()

    @classmethod
    def download_weights_on_node(cls, model_dir: str) -> None:
        """Download the weights for the InternVideo2 model on the node."""
        model_dir_path = Path(model_dir) / INTERNVIDEO2_MODEL_ID
        model_dir_path.mkdir(parents=True, exist_ok=True)
        if not model_dir_path.exists() or not any(model_dir_path.glob("*.pt")):
            download_model_from_hf(
                model_id=INTERNVIDEO2_MODEL_ID,
                local_dir=model_dir_path,
                revision=INTERNVIDEO2_MODEL_REVISION,
            )
            logger.info(f"InternVideo2 weights downloaded to: {model_dir_path}")

        # Download Bert weights
        bert_model_dir_path = Path(model_dir) / BERT_MODEL_ID
        bert_model_dir_path.mkdir(parents=True, exist_ok=True)
        if bert_model_dir_path.exists() and any(bert_model_dir_path.glob("*.safetensors")):
            return
        download_model_from_hf(
            model_id=BERT_MODEL_ID,
            local_dir=bert_model_dir_path,
            ignore_patterns=[
                "*.msgpack",
                "*.bin",
                "*.ot",
                "*.h5",
                "*.gz",
            ],  # Ignore all weight files except safetensors
            revision=BERT_MODEL_REVISION,
        )
        logger.info(f"Bert weights downloaded to: {bert_model_dir_path}")
