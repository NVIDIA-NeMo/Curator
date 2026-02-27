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

import importlib
import importlib.util
from collections.abc import Callable
from pathlib import Path
from typing import Final, Literal, cast

import numpy as np
import numpy.typing as npt
import torch
from loguru import logger
from transformers import AutoModel, AutoProcessor

from nemo_curator.utils.hf_download_utils import download_model_from_hf

from .base import ModelInterface

_COSMOS_EMBED1_VARIANTS_INFO: Final = {
    "224p": "nvidia/Cosmos-Embed1-224p",
    "336p": "nvidia/Cosmos-Embed1-336p",
    "448p": "nvidia/Cosmos-Embed1-448p",
}

COSMOS_EMBED1_MODEL_REVISION_INFO: Final = {
    "224p": "85f5627",
    "336p": "5d8309d",
    "448p": "9f4ff4d",
}


_HEAD_MASK_VECTOR_DIM: Final[int] = 1
_HEAD_MASK_MATRIX_DIM: Final[int] = 2
_HEAD_MASK_TARGET_DIM: Final[int] = 5


def _resolve_symbol(name: str) -> object | None:
    for module_name in ("transformers.pytorch_utils", "transformers.modeling_utils"):
        if importlib.util.find_spec(module_name) is None:
            continue
        module = importlib.import_module(module_name)
        if hasattr(module, name):
            return getattr(module, name)
    return None


def _fallback_apply_chunking_to_forward(
    forward_fn: Callable[..., object], chunk_size: int, chunk_dim: int, *input_tensors: torch.Tensor
) -> object:
    if len(input_tensors) == 0:
        msg = "input_tensors cannot be empty"
        raise ValueError(msg)

    if chunk_size <= 0:
        return forward_fn(*input_tensors)

    tensor_shape = input_tensors[0].shape[chunk_dim]
    if any(t.shape[chunk_dim] != tensor_shape for t in input_tensors):
        msg = "All input tensors must have the same shape on chunk_dim"
        raise ValueError(msg)
    if tensor_shape % chunk_size != 0:
        msg = "The dimension to be chunked must be a multiple of chunk_size"
        raise ValueError(msg)

    num_chunks = tensor_shape // chunk_size
    input_chunks = tuple(t.chunk(num_chunks, dim=chunk_dim) for t in input_tensors)
    output_chunks = tuple(forward_fn(*chunk_inputs) for chunk_inputs in zip(*input_chunks, strict=False))
    return torch.cat(output_chunks, dim=chunk_dim)


def _fallback_find_pruneable_heads_and_indices(
    heads: list[int] | set[int], n_heads: int, head_size: int, already_pruned_heads: set[int]
) -> tuple[set[int], torch.Tensor]:
    heads_to_prune = set(heads) - already_pruned_heads
    mask = torch.ones(n_heads, head_size)
    for head in heads_to_prune:
        adjusted_head = head - sum(1 for h in already_pruned_heads if h < head)
        mask[adjusted_head] = 0
    mask = mask.view(-1).contiguous().eq(1)
    index = torch.arange(mask.numel())[mask].long()
    return heads_to_prune, index


def _fallback_prune_linear_layer(layer: torch.nn.Linear, index: torch.Tensor, dim: int = 0) -> torch.nn.Linear:
    index = index.to(layer.weight.device)
    weight = layer.weight.index_select(dim, index).clone().detach()
    if layer.bias is not None:
        bias = layer.bias.clone().detach() if dim == 1 else layer.bias.index_select(0, index).clone().detach()
    new_size = list(layer.weight.size())
    new_size[dim] = index.size(0)
    new_layer = torch.nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None).to(layer.weight.device)
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(weight.contiguous())
    new_layer.weight.requires_grad = True
    if layer.bias is not None:
        new_layer.bias.requires_grad = False
        new_layer.bias.copy_(bias.contiguous())
        new_layer.bias.requires_grad = True
    return new_layer


def _ensure_all_tied_weights_keys(model: object) -> None:
    if hasattr(model, "all_tied_weights_keys"):
        return
    tied = getattr(model, "_tied_weights_keys", None)
    if tied is None:
        model.all_tied_weights_keys = {}
        return
    if isinstance(tied, dict):
        model.all_tied_weights_keys = tied
        return
    try:
        model.all_tied_weights_keys = dict.fromkeys(tied)
    except TypeError:
        model.all_tied_weights_keys = {}


def _convert_head_mask_to_5d(self: object, head_mask: torch.Tensor, num_hidden_layers: int) -> torch.Tensor:
    if head_mask.dim() == _HEAD_MASK_VECTOR_DIM:
        head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
    elif head_mask.dim() == _HEAD_MASK_MATRIX_DIM:
        head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
    if head_mask.dim() != _HEAD_MASK_TARGET_DIM:
        msg = f"head_mask.dim != {_HEAD_MASK_TARGET_DIM}, got {head_mask.dim()}"
        raise ValueError(msg)
    try:
        dtype = self.dtype
    except Exception:  # noqa: BLE001
        dtype = next(self.parameters()).dtype
    return head_mask.to(dtype=dtype)


def _get_head_mask(
    self: object, head_mask: torch.Tensor | None, num_hidden_layers: int, is_attention_chunked: bool = False
) -> torch.Tensor | list[None]:
    if head_mask is not None:
        head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
        if is_attention_chunked:
            head_mask = head_mask.unsqueeze(-1)
        return head_mask
    return [None] * num_hidden_layers


def _patch_modeling_utils_symbols(modeling_utils_module: object) -> None:
    if not hasattr(modeling_utils_module, "apply_chunking_to_forward"):
        symbol = _resolve_symbol("apply_chunking_to_forward") or _fallback_apply_chunking_to_forward
        modeling_utils_module.apply_chunking_to_forward = symbol
    if not hasattr(modeling_utils_module, "find_pruneable_heads_and_indices"):
        symbol = _resolve_symbol("find_pruneable_heads_and_indices") or _fallback_find_pruneable_heads_and_indices
        modeling_utils_module.find_pruneable_heads_and_indices = symbol
    if not hasattr(modeling_utils_module, "prune_linear_layer"):
        symbol = _resolve_symbol("prune_linear_layer") or _fallback_prune_linear_layer
        modeling_utils_module.prune_linear_layer = symbol


def _patch_pretrained_model_compat(pre_trained_model_cls: object) -> None:
    if not getattr(pre_trained_model_cls, "_nemo_curator_tied_weights_compat_patched", False):
        original_mark_tied = getattr(pre_trained_model_cls, "mark_tied_weights_as_initialized", None)
        if original_mark_tied is not None:

            def _patched_mark_tied_weights_as_initialized(self: object, *args: object, **kwargs: object) -> object:
                _ensure_all_tied_weights_keys(self)
                return original_mark_tied(self, *args, **kwargs)

            pre_trained_model_cls.mark_tied_weights_as_initialized = _patched_mark_tied_weights_as_initialized

        original_adjust_tied = getattr(pre_trained_model_cls, "_adjust_tied_keys_with_tied_pointers", None)
        if original_adjust_tied is not None:

            def _patched_adjust_tied_keys_with_tied_pointers(self: object, *args: object, **kwargs: object) -> object:
                _ensure_all_tied_weights_keys(self)
                return original_adjust_tied(self, *args, **kwargs)

            pre_trained_model_cls._adjust_tied_keys_with_tied_pointers = _patched_adjust_tied_keys_with_tied_pointers
        pre_trained_model_cls._nemo_curator_tied_weights_compat_patched = True

    if not hasattr(pre_trained_model_cls, "_convert_head_mask_to_5d"):
        pre_trained_model_cls._convert_head_mask_to_5d = _convert_head_mask_to_5d
    if not hasattr(pre_trained_model_cls, "get_head_mask"):
        pre_trained_model_cls.get_head_mask = _get_head_mask


def _patch_transformers_compat() -> None:
    """Patch moved Transformers symbols expected by older Cosmos remote code."""
    from transformers import modeling_utils
    from transformers.modeling_utils import PreTrainedModel

    _patch_modeling_utils_symbols(modeling_utils)
    _patch_pretrained_model_compat(PreTrainedModel)


def _patch_cosmos_embed1_modeling_vit(*, weights_dir: str, variant: Literal["224p", "336p", "448p"]) -> None:
    """Patch Cosmos-Embed1 ViT code to avoid meta-tensor .item() failures."""
    old_line = "dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule"
    new_line = 'dpr = torch.linspace(0, drop_path_rate, depth, device="cpu", dtype=torch.float32).tolist()'
    candidates = [
        Path(weights_dir) / "modeling_vit.py",
        Path.home()
        / ".cache/huggingface/modules/transformers_modules"
        / f"Cosmos_hyphen_Embed1_hyphen_{variant}"
        / "modeling_vit.py",
    ]

    for path in candidates:
        try:
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8")
            if old_line not in text:
                continue
            path.write_text(text.replace(old_line, new_line), encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Could not patch {path}: {e}")


class CosmosEmbed1(ModelInterface):
    """Cosmos-Embed1 embedding model."""

    def __init__(
        self,
        *,
        variant: Literal["224p", "336p", "448p"] = "336p",
        utils_only: bool = False,
        model_dir: str | None = None,
    ) -> None:
        """Initialize Cosmos-Embed1 model.

        Args:
            variant: Choose from "224p", "336p", "448p".
            utils_only: Whether to only initialize utility functions.

        """
        super().__init__()
        self.variant = variant
        self._weights_name = _COSMOS_EMBED1_VARIANTS_INFO[variant]
        self._weights_dir = str(Path(model_dir) / self._weights_name)
        self._utils_only = utils_only
        self._model: AutoModel | None = None

    @property
    def model_id_names(self) -> list[str]:
        """Get the model ID names.

        Returns:
            A list of model ID names.

        """
        return [self._weights_name]

    def setup(self) -> None:
        """Set up the Cosmos-Embed1 model.

        This method initializes the model and its configuration for processing video and text data.
        """
        logger.info("Setting up Cosmos-Embed1 model")
        _patch_transformers_compat()
        _patch_cosmos_embed1_modeling_vit(weights_dir=self._weights_dir, variant=self.variant)
        if not Path(self._weights_dir).exists():
            exception = f"Weights directory {self._weights_dir} not found!"
            raise FileNotFoundError(exception)
        if not self._utils_only:
            self._model = AutoModel.from_pretrained(
                self._weights_dir,
                trust_remote_code=True,
                local_files_only=True,
            ).to("cuda", dtype=torch.bfloat16)
            if self._model is None:
                msg = "Model failed to load"
                raise RuntimeError(msg)
            self._model.eval()  # type: ignore[attr-defined]
        self._processor = AutoProcessor.from_pretrained(  # type: ignore[no-untyped-call]
            self._weights_dir,
            trust_remote_code=True,
            local_files_only=True,
        )

    def get_target_num_frames(self) -> int:
        """Get the target number of frames for the model.

        Returns:
            The target number of frames.

        """
        return cast("int", self._processor.num_video_frames)

    def formulate_input_frames(self, frames: list[npt.NDArray[np.uint8]]) -> npt.NDArray[np.float32] | None:
        """Formulate input frames for the model.

        Args:
            frames: List of video frames.

        Returns:
            The formulated input frames.

        """
        fn = self.get_target_num_frames()
        if len(frames) < fn:
            logger.error(f"Frame count {len(frames)} is smaller than minimal requirement {fn}")
            return None
        step = len(frames) // fn
        video_batch = np.expand_dims(np.stack(frames[::step][:fn]), 0)
        video_batch = np.transpose(video_batch, (0, 1, 4, 2, 3))
        return cast(
            "npt.NDArray[np.float32]",
            self._processor(videos=video_batch, return_tensors="pt")["videos"].numpy(),
        )

    def encode_video_frames(self, frames: npt.NDArray[np.float32]) -> torch.Tensor:
        """Encode video frames for the model.

        Args:
            frames: The input video frames.

        Returns:
            The encoded video frames.

        """
        if self._model is None:
            msg = "Model is not loaded"
            raise RuntimeError(msg)
        if frames.size == 0:
            return torch.empty((0, self._model.config.embed_dim), dtype=torch.float16)  # type: ignore[attr-defined]

        with torch.no_grad():
            videos = torch.from_numpy(frames).to("cuda", dtype=torch.bfloat16)
            output = self._model.get_video_embeddings(videos=videos)  # type: ignore[attr-defined]
            return cast("torch.Tensor", output.visual_proj.to("cpu", dtype=torch.float16))

    def get_text_embedding(self, text: str) -> torch.Tensor:
        """Get the text embedding for the given text.

        Args:
            text: The input text.

        Returns:
            The text embedding.

        """
        if self._model is None:
            msg = "Model is not loaded"
            raise RuntimeError(msg)
        batch = self._processor(text=[text], return_tensors="pt").to("cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            output = self._model.get_text_embeddings(**batch)  # type: ignore[attr-defined]
            return cast("torch.Tensor", output.text_proj.to("cpu", dtype=torch.float16))

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
        if self._model is None:
            msg = "Model is not loaded"
            raise RuntimeError(msg)
        label_probs = (100.0 * video_embd @ text_embds_tensor.T).softmax(dim=-1)
        probs, idxs = label_probs.float().cpu().topk(count, dim=-1)
        return probs.cpu().numpy()[0].tolist(), idxs.cpu().long().numpy()[0].tolist()

    @classmethod
    def download_weights_on_node(cls, model_dir: str, variant: Literal["224p", "336p", "448p"] = "336p") -> None:
        """Download the weights for the CosmosEmbed1 model on the node."""
        model_dir_path = Path(model_dir) / _COSMOS_EMBED1_VARIANTS_INFO[variant]
        model_dir_path.mkdir(parents=True, exist_ok=True)
        if model_dir_path.exists() and any(model_dir_path.glob("*.safetensors")):
            return
        download_model_from_hf(
            model_id=_COSMOS_EMBED1_VARIANTS_INFO[variant],
            local_dir=model_dir_path,
        )
        logger.info(f"CosmosEmbed1 {variant} weights downloaded to: {model_dir_path}")

    @classmethod
    def download_processor_config_on_node(
        cls, model_dir: str, variant: Literal["224p", "336p", "448p"] = "336p"
    ) -> None:
        """Download the processor config for the CosmosEmbed1 model on the node."""
        model_dir_path = Path(model_dir) / _COSMOS_EMBED1_VARIANTS_INFO[variant]
        model_dir_path.mkdir(parents=True, exist_ok=True)

        processor_config_file = model_dir_path / "processor_config.json"
        if processor_config_file.exists():
            return
        download_model_from_hf(
            model_id=_COSMOS_EMBED1_VARIANTS_INFO[variant],
            local_dir=model_dir_path,
            ignore_patterns=["*.safetensors"],
            revision=COSMOS_EMBED1_MODEL_REVISION_INFO[variant],
        )
        logger.info(f"CosmosEmbed1 {variant} processor config downloaded to: {model_dir_path}")
