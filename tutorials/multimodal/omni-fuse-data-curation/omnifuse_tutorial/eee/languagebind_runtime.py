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

"""Runtime wrapper for the vendored LanguageBind source tree."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any

DEFAULT_LANGUAGEBIND_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "LanguageBind"


def _runtime_root() -> Path:
    return Path(os.environ.get("LANGUAGEBIND_ROOT") or DEFAULT_LANGUAGEBIND_ROOT).expanduser().resolve()


def _ensure_runtime_dirs() -> None:
    repo_tmp = Path(__file__).resolve().parents[2] / "tmp"
    repo_tmp.mkdir(exist_ok=True)
    torchinductor_dir = repo_tmp / "torchinductor_cache"
    torchinductor_dir.mkdir(exist_ok=True)
    pycache_dir = repo_tmp / "pycache"
    pycache_dir.mkdir(exist_ok=True)
    os.environ.setdefault("TMPDIR", str(repo_tmp))
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(torchinductor_dir))
    os.environ.setdefault("PYTHONPYCACHEPREFIX", str(pycache_dir))


def bootstrap_languagebind() -> None:
    """Make the LanguageBind checkout importable on modern torch/transformers."""

    root = _runtime_root()
    if not root.exists():
        raise RuntimeError(
            "LanguageBind source not found. Clone it at "
            f"{DEFAULT_LANGUAGEBIND_ROOT} or set LANGUAGEBIND_ROOT. "
            "For parity with Omni-Fuse, use the LanguageBind submodule from "
            "../embedsimclusterer-experiments/third_party/LanguageBind."
        )

    _ensure_runtime_dirs()
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    try:
        import torchaudio
        import transformers.models.clip.modeling_clip as clip_modeling
        from transformers.modeling_attn_mask_utils import AttentionMaskConverter
    except ImportError as exc:
        raise RuntimeError(
            "LanguageBind runtime requires torchaudio and transformers. "
            "Install the full local extras with `python -m pip install -e '.[full]'`."
        ) from exc

    if not hasattr(torchaudio, "set_audio_backend"):
        torchaudio.set_audio_backend = lambda *args, **kwargs: None

    if "torchvision.transforms.functional_tensor" not in sys.modules:
        try:
            import torchvision.transforms._functional_tensor as functional_tensor

            sys.modules["torchvision.transforms.functional_tensor"] = functional_tensor
        except ImportError:
            pass

    if not hasattr(clip_modeling, "_expand_mask"):
        clip_modeling._expand_mask = lambda mask, dtype, tgt_len=None: AttentionMaskConverter._expand_mask(
            mask=mask,
            dtype=dtype,
            tgt_len=tgt_len,
        )

    if not getattr(clip_modeling.CLIPVisionEmbeddings, "_omnifuse_languagebind_patch", False):

        def patched_forward(self: Any, pixel_values: Any, interpolate_pos_encoding: bool = False) -> Any:
            import torch

            batch_size, _, height, width = pixel_values.shape
            if isinstance(self.image_size, (tuple, list)):
                expected_height, expected_width = self.image_size
            else:
                expected_height = expected_width = self.image_size
            if not interpolate_pos_encoding and (height != expected_height or width != expected_width):
                raise ValueError(
                    f"Input image size ({height}*{width}) doesn't match model ({expected_height}*{expected_width})."
                )
            target_dtype = self.patch_embedding.weight.dtype
            patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))
            patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
            class_embeds = self.class_embedding.expand(batch_size, 1, -1)
            embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
            if interpolate_pos_encoding:
                embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
            else:
                embeddings = embeddings + self.position_embedding(self.position_ids)
            return embeddings

        clip_modeling.CLIPVisionEmbeddings.forward = patched_forward
        clip_modeling.CLIPVisionEmbeddings._omnifuse_languagebind_patch = True


class LanguageBindRuntime:
    """Thin adapter over upstream LanguageBind image, audio, and video models."""

    MODEL_REPOS = {
        "image": "LanguageBind/LanguageBind_Image",
        "audio": "LanguageBind/LanguageBind_Audio_FT",
        "video": "LanguageBind/LanguageBind_Video_FT",
    }

    def __init__(self, device: str, text_branch: str = "video", local_files_only: bool = False) -> None:
        bootstrap_languagebind()

        import numpy as np

        self.device = device
        self.text_branch = text_branch
        self.local_files_only = local_files_only
        self._models: dict[str, Any] = {}
        self._processors: dict[str, Any] = {}
        self._tokenizer: Any | None = None

        tokenizer_mod = importlib.import_module("languagebind.image.tokenization_image")
        self._tokenizer_cls = tokenizer_mod.LanguageBindImageTokenizer

        image_model_mod = importlib.import_module("languagebind.image.modeling_image")
        image_proc_mod = importlib.import_module("languagebind.image.processing_image")
        audio_model_mod = importlib.import_module("languagebind.audio.modeling_audio")
        audio_proc_mod = importlib.import_module("languagebind.audio.processing_audio")
        video_model_mod = importlib.import_module("languagebind.video.modeling_video")
        video_proc_mod = importlib.import_module("languagebind.video.processing_video")

        self._model_classes = {
            "image": image_model_mod.LanguageBindImage,
            "audio": audio_model_mod.LanguageBindAudio,
            "video": video_model_mod.LanguageBindVideo,
        }
        self._processor_classes = {
            "image": image_proc_mod.LanguageBindImageProcessor,
            "audio": audio_proc_mod.LanguageBindAudioProcessor,
            "video": video_proc_mod.LanguageBindVideoProcessor,
        }
        np.random.seed(13)

    def unload(self) -> None:
        self._processors.clear()
        self._tokenizer = None
        self._models.clear()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            return

    def _ensure_tokenizer(self) -> Any:
        if self._tokenizer is None:
            self._tokenizer = self._tokenizer_cls.from_pretrained(
                self.MODEL_REPOS["image"],
                local_files_only=self.local_files_only,
            )
        return self._tokenizer

    def _ensure_modality(self, modality: str) -> tuple[Any, Any]:
        if modality in self._models:
            return self._models[modality], self._processors[modality]

        tokenizer = self._ensure_tokenizer()
        model = self._model_classes[modality].from_pretrained(
            self.MODEL_REPOS[modality],
            local_files_only=self.local_files_only,
        )
        self._force_eager_attention(model)
        model = model.to(self.device)
        model.eval()
        processor = self._processor_classes[modality](model.config, tokenizer)
        if modality == "video":
            self._strip_random_video_flip(processor)
        self._models[modality] = model
        self._processors[modality] = processor
        return model, processor

    @staticmethod
    def _force_eager_attention(model: Any) -> None:
        for cfg in (
            getattr(model, "config", None),
            getattr(getattr(model, "text_model", None), "config", None),
            getattr(getattr(model, "vision_model", None), "config", None),
        ):
            if cfg is not None and getattr(cfg, "_attn_implementation", None) is None:
                cfg._attn_implementation = "eager"

    @staticmethod
    def _strip_random_video_flip(processor: Any) -> None:
        transform = getattr(processor, "transform", None)
        transforms = getattr(transform, "transforms", None)
        if not transforms:
            return
        kept = [step for step in transforms if step.__class__.__name__ != "RandomHorizontalFlipVideo"]
        if len(kept) != len(transforms):
            processor.transform = type(transform)(kept)

    def _to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        return {key: value.to(self.device) for key, value in batch.items()}

    def encode_text(self, text: str) -> Any:
        import torch
        import torch.nn.functional as F

        model, _ = self._ensure_modality(self.text_branch)
        tokenizer = self._ensure_tokenizer()
        batch = tokenizer([text], max_length=77, padding="max_length", truncation=True, return_tensors="pt")
        batch = self._to_device(batch)
        with torch.inference_mode():
            features = model.get_text_features(**batch)
        return F.normalize(features, dim=-1)[0]

    def encode_image(self, path: str) -> Any:
        return self._encode_media("image", path)

    def encode_audio(self, path: str) -> Any:
        return self._encode_media("audio", path)

    def encode_video(self, path: str) -> Any:
        return self._encode_media("video", path)

    def _encode_media(self, modality: str, path: str) -> Any:
        import torch
        import torch.nn.functional as F

        model, processor = self._ensure_modality(modality)
        batch = processor(images=[path], return_tensors="pt")
        batch = self._to_device(batch)
        with torch.inference_mode():
            features = model.get_image_features(**batch)
        return F.normalize(features, dim=-1)[0]
