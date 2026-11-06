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

"""Local embedding runtimes: Omni-Embed-Nemotron and LanguageBind."""

from __future__ import annotations

import contextlib
import logging
import os
import sys
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F

from omnifuse_tutorial.config import TUTORIAL_DIR, languagebind_root

if TYPE_CHECKING:
    from omnifuse_tutorial.config import RuntimeConfig

OMNI_PROCESSOR_FILES = ["*.json", "*.txt", "*.model", "*.py", "*.jinja", "additional_chat_templates/*.jinja"]
LANGUAGEBIND_REPOS = {
    "image": "LanguageBind/LanguageBind_Image",
    "audio": "LanguageBind/LanguageBind_Audio_FT",
    "video": "LanguageBind/LanguageBind_Video_FT",
}
# The Qwen2.5-Omni processor logs this through the root logger on every call; we never generate audio.
logging.getLogger().addFilter(lambda record: not str(record.msg).startswith("System prompt modified, audio output"))


def resolve_device(runtime: RuntimeConfig) -> str:
    if runtime.device and runtime.device != "auto":
        return runtime.device
    return "cuda" if torch.cuda.is_available() else "cpu"


def mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    """Attention-masked mean pooling followed by L2 normalization."""
    hidden = hidden.masked_fill(~mask[..., None].bool(), 0.0)
    return F.normalize(hidden.sum(dim=1) / mask.sum(dim=1)[..., None], dim=-1).float().cpu().numpy()


class OmniEmbed:
    """Lazy nvidia/omni-embed-nemotron-3b runtime for text, image, audio, and video embeddings."""

    def __init__(self, model_name: str, device: str):
        self.model_name, self.device = model_name, device
        self.model = self.processor = None

    def _load(self) -> None:
        if self.model is not None:
            return
        from huggingface_hub import snapshot_download
        from transformers import AutoModel, AutoProcessor

        # A processor-only snapshot works around AutoProcessor failing on the full repository.
        local = snapshot_download(self.model_name, allow_patterns=OMNI_PROCESSOR_FILES)
        self.processor = AutoProcessor.from_pretrained(local, trust_remote_code=True, local_files_only=True)
        dtype = torch.float32
        if self.device == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.model = AutoModel.from_pretrained(
            self.model_name, torch_dtype=dtype, attn_implementation="eager", trust_remote_code=True
        )
        self.model = self.model.to(self.device).eval()

    def encode_texts(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        self._load()
        batches = []
        for start in range(0, len(texts), batch_size):
            prompts = [
                self.processor.apply_chat_template(
                    [{"role": "user", "content": [{"type": "text", "text": text}]}],
                    add_generation_prompt=False,
                    tokenize=False,
                )
                for text in texts[start : start + batch_size]
            ]
            tokens = self.processor.tokenizer(
                prompts, return_tensors="pt", truncation=True, padding=True, max_length=2048
            )
            batches.append(self._pool(tokens))
        return np.concatenate(batches) if batches else np.empty((0, 2048), dtype=np.float32)

    def encode_media(self, path: str, modality: str) -> np.ndarray:
        import librosa
        from PIL import Image
        from qwen_vl_utils import process_vision_info

        self._load()
        image = Image.open(path).convert("RGB") if modality == "image" else None
        audio = librosa.load(str(path), sr=16000, mono=True)[0] if modality == "audio" else None
        if modality == "image":
            item = {"type": "image", "image": image}
        elif modality == "audio":
            item = {"type": "audio", "audio": audio}
        elif modality == "video":
            item = {"type": "video", "video": str(path), "fps": 1.0, "max_pixels": 128 * 128 * 64}
        else:
            raise ValueError(f"Unsupported modality for Omni-Embed: {modality}")
        messages = [{"role": "user", "content": [item]}]
        videos = process_vision_info(messages, return_video_kwargs=True)[1] if modality == "video" else None
        text = self.processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
        kwargs = {"text_kwargs": {"truncation": True, "max_length": 2048}} if modality == "audio" else {}
        batch = self.processor(
            text=[text], images=image, videos=videos, audio=audio, return_tensors="pt", padding=True, **kwargs
        )
        return self._pool(batch)[0]

    def _pool(self, batch: Any) -> np.ndarray:
        batch = {key: value.to(self.device) for key, value in batch.items() if value is not None}
        with torch.inference_mode():
            return mean_pool(self.model(**batch, output_hidden_states=True).hidden_states[-1], batch["attention_mask"])


def bootstrap_languagebind() -> None:
    """Make the LanguageBind checkout importable on current torch/transformers releases."""
    root = languagebind_root()
    if not root.exists():
        raise RuntimeError(
            f"LanguageBind source not found at {root}. Clone https://github.com/PKU-YuanGroup/LanguageBind to "
            f"{TUTORIAL_DIR / 'third_party' / 'LanguageBind'} or set LANGUAGEBIND_ROOT."
        )
    tmp = TUTORIAL_DIR / "tmp"
    for name, directory in (
        ("TMPDIR", tmp),
        ("TORCHINDUCTOR_CACHE_DIR", tmp / "torchinductor_cache"),
        ("PYTHONPYCACHEPREFIX", tmp / "pycache"),
    ):
        directory.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(name, str(directory))
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    import torchaudio
    import transformers.models.clip.modeling_clip as clip_modeling
    from transformers.modeling_attn_mask_utils import AttentionMaskConverter

    if not hasattr(torchaudio, "set_audio_backend"):
        torchaudio.set_audio_backend = lambda *args, **kwargs: None
    with contextlib.suppress(ImportError):
        import torchvision.transforms._functional_tensor as functional_tensor

        sys.modules.setdefault("torchvision.transforms.functional_tensor", functional_tensor)
    if not hasattr(clip_modeling, "_expand_mask"):
        clip_modeling._expand_mask = lambda mask, dtype, tgt_len=None: AttentionMaskConverter._expand_mask(
            mask=mask, dtype=dtype, tgt_len=tgt_len
        )
    clip_modeling.CLIPVisionEmbeddings.forward = _clip_vision_embeddings_forward


def _clip_vision_embeddings_forward(
    self: Any, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = False
) -> Any:
    """CLIP vision embeddings that cast pixels to the patch-embedding dtype, as LanguageBind expects."""
    batch_size, _, height, width = pixel_values.shape
    expected = tuple(self.image_size) if isinstance(self.image_size, (tuple, list)) else (self.image_size,) * 2
    if not interpolate_pos_encoding and (height, width) != expected:
        raise ValueError(f"Input image size ({height}*{width}) doesn't match model ({expected[0]}*{expected[1]}).")
    patch_embeds = self.patch_embedding(pixel_values.to(dtype=self.patch_embedding.weight.dtype))
    embeddings = torch.cat(
        [self.class_embedding.expand(batch_size, 1, -1), patch_embeds.flatten(2).transpose(1, 2)], dim=1
    )
    if interpolate_pos_encoding:
        return embeddings + self.interpolate_pos_encoding(embeddings, height, width)
    return embeddings + self.position_embedding(self.position_ids)


class LanguageBind:
    """LanguageBind image/audio/video towers; text uses the video tower's text encoder."""

    def __init__(self, device: str):
        bootstrap_languagebind()
        import languagebind

        self.device, self.loaded = device, {}
        self.classes = {
            "image": (languagebind.LanguageBindImage, languagebind.LanguageBindImageProcessor),
            "audio": (languagebind.LanguageBindAudio, languagebind.LanguageBindAudioProcessor),
            "video": (languagebind.LanguageBindVideo, languagebind.LanguageBindVideoProcessor),
        }
        self.tokenizer_class, self.tokenizer = languagebind.LanguageBindImageTokenizer, None
        np.random.seed(13)

    def _modality(self, modality: str) -> tuple[Any, Any]:
        if modality not in self.loaded:
            if self.tokenizer is None:
                self.tokenizer = self.tokenizer_class.from_pretrained(LANGUAGEBIND_REPOS["image"])
            model_class, processor_class = self.classes[modality]
            model = model_class.from_pretrained(LANGUAGEBIND_REPOS[modality])
            for module in (model, getattr(model, "text_model", None), getattr(model, "vision_model", None)):
                config = getattr(module, "config", None)
                if config is not None and getattr(config, "_attn_implementation", None) is None:
                    config._attn_implementation = "eager"
            model = model.to(self.device).eval()
            processor = processor_class(model.config, self.tokenizer)
            transforms = getattr(getattr(processor, "transform", None), "transforms", None)
            if modality == "video" and transforms:
                kept = [step for step in transforms if type(step).__name__ != "RandomHorizontalFlipVideo"]
                processor.transform = type(processor.transform)(kept)
            self.loaded[modality] = (model, processor)
        return self.loaded[modality]

    def encode_text(self, text: str) -> np.ndarray:
        model, _ = self._modality("video")
        batch = self.tokenizer([text], max_length=77, padding="max_length", truncation=True, return_tensors="pt")
        with torch.inference_mode():
            features = model.get_text_features(**{key: value.to(self.device) for key, value in batch.items()})
        return F.normalize(features, dim=-1)[0].float().cpu().numpy()

    def encode_media(self, path: str, modality: str) -> np.ndarray:
        model, processor = self._modality(modality)
        batch = processor(images=[path], return_tensors="pt")
        with torch.inference_mode():
            features = model.get_image_features(**{key: value.to(self.device) for key, value in batch.items()})
        return F.normalize(features, dim=-1)[0].float().cpu().numpy()
