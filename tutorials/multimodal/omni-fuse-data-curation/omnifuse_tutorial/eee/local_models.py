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

"""Full local Omni-Fuse embedding model stack."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg"}
TEXT_EXTENSIONS = {".txt", ".md", ".json", ".csv"}
OMNI_PROCESSOR_ALLOW_PATTERNS = [
    "*.json",
    "*.txt",
    "*.model",
    "*.py",
    "*.jinja",
    "additional_chat_templates/*.jinja",
]


def full_stack_dependency_error(component: str, exc: BaseException | None = None) -> RuntimeError:
    message = (
        f"Full local Omni-Fuse {component} is unavailable. Install the local model extras with "
        "`python -m pip install -e '.[full]'`, make sure Hugging Face model weights are cached "
        "or network access is available, and place required local checkpoints under `model_files/`."
    )
    error = RuntimeError(message)
    if exc is not None:
        error.__cause__ = exc
    return error


def resolve_device(runtime: Any | None = None) -> str:
    requested = str(getattr(runtime, "device", "auto") if runtime is not None else "auto")
    if requested and requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def resolve_offline_mode(runtime: Any | None = None) -> bool:
    if runtime is not None and bool(getattr(runtime, "offline_mode", False)):
        return True
    return os.environ.get("EMBEDSIM_OFFLINE_MODE", "0") == "1" or os.environ.get("HF_HUB_OFFLINE") == "1"


def infer_modality(value: Any, hint: str | None = None) -> str:
    if hint:
        return hint
    try:
        from PIL import Image

        if isinstance(value, Image.Image):
            return "image"
    except ImportError:
        pass
    if isinstance(value, dict):
        if "audio" in value or "waveform" in value or "sample_rate" in value:
            return "audio"
        if "frames" in value:
            return "video"
        if "file_path" in value:
            return infer_modality(value["file_path"])
    if isinstance(value, (str, Path)):
        value_str = str(value)
        if len(value_str) < 500 and "\n" not in value_str:
            suffix = Path(value_str).suffix.lower()
            if suffix in IMAGE_EXTENSIONS:
                return "image"
            if suffix in AUDIO_EXTENSIONS:
                return "audio"
            if suffix in VIDEO_EXTENSIONS:
                return "video"
        return "text"
    return "text"


def resolve_cached_hf_snapshot_path(model_name: str) -> str:
    if os.environ.get("HUGGINGFACE_HUB_CACHE"):
        cache_dir = Path(os.environ["HUGGINGFACE_HUB_CACHE"])
    elif os.environ.get("HF_HOME"):
        cache_dir = Path(os.environ["HF_HOME"]) / "hub"
    else:
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    model_cache = cache_dir / f"models--{model_name.replace('/', '--')}"
    snapshots_dir = model_cache / "snapshots"
    if not snapshots_dir.exists():
        raise RuntimeError(f"Model {model_name} is not cached at {snapshots_dir}.")
    snapshots = sorted(snapshots_dir.iterdir(), key=lambda path: path.stat().st_mtime, reverse=True)
    if not snapshots:
        raise RuntimeError(f"Model {model_name} has no cached snapshots under {snapshots_dir}.")
    return str(snapshots[0])


def load_omni_processor_with_workaround(model_name: str, offline_mode: bool) -> tuple[str, bool]:
    if offline_mode:
        return resolve_cached_hf_snapshot_path(model_name), True
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise full_stack_dependency_error("Omni-Embed processor download", exc)
    processor_path = snapshot_download(model_name, allow_patterns=OMNI_PROCESSOR_ALLOW_PATTERNS)
    return processor_path, True


_QWEN_OMNI_SYSTEM_PROMPT_WARNING_PREFIX = "System prompt modified, audio output may not work as expected."
_QWEN_OMNI_FILTER_INSTALLED = False


class _Qwen2_5OmniSystemPromptWarningFilter(logging.Filter):
    """Drop the noisy Qwen2.5-Omni 'system prompt modified' warning.

    transformers/models/qwen2_5_omni/processing_qwen2_5_omni.py logs this on
    every ``apply_chat_template`` call when the system prompt differs from the
    default Qwen one. We never use Qwen2.5-Omni's audio-generation mode (we
    only consume embeddings), so the warning is harmless and its per-sample
    repetition floods worker logs.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:
            return True
        return not message.startswith(_QWEN_OMNI_SYSTEM_PROMPT_WARNING_PREFIX)


def _silence_qwen_omni_system_prompt_warning() -> None:
    """Install a one-shot root-logger filter that suppresses the Qwen2.5-Omni
    ``apply_chat_template`` system-prompt warning.

    transformers issues this via ``logging.warning(...)`` (root logger), so the
    filter must live on the root logger. Idempotent across repeated calls
    within a single process.
    """
    global _QWEN_OMNI_FILTER_INSTALLED
    if _QWEN_OMNI_FILTER_INSTALLED:
        return
    logging.getLogger().addFilter(_Qwen2_5OmniSystemPromptWarningFilter())
    _QWEN_OMNI_FILTER_INSTALLED = True


class DimAdapter:
    """Deterministic projection to the configured embedding dimension."""

    def __init__(self, in_dim: int, out_dim: int, seed: int = 13):
        rng = np.random.default_rng(seed)
        self.weight = rng.standard_normal((in_dim, out_dim)).astype(np.float32) / np.sqrt(max(1, in_dim))
        self.out_dim = out_dim

    def __call__(self, values: np.ndarray) -> np.ndarray:
        return values @ self.weight


class NvidiaLlamaNemotronTextEncoder:
    """Local nvidia/llama-nemotron-embed-1b-v2 text encoder."""

    def __init__(
        self,
        model_name: str = "nvidia/llama-nemotron-embed-1b-v2",
        device: str = "cpu",
        offline_mode: bool = False,
    ):
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise full_stack_dependency_error("text expert encoder", exc)

        self.model_name = model_name
        self.device = device
        self.offline_mode = offline_mode
        self.torch = torch
        model_path = resolve_cached_hf_snapshot_path(model_name) if offline_mode else model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=offline_mode, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(model_path, local_files_only=offline_mode, trust_remote_code=True)
        self.model = self.model.to(device)
        self.model.eval()
        self.dim = int(getattr(getattr(self.model, "config", None), "hidden_size", 2048))

    def encode(self, texts: list[str], batch_size: int = 8, max_length: int = 2048) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        outputs: list[np.ndarray] = []
        prefixed = [f"passage: {text}" for text in texts]
        for start in range(0, len(prefixed), max(1, batch_size)):
            batch = prefixed[start : start + max(1, batch_size)]
            batch_dict = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(self.device)
            with self.torch.inference_mode():
                model_outputs = self.model(**batch_dict)
                hidden = model_outputs.last_hidden_state
                mask = batch_dict["attention_mask"]
                hidden = hidden.masked_fill(~mask[..., None].bool(), 0.0)
                embedding = hidden.sum(dim=1) / mask.sum(dim=1)[..., None]
                embedding = self.torch.nn.functional.normalize(embedding, dim=-1)
            outputs.append(embedding.float().cpu().numpy())
        return np.concatenate(outputs, axis=0).astype(np.float32)


class BlipCaptioner:
    def __init__(self, device: str = "cpu", offline_mode: bool = False):
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise full_stack_dependency_error("image captioner", exc)
        dev = 0 if device.startswith("cuda") else -1
        self.pipe = pipeline(
            "image-to-text",
            model="Salesforce/blip-image-captioning-base",
            device=dev,
            model_kwargs={"local_files_only": offline_mode},
        )

    def caption(self, image_obj: Any, prompt: str = "") -> str:
        from PIL import Image

        if isinstance(image_obj, (str, Path)):
            image = Image.open(image_obj).convert("RGB")
        elif isinstance(image_obj, Image.Image):
            image = image_obj.convert("RGB")
        else:
            image = Image.fromarray(image_obj).convert("RGB")
        result = self.pipe(image, generate_kwargs={"max_new_tokens": 64})
        return str(result[0].get("generated_text") or "").strip()


class WhisperASR:
    def __init__(self, device: str = "cpu", offline_mode: bool = False):
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise full_stack_dependency_error("audio ASR", exc)
        dev = 0 if device.startswith("cuda") else -1
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-tiny.en",
            device=dev,
            model_kwargs={"local_files_only": offline_mode},
        )

    def transcribe(self, audio_obj: Any, prompt: str = "") -> str:
        if isinstance(audio_obj, dict):
            if "audio" in audio_obj and "sample_rate" in audio_obj:
                audio_obj = {"array": audio_obj["audio"], "sampling_rate": audio_obj["sample_rate"]}
            elif "file_path" in audio_obj:
                audio_obj = audio_obj["file_path"]
        if isinstance(audio_obj, (str, Path)):
            import librosa

            audio, _ = librosa.load(str(audio_obj), sr=16000, mono=True)
            audio_obj = {"array": audio, "sampling_rate": 16000}
        try:
            import transformers.pipelines.automatic_speech_recognition as asr_pipeline

            asr_pipeline.is_torchcodec_available = lambda: False
        except Exception:
            pass
        result = self.pipe(audio_obj)
        return str(result.get("text") or "").strip()


class KeyframeVideoDescriber:
    def __init__(self, image_captioner: BlipCaptioner, num_frames: int = 4):
        self.image_captioner = image_captioner
        self.num_frames = max(1, int(num_frames))

    def describe(self, video_obj: Any, prompt: str = "") -> str:
        frames = sample_video_keyframes(video_obj, self.num_frames)
        captions = [self.image_captioner.caption(frame, prompt=prompt) for frame in frames]
        return " [SEP] ".join(caption for caption in captions if caption)


class OmniEmbedNemotronRuntime:
    """Local nvidia/omni-embed-nemotron-3b runtime for all modalities."""

    def __init__(
        self,
        model_name: str = "nvidia/omni-embed-nemotron-3b",
        device: str = "cpu",
        offline_mode: bool = False,
    ):
        self.model_name = model_name
        self.device = device
        self.offline_mode = offline_mode
        self._model: Any | None = None
        self._processor: Any | None = None
        self.dim = 2048

    def _initialize(self) -> None:
        if self._model is not None and self._processor is not None:
            return
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:
            raise full_stack_dependency_error("Omni-Embed e2e expert", exc)

        _silence_qwen_omni_system_prompt_warning()

        model_path = resolve_cached_hf_snapshot_path(self.model_name) if self.offline_mode else self.model_name
        processor_path, processor_local_only = load_omni_processor_with_workaround(self.model_name, self.offline_mode)
        self._processor = AutoProcessor.from_pretrained(
            processor_path,
            trust_remote_code=True,
            local_files_only=processor_local_only,
        )
        dtype = torch.float32
        if self.device == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self._model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=dtype,
            attn_implementation="eager",
            trust_remote_code=True,
            local_files_only=self.offline_mode,
        )
        self._model = self._model.to(self.device)
        self._model.eval()

    def encode_text(self, text: str) -> np.ndarray:
        return self._get_embedding(text=text)

    def encode_image(self, image: Any) -> np.ndarray:
        return self._get_embedding(image=image)

    def encode_audio(self, audio: Any) -> np.ndarray:
        return self._get_embedding(audio=audio)

    def encode_video(self, video: Any) -> np.ndarray:
        return self._get_embedding(video=video)

    def _get_embedding(
        self,
        text: str | None = None,
        image: Any | None = None,
        audio: Any | None = None,
        video: Any | None = None,
    ) -> np.ndarray:
        self._initialize()
        import torch

        processor = self._processor
        model = self._model
        content: list[dict[str, Any]] = []
        image_input = None
        audio_input = None
        video_inputs = None
        preloaded_video = False

        if text is not None:
            content.append({"type": "text", "text": text})
        if image is not None:
            image_input = _load_image(image)
            content.append({"type": "image", "image": image_input})
        if audio is not None:
            audio_input = _load_audio_waveform(audio)
            content.append({"type": "audio", "audio": audio_input})
        if video is not None:
            video_value = video
            if isinstance(video, dict) and "file_path" in video:
                video_value = video["file_path"]
            if isinstance(video_value, np.ndarray):
                video_inputs = video_value
                preloaded_video = True
                content.append({"type": "video", "video": "video.mp4"})
            else:
                content.append(
                    {
                        "type": "video",
                        "video": str(video_value),
                        "fps": 1.0,
                        "max_pixels": 128 * 128 * 64,
                    }
                )
        if not content:
            raise ValueError("At least one modality input must be provided")

        messages = [{"role": "user", "content": content}]
        if video is not None and not preloaded_video:
            try:
                from qwen_vl_utils import process_vision_info
            except ImportError as exc:
                raise full_stack_dependency_error("Omni-Embed video preprocessing", exc)
            _, video_inputs, _ = process_vision_info(messages, return_video_kwargs=True)

        text_input = processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
        processor_kwargs = {
            "text": [text_input],
            "images": image_input,
            "videos": video_inputs,
            "audio": audio_input,
            "return_tensors": "pt",
            "padding": True,
        }
        if image is None and video is None:
            processor_kwargs["text_kwargs"] = {"truncation": True, "max_length": 2048}
        batch_dict = processor(**processor_kwargs)
        batch_dict = {key: value.to(self.device) for key, value in batch_dict.items() if value is not None}
        with torch.inference_mode():
            outputs = model(**batch_dict, output_hidden_states=True)
            hidden = outputs.hidden_states[-1]
            mask = batch_dict["attention_mask"]
            hidden = hidden.masked_fill(~mask[..., None].bool(), 0.0)
            embedding = hidden.sum(dim=1) / mask.sum(dim=1)[..., None]
            embedding = torch.nn.functional.normalize(embedding, dim=-1)
        return embedding.float().cpu().numpy()[0].astype(np.float32)

    def unload(self) -> None:
        self._model = None
        self._processor = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            return


class FullLocalEEEBackend:
    """Production local EEE backend matching the Omni-Fuse expert stack."""

    def __init__(self, config: Any | None = None, runtime: Any | None = None, embedding_dim: int = 2048):
        self.config = config
        self.runtime = runtime
        self.embedding_dim = int(getattr(config, "embedding_dim", embedding_dim))
        self.batch_size = int(getattr(config, "batch_size", 8))
        self.device = resolve_device(runtime)
        self.offline_mode = resolve_offline_mode(runtime)
        self.text_prompt_base = str(getattr(config, "text_prompt_base", "") or "")
        self.text_prompt_prefix = str(getattr(config, "text_prompt_prefix", "") or "")
        self.text_model = str(getattr(config, "nvidia_embedding_model", "nvidia/llama-nemotron-embed-1b-v2"))
        self.omni_model = str(getattr(config, "nvidia_multimodal_model", "nvidia/omni-embed-nemotron-3b"))
        self._text_encoder: NvidiaLlamaNemotronTextEncoder | None = None
        self._image_captioner: BlipCaptioner | None = None
        self._audio_asr: WhisperASR | None = None
        self._video_describer: KeyframeVideoDescriber | None = None
        self._omni: OmniEmbedNemotronRuntime | None = None
        self._languagebind: Any | None = None
        self._languagebind_adapter: DimAdapter | None = None
        self._description_cache: dict[str, str] = {}

    def embed_raw(self, record: dict[str, Any], expert: str) -> list[float]:
        expert = _validate_expert(expert)
        modality = str(record.get("modality") or infer_modality(record.get("raw_path")))
        if expert == "text-based":
            text = self.describe_record(record)
            return self._encode_text_stack(text)
        if expert == "fusion":
            return self._encode_fusion(_raw_value(record, modality), modality)
        if expert == "e2e":
            return self._encode_omni(_raw_value(record, modality), modality)
        raise ValueError(f"Unsupported EEE expert: {expert}")

    def embed_annotation(self, record: dict[str, Any], expert: str) -> list[float]:
        return self.embed_query(_annotation_text(record), expert)

    def embed_query(self, query: str, expert: str = "text-based") -> list[float]:
        expert = _validate_expert(expert)
        if expert == "text-based":
            return self._encode_text_stack(query)
        if expert == "fusion":
            return self._encode_languagebind_text(query)
        if expert == "e2e":
            return _resize_and_normalize(self._ensure_omni().encode_text(query), self.embedding_dim)
        raise ValueError(f"Unsupported EEE expert: {expert}")

    def describe_record(self, record: dict[str, Any]) -> str:
        modality = str(record.get("modality") or infer_modality(record.get("raw_path")))
        raw_text = _text_or_none(record.get("sns_raw_text")) or _text_or_none(record.get("raw_text"))
        if raw_text:
            return raw_text
        raw_value = _raw_value(record, modality)
        if modality == "text":
            return str(raw_value)
        cache_key = f"{modality}:{raw_value}:{self.text_prompt_base}:{self.text_prompt_prefix}"
        if cache_key in self._description_cache:
            return self._description_cache[cache_key]
        prompt = f"{self.text_prompt_base}{self.text_prompt_prefix}"
        if modality == "image":
            description = self._ensure_image_captioner().caption(raw_value, prompt=prompt)
        elif modality == "audio":
            description = self._ensure_audio_asr().transcribe(raw_value, prompt=prompt)
        elif modality == "video":
            description = self._ensure_video_describer().describe(raw_value, prompt=prompt)
        else:
            description = str(raw_value)
        self._description_cache[cache_key] = description
        return description

    def unload(self) -> None:
        if self._languagebind is not None and hasattr(self._languagebind, "unload"):
            self._languagebind.unload()
        if self._omni is not None:
            self._omni.unload()
        self._text_encoder = None
        self._image_captioner = None
        self._audio_asr = None
        self._video_describer = None
        self._omni = None
        self._languagebind = None

    def _encode_text_stack(self, text: str) -> list[float]:
        vectors = self._ensure_text_encoder().encode([text], batch_size=self.batch_size)
        return _resize_and_normalize(vectors[0], self.embedding_dim)

    def _encode_omni(self, value: Any, modality: str) -> list[float]:
        omni = self._ensure_omni()
        if modality == "text":
            vector = omni.encode_text(str(value))
        elif modality == "image":
            vector = omni.encode_image(value)
        elif modality == "audio":
            vector = omni.encode_audio(value)
        elif modality == "video":
            vector = omni.encode_video(value)
        else:
            raise ValueError(f"Unsupported modality for Omni-Embed: {modality}")
        return _resize_and_normalize(vector, self.embedding_dim)

    def _encode_fusion(self, value: Any, modality: str) -> list[float]:
        if modality == "text":
            return self._encode_languagebind_text(str(value))
        runtime = self._ensure_languagebind()
        if modality == "image":
            vector = runtime.encode_image(_path_string(value, modality))
        elif modality == "audio":
            vector = runtime.encode_audio(_path_string(value, modality))
        elif modality == "video":
            vector = runtime.encode_video(_path_string(value, modality))
        else:
            raise ValueError(f"Unsupported modality for LanguageBind: {modality}")
        vector_np = vector.detach().cpu().numpy().astype(np.float32)
        return self._adapt_languagebind(vector_np)

    def _encode_languagebind_text(self, text: str) -> list[float]:
        runtime = self._ensure_languagebind()
        vector = runtime.encode_text(text)
        vector_np = vector.detach().cpu().numpy().astype(np.float32)
        return self._adapt_languagebind(vector_np)

    def _adapt_languagebind(self, vector: np.ndarray) -> list[float]:
        vector = np.asarray(vector, dtype=np.float32)
        if vector.ndim != 1:
            vector = vector.reshape(-1)
        if vector.shape[0] != self.embedding_dim:
            if self._languagebind_adapter is None or self._languagebind_adapter.weight.shape[0] != vector.shape[0]:
                self._languagebind_adapter = DimAdapter(vector.shape[0], self.embedding_dim, seed=13)
            vector = self._languagebind_adapter(vector[None, :])[0]
        return _resize_and_normalize(vector, self.embedding_dim)

    def _ensure_text_encoder(self) -> NvidiaLlamaNemotronTextEncoder:
        if self._text_encoder is None:
            self._text_encoder = NvidiaLlamaNemotronTextEncoder(
                model_name=self.text_model,
                device=self.device,
                offline_mode=self.offline_mode,
            )
        return self._text_encoder

    def _ensure_image_captioner(self) -> BlipCaptioner:
        if self._image_captioner is None:
            self._image_captioner = BlipCaptioner(self.device, self.offline_mode)
        return self._image_captioner

    def _ensure_audio_asr(self) -> WhisperASR:
        if self._audio_asr is None:
            self._audio_asr = WhisperASR(self.device, self.offline_mode)
        return self._audio_asr

    def _ensure_video_describer(self) -> KeyframeVideoDescriber:
        if self._video_describer is None:
            self._video_describer = KeyframeVideoDescriber(self._ensure_image_captioner(), num_frames=4)
        return self._video_describer

    def _ensure_omni(self) -> OmniEmbedNemotronRuntime:
        if self._omni is None:
            self._omni = OmniEmbedNemotronRuntime(self.omni_model, self.device, self.offline_mode)
        return self._omni

    def _ensure_languagebind(self) -> Any:
        if self._languagebind is None:
            from omnifuse_tutorial.eee.languagebind_runtime import LanguageBindRuntime

            self._languagebind = LanguageBindRuntime(
                device=self.device,
                text_branch="video",
                local_files_only=self.offline_mode,
            )
        return self._languagebind


def sample_video_keyframes(video_path: Any, num_frames: int = 4) -> list[Any]:
    try:
        import cv2
        from PIL import Image
    except ImportError as exc:
        raise full_stack_dependency_error("video keyframe sampling", exc)

    if isinstance(video_path, dict):
        if video_path.get("frames"):
            frames = video_path["frames"]
            indices = np.linspace(0, len(frames) - 1, num=min(num_frames, len(frames)), dtype=int)
            return [
                frame if hasattr(frame, "mode") else Image.fromarray(frame) for frame in (frames[i] for i in indices)
            ]
        if "file_path" in video_path:
            video_path = video_path["file_path"]
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Failed to open video: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        raise ValueError(f"Video has no frames: {video_path}")
    indices = set(np.linspace(0, frame_count - 1, num=min(num_frames, frame_count), dtype=int).tolist())
    frames = []
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index in indices:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
        index += 1
    cap.release()
    if not frames:
        raise ValueError(f"No keyframes sampled from video: {video_path}")
    return frames


def _load_image(value: Any) -> Any:
    from PIL import Image

    if isinstance(value, Image.Image):
        return value
    if isinstance(value, dict) and "file_path" in value:
        value = value["file_path"]
    if isinstance(value, (str, Path)):
        return Image.open(value).convert("RGB")
    return value


def _load_audio_waveform(value: Any) -> Any:
    if isinstance(value, dict):
        if "audio" in value:
            audio = value["audio"]
            sample_rate = int(value.get("sample_rate", 16000))
            if sample_rate != 16000:
                import librosa

                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
            return audio
        if "file_path" in value:
            value = value["file_path"]
    if isinstance(value, (str, Path)):
        import librosa

        audio, _ = librosa.load(str(value), sr=16000, mono=True)
        return audio
    return value


def _raw_value(record: dict[str, Any], modality: str) -> Any:
    if modality == "text":
        raw_text = _text_or_none(record.get("sns_raw_text")) or _text_or_none(record.get("raw_text"))
        if raw_text:
            return raw_text
        path = _path_or_none(record.get("raw_path"))
        if path and path.exists():
            return path.read_text(encoding="utf-8").strip()
    path = _path_or_none(record.get("raw_path"))
    if path:
        return str(path)
    return record.get("raw_path") or record.get("raw_text") or ""


def _annotation_text(record: dict[str, Any]) -> str:
    return _text_or_none(record.get("sns_annotation")) or _text_or_none(record.get("annotation")) or ""


def _text_or_none(value: Any) -> str | None:
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return None


def _path_or_none(value: Any) -> Path | None:
    if not isinstance(value, (str, Path)):
        return None
    try:
        return Path(value)
    except OSError:
        return None


def _path_string(value: Any, modality: str) -> str:
    if isinstance(value, dict) and "file_path" in value:
        value = value["file_path"]
    if isinstance(value, (str, Path)):
        return str(value)
    raise ValueError(f"LanguageBind {modality} encoding requires a file path, got {type(value)!r}")


def _validate_expert(expert: str) -> str:
    if expert not in {"text-based", "fusion", "e2e"}:
        raise ValueError(f"Unsupported EEE expert: {expert}")
    return expert


def _resize_and_normalize(vector: Any, dim: int) -> list[float]:
    values = np.asarray(vector, dtype=np.float32).reshape(-1)
    if values.size != dim:
        resized = np.zeros(dim, dtype=np.float32)
        limit = min(values.size, dim)
        resized[:limit] = values[:limit]
        values = resized
    norm = float(np.linalg.norm(values))
    if norm > 0:
        values = values / norm
    return [float(item) for item in values]
