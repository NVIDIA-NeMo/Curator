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

"""Full SNS forward extraction for images, audio, and video."""

from __future__ import annotations

import importlib.util
import io
import logging
import sys
import tempfile
import types
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


class ForwardModelStore:
    """Lazy loader for SNS forward extraction models."""

    def __init__(self, config: Any, device: str, offline_mode: bool = False):
        self.config = config
        self.device = device
        self.offline_mode = offline_mode
        self._amdetr_model: Any | None = None
        self._amdetr_available: bool | None = None
        self._videomomentdetr_model: Any | None = None
        self._videomomentdetr_available: bool | None = None

    def amdetr_model(self) -> Any:
        if self._amdetr_available is False:
            raise RuntimeError("AM-DETR model is unavailable from a previous load attempt")
        if self._amdetr_model is not None:
            return self._amdetr_model
        try:
            self._amdetr_model = self._load_amdetr_model()
            self._amdetr_available = True
            return self._amdetr_model
        except Exception as exc:
            self._amdetr_available = False
            raise RuntimeError(
                "Audio forward extraction requires AM-DETR. Install SNS/Lighthouse extras and "
                "allow Hugging Face access or cache lighthouse-emnlp2024/AM-DETR."
            ) from exc

    def video_moment_model(self) -> Any:
        if self._videomomentdetr_available is False:
            raise RuntimeError("CG-DETR model is unavailable from a previous load attempt")
        if self._videomomentdetr_model is not None:
            return self._videomomentdetr_model
        try:
            self._videomomentdetr_model = self._load_cgdetr_model()
            self._videomomentdetr_available = True
            return self._videomomentdetr_model
        except Exception as exc:
            self._videomomentdetr_available = False
            checkpoint = _checkpoint_path(self.config)
            raise RuntimeError(
                "Video forward extraction requires CG-DETR from the lighthouse package and "
                f"a checkpoint at {checkpoint}. Place the QVHighlights clip checkpoint there "
                "or set sns.cg_detr_checkpoint."
            ) from exc

    def unload(self) -> None:
        self._amdetr_model = None
        self._videomomentdetr_model = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            return

    def _load_cgdetr_model(self) -> Any:
        import easydict
        import torch
        from lighthouse.models import CGDETRPredictor

        checkpoint = _checkpoint_path(self.config)
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        torch.serialization.add_safe_globals([easydict.EasyDict])
        original_torch_load = torch.load

        def patched_torch_load(*args: Any, **kwargs: Any) -> Any:
            kwargs["weights_only"] = False
            return original_torch_load(*args, **kwargs)

        torch.load = patched_torch_load
        try:
            return CGDETRPredictor(str(checkpoint), device=self.device, feature_name="clip")
        finally:
            torch.load = original_torch_load

    def _load_amdetr_model(self) -> Any:
        import torch
        from huggingface_hub import snapshot_download

        repo_id = str(getattr(self.config, "amdetr_repo_id", "lighthouse-emnlp2024/AM-DETR"))
        model_dir = snapshot_download(repo_id, local_files_only=self.offline_mode)
        package_name = "omnifuse_amdetr_runtime"
        if package_name not in sys.modules:
            pkg = types.ModuleType(package_name)
            pkg.__path__ = [model_dir]
            pkg.__package__ = package_name
            sys.modules[package_name] = pkg
        config_module = _load_module(
            f"{package_name}.configuration_amdetr",
            Path(model_dir) / "configuration_amdetr.py",
            package_name,
        )
        sys.modules["configuration_amdetr"] = config_module
        model_module = _load_module(
            f"{package_name}.modeling_amdetr",
            Path(model_dir) / "modeling_amdetr.py",
            package_name,
        )
        sys.modules["modeling_amdetr"] = model_module
        config = config_module.AMDETRConfig(device="cpu")
        model = model_module.AMDETRPredictorWrapper(config)
        state_dict = torch.load(Path(model_dir) / "pytorch_model.bin", map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict, strict=False)
        try:
            model = model.to(self.device)
        except NotImplementedError:
            model = model.to_empty(device=self.device)
            model.load_state_dict(state_dict, strict=False, assign=True)
        model.eval()
        return model


def forward_extract_image(raw_data: Any, annotation: str, config: Any) -> tuple[Any, dict[str, Any]]:
    from PIL import Image, ImageOps

    from omnifuse_tutorial.sns.model_utils.grounding_dino import (
        calculate_lurl_from_xywh,
        calculate_min_span_bbox,
        get_bboxes,
    )

    original = raw_data
    image = Image.open(raw_data) if isinstance(raw_data, (str, Path)) else raw_data
    image = ImageOps.exif_transpose(image)
    if image.mode != "RGB":
        image = image.convert("RGB")
    components = annotation_components(annotation, config)
    kept: list[tuple[float, float, float, float]] = []
    component_scores: list[float] = []
    for component in components:
        boxes, scores = get_bboxes(
            image,
            component,
            model_id=str(getattr(config, "grounding_dino_model_id", "IDEA-Research/grounding-dino-tiny")),
            box_threshold=float(getattr(config, "tau_forward_image", 0.30)),
        )
        kept.extend(boxes)
        component_scores.extend(scores)
    if not kept:
        return original, {
            "reason": "no_bboxes_above_threshold",
            "components": components,
            "threshold": float(getattr(config, "tau_forward_image", 0.30)),
        }
    span = calculate_min_span_bbox(kept)
    if span is None:
        return original, {"reason": "no_bboxes_above_threshold", "components": components}
    padding = int(getattr(config, "bbox_padding_px", 0))
    crop_box = calculate_lurl_from_xywh(span, padding=padding, image_width=image.width, image_height=image.height)
    output_path = _output_dir(config) / f"{uuid.uuid4()}.jpg"
    image.crop(crop_box).save(output_path)
    return output_path, {
        "reason": "image_grounding_dino_crop",
        "components": components,
        "boxes": kept,
        "scores": component_scores,
        "union_bbox": crop_box,
        "threshold": float(getattr(config, "tau_forward_image", 0.30)),
        "bbox_padding_px": padding,
    }


def forward_extract_audio(
    raw_data: Any,
    annotation: str,
    config: Any,
    model_store: ForwardModelStore,
) -> tuple[Any, dict[str, Any]]:
    import librosa
    import soundfile as sf
    import torch
    import torch.nn.functional as F

    components = annotation_components(annotation, config)
    audio_input: str | io.BytesIO
    temp_path: Path | None = None
    try:
        if isinstance(raw_data, (str, Path)):
            audio_input = str(raw_data)
            waveform_np, sample_rate = librosa.load(str(raw_data), sr=None, mono=False)
        elif isinstance(raw_data, dict):
            audio_array = raw_data.get("audio")
            sample_rate = int(raw_data.get("sample_rate", 16000))
            if audio_array is None:
                return raw_data, {"reason": "no_audio_in_dict"}
            temp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            temp_path = Path(temp.name)
            temp.close()
            audio_to_save = audio_array if getattr(audio_array, "ndim", 1) == 1 else audio_array.T
            sf.write(str(temp_path), audio_to_save, sample_rate)
            audio_input = str(temp_path)
            waveform_np = audio_array if getattr(audio_array, "ndim", 1) == 2 else audio_array[None, :]
        else:
            audio_input = io.BytesIO(raw_data)
            waveform_np, sample_rate = librosa.load(io.BytesIO(raw_data), sr=None, mono=False)
        if waveform_np.ndim == 1:
            waveform_np = waveform_np[None, :]
        waveform = torch.from_numpy(waveform_np).float()
        model = model_store.amdetr_model()
        feats = model.encode_audio(audio_path=audio_input)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

    device = next(model.parameters()).device
    feats_device = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in feats.items()
    }
    all_windows: list[tuple[float, float, float]] = []
    for component in components:
        query_feats, query_mask = model._text_encoder.encode(component)
        if model._feature_name != "resnet_glove":
            query_feats = F.normalize(query_feats, dim=-1, eps=1e-5)
        query_feats = query_feats.to(device)
        query_mask = query_mask.to(device)
        model_inputs = {
            "src_vid": feats_device.get("video_feats", feats_device.get("audio_feats")),
            "src_vid_mask": feats_device.get("video_mask"),
            "src_txt": query_feats,
            "src_txt_mask": query_mask,
            "src_aud": feats_device.get("audio_feats"),
        }
        model_inputs = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in model_inputs.items()
        }
        with torch.inference_mode():
            outputs = model._model(**model_inputs)
        ranked_moments, _ = model._post_processing(model_inputs, outputs)
        for start, end, score in ranked_moments:
            if float(score) >= float(getattr(config, "tau_forward_audio", 0.25)):
                all_windows.append((float(start), float(end), float(score)))

    selected = _select_non_overlapping(
        all_windows,
        max_segments=int(getattr(config, "max_audio_segments", 5)),
        min_duration=float(getattr(config, "min_segment_duration", 2.0)),
    )
    if not selected:
        return raw_data, {"reason": "no_audio_segments_above_threshold", "total_segments": len(all_windows)}

    segments = [waveform[:, int(start * sample_rate) : int(end * sample_rate)] for start, end in selected]
    concatenated = torch.cat(segments, dim=1)
    output_path = _output_dir(config) / f"{uuid.uuid4()}_audio.wav"
    sf.write(str(output_path), concatenated.numpy().T, sample_rate, format="WAV")
    return output_path, {
        "reason": "audio_amdetr_segments",
        "components": components,
        "segments_kept": len(selected),
        "total_segments": len(all_windows),
        "kept_intervals": selected,
        "threshold": float(getattr(config, "tau_forward_audio", 0.25)),
    }


def forward_extract_video(
    raw_data: Any,
    annotation: str,
    config: Any,
    model_store: ForwardModelStore,
) -> tuple[Any, dict[str, Any]]:
    import cv2
    import numpy as np
    import torch

    components = annotation_components(annotation, config)
    if isinstance(raw_data, dict):
        video_path = raw_data.get("file_path")
        preloaded_frames = raw_data.get("frames")
    else:
        video_path = str(raw_data) if isinstance(raw_data, (str, Path)) else None
        preloaded_frames = None
    if not video_path:
        return raw_data, {"reason": "no_video_path"}

    model = model_store.video_moment_model()
    feats = model.encode_video(video_path=str(video_path))
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if preloaded_frames is not None:
        frames = preloaded_frames
        cap.release()
    else:
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
    if not frames:
        return raw_data, {"reason": "no_frames"}

    all_windows: list[tuple[float, float, float]] = []
    for component in components:
        prediction = model.predict(component, feats)
        for start, end, score in prediction.get("pred_relevant_windows", []):
            if float(score) >= float(getattr(config, "tau_forward_video", 0.20)):
                all_windows.append((float(start), float(end), float(score)))
    selected = _select_non_overlapping(
        all_windows,
        max_segments=int(getattr(config, "max_video_segments", 5)),
        min_duration=float(getattr(config, "min_segment_duration", 2.0)),
    )
    if not selected:
        return raw_data, {"reason": "no_video_segments_above_threshold", "total_segments": len(all_windows)}

    video_tensor = (
        torch.from_numpy(np.array(frames)).float() if isinstance(frames, list) else torch.from_numpy(frames).float()
    )
    segments = [video_tensor[int(start * fps) : int(end * fps)] for start, end in selected]
    concatenated = torch.cat(segments, dim=0)
    output_path = _output_dir(config) / f"{uuid.uuid4()}.mp4"
    height, width = int(concatenated.shape[1]), int(concatenated.shape[2])
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for index in range(concatenated.shape[0]):
        frame = concatenated[index].numpy().astype(np.uint8)
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    return output_path, {
        "reason": "video_cgdetr_segments",
        "components": components,
        "segments_kept": len(selected),
        "total_segments": len(all_windows),
        "kept_intervals": selected,
        "threshold": float(getattr(config, "tau_forward_video", 0.20)),
    }


def annotation_components(annotation: str, config: Any) -> list[str]:
    if not bool(getattr(config, "use_ann_components", True)):
        return [annotation] if annotation.strip() else []
    try:
        nlp = _spacy_model()
    except Exception as exc:
        raise RuntimeError(
            "SNS annotation component extraction requires spaCy model en_core_web_sm. "
            "Install it with `python -m spacy download en_core_web_sm`, or set "
            "sns.use_ann_components: false to use whole annotations."
        ) from exc
    doc = nlp(annotation)
    components: list[str] = []
    for chunk in doc.noun_chunks:
        tokens = [token.text for token in chunk if token.pos_ != "PRON"]
        if tokens:
            components.append(" ".join(tokens))
    for token in doc:
        if token.pos_ == "VERB":
            phrase_tokens = []
            for candidate in doc:
                if candidate == token or (
                    candidate.head == token and candidate.pos_ in {"VERB", "ADV", "NOUN", "PROPN", "ADJ"}
                ):
                    phrase_tokens.append(candidate.text)
            if phrase_tokens:
                components.append(" ".join(phrase_tokens))
    seen = set()
    unique = []
    for component in components or [annotation]:
        normalized = component.strip()
        lowered = normalized.lower()
        if normalized and lowered not in seen:
            seen.add(lowered)
            unique.append(normalized)
    return unique


@lru_cache(maxsize=1)
def _spacy_model() -> Any:
    import spacy

    return spacy.load("en_core_web_sm")


def _select_non_overlapping(
    windows: list[tuple[float, float, float]],
    max_segments: int,
    min_duration: float,
) -> list[tuple[float, float]]:
    selected: list[tuple[float, float]] = []
    for start, end, _ in sorted(windows, key=lambda item: item[2], reverse=True):
        if end - start < min_duration:
            continue
        if any(start < kept_end and end > kept_start for kept_start, kept_end in selected):
            continue
        selected.append((start, end))
        if len(selected) >= max_segments:
            break
    return sorted(selected, key=lambda item: item[0])


def _output_dir(config: Any) -> Path:
    output_dir = getattr(config, "sns_output_dir", None) or Path("outputs") / "sns"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def _checkpoint_path(config: Any) -> Path:
    value = getattr(config, "cg_detr_checkpoint", None) or Path("model_files") / "best.ckpt"
    return Path(value)


def _load_module(module_name: str, path: Path, package_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path, submodule_search_locations=[str(path.parent)])
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    module.__package__ = package_name
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
