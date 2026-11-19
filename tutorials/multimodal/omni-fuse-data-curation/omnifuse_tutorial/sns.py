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

"""Symmetric Nucleus Subsampling (SNS): keep only the raw and annotation content that explain each other."""

from __future__ import annotations

import functools
import importlib.util
import logging
import re
import sys
import types
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F

from omnifuse_tutorial.data import annotation_text, cosine, fingerprint, raw_text, text_or_none, unit
from omnifuse_tutorial.models import OmniEmbed, resolve_device
from omnifuse_tutorial.nvidia_api import NvidiaApi

if TYPE_CHECKING:
    from collections.abc import Callable

    from omnifuse_tutorial.config import ExperimentConfig, SNSConfig

logger = logging.getLogger(__name__)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")
MEDIA = ("image", "audio", "video")
SNS_PROMPTS = {
    "image": "Describe the annotation-relevant visual content in this image in detail.",
    "audio": "Transcribe and describe the annotation-relevant sounds in this audio in detail.",
    "video": "Describe the annotation-relevant events and actions in this video in detail.",
}


class SNS:
    """Forward SNS trims the raw side to annotation-relevant content; backward SNS trims annotation sentences.

    A candidate is kept only if it passes the mutual-information gate
    `sim(candidate) >= mi_ratio * max(sim(original), mi_eps)`.
    Descriptions come from the NVIDIA API, similarity from local Omni-Embed, and media extraction from
    Grounding-DINO (image), AM-DETR (audio), and CG-DETR (video).
    """

    def __init__(self, config: ExperimentConfig):
        self.config, self.dim, self.api = config.sns, config.eee.embedding_dim, NvidiaApi(config.eee)
        device = resolve_device(config.runtime)
        self.omni = OmniEmbed(config.sns.nvidia_model, device)
        self.extractor = ForwardExtractor(
            config.sns, config.sns.sns_output_dir or config.run_dir / "sns" / "media", device
        )
        self.text_cache, self.media_cache, self.descriptions = {}, {}, {}

    def process_record(self, record: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return the SNS-updated record and its manifest row."""
        try:
            return self._process(record)
        except Exception as exc:
            if not self.config.continue_on_error:
                raise
            logger.warning("[%s] SNS failed, keeping original: %s: %s", record.get("pair_id"), type(exc).__name__, exc)
            output = {
                **record,
                "sns_raw_text": text_or_none(record.get("raw_text")),
                "sns_annotation": text_or_none(record.get("annotation")) or "",
            }
            manifest = self._manifest(record, output, self.config.enabled, accepted=False, reason="processing_error")
            return output, {
                **manifest,
                "status": "error",
                "error": {"type": type(exc).__name__, "message": str(exc)[:1000]},
            }

    def _process(self, record: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        if not self.config.enabled:
            output = {**record, "sns_raw_text": record.get("raw_text"), "sns_annotation": record.get("annotation")}
            return output, self._manifest(record, output, enabled=False, accepted=False, reason="disabled")
        output = {
            **record,
            "sns_raw_text": text_or_none(record.get("raw_text")),
            "sns_annotation": text_or_none(record.get("annotation")) or "",
        }
        result, decisions = output, []
        if self.config.direction in {"forward", "bidirectional"}:
            result, decision = self._forward(output)
            decisions.append(decision)
        if self.config.direction in {"backward", "bidirectional"}:
            backward, decision = self._backward(output)
            result = {**result, "sns_annotation": backward["sns_annotation"]} if decisions else backward
            decisions.append(decision)
        accepted = any(decision.get("accepted") for decision in decisions)
        manifest = self._manifest(record, result, enabled=True, accepted=accepted, reason="processed")
        return result, {**manifest, "decisions": decisions}

    def _manifest(self, original: dict, output: dict, enabled: bool, accepted: bool, reason: str) -> dict[str, Any]:
        return {
            "pair_id": original["pair_id"],
            "pool": original["pool"],
            "modality": original["modality"],
            "enabled": enabled,
            "direction": self.config.direction,
            "accepted": accepted,
            "reason": reason,
            "original_raw_path": original.get("raw_path"),
            "original_annotation": original.get("annotation"),
            "sns_raw_text": output.get("sns_raw_text"),
            "sns_annotation": output.get("sns_annotation"),
            "thresholds": {key: value for key, value in vars(self.config).items() if key.startswith(("mi_", "tau_"))},
        }

    def _forward(self, record: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        annotation = annotation_text(record)
        if record["modality"] != "text":
            return self._forward_media(record, annotation)
        raw = text_or_none(record.get("raw_text")) or ""
        try:
            components = annotation_components(annotation, self.config)
        except RuntimeError:
            if self.config.require_forward_models:
                raise
            components = [annotation] if annotation.strip() else []
        sentences = _sentences(raw)
        kept = self._keep(sentences, components, self.config.tau_forward_text)
        if not kept:
            return record, {"direction": "forward", "accepted": False, "reason": "no_sentence_above_threshold"}
        candidate = " ".join(kept)
        if not self._mi_gate("text", raw, annotation, candidate, annotation)[0]:
            return record, {"direction": "forward", "accepted": False, "reason": "mi_gate_failed"}
        return {**record, "sns_raw_text": candidate}, {
            "direction": "forward",
            "accepted": True,
            "reason": "text_sentences_kept",
            "kept_sentences": len(kept),
            "total_sentences": len(sentences),
            "annotation_components": components,
        }

    def _backward(self, record: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        description = raw_text(record) or text_or_none(metadata.get("raw_description")) or self._describe(record)
        annotation = annotation_text(record)
        sentences = _sentences(annotation)
        kept = self._keep(sentences, _sentences(description) or [description], self.config.tau_backward)
        if not kept:
            return record, {
                "direction": "backward",
                "accepted": False,
                "reason": "no_annotation_sentence_above_threshold",
            }
        candidate = " ".join(kept)
        modality = str(record.get("modality") or "text")
        if modality == "text":
            raw = raw_text(record) or text_or_none(record.get("raw_path")) or ""
        else:
            raw = record.get("raw_path") or ""
        if not self._mi_gate(modality, raw, annotation, raw, candidate)[0]:
            return record, {"direction": "backward", "accepted": False, "reason": "mi_gate_failed"}
        return {**record, "sns_annotation": candidate}, {
            "direction": "backward",
            "accepted": True,
            "reason": "annotation_sentences_kept",
            "kept_sentences": len(kept),
            "total_sentences": len(sentences),
        }

    def _forward_media(self, record: dict[str, Any], annotation: str) -> tuple[dict[str, Any], dict[str, Any]]:
        modality, raw = str(record.get("modality") or ""), record.get("raw_path")
        if modality not in MEDIA:
            return record, {
                "direction": "forward",
                "accepted": False,
                "reason": "unsupported_media_modality",
                "modality": modality,
            }
        try:
            candidate, extraction = getattr(self.extractor, modality)(raw, annotation)
        except RuntimeError as exc:
            if self.config.require_forward_models:
                raise
            unavailable = {"reason": "forward_model_unavailable", "modality": modality, "error": str(exc)}
            return record, {"direction": "forward", "accepted": False, **unavailable}
        decision = {"direction": "forward", "accepted": False, "modality": modality, "extraction": extraction}
        if candidate is None or str(candidate) == str(raw):
            return record, {**decision, "reason": extraction.get("reason", "no_media_nucleus")}
        passed, gate = self._mi_gate(modality, raw, annotation, candidate, annotation)
        if not passed:
            return record, {**decision, **gate, "reason": "mi_gate_failed"}
        accepted = {
            **decision,
            **gate,
            "accepted": True,
            "reason": extraction.get("reason", "media_nucleus_extracted"),
        }
        return {**record, "raw_path": str(candidate), "sns_raw_text": None}, {
            **accepted,
            "output_path": str(candidate),
        }

    def _mi_gate(
        self, modality: str, raw: Any, annotation: str, candidate_raw: Any, candidate_annotation: str
    ) -> tuple[bool, dict]:
        original = self.similarity(raw, modality, annotation)
        candidate = self.similarity(candidate_raw, modality, candidate_annotation)
        threshold = self.config.mi_ratio * max(original, self.config.mi_eps)
        gate = {"original_similarity": original, "candidate_similarity": candidate, "mi_threshold": threshold}
        return candidate >= threshold, gate

    def similarity(self, raw: Any, modality: str, text: str) -> float:
        """Raw-vs-text Omni-Embed cosine; modalities other than text and media score 0."""
        if modality in MEDIA:
            return cosine(self._embed_media(raw, modality), self._embed_texts([text])[0])
        if modality != "text":
            return 0.0
        left, right = text_or_none(raw), text_or_none(text)
        return cosine(*self._embed_texts([left, right])) if left and right else 0.0

    def _keep(self, sentences: list[str], references: list[str], tau: float) -> list[str]:
        """Sentences whose best cosine similarity to any reference reaches `tau`."""
        if not sentences or not references:
            return []
        left = np.asarray(self._embed_texts(sentences), dtype=np.float32)
        right = np.asarray(self._embed_texts(references), dtype=np.float32)
        scores = left @ right.T
        return [sentence for sentence, row in zip(sentences, scores, strict=True) if float(row.max()) >= tau]

    def _embed_texts(self, texts: list[str]) -> list[np.ndarray]:
        if missing := list(dict.fromkeys(text for text in texts if text not in self.text_cache)):
            vectors = self.omni.encode_texts(missing, self.config.embedding_batch_size)
            self.text_cache.update(zip(missing, [unit(vector, self.dim) for vector in vectors], strict=True))
        return [self.text_cache[text] for text in texts]

    def _embed_media(self, path: Any, modality: str) -> np.ndarray:
        key = (modality, str(path))
        if key not in self.media_cache:
            self.media_cache[key] = unit(self.omni.encode_media(path, modality), self.dim)
        return self.media_cache[key]

    def _describe(self, record: dict[str, Any]) -> str:
        return raw_text(record) or self._describe_media(record.get("raw_path"), str(record.get("modality") or "text"))

    def _describe_media(self, value: Any, modality: str) -> str:
        if modality == "text":
            return text_or_none(value) or ""
        path = Path(value) if isinstance(value, (str, Path)) else None
        if path is None:
            return f"modality {modality} {value!r}"
        if not path.exists() or modality not in SNS_PROMPTS:
            return f"modality {modality} {' '.join(re.findall(r'[a-zA-Z0-9]+', path.stem))} {fingerprint(path)}"
        key = f"{modality}:{path}"
        if key not in self.descriptions:
            self.descriptions[key] = self.api.describe(path, modality, SNS_PROMPTS[modality])
        return self.descriptions[key]


class ForwardExtractor:
    """Forward SNS media extraction: Grounding-DINO crops, AM-DETR audio moments, CG-DETR video moments."""

    def __init__(self, config: SNSConfig, output_dir: Path, device: str):
        self.config, self.output_dir, self.device = config, Path(output_dir), device
        self.models, self.failed = {}, set()

    def image(self, raw: str, annotation: str) -> tuple[Any, dict[str, Any]]:
        from PIL import Image, ImageOps

        image = ImageOps.exif_transpose(Image.open(raw)).convert("RGB")
        components, tau = annotation_components(annotation, self.config), self.config.tau_forward_image
        boxes, scores = [], []
        for component in components:
            found, found_scores = grounding_dino_boxes(image, component, self.config.grounding_dino_model_id, tau)
            boxes += found
            scores += found_scores
        if not boxes:
            return raw, {"reason": "no_bboxes_above_threshold", "components": components, "threshold": tau}
        xs = [value for x, _, width, _ in boxes for value in (x, x + width)]
        ys = [value for _, y, _, height in boxes for value in (y, y + height)]
        left, upper = min(xs), min(ys)
        width, height = max(xs) - left, max(ys) - upper
        right, lower, padding = left + width, upper + height, self.config.bbox_padding_px
        if padding > 0:
            left, upper = max(0, left - padding), max(0, upper - padding)
            right, lower = min(image.width, right + padding), min(image.height, lower + padding)
        crop_box = (int(left), int(upper), int(right), int(lower))
        output = self._output_path(".jpg")
        image.crop(crop_box).save(output)
        return output, {
            "reason": "image_grounding_dino_crop",
            "components": components,
            "boxes": boxes,
            "scores": scores,
            "union_bbox": crop_box,
            "threshold": tau,
            "bbox_padding_px": padding,
        }

    def audio(self, raw: str, annotation: str) -> tuple[Any, dict[str, Any]]:
        import librosa
        import soundfile as sf

        components, tau = annotation_components(annotation, self.config), self.config.tau_forward_audio
        waveform, rate = librosa.load(str(raw), sr=None, mono=False)
        waveform = waveform[None, :] if waveform.ndim == 1 else waveform
        hint = (
            "Audio forward extraction requires AM-DETR. Install SNS/Lighthouse extras and allow Hugging Face access "
            f"or cache {self.config.amdetr_repo_id}."
        )
        model = self._model("AM-DETR", self._load_amdetr, hint)
        device = next(model.parameters()).device
        features = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in model.encode_audio(audio_path=str(raw)).items()
        }
        windows = []
        for component in components:
            query, mask = model._text_encoder.encode(component)
            if model._feature_name != "resnet_glove":
                query = F.normalize(query, dim=-1, eps=1e-5)
            inputs = {
                "src_vid": features.get("video_feats", features.get("audio_feats")),
                "src_vid_mask": features.get("video_mask"),
                "src_txt": query.to(device),
                "src_txt_mask": mask.to(device),
                "src_aud": features.get("audio_feats"),
            }
            with torch.inference_mode():
                outputs = model._model(**inputs)
            moments, _ = model._post_processing(inputs, outputs)
            windows += [
                (float(start), float(end), float(score)) for start, end, score in moments if float(score) >= tau
            ]
        selected = _select_windows(windows, self.config.max_audio_segments, self.config.min_segment_duration)
        if not selected:
            return raw, {"reason": "no_audio_segments_above_threshold", "total_segments": len(windows)}
        clip = np.concatenate([waveform[:, int(start * rate) : int(end * rate)] for start, end in selected], axis=1)
        output = self._output_path("_audio.wav")
        sf.write(str(output), clip.T, rate, format="WAV")
        return output, {
            "reason": "audio_amdetr_segments",
            "components": components,
            "segments_kept": len(selected),
            "total_segments": len(windows),
            "kept_intervals": selected,
            "threshold": tau,
        }

    def video(self, raw: str, annotation: str) -> tuple[Any, dict[str, Any]]:
        import cv2

        components, tau = annotation_components(annotation, self.config), self.config.tau_forward_video
        hint = (
            "Video forward extraction requires CG-DETR from the lighthouse package and a checkpoint at "
            f"{self.config.cg_detr_checkpoint}. Place the QVHighlights clip checkpoint there or set sns.cg_detr_checkpoint."
        )
        model = self._model("CG-DETR", self._load_cgdetr, hint)
        features = model.encode_video(video_path=str(raw))
        capture = cv2.VideoCapture(str(raw))
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        frames = []
        ok, frame = capture.read()
        while ok:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            ok, frame = capture.read()
        capture.release()
        if not frames:
            return raw, {"reason": "no_frames"}
        windows = [
            (float(start), float(end), float(score))
            for component in components
            for start, end, score in model.predict(component, features).get("pred_relevant_windows", [])
            if float(score) >= tau
        ]
        selected = _select_windows(windows, self.config.max_video_segments, self.config.min_segment_duration)
        if not selected:
            return raw, {"reason": "no_video_segments_above_threshold", "total_segments": len(windows)}
        video = np.asarray(frames)
        clip = np.concatenate([video[int(start * fps) : int(end * fps)] for start, end in selected])
        output = self._output_path(".mp4")
        writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (clip.shape[2], clip.shape[1]))
        for frame in clip:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        return output, {
            "reason": "video_cgdetr_segments",
            "components": components,
            "segments_kept": len(selected),
            "total_segments": len(windows),
            "kept_intervals": selected,
            "threshold": tau,
        }

    def _output_path(self, suffix: str) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self.output_dir / f"{uuid.uuid4()}{suffix}"

    def _model(self, name: str, load: Callable[[], Any], hint: str) -> Any:
        """Load a forward model once; a failed load is remembered so later records fail fast."""
        if name in self.failed:
            raise RuntimeError(f"{name} model is unavailable from a previous load attempt")
        if name not in self.models:
            try:
                self.models[name] = load()
            except Exception as exc:
                self.failed.add(name)
                raise RuntimeError(hint) from exc
        return self.models[name]

    def _load_cgdetr(self) -> Any:
        import easydict
        from lighthouse.models import CGDETRPredictor

        checkpoint = self.config.cg_detr_checkpoint
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        torch.serialization.add_safe_globals([easydict.EasyDict])
        torch_load = torch.load
        # Lighthouse checkpoints pickle EasyDict configs, which torch>=2.6 rejects with weights_only=True.
        torch.load = lambda *args, **kwargs: torch_load(*args, **{**kwargs, "weights_only": False})
        try:
            return CGDETRPredictor(str(checkpoint), device=self.device, feature_name="clip")
        finally:
            torch.load = torch_load

    def _load_amdetr(self) -> Any:
        from huggingface_hub import snapshot_download

        model_dir = Path(snapshot_download(self.config.amdetr_repo_id))
        package = "omnifuse_amdetr_runtime"
        if package not in sys.modules:
            module = types.ModuleType(package)
            module.__path__, module.__package__ = [str(model_dir)], package
            sys.modules[package] = module
        # The AM-DETR repository's remote code imports its own modules by bare name.
        for name in ("configuration_amdetr", "modeling_amdetr"):
            spec = importlib.util.spec_from_file_location(
                f"{package}.{name}", model_dir / f"{name}.py", submodule_search_locations=[str(model_dir)]
            )
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load module spec for {model_dir / f'{name}.py'}")
            module = importlib.util.module_from_spec(spec)
            module.__package__ = package
            sys.modules[f"{package}.{name}"] = sys.modules[name] = module
            spec.loader.exec_module(module)
        model = sys.modules["modeling_amdetr"].AMDETRPredictorWrapper(
            sys.modules["configuration_amdetr"].AMDETRConfig(device="cpu")
        )
        state_dict = torch.load(model_dir / "pytorch_model.bin", map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict, strict=False)
        try:
            model = model.to(self.device)
        except NotImplementedError:
            model = model.to_empty(device=self.device)
            model.load_state_dict(state_dict, strict=False, assign=True)
        return model.eval()


@functools.lru_cache(maxsize=4)
def _grounding_dino(model_id: str) -> tuple[Any, Any, torch.device]:
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    try:
        processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id, local_files_only=True).to(device)
    except (OSError, ValueError):
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    return processor, model.eval(), device


def grounding_dino_boxes(image: Any, prompt: str, model_id: str, threshold: float) -> tuple[list[tuple], list[float]]:
    """Grounding-DINO boxes as (x, y, width, height) tuples, plus their confidence scores."""
    prompt = prompt.strip()
    if not prompt:
        raise ValueError("Grounding-DINO prompt cannot be empty")
    processor, model, device = _grounding_dino(model_id)
    inputs = processor(images=image, text=prompt if prompt.endswith(".") else f"{prompt}.", return_tensors="pt")
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
    with torch.inference_mode():
        outputs = model(**inputs)
    result = processor.post_process_grounded_object_detection(
        outputs, inputs["input_ids"], threshold=threshold, target_sizes=[image.size[::-1]]
    )[0]
    boxes = [(x0, y0, x1 - x0, y1 - y0) for x0, y0, x1, y1 in result["boxes"].cpu().tolist()]
    return boxes, [float(score) for score in result["scores"].cpu().tolist()]


@functools.lru_cache(maxsize=1)
def _spacy() -> Any:
    import spacy

    return spacy.load("en_core_web_sm")


def annotation_components(annotation: str, config: SNSConfig) -> list[str]:
    """Noun chunks (without pronouns) and verb phrases of an annotation, deduplicated case-insensitively."""
    if not config.use_ann_components:
        return [annotation] if annotation.strip() else []
    try:
        nlp = _spacy()
    except Exception as exc:
        raise RuntimeError(
            "SNS annotation component extraction requires spaCy model en_core_web_sm. Install it with "
            "`python -m spacy download en_core_web_sm`, or set sns.use_ann_components: false to use whole annotations."
        ) from exc
    doc = nlp(annotation)
    components = [" ".join(token.text for token in chunk if token.pos_ != "PRON") for chunk in doc.noun_chunks]
    verb_children = {"VERB", "ADV", "NOUN", "PROPN", "ADJ"}
    for token in doc:
        if token.pos_ == "VERB":
            phrase = [
                item.text for item in doc if item == token or (item.head == token and item.pos_ in verb_children)
            ]
            components.append(" ".join(phrase))
    unique = {}
    for component in [item for item in components if item] or [annotation]:
        if component.strip():
            unique.setdefault(component.strip().lower(), component.strip())
    return list(unique.values())


def _select_windows(
    windows: list[tuple[float, float, float]], max_segments: int, min_duration: float
) -> list[tuple[float, float]]:
    """Highest-scoring non-overlapping windows of at least `min_duration`, returned in time order."""
    selected = []
    for start, end, _ in sorted(windows, key=lambda item: item[2], reverse=True):
        if end - start >= min_duration and not any(
            start < kept_end and end > kept_start for kept_start, kept_end in selected
        ):
            selected.append((start, end))
            if len(selected) >= max_segments:
                break
    return sorted(selected, key=lambda item: item[0])


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in SENTENCE_RE.split(text or "") if part.strip()]
