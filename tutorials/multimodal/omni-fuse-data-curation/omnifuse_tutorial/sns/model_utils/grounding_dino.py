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

"""Grounding-DINO helpers for image forward extraction."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)
DEFAULT_GROUNDING_DINO_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
DEFAULT_BOX_THRESHOLD = 0.05


@dataclass(frozen=True)
class GroundingDinoDetector:
    processor: Any
    model: Any
    device: Any


def _device() -> Any:
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend and mps_backend.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@lru_cache(maxsize=4)
def _load_detector(model_id: str) -> GroundingDinoDetector:
    try:
        import torch
        from huggingface_hub.utils import LocalEntryNotFoundError
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    except ImportError as exc:
        raise RuntimeError(
            "Grounding-DINO forward extraction requires transformers and huggingface_hub. "
            "Install the local extras with `python -m pip install -e '.[full]'`."
        ) from exc

    device = _device()
    try:
        processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id, local_files_only=True).to(device)
    except (LocalEntryNotFoundError, OSError, ValueError):
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    model.eval()
    return GroundingDinoDetector(processor=processor, model=model, device=device)


def get_bboxes(
    image: Any,
    prompt: str,
    model_id: str = DEFAULT_GROUNDING_DINO_MODEL_ID,
    box_threshold: float = DEFAULT_BOX_THRESHOLD,
) -> tuple[list[tuple[float, float, float, float]], list[float]]:
    """Return Grounding-DINO boxes as xywh tuples plus confidence scores."""

    import torch
    from PIL import Image

    if not isinstance(image, Image.Image):
        raise TypeError(f"Expected PIL.Image.Image, got {type(image)!r}")
    if image.mode != "RGB":
        image = image.convert("RGB")
    prompt = _normalize_prompt(prompt)
    detector = _load_detector(model_id)
    inputs = detector.processor(images=image, text=prompt, return_tensors="pt")
    model_inputs = {key: value.to(detector.device) if hasattr(value, "to") else value for key, value in inputs.items()}
    with torch.inference_mode():
        outputs = detector.model(**model_inputs)
    results = detector.processor.post_process_grounded_object_detection(
        outputs,
        model_inputs["input_ids"],
        threshold=box_threshold,
        target_sizes=[image.size[::-1]],
    )
    result = results[0]
    boxes = [_xyxy_to_xywh(box) for box in result["boxes"].detach().cpu().tolist()]
    scores = [float(score) for score in result["scores"].detach().cpu().tolist()]
    return boxes, scores


def calculate_min_span_bbox(
    bboxes: list[tuple[float, float, float, float]],
) -> tuple[float, float, float, float] | None:
    if not bboxes:
        return None
    min_x = np.inf
    min_y = np.inf
    max_x = -np.inf
    max_y = -np.inf
    for x, y, width, height in bboxes:
        min_x = min(min_x, x, x + width)
        min_y = min(min_y, y, y + height)
        max_x = max(max_x, x, x + width)
        max_y = max(max_y, y, y + height)
    return float(min_x), float(min_y), float(max_x - min_x), float(max_y - min_y)


def calculate_lurl_from_xywh(
    bbox: tuple[float, float, float, float],
    padding: int = 0,
    image_width: int = 0,
    image_height: int = 0,
) -> tuple[int, int, int, int]:
    left, upper, width, height = bbox
    right = left + width
    lower = upper + height
    if padding > 0:
        left = max(0, left - padding)
        upper = max(0, upper - padding)
        right = min(image_width, right + padding)
        lower = min(image_height, lower + padding)
    return int(left), int(upper), int(right), int(lower)


def _xyxy_to_xywh(box: list[float]) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = box
    return float(x0), float(y0), float(x1 - x0), float(y1 - y0)


def _normalize_prompt(prompt: str) -> str:
    normalized = prompt.strip()
    if not normalized:
        raise ValueError("Grounding-DINO prompt cannot be empty")
    if not normalized.endswith("."):
        normalized = f"{normalized}."
    return normalized
