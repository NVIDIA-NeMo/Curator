# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""Introspection utilities for discovering NeMo Curator stages and components.

This module provides functions to dynamically discover processing stages,
filters, classifiers, and other components from the NeMo Curator codebase.
When nemo_curator is not installed, it falls back to static definitions.
"""
from __future__ import annotations

import inspect
import re
from dataclasses import dataclass, field
from typing import Any, get_type_hints

# Try to import NeMo Curator
try:
    from nemo_curator.stages.base import CompositeStage, ProcessingStage
    from nemo_curator.stages.resources import Resources

    NEMO_CURATOR_AVAILABLE = True
except ImportError:
    ProcessingStage = None  # type: ignore[misc,assignment]
    CompositeStage = None  # type: ignore[misc,assignment]
    Resources = None  # type: ignore[misc,assignment]
    NEMO_CURATOR_AVAILABLE = False


@dataclass
class StageInfo:
    """Information about a NeMo Curator processing stage."""

    name: str
    module_path: str
    stage_type: str  # "ProcessingStage", "CompositeStage", "WorkflowBase"
    requires_gpu: bool
    description: str
    parameters: dict[str, dict[str, Any]] = field(default_factory=dict)
    stage_class: type | None = None

    @property
    def full_target(self) -> str:
        """Return the full target path for Hydra configuration."""
        return f"{self.module_path}.{self.name}"

    @property
    def source_path(self) -> str:
        """Return a relative source path for display."""
        # Convert module path to relative path
        path = self.module_path.replace("nemo_curator.", "").replace(".", "/")
        return path


def _get_type_name(annotation: Any) -> str:
    """Convert a type annotation to a readable string."""
    if annotation is inspect.Parameter.empty:
        return "Any"
    if hasattr(annotation, "__name__"):
        return annotation.__name__
    return str(annotation).replace("typing.", "")


def _extract_first_sentence(docstring: str | None) -> str:
    """Extract first sentence from docstring as description."""
    if not docstring:
        return "No description available"
    lines = docstring.strip().split("\n")
    first_para = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            break
        first_para.append(stripped)
    text = " ".join(first_para)
    match = re.match(r"^(.+?[.!?])\s", text + " ")
    if match:
        return match.group(1)
    return text[:200] + "..." if len(text) > 200 else text


def _check_requires_gpu(cls: type) -> bool:
    """Check if a stage class requires GPU resources."""
    if not NEMO_CURATOR_AVAILABLE or Resources is None:
        return False

    # Check class attribute 'resources'
    if hasattr(cls, "resources"):
        resources = getattr(cls, "resources", None)
        if isinstance(resources, Resources):
            return resources.requires_gpu

    # Check __init__ for gpu-related parameters
    try:
        sig = inspect.signature(cls.__init__)
        for param_name in sig.parameters:
            if "gpu" in param_name.lower():
                return True
    except (ValueError, TypeError):
        pass

    return False


def _get_stage_type(cls: type) -> str:
    """Determine the stage type from class hierarchy."""
    if not NEMO_CURATOR_AVAILABLE:
        return "ProcessingStage"

    class_name = cls.__name__

    # Check for Workflow pattern
    if "Workflow" in class_name:
        return "WorkflowBase"

    # Check inheritance
    if CompositeStage is not None and issubclass(cls, CompositeStage):
        return "CompositeStage"

    return "ProcessingStage"


def _introspect_parameters(cls: type) -> dict[str, dict[str, Any]]:
    """Introspect __init__ parameters of a class."""
    try:
        sig = inspect.signature(cls.__init__)
    except (ValueError, TypeError):
        return {}

    try:
        hints = get_type_hints(cls.__init__)
    except Exception:
        hints = {}

    parameters: dict[str, dict[str, Any]] = {}
    for param_name, param in sig.parameters.items():
        if param_name in ("self", "args", "kwargs"):
            continue

        param_type = hints.get(param_name, param.annotation)
        param_info: dict[str, Any] = {
            "type": _get_type_name(param_type),
        }

        if param.default is not inspect.Parameter.empty:
            param_info["default"] = param.default

        parameters[param_name] = param_info

    return parameters


def get_stage_info(cls: type) -> StageInfo:
    """Extract StageInfo from a stage class."""
    return StageInfo(
        name=cls.__name__,
        module_path=cls.__module__,
        stage_type=_get_stage_type(cls),
        requires_gpu=_check_requires_gpu(cls),
        description=_extract_first_sentence(cls.__doc__),
        parameters=_introspect_parameters(cls),
        stage_class=cls,
    )


def _is_stage_class(cls: type) -> bool:
    """Check if a class is a valid processing stage."""
    if not NEMO_CURATOR_AVAILABLE:
        return False

    if not isinstance(cls, type):
        return False

    # Must be a subclass of ProcessingStage or CompositeStage
    if ProcessingStage is not None and issubclass(cls, ProcessingStage):
        # Exclude the base class itself
        if cls is ProcessingStage:
            return False
        return True

    if CompositeStage is not None and issubclass(cls, CompositeStage):
        if cls is CompositeStage:
            return False
        return True

    # Check for Workflow classes (they may not inherit from ProcessingStage)
    if "Workflow" in cls.__name__ and hasattr(cls, "run"):
        return True

    return False


def discover_stages_in_module(module_path: str) -> list[StageInfo]:
    """Discover all stages in a given module path.

    Args:
        module_path: Dotted module path (e.g., "nemo_curator.stages.text.filters")

    Returns:
        List of StageInfo objects for discovered stages
    """
    if not NEMO_CURATOR_AVAILABLE:
        return []

    try:
        import importlib

        module = importlib.import_module(module_path)
    except ImportError:
        return []

    stages = []

    # Check __all__ if available
    if hasattr(module, "__all__"):
        names = module.__all__
    else:
        names = dir(module)

    for name in names:
        if name.startswith("_"):
            continue

        try:
            obj = getattr(module, name, None)
            if obj is None:
                continue

            if _is_stage_class(obj):
                # Only include if defined in this module or a submodule
                if obj.__module__.startswith(module_path.rsplit(".", 1)[0]):
                    stages.append(get_stage_info(obj))
        except Exception:
            continue

    return stages


def discover_all_stages() -> list[StageInfo]:
    """Discover all stages across all NeMo Curator modules.

    Returns:
        List of StageInfo objects for all discovered stages
    """
    if not NEMO_CURATOR_AVAILABLE:
        return []

    # Module paths to scan for stages
    module_paths = [
        # Text
        "nemo_curator.stages.text.io.reader",
        "nemo_curator.stages.text.io.writer",
        "nemo_curator.stages.text.filters",
        "nemo_curator.stages.text.filters.heuristic_filter",
        "nemo_curator.stages.text.filters.code",
        "nemo_curator.stages.text.filters.fasttext_filter",
        "nemo_curator.stages.text.classifiers",
        "nemo_curator.stages.text.modifiers",
        "nemo_curator.stages.text.modules",
        "nemo_curator.stages.text.embedders",
        "nemo_curator.stages.text.download",
        # Deduplication
        "nemo_curator.stages.deduplication.exact",
        "nemo_curator.stages.deduplication.fuzzy",
        "nemo_curator.stages.deduplication.fuzzy.workflow",
        "nemo_curator.stages.deduplication.semantic",
        # Video
        "nemo_curator.stages.video.io",
        "nemo_curator.stages.video.io.video_reader",
        "nemo_curator.stages.video.clipping",
        "nemo_curator.stages.video.clipping.transnetv2_extraction",
        "nemo_curator.stages.video.clipping.clip_extraction_stages",
        "nemo_curator.stages.video.caption",
        "nemo_curator.stages.video.caption.caption_generation",
        "nemo_curator.stages.video.embedding",
        "nemo_curator.stages.video.embedding.cosmos_embed1",
        "nemo_curator.stages.video.filtering",
        "nemo_curator.stages.video.filtering.motion_filter",
        # Image
        "nemo_curator.stages.image.embedders",
        "nemo_curator.stages.image.embedders.clip_embedder",
        "nemo_curator.stages.image.filters",
        # Audio
        "nemo_curator.stages.audio.inference",
        "nemo_curator.stages.audio.inference.asr_nemo",
        "nemo_curator.stages.audio.metrics",
    ]

    all_stages: dict[str, StageInfo] = {}  # Use dict to deduplicate by name

    for module_path in module_paths:
        for stage in discover_stages_in_module(module_path):
            # Use full target as key to avoid duplicates
            key = stage.full_target
            if key not in all_stages:
                all_stages[key] = stage

    return list(all_stages.values())
