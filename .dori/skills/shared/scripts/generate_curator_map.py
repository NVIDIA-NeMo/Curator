#!/usr/bin/env python3
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

"""Generate a comprehensive map of NeMo Curator stages, tasks, and pipelines.

This script introspects the nemo_curator package and generates a JSON/YAML map
that can be used by skills without requiring runtime imports. The map includes:

- All processing stages with metadata
- Task types and their compatibility
- Parameter schemas with defaults
- GPU memory requirements
- Import paths

The generated map enables:
- Offline pipeline validation
- Fast stage discovery without imports
- Version tracking of available features

Examples:
    # Generate JSON map
    python generate_curator_map.py --output curator_map.json

    # Generate YAML map
    python generate_curator_map.py --output curator_map.yaml --format yaml

    # Print to stdout
    python generate_curator_map.py --print

    # Generate with specific modality only
    python generate_curator_map.py --modality video --output video_map.json

    # Check what changed from existing map
    python generate_curator_map.py --diff curator_map.json
"""
from __future__ import annotations

import argparse
import inspect
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, get_type_hints

# Try to import NeMo Curator
try:
    import nemo_curator

    from nemo_curator.stages.base import CompositeStage, ProcessingStage
    from nemo_curator.stages.resources import Resources

    NEMO_CURATOR_AVAILABLE = True
    NEMO_CURATOR_VERSION = getattr(nemo_curator, "__version__", "unknown")
except ImportError:
    NEMO_CURATOR_AVAILABLE = False
    NEMO_CURATOR_VERSION = "not_installed"
    ProcessingStage = None  # type: ignore[misc,assignment]
    CompositeStage = None  # type: ignore[misc,assignment]
    Resources = None  # type: ignore[misc,assignment]


# Module paths to scan for each modality
MODALITY_MODULES: dict[str, list[str]] = {
    "text": [
        "nemo_curator.stages.text.io.reader",
        "nemo_curator.stages.text.io.writer",
        "nemo_curator.stages.text.filters",
        "nemo_curator.stages.text.filters.heuristic_filter",
        "nemo_curator.stages.text.filters.code",
        "nemo_curator.stages.text.filters.fasttext_filter",
        "nemo_curator.stages.text.classifiers",
        "nemo_curator.stages.text.classifiers.quality",
        "nemo_curator.stages.text.classifiers.domain",
        "nemo_curator.stages.text.classifiers.fineweb_edu",
        "nemo_curator.stages.text.classifiers.aegis",
        "nemo_curator.stages.text.classifiers.content_type",
        "nemo_curator.stages.text.classifiers.prompt_task_complexity",
        "nemo_curator.stages.text.modifiers",
        "nemo_curator.stages.text.modules",
        "nemo_curator.stages.text.embedders",
        "nemo_curator.stages.text.download",
    ],
    "video": [
        "nemo_curator.stages.video",
        "nemo_curator.stages.video.io",
        "nemo_curator.stages.video.io.video_reader",
        "nemo_curator.stages.video.io.clip_writer",
        "nemo_curator.stages.video.clipping",
        "nemo_curator.stages.video.clipping.transnetv2_extraction",
        "nemo_curator.stages.video.clipping.clip_extraction_stages",
        "nemo_curator.stages.video.clipping.video_frame_extraction",
        "nemo_curator.stages.video.clipping.clip_frame_extraction",
        "nemo_curator.stages.video.caption",
        "nemo_curator.stages.video.caption.caption_preparation",
        "nemo_curator.stages.video.caption.caption_generation",
        "nemo_curator.stages.video.caption.caption_enhancement",
        "nemo_curator.stages.video.embedding",
        "nemo_curator.stages.video.embedding.cosmos_embed1",
        "nemo_curator.stages.video.embedding.internvideo2",
        "nemo_curator.stages.video.filtering",
        "nemo_curator.stages.video.filtering.motion_filter",
        "nemo_curator.stages.video.filtering.clip_aesthetic_filter",
        "nemo_curator.stages.video.preview",
    ],
    "image": [
        "nemo_curator.stages.image",
        "nemo_curator.stages.image.io",
        "nemo_curator.stages.image.io.image_reader",
        "nemo_curator.stages.image.io.image_writer",
        "nemo_curator.stages.image.io.convert",
        "nemo_curator.stages.image.embedders",
        "nemo_curator.stages.image.embedders.clip_embedder",
        "nemo_curator.stages.image.filters",
        "nemo_curator.stages.image.filters.aesthetic_filter",
        "nemo_curator.stages.image.filters.nsfw_filter",
        "nemo_curator.stages.image.deduplication",
        "nemo_curator.stages.image.deduplication.removal",
    ],
    "audio": [
        "nemo_curator.stages.audio",
        "nemo_curator.stages.audio.io",
        "nemo_curator.stages.audio.io.convert",
        "nemo_curator.stages.audio.inference",
        "nemo_curator.stages.audio.inference.asr_nemo",
        "nemo_curator.stages.audio.metrics",
        "nemo_curator.stages.audio.metrics.get_wer",
        "nemo_curator.stages.audio.common",
    ],
    "deduplication": [
        "nemo_curator.stages.deduplication",
        "nemo_curator.stages.deduplication.exact",
        "nemo_curator.stages.deduplication.exact.identification",
        "nemo_curator.stages.deduplication.fuzzy",
        "nemo_curator.stages.deduplication.fuzzy.workflow",
        "nemo_curator.stages.deduplication.fuzzy.minhash",
        "nemo_curator.stages.deduplication.fuzzy.lsh",
        "nemo_curator.stages.deduplication.fuzzy.buckets_to_edges",
        "nemo_curator.stages.deduplication.fuzzy.connected_components",
        "nemo_curator.stages.deduplication.fuzzy.identify_duplicates",
        "nemo_curator.stages.deduplication.semantic",
        "nemo_curator.stages.deduplication.semantic.workflow",
    ],
}

# Task type mappings
TASK_TYPES: dict[str, dict[str, str]] = {
    "DocumentBatch": {
        "modality": "text",
        "data_type": "pd.DataFrame | pa.Table",
        "import": "from nemo_curator.tasks import DocumentBatch",
    },
    "VideoTask": {
        "modality": "video",
        "data_type": "Video",
        "import": "from nemo_curator.tasks.video import VideoTask",
    },
    "ImageBatch": {
        "modality": "image",
        "data_type": "list[ImageObject]",
        "import": "from nemo_curator.tasks.image import ImageBatch",
    },
    "AudioBatch": {
        "modality": "audio",
        "data_type": "dict | list[dict]",
        "import": "from nemo_curator.tasks.audio import AudioBatch",
    },
    "FileGroupTask": {
        "modality": "any",
        "data_type": "list[str]",
        "import": "from nemo_curator.tasks import FileGroupTask",
    },
}

# GPU memory estimates (GB)
GPU_MEMORY_ESTIMATES: dict[str, float] = {
    # Video
    "TransNetV2ClipExtractionStage": 10.0,
    "VideoFrameExtractionStage": 10.0,
    "CaptionGenerationStage": 16.0,
    "CaptionEnhancementStage": 16.0,
    "CosmosEmbed1EmbeddingStage": 20.0,
    "InternVideo2EmbeddingStage": 16.0,
    "ClipAestheticFilterStage": 4.0,
    # Image
    "ImageEmbeddingStage": 4.0,
    "ImageAestheticFilterStage": 4.0,
    "ImageNSFWFilterStage": 4.0,
    # Audio
    "InferenceAsrNemoStage": 16.0,
    # Text classifiers
    "QualityClassifier": 4.0,
    "DomainClassifier": 4.0,
    "FineWebEduClassifier": 8.0,
    "FineWebMixtralEduClassifier": 12.0,
    "FineWebNemotronEduClassifier": 12.0,
    "AegisClassifier": 16.0,
    "ContentTypeClassifier": 4.0,
    # Text embedders
    "EmbeddingCreatorStage": 8.0,
    # Deduplication
    "SemanticDeduplicationWorkflow": 8.0,
}


@dataclass
class ParameterInfo:
    """Information about a stage parameter."""

    name: str
    type: str
    default: Any = None
    required: bool = True
    description: str = ""


@dataclass
class StageInfo:
    """Information about a NeMo Curator processing stage."""

    name: str
    module_path: str
    stage_type: str  # ProcessingStage, CompositeStage, WorkflowBase
    modality: str
    category: str
    description: str
    requires_gpu: bool
    gpu_memory_gb: float
    input_task: str
    output_task: str
    parameters: list[dict[str, Any]] = field(default_factory=list)

    @property
    def import_statement(self) -> str:
        return f"from {self.module_path} import {self.name}"


def _get_type_name(annotation: Any) -> str:
    """Convert a type annotation to a readable string."""
    if annotation is inspect.Parameter.empty:
        return "Any"
    if hasattr(annotation, "__name__"):
        return annotation.__name__
    # Handle generic types
    type_str = str(annotation)
    # Clean up common patterns
    type_str = type_str.replace("typing.", "")
    type_str = type_str.replace("nemo_curator.tasks.", "")
    return type_str


def _extract_first_sentence(docstring: str | None) -> str:
    """Extract first sentence from docstring."""
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
    """Check if a stage requires GPU."""
    if not NEMO_CURATOR_AVAILABLE or Resources is None:
        return False

    # Check resources attribute
    if hasattr(cls, "resources"):
        resources = getattr(cls, "resources", None)
        if isinstance(resources, Resources):
            return resources.requires_gpu

    # Check for GPU-related parameters
    try:
        sig = inspect.signature(cls.__init__)
        for param_name in sig.parameters:
            if "gpu" in param_name.lower():
                return True
    except (ValueError, TypeError):
        pass

    return False


def _infer_modality(module_path: str) -> str:
    """Infer modality from module path."""
    if ".text." in module_path:
        return "text"
    if ".video." in module_path:
        return "video"
    if ".image." in module_path:
        return "image"
    if ".audio." in module_path:
        return "audio"
    if ".deduplication." in module_path:
        return "deduplication"
    return "other"


def _infer_category(module_path: str, class_name: str) -> str:
    """Infer category from module path or class name."""
    parts = module_path.split(".")
    categories = {
        "io", "reader", "writer", "filters", "classifiers", "modifiers",
        "modules", "embedders", "download", "clipping", "caption",
        "captioning", "embedding", "filtering", "deduplication",
        "inference", "metrics", "preview",
    }
    for part in parts:
        if part in categories:
            if part == "caption":
                return "captioning"
            return part

    # Infer from class name
    if "Reader" in class_name or "Writer" in class_name:
        return "io"
    if "Filter" in class_name:
        return "filtering"
    if "Classifier" in class_name:
        return "classifiers"
    if "Embedding" in class_name:
        return "embedding"

    return "other"


def _infer_task_types(cls: type) -> tuple[str, str]:
    """Infer input and output task types from class."""
    input_task = "Task"
    output_task = "Task"

    # Try to get from type hints on process method
    try:
        if hasattr(cls, "process"):
            hints = get_type_hints(cls.process)
            if "task" in hints:
                input_task = _get_type_name(hints["task"])
            if "return" in hints:
                output_task = _get_type_name(hints["return"])
    except Exception:
        pass

    # Infer from class name and modality
    modality = _infer_modality(cls.__module__)
    if modality == "video":
        input_task = output_task = "VideoTask"
    elif modality == "image":
        input_task = output_task = "ImageBatch"
    elif modality == "audio":
        input_task = output_task = "AudioBatch"
    elif modality == "text":
        input_task = output_task = "DocumentBatch"

    return input_task, output_task


def _introspect_parameters(cls: type) -> list[dict[str, Any]]:
    """Introspect parameters from __init__."""
    try:
        sig = inspect.signature(cls.__init__)
    except (ValueError, TypeError):
        return []

    try:
        hints = get_type_hints(cls.__init__)
    except Exception:
        hints = {}

    parameters = []
    for param_name, param in sig.parameters.items():
        if param_name in ("self", "args", "kwargs"):
            continue

        param_type = hints.get(param_name, param.annotation)
        param_info: dict[str, Any] = {
            "name": param_name,
            "type": _get_type_name(param_type),
            "required": param.default is inspect.Parameter.empty,
        }

        if param.default is not inspect.Parameter.empty:
            default = param.default
            # Make default JSON-serializable
            if callable(default) and not isinstance(default, type):
                param_info["default"] = f"<{type(default).__name__}>"
            else:
                try:
                    json.dumps(default)
                    param_info["default"] = default
                except (TypeError, ValueError):
                    param_info["default"] = str(default)

        parameters.append(param_info)

    return parameters


def _get_stage_type(cls: type) -> str:
    """Determine stage type from class hierarchy."""
    if not NEMO_CURATOR_AVAILABLE:
        return "ProcessingStage"

    name = cls.__name__
    if "Workflow" in name:
        return "WorkflowBase"

    if CompositeStage is not None and issubclass(cls, CompositeStage):
        return "CompositeStage"

    return "ProcessingStage"


def _is_stage_class(cls: type) -> bool:
    """Check if a class is a valid stage."""
    if not isinstance(cls, type):
        return False
    if not NEMO_CURATOR_AVAILABLE:
        return False

    # Check for ProcessingStage or CompositeStage
    if ProcessingStage is not None and issubclass(cls, ProcessingStage):
        if cls is ProcessingStage:
            return False
        return True

    if CompositeStage is not None and issubclass(cls, CompositeStage):
        if cls is CompositeStage:
            return False
        return True

    # Check for Workflow pattern
    if "Workflow" in cls.__name__ and hasattr(cls, "run"):
        return True

    return False


def _introspect_stage(cls: type) -> StageInfo:
    """Introspect a stage class."""
    name = cls.__name__
    module_path = cls.__module__
    modality = _infer_modality(module_path)
    input_task, output_task = _infer_task_types(cls)

    return StageInfo(
        name=name,
        module_path=module_path,
        stage_type=_get_stage_type(cls),
        modality=modality,
        category=_infer_category(module_path, name),
        description=_extract_first_sentence(cls.__doc__),
        requires_gpu=_check_requires_gpu(cls),
        gpu_memory_gb=GPU_MEMORY_ESTIMATES.get(name, 0.0),
        input_task=input_task,
        output_task=output_task,
        parameters=_introspect_parameters(cls),
    )


def discover_stages_in_module(module_path: str) -> list[StageInfo]:
    """Discover stages in a module."""
    if not NEMO_CURATOR_AVAILABLE:
        return []

    try:
        import importlib
        module = importlib.import_module(module_path)
    except ImportError:
        return []

    stages = []
    names = getattr(module, "__all__", dir(module))

    for name in names:
        if name.startswith("_"):
            continue

        try:
            obj = getattr(module, name, None)
            if obj is None:
                continue

            if _is_stage_class(obj):
                # Only include if defined in this module hierarchy
                if obj.__module__.startswith(module_path.rsplit(".", 1)[0]):
                    stages.append(_introspect_stage(obj))
        except Exception:
            continue

    return stages


def discover_all_stages(modalities: list[str] | None = None) -> dict[str, list[StageInfo]]:
    """Discover all stages, optionally filtered by modality."""
    if modalities is None:
        modalities = list(MODALITY_MODULES.keys())

    all_stages: dict[str, dict[str, StageInfo]] = {m: {} for m in modalities}

    for modality in modalities:
        if modality not in MODALITY_MODULES:
            continue

        for module_path in MODALITY_MODULES[modality]:
            for stage in discover_stages_in_module(module_path):
                key = f"{stage.module_path}.{stage.name}"
                if key not in all_stages[modality]:
                    all_stages[modality][key] = stage

    return {m: list(stages.values()) for m, stages in all_stages.items()}


def generate_curator_map(modalities: list[str] | None = None) -> dict[str, Any]:
    """Generate the complete curator map."""
    stages_by_modality = discover_all_stages(modalities)

    # Build task compatibility matrix
    task_compatibility: dict[str, list[str]] = {}
    for stages in stages_by_modality.values():
        for stage in stages:
            if stage.input_task not in task_compatibility:
                task_compatibility[stage.input_task] = []
            task_compatibility[stage.input_task].append(stage.name)

    # Build stage lookup
    stages_flat: dict[str, dict[str, Any]] = {}
    for stages in stages_by_modality.values():
        for stage in stages:
            stages_flat[stage.name] = {
                "module_path": stage.module_path,
                "import": stage.import_statement,
                "type": stage.stage_type,
                "modality": stage.modality,
                "category": stage.category,
                "description": stage.description,
                "requires_gpu": stage.requires_gpu,
                "gpu_memory_gb": stage.gpu_memory_gb,
                "input_task": stage.input_task,
                "output_task": stage.output_task,
                "parameters": stage.parameters,
            }

    return {
        "metadata": {
            "version": NEMO_CURATOR_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "generator": "generate_curator_map.py",
            "nemo_curator_available": NEMO_CURATOR_AVAILABLE,
        },
        "stages": stages_flat,
        "stages_by_modality": {
            m: [s.name for s in stages]
            for m, stages in stages_by_modality.items()
        },
        "task_types": TASK_TYPES,
        "task_compatibility": task_compatibility,
        "gpu_memory_estimates": GPU_MEMORY_ESTIMATES,
    }


def diff_maps(old_map: dict[str, Any], new_map: dict[str, Any]) -> dict[str, Any]:
    """Compare two curator maps and report differences."""
    old_stages = set(old_map.get("stages", {}).keys())
    new_stages = set(new_map.get("stages", {}).keys())

    added = new_stages - old_stages
    removed = old_stages - new_stages
    common = old_stages & new_stages

    changed = []
    for name in common:
        old_s = old_map["stages"][name]
        new_s = new_map["stages"][name]
        if old_s != new_s:
            changed.append(name)

    return {
        "old_version": old_map.get("metadata", {}).get("version", "unknown"),
        "new_version": new_map.get("metadata", {}).get("version", "unknown"),
        "stages_added": sorted(added),
        "stages_removed": sorted(removed),
        "stages_changed": sorted(changed),
        "total_old": len(old_stages),
        "total_new": len(new_stages),
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file path (JSON or YAML based on extension)",
    )
    parser.add_argument(
        "--format",
        choices=["json", "yaml"],
        default="json",
        help="Output format (default: json)",
    )
    parser.add_argument(
        "--modality",
        nargs="+",
        choices=list(MODALITY_MODULES.keys()),
        help="Filter to specific modalities",
    )
    parser.add_argument(
        "--print",
        action="store_true",
        dest="print_only",
        help="Print to stdout instead of writing to file",
    )
    parser.add_argument(
        "--diff",
        metavar="EXISTING_MAP",
        help="Compare with existing map and show differences",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show statistics about discovered stages",
    )

    args = parser.parse_args()

    if not NEMO_CURATOR_AVAILABLE:
        print("⚠️  nemo_curator not installed - map will have limited data")
        print("   Install with: pip install nemo-curator")
        print()

    # Generate map
    curator_map = generate_curator_map(args.modality)

    # Handle diff mode
    if args.diff:
        diff_path = Path(args.diff)
        if not diff_path.exists():
            print(f"❌ Diff file not found: {args.diff}")
            sys.exit(1)

        with open(diff_path) as f:
            if diff_path.suffix in (".yaml", ".yml"):
                try:
                    import yaml
                    old_map = yaml.safe_load(f)
                except ImportError:
                    print("❌ PyYAML required for YAML files")
                    sys.exit(1)
            else:
                old_map = json.load(f)

        diff = diff_maps(old_map, curator_map)
        print("📊 Curator Map Diff")
        print("=" * 50)
        print(f"Old version: {diff['old_version']} ({diff['total_old']} stages)")
        print(f"New version: {diff['new_version']} ({diff['total_new']} stages)")
        print()

        if diff["stages_added"]:
            print(f"✅ Added ({len(diff['stages_added'])}):")
            for s in diff["stages_added"]:
                print(f"   + {s}")
            print()

        if diff["stages_removed"]:
            print(f"❌ Removed ({len(diff['stages_removed'])}):")
            for s in diff["stages_removed"]:
                print(f"   - {s}")
            print()

        if diff["stages_changed"]:
            print(f"🔄 Changed ({len(diff['stages_changed'])}):")
            for s in diff["stages_changed"]:
                print(f"   ~ {s}")
            print()

        if not any([diff["stages_added"], diff["stages_removed"], diff["stages_changed"]]):
            print("✅ No differences found")
        return

    # Handle stats mode
    if args.stats:
        print("📊 NeMo Curator Stage Statistics")
        print("=" * 50)
        print(f"Version: {curator_map['metadata']['version']}")
        print(f"Total stages: {len(curator_map['stages'])}")
        print()

        # By modality
        print("By modality:")
        for modality, stages in curator_map["stages_by_modality"].items():
            print(f"  {modality}: {len(stages)} stages")
        print()

        # GPU stages
        gpu_stages = [n for n, s in curator_map["stages"].items() if s["requires_gpu"]]
        print(f"GPU stages: {len(gpu_stages)}")
        print()

        # By type
        types: dict[str, int] = {}
        for stage in curator_map["stages"].values():
            t = stage["type"]
            types[t] = types.get(t, 0) + 1
        print("By stage type:")
        for t, count in sorted(types.items()):
            print(f"  {t}: {count}")
        return

    # Output
    if args.format == "yaml" or (args.output and args.output.endswith((".yaml", ".yml"))):
        try:
            import yaml
            output_str = yaml.dump(curator_map, default_flow_style=False, sort_keys=False)
        except ImportError:
            print("❌ PyYAML required for YAML output. Install with: pip install pyyaml")
            sys.exit(1)
    else:
        output_str = json.dumps(curator_map, indent=2)

    if args.print_only or not args.output:
        print(output_str)
    else:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_str)
        print(f"✅ Generated: {output_path}")
        print(f"   Version: {curator_map['metadata']['version']}")
        print(f"   Stages: {len(curator_map['stages'])}")


if __name__ == "__main__":
    main()
