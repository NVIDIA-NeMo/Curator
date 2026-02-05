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

"""List available NeMo Curator text classifiers with introspected metadata.

Auto-discovers classifiers from nemo_curator.stages.text.classifiers and extracts:
- Class name and import path
- Parameters with types and defaults
- Output field names
- GPU memory requirements
- HuggingFace token requirements

Examples:
    # List all classifiers
    python list_classifiers.py

    # List with full parameter details
    python list_classifiers.py --verbose

    # Output as JSON (for use by other scripts)
    python list_classifiers.py --json

    # Search for specific classifiers
    python list_classifiers.py --search edu

    # Check if static fallback matches live introspection
    python list_classifiers.py --check-coverage
"""
from __future__ import annotations

import argparse
import inspect
import json
import re
import sys
from dataclasses import dataclass, field
from typing import Any, get_type_hints

# Try to import NeMo Curator
try:
    from nemo_curator.stages.base import CompositeStage
    from nemo_curator.stages.text import classifiers as classifiers_module

    NEMO_CURATOR_AVAILABLE = True
except ImportError:
    NEMO_CURATOR_AVAILABLE = False
    classifiers_module = None  # type: ignore[assignment]
    CompositeStage = None  # type: ignore[misc,assignment]


# GPU memory estimates (GB) - based on model sizes
GPU_MEMORY_ESTIMATES: dict[str, float] = {
    "QualityClassifier": 4.0,
    "DomainClassifier": 4.0,
    "MultilingualDomainClassifier": 4.0,
    "ContentTypeClassifier": 4.0,
    "FineWebEduClassifier": 8.0,
    "FineWebMixtralEduClassifier": 12.0,
    "FineWebNemotronEduClassifier": 12.0,
    "AegisClassifier": 16.0,
    "PromptTaskComplexityClassifier": 8.0,
    "InstructionDataGuardClassifier": 8.0,
}

# Output field mappings - extracted from classifier implementations
OUTPUT_FIELDS: dict[str, dict[str, str]] = {
    "QualityClassifier": {
        "field": "quality_pred",
        "type": "str",
        "values": "High, Medium, Low",
    },
    "DomainClassifier": {
        "field": "domain_pred",
        "type": "str",
        "values": "domain categories",
    },
    "MultilingualDomainClassifier": {
        "field": "domain_pred",
        "type": "str",
        "values": "multilingual domain categories",
    },
    "ContentTypeClassifier": {
        "field": "content_pred",
        "type": "str",
        "values": "content type labels",
    },
    "FineWebEduClassifier": {
        "field": "fineweb-edu-score-int",
        "type": "int",
        "values": "0-5",
    },
    "FineWebMixtralEduClassifier": {
        "field": "fineweb-edu-score-int",
        "type": "int",
        "values": "0-5",
    },
    "FineWebNemotronEduClassifier": {
        "field": "fineweb-edu-score-int",
        "type": "int",
        "values": "0-5",
    },
    "AegisClassifier": {
        "field": "aegis_pred",
        "type": "str",
        "values": "safe, O1-O13 (harm categories)",
    },
    "PromptTaskComplexityClassifier": {
        "field": "prompt_task_complexity_pred",
        "type": "str",
        "values": "complexity levels",
    },
    "InstructionDataGuardClassifier": {
        "field": "instruction_data_guard_pred",
        "type": "str",
        "values": "safety labels",
    },
}

# Classifiers requiring HuggingFace token
REQUIRES_HF_TOKEN: set[str] = {
    "AegisClassifier",
    "InstructionDataGuardClassifier",
}


@dataclass
class ClassifierInfo:
    """Information about a NeMo Curator text classifier."""

    name: str
    module_path: str
    description: str
    gpu_memory_gb: float
    requires_hf_token: bool
    output_field: str
    output_type: str
    output_values: str
    parameters: dict[str, dict[str, Any]] = field(default_factory=dict)
    classifier_class: type | None = None

    @property
    def import_statement(self) -> str:
        """Generate import statement for this classifier."""
        return f"from {self.module_path} import {self.name}"

    @property
    def short_name(self) -> str:
        """Generate a short CLI-friendly name."""
        # QualityClassifier -> quality
        # FineWebEduClassifier -> fineweb-edu
        name = self.name.replace("Classifier", "")
        # Insert hyphens before capitals
        name = re.sub(r"([a-z])([A-Z])", r"\1-\2", name)
        return name.lower()


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
            # Handle non-serializable defaults
            default = param.default
            if callable(default) and not isinstance(default, type):
                param_info["default"] = f"<{type(default).__name__}>"
            else:
                try:
                    json.dumps(default)  # Test if serializable
                    param_info["default"] = default
                except (TypeError, ValueError):
                    param_info["default"] = str(default)

        parameters[param_name] = param_info

    return parameters


def _introspect_classifier(cls: type) -> ClassifierInfo:
    """Introspect a classifier class to extract metadata."""
    name = cls.__name__
    module_path = cls.__module__

    # Get output field info
    output_info = OUTPUT_FIELDS.get(name, {})
    output_field = output_info.get("field", f"{name.lower()}_pred")
    output_type = output_info.get("type", "str")
    output_values = output_info.get("values", "varies")

    return ClassifierInfo(
        name=name,
        module_path=module_path,
        description=_extract_first_sentence(cls.__doc__),
        gpu_memory_gb=GPU_MEMORY_ESTIMATES.get(name, 8.0),
        requires_hf_token=name in REQUIRES_HF_TOKEN,
        output_field=output_field,
        output_type=output_type,
        output_values=output_values,
        parameters=_introspect_parameters(cls),
        classifier_class=cls,
    )


def _is_classifier_class(obj: Any) -> bool:
    """Check if an object is a classifier class."""
    if not isinstance(obj, type):
        return False
    if not NEMO_CURATOR_AVAILABLE or CompositeStage is None:
        return False

    # Must be a subclass of CompositeStage
    if not issubclass(obj, CompositeStage):
        return False

    # Must have "Classifier" in name
    if "Classifier" not in obj.__name__:
        return False

    # Exclude base classes
    if obj.__name__ in ("CompositeStage", "BaseClassifier"):
        return False

    return True


def discover_classifiers() -> list[ClassifierInfo]:
    """Discover all classifiers from NeMo Curator."""
    if not NEMO_CURATOR_AVAILABLE or classifiers_module is None:
        return _get_static_classifiers()

    discovered = []

    # Check __all__ if available
    if hasattr(classifiers_module, "__all__"):
        names = classifiers_module.__all__
    else:
        names = dir(classifiers_module)

    for name in names:
        if name.startswith("_"):
            continue

        obj = getattr(classifiers_module, name, None)
        if obj is None:
            continue

        if _is_classifier_class(obj):
            discovered.append(_introspect_classifier(obj))

    return discovered


def _get_static_classifiers() -> list[ClassifierInfo]:
    """Return static classifier list when NeMo Curator is not installed."""
    static = [
        ("QualityClassifier", "nemo_curator.stages.text.classifiers.quality",
         "General text quality scoring (High/Medium/Low)."),
        ("DomainClassifier", "nemo_curator.stages.text.classifiers.domain",
         "Domain/topic classification for text documents."),
        ("MultilingualDomainClassifier", "nemo_curator.stages.text.classifiers.domain",
         "Multilingual domain classification."),
        ("ContentTypeClassifier", "nemo_curator.stages.text.classifiers.content_type",
         "Content type detection (article, list, forum, etc.)."),
        ("FineWebEduClassifier", "nemo_curator.stages.text.classifiers.fineweb_edu",
         "Educational content scoring on a 0-5 scale."),
        ("FineWebMixtralEduClassifier", "nemo_curator.stages.text.classifiers.fineweb_edu",
         "Mixtral-based educational content scoring."),
        ("FineWebNemotronEduClassifier", "nemo_curator.stages.text.classifiers.fineweb_edu",
         "Nemotron-based educational content scoring."),
        ("AegisClassifier", "nemo_curator.stages.text.classifiers.aegis",
         "Content safety classification using Llama Guard."),
        ("PromptTaskComplexityClassifier", "nemo_curator.stages.text.classifiers.prompt_task_complexity",
         "Prompt and task complexity classification."),
        ("InstructionDataGuardClassifier", "nemo_curator.stages.text.classifiers.aegis",
         "Instruction data safety classification."),
    ]

    classifiers = []
    for name, module_path, description in static:
        output_info = OUTPUT_FIELDS.get(name, {})
        classifiers.append(ClassifierInfo(
            name=name,
            module_path=module_path,
            description=description,
            gpu_memory_gb=GPU_MEMORY_ESTIMATES.get(name, 8.0),
            requires_hf_token=name in REQUIRES_HF_TOKEN,
            output_field=output_info.get("field", f"{name.lower()}_pred"),
            output_type=output_info.get("type", "str"),
            output_values=output_info.get("values", "varies"),
            parameters={
                "text_field": {"type": "str", "default": "text"},
                "model_inference_batch_size": {"type": "int", "default": 256},
            },
        ))

    return classifiers


# Cache discovered classifiers
_CLASSIFIERS_CACHE: list[ClassifierInfo] | None = None


def get_classifiers() -> list[ClassifierInfo]:
    """Get all classifiers, using cache if available."""
    global _CLASSIFIERS_CACHE
    if _CLASSIFIERS_CACHE is None:
        _CLASSIFIERS_CACHE = discover_classifiers()
    return _CLASSIFIERS_CACHE


def search_classifiers(
    search: str | None = None,
    requires_hf_token: bool | None = None,
) -> list[ClassifierInfo]:
    """Search classifiers by various criteria."""
    results = get_classifiers().copy()

    if search:
        pattern = re.compile(search, re.IGNORECASE)
        results = [c for c in results if pattern.search(c.name) or pattern.search(c.description)]

    if requires_hf_token is not None:
        results = [c for c in results if c.requires_hf_token == requires_hf_token]

    return results


def to_dict(clf: ClassifierInfo) -> dict[str, Any]:
    """Convert ClassifierInfo to dictionary for JSON output."""
    return {
        "name": clf.name,
        "short_name": clf.short_name,
        "module_path": clf.module_path,
        "import": clf.import_statement,
        "description": clf.description,
        "gpu_memory_gb": clf.gpu_memory_gb,
        "requires_hf_token": clf.requires_hf_token,
        "output": {
            "field": clf.output_field,
            "type": clf.output_type,
            "values": clf.output_values,
        },
        "parameters": clf.parameters,
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--search",
        help="Search by classifier name or description (regex)",
    )
    parser.add_argument(
        "--hf-token-required",
        action="store_true",
        help="Show only classifiers requiring HuggingFace token",
    )
    parser.add_argument(
        "--no-hf-token",
        action="store_true",
        help="Show only classifiers NOT requiring HuggingFace token",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed information including parameters",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    parser.add_argument(
        "--check-coverage",
        action="store_true",
        help="Check which classifiers are missing metadata mappings",
    )

    args = parser.parse_args()

    # Determine HF token filter
    hf_filter = None
    if args.hf_token_required:
        hf_filter = True
    elif args.no_hf_token:
        hf_filter = False

    classifiers = search_classifiers(
        search=args.search,
        requires_hf_token=hf_filter,
    )

    if args.check_coverage:
        if not NEMO_CURATOR_AVAILABLE:
            print("❌ nemo_curator is not installed. Cannot check coverage.")
            print("   Install with: pip install nemo-curator")
            sys.exit(1)

        # Check for classifiers missing metadata
        missing_gpu = [c for c in classifiers if c.name not in GPU_MEMORY_ESTIMATES]
        missing_output = [c for c in classifiers if c.name not in OUTPUT_FIELDS]

        if missing_gpu:
            print("⚠️  Classifiers missing GPU memory estimates:")
            for c in missing_gpu:
                print(f"  - {c.name}")
            print()

        if missing_output:
            print("⚠️  Classifiers missing output field mappings:")
            for c in missing_output:
                print(f"  - {c.name}")
            print()

        if not missing_gpu and not missing_output:
            print("✅ All classifiers have complete metadata")
        return

    # Show warning if using static fallback
    if not NEMO_CURATOR_AVAILABLE and not args.json:
        print("⚠️  nemo_curator not installed - using static classifier list")
        print("   Install nemo-curator for live introspection with full parameter info\n")

    if args.json:
        output = [to_dict(c) for c in classifiers]
        print(json.dumps(output, indent=2))
    else:
        if not classifiers:
            print("No classifiers found matching criteria.")
            sys.exit(0)

        print("\n📊 NeMo Curator Text Classifiers")
        print("=" * 70)

        for clf in sorted(classifiers, key=lambda x: x.name):
            hf_marker = "🔑" if clf.requires_hf_token else "  "
            print(f"\n{hf_marker} {clf.name}")
            print(f"   {clf.description}")
            print(f"   Output: {clf.output_field} ({clf.output_type}) → {clf.output_values}")
            print(f"   GPU: ~{clf.gpu_memory_gb}GB")
            print(f"   Import: {clf.import_statement}")

            if args.verbose and clf.parameters:
                print("   Parameters:")
                for param_name, param_info in clf.parameters.items():
                    default_str = f" = {param_info['default']}" if "default" in param_info else ""
                    print(f"     - {param_name}: {param_info['type']}{default_str}")

        print(f"\n📊 Total: {len(classifiers)} classifiers")
        print("   🔑 = Requires HuggingFace token")
        if not NEMO_CURATOR_AVAILABLE:
            print("   (static list - install nemo-curator for live introspection)")


if __name__ == "__main__":
    main()
