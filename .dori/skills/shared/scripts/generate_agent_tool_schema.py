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

"""Generate Agent Tool Schema from a Python library.

This tool introspects a Python package and generates a machine-readable schema
describing its capabilities for AI agents. The schema follows the Agent Tool Schema
specification, enabling agents to:

- Discover available operations
- Understand parameter types and defaults
- Know resource requirements (GPU, credentials)
- Validate pipeline composition
- Generate correct code

Supports two introspection modes:
1. **Source mode** (--source): Parses Python files with AST - no installation needed
2. **Runtime mode** (default): Imports modules - requires package installed

Usage:
    # Generate from source (no installation needed!)
    python generate_agent_tool_schema.py --source ../../../nemo_curator --output schema.yaml

    # Generate with runtime introspection (requires package installed)
    python generate_agent_tool_schema.py --package nemo_curator --output schema.yaml

    # Use a config file for custom introspection rules
    python generate_agent_tool_schema.py --source ./src --config my_config.yaml --output schema.yaml
"""
from __future__ import annotations

import argparse
import ast
import importlib
import inspect
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, get_type_hints

# Schema version
SCHEMA_VERSION = "1.0.0"


@dataclass
class IntrospectionConfig:
    """Configuration for library introspection."""

    # Package to introspect
    package_name: str

    # Human-readable tool name
    tool_name: str = ""

    # Tool description
    description: str = ""

    # Installation command
    install_command: str = ""

    # Base classes that indicate an "operation"
    operation_base_classes: list[str] = field(default_factory=list)

    # Module paths to scan (for runtime mode)
    module_paths: list[str] = field(default_factory=list)

    # Source directories to scan (for source mode)
    source_paths: list[str] = field(default_factory=list)

    # Patterns to identify operation classes (regex)
    operation_patterns: list[str] = field(default_factory=lambda: [
        r".*Stage$",
        r".*Filter$",
        r".*Classifier$",
        r".*Modifier$",
        r".*Reader$",
        r".*Writer$",
        r".*Workflow$",
    ])

    # Patterns to exclude
    exclude_patterns: list[str] = field(default_factory=lambda: [
        r"^_",
        r"^Base",
        r"^Abstract",
    ])

    # Type mappings (class name -> schema type name)
    type_mappings: dict[str, str] = field(default_factory=dict)

    # GPU memory estimates by operation name
    gpu_memory_estimates: dict[str, float] = field(default_factory=dict)

    # Operations requiring specific credentials
    credential_requirements: dict[str, list[str]] = field(default_factory=dict)

    # Category inference rules (regex pattern -> category)
    category_rules: dict[str, list[str]] = field(default_factory=dict)

    # Hints to add for specific operations
    operation_hints: dict[str, dict[str, Any]] = field(default_factory=dict)


# Default configuration for NeMo Curator
NEMO_CURATOR_CONFIG = IntrospectionConfig(
    package_name="nemo_curator",
    tool_name="NeMo Curator",
    description="Scalable data curation library for training large language models",
    install_command="pip install nemo-curator",
    operation_base_classes=[
        "ProcessingStage",
        "CompositeStage",
    ],
    source_paths=[
        "stages/text/io/reader",
        "stages/text/io/writer",
        "stages/text/filters",
        "stages/text/classifiers",
        "stages/text/modifiers",
        "stages/text/modules",
        "stages/text/embedders",
        "stages/text/download",
        "stages/video/io",
        "stages/video/clipping",
        "stages/video/caption",
        "stages/video/embedding",
        "stages/video/filtering",
        "stages/video/preview",
        "stages/image/io",
        "stages/image/embedders",
        "stages/image/filters",
        "stages/image/deduplication",
        "stages/audio/inference",
        "stages/audio/metrics",
        "stages/audio/io",
        "stages/deduplication/exact",
        "stages/deduplication/fuzzy",
        "stages/deduplication/semantic",
    ],
    module_paths=[
        "nemo_curator.stages.text.io.reader",
        "nemo_curator.stages.text.io.writer",
        "nemo_curator.stages.text.filters",
        "nemo_curator.stages.text.classifiers",
        "nemo_curator.stages.text.modifiers",
        "nemo_curator.stages.text.modules",
        "nemo_curator.stages.text.embedders",
        "nemo_curator.stages.video.io",
        "nemo_curator.stages.video.clipping",
        "nemo_curator.stages.video.caption",
        "nemo_curator.stages.video.embedding",
        "nemo_curator.stages.video.filtering",
        "nemo_curator.stages.image.io",
        "nemo_curator.stages.image.embedders",
        "nemo_curator.stages.image.filters",
        "nemo_curator.stages.image.deduplication",
        "nemo_curator.stages.audio.inference",
        "nemo_curator.stages.audio.metrics",
        "nemo_curator.stages.deduplication.exact",
        "nemo_curator.stages.deduplication.fuzzy",
        "nemo_curator.stages.deduplication.semantic",
    ],
    type_mappings={
        "DocumentBatch": "DocumentBatch",
        "VideoTask": "VideoTask",
        "ImageBatch": "ImageBatch",
        "AudioBatch": "AudioBatch",
        "FileGroupTask": "FileGroupTask",
        "Task": "Task",
    },
    gpu_memory_estimates={
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
        "MultilingualDomainClassifier": 4.0,
        "FineWebEduClassifier": 8.0,
        "FineWebMixtralEduClassifier": 12.0,
        "FineWebNemotronEduClassifier": 12.0,
        "AegisClassifier": 16.0,
        "ContentTypeClassifier": 4.0,
        "PromptTaskComplexityClassifier": 8.0,
        "InstructionDataGuardClassifier": 8.0,
        "EmbeddingCreatorStage": 8.0,
    },
    credential_requirements={
        "AegisClassifier": ["HF_TOKEN"],
        "InstructionDataGuardClassifier": ["HF_TOKEN"],
    },
    category_rules={
        r"text[/\\]io[/\\]reader": ["text", "io", "reader"],
        r"text[/\\]io[/\\]writer": ["text", "io", "writer"],
        r"text[/\\]filters": ["text", "filter"],
        r"text[/\\]classifiers": ["text", "classifier", "ml"],
        r"text[/\\]modifiers": ["text", "modifier"],
        r"text[/\\]modules": ["text", "module"],
        r"text[/\\]embedders": ["text", "embedder", "ml"],
        r"text[/\\]download": ["text", "download"],
        r"video[/\\]io": ["video", "io"],
        r"video[/\\]clipping": ["video", "clipping"],
        r"video[/\\]caption": ["video", "captioning", "ml"],
        r"video[/\\]embedding": ["video", "embedding", "ml"],
        r"video[/\\]filtering": ["video", "filter"],
        r"video[/\\]preview": ["video", "preview"],
        r"image[/\\]io": ["image", "io"],
        r"image[/\\]embedders": ["image", "embedding", "ml"],
        r"image[/\\]filters": ["image", "filter", "ml"],
        r"image[/\\]deduplication": ["image", "deduplication"],
        r"audio[/\\]inference": ["audio", "inference", "ml"],
        r"audio[/\\]metrics": ["audio", "metrics"],
        r"audio[/\\]io": ["audio", "io"],
        r"deduplication[/\\]exact": ["deduplication", "exact"],
        r"deduplication[/\\]fuzzy": ["deduplication", "fuzzy"],
        r"deduplication[/\\]semantic": ["deduplication", "semantic", "ml"],
    },
    operation_hints={
        "WordCountFilter": {
            "purpose": "Remove very short or very long documents",
            "when_to_use": "Early in pipeline to reduce data volume",
            "typical_retention": "80-95%",
        },
        "QualityClassifier": {
            "purpose": "Score document quality as High/Medium/Low",
            "output_values": ["High", "Medium", "Low"],
            "typical_retention": "30-50% if keeping High only",
            "order_preference": "After heuristic filters, before deduplication",
        },
        "FineWebEduClassifier": {
            "purpose": "Score educational content quality 0-5",
            "output_values": ["0", "1", "2", "3", "4", "5"],
            "typical_retention": "20-40% if keeping score >= 3",
        },
        "AegisClassifier": {
            "purpose": "Detect harmful content",
            "output_values": ["safe", "O1", "O2", "O3", "O4", "O5", "O6", "O7", "O8", "O9", "O10", "O11", "O12", "O13"],
            "when_to_use": "Safety filtering before release",
        },
    },
)


@dataclass
class ParameterInfo:
    """Information about an operation parameter."""

    name: str
    type: str
    description: str = ""
    default: Any = None
    required: bool = True


@dataclass
class ResourceInfo:
    """Resource requirements for an operation."""

    requires_gpu: bool = False
    gpu_memory_gb: float = 0.0
    requires_credentials: list[str] = field(default_factory=list)


@dataclass
class OperationInfo:
    """Information about an operation (stage/filter/etc.)."""

    name: str
    import_path: str
    module_path: str
    description: str
    categories: list[str]
    input_type: str
    output_type: str
    parameters: list[ParameterInfo]
    resources: ResourceInfo
    hints: dict[str, Any] = field(default_factory=dict)
    base_classes: list[str] = field(default_factory=list)


# =============================================================================
# AST-based Source Introspection
# =============================================================================


class ASTClassVisitor(ast.NodeVisitor):
    """AST visitor that extracts class information."""

    def __init__(self, config: IntrospectionConfig, file_path: str, module_path: str):
        self.config = config
        self.file_path = file_path
        self.module_path = module_path
        self.classes: list[OperationInfo] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit a class definition."""
        name = node.name

        # Check exclude patterns
        for pattern in self.config.exclude_patterns:
            if re.match(pattern, name):
                return

        # Check if matches operation patterns
        matches_pattern = False
        for pattern in self.config.operation_patterns:
            if re.match(pattern, name):
                matches_pattern = True
                break

        # Check base classes
        base_names = []
        inherits_operation = False
        for base in node.bases:
            base_name = self._get_name(base)
            if base_name:
                base_names.append(base_name)
                if base_name in self.config.operation_base_classes:
                    inherits_operation = True

        # Must match pattern OR inherit from operation base
        if not matches_pattern and not inherits_operation:
            return

        # Extract class info
        op = self._extract_class_info(node, base_names)
        if op:
            self.classes.append(op)

    def _get_name(self, node: ast.expr) -> str | None:
        """Get name from AST node."""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        if isinstance(node, ast.Subscript):
            return self._get_name(node.value)
        return None

    def _extract_docstring(self, node: ast.ClassDef) -> str:
        """Extract docstring from class."""
        if node.body and isinstance(node.body[0], ast.Expr):
            if isinstance(node.body[0].value, ast.Constant):
                docstring = node.body[0].value.value
                if isinstance(docstring, str):
                    # Get first sentence
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
        return ""

    def _extract_init_params(self, node: ast.ClassDef) -> list[ParameterInfo]:
        """Extract __init__ parameters."""
        params = []

        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                args = item.args

                # Get defaults (aligned from the end)
                num_defaults = len(args.defaults)
                num_args = len(args.args)

                for i, arg in enumerate(args.args):
                    if arg.arg in ("self",):
                        continue

                    # Get type annotation
                    type_str = "any"
                    if arg.annotation:
                        type_str = self._annotation_to_str(arg.annotation)

                    # Check if has default
                    default_index = i - (num_args - num_defaults)
                    has_default = default_index >= 0
                    default_value = None

                    if has_default:
                        default_node = args.defaults[default_index]
                        default_value = self._extract_default(default_node)

                    params.append(ParameterInfo(
                        name=arg.arg,
                        type=type_str,
                        required=not has_default,
                        default=default_value,
                    ))

                break

        return params

    def _annotation_to_str(self, node: ast.expr) -> str:
        """Convert annotation AST to string."""
        if isinstance(node, ast.Name):
            name = node.id
            # Map Python types to JSON Schema types
            type_map = {
                "str": "string",
                "int": "integer",
                "float": "number",
                "bool": "boolean",
                "list": "array",
                "dict": "object",
                "None": "null",
            }
            return type_map.get(name, name)
        if isinstance(node, ast.Constant):
            if node.value is None:
                return "null"
            return str(node.value)
        if isinstance(node, ast.Subscript):
            base = self._annotation_to_str(node.value)
            return base
        if isinstance(node, ast.Attribute):
            return node.attr
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            # Union type (X | Y)
            left = self._annotation_to_str(node.left)
            right = self._annotation_to_str(node.right)
            if right == "null":
                return left  # X | None -> X (optional)
            return f"{left} | {right}"
        return "any"

    def _extract_default(self, node: ast.expr) -> Any:
        """Extract default value from AST."""
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id == "None":
                return None
            if node.id == "True":
                return True
            if node.id == "False":
                return False
            return f"<{node.id}>"
        if isinstance(node, ast.List):
            return []
        if isinstance(node, ast.Dict):
            return {}
        if isinstance(node, ast.Call):
            func_name = self._get_name(node.func)
            return f"<{func_name}()>"
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            if isinstance(node.operand, ast.Constant):
                return -node.operand.value
        return None

    def _infer_io_types(self, base_names: list[str]) -> tuple[str, str]:
        """Infer input/output types from file path."""
        path = self.file_path.lower()

        if "text" in path or "filter" in path or "classifier" in path or "modifier" in path:
            return "DocumentBatch", "DocumentBatch"
        if "video" in path:
            return "VideoTask", "VideoTask"
        if "image" in path:
            return "ImageBatch", "ImageBatch"
        if "audio" in path:
            return "AudioBatch", "AudioBatch"

        return "Task", "Task"

    def _infer_categories(self) -> list[str]:
        """Infer categories from file path."""
        for pattern, cats in self.config.category_rules.items():
            if re.search(pattern, self.file_path):
                return cats
        return ["other"]

    def _extract_class_info(self, node: ast.ClassDef, base_names: list[str]) -> OperationInfo | None:
        """Extract full class information."""
        name = node.name
        input_type, output_type = self._infer_io_types(base_names)
        categories = self._infer_categories()

        # Check GPU requirement
        requires_gpu = name in self.config.gpu_memory_estimates
        if not requires_gpu:
            # Infer from categories
            if "ml" in categories or "classifier" in categories or "embedding" in categories:
                requires_gpu = True

        gpu_memory = self.config.gpu_memory_estimates.get(name, 8.0 if requires_gpu else 0.0)
        credentials = self.config.credential_requirements.get(name, [])

        return OperationInfo(
            name=name,
            import_path=f"from {self.module_path} import {name}",
            module_path=self.module_path,
            description=self._extract_docstring(node),
            categories=categories,
            input_type=input_type,
            output_type=output_type,
            parameters=self._extract_init_params(node),
            resources=ResourceInfo(
                requires_gpu=requires_gpu,
                gpu_memory_gb=gpu_memory,
                requires_credentials=credentials,
            ),
            hints=self.config.operation_hints.get(name, {}),
            base_classes=base_names,
        )


def discover_from_source(source_root: Path, config: IntrospectionConfig) -> dict[str, OperationInfo]:
    """Discover operations by parsing source files."""
    operations: dict[str, OperationInfo] = {}

    # Find all Python files in source paths
    for source_path in config.source_paths:
        path = source_root / source_path

        if path.is_file() and path.suffix == ".py":
            files = [path]
        elif path.is_dir():
            files = list(path.rglob("*.py"))
        else:
            # Try as glob pattern
            files = list(source_root.glob(source_path + "/**/*.py"))
            if not files:
                files = list(source_root.glob(source_path + ".py"))

        for file_path in files:
            if file_path.name.startswith("_") and file_path.name != "__init__.py":
                continue

            try:
                source = file_path.read_text()
                tree = ast.parse(source)

                # Compute module path
                rel_path = file_path.relative_to(source_root)
                module_parts = list(rel_path.with_suffix("").parts)
                if module_parts[-1] == "__init__":
                    module_parts = module_parts[:-1]
                module_path = f"{config.package_name}.{'.'.join(module_parts)}"

                visitor = ASTClassVisitor(config, str(file_path), module_path)
                visitor.visit(tree)

                for op in visitor.classes:
                    if op.name not in operations:
                        operations[op.name] = op

            except SyntaxError as e:
                print(f"Warning: Syntax error in {file_path}: {e}", file=sys.stderr)
            except Exception as e:
                print(f"Warning: Error parsing {file_path}: {e}", file=sys.stderr)

    return operations


# =============================================================================
# Runtime Introspection (existing code, simplified)
# =============================================================================


def discover_from_runtime(config: IntrospectionConfig) -> dict[str, OperationInfo]:
    """Discover operations by importing modules at runtime."""
    operations: dict[str, OperationInfo] = {}

    # Try to import package
    try:
        pkg = importlib.import_module(config.package_name)
    except ImportError as e:
        print(f"Error: Could not import {config.package_name}: {e}", file=sys.stderr)
        return operations

    for module_path in config.module_paths:
        try:
            module = importlib.import_module(module_path)
        except ImportError:
            continue

        names = getattr(module, "__all__", dir(module))

        for name in names:
            if name.startswith("_"):
                continue

            # Check patterns
            matches = False
            for pattern in config.operation_patterns:
                if re.match(pattern, name):
                    matches = True
                    break

            if not matches:
                continue

            # Check exclude
            excluded = False
            for pattern in config.exclude_patterns:
                if re.match(pattern, name):
                    excluded = True
                    break

            if excluded:
                continue

            try:
                obj = getattr(module, name)
                if not isinstance(obj, type):
                    continue

                # Extract info
                op = _introspect_class_runtime(obj, config)
                if op and op.name not in operations:
                    operations[op.name] = op
            except Exception:
                continue

    return operations


def _introspect_class_runtime(cls: type, config: IntrospectionConfig) -> OperationInfo | None:
    """Introspect a class at runtime."""
    name = cls.__name__
    module_path = cls.__module__

    # Get docstring
    docstring = cls.__doc__ or ""
    if docstring:
        lines = docstring.strip().split("\n")
        first_para = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                break
            first_para.append(stripped)
        text = " ".join(first_para)
        match = re.match(r"^(.+?[.!?])\s", text + " ")
        description = match.group(1) if match else text[:200]
    else:
        description = ""

    # Infer types
    input_type, output_type = "Task", "Task"
    if ".text." in module_path:
        input_type = output_type = "DocumentBatch"
    elif ".video." in module_path:
        input_type = output_type = "VideoTask"
    elif ".image." in module_path:
        input_type = output_type = "ImageBatch"
    elif ".audio." in module_path:
        input_type = output_type = "AudioBatch"

    # Categories
    categories = ["other"]
    for pattern, cats in config.category_rules.items():
        if re.search(pattern.replace("[/\\\\]", "."), module_path):
            categories = cats
            break

    # Resources
    requires_gpu = name in config.gpu_memory_estimates
    if not requires_gpu and ("classifier" in categories or "ml" in categories):
        requires_gpu = True

    # Parameters
    params = []
    try:
        sig = inspect.signature(cls.__init__)
        try:
            hints = get_type_hints(cls.__init__)
        except Exception:
            hints = {}

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "args", "kwargs"):
                continue

            param_type = hints.get(param_name, param.annotation)
            type_str = "any"
            if param_type is not inspect.Parameter.empty:
                if hasattr(param_type, "__name__"):
                    type_str = param_type.__name__
                else:
                    type_str = str(param_type).replace("typing.", "")

            default = None
            required = param.default is inspect.Parameter.empty
            if not required:
                try:
                    json.dumps(param.default)
                    default = param.default
                except (TypeError, ValueError):
                    default = str(param.default)

            params.append(ParameterInfo(
                name=param_name,
                type=type_str,
                required=required,
                default=default,
            ))
    except (ValueError, TypeError):
        pass

    return OperationInfo(
        name=name,
        import_path=f"from {module_path} import {name}",
        module_path=module_path,
        description=description,
        categories=categories,
        input_type=input_type,
        output_type=output_type,
        parameters=params,
        resources=ResourceInfo(
            requires_gpu=requires_gpu,
            gpu_memory_gb=config.gpu_memory_estimates.get(name, 8.0 if requires_gpu else 0.0),
            requires_credentials=config.credential_requirements.get(name, []),
        ),
        hints=config.operation_hints.get(name, {}),
    )


# =============================================================================
# Schema Generation
# =============================================================================


def generate_schema(operations: dict[str, OperationInfo], config: IntrospectionConfig, version: str) -> dict[str, Any]:
    """Generate the complete Agent Tool Schema."""

    # Build type flow
    type_producers: dict[str, list[str]] = {}
    type_consumers: dict[str, list[str]] = {}

    for op in operations.values():
        if op.output_type not in type_producers:
            type_producers[op.output_type] = []
        type_producers[op.output_type].append(op.name)

        if op.input_type not in type_consumers:
            type_consumers[op.input_type] = []
        type_consumers[op.input_type].append(op.name)

    # Build operations dict
    ops_dict: dict[str, dict[str, Any]] = {}
    for name, op in sorted(operations.items()):
        op_dict: dict[str, Any] = {
            "import": op.import_path,
            "description": op.description,
            "category": op.categories,
            "input_type": op.input_type,
            "output_type": op.output_type,
        }

        if op.parameters:
            op_dict["parameters"] = {}
            for param in op.parameters:
                p: dict[str, Any] = {"type": param.type}
                if param.default is not None:
                    p["default"] = param.default
                if not param.required:
                    p["required"] = False
                op_dict["parameters"][param.name] = p

        op_dict["resources"] = {
            "requires_gpu": op.resources.requires_gpu,
        }
        if op.resources.gpu_memory_gb > 0:
            op_dict["resources"]["gpu_memory_gb"] = op.resources.gpu_memory_gb
        if op.resources.requires_credentials:
            op_dict["resources"]["requires_credentials"] = op.resources.requires_credentials

        if op.hints:
            op_dict["hints"] = op.hints

        ops_dict[name] = op_dict

    return {
        "$schema": "https://nvidia.github.io/agent-tool-schema/v1",
        "schema_version": SCHEMA_VERSION,
        "tool": {
            "name": config.tool_name or config.package_name,
            "package": config.package_name,
            "version": version,
            "description": config.description,
            "install": config.install_command or f"pip install {config.package_name}",
        },
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "generator": "generate_agent_tool_schema.py",
            "generator_version": SCHEMA_VERSION,
        },
        "types": {
            "DocumentBatch": {
                "description": "Batch of text documents",
                "runtime_type": "nemo_curator.tasks.DocumentBatch",
                "data_format": "pd.DataFrame | pa.Table",
                "common_columns": ["text", "id", "url"],
            },
            "VideoTask": {
                "description": "Video processing task",
                "runtime_type": "nemo_curator.tasks.video.VideoTask",
                "data_format": "Video object",
            },
            "ImageBatch": {
                "description": "Batch of images",
                "runtime_type": "nemo_curator.tasks.image.ImageBatch",
                "data_format": "list[ImageObject]",
            },
            "AudioBatch": {
                "description": "Batch of audio files",
                "runtime_type": "nemo_curator.tasks.audio.AudioBatch",
                "data_format": "dict | list[dict]",
            },
            "FileGroupTask": {
                "description": "Group of file paths",
                "runtime_type": "nemo_curator.tasks.FileGroupTask",
                "data_format": "list[str]",
            },
            "Task": {
                "description": "Generic task",
                "runtime_type": "nemo_curator.tasks.Task",
                "data_format": "varies",
            },
        },
        "operations": ops_dict,
        "composition": {
            "type_flow": {
                type_name: {
                    "producers": sorted(set(type_producers.get(type_name, []))),
                    "consumers": sorted(set(type_consumers.get(type_name, []))),
                }
                for type_name in sorted(set(type_producers.keys()) | set(type_consumers.keys()))
            },
        },
        "execution": {
            "environments": {
                "docker": {
                    "image": "nvcr.io/nvidia/nemo-curator:latest",
                    "gpu_flag": "--gpus all",
                },
                "native": {
                    "python_version": ">=3.10",
                    "cuda_version": ">=12.0",
                },
            },
        },
    }


def get_version_from_source(source_root: Path) -> str:
    """Try to extract version from source."""
    # Try pyproject.toml
    pyproject = source_root.parent / "pyproject.toml"
    if pyproject.exists():
        content = pyproject.read_text()
        match = re.search(r'version\s*=\s*["\']([^"\']+)["\']', content)
        if match:
            return match.group(1)

    # Try __init__.py
    init_file = source_root / "__init__.py"
    if init_file.exists():
        content = init_file.read_text()
        match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
        if match:
            return match.group(1)

    return "unknown"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source",
        help="Path to source directory (uses AST parsing, no installation needed)",
    )
    parser.add_argument(
        "--package",
        default="nemo_curator",
        help="Python package to introspect via runtime import (default: nemo_curator)",
    )
    parser.add_argument(
        "--config",
        help="Path to YAML/JSON config file with introspection rules",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file path (JSON or YAML based on extension)",
    )
    parser.add_argument(
        "--format",
        choices=["json", "yaml"],
        default="yaml",
        help="Output format (default: yaml)",
    )
    parser.add_argument(
        "--print",
        action="store_true",
        dest="print_only",
        help="Print to stdout instead of writing to file",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show statistics about discovered operations",
    )

    args = parser.parse_args()

    # Load config
    if args.config:
        config_path = Path(args.config)
        if config_path.suffix in (".yaml", ".yml"):
            try:
                import yaml
                config_data = yaml.safe_load(config_path.read_text())
            except ImportError:
                print("Error: PyYAML required for YAML config")
                sys.exit(1)
        else:
            config_data = json.loads(config_path.read_text())
        config = IntrospectionConfig(**config_data)
    elif args.package == "nemo_curator":
        config = NEMO_CURATOR_CONFIG
    else:
        config = IntrospectionConfig(package_name=args.package)

    # Discover operations
    if args.source:
        source_root = Path(args.source).resolve()
        print(f"📂 Scanning source: {source_root}", file=sys.stderr)
        operations = discover_from_source(source_root, config)
        version = get_version_from_source(source_root)
    else:
        print(f"📦 Importing package: {config.package_name}", file=sys.stderr)
        operations = discover_from_runtime(config)
        try:
            pkg = importlib.import_module(config.package_name)
            version = getattr(pkg, "__version__", "unknown")
        except ImportError:
            version = "unknown"

    if args.stats:
        print(f"\n📊 Agent Tool Schema Generator")
        print("=" * 50)
        print(f"Package: {config.package_name}")
        print(f"Version: {version}")
        print(f"Operations discovered: {len(operations)}")
        print()

        categories: dict[str, int] = {}
        gpu_count = 0
        for op in operations.values():
            for cat in op.categories:
                categories[cat] = categories.get(cat, 0) + 1
            if op.resources.requires_gpu:
                gpu_count += 1

        print("By category:")
        for cat, count in sorted(categories.items()):
            print(f"  {cat}: {count}")
        print()
        print(f"GPU operations: {gpu_count}")
        return

    # Generate schema
    schema = generate_schema(operations, config, version)

    # Output
    output_format = args.format
    if args.output:
        if args.output.endswith((".yaml", ".yml")):
            output_format = "yaml"
        elif args.output.endswith(".json"):
            output_format = "json"

    if output_format == "yaml":
        try:
            import yaml
            output_str = yaml.dump(schema, default_flow_style=False, sort_keys=False, allow_unicode=True)
        except ImportError:
            print("Warning: PyYAML not installed, using JSON", file=sys.stderr)
            output_str = json.dumps(schema, indent=2)
    else:
        output_str = json.dumps(schema, indent=2)

    if args.print_only or not args.output:
        print(output_str)
    else:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_str)
        print(f"✅ Generated: {output_path}", file=sys.stderr)
        print(f"   Tool: {schema['tool']['name']}", file=sys.stderr)
        print(f"   Version: {schema['tool']['version']}", file=sys.stderr)
        print(f"   Operations: {len(schema['operations'])}", file=sys.stderr)


if __name__ == "__main__":
    main()
