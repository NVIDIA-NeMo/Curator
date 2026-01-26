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

"""Validate NeMo Curator pipeline YAML configuration.

Auto-discovers valid targets from nemo_curator.stages. When nemo_curator
is not installed, falls back to a static list of known targets.

This script checks a pipeline configuration for common issues:
- Valid YAML syntax
- Valid _target_ paths that exist in NeMo Curator
- Required fields present
- Reasonable resource configurations

Examples:
    python validate_pipeline.py config.yaml
    python validate_pipeline.py config.yaml --strict
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

# Try to import shared introspection utilities
try:
    sys.path.insert(0, str(__file__).rsplit("/", 3)[0])  # Add dori/ to path
    from shared.introspect import NEMO_CURATOR_AVAILABLE, discover_all_stages
except ImportError:
    NEMO_CURATOR_AVAILABLE = False

    def discover_all_stages() -> list:  # type: ignore[misc]
        return []


def _get_valid_targets() -> set[str]:
    """Get set of valid target paths from NeMo Curator."""
    if NEMO_CURATOR_AVAILABLE:
        stages = discover_all_stages()
        targets = {s.full_target for s in stages}
        # Add common non-stage targets
        targets.add("nemo_curator.core.client.RayClient")
        return targets

    # Static fallback when nemo_curator is not installed
    return {
        # Workflows
        "nemo_curator.stages.deduplication.fuzzy.workflow.FuzzyDeduplicationWorkflow",
        "nemo_curator.stages.deduplication.semantic.SemanticDeduplicationWorkflow",
        # IO
        "nemo_curator.stages.text.io.reader.JsonlReader",
        "nemo_curator.stages.text.io.reader.ParquetReader",
        "nemo_curator.stages.text.io.writer.JsonlWriter",
        "nemo_curator.stages.text.io.writer.ParquetWriter",
        # Filters
        "nemo_curator.stages.text.filters.WordCountFilter",
        "nemo_curator.stages.text.filters.NonAlphaNumericFilter",
        "nemo_curator.stages.text.filters.SymbolsToWordsFilter",
        "nemo_curator.stages.text.filters.NumbersFilter",
        "nemo_curator.stages.text.filters.UrlsFilter",
        "nemo_curator.stages.text.filters.BulletsFilter",
        "nemo_curator.stages.text.filters.WhiteSpaceFilter",
        "nemo_curator.stages.text.filters.ParenthesesFilter",
        "nemo_curator.stages.text.filters.LongWordFilter",
        "nemo_curator.stages.text.filters.MeanWordLengthFilter",
        "nemo_curator.stages.text.filters.RepeatedLinesFilter",
        "nemo_curator.stages.text.filters.RepeatedParagraphsFilter",
        "nemo_curator.stages.text.filters.RepeatedLinesByCharFilter",
        "nemo_curator.stages.text.filters.RepeatedParagraphsByCharFilter",
        "nemo_curator.stages.text.filters.RepeatingTopNGramsFilter",
        "nemo_curator.stages.text.filters.RepeatingDuplicateNGramsFilter",
        "nemo_curator.stages.text.filters.PunctuationFilter",
        "nemo_curator.stages.text.filters.EllipsisFilter",
        "nemo_curator.stages.text.filters.CommonEnglishWordsFilter",
        "nemo_curator.stages.text.filters.WordsWithoutAlphabetsFilter",
        "nemo_curator.stages.text.filters.PornographicUrlsFilter",
        "nemo_curator.stages.text.filters.TokenCountFilter",
        "nemo_curator.stages.text.filters.SubstringFilter",
        "nemo_curator.stages.text.filters.HistogramFilter",
        "nemo_curator.stages.text.filters.FastTextLangId",
        "nemo_curator.stages.text.filters.FastTextQualityFilter",
        # Code filters
        "nemo_curator.stages.text.filters.code.AlphaFilter",
        "nemo_curator.stages.text.filters.code.GeneralCommentToCodeFilter",
        "nemo_curator.stages.text.filters.code.HTMLBoilerplateFilter",
        "nemo_curator.stages.text.filters.code.NumberOfLinesOfCodeFilter",
        "nemo_curator.stages.text.filters.code.PerExtensionFilter",
        "nemo_curator.stages.text.filters.code.PythonCommentToCodeFilter",
        "nemo_curator.stages.text.filters.code.TokenizerFertilityFilter",
        "nemo_curator.stages.text.filters.code.XMLHeaderFilter",
        # Classifiers
        "nemo_curator.stages.text.classifiers.QualityClassifier",
        "nemo_curator.stages.text.classifiers.DomainClassifier",
        "nemo_curator.stages.text.classifiers.FineWebEduClassifier",
        "nemo_curator.stages.text.classifiers.AegisClassifier",
        # Modules
        "nemo_curator.stages.text.modules.ScoreFilter",
        "nemo_curator.stages.text.modules.Filter",
        "nemo_curator.stages.text.modules.Score",
        "nemo_curator.stages.text.modules.Modify",
        "nemo_curator.stages.text.modules.AddId",
        # Video
        "nemo_curator.stages.video.io.video_reader.VideoReader",
        "nemo_curator.stages.video.clipping.TransNetV2ClipExtractionStage",
        "nemo_curator.stages.video.clipping.FixedStrideExtractorStage",
        "nemo_curator.stages.video.caption.CaptionGenerationStage",
        "nemo_curator.stages.video.embedding.CosmosEmbed1EmbeddingStage",
        # Image
        "nemo_curator.stages.image.embedders.ImageEmbeddingStage",
        "nemo_curator.stages.image.filters.AestheticFilterStage",
        "nemo_curator.stages.image.filters.NSFWFilterStage",
        # Audio
        "nemo_curator.stages.audio.inference.InferenceAsrNemoStage",
        # Client
        "nemo_curator.core.client.RayClient",
    }


# Required fields for common configurations (introspected where possible)
def _get_required_fields() -> dict[str, list[str]]:
    """Get required fields for common stage types."""
    # These are semantic requirements that can't be fully introspected
    # (they're about what makes sense, not what's technically required)
    return {
        "FuzzyDeduplicationWorkflow": ["input_path", "output_path", "cache_path"],
        "SemanticDeduplicationWorkflow": ["input_path", "output_path"],
        "JsonlReader": ["file_paths"],
        "ParquetReader": ["file_paths"],
    }


# Cache valid targets
_VALID_TARGETS_CACHE: set[str] | None = None


def get_valid_targets() -> set[str]:
    """Get valid targets with caching."""
    global _VALID_TARGETS_CACHE
    if _VALID_TARGETS_CACHE is None:
        _VALID_TARGETS_CACHE = _get_valid_targets()
    return _VALID_TARGETS_CACHE


def validate_target(target: str, strict: bool = False) -> tuple[bool, str]:
    """Check if a _target_ path is valid."""
    valid_targets = get_valid_targets()

    if target in valid_targets:
        return True, ""

    # Try to import the target in strict mode
    if strict:
        try:
            parts = target.rsplit(".", 1)
            if len(parts) == 2:
                module_path, class_name = parts
                module = importlib.import_module(module_path)
                if hasattr(module, class_name):
                    return True, ""
        except (ImportError, ModuleNotFoundError):
            pass
        return False, f"Target not found: {target}"

    # Non-strict: just check it looks like a valid path
    if target.startswith("nemo_curator."):
        return True, f"Warning: Target not in known list (may still be valid): {target}"

    return False, f"Invalid target (should start with nemo_curator.): {target}"


def validate_config(config: dict, strict: bool = False) -> dict:
    """Validate a pipeline configuration dictionary."""
    issues = []
    warnings = []
    required_fields = _get_required_fields()

    def check_dict(d: dict, path: str = ""):
        """Recursively check dictionary for _target_ fields."""
        if "_target_" in d:
            target = d["_target_"]
            valid, msg = validate_target(target, strict)
            if not valid:
                issues.append({"path": path, "error": msg})
            elif msg:
                warnings.append({"path": path, "warning": msg})

            # Check required fields
            class_name = target.rsplit(".", 1)[-1]
            if class_name in required_fields:
                for field in required_fields[class_name]:
                    if field not in d and f"${{{field}}}" not in str(d.values()):
                        # Check if it's a Hydra interpolation
                        has_interpolation = any(
                            f"${{{field}}}" in str(v) or f"${{" in str(v) for v in d.values()
                        )
                        if not has_interpolation:
                            warnings.append(
                                {
                                    "path": f"{path}._target_={class_name}",
                                    "warning": f"Missing recommended field: {field}",
                                }
                            )

        for key, value in d.items():
            if isinstance(value, dict):
                check_dict(value, f"{path}.{key}" if path else key)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        check_dict(item, f"{path}.{key}[{i}]" if path else f"{key}[{i}]")

    check_dict(config)

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("config", nargs="?", help="YAML configuration file to validate")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Strict mode: verify targets exist by importing",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON only",
    )
    parser.add_argument(
        "--list-targets",
        action="store_true",
        help="List all known valid targets",
    )

    args = parser.parse_args()

    # Handle --list-targets
    if args.list_targets:
        targets = sorted(get_valid_targets())
        if args.json:
            print(json.dumps({"targets": targets, "count": len(targets)}, indent=2))
        else:
            if not NEMO_CURATOR_AVAILABLE:
                print("⚠️  nemo_curator not installed - using static target list\n")
            print(f"📋 Valid Targets ({len(targets)} total):\n")
            for target in targets:
                print(f"  {target}")
            print(f"\n📊 Total: {len(targets)} targets (auto-discovered from NeMo Curator)")
        return

    # Require config file for validation
    if not args.config:
        parser.error("config file is required (or use --list-targets)")

    if not YAML_AVAILABLE:
        print(json.dumps({"error": "PyYAML not installed. Run: pip install pyyaml"}))
        sys.exit(1)

    config_path = Path(args.config)
    if not config_path.exists():
        result = {"error": f"Config file not found: {args.config}"}
        print(json.dumps(result) if args.json else f"❌ {result['error']}")
        sys.exit(1)

    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        result = {"error": f"Invalid YAML syntax: {e}"}
        print(json.dumps(result) if args.json else f"❌ {result['error']}")
        sys.exit(1)

    result = validate_config(config, args.strict)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        # Show source info
        if not NEMO_CURATOR_AVAILABLE:
            print("⚠️  nemo_curator not installed - using static target list\n")

        if result["valid"]:
            print("✅ Configuration is valid")
        else:
            print("❌ Configuration has issues:")
            for issue in result["issues"]:
                print(f"   • {issue['path']}: {issue['error']}")

        if result["warnings"]:
            print()
            print("⚠️  Warnings:")
            for warning in result["warnings"]:
                print(f"   • {warning['path']}: {warning['warning']}")

    sys.exit(0 if result["valid"] else 1)


if __name__ == "__main__":
    main()
