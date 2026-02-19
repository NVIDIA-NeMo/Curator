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

"""Validate Agent Tool Schema files against the specification.

This script validates generated Agent Tool Schema files to ensure they:
- Conform to the JSON Schema specification
- Have valid type references
- Have consistent composition rules
- Include required fields

Usage:
    # Validate a schema file
    python validate_agent_tool_schema.py schema.yaml

    # Validate with verbose output
    python validate_agent_tool_schema.py schema.yaml --verbose

    # Validate multiple files
    python validate_agent_tool_schema.py schema1.yaml schema2.json

    # Check composition consistency
    python validate_agent_tool_schema.py schema.yaml --check-composition

    # Output as JSON (for CI integration)
    python validate_agent_tool_schema.py schema.yaml --json
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Try to import jsonschema for validation
try:
    import jsonschema
    from jsonschema import Draft202012Validator, ValidationError

    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False
    Draft202012Validator = None  # type: ignore[misc,assignment]
    ValidationError = Exception  # type: ignore[misc,assignment]


@dataclass
class ValidationIssue:
    """A single validation issue."""

    level: str  # "error", "warning", "info"
    path: str  # JSON path to the issue
    message: str
    suggestion: str = ""


@dataclass
class ValidationResult:
    """Result of schema validation."""

    valid: bool
    schema_path: str
    issues: list[ValidationIssue] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.level == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.level == "warning")

    def add_error(self, path: str, message: str, suggestion: str = "") -> None:
        self.issues.append(ValidationIssue("error", path, message, suggestion))
        self.valid = False

    def add_warning(self, path: str, message: str, suggestion: str = "") -> None:
        self.issues.append(ValidationIssue("warning", path, message, suggestion))

    def add_info(self, path: str, message: str) -> None:
        self.issues.append(ValidationIssue("info", path, message))


def load_schema_file(path: str) -> dict[str, Any]:
    """Load a schema from JSON or YAML file."""
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Schema file not found: {path}")

    content = file_path.read_text()

    if file_path.suffix in (".yaml", ".yml"):
        try:
            import yaml

            return yaml.safe_load(content)
        except ImportError:
            raise ImportError("PyYAML required for YAML files. Install with: pip install pyyaml")
    else:
        return json.loads(content)


def load_agent_tool_schema_spec() -> dict[str, Any]:
    """Load the Agent Tool Schema JSON Schema specification."""
    # Look for the schema in common locations
    script_dir = Path(__file__).parent
    possible_paths = [
        script_dir.parent / "schemas" / "agent-tool-schema.json",
        script_dir / "agent-tool-schema.json",
        Path("agent-tool-schema.json"),
    ]

    for path in possible_paths:
        if path.exists():
            return json.loads(path.read_text())

    raise FileNotFoundError(
        "Agent Tool Schema specification not found. "
        "Expected at: skills/shared/schemas/agent-tool-schema.json"
    )


def validate_json_schema(schema: dict[str, Any], spec: dict[str, Any], result: ValidationResult) -> None:
    """Validate schema against JSON Schema specification."""
    if not JSONSCHEMA_AVAILABLE:
        result.add_warning(
            "$",
            "jsonschema package not installed - skipping JSON Schema validation",
            "Install with: pip install jsonschema",
        )
        return

    validator = Draft202012Validator(spec)
    errors = list(validator.iter_errors(schema))

    for error in errors:
        path = ".".join(str(p) for p in error.absolute_path) or "$"
        result.add_error(
            path,
            error.message,
            f"Expected: {error.schema.get('description', 'see schema')}" if error.schema else "",
        )


def validate_type_references(schema: dict[str, Any], result: ValidationResult) -> None:
    """Validate that all type references are valid."""
    defined_types = set(schema.get("types", {}).keys())

    # Also include common built-in types
    defined_types.update(["Task", "any", "string", "integer", "number", "boolean", "array", "object"])

    operations = schema.get("operations", {})

    for op_name, op in operations.items():
        # Check input_type
        input_type = op.get("input_type")
        if input_type and input_type not in defined_types:
            result.add_warning(
                f"operations.{op_name}.input_type",
                f"Referenced type '{input_type}' not defined in types section",
                f"Add '{input_type}' to the types section or use a defined type",
            )

        # Check output_type
        output_type = op.get("output_type")
        if output_type and output_type not in defined_types:
            result.add_warning(
                f"operations.{op_name}.output_type",
                f"Referenced type '{output_type}' not defined in types section",
                f"Add '{output_type}' to the types section or use a defined type",
            )


def validate_composition_consistency(schema: dict[str, Any], result: ValidationResult) -> None:
    """Validate that composition rules are internally consistent."""
    composition = schema.get("composition", {})
    type_flow = composition.get("type_flow", {})
    operations = schema.get("operations", {})

    # Check that all operations in type_flow exist
    for type_name, flow in type_flow.items():
        for producer in flow.get("producers", []):
            if producer not in operations:
                result.add_warning(
                    f"composition.type_flow.{type_name}.producers",
                    f"Operation '{producer}' listed as producer but not defined in operations",
                )

        for consumer in flow.get("consumers", []):
            if consumer not in operations:
                result.add_warning(
                    f"composition.type_flow.{type_name}.consumers",
                    f"Operation '{consumer}' listed as consumer but not defined in operations",
                )

    # Check that operations' declared types match type_flow
    for op_name, op in operations.items():
        output_type = op.get("output_type")
        input_type = op.get("input_type")

        if output_type and output_type in type_flow:
            producers = type_flow[output_type].get("producers", [])
            if op_name not in producers:
                result.add_info(
                    f"composition.type_flow.{output_type}.producers",
                    f"Operation '{op_name}' produces '{output_type}' but not listed in producers",
                )

        if input_type and input_type in type_flow:
            consumers = type_flow[input_type].get("consumers", [])
            if op_name not in consumers:
                result.add_info(
                    f"composition.type_flow.{input_type}.consumers",
                    f"Operation '{op_name}' consumes '{input_type}' but not listed in consumers",
                )


def validate_operation_completeness(schema: dict[str, Any], result: ValidationResult) -> None:
    """Check that operations have recommended fields."""
    operations = schema.get("operations", {})

    for op_name, op in operations.items():
        # Check for description
        if not op.get("description"):
            result.add_warning(
                f"operations.{op_name}.description",
                "Operation missing description",
                "Add a description explaining what this operation does",
            )

        # Check for import
        if not op.get("import"):
            result.add_error(
                f"operations.{op_name}.import",
                "Operation missing import statement",
                "Add an import statement (e.g., 'from module import Class')",
            )

        # Check GPU operations have memory estimate
        resources = op.get("resources", {})
        if resources.get("requires_gpu") and not resources.get("gpu_memory_gb"):
            result.add_warning(
                f"operations.{op_name}.resources.gpu_memory_gb",
                "GPU operation missing memory estimate",
                "Add gpu_memory_gb to help with resource planning",
            )

        # Check for hints on common operation types
        categories = op.get("category", [])
        if isinstance(categories, str):
            categories = [categories]

        if "classifier" in categories or "filter" in categories:
            hints = op.get("hints", {})
            if not hints.get("purpose"):
                result.add_info(
                    f"operations.{op_name}.hints.purpose",
                    "Classifier/filter missing 'purpose' hint",
                )


def validate_parameters(schema: dict[str, Any], result: ValidationResult) -> None:
    """Validate operation parameters."""
    operations = schema.get("operations", {})

    valid_types = {"string", "integer", "number", "boolean", "array", "object", "any", "null"}

    for op_name, op in operations.items():
        parameters = op.get("parameters", {})

        for param_name, param in parameters.items():
            param_type = param.get("type", "")

            # Check type is valid JSON Schema type
            if param_type.lower() not in valid_types:
                # Could be a custom type reference, that's okay
                pass

            # Check numeric constraints make sense
            if param.get("minimum") is not None and param.get("maximum") is not None:
                if param["minimum"] > param["maximum"]:
                    result.add_error(
                        f"operations.{op_name}.parameters.{param_name}",
                        f"minimum ({param['minimum']}) > maximum ({param['maximum']})",
                    )

            # Check default is within constraints
            if param.get("default") is not None:
                default = param["default"]
                if param.get("minimum") is not None and isinstance(default, (int, float)):
                    if default < param["minimum"]:
                        result.add_warning(
                            f"operations.{op_name}.parameters.{param_name}.default",
                            f"Default value {default} is below minimum {param['minimum']}",
                        )
                if param.get("maximum") is not None and isinstance(default, (int, float)):
                    if default > param["maximum"]:
                        result.add_warning(
                            f"operations.{op_name}.parameters.{param_name}.default",
                            f"Default value {default} is above maximum {param['maximum']}",
                        )

                # Check default is in enum
                if param.get("enum") and default not in param["enum"]:
                    result.add_warning(
                        f"operations.{op_name}.parameters.{param_name}.default",
                        f"Default value '{default}' not in enum {param['enum']}",
                    )


def collect_stats(schema: dict[str, Any]) -> dict[str, Any]:
    """Collect statistics about the schema."""
    operations = schema.get("operations", {})
    types = schema.get("types", {})

    # Count by category
    categories: dict[str, int] = {}
    gpu_count = 0
    total_params = 0

    for op in operations.values():
        cats = op.get("category", [])
        if isinstance(cats, str):
            cats = [cats]
        for cat in cats:
            categories[cat] = categories.get(cat, 0) + 1

        if op.get("resources", {}).get("requires_gpu"):
            gpu_count += 1

        total_params += len(op.get("parameters", {}))

    return {
        "schema_version": schema.get("schema_version", "unknown"),
        "tool_name": schema.get("tool", {}).get("name", "unknown"),
        "tool_version": schema.get("tool", {}).get("version", "unknown"),
        "type_count": len(types),
        "operation_count": len(operations),
        "gpu_operation_count": gpu_count,
        "total_parameters": total_params,
        "categories": categories,
    }


def validate_schema(schema_path: str, check_composition: bool = True, verbose: bool = False) -> ValidationResult:
    """Validate a schema file and return results."""
    result = ValidationResult(valid=True, schema_path=schema_path)

    # Load the schema
    try:
        schema = load_schema_file(schema_path)
    except Exception as e:
        result.add_error("$", f"Failed to load schema: {e}")
        return result

    # Load the specification
    try:
        spec = load_agent_tool_schema_spec()
    except FileNotFoundError as e:
        result.add_warning("$", str(e))
        spec = None

    # Collect stats first
    result.stats = collect_stats(schema)

    # Validate against JSON Schema spec
    if spec:
        validate_json_schema(schema, spec, result)

    # Validate type references
    validate_type_references(schema, result)

    # Validate composition consistency
    if check_composition:
        validate_composition_consistency(schema, result)

    # Validate operation completeness
    validate_operation_completeness(schema, result)

    # Validate parameters
    validate_parameters(schema, result)

    return result


def print_result(result: ValidationResult, verbose: bool = False) -> None:
    """Print validation result to console."""
    status = "✅ VALID" if result.valid else "❌ INVALID"
    print(f"\n{status}: {result.schema_path}")
    print("=" * 60)

    # Print stats
    stats = result.stats
    print(f"Tool: {stats.get('tool_name', 'unknown')} v{stats.get('tool_version', 'unknown')}")
    print(f"Operations: {stats.get('operation_count', 0)} ({stats.get('gpu_operation_count', 0)} GPU)")
    print(f"Types: {stats.get('type_count', 0)}")
    print(f"Parameters: {stats.get('total_parameters', 0)}")
    print()

    # Group issues by level
    errors = [i for i in result.issues if i.level == "error"]
    warnings = [i for i in result.issues if i.level == "warning"]
    infos = [i for i in result.issues if i.level == "info"]

    if errors:
        print(f"❌ Errors ({len(errors)}):")
        for issue in errors:
            print(f"   {issue.path}")
            print(f"      {issue.message}")
            if issue.suggestion and verbose:
                print(f"      💡 {issue.suggestion}")
        print()

    if warnings:
        print(f"⚠️  Warnings ({len(warnings)}):")
        for issue in warnings:
            print(f"   {issue.path}")
            print(f"      {issue.message}")
            if issue.suggestion and verbose:
                print(f"      💡 {issue.suggestion}")
        print()

    if verbose and infos:
        print(f"ℹ️  Info ({len(infos)}):")
        for issue in infos:
            print(f"   {issue.path}: {issue.message}")
        print()

    # Summary
    print("-" * 60)
    print(f"Summary: {result.error_count} errors, {result.warning_count} warnings")


def result_to_json(result: ValidationResult) -> dict[str, Any]:
    """Convert result to JSON-serializable dict."""
    return {
        "valid": result.valid,
        "schema_path": result.schema_path,
        "stats": result.stats,
        "error_count": result.error_count,
        "warning_count": result.warning_count,
        "issues": [
            {
                "level": i.level,
                "path": i.path,
                "message": i.message,
                "suggestion": i.suggestion,
            }
            for i in result.issues
        ],
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "schemas",
        nargs="+",
        help="Schema file(s) to validate (JSON or YAML)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed output including suggestions",
    )
    parser.add_argument(
        "--check-composition",
        action="store_true",
        default=True,
        help="Check composition rule consistency (default: true)",
    )
    parser.add_argument(
        "--no-check-composition",
        action="store_false",
        dest="check_composition",
        help="Skip composition rule checking",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON (for CI integration)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors (exit 1 if any warnings)",
    )

    args = parser.parse_args()

    if not JSONSCHEMA_AVAILABLE and not args.json:
        print("⚠️  jsonschema package not installed - some validations will be skipped")
        print("   Install with: pip install jsonschema")
        print()

    results = []
    all_valid = True

    for schema_path in args.schemas:
        result = validate_schema(
            schema_path,
            check_composition=args.check_composition,
            verbose=args.verbose,
        )
        results.append(result)

        if not result.valid:
            all_valid = False
        if args.strict and result.warning_count > 0:
            all_valid = False

    if args.json:
        output = {
            "all_valid": all_valid,
            "results": [result_to_json(r) for r in results],
        }
        print(json.dumps(output, indent=2))
    else:
        for result in results:
            print_result(result, verbose=args.verbose)

    sys.exit(0 if all_valid else 1)


if __name__ == "__main__":
    main()
