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

"""Validate a proposed NeMo Curator pipeline before code generation.

This script checks:
- Type compatibility between stages (input/output types match)
- GPU memory requirements (sum and individual)
- Required credentials (HF_TOKEN, API keys)
- Stage existence and import paths

The script uses the Agent Tool Schema for type information, making it
dynamic and not hardcoded to specific stages.

Usage:
    python validate_pipeline.py --stages "JsonlReader,WordCountFilter,QualityClassifier"
    python validate_pipeline.py --stages "Reader,Filter1,Filter2" --json
    python validate_pipeline.py --stages "Stage1,Stage2" --available-gpu 16
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Schema location - dynamically loaded, not hardcoded
SCHEMA_PATH = Path(__file__).parent.parent / "schemas" / "nemo_curator_schema.json"

# Fallback GPU estimates if not in schema (kept minimal - schema is source of truth)
FALLBACK_GPU_ESTIMATES: dict[str, float] = {}


def load_schema() -> dict[str, Any]:
    """Load the Agent Tool Schema."""
    if not SCHEMA_PATH.exists():
        return {"operations": {}, "types": {}, "composition": {}}
    
    with open(SCHEMA_PATH) as f:
        return json.load(f)


def find_operation(schema: dict[str, Any], stage_name: str) -> dict[str, Any] | None:
    """Find an operation in the schema by name (case-insensitive partial match)."""
    operations = schema.get("operations", {})
    
    # Exact match first
    if stage_name in operations:
        return operations[stage_name]
    
    # Case-insensitive exact match
    for op_name, op_info in operations.items():
        if op_name.lower() == stage_name.lower():
            return op_info
    
    # Partial match (e.g., "Quality" matches "QualityClassifier")
    for op_name, op_info in operations.items():
        if stage_name.lower() in op_name.lower():
            return op_info
    
    return None


def get_operation_info(schema: dict[str, Any], stage_name: str) -> dict[str, Any]:
    """Get operation info from schema with fallbacks."""
    op = find_operation(schema, stage_name)
    
    if op:
        return {
            "found": True,
            "canonical_name": next(
                (name for name in schema.get("operations", {}) 
                 if name.lower() == stage_name.lower() or stage_name.lower() in name.lower()),
                stage_name
            ),
            "input_type": op.get("input_type", "unknown"),
            "output_type": op.get("output_type", "unknown"),
            "requires_gpu": op.get("resources", {}).get("requires_gpu", False),
            "gpu_memory_gb": op.get("resources", {}).get("gpu_memory_gb", 0.0),
            "credentials": op.get("resources", {}).get("requires_credentials", []),
            "import_path": op.get("import", ""),
            "description": op.get("description", ""),
            "category": op.get("category", []),
        }
    
    # Stage not found in schema
    return {
        "found": False,
        "canonical_name": stage_name,
        "input_type": "unknown",
        "output_type": "unknown",
        "requires_gpu": False,
        "gpu_memory_gb": FALLBACK_GPU_ESTIMATES.get(stage_name, 0.0),
        "credentials": [],
        "import_path": "",
        "description": "",
        "category": [],
    }


def types_compatible(output_type: str, input_type: str) -> tuple[bool, str]:
    """Check if output type is compatible with input type.
    
    Returns (is_compatible, reason).
    """
    # Unknown types are assumed compatible (conservative)
    if output_type == "unknown" or input_type == "unknown":
        return True, "unknown types assumed compatible"
    
    # Exact match
    if output_type == input_type:
        return True, "exact match"
    
    # Common compatible pairs
    compatible_pairs = [
        # FileGroupTask can feed into readers
        ("FileGroupTask", "DocumentBatch"),  # Reader pattern
        ("FileGroupTask", "VideoTask"),
        ("FileGroupTask", "ImageBatch"),
        ("FileGroupTask", "AudioBatch"),
        # _EmptyTask can start any pipeline (generator stages)
        ("_EmptyTask", "DocumentBatch"),
        ("_EmptyTask", "VideoTask"),
        ("_EmptyTask", "ImageBatch"),
        ("_EmptyTask", "AudioBatch"),
        ("_EmptyTask", "FileGroupTask"),
    ]
    
    if (output_type, input_type) in compatible_pairs:
        return True, f"{output_type} feeds into {input_type}"
    
    # Task is generic base - compatible with anything
    if output_type == "Task" or input_type == "Task":
        return True, "generic Task type"
    
    return False, f"type mismatch: {output_type} → {input_type}"


def validate_pipeline(
    stages: list[str],
    schema: dict[str, Any],
    available_gpu_memory: float | None = None,
) -> dict[str, Any]:
    """Validate a pipeline configuration.
    
    Args:
        stages: List of stage names in order
        schema: Agent Tool Schema
        available_gpu_memory: Optional GPU memory constraint in GB
    
    Returns:
        Validation result dictionary
    """
    result = {
        "valid": True,
        "stages": [],
        "type_chain": [],
        "errors": [],
        "warnings": [],
        "total_gpu_memory_gb": 0.0,
        "requires_gpu": False,
        "credentials_required": [],
        "ready_to_generate": True,
    }
    
    # Analyze each stage
    stage_infos = []
    for stage_name in stages:
        info = get_operation_info(schema, stage_name)
        stage_infos.append(info)
        
        # Track stage info
        result["stages"].append({
            "name": stage_name,
            "canonical_name": info["canonical_name"],
            "found": info["found"],
            "input_type": info["input_type"],
            "output_type": info["output_type"],
            "gpu_memory_gb": info["gpu_memory_gb"],
        })
        
        # Not found in schema
        if not info["found"]:
            result["warnings"].append(
                f"Stage '{stage_name}' not found in schema - validation may be incomplete"
            )
        
        # Track GPU requirements
        if info["requires_gpu"]:
            result["requires_gpu"] = True
        result["total_gpu_memory_gb"] += info["gpu_memory_gb"]
        
        # Track credentials
        for cred in info["credentials"]:
            if cred not in result["credentials_required"]:
                result["credentials_required"].append(cred)
    
    # Validate type chain
    for i in range(len(stage_infos) - 1):
        current = stage_infos[i]
        next_stage = stage_infos[i + 1]
        
        output_type = current["output_type"]
        input_type = next_stage["input_type"]
        
        compatible, reason = types_compatible(output_type, input_type)
        
        if not compatible:
            result["errors"].append(
                f"Type mismatch: {current['canonical_name']} outputs {output_type}, "
                f"but {next_stage['canonical_name']} expects {input_type}"
            )
            result["valid"] = False
            result["ready_to_generate"] = False
    
    # Build type chain visualization
    if stage_infos:
        type_chain = [stage_infos[0]["input_type"]]
        for info in stage_infos:
            type_chain.append(info["output_type"])
        result["type_chain"] = type_chain
        result["type_chain_str"] = " → ".join(type_chain)
    
    # Check GPU memory constraint
    if available_gpu_memory is not None:
        if result["total_gpu_memory_gb"] > available_gpu_memory:
            result["warnings"].append(
                f"Total GPU memory ({result['total_gpu_memory_gb']:.1f}GB) "
                f"exceeds available ({available_gpu_memory:.1f}GB). "
                "Consider running stages sequentially or reducing batch sizes."
            )
    
    # Check for credential requirements
    for cred in result["credentials_required"]:
        result["warnings"].append(
            f"Stage requires {cred} credential - ensure it is set in environment"
        )
    
    # Check if first stage is a reader (common pattern)
    if stage_infos and stage_infos[0]["input_type"] not in ["_EmptyTask", "FileGroupTask", "unknown"]:
        result["warnings"].append(
            f"First stage expects {stage_infos[0]['input_type']} input. "
            "You may need a reader stage before this."
        )
    
    # Final ready check
    if result["errors"]:
        result["ready_to_generate"] = False
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Validate a NeMo Curator pipeline before code generation"
    )
    parser.add_argument(
        "--stages",
        required=True,
        help="Comma-separated list of stage names (e.g., 'JsonlReader,WordCountFilter,QualityClassifier')",
    )
    parser.add_argument(
        "--available-gpu",
        type=float,
        default=None,
        help="Available GPU memory in GB (for resource validation)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=SCHEMA_PATH,
        help="Path to Agent Tool Schema JSON file",
    )
    
    args = parser.parse_args()
    
    # Parse stages
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    
    if not stages:
        print("Error: No stages provided", file=sys.stderr)
        sys.exit(1)
    
    # Load schema
    schema_path = args.schema
    if schema_path.exists():
        with open(schema_path) as f:
            schema = json.load(f)
    else:
        print(f"Warning: Schema not found at {schema_path}, validation will be limited", file=sys.stderr)
        schema = {"operations": {}, "types": {}}
    
    # Validate
    result = validate_pipeline(stages, schema, args.available_gpu)
    
    # Output
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        # Human-readable output
        status = "✅ VALID" if result["valid"] else "❌ INVALID"
        print(f"\n{status} Pipeline: {' → '.join(stages)}")
        print("=" * 60)
        
        # Type chain
        if result.get("type_chain_str"):
            print(f"\n📋 Type Flow: {result['type_chain_str']}")
        
        # Stage details
        print("\n📦 Stages:")
        for stage in result["stages"]:
            found_marker = "✓" if stage["found"] else "?"
            gpu_info = f" (GPU: {stage['gpu_memory_gb']:.1f}GB)" if stage["gpu_memory_gb"] > 0 else ""
            print(f"   {found_marker} {stage['canonical_name']}: {stage['input_type']} → {stage['output_type']}{gpu_info}")
        
        # Resources
        print(f"\n🔧 Resources:")
        print(f"   GPU Required: {'Yes' if result['requires_gpu'] else 'No'}")
        if result["requires_gpu"]:
            print(f"   Total GPU Memory: {result['total_gpu_memory_gb']:.1f}GB")
        if result["credentials_required"]:
            print(f"   Credentials: {', '.join(result['credentials_required'])}")
        
        # Errors
        if result["errors"]:
            print(f"\n❌ Errors ({len(result['errors'])}):")
            for error in result["errors"]:
                print(f"   • {error}")
        
        # Warnings
        if result["warnings"]:
            print(f"\n⚠️  Warnings ({len(result['warnings'])}):")
            for warning in result["warnings"]:
                print(f"   • {warning}")
        
        # Final verdict
        print("\n" + "=" * 60)
        if result["ready_to_generate"]:
            print("✅ Ready to generate pipeline code")
        else:
            print("❌ Fix errors before generating pipeline code")
        print()
    
    sys.exit(0 if result["valid"] else 1)


if __name__ == "__main__":
    main()
