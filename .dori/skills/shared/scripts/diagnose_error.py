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

"""Diagnose NeMo Curator pipeline errors and suggest fixes.

This script analyzes error messages and provides:
- Root cause diagnosis
- Actionable fix suggestions
- Related documentation links

Error patterns are loaded from a configurable file, making this
extensible without code changes.

Usage:
    python diagnose_error.py --error "CUDA out of memory"
    python diagnose_error.py --error "ModuleNotFoundError: No module named 'nemo_curator'"
    python diagnose_error.py --error "KeyError: 'text'" --context '{"stage": "WordCountFilter"}'
    python diagnose_error.py --error-file error.log --json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

# Schema path for GPU memory info
SCHEMA_PATH = Path(__file__).parent.parent / "schemas" / "nemo_curator_schema.json"

# Error patterns - can be extended via external config
# Each pattern has: regex, diagnosis, fixes, severity
ERROR_PATTERNS: list[dict[str, Any]] = [
    # GPU/CUDA errors
    {
        "id": "cuda_oom",
        "pattern": r"CUDA out of memory|torch\.cuda\.OutOfMemoryError|CUDA error: out of memory",
        "diagnosis": "GPU ran out of memory during execution",
        "category": "gpu",
        "severity": "error",
        "fixes": [
            {
                "action": "Reduce batch size",
                "how": "Set model_inference_batch_size to a smaller value (try 32 or 16)",
                "example": "stage = QualityClassifier(model_inference_batch_size=16)",
            },
            {
                "action": "Run GPU stages sequentially",
                "how": "Split pipeline so GPU-heavy stages don't run in parallel",
                "example": "Run QualityClassifier separately from AegisClassifier",
            },
            {
                "action": "Use GPU memory fractions",
                "how": "Configure stages to use only part of GPU memory",
            },
            {
                "action": "Use a GPU with more memory",
                "how": "Check stage requirements in schema (typically 4-20GB per stage)",
            },
        ],
    },
    {
        "id": "cuda_not_available",
        "pattern": r"CUDA is not available|no CUDA-capable device|cuda runtime error",
        "diagnosis": "No GPU available or CUDA not properly installed",
        "category": "gpu",
        "severity": "error",
        "fixes": [
            {
                "action": "Check GPU availability",
                "how": "Run: python -c \"import torch; print(torch.cuda.is_available())\"",
            },
            {
                "action": "Install CUDA toolkit",
                "how": "See: https://developer.nvidia.com/cuda-downloads",
            },
            {
                "action": "Use CPU-only stages",
                "how": "Some stages have CPU fallback modes or are CPU-only",
            },
        ],
    },
    
    # Import/module errors
    {
        "id": "module_not_found",
        "pattern": r"ModuleNotFoundError: No module named ['\"]([^'\"]+)['\"]",
        "diagnosis": "Required Python package is not installed",
        "category": "installation",
        "severity": "error",
        "fixes": [
            {
                "action": "Install missing package",
                "how": "Run: pip install {captured_group_1}",
            },
            {
                "action": "Install NeMo Curator with extras",
                "how": "pip install 'nemo-curator[all]' for all dependencies",
            },
        ],
    },
    {
        "id": "import_error",
        "pattern": r"ImportError: cannot import name ['\"]([^'\"]+)['\"]",
        "diagnosis": "Import failed - likely version mismatch or missing dependency",
        "category": "installation",
        "severity": "error",
        "fixes": [
            {
                "action": "Check package versions",
                "how": "Run: pip list | grep nemo",
            },
            {
                "action": "Reinstall NeMo Curator",
                "how": "pip install --upgrade nemo-curator",
            },
        ],
    },
    
    # Data/column errors
    {
        "id": "key_error",
        "pattern": r"KeyError: ['\"]([^'\"]+)['\"]",
        "diagnosis": "Required column or key not found in data",
        "category": "data",
        "severity": "error",
        "fixes": [
            {
                "action": "Check input data columns",
                "how": "Verify your data has the required column: {captured_group_1}",
            },
            {
                "action": "Rename column to match expected name",
                "how": "Most stages expect a 'text' column for text content",
            },
            {
                "action": "Check stage inputs() method",
                "how": "See what columns the stage requires in its inputs() definition",
            },
        ],
    },
    {
        "id": "type_error_dataframe",
        "pattern": r"TypeError.*DataFrame|expected.*DataFrame|got.*instead of.*DataFrame",
        "diagnosis": "Stage received wrong data type (expected DataFrame)",
        "category": "data",
        "severity": "error",
        "fixes": [
            {
                "action": "Check pipeline type flow",
                "how": "Run: python validate_pipeline.py --stages 'YourStages'",
            },
            {
                "action": "Ensure reader produces correct type",
                "how": "JsonlReader and ParquetReader produce DocumentBatch with DataFrame",
            },
        ],
    },
    {
        "id": "empty_dataframe",
        "pattern": r"empty DataFrame|no rows|0 documents",
        "diagnosis": "Pipeline produced empty output - all data was filtered out",
        "category": "data",
        "severity": "warning",
        "fixes": [
            {
                "action": "Check filter thresholds",
                "how": "Your filter thresholds may be too aggressive",
            },
            {
                "action": "Test on sample first",
                "how": "Run: python test_pipeline.py --stages 'YourStages' --sample 20",
            },
            {
                "action": "Analyze input data",
                "how": "Run: python analyze_data.py --input your_data.jsonl",
            },
        ],
    },
    
    # Authentication errors
    {
        "id": "hf_token_missing",
        "pattern": r"HF_TOKEN|huggingface.*token|401.*huggingface|Access to model.*restricted",
        "diagnosis": "Hugging Face authentication token is missing or invalid",
        "category": "auth",
        "severity": "error",
        "fixes": [
            {
                "action": "Set HF_TOKEN environment variable",
                "how": "export HF_TOKEN='your_token_here'",
            },
            {
                "action": "Get a Hugging Face token",
                "how": "Visit: https://huggingface.co/settings/tokens",
            },
            {
                "action": "Accept model license",
                "how": "Some models require accepting a license on the HF website first",
            },
        ],
    },
    {
        "id": "api_rate_limit",
        "pattern": r"rate limit|429|too many requests",
        "diagnosis": "API rate limit exceeded",
        "category": "auth",
        "severity": "warning",
        "fixes": [
            {
                "action": "Reduce request rate",
                "how": "Add delays between requests or reduce batch size",
            },
            {
                "action": "Wait and retry",
                "how": "Rate limits usually reset after a few minutes",
            },
        ],
    },
    
    # File/IO errors
    {
        "id": "file_not_found",
        "pattern": r"FileNotFoundError|No such file or directory|Path does not exist",
        "diagnosis": "Input file or directory not found",
        "category": "io",
        "severity": "error",
        "fixes": [
            {
                "action": "Check file path",
                "how": "Verify the path exists: ls -la /path/to/file",
            },
            {
                "action": "Use absolute paths",
                "how": "Relative paths can be ambiguous in distributed execution",
            },
        ],
    },
    {
        "id": "permission_denied",
        "pattern": r"PermissionError|Permission denied",
        "diagnosis": "Insufficient permissions to read/write file",
        "category": "io",
        "severity": "error",
        "fixes": [
            {
                "action": "Check file permissions",
                "how": "Run: ls -la /path/to/file",
            },
            {
                "action": "Change output directory",
                "how": "Use a directory you have write access to",
            },
        ],
    },
    
    # Type/validation errors
    {
        "id": "type_mismatch",
        "pattern": r"type mismatch|expected.*but got|incompatible types",
        "diagnosis": "Pipeline has a type compatibility issue between stages",
        "category": "pipeline",
        "severity": "error",
        "fixes": [
            {
                "action": "Validate pipeline types",
                "how": "Run: python validate_pipeline.py --stages 'YourStages'",
            },
            {
                "action": "Check input/output types",
                "how": "Each stage's output type must match next stage's input type",
            },
        ],
    },
    
    # Memory errors
    {
        "id": "memory_error",
        "pattern": r"MemoryError|Cannot allocate memory|killed.*memory",
        "diagnosis": "System ran out of RAM",
        "category": "memory",
        "severity": "error",
        "fixes": [
            {
                "action": "Process data in smaller batches",
                "how": "Use batch processing or partition your data",
            },
            {
                "action": "Use streaming mode",
                "how": "Some stages support streaming to reduce memory footprint",
            },
            {
                "action": "Increase system memory",
                "how": "Or use a machine with more RAM",
            },
        ],
    },
]


def load_schema_gpu_info() -> dict[str, float]:
    """Load GPU memory requirements from schema."""
    if not SCHEMA_PATH.exists():
        return {}
    
    with open(SCHEMA_PATH) as f:
        schema = json.load(f)
    
    gpu_info = {}
    for op_name, op_info in schema.get("operations", {}).items():
        resources = op_info.get("resources", {})
        if resources.get("requires_gpu"):
            gpu_info[op_name] = resources.get("gpu_memory_gb", 0.0)
    
    return gpu_info


def match_error(error_text: str) -> list[dict[str, Any]]:
    """Match error text against known patterns."""
    matches = []
    
    for pattern_info in ERROR_PATTERNS:
        regex = pattern_info["pattern"]
        match = re.search(regex, error_text, re.IGNORECASE)
        
        if match:
            result = pattern_info.copy()
            result["matched"] = True
            result["matched_text"] = match.group(0)
            
            # Capture groups for template substitution
            result["captured_groups"] = match.groups()
            
            # Substitute captured groups in fixes
            if match.groups():
                for i, group in enumerate(match.groups(), 1):
                    if group:
                        for fix in result["fixes"]:
                            fix["how"] = fix["how"].replace(f"{{captured_group_{i}}}", group)
            
            matches.append(result)
    
    return matches


def diagnose_error(
    error_text: str,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Diagnose an error and provide fixes."""
    result = {
        "error_text": error_text[:500] + "..." if len(error_text) > 500 else error_text,
        "diagnosed": False,
        "diagnoses": [],
        "general_suggestions": [],
    }
    
    # Match against patterns
    matches = match_error(error_text)
    
    if matches:
        result["diagnosed"] = True
        
        for match in matches:
            diagnosis = {
                "pattern_id": match["id"],
                "category": match["category"],
                "severity": match["severity"],
                "diagnosis": match["diagnosis"],
                "matched_text": match["matched_text"],
                "fixes": match["fixes"],
            }
            result["diagnoses"].append(diagnosis)
    
    # Add context-specific analysis
    if context:
        pipeline = context.get("pipeline", context.get("stages", ""))
        if pipeline:
            # Check for GPU memory issues
            gpu_info = load_schema_gpu_info()
            stages = [s.strip() for s in str(pipeline).split(",")]
            
            total_gpu = 0.0
            gpu_stages = []
            for stage in stages:
                for op_name, mem in gpu_info.items():
                    if stage.lower() in op_name.lower() or op_name.lower() in stage.lower():
                        total_gpu += mem
                        gpu_stages.append((op_name, mem))
                        break
            
            if total_gpu > 0:
                result["gpu_analysis"] = {
                    "stages": gpu_stages,
                    "total_gpu_memory_gb": total_gpu,
                    "note": f"Combined GPU memory requirement: {total_gpu:.1f}GB",
                }
    
    # General suggestions if no specific match
    if not matches:
        result["general_suggestions"] = [
            {
                "action": "Check the full error traceback",
                "how": "Look at the last few lines for the actual error",
            },
            {
                "action": "Search NeMo Curator issues",
                "how": "https://github.com/NVIDIA/NeMo-Curator/issues",
            },
            {
                "action": "Validate your pipeline first",
                "how": "python validate_pipeline.py --stages 'YourStages'",
            },
        ]
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose NeMo Curator pipeline errors and suggest fixes"
    )
    parser.add_argument(
        "--error",
        type=str,
        default=None,
        help="Error message to diagnose",
    )
    parser.add_argument(
        "--error-file",
        type=Path,
        default=None,
        help="File containing error message/traceback",
    )
    parser.add_argument(
        "--context",
        type=str,
        default=None,
        help="JSON string with context (e.g., '{\"pipeline\": \"Stage1,Stage2\"}')",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    
    args = parser.parse_args()
    
    # Get error text
    error_text = None
    if args.error:
        error_text = args.error
    elif args.error_file:
        if args.error_file.exists():
            error_text = args.error_file.read_text()
        else:
            print(f"Error: File not found: {args.error_file}", file=sys.stderr)
            sys.exit(1)
    else:
        # Read from stdin
        print("Enter error message (Ctrl+D when done):", file=sys.stderr)
        error_text = sys.stdin.read()
    
    if not error_text or not error_text.strip():
        print("Error: No error text provided", file=sys.stderr)
        sys.exit(1)
    
    # Parse context
    context = None
    if args.context:
        try:
            context = json.loads(args.context)
        except json.JSONDecodeError as e:
            print(f"Warning: Could not parse context JSON: {e}", file=sys.stderr)
    
    # Diagnose
    result = diagnose_error(error_text, context)
    
    # Output
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        # Human-readable output
        print("\n🔍 Error Diagnosis")
        print("=" * 60)
        
        # Show matched error
        print(f"\n📋 Error:")
        print(f"   {result['error_text'][:200]}{'...' if len(result['error_text']) > 200 else ''}")
        
        if result["diagnosed"]:
            print(f"\n✅ Diagnosis Found ({len(result['diagnoses'])} match(es)):")
            
            for i, diag in enumerate(result["diagnoses"], 1):
                severity_icon = {"error": "❌", "warning": "⚠️", "info": "ℹ️"}.get(diag["severity"], "•")
                print(f"\n   {severity_icon} {diag['diagnosis']}")
                print(f"      Category: {diag['category']}")
                print(f"      Matched: \"{diag['matched_text'][:50]}...\"" if len(diag['matched_text']) > 50 else f"      Matched: \"{diag['matched_text']}\"")
                
                print(f"\n   💡 Fixes:")
                for j, fix in enumerate(diag["fixes"], 1):
                    print(f"      {j}. {fix['action']}")
                    print(f"         {fix['how']}")
                    if fix.get("example"):
                        print(f"         Example: {fix['example']}")
        else:
            print(f"\n❓ No specific diagnosis found")
            
            if result["general_suggestions"]:
                print(f"\n💡 General Suggestions:")
                for i, sug in enumerate(result["general_suggestions"], 1):
                    print(f"   {i}. {sug['action']}")
                    print(f"      {sug['how']}")
        
        # GPU analysis
        if result.get("gpu_analysis"):
            print(f"\n🔧 GPU Analysis:")
            for stage, mem in result["gpu_analysis"]["stages"]:
                print(f"   • {stage}: {mem:.1f}GB")
            print(f"   Total: {result['gpu_analysis']['total_gpu_memory_gb']:.1f}GB")
        
        print("\n" + "=" * 60)
        print()
    
    sys.exit(0)


if __name__ == "__main__":
    main()
