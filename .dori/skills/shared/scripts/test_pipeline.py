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

"""Test a NeMo Curator pipeline on sample data before full execution.

This script runs a pipeline on a small sample of data to verify:
- Stages execute without errors
- Type flow works correctly
- Output columns are created as expected
- Approximate retention rates

This requires NeMo Curator to be installed.

Usage:
    python test_pipeline.py --stages "WordCountFilter,QualityClassifier" --input data.jsonl --sample 20
    python test_pipeline.py --stages "Reader,Filter" --input data.jsonl --sample 10 --json
    python test_pipeline.py --dry-run --stages "Stage1,Stage2" --input data.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

# Check if NeMo Curator is available
NEMO_CURATOR_AVAILABLE = False
try:
    import nemo_curator
    from nemo_curator.tasks import DocumentBatch
    NEMO_CURATOR_AVAILABLE = True
except ImportError:
    pass


def get_stage_class(stage_name: str) -> type | None:
    """Dynamically import and return a stage class by name.
    
    Uses dynamic imports from the schema's import paths.
    """
    if not NEMO_CURATOR_AVAILABLE:
        return None
    
    # Load schema to get import paths
    schema_path = Path(__file__).parent.parent / "schemas" / "nemo_curator_schema.json"
    if schema_path.exists():
        with open(schema_path) as f:
            schema = json.load(f)
        
        operations = schema.get("operations", {})
        
        # Find the operation
        for op_name, op_info in operations.items():
            if op_name.lower() == stage_name.lower() or stage_name.lower() in op_name.lower():
                import_path = op_info.get("import", "")
                if import_path:
                    # Parse import statement
                    # Format: "from module.path import ClassName"
                    try:
                        parts = import_path.split()
                        if len(parts) >= 4 and parts[0] == "from" and parts[2] == "import":
                            module_path = parts[1]
                            class_name = parts[3]
                            
                            import importlib
                            module = importlib.import_module(module_path)
                            return getattr(module, class_name, None)
                    except Exception:
                        continue
    
    # Fallback: try common locations
    common_modules = [
        "nemo_curator.stages.text.filters.heuristic_filter",
        "nemo_curator.stages.text.filters",
        "nemo_curator.stages.text.classifiers",
        "nemo_curator.stages.text.classifiers.quality",
        "nemo_curator.stages.text.modifiers",
        "nemo_curator.stages.text.io.reader",
        "nemo_curator.stages.text.io.writer",
    ]
    
    for module_path in common_modules:
        try:
            import importlib
            module = importlib.import_module(module_path)
            if hasattr(module, stage_name):
                return getattr(module, stage_name)
        except ImportError:
            continue
    
    return None


def read_sample_data(
    input_path: Path,
    sample_size: int,
    text_column: str = "text",
) -> list[dict[str, Any]]:
    """Read sample documents from input file."""
    suffix = input_path.suffix.lower()
    docs = []
    
    if suffix == ".jsonl":
        with open(input_path) as f:
            for i, line in enumerate(f):
                if i >= sample_size:
                    break
                if line.strip():
                    try:
                        docs.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    
    elif suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
            table = pq.read_table(input_path)
            for i in range(min(sample_size, len(table))):
                row = {col: table[col][i].as_py() for col in table.column_names}
                docs.append(row)
        except ImportError:
            print("Warning: pyarrow not installed", file=sys.stderr)
    
    elif suffix == ".json":
        with open(input_path) as f:
            data = json.load(f)
        if isinstance(data, list):
            docs = data[:sample_size]
    
    return docs


def create_document_batch(docs: list[dict[str, Any]]) -> Any:
    """Create a DocumentBatch from documents."""
    if not NEMO_CURATOR_AVAILABLE:
        return None
    
    import pandas as pd
    df = pd.DataFrame(docs)
    
    return DocumentBatch(
        task_id="test_sample",
        dataset_name="test",
        data=df,
    )


def run_stage(
    stage_class: type,
    task: Any,
    stage_params: dict[str, Any] | None = None,
) -> tuple[Any | None, dict[str, Any]]:
    """Run a single stage on a task.
    
    Returns (output_task, stage_info).
    """
    info = {
        "stage_name": stage_class.__name__,
        "success": False,
        "error": None,
        "runtime_seconds": 0,
        "input_rows": 0,
        "output_rows": 0,
        "new_columns": [],
    }
    
    try:
        # Get input info
        if hasattr(task, "data") and hasattr(task.data, "__len__"):
            info["input_rows"] = len(task.data)
        
        # Record input columns
        input_columns = set()
        if hasattr(task, "data"):
            import pandas as pd
            if isinstance(task.data, pd.DataFrame):
                input_columns = set(task.data.columns)
        
        # Instantiate stage
        params = stage_params or {}
        stage = stage_class(**params)
        
        # Run stage
        start_time = time.time()
        result = stage.process(task)
        info["runtime_seconds"] = time.time() - start_time
        
        # Handle result
        if result is None:
            info["output_rows"] = 0
            info["success"] = True
            return None, info
        
        if isinstance(result, list):
            # Multiple output tasks
            total_rows = 0
            for r in result:
                if hasattr(r, "data") and hasattr(r.data, "__len__"):
                    total_rows += len(r.data)
            info["output_rows"] = total_rows
            info["success"] = True
            return result[0] if result else None, info
        
        # Single output task
        if hasattr(result, "data") and hasattr(result.data, "__len__"):
            info["output_rows"] = len(result.data)
            
            # Check for new columns
            import pandas as pd
            if isinstance(result.data, pd.DataFrame):
                output_columns = set(result.data.columns)
                info["new_columns"] = list(output_columns - input_columns)
        
        info["success"] = True
        return result, info
    
    except Exception as e:
        info["error"] = str(e)
        info["traceback"] = traceback.format_exc()
        return None, info


def test_pipeline(
    stage_names: list[str],
    input_path: Path,
    sample_size: int,
    text_column: str = "text",
    stage_params: dict[str, dict[str, Any]] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Test a pipeline on sample data."""
    result = {
        "success": False,
        "stages_tested": [],
        "input_docs": 0,
        "output_docs": 0,
        "retention": 0.0,
        "total_runtime_seconds": 0.0,
        "errors": [],
        "sample_output": [],
    }
    
    # Check NeMo Curator availability
    if not NEMO_CURATOR_AVAILABLE:
        result["errors"].append("NeMo Curator is not installed. Install with: pip install nemo-curator")
        return result
    
    # Load sample data
    docs = read_sample_data(input_path, sample_size, text_column)
    if not docs:
        result["errors"].append(f"Could not read any documents from {input_path}")
        return result
    
    result["input_docs"] = len(docs)
    
    # Dry run - just check stages exist
    if dry_run:
        for stage_name in stage_names:
            stage_class = get_stage_class(stage_name)
            if stage_class:
                result["stages_tested"].append({
                    "stage_name": stage_name,
                    "canonical_name": stage_class.__name__,
                    "found": True,
                    "dry_run": True,
                })
            else:
                result["stages_tested"].append({
                    "stage_name": stage_name,
                    "found": False,
                    "dry_run": True,
                })
                result["errors"].append(f"Stage not found: {stage_name}")
        
        result["success"] = len(result["errors"]) == 0
        return result
    
    # Create initial task
    current_task = create_document_batch(docs)
    if current_task is None:
        result["errors"].append("Could not create DocumentBatch")
        return result
    
    # Run each stage
    stage_params = stage_params or {}
    total_runtime = 0.0
    
    for stage_name in stage_names:
        stage_class = get_stage_class(stage_name)
        
        if stage_class is None:
            result["errors"].append(f"Stage not found: {stage_name}")
            result["stages_tested"].append({
                "stage_name": stage_name,
                "found": False,
                "success": False,
            })
            continue
        
        # Get params for this stage
        params = stage_params.get(stage_name, {})
        
        # Run stage
        output_task, stage_info = run_stage(stage_class, current_task, params)
        result["stages_tested"].append(stage_info)
        total_runtime += stage_info.get("runtime_seconds", 0)
        
        if not stage_info["success"]:
            result["errors"].append(f"Stage {stage_name} failed: {stage_info.get('error', 'unknown error')}")
            break
        
        if output_task is None:
            # All data filtered out
            result["stages_tested"][-1]["note"] = "All documents filtered out"
            break
        
        current_task = output_task
    
    result["total_runtime_seconds"] = total_runtime
    
    # Get final output info
    if current_task is not None and hasattr(current_task, "data"):
        import pandas as pd
        if isinstance(current_task.data, pd.DataFrame):
            result["output_docs"] = len(current_task.data)
            result["output_columns"] = list(current_task.data.columns)
            
            # Sample output (first few rows)
            sample_df = current_task.data.head(3)
            result["sample_output"] = sample_df.to_dict("records")
    
    # Calculate retention
    if result["input_docs"] > 0:
        result["retention"] = result["output_docs"] / result["input_docs"]
    
    # Overall success
    result["success"] = len(result["errors"]) == 0 and result["output_docs"] > 0
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Test a NeMo Curator pipeline on sample data"
    )
    parser.add_argument(
        "--stages",
        required=True,
        help="Comma-separated list of stage names",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to input data file",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=20,
        help="Number of documents to sample (default: 20)",
    )
    parser.add_argument(
        "--text-column",
        default="text",
        help="Column name containing text (default: 'text')",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only check if stages exist, don't run them",
    )
    parser.add_argument(
        "--params",
        type=str,
        default=None,
        help="JSON string with stage parameters, e.g., '{\"WordCountFilter\": {\"min_words\": 50}}'",
    )
    
    args = parser.parse_args()
    
    # Parse stages
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    
    if not stages:
        print("Error: No stages provided", file=sys.stderr)
        sys.exit(1)
    
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    
    # Parse params
    stage_params = None
    if args.params:
        try:
            stage_params = json.loads(args.params)
        except json.JSONDecodeError as e:
            print(f"Error parsing --params: {e}", file=sys.stderr)
            sys.exit(1)
    
    # Run test
    result = test_pipeline(
        stages,
        args.input,
        args.sample,
        args.text_column,
        stage_params,
        args.dry_run,
    )
    
    # Output
    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        # Human-readable output
        status = "✅ SUCCESS" if result["success"] else "❌ FAILED"
        print(f"\n{status} Pipeline Test: {' → '.join(stages)}")
        print("=" * 60)
        
        if result.get("errors") and "not installed" in result["errors"][0]:
            print(f"\n❌ {result['errors'][0]}")
            print("\nThis script requires NeMo Curator to be installed.")
            print("Install with: pip install nemo-curator")
            sys.exit(1)
        
        if args.dry_run:
            print("\n🔍 Dry Run (checking stage availability):")
            for stage in result["stages_tested"]:
                found = "✓" if stage.get("found") else "✗"
                name = stage.get("canonical_name", stage.get("stage_name", "unknown"))
                print(f"   {found} {name}")
        else:
            print(f"\n📊 Results:")
            print(f"   Input docs: {result['input_docs']}")
            print(f"   Output docs: {result['output_docs']}")
            print(f"   Retention: {result['retention']:.1%}")
            print(f"   Runtime: {result['total_runtime_seconds']:.2f}s")
            
            if result.get("output_columns"):
                print(f"\n📋 Output Columns:")
                print(f"   {', '.join(result['output_columns'])}")
            
            print(f"\n📦 Stage Details:")
            for stage in result["stages_tested"]:
                status_icon = "✓" if stage.get("success") else "✗"
                name = stage.get("stage_name", "unknown")
                in_rows = stage.get("input_rows", "?")
                out_rows = stage.get("output_rows", "?")
                runtime = stage.get("runtime_seconds", 0)
                
                print(f"   {status_icon} {name}: {in_rows} → {out_rows} ({runtime:.2f}s)")
                
                if stage.get("new_columns"):
                    print(f"      New columns: {', '.join(stage['new_columns'])}")
                
                if stage.get("error"):
                    print(f"      Error: {stage['error']}")
        
        if result.get("errors"):
            print(f"\n❌ Errors:")
            for error in result["errors"]:
                print(f"   • {error}")
        
        if result.get("sample_output") and not args.dry_run:
            print(f"\n📄 Sample Output (first {len(result['sample_output'])} docs):")
            for i, doc in enumerate(result["sample_output"], 1):
                # Show truncated text
                text = doc.get("text", "")
                if len(text) > 80:
                    text = text[:80] + "..."
                print(f"   {i}. {text[:60]}...")
                # Show added columns
                for key, value in doc.items():
                    if key not in ["text", "id", "url"]:
                        print(f"      {key}: {value}")
        
        print("\n" + "=" * 60)
        if result["success"]:
            print("✅ Pipeline test passed - ready for full execution")
        else:
            print("❌ Fix errors before running full pipeline")
        print()
    
    sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
