# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Simple pipeline runner for NeMo Curator.

This script demonstrates running a text curation pipeline with minimal filtering.

Usage:
    python run_pipeline.py --input sample_data/input.jsonl --output sample_data/output
"""

import argparse
import json
import sys
from pathlib import Path


def check_platform():
    """Check if running on a supported platform."""
    if sys.platform != "linux":
        print(f"WARNING: NeMo Curator requires Linux. Current platform: {sys.platform}")
        print("Running in Docker is recommended for macOS/Windows.")
        print()
        return False
    return True


def run_simple_filter(input_path: str, output_path: str, min_words: int = 50, max_non_alpha: float = 0.25):
    """Run a simple text filtering pipeline without NeMo Curator (for demo/testing).
    
    This is a standalone implementation that mimics the NeMo Curator filters
    for testing purposes when NeMo Curator isn't available.
    """
    import re
    
    input_file = Path(input_path)
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read input
    documents = []
    with open(input_file) as f:
        for line in f:
            documents.append(json.loads(line))
    
    print(f"Loaded {len(documents)} documents from {input_file}")
    
    # Apply filters
    passed = []
    filtered_reasons = {"word_count": 0, "non_alpha": 0}
    
    alphanum_pattern = re.compile(r'[a-zA-Z0-9]')
    
    for doc in documents:
        text = doc.get("text", "")
        
        # Word count filter
        words = text.split()
        if len(words) < min_words:
            filtered_reasons["word_count"] += 1
            continue
        
        # Non-alphanumeric filter
        if len(text) > 0:
            alpha_count = len(alphanum_pattern.findall(text))
            non_alpha_ratio = (len(text) - alpha_count) / len(text)
            if non_alpha_ratio > max_non_alpha:
                filtered_reasons["non_alpha"] += 1
                continue
        
        passed.append(doc)
    
    # Write output
    output_file = output_dir / "filtered.jsonl"
    with open(output_file, "w") as f:
        for doc in passed:
            f.write(json.dumps(doc) + "\n")
    
    print(f"\n=== Filtering Results ===")
    print(f"Input:  {len(documents)} documents")
    print(f"Output: {len(passed)} documents")
    print(f"Filtered: {len(documents) - len(passed)} documents")
    print(f"  - Word count (<{min_words}): {filtered_reasons['word_count']}")
    print(f"  - Non-alpha (>{max_non_alpha*100:.0f}%): {filtered_reasons['non_alpha']}")
    print(f"\nOutput written to: {output_file}")


def run_nemo_curator_pipeline(input_path: str, output_path: str):
    """Run the full NeMo Curator pipeline."""
    try:
        # Import NeMo Curator (will fail on non-Linux)
        from nemo_curator.stages.text.filters import NonAlphaNumericFilter, WordCountFilter
        from nemo_curator.stages.text.io.reader import JsonlReader
        from nemo_curator.stages.text.io.writer import ParquetWriter
        from nemo_curator.stages.text.modules import ScoreFilter
        from nemo_curator.pipeline import Pipeline
        
        print("NeMo Curator loaded successfully!")
        
        # Ensure output directory exists
        Path(output_path).mkdir(parents=True, exist_ok=True)
        
        # Build pipeline
        pipeline = Pipeline(
            name="text_quick_pass",
            stages=[
                JsonlReader(file_paths=input_path),
                ScoreFilter(
                    filter_obj=WordCountFilter(min_words=50, max_words=100000),
                    text_field="text"
                ),
                ScoreFilter(
                    filter_obj=NonAlphaNumericFilter(max_non_alpha_numeric_to_text_ratio=0.25),
                    text_field="text"
                ),
                ParquetWriter(path=output_path),  # Use 'path' not 'output_path'
            ]
        )
        
        # Run
        print("Running pipeline...")
        pipeline.run()
        print(f"Pipeline complete! Output written to: {output_path}")
        
    except ImportError as e:
        print(f"NeMo Curator import failed: {e}")
        print("Falling back to simple filter implementation...")
        run_simple_filter(input_path, output_path)
    except ValueError as e:
        if "Linux" in str(e):
            print(f"Platform error: {e}")
            print("Falling back to simple filter implementation...")
            run_simple_filter(input_path, output_path)
        else:
            raise


def main():
    parser = argparse.ArgumentParser(description="Run text curation pipeline")
    parser.add_argument("--input", type=str, default="sample_data/input.jsonl", help="Input file path")
    parser.add_argument("--output", type=str, default="sample_data/output", help="Output directory")
    parser.add_argument("--simple", action="store_true", help="Use simple filter (no NeMo Curator)")
    args = parser.parse_args()
    
    print("=" * 50)
    print("NeMo Curator - Text Quick Pass Pipeline")
    print("=" * 50)
    print()
    
    check_platform()
    
    if args.simple:
        run_simple_filter(args.input, args.output)
    else:
        run_nemo_curator_pipeline(args.input, args.output)


if __name__ == "__main__":
    main()
