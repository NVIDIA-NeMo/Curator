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

"""Example: Annotate code files with language detection, basic stats, and metrics.

This example demonstrates how to use the code annotation modifiers to:
1. Detect programming languages
2. Compute basic statistics (lines, bytes, patterns)
3. Compute software metrics (complexity, maintainability)
4. Compute OpenCoder-style comment fractions
5. Tokenize code

Usage:
    python annotate_code.py --input_dir /path/to/code --output_file annotated.parquet

    # Or with sample data:
    python annotate_code.py --use_sample_data --output_file sample_annotated.parquet
"""

import argparse
from pathlib import Path

import pandas as pd

from nemo_curator.stages.code import CodeAnnotator


def load_code_files(input_dir: str) -> pd.DataFrame:
    """Load code files from a directory into a DataFrame."""
    code_extensions = {
        ".py",
        ".rs",
        ".java",
        ".js",
        ".ts",
        ".cpp",
        ".c",
        ".h",
        ".go",
        ".rb",
        ".php",
        ".cs",
        ".swift",
        ".kt",
        ".scala",
        ".sh",
        ".bash",
    }

    records = []
    input_path = Path(input_dir)

    for file_path in input_path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in code_extensions:
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                records.append(
                    {
                        "content": content,
                        "representative_filename": file_path.name,
                        "file_path": str(file_path),
                    }
                )
            except OSError as e:
                print(f"Warning: Could not read {file_path}: {e}")

    return pd.DataFrame(records)


def get_sample_data() -> pd.DataFrame:
    """Generate sample code data for testing."""
    samples = [
        {
            "content": '''def factorial(n):
    """Calculate factorial recursively."""
    # Base case
    if n <= 1:
        return 1
    return n * factorial(n - 1)


def fibonacci(n):
    """Calculate nth Fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)
''',
            "representative_filename": "math_utils.py",
        },
        {
            "content": """fn main() {
    // Print hello world
    println!("Hello, world!");

    // Calculate sum
    let sum: i32 = (1..=100).sum();
    println!("Sum 1-100: {}", sum);
}

fn is_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    for i in 2..=(n as f64).sqrt() as u64 {
        if n % i == 0 {
            return false;
        }
    }
    true
}
""",
            "representative_filename": "main.rs",
        },
        {
            "content": """public class Calculator {
    /**
     * Add two numbers.
     * @param a First number
     * @param b Second number
     * @return Sum of a and b
     */
    public static int add(int a, int b) {
        return a + b;
    }

    public static int multiply(int a, int b) {
        return a * b;
    }

    public static void main(String[] args) {
        System.out.println("2 + 3 = " + add(2, 3));
        System.out.println("4 * 5 = " + multiply(4, 5));
    }
}
""",
            "representative_filename": "Calculator.java",
        },
        {
            "content": """// Simple Express.js server
const express = require('express');
const app = express();

// Middleware
app.use(express.json());

// Routes
app.get('/', (req, res) => {
    res.json({ message: 'Hello World!' });
});

app.get('/api/users', (req, res) => {
    // Return sample users
    res.json([
        { id: 1, name: 'Alice' },
        { id: 2, name: 'Bob' }
    ]);
});

// Start server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});
""",
            "representative_filename": "server.js",
        },
        {
            "content": """package main

import (
    "fmt"
    "sort"
)

// BinarySearch performs binary search on a sorted slice
func BinarySearch(arr []int, target int) int {
    left, right := 0, len(arr)-1

    for left <= right {
        mid := left + (right-left)/2

        if arr[mid] == target {
            return mid
        } else if arr[mid] < target {
            left = mid + 1
        } else {
            right = mid - 1
        }
    }

    return -1
}

func main() {
    numbers := []int{3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5}
    sort.Ints(numbers)
    fmt.Println("Sorted:", numbers)

    idx := BinarySearch(numbers, 5)
    fmt.Printf("Index of 5: %d\\n", idx)
}
""",
            "representative_filename": "search.go",
        },
    ]

    return pd.DataFrame(samples)


def _print_annotation_summary(annotated_df: pd.DataFrame) -> None:
    """Print summary statistics of the annotated DataFrame."""
    print("\n" + "=" * 60)
    print("ANNOTATION SUMMARY")
    print("=" * 60)

    print(f"\nTotal files: {len(annotated_df)}")
    print(f"Total columns: {len(annotated_df.columns)}")

    if "language" in annotated_df.columns:
        print("\nLanguage distribution:")
        lang_counts = annotated_df["language"].value_counts()
        for lang, count in lang_counts.items():
            print(f"  {lang}: {count}")

    if "basic_num_bytes" in annotated_df.columns:
        print("\nCode size stats:")
        print(f"  Total bytes: {annotated_df['basic_num_bytes'].sum():,}")
        print(f"  Total lines: {annotated_df['basic_num_lines'].sum():,}")
        print(f"  Avg lines/file: {annotated_df['basic_num_lines'].mean():.1f}")

    if "ors_comment_lines_frac" in annotated_df.columns:
        print("\nComment stats:")
        print(f"  Avg comment line fraction: {annotated_df['ors_comment_lines_frac'].mean():.2%}")

    if "software_metrics_cyclomatic_complexity" in annotated_df.columns:
        print("\nComplexity stats:")
        print(f"  Avg cyclomatic complexity: {annotated_df['software_metrics_cyclomatic_complexity'].mean():.2f}")


def _print_sample_results(annotated_df: pd.DataFrame) -> None:
    """Print sample results from the annotated DataFrame."""
    print("\n" + "=" * 60)
    print("SAMPLE RESULTS")
    print("=" * 60)

    display_cols = ["representative_filename"]
    for col in ["language", "basic_num_lines", "ors_comment_lines_frac", "software_metrics_cyclomatic_complexity"]:
        if col in annotated_df.columns:
            display_cols.append(col)

    print(annotated_df[display_cols].to_string(index=False))


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Annotate code files with NeMo Curator")
    parser.add_argument("--input_dir", type=str, help="Directory containing code files")
    parser.add_argument("--output_file", type=str, default="annotated_code.parquet", help="Output parquet file")
    parser.add_argument("--use_sample_data", action="store_true", help="Use sample data instead of input_dir")
    parser.add_argument("--detect_language", action="store_true", default=True, help="Detect programming language")
    parser.add_argument("--basic_stats", action="store_true", default=True, help="Compute basic statistics")
    parser.add_argument("--software_metrics", action="store_true", default=True, help="Compute software metrics")
    parser.add_argument("--opencoder_metrics", action="store_true", default=True, help="Compute OpenCoder metrics")
    parser.add_argument("--tokenize", action="store_true", default=True, help="Tokenize code")
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="github_o200k_base",
        help="Tokenizer to use (github_o200k_base or tiktoken_o200k_base)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # Load data
    if args.use_sample_data:
        print("Using sample data...")
        df = get_sample_data()
    elif args.input_dir:
        print(f"Loading code files from {args.input_dir}...")
        df = load_code_files(args.input_dir)
    else:
        print("Error: Either --input_dir or --use_sample_data must be specified")
        return

    print(f"Loaded {len(df)} code files")

    if len(df) == 0:
        print("No code files found!")
        return

    # Create annotator
    annotator = CodeAnnotator(
        detect_language=args.detect_language,
        basic_stats=args.basic_stats,
        software_metrics=args.software_metrics,
        opencoder_metrics=args.opencoder_metrics,
        tokenize=args.tokenize,
        tokenizer_name=args.tokenizer_name,
    )

    # Apply annotations
    print("Annotating code files...")
    annotated_df = annotator.modify_document(df)

    _print_annotation_summary(annotated_df)

    # Save results
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotated_df.to_parquet(output_path)
    print(f"\nSaved annotated data to {output_path}")

    _print_sample_results(annotated_df)


if __name__ == "__main__":
    main()
