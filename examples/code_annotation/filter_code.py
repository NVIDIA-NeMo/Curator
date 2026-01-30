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

"""Example: Filter code files based on quality signals.

This example demonstrates how to:
1. Annotate code files with quality signals
2. Apply filters based on annotation columns
3. Output filtered high-quality code

Usage:
    python filter_code.py --input_file annotated.parquet --output_file filtered.parquet

    # Or with inline annotation + filtering:
    python filter_code.py --use_sample_data --output_file filtered.parquet
"""

import argparse
from pathlib import Path

import pandas as pd

from nemo_curator.stages.code import CodeAnnotator
from nemo_curator.stages.text.filters.code import (
    AlphaPercentFilter,
    Base64ContentFilter,
    CommentFractionFilter,
    HexContentFilter,
    MaxLineLengthFilter,
    TokenCountFilter,
)
from nemo_curator.stages.text.filters.doc_filter import DocumentFilter


def get_sample_data() -> pd.DataFrame:
    """Generate sample code data with varying quality."""
    samples = [
        # Good quality code
        {
            "content": '''def quicksort(arr):
    """Implement quicksort algorithm."""
    if len(arr) <= 1:
        return arr

    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]

    return quicksort(left) + middle + quicksort(right)


def merge_sort(arr):
    """Implement merge sort algorithm."""
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])

    return merge(left, right)


def merge(left, right):
    """Merge two sorted arrays."""
    result = []
    i = j = 0

    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    result.extend(left[i:])
    result.extend(right[j:])
    return result
''',
            "representative_filename": "sorting.py",
        },
        # Too short - will be filtered
        {
            "content": "x = 1\n",
            "representative_filename": "tiny.py",
        },
        # Too many comments - will be filtered
        {
            "content": """# Comment line 1
# Comment line 2
# Comment line 3
# Comment line 4
# Comment line 5
# Comment line 6
# Comment line 7
# Comment line 8
# Comment line 9
x = 1
""",
            "representative_filename": "comments_only.py",
        },
        # Low alpha content (data file) - will be filtered
        {
            "content": """DATA = [
    0x1234567890abcdef,
    0xfedcba0987654321,
    0xaabbccdd11223344,
    0x5566778899aabbcc,
    0xddeeff0011223344,
    0x556677889900aabb,
]

VALUES = [
    123456789012345678,
    234567890123456789,
    345678901234567890,
    456789012345678901,
]
""",
            "representative_filename": "constants.py",
        },
        # Very long lines - will be filtered
        {
            "content": f"data = '{('x' * 2000)}'\nprint(data)\n",
            "representative_filename": "long_line.py",
        },
        # Base64 content - will be filtered
        {
            "content": '''import base64

ENCODED_DATA = """
SGVsbG8gV29ybGQhIFRoaXMgaXMgYSB0ZXN0IG1lc3NhZ2UgdGhhdCBoYXMgYmVlbiBlbmNvZGVk
IGludG8gYmFzZTY0IGZvcm1hdC4gSXQgaXMgdXNlZCB0byB0ZXN0IHRoZSBiYXNlNjQgZmlsdGVy
aW5nIGZ1bmN0aW9uYWxpdHkuIFRoaXMgdGV4dCBzaG91bGQgYmUgbG9uZyBlbm91Z2ggdG8gdHJp
Z2dlciB0aGUgZmlsdGVyLiBBZGRpbmcgbW9yZSB0ZXh0IHRvIG1ha2UgaXQgZXZlbiBsb25nZXIu
"""

def decode():
    return base64.b64decode(ENCODED_DATA)
''',
            "representative_filename": "base64_data.py",
        },
        # Good quality Rust code
        {
            "content": """/// A simple stack implementation
pub struct Stack<T> {
    items: Vec<T>,
}

impl<T> Stack<T> {
    /// Create a new empty stack
    pub fn new() -> Self {
        Stack { items: Vec::new() }
    }

    /// Push an item onto the stack
    pub fn push(&mut self, item: T) {
        self.items.push(item);
    }

    /// Pop an item from the stack
    pub fn pop(&mut self) -> Option<T> {
        self.items.pop()
    }

    /// Check if the stack is empty
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Get the number of items in the stack
    pub fn len(&self) -> usize {
        self.items.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_push_pop() {
        let mut stack = Stack::new();
        stack.push(1);
        stack.push(2);
        assert_eq!(stack.pop(), Some(2));
        assert_eq!(stack.pop(), Some(1));
        assert_eq!(stack.pop(), None);
    }
}
""",
            "representative_filename": "stack.rs",
        },
        # Good quality JavaScript
        {
            "content": """/**
 * A simple event emitter implementation
 */
class EventEmitter {
    constructor() {
        this.events = {};
    }

    /**
     * Subscribe to an event
     * @param {string} event - Event name
     * @param {Function} callback - Callback function
     */
    on(event, callback) {
        if (!this.events[event]) {
            this.events[event] = [];
        }
        this.events[event].push(callback);
    }

    /**
     * Emit an event
     * @param {string} event - Event name
     * @param {...any} args - Arguments to pass to callbacks
     */
    emit(event, ...args) {
        const callbacks = this.events[event];
        if (callbacks) {
            callbacks.forEach(cb => cb(...args));
        }
    }

    /**
     * Remove a listener
     * @param {string} event - Event name
     * @param {Function} callback - Callback to remove
     */
    off(event, callback) {
        const callbacks = this.events[event];
        if (callbacks) {
            this.events[event] = callbacks.filter(cb => cb !== callback);
        }
    }
}

module.exports = EventEmitter;
""",
            "representative_filename": "event_emitter.js",
        },
    ]

    return pd.DataFrame(samples)


def _get_filter_score(name: str, f: DocumentFilter, row: pd.Series, df: pd.DataFrame) -> float:
    """Get the filter score for a row based on the filter type."""
    score_map = {
        "CommentFraction": lambda: f.score_document(row.get("ors_comment_lines_frac")),
        "AlphaPercent": lambda: f.score_document(row.get("basic_alpha_percent")),
        "MaxLineLength": lambda: f.score_document(row.get("basic_max_line_length")),
        "HexContent": lambda: f.score_document(row.get("basic_hex_percent")),
        "Base64Content": lambda: f.score_document(row.get("basic_base64_percent")),
    }

    if name in score_map:
        return score_map[name]()
    if name == "TokenCount":
        token_col = [c for c in df.columns if c.startswith("num_tokens_")]
        return f.score_document(**{token_col[0]: row.get(token_col[0])}) if token_col else -1
    return 0


def _print_filter_results(df: pd.DataFrame, filter_cols: list[str]) -> None:
    """Print filter results summary."""
    print("\n" + "=" * 60)
    print("FILTER RESULTS")
    print("=" * 60)

    for _, row in df.iterrows():
        status = "✅ PASS" if row["passes_all_filters"] else "❌ FILTERED"
        failed_filters = [c.replace("filter_", "") for c in filter_cols if not row[c]]
        reason = f" ({', '.join(failed_filters)})" if failed_filters else ""
        print(f"  {row['representative_filename']}: {status}{reason}")

    passed = df["passes_all_filters"].sum()
    total = len(df)
    print(f"\nPassed: {passed}/{total} ({passed / total:.1%})")


def apply_filters(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Apply quality filters to annotated code DataFrame."""
    filters = [
        ("CommentFraction", CommentFractionFilter(min_comment_ratio=0.01, max_comment_ratio=0.80)),
        ("AlphaPercent", AlphaPercentFilter(min_alpha_percent=0.25)),
        ("MaxLineLength", MaxLineLengthFilter(max_line_length=1000)),
        ("TokenCount", TokenCountFilter(min_tokens=10, max_tokens=100000)),
        ("HexContent", HexContentFilter(max_hex_percent=0.40)),
        ("Base64Content", Base64ContentFilter(max_base64_percent=0.40)),
    ]

    # Create filter result columns
    for name, f in filters:
        col_name = f"filter_{name.lower()}"
        results = []
        for _, row in df.iterrows():
            score = _get_filter_score(name, f, row, df)
            results.append(f.keep_document(score))
        df[col_name] = results

    # Compute combined filter
    filter_cols = [c for c in df.columns if c.startswith("filter_")]
    df["passes_all_filters"] = df[filter_cols].all(axis=1)

    if verbose:
        _print_filter_results(df, filter_cols)

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter code files based on quality signals")
    parser.add_argument("--input_file", type=str, help="Input parquet file with annotated code")
    parser.add_argument("--output_file", type=str, default="filtered_code.parquet", help="Output parquet file")
    parser.add_argument("--use_sample_data", action="store_true", help="Use sample data instead of input file")
    parser.add_argument(
        "--keep_filtered", action="store_true", help="Keep filtered files in output (with filter columns)"
    )

    args = parser.parse_args()

    # Load or generate data
    if args.use_sample_data:
        print("Using sample data...")
        df = get_sample_data()

        # Annotate the sample data
        print("Annotating sample data...")
        annotator = CodeAnnotator(
            detect_language=True,
            basic_stats=True,
            opencoder_metrics=True,
            tokenize=True,
        )
        df = annotator.modify_document(df)
    elif args.input_file:
        print(f"Loading annotated data from {args.input_file}...")
        df = pd.read_parquet(args.input_file)
    else:
        print("Error: Either --input_file or --use_sample_data must be specified")
        return

    print(f"Loaded {len(df)} files")

    # Check required columns
    required_cols = ["ors_comment_lines_frac", "basic_alpha_percent", "basic_max_line_length"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"Warning: Missing columns {missing}. Some filters may not work correctly.")

    # Apply filters
    df = apply_filters(df, verbose=True)

    # Prepare output
    if args.keep_filtered:
        output_df = df
        print(f"\nKeeping all {len(output_df)} files with filter columns")
    else:
        output_df = df[df["passes_all_filters"]].copy()
        print(f"\nKeeping {len(output_df)} files that passed all filters")

    # Save results
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_parquet(output_path)
    print(f"Saved to {output_path}")

    # Show summary of kept files
    if len(output_df) > 0:
        print("\n" + "=" * 60)
        print("KEPT FILES")
        print("=" * 60)

        display_cols = ["representative_filename"]
        if "language" in output_df.columns:
            display_cols.append("language")
        if "basic_num_lines" in output_df.columns:
            display_cols.append("basic_num_lines")

        print(output_df[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
