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

"""Analyze JSONL data to help the agent make recommendations.

This script samples JSONL files and computes statistics that help the agent
understand the data and recommend appropriate curation stages.

Usage:
    python analyze_jsonl.py /path/to/data.jsonl
    python analyze_jsonl.py /path/to/data.jsonl --text-field content --sample 50
    python analyze_jsonl.py /path/to/data.jsonl --output json
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path


def count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split())


def count_urls(text: str) -> int:
    """Count URLs in text."""
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    return len(re.findall(url_pattern, text))


def url_ratio(text: str) -> float:
    """Calculate ratio of URL characters to total characters."""
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    urls = re.findall(url_pattern, text)
    url_chars = sum(len(url) for url in urls)
    return url_chars / len(text) if text else 0


def repeated_line_ratio(text: str) -> float:
    """Calculate ratio of repeated lines."""
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if not lines:
        return 0
    unique_lines = set(lines)
    return 1 - (len(unique_lines) / len(lines))


def non_alpha_ratio(text: str) -> float:
    """Calculate ratio of non-alphanumeric characters."""
    if not text:
        return 0
    non_alpha = sum(1 for c in text if not c.isalnum() and not c.isspace())
    return non_alpha / len(text)


def has_punctuation(text: str) -> bool:
    """Check if text has sentence-ending punctuation."""
    return bool(re.search(r'[.!?]', text))


def analyze_document(doc: dict, text_field: str) -> dict:
    """Analyze a single document."""
    text = doc.get(text_field, "")
    if not isinstance(text, str):
        text = str(text) if text else ""
    
    return {
        "word_count": count_words(text),
        "char_count": len(text),
        "url_count": count_urls(text),
        "url_ratio": url_ratio(text),
        "repeated_line_ratio": repeated_line_ratio(text),
        "non_alpha_ratio": non_alpha_ratio(text),
        "has_punctuation": has_punctuation(text),
        "line_count": text.count("\n") + 1 if text else 0,
    }


def analyze_file(
    filepath: str,
    text_field: str = "text",
    sample_size: int = 20,
    random_sample: bool = True,
) -> dict:
    """Analyze a JSONL file and return statistics.
    
    Args:
        filepath: Path to JSONL file
        text_field: Name of the text field in each document
        sample_size: Number of documents to sample
        random_sample: If True, sample randomly; if False, take first N
    
    Returns:
        Dictionary with analysis results
    """
    path = Path(filepath)
    if not path.exists():
        return {"error": f"File not found: {filepath}"}
    
    # Count total documents
    total_docs = 0
    with open(path) as f:
        for _ in f:
            total_docs += 1
    
    # Sample documents
    docs = []
    if random_sample and total_docs > sample_size:
        import random
        sample_indices = set(random.sample(range(total_docs), min(sample_size, total_docs)))
        with open(path) as f:
            for i, line in enumerate(f):
                if i in sample_indices:
                    try:
                        docs.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    else:
        with open(path) as f:
            for i, line in enumerate(f):
                if i >= sample_size:
                    break
                try:
                    docs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    
    if not docs:
        return {"error": "No valid documents found"}
    
    # Check if text field exists
    fields_present = Counter()
    for doc in docs:
        for key in doc.keys():
            fields_present[key] += 1
    
    if text_field not in fields_present:
        return {
            "error": f"Text field '{text_field}' not found",
            "available_fields": dict(fields_present),
            "suggestion": f"Try --text-field with one of: {list(fields_present.keys())}",
        }
    
    # Analyze each document
    analyses = [analyze_document(doc, text_field) for doc in docs]
    
    # Compute aggregate statistics
    word_counts = [a["word_count"] for a in analyses]
    char_counts = [a["char_count"] for a in analyses]
    url_ratios = [a["url_ratio"] for a in analyses]
    repeated_ratios = [a["repeated_line_ratio"] for a in analyses]
    non_alpha_ratios = [a["non_alpha_ratio"] for a in analyses]
    
    def percentile(data, p):
        """Compute percentile."""
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * p / 100
        f = int(k)
        c = f + 1 if f + 1 < len(sorted_data) else f
        return sorted_data[f] + (sorted_data[c] - sorted_data[f]) * (k - f)
    
    def mean(data):
        return sum(data) / len(data) if data else 0
    
    # Identify potential issues
    issues = []
    
    short_docs = sum(1 for w in word_counts if w < 50)
    if short_docs > 0:
        issues.append({
            "type": "short_documents",
            "count": short_docs,
            "percentage": round(100 * short_docs / len(word_counts), 1),
            "recommendation": "WordCountFilter(min_words=50)",
        })
    
    long_docs = sum(1 for w in word_counts if w > 100000)
    if long_docs > 0:
        issues.append({
            "type": "very_long_documents",
            "count": long_docs,
            "percentage": round(100 * long_docs / len(word_counts), 1),
            "recommendation": "WordCountFilter(max_words=100000)",
        })
    
    high_url_docs = sum(1 for r in url_ratios if r > 0.2)
    if high_url_docs > 0:
        issues.append({
            "type": "high_url_content",
            "count": high_url_docs,
            "percentage": round(100 * high_url_docs / len(url_ratios), 1),
            "recommendation": "UrlsFilter(max_url_to_text_ratio=0.2)",
        })
    
    high_repetition = sum(1 for r in repeated_ratios if r > 0.5)
    if high_repetition > 0:
        issues.append({
            "type": "high_repetition",
            "count": high_repetition,
            "percentage": round(100 * high_repetition / len(repeated_ratios), 1),
            "recommendation": "RepeatedLinesFilter(max_repeated_line_fraction=0.5)",
        })
    
    high_non_alpha = sum(1 for r in non_alpha_ratios if r > 0.25)
    if high_non_alpha > 0:
        issues.append({
            "type": "high_non_alphanumeric",
            "count": high_non_alpha,
            "percentage": round(100 * high_non_alpha / len(non_alpha_ratios), 1),
            "recommendation": "NonAlphaNumericFilter(max_non_alpha_numeric_to_text_ratio=0.25)",
        })
    
    no_punctuation = sum(1 for a in analyses if not a["has_punctuation"])
    if no_punctuation > len(analyses) * 0.1:
        issues.append({
            "type": "missing_punctuation",
            "count": no_punctuation,
            "percentage": round(100 * no_punctuation / len(analyses), 1),
            "recommendation": "PunctuationFilter(max_num_sentences_without_endmark_ratio=0.85)",
        })
    
    return {
        "file": str(filepath),
        "total_documents": total_docs,
        "sampled_documents": len(docs),
        "text_field": text_field,
        "fields_present": dict(fields_present),
        "statistics": {
            "word_count": {
                "min": min(word_counts),
                "max": max(word_counts),
                "mean": round(mean(word_counts), 1),
                "median": round(percentile(word_counts, 50), 1),
                "p10": round(percentile(word_counts, 10), 1),
                "p90": round(percentile(word_counts, 90), 1),
            },
            "char_count": {
                "min": min(char_counts),
                "max": max(char_counts),
                "mean": round(mean(char_counts), 1),
            },
            "url_ratio": {
                "mean": round(mean(url_ratios), 3),
                "max": round(max(url_ratios), 3),
            },
            "repeated_line_ratio": {
                "mean": round(mean(repeated_ratios), 3),
                "max": round(max(repeated_ratios), 3),
            },
            "non_alpha_ratio": {
                "mean": round(mean(non_alpha_ratios), 3),
                "max": round(max(non_alpha_ratios), 3),
            },
        },
        "issues_detected": issues,
        "recommendations": [issue["recommendation"] for issue in issues],
    }


def format_human_readable(results: dict) -> str:
    """Format results for human reading."""
    if "error" in results:
        output = f"Error: {results['error']}\n"
        if "available_fields" in results:
            output += f"Available fields: {list(results['available_fields'].keys())}\n"
        if "suggestion" in results:
            output += f"Suggestion: {results['suggestion']}\n"
        return output
    
    lines = [
        "=" * 60,
        "JSONL Data Analysis",
        "=" * 60,
        "",
        f"File: {results['file']}",
        f"Total documents: {results['total_documents']:,}",
        f"Sampled: {results['sampled_documents']}",
        f"Text field: {results['text_field']}",
        "",
        "Document Statistics:",
        f"  Word count: {results['statistics']['word_count']['min']} - {results['statistics']['word_count']['max']}",
        f"    Mean: {results['statistics']['word_count']['mean']}, Median: {results['statistics']['word_count']['median']}",
        f"    P10: {results['statistics']['word_count']['p10']}, P90: {results['statistics']['word_count']['p90']}",
        "",
        f"  URL ratio: mean={results['statistics']['url_ratio']['mean']}, max={results['statistics']['url_ratio']['max']}",
        f"  Repetition ratio: mean={results['statistics']['repeated_line_ratio']['mean']}, max={results['statistics']['repeated_line_ratio']['max']}",
        f"  Non-alpha ratio: mean={results['statistics']['non_alpha_ratio']['mean']}, max={results['statistics']['non_alpha_ratio']['max']}",
    ]
    
    if results["issues_detected"]:
        lines.extend([
            "",
            "Issues Detected:",
        ])
        for issue in results["issues_detected"]:
            lines.append(f"  - {issue['type']}: {issue['count']} docs ({issue['percentage']}%)")
        
        lines.extend([
            "",
            "Recommended Filters:",
        ])
        for rec in results["recommendations"]:
            lines.append(f"  - {rec}")
    else:
        lines.extend([
            "",
            "No significant issues detected.",
        ])
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("filepath", help="Path to JSONL file")
    parser.add_argument(
        "--text-field", "-t",
        default="text",
        help="Name of text field (default: text)",
    )
    parser.add_argument(
        "--sample", "-n",
        type=int,
        default=20,
        help="Number of documents to sample (default: 20)",
    )
    parser.add_argument(
        "--output", "-o",
        choices=["human", "json"],
        default="human",
        help="Output format (default: human)",
    )
    parser.add_argument(
        "--no-random",
        action="store_true",
        help="Don't randomize sample, take first N documents",
    )
    
    args = parser.parse_args()
    
    results = analyze_file(
        args.filepath,
        text_field=args.text_field,
        sample_size=args.sample,
        random_sample=not args.no_random,
    )
    
    if args.output == "json":
        print(json.dumps(results, indent=2))
    else:
        print(format_human_readable(results))
    
    # Exit with error code if there was an error
    if "error" in results:
        sys.exit(1)


if __name__ == "__main__":
    main()
