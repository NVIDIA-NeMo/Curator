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

"""Analyze input data and recommend NeMo Curator pipeline stages.

This script analyzes a sample of input data and provides data-driven
recommendations for which filters and classifiers to apply.

The recommendations are based on computed statistics, not hardcoded rules.
Each recommendation includes the reasoning based on the data.

Usage:
    python analyze_data.py --input data.jsonl --sample 100
    python analyze_data.py --input data.parquet --text-column content --json
    python analyze_data.py --input data.jsonl --sample 200 --threshold-config thresholds.yaml
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# Configurable thresholds - can be overridden via config file
DEFAULT_THRESHOLDS = {
    # Word count thresholds
    "short_doc_min_words": 50,
    "short_doc_alert_pct": 15,  # Alert if >15% docs are short
    "long_doc_max_words": 10000,
    "long_doc_alert_pct": 5,
    
    # URL/link density
    "high_url_ratio_threshold": 0.2,
    "high_url_alert_pct": 10,
    
    # Repeated content
    "high_repetition_threshold": 0.3,  # >30% repeated n-grams
    "high_repetition_alert_pct": 10,
    
    # Language (for non-English filtering)
    "non_target_lang_alert_pct": 10,
    
    # Bullets/boilerplate
    "high_bullet_ratio_threshold": 0.3,
    "high_bullet_alert_pct": 10,
}


def read_sample(
    input_path: Path,
    sample_size: int,
    text_column: str = "text",
) -> tuple[list[str], int]:
    """Read a sample of documents from input file.
    
    Returns (sample_texts, total_count_estimate).
    """
    suffix = input_path.suffix.lower()
    texts = []
    total_estimate = 0
    
    if suffix == ".jsonl":
        # Stream JSONL
        with open(input_path) as f:
            for i, line in enumerate(f):
                if line.strip():
                    try:
                        doc = json.loads(line)
                        text = doc.get(text_column, "")
                        if isinstance(text, str):
                            texts.append(text)
                    except json.JSONDecodeError:
                        continue
                
                if len(texts) >= sample_size:
                    # Estimate total by continuing to count lines
                    for j, _ in enumerate(f, start=i + 1):
                        pass
                    total_estimate = j + 1
                    break
        
        if total_estimate == 0:
            total_estimate = len(texts)
    
    elif suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
            
            table = pq.read_table(input_path, columns=[text_column])
            total_estimate = len(table)
            
            # Sample
            if len(table) > sample_size:
                import random
                indices = random.sample(range(len(table)), sample_size)
                for idx in indices:
                    texts.append(str(table[text_column][idx].as_py()))
            else:
                texts = [str(v.as_py()) for v in table[text_column]]
        except ImportError:
            print("Warning: pyarrow not installed, cannot read parquet files", file=sys.stderr)
            return [], 0
    
    elif suffix == ".json":
        # JSON array
        with open(input_path) as f:
            data = json.load(f)
        
        if isinstance(data, list):
            total_estimate = len(data)
            for doc in data[:sample_size]:
                if isinstance(doc, dict):
                    text = doc.get(text_column, "")
                elif isinstance(doc, str):
                    text = doc
                else:
                    continue
                if isinstance(text, str):
                    texts.append(text)
    
    else:
        # Try as plain text (one doc per file)
        with open(input_path) as f:
            texts.append(f.read())
        total_estimate = 1
    
    return texts, total_estimate


def compute_word_count(text: str) -> int:
    """Count words in text."""
    return len(text.split())


def compute_url_ratio(text: str) -> float:
    """Compute ratio of URL characters to total text."""
    url_pattern = r'https?://[^\s<>"\'{}|\\^`\[\]]+'
    urls = re.findall(url_pattern, text)
    url_chars = sum(len(url) for url in urls)
    return url_chars / len(text) if text else 0.0


def compute_bullet_ratio(text: str) -> float:
    """Compute ratio of bullet-point lines to total lines."""
    lines = text.strip().split("\n")
    if not lines:
        return 0.0
    
    bullet_patterns = [r"^\s*[-*•]\s", r"^\s*\d+[.)]\s"]
    bullet_count = 0
    for line in lines:
        for pattern in bullet_patterns:
            if re.match(pattern, line):
                bullet_count += 1
                break
    
    return bullet_count / len(lines)


def compute_repetition_score(text: str, n: int = 3) -> float:
    """Compute n-gram repetition score (0-1, higher = more repetition)."""
    words = text.lower().split()
    if len(words) < n:
        return 0.0
    
    ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
    if not ngrams:
        return 0.0
    
    counts = Counter(ngrams)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    return repeated / len(ngrams)


def detect_language_simple(text: str) -> str:
    """Simple language detection based on common words.
    
    For accurate detection, use FastText or langdetect library.
    """
    # Common English words
    english_words = {"the", "is", "at", "which", "on", "a", "an", "and", "or", "but", "in", "to", "for"}
    
    words = set(text.lower().split())
    english_overlap = len(words & english_words)
    
    # Very rough heuristic
    if english_overlap >= 3:
        return "en"
    return "unknown"


def analyze_sample(
    texts: list[str],
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    """Analyze a sample of texts and compute statistics."""
    if not texts:
        return {"error": "No texts to analyze"}
    
    stats = {
        "sample_size": len(texts),
        "word_counts": [],
        "url_ratios": [],
        "bullet_ratios": [],
        "repetition_scores": [],
        "languages": [],
    }
    
    for text in texts:
        stats["word_counts"].append(compute_word_count(text))
        stats["url_ratios"].append(compute_url_ratio(text))
        stats["bullet_ratios"].append(compute_bullet_ratio(text))
        stats["repetition_scores"].append(compute_repetition_score(text))
        stats["languages"].append(detect_language_simple(text))
    
    # Compute summary statistics
    word_counts = stats["word_counts"]
    url_ratios = stats["url_ratios"]
    bullet_ratios = stats["bullet_ratios"]
    repetition_scores = stats["repetition_scores"]
    languages = stats["languages"]
    
    n = len(texts)
    
    analysis = {
        "sample_size": n,
        "word_count": {
            "min": min(word_counts),
            "max": max(word_counts),
            "avg": sum(word_counts) / n,
            "median": sorted(word_counts)[n // 2],
            "short_pct": 100 * sum(1 for wc in word_counts if wc < thresholds["short_doc_min_words"]) / n,
            "long_pct": 100 * sum(1 for wc in word_counts if wc > thresholds["long_doc_max_words"]) / n,
        },
        "url_density": {
            "avg": sum(url_ratios) / n,
            "high_pct": 100 * sum(1 for ur in url_ratios if ur > thresholds["high_url_ratio_threshold"]) / n,
        },
        "bullet_ratio": {
            "avg": sum(bullet_ratios) / n,
            "high_pct": 100 * sum(1 for br in bullet_ratios if br > thresholds["high_bullet_ratio_threshold"]) / n,
        },
        "repetition": {
            "avg": sum(repetition_scores) / n,
            "high_pct": 100 * sum(1 for rs in repetition_scores if rs > thresholds["high_repetition_threshold"]) / n,
        },
        "language_distribution": dict(Counter(languages)),
    }
    
    return analysis


def generate_recommendations(
    analysis: dict[str, Any],
    thresholds: dict[str, Any],
) -> list[dict[str, Any]]:
    """Generate pipeline recommendations based on analysis."""
    recommendations = []
    
    # Word count filter
    short_pct = analysis["word_count"]["short_pct"]
    if short_pct > thresholds["short_doc_alert_pct"]:
        recommendations.append({
            "stage": "WordCountFilter",
            "priority": 1,  # Cheap CPU filter - run early
            "reason": f"{short_pct:.1f}% of documents have fewer than {thresholds['short_doc_min_words']} words",
            "params": {
                "min_words": thresholds["short_doc_min_words"],
            },
            "expected_removal_pct": short_pct,
            "import": "from nemo_curator.stages.text.filters.heuristic_filter import WordCountFilter",
        })
    
    long_pct = analysis["word_count"]["long_pct"]
    if long_pct > thresholds["long_doc_alert_pct"]:
        recommendations.append({
            "stage": "WordCountFilter",
            "priority": 1,
            "reason": f"{long_pct:.1f}% of documents have more than {thresholds['long_doc_max_words']} words",
            "params": {
                "max_words": thresholds["long_doc_max_words"],
            },
            "expected_removal_pct": long_pct,
            "import": "from nemo_curator.stages.text.filters.heuristic_filter import WordCountFilter",
        })
    
    # URL filter
    url_pct = analysis["url_density"]["high_pct"]
    if url_pct > thresholds["high_url_alert_pct"]:
        recommendations.append({
            "stage": "UrlsFilter",
            "priority": 1,
            "reason": f"{url_pct:.1f}% of documents have high URL density (>{thresholds['high_url_ratio_threshold']*100:.0f}%)",
            "params": {
                "max_url_to_text_ratio": thresholds["high_url_ratio_threshold"],
            },
            "expected_removal_pct": url_pct,
            "import": "from nemo_curator.stages.text.filters.heuristic_filter import UrlsFilter",
        })
    
    # Repetition filter
    rep_pct = analysis["repetition"]["high_pct"]
    if rep_pct > thresholds["high_repetition_alert_pct"]:
        recommendations.append({
            "stage": "RepeatedParagraphsFilter",
            "priority": 2,
            "reason": f"{rep_pct:.1f}% of documents have high repetition scores",
            "params": {},
            "expected_removal_pct": rep_pct,
            "import": "from nemo_curator.stages.text.filters.heuristic_filter import RepeatedParagraphsFilter",
        })
    
    # Bullet/list filter
    bullet_pct = analysis["bullet_ratio"]["high_pct"]
    if bullet_pct > thresholds["high_bullet_alert_pct"]:
        recommendations.append({
            "stage": "BulletsFilter",
            "priority": 2,
            "reason": f"{bullet_pct:.1f}% of documents are dominated by bullet points",
            "params": {
                "max_bullet_lines_ratio": thresholds["high_bullet_ratio_threshold"],
            },
            "expected_removal_pct": bullet_pct,
            "import": "from nemo_curator.stages.text.filters.heuristic_filter import BulletsFilter",
        })
    
    # Language filter
    lang_dist = analysis["language_distribution"]
    non_english_pct = 100 - lang_dist.get("en", 0) * 100 / analysis["sample_size"]
    if non_english_pct > thresholds["non_target_lang_alert_pct"] and lang_dist.get("unknown", 0) < analysis["sample_size"] * 0.5:
        recommendations.append({
            "stage": "FastTextLangId",
            "priority": 2,
            "reason": f"~{non_english_pct:.0f}% of documents may not be English",
            "params": {
                "keep_langs": ["en"],
            },
            "expected_removal_pct": non_english_pct,
            "import": "from nemo_curator.stages.text.filters.fasttext_filter import FastTextLangId",
            "note": "Consider using proper language detection for accurate filtering",
        })
    
    # Quality classifier (always suggest if GPU available)
    if analysis["sample_size"] >= 10:
        recommendations.append({
            "stage": "QualityClassifier",
            "priority": 3,  # ML - run after cheap filters
            "reason": "ML-based quality scoring for final quality gate",
            "params": {},
            "expected_removal_pct": "varies (typically 30-60%)",
            "import": "from nemo_curator.stages.text.classifiers.quality import QualityClassifier",
            "note": "Requires GPU (~4GB). Run after heuristic filters to reduce cost.",
        })
    
    # Sort by priority
    recommendations.sort(key=lambda x: x["priority"])
    
    return recommendations


def estimate_yield(
    total_docs: int,
    recommendations: list[dict[str, Any]],
) -> dict[str, Any]:
    """Estimate pipeline yield based on recommendations."""
    remaining = total_docs
    stages = []
    
    for rec in recommendations:
        removal_pct = rec.get("expected_removal_pct", 0)
        if isinstance(removal_pct, str):
            # Variable removal (e.g., QualityClassifier)
            removal_pct = 40  # Conservative estimate
        
        removed = int(remaining * removal_pct / 100)
        remaining = remaining - removed
        
        stages.append({
            "stage": rec["stage"],
            "input": remaining + removed,
            "output": remaining,
            "removed": removed,
        })
    
    return {
        "initial_docs": total_docs,
        "final_docs_estimate": remaining,
        "yield_pct": 100 * remaining / total_docs if total_docs > 0 else 0,
        "per_stage": stages,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Analyze input data and recommend NeMo Curator pipeline stages"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to input data file (JSONL, Parquet, JSON)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=100,
        help="Number of documents to sample (default: 100)",
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
        "--threshold-config",
        type=Path,
        default=None,
        help="Optional YAML file with custom thresholds",
    )
    
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    
    # Load thresholds
    thresholds = DEFAULT_THRESHOLDS.copy()
    if args.threshold_config and args.threshold_config.exists():
        try:
            import yaml
            with open(args.threshold_config) as f:
                custom = yaml.safe_load(f)
            thresholds.update(custom)
        except ImportError:
            print("Warning: PyYAML not installed, using default thresholds", file=sys.stderr)
    
    # Read sample
    texts, total_estimate = read_sample(args.input, args.sample, args.text_column)
    
    if not texts:
        print(f"Error: Could not read any documents from {args.input}", file=sys.stderr)
        sys.exit(1)
    
    # Analyze
    analysis = analyze_sample(texts, thresholds)
    analysis["total_docs_estimate"] = total_estimate
    
    # Generate recommendations
    recommendations = generate_recommendations(analysis, thresholds)
    
    # Estimate yield
    yield_estimate = estimate_yield(total_estimate, recommendations)
    
    # Build result
    result = {
        "input_file": str(args.input),
        "sample_size": len(texts),
        "total_docs_estimate": total_estimate,
        "analysis": analysis,
        "recommendations": recommendations,
        "yield_estimate": yield_estimate,
    }
    
    # Output
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        # Human-readable output
        print(f"\n📊 Data Analysis: {args.input.name}")
        print("=" * 60)
        
        print(f"\n📁 Dataset:")
        print(f"   Sampled: {len(texts)} docs")
        print(f"   Total (est): {total_estimate:,} docs")
        
        print(f"\n📈 Statistics:")
        wc = analysis["word_count"]
        print(f"   Word count: {wc['min']} - {wc['max']} (avg: {wc['avg']:.0f}, median: {wc['median']})")
        print(f"   Short docs (<{thresholds['short_doc_min_words']} words): {wc['short_pct']:.1f}%")
        print(f"   URL-heavy docs: {analysis['url_density']['high_pct']:.1f}%")
        print(f"   High repetition: {analysis['repetition']['high_pct']:.1f}%")
        print(f"   Languages: {analysis['language_distribution']}")
        
        print(f"\n💡 Recommended Pipeline ({len(recommendations)} stages):")
        for i, rec in enumerate(recommendations, 1):
            print(f"\n   {i}. {rec['stage']}")
            print(f"      Reason: {rec['reason']}")
            if rec.get("params"):
                params_str = ", ".join(f"{k}={v}" for k, v in rec["params"].items())
                print(f"      Params: {params_str}")
            if rec.get("note"):
                print(f"      Note: {rec['note']}")
        
        print(f"\n📉 Estimated Yield:")
        print(f"   Input: {total_estimate:,} docs")
        print(f"   Output: ~{yield_estimate['final_docs_estimate']:,} docs ({yield_estimate['yield_pct']:.0f}%)")
        
        print("\n" + "=" * 60)
        print("💡 Tip: Run with --json for machine-readable output")
        print()


if __name__ == "__main__":
    main()
