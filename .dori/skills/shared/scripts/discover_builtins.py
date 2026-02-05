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

"""
Discover Built-in NeMo Curator Operations

This script helps the agent discover available built-in operations
BEFORE generating custom code. Run this to find if NeMo Curator
already provides what you need.

Usage:
    # Find filters for a specific purpose
    python discover_builtins.py --query "punctuation" --category filter

    # Find modifiers
    python discover_builtins.py --query "quote" --category modifier

    # List all readers
    python discover_builtins.py --category reader

    # Search everything
    python discover_builtins.py --query "unicode"
"""

import argparse
import json
import sys
from pathlib import Path

# Add skills directory to path
SKILLS_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SKILLS_DIR))


# Built-in operations registry (extracted from NeMo Curator)
BUILTIN_OPERATIONS = {
    "filters": {
        "WordCountFilter": {
            "import": "from nemo_curator.stages.text.filters.heuristic_filter import WordCountFilter",
            "description": "Filter documents by word count (min/max)",
            "params": ["min_words=50", "max_words=100000", "lang='en'"],
            "gpu": False,
            "use_case": "Remove too-short or too-long documents",
        },
        "RepeatedLinesFilter": {
            "import": "from nemo_curator.stages.text.filters.heuristic_filter import RepeatedLinesFilter",
            "description": "Filter documents with too many repeated lines",
            "params": ["max_repeated_line_fraction=0.7"],
            "gpu": False,
            "use_case": "Remove documents with excessive line repetition",
        },
        "RepeatedParagraphsFilter": {
            "import": "from nemo_curator.stages.text.filters.heuristic_filter import RepeatedParagraphsFilter",
            "description": "Filter documents with too many repeated paragraphs",
            "params": ["max_repeated_paragraphs_ratio=0.7"],
            "gpu": False,
            "use_case": "Remove documents with paragraph repetition",
        },
        "PunctuationFilter": {
            "import": "from nemo_curator.stages.text.filters.heuristic_filter import PunctuationFilter",
            "description": "Filter documents where sentences don't end with punctuation",
            "params": ["max_num_sentences_without_endmark_ratio=0.85"],
            "gpu": False,
            "use_case": "Remove incomplete or malformed text",
        },
        "UrlsFilter": {
            "import": "from nemo_curator.stages.text.filters.heuristic_filter import UrlsFilter",
            "description": "Filter documents with too many URLs",
            "params": ["max_url_ratio=0.25"],
            "gpu": False,
            "use_case": "Remove URL-heavy spam/link content",
        },
        "BulletsFilter": {
            "import": "from nemo_curator.stages.text.filters.heuristic_filter import BulletsFilter",
            "description": "Filter documents with too many bullet points",
            "params": ["max_bullet_ratio=0.9"],
            "gpu": False,
            "use_case": "Remove list-heavy content",
        },
        "WhiteSpaceFilter": {
            "import": "from nemo_curator.stages.text.filters.heuristic_filter import WhiteSpaceFilter",
            "description": "Filter documents with too much whitespace",
            "params": ["max_whitespace_ratio=0.25"],
            "gpu": False,
            "use_case": "Remove poorly formatted text",
        },
        "BoilerPlateStringFilter": {
            "import": "from nemo_curator.stages.text.filters.heuristic_filter import BoilerPlateStringFilter",
            "description": "Filter documents with boilerplate text (terms of use, etc.)",
            "params": ["max_boilerplate_string_ratio=0.4"],
            "gpu": False,
            "use_case": "Remove web page boilerplate",
        },
        "MeanWordLengthFilter": {
            "import": "from nemo_curator.stages.text.filters.heuristic_filter import MeanWordLengthFilter",
            "description": "Filter by average word length",
            "params": ["min_mean_word_length=3", "max_mean_word_length=10"],
            "gpu": False,
            "use_case": "Remove gibberish or non-prose content",
        },
        "EllipsisFilter": {
            "import": "from nemo_curator.stages.text.filters.heuristic_filter import EllipsisFilter",
            "description": "Filter documents with too many ellipses",
            "params": ["max_num_lines_ending_with_ellipsis_ratio=0.3"],
            "gpu": False,
            "use_case": "Remove truncated or clickbait content",
        },
        "CommonEnglishFilter": {
            "import": "from nemo_curator.stages.text.filters.heuristic_filter import CommonEnglishFilter",
            "description": "Filter non-English documents using common word lists",
            "params": ["min_common_english_ratio=0.2"],
            "gpu": False,
            "use_case": "Basic language filtering",
        },
        "FastTextLangId": {
            "import": "from nemo_curator.stages.text.filters.fasttext_langid import FastTextLangId",
            "description": "Language identification using FastText",
            "params": ["languages=['en']", "threshold=0.5"],
            "gpu": False,
            "use_case": "Accurate language filtering",
        },
    },
    "modifiers": {
        "UnicodeReformatter": {
            "import": "from nemo_curator.stages.text.modifiers.unicode_reformatter import UnicodeReformatter",
            "description": "Fix Unicode issues, normalize text, uncurl quotes",
            "params": ["uncurl_quotes=True", "fix_encoding=True", "normalization='NFC'"],
            "gpu": False,
            "use_case": "Normalize curly quotes to straight, fix mojibake, Unicode normalization",
        },
        "UrlRemover": {
            "import": "from nemo_curator.stages.text.modifiers.url_remover import UrlRemover",
            "description": "Remove URLs from text",
            "params": [],
            "gpu": False,
            "use_case": "Clean URLs from documents",
        },
        "NewlineNormalizer": {
            "import": "from nemo_curator.stages.text.modifiers.newline_normalizer import NewlineNormalizer",
            "description": "Normalize different newline formats",
            "params": [],
            "gpu": False,
            "use_case": "Standardize line endings",
        },
        "MarkdownRemover": {
            "import": "from nemo_curator.stages.text.modifiers.markdown_remover import MarkdownRemover",
            "description": "Remove markdown formatting",
            "params": [],
            "gpu": False,
            "use_case": "Convert markdown to plain text",
        },
        "QuotationRemover": {
            "import": "from nemo_curator.stages.text.modifiers.quotation_remover import QuotationRemover",
            "description": "Remove quotation marks",
            "params": [],
            "gpu": False,
            "use_case": "Strip quotes from text",
        },
        "BoilerPlateStringModifier": {
            "import": "from nemo_curator.stages.text.modifiers.c4 import BoilerPlateStringModifier",
            "description": "Remove boilerplate text paragraphs",
            "params": [],
            "gpu": False,
            "use_case": "Strip web page boilerplate",
        },
    },
    "readers": {
        "JsonlReader": {
            "import": "from nemo_curator.stages.text.io.reader import JsonlReader",
            "description": "Read JSONL files into DocumentBatch",
            "params": ["input_path", "text_column='text'"],
            "gpu": False,
            "use_case": "Read JSONL datasets",
        },
        "ParquetReader": {
            "import": "from nemo_curator.stages.text.io.reader import ParquetReader",
            "description": "Read Parquet files into DocumentBatch",
            "params": ["input_path", "text_column='text'"],
            "gpu": False,
            "use_case": "Read Parquet datasets",
        },
    },
    "writers": {
        "JsonlWriter": {
            "import": "from nemo_curator.stages.text.io.writer import JsonlWriter",
            "description": "Write DocumentBatch to JSONL files",
            "params": ["output_path"],
            "gpu": False,
            "use_case": "Save curated data as JSONL",
        },
        "ParquetWriter": {
            "import": "from nemo_curator.stages.text.io.writer import ParquetWriter",
            "description": "Write DocumentBatch to Parquet files",
            "params": ["output_path"],
            "gpu": False,
            "use_case": "Save curated data as Parquet",
        },
    },
    "classifiers": {
        "QualityClassifier": {
            "import": "from nemo_curator.stages.text.classifiers.quality import QualityClassifier",
            "description": "Classify document quality (high/medium/low)",
            "params": [],
            "gpu": True,
            "gpu_memory_gb": 4,
            "use_case": "ML-based quality filtering",
        },
        "DomainClassifier": {
            "import": "from nemo_curator.stages.text.classifiers.domain import DomainClassifier",
            "description": "Classify document domain/topic",
            "params": [],
            "gpu": True,
            "gpu_memory_gb": 4,
            "use_case": "Categorize documents by domain",
        },
        "FineWebEduClassifier": {
            "import": "from nemo_curator.stages.text.classifiers.fineweb_edu import FineWebEduClassifier",
            "description": "Classify educational content quality",
            "params": [],
            "gpu": True,
            "gpu_memory_gb": 4,
            "use_case": "Filter for educational content",
        },
        "AegisClassifier": {
            "import": "from nemo_curator.stages.text.classifiers.aegis import AegisClassifier",
            "description": "NVIDIA AEGIS safety classifier",
            "params": ["hf_token"],
            "gpu": True,
            "gpu_memory_gb": 16,
            "credentials": ["HF_TOKEN"],
            "use_case": "Content safety filtering",
        },
    },
}


def search_operations(query: str | None, category: str | None) -> dict:
    """Search for operations matching query and/or category."""
    results = {}
    
    categories_to_search = [category] if category else BUILTIN_OPERATIONS.keys()
    
    for cat in categories_to_search:
        if cat not in BUILTIN_OPERATIONS:
            continue
            
        cat_results = {}
        for name, info in BUILTIN_OPERATIONS[cat].items():
            if query is None:
                cat_results[name] = info
            else:
                # Search in name, description, and use_case
                query_lower = query.lower()
                if (query_lower in name.lower() or 
                    query_lower in info.get("description", "").lower() or
                    query_lower in info.get("use_case", "").lower()):
                    cat_results[name] = info
        
        if cat_results:
            results[cat] = cat_results
    
    return results


def format_results(results: dict, json_output: bool) -> str:
    """Format search results for display."""
    if json_output:
        return json.dumps(results, indent=2)
    
    if not results:
        return "No matching operations found."
    
    lines = []
    for category, ops in results.items():
        lines.append(f"\n## {category.upper()}")
        lines.append("")
        for name, info in ops.items():
            lines.append(f"### {name}")
            lines.append(f"**Description**: {info.get('description', 'N/A')}")
            lines.append(f"**Use case**: {info.get('use_case', 'N/A')}")
            lines.append(f"**Import**: `{info.get('import', 'N/A')}`")
            if info.get("params"):
                lines.append(f"**Parameters**: {', '.join(info['params'])}")
            lines.append(f"**GPU**: {'Yes' if info.get('gpu') else 'No'}")
            if info.get("gpu_memory_gb"):
                lines.append(f"**GPU Memory**: {info['gpu_memory_gb']}GB")
            if info.get("credentials"):
                lines.append(f"**Credentials**: {', '.join(info['credentials'])}")
            lines.append("")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Discover built-in NeMo Curator operations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Find operations related to punctuation
  python discover_builtins.py --query "punctuation"
  
  # Find operations related to quotes
  python discover_builtins.py --query "quote"
  
  # List all modifiers
  python discover_builtins.py --category modifier
  
  # List all filters (JSON output)
  python discover_builtins.py --category filter --json
  
  # Find readers
  python discover_builtins.py --category reader
        """
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Search term to find matching operations"
    )
    parser.add_argument(
        "--category", "-c",
        type=str,
        choices=["filter", "modifier", "reader", "writer", "classifier"],
        help="Limit search to a specific category"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON"
    )
    
    args = parser.parse_args()
    
    # Map singular to plural for category lookup
    category_map = {
        "filter": "filters",
        "modifier": "modifiers", 
        "reader": "readers",
        "writer": "writers",
        "classifier": "classifiers",
    }
    category = category_map.get(args.category) if args.category else None
    
    results = search_operations(args.query, category)
    output = format_results(results, args.json)
    print(output)
    
    # Summary
    if not args.json:
        total = sum(len(ops) for ops in results.values())
        print(f"\n---\nFound {total} matching operation(s)")
        if args.query and total == 0:
            print("\nTip: If no built-in exists, you may need a custom implementation.")
            print("But first, try broader search terms or check the full list with --category")


if __name__ == "__main__":
    main()
