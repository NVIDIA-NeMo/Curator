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

"""List available NeMo Curator text filters.

Auto-discovers filters from nemo_curator.stages.text.filters and enriches
them with category metadata. Parameters, types, and defaults are introspected
directly from the filter classes.

Examples:
    # List all filters
    python list_filters.py

    # List with descriptions
    python list_filters.py --verbose

    # Search for filters
    python list_filters.py --search word

    # Filter by category
    python list_filters.py --category repetition
"""
import argparse
import inspect
import json
import re
import sys
from dataclasses import dataclass, field
from typing import Any, get_type_hints

# Try to import NeMo Curator - if not available, we'll use static fallback
try:
    from nemo_curator.stages.text.filters import __all__ as FILTER_NAMES
    from nemo_curator.stages.text.filters import DocumentFilter

    NEMO_CURATOR_AVAILABLE = True
except ImportError:
    FILTER_NAMES = []
    DocumentFilter = None  # type: ignore[misc,assignment]
    NEMO_CURATOR_AVAILABLE = False


# Category mappings - the only thing we need to maintain manually.
# Maps filter class name -> category. Filters not listed default to "other".
CATEGORY_MAP: dict[str, str] = {
    # Length
    "WordCountFilter": "length",
    "TokenCountFilter": "length",
    "MeanWordLengthFilter": "length",
    "LongWordFilter": "length",
    # Composition
    "NonAlphaNumericFilter": "composition",
    "SymbolsToWordsFilter": "composition",
    "NumbersFilter": "composition",
    "PunctuationFilter": "composition",
    "WhiteSpaceFilter": "composition",
    "ParenthesesFilter": "composition",
    # Content
    "UrlsFilter": "content",
    "BulletsFilter": "content",
    "EllipsisFilter": "content",
    "PornographicUrlsFilter": "content",
    "SubstringFilter": "content",
    "HistogramFilter": "content",
    "BoilerPlateStringFilter": "content",
    # Repetition
    "RepeatedLinesFilter": "repetition",
    "RepeatedParagraphsFilter": "repetition",
    "RepeatedLinesByCharFilter": "repetition",
    "RepeatedParagraphsByCharFilter": "repetition",
    "RepeatingTopNGramsFilter": "repetition",
    "RepeatingDuplicateNGramsFilter": "repetition",
    # Language
    "CommonEnglishWordsFilter": "language",
    "WordsWithoutAlphabetsFilter": "language",
    "FastTextLangId": "language",
    "FastTextQualityFilter": "language",
    # Code
    "AlphaFilter": "code",
    "GeneralCommentToCodeFilter": "code",
    "HTMLBoilerplateFilter": "code",
    "NumberOfLinesOfCodeFilter": "code",
    "PerExtensionFilter": "code",
    "PythonCommentToCodeFilter": "code",
    "TokenizerFertilityFilter": "code",
    "XMLHeaderFilter": "code",
}


@dataclass
class Filter:
    """Represents a text filter with introspected metadata."""

    name: str
    category: str
    description: str
    parameters: dict[str, dict[str, Any]] = field(default_factory=dict)
    filter_class: type | None = None

    @property
    def default_threshold(self) -> str:
        """Generate default threshold string from parameters."""
        defaults = []
        for param_name, param_info in self.parameters.items():
            if param_info.get("default") is not None:
                defaults.append(f"{param_info['default']}")
        return ", ".join(defaults) if defaults else "-"


def _get_type_name(annotation: Any) -> str:
    """Convert a type annotation to a readable string."""
    if annotation is inspect.Parameter.empty:
        return "Any"
    if hasattr(annotation, "__name__"):
        return annotation.__name__
    return str(annotation).replace("typing.", "")


def _extract_first_sentence(docstring: str | None) -> str:
    """Extract first sentence from docstring as description."""
    if not docstring:
        return "No description available"
    # Clean up whitespace
    lines = docstring.strip().split("\n")
    first_para = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            break
        first_para.append(stripped)
    text = " ".join(first_para)
    # Get first sentence
    match = re.match(r"^(.+?[.!?])\s", text + " ")
    if match:
        return match.group(1)
    return text[:200] + "..." if len(text) > 200 else text


def _introspect_filter(filter_class: type) -> Filter:
    """Introspect a filter class to extract its metadata."""
    name = filter_class.__name__
    category = CATEGORY_MAP.get(name, "other")
    description = _extract_first_sentence(filter_class.__doc__)

    # Get __init__ signature
    sig = inspect.signature(filter_class.__init__)
    parameters: dict[str, dict[str, Any]] = {}

    # Try to get type hints
    try:
        hints = get_type_hints(filter_class.__init__)
    except Exception:
        hints = {}

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "args", "kwargs"):
            continue

        param_type = hints.get(param_name, param.annotation)
        param_info: dict[str, Any] = {
            "type": _get_type_name(param_type),
        }

        if param.default is not inspect.Parameter.empty:
            param_info["default"] = param.default

        parameters[param_name] = param_info

    return Filter(
        name=name,
        category=category,
        description=description,
        parameters=parameters,
        filter_class=filter_class,
    )


def _get_static_filters() -> list[Filter]:
    """Return static filter list when NeMo Curator is not installed.

    This is a fallback for environments where nemo_curator is not available.
    Run with --check-coverage in an environment with nemo_curator to verify
    this list is up-to-date.
    """
    # Static definitions - update by running with nemo_curator installed
    static = [
        ("WordCountFilter", "length", "Filter documents by word count range."),
        ("TokenCountFilter", "length", "Filter documents by token count."),
        ("MeanWordLengthFilter", "length", "Filter by average word length range."),
        ("LongWordFilter", "length", "Filter documents containing words longer than threshold."),
        ("NonAlphaNumericFilter", "composition", "Filter by non-alphanumeric character ratio."),
        ("SymbolsToWordsFilter", "composition", "Filter by symbol-to-word ratio."),
        ("NumbersFilter", "composition", "Filter by number character ratio."),
        ("PunctuationFilter", "composition", "Filter by sentence punctuation ratio."),
        ("WhiteSpaceFilter", "composition", "Filter by whitespace character ratio."),
        ("ParenthesesFilter", "composition", "Filter by parentheses character ratio."),
        ("UrlsFilter", "content", "Filter by URL content ratio."),
        ("BulletsFilter", "content", "Filter by bullet-point line ratio."),
        ("EllipsisFilter", "content", "Filter by ellipsis-ending line ratio."),
        ("PornographicUrlsFilter", "content", "Filter documents containing pornographic URLs."),
        ("SubstringFilter", "content", "Filter by substring presence in text."),
        ("HistogramFilter", "content", "Filter by character histogram matching."),
        ("BoilerPlateStringFilter", "content", "Filter by boilerplate content ratio."),
        ("RepeatedLinesFilter", "repetition", "Filter by duplicate line ratio."),
        ("RepeatedParagraphsFilter", "repetition", "Filter by duplicate paragraph ratio."),
        ("RepeatedLinesByCharFilter", "repetition", "Filter by duplicate lines (by character count)."),
        ("RepeatedParagraphsByCharFilter", "repetition", "Filter by duplicate paragraphs (by character count)."),
        ("RepeatingTopNGramsFilter", "repetition", "Filter by top n-gram repetition ratio."),
        ("RepeatingDuplicateNGramsFilter", "repetition", "Filter by duplicate n-gram ratio."),
        ("CommonEnglishWordsFilter", "language", "Filter by common English word presence."),
        ("WordsWithoutAlphabetsFilter", "language", "Filter by words containing alphabetic characters."),
        ("FastTextLangId", "language", "Filter by FastText language identification."),
        ("FastTextQualityFilter", "language", "Filter by FastText quality score."),
        ("AlphaFilter", "code", "Filter by alphabetic character ratio."),
        ("GeneralCommentToCodeFilter", "code", "Filter by comment-to-code ratio."),
        ("HTMLBoilerplateFilter", "code", "Filter HTML by boilerplate detection."),
        ("NumberOfLinesOfCodeFilter", "code", "Filter by lines of code count."),
        ("PerExtensionFilter", "code", "Filter by file extension rules."),
        ("PythonCommentToCodeFilter", "code", "Filter Python by comment-to-code ratio."),
        ("TokenizerFertilityFilter", "code", "Filter by tokenizer fertility ratio."),
        ("XMLHeaderFilter", "code", "Filter files with XML headers."),
    ]
    return [
        Filter(name=name, category=cat, description=desc, parameters={})
        for name, cat, desc in static
    ]


def discover_filters() -> list[Filter]:
    """Discover and introspect all filters from NeMo Curator."""
    if not NEMO_CURATOR_AVAILABLE:
        return _get_static_filters()

    import nemo_curator.stages.text.filters as filters_module

    discovered = []
    for name in FILTER_NAMES:
        if name == "import_filter":
            continue
        cls = getattr(filters_module, name, None)
        if cls is None:
            continue
        # Only include actual filter classes (subclasses of DocumentFilter or classes with score_document)
        if isinstance(cls, type) and (
            issubclass(cls, DocumentFilter) or hasattr(cls, "score_document")
        ):
            discovered.append(_introspect_filter(cls))

    return discovered


# Cache discovered filters
_FILTERS_CACHE: list[Filter] | None = None


def get_filters() -> list[Filter]:
    """Get all filters, using cache if available."""
    global _FILTERS_CACHE
    if _FILTERS_CACHE is None:
        _FILTERS_CACHE = discover_filters()
    return _FILTERS_CACHE


def list_filters(
    category: str | None = None,
    search: str | None = None,
    verbose: bool = False,
) -> list[Filter]:
    """List filters matching criteria."""
    results = get_filters().copy()

    if category:
        results = [f for f in results if f.category == category.lower()]

    if search:
        pattern = re.compile(search, re.IGNORECASE)
        results = [f for f in results if pattern.search(f.name) or pattern.search(f.description)]

    return results


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--category",
        choices=["length", "composition", "content", "repetition", "language", "code", "other"],
        help="Filter by category",
    )
    parser.add_argument(
        "--search",
        help="Search by filter name or description",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed information including parameters",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    parser.add_argument(
        "--check-coverage",
        action="store_true",
        help="Check which filters are missing category mappings",
    )

    args = parser.parse_args()

    filters = list_filters(
        category=args.category,
        search=args.search,
        verbose=args.verbose,
    )

    if args.check_coverage:
        if not NEMO_CURATOR_AVAILABLE:
            print("❌ nemo_curator is not installed. Cannot check coverage.")
            print("   Install with: pip install nemo-curator")
            sys.exit(1)
        # Report filters that need category mapping
        unmapped = [f for f in filters if f.category == "other"]
        if unmapped:
            print("⚠️  Filters missing category mapping:")
            for f in unmapped:
                print(f"  - {f.name}")
            print(f"\nAdd these to CATEGORY_MAP in {__file__}")
        else:
            print("✅ All filters have category mappings")
        return

    # Show warning if using static fallback
    if not NEMO_CURATOR_AVAILABLE and not args.json:
        print("⚠️  nemo_curator not installed - using static filter list")
        print("   Install nemo-curator for live introspection with full parameter info\n")

    if args.json:
        output = [
            {
                "name": f.name,
                "category": f.category,
                "description": f.description,
                "parameters": f.parameters,
                "default_threshold": f.default_threshold,
            }
            for f in filters
        ]
        print(json.dumps(output, indent=2))
    else:
        # Group by category
        categories: dict[str, list[Filter]] = {}
        for f in filters:
            if f.category not in categories:
                categories[f.category] = []
            categories[f.category].append(f)

        for cat, cat_filters in sorted(categories.items()):
            print(f"\n📁 {cat.upper()}")
            print("-" * 60)
            for f in sorted(cat_filters, key=lambda x: x.name):
                if args.verbose:
                    print(f"  {f.name}")
                    print(f"    Description: {f.description}")
                    if f.parameters:
                        print("    Parameters:")
                        for param_name, param_info in f.parameters.items():
                            default_str = f" = {param_info['default']}" if "default" in param_info else ""
                            print(f"      - {param_name}: {param_info['type']}{default_str}")
                    print()
                else:
                    print(f"  • {f.name}: {f.description}")

        print(f"\n📊 Total: {len(filters)} filters (auto-discovered from NeMo Curator)")


if __name__ == "__main__":
    main()
