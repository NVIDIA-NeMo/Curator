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

"""Search NeMo Curator processing stages by modality, category, or name.

Auto-discovers stages from nemo_curator.stages and enriches them with
modality and category metadata. Stage types, GPU requirements, and
descriptions are introspected directly from the stage classes.

Examples:
    # List all text stages
    python search_stages.py --modality text

    # List text filters only
    python search_stages.py --modality text --category filters

    # Search for deduplication stages
    python search_stages.py --search dedup

    # Show only GPU stages
    python search_stages.py --gpu-only
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from typing import Any

# Try to import shared introspection utilities
try:
    sys.path.insert(0, str(__file__).rsplit("/", 3)[0])  # Add skills/ to path
    from shared.introspect import NEMO_CURATOR_AVAILABLE, StageInfo, discover_all_stages
except ImportError:
    NEMO_CURATOR_AVAILABLE = False
    StageInfo = None  # type: ignore[misc,assignment]

    def discover_all_stages() -> list:  # type: ignore[misc]
        return []


# Modality mappings - maps stage name -> modality
# Stages not listed are inferred from their module path
MODALITY_MAP: dict[str, str] = {
    # Explicitly multi-modality stages go here if needed
}

# Category mappings - maps stage name -> category
# These provide human-friendly groupings beyond what's in the module path
CATEGORY_MAP: dict[str, str] = {
    # Text IO
    "JsonlReader": "io",
    "ParquetReader": "io",
    "JsonlWriter": "io",
    "ParquetWriter": "io",
    "MegatronTokenizerWriter": "io",
    # Text Classifiers
    "QualityClassifier": "classifiers",
    "DomainClassifier": "classifiers",
    "MultilingualDomainClassifier": "classifiers",
    "ContentTypeClassifier": "classifiers",
    "FineWebEduClassifier": "classifiers",
    "FineWebMixtralEduClassifier": "classifiers",
    "FineWebNemotronEduClassifier": "classifiers",
    "AegisClassifier": "classifiers",
    "PromptTaskComplexityClassifier": "classifiers",
    "InstructionDataGuardClassifier": "classifiers",
    # Text Filters (heuristic)
    "WordCountFilter": "filters",
    "NonAlphaNumericFilter": "filters",
    "SymbolsToWordsFilter": "filters",
    "NumbersFilter": "filters",
    "UrlsFilter": "filters",
    "BulletsFilter": "filters",
    "WhiteSpaceFilter": "filters",
    "ParenthesesFilter": "filters",
    "LongWordFilter": "filters",
    "MeanWordLengthFilter": "filters",
    "RepeatedLinesFilter": "filters",
    "RepeatedParagraphsFilter": "filters",
    "RepeatedLinesByCharFilter": "filters",
    "RepeatedParagraphsByCharFilter": "filters",
    "RepeatingTopNGramsFilter": "filters",
    "RepeatingDuplicateNGramsFilter": "filters",
    "PunctuationFilter": "filters",
    "EllipsisFilter": "filters",
    "CommonEnglishWordsFilter": "filters",
    "WordsWithoutAlphabetsFilter": "filters",
    "PornographicUrlsFilter": "filters",
    "TokenCountFilter": "filters",
    "SubstringFilter": "filters",
    "HistogramFilter": "filters",
    "BoilerPlateStringFilter": "filters",
    # Text Filters (code)
    "AlphaFilter": "filters",
    "GeneralCommentToCodeFilter": "filters",
    "HTMLBoilerplateFilter": "filters",
    "NumberOfLinesOfCodeFilter": "filters",
    "PerExtensionFilter": "filters",
    "PythonCommentToCodeFilter": "filters",
    "TokenizerFertilityFilter": "filters",
    "XMLHeaderFilter": "filters",
    # Text Filters (FastText)
    "FastTextLangId": "filters",
    "FastTextQualityFilter": "filters",
    # Text Modifiers
    "BoilerPlateStringModifier": "modifiers",
    "MarkdownRemover": "modifiers",
    "NewlineNormalizer": "modifiers",
    "QuotationRemover": "modifiers",
    "UnicodeReformatter": "modifiers",
    "UrlRemover": "modifiers",
    "DocumentModifier": "modifiers",
    "FastTextLabelModifier": "modifiers",
    "LineRemover": "modifiers",
    "Slicer": "modifiers",
    # Text Modules
    "AddId": "modules",
    "DocumentJoiner": "modules",
    "DocumentSplitter": "modules",
    "Filter": "modules",
    "Modify": "modules",
    "Score": "modules",
    "ScoreFilter": "modules",
    # Text Deduplication
    "ExactDuplicateIdentification": "deduplication",
    "FuzzyDeduplicationWorkflow": "deduplication",
    "SemanticDeduplicationWorkflow": "deduplication",
    "MinHashStage": "deduplication",
    "LSHStage": "deduplication",
    "BucketsToEdgesStage": "deduplication",
    "ConnectedComponentsStage": "deduplication",
    "IdentifyDuplicatesStage": "deduplication",
    # Text Download
    "CommonCrawlDownloadExtractStage": "download",
    "WikipediaDownloadExtractStage": "download",
    "ArxivDownloadExtractStage": "download",
    # Text Embedders
    "EmbeddingCreatorStage": "embedders",
    # Text Deduplication (internal stages)
    "MinHashStage": "deduplication",
    "LSHStage": "deduplication",
    "BucketsToEdgesStage": "deduplication",
    "ConnectedComponentsStage": "deduplication",
    "IdentifyDuplicatesStage": "deduplication",
    # Video IO
    "VideoReader": "io",
    "VideoReaderStage": "io",
    "ClipWriterStage": "io",
    # Video Clipping
    "TransNetV2ClipExtractionStage": "clipping",
    "FixedStrideExtractorStage": "clipping",
    "ClipTranscodingStage": "clipping",
    # Video Captioning
    "CaptionPreparationStage": "captioning",
    "CaptionGenerationStage": "captioning",
    "CaptionEnhancementStage": "captioning",
    # Video Embedding
    "CosmosEmbed1FrameCreationStage": "embedding",
    "CosmosEmbed1EmbeddingStage": "embedding",
    "InternVideo2EmbeddingStage": "embedding",
    # Video Filtering
    "MotionVectorDecodeStage": "filtering",
    "MotionFilterStage": "filtering",
    "ClipAestheticFilterStage": "filtering",
    # Image
    "ImageEmbeddingStage": "embedding",
    "ImageAestheticFilterStage": "filtering",
    "ImageNSFWFilterStage": "filtering",
    "ImageReaderStage": "io",
    "ImageWriterStage": "io",
    "ConvertImageBatchToDocumentBatchStage": "io",
    "ImageDuplicatesRemovalStage": "deduplication",
    # Audio
    "InferenceAsrNemoStage": "inference",
    "GetPairwiseWerStage": "metrics",
}


@dataclass
class Stage:
    """Represents a NeMo Curator processing stage with enriched metadata."""

    name: str
    modality: str
    category: str
    stage_type: str
    gpu_required: bool
    description: str
    source_path: str
    full_target: str = ""


def _infer_modality(stage_info: Any) -> str:
    """Infer modality from module path."""
    module = stage_info.module_path if hasattr(stage_info, "module_path") else ""

    if ".text." in module or ".deduplication." in module:
        return "text"
    elif ".video." in module:
        return "video"
    elif ".image." in module:
        return "image"
    elif ".audio." in module:
        return "audio"

    return "other"


def _infer_category(stage_info: Any) -> str:
    """Infer category from module path."""
    module = stage_info.module_path if hasattr(stage_info, "module_path") else ""

    # Extract category from path like "nemo_curator.stages.text.filters.heuristic_filter"
    parts = module.split(".")
    for part in parts:
        if part in (
            "io",
            "filters",
            "classifiers",
            "modifiers",
            "modules",
            "deduplication",
            "download",
            "embedders",
            "clipping",
            "captioning",
            "embedding",
            "filtering",
            "inference",
            "metrics",
            "caption",
        ):
            # Normalize some names
            if part == "caption":
                return "captioning"
            return part

    return "other"


def _convert_stage_info(stage_info: Any) -> Stage:
    """Convert StageInfo to Stage with enriched metadata."""
    name = stage_info.name

    # Get modality (from map or infer from module path)
    modality = MODALITY_MAP.get(name) or _infer_modality(stage_info)

    # Get category (from map or infer from module path)
    category = CATEGORY_MAP.get(name) or _infer_category(stage_info)

    return Stage(
        name=name,
        modality=modality,
        category=category,
        stage_type=stage_info.stage_type,
        gpu_required=stage_info.requires_gpu,
        description=stage_info.description,
        source_path=stage_info.source_path,
        full_target=stage_info.full_target,
    )


def _get_static_stages() -> list[Stage]:
    """Return static stage list when NeMo Curator is not installed."""
    # Static definitions for fallback
    static = [
        # Text IO
        ("JsonlReader", "text", "io", "ProcessingStage", False, "Read JSONL files"),
        ("ParquetReader", "text", "io", "ProcessingStage", False, "Read Parquet files"),
        ("JsonlWriter", "text", "io", "ProcessingStage", False, "Write JSONL files"),
        ("ParquetWriter", "text", "io", "ProcessingStage", False, "Write Parquet files"),
        ("MegatronTokenizerWriter", "text", "io", "ProcessingStage", False, "Write for Megatron training"),
        # Text Classifiers
        ("QualityClassifier", "text", "classifiers", "ProcessingStage", True, "General quality scoring"),
        ("DomainClassifier", "text", "classifiers", "ProcessingStage", True, "Domain classification"),
        ("FineWebEduClassifier", "text", "classifiers", "ProcessingStage", True, "Educational quality scoring"),
        ("AegisClassifier", "text", "classifiers", "ProcessingStage", True, "Safety classification"),
        # Text Filters (sample)
        ("WordCountFilter", "text", "filters", "ProcessingStage", False, "Filter by word count"),
        ("NonAlphaNumericFilter", "text", "filters", "ProcessingStage", False, "Filter by non-alphanumeric ratio"),
        ("RepeatedLinesFilter", "text", "filters", "ProcessingStage", False, "Filter by duplicate line ratio"),
        ("FastTextLangId", "text", "filters", "ProcessingStage", False, "Language identification"),
        # Text Modifiers
        ("UnicodeReformatter", "text", "modifiers", "ProcessingStage", False, "Normalize Unicode"),
        ("UrlRemover", "text", "modifiers", "ProcessingStage", False, "Remove URLs"),
        # Text Modules
        ("AddId", "text", "modules", "ProcessingStage", False, "Add document IDs"),
        ("ScoreFilter", "text", "modules", "ProcessingStage", False, "Score and filter"),
        # Text Deduplication
        ("ExactDuplicateIdentification", "text", "deduplication", "ProcessingStage", False, "Exact dedup via hash matching"),
        ("FuzzyDeduplicationWorkflow", "text", "deduplication", "WorkflowBase", True, "MinHash + LSH fuzzy dedup"),
        ("SemanticDeduplicationWorkflow", "text", "deduplication", "WorkflowBase", True, "Embedding-based semantic dedup"),
        # Text Download
        ("CommonCrawlDownloadExtractStage", "text", "download", "CompositeStage", False, "Download Common Crawl"),
        ("WikipediaDownloadExtractStage", "text", "download", "CompositeStage", False, "Download Wikipedia"),
        ("ArxivDownloadExtractStage", "text", "download", "CompositeStage", False, "Download ArXiv papers"),
        # Text Embedders
        ("EmbeddingCreatorStage", "text", "embedders", "CompositeStage", True, "Generate text embeddings"),
        # Video IO
        ("VideoReader", "video", "io", "CompositeStage", False, "Read videos from path"),
        ("VideoReaderStage", "video", "io", "ProcessingStage", False, "Read video files and metadata"),
        ("ClipWriterStage", "video", "io", "ProcessingStage", False, "Write video clips to disk"),
        # Video Clipping
        ("TransNetV2ClipExtractionStage", "video", "clipping", "ProcessingStage", True, "ML-based scene detection"),
        ("FixedStrideExtractorStage", "video", "clipping", "ProcessingStage", False, "Fixed-length clip extraction"),
        ("ClipTranscodingStage", "video", "clipping", "ProcessingStage", False, "Transcode video clips"),
        # Video Captioning
        ("CaptionPreparationStage", "video", "captioning", "ProcessingStage", False, "Prepare video for captioning"),
        ("CaptionGenerationStage", "video", "captioning", "ProcessingStage", True, "Qwen VL captioning"),
        ("CaptionEnhancementStage", "video", "captioning", "ProcessingStage", True, "Enhance generated captions"),
        # Video Embedding
        ("CosmosEmbed1EmbeddingStage", "video", "embedding", "ProcessingStage", True, "NVIDIA Cosmos embeddings"),
        ("InternVideo2EmbeddingStage", "video", "embedding", "ProcessingStage", True, "InternVideo2 embeddings"),
        # Video Filtering
        ("MotionVectorDecodeStage", "video", "filtering", "ProcessingStage", False, "Decode motion vectors"),
        ("MotionFilterStage", "video", "filtering", "ProcessingStage", False, "Filter static/low-motion clips"),
        ("ClipAestheticFilterStage", "video", "filtering", "ProcessingStage", True, "Aesthetic quality filtering"),
        # Image
        ("ImageReaderStage", "image", "io", "ProcessingStage", False, "Read images from path"),
        ("ImageWriterStage", "image", "io", "ProcessingStage", False, "Write images to disk"),
        ("ImageEmbeddingStage", "image", "embedding", "ProcessingStage", True, "CLIP embeddings for images"),
        ("ImageAestheticFilterStage", "image", "filtering", "ProcessingStage", True, "Aesthetic quality scoring"),
        ("ImageNSFWFilterStage", "image", "filtering", "ProcessingStage", True, "NSFW content detection"),
        ("ImageDuplicatesRemovalStage", "image", "deduplication", "ProcessingStage", False, "Remove duplicate images"),
        ("ConvertImageBatchToDocumentBatchStage", "image", "io", "ProcessingStage", False, "Convert images to documents"),
        # Audio
        ("InferenceAsrNemoStage", "audio", "inference", "ProcessingStage", True, "NeMo ASR transcription"),
        ("GetPairwiseWerStage", "audio", "metrics", "ProcessingStage", False, "Word Error Rate calculation"),
    ]
    return [
        Stage(
            name=name,
            modality=modality,
            category=category,
            stage_type=stage_type,
            gpu_required=gpu,
            description=desc,
            source_path=f"stages/{modality}/{category}/",
            full_target=f"nemo_curator.stages.{modality}.{category}.{name}",
        )
        for name, modality, category, stage_type, gpu, desc in static
    ]


def discover_stages() -> list[Stage]:
    """Discover all stages from NeMo Curator or use static fallback."""
    if not NEMO_CURATOR_AVAILABLE:
        return _get_static_stages()

    discovered = discover_all_stages()
    return [_convert_stage_info(s) for s in discovered]


# Cache discovered stages
_STAGES_CACHE: list[Stage] | None = None


def get_stages() -> list[Stage]:
    """Get all stages, using cache if available."""
    global _STAGES_CACHE
    if _STAGES_CACHE is None:
        _STAGES_CACHE = discover_stages()
    return _STAGES_CACHE


def search_stages(
    modality: str | None = None,
    category: str | None = None,
    search: str | None = None,
    gpu_only: bool = False,
    cpu_only: bool = False,
) -> list[Stage]:
    """Search stages by various criteria."""
    results = get_stages().copy()

    if modality:
        results = [s for s in results if s.modality == modality.lower()]

    if category:
        results = [s for s in results if s.category == category.lower()]

    if search:
        pattern = re.compile(search, re.IGNORECASE)
        results = [s for s in results if pattern.search(s.name) or pattern.search(s.description)]

    if gpu_only:
        results = [s for s in results if s.gpu_required]

    if cpu_only:
        results = [s for s in results if not s.gpu_required]

    return results


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--modality",
        choices=["text", "video", "image", "audio", "other"],
        help="Filter by modality",
    )
    parser.add_argument(
        "--category",
        help="Filter by category (io, filters, classifiers, deduplication, etc.)",
    )
    parser.add_argument(
        "--search",
        help="Search by stage name or description (regex)",
    )
    parser.add_argument(
        "--gpu-only",
        action="store_true",
        help="Show only GPU stages",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Show only CPU stages",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    parser.add_argument(
        "--check-coverage",
        action="store_true",
        help="Check which stages are missing category mappings",
    )

    args = parser.parse_args()

    if args.check_coverage:
        if not NEMO_CURATOR_AVAILABLE:
            print("❌ nemo_curator is not installed. Cannot check coverage.")
            print("   Install with: pip install nemo-curator")
            sys.exit(1)
        stages = get_stages()
        # Check for stages that fell back to inference
        discovered = discover_all_stages()
        unmapped = []
        for s in discovered:
            if s.name not in CATEGORY_MAP:
                unmapped.append(s.name)
        if unmapped:
            print("⚠️  Stages missing category mapping (using inference):")
            for name in sorted(unmapped):
                print(f"  - {name}")
            print(f"\nAdd these to CATEGORY_MAP in {__file__}")
        else:
            print("✅ All stages have category mappings")
        return

    results = search_stages(
        modality=args.modality,
        category=args.category,
        search=args.search,
        gpu_only=args.gpu_only,
        cpu_only=args.cpu_only,
    )

    # Show warning if using static fallback
    if not NEMO_CURATOR_AVAILABLE and not args.json:
        print("⚠️  nemo_curator not installed - using static stage list")
        print("   Install nemo-curator for live introspection\n")

    if args.json:
        output = [
            {
                "name": s.name,
                "modality": s.modality,
                "category": s.category,
                "type": s.stage_type,
                "gpu_required": s.gpu_required,
                "description": s.description,
                "source": s.source_path,
                "target": s.full_target,
            }
            for s in results
        ]
        print(json.dumps(output, indent=2))
    else:
        if not results:
            print("No stages found matching criteria.")
            sys.exit(0)

        # Group by category for display
        categories: dict[str, list[Stage]] = {}
        for s in results:
            key = f"{s.modality}/{s.category}"
            if key not in categories:
                categories[key] = []
            categories[key].append(s)

        for cat_key, stages in sorted(categories.items()):
            print(f"\n📁 {cat_key}")
            print("-" * 60)
            for s in sorted(stages, key=lambda x: x.name):
                gpu_marker = "🔶" if s.gpu_required else "  "
                type_marker = {
                    "ProcessingStage": "S",
                    "CompositeStage": "C",
                    "WorkflowBase": "W",
                }.get(s.stage_type, "?")
                print(f"  {gpu_marker} [{type_marker}] {s.name}")
                print(f"       {s.description}")

        print(f"\n📊 Total: {len(results)} stages (auto-discovered from NeMo Curator)")
        print("   🔶 = GPU required, [S] = Stage, [C] = Composite, [W] = Workflow")


if __name__ == "__main__":
    main()
