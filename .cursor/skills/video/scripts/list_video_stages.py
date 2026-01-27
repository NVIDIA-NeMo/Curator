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

"""List available video processing stages in NeMo Curator.

Examples:
    # List all video stages
    python list_video_stages.py

    # List with descriptions
    python list_video_stages.py --verbose

    # Filter by category
    python list_video_stages.py --category clipping

    # Search for stages
    python list_video_stages.py --search embed
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.introspect import estimate_gpu_memory, get_stages_by_modality

# Category mappings for video stages
VIDEO_CATEGORIES = {
    "io": ["VideoReader", "VideoReaderStage", "ClipWriterStage"],
    "clipping": [
        "TransNetV2ClipExtractionStage",
        "FixedStrideExtractorStage",
        "ClipTranscodingStage",
        "VideoFrameExtractionStage",
        "ClipFrameExtractionStage",
    ],
    "filtering": [
        "MotionVectorDecodeStage",
        "MotionFilterStage",
        "ClipAestheticFilterStage",
    ],
    "caption": [
        "CaptionPreparationStage",
        "CaptionGenerationStage",
        "CaptionEnhancementStage",
    ],
    "embedding": [
        "CosmosEmbed1FrameCreationStage",
        "CosmosEmbed1EmbeddingStage",
        "InternVideo2FrameCreationStage",
        "InternVideo2EmbeddingStage",
    ],
    "preview": ["PreviewStage"],
}


def get_category(stage_name: str) -> str:
    """Get the category for a stage."""
    for category, stages in VIDEO_CATEGORIES.items():
        if stage_name in stages:
            return category
    return "other"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="List available video processing stages"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed descriptions"
    )
    parser.add_argument(
        "--category",
        "-c",
        choices=list(VIDEO_CATEGORIES.keys()) + ["all"],
        default="all",
        help="Filter by category",
    )
    parser.add_argument(
        "--search", "-s", type=str, help="Search for stages by name"
    )
    parser.add_argument(
        "--json", action="store_true", help="Output as JSON"
    )
    args = parser.parse_args()

    stages = get_stages_by_modality("video")

    # Filter by category
    if args.category != "all":
        category_stages = VIDEO_CATEGORIES.get(args.category, [])
        stages = [s for s in stages if s.name in category_stages]

    # Search filter
    if args.search:
        search_lower = args.search.lower()
        stages = [s for s in stages if search_lower in s.name.lower()]

    # Sort by category then name
    stages.sort(key=lambda s: (get_category(s.name), s.name))

    if args.json:
        import json

        output = []
        for stage in stages:
            output.append({
                "name": stage.name,
                "category": get_category(stage.name),
                "module": stage.module_path,
                "gpu_memory_gb": estimate_gpu_memory(stage.name),
                "description": stage.description,
            })
        print(json.dumps(output, indent=2))
        return

    if not stages:
        print("No video stages found.")
        print("\nNote: NeMo Curator must be installed to discover stages.")
        print("Install with: pip install nemo-curator[video]")
        return

    print(f"Found {len(stages)} video stages:\n")

    current_category = None
    for stage in stages:
        category = get_category(stage.name)
        if category != current_category:
            current_category = category
            print(f"\n## {category.upper()}")
            print("-" * 40)

        gpu_mem = estimate_gpu_memory(stage.name)
        gpu_str = f" (GPU: {gpu_mem}GB)" if gpu_mem > 0 else " (CPU)"

        print(f"  {stage.name}{gpu_str}")

        if args.verbose:
            print(f"    Module: {stage.module_path}")
            print(f"    Description: {stage.description}")
            if stage.parameters:
                params = ", ".join(stage.parameters.keys())
                print(f"    Parameters: {params}")
            print()


if __name__ == "__main__":
    main()
