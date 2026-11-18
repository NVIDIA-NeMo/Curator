# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Example: Image download pipeline using NeMo Curator.

This demonstrates how the cc-img-dl pipeline maps to Curator stages.
Two approaches are shown:

1. Simple (CompositeStage) — one stage, one line
2. Fine-grained — individual stages for customization

Usage:
    python image_download_example.py --manifest url_manifest.parquet --output-dir /data/images
"""

import argparse
import sys
from pathlib import Path

# Add the PoC package to the path so we can import the new stages
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nemo_curator.pipeline import Pipeline

from nemo_curator_stages.image.download import ImageDownloadCompositeStage  # noqa: E402
from nemo_curator_stages.image.download.compliance import ComplianceFilterStage  # noqa: E402
from nemo_curator_stages.image.download.downloader import ImageDownloaderStage  # noqa: E402
from nemo_curator_stages.image.download.manifest import ManifestReaderStage  # noqa: E402


def simple_pipeline(manifest_path: str, output_dir: str, limit: int | None = None) -> Pipeline:
    """Approach 1: Single CompositeStage — the simplest way to use the pipeline.

    This is equivalent to running the entire cc-img-dl CLI:
        cc-img-dl --manifest url_manifest.parquet --output-dir /data/images --num-shards 8
    """
    return Pipeline(
        name="image_download_simple",
        description="Download images from URL manifest with compliance checking",
        stages=[
            ImageDownloadCompositeStage(
                manifest_path=manifest_path,
                output_dir=output_dir,
                limit=limit,
                enable_compliance=True,
                user_agent="my-team/1.0 (+https://example.org)",
                compliance_failure_policy="conservative",
                urls_per_task=500,
                download_concurrency=20,
            ),
        ],
    )


def finegrained_pipeline(manifest_path: str, output_dir: str, limit: int | None = None) -> Pipeline:
    """Approach 2: Individual stages — full control over each step.

    This lets you:
    - Customize resources per stage (e.g., more CPUs for compliance checking)
    - Insert additional stages (e.g., URL deduplication between manifest and compliance)
    - Disable compliance checking entirely by omitting ComplianceFilterStage
    - Use different compliance policies for different URL patterns
    """
    return Pipeline(
        name="image_download_finegrained",
        description="Download images with per-stage control",
        stages=[
            # Stage 1: Read manifest and produce URL batches
            ManifestReaderStage(
                manifest_path=manifest_path,
                urls_per_task=1000,
                limit=limit,
            ),
            # Stage 2: Filter out non-compliant URLs (opt-in)
            ComplianceFilterStage(
                user_agent="my-team/1.0 (+https://example.org)",
                compliance_failure_policy="conservative",
                robots_ttl=86400,
                tdm_ttl=86400,
                max_workers=8,
            ),
            # Stage 3: Download images
            ImageDownloaderStage(
                output_dir=output_dir,
                user_agent="my-team/1.0 (+https://example.org)",
                max_retries=3,
                max_workers=20,
                max_image_bytes=50_000_000,
            ),
        ],
    )


def main():
    parser = argparse.ArgumentParser(description="Image download via NeMo Curator")
    parser.add_argument("--manifest", required=True, help="Path to url_manifest.parquet")
    parser.add_argument("--output-dir", required=True, help="Output directory for images")
    parser.add_argument("--limit", type=int, help="Limit number of URLs (for testing)")
    parser.add_argument("--mode", choices=["simple", "finegrained"], default="simple")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.mode == "simple":
        pipeline = simple_pipeline(args.manifest, args.output_dir, args.limit)
    else:
        pipeline = finegrained_pipeline(args.manifest, args.output_dir, args.limit)

    print(f"\n{'='*60}")
    print(f"Pipeline: {pipeline.name}")
    print(f"{'='*60}")
    pipeline.describe()
    print(f"{'='*60}\n")

    # In a real deployment, you'd use XennaExecutor for distributed execution.
    # For local testing, the default executor works.
    pipeline.run()

    print("\nDone.")


if __name__ == "__main__":
    main()
