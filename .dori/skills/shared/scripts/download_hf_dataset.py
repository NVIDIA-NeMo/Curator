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

"""Download any HuggingFace dataset and convert for NeMo Curator.

Generic utility to download datasets from HuggingFace Hub and prepare them
for use with NeMo Curator pipelines. Supports format conversion, field mapping,
streaming for large datasets, and sampling.

Usage:
    # Download dataset to JSONL
    python download_hf_dataset.py allenai/c4 --split train --samples 1000

    # Specify text field mapping
    python download_hf_dataset.py wikitext --config wikitext-103-v1 --text-field text

    # Download full dataset as parquet
    python download_hf_dataset.py HuggingFaceFW/fineweb --format parquet --split train

    # Search for datasets
    python download_hf_dataset.py --search "code python"

    # Get dataset info without downloading
    python download_hf_dataset.py bigcode/starcoderdata --info
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Check for required dependencies
MISSING_DEPS = []
try:
    from datasets import load_dataset, get_dataset_config_names, get_dataset_split_names
except ImportError:
    MISSING_DEPS.append("datasets")

try:
    from huggingface_hub import HfApi, list_datasets
except ImportError:
    MISSING_DEPS.append("huggingface_hub")


def check_dependencies() -> None:
    """Check if required dependencies are installed."""
    if MISSING_DEPS:
        print("Missing required dependencies. Install with:")
        print(f"  pip install {' '.join(MISSING_DEPS)}")
        sys.exit(1)


def search_datasets(query: str, limit: int = 20) -> list[dict[str, Any]]:
    """Search HuggingFace Hub for datasets.

    Args:
        query: Search query (keywords)
        limit: Maximum results to return

    Returns:
        List of dataset metadata dictionaries
    """
    check_dependencies()

    print(f"🔍 Searching HuggingFace for: '{query}'")
    api = HfApi()

    results = []
    for ds in list_datasets(search=query, limit=limit, sort="downloads", direction=-1):
        results.append({
            "id": ds.id,
            "downloads": ds.downloads,
            "likes": ds.likes,
            "tags": ds.tags[:5] if ds.tags else [],
        })

    return results


def get_dataset_info(dataset_path: str, config: str | None = None) -> dict[str, Any]:
    """Get metadata about a dataset without downloading.

    Args:
        dataset_path: HuggingFace dataset path (e.g., 'allenai/c4')
        config: Optional dataset configuration name

    Returns:
        Dictionary with dataset information
    """
    check_dependencies()

    print(f"📊 Getting info for: {dataset_path}")

    info: dict[str, Any] = {"path": dataset_path}

    # Get available configs
    try:
        configs = get_dataset_config_names(dataset_path)
        info["configs"] = configs
    except Exception:
        info["configs"] = ["default"]

    # Get splits for the specified config
    target_config = config or (info["configs"][0] if info["configs"] else None)
    try:
        splits = get_dataset_split_names(dataset_path, target_config)
        info["splits"] = splits
    except Exception:
        info["splits"] = ["train"]

    # Try to load a tiny sample to get column info
    try:
        ds = load_dataset(
            dataset_path,
            target_config,
            split=f"{info['splits'][0]}[:1]",
            trust_remote_code=True,
        )
        info["columns"] = ds.column_names
        info["num_rows_sample"] = len(ds)

        # Try to detect text field
        text_candidates = ["text", "content", "document", "passage", "sentence", "raw_content"]
        for candidate in text_candidates:
            if candidate in info["columns"]:
                info["suggested_text_field"] = candidate
                break
    except Exception as e:
        info["columns"] = f"Could not inspect: {e}"

    return info


def download_dataset(
    dataset_path: str,
    output_dir: Path,
    config: str | None = None,
    split: str = "train",
    text_field: str | None = None,
    id_field: str | None = None,
    samples: int | None = None,
    output_format: str = "jsonl",
    streaming: bool = True,
    include_metadata: bool = False,
) -> dict[str, Any]:
    """Download dataset and convert to NeMo Curator format.

    Args:
        dataset_path: HuggingFace dataset path
        output_dir: Directory to save output
        config: Dataset configuration name
        split: Dataset split (train/validation/test)
        text_field: Column containing main text content
        id_field: Column containing document ID (auto-generated if None)
        samples: Number of samples to download (None for all)
        output_format: Output format (jsonl or parquet)
        streaming: Use streaming mode (recommended for large datasets)
        include_metadata: Include all columns, not just text

    Returns:
        Dictionary with download statistics
    """
    check_dependencies()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build dataset name for output file
    safe_name = dataset_path.replace("/", "_")
    if config:
        safe_name = f"{safe_name}_{config}"

    print(f"📥 Downloading: {dataset_path}")
    if config:
        print(f"   Config: {config}")
    print(f"   Split: {split}")
    if samples:
        print(f"   Samples: {samples}")

    # Load dataset
    load_kwargs: dict[str, Any] = {
        "path": dataset_path,
        "split": split,
        "streaming": streaming,
        "trust_remote_code": True,
    }
    if config:
        load_kwargs["name"] = config

    ds = load_dataset(**load_kwargs)

    # Auto-detect text field if not specified
    if text_field is None:
        if streaming:
            # Get first item to inspect columns
            first_item = next(iter(ds))
            columns = list(first_item.keys())
        else:
            columns = ds.column_names

        text_candidates = ["text", "content", "document", "passage", "sentence", "raw_content"]
        for candidate in text_candidates:
            if candidate in columns:
                text_field = candidate
                print(f"   Auto-detected text field: '{text_field}'")
                break

        if text_field is None:
            text_field = columns[0]
            print(f"   Using first column as text: '{text_field}'")

    # Process and save
    stats = {"total": 0, "text_field": text_field, "format": output_format}

    if output_format == "jsonl":
        output_file = output_dir / f"{safe_name}_{split}.jsonl"
        with open(output_file, "w") as f:
            for i, row in enumerate(ds):
                if samples and i >= samples:
                    break

                # Build document
                doc: dict[str, Any] = {
                    "id": row.get(id_field, f"doc_{i:08d}") if id_field else f"doc_{i:08d}",
                    "text": row.get(text_field, ""),
                }

                # Include source metadata
                doc["_source"] = {
                    "dataset": dataset_path,
                    "split": split,
                }
                if config:
                    doc["_source"]["config"] = config

                # Include all fields if requested
                if include_metadata:
                    for key, value in row.items():
                        if key not in [text_field, id_field]:
                            # Skip non-serializable fields
                            if isinstance(value, (str, int, float, bool, list, dict, type(None))):
                                doc[key] = value

                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                stats["total"] += 1

                # Progress indicator
                if stats["total"] % 1000 == 0:
                    print(f"   Processed {stats['total']} documents...")

        stats["output_file"] = str(output_file)

    elif output_format == "parquet":
        # For parquet, we need to materialize the dataset
        output_file = output_dir / f"{safe_name}_{split}.parquet"

        if streaming:
            # Collect into list first
            rows = []
            for i, row in enumerate(ds):
                if samples and i >= samples:
                    break
                rows.append(row)
                if len(rows) % 1000 == 0:
                    print(f"   Collected {len(rows)} documents...")

            # Convert to dataset and save
            import pandas as pd

            df = pd.DataFrame(rows)
            df.to_parquet(output_file)
            stats["total"] = len(rows)
        else:
            if samples:
                ds = ds.select(range(min(samples, len(ds))))
            ds.to_parquet(output_file)
            stats["total"] = len(ds)

        stats["output_file"] = str(output_file)

    print(f"✅ Downloaded {stats['total']} documents")
    print(f"💾 Saved to: {stats['output_file']}")

    return stats


def print_search_results(results: list[dict[str, Any]]) -> None:
    """Pretty print search results."""
    print("\n" + "=" * 70)
    print("Search Results")
    print("=" * 70)

    for i, ds in enumerate(results, 1):
        print(f"\n{i}. {ds['id']}")
        print(f"   Downloads: {ds['downloads']:,} | Likes: {ds['likes']}")
        if ds["tags"]:
            print(f"   Tags: {', '.join(ds['tags'])}")

    print("\n" + "-" * 70)
    print("To download: python download_hf_dataset.py <dataset_id> --split train")


def print_dataset_info(info: dict[str, Any]) -> None:
    """Pretty print dataset info."""
    print("\n" + "=" * 70)
    print(f"Dataset: {info['path']}")
    print("=" * 70)

    if isinstance(info.get("configs"), list):
        print(f"\nConfigs: {', '.join(info['configs'][:10])}")
        if len(info.get("configs", [])) > 10:
            print(f"   ... and {len(info['configs']) - 10} more")

    if isinstance(info.get("splits"), list):
        print(f"Splits: {', '.join(info['splits'])}")

    if isinstance(info.get("columns"), list):
        print(f"Columns: {', '.join(info['columns'])}")
        if "suggested_text_field" in info:
            print(f"Suggested text field: '{info['suggested_text_field']}'")

    print("\n" + "-" * 70)
    suggested_cmd = f"python download_hf_dataset.py {info['path']}"
    if info.get("suggested_text_field"):
        suggested_cmd += f" --text-field {info['suggested_text_field']}"
    print(f"Download: {suggested_cmd} --split train --samples 1000")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Download HuggingFace datasets for NeMo Curator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Search for datasets
  python download_hf_dataset.py --search "wikipedia"

  # Get dataset info (columns, splits, configs)
  python download_hf_dataset.py wikitext --info

  # Download with auto-detected text field
  python download_hf_dataset.py allenai/c4 --split train --samples 1000

  # Download with specific config and text field
  python download_hf_dataset.py wikitext --config wikitext-103-v1 --text-field text

  # Download as parquet
  python download_hf_dataset.py HuggingFaceFW/fineweb --format parquet --samples 5000

  # Include all metadata columns
  python download_hf_dataset.py pile-of-law/pile-of-law --include-metadata

Common datasets for LLM training:
  - allenai/c4 (Common Crawl, cleaned)
  - HuggingFaceFW/fineweb (High-quality web data)
  - bigcode/starcoderdata (Code)
  - togethercomputer/RedPajama-Data-1T (Mix)
  - pile-of-law/pile-of-law (Legal)
  - wikipedia (Encyclopedia)
        """,
    )

    # Positional argument for dataset path
    parser.add_argument(
        "dataset",
        nargs="?",
        help="HuggingFace dataset path (e.g., 'allenai/c4')",
    )

    # Search mode
    parser.add_argument(
        "--search",
        type=str,
        metavar="QUERY",
        help="Search HuggingFace for datasets by keyword",
    )

    # Info mode
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show dataset info without downloading",
    )

    # Dataset options
    parser.add_argument(
        "--config",
        type=str,
        help="Dataset configuration name",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split (default: train)",
    )
    parser.add_argument(
        "--text-field",
        type=str,
        help="Column containing text (auto-detected if not specified)",
    )
    parser.add_argument(
        "--id-field",
        type=str,
        help="Column containing document ID (auto-generated if not specified)",
    )

    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data",
        help="Output directory (default: ./data)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["jsonl", "parquet"],
        default="jsonl",
        help="Output format (default: jsonl)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        help="Number of samples to download (default: all)",
    )
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Disable streaming mode (loads full dataset into memory)",
    )
    parser.add_argument(
        "--include-metadata",
        action="store_true",
        help="Include all columns in output, not just text",
    )

    args = parser.parse_args()

    # Handle search mode
    if args.search:
        check_dependencies()
        results = search_datasets(args.search)
        print_search_results(results)
        return

    # Require dataset for other modes
    if not args.dataset:
        parser.print_help()
        print("\n❌ Please specify a dataset path or use --search")
        sys.exit(1)

    check_dependencies()

    # Handle info mode
    if args.info:
        info = get_dataset_info(args.dataset, args.config)
        print_dataset_info(info)
        return

    # Download mode
    stats = download_dataset(
        dataset_path=args.dataset,
        output_dir=Path(args.output_dir),
        config=args.config,
        split=args.split,
        text_field=args.text_field,
        id_field=args.id_field,
        samples=args.samples,
        output_format=args.format,
        streaming=not args.no_streaming,
        include_metadata=args.include_metadata,
    )

    print("\n" + "=" * 70)
    print("Download Complete")
    print("=" * 70)
    print(f"Documents: {stats['total']:,}")
    print(f"Text field: {stats['text_field']}")
    print(f"Output: {stats['output_file']}")
    print("\nReady for NeMo Curator processing!")


if __name__ == "__main__":
    main()
