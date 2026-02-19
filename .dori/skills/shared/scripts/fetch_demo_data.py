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

"""Fetch demo datasets from HuggingFace for NeMo Curator skill demos.

This script downloads small sample datasets suitable for demonstrating
NeMo Curator pipelines. Supports text, audio, image, and video modalities.

Usage:
    # Fetch all modalities with defaults
    python fetch_demo_data.py --all --output-dir ./demo_data

    # Fetch specific modality
    python fetch_demo_data.py --text --samples 100 --output-dir ./demo_data
    python fetch_demo_data.py --audio --language en_us --samples 50

    # Fetch multiple modalities
    python fetch_demo_data.py --text --audio --image --samples 50

    # List available datasets without downloading
    python fetch_demo_data.py --list
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

# Check for required dependencies
MISSING_DEPS = []
try:
    from datasets import load_dataset
except ImportError:
    MISSING_DEPS.append("datasets")

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    MISSING_DEPS.append("huggingface_hub")

# Dataset registry with metadata
DATASET_REGISTRY: dict[str, dict[str, Any]] = {
    "text": {
        "tiny_stories": {
            "hf_path": "roneneldan/TinyStories",
            "description": "Small synthetic stories for text processing demos",
            "split": "train",
            "text_field": "text",
            "default_samples": 100,
            "license": "CDLA-Sharing-1.0",
        },
        "wikipedia_sample": {
            "hf_path": "wikipedia",
            "hf_config": "20220301.simple",
            "description": "Simple English Wikipedia articles",
            "split": "train",
            "text_field": "text",
            "default_samples": 50,
            "license": "CC BY-SA 3.0",
        },
    },
    "audio": {
        "fleurs": {
            "hf_path": "google/fleurs",
            "description": "Multilingual speech dataset (102 languages)",
            "split": "train",
            "audio_field": "audio",
            "text_field": "transcription",
            "default_samples": 50,
            "default_language": "en_us",
            "available_languages": [
                "en_us",
                "es_419",
                "fr_fr",
                "de_de",
                "zh_cn",
                "ja_jp",
                "ko_kr",
                "hi_in",
                "ar_eg",
                "ru_ru",
            ],
            "license": "CC BY 4.0",
        },
        "common_voice": {
            "hf_path": "mozilla-foundation/common_voice_16_1",
            "description": "Mozilla Common Voice speech dataset",
            "split": "train",
            "audio_field": "audio",
            "text_field": "sentence",
            "default_samples": 50,
            "default_language": "en",
            "available_languages": ["en", "es", "fr", "de", "zh-CN", "ja", "ko"],
            "license": "CC0",
            "requires_auth": True,
        },
    },
    "image": {
        "flickr30k_sample": {
            "hf_path": "nlphuji/flickr30k",
            "description": "Flickr30k image-caption pairs",
            "split": "test",
            "image_field": "image",
            "caption_field": "caption",
            "default_samples": 50,
            "license": "Custom (research only)",
        },
        "coco_captions": {
            "hf_path": "HuggingFaceM4/COCO",
            "description": "MS COCO image-caption dataset",
            "split": "train",
            "image_field": "image",
            "caption_field": "sentences_raw",
            "default_samples": 50,
            "license": "CC BY 4.0",
        },
    },
    "video": {
        "webvid_sample": {
            "hf_path": "TempoFunk/webvid-10M",
            "description": "WebVid video-text pairs (URLs only, videos not included)",
            "split": "train",
            "url_field": "contentUrl",
            "caption_field": "name",
            "default_samples": 20,
            "license": "Custom",
            "note": "Contains URLs to videos, not actual video files",
        },
    },
}


def check_dependencies() -> None:
    """Check if required dependencies are installed."""
    if MISSING_DEPS:
        print("Missing required dependencies. Install with:")
        print(f"  pip install {' '.join(MISSING_DEPS)}")
        sys.exit(1)


def list_available_datasets() -> None:
    """Print available datasets by modality."""
    print("\n" + "=" * 60)
    print("Available Demo Datasets for NeMo Curator")
    print("=" * 60)

    for modality, datasets in DATASET_REGISTRY.items():
        print(f"\n📁 {modality.upper()}")
        print("-" * 40)
        for name, info in datasets.items():
            print(f"  • {name}")
            print(f"    Dataset: {info['hf_path']}")
            print(f"    Description: {info['description']}")
            print(f"    Default samples: {info['default_samples']}")
            if "available_languages" in info:
                langs = ", ".join(info["available_languages"][:5])
                print(f"    Languages: {langs}...")
            if info.get("requires_auth"):
                print("    ⚠️  Requires HuggingFace authentication")
            if info.get("note"):
                print(f"    Note: {info['note']}")
            print()


def fetch_text_dataset(
    dataset_name: str = "tiny_stories",
    num_samples: int | None = None,
    output_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Fetch text dataset samples.

    Args:
        dataset_name: Name of text dataset from registry
        num_samples: Number of samples to fetch (None for default)
        output_dir: Directory to save output (None for no file output)

    Returns:
        List of document dictionaries with 'id' and 'text' fields
    """
    if dataset_name not in DATASET_REGISTRY["text"]:
        available = list(DATASET_REGISTRY["text"].keys())
        raise ValueError(f"Unknown text dataset: {dataset_name}. Available: {available}")

    config = DATASET_REGISTRY["text"][dataset_name]
    samples = num_samples or config["default_samples"]

    print(f"📥 Fetching {samples} samples from {config['hf_path']}...")

    # Load with streaming to avoid downloading entire dataset
    ds_kwargs = {"path": config["hf_path"], "split": config["split"], "streaming": True}
    if "hf_config" in config:
        ds_kwargs["name"] = config["hf_config"]

    ds = load_dataset(**ds_kwargs)

    # Extract samples
    documents = []
    for i, row in enumerate(ds):
        if i >= samples:
            break
        documents.append({
            "id": f"doc_{i:06d}",
            "text": row[config["text_field"]],
            "source": config["hf_path"],
        })

    print(f"✅ Fetched {len(documents)} text documents")

    # Save if output directory specified
    if output_dir:
        output_dir = Path(output_dir) / "text"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{dataset_name}.jsonl"
        with open(output_file, "w") as f:
            for doc in documents:
                f.write(json.dumps(doc) + "\n")
        print(f"💾 Saved to {output_file}")

    return documents


def fetch_audio_dataset(
    dataset_name: str = "fleurs",
    language: str | None = None,
    num_samples: int | None = None,
    output_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Fetch audio dataset samples.

    Args:
        dataset_name: Name of audio dataset from registry
        language: Language code (e.g., 'en_us' for FLEURS)
        num_samples: Number of samples to fetch
        output_dir: Directory to save output

    Returns:
        List of audio sample dictionaries
    """
    if dataset_name not in DATASET_REGISTRY["audio"]:
        available = list(DATASET_REGISTRY["audio"].keys())
        raise ValueError(f"Unknown audio dataset: {dataset_name}. Available: {available}")

    config = DATASET_REGISTRY["audio"][dataset_name]
    samples = num_samples or config["default_samples"]
    lang = language or config.get("default_language", "en")

    if config.get("requires_auth"):
        print(f"⚠️  {dataset_name} requires HuggingFace authentication")
        print("   Run: huggingface-cli login")

    print(f"📥 Fetching {samples} audio samples from {config['hf_path']} ({lang})...")

    # Load with streaming
    ds = load_dataset(config["hf_path"], lang, split=config["split"], streaming=True)

    # Extract samples (metadata only, audio paths)
    audio_samples = []
    for i, row in enumerate(ds):
        if i >= samples:
            break

        sample = {
            "id": f"audio_{i:06d}",
            "transcription": row.get(config["text_field"], ""),
            "language": lang,
            "source": config["hf_path"],
        }

        # Include audio metadata if available
        if config["audio_field"] in row:
            audio_data = row[config["audio_field"]]
            if isinstance(audio_data, dict):
                sample["sampling_rate"] = audio_data.get("sampling_rate")
                sample["audio_path"] = audio_data.get("path")

        audio_samples.append(sample)

    print(f"✅ Fetched {len(audio_samples)} audio samples")

    # Save metadata if output directory specified
    if output_dir:
        output_dir = Path(output_dir) / "audio"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{dataset_name}_{lang}.jsonl"
        with open(output_file, "w") as f:
            for sample in audio_samples:
                f.write(json.dumps(sample) + "\n")
        print(f"💾 Saved metadata to {output_file}")

    return audio_samples


def fetch_image_dataset(
    dataset_name: str = "flickr30k_sample",
    num_samples: int | None = None,
    output_dir: Path | None = None,
    download_images: bool = False,
) -> list[dict[str, Any]]:
    """Fetch image dataset samples.

    Args:
        dataset_name: Name of image dataset from registry
        num_samples: Number of samples to fetch
        output_dir: Directory to save output
        download_images: Whether to download actual image files

    Returns:
        List of image sample dictionaries
    """
    if dataset_name not in DATASET_REGISTRY["image"]:
        available = list(DATASET_REGISTRY["image"].keys())
        raise ValueError(f"Unknown image dataset: {dataset_name}. Available: {available}")

    config = DATASET_REGISTRY["image"][dataset_name]
    samples = num_samples or config["default_samples"]

    print(f"📥 Fetching {samples} image samples from {config['hf_path']}...")

    ds = load_dataset(config["hf_path"], split=config["split"], streaming=True)

    image_samples = []
    for i, row in enumerate(ds):
        if i >= samples:
            break

        caption = row.get(config["caption_field"], "")
        if isinstance(caption, list):
            caption = caption[0] if caption else ""

        sample = {
            "id": f"image_{i:06d}",
            "caption": caption,
            "source": config["hf_path"],
        }

        # Handle image data
        if download_images and output_dir:
            image_data = row.get(config["image_field"])
            if image_data:
                img_dir = Path(output_dir) / "image" / "files"
                img_dir.mkdir(parents=True, exist_ok=True)
                img_path = img_dir / f"image_{i:06d}.jpg"
                image_data.save(img_path)
                sample["image_path"] = str(img_path)

        image_samples.append(sample)

    print(f"✅ Fetched {len(image_samples)} image samples")

    # Save metadata
    if output_dir:
        output_dir = Path(output_dir) / "image"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{dataset_name}.jsonl"
        with open(output_file, "w") as f:
            for sample in image_samples:
                f.write(json.dumps(sample) + "\n")
        print(f"💾 Saved metadata to {output_file}")
        if download_images:
            print(f"🖼️  Images saved to {output_dir}/files/")

    return image_samples


def fetch_video_dataset(
    dataset_name: str = "webvid_sample",
    num_samples: int | None = None,
    output_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Fetch video dataset samples (URLs/metadata only).

    Args:
        dataset_name: Name of video dataset from registry
        num_samples: Number of samples to fetch
        output_dir: Directory to save output

    Returns:
        List of video sample dictionaries with URLs
    """
    if dataset_name not in DATASET_REGISTRY["video"]:
        available = list(DATASET_REGISTRY["video"].keys())
        raise ValueError(f"Unknown video dataset: {dataset_name}. Available: {available}")

    config = DATASET_REGISTRY["video"][dataset_name]
    samples = num_samples or config["default_samples"]

    print(f"📥 Fetching {samples} video samples from {config['hf_path']}...")
    print("   Note: This fetches metadata/URLs only, not actual video files")

    ds = load_dataset(config["hf_path"], split=config["split"], streaming=True)

    video_samples = []
    for i, row in enumerate(ds):
        if i >= samples:
            break

        sample = {
            "id": f"video_{i:06d}",
            "caption": row.get(config["caption_field"], ""),
            "url": row.get(config["url_field"], ""),
            "source": config["hf_path"],
        }
        video_samples.append(sample)

    print(f"✅ Fetched {len(video_samples)} video samples")

    if output_dir:
        output_dir = Path(output_dir) / "video"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{dataset_name}.jsonl"
        with open(output_file, "w") as f:
            for sample in video_samples:
                f.write(json.dumps(sample) + "\n")
        print(f"💾 Saved to {output_file}")

    return video_samples


def main() -> None:
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Fetch demo datasets from HuggingFace for NeMo Curator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available datasets
  python fetch_demo_data.py --list

  # Fetch text data (100 samples)
  python fetch_demo_data.py --text --output-dir ./demo_data

  # Fetch specific text dataset with custom sample count
  python fetch_demo_data.py --text --dataset tiny_stories --samples 200

  # Fetch audio data in Spanish
  python fetch_demo_data.py --audio --language es_419 --samples 50

  # Fetch multiple modalities
  python fetch_demo_data.py --text --audio --image --output-dir ./demo_data

  # Fetch all modalities
  python fetch_demo_data.py --all --output-dir ./demo_data

  # Fetch images with actual image files
  python fetch_demo_data.py --image --download-images --output-dir ./demo_data
        """,
    )

    # Modality selection
    parser.add_argument("--text", action="store_true", help="Fetch text dataset")
    parser.add_argument("--audio", action="store_true", help="Fetch audio dataset")
    parser.add_argument("--image", action="store_true", help="Fetch image dataset")
    parser.add_argument("--video", action="store_true", help="Fetch video dataset (URLs only)")
    parser.add_argument("--all", action="store_true", help="Fetch all modalities")

    # Options
    parser.add_argument(
        "--dataset",
        type=str,
        help="Specific dataset name within modality (see --list)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        help="Number of samples to fetch (default varies by dataset)",
    )
    parser.add_argument(
        "--language",
        type=str,
        help="Language code for audio datasets (e.g., en_us, es_419)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./demo_data",
        help="Output directory (default: ./demo_data)",
    )
    parser.add_argument(
        "--download-images",
        action="store_true",
        help="Download actual image files (not just metadata)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available datasets without downloading",
    )

    args = parser.parse_args()

    # Handle --list
    if args.list:
        list_available_datasets()
        return

    # Check dependencies before fetching
    check_dependencies()

    # Determine which modalities to fetch
    fetch_text = args.text or args.all
    fetch_audio = args.audio or args.all
    fetch_image = args.image or args.all
    fetch_video = args.video or args.all

    if not any([fetch_text, fetch_audio, fetch_image, fetch_video]):
        parser.print_help()
        print("\n❌ Please specify at least one modality (--text, --audio, --image, --video, or --all)")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else None

    print("\n" + "=" * 60)
    print("NeMo Curator Demo Data Fetcher")
    print("=" * 60)

    results: dict[str, list[dict[str, Any]]] = {}

    if fetch_text:
        dataset_name = args.dataset if args.dataset in DATASET_REGISTRY["text"] else "tiny_stories"
        results["text"] = fetch_text_dataset(
            dataset_name=dataset_name,
            num_samples=args.samples,
            output_dir=output_dir,
        )

    if fetch_audio:
        dataset_name = args.dataset if args.dataset in DATASET_REGISTRY["audio"] else "fleurs"
        results["audio"] = fetch_audio_dataset(
            dataset_name=dataset_name,
            language=args.language,
            num_samples=args.samples,
            output_dir=output_dir,
        )

    if fetch_image:
        dataset_name = args.dataset if args.dataset in DATASET_REGISTRY["image"] else "flickr30k_sample"
        results["image"] = fetch_image_dataset(
            dataset_name=dataset_name,
            num_samples=args.samples,
            output_dir=output_dir,
            download_images=args.download_images,
        )

    if fetch_video:
        dataset_name = args.dataset if args.dataset in DATASET_REGISTRY["video"] else "webvid_sample"
        results["video"] = fetch_video_dataset(
            dataset_name=dataset_name,
            num_samples=args.samples,
            output_dir=output_dir,
        )

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for modality, samples in results.items():
        print(f"  {modality}: {len(samples)} samples")
    if output_dir:
        print(f"\n📁 Output saved to: {output_dir}")
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
