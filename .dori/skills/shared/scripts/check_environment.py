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

"""Check NeMo Curator environment and available features.

This script verifies that NeMo Curator is properly installed and reports
what features are available in the current environment.

Usage:
    python check_environment.py
    python check_environment.py --output json
    python check_environment.py --check-gpu
"""

import argparse
import json
import sys


def check_nemo_curator() -> dict:
    """Check if NeMo Curator is installed and get version."""
    try:
        import nemo_curator
        version = getattr(nemo_curator, "__version__", "unknown")
        return {"installed": True, "version": version}
    except ImportError as e:
        return {"installed": False, "error": str(e)}


def check_gpu() -> dict:
    """Check GPU availability."""
    result = {
        "cuda_available": False,
        "gpu_count": 0,
        "gpus": [],
    }
    
    try:
        import torch
        result["pytorch_version"] = torch.__version__
        result["cuda_available"] = torch.cuda.is_available()
        
        if result["cuda_available"]:
            result["gpu_count"] = torch.cuda.device_count()
            result["cuda_version"] = torch.version.cuda
            
            for i in range(result["gpu_count"]):
                props = torch.cuda.get_device_properties(i)
                result["gpus"].append({
                    "index": i,
                    "name": props.name,
                    "memory_gb": round(props.total_memory / (1024**3), 1),
                    "compute_capability": f"{props.major}.{props.minor}",
                })
    except ImportError:
        result["pytorch_installed"] = False
    except Exception as e:
        result["error"] = str(e)
    
    return result


def check_text_filters() -> dict:
    """Check if text filter stages are available."""
    filters = {}
    
    filter_classes = [
        "WordCountFilter",
        "NonAlphaNumericFilter",
        "SymbolsToWordsFilter",
        "UrlsFilter",
        "MeanWordLengthFilter",
        "RepeatedLinesFilter",
        "RepeatedParagraphsFilter",
        "PunctuationFilter",
        "CommonEnglishWordsFilter",
    ]
    
    try:
        from nemo_curator.stages.text import filters as filter_module
        for name in filter_classes:
            filters[name] = hasattr(filter_module, name)
        filters["_module_available"] = True
    except ImportError as e:
        filters["_module_available"] = False
        filters["_error"] = str(e)
    
    return filters


def check_classifiers() -> dict:
    """Check if classifier stages are available."""
    classifiers = {}
    
    classifier_info = [
        ("QualityClassifier", "nemo_curator.stages.text.classifiers.quality"),
        ("FineWebEduClassifier", "nemo_curator.stages.text.classifiers.fineweb_edu"),
        ("DomainClassifier", "nemo_curator.stages.text.classifiers.domain"),
        ("AegisClassifier", "nemo_curator.stages.text.classifiers.aegis"),
        ("ContentTypeClassifier", "nemo_curator.stages.text.classifiers.content_type"),
    ]
    
    for name, module_path in classifier_info:
        try:
            module = __import__(module_path, fromlist=[name])
            classifiers[name] = hasattr(module, name)
        except ImportError:
            classifiers[name] = False
    
    return classifiers


def check_deduplication() -> dict:
    """Check if deduplication stages are available."""
    dedup = {}
    
    try:
        from nemo_curator.stages.deduplication.fuzzy import FuzzyDeduplicationWorkflow
        dedup["FuzzyDeduplicationWorkflow"] = True
    except ImportError:
        dedup["FuzzyDeduplicationWorkflow"] = False
    
    return dedup


def check_video() -> dict:
    """Check if video stages are available."""
    video = {}
    
    stage_info = [
        ("VideoReader", "nemo_curator.stages.video.io.video_reader"),
        ("TransNetV2ClipExtractionStage", "nemo_curator.stages.video.clipping.transnetv2_extraction"),
        ("CaptionGenerationStage", "nemo_curator.stages.video.caption.caption_generation"),
        ("CosmosEmbed1EmbeddingStage", "nemo_curator.stages.video.embedding.cosmos_embed1"),
    ]
    
    for name, module_path in stage_info:
        try:
            module = __import__(module_path, fromlist=[name])
            video[name] = hasattr(module, name)
        except ImportError:
            video[name] = False
    
    return video


def check_image() -> dict:
    """Check if image stages are available."""
    image = {}
    
    stage_info = [
        ("ImageEmbeddingStage", "nemo_curator.stages.image.embedders.clip_embedder"),
        ("ImageAestheticFilterStage", "nemo_curator.stages.image.filters.aesthetic_filter"),
        ("ImageNSFWFilterStage", "nemo_curator.stages.image.filters.nsfw_filter"),
    ]
    
    for name, module_path in stage_info:
        try:
            module = __import__(module_path, fromlist=[name])
            image[name] = hasattr(module, name)
        except ImportError:
            image[name] = False
    
    return image


def check_audio() -> dict:
    """Check if audio stages are available."""
    audio = {}
    
    stage_info = [
        ("InferenceAsrNemoStage", "nemo_curator.stages.audio.inference.asr_nemo"),
        ("GetPairwiseWerStage", "nemo_curator.stages.audio.metrics.get_wer"),
        ("PreserveByValueStage", "nemo_curator.stages.audio.common"),
    ]
    
    for name, module_path in stage_info:
        try:
            module = __import__(module_path, fromlist=[name])
            audio[name] = hasattr(module, name)
        except ImportError:
            audio[name] = False
    
    return audio


def check_pipeline() -> dict:
    """Check if Pipeline and core classes are available."""
    pipeline = {}
    
    try:
        from nemo_curator.pipeline import Pipeline
        pipeline["Pipeline"] = True
    except ImportError:
        pipeline["Pipeline"] = False
    
    try:
        from nemo_curator.tasks import DocumentBatch
        pipeline["DocumentBatch"] = True
    except ImportError:
        pipeline["DocumentBatch"] = False
    
    return pipeline


def run_all_checks(include_gpu: bool = True) -> dict:
    """Run all environment checks."""
    results = {
        "nemo_curator": check_nemo_curator(),
        "pipeline": check_pipeline(),
        "text_filters": check_text_filters(),
        "classifiers": check_classifiers(),
        "deduplication": check_deduplication(),
        "video": check_video(),
        "image": check_image(),
        "audio": check_audio(),
    }
    
    if include_gpu:
        results["gpu"] = check_gpu()
    
    # Compute summary
    summary = {
        "nemo_curator_installed": results["nemo_curator"]["installed"],
        "gpu_available": results.get("gpu", {}).get("cuda_available", False),
        "text_processing": results["text_filters"].get("_module_available", False),
        "classifiers_available": any(v for k, v in results["classifiers"].items() if isinstance(v, bool)),
        "video_available": any(v for v in results["video"].values() if isinstance(v, bool)),
        "image_available": any(v for v in results["image"].values() if isinstance(v, bool)),
        "audio_available": any(v for v in results["audio"].values() if isinstance(v, bool)),
    }
    results["summary"] = summary
    
    return results


def format_human_readable(results: dict) -> str:
    """Format results for human reading."""
    lines = [
        "=" * 60,
        "NeMo Curator Environment Check",
        "=" * 60,
        "",
    ]
    
    # NeMo Curator
    nc = results["nemo_curator"]
    if nc["installed"]:
        lines.append(f"NeMo Curator: Installed (v{nc['version']})")
    else:
        lines.append(f"NeMo Curator: NOT INSTALLED - {nc.get('error', 'unknown error')}")
    
    # GPU
    if "gpu" in results:
        gpu = results["gpu"]
        if gpu["cuda_available"]:
            lines.append(f"GPU: Available ({gpu['gpu_count']} GPU(s))")
            for g in gpu["gpus"]:
                lines.append(f"  - {g['name']} ({g['memory_gb']}GB)")
        else:
            lines.append("GPU: Not available (CPU only)")
    
    lines.append("")
    
    # Features
    lines.append("Available Features:")
    
    # Text filters
    tf = results["text_filters"]
    if tf.get("_module_available"):
        available = sum(1 for k, v in tf.items() if not k.startswith("_") and v)
        lines.append(f"  Text Filters: {available} available")
    else:
        lines.append("  Text Filters: NOT AVAILABLE")
    
    # Classifiers
    cl = results["classifiers"]
    available = sum(1 for v in cl.values() if v)
    if available > 0:
        lines.append(f"  ML Classifiers: {available} available (GPU required)")
    else:
        lines.append("  ML Classifiers: NOT AVAILABLE")
    
    # Deduplication
    dd = results["deduplication"]
    if dd.get("FuzzyDeduplicationWorkflow"):
        lines.append("  Fuzzy Deduplication: Available (GPU recommended)")
    else:
        lines.append("  Fuzzy Deduplication: NOT AVAILABLE")
    
    # Video
    vid = results["video"]
    available = sum(1 for v in vid.values() if v)
    if available > 0:
        lines.append(f"  Video Processing: {available} stages available (GPU required)")
    else:
        lines.append("  Video Processing: NOT AVAILABLE")
    
    # Image
    img = results["image"]
    available = sum(1 for v in img.values() if v)
    if available > 0:
        lines.append(f"  Image Processing: {available} stages available (GPU required)")
    else:
        lines.append("  Image Processing: NOT AVAILABLE")
    
    # Audio
    aud = results["audio"]
    available = sum(1 for v in aud.values() if v)
    if available > 0:
        lines.append(f"  Audio Processing: {available} stages available (GPU required)")
    else:
        lines.append("  Audio Processing: NOT AVAILABLE")
    
    lines.append("")
    
    # Recommendations
    lines.append("Recommendations:")
    summary = results["summary"]
    
    if not summary["nemo_curator_installed"]:
        lines.append("  - Install NeMo Curator: pip install nemo-curator")
    elif not summary["gpu_available"]:
        lines.append("  - Text filtering works on CPU")
        lines.append("  - GPU needed for: classifiers, dedup, video, image, audio")
        lines.append("  - Use Docker with GPU: docker run --gpus all nvcr.io/nvidia/nemo-curator:latest")
    else:
        lines.append("  - Environment ready for all features")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output", "-o",
        choices=["human", "json"],
        default="human",
        help="Output format (default: human)",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Skip GPU check",
    )
    parser.add_argument(
        "--check-gpu",
        action="store_true",
        help="Only check GPU availability",
    )
    
    args = parser.parse_args()
    
    if args.check_gpu:
        results = {"gpu": check_gpu()}
    else:
        results = run_all_checks(include_gpu=not args.no_gpu)
    
    if args.output == "json":
        print(json.dumps(results, indent=2))
    else:
        if args.check_gpu:
            gpu = results["gpu"]
            if gpu["cuda_available"]:
                print(f"GPU available: {gpu['gpu_count']} GPU(s)")
                for g in gpu["gpus"]:
                    print(f"  - {g['name']} ({g['memory_gb']}GB)")
            else:
                print("No GPU available")
        else:
            print(format_human_readable(results))
    
    # Exit with error if NeMo Curator not installed
    if not args.check_gpu and not results.get("summary", {}).get("nemo_curator_installed", False):
        sys.exit(1)


if __name__ == "__main__":
    main()
