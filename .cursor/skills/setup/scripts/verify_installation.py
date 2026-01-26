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

"""Verify NeMo Curator installation.

This script checks that NeMo Curator and its dependencies are properly installed.

Examples:
    # Full verification
    python verify_installation.py

    # Core only
    python verify_installation.py --core

    # Specific modality
    python verify_installation.py --text
    python verify_installation.py --video
"""

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field


@dataclass
class VerificationResult:
    """Result of a verification check."""

    name: str
    passed: bool
    message: str
    details: str = ""


@dataclass
class VerificationReport:
    """Complete verification report."""

    results: list[VerificationResult] = field(default_factory=list)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    @property
    def total(self) -> int:
        return len(self.results)


def check_import(module: str, name: str | None = None) -> VerificationResult:
    """Check if a module can be imported."""
    name = name or module
    try:
        __import__(module)
        return VerificationResult(name, True, f"{name} available")
    except ImportError as e:
        return VerificationResult(name, False, f"{name} not available", str(e))


def check_version(module: str, name: str | None = None) -> VerificationResult:
    """Check module version."""
    name = name or module
    try:
        mod = __import__(module)
        version = getattr(mod, "__version__", "unknown")
        return VerificationResult(name, True, f"{name} version: {version}")
    except ImportError as e:
        return VerificationResult(name, False, f"{name} not available", str(e))


def verify_core() -> list[VerificationResult]:
    """Verify core NeMo Curator installation."""
    results = []

    # NeMo Curator version
    results.append(check_version("nemo_curator", "NeMo Curator"))

    # Core modules
    try:
        from nemo_curator.pipeline import Pipeline
        from nemo_curator.stages.base import ProcessingStage
        from nemo_curator.tasks import DocumentBatch, Task

        results.append(VerificationResult("Core modules", True, "Pipeline, Task, Stage available"))
    except ImportError as e:
        results.append(VerificationResult("Core modules", False, "Core import failed", str(e)))

    # Ray
    results.append(check_version("ray", "Ray"))

    return results


def verify_gpu() -> list[VerificationResult]:
    """Verify GPU availability."""
    results = []

    try:
        import torch

        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            cuda_version = torch.version.cuda
            results.append(
                VerificationResult(
                    "GPU",
                    True,
                    f"GPU available: {device_name}",
                    f"Memory: {memory_gb:.1f} GB, CUDA: {cuda_version}",
                )
            )
        else:
            results.append(VerificationResult("GPU", False, "No GPU detected", "Will use CPU only"))
    except ImportError:
        results.append(VerificationResult("PyTorch", False, "PyTorch not installed"))

    # RAPIDS
    try:
        import cudf

        results.append(check_version("cudf", "cuDF (RAPIDS)"))
    except ImportError:
        results.append(
            VerificationResult("cuDF (RAPIDS)", False, "cuDF not available", "GPU deduplication unavailable")
        )

    return results


def verify_text() -> list[VerificationResult]:
    """Verify text curation modules."""
    results = []

    # Filters
    try:
        from nemo_curator.stages.text.filters import WordCountFilter

        results.append(VerificationResult("Text filters", True, "25+ filters available"))
    except ImportError as e:
        results.append(VerificationResult("Text filters", False, "Filters not available", str(e)))

    # Classifiers
    try:
        from nemo_curator.stages.text.classifiers import QualityClassifier

        results.append(VerificationResult("Text classifiers", True, "Classifiers available"))
    except ImportError as e:
        results.append(VerificationResult("Text classifiers", False, "Classifiers not available", str(e)))

    # FastText
    results.append(check_import("fasttext", "FastText"))

    # vLLM
    results.append(check_import("vllm", "vLLM"))

    # Deduplication
    try:
        from nemo_curator.stages.deduplication.fuzzy import FuzzyDeduplicationWorkflow

        results.append(VerificationResult("Fuzzy deduplication", True, "FuzzyDeduplicationWorkflow available"))
    except ImportError as e:
        results.append(VerificationResult("Fuzzy deduplication", False, "Not available", str(e)))

    return results


def verify_video() -> list[VerificationResult]:
    """Verify video curation modules."""
    results = []

    # Video IO
    try:
        from nemo_curator.stages.video.io.video_reader import VideoReader

        results.append(VerificationResult("Video IO", True, "VideoReader available"))
    except ImportError as e:
        results.append(VerificationResult("Video IO", False, "Not available", str(e)))

    # PyAV
    results.append(check_version("av", "PyAV"))

    # OpenCV
    results.append(check_version("cv2", "OpenCV"))

    # PyNvVideoCodec
    results.append(check_import("PyNvVideoCodec", "NVIDIA Video Codec"))

    # flash-attn
    results.append(check_import("flash_attn", "flash-attn"))

    # FFmpeg
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            version_line = result.stdout.split("\n")[0]
            results.append(VerificationResult("FFmpeg", True, version_line[:50]))
        else:
            results.append(VerificationResult("FFmpeg", False, "FFmpeg error"))
    except FileNotFoundError:
        results.append(VerificationResult("FFmpeg", False, "FFmpeg not installed"))
    except subprocess.TimeoutExpired:
        results.append(VerificationResult("FFmpeg", False, "FFmpeg timeout"))

    return results


def verify_audio() -> list[VerificationResult]:
    """Verify audio curation modules."""
    results = []

    # Audio stages
    try:
        from nemo_curator.stages.audio.inference import InferenceAsrNemoStage

        results.append(VerificationResult("Audio stages", True, "ASR stage available"))
    except ImportError as e:
        results.append(VerificationResult("Audio stages", False, "Not available", str(e)))

    # NeMo Toolkit
    try:
        import nemo.collections.asr as nemo_asr

        results.append(VerificationResult("NeMo ASR", True, "NeMo ASR available"))
    except ImportError as e:
        results.append(VerificationResult("NeMo ASR", False, "Not available", str(e)))

    return results


def verify_image() -> list[VerificationResult]:
    """Verify image curation modules."""
    results = []

    # Image stages
    try:
        from nemo_curator.stages.image.embedders import ImageEmbeddingStage

        results.append(VerificationResult("Image stages", True, "Embedding stage available"))
    except ImportError as e:
        results.append(VerificationResult("Image stages", False, "Not available", str(e)))

    # torchvision
    results.append(check_version("torchvision", "torchvision"))

    # NVIDIA DALI
    try:
        import nvidia.dali as dali

        results.append(VerificationResult("NVIDIA DALI", True, f"DALI version: {dali.__version__}"))
    except ImportError:
        results.append(VerificationResult("NVIDIA DALI", False, "Not available (image_cuda12)"))

    return results


def print_report(report: VerificationReport, verbose: bool = False) -> None:
    """Print verification report."""
    print("\n" + "=" * 60)
    print("NeMo Curator Installation Verification")
    print("=" * 60 + "\n")

    for result in report.results:
        status = "✓" if result.passed else "✗"
        print(f"  {status} {result.message}")
        if verbose and result.details:
            print(f"      {result.details}")

    print("\n" + "-" * 60)
    print(f"Summary: {report.passed}/{report.total} checks passed")

    if report.failed > 0:
        print(f"\n⚠ {report.failed} check(s) failed. See above for details.")
        print("  See references/TROUBLESHOOTING.md for solutions.")
    else:
        print("\n✓ All checks passed! NeMo Curator is ready to use.")

    print()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--core", action="store_true", help="Verify core only")
    parser.add_argument("--gpu", action="store_true", help="Verify GPU and RAPIDS")
    parser.add_argument("--text", action="store_true", help="Verify text modules")
    parser.add_argument("--video", action="store_true", help="Verify video modules")
    parser.add_argument("--audio", action="store_true", help="Verify audio modules")
    parser.add_argument("--image", action="store_true", help="Verify image modules")
    parser.add_argument("--all", action="store_true", help="Verify everything (default)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    # Default to --all if no specific flags
    if not any([args.core, args.gpu, args.text, args.video, args.audio, args.image, args.all]):
        args.all = True

    report = VerificationReport()

    # Run selected verifications
    if args.all or args.core:
        report.results.extend(verify_core())

    if args.all or args.gpu:
        report.results.extend(verify_gpu())

    if args.all or args.text:
        report.results.extend(verify_text())

    if args.all or args.video:
        report.results.extend(verify_video())

    if args.all or args.audio:
        report.results.extend(verify_audio())

    if args.all or args.image:
        report.results.extend(verify_image())

    # Output
    if args.json:
        output = {
            "passed": report.passed,
            "failed": report.failed,
            "total": report.total,
            "results": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "message": r.message,
                    "details": r.details,
                }
                for r in report.results
            ],
        }
        print(json.dumps(output, indent=2))
    else:
        print_report(report, args.verbose)

    # Exit code
    sys.exit(0 if report.failed == 0 else 1)


if __name__ == "__main__":
    main()
