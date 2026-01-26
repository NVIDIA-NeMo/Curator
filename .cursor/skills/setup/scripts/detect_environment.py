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

"""Detect environment for NeMo Curator installation.

This script analyzes the current environment and recommends appropriate
NeMo Curator installation options.

Examples:
    # JSON output for agent parsing
    python detect_environment.py

    # Human-readable output
    python detect_environment.py --human

    # Check specific modality requirements
    python detect_environment.py --modality video
"""

import argparse
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field


@dataclass
class EnvironmentInfo:
    """Environment detection results."""

    python_version: str
    cuda_version: str | None = None
    cuda_driver: str | None = None
    gpu_name: str | None = None
    gpu_memory_gb: float | None = None
    gpu_count: int = 0
    ffmpeg_installed: bool = False
    ffmpeg_nvenc: bool = False
    existing_packages: list[str] = field(default_factory=list)
    recommended_extras: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def detect_python() -> str:
    """Get Python version."""
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def detect_gpu() -> tuple[str | None, str | None, str | None, float | None, int]:
    """Detect GPU information using nvidia-smi.

    Returns:
        Tuple of (cuda_version, driver_version, gpu_name, memory_gb, gpu_count)
    """
    try:
        # Get GPU info
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode != 0:
            return None, None, None, None, 0

        lines = result.stdout.strip().split("\n")
        gpu_count = len(lines)

        if not lines or not lines[0].strip():
            return None, None, None, None, 0

        # Parse first GPU
        parts = lines[0].split(", ")
        gpu_name = parts[0].strip() if len(parts) > 0 else None
        memory_mb = float(parts[1].strip()) if len(parts) > 1 else None
        driver_version = parts[2].strip() if len(parts) > 2 else None

        memory_gb = memory_mb / 1024 if memory_mb else None

        # Infer CUDA version from driver
        # Driver 535+ supports CUDA 12.x, 525+ supports 12.0
        cuda_version = None
        if driver_version:
            driver_major = int(driver_version.split(".")[0])
            if driver_major >= 535:
                cuda_version = "12.x"
            elif driver_major >= 525:
                cuda_version = "12.0"
            elif driver_major >= 450:
                cuda_version = "11.x"

        return cuda_version, driver_version, gpu_name, memory_gb, gpu_count

    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        return None, None, None, None, 0


def detect_ffmpeg() -> tuple[bool, bool]:
    """Check FFmpeg installation and NVENC support.

    Returns:
        Tuple of (ffmpeg_installed, has_nvenc)
    """
    try:
        # Check if ffmpeg exists
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode != 0:
            return False, False

        # Check for NVENC encoder
        encoder_result = subprocess.run(
            ["ffmpeg", "-encoders"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        has_nvenc = "h264_nvenc" in encoder_result.stdout

        return True, has_nvenc

    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False, False


def detect_existing_packages() -> list[str]:
    """Check for existing NeMo Curator installation."""
    packages = []

    try:
        import nemo_curator

        packages.append(f"nemo-curator=={nemo_curator.__version__}")
    except ImportError:
        pass

    # Check for key dependencies
    dep_checks = [
        ("cudf", "cudf"),
        ("vllm", "vllm"),
        ("fasttext", "fasttext"),
        ("av", "PyAV"),
        ("cv2", "opencv-python"),
        ("flash_attn", "flash-attn"),
        ("nemo.collections.asr", "nemo-toolkit[asr]"),
    ]

    for module, package_name in dep_checks:
        try:
            __import__(module)
            packages.append(package_name)
        except ImportError:
            pass

    return packages


def recommend_extras(
    cuda_version: str | None,
    modality: str | None = None,
) -> list[str]:
    """Recommend installation extras based on environment."""
    extras = []

    if modality:
        # Specific modality requested
        suffix = "_cuda12" if cuda_version and "12" in cuda_version else "_cpu"
        extras.append(f"{modality}{suffix}")
    else:
        # Recommend based on GPU availability
        if cuda_version and "12" in cuda_version:
            extras = ["text_cuda12", "video_cuda12", "image_cuda12", "audio_cuda12"]
        elif cuda_version:
            # CUDA 11.x - limited support
            extras = ["text_cpu", "video_cpu", "image_cpu", "audio_cpu"]
        else:
            extras = ["text_cpu", "video_cpu", "image_cpu", "audio_cpu"]

    return extras


def generate_warnings(info: EnvironmentInfo, modality: str | None = None) -> list[str]:
    """Generate warnings based on environment."""
    warnings = []

    # Python version
    py_major, py_minor, _ = info.python_version.split(".")
    if int(py_minor) < 10:
        warnings.append(f"Python {info.python_version} not supported. Requires 3.10+")

    # GPU warnings
    if not info.cuda_version:
        warnings.append("No CUDA GPU detected. Will use CPU-only packages.")

    if info.gpu_memory_gb and info.gpu_memory_gb < 16:
        warnings.append(f"GPU has {info.gpu_memory_gb:.1f}GB VRAM. Some stages need 16GB+")

    # Video-specific
    if modality == "video" or not modality:
        if not info.ffmpeg_installed:
            warnings.append("FFmpeg not installed. Required for video processing.")
        elif not info.ffmpeg_nvenc:
            warnings.append("FFmpeg lacks NVENC. GPU encoding unavailable.")

    return warnings


def detect_environment(modality: str | None = None) -> EnvironmentInfo:
    """Run full environment detection."""
    python_version = detect_python()
    cuda_version, cuda_driver, gpu_name, gpu_memory_gb, gpu_count = detect_gpu()
    ffmpeg_installed, ffmpeg_nvenc = detect_ffmpeg()
    existing_packages = detect_existing_packages()
    recommended_extras = recommend_extras(cuda_version, modality)

    info = EnvironmentInfo(
        python_version=python_version,
        cuda_version=cuda_version,
        cuda_driver=cuda_driver,
        gpu_name=gpu_name,
        gpu_memory_gb=gpu_memory_gb,
        gpu_count=gpu_count,
        ffmpeg_installed=ffmpeg_installed,
        ffmpeg_nvenc=ffmpeg_nvenc,
        existing_packages=existing_packages,
        recommended_extras=recommended_extras,
    )

    info.warnings = generate_warnings(info, modality)

    return info


def print_human_readable(info: EnvironmentInfo) -> None:
    """Print human-readable environment report."""
    print("\n" + "=" * 60)
    print("NeMo Curator Environment Detection")
    print("=" * 60 + "\n")

    # Python
    print(f"Python: {info.python_version}")

    # GPU
    if info.gpu_name:
        print(f"GPU: {info.gpu_name} ({info.gpu_memory_gb:.1f} GB)")
        print(f"CUDA: {info.cuda_version} (Driver: {info.cuda_driver})")
        print(f"GPU Count: {info.gpu_count}")
    else:
        print("GPU: Not detected")

    # FFmpeg
    if info.ffmpeg_installed:
        nvenc_status = "with NVENC" if info.ffmpeg_nvenc else "without NVENC"
        print(f"FFmpeg: Installed ({nvenc_status})")
    else:
        print("FFmpeg: Not installed")

    # Existing packages
    if info.existing_packages:
        print(f"\nExisting packages: {', '.join(info.existing_packages)}")

    # Recommendations
    print(f"\nRecommended extras: {', '.join(info.recommended_extras)}")

    # Warnings
    if info.warnings:
        print("\n⚠ Warnings:")
        for warning in info.warnings:
            print(f"  - {warning}")

    # Install command
    print("\n" + "-" * 60)
    print("Suggested install command:")
    extras = ",".join(info.recommended_extras)
    if "video_cuda12" in info.recommended_extras:
        print(f"  uv pip install --no-build-isolation nemo-curator[{extras}]")
    else:
        print(f"  uv pip install nemo-curator[{extras}]")
    print()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--modality",
        choices=["text", "video", "image", "audio"],
        help="Check requirements for specific modality",
    )
    parser.add_argument(
        "--human",
        action="store_true",
        help="Output human-readable format instead of JSON",
    )

    args = parser.parse_args()

    info = detect_environment(args.modality)

    if args.human:
        print_human_readable(info)
    else:
        print(json.dumps(asdict(info), indent=2))


if __name__ == "__main__":
    main()
