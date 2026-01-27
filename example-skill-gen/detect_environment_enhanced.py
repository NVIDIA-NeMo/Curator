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

"""Enhanced environment detection for NeMo Curator.

This script analyzes the current environment including Docker images
and recommends the best way to run NeMo Curator.

Examples:
    # Full analysis with recommendations
    python detect_environment_enhanced.py

    # JSON output for agent parsing
    python detect_environment_enhanced.py --json

    # Quick check - just show recommendation
    python detect_environment_enhanced.py --quick
"""

import argparse
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta


@dataclass
class DockerImageInfo:
    """Information about a Docker image."""

    name: str
    tag: str
    size_gb: float
    created: str
    image_id: str
    age_days: int | None = None


@dataclass
class DockerInfo:
    """Docker environment information."""

    installed: bool = False
    running: bool = False
    images: list[DockerImageInfo] = field(default_factory=list)
    recommended_image: str | None = None
    recommendation_reason: str | None = None


@dataclass
class EnvironmentInfo:
    """Environment detection results."""

    python_version: str
    platform: str = ""
    platform_supported: bool = True
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
    docker: DockerInfo = field(default_factory=DockerInfo)
    recommended_scenario: str = ""
    scenario_command: str = ""
    scenario_explanation: str = ""


def detect_python() -> str:
    """Get Python version."""
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def detect_platform() -> tuple[str, bool]:
    """Detect platform and whether it's supported.
    
    Returns:
        Tuple of (platform_name, is_supported)
    """
    platform = sys.platform
    supported = platform == "linux"
    return platform, supported


def detect_docker() -> DockerInfo:
    """Detect Docker installation and available images.
    
    Returns:
        DockerInfo with installation status and available images
    """
    info = DockerInfo()
    
    # Check if docker is installed
    try:
        result = subprocess.run(
            ["docker", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        info.installed = result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return info
    
    if not info.installed:
        return info
    
    # Check if docker daemon is running
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        info.running = result.returncode == 0
    except subprocess.TimeoutExpired:
        return info
    
    if not info.running:
        return info
    
    # Get NeMo Curator related images
    try:
        result = subprocess.run(
            ["docker", "images", "--format", "{{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}\t{{.ID}}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        
        if result.returncode != 0:
            return info
        
        curator_keywords = ["nemo", "curator", "nvidia"]
        
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            
            parts = line.split("\t")
            if len(parts) < 5:
                continue
            
            repo, tag, size_str, created_str, image_id = parts[:5]
            
            # Check if this is a relevant image
            repo_lower = repo.lower()
            if not any(kw in repo_lower for kw in curator_keywords):
                continue
            
            # Parse size (e.g., "52.8GB", "212MB")
            size_gb = 0.0
            if "GB" in size_str:
                size_gb = float(size_str.replace("GB", "").strip())
            elif "MB" in size_str:
                size_gb = float(size_str.replace("MB", "").strip()) / 1024
            
            # Parse creation date and calculate age
            age_days = None
            try:
                # Format: "2025-01-13 10:30:45 -0700 MST"
                date_part = " ".join(created_str.split()[:2])
                created_dt = datetime.strptime(date_part, "%Y-%m-%d %H:%M:%S")
                age_days = (datetime.now() - created_dt).days
            except (ValueError, IndexError):
                pass
            
            info.images.append(DockerImageInfo(
                name=repo,
                tag=tag,
                size_gb=size_gb,
                created=created_str,
                image_id=image_id[:12],
                age_days=age_days,
            ))
        
        # Recommend best image
        if info.images:
            # Prioritize: nemo-curator-local > nemo-curator-env > others
            # Also prefer newer images
            priority_order = ["nemo-curator-local", "nemo-curator-env", "nvcr.io/nvidia/nemo-curator"]
            
            best_image = None
            for priority_name in priority_order:
                for img in info.images:
                    if priority_name in img.name:
                        best_image = img
                        break
                if best_image:
                    break
            
            # Fallback to largest image (likely most complete)
            if not best_image and info.images:
                best_image = max(info.images, key=lambda x: x.size_gb)
            
            if best_image:
                info.recommended_image = f"{best_image.name}:{best_image.tag}"
                
                # Check age
                if best_image.age_days is not None:
                    if best_image.age_days > 90:
                        info.recommendation_reason = f"Found but {best_image.age_days} days old - consider updating"
                    elif best_image.age_days > 30:
                        info.recommendation_reason = f"Found, {best_image.age_days} days old"
                    else:
                        info.recommendation_reason = "Found and recent"
                else:
                    info.recommendation_reason = "Found"
    
    except subprocess.TimeoutExpired:
        pass
    
    return info


def detect_gpu() -> tuple[str | None, str | None, str | None, float | None, int]:
    """Detect GPU information using nvidia-smi.

    Returns:
        Tuple of (cuda_version, driver_version, gpu_name, memory_gb, gpu_count)
    """
    try:
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

        parts = lines[0].split(", ")
        gpu_name = parts[0].strip() if len(parts) > 0 else None
        memory_mb = float(parts[1].strip()) if len(parts) > 1 else None
        driver_version = parts[2].strip() if len(parts) > 2 else None

        memory_gb = memory_mb / 1024 if memory_mb else None

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
    """Check FFmpeg installation and NVENC support."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode != 0:
            return False, False

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

    dep_checks = [
        ("cudf", "cudf"),
        ("vllm", "vllm"),
        ("fasttext", "fasttext"),
        ("av", "PyAV"),
        ("cv2", "opencv-python"),
        ("flash_attn", "flash-attn"),
    ]

    for module, package_name in dep_checks:
        try:
            __import__(module)
            packages.append(package_name)
        except ImportError:
            pass

    return packages


def recommend_scenario(info: EnvironmentInfo) -> tuple[str, str, str]:
    """Recommend the best scenario based on environment.
    
    Returns:
        Tuple of (scenario_name, command, explanation)
    """
    # Scenario 1: Native Linux with existing install
    if info.platform_supported and info.existing_packages:
        if any("nemo-curator" in p for p in info.existing_packages):
            return (
                "native_existing",
                "python your_script.py",
                "NeMo Curator is already installed locally. You're ready to go!"
            )
    
    # Scenario 2: Native Linux without install
    if info.platform_supported:
        extras = ",".join(info.recommended_extras) if info.recommended_extras else "text_cpu"
        return (
            "native_install",
            f"uv pip install nemo-curator[{extras}]",
            "Linux detected. Install NeMo Curator directly."
        )
    
    # Scenario 3: Non-Linux with Docker image available
    if info.docker.running and info.docker.recommended_image:
        img = info.docker.recommended_image
        age_note = ""
        if info.docker.recommendation_reason and "days old" in info.docker.recommendation_reason:
            age_note = f" ({info.docker.recommendation_reason})"
        
        return (
            "docker_existing",
            f"docker run --rm -v $(pwd):/workspace -w /workspace {img} python your_script.py",
            f"Use your existing Docker image: {img}{age_note}"
        )
    
    # Scenario 4: Non-Linux with Docker but no image
    if info.docker.running:
        return (
            "docker_pull",
            "docker pull nvcr.io/nvidia/nemo-curator:latest",
            "Docker is running but no NeMo Curator image found. Pull the official image."
        )
    
    # Scenario 5: Non-Linux with Docker installed but not running
    if info.docker.installed and not info.docker.running:
        return (
            "docker_start",
            "# Start Docker Desktop, then run:\ndocker pull nvcr.io/nvidia/nemo-curator:latest",
            "Docker is installed but not running. Start Docker Desktop first."
        )
    
    # Scenario 6: Non-Linux without Docker
    return (
        "docker_install",
        "# Install Docker Desktop from https://docker.com\n# Then: docker pull nvcr.io/nvidia/nemo-curator:latest",
        f"NeMo Curator requires Linux. On {info.platform}, use Docker."
    )


def generate_warnings(info: EnvironmentInfo, modality: str | None = None) -> list[str]:
    """Generate warnings based on environment."""
    warnings = []

    # Platform warning
    if not info.platform_supported:
        warnings.append(f"Platform '{info.platform}' not supported. Use Docker.")

    # Python version
    py_major, py_minor, _ = info.python_version.split(".")
    if int(py_minor) < 10:
        warnings.append(f"Python {info.python_version} not supported. Requires 3.10+")

    # GPU warnings
    if info.platform_supported and not info.cuda_version:
        warnings.append("No CUDA GPU detected. Will use CPU-only packages.")

    if info.gpu_memory_gb and info.gpu_memory_gb < 16:
        warnings.append(f"GPU has {info.gpu_memory_gb:.1f}GB VRAM. Some stages need 16GB+")

    # Docker image age warning
    if info.docker.recommended_image:
        for img in info.docker.images:
            if f"{img.name}:{img.tag}" == info.docker.recommended_image:
                if img.age_days and img.age_days > 90:
                    warnings.append(f"Docker image is {img.age_days} days old. Consider: docker pull {info.docker.recommended_image}")
                break

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
    platform, platform_supported = detect_platform()
    cuda_version, cuda_driver, gpu_name, gpu_memory_gb, gpu_count = detect_gpu()
    ffmpeg_installed, ffmpeg_nvenc = detect_ffmpeg()
    existing_packages = detect_existing_packages()
    docker = detect_docker()

    # Recommend extras based on GPU
    recommended_extras = []
    if modality:
        suffix = "_cuda12" if cuda_version and "12" in cuda_version else "_cpu"
        recommended_extras.append(f"{modality}{suffix}")
    else:
        suffix = "_cuda12" if cuda_version and "12" in cuda_version else "_cpu"
        recommended_extras = [f"text{suffix}"]

    info = EnvironmentInfo(
        python_version=python_version,
        platform=platform,
        platform_supported=platform_supported,
        cuda_version=cuda_version,
        cuda_driver=cuda_driver,
        gpu_name=gpu_name,
        gpu_memory_gb=gpu_memory_gb,
        gpu_count=gpu_count,
        ffmpeg_installed=ffmpeg_installed,
        ffmpeg_nvenc=ffmpeg_nvenc,
        existing_packages=existing_packages,
        recommended_extras=recommended_extras,
        docker=docker,
    )

    info.warnings = generate_warnings(info, modality)
    
    # Get recommendation
    scenario, command, explanation = recommend_scenario(info)
    info.recommended_scenario = scenario
    info.scenario_command = command
    info.scenario_explanation = explanation

    return info


def print_human_readable(info: EnvironmentInfo, quick: bool = False) -> None:
    """Print human-readable environment report."""
    
    if quick:
        # Quick mode - just show recommendation
        print("\n" + "=" * 60)
        print("🎯 RECOMMENDATION")
        print("=" * 60)
        print(f"\n{info.scenario_explanation}\n")
        print(f"Command:\n  {info.scenario_command.replace(chr(10), chr(10) + '  ')}\n")
        
        if info.warnings:
            print("⚠️  Warnings:")
            for w in info.warnings[:3]:  # Show top 3
                print(f"   - {w}")
            print()
        return
    
    print("\n" + "=" * 60)
    print("NeMo Curator Environment Detection")
    print("=" * 60 + "\n")

    # Platform
    status = "✅" if info.platform_supported else "❌"
    print(f"Platform: {info.platform} {status}")

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

    # Docker
    print("\n--- Docker ---")
    if info.docker.running:
        print("Docker: ✅ Running")
        if info.docker.images:
            print("Available images:")
            for img in info.docker.images:
                age_str = f", {img.age_days}d old" if img.age_days else ""
                recommended = " ⭐" if f"{img.name}:{img.tag}" == info.docker.recommended_image else ""
                print(f"  - {img.name}:{img.tag} ({img.size_gb:.1f}GB{age_str}){recommended}")
        else:
            print("  No NeMo Curator images found")
    elif info.docker.installed:
        print("Docker: ⚠️  Installed but not running")
    else:
        print("Docker: ❌ Not installed")

    # Existing packages
    if info.existing_packages:
        print(f"\nExisting packages: {', '.join(info.existing_packages)}")

    # Recommendation
    print("\n" + "=" * 60)
    print("🎯 RECOMMENDATION")
    print("=" * 60)
    print(f"\n{info.scenario_explanation}\n")
    print(f"Command:\n  {info.scenario_command.replace(chr(10), chr(10) + '  ')}\n")

    # Warnings
    if info.warnings:
        print("⚠️  Warnings:")
        for warning in info.warnings:
            print(f"  - {warning}")
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
        "--json",
        action="store_true",
        help="Output JSON format for agent parsing",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode - just show recommendation",
    )

    args = parser.parse_args()

    info = detect_environment(args.modality)

    if args.json:
        # Convert to dict, handling nested dataclasses
        data = asdict(info)
        print(json.dumps(data, indent=2, default=str))
    else:
        print_human_readable(info, quick=args.quick)


if __name__ == "__main__":
    main()
