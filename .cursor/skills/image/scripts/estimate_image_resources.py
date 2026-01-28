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

"""Estimate resources for image processing pipelines."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.introspect import estimate_gpu_memory


def count_images(input_path: str) -> tuple[int, int]:
    """Count tar files and estimate image count."""
    path = Path(input_path)
    if not path.exists():
        return 0, 0

    tar_count = len(list(path.glob("*.tar")))
    # Estimate ~1000 images per tar file
    estimated_images = tar_count * 1000

    return tar_count, estimated_images


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate image pipeline resources")
    parser.add_argument("--input-path", "-i", required=True)
    parser.add_argument("--aesthetic", action="store_true", default=True)
    parser.add_argument("--nsfw", action="store_true", default=True)
    parser.add_argument("--dedup", action="store_true", default=False)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    tar_count, image_count = count_images(args.input_path)

    if tar_count == 0:
        print(f"No tar files found in {args.input_path}")
        sys.exit(1)

    # Image stages share CLIP model, so max GPU is ~4GB
    max_gpu = 4.0 if (args.aesthetic or args.nsfw) else 0.0

    # ~100 images/sec throughput
    time_hours = image_count / (100 * 3600)

    estimate = {
        "tar_files": tar_count,
        "estimated_images": image_count,
        "max_gpu_memory_gb": max_gpu,
        "estimated_time_hours": round(time_hours, 1),
        "recommended_batch_size": 100,
    }

    if args.json:
        import json
        print(json.dumps(estimate, indent=2))
        return

    print("=" * 50)
    print("IMAGE PIPELINE RESOURCE ESTIMATE")
    print("=" * 50)
    print(f"Tar Files: {estimate['tar_files']}")
    print(f"Estimated Images: {estimate['estimated_images']}")
    print(f"GPU Memory: {estimate['max_gpu_memory_gb']} GB")
    print(f"Estimated Time: {estimate['estimated_time_hours']} hours")


if __name__ == "__main__":
    main()
