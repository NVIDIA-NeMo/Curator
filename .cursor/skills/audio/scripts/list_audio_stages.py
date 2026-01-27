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

"""List available audio processing stages in NeMo Curator."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.introspect import estimate_gpu_memory, get_stages_by_modality

AUDIO_CATEGORIES = {
    "io": ["AudioToDocumentStage"],
    "inference": ["InferenceAsrNemoStage"],
    "metrics": ["GetPairwiseWerStage", "GetAudioDurationStage"],
    "filtering": ["PreserveByValueStage"],
    "datasets": ["CreateInitialManifestFleursStage"],
}


def get_category(stage_name: str) -> str:
    for category, stages in AUDIO_CATEGORIES.items():
        if stage_name in stages:
            return category
    return "other"


def main() -> None:
    parser = argparse.ArgumentParser(description="List available audio stages")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    stages = get_stages_by_modality("audio")
    stages.sort(key=lambda s: (get_category(s.name), s.name))

    if args.json:
        import json
        output = [{"name": s.name, "category": get_category(s.name),
                   "gpu_memory_gb": estimate_gpu_memory(s.name)} for s in stages]
        print(json.dumps(output, indent=2))
        return

    if not stages:
        print("No audio stages found. Install nemo-curator[audio].")
        return

    print(f"Found {len(stages)} audio stages:\n")

    current_category = None
    for stage in stages:
        category = get_category(stage.name)
        if category != current_category:
            current_category = category
            print(f"\n## {category.upper()}")

        gpu_mem = estimate_gpu_memory(stage.name)
        gpu_str = f" (GPU: {gpu_mem}GB)" if gpu_mem > 0 else " (CPU)"
        print(f"  {stage.name}{gpu_str}")

        if args.verbose:
            print(f"    {stage.description}")


if __name__ == "__main__":
    main()
