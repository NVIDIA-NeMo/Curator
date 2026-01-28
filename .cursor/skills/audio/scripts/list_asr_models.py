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

"""List available NeMo ASR models for audio transcription."""
import argparse

# Common NeMo ASR models
ASR_MODELS = [
    {
        "name": "nvidia/stt_en_fastconformer_hybrid_large_pc",
        "language": "en",
        "description": "English FastConformer Hybrid Large (Punctuation & Capitalization)",
        "gpu_memory_gb": 16,
    },
    {
        "name": "nvidia/stt_en_fastconformer_ctc_large",
        "language": "en",
        "description": "English FastConformer CTC Large",
        "gpu_memory_gb": 12,
    },
    {
        "name": "nvidia/stt_en_conformer_ctc_large",
        "language": "en",
        "description": "English Conformer CTC Large",
        "gpu_memory_gb": 12,
    },
    {
        "name": "nvidia/stt_de_fastconformer_hybrid_large_pc",
        "language": "de",
        "description": "German FastConformer Hybrid Large",
        "gpu_memory_gb": 16,
    },
    {
        "name": "nvidia/stt_es_fastconformer_hybrid_large_pc",
        "language": "es",
        "description": "Spanish FastConformer Hybrid Large",
        "gpu_memory_gb": 16,
    },
    {
        "name": "nvidia/stt_fr_fastconformer_hybrid_large_pc",
        "language": "fr",
        "description": "French FastConformer Hybrid Large",
        "gpu_memory_gb": 16,
    },
    {
        "name": "nvidia/stt_hy_fastconformer_hybrid_large_pc",
        "language": "hy",
        "description": "Armenian FastConformer Hybrid Large",
        "gpu_memory_gb": 16,
    },
    {
        "name": "nvidia/stt_multilingual_fastconformer_hybrid_large_pc",
        "language": "multi",
        "description": "Multilingual FastConformer Hybrid Large",
        "gpu_memory_gb": 16,
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(description="List available NeMo ASR models")
    parser.add_argument("--language", "-l", help="Filter by language code")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    models = ASR_MODELS
    if args.language:
        models = [m for m in models if m["language"] == args.language or m["language"] == "multi"]

    if args.json:
        import json
        print(json.dumps(models, indent=2))
        return

    print("Available NeMo ASR Models:\n")
    print(f"{'Model Name':<55} {'Lang':<6} {'GPU':<6}")
    print("-" * 70)

    for model in models:
        print(f"{model['name']:<55} {model['language']:<6} {model['gpu_memory_gb']}GB")

    print("\nUsage:")
    print("  python generate_audio_config.py --model-name <model_name> ...")


if __name__ == "__main__":
    main()
