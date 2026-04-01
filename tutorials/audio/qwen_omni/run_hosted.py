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

"""Run audio processing pipeline against the NVIDIA inference API.

Unlike ``run_pipeline.py`` (which needs a local vLLM server), this script
uses hosted models (e.g. Gemini via NVIDIA inference API) that accept
the ``input_audio`` format over HTTP — no local GPU required.

Usage::

    export API_KEY="$NVIDIA_API_KEY"
    python run_hosted.py --input_manifest /path/to/input.jsonl

The input JSONL should have rows with ``audio_url`` pointing to local
audio files, and optionally ``text`` and ``system_text`` fields.
"""

import argparse
import os

import ray

from nemo_curator.backends.experimental.ray_data import RayDataExecutor
from nemo_curator.models.client import OpenAIClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio.request.omni_llm_request import OmniLLMRequestStage
from nemo_curator.stages.audio.request.prepare_omni_request import PrepareOmniRequestStage
from nemo_curator.stages.text.io.reader.jsonl import JsonlReader
from nemo_curator.stages.text.io.writer.jsonl import JsonlWriter
from nemo_curator.stages.audio.request.prepare_omni_lhotse import PrepareOmniLhotseStage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run audio pipeline via NVIDIA inference API.")
    parser.add_argument(
        "--input_manifest",
        type=str,
        required=True,
        help="Input JSONL path. Can be a single file or a directory containing multiple files.",
    )
    parser.add_argument("--input_tar", type=str, default="", help="Input tar file range")
    parser.add_argument("--input_index", type=str, default="", help="Input dali index for reading from tar file")
    parser.add_argument("--output_path", type=str, default="output/hosted_audio/", help="Output directory")
    parser.add_argument("--user_prompt", type=str, default="Transcribe audio.", help="User prompt")
    parser.add_argument("--system_prompt", type=str, default="You are a helpful assistant.", help="System prompt")
    parser.add_argument(
        "--base-url",
        type=str,
        default="https://inference-api.nvidia.com",
        help="API base URL",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="gcp/google/gemini-2.5-pro",
        help="Model name (must support input_audio)",
    )
    parser.add_argument("--api-key", type=str, default=None, help="API key (or set API_KEY / NVIDIA_API_KEY env var)")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Max tokens")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p")
    parser.add_argument("--no-ray-local", action="store_true", help="Skip ray.init(local); use existing cluster")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    api_key = args.api_key or os.getenv("API_KEY") or os.getenv("NVIDIA_API_KEY")
    if not api_key:
        print("ERROR: API key required. Set --api-key, API_KEY, or NVIDIA_API_KEY.")
        raise SystemExit(1)

    if not args.no_ray_local:
        ray.init(address="local", ignore_reinit_error=True)

    llm_client = OpenAIClient(base_url=args.base_url, api_key=api_key)
    generation_config = {
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }

    pipeline = Pipeline(name="hosted_audio")
    if not args.lhotse_mode:
        pipeline.add_stage(JsonlReader(file_paths=args.input_manifest))
        pipeline.add_stage(
            PrepareOmniRequestStage(
                format="input_data",
                input_tar=args.input_tar,
                input_index=args.input_index,
                user_prompt=args.user_prompt,
                system_prompt=args.system_prompt,
            )
        )
    else:
        pipeline.add_stage(
            PrepareOmniLhotseStage(
                lhotse_mode=args.lhotse_mode,
                input_manifest=args.input_manifest,
                input_tar=args.input_tar,
                user_prompt=args.user_prompt,
                user_prompt_key="",
                system_prompt=args.system_prompt,
            )
        )
    pipeline.add_stage(
        OmniLLMRequestStage(
            client=llm_client,
            model_name=args.model_name,
            generation_config=generation_config,
        )
    )
    pipeline.add_stage(JsonlWriter(path=args.output_path, write_kwargs={"force_ascii": False}).with_(batch_size=1))
    pipeline.run(executor=RayDataExecutor())


if __name__ == "__main__":
    main()
