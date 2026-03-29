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


import argparse
import ray

from nemo_curator.backends.experimental.ray_data import RayDataExecutor
from nemo_curator.models.client import OpenAIClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio.request.onmi_llm_request import OmniLLMRequestStage
from nemo_curator.stages.audio.request.prepare_omni_request import PrepareOmniRequestStage
from nemo_curator.stages.text.io.reader.jsonl import JsonlReader
from nemo_curator.stages.text.io.writer.jsonl import JsonlWriter
from nemo_curator.stages.audio.request.prepare_omni_lhotse import PrepareOmniLhotseStage


# Use a local in-process Ray runtime so we don't try to connect to an existing cluster (e.g. GCS at 127.0.1.1:6379).
# Omit this block if you have started a cluster with `ray start --head` and want to use it.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen3-Omni pipeline with JSONL input/output.")
    parser.add_argument(
        "--input_manifest",
        type=str,
        required=True,
        help="Input JSONL path. Can be a single file or a directory containing multiple files.",
    )
    parser.add_argument("--input_tar", type=str, default="", help="Input tar file range")
    parser.add_argument("--input_index", type=str, default="", help="Input dali index for reading from tar file")
    parser.add_argument("--output_path", type=str, default="output/qwen3_omni/", help="Output directory")
    parser.add_argument("--host", type=str, default="localhost", help="vLLM API host")
    parser.add_argument("--port", type=int, default=8200, help="vLLM API port")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-Omni-30B-A3B-Instruct", help="Model name")
    parser.add_argument("--user_prompt", type=str, default="Transcribe audio.", help="User prompt")
    parser.add_argument("--lhotse_mode", type=str, default="", help="Lhotse mode, nemo_tarred or lhotse_shar")
    parser.add_argument("--system_prompt", type=str, default="You are a helpful assistant.", help="System prompt")
    parser.add_argument("--api-key", type=str, default="dummy-key", help="API key")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p")
    parser.add_argument("--no-ray-local", action="store_true", help="Skip ray.init(local); use existing cluster")

    # Optional: launch vLLM as a subprocess (for single-container setups)
    parser.add_argument(
        "--start-server", action="store_true", help="Launch vLLM as a subprocess before running the pipeline"
    )
    parser.add_argument("--vllm-python", type=str, default="python", help="Python executable for vLLM subprocess")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="vLLM dtype (with --start-server)")
    parser.add_argument("--max-model-len", type=int, default=65536, help="vLLM max model length (with --start-server)")
    parser.add_argument(
        "--tensor-parallel-size", type=int, default=1, help="vLLM tensor parallel size (with --start-server)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    server = None
    if args.start_server:
        from nemo_curator.core.vllm_server import VLLMSubprocessServer

        server = VLLMSubprocessServer(
            model=args.model_name,
            port=args.port,
            python_executable=args.vllm_python,
            extra_args=[
                "--dtype",
                args.dtype,
                "--max-model-len",
                str(args.max_model_len),
                "--allowed-local-media-path",
                "/",
                "-tp",
                str(args.tensor_parallel_size),
            ],
        )
        server.start()

    try:
        if not args.no_ray_local:
            ray.init(address="local", ignore_reinit_error=True)

        base_url = f"http://{args.host}:{args.port}/v1"
        llm_client = OpenAIClient(base_url=base_url, api_key=args.api_key)
        generation_config = {
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
        }

        pipeline = Pipeline(name="qwen3_omni")
        if not args.lhotse_mode:
            pipeline.add_stage(JsonlReader(file_paths=args.input_manifest))
            pipeline.add_stage(
                PrepareOmniRequestStage(
                    format="data_url",
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
    finally:
        if server is not None:
            server.stop()


if __name__ == "__main__":
    main()
