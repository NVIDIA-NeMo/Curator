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
from nemo_curator.core.client import RayClient
from nemo_curator.backends.experimental.ray_data import RayDataExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.models.client import AsyncOpenAIClient
from nemo_curator.stages.audio.request.omni_llm_request import OmniLLMRequestStage
from nemo_curator.stages.text.io.writer.jsonl import JsonlWriter
from nemo_curator.stages.audio.request.prepare_omni_lhotse import PrepareOmniLhotseStage
from nemo_curator.core.serve import InferenceModelConfig, InferenceServer

"""
FROM nvcr.io/nvidian/nemo-curator:nightly
uv pip install texterrors
python tutorials/audio/qwen_omni/run_serve_pipeline.py --input_manifest ./input.jsonl --output_pat /tmp/ --host localhost --port 8888 --lhotse_mode nemo_raw --start-server
"""


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
    parser.add_argument("--port", type=int, default=8200, help="vLLM API port")
    parser.add_argument("--host", type=str, default="localhost", help="vLLM API host")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-Omni-30B-A3B-Instruct", help="Model name")
    parser.add_argument("--user_prompt", type=str, default="Transcribe audio.", help="User prompt")
    parser.add_argument("--lhotse_mode", type=str, default="", help="Lhotse mode, nemo_tarred or lhotse_shar")
    parser.add_argument("--system_prompt", type=str, default="You are a helpful assistant.", help="System prompt")
    parser.add_argument("--api-key", type=str, default="dummy-key", help="API key")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p")
    parser.add_argument("--no-ray-local", action="store_true", help="Skip ray.init(local); use existing cluster")
    parser.add_argument(
        "--s3cfg",
        type=str,
        default="",
        help="Path to AIS/S3 config file and section, e.g. ~/.s3cfg[default]. "
        "Endpoint and token can also be set via AIS_ENDPOINT and AIS_AUTHN_TOKEN env vars.",
    )

    # Optional: launch vLLM as a subprocess (for single-container setups)
    parser.add_argument(
        "--start-server", action="store_true", help="Launch vLLM as a subprocess before running the pipeline"
    )
    parser.add_argument("--vllm-python", type=str, default="python", help="Python executable for vLLM subprocess")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="vLLM dtype (with --start-server)")
    parser.add_argument("--max-model-len", type=int, default=65536, help="vLLM max model length (with --start-server)")
    parser.add_argument(
        "--tensor-parallel-size", type=int, default=2, help="vLLM tensor parallel size (with --start-server)"
    )
    parser.add_argument(
        "--pipeline-parallel-size", type=int, default=1, help="vLLM pipeline parallel size (with --start-server)"
    )
    parser.add_argument(
        "--data-parallel-size", type=int, default=1, help="vLLM data parallel size (with --start-server)"
    )
    parser.add_argument("--replicas", type=int, default=1, help="Number of replicas")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    if not args.no_ray_local:
        client = RayClient()
        client.start()

    server = None
    if args.start_server:
        args.host = "localhost"
        engine_kwargs = {
            "tensor_parallel_size": args.tensor_parallel_size,
            "max_model_len": args.max_model_len,
            "pipeline_parallel_size": args.pipeline_parallel_size,
            "data_parallel_size": args.data_parallel_size,
        }
        deployment_config = {"autoscaling_config": {"min_replicas": args.replicas, "max_replicas": args.replicas}}
        config = InferenceModelConfig(
            model_identifier=args.model_name,
            deployment_config=deployment_config,
            engine_kwargs=engine_kwargs,
        )
        server = InferenceServer(models=[config], port=args.port, verbose=True)
        server.start()

    base_url = f"http://{args.host}:{args.port}/v1"
    llm_client = AsyncOpenAIClient(base_url=base_url, api_key=args.api_key)
    generation_config = {
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }

    pipeline = Pipeline(name="qwen3_omni")
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

    if server is not None:
        server.stop()


if __name__ == "__main__":
    args = parse_args()
    main(args)
