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

"""Run the Nemotron-Parse PDF tutorial with NVIDIA Dynamo serving."""

from __future__ import annotations

import os

from main import create_nemotron_parse_pdf_argparser, create_nemotron_parse_pdf_pipeline

from nemo_curator.backends.ray_data import RayDataExecutor
from nemo_curator.backends.utils import get_available_cpu_gpu_resources
from nemo_curator.core.serve import (
    DynamoRouterConfig,
    DynamoServerConfig,
    DynamoVLLMModelConfig,
    InferenceServer,
)


def main() -> None:
    parser = create_nemotron_parse_pdf_argparser()
    parser.set_defaults(inference_batch_size=32)
    args = parser.parse_args()

    if args.backend != "vllm":
        parser.error("Dynamo serving requires --backend vllm")

    args.output_dir = os.path.abspath(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    _, available_gpus = get_available_cpu_gpu_resources(init_and_shutdown=True)
    num_gpus = int(available_gpus)
    if num_gpus < 1:
        parser.error("Dynamo serving requires at least one Ray-visible GPU")

    model_name = args.model_path
    model = DynamoVLLMModelConfig(
        model_identifier=args.model_path,
        num_replicas=num_gpus,
        engine_kwargs={
            "trust_remote_code": True,
            "dtype": "bfloat16",
            "limit_mm_per_prompt": {"image": 1},
            "enable_prefix_caching": False,
            "disable_hybrid_kv_cache_manager": False,
        },
        dynamo_kwargs={"enable_multimodal": True},
        runtime_env={"uv": {"packages": ["albumentations==2.0.8"]}},
    )
    backend = DynamoServerConfig(
        request_plane="tcp",
        router=DynamoRouterConfig(router_kwargs={"trust_remote_code": True}),
        subprocess_env={"DYN_TCP_REQUEST_TIMEOUT": "300"},
    )

    with InferenceServer(models=[model], backend=backend, health_check_timeout_s=900) as server:
        pipeline = create_nemotron_parse_pdf_pipeline(
            args,
            inference_server_endpoint=server.endpoint,
            inference_server_model_name=model_name,
            inference_server_client_num_workers=4 * num_gpus,
        )
        pipeline.run(RayDataExecutor())


if __name__ == "__main__":
    main()
