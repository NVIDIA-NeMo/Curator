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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse

from pipeline_utils import create_nemotron_parse_pdf_argparser, create_nemotron_parse_pdf_pipeline

from nemo_curator.backends.ray_data import RayDataExecutor
from nemo_curator.backends.utils import get_available_cpu_gpu_resources
from nemo_curator.stages.interleaved.pdf.nemotron_parse import create_nemotron_parse_inference_server


def main() -> None:
    parser = create_nemotron_parse_pdf_argparser()
    parser.set_defaults(inference_batch_size=32)
    args = parser.parse_args()

    args.output_dir = os.path.abspath(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.inference_server_endpoint is not None:
        _run_pipeline(
            args,
            args.inference_server_endpoint,
            args.inference_server_model_name or args.model_path,
            args.client_num_workers or 4,
        )
        return

    _, available_gpus = get_available_cpu_gpu_resources(init_and_shutdown=True)
    num_gpus = int(available_gpus)
    if num_gpus < 1:
        parser.error("Dynamo serving requires at least one Ray-visible GPU")

    model_name = args.model_path
    engine_kwargs = {"enforce_eager": True} if args.enforce_eager else {}
    if args.parse_mode is not None:
        engine_kwargs.update(
            {
                "distributed_executor_backend": "uni",
                "tensor_parallel_size": 1,
                "max_model_len": 16384,
                "max_num_seqs": args.max_num_seqs,
                "max_num_batched_tokens": 16384,
                "attention_config": {"flash_attn_version": 3},
            }
        )
    server = create_nemotron_parse_inference_server(
        model_path=args.model_path,
        model_name=model_name,
        backend="dynamo",
        num_replicas=num_gpus,
        engine_kwargs=engine_kwargs or None,
    )

    with server:
        _run_pipeline(args, server.endpoint, model_name, args.client_num_workers or 4 * num_gpus)


def _run_pipeline(args: argparse.Namespace, endpoint: str, model_name: str, client_num_workers: int) -> None:
    pipeline = create_nemotron_parse_pdf_pipeline(
        args,
        inference_server_endpoint=endpoint,
        inference_server_model_name=model_name,
        inference_server_client_num_workers=client_num_workers,
    )
    if args.executor == "xenna":
        from nemo_curator.backends.xenna import XennaExecutor

        executor = XennaExecutor()
    else:
        executor = RayDataExecutor()
    pipeline.run(executor)


if __name__ == "__main__":
    main()
