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

from pipeline_utils import (
    create_nemotron_parse_pdf_argparser,
    create_nemotron_parse_pdf_pipeline,
    validate_manifest_path,
    validate_output_directory,
)

from nemo_curator.backends.ray_data import RayDataExecutor
from nemo_curator.backends.utils import get_available_cpu_gpu_resources
from nemo_curator.stages.interleaved.pdf.nemotron_parse import create_nemotron_parse_inference_server
from nemo_curator.tasks.utils import TaskPerfUtils


def main() -> None:
    parser = create_nemotron_parse_pdf_argparser()
    parser.set_defaults(inference_batch_size=32)
    parser.add_argument("--executor", choices=["ray-data", "xenna"], default="ray-data")
    args = parser.parse_args()

    if args.reuse_encoded_images and args.parse_mode != "single":
        parser.error("--reuse-encoded-images requires --parse-mode single")

    args.output_dir = os.path.abspath(args.output_dir)
    if args.pages_per_task is None:
        os.makedirs(args.output_dir, exist_ok=True)
    elif os.path.exists(args.output_dir):
        msg = f"Output directory {args.output_dir} already exists"
        raise FileExistsError(msg)
    validate_output_directory(args.output_dir)

    if args.parse_mode is not None and args.inference_server_endpoint is None and args.nemotron_parse_path is None:
        parser.error("--parse-mode requires --inference-server-endpoint or --nemotron-parse-path")
    validate_manifest_path(args.manifest)
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
    nemotron_parse_path = args.nemotron_parse_path if args.parse_mode is not None else None
    engine_kwargs = {"enforce_eager": True} if args.enforce_eager else {}
    if nemotron_parse_path is not None:
        engine_kwargs["max_num_seqs"] = args.max_num_seqs
    server = create_nemotron_parse_inference_server(
        model_path=args.model_path,
        model_name=model_name,
        backend="dynamo",
        num_replicas=num_gpus,
        engine_kwargs=engine_kwargs or None,
        nemotron_parse_path=nemotron_parse_path,
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
    output_tasks = pipeline.run(executor)
    if args.parse_mode is not None:
        valid_pages = TaskPerfUtils.get_aggregated_stage_stat(
            output_tasks, "nemotron_parse_inference", "custom.num_valid_pages"
        )
        if valid_pages <= 0:
            msg = "Parse 3.5 pipeline produced no valid pages; inspect the retained output errors"
            raise RuntimeError(msg)


if __name__ == "__main__":
    main()
