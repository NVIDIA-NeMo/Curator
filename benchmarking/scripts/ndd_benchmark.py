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

# ruff: noqa: PLR0913

"""NeMo Data Designer (NDD) benchmarking script.

Key args:
  --inference-server-type  ray-serve | nvidia-nim
  --engine-kwargs          JSON vLLM kwargs, e.g. '{"tensor_parallel_size": 4}'
  --autoscaling-config     JSON Ray Serve autoscaling, e.g. '{"min_replicas": 1, "max_replicas": 1}'
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import data_designer.config as dd
from loguru import logger
from utils import setup_executor, write_benchmark_results

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.synthetic.nemo_data_designer.data_designer import DataDesignerStage
from nemo_curator.stages.text.io.reader.jsonl import JsonlReader
from nemo_curator.stages.text.io.writer.jsonl import JsonlWriter
from nemo_curator.tasks.utils import TaskPerfUtils
from nemo_curator.utils.file_utils import get_all_file_paths_under

if TYPE_CHECKING:
    from nemo_curator.core.serve import InferenceServer


# ---------------------------------------------------------------------------
# Data Designer config builder
# ---------------------------------------------------------------------------


def _build_config(model_id: str, provider_name: str) -> dd.DataDesignerConfigBuilder:
    """Build the DataDesigner config for the medical-notes generation task."""
    model_alias = model_id

    model_configs = [
        dd.ModelConfig(
            alias=model_alias,
            model=model_id,
            provider=provider_name,
            skip_health_check=True,
            inference_parameters=dd.ChatCompletionInferenceParams(
                temperature=1.0,
                top_p=1.0,
                max_tokens=2048,
            ),
        ),
    ]

    config_builder = dd.DataDesignerConfigBuilder(model_configs=model_configs)

    # -- Sampler columns ------------------------------------------------
    config_builder.add_column(
        dd.SamplerColumnConfig(
            name="patient_sampler",
            sampler_type=dd.SamplerType.PERSON_FROM_FAKER,
            params=dd.PersonFromFakerSamplerParams(),
        )
    )
    config_builder.add_column(
        dd.SamplerColumnConfig(
            name="doctor_sampler",
            sampler_type=dd.SamplerType.PERSON_FROM_FAKER,
            params=dd.PersonFromFakerSamplerParams(),
        )
    )
    config_builder.add_column(
        dd.SamplerColumnConfig(
            name="patient_id",
            sampler_type=dd.SamplerType.UUID,
            params=dd.UUIDSamplerParams(prefix="PT-", short_form=True, uppercase=True),
        )
    )

    # -- Expression columns ---------------------------------------------
    config_builder.add_column(dd.ExpressionColumnConfig(name="first_name", expr="{{ patient_sampler.first_name}}"))
    config_builder.add_column(dd.ExpressionColumnConfig(name="last_name", expr="{{ patient_sampler.last_name }}"))
    config_builder.add_column(dd.ExpressionColumnConfig(name="dob", expr="{{ patient_sampler.birth_date }}"))
    config_builder.add_column(
        dd.SamplerColumnConfig(
            name="symptom_onset_date",
            sampler_type=dd.SamplerType.DATETIME,
            params=dd.DatetimeSamplerParams(start="2024-01-01", end="2024-12-31"),
        )
    )
    config_builder.add_column(
        dd.SamplerColumnConfig(
            name="date_of_visit",
            sampler_type=dd.SamplerType.TIMEDELTA,
            params=dd.TimeDeltaSamplerParams(dt_min=1, dt_max=30, reference_column_name="symptom_onset_date"),
        )
    )
    config_builder.add_column(dd.ExpressionColumnConfig(name="physician", expr="Dr. {{ doctor_sampler.last_name }}"))

    # -- LLM column -----------------------------------------------------
    config_builder.add_column(
        dd.LLMTextColumnConfig(
            name="physician_notes",
            prompt="""\
You are a primary-care physician who just had an appointment with {{ first_name }} {{ last_name }},
who has been struggling with symptoms from {{ diagnosis }} since {{ symptom_onset_date }}.
The date of today's visit is {{ date_of_visit }}.

{{ patient_summary }}

Write careful notes about your visit with {{ first_name }},
as Dr. {{ doctor_sampler.first_name }} {{ doctor_sampler.last_name }}.

Format the notes as a busy doctor might.
Respond with only the notes, no other text.
""",
            model_alias=model_alias,
        )
    )

    return config_builder


# ---------------------------------------------------------------------------
# InferenceServer helpers
# ---------------------------------------------------------------------------


def _start_inference_server(
    model_id: str,
    engine_kwargs: dict[str, Any] | None = None,
    autoscaling_config: dict[str, Any] | None = None,
) -> "InferenceServer":
    """Start a local InferenceServer and return it."""
    from nemo_curator.core.serve import InferenceModelConfig, InferenceServer

    engine_kwargs = engine_kwargs or {}
    autoscaling_config = autoscaling_config or {"min_replicas": 1, "max_replicas": 1}

    server_config = InferenceModelConfig(
        model_identifier=model_id,
        deployment_config={"autoscaling_config": autoscaling_config},
        engine_kwargs=engine_kwargs,
    )

    server = InferenceServer(models=[server_config])
    server.start()
    return server


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def run_ndd_benchmark(  # noqa: PLR0915
    inference_server_type: str,
    model_id: str,
    input_path: str,
    output_path: str,
    executor: str,
    num_files: int | None,
    engine_kwargs: dict[str, Any] | None = None,
    autoscaling_config: dict[str, Any] | None = None,
    **kwargs,  # noqa: ARG001
) -> dict[str, Any]:
    """Run the NDD benchmark and collect metrics."""
    input_path = Path(input_path)
    output_path = Path(output_path).absolute()
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Model type: {inference_server_type}")
    logger.info(f"Model ID: {model_id}")
    logger.info(f"Input path: {input_path}")
    logger.info(f"Output path: {output_path}")
    logger.info(f"Executor: {executor}")

    # Resolve input files using Curator utility
    input_files = get_all_file_paths_under(str(input_path), keep_extensions="jsonl")
    if num_files is not None and num_files > 0:
        logger.info(f"Using {num_files} of {len(input_files)} input files")
        input_files = input_files[:num_files]

    inference_server = None
    model_providers = None
    serve_startup_s = 0.0

    if inference_server_type == "ray-serve":
        logger.info(f"Starting local InferenceServer with engine_kwargs={engine_kwargs}")
        serve_start = time.perf_counter()
        inference_server = _start_inference_server(model_id, engine_kwargs, autoscaling_config)
        serve_startup_s = time.perf_counter() - serve_start
        logger.info(f"InferenceServer ready at {inference_server.endpoint} (startup: {serve_startup_s:.1f}s)")

        provider_name = "local"
        model_providers = [
            dd.ModelProvider(
                name=provider_name,
                endpoint=inference_server.endpoint,
                api_key="unused",  # pragma: allowlist secret
            )
        ]
    elif inference_server_type == "nvidia-nim":
        if not os.environ.get("NVIDIA_API_KEY"):
            msg = "NVIDIA_API_KEY must be set for nvidia-nim model type"
            raise OSError(msg)
        provider_name = "nvidia"
    else:
        msg = f"Unknown inference_server_type: {inference_server_type}"
        raise ValueError(msg)

    # -- Build config and run pipeline ----------------------------------
    config_builder = _build_config(model_id, provider_name)

    executor_obj = setup_executor(executor)

    pipeline = Pipeline(
        name="ndd_benchmark_pipeline",
        stages=[
            JsonlReader(file_paths=input_files, fields=["diagnosis", "patient_summary"]),
            DataDesignerStage(config_builder=config_builder, model_providers=model_providers),
            JsonlWriter(path=str(output_path)),
        ],
    )

    logger.info("Starting NDD pipeline...")
    run_start_time = time.perf_counter()
    try:
        output_tasks = pipeline.run(executor_obj)
    finally:
        run_time_taken = time.perf_counter() - run_start_time

        if inference_server is not None:
            inference_server.stop()

    # -- Post-run: extract metrics from _stage_perf ----------------------
    input_row_count = int(
        TaskPerfUtils.get_aggregated_stage_stat(output_tasks, "DataDesignerStage", "custom.num_input_records")
    )
    output_row_count = int(
        TaskPerfUtils.get_aggregated_stage_stat(output_tasks, "DataDesignerStage", "custom.num_output_records")
    )
    input_tokens_median_per_record = float(
        TaskPerfUtils.get_aggregated_stage_stat(
            output_tasks, "DataDesignerStage", "custom.input_tokens_median_per_record"
        )
    )
    output_tokens_median_per_record = float(
        TaskPerfUtils.get_aggregated_stage_stat(
            output_tasks, "DataDesignerStage", "custom.output_tokens_median_per_record"
        )
    )
    throughput_rows_per_sec = output_row_count / run_time_taken if run_time_taken > 0 else 0

    logger.success(f"NDD benchmark completed in {run_time_taken:.2f}s")
    logger.success(f"Input:  {input_row_count} rows")
    logger.success(f"Output: {output_row_count} rows")
    logger.success(f"Input tokens median per record: {input_tokens_median_per_record:,}")
    logger.success(f"Output tokens median per record: {output_tokens_median_per_record:,}")
    logger.success(f"Throughput: {throughput_rows_per_sec:.2f} rows/sec")

    return {
        "metrics": {
            "is_success": True,
            "time_taken_s": run_time_taken,
            "inference_server_type": inference_server_type,
            "model_id": model_id,
            "input_row_count": input_row_count,
            "output_row_count": output_row_count,
            "input_tokens_median_per_record": input_tokens_median_per_record,
            "output_tokens_median_per_record": output_tokens_median_per_record,
            "throughput_rows_per_sec": throughput_rows_per_sec,
            "serve_startup_s": serve_startup_s,
            "num_files": num_files or "all",
        },
        "tasks": output_tasks,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="NeMo Data Designer (NDD) benchmark")
    parser.add_argument("--benchmark-results-path", required=True, help="Path to write benchmark results")
    parser.add_argument("--input-path", required=True, help="Path to input JSONL seed data")
    parser.add_argument("--output-path", required=True, help="Path to write generated output")
    parser.add_argument(
        "--inference-server-type",
        required=True,
        choices=["ray-serve", "nvidia-nim"],
        help="Model serving backend",
    )
    parser.add_argument("--model-id", default="openai/gpt-oss-20b", help="Model identifier")
    parser.add_argument("--executor", default="ray_data", choices=["ray_data", "xenna"], help="Pipeline executor")
    parser.add_argument("--num-files", type=int, default=None, help="Limit number of input files (default: all)")
    parser.add_argument(
        "--engine-kwargs",
        type=str,
        default=None,
        help="JSON string of vLLM engine kwargs (e.g. '{\"tensor_parallel_size\": 4}')",
    )
    parser.add_argument(
        "--autoscaling-config",
        type=str,
        default=None,
        help='JSON string of Ray Serve autoscaling config (e.g. \'{"min_replicas": 1, "max_replicas": 4}\')',
    )

    args = parser.parse_args()

    logger.info("=== NDD Benchmark Starting ===")
    logger.info(f"Arguments: {vars(args)}")

    # Parse JSON string args
    engine_kwargs = json.loads(args.engine_kwargs) if args.engine_kwargs else None
    autoscaling_config = json.loads(args.autoscaling_config) if args.autoscaling_config else None

    success_code = 1
    result_dict: dict[str, Any] = {
        "params": vars(args),
        "metrics": {"is_success": False},
        "tasks": [],
    }
    try:
        result_dict.update(
            run_ndd_benchmark(
                inference_server_type=args.inference_server_type,
                model_id=args.model_id,
                input_path=args.input_path,
                output_path=args.output_path,
                executor=args.executor,
                num_files=args.num_files,
                engine_kwargs=engine_kwargs,
                autoscaling_config=autoscaling_config,
            )
        )
        success_code = 0 if result_dict["metrics"]["is_success"] else 1
    finally:
        write_benchmark_results(result_dict, args.benchmark_results_path)
    return success_code


if __name__ == "__main__":
    raise SystemExit(main())
