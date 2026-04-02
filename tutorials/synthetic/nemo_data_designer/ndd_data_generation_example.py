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

"""
Synthetic data generation example using NeMo Data Designer (NDD).

This tutorial generates synthetic medical notes from a seed CSV of symptom
descriptions.  It supports two modes:

  1. **Local InferenceServer (default)** - starts a Ray Serve + vLLM server on
     the local cluster and points NDD at it via a custom ModelProvider.  No
     external API key is required.

  2. **Remote provider** (e.g. NVIDIA NIM) - uses a hosted API endpoint.
     Requires the appropriate API key in the environment (e.g. NVIDIA_API_KEY).

Usage examples::

    # Local model (default) - serves openai/gpt-oss-20b locally:
    python ndd_data_generation_example.py

    # Local model with a different HuggingFace model:
    python ndd_data_generation_example.py --model google/gemma-3-27b-it

    # Remote NVIDIA NIM API:
    python ndd_data_generation_example.py \
        --provider nvidia \
        --model meta/llama-3.3-70b-instruct

    # Use a Data Designer YAML config file:
    python ndd_data_generation_example.py --data-designer-config-file config.yaml
"""

import argparse
import time
from pathlib import Path

import data_designer.config as dd
import pandas as pd

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.synthetic.nemo_data_designer.data_designer import DataDesignerStage
from nemo_curator.stages.text.io.reader.jsonl import JsonlReader
from nemo_curator.stages.text.io.writer.jsonl import JsonlWriter
from nemo_curator.utils.file_utils import get_all_file_paths_under


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic data using Nemo Data Designer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data-designer-config-file",
        type=str,
        default=None,
        help="Path to the data designer config file",
    )

    parser.add_argument(
        "--output-path",
        type=str,
        default="./synthetic_output",
        help="Directory path to save the generated synthetic data in JSONL format",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-oss-20b",
        help="Model identifier (HuggingFace ID or local path)",
    )

    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        help="Model provider name (e.g. 'nvidia'). If not set, a local InferenceServer is started.",
    )

    return parser.parse_args()


SEED_CSV_URL = "https://raw.githubusercontent.com/NVIDIA/GenerativeAIExamples/refs/heads/main/nemo/NeMo-Data-Designer/data/gretelai_symptom_to_diagnosis.csv"


def download_and_convert_seed_data(
    output_dir: str | Path | None = None,
    records_per_file: int = 10,
    number_of_records: int = 20,
) -> str:
    """Download seed CSV from URL, convert to JSONL (chunked), return output dir path."""
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "processed_seed_data"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SEED_CSV_URL, sep=",", encoding="utf-8")
    df = df.head(number_of_records)
    for i, start in enumerate(range(0, len(df), records_per_file)):
        chunk = df.iloc[start : start + records_per_file]
        chunk.to_json(
            output_dir / f"{i:06d}.jsonl",
            orient="records",
            lines=True,
            force_ascii=False,
            date_format="iso",
        )
    return str(output_dir)


def _build_config(model_id: str, provider_name: str, model_alias: str) -> dd.DataDesignerConfigBuilder:
    """Build the Data Designer config with medical notes generation."""
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
        )
    ]

    config_builder = dd.DataDesignerConfigBuilder(model_configs=model_configs)

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
            params=dd.UUIDSamplerParams(
                prefix="PT-",
                short_form=True,
                uppercase=True,
            ),
        )
    )

    config_builder.add_column(
        dd.ExpressionColumnConfig(
            name="first_name",
            expr="{{ patient_sampler.first_name}}",
        )
    )

    config_builder.add_column(
        dd.ExpressionColumnConfig(
            name="last_name",
            expr="{{ patient_sampler.last_name }}",
        )
    )

    config_builder.add_column(
        dd.ExpressionColumnConfig(
            name="dob",
            expr="{{ patient_sampler.birth_date }}",
        )
    )

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
            params=dd.TimeDeltaSamplerParams(
                dt_min=1,
                dt_max=30,
                reference_column_name="symptom_onset_date",
            ),
        )
    )

    config_builder.add_column(
        dd.ExpressionColumnConfig(
            name="physician",
            expr="Dr. {{ doctor_sampler.last_name }}",
        )
    )

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


def _collect_output_info(output_path: str) -> tuple[list, list]:
    """Collect output file paths and dataframes from files under output_path."""
    output_files = get_all_file_paths_under(output_path, recurse_subdirectories=True, keep_extensions=".jsonl")
    print(f"\nGenerated data saved to: {output_path}")
    for file_path in output_files:
        print(f"  - {file_path}")
    all_data_frames = [pd.read_json(f, lines=True) for f in output_files]
    return output_files, all_data_frames


def _print_sample_documents(output_files: list, all_data_frames: list) -> None:
    """Print a sample of generated documents."""
    print("\n" + "=" * 50)
    print("Sample of generated documents:")
    print("=" * 50)
    for i, df in enumerate(all_data_frames):
        print(f"\nFile {i + 1}: {output_files[i]}")
        print(f"Number of documents: {len(df)}")
        print("\nGenerated text (showing first 5):")
        for j, row in enumerate(df.head(3).to_dict(orient="records")):
            print(f"Document {j + 1}:")
            for key, value in row.items():
                print(f"[{key}]:")
                print(f"{value}")
            print("-" * 40)
        break


def main() -> None:  # noqa: PLR0915
    """Main function to run the synthetic data generation pipeline."""
    args = parse_args()
    print("Preparing seed data (download + CSV→JSONL)...")
    seed_dir = download_and_convert_seed_data()
    print(f"Seed data ready: {seed_dir}")

    import torch

    from nemo_curator.backends.ray_data import RayDataExecutor
    from nemo_curator.core.client import RayClient

    NUM_GPUS = 4  # noqa: N806

    if args.provider is None and torch.cuda.device_count() < NUM_GPUS:
        error_msg = "The number of GPUs on this machine are lesser than the default this tutorial was tested with, please update `num_gpus` passed into `RayClient`"
        raise ValueError(error_msg)

    client = RayClient(num_cpus=16, num_gpus=NUM_GPUS)
    client.start()

    model_alias = "local-llm"
    provider_name = args.provider or "local"
    inference_server = None
    model_providers = None

    # If no remote provider specified, start a local InferenceServer
    if args.provider is None:
        from nemo_curator.backends.experimental.utils import get_available_cpu_gpu_resources
        from nemo_curator.core.serve import InferenceModelConfig, InferenceServer

        _, num_gpus = get_available_cpu_gpu_resources()
        num_gpus = int(num_gpus)
        print(f"Detected {num_gpus} GPUs, using tensor_parallel_size={num_gpus}")

        server_config = InferenceModelConfig(
            model_identifier=args.model,
            deployment_config={
                "autoscaling_config": {
                    "min_replicas": 1,
                    "max_replicas": 1,
                },
            },
            engine_kwargs={
                "tensor_parallel_size": num_gpus,
            },
        )

        inference_server = InferenceServer(models=[server_config])
        inference_server.start()

        # Create a custom ModelProvider pointing at the local server
        model_providers = [
            dd.ModelProvider(
                name=provider_name,
                endpoint=inference_server.endpoint,
                api_key="unused",  # pragma: allowlist secret
            )
        ]
        print(f"Local InferenceServer ready at {inference_server.endpoint}")

    # Build the NDD config
    if args.data_designer_config_file is not None:
        config_builder = dd.DataDesignerConfigBuilder.from_config(args.data_designer_config_file)
    else:
        config_builder = _build_config(
            model_id=args.model,
            provider_name=provider_name,
            model_alias=model_alias,
        )

    pipeline = Pipeline(
        name="ndd_data_generation", description="Generate synthetic text data using NeMo Data Designer"
    )

    pipeline.add_stage(
        JsonlReader(
            file_paths=seed_dir + "/*.jsonl",
            fields=["diagnosis", "patient_summary"],
        )
    )

    ndd_stage = DataDesignerStage(config_builder=config_builder, model_providers=model_providers)
    ndd_stage.xenna_stage_spec = lambda: {"num_workers": 2}
    pipeline.add_stage(ndd_stage)

    pipeline.add_stage(
        JsonlWriter(
            path=args.output_path,
        )
    )

    print(pipeline.describe())
    print("\n" + "=" * 50 + "\n")

    print("Starting synthetic data generation pipeline...")
    start_time = time.perf_counter()
    try:
        pipeline.run(executor=RayDataExecutor())
    finally:
        end_time = time.perf_counter()
        if inference_server is not None:
            inference_server.stop()
        client.stop()

    elapsed_time = end_time - start_time

    print("\nPipeline completed!")
    print(f"Total execution time: {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes)")
    output_files, all_data_frames = _collect_output_info(args.output_path)
    _print_sample_documents(output_files, all_data_frames)


if __name__ == "__main__":
    main()
