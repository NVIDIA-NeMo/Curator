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
Quick synthetic data generation example for Nemo Data Designer
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

    return parser.parse_args()


SEED_CSV_URL = "https://raw.githubusercontent.com/NVIDIA/GenerativeAIExamples/refs/heads/main/nemo/NeMo-Data-Designer/data/gretelai_symptom_to_diagnosis.csv"


def download_and_convert_seed_data(
    output_dir: str | Path | None = None,
    records_per_file: int = 10,
) -> str:
    """Download seed CSV from URL, convert to JSONL (chunked), return output dir path."""
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "processed_seed_data"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SEED_CSV_URL, sep=",", encoding="utf-8")
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

def _build_config_manually() -> dd.DataDesignerConfigBuilder:
    """Build the default Data Designer config with medical notes generation."""
    model_provider = "nvidia"
    model_id = "meta/llama-3.3-70b-instruct"
    model_alias = "llama-3.3-70b"
    model_configs = [
        dd.ModelConfig(
            alias=model_alias,
            model=model_id,
            provider=model_provider,
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
    output_files = get_all_file_paths_under(
        output_path, recurse_subdirectories=True, keep_extensions=".jsonl"
    )
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
        for j, row in enumerate(df.head(5).to_dict(orient="records")):
            print(f"Document {j + 1}:")
            for key, value in row.items():
                print(f"[{key}]:")
                print(f"{value}")
            print("-" * 40)


def main() -> None:
    """Main function to run the synthetic data generation pipeline."""
    args = parse_args()
    print("Preparing seed data (download + CSV→JSONL)...")
    seed_dir = download_and_convert_seed_data()
    print(f"Seed data ready: {seed_dir}")

    pipeline = Pipeline(name="ndd_data_generation", description="Generate synthetic text data using Nemo Data Designer")

    # Add reader stage to read the seed dataset
    pipeline.add_stage(
        JsonlReader(
            file_paths=seed_dir + "/*.jsonl",
            fields=["diagnosis", "patient_summary"],
        )
    )

    # Define Nemo Data Designer config builder
    if args.data_designer_config_file is not None:
        config_builder = dd.DataDesignerConfigBuilder.from_config(args.data_designer_config_file)
    # Manually define the config builder
    else:
        config_builder = _build_config_manually()


    # Add the Nemo Data Designer stage
    pipeline.add_stage(
        DataDesignerStage(config_builder=config_builder)
    )

    # Add JSONL writer to save the generated data
    pipeline.add_stage(
        JsonlWriter(
            path=args.output_path,
        )
    )

    # Print pipeline description
    print(pipeline.describe())
    print("\n" + "=" * 50 + "\n")

    # Execute pipeline with timing
    print("Starting synthetic data generation pipeline...")
    start_time = time.time()
    pipeline.run()
    end_time = time.time()

    elapsed_time = end_time - start_time

    # Print results
    print("\nPipeline completed!")
    print(f"Total execution time: {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes)")
    output_files, all_data_frames = _collect_output_info(args.output_path)
    _print_sample_documents(output_files, all_data_frames)

if __name__ == "__main__":
    main()
