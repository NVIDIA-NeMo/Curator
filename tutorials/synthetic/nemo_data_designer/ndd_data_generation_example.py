# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import os
import time

import pandas as pd
import data_designer.config as dd
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.synthetic.nemo_data_designer.base import BaseDataDesignerStage
from nemo_curator.stages.text.io.reader.jsonl import JsonlReader
from nemo_curator.stages.text.io.writer.jsonl import JsonlWriter


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic data using Nemo Data Designer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--seed-dataset-path",
        type=str,
        default=None,
        help="Path to the seed dataset in JSONL format",
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


def main() -> None:
    """Main function to run the synthetic data generation pipeline."""
    args = parse_args()

    # Create pipeline
    pipeline = Pipeline(name="ndd_data_generation", description="Generate synthetic text data using Nemo Data Designer")

    # Add reader stage to read the seed dataset
    pipeline.add_stage(
        JsonlReader(
            file_paths=args.seed_dataset_path + "/*.jsonl",
            fields=["diagnosis", "patient_summary"], # Specify fields to read
        )
    )

    # Define Nemo Data Designer config builder
    if args.data_designer_config_file is not None:
        config_builder = dd.DataDesignerConfigBuilder.from_config(args.data_designer_config_file)
    # Manually define the config builder
    else:
        MODEL_PROVIDER = "nvidia"
        MODEL_ID = "meta/llama-3.3-70b-instruct"
        MODEL_ALIAS = "llama-3.3-70b"
        model_configs = [
            dd.ModelConfig(
                alias=MODEL_ALIAS,
                model=MODEL_ID,
                provider=MODEL_PROVIDER,
                inference_parameters=dd.ChatCompletionInferenceParams(
                    temperature=1.0,
                    top_p=1.0,
                    max_tokens=2048,
                ),
            )
        ]

        config_builder = dd.DataDesignerConfigBuilder(model_configs=model_configs)

        config_builder.add_column(
            name="patient_sampler",
            column_type="sampler",
            sampler_type="person_from_faker",
        )

        config_builder.add_column(
            name="doctor_sampler",
            column_type="sampler",
            sampler_type="person_from_faker",
        )

        config_builder.add_column(
            name="patient_id",
            column_type="sampler",
            sampler_type="uuid",
            params={
                "prefix": "PT-",
                "short_form": True,
                "uppercase": True,
            },
        )

        config_builder.add_column(
            name="first_name",
            column_type="expression",
            expr="{{ patient_sampler.first_name}}",
        )

        config_builder.add_column(
            name="last_name",
            column_type="expression",
            expr="{{ patient_sampler.last_name }}",
        )

        config_builder.add_column(
            name="dob",
            column_type="expression",
            expr="{{ patient_sampler.birth_date }}",
        )

        config_builder.add_column(
            name="symptom_onset_date",
            column_type="sampler",
            sampler_type="datetime",
            params={"start": "2024-01-01", "end": "2024-12-31"},
        )

        config_builder.add_column(
            name="date_of_visit",
            column_type="sampler",
            sampler_type="timedelta",
            params={"dt_min": 1, "dt_max": 30, "reference_column_name": "symptom_onset_date"},
        )

        config_builder.add_column(
            name="physician",
            column_type="expression",
            expr="Dr. {{ doctor_sampler.last_name }}",
        )

        config_builder.add_column(
            name="physician_notes",
            column_type="llm-text",
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
            model_alias=MODEL_ALIAS,
        )


    # Add the Nemo Data Designer stage
    pipeline.add_stage(
        BaseDataDesignerStage(
            config_builder=config_builder,
            data_designer_config_file=args.data_designer_config_file,
        )
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
    results = pipeline.run()
    end_time = time.time()

    elapsed_time = end_time - start_time

    # Print results
    print("\nPipeline completed!")
    print(f"Total execution time: {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes)")

    # Collect output file paths and read generated data
    output_files = []
    all_data_frames = []
    if results:
        print(f"\nGenerated data saved to: {args.output_path}")
        for result in results:
            if hasattr(result, "data") and result.data:
                for file_path in result.data:
                    print(f"  - {file_path}")
                    output_files.append(file_path)
                    # Read the JSONL file to get the actual data
                    df = pd.read_json(file_path, lines=True)
                    all_data_frames.append(df)

    # Display sample of generated documents
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

if __name__ == "__main__":
    main()
