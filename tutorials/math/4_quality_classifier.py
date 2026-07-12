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

from loguru import logger

from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.math.classifiers.finemath import FineMathClassifier
from nemo_curator.stages.resources import Resources
from nemo_curator.stages.text.io.reader import JsonlReader
from nemo_curator.stages.text.io.writer import JsonlWriter


def build_pipeline(input_glob: str, output_dir: str, text_field: str = "text") -> Pipeline:
    p = Pipeline(
        name="math_quality_classifier",
        description="Classify mathematical content quality using the FineMath model",
    )

    # Reader (composite): tune both file partitioning and jsonl_reader stages
    p.add_stage(
        JsonlReader(file_paths=input_glob).with_(
            {
                "file_partitioning": {"resources": Resources(cpus=1.0)},
                "jsonl_reader": {"resources": Resources(cpus=1.0)},
            }
        )
    )

    # Classifier (composite): tune tokenizer and model sub-stages
    p.add_stage(
        FineMathClassifier(text_field=text_field).with_(
            {
                "finemath_classifier_tokenizer": {"resources": Resources(cpus=1.0)},
                "finemath_classifier_model": {"resources": Resources(cpus=1.0, gpus=1.0)},
            }
        )
    )

    # Writer (single stage)
    p.add_stage(JsonlWriter(path=output_dir).with_(resources=Resources(cpus=1.0)))

    return p


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Math (FineMath) classifier on JSONL files",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Glob or directory for JSONL input files",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for JSONL results",
    )
    parser.add_argument(
        "--text-field",
        default="text",
        help="Column to classify (default: 'text', use 'cleaned_text' after LLM cleanup)",
    )
    args = parser.parse_args()

    ray_client = RayClient()
    ray_client.start()

    try:
        pipeline = build_pipeline(args.input, args.output, text_field=args.text_field)
        logger.info(pipeline.describe())

        pipeline.run()
    finally:
        ray_client.stop()


if __name__ == "__main__":
    main()
