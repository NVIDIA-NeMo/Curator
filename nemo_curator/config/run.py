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

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.reader import JsonlReader, ParquetReader
from nemo_curator.stages.text.io.writer import JsonlWriter, ParquetWriter


def create_pipeline_from_yaml(cfg: DictConfig) -> Pipeline:
    logger.info(f"Hydra config: {OmegaConf.to_yaml(cfg)}")
    pipeline = Pipeline(name="yaml_pipeline", description="Create and execute a pipeline from a YAML file")

    # Add stages to the pipeline
    for p in cfg.stages:
        if "input_file_type" in p:  # Text-specific
            if p.input_file_type not in ["jsonl", "parquet"]:
                raise ValueError(f"Invalid input file type: {p.input_file_type}")
            reader_stage = JsonlReader if p.input_file_type == "jsonl" else ParquetReader
            stage = reader_stage(
                file_paths=p.file_paths,
                files_per_partition=p.files_per_partition,
                blocksize=p.blocksize,
                fields=p.fields,
            )
        elif "output_file_type" in p:  # Text-specific
            if p.output_file_type not in ["jsonl", "parquet"]:
                raise ValueError(f"Invalid output file type: {p.output_file_type}")
            writer_stage = JsonlWriter if p.output_file_type == "jsonl" else ParquetWriter
            stage = writer_stage(path=p.path, fields=p.fields)
        else:
            stage = hydra.utils.instantiate(p)
        pipeline.add_stage(stage)

    return pipeline


@hydra.main(version_base=None)
def main(cfg: DictConfig) -> None:
    ray_client = RayClient()
    ray_client.start()

    pipeline = create_pipeline_from_yaml(cfg)

    # Execute pipeline
    print("Starting pipeline execution...")
    results = pipeline.run()

    # Print results
    print("\nPipeline completed!")
    print(f"Total output tasks: {len(results) if results else 0}")

    ray_client.stop()


if __name__ == "__main__":
    main()
